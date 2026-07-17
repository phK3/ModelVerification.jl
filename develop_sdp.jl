
using ModelVerification, LazySets, Flux, LinearAlgebra, Parameters, MosekTools, Plots, JuMP, SCS
import ModelVerification: prepare_problem, search_branches, ForwardProp, BackwardProp, ModelGraph

const MV = ModelVerification

@with_kw struct LayerSDP <: ForwardProp
    optimizer = SCS.Optimizer
    pre_bound_method::Union{ForwardProp,BackwardProp,Nothing} = nothing
end


function add_psd_input_set_constraint(opt_model, batch_info, start_node, input_set::AbstractHyperrectangle)
    # don't do anything, all the constraints will be handled in the first activation layer we encounter 
    # (or the last layer if no activation layer is found in the network)
    batch_info[start_node][:pre_lower] = low(input_set)
    batch_info[start_node][:pre_upper] = high(input_set)
end


function MV.prepare_method(prop_method::LayerSDP, batch_input::AbstractVector, batch_output::AbstractVector,
    batch_inheritance::AbstractVector, model_info::ModelGraph)::Tuple{AbstractVector,Dict}
    batch_info = MV.init_propagation(prop_method, batch_input, batch_output, model_info)

    # store :size_after_layer
    batch_info = MV.get_all_layer_output_size(model_info, batch_info, size(LazySets.center(batch_input[1])))

    # store previous nodes
    for node in model_info.all_nodes
        batch_info[node][:prev_nodes] = model_info.node_prevs[node]
    end

    # compute pre-activation bounds
    if !isnothing(prop_method.pre_bound_method)
        # compute bounds using specified method
        _, pre_batch_info = MV.prepare_method(
            prop_method.pre_bound_method, batch_input, batch_output, batch_inheritance, model_info)
        _, pre_batch_info = MV.propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            pre_bound = pre_batch_info[prev_node][:bound]
            l, u = MV.compute_bound(pre_bound)
            # convert CUDA.CuArray to Vector for indexing in relu propagation
            batch_info[node][:pre_lower] = collect(l[:, 1])
            batch_info[node][:pre_upper] = collect(u[:, 1])
        end
    end

    # create optimization model
    opt_model = Model(prop_method.optimizer)
    batch_info[:opt_model] = opt_model

    # create input variables and add constraints (suppose there is only one input)
    start_node = model_info.start_nodes[1]

    add_psd_input_set_constraint(opt_model, batch_info, start_node, batch_input[1])
    batch_info[start_node][:last_act_node] = start_node

    # objective function has to be added later
    # if we want to maximize violation, we first have to wait until the output layer
    # is encoded.

    return batch_output, batch_info
end


function MV.propagate_layer_batch(prop_method::LayerSDP, layer::Dense, bound::AbstractVector, batch_info::Dict)::AbstractVector
    # create optimization variable of the current node
    node = batch_info[:current_node]

    prev_nodes = batch_info[node][:prev_nodes]
    @assert length(prev_nodes) == 1
    prev_node = prev_nodes[1]
    if !haskey(batch_info[prev_node], :weight)
        batch_info[node][:weight] = layer.weight
        batch_info[node][:bias]   = layer.bias
    else
        # two chained linear layers
        W = batch_info[prev_node][:weight]
        b = batch_info[prev_node][:bias]
        batch_info[node][:weight] = layer.weight * W
        batch_info[node][:bias]   = layer.weight * b .+ layer.bias
    end

    batch_info[node][:last_act_node] = batch_info[prev_node][:last_act_node]

    return bound
end


function MV.propagate_layer_batch(prop_method::LayerSDP, layer::typeof(relu), bound::AbstractVector, batch_info::Dict)::AbstractVector
    # create optimization variable of the current node
    node = batch_info[:current_node]
    opt_model = batch_info[:opt_model]

    batch_info[node][:last_act_node] = node

    # get optimization variable of the previous actiation node
    # (since we don't really do anything in the linear layers)
    prev_nodes = batch_info[node][:prev_nodes]
    @assert length(prev_nodes) == 1
    prev_node = prev_nodes[1]
    prev_act_node = batch_info[prev_node][:last_act_node]

    n_neurons = batch_info[node][:size_after_layer][1]
    n_neurons_prev = batch_info[prev_act_node][:size_after_layer][1]
    P = @variable(opt_model, [1:1 + n_neurons_prev + n_neurons, 1:1 + n_neurons_prev + n_neurons], PSD, base_name="P_$(node)")
    batch_info[node][:opt_vars] = Dict(:z => P)
    
    # idxs to access the previous and the last layer's entries in the matrix variable
    x_prev = 2:2+n_neurons_prev-1
    x      = 2+n_neurons_prev:2+n_neurons_prev+n_neurons-1

    W = batch_info[prev_node][:weight]
    b = batch_info[prev_node][:bias]

    if haskey(batch_info[node], :pre_lower)
        # get pre-activation bound
        l = batch_info[node][:pre_lower]
        u = batch_info[node][:pre_upper]

        # add constraint of relu layer
        active = l .>= 0.0
        inactive = u .<= 0.0
        unstable = .~active .& .~inactive

        # triangle ReLU upper bound
        λ, β = MV.relu_upper_bound(l[unstable], u[unstable])
        @constraint(opt_model, P[1,x[unstable]] .<= λ .* (W*P[1,x_prev] .+ b)[unstable] .+ β, base_name="relu_Δ_$(node)")
    else
        active = fill(false, n_neurons)
        inactive = fill(false, n_neurons)
        unstable = fill(true, n_neurons)
    end

    # fixed ReLUs
    # TODO: also constraints on quadratic vars here?
    # if y == Wx+b, then yy == (Wx + b)(Wx + b), xy == x(Wx + b)
    @constraint(opt_model, P[1,x[active]] .== (W*P[1,x_prev])[active,:] .+ b[active], base_name="fixed_active_$(node)")

    # TODO: does it help, if we also set x² == 0?
    # basically, we could just set all x == 0, xx == 0, xy == 0
    @constraint(opt_model, P[1,x[inactive]] .== 0.0, base_name="fixed_inactive_$(node)")

    # SDP relaxation
    @constraint(opt_model, P[1,x[unstable]] .>= 0.0, base_name="relu_geq_0_$(node)")
    @constraint(opt_model, P[1,x[unstable]] .>= (W*P[1,x_prev])[unstable,:] .+ b[unstable], base_name="relu_geq_x_$(node)")
    # y(y - (Wx+b)) == 0
    # yy - yWx - yb == 0
    # yy - W(xy') == yb
    @constraint(opt_model, diag(P[x[unstable],x[unstable]] .- (W*P[x_prev,x[unstable]])[unstable,:]) .== b[unstable] .* P[1,x[unstable]], base_name="relu_quad_$(node)")

    if haskey(batch_info[prev_act_node], :pre_lower)
        # only add bound constraints if we have bounds
        # i.e. always for the input layer
        l_prev = batch_info[prev_act_node][:pre_lower]
        u_prev = batch_info[prev_act_node][:pre_upper]

        # bound constraints
        # xx - (l + u)x + lu <= 0
        # xx - lx - ux + lu == (x - l)(x - u) <= 0
        @constraint(opt_model, diag(P[x_prev, x_prev]) .- (l_prev .+ u_prev) .* P[1,x_prev] .+ l_prev .* u_prev .<= 0, base_name="bounds_$(node)")
    end

    if haskey(batch_info[prev_act_node], :opt_vars)
        # connect this matrix var to the previous layer's matrix var
        # if the previous layer has no matrix variable, it might have
        # been the input layer.
        # We handled that by encoding the bounds on the variables above
        
        # TODO: rename to :z to :P ??    
        P_prev = batch_info[prev_act_node][:opt_vars][:z]
        
        # outputs of previous layer are at the end of the matrix var
        n_prev = size(P_prev, 1)
        x̂_prev = [1] ∪ (n_prev - (n_neurons_prev - 1):n_prev)
        x̂      = [1] ∪ x_prev
        @constraint(opt_model, P_prev[x̂_prev, x̂_prev] .== P[x̂, x̂], base_name="connect_$(node)")
    end

    @constraint(opt_model, P[1,1] == 1.)

    return bound
end


"""
Appends a dedicated variable to the SDP that is equal to the output of the NN and returns that variable.
"""
function extract_output_variable(prop_method::LayerSDP, batch_info::Dict, model_info::ModelGraph)
    @assert length(model_info.final_nodes) == 1 "Only models with a single output are supported!"
    final_node = model_info.final_nodes[1]
    W = batch_info[final_node][:weight]
    b = batch_info[final_node][:bias]

    prev_act_node = batch_info[final_node][:last_act_node]
    P = batch_info[prev_act_node][:opt_vars][:z]

    n_prev = size(P, 1)
    n_neurons_prev = batch_info[prev_act_node][:size_after_layer][1]
    x_prev = (n_prev - (n_neurons_prev - 1):n_prev)

    opt_model = batch_info[:opt_model]
    y_out = @variable(opt_model, [1:length(b)])
    @constraint(opt_model, y_out .== W*P[1,x_prev] .+ b)

    return y_out
end



function create_riai_example()
    toymodel = Chain(Dense(2 => 2, relu), Dense(2 => 2, relu), Dense(2 => 2)) |> f64

    toymodel[1].weight .= [1. 1; 1 -1]
    toymodel[1].bias .= [0.0, 0]
    toymodel[2].weight .= [1.0 1; 1 -1]
    toymodel[2].bias .= [-0.5, 0]
    toymodel[3].weight .= [-1.0 1; 0 1]
    toymodel[3].bias .= [3.0, 0]

    return toymodel
end


## Example 
riai = create_riai_example()

input_set = Hyperrectangle(low=-ones(2), high=ones(2))
output_set = HPolytope([1. 1.], [6.5])

problem = Problem(riai, input_set, output_set)

### SDP doesn't need preactivation bounds
solver = LayerSDP(optimizer=SCS.Optimizer, pre_bound_method=nothing)
search_method = BFS(max_iter=100, batch_size=1)  # not needed for SDP
split_method = Bisect(1)  # not needed for SDP

model_info, problem = prepare_problem(search_method, split_method, solver, problem)
batch_output, batch_info = MV.prepare_method(solver, [input_set], [output_set], [nothing], model_info)
batch_bound, batch_info = MV.propagate(solver, model_info, batch_info);


### Now with bounds
solver2 = LayerSDP(optimizer=SCS.Optimizer, pre_bound_method=Crown(use_gpu=false, bound_heuristics=MV.parallel_slope))

model_info2, problem2 = prepare_problem(search_method, split_method, solver2, problem)
batch_output2, batch_info2 = MV.prepare_method(solver2, [input_set], [output_set], [nothing], model_info2)
batch_bound2, batch_info2 = MV.propagate(solver2, model_info2, batch_info2);

### Maximize 1st Output Dimension
y_out = extract_output_variable(solver, batch_info, model_info)

opt_model = batch_info[:opt_model]

c = [1., 0]
@objective(opt_model, Max, c'*y_out)
optimize!(opt_model)
objective_value(opt_model)


### Plot the overapproximated output set 
jw = JuMPWrapper{Float64}(objective_function(opt_model), opt_model, y_out)

set_silent(jw.model)
plot(jw, label="SDP")
unset_silent(jw.model)

xs = sample(input_set, 10000)
xs = hcat(xs...)
ys = riai(xs)
scatter!(ys[1,:], ys[2,:], label="samples")


### Plot comparison with and w/o bounds
set_silent(jw.model)
plot(jw, label="SDP")
unset_silent(jw.model)

y_out2 = extract_output_variable(solver2, batch_info2, model_info2)
opt_model2 = batch_info2[:opt_model]
jw2 = JuMPWrapper{Float64}(objective_function(opt_model2), opt_model2, y_out2)

set_silent(jw2.model)
plot!(jw2, label="SDP+bounds")
unset_silent(jw2.model)
current()


### ACAS Debug
include("./vnncomp_scripts/vnnlib_parser.jl");
net = MV.build_flux_model("../vnncomp2022_benchmarks/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000_simple.onnx");

specs = read_vnnlib_simple("../vnncomp2022_benchmarks/benchmarks/acasxu/vnnlib/prop_1.vnnlib", 5, 5)
X_range, Y_cons = specs[1]
lb = [bd[1] for bd in X_range]
ub = [bd[2] for bd in X_range]

Y_con = Y_cons[1]

input_set = Hyperrectangle(low=lb, high=ub)
input_set = Hyperrectangle(input_set.center, 0.1 .* input_set.radius)
A = hcat(Y_con[1]...)'
b = Y_con[2]

Yc = HPolytope(A, b)
Y  = Complement(Yc)

l = Dense(50, 50)
l.weight .= I(50)
l.bias .= zeros(50)
net_partial = Chain(net.layers[1:4]..., l)
# don't care about output constraint here anyways, so don't modify it
# problem = Problem(net_partial, input_set, Y);
problem = Problem(net, input_set, Y);


# The SDP solvers fine, when no bound propagation is used, and is infeasible, when I use bound propagation
# ==> I suspect, that the condition number of the matrix is horrible, if I have large bounds and then even square them for the constraints!!!
solver = LayerSDP(optimizer=SCS.Optimizer, pre_bound_method=nothing) # Crown(use_gpu=false, bound_heuristics=MV.parallel_slope))
search_method = BFS(max_iter=100, batch_size=1)  # actually not needed for SDP
split_method = Bisect(1)  # actually not needed for SDP

model_info, problem = prepare_problem(search_method, split_method, solver, problem)
batch_output, batch_info = MV.prepare_method(solver, [input_set], [output_set], [nothing], model_info)
batch_bound, batch_info = MV.propagate(solver, model_info, batch_info);


y_out = extract_output_variable(solver, batch_info, model_info)
opt_model = batch_info[:opt_model]

# c = zeros(50)
c = zeros(5)
c[1] = 1.
@objective(opt_model, Max, c'*y_out)
optimize!(opt_model)
objective_value(opt_model)