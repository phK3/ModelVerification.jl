


"""
Constructs a valid upper bound u(x) = α*x + δ ≥ f(x) over x ∈ [l, u] given fixed slope α

Requires a function fmax(α, l, u) that computes a value δ ≥ maxₓ f(x) - α*x over x ∈ [l, u] taylored to the function you want to overapproximate.

args:
    fmax - function fmax(α, l, u) computes overapproximate bound on f(x) - α*x
    α - slope of the upper relaxation to be constructed
    l - lower bound of the domain of interest
    u - upper bound of the domain of interest
"""
function shift_upper_linear(fmax, α, l, u)
    δ = fmax(α, l, u)
    λ = α
    β = δ
    return λ, β
end


function shift_lower_linear(fmin, α, l, u)
    δ = fmin(α, l, u)
    λ = α
    β = δ
    return λ, β
end


# TODO: is α > 0 ? ... construction efficient?
function exp_critical_points_linear(α, l, u)
    x = clamp(α > 0 ? log(α) : -Inf, l, u)
    return [x, l, u]
end

function elu_critical_points_linear(α, l, u)
    # no critical points other than 0 and u on positive side (or infinitely many, but then 0, u are also critical with same value)
    x = clamp(α > 0 ? log(α) : -Inf, l, 0)
    return [x, l, u, 0]
end

σ_inv(x) = -log(1/x - 1)

function σ_critical_points_linear(α, l, u)
    # solve σ(x)(1 - σ(x)) - α = 0
    # subs σ(x) = s
    # pq formula
    r = 0.25 - α
    if r >= 0
        s₁ = clamp(0.5 + sqrt(0.25 - α), 0, 1)
        s₂ = clamp(0.5 - sqrt(0.25 - α), 0, 1)
    else
        # will map to -Inf after σ_inv and be discarded with clamp to [l, u]
        s₁ = 0
        s₂ = 0
    end

    x₁ = clamp(σ_inv(s₁), l, u)
    x₂ = clamp(σ_inv(s₂), l, u)

    return [x₁, x₂, l, u]
end 

function tanh_critical_points_linear(α, l, u)
    # d/dx tanh(x) - αx = d/dx 2σ(2x) - αx = 0 <--> d/dy 2σ(y) - α/2 y = 0 (with y = 2x)
    # <--> d/dy σ(y) - α/4 y == 0 (just divide both sides by 2)
    # then account for y = 2x for bounds and for returned critical points
    return σ_critical_points_linear(α / 4, 2*l, 2*u) ./ 2
end 

function sqrt_critical_points_linear(α, l, u)
    x = clamp(1 / (4*α^2), l, u)
    # TODO: do we want to return l, even if it is less than 0?
    return [x, l, u]
end

function sin_critical_points_linear(α, l, u)
    # ensure we only call acos() with valid values, but remember when we had to replace it
    α_valid = (-1 <= α) & (α <= 1)
    α = clamp(α, -1, 1)
    x = acos(α)

    k_min₁ = ceil((l - x) / (2π))
    k_max₁ = floor((u - x) / (2π))
    k_min₂ = ceil((l + x) / (2π))
    k_max₂ = floor((u + x) / (2π))

    x1 = clamp(ifelse(α_valid, 2*k_max₁*π + x, l), l, u)
    x2 = clamp(ifelse(α_valid, 2*k_min₁*π + x, l), l, u)
    x3 = clamp(ifelse(α_valid, 2*k_max₂*π - x, l), l, u)
    x4 = clamp(ifelse(α_valid, 2*k_min₂*π - x, l), l, u)

    return [l, x1, x2, x3, x4, u]
end

function cos_critical_points_linear(α, l, u)
    # d/dx cos(x) - αx = d/dx sin(x + 0.5π) - α*x = 0
    # <--> d/dy sin(y) - α*(y - 0.5π) = 0   (y = x + 0.5π)
    # <--> d/dy sin(y) - αy + 0.5απ = 0
    # <--> d/dy sin(y) - αy = 0
    return sin_critical_points_linear(α, l + 0.5π, u + 0.5π) .- 0.5π
end


abstract type AbstractContinuousPiecewiseLinear end

"""
Type representing continuous piecewise linear functions.

Such a function f can be represented as 
    f = λ₁x + β₁ if b₁ ≤ x ≤ b₂ 
        λ₂x + β₂ if b₂ ≤ x ≤ b₃
        ...

args:
    slopes - vector of n linear slopes λᵢ
    biases - vector of n biases βᵢ
    breakpoints - vector of n+1 breakpoints bᵢ (**including** -Inf or Inf if needed for the function!)
"""
struct ContinuousPiecewiseLinear{N} <: AbstractContinuousPiecewiseLinear
    slopes::Vector{N}
    biases::Vector{N}
    breakpoints::Vector{N}
end

Base.broadcastable(f::AbstractContinuousPiecewiseLinear) = Ref(f)

const MyAbs = ContinuousPiecewiseLinear([-1., 1], zeros(2), [-Inf, 0, Inf])
const MyHardSigmoid = ContinuousPiecewiseLinear([0, 1/6, 0], [0, 0.5, 1], [-Inf, -3, 3, Inf])
const MyReLU = ContinuousPiecewiseLinear([0, 1.], zeros(2), [-Inf, 0, Inf])

struct MyLeakyReLU{N} <: AbstractContinuousPiecewiseLinear
    slopes::Vector{N}
    biases::Vector{N}
    breakpoints::Vector{N}
end

MyLeakyReLU(slope::Number) = MyLeakyReLU([slope, 1.], zeros(2), [-Inf, 0, Inf]) 

function piecewise_lin_critical_points_linear(f::AbstractContinuousPiecewiseLinear, α, l, u)
    # TODO: this only works for **continuous** functions 
    # if we have f(x) = f1(x) at [a, b) and f2(x) at [b, c)
    # then we return [a, b, c] as critical points 
    # and we only evaluate f at those points.
    # This means that we only evaluate f2 at b, but not f1!
    relevant_mask = (f.breakpoints .>= l) .& (f.breakpoints .<= u)
    breakpoints = f.breakpoints[relevant_mask]
    return [l; clamp.(breakpoints, l, u); u]
end

     
function test_correctness(fun, crit_fun, l, u; n_test=10, n_points=200)
    xs = range(l, u, n_points)

    for i in 1:n_test
        α = randn()
        cs = crit_fun(α, l, u)

        vals_xs = fun.(xs) .- α .* xs
        vals_cs = fun.(cs) .- α .* cs
        i_xs_max = argmax(vals_xs)
        i_cs_max = argmax(vals_cs)
        i_xs_min = argmin(vals_xs)
        i_cs_min = argmin(vals_cs)

        if vals_xs[i_xs_max] > vals_cs[i_cs_max]
            println("Maximum not correct!")
            println("\tcritical point is ", cs[i_cs_max], " with value ", vals_cs[i_cs_max])
            println("\tbut found point   ", xs[i_xs_max], " with value ", vals_xs[i_xs_max])
        elseif vals_xs[i_xs_min] < vals_cs[i_cs_min]
            println("Minimum not correct!")
            println("\tcritical point is ", cs[i_cs_min], " with value ", vals_cs[i_cs_min])
            println("\tbut found point   ", xs[i_xs_min], " with value ", vals_xs[i_xs_min])
        end
    end
end
     
     
