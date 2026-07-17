
using JuMP, LazySets

# need dependence on N because just LazySet (without parametrisation on number datatype is not a type)
struct JuMPWrapper{N<:Number} <: LazySet{N}
    objfun  # original objective of the optimization problem
    model   # JuMP model containing the constraints
    outvars  # variables describing the current dim of the set
end


LazySets.dim(S::JuMPWrapper) = length(S.outvars)


function LazySets.σ(a::AbstractVector, S::JuMPWrapper)
    @objective(S.model, Max, a' * S.outvars)
    # can't use set_silent(), unset_silent() here,
    # somehow JuMP will complain that optimize! has not been called
    #set_silent(S.model)
    optimize!(S.model)
    #unset_silent(S.model)

    return value.(S.outvars)
end 