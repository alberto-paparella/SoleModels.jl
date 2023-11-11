import SoleModels: AbstractFeature

using SoleData: channelvariable

import Base: show
import SoleLogics: syntaxstring

# Feature parentheses (e.g., for parsing/showing "main[V2]")
const UVF_OPENING_PARENTHESIS = "["
const UVF_CLOSING_PARENTHESIS = "]"
# Default prefix for variables
const UVF_VARPREFIX = "V"

"""
    abstract type VarFeature <: AbstractFeature end

Abstract type for feature functions that can be computed on (multi)variate data.
Instances of multivariate datasets have values for a number of *variables*,
which can be used to define logical features.

For example, with dimensional data (e.g., multivariate time series, digital images
and videos), features can be computed as the minimum value for a given variable
on a specific interval/rectangle/cuboid (in general, a [`SoleLogics.GeometricalWorld](@ref)).

As an example of a dimensional feature, consider *min[V1]*,
which computes the minimum for variable 1 for a given world.
`ScalarCondition`s such as *min[V1] >= 10* can be, then, evaluated on worlds.

See also
[`scalarlogiset`](@ref),
[`featvaltype`](@ref),
[`computefeature`](@ref),
[`SoleLogics.Interval`](@ref).
"""
abstract type VarFeature <: AbstractFeature end

DEFAULT_VARFEATVALTYPE = Real

"""
    featvaltype(dataset, f::VarFeature)

Return the type of the values returned by feature `f` on logiseed `dataset`.

See also [`VarFeature`](@ref).
"""
function featvaltype(dataset, f::VarFeature)
    return error("Please, provide method featvaltype(::$(typeof(dataset)), ::$(typeof(f))).")
end

"""
    computefeature(f::VarFeature, featchannel; kwargs...)

Compute a feature on a featchannel (i.e., world reading) of an instance.

See also [`VarFeature`](@ref).
"""
function computefeature(f::VarFeature, featchannel; kwargs...)
    return error("Please, provide method computefeature(::$(typeof(f)), featchannel::$(typeof(featchannel)); kwargs...).")
end


@inline (f::AbstractFeature)(args...) = computefeature(f, args...)

############################################################################################

"""
    struct MultivariateFeature <: VarFeature
        f::Function
    end

A dimensional feature represented by the application of a function to a dimensional channel.
For example, it can wrap a scalar function computing
how much a `Interval2D` world, when interpreted on an image, resembles a horse.
Note that the image has a number of spatial variables (3, for the case of RGB),
and "resembling a horse" may require a computation involving all variables.

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`AbstractUnivariateFeature`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
struct MultivariateFeature{U} <: VarFeature
    f::Function
end
syntaxstring(f::MultivariateFeature, args...; kwargs...) = "$(f.f)"

function featvaltype(dataset, f::MultivariateFeature{U}) where {U}
    return U
end

############################################################################################

"""
    abstract type AbstractUnivariateFeature <: VarFeature end

A dimensional feature represented by the application of a function to a single variable of a
dimensional channel.
For example, it can wrap a scalar function computing
how much red a `Interval2D` world, when interpreted on an image, contains.

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`UnivariateFeature`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
abstract type AbstractUnivariateFeature <: VarFeature end

"""
    computeunivariatefeature(f::AbstractUnivariateFeature, varchannel; kwargs...)

Compute a feature on a variable channel (i.e., world reading) of an instance.

See also [`AbstractUnivariateFeature`](@ref).
"""
function computeunivariatefeature(f::AbstractUnivariateFeature, varchannel::Any; kwargs...)
    return error("Please, provide method computeunivariatefeature(::$(typeof(f)), varchannel::$(typeof(varchannel)); kwargs...).")
end

i_variable(f::AbstractUnivariateFeature) = f.i_variable

function computefeature(f::AbstractUnivariateFeature, featchannel::Any)
    computeunivariatefeature(f, channelvariable(featchannel, i_variable(f)))
end

"""
    variable_name(
        f::AbstractUnivariateFeature;
        variable_names_map::Union{Nothing,AbstractDict,AbstractVector} = nothing,
        variable_name_prefix::Union{Nothing,String} = $(repr(UVF_VARPREFIX)),
    )::String

Return the name of the variable targeted by a univariate feature.
By default, an variable name is a number prefixed by $(repr(UVF_VARPREFIX));
however, `variable_names_map` or `variable_name_prefix` can be used to
customize variable names.
The prefix can be customized by specifying `variable_name_prefix`.
Alternatively, a mapping from string to integer (either via a Dictionary or a Vector)
can be passed as `variable_names_map`.
Note that only one in `variable_names_map` and `variable_name_prefix` should be provided.


See also
[`parsecondition`](@ref),
[`ScalarCondition`](@ref),
[`syntaxstring`](@ref).
"""
function variable_name(
    f::AbstractUnivariateFeature;
    variable_names_map::Union{Nothing,AbstractDict,AbstractVector} = nothing,
    variable_name_prefix::Union{Nothing,String} = nothing,
    kwargs..., # TODO remove this.
)
    if isnothing(variable_names_map)
        variable_name_prefix = isnothing(variable_name_prefix) ? UVF_VARPREFIX : variable_name_prefix
        "$(variable_name_prefix)$(i_variable(f))"
    else
        @assert isnothing(variable_name_prefix)
        "$(variable_names_map[i_variable(f)])"
    end
end

function featurename(f::AbstractFeature; kwargs...)
    return error("Please, provide method featurename(::$(typeof(f)); kwargs...).")
end

function syntaxstring(
    f::AbstractUnivariateFeature;
    opening_parenthesis::String = UVF_OPENING_PARENTHESIS,
    closing_parenthesis::String = UVF_CLOSING_PARENTHESIS,
    kwargs...
)
    n = variable_name(f; kwargs...)
    "$(featurename(f))$opening_parenthesis$n$closing_parenthesis"
end

############################################################################################

"""
    struct UnivariateFeature{U} <: AbstractUnivariateFeature
        i_variable::Integer
        f::Function
    end

A dimensional feature represented by the application of a generic function `f`
to a single variable of a dimensional channel.
For example, it can wrap a scalar function computing
how much red a `Interval2D` world, when interpreted on an image, contains.

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`AbstractUnivariateFeature`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
struct UnivariateFeature{U} <: AbstractUnivariateFeature
    i_variable::Integer
    f::Function
    function UnivariateFeature{U}(feat::UnivariateFeature) where {U<:Real}
        return new{U}(i_variable(f), feat.f)
    end
    function UnivariateFeature{U}(i_variable::Integer, f::Function) where {U<:Real}
        return new{U}(i_variable, f)
    end
    function UnivariateFeature(i_variable::Integer, f::Function)
        return UnivariateFeature{DEFAULT_VARFEATVALTYPE}(i_variable, f)
    end
end
featurename(f::UnivariateFeature) = string(f.f)

function featvaltype(dataset, f::UnivariateFeature{U}) where {U}
    return U
end

"""
    struct UnivariateNamedFeature{U} <: AbstractUnivariateFeature
        i_variable::Integer
        name::String
    end

A univariate feature solely identified by its name and reference variable.

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`AbstractUnivariateFeature`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
struct UnivariateNamedFeature{U} <: AbstractUnivariateFeature
    i_variable::Integer
    name::String
    function UnivariateNamedFeature{U}(f::UnivariateNamedFeature) where {U<:Real}
        return new{U}(i_variable(f), f.name)
    end
    function UnivariateNamedFeature{U}(i_variable::Integer, name::String) where {U<:Real}
        return new{U}(i_variable, name)
    end
    function UnivariateNamedFeature(i_variable::Integer, name::String)
        return UnivariateNamedFeature{DEFAULT_VARFEATVALTYPE}(i_variable, name)
    end
end
featurename(f::UnivariateNamedFeature) = f.name

function featvaltype(dataset, f::UnivariateNamedFeature{U}) where {U}
    return U
end

############################################################################################

"""
    struct UnivariateValue <: AbstractUnivariateFeature
        i_variable::Integer
    end

Simply the value of a scalar variable
(propositional case, when the frame has a single world).

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`AbstractUnivariateFeature`](@ref),
[`UnivariateMax`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
struct UnivariateValue <: AbstractUnivariateFeature
    i_variable::Integer
    function UnivariateValue(f::UnivariateValue)
        return new(i_variable(f))
    end
    function UnivariateValue(i_variable::Integer)
        return new(i_variable)
    end
end
featurename(f::UnivariateValue) = ""

function syntaxstring(
    f::UnivariateValue;
    kwargs...
)
    variable_name(f; kwargs...)
end

function featvaltype(dataset, f::UnivariateValue)
    return vareltype(dataset, f.i_variable)
end

############################################################################################

"""
    struct UnivariateMin <: AbstractUnivariateFeature
        i_variable::Integer
    end

Notable univariate feature computing the minimum value for a given variable.

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`AbstractUnivariateFeature`](@ref),
[`UnivariateMax`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
struct UnivariateMin <: AbstractUnivariateFeature
    i_variable::Integer
    function UnivariateMin(f::UnivariateMin)
        return new(i_variable(f))
    end
    function UnivariateMin(i_variable::Integer)
        return new(i_variable)
    end
end
featurename(f::UnivariateMin) = "min"

function featvaltype(dataset, f::UnivariateMin)
    return vareltype(dataset, f.i_variable)
end

"""
    struct UnivariateMax <: AbstractUnivariateFeature
        i_variable::Integer
    end

Notable univariate feature computing the maximum value for a given variable.

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`AbstractUnivariateFeature`](@ref),
[`UnivariateMin`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
struct UnivariateMax <: AbstractUnivariateFeature
    i_variable::Integer
    function UnivariateMax(f::UnivariateMax)
        return new(i_variable(f))
    end
    function UnivariateMax(i_variable::Integer)
        return new(i_variable)
    end
end
featurename(f::UnivariateMax) = "max"

function featvaltype(dataset, f::UnivariateMax)
    return vareltype(dataset, f.i_variable)
end

############################################################################################

"""
    struct UnivariateSoftMin{T<:AbstractFloat} <: AbstractUnivariateFeature
        i_variable::Integer
        alpha::T
    end

Univariate feature computing a "softened" version of the minimum value for a given variable.

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`AbstractUnivariateFeature`](@ref),
[`UnivariateMin`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
struct UnivariateSoftMin{T<:AbstractFloat} <: AbstractUnivariateFeature
    i_variable::Integer
    alpha::T
    function UnivariateSoftMin(f::UnivariateSoftMin)
        return new{typeof(alpha(f))}(i_variable(f), alpha(f))
    end
    function UnivariateSoftMin(i_variable::Integer, alpha::T) where {T}
        @assert !(alpha > 1.0 || alpha < 0.0) "Cannot instantiate UnivariateSoftMin with alpha = $(alpha)"
        @assert !isone(alpha) "Cannot instantiate UnivariateSoftMin with alpha = $(alpha). Use UnivariateMin instead!"
        new{T}(i_variable, alpha)
    end
end
alpha(f::UnivariateSoftMin) = f.alpha
featurename(f::UnivariateSoftMin) = "min" * utils.subscriptnumber(rstrip(rstrip(string(alpha(f)*100), '0'), '.'))

function featvaltype(dataset, f::UnivariateSoftMin)
    return vareltype(dataset, f.i_variable)
end

"""
    struct UnivariateSoftMax{T<:AbstractFloat} <: AbstractUnivariateFeature
        i_variable::Integer
        alpha::T
    end

Univariate feature computing a "softened" version of the maximum value for a given variable.

See also [`SoleLogics.Interval`](@ref),
[`SoleLogics.Interval2D`](@ref),
[`AbstractUnivariateFeature`](@ref),
[`UnivariateMax`](@ref),
[`VarFeature`](@ref), [`AbstractFeature`](@ref).
"""
struct UnivariateSoftMax{T<:AbstractFloat} <: AbstractUnivariateFeature
    i_variable::Integer
    alpha::T
    function UnivariateSoftMax(f::UnivariateSoftMax)
        return new{typeof(alpha(f))}(i_variable(f), alpha(f))
    end
    function UnivariateSoftMax(i_variable::Integer, alpha::T) where {T}
        @assert !(alpha > 1.0 || alpha < 0.0) "Cannot instantiate UnivariateSoftMax with alpha = $(alpha)"
        @assert !isone(alpha) "Cannot instantiate UnivariateSoftMax with alpha = $(alpha). Use UnivariateMax instead!"
        new{T}(i_variable, alpha)
    end
end
alpha(f::UnivariateSoftMax) = f.alpha
featurename(f::UnivariateSoftMax) = "max" * utils.subscriptnumber(rstrip(rstrip(string(alpha(f)*100), '0'), '.'))

function featvaltype(dataset, f::UnivariateSoftMax)
    return vareltype(dataset, f.i_variable)
end

############################################################################################

# These features collapse to a single value; it can be useful to know this
is_collapsing_univariate_feature(f::Union{UnivariateMin,UnivariateMax,UnivariateSoftMin,UnivariateSoftMax}) = true
is_collapsing_univariate_feature(f::UnivariateFeature) = (f.f in [minimum, maximum, mean])


_st_featop_abbr(f::UnivariateMin,     ::typeof(≥); kwargs...) = "$(variable_name(f; kwargs...)) ⪴"
_st_featop_abbr(f::UnivariateMax,     ::typeof(≤); kwargs...) = "$(variable_name(f; kwargs...)) ⪳"
_st_featop_abbr(f::UnivariateSoftMin, ::typeof(≥); kwargs...) = "$(variable_name(f; kwargs...)) $("⪴" * utils.subscriptnumber(rstrip(rstrip(string(alpha(f)*100), '0'), '.')))"
_st_featop_abbr(f::UnivariateSoftMax, ::typeof(≤); kwargs...) = "$(variable_name(f; kwargs...)) $("⪳" * utils.subscriptnumber(rstrip(rstrip(string(alpha(f)*100), '0'), '.')))"

_st_featop_abbr(f::UnivariateMin,     ::typeof(<); kwargs...) = "$(variable_name(f; kwargs...)) ⪶"
_st_featop_abbr(f::UnivariateMax,     ::typeof(>); kwargs...) = "$(variable_name(f; kwargs...)) ⪵"
_st_featop_abbr(f::UnivariateSoftMin, ::typeof(<); kwargs...) = "$(variable_name(f; kwargs...)) $("⪶" * utils.subscriptnumber(rstrip(rstrip(string(alpha(f)*100), '0'), '.')))"
_st_featop_abbr(f::UnivariateSoftMax, ::typeof(>); kwargs...) = "$(variable_name(f; kwargs...)) $("⪵" * utils.subscriptnumber(rstrip(rstrip(string(alpha(f)*100), '0'), '.')))"

_st_featop_abbr(f::UnivariateMin,     ::typeof(≤); kwargs...) = "$(variable_name(f; kwargs...)) ↘"
_st_featop_abbr(f::UnivariateMax,     ::typeof(≥); kwargs...) = "$(variable_name(f; kwargs...)) ↗"
_st_featop_abbr(f::UnivariateSoftMin, ::typeof(≤); kwargs...) = "$(variable_name(f; kwargs...)) $("↘" * utils.subscriptnumber(rstrip(rstrip(string(alpha(f)*100), '0'), '.')))"
_st_featop_abbr(f::UnivariateSoftMax, ::typeof(≥); kwargs...) = "$(variable_name(f; kwargs...)) $("↗" * utils.subscriptnumber(rstrip(rstrip(string(alpha(f)*100), '0'), '.')))"

############################################################################################

import SoleModels: parsefeature

using StatsBase

"""
Syntaxstring aliases for standard features, such as "min", "max", "avg".
"""
const BASE_FEATURE_ALIASES = Dict{String,Base.Callable}(
    #
    "minimum" => UnivariateMin,
    "min"     => UnivariateMin,
    "maximum" => UnivariateMax,
    "max"     => UnivariateMax,
    #
    "avg"     => StatsBase.mean,
    "mean"    => StatsBase.mean,
)

"""
    parsefeature(FT::Type{<:VarFeature}, expr::String; kwargs...)

Parse a [`VarFeature`](@ref) of type `FT` from its [`syntaxstring`](@ref) representation.

# Keyword Arguments
- `featvaltype::Union{Nothing,Type} = nothing`: the feature's featvaltype
    (recommended for some features, e.g., [`UnivariateFeature`](@ref));
- `opening_parenthesis::String = $(repr(UVF_OPENING_PARENTHESIS))`:
    the string signaling the opening of an expression block (e.g., `"min[V2]"`);
- `closing_parenthesis::String = $(repr(UVF_CLOSING_PARENTHESIS))`:
    the string signaling the closing of an expression block (e.g., `"min[V2]"`);
- `additional_feature_aliases = Dict{String,Base.Callable}()`: A dictionary mapping strings to
    callables, useful when parsing custom-made, non-standard features.
    By default, features such as "avg" or "min" are provided for
    (see `SoleModels.BASE_FEATURE_ALIASES`);
    note that, in case of clashing `string`'s,
    the provided additional aliases will override the standard ones;
- `variable_names_map::Union{Nothing,AbstractDict,AbstractVector} = nothing`:
    mapping from variable name to variable index, useful when parsing from
    `syntaxstring`'s with variable names (e.g., `"min[Heart rate]"`);
- `variable_name_prefix::String = $(repr(UVF_VARPREFIX))`:
    prefix used with variable indices (e.g., "$(UVF_VARPREFIX)10").

Note that at most one argument in `variable_names_map` and `variable_name_prefix`
should be provided.

!!! note
    The default parentheses, here, differ from those of [`SoleLogics.parseformula](@ref),
    since features are typically wrapped into `Atom`'s, and `parseformula` does not
    allow parenthesis characters in atoms' `syntaxstring`'s.

See also [`VarFeature`](@ref), [`featvaltype`](@ref), [`parsecondition`](@ref).
"""

function parsefeature(
    ::Type{FT},
    expr::String;
    featvaltype::Union{Nothing,Type} = nothing,
    opening_parenthesis::String = UVF_OPENING_PARENTHESIS,
    closing_parenthesis::String = UVF_CLOSING_PARENTHESIS,
    additional_feature_aliases = Dict{String,Base.Callable}(),
    variable_names_map::Union{Nothing,AbstractDict,AbstractVector} = nothing,
    variable_name_prefix::Union{Nothing,String} = nothing,
    kwargs...
) where {FT<:VarFeature}
    @assert isnothing(variable_names_map) || isnothing(variable_name_prefix) "" *
        "Cannot parse variable with both variable_names_map and variable_name_prefix. " *
        "(expr = $(repr(expr)))"

    @assert length(opening_parenthesis) == 1 || length(closing_parenthesis)
        "Parentheses must be single-character strings! " *
        "$(repr(opening_parenthesis)) and $(repr(closing_parenthesis)) encountered."

    featdict = merge(BASE_FEATURE_ALIASES, additional_feature_aliases)

    variable_name_prefix = isnothing(variable_name_prefix) &&
        isnothing(variable_names_map) ? UVF_VARPREFIX : variable_name_prefix
    variable_name_prefix = isnothing(variable_name_prefix) ? "" : variable_name_prefix

    r = Regex("^\\s*(\\w+)\\s*\\$(opening_parenthesis)\\s*$(variable_name_prefix)(\\S+)\\s*\\$(closing_parenthesis)\\s*\$")
    slices = match(r, expr)

    # Assert for malformed strings (e.g. "123.4<avg[V189]>250.2")
    @assert !isnothing(slices) && length(slices) == 2 "Could not parse variable " *
        "feature from expression $(repr(expr))."

    slices = string.(slices)
    (_feature, _variable) = (slices[1], slices[2])

    feature = begin
        i_var = begin
            if isnothing(variable_names_map)
                parse(Int, _variable)
            elseif variable_names_map isa Union{AbstractDict,AbstractVector}
                i_var = findfirst(variable_names_map, variable)
                @assert !isnothing(i_var) "Could not find variable $variable in the " *
                    "specified map. ($(@show variable_names_map))"
            else
                error("Unexpected variable_names_map of type $(typeof(variable_names_map)) " *
                    "encountered.")
            end
        end
        if haskey(featdict, _feature)
            # If it is a known feature get it as
            #  a type (e.g., `UnivariateMin`), or Julia function (e.g., `minimum`).
            feat_or_fun = featdict[_feature]
            # If it is a function, wrap it into a UnivariateFeature
            #  otherwise, it is a feature, and it is used as a constructor.
            if feat_or_fun isa Function
                if isnothing(featvaltype)
                    featvaltype = DEFAULT_VARFEATVALTYPE
                    @warn "Please, specify a type for the feature values (featvaltype = ...). " *
                        "$(featvaltype) will be used, but note that this may raise type errors. " *
                        "(expression = $(repr(expr)))"
                end

                UnivariateFeature{featvaltype}(i_var, feat_or_fun)
            else
                feat_or_fun(i_var) # TODO do this
                # feat_or_fun{featvaltype}(i_var)
            end
        else
            # If it is not a known feature, interpret it as a Julia function,
            #  and wrap it into a UnivariateFeature.
            f = eval(Meta.parse(_feature))
            if isnothing(featvaltype)
                featvaltype = DEFAULT_VARFEATVALTYPE
                @warn "Please, specify a type for the feature values (featvaltype = ...). " *
                    "$(featvaltype) will be used, but note that this may raise type errors. " *
                    "(expression = $(repr(expr)))"
            end

            UnivariateFeature{featvaltype}(i_var, f)
        end
    end

    # if !(feature isa FT)
    #     @warn "Could not parse expression $(repr(expr)) as feature of type $(FT); " *
    #         " $(typeof(feature)) was used."
    # end

    return feature
end
