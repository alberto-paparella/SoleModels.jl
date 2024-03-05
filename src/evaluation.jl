using SoleModels
using SoleModels: LeafModel
import SoleLogics: natoms

####################### Utility Functions #####################

function accuracy(
    y_true::AbstractArray,
    y_pred::AbstractArray,
)
    @assert length(y_true) == length(y_pred) "True labels and Predicted labels don't have the same length"
    @assert length(y_pred) > 0 "Don't pass the labels, the two vectors are empty"

    return sum(y_pred .== y_true)/length(y_pred)
end

function error(
    y_true::AbstractArray,
    y_pred::AbstractArray,
)

    return 1.0 - accuracy(y_true, y_pred)
end

function mae(
    y_true::AbstractArray,
    y_pred::AbstractArray,
)
    @assert length(y_true) == length(y_pred) "True labels and Predicted labels don't have the same length"
    @assert length(y_pred) > 0 "Don't pass the labels, the two vectors are empty"

    return sum(abs.(y_pred.-y_true)) / length(y_pred)
end

function mse(
    y_true::AbstractArray,
    y_pred::AbstractArray,
)
    @assert length(y_true) == length(y_pred) "True labels and Predicted labels don't have the same length"
    @assert length(y_pred) > 0 "Don't pass the labels, the two vectors are empty"

    return sum((y_pred.-y_true).^2) / length(y_pred)
end

#############################################################

"""
    readmetrics(m::AbstractModel; digits = 2)

Return a `NamedTuple` with some performance metrics for the given symbolic model.
Performance metrics can be computed when the `info` structure of the model has the
    following keys:
    - :supporting_labels
    - :supporting_predictions

The `digits` keyword argument is used to `round` accuracy/confidence metrics.
"""
function readmetrics(m::LeafModel{L}; digits = 2, dictmetrics::Dict{Symbol,Function} = Dict{Symbol,Function}()) where {L<:Label}
    merge(if haskey(info(m), :supporting_labels) && haskey(info(m), :supporting_predictions)
        _gts = info(m).supporting_labels
        _preds = info(m).supporting_predictions
        if L <: CLabel
            (; ninstances = length(_gts), confidence = round(accuracy(_gts, _preds); digits = digits))
        elseif L <: RLabel
            (; ninstances = length(_gts), mae = round(mae(_gts, _preds); digits = digits))
        else
            error("Could not compute readmetrics with unknown label type: $(L).")
        end
    elseif haskey(info(m), :supporting_labels)
        return (; ninstances = length(info(m).supporting_labels))
    else
        return (;)
    end, (; coverage = 1.0))
end

function readmetrics(m::Rule; digits = 2, dictmetrics::Dict{Symbol,Function} = Dict{Symbol,Function}(), kwargs...)
    if haskey(info(m), :supporting_labels) && haskey(info(consequent(m)), :supporting_labels)
        _gts = info(m).supporting_labels
        _gts_leaf = info(consequent(m)).supporting_labels
        coverage = length(_gts_leaf)/length(_gts)
        
        merge(
            readmetrics(consequent(m); digits = digits, kwargs...), 
            (; coverage = round(coverage; digits = digits)),
            if haskey(info(m), :supporting_predictions)
                _preds = info(m).supporting_predictions
                metrics = (;)

                for (key,func) in dictmetrics
                    if Base.nargs(func) == 2
                        merge(metrics, (; key => func(_gts,_preds)))
                    end
                end

                metrics
            end
        )
    elseif haskey(info(m), :supporting_labels)
        _gts = info(m).supporting_labels
        
        return merge(
            (; ninstances = length(_gts)),
            begin
                metrics = (;)

                for (key,func) in dictmetrics
                    if Base.nargs(func) == 1
                        merge(metrics, (; key => func(_gts)))
                    end
                end

                metrics
            end
        )
    elseif haskey(info(consequent(m)), :supporting_labels)
        return (; ninstances = length(info(m).supporting_labels))
    else
        return (;)
    end
end


"""
    evaluaterule(
        r::Rule{O},
        X::AbstractInterpretationSet,
        Y::AbstractVector{L}
    ) where {O,L<:Label}

Evaluate the rule on a labeled dataset, and return a `NamedTuple` consisting of:
- `antsat::Vector{Bool}`: satsfaction of the antecedent for each instance in the dataset;
- `ys::Vector{Union{Nothing,O}}`: rule prediction. For each instance in X:
    - `consequent(rule)` if the antecedent is satisfied,
    - `nothing` otherwise.

See also
[`Rule`](@ref),
[`SoleLogics.AbstractInterpretationSet`](@ref),
[`Label`](@ref),
[`checkantecedent`](@ref).
"""
function evaluaterule(
    rule::Rule,
    X::AbstractInterpretationSet,
    Y::AbstractVector{<:Label};
    kwargs...,
)
    #println("Evaluation rule in time...")
    ys = apply(rule,X)
    #if X isa SupportedLogiset
    #    println("# Memoized Values: $(nmemoizedvalues(X))")
    #end

    antsat = ys .!= nothing

    #=
    cons_sat = begin
        idxs_sat = findall(antsat .== true)
        cons_sat = Vector{Union{Bool,Nothing}}(fill(nothing, length(Y)))

        idxs_true = begin
            idx_cons = findall(outcome(consequent(rule)) .== Y)
            intersect(idxs_sat, idx_cons)
        end
        idxs_false = begin
            idx_cons = findall(outcome(consequent(rule)) .!= Y)
            intersect(idxs_sat, idx_cons)
        end
        cons_sat[idxs_true]  .= true
        cons_sat[idxs_false] .= false
        cons_sat
    end
    =#

    # - `cons_sat::Vector{Union{Nothing,Bool}}`: for each instance in the dataset:
    #     - `nothing` if antecedent is not satisfied.
    #     - `false` if the antecedent is satisfied, but the consequent does not match the ground-truth label,
    #     - `true` if the antecedent is satisfied, and the consequent matches the ground-truth label,

    return (;
        ys = ys,
        antsat     = antsat,
        # cons_sat    = cons_sat,
    )
end

# """
#     natoms(rule::Rule{O,<:Formula}) where {O}

# See also
# [`Rule`](@ref),
# [`SoleLogics.Formula`](@ref),
# [`antecedent`](@ref),
# """
# natoms(rule::Rule{O,<:Formula}) where {O} = natoms(antecedent(rule))

"""
    rulemetrics(
        r::Rule,
        X::AbstractInterpretationSet,
        Y::AbstractVector{<:Label}
    )

Compute metrics for a rule with respect to a labeled dataset and returns a `NamedTuple` consisting of:
- `support`: number of instances satisfying the antecedent of the rule divided by
    the total number of instances;
- `error`:
    - For classification problems: number of instances that were not classified
    correctly divided by the total number of instances;
    - For regression problems: mean squared error;
- `length`: number of atoms in the rule's antecedent.

See also
[`Rule`](@ref),
[`SoleLogics.AbstractInterpretationSet`](@ref),
[`Label`](@ref),
[`evaluaterule`](@ref),
[`outcometype`](@ref),
[`consequent`](@ref).
"""
function rulemetrics(
    rule::Rule,
    X::AbstractInterpretationSet,
    Y::AbstractVector{<:Label};
    kwargs...,
)
    eval_result = evaluaterule(rule, X, Y; kwargs...)
    ys = eval_result[:ys]
    antsat = eval_result[:antsat]
    n_instances = ninstances(X)
    n_satisfy = sum(antsat)

    rule_support =  n_satisfy / n_instances

    rule_error = begin
        if outcometype(consequent(rule)) <: CLabel
            # Number of incorrectly classified instances divided by number of instances
            # satisfying the rule condition.
            error(ys[antsat],Y[antsat])
        elseif outcometype(consequent(rule)) <: RLabel
            # Mean Squared Error (mse)
            mse(ys[antsat],Y[antsat])
        else
            error("The outcome type of the consequent of the input rule $(outcometype(consequent(rule))) is not among those accepted")
        end
    end

    return (;
        antsat    = antsat,
        support   = rule_support,
        error     = rule_error,
        length    = natoms(antecedent(rule)),
    )
end

############################################################################################
############################################################################################
############################################################################################

function metrics(
    rule::Rule,
    X::Unione{Nothing,AbstractInterpretationSet},
    Y::Unione{Nothing,AbstractVector{<:Label}};
    return_model::Bool = false,
    dictmetrics::Dict{Symbol,Function} = Dict{Symbol,Function}(),
    kwargs...,
)
    info_rule = info(rule)

    supporting_labels = begin
        if !isnothing(Y)
            Y 
        elseif haskey(info_rule,:supporting_labels)
            info_rule.supporting_labels
        else 
            nothing
        end
    end

    supporting_predictions = begin
        if !isnothing(X)
            apply(rule,X)
        elseif haskey(info_rule,:supporting_predictions)
            info_rule.supporting_predictions
        else 
            nothing
        end
    end

    new_inforule = begin
        if !isnothing(supporting_labels) && !isnothing(supporting_predictions)
            (info_rule..., supporting_labels = supporting_labels, supporting_predictions = supporting_predictions)
        elseif !isnothing(supporting_labels)
            (info_rule..., supporting_labels = supporting_labels)
        elseif !isnothing(supporting_predictions)
            (info_rule..., supporting_predictions = supporting_predictions)
        else
            info_rule
        end
    end

    newrule = Rule(antecedent(rule),consequent(rule),new_inforule)
    readmetrics(newrule; dictmetrics=dictmetrics)
end
