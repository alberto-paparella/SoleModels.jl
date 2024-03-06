module SoleModels

using Reexport

using SoleBase

@reexport using SoleData

using SoleData: AbstractLogiset, ismultilogiseed

@reexport using SoleLogics
using SoleLogics
using SoleLogics: AbstractInterpretation, AbstractInterpretationSet
using SoleLogics: SyntaxToken
using SoleLogics: Formula, synstruct
using SoleLogics: ⊤, ¬, ∧

using FunctionWrappers: FunctionWrapper
using StatsBase
using ThreadSafeDicts
using Lazy

using SoleData: load_arff_dataset

############################################################################################
############################################################################################
############################################################################################

using SoleData.DimensionalDatasets: OneWorld, Interval, Interval2D
using SoleData.DimensionalDatasets: IARelations
using SoleData.DimensionalDatasets: IA2DRelations
using SoleData.DimensionalDatasets: identityrel
using SoleData.DimensionalDatasets: globalrel

############################################################################################
############################################################################################
############################################################################################

export outcometype, outputtype

export Rule, Branch
export checkantecedent
export antecedent, consequent
export posconsequent, negconsequent

export DecisionList
export rulebase, defaultconsequent

export apply, apply!

export DecisionTree
export root

export MixedSymbolicModel, DecisionForest

include("base.jl")

export printmodel, displaymodel

include("print.jl")

export immediatesubmodels, listimmediaterules
export listrules, joinrules

include("symbolic-utils.jl")

export AssociationRule, ClassificationRule, RegressionRule

include("machine-learning.jl")

export rulemetrics, readmetrics, metrics

include("evaluation.jl")

include("experimentals.jl")

end
