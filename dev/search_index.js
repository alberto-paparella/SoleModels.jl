var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SoleModels","category":"page"},{"location":"#SoleModels","page":"Home","title":"SoleModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SoleModels.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SoleModels]","category":"page"},{"location":"#SoleModels.CLabel","page":"Home","title":"SoleModels.CLabel","text":"const CLabel  = Union{String,Integer}\nconst RLabel  = AbstractFloat\nconst Label   = Union{CLabel,RLabel}\n\nTypes for supervised machine learning labels (classification and regression).\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.Label","page":"Home","title":"SoleModels.Label","text":"const CLabel  = Union{String,Integer}\nconst RLabel  = AbstractFloat\nconst Label   = Union{CLabel,RLabel}\n\nTypes for supervised machine learning labels (classification and regression).\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.AbstractActiveConditionalDataset","page":"Home","title":"SoleModels.AbstractActiveConditionalDataset","text":"abstract type AbstractActiveConditionalDataset{\n    W<:AbstractWorld,\n    A<:AbstractCondition,\n    T<:TruthValue,\n    FR<:AbstractFrame{W,T},\n} <: AbstractConditionalDataset{W,A,T,FR} end\n\nAbstract type for active conditional datasets, that is, conditional datasets that can be used in machine learning algorithms (e.g., they have an alphabet, can enumerate propositions and learn formulas from).\n\nSee also AbstractConditionalDataset, AbstractCondition.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.AbstractBooleanCondition","page":"Home","title":"SoleModels.AbstractBooleanCondition","text":"abstract type AbstractBooleanCondition end\n\nA boolean condition is a condition that evaluates to a boolean truth value (true/false), when checked on a logical interpretation.\n\nSee also TrueCondition, LogicalTruthCondition, check, syntaxstring.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.AbstractCondition","page":"Home","title":"SoleModels.AbstractCondition","text":"abstract type AbstractCondition end\n\nAbstract type for representing conditions that can be interpreted and evaluated on worlds of instances of a conditional dataset. In logical contexts, these are wrapped into Propositions.\n\nSee also Proposition, syntaxstring, FeatMetaCondition, FeatCondition.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.AbstractConditionalAlphabet","page":"Home","title":"SoleModels.AbstractConditionalAlphabet","text":"abstract type AbstractConditionalAlphabet{C<:FeatCondition} <: AbstractAlphabet{C} end\n\nAbstract type for alphabets of conditions.\n\nSee also FeatCondition, FeatMetaCondition, AbstractAlphabet.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.AbstractConditionalDataset","page":"Home","title":"SoleModels.AbstractConditionalDataset","text":"abstract type AbstractConditionalDataset{\n    W<:AbstractWorld,\n    A<:AbstractCondition,\n    T<:TruthValue,\n    FR<:AbstractFrame{W,T},\n} <: AbstractInterpretationSet{AbstractKripkeStructure{W,A,T,FR}} end\n\nAbstract type for conditional datasets, that is, symbolic learning datasets where each instance is a Kripke model where conditions (see AbstractCondition), and logical formulas with conditional letters can be checked on worlds.\n\nSee also AbstractInterpretationSet, AbstractCondition.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.AbstractFeature","page":"Home","title":"SoleModels.AbstractFeature","text":"abstract type AbstractFeature{U<:Real} end\n\nAbstract type for features, representing a scalar functions that can be computed on a world.\n\nSee also featvaltype, computefeature, AbstractWorld.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.AbstractLogicalBooleanCondition","page":"Home","title":"SoleModels.AbstractLogicalBooleanCondition","text":"abstract type AbstractLogicalBooleanCondition <: AbstractBooleanCondition end\n\nA boolean condition based on a formula of a given logic, that is to be checked on a logical interpretation.\n\nSee also formula, syntaxstring, check, AbstractBooleanCondition.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.AbstractModel","page":"Home","title":"SoleModels.AbstractModel","text":"abstract type AbstractModel{O} end\n\nAbstract type for mathematical models that, given an instance object (i.e., a piece of data), output an outcome of type O.\n\nSee also Rule, Branch, isopen, apply, issymbolic, info, outcometype.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.Branch","page":"Home","title":"SoleModels.Branch","text":"struct Branch{\n    O,\n    C<:AbstractBooleanCondition,\n    FM<:AbstractModel\n} <: ConstrainedModel{O,FM}\n    antecedent::C\n    posconsequent::FM\n    negconsequent::FM\n    info::NamedTuple\nend\n\nA Branch is one of the fundamental building blocks of symbolic modeling, and has the semantics:\n\nIF (antecedent) THEN (consequent_1) ELSE (consequent_2) END\n\nwhere the antecedent is boolean condition to be tested and the consequents are the feasible local outcomes of the block.\n\nNote that FM refers to the Feasible Models (FM) allowed in the model's sub-tree.\n\nSee also antecedent, posconsequent, negconsequent, AbstractBooleanCondition, Rule, ConstrainedModel, AbstractModel.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.ConstantModel","page":"Home","title":"SoleModels.ConstantModel","text":"struct ConstantModel{O} <: FinalModel{O}\n    outcome::O\n    info::NamedTuple\nend\n\nThe simplest type of model is the ConstantModel; it is a FinalModel that always outputs the same outcome.\n\nExamples\n\njulia> SoleModels.FinalModel(2) isa SoleModels.ConstantModel\n\njulia> SoleModels.FinalModel(sum) isa SoleModels.FunctionModel\n┌ Warning: Over efficiency concerns, please consider wrappingJulia Function's into FunctionWrapper{O,Tuple{SoleModels.AbstractInterpretation}} structures,where O is their return type.\n└ @ SoleModels ~/.julia/dev/SoleModels/src/models/base.jl:337\ntrue\n\n\nSee also apply, FunctionModel, FinalModel.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.ConstrainedModel","page":"Home","title":"SoleModels.ConstrainedModel","text":"An AbstractModel can wrap another AbstractModel, and use it to compute the outcome. As such, an AbstractModel can actually be the result of a composition of many models, and enclose a tree of AbstractModels (with FinalModels at the leaves). In order to typebound the Feasible Models (FM) allowed in the sub-tree, the ConstrainedModel type is introduced:\n\nabstract type ConstrainedModel{O,FM<:AbstractModel} <: AbstractModel{O} end\n\nFor example, ConstrainedModel{String, Union{Branch{String}, ConstantModel{String}}} supertypes models that with String outcomes that make use of Branch{String} and ConstantModel{String} (essentially, a decision trees with Strings at the leaves).\n\nSee also FinalModel, AbstractModel.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.DecisionForest","page":"Home","title":"SoleModels.DecisionForest","text":"A Decision Forest is a symbolic model that wraps an ensemble of models\n\nstruct DecisionForest{\n    O,\n    C<:AbstractBooleanCondition,\n    FFM<:FinalModel\n} <: ConstrainedModel{O, Union{<:Branch{<:O,<:C}, <:FFM}}\n    trees::Vector{<:DecisionTree}\n    info::NamedTuple\nend\n\nSee also ConstrainedModel, MixedSymbolicModel, DecisionList, DecisionTree\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.DecisionList","page":"Home","title":"SoleModels.DecisionList","text":"struct DecisionList{\n    O,\n    C<:AbstractBooleanCondition,\n    FM<:AbstractModel\n} <: ConstrainedModel{O,FM}\n    rulebase::Vector{Rule{_O,_C,_FM} where {_O<:O,_C<:C,_FM<:FM}}\n    defaultconsequent::FM\n    info::NamedTuple\nend\n\nA DecisionList (or decision table, or rule-based model) is a symbolic model that has the semantics of an IF-ELSEIF-ELSE block:\n\nIF (antecedent_1)     THEN (consequent_1)\nELSEIF (antecedent_2) THEN (consequent_2)\n...\nELSEIF (antecedent_n) THEN (consequent_n)\nELSE (consequent_default) END\n\nwhere the antecedents are conditions to be tested and the consequents are the feasible local outcomes of the block. Using the classical semantics, the antecedents are evaluated in order, and a consequent is returned as soon as a valid antecedent is found, or when the computation reaches the ELSE clause.\n\nNote that FM refers to the Feasible Models (FM) allowed in the model's sub-tree.\n\nSee also Rule, ConstrainedModel, DecisionTree, AbstractModel.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.DecisionTree","page":"Home","title":"SoleModels.DecisionTree","text":"A DecisionTree is a symbolic model that operates as a nested structure of IF-THEN-ELSE blocks:\n\nIF (antecedent_1) THEN\n    IF (antecedent_2) THEN\n        (consequent_1)\n    ELSE\n        (consequent_2)\n    END\nELSE\n    IF (antecedent_3) THEN\n        (consequent_3)\n    ELSE\n        (consequent_4)\n    END\nEND\n\nwhere the antecedents are conditions to be tested and the consequents are the feasible local outcomes of the block.\n\nIn practice, a DecisionTree simply wraps a constrained sub-tree of Branch and FinalModel:\n\nstruct DecisionTree{\nO,\n    C<:AbstractBooleanCondition,\n    FFM<:FinalModel\n} <: ConstrainedModel{O, Union{<:Branch{<:O,<:C}, <:FFM}}\n    root::M where {M<:Union{FFM,Branch}}\n    info::NamedTuple\nend\n\nNote that FM refers to the Feasible Models (FM) allowed in the model's sub-tree. Also note that this structure also includes an info::NamedTuple for storing additional information.\n\nSee also ConstrainedModel, MixedSymbolicModel, DecisionList.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.ExternalFWDFeature","page":"Home","title":"SoleModels.ExternalFWDFeature","text":"struct ExternalFWDFeature{U} <: AbstractFeature{U}\n    name::String\n    fwd::Any\nend\n\nA feature encoded explicitly as (a slice of) an FWD structure (see AbstractFWD).\n\nSee also AbstractFWD, AbstractFeature.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.FeatCondition","page":"Home","title":"SoleModels.FeatCondition","text":"struct FeatCondition{U,M<:FeatMetaCondition} <: AbstractCondition\n    metacond::M\n    a::U\nend\n\nA scalar condition comparing a computed feature value (see FeatMetaCondition) and a threshold value a. It can be evaluated on a world of an instance of a conditional dataset.\n\nExample: min(V1)  10, which translates to \"Within this world, the minimum of variable 1 is greater or equal than 10.\"\n\nSee also AbstractCondition, negation, FeatMetaCondition.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.FeatMetaCondition","page":"Home","title":"SoleModels.FeatMetaCondition","text":"struct FeatMetaCondition{F<:AbstractFeature,O<:TestOperator} <: AbstractCondition\n    feature::F\n    test_operator::O\nend\n\nA metacondition representing a scalar comparison method. A feature is a scalar function that can be computed on a world of an instance of a conditional dataset. A test operator is a binary mathematical relation, comparing the computed feature value and an external threshold value (see FeatCondition). A metacondition can also be used for representing the infinite set of conditions that arise with a free threshold (see UnboundedExplicitConditionalAlphabet): min(V1)  a a  ℝ.\n\nSee also AbstractCondition, negation, FeatCondition.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.FinalModel","page":"Home","title":"SoleModels.FinalModel","text":"abstract type FinalModel{O} <: AbstractModel{O} end\n\nA FinalModel is a model which outcomes do not depend on another model. An AbstractModel can generally wrap other AbstractModels. In such case, the outcome can depend on the inner models being applied on the instance object. Otherwise, the model is considered final; that is, it is a leaf of a tree of AbstractModels.\n\nSee also ConstantModel, FunctionModel, AbstractModel.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.FunctionModel","page":"Home","title":"SoleModels.FunctionModel","text":"struct FunctionModel{O} <: FinalModel{O}\n    f::FunctionWrapper{O}\n    info::NamedTuple\nend\n\nA FunctionModel is a FinalModel that applies a native Julia Function in order to compute the outcome. Over efficiency concerns, it is mandatory to make explicit the output type O by wrapping the Function into an object of type FunctionWrapper{O}.\n\nSee also ConstantModel, FunctionWrapper, FinalModel.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.LogicalTruthCondition","page":"Home","title":"SoleModels.LogicalTruthCondition","text":"struct LogicalTruthCondition{F<:AbstractFormula} <: AbstractLogicalBooleanCondition\n    formula::F\nend\n\nA boolean condition that, on a given logical interpretation, a logical formula evaluates to the top of the logic's algebra.\n\nSee also formula, AbstractLogicalBooleanCondition.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.MixedSymbolicModel","page":"Home","title":"SoleModels.MixedSymbolicModel","text":"A MixedSymbolicModel is a symbolic model that operaters as a free nested structure of IF-THEN-ELSE and IF-ELSEIF-ELSE blocks:\n\nIF (antecedent_1) THEN\n    IF (antecedent_1)     THEN (consequent_1)\n    ELSEIF (antecedent_2) THEN (consequent_2)\n    ELSE (consequent_1_default) END\nELSE\n    IF (antecedent_3) THEN\n        (consequent_3)\n    ELSE\n        (consequent_4)\n    END\nEND\n\nwhere the antecedents are conditinos and the consequents are the feasible local outcomes of the block.\n\nIn Sole.jl, this logic can implemented using ConstrainedModels such as Rules, Branchs, DecisionLists, DecisionTrees, and the be wrapped into a MixedSymbolicModel:\n\nstruct MixedSymbolicModel{O,FM<:AbstractModel} <: ConstrainedModel{O,FM}\n    root::M where {M<:Union{FinalModel{<:O},ConstrainedModel{<:O,<:FM}}}\n    info::NamedTuple\nend\n\nNote that FM refers to the Feasible Models (FM) allowed in the model's sub-tree.\n\nSee also ConstrainedModel, DecisionTree, DecisionList.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.MultiFrameConditionalDataset","page":"Home","title":"SoleModels.MultiFrameConditionalDataset","text":"struct MultiFrameConditionalDataset{MD<:AbstractConditionalDataset}\n    modalities  :: Vector{<:MD}\nend\n\nA multi-frame conditional dataset. This structure is useful for representing multimodal datasets in logical terms.\n\nSee also AbstractConditionalDataset, minify.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.NamedFeature","page":"Home","title":"SoleModels.NamedFeature","text":"struct NamedFeature{U} <: AbstractFeature{U}\n    name::String\nend\n\nA feature solely identified by its name.\n\nSee also AbstractFeature.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.RLabel","page":"Home","title":"SoleModels.RLabel","text":"const CLabel  = Union{String,Integer}\nconst RLabel  = AbstractFloat\nconst Label   = Union{CLabel,RLabel}\n\nTypes for supervised machine learning labels (classification and regression).\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.Rule","page":"Home","title":"SoleModels.Rule","text":"struct Rule{\n    O,\n    C<:AbstractBooleanCondition,\n    FM<:AbstractModel\n} <: ConstrainedModel{O,FM}\n    antecedent::C\n    consequent::FM\n    info::NamedTuple\nend\n\nA Rule is one of the fundamental building blocks of symbolic modeling, and has the semantics:\n\nIF (antecedent) THEN (consequent) END\n\nwhere the antecedent is a condition to be tested and the consequent is the local outcome of the block.\n\nNote that FM refers to the Feasible Models (FM) allowed in the model's sub-tree.\n\nSee also antecedent, consequent, AbstractBooleanCondition, ConstrainedModel, AbstractModel.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.TestOperator","page":"Home","title":"SoleModels.TestOperator","text":"const TestOperator = Function\n\nA test operator is a binary Julia Function used for comparing a feature value and a threshold. In a crisp (i.e., boolean, non-fuzzy) setting, the test operator returns a boolean value, and <, >, ≥, ≤, !=, and == are typically used.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.TrueCondition","page":"Home","title":"SoleModels.TrueCondition","text":"struct TrueCondition <: AbstractLogicalBooleanCondition end\n\nA true condition is the boolean condition that always yields true.\n\nSee also LogicalTruthCondition, AbstractLogicalBooleanCondition.\n\n\n\n\n\n","category":"type"},{"location":"#SoleModels.UnboundedExplicitConditionalAlphabet","page":"Home","title":"SoleModels.UnboundedExplicitConditionalAlphabet","text":"struct UnboundedExplicitConditionalAlphabet{C<:FeatCondition} <: AbstractConditionalAlphabet{C}\n    metaconditions::Vector{<:FeatMetaCondition}\nend\n\nAn infinite alphabet of conditions induced from a finite set of metaconditions. For example, if metaconditions = [FeatMetaCondition(UnivariateMin(1), ≥)], the alphabet represents the (infinite) set: min(V1)  a a  ℝ.\n\nSee also BoundedExplicitConditionalAlphabet, FeatCondition, FeatMetaCondition, AbstractAlphabet.\n\n\n\n\n\n","category":"type"},{"location":"#Base.isopen-Tuple{SoleModels.AbstractModel}","page":"Home","title":"Base.isopen","text":"isopen(::AbstractModel)::Bool\n\nReturn whether a model is open. An AbstractModel{O} is closed if it is always able to provide an outcome of type O. Otherwise, the model can output nothing values and is referred to as open.\n\nRule is an example of an open model, while Branch is an example of closed model.\n\nSee also AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Tuple{Random.AbstractRNG, SoleModels.BoundedExplicitConditionalAlphabet}","page":"Home","title":"Base.rand","text":"function Base.rand(\n    rng::AbstractRNG,\n    a::BoundedExplicitConditionalAlphabet;\n    metaconditions::Union{Nothing,FeatMetaCondition,AbstractVector{<:FeatMetaCondition}} = nothing,\n    feature::Union{Nothing,AbstractFeature,AbstractVector{<:AbstractFeature}} = nothing,\n    test_operator::Union{Nothing,TestOperator,AbstractVector{<:TestOperator}} = nothing,\n)::Proposition\n\nRandomly sample a Proposition holding a FeatCondition from conditional alphabet a, such that:\n\nif metaconditions are specified, then the set of metaconditions (feature-operator pairs)\n\nis limited to metaconditions;\n\nif feature is specified, then the set of metaconditions (feature-operator pairs)\n\nis limited to those with feature;\n\nif test_operator is specified, then the set of metaconditions (feature-operator pairs)\n\nis limited to those with test_operator.\n\nSee also BoundedExplicitConditionalAlphabet, FeatCondition, FeatMetaCondition, `AbstractAlphabet'.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.antecedent-Tuple{Rule}","page":"Home","title":"SoleModels.antecedent","text":"antecedent(m::Union{Rule,Branch})::AbstractBooleanCondition\n\nReturn the antecedent of a rule/branch; that is, the condition to be evaluated upon applying the model.\n\nSee also apply, consequent, check_antecedent, Rule, Branch.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.apply-Tuple{SoleModels.AbstractModel, SoleLogics.AbstractInterpretation}","page":"Home","title":"SoleModels.apply","text":"apply(\n    m::AbstractModel,\n    i::AbstractInterpretation;\n    check_args::Tuple = (),\n    check_kwargs::NamedTuple = (;),\n    functional_args::Tuple = (),\n    functional_kwargs::NamedTuple = (;),\n    kwargs...\n)::outputtype(m)\n\napply(\n    m::AbstractModel,\n    d::AbstractInterpretationSet;\n    check_args::Tuple = (),\n    check_kwargs::NamedTuple = (; use_memo = [ThreadSafeDict{SyntaxTree,WorldSet{worldtype(d)}}() for i in 1:ninstances(d)]),\n    functional_args::Tuple = (),\n    functional_kwargs::NamedTuple = (;),\n    kwargs...\n)::AbstractVector{<:outputtype(m)}\n\nReturn the output prediction of the model on an instance, or on each instance of a dataset. The predictions can be nothing if the model is open.\n\ncheck_args and check_kwargs can influence check's behavior at the time of its computation (see check)\n\nfunctional_args and functional_kwargs can influence FunctionModel's behavior when the corresponding function is applied to AbstractInterpretation (see FunctionModel, AbstractInterpretation)\n\nSee also isopen, outcometype, outputtype, AbstractModel, AbstractInterpretation, AbstractInterpretationSet.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.apply_test_operator-Union{Tuple{T}, Tuple{Function, T, T}} where T","page":"Home","title":"SoleModels.apply_test_operator","text":"Apply a test operator by simply passing the feature value and threshold to the (binary) test operator function.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.balanced_weights-Union{Tuple{AbstractVector{L}}, Tuple{L}} where L<:Union{Integer, String}","page":"Home","title":"SoleModels.balanced_weights","text":"default_weights(Y::AbstractVector{L}) where {L<:CLabel}::AbstractVector{<:Number}\n\nReturn a class-rebalancing weight vector, given a label vector Y.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.bestguess","page":"Home","title":"SoleModels.bestguess","text":"bestguess(\n    labels::AbstractVector{<:Label},\n    weights::Union{Nothing,AbstractVector} = nothing;\n    suppress_parity_warning = false,\n)\n\nReturn the best guess for a set of labels; that is, the label that best approximates the labels provided. For classification labels, this function returns the majority class; for regression labels, the average value. If no labels are provided, nothing is returned. The computation can be weighted.\n\nSee also CLabel, RLabel, Label.\n\n\n\n\n\n","category":"function"},{"location":"#SoleModels.check_antecedent-Tuple{Rule, Vararg{Any}}","page":"Home","title":"SoleModels.check_antecedent","text":"function check_antecedent(\n    m::Union{Rule,Branch},\n    args...;\n    kwargs...\n)\n    check(antecedent(m), id, args...; kwargs...)\nend\n\nSimply checks the antecedent of a rule on an instance or dataset.\n\nSee also antecedent, Rule, Branch.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.check_model_constraints","page":"Home","title":"SoleModels.check_model_constraints","text":"This function is used when constructing ConstrainedModels to check that the inner models satisfy the desired type constraints.\n\nSee also ConstrainedModel, Rule, Branch.\n\n\n\n\n\n","category":"function"},{"location":"#SoleModels.computefeature-Union{Tuple{U}, Tuple{AbstractFeature{U}, Any}} where U","page":"Home","title":"SoleModels.computefeature","text":"computefeature(f::AbstractFeature{U}, channel; kwargs...)::U where {U}\n\nCompute a feature on a channel of an instance.\n\nSee also AbstractFeature.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.consequent-Tuple{Rule}","page":"Home","title":"SoleModels.consequent","text":"consequent(m::Rule)::AbstractModel\n\nReturn the consequent of a rule.\n\nSee also antecedent, Rule.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.default_weights-Tuple{Integer}","page":"Home","title":"SoleModels.default_weights","text":"default_weights(n::Integer)::AbstractVector{<:Number}\n\nReturn a default weight vector of n values.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.displaymodel-Tuple{SoleModels.AbstractModel}","page":"Home","title":"SoleModels.displaymodel","text":"printmodel(io::IO, m::AbstractModel; kwargs...)\ndisplaymodel(m::AbstractModel; kwargs...)\n\nprints or returns a string representation of model m.\n\nArguments\n\nheader::Bool = true: when set to true, a header is printed, displaying\n\nthe info structure for m;\n\nshow_subtree_info::Bool = false: when set to true, the header is printed for\n\nmodels in the sub-tree of m;\n\nmax_depth::Union{Nothing,Int} = nothing: when it is an Int, models in the sub-tree\n\nwith a depth higher than max_depth are ellipsed with \"...\";\n\nsyntaxstring_kwargs::NamedTuple = (;): kwargs to be passed to syntaxstring for\n\nformatting logical formulas.\n\nSee also SoleLogics.syntaxstring, AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.evaluaterule-Union{Tuple{FM}, Tuple{C}, Tuple{O}, Tuple{Rule{O, C, FM}, SoleLogics.AbstractInterpretationSet, AbstractVector{<:Union{AbstractFloat, Integer, String}}}} where {O, C, FM<:SoleModels.AbstractModel}","page":"Home","title":"SoleModels.evaluaterule","text":"evaluaterule(\n    r::Rule{O},\n    X::AbstractInterpretationSet,\n    Y::AbstractVector{L}\n) where {O,L<:Label}\n\nEvaluate the rule on a labelled dataset, and return a NamedTuple consisting of:\n\nantsat::Vector{Bool}: satsfaction of the antecedent for each instance in the dataset;\nys::Vector{Union{Nothing,O}}: rule prediction. For each instance in X:\nconsequent(rule) if the antecedent is satisfied,\nnothing otherwise.\n\nSee also Rule, AbstractInterpretationSet, Label, check_antecedent.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.feasiblemodelstype-Union{Tuple{Type{M}}, Tuple{M}, Tuple{O}} where {O, M<:SoleModels.AbstractModel{O}}","page":"Home","title":"SoleModels.feasiblemodelstype","text":"feasiblemodelstype(m::AbstractModel)\n\nReturn a Union of the Feasible Models (FM) allowed in the sub-tree of any AbstractModel. Note that for a ConstrainedModel{O,FM<:AbstractModel}, it simply returns FM.\n\nSee also ConstrainedModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.featvaltype-Union{Tuple{Type{<:AbstractFeature{U}}}, Tuple{U}} where U","page":"Home","title":"SoleModels.featvaltype","text":"featvaltype(::Type{<:AbstractFeature{U}}) where {U} = U\nfeatvaltype(::AbstractFeature{U}) where {U} = U\n\nReturn the type returned by the feature.\n\nSee also AbstractWorld.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.formula-Tuple{SoleModels.AbstractLogicalBooleanCondition}","page":"Home","title":"SoleModels.formula","text":"formula(c::AbstractLogicalBooleanCondition)::AbstractFormula\n\nReturn the logical formula (see SoleLogics package) of a given logical boolean condition.\n\nSee also syntaxstring, AbstractLogicalBooleanCondition.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.immediatesubmodels-Union{Tuple{SoleModels.AbstractModel{O}}, Tuple{O}} where O","page":"Home","title":"SoleModels.immediatesubmodels","text":"immediatesubmodels(m::AbstractModel)\n\nReturn the list of immediate child models. Note: if the model is final, then the returned list will be empty.\n\nExamples\n\njulia> using SoleLogics\n\njulia> branch = Branch(SoleLogics.parseformula(\"p∧q∨r\"), \"YES\", \"NO\");\n\njulia> immediatesubmodels(branch)\n2-element Vector{SoleModels.ConstantModel{String}}:\n SoleModels.ConstantModel{String}\nYES\n\n SoleModels.ConstantModel{String}\nNO\n\njulia> branch2 = Branch(SoleLogics.parseformula(\"s→p\"), branch, 42);\n\n\njulia> printmodel.(immediatesubmodels(branch2));\nBranch\n┐ p ∧ (q ∨ r)\n├ ✔ YES\n└ ✘ NO\n\nConstantModel\n42\n\nSee also submodels, FinalModel, AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.info-Tuple{SoleModels.AbstractModel}","page":"Home","title":"SoleModels.info","text":"info(m::AbstractModel)::NamedTuple = m.info\n\nReturn the info structure for model m; this structure is used for storing additional information that does not affect the model's behavior. This structure can hold, for example, information about the model's statistical performance during the learning phase.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.isminifiable-Tuple{Any}","page":"Home","title":"SoleModels.isminifiable","text":"isminifiable(::Any)::Bool\n\nReturn whether minification can be applied on a dataset structure. See also minify.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.issymbolic-Tuple{SoleModels.AbstractModel}","page":"Home","title":"SoleModels.issymbolic","text":"issymbolic(::AbstractModel)::Bool\n\nReturn whether a model is symbolic or not. A model is said to be symbolic when its functioning simply relies on checking conditions (e.g., based on formulas of a certain logical language, see SoleLogics package) on the instance. Essentially, symbolic models have a rule-based structure, and provide a form of transparent and interpretable computation.\n\nExamples of purely symbolic models are Rules, Branch, DecisionLists and DecisionTrees. Examples of non-symbolic models are those encoding algebraic mathematical functions (e.g., a neural networks). Note that DecisionForests are not purely symbolic.\n\nSee also apply, listrules, AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.listimmediaterules-Tuple{SoleModels.AbstractModel}","page":"Home","title":"SoleModels.listimmediaterules","text":"listimmediaterules(m::AbstractModel{O} where {O})::Rule{<:O}\n\nList the immediate rules equivalent to a symbolic model.\n\nSee also listrules, issymbolic, AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.listrules-Tuple{SoleModels.AbstractModel}","page":"Home","title":"SoleModels.listrules","text":"listrules(m::AbstractModel; force_syntaxtree::Bool = false)::Vector{<:Rule}\n\nReturn a list of rules capturing the knowledge enclosed in symbolic model. The behavior of a symbolic model can be extracted and represented as a set of mutually exclusive (and jointly exaustive, if the model is closed) rules, which can be useful for many purposes.\n\nThe keyword argument force_syntaxtree, when set to true, causes the logical antecedents in the returned rules to be represented as SyntaxTrees, as opposed to other syntax structure (e.g., LeftmostConjunctiveForm).\n\nSee also listimmediaterules, issymbolic, FinalModel, AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.minify-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T<:Union{Missing, Nothing, Real}","page":"Home","title":"SoleModels.minify","text":"minify(dataset::D1)::Tuple{D2,Function} where {D1,D2}\n\nReturn a minified version of a dataset, as well as a backmap for reverting to the original dataset. Dataset minification remaps each scalar values in the dataset to a new value such that the overall order of the values is preserved; the output dataset is smaller in size, since it relies on values of type UInt8, UInt16, UInt32, etc.\n\nSee also isminifiable.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.negconsequent-Tuple{Branch}","page":"Home","title":"SoleModels.negconsequent","text":"negconsequent(m::Branch)::AbstractModel\n\nReturn the negative consequent of a branch; that is, the model to be applied if the antecedent evaluates to false.\n\nSee also antecedent, Branch.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.outcometype-Union{Tuple{Type{<:SoleModels.AbstractModel{O}}}, Tuple{O}} where O","page":"Home","title":"SoleModels.outcometype","text":"outcometype(::Type{<:AbstractModel{O}}) where {O} = O\noutcometype(m::AbstractModel) = outcometype(typeof(m))\n\nReturn the outcome type of a model (type).\n\nSee also AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.outputtype-Tuple{SoleModels.AbstractModel}","page":"Home","title":"SoleModels.outputtype","text":"outputtype(m::AbstractModel)\n\nReturn a supertype for the outputs obtained when applying a model. The result depends on whether the model is open or closed:\n\noutputtype(M::AbstractModel{O}) = isopen(M) ? Union{Nothing,O} : O\n\nNote that if the model is closed, then outputtype(m) is equal to outcometype(m).\n\nSee also isopen, apply, outcometype, AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.posconsequent-Tuple{Branch}","page":"Home","title":"SoleModels.posconsequent","text":"posconsequent(m::Branch)::AbstractModel\n\nReturn the positive consequent of a branch; that is, the model to be applied if the antecedent evaluates to true.\n\nSee also antecedent, Branch.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.printmodel-Tuple{IO, SoleModels.AbstractModel}","page":"Home","title":"SoleModels.printmodel","text":"printmodel(io::IO, m::AbstractModel; kwargs...)\ndisplaymodel(m::AbstractModel; kwargs...)\n\nprints or returns a string representation of model m.\n\nArguments\n\nheader::Bool = true: when set to true, a header is printed, displaying\n\nthe info structure for m;\n\nshow_subtree_info::Bool = false: when set to true, the header is printed for\n\nmodels in the sub-tree of m;\n\nmax_depth::Union{Nothing,Int} = nothing: when it is an Int, models in the sub-tree\n\nwith a depth higher than max_depth are ellipsed with \"...\";\n\nsyntaxstring_kwargs::NamedTuple = (;): kwargs to be passed to syntaxstring for\n\nformatting logical formulas.\n\nSee also SoleLogics.syntaxstring, AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.representatives-Union{Tuple{W}, Tuple{SoleLogics.AbstractMultiModalFrame{W}, W, SoleLogics.AbstractRelation, SoleModels.FeatMetaCondition}} where W<:SoleLogics.AbstractWorld","page":"Home","title":"SoleModels.representatives","text":"function representatives(\n    fr::AbstractMultiModalFrame{W},\n    S::W,\n    ::AbstractRelation,\n    ::FeatMetaCondition\n) where {W<:AbstractWorld}\n\nReturn an iterator to the (few) representative accessible worlds that are really necessary, upon collation, for computing and propagating truth values through existential modal operators.\n\nThis allows for some optimizations when model checking specific conditional formulas. For example, it turns out that when you need to test a formula \"⟨L⟩ (MyFeature ≥ 10)\" on a world w, instead of computing \"MyFeature\" on all worlds and then maximizing, computing it on a single world is enough to decide the truth. A few cases arise depending on the relation, the feature and the test operator (or, better, its aggregator).\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.rulemetrics-Union{Tuple{FM}, Tuple{C}, Tuple{O}, Tuple{Rule{O, C, FM}, SoleLogics.AbstractInterpretationSet, AbstractVector{<:Union{AbstractFloat, Integer, String}}}} where {O, C, FM<:SoleModels.AbstractModel}","page":"Home","title":"SoleModels.rulemetrics","text":"rulemetrics(\n    r::Rule,\n    X::AbstractInterpretationSet,\n    Y::AbstractVector{<:Label}\n)\n\nCalculate metrics for a rule with respect to a labelled dataset and returns a NamedTuple consisting of:\n\nsupport: number of instances satisfying the antecedent of the rule divided by   the total number of instances;\nerror:\nFor classification problems: number of instances that were not classified\ncorrectly divided by the total number of instances;\nFor regression problems: mean squared error;\nlength: number of propositions in the rule's antecedent.\n\nSee also Rule, AbstractInterpretationSet, Label, evaluaterule, ninstances, outcometype, consequent.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.submodels-Tuple{SoleModels.AbstractModel}","page":"Home","title":"SoleModels.submodels","text":"submodels(m::AbstractModel)\n\nEnumerate all submodels in the sub-tree. This function is the transitive closure of immediatesubmodels; in fact, the returned list includes the immediate submodels (immediatesubmodels(m)), but also their immediate submodels, and so on.\n\nExamples\n\njulia> using SoleLogics\n\njulia> branch = Branch(SoleLogics.parseformula(\"p∧q∨r\"), \"YES\", \"NO\");\n\njulia> submodels(branch)\n2-element Vector{SoleModels.ConstantModel{String}}:\n ConstantModel\nYES\n\n ConstantModel\nNO\n\n\njulia> branch2 = Branch(SoleLogics.parseformula(\"s→p\"), branch, 42);\n\njulia> printmodel.(submodels(branch2));\nBranch\n┐ p ∧ (q ∨ r)\n├ ✔ YES\n└ ✘ NO\n\nConstantModel\nYES\n\nConstantModel\nNO\n\nConstantModel\n42\n\njulia> submodels(branch) == immediatesubmodels(branch)\ntrue\n\njulia> submodels(branch2) == immediatesubmodels(branch2)\nfalse\n\nSee also immediatesubmodels, FinalModel, AbstractModel.\n\n\n\n\n\n","category":"method"},{"location":"#SoleModels.wrap-Tuple{Any, Type{<:SoleModels.AbstractModel}}","page":"Home","title":"SoleModels.wrap","text":"wrap(o::Any)::AbstractModel\n\nThis function wraps anything into an AbstractModel. The default behavior is the following:\n\nwhen called on an AbstractModel, the model is\n\nsimply returned (no wrapping is performed);\n\nFunctions and FunctionWrappers are wrapped into a FunctionModel;\nevery other object is wrapped into a ConstantModel.\n\nSee also ConstantModel, FunctionModel, ConstrainedModel, FinalModel.\n\n\n\n\n\n","category":"method"}]
}
