using Test
using Revise
using Random
using DataFrames
using SoleLogics
using SoleModels
using SoleData
using SoleData.DimensionalDatasets

n_instances = 20
rng = MersenneTwister(42)

# Dataset Construction
attributes = ["fever","pressure"]
attributes_values = [
    [rand(rng,collect(36:0.5:40)) for i in 1:n_instances],
    [rand(rng,collect(60:2:130)) for i in 1:n_instances],
]
y_true = [attributes_values[1][i] >= 37.5 && attributes_values[2][i] <= 100 ? "sick" : "not sick" for i in 1:n_instances]
dataset = DataFrame(; NamedTuple([Symbol(attributes[i]) => attributes_values[i] for i in 1:length(attributes)])...)

# Logiset Definition
nvars = nvariables(dataset)
features = collect(Iterators.flatten([[UnivariateMin(i_var)] for i_var in 1:nvars]))
logiset = scalarlogiset(dataset, features; use_full_memoization = false, use_onestep_memoization = false)

# Rule Definition: max[V1] >= 38, max[V2] < 110

# Build a formula on scalar conditions
condition1 = ScalarCondition(features[1], >=, 38.0)
condition2 = ScalarCondition(features[2], <, 110)
antecedentrule = Atom(condition1) ∧ Atom(condition2)

# Build consequent
consequentrule = "sick"

# Build Rule without info
rule = Rule(antecedentrule, consequentrule)

inforule = metrics(rule)
inforulelogiset = metrics(rule,logiset)
inforuley = metrics(rule,Y = y_true)
inforuleall = metrics(rule,logiset,y_true)
newrule = metrics(rule; return_model=true)
newrulelogiset = metrics(rule,logiset; return_model=true)
newruley = metrics(rule,Y = y_true; return_model=true)
newruleall = metrics(rule,logiset,y_true; return_model=true)


@test inforule == NamedTuple()
@test inforulelogiset == NamedTuple()
@test inforuley == NamedTuple()
@test inforuleall == (ninstances = 8, accuracy = 1.0,)
@test SoleModels.info(newrule) == NamedTuple()
@test SoleModels.info(newrule) == NamedTuple()
@test SoleModels.info(newrulelogiset) == NamedTuple()
@test SoleModels.info(newruley) == NamedTuple()
@test SoleModels.info(newruleall) == (ninstances = 8, accuracy = 1.0,)

# Build Rule with info
rule = Rule(antecedentrule,consequentrule,(; supporting_labels = y_true))

inforule = metrics(rule)
inforulelogiset = metrics(rule,logiset)
inforuley = metrics(rule,Y = y_true)
inforuleall = metrics(rule,logiset,y_true)
newrule = metrics(rule; return_model=true)
newrulelogiset = metrics(rule,logiset; return_model=true)
newruley = metrics(rule,Y = y_true; return_model=true)
newruleall = metrics(rule,logiset,y_true; return_model=true)


@test inforule == (ninstances = 20,)
@test inforulelogiset == (ninstances = 8, accuracy = 1.0,)
@test inforuley == (ninstances = 20,)
@test inforuleall == (ninstances = 8, accuracy = 1.0,)
@test SoleModels.info(newrule) == (ninstances = 20,)
@test SoleModels.info(newrule) == (ninstances = 20,)
@test SoleModels.info(newrulelogiset) == (ninstances = 8, accuracy = 1.0,)
@test SoleModels.info(newruley) == (ninstances = 20,)
@test SoleModels.info(newruleall) == (ninstances = 8, accuracy = 1.0,)

# Build Rule with info
supp_preds = apply(Rule(antecedentrule,consequentrule),logiset)
rule = Rule(antecedentrule,consequentrule,(; supporting_labels = y_true, supporting_predictions = supp_preds))

inforule = metrics(rule)
inforulelogiset = metrics(rule,logiset)
inforuley = metrics(rule,Y = y_true)
inforuleall = metrics(rule,logiset,y_true)
newrule = metrics(rule; return_model=true)
newrulelogiset = metrics(rule,logiset; return_model=true)
newruley = metrics(rule,Y = y_true; return_model=true)
newruleall = metrics(rule,logiset,y_true; return_model=true)


@test inforule == (ninstances = 8, accuracy = 1.0)
@test inforulelogiset == (ninstances = 8, accuracy = 1.0,)
@test inforuley == (ninstances = 8, accuracy = 1.0)
@test inforuleall == (ninstances = 8, accuracy = 1.0,)
@test SoleModels.info(newrule) == (ninstances = 8, accuracy = 1.0)
@test SoleModels.info(newrule) == (ninstances = 8, accuracy = 1.0)
@test SoleModels.info(newrulelogiset) == (ninstances = 8, accuracy = 1.0,)
@test SoleModels.info(newruley) == (ninstances = 8, accuracy = 1.0)
@test SoleModels.info(newruleall) == (ninstances = 8, accuracy = 1.0,)


