# **Assumption.** This file has not been separated from the
# Project.toml file that
# [originally](https://github.com/JuliaAI/MLJTestIntegration.jl/blob/dev/examples/bigtest/Project.toml)
# accompanied it.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MLJTestIntegration
using MLJModels
using Test
using DataFrames # for displaying tables

if false
# # Regression

known_issues = models() do model
    any([
        # https://github.com/lalvim/PartialLeastSquaresRegressor.jl/issues/29
        model.package_name == "PartialLeastSquaresRegressor",

        #https://github.com/JuliaAI/MLJXGBoostInterface.jl/issues/17
        model.name == "XGBoostRegressor",
    ])
end

MLJTestIntegration.test_single_target_regressors(ignore=known_issues, level=1)
fails, summary =
    MLJTestIntegration.test_single_target_regressors(ignore=known_issues, level=3)

@test isempty(fails)
summary |> DataFrame
end


# # Classification

# https://github.com/alan-turing-institute/MLJ.jl/issues/939
known_issues = [
    #
    (name = "DecisionTreeClassifier", package_name="BetaML"),
    (name = "NuSVC", package_name="LIBSVM"),
    (name="PegasosClassifier", package_name="BetaML"),
    (name="RandomForestClassifier", package_name="BetaML"),
    (name="SVMNuClassifier", package_name="ScikitLearn"),
]

MLJTestIntegration.test_single_target_classifiers(ignore=known_issues, level=1)
fails, summary =
    MLJTestIntegration.test_single_target_classifiers(ignore=known_issues, level=3)
