# **Assumption.** This file has not been separated from the
# Project.toml file that
# [originally](https://github.com/JuliaAI/MLJTest.jl/blob/dev/examples/bigtest/Project.toml)
# accompanied it.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MLJTest
using MLJModels
using Test
using DataFrames # for displaying tables

# # Regression

known_issues = models() do model
    any([
        # https://github.com/lalvim/PartialLeastSquaresRegressor.jl/issues/29
        model.package_name == "PartialLeastSquaresRegressor",

        #https://github.com/JuliaAI/MLJXGBoostInterface.jl/issues/17
        model.name == "XGBoostRegressor",
    ])
end

MLJTest.test_single_target_regressors(ignore=known_issues, level=1)
fails, summary =
    MLJTest.test_single_target_regressors(ignore=known_issues, level=3)

@test isempty(fails)
summary |> DataFrame

# # Classification

known_issues = models() do m
    any([
    ])
end

#     # TODO: investigate this exclusion:
#     !(m.abstract_type <: MLJ.MLJBase.SupervisedAnnotator)

MLJTest.test_single_target_classifiers(ignore=known_issues, level=1)
fails, summary =
    MLJTest.test_single_target_classifiers(ignore=known_issues, level=3)


