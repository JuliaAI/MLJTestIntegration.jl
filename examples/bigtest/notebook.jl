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


# # Regression

known_problems = models() do model
    any([
        # https://github.com/lalvim/PartialLeastSquaresRegressor.jl/issues/29
        model.package_name == "PartialLeastSquaresRegressor",

        #https://github.com/JuliaAI/MLJXGBoostInterface.jl/issues/17
        model.name == "XGBoostRegressor",
    ])
end

MLJTestIntegration.test_single_target_regressors(
    known_problems,
    ignore=true,
    level=1
)

fails1, report1 =
    MLJTestIntegration.test_single_target_regressors(
        known_problems,
        ignore=true,
        level=4
    )

fails1 |> DataFrame

#-

report1 |> DataFrame


# # Classification

# https://github.com/alan-turing-institute/MLJ.jl/issues/939
known_problems = [
    (name = "KernelPerceptronClassifier", package_name="BetaML"),
    (name = "DecisionTreeClassifier", package_name="BetaML"),
    (name = "PerceptronClassifier", package_name="BetaML"),
    (name = "NuSVC", package_name="LIBSVM"),
    (name="PegasosClassifier", package_name="BetaML"),
    (name="RandomForestClassifier", package_name="BetaML"),
    (name="SVMNuClassifier", package_name="ScikitLearn"),
    (name="KernelPerceptronClassifier", package_name="BetaML"),
    (name="LinearSVC", package_name="LIBSVM"),
    (name= "MultinomialClassifier", "MLJLinearModels"),
    (name="SVMLinearClassifier", package_name="ScikitLearn"),
]

MLJTestIntegration.test_single_target_classifiers(
    known_problems,
    level=1,
    ignore=true,
)
fails2, report2 =
    MLJTestIntegration.test_single_target_classifiers(
        known_problems,
        ignore=true,
        level=4,
    )

fails2 |> DataFrame

#-

report2 |> DataFrame
