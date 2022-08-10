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
    model.package_name in [
        "ScikitLearn",
        "LIBSVM",
        "XGBoost",
    ] ||
    any([
        # https://github.com/lalvim/PartialLeastSquaresRegressor.jl/issues/29
        model.package_name == "PartialLeastSquaresRegressor",
    ])
end

MLJTestIntegration.test_single_target_regressors(
    known_problems,
    ignore=true,
    level=1,
)

fails1, report1 =
    MLJTestIntegration.test_single_target_regressors(
        known_problems,
        ignore=true,
        level=4,
        verbosity=2,
    )

fails1 |> DataFrame

#-

report1 |> DataFrame


# # Classification

known_problems = models() do model
    model.package_name in [
        "ScikitLearn",
        "LIBSVM",
        "XGBoost",
    ] || (name = model.name, package_name = model.package_name) in [

        # https://github.com/JuliaAI/MLJMultivariateStatsInterface.jl/issues/41
        (name = "LDA", package_name = "MultivariateStats"),
        (name = "SubspaceLDA", package_name = "MultivariateStats"),
        (name = "BayesianLDA", package_name = "MultivariateStats"),
        (name = "BayesianSubspaceLDA", package_name = "MultivariateStats"),

        # https://github.com/JuliaAI/MLJBase.jl/issues/781
        (name = "DecisionTreeClassifier", package_name="BetaML"),
        (name="RandomForestClassifier", package_name="BetaML"),

        # https://github.com/alan-turing-institute/MLJ.jl/issues/939
        (name = "NuSVC", package_name="LIBSVM"),
    ]
end

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
        verbosity=2
    )

fails2 |> DataFrame

#-

report2 |> DataFrame
