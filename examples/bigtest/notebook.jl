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
        # https://github.com/jeremiedb/EvoLinear.jl/issues/12
        model.package_name == "EvoLinear",
        # waiting for BetaML 0.9.1:
        model.package_name == "BetaML",
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

@assert isempty(fails1)

#-


# # Classification

known_problems = models() do model
    model.package_name in [
        "ScikitLearn",
        "LIBSVM",
        "XGBoost",
        "BetaML", # waiting for BetaML 0.9.1:
        "EvoLinear", # https://github.com/jeremiedb/EvoLinear.jl/issues/12
    ] || (name = model.name, package_name = model.package_name) in []
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

@assert isempty(fails2)
