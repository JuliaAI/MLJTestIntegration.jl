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
        # as python models, problems with multithreading:
        "MLJScikitLearnInterface",
        "LIBSVM",
        "XGBoost",
        "CatBoost",
        # https://github.com/lalvim/PartialLeastSquaresRegressor.jl/issues/29]
        # waiting for a release > 2.2.0 (issue addressed on master)
        "PartialLeastSquaresRegressor",
    ] || (name = model.name, package_name = model.package_name) in []
end

MLJTestIntegration.test_single_target_regressors(
    known_problems,
    ignore=true,
    level=1,
    throw=true,
)

fails1, report1 =
    MLJTestIntegration.test_single_target_regressors(
        known_problems,
        ignore=true,
        level=4,
        throw=false,
        verbosity=2,
    )

@assert isempty(fails1)

#-


# # Classification

known_problems = models() do model
    model.package_name in [
        # as python models, problems with multithreading:
        "MLJScikitLearnInterface",
        "LIBSVM",
        "XGBoost",
        "CatBoost",
    ] || (name = model.name, package_name = model.package_name) in [
        # https://github.com/OutlierDetectionJL/OutlierDetectionNetworks.jl/issues/8
        (name = "ESADDetector", package_name="OutlierDetectionNetworks"),
        (name = "DSADDetector", package_name="OutlierDetectionNetworks"),
        # https://github.com/rikhuijzer/SIRUS.jl/issues/43
        (name = "StableForestClassifier", package_name="SIRUS"),
        (name = "StableRulesClassifier", package_name="SIRUS"),
    ]
end

MLJTestIntegration.test_single_target_classifiers(
    known_problems,
    level=1,
    ignore=true,
    throw=true,
)

fails2, report2 =
    MLJTestIntegration.test_single_target_classifiers(
        known_problems,
        ignore=true,
        level=4,
        verbosity=2
    )

@assert isempty(fails2)
