@testset "model_type with model proxy instead of type" begin

    # test error thrown (not caught) if pkg missing from environment:
    @test_throws ArgumentError MLJTestIntegration.MLJTestInterface.model_type(
        (name="PCA", package_name="MultivariateStats"),
        @__MODULE__
    )

    M, outcome = MLJTestIntegration.MLJTestInterface.model_type(
        (name="DecisionTreeClassifier", package_name="DecisionTree"),
        @__MODULE__;
        verbosity=0
    )

end

clf = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
rgs = (@load DecisionTreeRegressor pkg=DecisionTree verbosity=0)()
Xclf, yclf = make_moons(rng=123)
Xrgs, yrgs = make_regression(rng=123)

@testset "stack_evaluation" begin

    # with probablistic classifier:
    e, outcome = MTI.stack_evaluation(
        clf,
        [CPU1(), CPUThreads()],
        Xclf, yclf;
        throw=true,
        verbosity=0,
    )
    @test outcome == "✓"

    # with deterministic regressor:
    e, outcome = MTI.stack_evaluation(
        rgs,
        [CPU1(), CPUThreads()],
        Xrgs, yrgs;
        throw=false,
        verbosity=0,
    )
    @test outcome == "✓"

end

true
