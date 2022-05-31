@testset "attempt()" begin
    e = ArgumentError("elephant")
    bad() = throw(e)
    good() = 42

    @test (@test_logs MLJTestIntegration.attempt(bad, "")) == (e, "×")
    @test(@test_logs(
        (:info, "look ×"),
        MLJTestIntegration.attempt(bad, "look "),
    )  == (e, "×"))
    @test (@test_logs MLJTestIntegration.attempt(good, "")) == (42, "✓")
    @test (@test_logs(
        (:info, "look ✓"),
        MLJTestIntegration.attempt(good, "look "),
    )  == (42, "✓"))
    @test_throws e MLJTestIntegration.attempt(bad, ""; throw=true)
end

@testset "model_type" begin

    # test error thrown (not caught) if pkg missing from environment:
    @test_throws ArgumentError MLJTestIntegration.model_type(
        (name="PCA", package_name="MultivariateStats"),
        @__MODULE__
    )

    M, outcome = MLJTestIntegration.model_type(
        (name="DecisionTreeClassifier", package_name="DecisionTree"),
        @__MODULE__;
        verbosity=0
    )

end
