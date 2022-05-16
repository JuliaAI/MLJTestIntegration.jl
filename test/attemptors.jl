@testset "attempt()" begin
    e = ArgumentError("elephant")
    bad() = throw(e)
    good() = 42

    @test (@test_logs MLJTest.attempt(bad)) == (e, "×")
    @test (@test_logs (:info, "look ×") MLJTest.attempt(bad, "look "))  == (e, "×")
    @test (@test_logs MLJTest.attempt(good)) == (42, "✓")
    @test (@test_logs (:info, "look ✓") MLJTest.attempt(good, "look "))  == (42, "✓")
end

@testset "model_type" begin

    # test error thrown (not caught) if pkg missing from environment:
    @test_throws ArgumentError MLJTest.model_type(
        (name="PCA", package_name="MultivariateStats"),
        @__MODULE__
    )

    M, outcome = MLJTest.model_type(
        (name="DecisionTreeClassifier", package_name="DecisionTree"),
        @__MODULE__;
        verbosity=0
    )

end
