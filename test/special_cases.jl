classifiers = [
    (name = "ConstantClassifier", package_name = "MLJModels"),
    (name = "DeterministicConstantClassifier", package_name = "MLJModels")
]

regressors = [
    (name = "ConstantRegressor", package_name = "MLJModels"),
    (name = "DeterministicConstantRegressor", package_name = "MLJModels")
]

@testset "actual_proxies" begin
    data =  MTI._make_baby_boston()
    proxies = @test_logs MTI.actual_proxies(regressors, data, false, 1)
    @test proxies == regressors
    proxies2 = @test_logs MTI.actual_proxies(regressors, data, true, 1)
    @test proxies2 == setdiff(MTI.strip.(models(matching(data...))), regressors)
    proxies = @test_logs(
        (:warn, MTI.warn_not_testing_these(classifiers)),
        MTI.actual_proxies(vcat(regressors, classifiers), data, false, 1),
    )
    @test proxies == regressors
    proxies = @test_logs(
        MTI.actual_proxies(vcat(regressors, classifiers), data, true, 1),
    )
    @test proxies == proxies2
end
