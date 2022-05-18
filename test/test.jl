# grab some classifiers from MLJModels:
classifiers = [
    (name = "ConstantClassifier", package_name = "MLJModels"),
    (name = "DeterministicConstantClassifier", package_name = "MLJModels")
]

expected_summary1 = (
    name = "ConstantClassifier",
    package = "MLJModels",
    model_type = "✓",
    model_instance = "✓",
    fitted_machine = "✓",
    operations = "predict",
    evaluation = "✓",
    tuned_pipe_evaluation = "✓",
    threshold_prediction = "✓",
    ensemble_prediction = "✓",
    iteration_prediction = "-"
)

expected_summary2 = (
    name = "DeterministicConstantClassifier",
    package = "MLJModels",
    model_type = "✓",
    model_instance = "✓",
    fitted_machine = "✓",
    operations = "predict",
    evaluation = "✓",
    tuned_pipe_evaluation = "✓",
    threshold_prediction = "-",
    ensemble_prediction = "✓",
    iteration_prediction = "-"
)

@testset "test classifiers on valid data (models are proxies)" begin
    X, y0 = make_moons();
    y = coerce(y0, OrderedFactor);

    fails, summary  =
        @test_logs MLJTest.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=3,
            verbosity=0
        )
    @test isempty(fails)
    @test summary[1] == expected_summary1
    @test summary[2] == expected_summary2
end

@testset "test classifiers on valid data (models are types)" begin

    classifiers = [ConstantClassifier, DeterministicConstantClassifier]
    X, y0 = make_moons();
    y = coerce(y0, OrderedFactor);

    fails, summary  =
        @test_logs MLJTest.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=3,
            verbosity=0
        )
    @test isempty(fails)
    @test summary[1] == expected_summary1
    @test summary[2] == expected_summary2
end

@testset "test classifiers on invalid data" begin
    X, y = make_regression(); # wrong data for a classifier
    fails, summary = @test_logs(
        (:error, r""),
        match_mode=:any,
        MLJTest.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=3,
            verbosity=0
        )
    )

    @test length(fails) === 2
    @test fails[1].exception isa ErrorException
    @test merge(fails[1], (; exception="")) == (
        name = "ConstantClassifier",
        package = "MLJModels",
        test = "fitted_machine",
        exception = ""
    )

    @test fails[2].exception isa ArgumentError
    @test merge(fails[2], (; exception="")) == (
        name = "DeterministicConstantClassifier",
        package = "MLJModels",
        test = "evaluation",
        exception = ""
    )

    @test summary[1] == (
        name = "ConstantClassifier",
        package = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "×",
        operations = "-",
        evaluation = "-",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-"
    )

    @test summary[2] == (
        name = "DeterministicConstantClassifier",
        package = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
        evaluation = "×",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-"
    )

end

classifiers = [ConstantClassifier,]
X, y0 = make_moons();
y = coerce(y0, OrderedFactor);

@testset "verbose logging" begin
    # progress meter:
    @test_logs MLJTest.test(
        fill(classifiers[1], 500),
        X,
        y;
        mod=@__MODULE__,
        level=1,
        verbosity=1);

    # verbosity high:
    @test_logs(
        (:info, r"Testing ConstantClassifier"),
        (:info, r"model_type"),
        (:info, r"model_instance"),
        (:info, r"fitted_machine"),
        (:info, r"operations"),
        (:info, r"threshold_predictor"),
        (:info, r"evaluation"),
        (:info, r"tuned_pipe_evaluation"),
        (:info, r"ensemble_prediction"),
        MLJTest.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=3,
            verbosity=2)
    )
end

@testset "level" begin
    # level=1:
    fails, summary  =
        @test_logs MLJTest.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=1,
            verbosity=0)
    @test isempty(fails)
    @test summary[1] == (
        name = "ConstantClassifier",
        package = "MLJModels",
        model_type = "✓",
        model_instance = "-",
        fitted_machine = "-",
        operations = "-",
        evaluation = "-",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-",
    )

    # level=2:
    fails, summary  =
        @test_logs MLJTest.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=2,
            verbosity=0)
    @test isempty(fails)
    @test summary[1] == (
        name = "ConstantClassifier",
        package = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
        evaluation = "-",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-",
    )
end

@testset "iterative model" begin
    X, y = MLJTest.make_dummy();
    fails, summary =
        MLJTest.test(
            [MLJTest.DummyIterativeModel,],
            X,
            y;
            mod=@__MODULE__,
            level=3,
            verbosity=0
        )
    @test isempty(fails)
    @test summary[1] == (
        name = "DummyIterativeModel",
        package = "MLJTest",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
        evaluation = "✓",
        tuned_pipe_evaluation = "✓",
        threshold_prediction = "-",
        ensemble_prediction = "✓",
        iteration_prediction = "✓",)
end

# if false

# if false
#     MLJModels.activate_registry_environment()
#     Pkg.instantiate()
# else
#     Pkg.activate(temp=true)
#     Pkg.add("MLJDecisionTreeInterface")
#     Pkg.instantiate()
# end

# # pre-load all model types (needed to avoid "alternative world"
# # conflicts and other weird stuff)

# verbosity = 0

# @info "Loading all model types..."
# for model_metadata in models(m->m.package_name == "DecisionTree")
#     MLJTest.model_type(model_metadata, verbosity=verbosity)
# end

# # ## Grab data and define some collections of models

# # supervised models with a single binary target, excluding supervised
# # outlier detectors:
# const X, y0 = make_moons();
# const y = coerce(y0, OrderedFactor)
# const classifiers = models(matching(X, y)) do m
#     !(m.abstract_type <: MLJ.MLJBase.SupervisedAnnotator) &&
#         package_name == "DecisionTree"
# end

# # supervised models with a single `Continuous` target:
# const X2, y2 = make_regression();
# const regressors = models(matching(X2, y2)) do m
#     package_name == "DecisionTree"
# end

# if false
#     # supervised models with a single `Count` target
#     const count_regressors = models(matching(X, eachindex(y)))

#     # unsupervised models, excluding some outlier detectors
#     const transformers = models(matching(X)) do m
#         m.package_name != "OutlierDetectionPython"
#     end
# end

# # ## run the tests

# classifier_fails, t1 = MLJTest.test(classifiers, X, y; verbosity);
# regressor_fails, t2 = MLJTest.test(regressors, X2, y2; verbosity);
# if false
#     transformer_fails, t3 = MLJTest.test(transformers, X; verbosity);
# end
# if false
#     count_regressor_fails, t4 =
#         MLJTest.test(count_regressors[1:2], X,
#                                  rand([1, 2, 3], length(y)); verbosity);
# end

# #-
# if false

# untested_models =
#     setdiff(models(), union(classifiers, regressors, count_regressors, transformers))

# end

# @test classifier_fails |> isempty

# end

# true
