using Test
using Pkg
using MLJTestIntegration
using MLJ
using MLJModels
const MTI = MLJTestIntegration

classifiers = [
    (name = "ConstantClassifier", package_name = "MLJModels"),
    (name = "DeterministicConstantClassifier", package_name = "MLJModels")
]

expected_report1 = (
    name = "ConstantClassifier",
    package_name = "MLJModels",
    model_type = "✓",
    model_instance = "✓",
    fitted_machine = "✓",
    operations = "predict",
    evaluation = "✓",
    accelerated_evaluation = "✓",
    tuned_pipe_evaluation = "✓",
    threshold_prediction = "✓",
    ensemble_prediction = "✓",
    iteration_prediction = "-",
    stack_evaluation = "✓",
    accelerated_stack_evaluation = "✓",
)

expected_report2 = (
    name = "DeterministicConstantClassifier",
    package_name = "MLJModels",
    model_type = "✓",
    model_instance = "✓",
    fitted_machine = "✓",
    operations = "predict",
    evaluation = "✓",
    accelerated_evaluation = "✓",
    tuned_pipe_evaluation = "✓",
    threshold_prediction = "-",
    ensemble_prediction = "✓",
    iteration_prediction = "-",
    stack_evaluation = "-",
    accelerated_stack_evaluation = "-",
)

@testset "test classifiers on valid data (models are proxies)" begin
    X, y0 = make_moons();
    y = coerce(y0, OrderedFactor);

    fails, report  =
        @test_logs MLJTestIntegration.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=4,
            verbosity=0
        )
    @test isempty(fails)
    @test report[1] == expected_report1
    @test report[2] == expected_report2
end

@testset "test classifiers on valid data (models are types)" begin

    classifiers = [ConstantClassifier, DeterministicConstantClassifier]
    X, y0 = make_moons();
    y = coerce(y0, OrderedFactor);

    fails, report  =
        @test_logs MLJTestIntegration.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=4,
            verbosity=0
        )
    @test isempty(fails)
    @test report[1] == expected_report1
    @test report[2] == expected_report2
end

@testset "test classifiers on invalid data" begin
    X, y = make_regression(); # wrong data for a classifier
    fails, report = @test_logs(
        (:error, r""),
        match_mode=:any,
        MLJTestIntegration.test(
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
        package_name = "MLJModels",
        test = "fitted_machine",
        exception = ""
    )

    @test fails[2].exception isa ArgumentError
    @test merge(fails[2], (; exception="")) == (
        name = "DeterministicConstantClassifier",
        package_name = "MLJModels",
        test = "evaluation",
        exception = ""
    )

    @test report[1] == (
        name = "ConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "×",
        operations = "-",
        evaluation = "-",
        accelerated_evaluation = "-",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-",
        stack_evaluation = "-",
        accelerated_stack_evaluation = "-",
    )

    @test report[2] == (
        name = "DeterministicConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
        evaluation = "×",
        accelerated_evaluation = "-",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-",
        stack_evaluation = "-",
        accelerated_stack_evaluation = "-",
    )

    # throw=true:
    @test_logs(
        (:error, r""), match_mode=:any,
        @test_throws(
            ErrorException,
            MLJTestIntegration.test(
                classifiers,
                X,
                y;
                mod=@__MODULE__,
                level=3,
                throw=true,
                verbosity=0
            )
        )
    )
end

classifiers = [ConstantClassifier,]
X, y0 = make_moons();
y = coerce(y0, OrderedFactor);

@testset "verbose logging" begin
    # progress meter:
    @test_logs MLJTestIntegration.test(
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
        (:info, r"stack_evaluation"),
        MLJTestIntegration.test(
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
    fails, report  =
        @test_logs MLJTestIntegration.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=1,
            verbosity=0)
    @test isempty(fails)
    @test report[1] == (
        name = "ConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "-",
        fitted_machine = "-",
        operations = "-",
        evaluation = "-",
        accelerated_evaluation = "-",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-",
        stack_evaluation = "-",
        accelerated_stack_evaluation = "-",
    )

    # level=2:
    fails, report  =
        @test_logs MLJTestIntegration.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=2,
            verbosity=0)
    @test isempty(fails)
    @test report[1] == (
        name = "ConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
        evaluation = "-",
        accelerated_evaluation = "-",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-",
        stack_evaluation = "-",
        accelerated_stack_evaluation = "-",
    )

    # level=4:
    fails, report  =
        @test_logs MLJTestIntegration.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=4,
            verbosity=0)
    @test isempty(fails)
    @test report[1] == (
        name = "ConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
        evaluation = "✓",
        accelerated_evaluation = "✓",
        tuned_pipe_evaluation = "✓",
        threshold_prediction = "✓",
        ensemble_prediction = "✓",
        iteration_prediction = "-",
        stack_evaluation = "✓",
        accelerated_stack_evaluation = "✓",
    )
end

@testset "iterative model" begin
    X, y = MLJTestIntegration.make_dummy();
    fails, report =
        MLJTestIntegration.test(
            [MLJTestIntegration.DummyIterativeModel,],
            X,
            y;
            mod=@__MODULE__,
            level=3,
            verbosity=0
        )
    @test isempty(fails)
    @test report[1] == (
        name = "DummyIterativeModel",
        package_name = "MLJTestIntegration",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
        evaluation = "✓",
        accelerated_evaluation = "-",
        tuned_pipe_evaluation = "✓",
        threshold_prediction = "-",
        ensemble_prediction = "✓",
        iteration_prediction = "✓",
        stack_evaluation = "-",
        accelerated_stack_evaluation = "-",
    )
end

true
