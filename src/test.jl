_package_name(model) = MLJ.package_name(model)
_name(model) = MLJ.name(model)
_package_name(model::NamedTuple) = model.package_name
_name(model::NamedTuple) = model.name

# to update progress meter:
function next!(p)
    p.counter +=1
    MLJ.ProgressMeter.updateProgress!(p)
end

"""
    test(models, data...; mod=Main, level=2, throw=false, verbosity=1)

Apply a battery of MLJ integration tests to a collection of models,
using `data` for training. Here `mod` should be the module from which
`test` is called (generally, `mod=@__MODULE__` will work). Here
`models` is either:

1. A collection of model types implementing the [MLJ model
interface](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/).

2. A collection of named tuples, where each tuple includes `:name` and
  `:package_name` as keys, and whose corresponding values point to a
  model types appearing in the [MLJ Model
  Registry](https://github.com/JuliaAI/MLJModels.jl/tree/dev/src/registry).
  `MLJ.models(...)` always returns such a collection.

Ordinarily, code defining the model types to be tested must already be
loaded into the module `mod`. An exception is described under "Testing
with automatic code loading" below.

  interface packages providing the models must be in the current
  environment, but the packages need not be loaded.

The extent of testing is controlled by `level`:

|`level`          | description                      | tests (full list below) |
|:----------------|:---------------------------------|:------------------------|
| 1               | test code loading                | `:model_type`           |
| 2 (default)     | basic test of model interface    | first four tests        |
| 3               | comprehensive                    | all applicable tests    |

By default, exceptions caught in tests are not thrown. If
`throw=true`, testing will terminate at the first execption
encountered, after throwing that exception (useful to obtain stack
traces).

# Return value

Returns `(failures, summary)` where:

- `failures`: table of exceptions thrown

- `summary`: table summarizing the outcomes of each test, where
  outcomes are indicated as below:

| entry | interpretation                     |
|:------|:-----------------------------------|
| ✓     | test succesful                     |
| ×     | test unsuccessful                  |
| n/a   | skipped because not applicable     |
| -     | test skipped for some other reason |

# Testing with automatic code loading

World Age issues pose challenges for testing Julia code if some code
is to be loaded "on demand". Nevertheless, in case 2 mentioned above,
model types to be tested need not be loaded, provided testing is
carried out in two stages, as shown in the second example below. In
this case, however, the necessary model interface packages need
to be listed in the current Julia environment, and the `test` calls
must appear in global scope.

# Examples

## Testing models in a new MLJ model interface implementation

The following tests the model interface implemented by some model type
`MyClassifier`, as might appear in tests for a package providing that
type:

```julia
import MLJTest
using Test
X, y = MLJTest.MLJ.make_blobs()
failures, summary = MLJTest.test([MyClassifier, ], X, y, verbosity=1, mod=@__MODULE__)
@test isempty(failures)
```

## Testing models after filtering models in the registry

The following applies comprehensive integration tests to all
regressors provided by the package GLM.jl appearing in the MLJ Model
Registry. Since GLM.jl models are provided through the interface
package `MLJGLMInterface`, this must be in the current environment:

```julia
Pkg.add("MLJGLMInterface")
import MLJBase, MLJTest
using DataFrames # to view summary
X, y = MLJTest.MLJ.make_regression();
regressors = MLJTest.MLJ.models(matching(X, y)) do m
    m.package_name == "GLM"
end

# to test code loading *and* load code:
MLJTest.test(regressors, X, y, verbosity=1, mod=@__MODULE__, level=1)

# comprehensive tests:
failures, summary =
    MLJTest.test(regressors, X, y, verbosity=3, mod=@__MODULE__, level=1)

summary |> DataFrame
```

# List of tests

Tests are applied in sequence. When a test fails, subsequent tests for
that model are skipped. The following are applied to all models:

- `:model_type`: Load model type using registry (if named tuples are
  provided) or using `load_path(model_type)` (if types are provided, to
  check `load_path` trait is correctly overloaded).

- `:model_instance`: Create a default instance.

- `:fitted_machine`: Bind instance to data in a machine and `fit!`

- `:operations`: Call implemented operations, such as `predict` and `transform`

These additional tests are applied to `Supervised` models:

- `:threshold_prediction`: If the model is `Probablisitic` and
  `scitype(data[2]) <: Finite{2}` (binary classification) then wrap
  model using `BinaryThresholdPredictor` and `fit!`.

- `:evaluation`: Assuming MLJ is able to infer a suitable `measure`
  (metric), evaluate the performance of the model using `evaluate!`
  and a `Holdout` set.

- `:tuned_pipe_evaluation`: Repeat the `:evauation` test but first
  insert model in a pipeline with a trivial pre-processing step
  (applies the identity transformation) and wrap in `TunedModel` (only
  the default instance is actually evaluated)

- Repeat the `:evaluation` test but first wrap as an `EnsembleModel`.

- If the model is iterable, repeat the `:evaluation` test
  but first wrap as an `IteratedModel`.

"""
function test(model_proxies, data...; mod=Main, level=2, throw=false, verbosity=1,)

    nproxies = length(model_proxies)

    # initiate return objects:
    failures = NamedTuple{(:name, :package_name, :test, :exception), NTuple{4, Any}}[]
    summary = Vector{NamedTuple{(
        :name,
        :package_name,
        :model_type,
        :model_instance,
        :fitted_machine,
        :operations,
        :evaluation,
        :tuned_pipe_evaluation,
        :threshold_prediction,
        :ensemble_prediction,
        :iteration_prediction
    ), NTuple{11, String}}}(undef, nproxies)

    # summary table row corresponding to all tests skipped:
    row0 = (
        ; name="undefined",
        package_name= "undefined",
        model_type = "-",
        model_instance = "-",
        fitted_machine = "-",
        operations = "-",
        evaluation = "-",
        tuned_pipe_evaluation = "-",
        threshold_prediction = "-",
        ensemble_prediction = "-",
        iteration_prediction = "-"
    )

    # for updating `failures` and `summary` tables; returns the updated row,
    # as added to `summary`:
    function update(row, i, test, value_or_exception, outcome)
        outcome_nt = NamedTuple{(test,)}((outcome,))
        updated_row = merge(row, outcome_nt)
        summary[i] = updated_row
        if outcome == "×"
            failures_row = (
                ; name=row.name,
                package_name=row.package_name,
                test=string(test),
                exception=value_or_exception
            )
            push!(failures, failures_row)
        end
        return updated_row
    end

    if verbosity == 1
        meter = MLJ.ProgressMeter.Progress(
            nproxies,
            desc = "Testing $nproxies models: ",
            barglyphs = MLJ.ProgressMeter.BarGlyphs("[=> ]"),
            barlen = 25,
            color = :yellow
        )
    end

    for (i, model_proxy) in enumerate(model_proxies)

        verbosity == 1 && next!(meter)

        package_name = _package_name(model_proxy)
        name = _name(model_proxy)

        verbosity > 1 && @info "\nTesting $name from $package_name"

        row = merge(row0, (; name, package_name))

        # model_type:
        model_type, outcome = MLJTest.model_type(model_proxy, mod; throw, verbosity)
        row = update(row, i, :model_type, model_type, outcome)
        outcome == "×" && continue

        level > 1 || continue

        # model_instance:
        model_instance, outcome =
            MLJTest.model_instance(model_type; throw, verbosity)
        row = update(row, i, :model_instance, model_instance, outcome)
        outcome == "×" && continue

        # fitted_machine:
        fitted_machine, outcome =
            MLJTest.fitted_machine(model_instance, data...; throw, verbosity)
        row = update(row, i, :fitted_machine, fitted_machine, outcome)
        outcome == "×" && continue

        # operations:
        operations, outcome =
            MLJTest.operations(fitted_machine, data...; throw, verbosity)
        # special treatment to get list of operations in `summary`:
        if operations == "×"
            row = update(row, i, :operations, operations, outcome)
            continue
        else
            row = update(row, i, :operations, operations, operations)
        end

        level > 2 || continue
        model_instance isa Supervised || continue

        # supervised tests:

        # threshold_prediction:
        if prediction_type(model_instance) == :probabilistic &&
            target_scitype(model_instance) <: AbstractArray{<:Finite} &&
            length(data) > 1 &&
            scitype(data[2]) <: AbstractVector{<:Finite{2}}

            threshold_prediction, outcome =
                MLJTest.threshold_prediction(model_instance, data...; throw, verbosity)
            row = update(row, i, :threshold_prediction, threshold_prediction, outcome)
            outcome == "×" && continue
        end

        measure = MLJ.MLJBase.default_measure(model_instance)

        isnothing(measure) && continue

        # evaluation:
        evaluation, outcome =
            MLJTest.evaluation(measure, model_instance, data...; throw, verbosity)
        row = update(row, i, :evaluation, evaluation, outcome)
        outcome == "×" && continue

        # tuned_pipe_evaluation:
        tuned_pipe_evaluation, outcome =
            MLJTest.tuned_pipe_evaluation(
                measure,
                model_instance,
                data...;
                throw,
                verbosity
            )
        row = update(row, i, :tuned_pipe_evaluation, tuned_pipe_evaluation, outcome)
        outcome == "×" && continue

        # ensemble_prediction:
        ensemble_prediction, outcome =
            MLJTest.ensemble_prediction(model_instance, data...; throw, verbosity)
        row = update(row, i, :ensemble_prediction, ensemble_prediction, outcome)
        outcome == "×" && continue

        isnothing(iteration_parameter(model_instance)) &&  continue

        # iteration prediction:
        iteration_prediction, outcome =
            MLJTest.iteration_prediction(measure, model_instance, data...; throw, verbosity)
        row = update(row, i, :iteration_prediction, iteration_prediction, outcome)
        outcome == "×" && continue
    end

    return failures, summary
end
