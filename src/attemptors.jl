const ERR_INCONSISTENT_RESULTS =
    "Different computational resources are giving different results. "


# # NEW ATTEMPTORS

root(load_path) = split(load_path, '.') |> first

function threshold_prediction(model, data...; throw=false, verbosity=1)
    message = "[:threshold_predictor] Calling fit!/predict for threshold predictor "*
        "test) "
    attempt(MTI.finalize(message, verbosity); throw) do
        tmodel = BinaryThresholdPredictor(model)
        mach = machine(tmodel, data...)
        fit!(mach, verbosity=0)
        predict(mach, first(data))
    end
end

function evaluation(measure, model, resources, data...; throw=false, verbosity=1)
    L = length(resources)
    message = L > 1 ? "[:accelerated_evaluation] " : "[evaluation] "
    message *=  "Evaluating model performance using $L different resources. "
    attempt(MTI.finalize(message, verbosity); throw) do
        es = map(resources) do resource
            evaluate(model, data...;
                     measure=measure,
                     resampling=CV(nfolds=4),
                     acceleration=resource,
                     verbosity=0)
        end
        ms = map(e->sort(e.per_fold[1]), es)
        m = first(ms)
        @assert all(≈(m), collect(ms)[2:end]) ERR_INCONSISTENT_RESULTS
        return first(es)
    end
end

function tuned_pipe_evaluation(
    measure,
    model,
    data...;
    throw=false,
    verbosity=1,
)
    message = "[:tuned_pipe_evaluation] Evaluating perfomance in a tuned pipeline "
    attempt(MTI.finalize(message, verbosity); throw) do
        pipe = identity |> model
        tuned_pipe = TunedModel(
            models=fill(pipe, 3),
            measure=measure,
        )
        evaluate(
            tuned_pipe, data...;
            measure=measure,
            verbosity=0,
        )
    end
end

function ensemble_prediction(model, data...; throw=false, verbosity=1)
    attempt(MTI.finalize("[:ensemble_prediction] Ensembling ", verbosity); throw) do
        imodel = EnsembleModel(
            model=model,
            n=2,
        )
        mach = machine(imodel, data...)
        fit!(mach, verbosity=0)
        predict(mach, first(data))
    end
end

# the `model` must support iteration (`!isnothing(iteration_paramater(model))`)
function iteration_prediction(measure, model, data...; throw=false, verbosity=1)
    message =  "[:iteration_prediction] Iterating with controls "
    attempt(MTI.finalize(message, verbosity); throw) do
        imodel = IteratedModel(model=model,
                               measure=measure,
                               controls=[Step(1),
                                         InvalidValue(),
                                         NumberLimit(2)])
        mach = machine(imodel, data...)
        fit!(mach, verbosity=0)
        predict(mach, first(data))
    end
end

function _stack(model, resource, isregressor)
    if isregressor
        models = (knn1=KNNRegressor(K=4),
                  knn2=KNNRegressor(K=6),
                  tmodel=model)
        metalearner = KNNRegressor()
    else
        models = (knn1=KNNClassifier(K=4),
                  knn2=KNNClassifier(K=6),
                  tmodel=model)
        metalearner = KNNClassifier()
    end
    Stack(;
        metalearner,
        resampling=CV(;nfolds=2),
        acceleration=resource,
        models...
    )
end

# return a nested stack in which `model` appears at two levels, with
# both layers accelerated using `resource`:
_double_stack(model, resource, isregressor) =
    _stack(_stack(model, resource, isregressor), resource, isregressor)

# the `model` can only be single-target deterministic regressor or
# probabilistic classifier.
function stack_evaluation(
    model,
    resources,
    data...;
    throw=false,
    verbosity=1
)
    L = length(resources)
    message = L > 1 ? "[:accelerated_stack_evaluation] " : "[stack_evaluation] "
    message *=  "Evaluating a nested stack containing model "*
        "using $L different resources. "
    target_scitype = MLJ.target_scitype(model)
    isregressor = AbstractVector{Continuous} <: target_scitype
    measure = isregressor ? LPLoss(2) : BrierScore()

    attempt(MTI.finalize(message, verbosity); throw) do
        es = map(resources) do resource
            stack = _stack(model, resource, isregressor)
            evaluate(
                stack,
                data...;
                measure=measure,
                resampling=Holdout(),
                verbosity=0,
            )
        end |> collect
        ms = map(e->sort(e.per_fold[1]), es)
        m = first(ms)
        @assert all(≈(m), ms[2:end]) ERR_INCONSISTENT_RESULTS
        first(es)
    end
end
