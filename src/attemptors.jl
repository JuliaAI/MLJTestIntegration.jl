str(model_metadata) = "$(model_metadata.name) from $(model_metadata.package_name)"

"""
    attempt(f, message="")

Return `(f(), "✓") if `f()` executes without throwing an
exception. Otherwise, return `(ex, "×"), where `ex` is the exception
thrown.

If `message` is not empty, then it is logged to `Info`, together with
the second return value ("✓" or "×").

"""
function attempt(f, message="")
    ret = try
        (f(), "✓")
    catch ex
        (ex, "×")
    end
    isempty(message) || @info message*last(ret)
    return ret
end

finalize(message, verbosity) = verbosity < 2 ? "" : message


# # ATTEMPTORS

function model_type(proxy, mod; verbosity=1)
    message = "Loading model type "
    attempt(finalize(message, verbosity)) do
        load_path = MLJ.load_path(proxy)
        import_ex = "import "*load_path |> Meta.parse
        path_ex = load_path |> Meta.parse
        quote
            $import_ex
            $path_ex
        end |>  mod.eval
    end
end

function model_instance(model_type; verbosity=1)
    message = "Instantiating default model "
    attempt(finalize(message, verbosity))  do
        model_type()
    end
end

function fitted_machine(model, data...; verbosity=1)
    message = "Fitting machine "
    attempt(finalize(message, verbosity))  do
        mach = machine(model, data...)
        fit!(mach, verbosity=-1)
    end
end

function operations(fitted_machine, data...; verbosity=1)
    message = "Calling `predict`, `transform` and/or `inverse_transform` "
    attempt(finalize(message, verbosity))  do
        operations = String[]
        methods = MLJ.implemented_methods(fitted_machine.model)
        if :predict in methods
            predict(fitted_machine, first(data))
            push!(operations, "predict")
        end
        if :transform in methods
            W = transform(fitted_machine, first(data))
            push!(operations, "transform")
            if :inverse_transform in methods
                inverse_transform(fitted_machine, W)
                push!(operations, "inverse_transform")
            end
        end
        join(operations, ", ")
    end
end

function threshold_prediction(model, data...; verbosity=1)
    message = "Calling fit!/predict for threshold predictor "
    attempt(finalize(message, verbosity)) do
        tmodel = BinaryThresholdPredictor(model)
        mach = machine(tmodel, data...)
        fit!(mach, verbosity=0)
        predict(mach, first(data))
    end
end

function evaluation(measure, model, data...; verbosity=1)
    message = "Evaluating performance "
    attempt(finalize(message, verbosity)) do
        evaluate(model, data...;
                 measure=measure,
                 resampling=Holdout(),
                 verbosity=0)
    end
end

function tuned_pipe_evaluation(measure, model, data...; verbosity=1)
    message = "Evaluating perfomance in a tuned pipeline "
    attempt(finalize(message, verbosity)) do
        pipe = Standardizer() |> model
        tuned_pipe = TunedModel(models=[pipe,],
                                measure=measure)
        evaluate(tuned_pipe, data...;
                 measure=measure,
                 verbosity=0);
    end
end

function ensemble_prediction(model, data...; verbosity=1)
    attempt(finalize("Ensembling ", verbosity)) do
        imodel = EnsembleModel(model=model,
                               n=2)
        mach = machine(imodel, data...)
        fit!(mach, verbosity=0)
        predict(mach, first(data))
    end
end

function iteration_prediction(measure, model, data...; verbosity=1)
    message =  "Iterating with controls "
    attempt(finalize(message, verbosity)) do
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
