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

# TODO: Instead, in ****** below, use `MLJ.load_path`, after MLJModels
# is updated to 0.16. And delete the two methods immediately
# following. What's required will already be in MLJModels 0.15.10, but
# the current implementation avoids an explicit MLJModels dependency
# for MLJTest.
load_path(model_type) = MLJ.load_path(model_type)
function load_path(proxy::NamedTuple)
    handle = (name=proxy.name, pkg=proxy.package_name)
    return MLJ.MLJModels.INFO_GIVEN_HANDLE[handle][:load_path]
end

function model_type(proxy, mod; verbosity=1)
    # check interface package really is in current environment:
    message = "Loading model type "
    model_type, outcome = attempt(finalize(message, verbosity)) do
        load_path = MLJTest.load_path(proxy) # MLJ.load_path(proxy) *****
        path_components = split(load_path, '.')
        api_pkg_ex = first(path_components) |> Symbol
        import_ex = :(import $api_pkg_ex)
        path_ex = load_path |> Meta.parse
        quote
            $import_ex
            $path_ex
        end |>  mod.eval
    end

    # catch case of interface package not in current environment:
    if outcome == "×" &&
        model_type isa ArgumentError &&
        contains(model_type.msg, "not found in current path")
        throw(model_type)
    end

    return model_type, outcome
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
