"""
    attempt(f, message; throw=false)

Return `(f(), "✓") if `f()` executes without throwing an
exception. Otherwise, return `(ex, "×"), where `ex` is the exception
caught. Only truly throw the exception if `throw=true`. 

If `message` is not empty, then it is logged to `Info`, together with
the second return value ("✓" or "×").


"""
function attempt(f, message; throw=false)
    ret = try
        (f(), "✓")
    catch ex
        throw && Base.throw(ex)
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
# for MLJTestIntegration.
load_path(model_type) = MLJ.load_path(model_type)
function load_path(proxy::NamedTuple)
    handle = (name=proxy.name, pkg=proxy.package_name)
    return MLJ.MLJModels.INFO_GIVEN_HANDLE[handle][:load_path]
end

root(load_path) = split(load_path, '.') |> first

function model_type(proxy, mod; throw=false, verbosity=1)
    # check interface package really is in current environment:
    message = "[:model_type] Loading model type "
    model_type, outcome = attempt(finalize(message, verbosity); throw) do
        load_path = MLJTestIntegration.load_path(proxy) # MLJ.load_path(proxy) *****
        load_path_ex = load_path |> Meta.parse
        api_pkg_ex = root(load_path) |> Symbol
        import_ex = :(import $api_pkg_ex)
        quote
            $import_ex
            $load_path_ex
        end |>  mod.eval
    end

    # catch case of interface package not in current environment:
    if outcome == "×" && model_type isa ArgumentError
        # try to get the name of interface package; if this fails we
        # catch the exception thrown but take no further
        # action. Otherwise, we test if the original exception caught
        # above, `model_type`, was triggered because of API package is
        # missing from in environment.
        api_pkg = try
            load_path = MLJTestIntegration.load_path(proxy) # MLJ.load_path(proxy) *****
            api_pkg = root(load_path)
        catch
            nothing
        end
        if !isnothing(api_pkg) &&
               api_pkg != "unknown" &&
               contains(model_type.msg, "$api_pkg not found in")
            Base.throw(model_type)
        end
    end

    return model_type, outcome
end

function model_instance(model_type; throw=false, verbosity=1)
    message = "[:model_instance] Instantiating default model "
    attempt(finalize(message, verbosity); throw)  do
        model_type()
    end
end

function fitted_machine(model, data...; throw=false, verbosity=1)
    message = "[:fitted_machine] Fitting machine "
    attempt(finalize(message, verbosity); throw)  do
        mach = machine(model, data...)
        fit!(mach, verbosity=-1)
    end
end

function operations(fitted_machine, data...; throw=false, verbosity=1)
    message = "[:operations] Calling `predict`, `transform` and/or `inverse_transform` "
    attempt(finalize(message, verbosity); throw)  do
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

function threshold_prediction(model, data...; throw=false, verbosity=1)
    message = "[:threshold_predictor] Calling fit!/predict for threshold predictor "*
        "test) "
    attempt(finalize(message, verbosity); throw) do
        tmodel = BinaryThresholdPredictor(model)
        mach = machine(tmodel, data...)
        fit!(mach, verbosity=0)
        predict(mach, first(data))
    end
end

function evaluation(measure, model, data...; throw=false, verbosity=1)
    message = "[:evaluation] Evaluating performance "
    attempt(finalize(message, verbosity); throw) do
        evaluate(model, data...;
                 measure=measure,
                 resampling=Holdout(),
                 verbosity=0)
    end
end

function tuned_pipe_evaluation(measure, model, data...; throw=false, verbosity=1)
    message = "[:tuned_pipe_evaluation] Evaluating perfomance in a tuned pipeline "
    attempt(finalize(message, verbosity); throw) do
        pipe = identity |> model
        tuned_pipe = TunedModel(models=[pipe,],
                                measure=measure)
        evaluate(tuned_pipe, data...;
                 measure=measure,
                 verbosity=0);
    end
end

function ensemble_prediction(model, data...; throw=false, verbosity=1)
    attempt(finalize("[:ensemble_prediction] Ensembling ", verbosity); throw) do
        imodel = EnsembleModel(model=model,
                               n=2)
        mach = machine(imodel, data...)
        fit!(mach, verbosity=0)
        predict(mach, first(data))
    end
end

function iteration_prediction(measure, model, data...; throw=false, verbosity=1)
    message =  "[:iteration_prediction] Iterating with controls "
    attempt(finalize(message, verbosity); throw) do
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
