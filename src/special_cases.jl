# # HELPERS

function warn_not_testing_these(models)
    "Not testing the following models, as incompatible with testing data:\n"*
        "$models"
end

strip(proxy) = (name=proxy.name, package_name=proxy.package_name)

function actual_proxies(raw_proxies, data, ignore, verbosity)
    proxies = strip.(raw_proxies)
    from_registry = strip.(models(matching(data...)))
    if ignore
        actual_proxies = setdiff(from_registry, proxies)
    else
        actual_proxies = intersect(proxies, from_registry)
        rejected = setdiff(proxies, actual_proxies)
        if !isempty(rejected) && verbosity > 0
            @warn warn_not_testing_these(rejected)
        end
    end
    return actual_proxies
end

function _test(proxies, data; ignore::Bool=false, verbosity=1, kwargs...)
    test(actual_proxies(proxies, data, ignore, verbosity), data...; kwargs...)
end
_test(data; ignore=true, kwargs...) = _test([], data; ignore, kwargs...)


# # SINGLE TARGET CLASSIFICATION

function _make_binary()
    data = MLJ.load_crabs()
    y_, X = unpack(data, ==(:sp), col->col in [:FL, :RW])
    y = coerce(y_, MLJ.OrderedFactor)
    return X, y
end

test_single_target_classifiers(args...; kwargs...) =
    _test(args..., _make_binary(); kwargs...)


# # SINGLE TARGET REGRESSION

function _make_baby_boston()
    data = MLJ.load_boston()
    y, X = unpack(data, ==(:MedV), col->col in [:LStat, :Rm])
    return X, y
end

test_single_target_regressors(args...; kwargs...) =
    _test(args..., _make_baby_boston(); kwargs...)


# # SINGLE TARGET COUNT REGRESSORS

function _make_count()
    X, y_ = _make_baby_boston()
    y = map(η -> round(Int, η), y_)
    return X, y
end

test_single_target_count_regressors(args...; kwargs...) =
    _test(args..., _make_count(); kwargs...)


# # CONTINUOUS TABLE TRANSFORMERS

_make_transformer() = (first(_make_baby_boston()),)

test_continuous_table_transformers(args...; kwargs...) =
    _test(args..., _make_transformer(); kwargs...)
