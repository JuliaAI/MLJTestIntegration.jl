# # HELPERS

const DOC_AS_ABOVE =
    """
    The same as above, but restricting to those registered models that are also
    in `models`, a vector of named tuples inlcuding `:name` and
    `:package_name` as keys. If `ignore=true`, then instead apply tests
    to all models but *excluding* those in `models`.
    """

function warn_not_testing_these(models)
    "Not testing the following models, as incompatible with testing data:\n"*
        "$models"
end

strip(proxy) = (name=proxy.name, package_name=proxy.package_name)

function actual_proxies(raw_proxies, data, ignore, verbosity)
    if !(raw_proxies isa Vector)
        raw_proxies = [raw_proxies, ]
    end
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
    test(actual_proxies(proxies, data, ignore, verbosity), data...; verbosity, kwargs...)
end
_test(data; ignore=true, kwargs...) = _test([], data; ignore, kwargs...)


# # SINGLE TARGET CLASSIFICATION

function _make_binary()
    data = MLJ.load_crabs()
    y_, X = unpack(data, ==(:sp), col->col in [:FL, :RW])
    y = coerce(y_, MLJ.OrderedFactor)
    return X, y
end

"""
    MLJTestIntegration.test_single_target_classifiers(; keyword_options...)

Apply [`MLJTestIntegration.test`](@ref) to all models in the MLJ Model
Registry that support single target classification, using a
two-feature selection of the Crab dataset. The specifed
`keyword_options` are passed to onto to `MLJTestIntegration.test`.

    MLJTestIntegration.test_single_target_classifiers(models; ignore=false, keyword_options...)

$DOC_AS_ABOVE

"""
test_single_target_classifiers(args...; kwargs...) =
    _test(args..., _make_binary(); kwargs...)


# # SINGLE TARGET REGRESSION

function _make_baby_boston()
    data = MLJ.load_boston()
    y, X = unpack(data, ==(:MedV), col->col in [:LStat, :Rm])
    return X, y
end

"""
    MLJTestIntegration.test_single_target_regressors(; keyword_options...)

Apply [`MLJTestIntegration.test`](@ref) to all models in the MLJ Model
Registry that support single target regression, using a two-feature
selection of the Boston dataset. The specifed `keyword_options` are
passed onto `MLJTestIntegration.test`.

    MLJTestIntegration.test_single_target_regressors(models; ignore=false, keyword_options...)

$DOC_AS_ABOVE

"""
test_single_target_regressors(args...; kwargs...) =
    _test(args..., _make_baby_boston(); kwargs...)


# # SINGLE TARGET COUNT REGRESSORS

function _make_count()
    X, y_ = _make_baby_boston()
    y = map(η -> round(Int, η), y_)
    return X, y
end

"""
    MLJTestIntegration.test_single_count_regressors(; keyword_options...)

Apply [`MLJTestIntegration.test`](@ref) to all models in the MLJ Model
Registry that support single target count regressors
(`AbstractVector{Count}` target scitype) using a two-feature selection
of the Boston datasetand the target variable discretized.  The
specifed `keyword_options` are passed onto
`MLJTestIntegration.test`.

    MLJTestIntegration.test_single_target_regressors(models; ignore=false, keyword_options...)

$DOC_AS_ABOVE

"""
test_single_target_count_regressors(args...; kwargs...) =
    _test(args..., _make_count(); kwargs...)


# # CONTINUOUS TABLE TRANSFORMERS

_make_transformer() = (first(_make_baby_boston()),)

"""
    test_continuous_table_transformers(; keyword_options...)

Apply [`MLJTestIntegration.test`](@ref) to all models in the MLJ
Model Registry that train on a single table with continuous features,
using a two-feature selection of the Boston dataset.  The specifed
`keyword_options` are passed onto `MLJTestIntegration.test`.

    test_continuous_table_transformers(models; ignore=false, keyword_options...)

$DOC_AS_ABOVE
"""
test_continuous_table_transformers(args...; kwargs...) =
    _test(args..., _make_transformer(); kwargs...)
