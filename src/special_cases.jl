# # HELPERS

_strip(proxy) = (name=proxy.name, package_name=proxy.package_name)

function _filter(proxies, bad)
    sbad = _strip.(bad)
    filter(proxies) do proxy
        !(_strip(proxy) in sbad)
    end
end

# fallback:
function _test(data, ignore; kwargs...)
    proxies = _filter(models(matching(data...)), ignore)
    test(proxies, data...; kwargs...)
end

# when there are no models to exclude:
function _test(data, ignore::Nothing; kwargs...)
    proxies = models(matching(data...))
    test(proxies, data...; kwargs...)
end


# # SINGLE TARGET CLASSIFICATION

function _make_binary()
    data = MLJ.load_crabs()
    y_, X = unpack(data, ==(:sp), col->col in [:FL, :RW])
    y = coerce(y_, MLJ.OrderedFactor)
    return X, y
end

test_single_target_classifiers(; ignore=nothing, kwargs...) =
    _test(_make_binary(), ignore; kwargs...)


# # SINGLE TARGET REGRESSION

function _make_baby_boston()
    data = MLJ.load_boston()
    y, X = unpack(data, ==(:MedV), col->col in [:LStat, :Rm])
    return X, y
end

test_single_target_regressors(; ignore=nothing, kwargs...) =
    _test(_make_baby_boston(), ignore; kwargs...)


# # SINGLE TARGET COUNT REGRESSORS

function _make_count()
    X, y_ = _make_baby_boston()
    y = map(η -> round(Int, η), y_)
    return X, y
end

test_single_target_count_regressors(; ignore=nothing, kwargs...) =
    _test(_make_count(), ignore; kwargs...)


# # CONTINUOUS TABLE TRANSFORMERS

_make_transformer() = (first(_make_baby_boston()),)

test_continuous_table_transformers(; ingore=nothing, kwargs...) =
    _test(_make_transformer(), ignore; kwargs...)
