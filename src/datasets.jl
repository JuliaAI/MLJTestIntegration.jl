const SUPERVISED_DATASETS = Any[
    MLJTestIntegration.make_regression(),
    MLJTestIntegration.make_binary(),
    MLJTestIntegration.make_multiclass(),
    MLJTestIntegration.make_count(),
    MLJTestIntegration.make_regression(row_table=true),
    MLJTestIntegration.make_binary(row_table=true),
    MLJTestIntegration.make_multiclass(row_table=true),
    MLJTestIntegration.make_count(row_table=true),
]
const DATASETS = Any[
    SUPERVISED_DATASETS...,
    (first(MLJTestIntegration.make_regression()),),
    (first(MLJTestIntegration.make_regression(row_table=true)),),
]

# Sometimes `(X, )` is a table, when `X` is a table, which leads to `scitype((X,)) =
# Table(...)` where `Tuple{scitype(X)}` is wanted. So, as we know data is *always* going
# to be a tuple, we use `_scitype(data)` instead of `scitype(data)`, where:
_scitype(data) = Tuple{scitype.(data)...}

const DATASET_SCITYPES = _scitype.(DATASETS)

_input_scitype(modeltype) = input_scitype(modeltype)
_input_scitype(modelproxy::NamedTuple) = modelproxy.input_scitype
_fit_data_scitype(modeltype) = fit_data_scitype(modeltype)
_fit_data_scitype(modelproxy::NamedTuple) = modelproxy.fit_data_scitype
is_static(modeltype) = modeltype <: Static
is_static(modelproxy::NamedTuple) = modelproxy.fit_data_scitype == Tuple{}


"""
    datasets(model)

Return a list of datasets available in MLJTestIntegration that appear to be valid for
testing with the specified `model`, which is a model type or model proxy.

"""
function datasets(model)
    datasets = []
    T = is_static(model) ?
        _input_scitype(model) :
        _fit_data_scitype(model)
    for (dataset, datascitype) in zip(DATASETS, DATASET_SCITYPES)
        datascitype <: T && push!(datasets, dataset)
    end
    return datasets
end
