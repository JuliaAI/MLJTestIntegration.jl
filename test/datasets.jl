using Test
using MLJTestIntegration
using MLJ
using MLJModels
const MTI = MLJTestIntegration

Rgs = @load KNNRegressor pkg=NearestNeighborModels verbosity=0
Clf = @load KNNClassifier pkg= NearestNeighborModels verbosity=0
Trf = @load Standardizer pkg=MLJModels verbosity=0

Rgs_proxy = models() do m
    m.name == "KNNRegressor" && m.package_name == "NearestNeighborModels"
end |> only

# a Static model with two arguments:
struct StaticModel <: Static end
MLJ.input_scitype(::Type{<:StaticModel}) =
    Tuple{Table(Continuous), AbstractVector{<:Finite}}

sets = MTI.datasets(Rgs)
@test length(sets) == 2
@test scitype.(sets) |> Set ==
    scitype.([
        MTI.make_regression(),
        MTI.make_regression(row_table=true),
    ]) |> Set

sets = MTI.datasets(Rgs_proxy)
@test length(sets) == 2
@test scitype.(sets) |> Set ==
    scitype.([
        MTI.make_regression(),
        MTI.make_regression(row_table=true),
    ]) |> Set

sets = MTI.datasets(Rgs_proxy)
@test length(sets) == 2
@test scitype.(sets) |> Set ==
    scitype.([
        MTI.make_regression(),
        MTI.make_regression(row_table=true),
    ]) |> Set

sets = MTI.datasets(Clf)
@test length(sets) == 4
@test scitype.(sets) |> Set ==
    scitype.([
        MTI.make_binary(),
        MTI.make_multiclass(),
        MTI.make_binary(row_table=true),
        MTI.make_multiclass(row_table=true),
    ]) |> Set

sets = MTI.datasets(Trf)
@test length(sets) == 2
@test scitype(sets[1][1]) == scitype(MTI.make_regression()[1])

sets = MTI.datasets(StaticModel)
@test length(sets) == 4
@test scitype.(sets) |> Set ==
    scitype.([
        MTI.make_binary(),
        MTI.make_multiclass(),
        MTI.make_binary(row_table=true),
        MTI.make_multiclass(row_table=true),
    ]) |> Set

true
