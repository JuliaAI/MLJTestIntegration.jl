module MLJTestIntegration

const N_MODELS_FOR_REPEATABILITY_TEST = 20

using MLJ
using Pkg
using .Threads
using Test
using NearestNeighborModels # needed for building stacks
import MLJTestInterface
const MTI = MLJTestInterface
import MLJTestInterface.attempt
import MLJTestInterface: make_binary, make_multiclass, make_regression, make_count

include("datasets.jl")
include("attemptors.jl")
include("test.jl")
include("dummy_model.jl")

function __init__()
    global RESOURCES = (CPU1(), CPUThreads())
end

using .DummyModel

end # module
