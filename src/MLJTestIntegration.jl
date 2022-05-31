module MLJTestIntegration

const N_MODELS_FOR_REPEATABILITY_TEST = 3

using MLJ
using Pkg
using .Threads
using Test

include("attemptors.jl")
include("test.jl")
include("special_cases.jl")
include("dummy_model.jl")

function __init__()
    global RESOURCES = (CPU1(), CPUThreads())
    @info "Testing with $(nthreads()) threads. "
end

using .DummyModel

end # module
