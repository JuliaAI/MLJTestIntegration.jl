module MLJTest

using MLJ
using Pkg

include("attemptors.jl")
include("test.jl")
include("special_cases.jl")
include("dummy_model.jl")

using .DummyModel

end # module
