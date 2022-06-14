using Test
using Pkg
using MLJTestIntegration
using MLJTestIntegration.MLJ
using MLJTestIntegration.MLJ.MLJModels
const MTI = MLJTestIntegration

# enable conditional testing of modules by providing test_args
# e.g. `Pkg.test("MLJBase", test_args=["misc"])`

const RUN_ALL_TESTS = isempty(ARGS)
macro conditional_testset(name, expr)
    name = string(name)
    esc(quote
        if RUN_ALL_TESTS || $name in ARGS
            @testset $name $expr
        end
    end)
end

@conditional_testset "attemptors" begin
    include("attemptors.jl")
end

@conditional_testset "test" begin
    include("test.jl")
end
