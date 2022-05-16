# MLJTest.jl

Package for applying integration tests to models implementing the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) model
interface.

| Linux         | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/MLJTestInterface.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJTestInterface.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/MLJTestInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLJTestInterface.jl?branch=master) |

# Installation

```julia
using Pkg
Pkg.add("MLJTest")
```

# Usage

To start using the package, run `using MLJTest`. The package provides
a single public method, `MLJTest.test`. Query it's document string for
details.

```julia
using MLJTest
@doc MLJTest.test
```

