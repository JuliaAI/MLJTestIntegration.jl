# MLJTest.jl

Package for applying integration tests to models implementing the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) model
interface.

[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](https://github.com/bcgov/repomountie/blob/master/doc/lifecycle-badges.md) [![Build Status](https://github.com/JuliaAI/MLJTest.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJTest.jl/actions) [![Coverage](https://codecov.io/gh/JuliaAI/MLJTest.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLJTest.jl?branch=master) 

# Installation

```julia
using Pkg
Pkg.add("MLJTest")
```

# Usage

This package provides a single method for testing a collection of
`models` (types or named tuples with keys `:name` and `:package_name`)
using the specified training `data`:

```
MLJTest.test(models, data...; mod=Main, level=2, throw=false, verbosity=1) 
    -> failures, summary
```

For detailed documentation, run `using MLJTest; @doc MLJTest.test`.


# Examples

## Testing models in a new MLJ model interface implementation

The following tests the model interface implemented by some model type
`MyClassifier`, as might appear in tests for a package providing that
type:

```
import MLJTest
using Test
X, y = MLJTest.MLJ.make_blobs()
failures, summary = MLJTest.test([MyClassifier, ], X, y, verbosity=1, mod=@__MODULE__)
@test isempty(failures)
```

## Testing models after filtering models in the registry

The following applies comprehensive integration tests to all
regressors provided by the package GLM.jl appearing in the MLJ Model
Registry. Since GLM.jl models are provided through the interface
package `MLJGLMInterface`, this must be in the current environment:

```
Pkg.add("MLJGLMInterface")
import MLJBase, MLJTest
using DataFrames # to view summary
X, y = MLJTest.MLJ.make_regression();
regressors = MLJTest.MLJ.models(matching(X, y)) do m
    m.package_name == "GLM"
end
failures, summary = MLJTest.test(
    regressors, 
    X, 
    y, 
    verbosity=1, 
    mod=@__MODULE__,
    level=3)
summary |> DataFrame
```
