# MLJTestIntegration.jl

Package for applying integration tests to models implementing the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) model
interface.

[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](https://github.com/bcgov/repomountie/blob/master/doc/lifecycle-badges.md) [![Build Status](https://github.com/JuliaAI/MLJTestIntegration.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJTestIntegration.jl/actions) [![Coverage](https://codecov.io/gh/JuliaAI/MLJTestIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLJTestIntegration.jl?branch=master) 

# Installation

```julia
using Pkg
Pkg.add("MLJTestIntegration")
```

# Usage

This package provides a method for testing a collection of `models`
(types or named tuples with keys `:name` and `:package_name`) using
the specified training `data`:

```julia
MLJTestIntegration.test(models, data...; mod=Main, level=2, throw=false, verbosity=1) 
    -> failures, summary
```

For detailed documentation, run `using MLJTestIntegration; @doc MLJTestIntegration.test`.

For convenience, a number of specializations of this method are also provided: 

- `test_single_target_classifiers`
- `test_single_target_regressors`
- `test_single_target_count_regressors`
- `test_continuous_table_transformers`

Query the document strings for details, or see
[examples/bigtest/notebook.jl](examples/bigtest/notebook.jl).


# Examples

## Testing models in a new MLJ model interface implementation

The following tests the model interface implemented by some model type
`MyClassifier`, as might appear in tests for a package providing that
type:

```julia
import MLJTestIntegration
using Test
X, y = MLJTestIntegration.MLJ.make_blobs()
failures, summary = MLJTestIntegration.test([MyClassifier, ], X, y, verbosity=1, mod=@__MODULE__)
@test isempty(failures)
```

## Testing models after filtering models in the registry

The following applies comprehensive integration tests to all
regressors provided by the package GLM.jl appearing in the MLJ Model
Registry. Since GLM.jl models are provided through the interface
package `MLJGLMInterface`, this must be in the current environment:

```julia
Pkg.add("MLJGLMInterface")
import MLJBase, MLJTestIntegration
using DataFrames # to view summary
X, y = MLJTestIntegration.MLJ.make_regression();
regressors = MLJTestIntegration.MLJ.models(matching(X, y)) do m
    m.package_name == "GLM"
end

# to test code loading *and* load code:
MLJTestIntegration.test(regressors, X, y, verbosity=1, mod=@__MODULE__, level=4)

# comprehensive tests:
failures, summary =
    MLJTestIntegration.test(regressors, X, y, verbosity=3, mod=@__MODULE__, level=4)

summary |> DataFrame
```
