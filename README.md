# MLJTestIntegration.jl

Package for applying integration tests to models implementing the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) model
interface.

For to apply minimal tests for an implementations of the MLJ model interface, use
[MLJTestInterface.jl](https://github.com/JuliaAI/MLJTestInterface.jl) instead.

However, *implementations are now encouraged to include the comprehensive tests* provided
in this package instead.

[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](https://github.com/bcgov/repomountie/blob/master/doc/lifecycle-badges.md) [![Build Status](https://github.com/JuliaAI/MLJTestIntegration.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJTestIntegration.jl/actions) [![Coverage](https://codecov.io/gh/JuliaAI/MLJTestIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLJTestIntegration.jl?branch=master)

# Installation

```julia
using Pkg
Pkg.add("MLJTestIntegration")
```

# Usage

This package provides a method for testing a collection of `models`
(more, specifically model types) using the specified training `data`:

```julia
MLJTestIntegration.test(models, data...; mod=Main, level=2, throw=false, verbosity=1)
    -> failures, summary
```

For pure Julia models, `level=4` (the maximimum) is recommended. Otherwise `level=3` is
recommended.

For detailed documentation, run `using MLJTestIntegration; @doc MLJTestIntegration.test`.


# Example

The following tests the model interface implemented by the `DecisionTreeClassifier` model
implemented in the package MLJDecisionTreeInterface.jl.

```julia
import MLJDecisionTreeInterface
import MLJTestIntegration
using Test
X, y = MLJTestIntegration.make_multiclass()
failures, summary = MLJTestIntegration.test(
    [MLJDecisionTreeInterface.DecisionTreeClassifier, ],
    X, y,
    verbosity=0, # set to 2 when debugging
    throw=false, # set to `true` when debugging
    mod=@__MODULE__,
)
@test isempty(failures)
```

For models supporting tabular data `X`, we recommend repeating the tests with `(X, y)`
replaced with `(Tables.rowtable(X)), y)` to ensure support for different table types,
which new implementation sometimes fail to do.

# Datasets

The following commands generate datasets of the form `(X, y)` suitable for integration
tests:

- `MLJTestIntegration.make_binary`

- `MLJTestIntegration.make_multiclass`

- `MLJTestIntegration.make_regression`

- `MLJTestIntegration.make_count`

The binary and multiclass sets are deliberately small and will catch failures to properly
track target class pools, a common gotcha in new implementations.
