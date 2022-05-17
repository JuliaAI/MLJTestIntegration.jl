# MLJTest.jl

Package for applying integration tests to models implementing the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) model
interface.

[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](https://github.com/bcgov/repomountie/blob/master/doc/lifecycle-badges.md)

| Linux         | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/MLJTest.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJTest.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/MLJTest.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLJTest.jl?branch=master) |

# Installation

```julia
using Pkg
Pkg.add("MLJTest")
```

# Usage

This package provides a single method for testing a collection of
`models` (types or MLJ Model Registry entries) using training `data`:

```
MLJTest.test(models, data...; verbosity=1, mod=Main, loading_only=false) -> failures, summary
```

See the method document string for details. 
