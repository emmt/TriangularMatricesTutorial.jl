# TriangularMatricesTutorial [![Build Status](https://github.com/emmt/TriangularMatricesTutorial.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/TriangularMatricesTutorial.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/TriangularMatricesTutorial.jl?svg=true)](https://ci.appveyor.com/project/emmt/TriangularMatricesTutorial-jl) [![Coverage](https://codecov.io/gh/emmt/TriangularMatricesTutorial.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/TriangularMatricesTutorial.jl)

This repository is to experiment algorithms involving triangular matrices.  The
main objective is to provide reference algorithms that can be resonnably fast
and yet whose code remains readable.  In some sense, this shows how Julia can
be good at helping you to write efficient code that is almost as simple as its
mathematical counterpart.  We encourage users to have a look at
[`src/lmul.jl`](src/lmul.jl) which implements left-multiplication and
[`src/ldiv.jl`](src/ldiv.jl) which implements left-division by triangular
matrices.

The following is provided:

- In-place row-wise and column-wise left-multiplication (`lmul` and `lmul!`
  methods) and left-division (`ldiv` and `ldiv!` methods) of a vector by a
  triangular matrix, by its adjoint, or by its transpose.

This package is largely a work in progress, and we consider implementing the
following things:

- Algorithms for the Cholesky decomposition.

- Linear least-squares.

- Sparse Cholesky factors with optional permutation.


## Usage

Left-multiplication and left-division of vector `b` by triangular matrix `A`
can be computed by:

```julia
using TriangularMatricesTutorial
lmul(A, b) # -> A*b
ldiv(A, b) # -> A\b
```

Nothing terrible, yet you can choose a row-wise or column-wise method:

```julia
lmul(RowWise(), A, b) # -> A*b computed by a row-wise algorithm
ldiv(RowWise(), A, b) # -> A\b computed by a row-wise algorithm
lmul(ColumnWise(), A, b) # -> A*b computed by a column-wise algorithm
ldiv(ColumnWise(), A, b) # -> A\b computed by a column-wise algorithm
```

Which algorithm is the fastest depends largely on the storage order of the
entries of `A` (colum-major or row-major).  Column-wise methods addresses the
matrix entries in column-major order so you might think that they are the
methods of choice since Julia stores regular arrays in column-major order.
Julia is however quite flexible about the definition of a matrix so the most
suitable method depends on the type of `A`.  For example, Julia makes use of
annotations to lazily represent the transpose or the adjoint of a matrix, which
lazily changes the storage order of entries.  If no specific row-/column-wise
method is chosen, `lmul` and `ldiv` will automatically peek a suitable one
(hopefully).  This automatic choice may not always be accurate (e.g. gor small
matrices) and you may want to try a different one.  You may also want to play
with this option.  This is precisely one of the purpose of this package!

The `RowWise` and `ColumnWise` methods take an optional argument to specify the
optimization level for loops:

- `Debug` to perform bounds checking.  This yields the safest but usually
  the slowest code.

- `InBounds` to assume that all indices are correct and avoid bounds checking.
  This yields code that is a bit faster than with `Debug`, although Julia has a
  tendency to be very smart and fast for bounds checking.

- `Vectorize` to perform SIMD loop optimization.  This also avoid bounds
  checking.  This usually yields the fastest code.

The default optimization level is `Vectorize`, but you may want to play with
this parameter.

If you want to completely avoid allocations, you may use the in-place versions
of the `lmul` and `ldiv` methods:

```julia
lmul!(opt, [dst=b,] A, b) -> dst
ldiv!(opt, [dst=b,] A, b) -> dst
```

which overwrite `dst` with the result of `A*b` and of `A\b`.  If `dst` is not
specified and if `A` is a triangular matrix, the operation can be applied
in-place, overwriting `b`.  Argument `opt` is either an optimization level
(e.g. `Debug`, `InBounds`, or `Vectorize`) or an instance of `RowWise` or
`ColumnWise`.  This argument is required to avoid type-piracy.  If you do not
specify it, you will be calling the methods implemented by Julia.

The following table summarizes some results for `L` lower triangular of size
`n×n` with for `n=100` and elements of type `T=Float64` (the number of
operations is `n²` for multiplying by a triangular matrix, `2n²-n` for a full
square matrix).

| Code                                                     | Time     | Power       |
|:---------------------------------------------------------|:---------|:------------|
| `@btime lmul!($(RowWise(Debug)),$dst,$L,$b);`            | 3.369 μs | 2.97 Gflops |
| `@btime lmul!($(RowWise(InBounds)),$dst,$L,$b);`         | 2.783 μs | 3.59 Gflops |
| `@btime lmul!($(RowWise(Vectorize)),$dst,$L,$b);`        | 4.394 μs | 2.28 Gflops |
| `@btime lmul!($(ColumnWiseWise(Debug)),$dst,$L,$b);`     | 2.895 μs | 3.45 Gflops |
| `@btime lmul!($(ColumnWiseWise(InBounds)),$dst,$L,$b);`  | 1.338 μs | 7.47 Gflops |
| `@btime lmul!($(ColumnWiseWise(Vectorize)),$dst,$L,$b);` | 1.303 μs | 7.67 Gflops |

For the left-division:

| Code                                                     | Time     | Power       |
|:---------------------------------------------------------|:---------|:------------|
| `@btime ldiv!($dst,$L,$b);`                              | 1.596 μs | 6.27 Gflops |
| `@btime ldiv!($(RowWise(Debug)),$dst,$L,$b);`            | 3.290 μs | 3.04 Gflops |
| `@btime ldiv!($(RowWise(InBounds)),$dst,$L,$b);`         | 2.832 μs | 3.53 Gflops |
| `@btime ldiv!($(RowWise(Vectorize)),$dst,$L,$b);`        | 3.954 μs | 2.53 Gflops |
| `@btime ldiv!($(ColumnWiseWise(Debug)),$dst,$L,$b);`     | 3.692 μs | 2.71 Gflops |
| `@btime ldiv!($(ColumnWiseWise(InBounds)),$dst,$L,$b);`  | 1.308 μs | 7.65 Gflops |
| `@btime ldiv!($(ColumnWiseWise(Vectorize)),$dst,$L,$b);` | 1.411 μs | 7.09 Gflops |

The first line is using the method in `LinearAlgebra` (based on BLAS) which is
about 20% slower than the best method (here `ColumnWiseWise(InBounds)`).  Not
too bad for pure Julia code.  Note that BLAS will win for large matrix
(typically for `n` larger than a few hundreds).


## Implementation notes

We use [`MayOptimize`]() package for flexible loop optimizations and specific
types (based on `TriangularMatricesTutorial.Algorithm`) to avoid type-piracy.

Julia annotates matrix for lazy transpose and adjoint and to indicate specific
matrix structure (`LowerTriangular`, `UnitLowerTriangular`, etc.).  The
triangle matrix annotation has some overheads in indexing operations (to ensure
that the results of the `getindex` and `setindex!` calls are consistent with
the structure of the matrix).  To avoid these overheads, we unveil the parent
matrix of triangular matrices and account for the specific structure by
restricting the indices ised.  Adjoint and transpose are handled without
penalty so this is left unchanged (it reduces a lot the number of algorithms to
code).

Since Julia 1.6, the triangular annotation is always kept on top of the other
annotations so unveilling it is easy but this has to be explicitely done for
other.

In spite of this, implemented algorithms remain quite readable and simple and
yet efficient.

## Installation
