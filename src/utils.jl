# Errors.
const NON_SQUARE_MATRIX = ArgumentError("non-square matrix")
const BAD_OUTPUT_SIZE = DimensionMismatch("output array has incompatible size")
const BAD_INPUT_SIZE = DimensionMismatch("input array has incompatible size")
const HAVE_OFFSET_AXES = ArgumentError("array(s) have offset axes")

"""
    Optimization

is the union of types acceptable for building an algorithm object.

"""
const Optimization = Union{OptimLevel,Type{<:OptimLevel}}

"""
    Algorithm{opt}

is the abstract type of an algorithm executed with optimization level `opt`.
This type may also be used to extend methods from other packages without
"type-piracy".

"""
abstract type Algorithm{opt<:OptimLevel} end
struct RowWise{opt<:OptimLevel} <: Algorithm{opt} end
struct ColumnWise{opt<:OptimLevel} <: Algorithm{opt} end

"""
    RowWise{opt}()
    RowWise(opt=Vectorize)

yields an object representing a row-wise algorithm executed with optimization
level `opt`.

"""
RowWise(opt::Type{<:OptimLevel} = Vectorize) = RowWise{opt}()
RowWise(opt::OptimLevel) = RowWise(typeof(opt))

"""
    ColumnWise{opt}()
    ColumnWise(opt)

yields an object representing a column-wise algorithm executed with
optimization level `opt`.

"""
ColumnWise(opt::Type{<:OptimLevel} = Vectorize) = ColumnWise{opt}()
ColumnWise(opt::OptimLevel) = ColumnWise(typeof(opt))

"""
    is_column_major(A) -> bool

yields whether `A` seems to be stored in column-major order.

"""
is_column_major(A::AbstractMatrix) = (stride(A,1) ≤ stride(A,2))
is_column_major(A::AbstractTriangular) = is_column_major(parent(A))
is_column_major(A::Adjoint) = ! is_column_major(parent(A))
is_column_major(A::Transpose) = ! is_column_major(parent(A))

"""
    is_row_major(A) -> bool

yields whether `A` seems to be stored in row-major order.

"""
is_row_major(A::AbstractMatrix) = ! is_column_major(A)

"""
    promote_eltype(args...) -> T

yields the promoted element type of arguments `args...`.  All arguments must
implement the `eltype` method.

"""
@inline promote_eltype(args...) = promote_type(map(eltype, args)...)

"""
    copy!(dst, src) -> dst

copies the entries of `src` into `dst` and returns `dst`.

"""
function copy!(dst::AbstractArray{<:Any,N},
               src::AbstractArray{<:Any,N}) where {N}
    if dst !== src
        # Only copy entries if not the same object.
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = src[i]
        end
    end
    return dst
end
