"""
    lmul([opt=Vectorize,] A, b)

yields the result of the left-multiplication `A*b` computed with optimization
level `opt` (use vectorization by default).

"""
lmul(A::AbstractMatrix, b::AbstractVector) = lmul(Vectorize, A, b)
lmul(opt::Optimization, A::AbstractMatrix, b::AbstractVector) =
    lmul!(opt, lmul_output(A, b), A, b)
lmul(alg::Algorithm, A::AbstractMatrix, b::AbstractVector) =
    lmul!(alg, lmul_output(A, b), A, b)

"""
    lmul_output(A, b) -> dst

yields array to store the result of the left-multiplication of `b` by `A`.

"""
lmul_output(A::AbstractMatrix, b::AbstractVector) =
    Array{promote_eltype(A, b)}(undef, size(A,1))

"""
    lmul_size([dst,] A, b) -> m, n

checks the sizes and indexing of arguments to perform left-multiplication `a*b`
and returns the number of rows and columns of `A`.  If `dst` is specified, it
is checked that its size and indexing are suitable to store the result of the
operation.

"""
function lmul_size(A::AbstractMatrix, b::AbstractVector)
    Base.has_offset_axes(A, b) && throw(HAVE_OFFSET_AXES)
    m, n = size(A)
    length(b) == n || throw(BAD_INPUT_SIZE)
    return (m, n)
end

function lmul_size(dst::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    Base.has_offset_axes(dst, A, b) && throw(HAVE_OFFSET_AXES)
    m, n = size(A)
    length(b) == n || throw(BAD_INPUT_SIZE)
    length(dst) == m || throw(BAD_OUTPUT_SIZE)
    return (m, n)
end

"""
    lmul!(opt, A::AbstractTriangular, b) -> b

overwrites `b` with the result of the left-multiplication `A*b`.  Argument
`opt` is to select the optimization level for loops and/or a specific algorithm
to perform the operation (see [`RowWise`](@ref) or [`ColumnWise`](@ref)).

"""
lmul!(opt::Union{Optimization,Algorithm}, A::AbstractTriangular, b) =
    lmul!(opt, b, A, b)

"""
    lmul!(opt, dst, A, b) -> dst

overwrites `dst` with the result of the left-multiplication `A*b`.  If `A` is a
triangular matrix, `dst` can be the same as `b`.  Argument `opt` is to select
the optimization level for loops and/or a specific algorithm to perform the
operation (see [`RowWise`](@ref) or [`ColumnWise`](@ref)).

"""
function lmul!(opt::Optimization, dst::AbstractVector,
               A::AbstractMatrix, b::AbstractVector)
    if is_column_major(A)
        lmul!(ColumnWise(opt), dst, A, b)
    else
        lmul!(RowWise(opt), dst, A, b)
    end
    return dst
end

# Row-wise left-multiplication by a lower triangular matrix.
#
# Notes:
# - Efficient for row-major storage order.
# - Can be naturally applied in-place.
function lmul!(::RowWise{opt}, dst::AbstractVector,
               A::LowerTriangular, b::AbstractVector) where {opt}
    L = parent(A) # strip decoration to avoid indexing overheads
    m, n = lmul_size(dst, L, b)
    T = promote_eltype(L, b)
    @maybe_inbounds opt for i in reverse(1:n)
        s = zero(T)
        @maybe_vectorized opt for j in 1:i
            s += L[i,j]*b[j]
        end
        dst[i] = s
    end
    return dst
end

# Row-wise left-multiplication by an upper triangular matrix.
#
# Notes:
# - Efficient for row-major storage order.
# - Can be naturally applied in-place.
function lmul!(::RowWise{opt}, dst::AbstractVector,
               A::UpperTriangular, b::AbstractVector) where {opt}
    U = parent(A) # strip decoration to avoid indexing overheads
    m, n = lmul_size(dst, U, b)
    T = promote_eltype(U, b)
    @maybe_inbounds opt for i in 1:n
        s = zero(T)
        @maybe_vectorized opt for j in i:n
            s += U[i,j]*b[j]
        end
        dst[i] = s
    end
    return dst
end

# The column-wise left-multiplication by a triangular matrix is intrinsically
# an in-place operation.
function lmul!(alg::ColumnWise, dst::AbstractVector,
               A::AbstractTriangular, b::AbstractVector)
    return lmul!(alg, A, copy!(dst, b))
end

# Column-wise left-multiplication by a lower triangular matrix.
#
# Notes:
# - Efficient for column-major order.
# - Intrinsically in-place algorithm.
function lmul!(::ColumnWise{opt},
               A::LowerTriangular, b::AbstractVector) where {opt}
    L = parent(A) # strip decoration to avoid indexing overheads
    m, n = lmul_size(L, b)
    T = promote_eltype(L, b)
    @maybe_inbounds opt for j in reverse(1:n)
        if !iszero(b[j])
            b_j = convert(T, b[j])::T
            b[j] = L[j,j]*b_j
            @maybe_vectorized opt for i in j+1:n
                b[i] += L[i,j]*b_j
            end
        end
    end
    return b
end

# Column-wise left-multiplication by an upper triangular matrix.
#
# Notes:
# - Efficient for column-major order.
# - Intrinsically in-place algorithm.
function lmul!(::ColumnWise{opt},
               A::UpperTriangular, b::AbstractVector) where {opt}
    U = parent(A) # strip decoration to avoid indexing overheads
    m, n = lmul_size(U, b)
    T = promote_eltype(U, b)
    @maybe_inbounds opt for j in 1:n
        if !iszero(b[j])
            b_j = convert(T, b[j])::T
            @maybe_vectorized opt for i in 1:j-1
                b[i] += U[i,j]*b_j
            end
            b[j] = U[j,j]*b_j
        end
    end
    return b
end
