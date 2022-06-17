"""
    ldiv([opt=Vectorize,] A, b)

yields the result of the left-division `A\b` computed with optimization level
`opt` (use vectorization by default).

"""
ldiv(A::AbstractMatrix, b::AbstractVector) = ldiv(Vectorize, A, b)
ldiv(opt::Optimization, A::AbstractMatrix, b::AbstractVector) =
    ldiv!(opt, ldiv_output(A, b), A, b)
ldiv(alg::Algorithm, A::AbstractMatrix, b::AbstractVector) =
    ldiv!(alg, ldiv_output(A, b), A, b)

"""
    ldiv_output(A, b) -> dst

yields array to store the result of the left-division of `b` by `A`.

"""
ldiv_output(A::AbstractMatrix, b::AbstractVector) =
    Array{promote_eltype(A, b)}(undef, size(A,2))

"""
    ldiv_size([dst,] A, b) -> n

checks the sizes and indexing of arguments to perform left-division `A\b` and
returns the number of rows and columns of `A`.  If `dst` is specified, it is
checked that its size and indexing are suitable to store the result of the
operation.

"""
function ldiv_size(A::AbstractMatrix, b::AbstractVector)
    Base.has_offset_axes(A, b) && throw(HAVE_OFFSET_AXES)
    m, n = size(A)
    m == n || throw(NON_SQUARE_MATRIX)
    length(b) == n || throw(BAD_INPUT_SIZE)
    return n
end

function ldiv_size(dst::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    Base.has_offset_axes(dst, A, b) && throw(HAVE_OFFSET_AXES)
    m, n = size(A)
    m == n || throw(NON_SQUARE_MATRIX)
    length(b) == n || throw(BAD_INPUT_SIZE)
    length(dst) == n || throw(BAD_OUTPUT_SIZE)
    return n
end

"""
    ldiv!(opt, A::AbstractTriangular, b) -> b

overwrites `b` with the result of the left-division `A\b`.  Argument `opt` is
to select the optimization level for loops and/or a specific algorithm to
perform the operation (see [`RowWise`](@ref) or [`ColumnWise`](@ref)).

"""
ldiv!(opt::Union{Optimization,Algorithm}, A::AbstractTriangular, b) =
    ldiv!(opt, b, A, b)

"""
    ldiv!(opt, dst, A, b) -> dst

overwrites `dst` with the result of the left-division `A\b`.  If `A` is a
triangular matrix, `dst` can be the same as `b`.  Argument `opt` is to select
the optimization level for loops and/or a specific algorithm to perform the
operation (see [`RowWise`](@ref) or [`ColumnWise`](@ref)).

"""
function ldiv!(opt::Optimization, dst::AbstractVector,
               A::AbstractMatrix, b::AbstractVector)
    if is_column_major(A)
        ldiv!(ColumnWise(opt), dst, A, b)
    else
        ldiv!(RowWise(opt), dst, A, b)
    end
    return dst
end

# Row-wise left-division by a lower triangular matrix.
#
# Notes:
# - Efficient for row-major storage order.
# - Can be naturally applied in-place.
function ldiv!(::RowWise{opt}, dst::AbstractVector,
               A::LowerTriangular, b::AbstractVector) where {opt}
    L = parent(A) # strip decoration to avoid indexing overheads
    n = ldiv_size(dst, L, b)
    T = promote_eltype(L, b)
    @maybe_inbounds opt for i in 1:n
        s = zero(T)
        @maybe_vectorized opt for j in 1:i-1
            s += L[i,j]*dst[j]
        end
        dst[i] = (b[i] - s)/L[i,i]
    end
    return dst
end

# Row-wise left-division by an upper triangular matrix.
#
# Notes:
# - Efficient for row-major storage order.
# - Can be naturally applied in-place.
function ldiv!(::RowWise{opt}, dst::AbstractVector,
               A::UpperTriangular, b::AbstractVector) where {opt}
    U = parent(A) # strip decoration to avoid indexing overheads
    n = ldiv_size(dst, U, b)
    T = promote_eltype(U, b)
    @maybe_inbounds opt for i in reverse(1:n)
        s = zero(T)
        @maybe_vectorized opt for j in i+1:n
            s += U[i,j]*dst[j]
        end
        dst[i] = (b[i] - s)/U[i,i]
    end
    return dst
end

# The column-wise left-division by a triangular matrix is intrinsically
# an in-place operation.
function ldiv!(alg::ColumnWise, dst::AbstractVector,
               A::AbstractTriangular, b::AbstractVector)
    return ldiv!(alg, A, copy!(dst, b))
end

# Column-wise left-division by a lower triangular matrix.
#
# Notes:
# - Efficient for column-major order.
# - Intrinsically in-place algorithm.
function ldiv!(::ColumnWise{opt},
               A::LowerTriangular, b::AbstractVector) where {opt}
    L = parent(A) # strip decoration to avoid indexing overheads
    n = ldiv_size(L, b)
    @maybe_inbounds opt for j in 1:n
        b_j = b[j]/L[j,j]
        b[j] = b_j
        if !iszero(b_j)
            @maybe_vectorized opt for i in j+1:n
                b[i] -= L[i,j]*b_j
            end
        end
    end
    return b
end

# Column-wise left-division by an upper triangular matrix.
#
# Notes:
# - Efficient for column-major order.
# - Intrinsically in-place algorithm.
function ldiv!(::ColumnWise{opt},
               A::UpperTriangular, b::AbstractVector) where {opt}
    U = parent(A) # strip decoration to avoid indexing overheads
    n = ldiv_size(U, b)
    @maybe_inbounds opt for j in reverse(1:n)
        b_j = b[j]/U[j,j]
        b[j] = b_j
        if !iszero(b_j)
            @maybe_vectorized opt for i in 1:j-1
                b[i] -= U[i,j]*b_j
            end
        end
    end
    return b
end
