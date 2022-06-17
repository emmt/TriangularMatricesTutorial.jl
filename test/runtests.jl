module TestingTriangularMatricesTutorial

using TriangularMatricesTutorial
using Test
using MayOptimize
using LinearAlgebra

# Narrow is used to reduce integer range for generating small random integers
# that can be converted to a widen type for exact calculations.
narrow(T::Type) = T
narrow(::Type{<:Union{Signed,AbstractFloat}}) = Int8
narrow(::Type{<:Unsigned}) = UInt8
narrow(::Type{Complex{T}}) where {T} = Complex{narrow(T)}

brief(::typeof(identity),  S::Type{<:AbstractMatrix}, T::Type) = brief(S,T)
brief(::typeof(transpose), S::Type{<:AbstractMatrix}, T::Type) = "transpose($(brief(S,T)))"
brief(::typeof(adjoint),   S::Type{<:AbstractMatrix}, T::Type) = "adjoint($(brief(S,T)))"
brief(::Type{<:LowerTriangular}, T::Type) = "LowerTriangular{$T}"
brief(::Type{<:UpperTriangular}, T::Type) = "UpperTriangular{$T}"
brief(::Type{<:UnitLowerTriangular}, T::Type) = "UnitLowerTriangular{$T}"
brief(::Type{<:UnitUpperTriangular}, T::Type) = "UnitUpperTriangular{$T}"

@testset "TriangularMatricesTutorial" begin
    n = 7
    TYPES = (Float64, Complex{Float64})
    STRUCTS = (LowerTriangular, UpperTriangular)
    FUNCS = (identity, transpose, adjoint)
    @testset "$(brief(f,s,T))" for T in TYPES, s in STRUCTS, f in FUNCS
        A = convert(Matrix{T}, rand(narrow(T), n, n))
        b = convert(Vector{T}, rand(narrow(T), n))
        b_cpy = copy(b) # to check that b was left unchanged
        dst = similar(b)

        # Make the triangular parts of A invertible.
        for i in 1:n
            if A[i,i] == 0
                A[i,i] = 1
            end
        end

        # Structured matrix.
        S = f(s(A))
        Sb = S*b
        Sb_cpy = copy(Sb)

        # Test out-of-place operations.
        @test lmul(S,b) == Sb
        @test b == b_cpy
        @test lmul(RowWise(Debug),S,b) == Sb
        @test b == b_cpy
        @test lmul(ColumnWise(Debug),S,b) == Sb
        @test b == b_cpy
        @test ldiv(S,Sb) ≈ b
        @test Sb == Sb_cpy
        @test ldiv(RowWise(Debug),S,Sb) ≈ b
        @test Sb == Sb_cpy
        @test ldiv(ColumnWise(Debug),S,Sb) ≈ b
        @test Sb == Sb_cpy

        # Test in-place operations.
        @test lmul!(Debug,            S,copyto!(dst,b)) == Sb
        @test lmul!(RowWise(Debug),   S,copyto!(dst,b)) == Sb
        @test lmul!(ColumnWise(Debug),S,copyto!(dst,b)) == Sb
        @test ldiv!(Debug,            S,copyto!(dst,Sb)) ≈ b
        @test ldiv!(RowWise(Debug),   S,copyto!(dst,Sb)) ≈ b
        @test ldiv!(ColumnWise(Debug),S,copyto!(dst,Sb)) ≈ b
    end
end

end # module
