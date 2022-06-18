module TestingTriangularMatricesTutorial

using TriangularMatricesTutorial
using TriangularMatricesTutorial: strip_triangular
using Test
using MayOptimize
using LinearAlgebra
using LinearAlgebra: AbstractTriangular

# Narrow is used to reduce integer range for generating small random integers
# that can be converted to a widen type for exact calculations.
narrow(T::Type) = T
narrow(::Type{<:Union{Signed,AbstractFloat}}) = Int8
narrow(::Type{<:Unsigned}) = UInt8
narrow(::Type{Complex{T}}) where {T} = Complex{narrow(T)}

brief(args...) = string(map(brief, args)...)
brief(str::AbstractString) = str
brief(arg::Union{Char,Symbol}) = string(arg)
brief(T::Type{<:Number}) = string(T.name)
for S in (:LowerTriangular, :UnitLowerTriangular,
          :UpperTriangular, :UnitUpperTriangular)
    @eval begin
        brief(::Type{<:$S}) = $(string(S))
        brief(::Type{<:$S}, arg) = $(brief(S)*"(")*brief(arg)*")"
    end
end
brief(::typeof(identity),  args...) = brief(args...)
brief(::typeof(transpose), args...) = "transpose($(brief(args...)))"
brief(::typeof(adjoint),   args...) = "adjoint($(brief(args...)))"

@testset "TriangularMatricesTutorial" begin
    TYPES = (Float64, Complex{Float64})
    STRUCTS = (LowerTriangular, UpperTriangular)
    FUNCS = (identity, transpose, adjoint)
    let A = [1 2; 3 4]
        @testset "Type aliases for S = $(brief(s))($(brief(f,"A")))" for f in FUNCS, s in STRUCTS
            let S = s(f(A))
                @test isa(S,            AbstractMatrix)
                @test isa(adjoint(S),   AbstractMatrix)
                @test isa(transpose(S), AbstractMatrix)
                @test isa(S,            TriangularMatrix) == (s <: AbstractTriangular)
                @test isa(adjoint(S),   TriangularMatrix) == (s <: AbstractTriangular)
                @test isa(transpose(S), TriangularMatrix) == (s <: AbstractTriangular)
                @test isa(S,            LowerTriangularMatrix) == (s <: LowerTriangular)
                @test isa(adjoint(S),   LowerTriangularMatrix) == (s <: UpperTriangular)
                @test isa(transpose(S), LowerTriangularMatrix) == (s <: UpperTriangular)
                @test isa(S,            UpperTriangularMatrix) == (s <: UpperTriangular)
                @test isa(adjoint(S),   UpperTriangularMatrix) == (s <: LowerTriangular)
                @test isa(transpose(S), UpperTriangularMatrix) == (s <: LowerTriangular)
                @test isa(S,            UnitLowerTriangularMatrix) == (s <: UnitLowerTriangular)
                @test isa(adjoint(S),   UnitLowerTriangularMatrix) == (s <: UnitUpperTriangular)
                @test isa(transpose(S), UnitLowerTriangularMatrix) == (s <: UnitUpperTriangular)
                @test isa(S,            UnitUpperTriangularMatrix) == (s <: UnitUpperTriangular)
                @test isa(adjoint(S),   UnitUpperTriangularMatrix) == (s <: UnitLowerTriangular)
                @test isa(transpose(S), UnitUpperTriangularMatrix) == (s <: UnitLowerTriangular)
            end
        end
        @testset "strip_triangular($(brief(f,brief(s,brief(g,"A")))))" for f in FUNCS, s in (identity, STRUCTS...), g in FUNCS
            @test strip_triangular(f(s(g(A)))) === f(g(A))
        end
    end

    @testset "Operations with S = $(brief(f,brief(s),"{",brief(T),"}(A)"))" for T in TYPES, s in STRUCTS, f in FUNCS
        n = 7
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
        @test lmul(RowWise(Debug),   S,b) == Sb
        @test b == b_cpy
        @test lmul(ColumnWise(Debug),S,b) == Sb
        @test b == b_cpy
        @test ldiv(S,Sb) ≈ b
        @test Sb == Sb_cpy
        @test ldiv(RowWise(Debug),   S,Sb) ≈ b
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
