module TriangularMatricesTutorial

export
    # Re-exports from MayOptimize:
    Debug,
    InBounds,
    Vectorize,

    # Re-exports from LinearAlgebra:
    AbstractTriangular,
    LowerTriangular,
    UpperTriangular,
    UnitLowerTriangular,
    UnitUpperTriangular,
    lmul!,
    ldiv!,

    # Exports from this package:
    RowWise,
    ColumnWise,
    lmul,
    ldiv

using MayOptimize

using LinearAlgebra
using LinearAlgebra: AbstractTriangular
import LinearAlgebra: lmul!, ldiv!

using Base: @propagate_inbounds, promote_op

include("utils.jl")
include("lmul.jl")
include("ldiv.jl")

end # module
