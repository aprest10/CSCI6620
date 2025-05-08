from enum import Enum

class SparseMatrixType(Enum):
    """Sparse Matrix Types"""
    RANDOM = "random"
    BANDED = "banded"
    BLOCK_DIAGONAL = "block_diagonal"
    TRIDIAGONAL = "tridiagonal"
    CHECKERBOARD = "checkerboard"