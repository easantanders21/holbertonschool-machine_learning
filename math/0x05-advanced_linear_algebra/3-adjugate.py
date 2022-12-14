#!/usr/bin/env python3
"""
3-adjugate module
contains functions: matmin, determinant, minor, cofactor and adjugate
"""


def matmin(m, i, j):
    """Slices matrix for minor calculation
    Args:
        matrix (list): list of lists whose minor should be calculated
        i (int):
        j (int):
    Returns:
        (list): the sliced matrix for minor calculation
    """
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]


def determinant(matrix):
    """Calculates the determinant of a matrix
    Args:
        matrix (list): list of lists whose determinant should be calculated
    Returns:
        (float): the determinant of matrix
    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")
    if any(type(row) is not list for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if any(n != len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    det = 0
    for c in range(n):
        det += ((-1)**c)*matrix[0][c] * determinant(matmin(matrix, 0, c))
    return det


def minor(matrix):
    """Calculates the minor of a matrix
    Args:
        matrix (list): list of lists whose minor should be calculated
    Returns:
        (list): the minor matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")
    if any(type(row) is not list for row in matrix):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if any((not row or n != len(row)) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]
    my_min = [[determinant(matmin(matrix, i, j)) for j in range(n)]
              for i in range(n)]
    return my_min


def cofactor(matrix):
    """Calculates the cofactor of a matrix
    Args:
        matrix (list): list of lists whose cofactor should be calculated
    Returns:
        (list): the cofactor matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")
    if any(type(row) is not list for row in matrix):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if any((not row or n != len(row)) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    my_min = minor(matrix)

    cof = [[((-1)**(i+j)) * my_min[i][j] for j in range(n)]
           for i in range(n)]
    return cof


def adjugate(matrix):
    """Calculates the adjugate of a matrix
    Args:
        matrix (list): list of lists whose adjugate should be calculated
    Returns:
        (list): the adjugate matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")
    if any(type(row) is not list for row in matrix):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if any((not row or n != len(row)) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cof = cofactor(matrix)
    adj = [[row[i] for row in cof] for i in range(n)]
    return adj
