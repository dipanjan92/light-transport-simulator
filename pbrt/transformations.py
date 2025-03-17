"""
This module parses transformation data from PBRT scene descriptions and converts 
them into structured representations. It ensures the transformation matrix is valid 
and properly formatted for use in rendering calculations.
"""

def parse_transform(transform_list):
    """
    Parses a transformation matrix from a PBRT transformation list.

    Args:
        transform_list (list): List of transformation dictionaries from PBRT.

    Returns:
        tuple: A 16-element tuple representing the transformation matrix.

    Raises:
        ValueError: If the input is not a list or does not contain a valid 16-element matrix.
    """
    if not transform_list or not isinstance(transform_list, list):
        raise ValueError("Expected a non-empty list for transform")

    # Get the first (and assumed only) transform dictionary.
    # This assumes that the list contains at least one transform entry.
    transform_entry = transform_list[0]
    properties = transform_entry.get("properties", {})
    matrix = properties.get("matrix")
    
    # Check that the matrix is present and has exactly 16 elements.
    # This is necessary for proper transformation representation.
    if not matrix or len(matrix) != 16:
        raise ValueError("Transform matrix must be a list of 16 numbers")

    # Convert to tuple for immutability and compatibility with rendering functions.
    return tuple(float(x) for x in matrix)