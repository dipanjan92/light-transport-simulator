"""
This module tokenizes PBRT-V4 scene files by extracting keywords, quoted strings, 
bracketed content, and numbers. It supports PBRT-v4 syntax for parsing light transport 
simulation input files.
"""

import re

# Updated Token types with extended keywords for pbrt-v4.
TOKEN_TYPES = {
    'KEYWORD': r'\b(?:Camera|Shape|LightSource|AreaLightSource|Material|MakeNamedMaterial|NamedMaterial|Texture|Integrator|Sampler|PixelFilter|Film|WorldBegin|WorldEnd|AttributeBegin|AttributeEnd|Transform|ConcatTransform|LookAt|MediumInterface|MakeNamedMedium|Translate|Scale|Rotate|ActiveTransform|CoordinateSystem|CoordSysTransform)\b',  # Represents keywords in PBRT
    'QUOTED': r'"[^"]*"',  # matches quoted strings
    # Use a non-greedy pattern with DOTALL to capture multiline bracketed content.
    'BRACKET': r'\[(.*?)\]',  # Represents bracketed content
    'NUMBER': r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?'  # Represents numeric values
}

# Combine into a single regex pattern with named groups.
TOKEN_PATTERN = re.compile(
    r'(?P<KEYWORD>' + TOKEN_TYPES['KEYWORD'] + r')'
    r'|(?P<QUOTED>' + TOKEN_TYPES['QUOTED'] + r')'
    r'|(?P<BRACKET>' + TOKEN_TYPES['BRACKET'] + r')'
    r'|(?P<NUMBER>' + TOKEN_TYPES['NUMBER'] + r')',
    re.DOTALL  # Allows matching across multiple lines
)

def tokenize(text):
    """
    Tokenizes a given PBRT scene file content into recognized tokens.

    Args:
        text (str): The content of a PBRT scene file.

    Returns:
        list of dict: A list of tokens, each represented as a dictionary with 'type' and 'value'.
    """
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):  # Iterate over all regex matches
        token_type = match.lastgroup  # Get the type of the matched token
        token_value = match.group().strip()  # Get the matched token value
        tokens.append({'type': token_type, 'value': token_value})  # Store token as a dictionary
    return tokens

def tokenize_file(filename):
    """
    Reads a PBRT scene file and tokenizes its content.

    Args:
        filename (str): Path to the PBRT scene file.

    Returns:
        list of dict: A list of extracted tokens from the file.
    """
    with open(filename, 'r') as f:
        text = f.read()  # Read the entire file content
    return tokenize(text)  # Tokenize the file content
