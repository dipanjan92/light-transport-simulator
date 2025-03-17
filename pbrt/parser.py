"""
This module provides parsing utilities for PBRT-V4 scene files.
It defines functions to tokenize, parse, and post-process PBRT data into structured dictionaries.
"""
import re
from pbrt.lexer import tokenize_file

IDENTITY_4x4 = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
)

def parse_value(token):
    """
    Parse a token into a corresponding Python value.
    
    - If enclosed in brackets, convert it to a list of floats (or strings if conversion fails).
    - If quoted, strip the quotes and return as a string.
    - Otherwise, attempt to convert it to a float, defaulting to a string if conversion fails.
    
    Args:
        token (str): The input token from PBRT.
    
    Returns:
        Union[str, float, list]: The parsed value.
    """
    token = token.strip()
    if token.startswith('[') and token.endswith(']'):
        inner = token[1:-1].strip()
        # Split on any whitespace (including newlines)
        values = re.split(r'\s+', inner)
        try:
            return [float(v) for v in values if v]
        except ValueError:
            return values
    elif token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    else:
        try:
            return float(token)
        except ValueError:
            return token

def parse_tokens(tokens):
    """
    Convert a sequence of tokens into structured blocks.
    
    Each block is a dictionary containing:
    - 'type': The PBRT directive (e.g., 'Shape', 'Material').
    - 'name' (optional): The identifier if specified.
    - 'properties': A dictionary of parameters.
    - 'children' (optional): Nested attributes if applicable.
    
    Supports handling nested AttributeBegin/End blocks.
    
    Args:
        tokens (list): List of tokenized elements.
    
    Returns:
        list: A list of structured blocks.
    """
    idx = 0
    n = len(tokens)
    blocks = []

    def parse_block(stop_tokens=None):
        nonlocal idx
        block_list = []
        while idx < n:
            current = tokens[idx]
            # Check for stop tokens
            if stop_tokens and current['type'] == 'KEYWORD' and current['value'] in stop_tokens:
                idx += 1  # consume the stop token
                break
            # Handle Attribute nesting.
            if current['value'] == 'AttributeBegin':
                idx += 1
                children = parse_block(stop_tokens={'AttributeEnd'})
                block_list.append({'type': 'Attribute', 'children': children})
                continue
            if current['value'] == 'AttributeEnd':
                idx += 1
                break
            # If a new block starts with a KEYWORD.
            if current['type'] == 'KEYWORD':
                key = current['value']
                idx += 1
                identifier = None
                if idx < n and tokens[idx]['type'] == 'QUOTED':
                    identifier = parse_value(tokens[idx]['value'])
                    idx += 1
                sub_block = {'type': key, 'properties': {}}
                if identifier:
                    sub_block['name'] = identifier
                # Special handling for Transform: use next BRACKET token as matrix if available.
                if key == 'Transform' and idx < n and tokens[idx]['type'] == 'BRACKET':
                    sub_block['properties']['matrix'] = parse_value(tokens[idx]['value'])
                    idx += 1
                if key == 'Transform' and 'matrix' not in sub_block['properties']:
                    sub_block['properties']['matrix'] = IDENTITY_4x4

                # Collect properties until we hit the next KEYWORD.
                while idx < n and tokens[idx]['type'] != 'KEYWORD':
                    if tokens[idx]['type'] == 'QUOTED':
                        prop_key = parse_value(tokens[idx]['value'])
                        idx += 1
                        prop_value = None
                        if idx < n and tokens[idx]['type'] == 'BRACKET':
                            prop_value = parse_value(tokens[idx]['value'])
                            idx += 1
                        # For keys like "float fov", take the second word as the property name.
                        if isinstance(prop_key, str) and ' ' in prop_key:
                            key_parts = prop_key.split()
                            prop_key_clean = key_parts[1]
                        else:
                            prop_key_clean = prop_key
                        sub_block['properties'][prop_key_clean] = prop_value
                    else:
                        idx += 1
                block_list.append(sub_block)
            else:
                idx += 1
        return block_list

    blocks = parse_block(stop_tokens={'WorldEnd'})
    return blocks

def post_process_blocks(blocks):
    """
    Organize blocks so that NamedMaterial blocks correctly group associated Shape blocks.
    
    If a NamedMaterial block is immediately followed by one or more Shape blocks,
    those Shape blocks are nested as its children.
    
    Args:
        blocks (list): List of parsed blocks.
    
    Returns:
        list: Processed blocks with proper hierarchy.
    """
    processed = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.get('type') == 'NamedMaterial':
            block.setdefault('children', [])
            i += 1
            while i < len(blocks) and blocks[i].get('type') == 'Shape':
                block['children'].append(blocks[i])
                i += 1
            processed.append(block)
        else:
            processed.append(block)
            i += 1
    return processed

def pbrt_to_dict(filename):
    """
    Convert a PBRT scene file into a structured dictionary representation.
    
    - Tokenizes the file.
    - Parses tokens into structured blocks.
    - Post-processes blocks into a hierarchical scene structure.
    
    Args:
        filename (str): The path to the PBRT scene file.
    
    Returns:
        dict: A dictionary where keys represent block types and values are lists of corresponding elements.
    """
    tokens = tokenize_file(filename)
    flat_blocks = parse_tokens(tokens)
    nested = post_process_blocks(flat_blocks)
    scene_dict = {}
    for block in nested:
        block_type = block.get('type')
        scene_dict.setdefault(block_type, []).append(block)
    return scene_dict

if __name__ == "__main__":
    # Replace 'scene.pbrt' with the path to your PBRT v4 file.
    filename = "scene.pbrt"
    import pprint
    scene_dict = pbrt_to_dict(filename)
    pprint.pprint(scene_dict)
