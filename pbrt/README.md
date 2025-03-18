# PBRT v4 Scene Parser (Work in Progress)

This is a **work-in-progress** parser for **simple PBRT v4 scenes**. It is designed to extract and process scene data, but currently supports only a limited set of PBRT v4 features. This parser is **not a full PBRT implementation** but is useful for basic scene parsing and transformation handling.

For more details on PBRT v4 scene files, refer to the official documentation:  
➡️ [PBRT v4 File Format](https://pbrt.org/fileformat-v4)  

## Features & Limitations

✅ Parses basic shapes, materials, lights, and transformations  
✅ Extracts camera parameters and object properties  
✅ Converts PBRT v4 structures into a structured format  
❌ Does not support complex PBRT features like volume rendering, advanced BSDFs, or procedural textures  

## File Overview

Each file in this repository serves a specific purpose in parsing PBRT v4 scenes:

- **`lexer.py`** – Tokenizes PBRT scene files by extracting keywords, quoted strings, bracketed content, and numbers.
- **`shape_parser.py`** – Parses shape descriptions (e.g., `trianglemesh`, `sphere`), applying transformations and material properties.
- **`material_parser.py`** – Processes material definitions and converts them into a structured format.
- **`light_parser.py`** – Extracts light sources from PBRT files and categorizes them by type.
- **`camera_parser.py`** – Reads and extracts camera parameters (position, look-at, up vector, FOV) from the PBRT scene.
- **`transformations.py`** – Parses and formats transformation matrices used in PBRT scene descriptions.
- **`tests.ipynb`** – Provides example usages and test cases for the parser.  

Check **`tests.ipynb`** to see how this parser works with example PBRT v4 scenes.

## Reference

This project draws heavily from:

> Pharr, M., Jakob, W. and Humphreys, G., 2023. *Physically Based Rendering: From Theory to Implementation.* 4th ed. Available at: [https://pbr-book.org/4ed/contents](https://pbr-book.org/4ed/contents).  