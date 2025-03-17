"""
This module parses material definitions from PBRT-V4 scene descriptions and converts 
them into Taichi-compatible Material objects for rendering. It supports diffuse, 
diffuse transmission, dielectric, and conductor materials.
"""

import taichi as ti
from taichi.math import vec3

from base.materials import Material
from pbrt.parse_utils import to_vec3

# Define material type constants (you can expand these as needed)
DIFFUSE = 0
DIFFUSE_TRANSMISSION = 1
DIELECTRIC = 2
CONDUCTOR = 3


# Create functions for each material type.
def create_diffuse(mat_dict, emission=vec3(0.0)):
    """
    Creates a diffuse material with a specified reflectance color.

    Args:
        mat_dict (dict): Dictionary containing material properties.
        emission (vec3, optional): Emission color. Defaults to vec3(0.0).

    Returns:
        Material: A diffuse material with the given reflectance and emission.
    """
    # For a diffuse material, we typically use the "reflectance" parameter.
    reflectance = to_vec3(mat_dict.get("reflectance", [1.0, 1.0, 1.0]))  # Extracting reflectance from mat_dict
    return Material(material_type=DIFFUSE,
                    reflectance=reflectance,
                    transmittance=vec3(0.0),
                    uroughness=0.0,
                    vroughness=0.0,
                    eta=vec3(1.0),
                    k=vec3(0.0),
                    emission=emission)

def create_diffuse_transmission(mat_dict, emission=vec3(0.0)):
    """
    Creates a diffuse transmission material with reflectance and transmittance.

    Args:
        mat_dict (dict): Dictionary containing material properties.
        emission (vec3, optional): Emission color. Defaults to vec3(0.0).

    Returns:
        Material: A diffuse transmission material with specified reflectance and transmittance.
    """
    # Diffuse transmission might combine a diffuse response with a transmission component.
    # For this example, we use the "reflectance" parameter for the transmitted color.
    reflectance = to_vec3(mat_dict.get("reflectance", [1.0, 1.0, 1.0]))
    transmittance = to_vec3(mat_dict.get("transmittance", [1.0, 1.0, 1.0]))
    return Material(material_type=DIFFUSE_TRANSMISSION,
                    reflectance=reflectance,
                    transmittance=transmittance,
                    uroughness=0.0,
                    vroughness=0.0,
                    eta=1.0,
                    k=to_vec3([0.0, 0.0, 0.0]),
                    emission=emission)

def create_dielectric(mat_dict, emission=vec3(0.0)):
    """
    Creates a dielectric material with a specified index of refraction (eta).

    Args:
        mat_dict (dict): Dictionary containing material properties.
        emission (vec3, optional): Emission color. Defaults to vec3(0.0).

    Returns:
        Material: A dielectric material with the given reflectance and eta.
    """
    # For a dielectric material, we might use "reflectance" as color
    # and an "eta" parameter (index of refraction).
    reflectance = to_vec3(mat_dict.get("reflectance", [1.0, 1.0, 1.0]))
    eta = to_vec3(mat_dict.get("eta", [1.0, 1.0, 1.0]))
    # Roughness is usually zero for an ideal dielectric.
    return Material(material_type=DIELECTRIC,
                    reflectance=reflectance,
                    transmittance=vec3(0.0),
                    uroughness=0.0,
                    vroughness=0.0,
                    eta=vec3(eta[0]),
                    k=vec3(0.0),
                    emission=emission)

def create_conductor(mat_dict, emission=vec3(0.0)):
    """
    Creates a conductor material with specified reflectance, roughness, and complex refractive indices (eta and k).

    Args:
        mat_dict (dict): Dictionary containing material properties.
        emission (vec3, optional): Emission color. Defaults to vec3(0.0).

    Returns:
        Material: A conductor material with reflectance, roughness, and refractive indices.
    """
    # For conductor materials, you typically need roughness,
    # and complex refractive index components "eta" and "k".
    reflectance = to_vec3(mat_dict.get("reflectance", [1.0, 1.0, 1.0]))
    urough = float(mat_dict.get("uroughness", [0.0])[0])  # Extracting uroughness from mat_dict
    vrough = float(mat_dict.get("vroughness", [0.0])[0])  # Extracting vroughness from mat_dict
    eta = to_vec3(mat_dict.get("eta", [1.0, 1.0, 1.0]))
    k = to_vec3(mat_dict.get("k", [0.0, 0.0, 0.0]))
    return Material(material_type=CONDUCTOR,
                    reflectance=reflectance,
                    transmittance=vec3(0.0),
                    uroughness=urough,
                    vroughness=vrough,
                    eta=eta,
                    k=k,
                    emission=emission)

# A mapping from material type strings to create functions.
TYPE_MAP = {
    "diffuse": create_diffuse,
    "diffusetransmission": create_diffuse_transmission,
    "dielectric": create_dielectric,
    "conductor": create_conductor
}

# Main parser function: choose create function based on material type.
def parse_materials(material_list):
    """
    Parses a list of material dictionaries and creates Material objects.

    Args:
        material_list (list of dict): List of material descriptions from PBRT.

    Returns:
        list of Material: A list of parsed Material objects.
    """
    # material_list is a list of material dictionaries (as shown in your example).
    parsed_materials = []
    for m in material_list:
        mat_props = m.get("properties", {})
        # Assume the type is provided as a list with one element, e.g., ['"diffuse"'].
        raw_type = mat_props.get("type", ["diffuse"])
        # Remove quotes if necessary.
        mat_type = raw_type[0].strip('"').lower() if isinstance(raw_type, list) and raw_type else "diffuse"
        create_fn = TYPE_MAP.get(mat_type, create_diffuse)  # Mapping material type to create function
        parsed_materials.append(create_fn(mat_props))
    return parsed_materials

# Finally, create a Taichi field populated with these materials.
def create_material_field(material_list, material_field):
    """
    Populates a Taichi field with parsed materials.

    Args:
        material_list (list of dict): List of material descriptions.
        material_field (ti.field): Taichi field to store the materials.

    Returns:
        ti.field: The populated material field.
    """
    parsed_materials = parse_materials(material_list)
    for i, mat in enumerate(parsed_materials):
        material_field[i] = mat
    return material_field

def create_material_by_name(material_list, target_name):
    """
    Searches for a material by name in the provided list and creates it.

    Args:
        material_list (list of dict): List of material descriptions.
        target_name (str): The name of the material to find.

    Returns:
        Material: The created Material object.

    Raises:
        ValueError: If the specified material name is not found.
    """
    for m in material_list:
        if m.get("name") == target_name:  # Matching material name to find the correct material
            mat_props = m.get("properties", {})
            raw_type = mat_props.get("type", ["diffuse"])
            mat_type = raw_type[0].strip('"').lower() if isinstance(raw_type, list) and raw_type else "diffuse"
            create_fn = TYPE_MAP.get(mat_type, create_diffuse)
            return create_fn(mat_props)
    raise ValueError(f"Material with name '{target_name}' not found.")
