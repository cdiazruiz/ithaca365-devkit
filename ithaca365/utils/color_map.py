from typing import Dict, Tuple


def get_colormap() -> Dict[str, Tuple[int, int, int]]:
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "car": (255, 158, 0),  # Orange
        "truck": (255, 99, 71),  # Tomato
        "pedestrian": (0, 0, 230),  # Blue
        "bicyclist": (220, 20, 60),  # Crimson
        "bus": (255, 69, 0),  # Orangered
        "motorcyclist": (255, 61, 99),  # Red
        "road": (255,255,0)   # Yellow
    }

    return classname_to_color
