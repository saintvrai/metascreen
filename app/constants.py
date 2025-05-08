# app/constants.py

import numpy as np

DISPLAY_NAMES = {
    "kvadrat": "Квадрат",
    "krug":    "Круг",
    "romb":    "Ромб"
}

STRUCTURES = list(DISPLAY_NAMES.keys())

CONSTRAINTS = {
    "kvadrat": {"H": (0.1, 0.6), "K": (1.0, 2.0), "L": (2.8, 3.6), "P": (0.1, 0.2)},
    "krug":    {"H": (0.6, 3.0), "K": (0.6, 3.0), "L": (0.6, 3.0), "P": (0.6, 3.0)},
    "romb":    {"H": (0.4, 1.5), "K": (0.4, 1.5), "L": (0.4, 1.5), "P": (0.4, 1.5)},
}

GA_DEFAULTS = {
    "pop_size": 150,
    "generations": 200,
    "mutation_rate": 0.1,
}

PARAM_ORDER = ['H', 'K', 'L', 'P']
