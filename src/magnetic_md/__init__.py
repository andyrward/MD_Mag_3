"""Magnetic Nanoparticle MD Simulation Package.

A package for simulating magnetic nanoparticle chain formation 
with field ON/OFF cycling.
"""

from .particles import LinkType, Link, Particle, MagneticChain
from .simulation import MagneticSimulation
from .fields import constant_field, square_wave, delayed_activation

__all__ = [
    "LinkType",
    "Link",
    "Particle",
    "MagneticChain",
    "MagneticSimulation",
    "constant_field",
    "square_wave",
    "delayed_activation",
]
