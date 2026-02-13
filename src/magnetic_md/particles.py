"""Data structures for particles, links, and chains in magnetic nanoparticle simulations."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import numpy as np


class LinkType(Enum):
    """Type of link between particles."""
    MAGNETIC = "magnetic"  # Field-dependent magnetic link
    ANTIGEN = "antigen"    # Biochemical link (for future use)


@dataclass
class Link:
    """Represents a connection between two particles.
    
    Attributes:
        particle1_id: ID of first particle
        particle2_id: ID of second particle
        link_type: Type of link (MAGNETIC or ANTIGEN)
        pole1: Pole of particle1 ("front" or "back")
        pole2: Pole of particle2 ("front" or "back")
        time_formed: Simulation time when link was formed
    """
    particle1_id: int
    particle2_id: int
    link_type: LinkType
    pole1: str  # "front" or "back"
    pole2: str  # "front" or "back"
    time_formed: float


@dataclass
class Particle:
    """Represents a magnetic nanoparticle.
    
    Attributes:
        id: Unique particle identifier
        position: 3D position vector [x, y, z] in nm
        np_type: Particle type ('A' or 'B')
        front_linked_to: ID of particle linked to front pole, or None
        back_linked_to: ID of particle linked to back pole, or None
        front_link_type: Type of front link, or None
        back_link_type: Type of back link, or None
    """
    id: int
    position: np.ndarray
    np_type: str  # 'A' or 'B'
    front_linked_to: Optional[int] = None
    back_linked_to: Optional[int] = None
    front_link_type: Optional[LinkType] = None
    back_link_type: Optional[LinkType] = None


@dataclass
class MagneticChain:
    """Represents a chain of magnetically linked particles.
    
    Attributes:
        id: Unique chain identifier
        particle_ids: Ordered list of particle IDs (back to front, increasing z)
        center_of_mass: 3D position of chain center of mass
        D_eff: Effective diffusion coefficient
    """
    id: int
    particle_ids: List[int] = field(default_factory=list)
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))
    D_eff: float = 0.0
    
    def update_center_of_mass(self, particles: List[Particle]) -> None:
        """Update the center of mass of the chain.
        
        Args:
            particles: List of all particles in the simulation
        """
        if not self.particle_ids:
            self.center_of_mass = np.zeros(3)
            return
        
        # Create a dictionary for fast particle lookup
        particle_dict = {p.id: p for p in particles}
        
        # Calculate center of mass
        positions = [particle_dict[pid].position for pid in self.particle_ids]
        self.center_of_mass = np.mean(positions, axis=0)
    
    def calculate_diffusion(self, D_single: float) -> None:
        """Calculate effective diffusion coefficient for the chain.
        
        Uses D_eff = D_single / N where N is the chain length.
        
        Args:
            D_single: Diffusion coefficient of a single particle
        """
        N = len(self.particle_ids)
        if N > 0:
            self.D_eff = D_single / N
        else:
            self.D_eff = 0.0
