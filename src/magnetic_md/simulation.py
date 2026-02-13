"""Main simulation engine for magnetic nanoparticle MD."""

from typing import List, Callable, Dict, Any
import numpy as np
from .particles import Particle, Link, LinkType, MagneticChain


class MagneticSimulation:
    """Magnetic nanoparticle simulation with field ON/OFF cycling.
    
    Physical constants:
        kT = 4.14 pN·nm at 300K
        eta = 1e-9 pN·s/nm² for water
        D = kT / (3 * pi * eta * d_np)
    """
    
    def __init__(
        self,
        N_A: int,
        N_B: int,
        box_size: float,
        d_np: float = 100.0,
        d_chain: float = 10.0,
        r_cone: float = 20.0,
        theta_cone: float = 0.524,  # 30 degrees in radians
        D_trans: float = None,
    ):
        """Initialize the magnetic simulation.
        
        Args:
            N_A: Number of type A particles
            N_B: Number of type B particles
            box_size: Simulation box size (nm)
            d_np: Particle diameter (nm)
            d_chain: Spacing between particles in chains (nm)
            r_cone: Interaction cone radius (nm)
            theta_cone: Cone half-angle (radians)
            D_trans: Translational diffusion coefficient (nm²/s), calculated if None
        """
        self.N_A = N_A
        self.N_B = N_B
        self.box_size = box_size
        self.d_np = d_np
        self.d_chain = d_chain
        self.r_cone = r_cone
        self.theta_cone = theta_cone
        
        # Calculate diffusion coefficient using Stokes-Einstein
        if D_trans is None:
            kT = 4.14  # pN·nm
            eta = 1e-9  # pN·s/nm²
            self.D_trans = kT / (3 * np.pi * eta * d_np)
        else:
            self.D_trans = D_trans
        
        # Initialize particles randomly in box
        self.particles = []
        self._initialize_particles()
        
        # Initialize chains
        self.chains = []
        self.links = []
        
        # Simulation state
        self.time = 0.0
    
    def _initialize_particles(self) -> None:
        """Initialize particles with random positions in the box."""
        particle_id = 0
        
        # Create type A particles
        for _ in range(self.N_A):
            position = np.random.uniform(0, self.box_size, 3)
            self.particles.append(Particle(
                id=particle_id,
                position=position,
                np_type='A'
            ))
            particle_id += 1
        
        # Create type B particles
        for _ in range(self.N_B):
            position = np.random.uniform(0, self.box_size, 3)
            self.particles.append(Particle(
                id=particle_id,
                position=position,
                np_type='B'
            ))
            particle_id += 1
    
    def _minimum_image_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """Calculate minimum image distance vector with periodic boundary conditions.
        
        Args:
            pos1: Position of first particle
            pos2: Position of second particle
            
        Returns:
            Minimum image distance vector from pos1 to pos2
        """
        dr = pos2 - pos1
        dr = dr - self.box_size * np.round(dr / self.box_size)
        return dr
    
    def is_in_interaction_cone(
        self,
        p1: Particle,
        p2: Particle,
        pole_direction: str,
        r_cone: float,
        theta_cone: float
    ) -> bool:
        """Check if p2 is in p1's interaction cone.
        
        Args:
            p1: First particle
            p2: Second particle
            pole_direction: "front" (+z) or "back" (-z)
            r_cone: Cone radius (nm)
            theta_cone: Cone half-angle (radians)
            
        Returns:
            True if p2 is in p1's cone
        """
        # Calculate minimum image distance
        dr = self._minimum_image_distance(p1.position, p2.position)
        distance = np.linalg.norm(dr)
        
        # Check distance constraint
        if distance > r_cone or distance < 1e-10:  # Avoid self-interaction
            return False
        
        # Define cone axis direction
        if pole_direction == "front":
            axis = np.array([0.0, 0.0, 1.0])  # +z direction
        else:  # "back"
            axis = np.array([0.0, 0.0, -1.0])  # -z direction
        
        # Calculate angle between dr and axis
        cos_angle = np.dot(dr, axis) / distance
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Check angle constraint
        return angle <= theta_cone
    
    def update_magnetic_links(self) -> None:
        """Scan for particles in interaction cones and create magnetic links."""
        # Clear existing magnetic links
        self.links = [link for link in self.links if link.link_type != LinkType.MAGNETIC]
        
        # Reset magnetic link information in particles
        for p in self.particles:
            if p.front_link_type == LinkType.MAGNETIC:
                p.front_linked_to = None
                p.front_link_type = None
            if p.back_link_type == LinkType.MAGNETIC:
                p.back_linked_to = None
                p.back_link_type = None
        
        # Check all particle pairs for potential links
        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                # Check front pole of p1
                if p1.front_linked_to is None and p2.back_linked_to is None:
                    if self.is_in_interaction_cone(p1, p2, "front", self.r_cone, self.theta_cone):
                        # Create link
                        link = Link(
                            particle1_id=p1.id,
                            particle2_id=p2.id,
                            link_type=LinkType.MAGNETIC,
                            pole1="front",
                            pole2="back",
                            time_formed=self.time
                        )
                        self.links.append(link)
                        p1.front_linked_to = p2.id
                        p1.front_link_type = LinkType.MAGNETIC
                        p2.back_linked_to = p1.id
                        p2.back_link_type = LinkType.MAGNETIC
                
                # Check back pole of p1
                if p1.back_linked_to is None and p2.front_linked_to is None:
                    if self.is_in_interaction_cone(p1, p2, "back", self.r_cone, self.theta_cone):
                        # Create link
                        link = Link(
                            particle1_id=p1.id,
                            particle2_id=p2.id,
                            link_type=LinkType.MAGNETIC,
                            pole1="back",
                            pole2="front",
                            time_formed=self.time
                        )
                        self.links.append(link)
                        p1.back_linked_to = p2.id
                        p1.back_link_type = LinkType.MAGNETIC
                        p2.front_linked_to = p1.id
                        p2.front_link_type = LinkType.MAGNETIC
    
    def identify_magnetic_chains(self) -> List[MagneticChain]:
        """Trace connectivity to build chain objects.
        
        Returns:
            List of identified magnetic chains
        """
        # Track which particles have been assigned to chains
        assigned = set()
        chains = []
        chain_id = 0
        
        for particle in self.particles:
            if particle.id in assigned:
                continue
            
            # Start a new chain from this particle if it has magnetic links
            if (particle.front_link_type == LinkType.MAGNETIC or 
                particle.back_link_type == LinkType.MAGNETIC):
                
                # Trace to the back end of the chain
                current = particle
                while (current.back_linked_to is not None and 
                       current.back_link_type == LinkType.MAGNETIC):
                    particle_dict = {p.id: p for p in self.particles}
                    current = particle_dict[current.back_linked_to]
                
                # Now trace forward from back to front
                chain_particles = []
                while current is not None:
                    chain_particles.append(current.id)
                    assigned.add(current.id)
                    
                    if (current.front_linked_to is not None and 
                        current.front_link_type == LinkType.MAGNETIC):
                        particle_dict = {p.id: p for p in self.particles}
                        current = particle_dict[current.front_linked_to]
                    else:
                        break
                
                # Sort particles by z-coordinate to ensure ordering
                particle_dict = {p.id: p for p in self.particles}
                chain_particles.sort(key=lambda pid: particle_dict[pid].position[2])
                
                # Create chain object
                chain = MagneticChain(id=chain_id, particle_ids=chain_particles)
                chain.update_center_of_mass(self.particles)
                chain.calculate_diffusion(self.D_trans)
                chains.append(chain)
                chain_id += 1
        
        self.chains = chains
        return chains
    
    def break_magnetic_links(self) -> None:
        """Remove all magnetic links (preserves antigen links for future)."""
        # Remove magnetic links
        self.links = [link for link in self.links if link.link_type != LinkType.MAGNETIC]
        
        # Clear magnetic link information from particles
        for p in self.particles:
            if p.front_link_type == LinkType.MAGNETIC:
                p.front_linked_to = None
                p.front_link_type = None
            if p.back_link_type == LinkType.MAGNETIC:
                p.back_linked_to = None
                p.back_link_type = None
        
        # Clear chains
        self.chains = []
    
    def move_chains_brownian(self, chains: List[MagneticChain], dt: float) -> None:
        """Move chains with rigid body Brownian motion.
        
        Args:
            chains: List of chains to move
            dt: Time step (seconds)
        """
        particle_dict = {p.id: p for p in self.particles}
        
        for chain in chains:
            # Calculate displacement with scaled diffusion
            sigma = np.sqrt(2 * chain.D_eff * dt)
            displacement = np.random.normal(0, sigma, 3)
            
            # Move all particles in chain together
            for pid in chain.particle_ids:
                particle = particle_dict[pid]
                particle.position += displacement
                # Apply periodic boundary conditions
                particle.position = particle.position % self.box_size
    
    def move_free_particles_brownian(self, dt: float) -> None:
        """Move free (unchained) particles with Brownian motion.
        
        Args:
            dt: Time step (seconds)
        """
        # Get set of particles in chains
        chained_particles = set()
        for chain in self.chains:
            chained_particles.update(chain.particle_ids)
        
        # Move free particles
        sigma = np.sqrt(2 * self.D_trans * dt)
        for particle in self.particles:
            if particle.id not in chained_particles:
                displacement = np.random.normal(0, sigma, 3)
                particle.position += displacement
                # Apply periodic boundary conditions
                particle.position = particle.position % self.box_size
    
    def enforce_chain_geometry(self, chains: List[MagneticChain]) -> None:
        """Position particles along z-axis with d_chain spacing.
        
        Args:
            chains: List of chains to enforce geometry on
        """
        particle_dict = {p.id: p for p in self.particles}
        
        for chain in chains:
            if len(chain.particle_ids) < 2:
                continue
            
            # Update center of mass
            chain.update_center_of_mass(self.particles)
            
            # Calculate positions along z-axis
            N = len(chain.particle_ids)
            total_length = (N - 1) * self.d_chain
            
            # Starting z position (centered around chain COM)
            z_start = chain.center_of_mass[2] - total_length / 2
            
            # Position particles
            for i, pid in enumerate(chain.particle_ids):
                particle = particle_dict[pid]
                # Keep x, y at chain center of mass
                particle.position[0] = chain.center_of_mass[0]
                particle.position[1] = chain.center_of_mass[1]
                # Set z position with proper spacing
                particle.position[2] = z_start + i * self.d_chain
                # Apply periodic boundary conditions
                particle.position = particle.position % self.box_size
    
    def step(self, dt: float, field_on: bool) -> None:
        """Execute one simulation timestep.
        
        Args:
            dt: Time step (seconds)
            field_on: Whether magnetic field is ON
        """
        if field_on:
            # Update magnetic links
            self.update_magnetic_links()
            
            # Identify chains
            self.identify_magnetic_chains()
            
            # Move chains with rigid body motion
            self.move_chains_brownian(self.chains, dt)
            
            # Move free particles
            self.move_free_particles_brownian(dt)
            
            # Enforce chain geometry
            self.enforce_chain_geometry(self.chains)
        else:
            # Break magnetic links
            self.break_magnetic_links()
            
            # Move all particles as free particles
            self.move_free_particles_brownian(dt)
        
        # Update time
        self.time += dt
    
    def run(
        self,
        duration: float,
        dt: float,
        field_protocol: Callable[[float], bool]
    ) -> List[Dict[str, Any]]:
        """Run the main simulation loop.
        
        Args:
            duration: Total simulation time (seconds)
            dt: Time step (seconds)
            field_protocol: Function mapping time to field state
            
        Returns:
            List of dictionaries containing time series data
        """
        trajectory = []
        n_steps = int(duration / dt)
        
        for step in range(n_steps):
            # Get field state
            field_on = field_protocol(self.time)
            
            # Execute timestep
            self.step(dt, field_on)
            
            # Record data
            data = {
                'time': self.time,
                'field_on': field_on,
                'n_chains': len(self.chains),
                'avg_chain_length': np.mean([len(c.particle_ids) for c in self.chains]) if self.chains else 0.0,
                'max_chain_length': max([len(c.particle_ids) for c in self.chains]) if self.chains else 0,
            }
            trajectory.append(data)
        
        return trajectory
