"""Main simulation engine for magnetic nanoparticle MD."""

from typing import List, Callable, Dict, Any
import warnings
import numpy as np
from .particles import Particle, Link, LinkType, MagneticChain


class MagneticSimulation:
    """Magnetic nanoparticle simulation with field ON/OFF cycling.
    
    Physical constants:
        kT = 4.14 pN·nm at 300K
        eta = 1e-9 pN·s/nm² for water
        D = kT / (3 * pi * eta * d_np)
    
    Simulation parameters:
        CHAIN_SPACING_TOLERANCE: Relative tolerance (10%) for checking whether
            chains are already properly spaced before applying adjustments.
    """
    
    CHAIN_SPACING_TOLERANCE = 0.1  # Relative tolerance (10%) for chain spacing checks
    
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
        
        # Create particle dictionary once for efficiency
        particle_dict = {p.id: p for p in self.particles}
        
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
                    current = particle_dict[current.back_linked_to]
                
                # Now trace forward from back to front, maintaining connectivity order
                chain_particles = []
                while current is not None:
                    chain_particles.append(current.id)
                    assigned.add(current.id)
                    
                    if (current.front_linked_to is not None and 
                        current.front_link_type == LinkType.MAGNETIC):
                        current = particle_dict[current.front_linked_to]
                    else:
                        break
                
                # Create chain object using connectivity-based ordering
                # (already ordered from back to front by traversal)
                chain = MagneticChain(id=chain_id, particle_ids=chain_particles)
                chain.update_center_of_mass(self.particles, self.box_size)
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
    
    def enforce_chain_geometry_initial(self, chains: List[MagneticChain]) -> None:
        """Set initial z-spacing for newly formed chains while preserving x,y structure.
        
        This method checks if chains are already properly initialized and skips them,
        so it can be safely called every timestep. After initial setup, Brownian motion
        preserves rigid body structure.
        
        Args:
            chains: List of chains to initialize geometry for
        """
        particle_dict = {p.id: p for p in self.particles}
        
        for chain in chains:
            if len(chain.particle_ids) < 2:
                continue
            
            # Skip if chain already has proper spacing (already formed)
            # Check if particles are already aligned along z-axis with proper spacing
            positions = [particle_dict[pid].position for pid in chain.particle_ids]
            z_positions = [pos[2] for pos in positions]
            
            # Sort by z to check spacing
            sorted_indices = np.argsort(z_positions)
            sorted_z = [z_positions[i] for i in sorted_indices]
            
            # Check if already properly spaced (within tolerance)
            is_properly_initialized = True
            if len(sorted_z) > 1:
                for i in range(len(sorted_z) - 1):
                    spacing = sorted_z[i + 1] - sorted_z[i]
                    # Handle periodic boundaries using minimum image convention
                    if spacing > self.box_size / 2:
                        spacing -= self.box_size
                    elif spacing < -self.box_size / 2:
                        spacing += self.box_size
                    if not np.isclose(abs(spacing), self.d_chain, rtol=self.CHAIN_SPACING_TOLERANCE):
                        is_properly_initialized = False
                        break
            
            if is_properly_initialized:
                continue  # Skip this chain, it's already properly formed
            
            # Update center of mass
            chain.update_center_of_mass(self.particles, self.box_size)
            
            # Calculate ideal z-positions along chain axis
            N = len(chain.particle_ids)
            total_length = (N - 1) * self.d_chain
            
            com_wrapped = chain.center_of_mass % self.box_size
            z_center = com_wrapped[2]
            z_start = z_center - total_length / 2.0
            
            # Wrap z_start into [0, box_size) to handle all chain lengths robustly
            z_start = z_start % self.box_size
            
            # Position particles: maintain x,y from COM, set ideal z-spacing
            for i, pid in enumerate(chain.particle_ids):
                particle = particle_dict[pid]
                
                # Calculate current offset from COM in x,y
                # Use unwrapped positions to handle periodic boundaries correctly
                dx = particle.position[0] - chain.center_of_mass[0]
                dy = particle.position[1] - chain.center_of_mass[1]
                
                # Apply periodic boundary correction if needed
                if dx > self.box_size / 2:
                    dx -= self.box_size
                elif dx < -self.box_size / 2:
                    dx += self.box_size
                    
                if dy > self.box_size / 2:
                    dy -= self.box_size
                elif dy < -self.box_size / 2:
                    dy += self.box_size
                
                # Set new position: COM + offset in x,y, ideal z-spacing
                particle.position[0] = (com_wrapped[0] + dx) % self.box_size
                particle.position[1] = (com_wrapped[1] + dy) % self.box_size
                particle.position[2] = (z_start + i * self.d_chain) % self.box_size
    
    def enforce_chain_geometry_deprecated(self, chains: List[MagneticChain]) -> None:
        """DEPRECATED: Position particles along z-axis with d_chain spacing.
        
        WARNING: This method collapses all particles to the same (x,y) coordinates,
        which destroys the rigid body structure. Use enforce_chain_geometry_initial()
        instead, which preserves x,y offsets.
        
        This function ensures chains maintain proper geometry without breaking
        across periodic boundaries. The entire chain is kept contiguous.
        
        Args:
            chains: List of chains to enforce geometry on
        """
        warnings.warn(
            "enforce_chain_geometry_deprecated is deprecated. "
            "Use enforce_chain_geometry_initial instead.",
            DeprecationWarning,
            stacklevel=2
        )
        particle_dict = {p.id: p for p in self.particles}
        
        for chain in chains:
            if len(chain.particle_ids) < 2:
                continue
            
            # Update center of mass (handles periodic boundaries)
            chain.update_center_of_mass(self.particles, self.box_size)
            
            # Calculate positions along z-axis
            N = len(chain.particle_ids)
            total_length = (N - 1) * self.d_chain
            
            # Wrap the chain center of mass into the primary simulation box
            com_wrapped = chain.center_of_mass % self.box_size
            
            # Choose a starting z position so the entire chain lies within the box
            # and remains contiguous (no wrapping within a single chain)
            z_center = com_wrapped[2]
            z_start = z_center - total_length / 2.0
            
            # Ensure chain stays within box bounds - wrap if needed
            if z_start < 0:
                z_start += self.box_size
            elif z_start + total_length >= self.box_size:
                z_start -= self.box_size
            
            # Position particles with proper spacing along z
            for i, pid in enumerate(chain.particle_ids):
                particle = particle_dict[pid]
                # Keep x, y at wrapped chain center of mass
                particle.position[0] = com_wrapped[0]
                particle.position[1] = com_wrapped[1]
                # Set z position with proper spacing
                particle.position[2] = (z_start + i * self.d_chain) % self.box_size
    
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
            
            # Enforce geometry BEFORE motion (only for newly formed chains)
            # This sets initial z-spacing when chains first form
            self.enforce_chain_geometry_initial(self.chains)
            
            # Move chains with rigid body motion (preserves relative positions!)
            self.move_chains_brownian(self.chains, dt)
            
            # Move free particles
            self.move_free_particles_brownian(dt)
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
