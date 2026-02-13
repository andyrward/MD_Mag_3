"""Tests for the main simulation engine."""

import numpy as np
from src.magnetic_md.simulation import MagneticSimulation
from src.magnetic_md.particles import Particle, LinkType
from src.magnetic_md.fields import constant_field


class TestMagneticSimulation:
    """Tests for MagneticSimulation class."""
    
    def test_initialization(self):
        """Test simulation initialization."""
        sim = MagneticSimulation(N_A=10, N_B=10, box_size=1000.0)
        
        assert sim.N_A == 10
        assert sim.N_B == 10
        assert sim.box_size == 1000.0
        assert len(sim.particles) == 20
        assert sim.time == 0.0
        
        # Check particle types
        type_A = [p for p in sim.particles if p.np_type == 'A']
        type_B = [p for p in sim.particles if p.np_type == 'B']
        assert len(type_A) == 10
        assert len(type_B) == 10
    
    def test_diffusion_calculation(self):
        """Test that diffusion coefficient is calculated correctly."""
        sim = MagneticSimulation(N_A=5, N_B=5, box_size=1000.0, d_np=100.0)
        
        # D = kT / (3 * pi * eta * d_np)
        kT = 4.14
        eta = 1e-9
        d_np = 100.0
        expected_D = kT / (3 * np.pi * eta * d_np)
        
        assert np.isclose(sim.D_trans, expected_D)
    
    def test_minimum_image_distance(self):
        """Test periodic boundary condition distance calculation."""
        sim = MagneticSimulation(N_A=2, N_B=0, box_size=1000.0)
        
        # Test simple distance
        pos1 = np.array([100.0, 100.0, 100.0])
        pos2 = np.array([200.0, 200.0, 200.0])
        dr = sim._minimum_image_distance(pos1, pos2)
        expected = np.array([100.0, 100.0, 100.0])
        assert np.allclose(dr, expected)
        
        # Test periodic boundary
        pos1 = np.array([50.0, 50.0, 50.0])
        pos2 = np.array([950.0, 950.0, 950.0])
        dr = sim._minimum_image_distance(pos1, pos2)
        expected = np.array([-100.0, -100.0, -100.0])
        assert np.allclose(dr, expected)
    
    def test_interaction_cone_detection(self):
        """Test interaction cone detection."""
        sim = MagneticSimulation(N_A=2, N_B=0, box_size=1000.0, r_cone=50.0, theta_cone=0.524)
        
        # Place two particles aligned along z-axis
        p1 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A')
        p2 = Particle(id=1, position=np.array([500.0, 500.0, 530.0]), np_type='A')
        
        # p2 should be in front cone of p1
        assert sim.is_in_interaction_cone(p1, p2, "front", sim.r_cone, sim.theta_cone)
        
        # p2 should not be in back cone of p1
        assert not sim.is_in_interaction_cone(p1, p2, "back", sim.r_cone, sim.theta_cone)
        
        # p1 should be in back cone of p2
        assert sim.is_in_interaction_cone(p2, p1, "back", sim.r_cone, sim.theta_cone)
    
    def test_interaction_cone_distance_limit(self):
        """Test that interaction cone respects distance limits."""
        sim = MagneticSimulation(N_A=2, N_B=0, box_size=1000.0, r_cone=50.0)
        
        # Place particles too far apart
        p1 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A')
        p2 = Particle(id=1, position=np.array([500.0, 500.0, 600.0]), np_type='A')
        
        # Should not be in cone (too far)
        assert not sim.is_in_interaction_cone(p1, p2, "front", sim.r_cone, sim.theta_cone)
    
    def test_interaction_cone_angle_limit(self):
        """Test that interaction cone respects angle limits."""
        sim = MagneticSimulation(N_A=2, N_B=0, box_size=1000.0, r_cone=50.0, theta_cone=0.524)
        
        # Place particles at an angle
        p1 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A')
        p2 = Particle(id=1, position=np.array([530.0, 530.0, 510.0]), np_type='A')
        
        # Should not be in cone (too large angle)
        assert not sim.is_in_interaction_cone(p1, p2, "front", sim.r_cone, sim.theta_cone)
    
    def test_update_magnetic_links(self):
        """Test magnetic link creation."""
        sim = MagneticSimulation(N_A=0, N_B=0, box_size=1000.0, r_cone=50.0)
        
        # Manually add particles in alignment
        p1 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A')
        p2 = Particle(id=1, position=np.array([500.0, 500.0, 520.0]), np_type='A')
        sim.particles = [p1, p2]
        
        # Update links
        sim.update_magnetic_links()
        
        # Check that link was created
        assert len(sim.links) == 1
        assert p1.front_linked_to == 1
        assert p2.back_linked_to == 0
        assert p1.front_link_type == LinkType.MAGNETIC
        assert p2.back_link_type == LinkType.MAGNETIC
    
    def test_break_magnetic_links(self):
        """Test breaking magnetic links."""
        sim = MagneticSimulation(N_A=0, N_B=0, box_size=1000.0, r_cone=50.0)
        
        # Manually create linked particles
        p1 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A',
                     front_linked_to=1, front_link_type=LinkType.MAGNETIC)
        p2 = Particle(id=1, position=np.array([500.0, 500.0, 520.0]), np_type='A',
                     back_linked_to=0, back_link_type=LinkType.MAGNETIC)
        sim.particles = [p1, p2]
        
        # Break links
        sim.break_magnetic_links()
        
        # Check that links were removed
        assert len(sim.links) == 0
        assert p1.front_linked_to is None
        assert p2.back_linked_to is None
        assert p1.front_link_type is None
        assert p2.back_link_type is None
    
    def test_identify_chains(self):
        """Test chain identification."""
        sim = MagneticSimulation(N_A=0, N_B=0, box_size=1000.0)
        
        # Create a chain of 3 particles
        p1 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A',
                     front_linked_to=1, front_link_type=LinkType.MAGNETIC)
        p2 = Particle(id=1, position=np.array([500.0, 500.0, 510.0]), np_type='A',
                     back_linked_to=0, back_link_type=LinkType.MAGNETIC,
                     front_linked_to=2, front_link_type=LinkType.MAGNETIC)
        p3 = Particle(id=2, position=np.array([500.0, 500.0, 520.0]), np_type='A',
                     back_linked_to=1, back_link_type=LinkType.MAGNETIC)
        sim.particles = [p1, p2, p3]
        
        # Identify chains
        chains = sim.identify_magnetic_chains()
        
        # Should find one chain with 3 particles
        assert len(chains) == 1
        assert len(chains[0].particle_ids) == 3
        assert chains[0].particle_ids == [0, 1, 2]
    
    def test_step_field_on(self):
        """Test simulation step with field ON."""
        sim = MagneticSimulation(N_A=5, N_B=5, box_size=1000.0)
        initial_time = sim.time
        
        # Take one step with field ON
        sim.step(dt=0.01, field_on=True)
        
        # Time should advance
        assert sim.time > initial_time
        
        # Particles should have moved
        # (hard to test exact positions due to randomness)
    
    def test_step_field_off(self):
        """Test simulation step with field OFF."""
        sim = MagneticSimulation(N_A=5, N_B=5, box_size=1000.0)
        
        # Manually create some links
        sim.particles[0].front_linked_to = 1
        sim.particles[0].front_link_type = LinkType.MAGNETIC
        
        # Take one step with field OFF
        sim.step(dt=0.01, field_on=False)
        
        # Links should be broken
        assert sim.particles[0].front_linked_to is None
        assert len(sim.chains) == 0
    
    def test_run_simulation(self):
        """Test running a full simulation."""
        sim = MagneticSimulation(N_A=5, N_B=5, box_size=1000.0)
        
        # Run simulation with constant field
        field_protocol = constant_field(True)
        trajectory = sim.run(duration=0.1, dt=0.01, field_protocol=field_protocol)
        
        # Check trajectory
        assert len(trajectory) == 10  # 0.1 / 0.01
        assert all('time' in d for d in trajectory)
        assert all('field_on' in d for d in trajectory)
        assert all('n_chains' in d for d in trajectory)
