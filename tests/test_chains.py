"""Tests for chain formation and dynamics."""

import numpy as np
from src.magnetic_md.simulation import MagneticSimulation
from src.magnetic_md.particles import Particle, MagneticChain, LinkType
from src.magnetic_md.fields import constant_field, square_wave, delayed_activation


class TestChainFormation:
    """Tests for chain formation dynamics."""
    
    def test_chain_forms_with_field_on(self):
        """Test that chains form when field is ON."""
        # Create simulation with particles close together
        sim = MagneticSimulation(N_A=3, N_B=3, box_size=200.0, r_cone=100.0)
        
        # Position particles in a line along z-axis
        sim.particles[0].position = np.array([100.0, 100.0, 50.0])
        sim.particles[1].position = np.array([100.0, 100.0, 70.0])
        sim.particles[2].position = np.array([100.0, 100.0, 90.0])
        
        # Update magnetic links
        sim.update_magnetic_links()
        
        # Should create links
        assert len(sim.links) >= 1
        
        # Identify chains
        chains = sim.identify_magnetic_chains()
        
        # Should have at least one chain
        assert len(chains) >= 1
    
    def test_chains_dissolve_with_field_off(self):
        """Test that chains dissolve when field turns OFF."""
        sim = MagneticSimulation(N_A=5, N_B=5, box_size=1000.0)
        
        # Create some magnetic links
        sim.particles[0].front_linked_to = 1
        sim.particles[0].front_link_type = LinkType.MAGNETIC
        sim.particles[1].back_linked_to = 0
        sim.particles[1].back_link_type = LinkType.MAGNETIC
        
        # Identify chains
        sim.identify_magnetic_chains()
        assert len(sim.chains) > 0
        
        # Break links (field OFF)
        sim.break_magnetic_links()
        
        # Chains should be gone
        assert len(sim.chains) == 0
        assert len(sim.links) == 0
    
    def test_chain_diffusion_scaling(self):
        """Test that chain diffusion scales as 1/N."""
        D_single = 100.0
        
        # Chain with different lengths
        chain1 = MagneticChain(id=0, particle_ids=[0])
        chain2 = MagneticChain(id=1, particle_ids=[0, 1])
        chain3 = MagneticChain(id=2, particle_ids=[0, 1, 2])
        
        chain1.calculate_diffusion(D_single)
        chain2.calculate_diffusion(D_single)
        chain3.calculate_diffusion(D_single)
        
        # Check scaling
        assert chain1.D_eff == D_single
        assert chain2.D_eff == D_single / 2
        assert chain3.D_eff == D_single / 3
        
        # Check ratios
        assert np.isclose(chain1.D_eff / chain2.D_eff, 2.0)
        assert np.isclose(chain1.D_eff / chain3.D_eff, 3.0)
    
    def test_enforce_chain_geometry(self):
        """Test that particles in chains maintain x,y offsets while z-spacing is enforced."""
        sim = MagneticSimulation(N_A=0, N_B=0, box_size=1000.0, d_chain=10.0)
        
        # Create particles with different x,y positions (not aligned)
        p1 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A')
        p2 = Particle(id=1, position=np.array([510.0, 505.0, 510.0]), np_type='A')
        p3 = Particle(id=2, position=np.array([505.0, 495.0, 490.0]), np_type='A')
        sim.particles = [p1, p2, p3]
        
        # Store initial x,y positions
        initial_xy = [(p.position[0], p.position[1]) for p in sim.particles]
        
        # Create chain
        chain = MagneticChain(id=0, particle_ids=[0, 1, 2])
        chain.update_center_of_mass(sim.particles, sim.box_size)
        
        # Enforce geometry using the new method
        sim.enforce_chain_geometry_initial([chain])
        
        # Check that x,y positions maintained relative structure (not all same!)
        # They should be clustered around COM but NOT identical
        x_positions = [p.position[0] for p in sim.particles]
        y_positions = [p.position[1] for p in sim.particles]
        
        # Check they're not all the same (should have variation)
        # Round to 2 decimal places (0.01 nm precision) to account for floating point errors
        assert len(set(np.round(x_positions, 2))) > 1, "All x positions are identical!"
        assert len(set(np.round(y_positions, 2))) > 1, "All y positions are identical!"
        
        # Check z-spacing is correct
        z_positions = sorted([p.position[2] for p in sim.particles])
        assert np.isclose(z_positions[1] - z_positions[0], sim.d_chain, rtol=sim.CHAIN_SPACING_TOLERANCE)
        assert np.isclose(z_positions[2] - z_positions[1], sim.d_chain, rtol=sim.CHAIN_SPACING_TOLERANCE)
    
    def test_rigid_body_motion_preserves_structure(self):
        """Test that Brownian motion of chains preserves relative positions."""
        sim = MagneticSimulation(N_A=0, N_B=0, box_size=10000.0, d_chain=10.0)
        
        # Create a chain with varied x,y positions
        p1 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A')
        p2 = Particle(id=1, position=np.array([505.0, 502.0, 510.0]), np_type='A')
        p3 = Particle(id=2, position=np.array([503.0, 498.0, 520.0]), np_type='A')
        sim.particles = [p1, p2, p3]
        
        # Store relative positions
        com_initial = np.mean([p.position for p in sim.particles], axis=0)
        relative_positions = [p.position - com_initial for p in sim.particles]
        
        # Create chain
        chain = MagneticChain(id=0, particle_ids=[0, 1, 2])
        chain.update_center_of_mass(sim.particles, sim.box_size)
        chain.calculate_diffusion(sim.D_trans)
        sim.chains = [chain]
        
        # Move chain
        sim.move_chains_brownian([chain], dt=0.01)
        
        # Check that relative positions are preserved
        com_final = np.mean([p.position for p in sim.particles], axis=0)
        relative_positions_final = [p.position - com_final for p in sim.particles]
        
        for i in range(3):
            for j in range(3):  # x, y, z
                assert np.isclose(relative_positions[i][j], relative_positions_final[i][j], atol=0.1), \
                    f"Relative position changed for particle {i}, component {j}"


class TestFieldProtocols:
    """Tests for field protocol functions."""
    
    def test_constant_field_on(self):
        """Test constant field ON protocol."""
        protocol = constant_field(True)
        assert protocol(0.0) is True
        assert protocol(1.0) is True
        assert protocol(10.0) is True
    
    def test_constant_field_off(self):
        """Test constant field OFF protocol."""
        protocol = constant_field(False)
        assert protocol(0.0) is False
        assert protocol(1.0) is False
        assert protocol(10.0) is False
    
    def test_square_wave_start_off(self):
        """Test square wave starting with field OFF."""
        protocol = square_wave(t_on=5.0, t_off=5.0, start_on=False)
        
        # First 5 seconds: OFF
        assert protocol(0.0) is False
        assert protocol(2.5) is False
        assert protocol(4.9) is False
        
        # Next 5 seconds: ON
        assert protocol(5.0) is True
        assert protocol(7.5) is True
        assert protocol(9.9) is True
        
        # Next cycle: OFF
        assert protocol(10.0) is False
        assert protocol(12.5) is False
    
    def test_square_wave_start_on(self):
        """Test square wave starting with field ON."""
        protocol = square_wave(t_on=5.0, t_off=5.0, start_on=True)
        
        # First 5 seconds: ON
        assert protocol(0.0) is True
        assert protocol(2.5) is True
        assert protocol(4.9) is True
        
        # Next 5 seconds: OFF
        assert protocol(5.0) is False
        assert protocol(7.5) is False
        assert protocol(9.9) is False
        
        # Next cycle: ON
        assert protocol(10.0) is True
    
    def test_delayed_activation(self):
        """Test delayed activation protocol."""
        protocol = delayed_activation(t_delay=5.0)
        
        # Before delay: OFF
        assert protocol(0.0) is False
        assert protocol(2.5) is False
        assert protocol(4.9) is False
        
        # After delay: ON
        assert protocol(5.0) is True
        assert protocol(7.5) is True
        assert protocol(10.0) is True
    
    def test_simulation_with_square_wave(self):
        """Test simulation with square wave field protocol."""
        sim = MagneticSimulation(N_A=5, N_B=5, box_size=1000.0)
        
        # Run with square wave
        protocol = square_wave(t_on=0.05, t_off=0.05, start_on=False)
        trajectory = sim.run(duration=0.2, dt=0.01, field_protocol=protocol)
        
        # Check that field state changes
        field_states = [d['field_on'] for d in trajectory]
        assert True in field_states  # Field was ON at some point
        assert False in field_states  # Field was OFF at some point
        
        # Check that field changes at expected times
        # Note: trajectory records state after each step completes
        assert field_states[0] is False  # At t=0.01, still in OFF phase
        assert field_states[5] is True   # At t=0.06, in ON phase
        assert field_states[11] is False  # At t=0.12, back to OFF phase
