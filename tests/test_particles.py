"""Tests for particle, link, and chain data structures."""

import numpy as np
from src.magnetic_md.particles import LinkType, Link, Particle, MagneticChain


class TestLinkType:
    """Tests for LinkType enum."""
    
    def test_link_types_exist(self):
        """Test that link types are defined."""
        assert LinkType.MAGNETIC.value == "magnetic"
        assert LinkType.ANTIGEN.value == "antigen"


class TestLink:
    """Tests for Link dataclass."""
    
    def test_create_magnetic_link(self):
        """Test creating a magnetic link."""
        link = Link(
            particle1_id=0,
            particle2_id=1,
            link_type=LinkType.MAGNETIC,
            pole1="front",
            pole2="back",
            time_formed=0.0
        )
        assert link.particle1_id == 0
        assert link.particle2_id == 1
        assert link.link_type == LinkType.MAGNETIC
        assert link.pole1 == "front"
        assert link.pole2 == "back"
        assert link.time_formed == 0.0
    
    def test_create_antigen_link(self):
        """Test creating an antigen link."""
        link = Link(
            particle1_id=0,
            particle2_id=1,
            link_type=LinkType.ANTIGEN,
            pole1="front",
            pole2="back",
            time_formed=1.5
        )
        assert link.link_type == LinkType.ANTIGEN
        assert link.time_formed == 1.5


class TestParticle:
    """Tests for Particle dataclass."""
    
    def test_create_particle(self):
        """Test creating a particle."""
        position = np.array([100.0, 200.0, 300.0])
        particle = Particle(
            id=0,
            position=position,
            np_type='A'
        )
        assert particle.id == 0
        assert np.array_equal(particle.position, position)
        assert particle.np_type == 'A'
        assert particle.front_linked_to is None
        assert particle.back_linked_to is None
    
    def test_particle_with_links(self):
        """Test particle with link information."""
        position = np.array([0.0, 0.0, 0.0])
        particle = Particle(
            id=0,
            position=position,
            np_type='B',
            front_linked_to=1,
            back_linked_to=2,
            front_link_type=LinkType.MAGNETIC,
            back_link_type=LinkType.MAGNETIC
        )
        assert particle.front_linked_to == 1
        assert particle.back_linked_to == 2
        assert particle.front_link_type == LinkType.MAGNETIC
        assert particle.back_link_type == LinkType.MAGNETIC


class TestMagneticChain:
    """Tests for MagneticChain dataclass."""
    
    def test_create_chain(self):
        """Test creating a magnetic chain."""
        chain = MagneticChain(id=0, particle_ids=[0, 1, 2])
        assert chain.id == 0
        assert chain.particle_ids == [0, 1, 2]
        assert chain.D_eff == 0.0
    
    def test_update_center_of_mass(self):
        """Test updating chain center of mass."""
        # Create particles
        particles = [
            Particle(id=0, position=np.array([0.0, 0.0, 0.0]), np_type='A'),
            Particle(id=1, position=np.array([0.0, 0.0, 10.0]), np_type='B'),
            Particle(id=2, position=np.array([0.0, 0.0, 20.0]), np_type='A'),
        ]
        
        # Create chain
        chain = MagneticChain(id=0, particle_ids=[0, 1, 2])
        chain.update_center_of_mass(particles)
        
        # Check center of mass
        expected_com = np.array([0.0, 0.0, 10.0])
        assert np.allclose(chain.center_of_mass, expected_com)
    
    def test_calculate_diffusion(self):
        """Test diffusion coefficient calculation."""
        D_single = 100.0
        
        # Chain with 1 particle
        chain1 = MagneticChain(id=0, particle_ids=[0])
        chain1.calculate_diffusion(D_single)
        assert chain1.D_eff == D_single
        
        # Chain with 2 particles
        chain2 = MagneticChain(id=1, particle_ids=[0, 1])
        chain2.calculate_diffusion(D_single)
        assert chain2.D_eff == D_single / 2
        
        # Chain with 4 particles
        chain4 = MagneticChain(id=2, particle_ids=[0, 1, 2, 3])
        chain4.calculate_diffusion(D_single)
        assert chain4.D_eff == D_single / 4
    
    def test_empty_chain(self):
        """Test behavior of empty chain."""
        chain = MagneticChain(id=0)
        chain.update_center_of_mass([])
        chain.calculate_diffusion(100.0)
        
        assert len(chain.particle_ids) == 0
        assert np.array_equal(chain.center_of_mass, np.zeros(3))
        assert chain.D_eff == 0.0
