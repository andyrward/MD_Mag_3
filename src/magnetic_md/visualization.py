"""Visualization tools for magnetic nanoparticle simulations."""

from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from .particles import Particle, MagneticChain


def plot_particles_3d(
    particles: List[Particle],
    chains: List[MagneticChain],
    box_size: float,
    title: str = "Particle Configuration",
    show_chains: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """Create 3D scatter plot of particles colored by type.
    
    Args:
        particles: List of particles to plot
        chains: List of chains
        box_size: Size of simulation box
        title: Plot title
        show_chains: Whether to show chain connections
        
    Returns:
        Figure and axes objects
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate particles by type
    type_A = [p for p in particles if p.np_type == 'A']
    type_B = [p for p in particles if p.np_type == 'B']
    
    # Plot particles
    if type_A:
        pos_A = np.array([p.position for p in type_A])
        ax.scatter(pos_A[:, 0], pos_A[:, 1], pos_A[:, 2], 
                  c='blue', s=50, alpha=0.6, label='Type A')
    
    if type_B:
        pos_B = np.array([p.position for p in type_B])
        ax.scatter(pos_B[:, 0], pos_B[:, 1], pos_B[:, 2], 
                  c='red', s=50, alpha=0.6, label='Type B')
    
    # Draw chain connections
    if show_chains and chains:
        particle_dict = {p.id: p for p in particles}
        for chain in chains:
            if len(chain.particle_ids) > 1:
                positions = [particle_dict[pid].position for pid in chain.particle_ids]
                positions = np.array(positions)
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       'k-', linewidth=2, alpha=0.5)
    
    # Set labels and limits
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    ax.set_title(title)
    ax.legend()
    
    return fig, ax


def plot_chain_statistics(trajectory: List[Dict[str, Any]]) -> plt.Figure:
    """Create time series plots of chain statistics.
    
    Args:
        trajectory: List of dictionaries with time series data
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Extract data
    times = [d['time'] for d in trajectory]
    field_states = [d['field_on'] for d in trajectory]
    n_chains = [d['n_chains'] for d in trajectory]
    avg_lengths = [d['avg_chain_length'] for d in trajectory]
    
    # Plot number of chains
    axes[0].plot(times, n_chains, 'b-', linewidth=2)
    axes[0].set_ylabel('Number of Chains')
    axes[0].grid(True, alpha=0.3)
    
    # Plot average chain length
    axes[1].plot(times, avg_lengths, 'g-', linewidth=2)
    axes[1].set_ylabel('Average Chain Length')
    axes[1].grid(True, alpha=0.3)
    
    # Plot field state with shaded regions
    axes[2].fill_between(times, 0, 1, where=field_states, 
                         alpha=0.3, color='orange', label='Field ON')
    axes[2].set_ylabel('Field State')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['OFF', 'ON'])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_particle_positions_2d(
    particles: List[Particle],
    chains: List[MagneticChain],
    box_size: float,
    projection: str = 'xy',
    title: str = "Particle Positions (2D)"
) -> Tuple[plt.Figure, plt.Axes]:
    """Create 2D projection of particle positions.
    
    Args:
        particles: List of particles to plot
        chains: List of chains
        box_size: Size of simulation box
        projection: Which plane to project onto ('xy', 'xz', or 'yz')
        title: Plot title
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Determine projection axes
    if projection == 'xy':
        idx1, idx2 = 0, 1
        xlabel, ylabel = 'X (nm)', 'Y (nm)'
    elif projection == 'xz':
        idx1, idx2 = 0, 2
        xlabel, ylabel = 'X (nm)', 'Z (nm)'
    elif projection == 'yz':
        idx1, idx2 = 1, 2
        xlabel, ylabel = 'Y (nm)', 'Z (nm)'
    else:
        raise ValueError(f"Invalid projection: {projection}")
    
    # Separate particles by type
    type_A = [p for p in particles if p.np_type == 'A']
    type_B = [p for p in particles if p.np_type == 'B']
    
    # Plot particles
    if type_A:
        pos_A = np.array([p.position for p in type_A])
        ax.scatter(pos_A[:, idx1], pos_A[:, idx2], 
                  c='blue', s=50, alpha=0.6, label='Type A')
    
    if type_B:
        pos_B = np.array([p.position for p in type_B])
        ax.scatter(pos_B[:, idx1], pos_B[:, idx2], 
                  c='red', s=50, alpha=0.6, label='Type B')
    
    # Draw chain connections
    if chains:
        particle_dict = {p.id: p for p in particles}
        for chain in chains:
            if len(chain.particle_ids) > 1:
                positions = [particle_dict[pid].position for pid in chain.particle_ids]
                positions = np.array(positions)
                ax.plot(positions[:, idx1], positions[:, idx2], 
                       'k-', linewidth=2, alpha=0.5)
    
    # Set labels and limits
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def animate_simulation(
    trajectory: List[Dict[str, Any]],
    particles_history: List[List[Particle]],
    box_size: float,
    interval: int = 100,
    projection: str = 'xy'
) -> FuncAnimation:
    """Create animation of particle motion and chain formation.
    
    Args:
        trajectory: List of trajectory data
        particles_history: List of particle states at each timestep
        box_size: Size of simulation box
        interval: Time between frames in milliseconds
        projection: Which plane to project onto ('xy', 'xz', or 'yz')
        
    Returns:
        Animation object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Determine projection axes
    if projection == 'xy':
        idx1, idx2 = 0, 1
        xlabel, ylabel = 'X (nm)', 'Y (nm)'
    elif projection == 'xz':
        idx1, idx2 = 0, 2
        xlabel, ylabel = 'X (nm)', 'Z (nm)'
    elif projection == 'yz':
        idx1, idx2 = 1, 2
        xlabel, ylabel = 'Y (nm)', 'Z (nm)'
    else:
        raise ValueError(f"Invalid projection: {projection}")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Initialize scatter plots
    scatter_A = ax.scatter([], [], c='blue', s=50, alpha=0.6, label='Type A')
    scatter_B = ax.scatter([], [], c='red', s=50, alpha=0.6, label='Type B')
    ax.legend()
    
    title_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        verticalalignment='top')
    
    def init():
        scatter_A.set_offsets(np.empty((0, 2)))
        scatter_B.set_offsets(np.empty((0, 2)))
        title_text.set_text('')
        return scatter_A, scatter_B, title_text
    
    def update(frame):
        particles = particles_history[frame]
        data = trajectory[frame]
        
        # Update particle positions
        type_A = [p for p in particles if p.np_type == 'A']
        type_B = [p for p in particles if p.np_type == 'B']
        
        if type_A:
            pos_A = np.array([p.position for p in type_A])
            scatter_A.set_offsets(pos_A[:, [idx1, idx2]])
        
        if type_B:
            pos_B = np.array([p.position for p in type_B])
            scatter_B.set_offsets(pos_B[:, [idx1, idx2]])
        
        # Update title
        field_str = "ON" if data['field_on'] else "OFF"
        title_text.set_text(f"t={data['time']:.2f}s, Field={field_str}, Chains={data['n_chains']}")
        
        return scatter_A, scatter_B, title_text
    
    anim = FuncAnimation(fig, update, init_func=init, 
                        frames=len(particles_history),
                        interval=interval, blit=True)
    
    return anim
