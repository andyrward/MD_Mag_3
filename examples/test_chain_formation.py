"""Example script demonstrating magnetic chain formation.

This example shows:
1. Initializing a magnetic nanoparticle simulation
2. Defining a square wave field protocol (OFF 5s, ON 5s)
3. Running the simulation
4. Visualizing chain formation statistics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from magnetic_md import (
    MagneticSimulation,
    square_wave,
)
from magnetic_md.visualization import (
    plot_chain_statistics,
    plot_particles_3d,
    plot_particle_positions_2d,
)


def main():
    """Run example simulation and generate visualizations."""
    print("=" * 60)
    print("Magnetic Nanoparticle Chain Formation Example")
    print("=" * 60)
    
    # Initialize simulation
    print("\nInitializing simulation...")
    print("  - N_A = 20 (type A particles)")
    print("  - N_B = 20 (type B particles)")
    print("  - Box size = 5000 nm")
    print("  - Particle diameter = 100 nm")
    
    np.random.seed(42)  # For reproducibility
    sim = MagneticSimulation(
        N_A=20,
        N_B=20,
        box_size=5000,
        d_np=100,
        d_chain=10,
        r_cone=150,
        theta_cone=0.524  # 30 degrees
    )
    
    print(f"  - Diffusion coefficient: {sim.D_trans:.2f} nm²/s")
    
    # Define field protocol: OFF 5s, ON 5s
    print("\nField protocol: Square wave")
    print("  - OFF for 5 seconds")
    print("  - ON for 5 seconds")
    print("  - Repeating...")
    
    field_protocol = square_wave(t_on=5.0, t_off=5.0, start_on=False)
    
    # Run simulation
    print("\nRunning simulation...")
    print("  - Duration: 10 seconds")
    print("  - Time step: 0.01 seconds")
    
    trajectory = sim.run(duration=10.0, dt=0.01, field_protocol=field_protocol)
    
    print(f"  - Completed {len(trajectory)} steps")
    
    # Analyze results
    print("\nSimulation results:")
    final_state = trajectory[-1]
    print(f"  - Final number of chains: {final_state['n_chains']}")
    print(f"  - Average chain length: {final_state['avg_chain_length']:.2f}")
    print(f"  - Maximum chain length: {final_state['max_chain_length']}")
    
    # Find peak statistics
    max_chains = max(d['n_chains'] for d in trajectory)
    max_avg_length = max(d['avg_chain_length'] for d in trajectory)
    
    print(f"\nPeak statistics:")
    print(f"  - Maximum number of chains: {max_chains}")
    print(f"  - Maximum average chain length: {max_avg_length:.2f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # 1. Chain statistics over time
    fig1 = plot_chain_statistics(trajectory)
    fig1.suptitle("Chain Formation Dynamics", fontsize=14, fontweight='bold')
    plt.savefig('chain_statistics.png', dpi=150, bbox_inches='tight')
    print("  - Saved: chain_statistics.png")
    
    # 2. Final 3D particle configuration
    plot_particles_3d(
        sim.particles,
        sim.chains,
        sim.box_size,
        title="Final Particle Configuration (3D)",
        show_chains=True
    )
    plt.savefig('particles_3d.png', dpi=150, bbox_inches='tight')
    print("  - Saved: particles_3d.png")
    
    # 3. 2D projections
    plot_particle_positions_2d(
        sim.particles,
        sim.chains,
        sim.box_size,
        projection='xy',
        title="Particle Positions (XY Projection)"
    )
    plt.savefig('particles_xy.png', dpi=150, bbox_inches='tight')
    print("  - Saved: particles_xy.png")
    
    plot_particle_positions_2d(
        sim.particles,
        sim.chains,
        sim.box_size,
        projection='xz',
        title="Particle Positions (XZ Projection)"
    )
    plt.savefig('particles_xz.png', dpi=150, bbox_inches='tight')
    print("  - Saved: particles_xz.png")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Optionally display plots
    # plt.show()


if __name__ == "__main__":
    main()
