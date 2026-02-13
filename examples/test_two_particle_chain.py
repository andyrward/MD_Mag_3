"""
Two-particle chain formation test.

This script tests the most basic chain mechanics:
- Do two properly-spaced particles form a magnetic link?
- Do they move together as a rigid body?
- Is the diffusion coefficient correctly scaled?

This is the fundamental test that must pass before testing multi-particle systems.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from magnetic_md import MagneticSimulation, Particle

def test_two_particle_chain():
    """Run two-particle chain formation test with detailed diagnostics."""
    
    print("=" * 70)
    print("TWO-PARTICLE MAGNETIC CHAIN TEST")
    print("=" * 70)
    
    # Initialize empty simulation
    sim = MagneticSimulation(
        N_A=0, 
        N_B=0, 
        box_size=2000.0,  # Large box to avoid boundary effects
        d_np=100.0, 
        d_chain=10.0,
        r_cone=150.0,  # Search radius for link formation
        theta_cone=0.524  # 30 degrees
    )
    
    print("\n" + "=" * 70)
    print("STEP 1: INITIAL SETUP")
    print("=" * 70)
    
    # Create two particles with proper spacing
    # Center-to-center distance = d_chain = 10 nm
    # Note: d_chain is the center-to-center spacing in the simulation code
    p0 = Particle(id=0, position=np.array([500.0, 500.0, 500.0]), np_type='A')
    p1 = Particle(id=1, position=np.array([500.0, 500.0, 510.0]), np_type='A')
    
    sim.particles = [p0, p1]
    
    print(f"Box size: {sim.box_size} nm")
    print(f"Particle diameter: {sim.d_np} nm")
    print(f"Chain spacing (d_chain): {sim.d_chain} nm")
    print(f"Single particle diffusion: {sim.D_trans:.2f} nm²/s")
    print(f"\nParticle 0 position: {p0.position}")
    print(f"Particle 1 position: {p1.position}")
    
    center_to_center = np.linalg.norm(p1.position - p0.position)
    surface_to_surface = center_to_center - sim.d_np
    
    print(f"\nCenter-to-center distance: {center_to_center:.2f} nm")
    print(f"Surface-to-surface gap: {surface_to_surface:.2f} nm")
    print(f"Expected center-to-center (d_chain): {sim.d_chain} nm")
    
    if not np.isclose(center_to_center, sim.d_chain, atol=1.0):
        print("⚠️  WARNING: Center-to-center distance does not match d_chain!")
    else:
        print("✅ Center-to-center distance matches d_chain")
    
    # Test 1: Link Formation
    print("\n" + "=" * 70)
    print("STEP 2: TESTING LINK FORMATION")
    print("=" * 70)
    
    sim.update_magnetic_links()
    
    print(f"Number of links formed: {len(sim.links)}")
    print(f"\nParticle 0 link status:")
    print(f"  - front_linked_to: {p0.front_linked_to}")
    print(f"  - back_linked_to: {p0.back_linked_to}")
    print(f"  - front_link_type: {p0.front_link_type}")
    print(f"  - back_link_type: {p0.back_link_type}")
    
    print(f"\nParticle 1 link status:")
    print(f"  - front_linked_to: {p1.front_linked_to}")
    print(f"  - back_linked_to: {p1.back_linked_to}")
    print(f"  - front_link_type: {p1.front_link_type}")
    print(f"  - back_link_type: {p1.back_link_type}")
    
    if len(sim.links) == 0:
        print("\n❌ FAILED: No magnetic link formed!")
        print("   Possible issues:")
        print("   - Interaction cone parameters incorrect")
        print("   - Link formation logic broken")
        print("   - Particle spacing outside detection range")
        return False
    elif len(sim.links) == 1:
        print("\n✅ PASSED: Magnetic link formed successfully")
    else:
        print(f"\n⚠️  WARNING: Expected 1 link, found {len(sim.links)}")
    
    # Test 2: Chain Identification
    print("\n" + "=" * 70)
    print("STEP 3: TESTING CHAIN IDENTIFICATION")
    print("=" * 70)
    
    sim.identify_magnetic_chains()
    
    print(f"Number of chains: {len(sim.chains)}")
    
    if len(sim.chains) == 0:
        print("\n❌ FAILED: No chain identified despite link existing!")
        return False
    elif len(sim.chains) > 1:
        print(f"\n⚠️  WARNING: Expected 1 chain, found {len(sim.chains)}")
    
    chain = sim.chains[0]
    print(f"\nChain 0 properties:")
    print(f"  - Particle IDs: {chain.particle_ids}")
    print(f"  - Number of particles: {len(chain.particle_ids)}")
    print(f"  - Center of mass: {chain.center_of_mass}")
    print(f"  - D_eff: {chain.D_eff:.2f} nm²/s")
    print(f"  - D_single: {sim.D_trans:.2f} nm²/s")
    print(f"  - Ratio D_eff/D_single: {chain.D_eff / sim.D_trans:.3f}")
    
    expected_ratio = 1.0 / len(chain.particle_ids)
    print(f"  - Expected ratio: {expected_ratio:.3f}")
    
    if len(chain.particle_ids) != 2:
        print(f"\n❌ FAILED: Chain should have 2 particles, has {len(chain.particle_ids)}")
        return False
    else:
        print("\n✅ PASSED: Chain contains both particles")
    
    if not np.isclose(chain.D_eff / sim.D_trans, expected_ratio, rtol=0.01):
        print("\n❌ FAILED: Diffusion scaling incorrect!")
        return False
    else:
        print("✅ PASSED: Diffusion coefficient correctly scaled")
    
    # Test 3: Rigid Body Motion
    print("\n" + "=" * 70)
    print("STEP 4: TESTING RIGID BODY MOTION")
    print("=" * 70)
    
    n_steps = 100
    dt = 0.01  # seconds
    
    print(f"Running {n_steps} Brownian motion steps...")
    print(f"Time step: {dt} s")
    print(f"Total time: {n_steps * dt} s")
    
    # Track positions
    times = [0.0]
    positions_0 = [sim.particles[0].position.copy()]
    positions_1 = [sim.particles[1].position.copy()]
    separations = []
    com_positions = []
    
    # Initial separation
    sep_vec = sim.particles[1].position - sim.particles[0].position
    separations.append(np.linalg.norm(sep_vec))
    com_positions.append((sim.particles[0].position + sim.particles[1].position) / 2.0)
    
    # Run simulation
    for step in range(n_steps):
        sim.step(dt=dt, field_on=True)
        
        times.append((step + 1) * dt)
        positions_0.append(sim.particles[0].position.copy())
        positions_1.append(sim.particles[1].position.copy())
        
        sep_vec = sim.particles[1].position - sim.particles[0].position
        separations.append(np.linalg.norm(sep_vec))
        
        com = (sim.particles[0].position + sim.particles[1].position) / 2.0
        com_positions.append(com)
    
    # Convert to arrays
    times = np.array(times)
    positions_0 = np.array(positions_0)
    positions_1 = np.array(positions_1)
    separations = np.array(separations)
    com_positions = np.array(com_positions)
    
    print("\n--- Rigid Body Motion Analysis ---")
    
    # Check first displacement
    disp_0 = positions_0[1] - positions_0[0]
    disp_1 = positions_1[1] - positions_1[0]
    disp_diff = disp_1 - disp_0
    
    print(f"\nFirst step displacements:")
    print(f"  Particle 0: [{disp_0[0]:+.4f}, {disp_0[1]:+.4f}, {disp_0[2]:+.4f}] nm")
    print(f"  Particle 1: [{disp_1[0]:+.4f}, {disp_1[1]:+.4f}, {disp_1[2]:+.4f}] nm")
    print(f"  Difference: [{disp_diff[0]:+.4f}, {disp_diff[1]:+.4f}, {disp_diff[2]:+.4f}] nm")
    print(f"  |Difference|: {np.linalg.norm(disp_diff):.4f} nm")
    
    if np.linalg.norm(disp_diff) > 0.1:
        print("\n❌ FAILED: Particles not moving together (displacements differ)!")
        rigid_body_test = False
    else:
        print("\n✅ PASSED: Particles move with identical displacements")
        rigid_body_test = True
    
    # Check all displacements
    max_diff = 0.0
    for i in range(1, len(positions_0)):
        d0 = positions_0[i] - positions_0[i-1]
        d1 = positions_1[i] - positions_1[i-1]
        diff = np.linalg.norm(d1 - d0)
        if diff > max_diff:
            max_diff = diff
    
    print(f"\nMaximum displacement difference over {n_steps} steps: {max_diff:.4f} nm")
    
    if max_diff > 1.0:
        print("❌ FAILED: Large displacement differences detected")
        rigid_body_test = False
    else:
        print("✅ PASSED: All displacements consistent with rigid body motion")
    
    # Test 4: Separation Stability
    print("\n" + "=" * 70)
    print("STEP 5: TESTING SEPARATION STABILITY")
    print("=" * 70)
    
    mean_sep = np.mean(separations)
    std_sep = np.std(separations)
    min_sep = np.min(separations)
    max_sep = np.max(separations)
    
    print(f"\nSeparation statistics:")
    print(f"  Mean: {mean_sep:.4f} nm")
    print(f"  Std:  {std_sep:.4f} nm")
    print(f"  Min:  {min_sep:.4f} nm")
    print(f"  Max:  {max_sep:.4f} nm")
    print(f"  Expected (d_chain, center-to-center): {sim.d_chain} nm")
    print(f"  Deviation from expected: {abs(mean_sep - sim.d_chain):.4f} nm")
    
    if std_sep > 1.0:
        print("\n❌ FAILED: Separation is not stable (high variance)")
        separation_test = False
    else:
        print("\n✅ PASSED: Separation remains stable")
        separation_test = True
    
    if abs(mean_sep - sim.d_chain) > 2.0:
        print("⚠️  WARNING: Mean separation differs significantly from d_chain")
    
    # Test 5: Diffusion Coefficient Measurement
    print("\n" + "=" * 70)
    print("STEP 6: MEASURING DIFFUSION FROM TRAJECTORY")
    print("=" * 70)
    
    # Calculate MSD of center of mass
    com_displacements = com_positions - com_positions[0]
    msd = np.mean(com_displacements**2, axis=1)
    
    # Fit to MSD = 6*D*t (3D diffusion)
    # Use first half of trajectory for linear regime
    fit_points = n_steps // 2
    coeffs = np.polyfit(times[:fit_points], msd[:fit_points], 1)
    D_measured = coeffs[0] / 6.0  # MSD = 6*D*t
    
    print(f"\nDiffusion analysis:")
    print(f"  Theoretical D_chain: {chain.D_eff:.2f} nm²/s")
    print(f"  Measured D_chain: {D_measured:.2f} nm²/s")
    print(f"  Ratio: {D_measured / chain.D_eff:.3f}")
    
    if abs(D_measured / chain.D_eff - 1.0) > 0.3:
        print("\n⚠️  WARNING: Measured diffusion differs from theoretical")
        print("   (This is expected for short trajectories)")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING DIAGNOSTIC PLOTS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: X vs time
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(times, positions_0[:, 0], 'b-', label='Particle 0', linewidth=2)
    ax1.plot(times, positions_1[:, 0], 'r--', label='Particle 1', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (nm)')
    ax1.set_title('X Coordinate vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Y vs time
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(times, positions_0[:, 1], 'b-', label='Particle 0', linewidth=2)
    ax2.plot(times, positions_1[:, 1], 'r--', label='Particle 1', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (nm)')
    ax2.set_title('Y Coordinate vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Z vs time
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(times, positions_0[:, 2], 'b-', label='Particle 0', linewidth=2)
    ax3.plot(times, positions_1[:, 2], 'r--', label='Particle 1', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position (nm)')
    ax3.set_title('Z Coordinate vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: XY trajectory
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(positions_0[:, 0], positions_0[:, 1], 'b-o', label='Particle 0', markersize=3)
    ax4.plot(positions_1[:, 0], positions_1[:, 1], 'r-s', label='Particle 1', markersize=3)
    ax4.set_xlabel('X (nm)')
    ax4.set_ylabel('Y (nm)')
    ax4.set_title('XY Trajectory')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # Plot 5: XZ trajectory
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(positions_0[:, 0], positions_0[:, 2], 'b-o', label='Particle 0', markersize=3)
    ax5.plot(positions_1[:, 0], positions_1[:, 2], 'r-s', label='Particle 1', markersize=3)
    ax5.set_xlabel('X (nm)')
    ax5.set_ylabel('Z (nm)')
    ax5.set_title('XZ Trajectory')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # Plot 6: YZ trajectory
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(positions_0[:, 1], positions_0[:, 2], 'b-o', label='Particle 0', markersize=3)
    ax6.plot(positions_1[:, 1], positions_1[:, 2], 'r-s', label='Particle 1', markersize=3)
    ax6.set_xlabel('Y (nm)')
    ax6.set_ylabel('Z (nm)')
    ax6.set_title('YZ Trajectory')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    # Plot 7: Separation vs time
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(times, separations, 'k-', linewidth=2)
    ax7.axhline(y=sim.d_chain, color='r', linestyle='--', 
                label=f'd_chain={sim.d_chain} nm', linewidth=2)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Separation (nm)')
    ax7.set_title('Particle Separation vs Time')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: COM trajectory XY
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(com_positions[:, 0], com_positions[:, 1], 'g-o', markersize=3)
    ax8.set_xlabel('X (nm)')
    ax8.set_ylabel('Y (nm)')
    ax8.set_title('Center of Mass XY Trajectory')
    ax8.grid(True, alpha=0.3)
    ax8.axis('equal')
    
    # Plot 9: MSD vs time
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(times, msd, 'ko', markersize=4, label='MSD')
    ax9.plot(times[:fit_points], np.polyval(coeffs, times[:fit_points]), 
             'r-', linewidth=2, label=f'Fit: D={D_measured:.2f} nm²/s')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('MSD (nm²)')
    ax9.set_title('Mean Squared Displacement')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('two_particle_chain_test.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: two_particle_chain_test.png")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    all_tests_passed = (
        len(sim.links) == 1 and
        len(sim.chains) == 1 and
        len(chain.particle_ids) == 2 and
        rigid_body_test and
        separation_test
    )
    
    print("\nTest Results:")
    print(f"  ✅ Link formation: {'PASS' if len(sim.links) == 1 else 'FAIL'}")
    print(f"  ✅ Chain identification: {'PASS' if len(sim.chains) == 1 else 'FAIL'}")
    print(f"  ✅ Rigid body motion: {'PASS' if rigid_body_test else 'FAIL'}")
    print(f"  ✅ Separation stability: {'PASS' if separation_test else 'FAIL'}")
    
    if all_tests_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! Chain mechanics working correctly.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED - See diagnostic output above")
        print("=" * 70)
    
    return all_tests_passed


if __name__ == "__main__":
    success = test_two_particle_chain()
    sys.exit(0 if success else 1)
