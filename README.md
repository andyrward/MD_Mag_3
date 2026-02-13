# MD_Mag_3: Magnetic Nanoparticle MD Simulation

A Python package for simulating magnetic nanoparticle chain formation with field ON/OFF cycling.

## Overview

MD_Mag_3 implements molecular dynamics (MD) simulations of magnetic nanoparticles that form chains when an external magnetic field is applied. The simulation includes:

- **Magnetic chain formation**: Particles link together along the field direction (z-axis) when field is ON
- **Field cycling**: Support for various field protocols (constant, square wave, delayed activation)
- **Brownian dynamics**: Realistic particle motion with scaled diffusion for chains
- **Visualization tools**: 3D plots, 2D projections, and time series analysis

## Features

- Two particle types (A and B) for future biochemical interactions
- Periodic boundary conditions for bulk behavior
- Interaction cone model for magnetic linking
- Rigid body chain dynamics with 1/N diffusion scaling
- Automatic chain identification and tracking
- Comprehensive visualization and analysis tools

## Installation

### Requirements

- Python >= 3.12
- numpy >= 2.4.2
- matplotlib >= 3.10.8
- pytest >= 9.0.2 (for testing)

### Install from source

```bash
git clone https://github.com/andyrward/MD_Mag_3.git
cd MD_Mag_3
pip install -e .
```

Or using uv (recommended):

```bash
git clone https://github.com/andyrward/MD_Mag_3.git
cd MD_Mag_3
uv sync
```

## Quick Start

```python
from magnetic_md import MagneticSimulation, square_wave
from magnetic_md.visualization import plot_chain_statistics, plot_particles_3d

# Initialize simulation
sim = MagneticSimulation(
    N_A=20,           # 20 type A particles
    N_B=20,           # 20 type B particles
    box_size=5000,    # 5000 nm box
    d_np=100          # 100 nm particle diameter
)

# Define field protocol: OFF 5s, ON 5s, repeating
field_protocol = square_wave(t_on=5.0, t_off=5.0, start_on=False)

# Run simulation
trajectory = sim.run(duration=10.0, dt=0.01, field_protocol=field_protocol)

# Visualize results
plot_chain_statistics(trajectory)
plot_particles_3d(sim.particles, sim.chains, sim.box_size, "Final State")
```

## Project Structure

```
MD_Mag_3/
├── src/
│   └── magnetic_md/
│       ├── __init__.py          # Package initialization
│       ├── particles.py         # Particle, Link, Chain data structures
│       ├── simulation.py        # Main simulation engine
│       ├── fields.py            # Magnetic field protocols
│       └── visualization.py     # Plotting and analysis tools
├── tests/
│   ├── test_particles.py        # Tests for data structures
│   ├── test_simulation.py       # Tests for simulation engine
│   └── test_chains.py           # Tests for chain dynamics
├── examples/
│   └── test_chain_formation.py  # Example script
└── README.md                    # This file
```

## Physics Model

### Diffusion

Particles undergo Brownian motion with diffusion coefficient calculated using the Stokes-Einstein relation:

```
D = kT / (3 * π * η * d_np)
```

where:
- kT = 4.14 pN·nm (thermal energy at 300K)
- η = 1e-9 pN·s/nm² (water viscosity)
- d_np = particle diameter

### Chain Formation

Particles form magnetic links when they are within an interaction cone:
- **Cone radius**: r_cone (default 20 nm)
- **Cone half-angle**: θ_cone (default 30° = 0.524 rad)
- **Direction**: Front (+z) or back (-z) pole

### Chain Dynamics

Chains move as rigid bodies with scaled diffusion:
```
D_chain = D_single / N
```

where N is the number of particles in the chain.

## Examples

See the `examples/` directory for complete examples:

- `test_chain_formation.py`: Basic chain formation with square wave field protocol

Run an example:

```bash
python examples/test_chain_formation.py
```

## Testing

### Basic Chain Test

Validate fundamental chain mechanics with 2 particles before testing complex multi-particle scenarios:

```bash
python examples/test_two_particle_chain.py
```

This validates:
- Magnetic link formation between properly-spaced particles
- Chain identification and tracking
- Rigid body motion (particles move together)
- Separation stability (constant spacing maintained)
- Diffusion coefficient scaling (D_chain = D_single / N)

### Unit Tests

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=src/magnetic_md --cov-report=html
```

## Documentation

### Field Protocols

Create custom field protocols using the provided functions:

```python
from magnetic_md.fields import constant_field, square_wave, delayed_activation

# Constant field ON
protocol1 = constant_field(True)

# Square wave: 5s ON, 5s OFF
protocol2 = square_wave(t_on=5.0, t_off=5.0, start_on=False)

# Delayed activation: OFF until t=10s, then ON
protocol3 = delayed_activation(t_delay=10.0)
```

### Visualization

Available plotting functions:

- `plot_particles_3d()`: 3D scatter plot with chain connections
- `plot_chain_statistics()`: Time series of chain formation
- `plot_particle_positions_2d()`: 2D projections (xy, xz, yz)
- `animate_simulation()`: Create animations (requires particle history)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{md_mag_3,
  title = {MD_Mag_3: Magnetic Nanoparticle MD Simulation},
  author = {Ward, Andy R.},
  year = {2026},
  url = {https://github.com/andyrward/MD_Mag_3}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
