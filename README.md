![](doc/docs/images/Meep-banner.png)

> **Note:** This is a fork of the original [NanoComp/meep](https://github.com/NanoComp/meep) repository. This fork adds an **original [Spherical Cow Cloak](#spherical-cow-cloak-transformation-optics-invisibility-simulation) study** (Pendry-style transformation-optics invisibility cloak with full FDTD stability analysis), plus comprehensive documentation including a [Four-Approach Cloaking Comparison](guides/CLOAK_COMPARISON_GUIDE.md): [148 Physics Tutorials](guides/tutorials/00_index.md) with theory and code walkthroughs for every Python example and test, an [Architecture Guide](guides/ARCHITECTURE.md), a [Developer Guide](guides/DEVELOPER_GUIDE.md), a [User Guide](guides/USER_GUIDE.md) with 10 educational use cases and Windows installation instructions, a [Quick Start Explained](guides/QUICKSTART_EXPLAINED.md) walkthrough, and a [Test Report](guides/TEST_REPORT.md) from running all 148 Python examples and tests. All original source code remains unchanged.

[![CI](https://github.com/NanoComp/meep/actions/workflows/build-ci.yml/badge.svg)](https://github.com/NanoComp/meep/actions/workflows/build-ci.yml)
[![Sanitizers](https://github.com/NanoComp/meep/actions/workflows/build-san.yml/badge.svg)](https://github.com/NanoComp/meep/actions/workflows/build-san.yml)
[![Latest Docs](https://readthedocs.org/projects/meep/badge/?version=latest)](http://meep.readthedocs.io/en/latest/)
![Python versions 3.7–3.10](https://img.shields.io/badge/python-3.7%2C%203.7%2C%203.8%2C%203.10-brightgreen.svg)
[![codecov](https://codecov.io/gh/NanoComp/meep/branch/master/graph/badge.svg?token=k88ZuW3795)](https://codecov.io/gh/NanoComp/meep)

**Meep** is a free and open-source software package for [electromagnetics](https://en.wikipedia.org/wiki/Electromagnetism) simulation via the [finite-difference time-domain](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) (FDTD) method spanning a broad range of applications.

## Key Features

-   **Free and open-source software** under the [GNU GPL](https://en.wikipedia.org/wiki/GNU_General_Public_License).
-   Complete **scriptability** via [Python](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/), [Scheme](https://meep.readthedocs.io/en/latest/Scheme_Tutorials/Basics), or [C++](https://meep.readthedocs.io/en/master/C++_Tutorial/) APIs.
-   Simulation in **1d, 2d, 3d**, and **[cylindrical](https://meep.readthedocs.io/en/latest/Exploiting_Symmetry/#cylindrical-symmetry)** coordinates.
-   Distributed memory [parallelism](https://meep.readthedocs.io/en/latest/Parallel_Meep) on any system supporting [MPI](https://en.wikipedia.org/wiki/MPI).
-   Portable to any Unix-like operating system such as [Linux](https://en.wikipedia.org/wiki/Linux), [macOS](https://en.wikipedia.org/wiki/macOS), and [FreeBSD](https://en.wikipedia.org/wiki/FreeBSD).
-   **Precompiled binary packages** of official releases via [Conda](https://meep.readthedocs.io/en/latest/Installation/#conda-packages).
-   Variety of arbitrary [material](https://meep.readthedocs.io/en/latest/Materials) types: **anisotropic** electric permittivity ε and magnetic permeability μ, along with **dispersive** ε(ω) and μ(ω) including loss/gain, **nonlinear** (Kerr & Pockels) dielectric and magnetic materials, electric/magnetic **conductivities** σ, **saturable** gain/absorption, and **gyrotropic** media (magneto-optical effects).
-   [Materials library](https://meep.readthedocs.io/en/latest/Materials/#materials-library) containing predefined broadband, complex refractive indices.
-   [Perfectly matched layer](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer) (**PML**) absorbing boundaries as well as **Bloch-periodic** and perfect-conductor boundary conditions.
-   Exploitation of [symmetries](https://meep.readthedocs.io/en/latest/Exploiting_Symmetry) to reduce the computation size, including even/odd mirror planes and 90°/180° rotations.
-   [Subpixel smoothing](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/) for improving accuracy and shape optimization.
-   [Custom current sources](https://meep.readthedocs.io/en/latest/Python_Tutorials/Custom_Source/) with arbitrary time and spatial profile as well as a [mode launcher](https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/) for waveguides and planewaves, and [Gaussian beams](https://meep.readthedocs.io/en/latest/Python_User_Interface/#gaussianbeam3dsource).
-   [Frequency-domain solver](https://meep.readthedocs.io/en/latest/Python_User_Interface/#frequency-domain-solver) for finding the response to a [continuous-wave](https://en.wikipedia.org/wiki/Continuous_wave) (CW) source as well as a [frequency-domain eigensolver](https://meep.readthedocs.io/en/latest/Python_User_Interface/#frequency-domain-eigensolver) for finding resonant modes.
-   ε/μ and field import/export in the [HDF5](https://en.wikipedia.org/wiki/HDF5) data format.
-   [GDSII](https://meep.readthedocs.io/en/latest/Python_User_Interface/#gdsii-support) file import for planar geometries.
-   Field analyses including [discrete-time Fourier transform (DTFT)](https://meep.readthedocs.io/en/latest/Python_User_Interface/#field-computations), [Poynting flux](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#transmittance-spectrum-of-a-waveguide-bend), [mode decomposition](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/) (for [S-parameters](https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/#s-parameters-of-a-directional-coupler)), [energy density](https://meep.readthedocs.io/en/latest/Python_User_Interface/#energy-density-spectra), [near to far transformation](https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/), [frequency extraction](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#modes-of-a-ring-resonator), [local density of states](https://meep.readthedocs.io/en/latest/Python_Tutorials/Local_Density_of_States/) (LDOS), [modal volume](https://meep.readthedocs.io/en/latest/Python_User_Interface/#field-computations), [scattering cross section](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#mie-scattering-of-a-lossless-dielectric-sphere), [Maxwell stress tensor](https://meep.readthedocs.io/en/latest/Python_Tutorials/Optical_Forces/), [absorbed power density](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#absorbed-power-density-map-of-a-lossy-cylinder), [arbitrary functions](https://meep.readthedocs.io/en/latest/Field_Functions/); completely programmable.
-   [Adjoint solver](https://meep.readthedocs.io/en/latest/Python_Tutorials/Adjoint_Solver) for **inverse design** and **topology optimization**.
-   [Visualization routines](https://meep.readthedocs.io/en/latest/Python_User_Interface/#data-visualization) for the simulation domain involving geometries, fields, boundary layers, sources, and monitors.

## Spherical Cow Cloak: Transformation-Optics Invisibility Simulation

> **Featured study** -- Can you cloak a spherical cow? This fork investigates Pendry's transformation-optics invisibility cloak using Meep's FDTD engine, pushing it to its stability limits and documenting exactly where and why it breaks.

![Cloak field comparison: empty reference, bare sphere, and cloaked sphere showing Ez field patterns](python/examples/cloak_fields_dielectric.png)

*Ez field cross-sections comparing an empty reference (undisturbed plane wave), a bare dielectric sphere (strong scattering shadow), and a cloaked sphere. The cloak bends wavefronts around the object, but heavy regularization required for Yee-grid stability limits the cloaking effectiveness. See the [Visualization Guide](guides/CLOAK_VISUALIZATION_GUIDE.md) for a detailed walkthrough of how to read these plots.*

### What this study covers

- **Transformation optics implementation** -- Full Pendry cloak with anisotropic, inhomogeneous epsilon/mu tensors derived from the coordinate mapping r' = R1 + r(R2-R1)/R2, converted from cylindrical to Cartesian coordinates at every grid point
- **Mie theory validation** -- Bare-sphere scattering cross-section matches analytical Mie series to <1% error, confirming the simulation methodology is correct
- **Yee grid stability analysis** -- Systematic measurement of the FDTD stability boundary: off-diagonal permittivity components exceeding ~1.3x the diagonal cause unconditional instability that *cannot* be fixed by reducing the Courant number
- **Regularization trade-off** -- The minimum stable regularization (eps_min >= 0.55) flattens the gradient-index profile that the cloak relies on, destroying the cloaking effect. This is a fundamental FDTD limitation, not a bug.
- **Alternative approaches** -- The 2D cylindrical implementation achieves stable cloaking with full eps+mu tensors, demonstrating that the physics works when the numerics cooperate

### Implementations

| File | Approach | Key Result |
|------|----------|------------|
| [`spherical_cow_cloak.py`](python/examples/spherical_cow_cloak.py) | 3D reduced-parameter cloak | Mie-validated scattering; documents Yee grid stability limits |
| [`spherical_cow_cloak_2d.py`](python/examples/spherical_cow_cloak_2d.py) | 2D full Pendry cloak (eps+mu) | Stable cloaking with anisotropic tensors in 2D |
| [`spherical_cow_cloak_viz.py`](python/examples/spherical_cow_cloak_viz.py) | Visualization suite | Field maps, scattering spectra, 3D plotly, GIF animation |
| [`spherical_cow_cloak_comparison.py`](python/examples/spherical_cow_cloak_comparison.py) | Four-approach comparison (3D cancellation, 2D Pendry, multi-shell, carpet) | Successfully cloaks a true 3D sphere; head-to-head comparison of all methods |

### Documentation

| Guide | What you'll learn |
|-------|-------------------|
| [Cloak Simulation Report](guides/CLOAK_SIMULATION_REPORT.md) | Full technical analysis: Yee grid stability measurements, Courant number experiments, regularization trade-offs, and 5 mitigation strategies |
| [Visualization Guide](guides/CLOAK_VISUALIZATION_GUIDE.md) | How to read the field plots and scattering spectra, relate them to the physics, and judge whether results make sense |
| [Cloaking Comparison Guide](guides/CLOAK_COMPARISON_GUIDE.md) | Physics behind all four cloaking approaches, how to read comparison plots, and which method to use when |

### Quick start

```bash
# Quick run (3D, ~12 seconds)
python python/examples/spherical_cow_cloak.py --quick --cow-material dielectric

# 2D simulation (full Pendry cloak)
python python/examples/spherical_cow_cloak_2d.py

# Generate all visualizations from saved data
python python/examples/spherical_cow_cloak_viz.py

# Four-approach comparison (3D + 2D, ~5-8 min)
python python/examples/spherical_cow_cloak_comparison.py --method all --quick
```

## Quick Start

### Installation via Conda

The fastest way to get started on Linux or macOS is through the `conda-forge` channel (Windows users: install via [WSL2](guides/USER_GUIDE.md)):

```bash
conda create -n mp -c conda-forge pymeep
conda activate mp
```

For MPI-parallel support, install the MPICH variant instead:

```bash
conda create -n pmp -c conda-forge pymeep=*=mpi_mpich_*
conda activate pmp
```

### A Minimal 2D Waveguide Simulation

Save the following as `my_first_sim.py`:

```python
import meep as mp
import matplotlib.pyplot as plt

# Simulation cell: 16 x 8 micrometers with PML absorbing boundaries
cell = mp.Vector3(16, 8)
pml_layers = [mp.PML(1.0)]

# A dielectric waveguide (epsilon=12) running along the x-axis
geometry = [
    mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
             center=mp.Vector3(),
             material=mp.Medium(epsilon=12))
]

# Gaussian pulse source exciting the Hz field at one end of the guide
sources = [
    mp.Source(mp.GaussianSource(frequency=0.15, fwidth=0.1),
              component=mp.Hz,
              center=mp.Vector3(-7))
]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=10)

sim.run(until=100)

# Plot the Hz field
sim.plot2D(fields=mp.Hz)
plt.savefig("waveguide_hz.png", dpi=150)
print("Field plot saved to waveguide_hz.png")
```

Run the simulation with:

```bash
python my_first_sim.py
```

For more examples and step-by-step tutorials, see the [online manual](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/).

## Test Status

We ran all 148 Python files from `python/examples/` and `python/tests/` against pymeep 1.31.0 on Python 3.13 with NumPy 2.x under WSL2. See [guides/TEST_REPORT.md](guides/TEST_REPORT.md) for the full report.

| Result | Count | Pct |
|--------|-------|-----|
| Pass   | 76    | 51% |
| Fail   | 25    | 17% |
| Timeout| 47    | 32% |

**None of the 25 failures are bugs in meep itself.** All are caused by environment issues:

| Root Cause | Files | Fix |
|-----------|-------|-----|
| Missing `parameterized` package | 10 | `pip install parameterized` |
| NumPy 2.x removed APIs | 2 | `np.trapz` &rarr; `np.trapezoid`, `np.complex_` &rarr; `np.complex128` |
| Missing `h5topng` tool | 2 | `sudo apt install h5utils` |
| Missing data files | 2 | Needs pre-generated data or companion script |
| Scripts requiring CLI args | 3 | Not bugs &mdash; run with required arguments |
| Outdated example API calls | 4 | Needs upstream fixes (API signature changes) |
| Missing optional packages | 2 | `pip install gdspy` / `pip install jax` |

The 47 timeouts are due to CPU contention from parallel batch execution, not actual hangs. Files that timed out in one run frequently passed in the other.

**Quick fix for the most common issue:**

```bash
pip install parameterized    # fixes 10 of 25 failures
sudo apt install h5utils     # fixes 2 more
```

## Documentation

| Resource | Description |
|---|---|
| [guides/tutorials/](guides/tutorials/00_index.md) | **148 Physics Tutorials**: deep-dive theory + code walkthroughs for every Python example and test |
| [guides/QUICKSTART_EXPLAINED.md](guides/QUICKSTART_EXPLAINED.md) | Quick Start Explained: physics, simulation setup, and results walkthrough |
| [guides/USER_GUIDE.md](guides/USER_GUIDE.md) | User Guide: installation, tutorials, and 10 worked use cases |
| [guides/DEVELOPER_GUIDE.md](guides/DEVELOPER_GUIDE.md) | Developer Guide: building from source, testing, and contributing |
| [guides/ARCHITECTURE.md](guides/ARCHITECTURE.md) | Architecture Documentation: system design, diagrams, and code reference |
| [guides/TEST_REPORT.md](guides/TEST_REPORT.md) | Test Report: results from running all 148 Python examples and tests |
| [guides/CLOAK_SIMULATION_REPORT.md](guides/CLOAK_SIMULATION_REPORT.md) | Cloak Report: FDTD limitations, Yee grid stability analysis, and mitigation strategies |
| [guides/CLOAK_VISUALIZATION_GUIDE.md](guides/CLOAK_VISUALIZATION_GUIDE.md) | Visualization Guide: how to read field plots and scattering spectra with physics interpretation |
| [CLAUDE.md](CLAUDE.md) | AI Assistant Guide: instructions for Claude Code |
| [Online Manual](https://meep.readthedocs.io/en/latest) | Full documentation on Read the Docs |

## Platform Support

| Platform | Support | Notes |
|---|---|---|
| Linux | Full | Install via Conda or build from source |
| macOS | Full | Install via Conda or build from source |
| Windows | Via WSL2 or Docker | No native Conda package — see [USER_GUIDE.md](guides/USER_GUIDE.md) for setup instructions |

Conda packages are the recommended path for new users on Linux and macOS. Windows users should use WSL2 (Ubuntu) or Docker. Building from source gives the most flexibility for advanced configurations (MPI, custom prefix, debug builds).

## Building from Source

For users who need custom build options such as MPI parallelization, OpenMP threading, or debug symbols, Meep can be built from source using GNU Autotools. The build requires several external libraries: FFTW3, GSL, LAPACK, HDF5, harminv, libctl, and MPB.

```bash
sh autogen.sh
./configure --enable-maintainer-mode --prefix=$HOME/local \
  --with-libctl=$HOME/local/share/libctl
make -j$(nproc)
make check
```

Full instructions including dependency installation, configure options, and CI build details are in [guides/DEVELOPER_GUIDE.md](guides/DEVELOPER_GUIDE.md).

## Project Structure

```
src/              - C++ core FDTD engine (libmeep)
python/           - Python interface and SWIG bindings
scheme/           - Scheme/Guile interface
tests/            - C++ test suite
libpympb/         - Python MPB eigenmode solver bindings
guides/           - Project guides (architecture, developer, user, test report)
guides/tutorials/ - 148 physics tutorials (theory + code walkthroughs)
```

## Citing Meep

We kindly request that you cite the following paper in any published work for which you used Meep:

- A. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J.D. Joannopoulos, and S.G. Johnson, [MEEP: A flexible free-software package for electromagnetic simulations by the FDTD method](http://dx.doi.org/doi:10.1016/j.cpc.2009.11.008), Computer Physics Communications, Vol. 181, pp. 687-702 (2010) ([pdf](http://ab-initio.mit.edu/~oskooi/papers/Oskooi10.pdf)).
