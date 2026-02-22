# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meep (MIT Electromagnetic Equation Propagation) is an open-source FDTD (finite-difference time-domain) electromagnetics simulation engine. It provides C++ core library with Python (via SWIG) and Scheme (via libctl/Guile) interfaces. The project lives at https://github.com/NanoComp/meep.

## Build System

Meep uses GNU Autotools. The build requires several external libraries: FFTW3, GSL, LAPACK, HDF5, harminv, libctl, MPB, and optionally MPI for parallelism.

### Build Commands

```bash
# Generate configure script (requires autoconf, automake, libtool, swig)
sh autogen.sh

# Configure (typical development build)
./configure --enable-maintainer-mode --prefix=$HOME/local \
  --with-libctl=$HOME/local/share/libctl

# Common configure variants
./configure ... --with-mpi          # Enable MPI parallelization
./configure ... --enable-single     # Single-precision floats
./configure ... --with-openmp       # OpenMP threading
./configure ... --enable-debug      # Debug symbols
./configure ... --with-coverage     # Python coverage

# Build
make -j$(nproc)

# Out-of-tree build (used in CI)
mkdir build && cd build && ../configure ... && make
```

### Running Tests

```bash
# All C++ tests
make check                    # from root or tests/

# All Python tests
cd python && make check       # runs unittest-based tests

# Single Python test
python python/tests/test_simulation.py
# Or with MPI:
mpirun -np 2 python python/tests/test_ring.py

# Distcheck (full build+test cycle, used in CI)
make distcheck DISTCHECK_CONFIGURE_FLAGS="--with-libctl=..."
```

C++ tests are standalone executables in `tests/` linked against `libmeep.la`. Python tests use `unittest` and live in `python/tests/test_*.py`. Some Python tests require MPB (`test_mode_coeffs.py`, `test_mode_decomposition.py`, etc.) and are skipped without it. `test_material_dispersion.py` is excluded from MPI builds.

## Architecture

### C++ Core (`src/`)

The core builds `libmeep.la` (shared library). Key components:

- **`meep.hpp`** — Main public API header (~2500 lines). Defines all core classes: `fields`, `structure`, `susceptibility` hierarchy, DFT monitors, sources, etc.
- **`meep/vec.hpp`** — Vector math, grid coordinates, `component` enum (Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz, Bx, By, Bz)
- **`meep_internals.hpp`** — Internal implementation details
- **`meepgeom.hpp` / `material_data.hpp`** — Geometry processing and material data structures

**FDTD time-stepping loop:** `step.cpp` → `step_generic.cpp` (field updates) → `step_db.cpp` (dispersive materials) → `update_eh.cpp` / `update_pols.cpp`

**Domain decomposition:** `structure` splits into `structure_chunk` per MPI rank. `binary_partition` handles spatial partitioning. Communication via `mympi.cpp`.

**Key subsystems by file:**
- `fields.cpp` / `fields_dump.cpp` — Field storage, checkpoint/restart
- `boundaries.cpp` — PML, periodic, Bloch boundary conditions
- `dft.cpp` — All DFT monitors (flux, energy, force, near2far, fields)
- `sources.cpp` — Source injection into fields
- `susceptibility.cpp` — Lorentzian, Drude, gyrotropic, nonlinear materials
- `meepgeom.cpp` — Geometry → epsilon grid conversion, subpixel smoothing
- `cw_fields.cpp` — Continuous-wave frequency-domain solver
- `array_slice.cpp` — Field interpolation and slicing
- `near2far.cpp` — Near-to-far-field transformations

### Python Interface (`python/`)

SWIG bridges C++ to Python. The binding chain:

1. `meep.i` (SWIG interface, ~65K lines) + `numpy.i`, `vec.i` → generates `meep-python.cxx`
2. `meep-python.hpp` — C++ helper functions for the Python layer
3. Compiled into `_meep.so` module

**High-level Python modules** (the user-facing API):
- `simulation.py` — `Simulation` class, main orchestrator for FDTD runs
- `geom.py` — `Vector3`, `Medium`, geometric objects (`Block`, `Cylinder`, `Prism`, etc.)
- `source.py` — Source definitions (`EigenModeSource`, `GaussianSource`, etc.)
- `visualization.py` — matplotlib-based plotting
- `materials.py` — Predefined material library (broadband refractive indices)
- `adjoint/` — Adjoint solver for inverse design/topology optimization (JAX integration via `wrapper.py`)

### MPB Integration (`libpympb/`)

Python bindings for the MIT Photonic Bands eigenmode solver. `pympb.hpp`/`pympb.cpp` define the `mode_solver` class wrapping MPB's eigenmode calculations. Used by `python/solver.py` and for eigenmode sources in meep.

### Scheme Interface (`scheme/`)

Alternative scripting interface via Guile/libctl. Built only with `--with-libctl --with-scheme`.

## Code Style

- **C++:** LLVM-based clang-format (see `.clang-format`). 2-space indent, 100-char line limit, `Cpp03` standard mode, no include sorting.
- **Python:** Black formatter, flake8 linting (max line 88), pyupgrade for Python 3.7+ syntax.
- **Pre-commit hooks:** clang-format for C++, Black + pyupgrade + bandit for Python. Run `pre-commit run -a` to check all files.

## CI

GitHub Actions (`.github/workflows/build-ci.yml`): Tests Python 3.9 and 3.11, with and without MPI. Builds dependencies (libctl, harminv, MPB, libGDSII) from source. Runs both `make check` (C++ tests) and Python test suite.

## Key Conventions

- SWIG regeneration only happens in `--enable-maintainer-mode`. Release tarballs ship pre-generated `.cxx` files.
- `RUNCODE` variable in Makefiles prepends `mpirun -np 2` for MPI builds and `env OMP_NUM_THREADS=2` for OpenMP.
- Shared library versioning tracked via `SHARED_VERSION_INFO` in `configure.ac` (currently `36:0:1`). Any `.hpp` change generally breaks binary compatibility.
- The `step_generic_stride1.cpp` file is auto-generated from `step_generic.cpp` via sed transforms — do not edit directly.

## Known Issues (pymeep 1.31.0 + Python 3.13 + NumPy 2.x)

Full details in `guides/TEST_REPORT.md`. Summary of issues to be aware of when running examples/tests:

- **10 tests need `parameterized`**: `pip install parameterized` (not included in conda-forge pymeep). Affects: `test_adjoint_cyl`, `test_adjoint_utils`, `test_binary_grating`, `test_binary_partition_utils`, `test_chunk_balancer`, `test_get_epsilon_grid`, `test_mode_decomposition`, `test_pml_cyl`, `test_refl_angular`, `test_special_kz`.
- **NumPy 2.x breaking changes**: `np.trapz` removed (use `np.trapezoid`), `np.complex_` removed (use `np.complex128`). Affects `antenna-radiation.py` and `solve-cw.py`.
- **`h5topng` not installed by default**: `sudo apt install h5utils`. Affects `cherenkov-radiation.py` and `wvg-src.py`.
- **Outdated API calls in examples**: `cavity_arrayslice.py` (get_array signature changed), `mpb_line_defect.py` (`fix_efield_phase` → `fix_field_phase`), `waveguide_crossing.py` and `binary_grating_levelset.py` (EigenModeSource amplitude type mismatch).
- **Optional packages**: `ring_gds.py` needs `gdspy`, `test_adjoint_jax.py` needs `jax`.
- **CLI-argument scripts**: `dipole_in_vacuum_1D.py`, `dipole_in_vacuum_cyl_off_axis.py`, `dipole_in_vacuum_cyl_on_axis.py` require command-line arguments.

## Documentation

- `guides/ARCHITECTURE.md` — System architecture with ASCII diagrams, data flows, and code references
- `guides/DEVELOPER_GUIDE.md` — Step-by-step build, test, and contribution guide for new developers
- `guides/USER_GUIDE.md` — Installation (including Windows), tutorials, and 10 educational use cases
- `guides/QUICKSTART_EXPLAINED.md` — Detailed walkthrough of the Quick Start example
- `guides/TEST_REPORT.md` — Results from running all 148 Python examples and tests
- [Online Manual](https://meep.readthedocs.io/en/latest) — Full reference documentation
