# Meep Developer Guide

A comprehensive guide for developers who are new to the Meep FDTD electromagnetic simulation project. This guide assumes you have experience with C/C++ or Java, but are new to GNU Autotools, SWIG, FDTD simulation, MPI, and scientific computing.

---

## Table of Contents

1. [Prerequisites and Environment Setup](#1-prerequisites-and-environment-setup)
2. [Building Dependencies from Source](#2-building-dependencies-from-source)
3. [Building Meep](#3-building-meep)
4. [Running Tests](#4-running-tests)
5. [Understanding the Codebase](#5-understanding-the-codebase)
6. [Making Changes](#6-making-changes)
7. [Code Style and Formatting](#7-code-style-and-formatting)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Key Conventions and Gotchas](#9-key-conventions-and-gotchas)
10. [Debugging Tips](#10-debugging-tips)

---

## 1. Prerequisites and Environment Setup

### What is Meep?

Meep (MIT Electromagnetic Equation Propagation) is a C++ library that simulates how electromagnetic waves travel through materials. Think of it like a physics simulator: you describe a geometry (e.g., a silicon waveguide), place electromagnetic sources, run time forward in discrete steps, and observe how fields evolve. It also has Python and Scheme scripting interfaces.

The core algorithm is called FDTD (Finite-Difference Time-Domain). This is explained further in Section 5.

### Why So Many Dependencies?

Meep relies on several specialized scientific libraries, each serving a specific purpose. Unlike a typical web application, scientific computing software tends to be built in layers of specialized libraries rather than all-in-one frameworks. You need to install all of them before you can build Meep.

### System Package Dependencies

#### Ubuntu / Debian

Run the following command to install all required system packages:

```bash
sudo apt-get update
sudo apt-get install -y \
    autoconf \
    automake \
    libtool \
    swig \
    gcc \
    g++ \
    gfortran \
    libfftw3-dev \
    libgsl-dev \
    liblapack-dev \
    libhdf5-dev \
    guile-3.0-dev \
    libpng-dev \
    pkg-config \
    git \
    make
```

For MPI support (optional but recommended for parallel simulations):

```bash
sudo apt-get install -y \
    libopenmpi-dev \
    mpi-default-bin \
    openmpi-bin \
    libhdf5-openmpi-dev
```

If you want MPI support, replace `libhdf5-dev` in the first command with `libhdf5-openmpi-dev` from the MPI command, as you need the MPI-aware version of HDF5.

#### macOS (Homebrew)

First, install [Homebrew](https://brew.sh) if you have not already:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install the required packages:

```bash
brew install \
    autoconf \
    automake \
    libtool \
    swig \
    gcc \
    gfortran \
    fftw \
    gsl \
    lapack \
    hdf5 \
    guile \
    libpng \
    pkg-config \
    git
```

For MPI support on macOS:

```bash
brew install open-mpi
```

Note: On macOS, Homebrew installs packages into `/opt/homebrew` (Apple Silicon) or `/usr/local` (Intel). You may need to add Homebrew's bin directory to your PATH:

```bash
# For Apple Silicon Macs:
export PATH="/opt/homebrew/bin:$PATH"

# For Intel Macs:
export PATH="/usr/local/bin:$PATH"
```

### What Each Dependency Does

**autoconf** - Generates the `configure` script from `configure.ac`. Think of it as a meta-build tool: you write a template describing your build requirements, and autoconf generates a portable shell script that detects your system's capabilities. Without this, you cannot regenerate the `configure` script after editing `configure.ac`.

**automake** - Generates `Makefile` files from `Makefile.am` template files. Works alongside autoconf. You need this to regenerate Makefiles after editing any `Makefile.am`.

**libtool** - A portable interface for building shared libraries (`.so` on Linux, `.dylib` on macOS). Meep compiles as a shared library (`libmeep.la`) so Python can load it dynamically. Libtool hides the platform differences in how shared libraries are built.

**swig** - Simplified Wrapper and Interface Generator. This tool reads C++ header files and `.i` interface files, then automatically generates Python (and other language) wrapper code. Without SWIG, you would have to manually write all the Python-to-C++ bridge code by hand - tens of thousands of lines.

**gcc / g++** - The GNU C and C++ compilers. Meep's core is written in C++ and must be compiled. If you prefer Clang (on macOS this is common), Meep supports it too.

**gfortran** - The GNU Fortran compiler. Meep does not contain Fortran code itself, but it links against LAPACK and BLAS, which are traditionally written in Fortran. The Fortran compiler is needed to determine how the Fortran libraries export their symbols so the C++ linker can find them.

**libfftw3-dev** - FFTW3 (Fastest Fourier Transform in the West) is a highly optimized library for computing Discrete Fourier Transforms. Meep uses it for frequency-domain analysis: converting time-domain field data into frequency-domain spectra. This is how Meep computes transmission spectra, reflection spectra, etc.

**libgsl-dev** - GNU Scientific Library. Provides mathematical routines including special functions, numerical integration, interpolation, and random number generation. Meep uses GSL for various numerical algorithms in the simulation engine.

**liblapack-dev** - Linear Algebra PACKage. Provides routines for solving systems of linear equations, computing eigenvalues, and matrix factorizations. Meep uses it indirectly through Harminv (see below) for resonant frequency extraction.

**libhdf5-dev** - Hierarchical Data Format 5. HDF5 is a file format designed for storing large, multi-dimensional scientific datasets. Meep outputs field data, epsilon distributions, and other volumetric data as HDF5 files (`.h5`). These files can then be visualized with tools like h5utils or read into Python with h5py.

**guile-3.0-dev** - Guile is an implementation of the Scheme programming language. Meep has a Scheme scripting interface built on top of Guile. Even if you only plan to use Python, Guile is needed to build libctl (see Section 2), which is required for the Python interface.

**libpng-dev** - PNG image format library. Used for producing image output from field visualizations.

**openmpi / libopenmpi-dev** - Open MPI is an implementation of the Message Passing Interface (MPI) standard. MPI enables distributed-memory parallelism: running a single simulation across many processors on a cluster, each handling a different spatial region of the simulation. This is optional for development but important for production simulations. With MPI, a simulation that takes 8 hours on one core might take 20 minutes on 24 cores.

### Python Requirements

Meep requires Python 3.7 or later. Install the required Python packages:

```bash
pip3 install \
    numpy \
    scipy \
    matplotlib \
    h5py \
    autograd \
    jax \
    jaxlib \
    parameterized \
    pytest
```

For MPI support, also install:

```bash
pip3 install mpi4py
```

For code coverage reporting:

```bash
pip3 install coverage
```

What each Python package does:

**numpy** - Numerical Python. Provides multi-dimensional arrays and mathematical operations on them. Meep's Python interface returns field data as NumPy arrays, so nearly every simulation script uses NumPy.

**scipy** - Scientific Python. Provides additional scientific algorithms built on NumPy (signal processing, optimization, statistics, etc.). Used in some simulation analysis scripts and tests.

**matplotlib** - The standard Python plotting library. Meep's `visualization.py` module uses matplotlib to plot field distributions, spectra, and geometry.

**h5py** - Python interface to HDF5. Lets Python code read and write `.h5` files directly. Used to read Meep's output files for post-processing and visualization.

**autograd / jax / jaxlib** - Automatic differentiation libraries. Required for Meep's adjoint solver (`python/adjoint/`), which computes gradients of electromagnetic objectives for inverse design and topology optimization. JAX is a newer, faster alternative to autograd; both are supported.

**mpi4py** - MPI for Python. Required for running parallel Meep simulations from Python. It wraps the MPI C library and provides Python bindings for sending and receiving data between MPI processes.

**parameterized** - A Python library for parameterizing unit tests. Used in Meep's Python test suite.

**pytest** - Python test runner. Used to discover and run the test suite (in addition to the built-in `unittest` framework which some tests use directly).

---

## 2. Building Dependencies from Source

Several dependencies are not available in standard package managers and must be built from source: `libctl`, `harminv`, `MPB`, and `libGDSII`. These are all NanoComp/MIT projects that work together.

### The Install Prefix Convention

When building scientific software without root access, the standard practice is to install everything under a personal directory, conventionally `$HOME/local`. This means:

- Libraries go to `$HOME/local/lib`
- Headers go to `$HOME/local/include`
- Executables go to `$HOME/local/bin`
- Data files go to `$HOME/local/share`

You tell each `configure` script about this location with `--prefix=$HOME/local`.

Create this directory first:

```bash
mkdir -p $HOME/local
```

### Setting Up Environment Variables

Before building anything, set these environment variables. Add them to your `~/.bashrc` or `~/.zshrc` to make them permanent:

```bash
# Where our custom-built libraries are installed
export PREFIX=$HOME/local

# Tell the compiler where to find header files
export CPPFLAGS="-I$PREFIX/include"

# Tell the linker where to find library files
export LDFLAGS="-L$PREFIX/lib"

# Tell the runtime linker where to find shared libraries when programs run
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"

# On macOS, use DYLD_LIBRARY_PATH instead of LD_LIBRARY_PATH:
# export DYLD_LIBRARY_PATH="$PREFIX/lib:$DYLD_LIBRARY_PATH"

# Tell pkg-config where to find .pc metadata files for our libraries
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# Add our installed executables to PATH
export PATH="$PREFIX/bin:$PATH"
```

**Why LD_LIBRARY_PATH?** When a compiled program runs, the operating system's dynamic linker must find the shared libraries the program needs. By default it only looks in standard system paths like `/usr/lib`. If you install a library into `$HOME/local/lib`, you must tell the linker to look there too. `LD_LIBRARY_PATH` is a colon-separated list of additional directories to search.

**Why PKG_CONFIG_PATH?** `pkg-config` is a tool that tells the compiler where libraries are installed. Build systems call it like `pkg-config --cflags fftw3` to get the right compiler flags automatically. Custom-installed libraries include `.pc` files in their `lib/pkgconfig/` directory, and `PKG_CONFIG_PATH` tells `pkg-config` where to look.

**Note on macOS System Integrity Protection:** On macOS, `DYLD_LIBRARY_PATH` is often ignored for security reasons. A more reliable approach is to use `-Wl,-rpath,$PREFIX/lib` in your `LDFLAGS` to bake the library paths directly into the compiled binaries:

```bash
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"
```

### 2.1 Building libctl

**What is libctl?** libctl is a library that provides a Scheme-based control language for scientific simulations. It handles the communication between Guile (the Scheme interpreter) and C++ programs, converting complex data structures automatically. Even if you only use the Python interface, libctl is required because Meep's Python interface is built on top of it.

```bash
cd $HOME
mkdir -p src && cd src

# Clone the repository
git clone https://github.com/NanoComp/libctl.git
cd libctl

# Run autogen.sh to generate the configure script from configure.ac
# (This is needed because we cloned from git, not a release tarball)
sh autogen.sh --prefix=$HOME/local --enable-shared

# Build using all available processor cores
make -j$(nproc)

# Install to $HOME/local
make install
```

The `--enable-shared` flag tells the build system to build a shared library (`.so` file) in addition to or instead of a static library (`.a` file). Shared libraries are required because Python extension modules load libraries at runtime.

Verify the installation:

```bash
ls $HOME/local/share/libctl
# Should show files like base.ctl, geom.ctl, etc.
```

### 2.2 Building harminv

**What is harminv?** Harminv (Harmonic Inversion) is a library for extracting resonant frequencies, decay rates, and amplitudes from a time-domain signal. In FDTD simulations, you often want to find the resonant modes of a cavity. You run the simulation, record the fields at a point over time, then feed that time series to harminv to extract the resonant frequencies. It uses a sophisticated algorithm (filter diagonalization method) that is much more efficient than a simple Fourier transform for this purpose.

```bash
cd $HOME/src

git clone https://github.com/NanoComp/harminv.git
cd harminv

sh autogen.sh --prefix=$HOME/local --enable-shared

make -j$(nproc)
make install
```

Harminv depends on BLAS and LAPACK (which you installed as system packages). If configure cannot find them, you may need to be explicit:

```bash
sh autogen.sh --prefix=$HOME/local --enable-shared \
    --with-blas=openblas  # or whatever your BLAS library is named
```

### 2.3 Building MPB

**What is MPB?** MPB (MIT Photonic Bands) is a photonic band structure computation package. It solves for the electromagnetic eigenmodes (standing wave solutions) of a periodic structure. In Meep, MPB is used for:

1. **Eigenmode sources**: Instead of injecting a broadband pulse and waiting for modes to form, you can use MPB to pre-compute the mode profile and inject it directly. This gives cleaner excitation.
2. **Mode decomposition**: After running a simulation, decompose the transmitted/reflected fields into individual waveguide modes to calculate mode-specific transmission/reflection coefficients.

MPB is only required if you use these features; simpler simulations without eigenmode sources work without it.

**Important:** Meep can only link to the *serial* (non-MPI) version of MPB. Do not build MPB with MPI.

```bash
cd $HOME/src

git clone https://github.com/NanoComp/mpb.git
cd mpb

# --with-hermitian-eps: required for handling complex-valued permittivities
#   (metals, gain media) correctly in eigenmode calculations
# --with-libctl: tells MPB where to find the libctl data files we installed
sh autogen.sh \
    --prefix=$HOME/local \
    --enable-shared \
    --with-libctl=$HOME/local/share/libctl \
    --with-hermitian-eps \
    LIBS=-ldl

make -j$(nproc)
make install
```

The `LIBS=-ldl` flag explicitly links against `libdl` (the dynamic linking library), which is needed on some Linux systems to load shared libraries at runtime.

Verify MPB installed correctly:

```bash
$HOME/local/bin/mpb --version
```

### 2.4 Building libGDSII

**What is libGDSII?** GDSII is a binary file format used by electronic design automation (EDA) tools like Cadence Virtuoso to describe chip layouts. It is a standard format used in semiconductor foundries for 2D/planar photonic integrated circuit (PIC) designs. libGDSII lets Meep import these industry-standard layout files directly, so you can simulate geometries designed in chip design tools without manually re-entering all the geometry.

```bash
cd $HOME/src

git clone https://github.com/HomerReid/libGDSII.git
cd libGDSII

sh autogen.sh --prefix=$HOME/local

make -j$(nproc)
make install
```

### Dependency Build Order

Always build in this order, as each depends on the previous:

1. libctl (no local dependencies)
2. harminv (no local dependencies)
3. MPB (depends on libctl and harminv)
4. libGDSII (no local dependencies)
5. Meep (depends on all of the above)

---

## 3. Building Meep

### Getting the Source

```bash
cd $HOME/src
git clone https://github.com/NanoComp/meep.git
cd meep
```

### Understanding autogen.sh

When you clone from git, there is no `configure` script yet. The `configure` script is *generated* by autoconf from `configure.ac`. The `autogen.sh` script runs the full Autotools chain to produce `configure` and the `Makefile.in` templates:

```
autogen.sh runs:
  aclocal      -> generates aclocal.m4 (collects m4 macro files)
  autoheader   -> generates config.h.in (template for config.h)
  automake     -> generates Makefile.in files from Makefile.am files
  autoconf     -> generates configure script from configure.ac
```

You can also run this chain manually using `autoreconf -vif` (verbose, install missing files, force regeneration), which is what the CI system does.

### Step-by-Step Build

**Step 1: Generate the build system**

```bash
# From inside the meep/ directory
sh autogen.sh
```

Or equivalently:

```bash
autoreconf --verbose --install --symlink --force
```

If this fails, check that autoconf, automake, libtool, and swig are all installed.

**Step 2: Configure**

The `configure` script detects your system's capabilities, finds installed libraries, and generates the final `Makefile` files.

For a typical development build:

```bash
./configure \
    --enable-maintainer-mode \
    --prefix=$HOME/local \
    --with-libctl=$HOME/local/share/libctl
```

For an MPI-enabled build:

```bash
./configure \
    --enable-maintainer-mode \
    --prefix=$HOME/local \
    --with-libctl=$HOME/local/share/libctl \
    --with-mpi
```

**Step 3: Build**

```bash
make -j$(nproc)
```

The `-j$(nproc)` flag parallelizes the build across all available CPU cores. `nproc` prints the number of processors.

**Step 4: Install (optional)**

```bash
make install
```

This copies the compiled libraries, executables, and Python modules to `$HOME/local`. If you used a system prefix like `/usr/local`, you would need `sudo make install`.

### In-Tree vs Out-of-Tree Builds

**In-tree build** (source and build in the same directory, as above): Simple but the build artifacts mix with source files. Running `make clean` removes the build artifacts.

**Out-of-tree build** (recommended, and used by CI): The build artifacts go in a separate directory, keeping the source tree pristine. This also allows building multiple configurations simultaneously (e.g., serial and MPI) from the same source.

```bash
# Create a separate build directory
mkdir -p $HOME/src/meep-build
cd $HOME/src/meep-build

# Run configure from the build directory, pointing to the source
$HOME/src/meep/configure \
    --enable-maintainer-mode \
    --prefix=$HOME/local \
    --with-libctl=$HOME/local/share/libctl

make -j$(nproc)
```

### What --enable-maintainer-mode Does

This is critical for developers. Without it:

- Autotools will NOT regenerate `Makefile` files when you edit `Makefile.am`
- SWIG will NOT regenerate the Python binding code when you edit `meep.i`
- The pre-generated `.cxx` files shipped in the source tree will be used as-is

With `--enable-maintainer-mode`:

- Editing `Makefile.am` and then running `make` will automatically regenerate the Makefile
- Editing `meep.i` and then running `make` will automatically regenerate `meep-python.cxx`
- This requires that autoconf, automake, libtool, and swig are installed

Release tarballs (downloaded from the GitHub releases page, not cloned from git) ship with pre-generated files. When building from a release tarball, you do NOT need `--enable-maintainer-mode` and you do NOT need SWIG, autoconf, or automake - just a compiler and the library dependencies.

### All configure Flags Explained

| Flag | Effect |
|------|--------|
| `--enable-maintainer-mode` | Enables automatic regeneration of generated files when sources change. Required for development. |
| `--prefix=DIR` | Install to DIR instead of the default `/usr/local`. Use `$HOME/local` for non-root installs. |
| `--with-libctl=DIR` | Path to the libctl shared data files (`DIR/base.ctl` etc.). Must be `PREFIX/share/libctl` where PREFIX is libctl's install prefix. |
| `--with-mpi` | Enable MPI parallelization. Requires MPI to be installed. |
| `--with-openmp` | Enable OpenMP multi-threading. Can be combined with MPI. |
| `--enable-single` | Use single-precision (32-bit) floating point instead of double-precision (64-bit). Reduces memory by half and can double performance on memory-bandwidth-limited problems, with little loss in accuracy. |
| `--enable-debug` | Compile with debug symbols (`-g`) and extra runtime checks. Disables optimization. Makes the executable much easier to debug with GDB. |
| `--with-coverage` | Enable Python code coverage instrumentation. |
| `--without-python` | Do not build the Python interface. Useful for CI or headless builds. |
| `--without-scheme` | Do not build the Scheme/Guile interface. |
| `--without-hdf5` | Build without HDF5 support. You cannot output field data without HDF5. |
| `--enable-shared` | Build shared libraries. Usually on by default; required for Python bindings. |
| `--enable-swig-python-threads` | Build Python bindings with GIL released during C++ calls (experimental). |

### Troubleshooting Common Build Errors

**"configure: error: could not find libctl"**

The `--with-libctl` path is wrong. The path should point to the `share/libctl` subdirectory of your libctl install prefix, not the prefix itself:

```bash
# Wrong:
--with-libctl=$HOME/local

# Correct:
--with-libctl=$HOME/local/share/libctl
```

**"configure: error: cannot find -lfftw3"**

FFTW3 is not installed, or it is installed in a non-standard location. If you installed it to `$HOME/local`:

```bash
./configure ... CPPFLAGS="-I$HOME/local/include" LDFLAGS="-L$HOME/local/lib"
```

**"configure: error: SWIG not found" or SWIG version error**

Install SWIG: `sudo apt-get install swig` (Ubuntu) or `brew install swig` (macOS). The version must be recent enough (3.0+). Check with `swig -version`.

**"make: *** [meep-python.cxx] Error 1"**

SWIG failed to process `meep.i`. This is often caused by an incompatible SWIG version or missing `numpy.i`. Check the error output from SWIG carefully - it will tell you which line failed and why.

**Linker errors: "undefined reference to ..."**

Usually means a library was not found at link time. Check that `LDFLAGS` includes the right paths. For example, if harminv is in `$HOME/local`:

```bash
./configure ... LDFLAGS="-L$HOME/local/lib"
```

**Python module not found after install**

If `import meep` fails after `make install`, Python cannot find the installed module. Check:

```bash
# Find where meep installed its Python files
find $HOME/local -name "_meep*.so" 2>/dev/null

# Add that directory to PYTHONPATH (adjust the path to match)
export PYTHONPATH=$HOME/local/lib/python3.X/site-packages:$PYTHONPATH
```

Or use the build tree directly without installing:

```bash
export PYTHONPATH=$HOME/src/meep-build/python:$PYTHONPATH
```

---

## 4. Running Tests

Meep has two test suites: a C++ test suite in `tests/` and a Python test suite in `python/tests/`.

### 4.1 Running All C++ Tests

The C++ tests are standalone executables linked against `libmeep.la`. They test low-level physics directly without going through Python.

```bash
# From the build directory root:
make check

# Or equivalently, from the tests/ subdirectory:
cd tests && make check
```

This compiles all test executables and runs them. Output appears in `tests/test-suite.log`. If a test fails, the log will contain the failure details.

To run just the C++ tests with verbose output:

```bash
make check VERBOSE=1
```

To run a single C++ test executable directly:

```bash
# After building, find the executable in the build directory
./tests/known_results

# Or with verbose output:
./tests/known_results --verbose
```

### 4.2 Running All Python Tests

```bash
# From the python/ subdirectory of your build directory:
cd python && make check
```

This runs the full Python test suite using the `unittest` framework. Results are written to `python/test-suite.log`.

The Makefile sets up `PYTHONPATH` automatically so the tests use the just-built Meep module (not any installed version). This is important: always run tests from the build directory, not by directly invoking Python from an arbitrary location.

### 4.3 Running a Single Python Test

During development, you usually want to run just the test for the subsystem you are working on:

```bash
# From the build directory root, set PYTHONPATH first
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

# Run a specific test file
python python/tests/test_simulation.py

# Run a specific test class
python python/tests/test_simulation.py SimulationTest

# Run a specific test method
python python/tests/test_simulation.py SimulationTest.test_convenience_methods
```

During development with an out-of-tree build, the workflow is:

```bash
# Go to the python subdirectory of the build tree
cd $HOME/src/meep-build/python

# Set PYTHONPATH to the build directory's python package
export PYTHONPATH=$(pwd)

# Edit source files in the source tree (e.g., $HOME/src/meep/python/simulation.py)
# Then rebuild the Python package to pick up changes
make

# Run the specific test you are working on
python $HOME/src/meep/python/tests/test_simulation.py
```

### 4.4 Running Tests with MPI

Some tests verify that Meep works correctly when parallelized across multiple processes. To run a test under MPI:

```bash
mpirun -np 2 python python/tests/test_ring.py
```

The `-np 2` flag tells MPI to spawn 2 processes. Meep will divide the simulation domain between them. Most test scripts work correctly with any number of processes, but 2 is the standard for CI.

For more reliable error propagation when one MPI process fails:

```bash
mpirun -np 2 python -m mpi4py python/tests/test_ring.py
```

The `-m mpi4py` flag causes mpi4py to cleanly abort all processes if any one of them raises an exception, rather than hanging.

### 4.5 Running with Coverage

Coverage reports show what fraction of the Python code is exercised by the test suite. This helps identify untested code paths.

First configure with coverage enabled:

```bash
./configure --enable-maintainer-mode --with-coverage \
    --prefix=$HOME/local --with-libctl=$HOME/local/share/libctl
make

cd python && make check
```

When `--with-coverage` is set, tests are run with `coverage run` instead of plain `python`. After tests complete, collect and display the coverage report:

```bash
coverage combine -a
coverage report -i
coverage xml -i   # generates coverage.xml for tools like Codecov
```

Note: Coverage mode always runs in serial (no MPI), because running `coverage run` under MPI is not straightforward.

### 4.6 Which Tests Require MPB

Several Python tests exercise eigenmode sources and mode decomposition, which require MPB. These tests are conditionally excluded from the test suite if MPB was not found at configure time:

Tests that require MPB (excluded without it):
- `test_mpb.py` - Direct MPB interface tests
- `test_mode_coeffs.py` - Mode coefficient calculations
- `test_mode_decomposition.py` - Mode decomposition analysis
- `test_binary_grating.py` - Binary grating diffraction using mode decomposition
- `test_diffracted_planewave.py` - Diffracted planewave calculations
- `test_dispersive_eigenmode.py` - Eigenmode sources with dispersive materials
- `test_kdom.py` - Wavenumber domain calculations
- `test_wvg_src.py` - Waveguide source tests

If you need to work on these features, make sure MPB is built and installed correctly (see Section 2.3).

### 4.7 test_material_dispersion.py and MPI Builds

The test `test_material_dispersion.py` is explicitly excluded from MPI builds. This is a known limitation: testing material dispersion with MPI requires synchronizing random number generators across processes in a way that this test does not currently handle. The exclusion is automatic: when Meep is configured with `--with-mpi`, the `MDPYTEST` variable in `python/Makefile.am` is set to empty, so this test is simply not included in the test list.

### 4.8 Known Test Failures and Environment Setup

We ran all 148 Python files (`python/examples/` and `python/tests/`) against pymeep 1.31.0 on Python 3.13 with NumPy 2.x. The full report is in `guides/TEST_REPORT.md`. Here is what you need to know as a developer:

**Install these packages before running the full test suite:**

```bash
pip install parameterized    # required by 10 tests (not in conda-forge pymeep)
sudo apt install h5utils     # provides h5topng, used by 2 examples
```

**NumPy 2.x compatibility**: Two examples use NumPy APIs removed in 2.0:
- `python/examples/antenna-radiation.py`: `np.trapz` → `np.trapezoid`
- `python/examples/solve-cw.py`: `np.complex_` → `np.complex128`

**Outdated example API calls** (need upstream patches):
- `python/examples/cavity_arrayslice.py`: `get_array()` positional argument conflicts with keyword `component`
- `python/examples/mpb_line_defect.py`: uses `fix_efield_phase` (renamed to `fix_field_phase`)
- `python/examples/waveguide_crossing.py` and `binary_grating_levelset.py`: `EigenModeSource` amplitude receives an array instead of a scalar

**Scripts that require CLI arguments** (not bugs):
- `dipole_in_vacuum_1D.py {x,y}`
- `dipole_in_vacuum_cyl_off_axis.py {x,y} dipole_pos_r`
- `dipole_in_vacuum_cyl_on_axis.py {x,z}`

**Optional packages**: `ring_gds.py` needs `gdspy`, `test_adjoint_jax.py` needs `jax`.

---

## 5. Understanding the Codebase

### 5.1 Directory Structure

```
meep/
├── src/                    # C++ core library (libmeep.la)
│   ├── meep.hpp            # Main public API header (~2500 lines) - START HERE
│   ├── meep_internals.hpp  # Internal implementation details
│   ├── meepgeom.hpp        # Geometry processing (libctl interface)
│   ├── material_data.hpp   # Material data structures
│   ├── meep/
│   │   ├── vec.hpp         # Vector math, grid coordinates, field component enum
│   │   └── mympi.hpp       # MPI abstraction layer
│   ├── step.cpp            # Main FDTD time-stepping loop
│   ├── step_generic.cpp    # Generic field update kernels
│   ├── step_db.cpp         # Time-stepping for dispersive materials
│   ├── update_eh.cpp       # E and H field updates
│   ├── update_pols.cpp     # Polarization field updates
│   ├── fields.cpp          # Field storage, initialization, chunking
│   ├── fields_dump.cpp     # Checkpoint/restart (save/load field state)
│   ├── structure.cpp       # Structure (epsilon grid) storage
│   ├── boundaries.cpp      # PML, periodic, Bloch boundary conditions
│   ├── dft.cpp             # DFT (Discrete Fourier Transform) monitors
│   ├── sources.cpp         # Source injection into the FDTD grid
│   ├── susceptibility.cpp  # Material models (Lorentzian, Drude, etc.)
│   ├── meepgeom.cpp        # Geometry-to-epsilon-grid conversion
│   ├── cw_fields.cpp       # Continuous-wave frequency-domain solver
│   ├── array_slice.cpp     # Field interpolation and extraction
│   ├── near2far.cpp        # Near-to-far field transformations
│   ├── mympi.cpp           # MPI communication implementation
│   └── vec.cpp             # Vector math implementation
│
├── python/                 # Python interface
│   ├── meep.i              # SWIG interface file (~65K lines) - defines Python bindings
│   ├── meep-python.hpp     # C++ helper functions for the Python layer
│   ├── typemap_utils.cpp   # SWIG typemap utilities (shared between meep.i and mpb.i)
│   ├── numpy.i             # NumPy array typemaps
│   ├── vec.i               # SWIG bindings for vec.hpp
│   ├── simulation.py       # Simulation class - the main Python API
│   ├── geom.py             # Vector3, Medium, Block, Cylinder, etc.
│   ├── source.py           # GaussianSource, EigenModeSource, etc.
│   ├── visualization.py    # matplotlib-based field visualization
│   ├── materials.py        # Predefined material library
│   ├── adjoint/            # Adjoint solver for inverse design
│   │   └── wrapper.py      # JAX integration for gradient computation
│   ├── tests/              # Python test suite
│   │   └── test_*.py       # Test files
│   └── examples/           # Example scripts
│
├── tests/                  # C++ test suite
│   └── *.cpp               # Test programs linked against libmeep.la
│
├── libpympb/               # Python bindings for MPB
│   ├── pympb.hpp           # MPB Python interface header
│   └── pympb.cpp           # MPB mode_solver class implementation
│
├── scheme/                 # Scheme/Guile interface (optional)
│
├── doc/                    # Documentation
│   └── docs/               # Markdown documentation files
│
├── .github/
│   └── workflows/          # GitHub Actions CI configuration
│       ├── build-ci.yml    # Main test workflow
│       ├── build-san.yml   # Sanitizer (ASAN/UBSAN) workflow
│       └── pre-commit.yml  # Code formatting checks
│
├── configure.ac            # Autoconf input: describes build requirements
├── Makefile.am             # Automake input: top-level build rules
├── autogen.sh              # Runs the full Autotools chain
├── .clang-format           # C++ code style configuration
└── .pre-commit-config.yaml # Pre-commit hook configuration
```

### 5.2 Navigating the C++ Core

Start with `src/meep.hpp`. This is the main public API header - the "front door" to the C++ library. At about 2500 lines, it is large but well-organized. Look for these key classes:

**`meep::structure`** - Represents the geometry and materials. Stores the epsilon (permittivity) and mu (permeability) values at every grid point. You create a `structure` object, then fill it with geometric objects (boxes, spheres, cylinders) using the `meepgeom` interface.

**`meep::fields`** - Represents the electromagnetic fields (Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz, Bx, By, Bz) at every grid point. The `fields` object is created from a `structure` object and then evolved in time using `step()`.

**`meep::fields_chunk`** and **`meep::structure_chunk`** - These represent one spatial sub-region of the simulation domain, assigned to a single MPI process. Both `fields` and `structure` contain arrays of these chunks. This is the key data structure for MPI parallelization.

**`meep::component`** (in `meep/vec.hpp`) - An enum listing all field components: `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`, `Dx`, `Dy`, `Dz`, `Bx`, `By`, `Bz`, `Dielectric` (for epsilon), etc. You pass these values to functions that take or return specific field components.

**`meep::grid_volume`** (in `meep/vec.hpp`) - Represents a discretized computational volume: the resolution, number of grid points in each direction, and origin. Understands the Yee lattice layout (see Section 5.4).

The FDTD time-stepping loop follows this call chain:
```
fields::step()              [step.cpp]
  -> fields_chunk::step_h() [step_generic.cpp] - update H from curl(E)
  -> fields_chunk::step_e() [step_generic.cpp] - update E from curl(H)
  -> step_db()              [step_db.cpp]       - update dispersive material polarizations
  -> update_eh()            [update_eh.cpp]     - apply constitutive relations E=D/eps, H=B/mu
  -> update_pols()          [update_pols.cpp]   - update polarization P
```

### 5.3 Navigating the Python Interface

Start with `python/simulation.py`. The `Simulation` class is the main entry point for any Python-based Meep simulation. Its `__init__` method shows all the simulation parameters; its `run()` method starts the time-stepping loop.

A minimal simulation script looks like:

```python
import meep as mp

sim = mp.Simulation(
    cell_size=mp.Vector3(10, 10),
    boundary_layers=[mp.PML(1.0)],
    sources=[mp.Source(mp.GaussianSource(0.15, fwidth=0.1),
                       component=mp.Ez,
                       center=mp.Vector3())],
    resolution=10,
)
sim.run(until=100)
```

Key Python modules and what they contain:

`simulation.py` - `Simulation` class, step functions (`output_efield_z`, `at_every`, etc.)
`geom.py` - `Vector3`, `Medium`, `Block`, `Cylinder`, `Sphere`, `Prism`, `PML`
`source.py` - `Source`, `GaussianSource`, `ContinuousSource`, `EigenModeSource`
`visualization.py` - `plot2D()`, `plot_eps()`, animation helpers

### 5.4 How SWIG Works

SWIG (Simplified Wrapper and Interface Generator) is the bridge between C++ and Python. Here is the flow:

```
Step 1: SWIG reads interface files and C++ headers
  meep.i + vec.i + numpy.i + meep.hpp + vec.hpp
                    |
                    v (SWIG runs)
Step 2: Generates two files:
  meep-python.cxx  - C++ code that wraps every class and function
  meep.py          - Python code with "proxy" classes

Step 3: The C++ wrapper is compiled
  meep-python.cxx -> _meep.so (a Python extension module)

Step 4: Python uses the package
  import meep
  # meep/__init__.py is renamed from meep.py
  # meep._meep is the compiled _meep.so
```

The `.i` file (SWIG interface file) tells SWIG:
- Which headers to process: `%include "meep.hpp"`
- How to convert between Python and C++ types: `%typemap` directives
- Extra Python code to inject: `%pythoncode` blocks
- Which functions to ignore: `%ignore` directives

When you see `meep.fields` in Python and call a method on it, Python actually:
1. Looks up the `fields` proxy class defined in `meep.py`
2. The proxy has a `this` pointer (a wrapped C++ pointer)
3. Method calls dispatch through `_meep.so` to the actual C++ methods

### 5.5 The FDTD Algorithm: A Brief Explanation

FDTD stands for Finite-Difference Time-Domain. Here is what it actually does, step by step:

**The Physical Problem**

Maxwell's equations describe how electric fields (E) and magnetic fields (H) evolve in time and space. In simplified form:

```
dH/dt = -curl(E) / mu        (H changes due to curl of E)
dE/dt = +curl(H) / epsilon   (E changes due to curl of H)
```

**Discretization: The Yee Grid**

To simulate these equations on a computer, discretize space into a 3D grid of voxels. But there is a key insight: E and H fields should be stored at *different* spatial locations within each voxel (this is the Yee lattice):

- **E fields** (Ex, Ey, Ez) are stored on the *edges* of each cubic voxel
- **H fields** (Hx, Hy, Hz) are stored on the *faces* of each cubic voxel

This staggered arrangement makes the finite-difference approximation of `curl` second-order accurate.

**Time Stepping: The "Leap-Frog" Algorithm**

Similarly, E and H are also staggered in *time*:

```
... H(t=0.5) ... E(t=1.0) ... H(t=1.5) ... E(t=2.0) ...
```

At each time step:
1. Update all H values at time `t + 0.5*dt` using E values at time `t`
2. Apply boundary conditions and inject sources into H
3. Update all E values at time `t + dt` using the just-updated H values
4. Apply boundary conditions and inject sources into E
5. Accumulate DFT (Fourier transform) values for frequency-domain monitors

This is extremely simple and efficient. The computational cost scales linearly with the number of voxels and linearly with the number of time steps.

**The Courant Condition**

The time step `dt` is not a free parameter. For stability, it must satisfy:

```
dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
```

where `c` is the speed of light and `dx, dy, dz` are the grid spacings. Meep automatically chooses `dt = 0.5 * dx` (in Meep's normalized units where c=1 by default, and assuming cubic voxels).

**PML: Absorbing Boundary Conditions**

Real simulations need to absorb outgoing waves at the simulation boundaries (otherwise they reflect back in). PML (Perfectly Matched Layer) is a special material region at the boundary that absorbs incoming waves with theoretically zero reflection. In Meep's implementation, PML adds extra "stretched coordinate" terms to Maxwell's equations that cause the waves to decay exponentially as they enter the PML region.

**DFT Monitors**

To compute frequency-domain quantities (transmission spectra, reflection spectra, near-to-far-field transforms), Meep accumulates a running Discrete Fourier Transform of the fields at each time step. At the end of the simulation, these accumulated sums give the Fourier-transformed fields at each requested frequency. This is done in `dft.cpp`.

---

## 6. Making Changes

### 6.1 Modifying C++ Code

After editing any `.cpp` or `.hpp` file in `src/`, rebuild:

```bash
# From your build directory
make -j$(nproc)
```

Autotools tracks dependencies automatically. If you only changed one `.cpp` file, only that file recompiles and the library re-links. Full rebuilds are rare.

If you edited a header file that many `.cpp` files include, all those files will recompile. This can be slow for headers like `meep.hpp` that are included nearly everywhere.

After rebuilding, Python tests pick up the new C++ code automatically because `_meep.so` is regenerated.

### 6.2 Modifying Python Code

Pure Python files (`simulation.py`, `geom.py`, `source.py`, `visualization.py`, `materials.py`) do not require recompilation. However, you still need to make sure your modified files are what Python finds when it imports `meep`.

If running from the build directory with `PYTHONPATH` set to the build's `python/` directory:

- Changes to `simulation.py`, `geom.py`, etc. in the *source tree* are NOT picked up automatically
- These files are symlinked or copied into the build directory's `python/meep/` directory by `make`
- Run `make` in the python build directory to sync the files, then run your test

Alternatively, point `PYTHONPATH` directly at the source tree's `python/` directory, but be careful that the compiled `_meep.so` from the build directory is also on the path:

```bash
# From the build directory
export PYTHONPATH=$HOME/src/meep/python:$(pwd)/python:$PYTHONPATH
```

### 6.3 Modifying SWIG Bindings

The SWIG interface file is `python/meep.i`. This is a large (~65K lines) file that defines the Python bindings for all C++ classes and functions. Editing it requires:

1. `--enable-maintainer-mode` must have been used during configure (otherwise SWIG will not be invoked)
2. SWIG must be installed

After editing `meep.i`, simply run `make` in the build directory. Autotools detects that `meep.i` changed and invokes SWIG to regenerate `meep-python.cxx`, then recompiles it into `_meep.so`.

Common things to do in `meep.i`:

**Expose a new C++ function to Python:**
The function is usually already exposed if it is declared in `meep.hpp`, because `meep.i` does `%include "meep.hpp"`. If the function uses types that SWIG cannot handle automatically, you need to add a `%typemap` or a custom wrapper.

**Add a Python method to a wrapped class:**
Use `%extend` in `meep.i`:

```swig
%extend meep::fields {
    void my_new_method(double x) {
        // C++ code here, $self is the fields* pointer
        $self->some_existing_method(x);
    }
}
```

**Inject Python-only code into the module:**
Use `%pythoncode` blocks in `meep.i`:

```swig
%pythoncode %{
def my_helper_function(sim):
    """A Python helper that has no C++ equivalent."""
    return sim.get_some_value()
%}
```

**Import a new high-level module:**
At the end of `meep.i`, there is a large `%pythoncode` block that imports from `simulation.py`, `geom.py`, etc. If you add a new `.py` module, add an import here so its contents are accessible from the `meep` namespace.

### 6.4 Adding a New C++ Source File

Say you want to add `src/my_feature.cpp`:

1. Create the file `src/my_feature.cpp`

2. Edit `src/Makefile.am` to add it to `libmeep_la_SOURCES`:

```makefile
libmeep_la_SOURCES = array_slice.cpp ... \
    my_feature.cpp \       # Add this line
    $(HDRS) $(BUILT_SOURCES)
```

3. Rebuild the build system (with `--enable-maintainer-mode`, this happens automatically when you run `make`):

```bash
make -j$(nproc)
```

If not in maintainer mode, run manually:

```bash
autoreconf -vif && ./configure [your flags] && make -j$(nproc)
```

If you are adding a new header `src/my_feature.hpp` that should be installed (i.e., be part of the public API), add it to `include_HEADERS` or `pkginclude_HEADERS` in `src/Makefile.am`.

### 6.5 Adding a New Python Test

1. Create the test file `python/tests/test_my_feature.py`:

```python
import unittest
import meep as mp


class MyFeatureTest(unittest.TestCase):
    def test_basic_behavior(self):
        # Set up a simple simulation
        sim = mp.Simulation(
            cell_size=mp.Vector3(5, 5),
            resolution=10,
        )
        # Test something
        self.assertIsNotNone(sim)

    def test_specific_value(self):
        # ... test that a specific value is what we expect
        result = some_computation()
        self.assertAlmostEqual(result, expected_value, places=3)


if __name__ == "__main__":
    unittest.main()
```

2. Add it to the `TESTS` list in `python/Makefile.am`:

```makefile
TESTS = \
    $(TEST_DIR)/test_3rd_harm_1d.py \
    ...
    $(TEST_DIR)/test_my_feature.py \   # Add this line
    ...
```

3. Rebuild (with `--enable-maintainer-mode`) and run:

```bash
make
python python/tests/test_my_feature.py
```

If your test requires MPB, use the conditional variable pattern already in `python/Makefile.am`:

```makefile
if WITH_MPB
  MY_FEATURE_TEST = $(TEST_DIR)/test_my_feature.py
else
  MY_FEATURE_TEST =
endif

TESTS = \
    ...
    $(MY_FEATURE_TEST) \
    ...
```

### 6.6 Adding a New Python Module

Say you want to add `python/meep/my_module.py`:

1. Create the file in the source tree at `python/my_module.py` (the `python/meep/` directory at install time is assembled by the build system, do not create files there directly in the source tree).

2. Add it to `python/Makefile.am`. Find the `pkgpython_PYTHON` variable:

```makefile
pkgpython_PYTHON = \
    geom.py \
    simulation.py \
    source.py \
    visualization.py \
    materials.py \
    my_module.py    # Add this line
```

Also add it to `EXTRA_DIST` if it is not already covered:

```makefile
EXTRA_DIST = ... meep.i meep-python.hpp typemap_utils.cpp \
    materials.py my_module.py examples tests
```

3. If you want `meep.my_function` to work (i.e., your module's contents accessible directly from the `meep` namespace), add an import in the `%pythoncode` block at the end of `python/meep.i`:

```swig
%pythoncode %{
from .my_module import MyClass, my_function
%}
```

4. Add the new module to the `HL_IFACE` variable in `python/Makefile.am` if you want it to trigger a rebuild of the high-level interface:

```makefile
HL_IFACE = simulation.py geom.py source.py visualization.py my_module.py
```

---

## 7. Code Style and Formatting

### 7.1 C++ Style

Meep uses LLVM-based clang-format for consistent C++ formatting. The configuration is in `.clang-format` at the repository root.

Key style rules:
- **Indent width**: 2 spaces (not tabs)
- **Line length limit**: 100 characters
- **Standard**: Cpp03 mode (the formatter does not use C++11-specific formatting rules, even though the code itself uses C++11 features)
- **Include sorting**: Disabled (includes appear in whatever order the author chose)
- **Brace style**: Opening brace on the same line as the statement (K&R style)
- **Short blocks**: Allowed on a single line (e.g., `if (x) return y;`)

To format a file manually:

```bash
clang-format -i src/my_feature.cpp
```

To check formatting without modifying (exit code 1 if not formatted):

```bash
clang-format --dry-run --Werror src/my_feature.cpp
```

### 7.2 Python Style

Python code uses:

- **Black** for code formatting (max line length 88 characters)
- **flake8** for linting (checks for style errors, unused imports, etc.)
- **pyupgrade** to automatically upgrade syntax to Python 3.7+ idioms

To format Python code:

```bash
black python/simulation.py
```

To check Python style:

```bash
flake8 python/simulation.py
```

The `.flake8` configuration at the repository root sets:
- `max-line-length = 88` (matches Black's default)
- `max-complexity = 57` (high, to accommodate existing complex functions)
- Several warnings are silenced (E501, E203, W503, etc.) to avoid conflicts with Black's formatting choices

### 7.3 Pre-commit Hooks

Pre-commit hooks run style checks automatically before every git commit, preventing improperly formatted code from entering the repository.

Install pre-commit and the repository's hooks:

```bash
pip3 install pre-commit
cd /path/to/meep
pre-commit install
```

After installation, every `git commit` automatically runs:
- clang-format on changed C++ files
- Black on changed Python files
- pyupgrade on changed Python files
- bandit (security checker) on changed Python files (excluding tests)
- YAML validity checks
- End-of-file fixers

If a hook fails, the commit is aborted. The hook may also auto-fix the file (Black does this). In that case, review the changes, `git add` them, and commit again.

To run all hooks on all files without committing (useful for checking the entire codebase or after setting up):

```bash
pre-commit run -a
```

To run just one hook:

```bash
pre-commit run black --all-files
pre-commit run clang-format --all-files
```

To temporarily skip hooks (use sparingly):

```bash
git commit --no-verify -m "WIP: work in progress"
```

### 7.4 Config Files Reference

| File | Purpose |
|------|---------|
| `.clang-format` | C++ formatter configuration (clang-format reads this automatically) |
| `.flake8` | Python linter configuration (flake8 reads this automatically) |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |

---

## 8. CI/CD Pipeline

### 8.1 GitHub Actions Workflow

Meep uses GitHub Actions for continuous integration. The workflows live in `.github/workflows/`.

**`build-ci.yml`** - The main test workflow. Runs on:
- Every pull request
- Every push to any branch
- On a daily schedule (2 AM UTC)
- On manual trigger (workflow_dispatch)

**`build-san.yml`** - Sanitizer tests. Runs daily and on manual trigger. Builds with AddressSanitizer and UBSan to detect memory errors and undefined behavior.

**`pre-commit.yml`** - Code formatting checks. Runs pre-commit hooks to verify all code is properly formatted.

### 8.2 Test Matrix

The main workflow builds a matrix of configurations:

| Python version | MPI enabled | What it tests |
|---------------|-------------|---------------|
| 3.9 | false | Serial build, Python tests with coverage, full `make check` |
| 3.11 | false | Serial out-of-tree build, `make distcheck` with OpenMP |
| 3.9 | true | MPI parallel build, `make distcheck` |
| 3.11 | true | MPI parallel build, `make distcheck` |

Each matrix entry is an independent GitHub Actions job running on `ubuntu-latest`.

### 8.3 What Each CI Job Does

Every job follows this sequence:

1. Install system packages (autoconf, automake, fftw3, gsl, lapack, guile, png, libtool, swig, and conditionally openmpi + parallel hdf5)
2. Build libctl from source (using the NanoComp/libctl GitHub repository)
3. Build harminv from source
4. Build MPB from source
5. Build libGDSII from source
6. Check out the Meep source
7. Set up the specified Python version
8. Install Python dependencies from `python/requirements.txt`
9. Install nlopt (for adjoint solver)
10. Install mpi4py (MPI jobs only)
11. Run `autoreconf` to generate the build system
12. Run `configure` with appropriate flags
13. Build and test

For the serial Python 3.9 job (the coverage job):
- Runs `make` in-tree, then `make check` in the `python/` subdirectory
- Generates a coverage report and uploads to Codecov

For all other jobs:
- Runs an out-of-tree build with `make distcheck`

### 8.4 What distcheck Does

`make distcheck` is a comprehensive quality check that verifies the release tarball is self-contained and buildable. It:

1. Creates a source tarball with `make dist` (everything needed to build from source)
2. Extracts the tarball into a fresh directory
3. Runs configure, make, and make check from the extracted directory
4. Verifies that `make install` and `make uninstall` work correctly
5. Verifies that the source tree is clean (no generated files left over)

This catches problems like:
- Missing files that were not included in the tarball
- Build rules that depend on files only in the git repository
- Tests that depend on absolute paths or environment-specific settings

To run distcheck locally:

```bash
cd your-build-directory
make distcheck DISTCHECK_CONFIGURE_FLAGS="--with-libctl=$HOME/local/share/libctl"
```

### 8.5 Reading Test Failure Logs

When a CI job fails, logs are uploaded as artifacts. Look for:

- **C++ test failures**: The artifact is named `cpp-tests-mpi-[true/false]-log` and contains `tests/test-suite.log`. Each test program reports PASS or FAIL with details.

- **Python test failures**: The artifact is named `py[version]-tests-mpi-[true/false]-log` and contains `python/test-suite.log`. This shows the output from each test script, including Python tracebacks.

To reproduce a CI failure locally, match the configure flags from the failing job. For example, for a Python 3.11 MPI failure:

```bash
cd $HOME/src/meep-build
../configure \
    --enable-maintainer-mode \
    --prefix=$HOME/local \
    --with-libctl=$HOME/local/share/libctl \
    --with-mpi \
    --with-openmp

make distcheck DISTCHECK_CONFIGURE_FLAGS="--with-libctl=$HOME/local/share/libctl --with-mpi"
```

---

## 9. Key Conventions and Gotchas

### 9.1 Do NOT Edit step_generic_stride1.cpp

`src/step_generic_stride1.cpp` is automatically generated from `src/step_generic.cpp` by a sed transformation in `src/Makefile.am`:

```makefile
step_generic_stride1.cpp: step_generic.cpp
    (echo $(PRELUDE); echo; \
     sed 's/LOOP_OVER/S1LOOP_OVER/g' $(top_srcdir)/src/step_generic.cpp \
     | sed 's/step_curl/step_curl_stride1/' \
     | sed 's/step_update_EDHB/step_update_EDHB_stride1/' \
     | sed 's/step_beta/step_beta_stride1/' \
     | sed 's/step_bfast/step_bfast_stride1/') > $@
```

This file contains a stride-1 optimized variant of the main field update kernels. "Stride-1" means accessing memory in sequential order (stride 1), which is cache-friendly and allows compiler auto-vectorization (SIMD instructions).

Any edits to `step_generic_stride1.cpp` will be overwritten the next time `make` runs. Make your changes to `step_generic.cpp` only.

### 9.2 SWIG Regeneration Requires Maintainer Mode

The generated files `python/meep-python.cxx` and `python/meep.py` are checked into the repository. This allows building Meep from a release tarball *without* having SWIG installed.

However, if you edit `python/meep.i` and try to rebuild WITHOUT `--enable-maintainer-mode`, your changes will be silently ignored and the old pre-generated `.cxx` file will be used. This is a subtle trap. Always use `--enable-maintainer-mode` for development.

When you edit `meep.i` and run `make` with maintainer mode, SWIG runs and regenerates both `meep-python.cxx` and `meep.py`. The new `meep-python.cxx` is then compiled. The generated files are not automatically `git add`'d; you must add them manually when committing changes to `meep.i`.

### 9.3 SHARED_VERSION_INFO and Binary Compatibility

In `configure.ac`:

```
SHARED_VERSION_INFO="36:0:1"  # CURRENT:REVISION:AGE
```

This controls the shared library version number embedded in `libmeep.so`. The three numbers mean:

- **CURRENT**: The current interface version. Increment this when the API changes in any way.
- **REVISION**: How many times the library has been revised since the current interface was introduced. Increment this for bug fixes that do not change the interface.
- **AGE**: How many previous interface versions this library is still compatible with. Increment when adding new (backward-compatible) interfaces.

**The critical rule:** Any change to a C++ class definition in a `.hpp` file (adding a member variable, changing the base class, adding a virtual function, etc.) generally breaks binary compatibility and requires incrementing CURRENT and resetting AGE to 0.

Why does this matter? If you install a new `libmeep.so` with a changed class layout but keep the old `_meep.so` Python extension, you get crashes and undefined behavior because the Python extension was compiled against the old class layout.

For development work, you do not need to worry about this. It matters when making public releases. The version is checked by the dynamic linker at runtime to catch mismatches.

### 9.4 The RUNCODE Variable

In the Makefiles, tests are run via:

```makefile
PY_LOG_COMPILER = $(RUNCODE) $(PYTHON)
```

The `RUNCODE` variable is set automatically by configure:

- For serial builds: empty (tests run as plain `python test_foo.py`)
- For MPI builds: `mpirun -np 2` (tests run as `mpirun -np 2 python test_foo.py`)
- For OpenMP builds: `env OMP_NUM_THREADS=2` (tests run with 2 threads)
- For MPI + OpenMP: `env OMP_NUM_THREADS=2 mpirun -np 2`

You can override RUNCODE when invoking make:

```bash
# Run tests with 4 MPI processes and 4 OpenMP threads
make check RUNCODE="env OMP_NUM_THREADS=4 mpirun -np 4"
```

### 9.5 Release Tarballs vs Git Clones

Release tarballs (downloaded from https://github.com/NanoComp/meep/releases) contain pre-generated files:
- `python/meep-python.cxx` (generated by SWIG from `meep.i`)
- `python/meep.py` (generated by SWIG)
- `configure` script (generated by autoconf from `configure.ac`)
- Various `Makefile.in` files (generated by automake from `Makefile.am`)

Building from a release tarball does NOT require:
- autoconf, automake, libtool
- SWIG

Git clones do NOT include pre-generated files and DO require those tools. Run `sh autogen.sh` to generate them.

### 9.6 Compiler Consistency

Meep and all its dependencies must be compiled with the same C++ compiler (and ideally the same version). This is especially important on supercomputers that have multiple compiler versions. Mixing compilers causes linker errors or subtle runtime crashes because different compilers may implement C++ features (especially virtual function tables, exception handling, and name mangling) differently.

If building with MPI, use the MPI compiler wrappers (`mpicc`, `mpic++`) consistently for all dependencies that need MPI awareness (HDF5, Meep itself).

---

## 10. Debugging Tips

### 10.1 Building with Debug Symbols

For debugging, configure with debug mode:

```bash
./configure \
    --enable-maintainer-mode \
    --enable-debug \
    --prefix=$HOME/local \
    --with-libctl=$HOME/local/share/libctl
make -j$(nproc)
```

`--enable-debug` does two things:
1. Adds `-g` to compiler flags, embedding debug symbols (function names, line numbers) in the compiled binary
2. Disables optimization (`-O0` instead of `-O2`), making the compiled code match the source code line-by-line (optimization moves and inlines code in ways that confuse debuggers)

The trade-off: debug builds run 5-10x slower and use more memory.

### 10.2 Using GDB with Meep

GDB (GNU Debugger) can debug C++ programs, including Meep simulations.

**Debugging a C++ test directly:**

```bash
# Build with debug symbols first
./configure --enable-debug ... && make

# Run the test under GDB
gdb ./tests/known_results
(gdb) run
# If it crashes, GDB stops at the crash point
(gdb) backtrace    # Show the call stack
(gdb) list         # Show source code around crash
(gdb) print x      # Print the value of variable x
(gdb) quit
```

**Setting breakpoints:**

```bash
gdb ./tests/known_results
(gdb) break meep::fields::step     # Break at start of step()
(gdb) break step.cpp:150           # Break at line 150 of step.cpp
(gdb) run
# Program stops at the breakpoint
(gdb) next         # Execute one line
(gdb) step         # Execute one line, entering function calls
(gdb) continue     # Continue until next breakpoint or end
```

**Catching crashes (SIGSEGV) and examining state:**

```bash
(gdb) run
# Program crashes with SIGSEGV
(gdb) backtrace full    # Full stack trace with local variables
(gdb) frame 3           # Switch to frame #3 in the backtrace
(gdb) info locals       # Print all local variables in current frame
(gdb) print my_array[0] # Print array element
```

### 10.3 Debugging Python + C++ Together

When the crash happens inside a C++ function called from Python, you need to debug both levels simultaneously.

**Method 1: Attach GDB to a Python process**

```bash
# Start Python normally
python python/tests/test_simulation.py &
PID=$!

# Attach GDB to the running Python process
gdb python $PID

# In GDB, set a C++ breakpoint
(gdb) break meep::fields::step
(gdb) continue

# Python continues running until the breakpoint is hit
```

**Method 2: Run Python inside GDB directly**

```bash
gdb python
(gdb) run python/tests/test_simulation.py
# If it crashes in C++, GDB stops at the crash
(gdb) backtrace
```

**Method 3: Use the PYTHONFAULTHANDLER**

Before running your test, set:

```bash
export PYTHONFAULTHANDLER=1
python test_my_script.py
```

On a crash (SIGSEGV, SIGFPE, etc.), Python will print a stack trace showing the Python call stack at the point of crash. This does not give C++ line numbers but shows which Python line triggered the crash.

**Checking for Python-side errors first:**

Many apparent "crashes" are actually Python exceptions that are not being handled. Before using GDB, verify the problem is actually in C++ by adding Python-level error handling:

```python
import traceback
try:
    sim.run(until=100)
except Exception:
    traceback.print_exc()
```

### 10.4 Common Segfault Causes in FDTD Codes

**Out-of-bounds array access:** The Yee grid has staggered field components. Accessing a field component at the wrong grid location (off-by-one in an index) is the most common source of crashes. Check that index arithmetic accounts for the half-grid offsets.

**MPI rank mismatch:** Code that should only run on rank 0 runs on all ranks (or vice versa). For example, printing to stdout is usually fine from all ranks, but writing to a file should typically only happen from rank 0. Use `meep::am_master()` checks.

**Uninitialized memory:** C++ does not zero-initialize arrays by default. FDTD codes allocate large arrays frequently. Accessing an uninitialized array produces garbage values that can cause unexpected behavior or NaN propagation.

**NaN propagation in fields:** If any field value becomes NaN (Not a Number), it spreads to neighboring cells on subsequent time steps because the FDTD update kernel uses local field values. By the time you notice the problem (all fields are NaN), the source of the NaN may be long gone. To catch it early, add a periodic check:

```python
# In a run() step function callback
def check_fields(sim):
    field_max = sim.max_abs_field_components([mp.Ez])
    if np.isnan(field_max) or np.isinf(field_max):
        raise RuntimeError("Fields went to NaN/Inf!")

sim.run(mp.at_every(10, check_fields), until=1000)
```

**Fourier transform of zeroed-out regions:** If a DFT monitor region overlaps a PML, the fields in the PML are artificially attenuated. The DFT accumulates these attenuated values, giving incorrect spectra. Keep DFT monitors away from PML regions.

### 10.5 Using the Sanitizer Build

The `build-san.yml` CI workflow builds Meep with compiler sanitizers. You can replicate this locally to find subtle bugs.

**AddressSanitizer (ASAN)** - Detects memory errors: buffer overflows, use-after-free, use-after-return, heap corruption.

```bash
./configure \
    --enable-maintainer-mode \
    --without-python \
    --without-scheme \
    --with-hdf5 \
    --without-mpi \
    CXXFLAGS="-fsanitize=address -g1" \
    LDFLAGS="-fsanitize=address"

make -j$(nproc)
make check
```

When a memory error occurs, ASAN prints a detailed report showing exactly where the bad access happened, including a full stack trace and information about when/where the memory was allocated and freed.

**UBSan (Undefined Behavior Sanitizer)** - Detects undefined behavior: signed integer overflow, null pointer dereference, misaligned memory access, invalid enum values.

```bash
./configure \
    --enable-maintainer-mode \
    --without-python \
    --without-scheme \
    CC=clang \
    CXX=clang++ \
    LD=clang++ \
    CXXFLAGS="-fsanitize=undefined -gmlt" \
    LDFLAGS="-fsanitize=undefined"

make -j$(nproc)
make check
```

Note: Sanitizer builds run 2-10x slower and use significantly more memory. They are not suitable for production simulations, only for debugging.

**Note:** Sanitizers currently only work for the C++ test suite (without Python), as shown in `build-san.yml`. Combining sanitizers with Python is more complex because the Python interpreter itself may trigger sanitizer warnings.

### 10.6 Verbose Meep Output

Meep supports different verbosity levels for diagnostic output:

```python
import meep as mp

# Set verbosity level (0=quiet, 1=default, 2=verbose, 3=debug)
mp.verbosity(2)

sim = mp.Simulation(...)
sim.run(until=100)
```

At verbosity 2, Meep prints detailed timing information for each step: how long was spent on field updates, boundary conditions, DFT accumulation, source injection, etc. This helps identify performance bottlenecks.

From C++, verbosity is controlled by `meep::verbosity` global variable in `adjust_verbosity.hpp`.

## See Also

- [148 Physics Tutorials](tutorials/00_index.md) — Deep-dive theory and code walkthroughs for every Python example and test
- [Architecture Guide](ARCHITECTURE.md) — System design, ASCII diagrams, data flows, and code reference
- [User Guide](USER_GUIDE.md) — Installation, tutorials, and 10 worked use cases
- [Quick Start Explained](QUICKSTART_EXPLAINED.md) — Detailed walkthrough of the Quick Start example
- [Test Report](TEST_REPORT.md) — Results from running all 148 Python examples and tests
- [Online Manual](https://meep.readthedocs.io/en/latest) — Full reference documentation

---

*This guide was written for the Meep codebase as of early 2026. For the most up-to-date information, consult the official documentation at https://meep.readthedocs.io and the source code itself, particularly `CLAUDE.md` in the repository root.*
