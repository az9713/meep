# Meep User Guide

A practical guide for using Meep to simulate electromagnetic phenomena.
This guide assumes basic Python knowledge but no prior experience with
FDTD simulation or computational electromagnetics.

---

## Table of Contents

1. [What is Meep?](#1-what-is-meep)
2. [Installation](#2-installation)
3. [Core Concepts](#3-core-concepts)
4. [Quick Start: Your First Simulation](#4-quick-start-your-first-simulation)
5. [Ten Educational Use Cases](#5-ten-educational-use-cases)
6. [Common Patterns and Recipes](#6-common-patterns-and-recipes)
7. [Troubleshooting](#7-troubleshooting)
8. [Where to Go Next](#8-where-to-go-next)

---

## 1. What is Meep?

### Plain English: What is FDTD Simulation?

Imagine you could watch light move through a material in slow motion, frame by
frame, and measure the electric and magnetic fields at every point in space at
every moment in time. That is exactly what Finite-Difference Time-Domain (FDTD)
simulation does — in software.

Maxwell's equations (the laws of electromagnetism) describe how electric fields
(E) and magnetic fields (H) evolve over time and interact with materials. FDTD
turns these equations into a simple recipe: divide space into a grid of tiny
cells, and at each time step, update each cell's fields based on its neighbors.
Repeat this thousands of times and you have simulated how light (or radio waves,
or microwaves) travels through your structure.

Meep is MIT's free, open-source implementation of FDTD. "Meep" stands for MIT
Electromagnetic Equation Propagation. It has been developed since the early
2000s and is used worldwide in academic research and industry.

### What is Electromagnetic Simulation Used For?

Electromagnetic simulation is used whenever you need to understand how light or
radio waves interact with structures whose features are comparable in size to
the wavelength. Common applications include:

- **Photonics and integrated optics**: Designing waveguides, splitters,
  modulators, and optical chips that route light on a silicon wafer
- **Nanophotonics**: Studying metal nanoparticles, plasmonic structures, and
  nanoantennas that confine light to nanometer scales
- **Photonic crystals**: Arrays of holes or pillars that create "bandgaps"
  where certain frequencies of light cannot propagate
- **Optical resonators**: Ring resonators and cavities that trap light and
  build up intensity, used in lasers and sensors
- **Antennas**: Designing and optimizing antenna radiation patterns from
  microwave to optical frequencies
- **Solar cells**: Understanding light trapping and absorption in
  photovoltaic materials
- **Metamaterials**: Engineered materials with unusual optical properties such
  as negative refractive index
- **Fiber optics**: Mode analysis in optical fibers and tapers

### What Meep Can Do

- Simulate electromagnetic fields in 1D, 2D, 3D, and cylindrical coordinates
- Handle arbitrary geometries (blocks, cylinders, spheres, prisms, and more)
- Model dispersive materials (frequency-dependent permittivity and permeability)
  including metals like gold and silver at optical frequencies
- Compute transmitted and reflected power spectra (flux)
- Find resonant frequencies and quality factors (Q factors) using Harminv
- Transform near-field measurements to far-field radiation patterns
- Launch specific waveguide modes using eigenmode sources (requires MPB)
- Run in parallel across multiple CPU cores or a cluster using MPI
- Export field data for visualization and post-processing

### What Meep Cannot Do

- Simulate quantum effects (Meep is entirely classical electromagnetics)
- Handle thermal or mechanical effects directly
- Automatically design structures for you (though the adjoint solver can help
  with inverse design — see Section 8)
- Simulate non-electromagnetic physics such as fluid dynamics or structural
  mechanics

### Key Concepts

**Wavelength and frequency**: Light has a frequency (how many oscillations per
second) and a wavelength (the physical distance between wave peaks). They are
related by the speed of light c: wavelength = c / frequency. In Meep's unit
system, both are measured in micrometers (microns, μm).

**Permittivity (epsilon, epsilon_r)**: A number that describes how a material
responds to an electric field. In vacuum, epsilon = 1. Glass has epsilon around
2.25. Silicon has epsilon around 12. A material with higher epsilon slows light
down (its refractive index n = sqrt(epsilon)) and can guide or trap it.

**Electric field (E) and magnetic field (H)**: The two components of an
electromagnetic wave. In Meep they are split into six components: Ex, Ey, Ez
(electric) and Hx, Hy, Hz (magnetic), each pointing along a coordinate axis.

**PML (Perfectly Matched Layer)**: An artificial absorbing boundary placed at
the edges of the simulation domain. It absorbs outgoing radiation without
reflecting it back, so the simulation region can be finite even when simulating
waves that would travel to infinity. You almost always need PML.

---

## 2. Installation

### Windows Users

Meep is a C++ library originally designed for Linux/macOS. It does not have a
native Windows build. However, there are three practical ways to run Meep on
Windows, listed from easiest to most involved.

#### Option A: Conda (Recommended for Windows)

Conda (via Miniconda or Anaconda) provides a pre-compiled binary of Meep that
runs inside a Linux-compatible environment. No C++ compilation is required.

**Step 1: Install Miniconda**

Download the Miniconda installer for Windows from:
https://docs.conda.io/en/latest/miniconda.html

Run the installer. When asked, add Miniconda to your PATH (check the box even
though the installer warns against it — it makes things easier for beginners).

**Step 2: Open Anaconda Prompt or Windows Terminal**

Open "Anaconda Prompt" from the Start menu. This gives you a shell where conda
commands work.

**Step 3: Create a conda environment with Meep**

```bash
conda create -n mp -c conda-forge pymeep=*=fftw* python=3.11
conda activate mp
```

The `-n mp` names the environment "mp". The `-c conda-forge` tells conda to
use the community package repository where Meep lives. The `fftw*` selector
picks the serial (non-MPI) build, which is the easiest to start with.

This step downloads about 1-2 GB of packages. It may take several minutes.

**Step 4: Verify the installation**

```bash
python -c "import meep; print(meep.__version__)"
```

You should see a version number printed, such as `1.29.0`. If you see an
import error, try deactivating and reactivating the environment:

```bash
conda deactivate
conda activate mp
```

**Step 5: Run your first simulation**

```bash
python -c "
import meep as mp
sim = mp.Simulation(cell_size=mp.Vector3(10, 10), resolution=10)
print('Meep is working!')
"
```

**What you get**: A complete, working Meep installation with all dependencies
pre-compiled. You do not need a C++ compiler, and you do not need to understand
the build system.

**Limitation**: Conda on Windows uses a compatibility layer. Very occasionally
you may find a feature that requires a newer Meep version than conda provides.
In that case, use WSL2 (Option B).

#### Option B: WSL2 (Windows Subsystem for Linux)

WSL2 runs a real Linux kernel inside Windows. This gives you the full Linux
experience and the most up-to-date Meep via conda, or the ability to build from
source.

**Step 1: Enable WSL2**

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

This installs Ubuntu by default. Restart your computer when prompted.

**Step 2: Set up Ubuntu**

After restart, Ubuntu opens automatically and asks you to create a username and
password. Choose anything you like; these are separate from your Windows
credentials.

**Step 3: Update Ubuntu packages**

```bash
sudo apt update && sudo apt upgrade -y
```

**Step 4: Install Miniconda inside Ubuntu**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts, accept the license, and let it initialize conda in your
shell. Close and reopen the Ubuntu terminal.

**Step 5: Install Meep**

```bash
conda create -n mp -c conda-forge pymeep=*=fftw* python=3.11
conda activate mp
python -c "import meep; print(meep.__version__)"
```

**Accessing your Windows files from WSL2**

Your Windows drives are mounted at `/mnt/c`, `/mnt/d`, etc. Your Windows
Desktop is at `/mnt/c/Users/YourName/Desktop`. You can copy simulation output
files there to open them in Windows:

```bash
cp my_output.png /mnt/c/Users/YourName/Desktop/
```

For best performance, keep your simulation scripts inside the WSL2 filesystem
(e.g., `~/simulations/`) rather than on `/mnt/c/...`.

#### Option C: Docker

Docker runs Meep inside an isolated container. This is useful if you want a
reproducible environment or already use Docker.

**Step 1: Install Docker Desktop**

Download from https://www.docker.com/products/docker-desktop and install it.
Make sure Docker Desktop is running (you should see the Docker icon in the
system tray).

**Step 2: Pull a Meep image**

Open a Command Prompt or PowerShell and run:

```bash
docker pull continuumio/miniconda3
docker run -it continuumio/miniconda3 bash
```

Inside the container:

```bash
conda install -c conda-forge pymeep -y
python -c "import meep; print(meep.__version__)"
```

**Step 3: Run simulations with file sharing**

To share files between your Windows machine and the container:

```bash
docker run -it -v C:\Users\YourName\simulations:/work continuumio/miniconda3 bash
```

This mounts `C:\Users\YourName\simulations` on Windows to `/work` inside the
container. Files you create in `/work` appear in your Windows folder.

### Linux Users

**Conda (easiest)**

```bash
conda create -n mp -c conda-forge pymeep=*=fftw* python=3.11
conda activate mp
python -c "import meep; print(meep.__version__)"
```

For MPI-parallel Meep (recommended for serious simulations):

```bash
conda create -n mpimpi -c conda-forge pymeep=*=mpi_mpich* python=3.11
conda activate mpimpi
```

**From source**: See `DEVELOPER_GUIDE.md` for building from source on Linux,
which gives you the latest code and all optional features.

### macOS Users

**Conda (easiest)**

```bash
conda create -n mp -c conda-forge pymeep=*=fftw* python=3.11
conda activate mp
python -c "import meep; print(meep.__version__)"
```

On Apple Silicon (M1/M2/M3) Macs, use the same command — conda-forge provides
ARM builds. If you encounter issues, try installing Rosetta first:

```bash
softwareupdate --install-rosetta
```

**From source**: See `DEVELOPER_GUIDE.md`.

---

## 3. Core Concepts

### The FDTD Grid (Yee Cell)

Meep divides the simulation domain into a regular grid of cubic (or rectangular)
cells. Each cell stores field values. The electric and magnetic field components
are staggered in space — they sit at the faces and edges of the cells, not all
at the same point. This arrangement is called the "Yee cell" (after Kane Yee,
who invented it in 1966).

```
2D Yee Cell (Ez/Hx/Hy polarization):

    +----------+----------+
    |          |          |
    |   Hx     |   Hx     |   <- H fields at cell centers (for 2D: Hz at center)
    |          |          |
    +----Ey----+----Ey----+   <- Ey at left/right edges
    |          |          |
    |   Hx     |   Hx     |
    |          |          |
    +----------+----------+
         Ey         Ey

    (Ex at top/bottom edges, Ez out of the page at corners for 3D)
```

In 2D simulations, you typically work with one of two polarizations:
- **TM (Ez polarization)**: Ez, Hx, Hy are nonzero. The electric field points
  perpendicular to the 2D plane.
- **TE (Hz polarization)**: Hz, Ex, Ey are nonzero. The magnetic field points
  perpendicular to the 2D plane.

For most photonics work on chip-scale devices, TM (Ez) is common because it
gives a clear picture of the electric field intensity.

### How Time-Stepping Works

At each time step, Meep:

1. Updates the H field at every grid point using the current E field and
   Maxwell's curl equation: dH/dt = -(1/mu) * curl(E)
2. Updates the E field at every grid point using the updated H field:
   dE/dt = (1/epsilon) * (curl(H) - J)   where J is any current source
3. Applies boundary conditions (PML absorbs at edges, periodic boundaries
   for Bloch-periodic problems)
4. Records any monitors (flux, fields, etc.)

The time step size is limited by the "Courant condition": the wave cannot travel
more than one grid cell in one time step or the simulation goes unstable.
Meep chooses the Courant factor automatically (default 0.5 in 2D, varies in 3D).

### Meep's Unit System

This is the most common source of confusion for beginners. Meep uses a
unit system where:

- **Length unit**: 1 μm (micrometer) by convention, but really arbitrary.
  You choose the scale. All distances (cell size, waveguide width, etc.) are
  in your chosen length unit.
- **Frequency unit**: c / (length unit) = c/μm. So a frequency of 0.15 means
  the wave oscillates at 0.15 * c/μm = 0.15 * (3e8 m/s) / (1e-6 m).
- **Time unit**: (length unit) / c = μm/c. One time unit is the time for light
  to travel 1 μm in vacuum.

**The key formula**: `frequency = 1 / wavelength`. If your wavelength is 1.0 μm
(telecom O-band), your Meep frequency is 1.0 / 1.0 = 1.0. If your wavelength is
1.55 μm (telecom C-band), your Meep frequency is 1.0 / 1.55 ≈ 0.645.

**Resolution** is in pixels (grid cells) per length unit. `resolution=10`
means 10 grid cells per μm, so each cell is 0.1 μm = 100 nm wide.

**Rule of thumb for resolution**: You need at least 8 pixels per wavelength
in each material. The wavelength in a material with refractive index n is
lambda_0 / n, where lambda_0 is the free-space wavelength. For silicon (n≈3.5)
at lambda_0 = 1.55 μm, the wavelength in silicon is about 0.44 μm. With
resolution = 20 (pixels per μm), each pixel is 50 nm, giving about 9 pixels
per wavelength in silicon — barely enough. Use 30+ for accurate results.

### PML Absorbing Boundaries

PML stands for Perfectly Matched Layer. It is a layer of artificial absorbing
material placed at the edges of the simulation domain. When a wave enters the
PML, it is absorbed without reflection (for all angles and frequencies, in the
ideal mathematical case).

```
+--PML--+------------------+--PML--+
|///////|                  |///////|
|/PML///|  Active region   |/PML///|
|///////|  (your device)   |///////|
+--PML--+------------------+--PML--+
|///////|                  |///////|
|/PML///|                  |/PML///|
|///////|                  |///////|
+--PML--+------------------+--PML--+
```

**PML thickness**: Use at least half a wavelength of PML, or at least 1 μm,
whichever is larger. `mp.PML(1.0)` creates a 1 μm PML on all sides. You should
not place sources or monitors inside the PML.

**Important**: Never put high-contrast interfaces (like a waveguide) right up
to the PML edge. Leave some buffer (padding) of at least half a wavelength
between your structure and the PML.

### Sources

Meep supports several source types:

**Gaussian pulse** (`mp.GaussianSource`): A broadband source that turns on,
peaks, then decays. It excites a range of frequencies simultaneously. Use this
when you want spectra or when using Harminv to find resonances.

```python
mp.GaussianSource(fcen, fwidth=df)
# fcen: center frequency
# fwidth: frequency width (1/e half-width)
```

**Continuous wave** (`mp.ContinuousSource`): A pure sinusoidal source that
runs forever at one frequency. Use this when you want steady-state field
profiles.

```python
mp.ContinuousSource(frequency=0.15, width=10)
# width: turn-on time in Meep time units (larger = slower turn-on, less ringing)
```

**Eigenmode source** (`mp.EigenModeSource`): Launches a specific guided mode
of a waveguide. Requires MPB to be installed. This is the cleanest way to
excite exactly one mode of a waveguide.

**Source placement**: Sources are placed at a `center` position with a `size`.
A zero-size source is a point source. A line source (size in y only) excites a
plane wave in 2D.

### Monitors

**Flux monitor** (`sim.add_flux`): Measures the time-averaged power (Poynting
flux) flowing through a surface. Use two flux monitors to compute transmittance
and reflectance. Must normalize against a reference run without the scattering
structure.

**DFT field monitor** (`sim.add_dft_fields`): Records the Fourier transform of
fields at specified frequencies. Use this to get the spatial field profile at a
specific wavelength after the simulation runs.

**Near-to-far-field monitor** (`sim.add_near2far`): Records near fields that
can later be transformed to far-field radiation patterns using Huygens' principle.

**Harminv**: Not a monitor in the traditional sense, but a signal processing
tool that analyzes the time-domain field to extract resonant frequencies and
their decay rates (Q factors). Use it after a Gaussian pulse excitation.

---

## 4. Quick Start: Your First Simulation

Let's simulate a 2D dielectric waveguide — the simplest useful photonics
structure. We will launch a continuous wave, run the simulation, and visualize
the fields.

```
Simulation layout:

Y
^
|    +-----PML (1 μm)-----+
|    |                     |
|    |   +-waveguide-+     |
|    |   |  epsilon=12     |  <- light travels right
|    |   |  width = 1 μm   |
|    |   +-----------+     |
|    |   ^           ^     |
|    | source      monitor  |
|    +---------------------+
+---------------------------------> X
     (cell: 16 μm wide, 8 μm tall)
```

```python
# first_simulation.py
# A 2D dielectric waveguide: the "Hello, World" of photonics simulation.

import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------
# Step 1: Define the simulation cell (the computational domain).
# mp.Vector3(x, y, z) — z=0 means 2D (infinite in z, but no z variation).
# All distances are in micrometers (μm) by convention.
# -----------------------------------------------------------------------
cell = mp.Vector3(16, 8, 0)   # 16 μm wide, 8 μm tall, 2D

# -----------------------------------------------------------------------
# Step 2: Define the geometry (what materials go where).
# mp.Block is a rectangular region. mp.inf means the block extends to
# the boundary of the simulation in that direction.
# epsilon=12 is roughly the permittivity of silicon (refractive index ~3.46).
# -----------------------------------------------------------------------
geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, 1, mp.inf),   # infinitely long, 1 μm tall
        center=mp.Vector3(0, 0),              # centered at the origin
        material=mp.Medium(epsilon=12),       # high-index dielectric (silicon-like)
    )
]

# -----------------------------------------------------------------------
# Step 3: Define the source.
# ContinuousSource runs at a single frequency indefinitely.
# frequency=0.15 means wavelength = 1/0.15 ≈ 6.67 μm.
# component=mp.Ez means we excite the Ez field (TM polarization).
# center=mp.Vector3(-7, 0) places the source near the left edge.
# -----------------------------------------------------------------------
sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=0.15),
        component=mp.Ez,
        center=mp.Vector3(-7, 0),   # 7 μm left of center (inside the waveguide)
        size=mp.Vector3(0, 0, 0),   # point source
    )
]

# -----------------------------------------------------------------------
# Step 4: Define PML boundary layers.
# 1.0 μm PML on all sides — absorbs outgoing waves without reflection.
# At frequency 0.15 (wavelength 6.67 μm), 1 μm is thin but sufficient
# for this demonstration. Use 1-2 wavelengths for production work.
# -----------------------------------------------------------------------
pml_layers = [mp.PML(1.0)]

# -----------------------------------------------------------------------
# Step 5: Set resolution.
# 10 pixels per μm means each grid cell is 100 nm.
# Wavelength = 6.67 μm, so we have ~67 pixels per wavelength.
# This is very well resolved (8 is the minimum).
# -----------------------------------------------------------------------
resolution = 10

# -----------------------------------------------------------------------
# Step 6: Create the Simulation object.
# This is where everything comes together.
# -----------------------------------------------------------------------
sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# -----------------------------------------------------------------------
# Step 7: Run the simulation.
# until=200 means run for 200 time units (200 μm/c ≈ 0.67 ps).
# At frequency 0.15, one period = 1/0.15 ≈ 6.67 time units, so we run
# for about 30 oscillation periods — enough for steady state.
# -----------------------------------------------------------------------
sim.run(until=200)

# -----------------------------------------------------------------------
# Step 8: Visualize results.
# get_array extracts field data as a NumPy array for plotting.
# -----------------------------------------------------------------------

# Get the permittivity (epsilon) distribution
eps_data = sim.get_array(
    center=mp.Vector3(),   # center of the array to extract
    size=cell,             # size of the region to extract
    component=mp.Dielectric  # extract epsilon
)

# Get the Ez electric field
ez_data = sim.get_array(
    center=mp.Vector3(),
    size=cell,
    component=mp.Ez
)

# Plot epsilon (shows the waveguide geometry)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(
    eps_data.transpose(),   # transpose because arrays are [x,y] but imshow wants [y,x]
    interpolation="spline36",
    cmap="binary",
    origin="lower"
)
plt.title("Permittivity (epsilon)")
plt.colorbar()
plt.axis("off")

# Plot Ez field overlaid on epsilon
plt.subplot(1, 2, 2)
plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary", origin="lower")
plt.imshow(
    ez_data.transpose(),
    interpolation="spline36",
    cmap="RdBu",      # red-blue colormap for +/- fields
    alpha=0.9,        # semi-transparent so geometry shows through
    origin="lower"
)
plt.title("Ez field (steady state)")
plt.axis("off")

plt.tight_layout()
plt.savefig("first_simulation.png", dpi=150)
plt.show()

print("Simulation complete! Output saved to first_simulation.png")
```

**How to run this**:

```bash
conda activate mp
python first_simulation.py
```

**Expected output**: A PNG image showing two panels. The left panel shows a
thin horizontal white bar (the waveguide, epsilon=12 against background
epsilon=1). The right panel shows the waveguide with alternating red and blue
bands propagating to the right — the oscillating Ez field guided by the
waveguide. Near the source (left side) you may see some unguided radiation;
this decays and only the guided mode remains farther along the waveguide.

---

## 5. Ten Educational Use Cases

### Use Case 1: Point Source in Free Space (1D)

**Physical concept**: The simplest FDTD simulation possible. A dipole source
radiates a Gaussian pulse into free space. The fields radiate outward, pass
through PML, and decay. We detect when the field has fully decayed.

**What you will learn**: Basic simulation setup, PML, `until_after_sources`,
`stop_when_fields_decayed`, and reading field data.

```
Simulation layout (1D cross-section, viewed from side):

|PML| ... vacuum ... |*source*| ... vacuum ... |PML|
  1 μm                              (detects field decay here)
```

```python
# use_case_1_point_source_1d.py
# A Gaussian pulse radiating in 1D free space.

import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------
# 1D simulation: cell has y=0 and z=0, only x has extent.
# -----------------------------------------------------------------------
sx = 16             # cell length in x, in μm
cell = mp.Vector3(sx, 0, 0)   # y=0, z=0 means 1D

dpml = 1.0          # PML thickness in μm
pml_layers = [mp.PML(dpml)]

# Center frequency and bandwidth of the Gaussian pulse.
# fcen=0.5 means wavelength = 2 μm. df=0.4 gives a broadband pulse.
fcen = 0.5
df = 0.4

sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ex,        # 1D: use Ex (or Hz)
        center=mp.Vector3(0),   # source at the center
    )
]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    sources=sources,
    resolution=20,              # 20 pixels/μm: good for wavelength ~2 μm
)

# Record Ex at a point to the right of the source over time.
monitor_pt = mp.Vector3(sx / 2 - dpml - 0.5)  # 0.5 μm inside PML edge
t_data = []
ex_data = []

def record_field(sim):
    t_data.append(sim.meep_time())
    ex_data.append(sim.get_field_point(mp.Ex, monitor_pt))

# Run until the source has peaked and the fields have decayed to
# 1e-3 of their peak value at the monitor point.
# stop_when_fields_decayed(decay_time, component, point, threshold)
# checks every decay_time units if fields at point are below threshold * max.
sim.run(
    mp.at_every(0.1, record_field),    # record field every 0.1 time units
    until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, monitor_pt, 1e-3),
)

# Convert to numpy arrays for plotting
t_data = np.array(t_data, dtype=float)
ex_data = np.real(np.array(ex_data, dtype=complex))

print(f"Simulation ran until t = {t_data[-1]:.1f} μm/c")
print(f"Total time steps: {len(t_data)}")

# Plot the field vs time at the monitor point
plt.figure(figsize=(8, 4))
plt.plot(t_data, ex_data)
plt.xlabel("Time (μm/c)")
plt.ylabel("Ex field amplitude")
plt.title("Gaussian pulse detected at monitor point (1D free space)")
plt.grid(True)
plt.tight_layout()
plt.savefig("uc1_pulse.png", dpi=150)
plt.show()
```

**Expected output**: A plot showing a Gaussian-shaped pulse that rises, peaks,
then decays. The pulse width in time corresponds to the source bandwidth df.
The simulation stops automatically once the fields have decayed.

---

### Use Case 2: Dielectric Slab Waveguide (2D)

**Physical concept**: A high-index slab surrounded by lower-index material
confines light by total internal reflection. This is the building block of
integrated photonic circuits.

**What you will learn**: Geometry definition, `mp.Block`, continuous source,
field visualization, the concept of guided modes.

```
Simulation layout:

    +--PML--+---------------------------+--PML--+
    |///////|     cladding (air, n=1)   |///////|
    |/PML///+===========================+/PML///|
    |///////|   waveguide core (n=3.46) |///////|  <- guided mode here
    |/PML///+===========================+/PML///|
    |///////|     cladding (air, n=1)   |///////|
    +--PML--+---------------------------+--PML--+
                   ^
              source here (left)
```

```python
# use_case_2_slab_waveguide.py
# Light guided in a 2D dielectric slab waveguide.

import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# Cell size: 20 μm wide, 10 μm tall, 2D
cell = mp.Vector3(20, 10, 0)

dpml = 1.0                    # PML thickness
pml_layers = [mp.PML(dpml)]

# Waveguide parameters
wg_width = 1.0                # waveguide width in μm
wg_eps = 12.0                 # epsilon = 12 -> n = sqrt(12) ≈ 3.46 (like silicon)

# The waveguide is a Block that spans the full x extent.
geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, wg_width, mp.inf),  # infinite in x and z
        center=mp.Vector3(0, 0),                    # centered on y=0
        material=mp.Medium(epsilon=wg_eps),
    )
]

# Continuous source at frequency 0.15 (wavelength = 6.67 μm).
# The source size covers the waveguide cross-section in y.
fcen = 0.15
sources = [
    mp.Source(
        src=mp.ContinuousSource(fcen, width=20),  # width=20: slow turn-on
        component=mp.Ez,
        center=mp.Vector3(-8, 0),                 # 8 μm left of center
        size=mp.Vector3(0, wg_width * 2),         # spans ±1 μm in y
    )
]

resolution = 10

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# Run long enough for steady state (100+ periods at fcen=0.15 -> period ≈ 6.67)
sim.run(until=400)

# Extract data for visualization
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

plt.figure(figsize=(12, 5))

# Epsilon plot (geometry)
plt.subplot(1, 2, 1)
plt.imshow(
    eps_data.transpose(),
    interpolation="spline36",
    cmap="binary",
    origin="lower",
    extent=[-10, 10, -5, 5],  # match cell coordinates
)
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.title("Geometry (epsilon)")
plt.colorbar(label="epsilon")

# Ez field plot
plt.subplot(1, 2, 2)
plt.imshow(
    eps_data.transpose(),
    interpolation="spline36",
    cmap="binary",
    origin="lower",
    extent=[-10, 10, -5, 5],
    alpha=0.3,
)
im = plt.imshow(
    ez_data.transpose(),
    interpolation="spline36",
    cmap="RdBu",
    origin="lower",
    extent=[-10, 10, -5, 5],
)
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.title("Ez field — guided mode")
plt.colorbar(im, label="Ez")

plt.tight_layout()
plt.savefig("uc2_waveguide.png", dpi=150)
plt.show()

print(
    f"Wavelength = {1/fcen:.2f} μm, waveguide n = {wg_eps**0.5:.2f}, "
    f"wavelength in core = {1/fcen/wg_eps**0.5:.2f} μm"
)
```

**Expected output**: The field plot shows an oscillating wave (Ez) that is
confined to the waveguide slab and propagates in the +x direction. The field
decays exponentially above and below the slab — this is the evanescent field.
The guided mode has a sinusoidal shape inside the core.

---

### Use Case 3: Waveguide Bend — Measuring Transmission (2D)

**Physical concept**: A 90-degree bend in a waveguide is a common test
structure. Light that cannot navigate the bend is reflected or radiated away.
We measure how much power makes it through.

**What you will learn**: Flux monitors, `add_flux`, normalization (two-run
technique), `get_fluxes`, reflectance/transmittance calculation.

**The key insight about normalization**: To get transmittance as a fraction
(0 to 1), you need to know the incident power. You run the simulation twice:
(1) straight waveguide to measure incident power, (2) bent waveguide to measure
transmitted power. Dividing gives transmittance.

```
Simulation layout:

    +--PML--+-------------------------+--PML--+
    |///////|                         |///////|
    |/PML///|   horizontal waveguide  |/PML///|
  S |///////|===========================|///////|
  R |/PML///| bend        |           |/PML///|  T
  C |///////|             |           |///////|  r
    |/PML///|             |           |/PML///|  a
    |///////|             v waveguide |///////|  n
    |/PML///|             (vertical)  |/PML///|  s
    |///////|                         |///////|
    +--PML--+-------------------------+--PML--+
         ^flux_refl                        ^flux_tran
```

```python
# use_case_3_waveguide_bend.py
# Measure transmission through a 90-degree waveguide bend.
# Implements the standard two-run normalization procedure.

import meep as mp
import matplotlib.pyplot as plt
import numpy as np

resolution = 10    # pixels/μm

sx = 16            # cell x size in μm
sy = 32            # cell y size in μm (taller to fit vertical arm)
cell = mp.Vector3(sx, sy, 0)

dpml = 1.0         # PML thickness in μm
pad = 4            # gap between waveguide and PML edge in μm
w = 1              # waveguide width in μm

# Calculate center positions for the waveguide arms.
# The horizontal arm runs at y = wvg_ycen.
# The vertical arm runs at x = wvg_xcen.
wvg_xcen = 0.5 * (sx - w - 2 * pad)    #  x center of the vertical arm
wvg_ycen = -0.5 * (sy - w - 2 * pad)  #  y center of the horizontal arm

pml_layers = [mp.PML(dpml)]

fcen = 0.15        # center frequency
df = 0.1           # frequency bandwidth

# Gaussian pulse source on the left end of the horizontal arm.
# The source size covers the waveguide in y.
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(-0.5 * sx + dpml, wvg_ycen, 0),
        size=mp.Vector3(0, w, 0),
    )
]

# ===================================================================
# RUN 1: Straight waveguide (no bend) to get incident power reference.
# ===================================================================

# Straight horizontal waveguide only.
geometry_straight = [
    mp.Block(
        size=mp.Vector3(mp.inf, w, mp.inf),
        center=mp.Vector3(0, wvg_ycen, 0),
        material=mp.Medium(epsilon=12),
    )
]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry_straight,
    sources=sources,
    resolution=resolution,
)

nfreq = 100   # number of frequency points in the spectrum

# Reflection monitor: just to the right of the source.
refl_fr = mp.FluxRegion(
    center=mp.Vector3(-0.5 * sx + dpml + 0.5, wvg_ycen, 0),
    size=mp.Vector3(0, 2 * w, 0),
)
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# Transmission monitor: near the right edge.
tran_fr_straight = mp.FluxRegion(
    center=mp.Vector3(0.5 * sx - dpml - 0.5, wvg_ycen, 0),
    size=mp.Vector3(0, 2 * w, 0),
)
tran = sim.add_flux(fcen, df, nfreq, tran_fr_straight)

# Wait for the pulse to pass through completely.
pt = mp.Vector3(0.5 * sx - dpml - 0.5, wvg_ycen)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

# Save the reflection flux data so we can subtract it in Run 2
# (to get only the reflected power from the bend, not the incident power).
straight_refl_data = sim.get_flux_data(refl)

# Save the incident (transmitted) power spectrum from the straight run.
straight_tran_flux = mp.get_fluxes(tran)

print("Run 1 (straight waveguide) complete.")

# ===================================================================
# RUN 2: Bent waveguide (the actual structure we want to measure).
# ===================================================================

sim.reset_meep()   # clear all fields and structures

# The bent waveguide: horizontal arm + vertical arm.
geometry_bent = [
    # Horizontal arm: from left to the bend point
    mp.Block(
        size=mp.Vector3(sx - pad, w, mp.inf),
        center=mp.Vector3(-0.5 * pad, wvg_ycen, 0),
        material=mp.Medium(epsilon=12),
    ),
    # Vertical arm: from the bend point upward
    mp.Block(
        size=mp.Vector3(w, sy - pad, mp.inf),
        center=mp.Vector3(wvg_xcen, 0.5 * pad, 0),
        material=mp.Medium(epsilon=12),
    ),
]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry_bent,
    sources=sources,
    resolution=resolution,
)

refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# Transmission monitor is now at the top of the vertical arm.
tran_fr_bent = mp.FluxRegion(
    center=mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5, 0),
    size=mp.Vector3(2 * w, 0, 0),
)
tran = sim.add_flux(fcen, df, nfreq, tran_fr_bent)

# Load the negated straight-run reflection data.
# This subtracts the incident wave from the reflected measurement,
# leaving only the wave reflected by the bend.
sim.load_minus_flux_data(refl, straight_refl_data)

pt = mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

bend_refl_flux = mp.get_fluxes(refl)
bend_tran_flux = mp.get_fluxes(tran)
flux_freqs = mp.get_flux_freqs(refl)

print("Run 2 (bent waveguide) complete.")

# ===================================================================
# Compute and plot transmittance and reflectance spectra.
# ===================================================================

wavelengths = [1 / f for f in flux_freqs]
reflectances = [-r / i for r, i in zip(bend_refl_flux, straight_tran_flux)]
transmittances = [t / i for t, i in zip(bend_tran_flux, straight_tran_flux)]
losses = [1 - r - t for r, t in zip(reflectances, transmittances)]

plt.figure(figsize=(8, 5))
plt.plot(wavelengths, reflectances, "bo-", label="Reflectance", markersize=3)
plt.plot(wavelengths, transmittances, "ro-", label="Transmittance", markersize=3)
plt.plot(wavelengths, losses, "go-", label="Loss (radiation)", markersize=3)
plt.xlim([min(wavelengths), max(wavelengths)])
plt.ylim([0, 1])
plt.xlabel("Wavelength (μm)")
plt.ylabel("Fraction of incident power")
plt.title("90-degree Waveguide Bend: Transmission Spectrum")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("uc3_bend_transmission.png", dpi=150)
plt.show()
```

**Expected output**: A spectrum plot showing transmittance, reflectance, and
loss versus wavelength. For a sharp 90-degree bend in a silicon waveguide (n≈3.5,
width=1 μm), transmission is imperfect. Longer wavelengths (approaching cutoff)
transmit less. The three curves should sum to 1.0 at each wavelength (energy
conservation).

---

### Use Case 4: Ring Resonator — Finding Resonances (2D)

**Physical concept**: A ring-shaped waveguide traps light at specific
resonant frequencies (those where the round-trip phase is a multiple of 2π).
At resonance, light builds up in the ring and slowly leaks out. The quality
factor Q tells you how many oscillations the light makes before escaping.

**What you will learn**: Cylinder geometry, Harminv signal analysis, resonant
frequencies, Q factors.

```
Simulation layout:

    +--PML--+-------------------+--PML--+
    |///////|                   |///////|
    |/PML///|     +---+         |/PML///|
    |///////|   /       \       |///////|
    |/PML///|  |  (air)  |      |/PML///|
    |///////|   \       /       |///////|
    |/PML///|     +---+         |/PML///|
    |///////|   *src              |///////|  <- point source excites the ring
    +--PML--+-------------------+--PML--+

    The ring: outer cylinder (n=3.4) minus inner cylinder (air)
```

```python
# use_case_4_ring_resonator.py
# Find resonant frequencies and Q factors of a 2D ring resonator.

import meep as mp

def main():
    # Ring resonator parameters (all in μm)
    n = 3.4           # refractive index of ring material
    w = 1             # width of the ring (annular region)
    r = 1             # inner radius
    pad = 4           # padding from waveguide to PML
    dpml = 2          # PML thickness

    # Cell is square, big enough to fit the ring + padding + PML
    sxy = 2 * (r + w + pad + dpml)
    cell = mp.Vector3(sxy, sxy, 0)

    # Build the ring as two overlapping cylinders.
    # Later objects take precedence over earlier ones:
    # outer cylinder (dielectric) is placed first,
    # inner cylinder (air, no material = vacuum) is placed second
    # and cuts out the hole.
    outer_cylinder = mp.Cylinder(
        radius=r + w,
        material=mp.Medium(index=n),   # use index= to specify refractive index
    )
    inner_cylinder = mp.Cylinder(
        radius=r,
        # No material argument = vacuum (epsilon=1)
    )
    geometry = [outer_cylinder, inner_cylinder]

    # Gaussian pulse source at the inner edge of the ring.
    # This excites many resonances at once.
    fcen = 0.15       # center frequency
    df = 0.1          # bandwidth (covers multiple ring modes)
    src = mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(r + 0.1),   # just inside the ring (slightly past r)
    )

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=[src],
        resolution=10,
        # Mirror symmetry about y=0: the modes we excite are symmetric/antisymmetric
        # in y. This halves the computational cost.
        symmetries=[mp.Mirror(mp.Y)],
        boundary_layers=[mp.PML(dpml)],
    )

    # Harminv watches a field component at a point and extracts frequencies.
    # It runs during the simulation via mp.after_sources (starts after the source
    # has mostly decayed, so only the ring's natural oscillations are analyzed).
    harminv_monitor = mp.Harminv(
        mp.Ez,              # field component to analyze
        mp.Vector3(r + 0.1),  # same point as source
        fcen,               # center of frequency range to search
        df,                 # width of frequency range
    )

    # Run:
    # - at_beginning: output epsilon at t=0 (one snapshot of geometry)
    # - after_sources: start Harminv once source has peaked
    # - until_after_sources=300: run 300 time units after source peak
    sim.run(
        mp.at_beginning(mp.output_epsilon),
        mp.after_sources(harminv_monitor),
        until_after_sources=300,
    )

    # Harminv results are printed automatically during the run.
    # They look like:
    #   harminv0:, frequency, imag. freq., Q, |amp|, amplitude, error
    # where frequency is the resonant frequency and Q is the quality factor.

    print("\nHarminv found the following resonances:")
    print("freq (μm^-1) | wavelength (μm) | Q factor")
    print("-" * 45)
    for mode in harminv_monitor.modes:
        freq = mode.freq
        Q = mode.Q
        if Q > 0:   # positive Q means a real resonance
            print(f"  {freq:.6f}   |   {1/freq:.4f}        |  {Q:.1f}")

if __name__ == "__main__":
    main()
```

**Expected output**: Printed lines listing resonant frequencies and Q factors
of the ring resonator. For the parameters above (r=1 μm, w=1 μm, n=3.4), you
should find several resonances in the frequency range 0.1-0.2. Higher Q means
the mode is more tightly confined. Q values of 100-1000 are typical for simple
2D ring resonators without sophisticated design.

---

### Use Case 5: Mie Scattering from a Sphere (3D)

**Physical concept**: When a sphere is illuminated by a plane wave, it scatters
some of the light in all directions. The efficiency of scattering depends on
the sphere size relative to the wavelength. Mie theory gives the exact
analytical answer; this simulation verifies it.

**What you will learn**: 3D simulations, Sphere geometry, plane wave sources,
scattering cross-section, flux box technique, comparison with theory.

```
Simulation layout (3D, shown as 2D cross-section):

    +---PML---+------------------------------+---PML---+
    |/////////|                              |/////////|
    |/////////|       plane wave ->          |/////////|
    |/////////|          [sphere]            |/////////|
    |/////////|      /----+----\             |/////////|
    |/////////|     |  n=2.0   |             |/////////|
    |/////////|      \---------/             |/////////|
    |/////////|   flux box (6 faces)         |/////////|
    +---PML---+------------------------------+---PML---+
    source here ->
```

```python
# use_case_5_mie_scattering.py
# Mie scattering from a dielectric sphere: comparing Meep to analytical theory.

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Sphere parameters
r = 1.0           # sphere radius in μm
n_sphere = 2.0    # refractive index of sphere

# Frequency range: circumference/wavelength from 0.2 to 5 (covers Mie regime)
wvl_min = 2 * np.pi * r / 10    # minimum wavelength in μm
wvl_max = 2 * np.pi * r / 2     # maximum wavelength in μm
frq_min = 1 / wvl_max
frq_max = 1 / wvl_min
frq_cen = 0.5 * (frq_min + frq_max)   # center frequency
dfrq = frq_max - frq_min               # frequency width
nfrq = 50                              # number of frequency points

# Resolution: at least 8 pixels per wavelength inside the sphere.
# Wavelength in sphere = wvl_min / n_sphere ≈ 0.63/2 = 0.31 μm.
# resolution=25 gives 25 px/μm = 1 px per 0.04 μm → ~8 px per 0.31 μm.
resolution = 25

dpml = 0.5 * wvl_max   # PML thickness: half the longest wavelength
dair = 0.5 * wvl_max   # gap between sphere and PML

pml_layers = [mp.PML(thickness=dpml)]

# Use symmetry to halve (and quarter) the simulation cost.
# Mirror(Y) and Mirror(Z, phase=-1) exploit the plane wave's polarization.
symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

s = 2 * (dpml + dair + r)    # total cell size (cube)
cell_size = mp.Vector3(s, s, s)

# Plane wave source extending across the full yz face of the cell.
# is_integrated=True is required for sources that extend into PML.
sources = [
    mp.Source(
        src=mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        center=mp.Vector3(-0.5 * s + dpml),   # at the left PML face
        size=mp.Vector3(0, s, s),              # full yz cross-section
        component=mp.Ez,
    )
]

# ===================================================================
# RUN 1: No sphere (measure incident field for normalization).
# ===================================================================

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),     # needed for plane wave sources
    symmetries=symmetries,
)

# Six flux monitors forming a closed box around where the sphere will be.
# Net outward flux from this box = power scattered by sphere.
box_x1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2*r, 2*r)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2*r, 2*r)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2*r, 0, 2*r)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2*r, 0, 2*r)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2*r, 2*r, 0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2*r, 2*r, 0)))

sim.run(until_after_sources=10)

# Save incident flux data (needed to subtract from scattering measurement)
box_x1_data = sim.get_flux_data(box_x1)
box_x2_data = sim.get_flux_data(box_x2)
box_y1_data = sim.get_flux_data(box_y1)
box_y2_data = sim.get_flux_data(box_y2)
box_z1_data = sim.get_flux_data(box_z1)
box_z2_data = sim.get_flux_data(box_z2)

# The incident power (per unit area) measured through the -x face.
# We use this to compute the cross-section later.
box_x1_flux0 = mp.get_fluxes(box_x1)
freqs = mp.get_flux_freqs(box_x1)

print("Run 1 (no sphere) complete.")

sim.reset_meep()

# ===================================================================
# RUN 2: With sphere.
# ===================================================================

geometry = [
    mp.Sphere(
        material=mp.Medium(index=n_sphere),
        center=mp.Vector3(),
        radius=r,
    )
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),
    symmetries=symmetries,
    geometry=geometry,
)

# Same six flux monitors
box_x1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2*r, 2*r)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2*r, 2*r)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2*r, 0, 2*r)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2*r, 0, 2*r)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2*r, 2*r, 0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2*r, 2*r, 0)))

# Load negated incident flux data: now monitors measure SCATTERED field only.
sim.load_minus_flux_data(box_x1, box_x1_data)
sim.load_minus_flux_data(box_x2, box_x2_data)
sim.load_minus_flux_data(box_y1, box_y1_data)
sim.load_minus_flux_data(box_y2, box_y2_data)
sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)

sim.run(until_after_sources=100)

# Net scattered power = sum of outward fluxes (with sign convention).
# Convention: +x face has positive outward normal, -x face has negative.
# Meep's add_flux measures flux in the +normal direction, so:
#   total = (+x2) - (-x1) + (+y2) - (-y1) + (+z2) - (-z1)
#   But with load_minus_flux_data, the scattered field contributions flip.
box_x1_flux = mp.get_fluxes(box_x1)
box_x2_flux = mp.get_fluxes(box_x2)
box_y1_flux = mp.get_fluxes(box_y1)
box_y2_flux = mp.get_fluxes(box_y2)
box_z1_flux = mp.get_fluxes(box_z1)
box_z2_flux = mp.get_fluxes(box_z2)

scatt_flux = np.array(box_x1_flux) - np.array(box_x2_flux) \
           + np.array(box_y1_flux) - np.array(box_y2_flux) \
           + np.array(box_z1_flux) - np.array(box_z2_flux)

# Incident intensity = incident power / area of the -x face that surrounds sphere
intensity = np.array(box_x1_flux0) / (2 * r) ** 2
scatt_cross_section = scatt_flux / intensity
# Scattering efficiency Q_sca = cross-section / geometric cross-section
scatt_eff_meep = -scatt_cross_section / (np.pi * r ** 2)

print("Run 2 (with sphere) complete.")

# ===================================================================
# Compare with Mie theory using scipy or a simple approximation.
# For a rigorous comparison, install PyMieScatt: pip install PyMieScatt
# ===================================================================

try:
    import PyMieScatt as ps
    scatt_eff_theory = [
        ps.MieQ(n_sphere, 1000 / f, 2 * r * 1000, asDict=True)["Qsca"]
        for f in freqs
    ]
    has_theory = True
except ImportError:
    has_theory = False
    print("PyMieScatt not installed; skipping theory comparison.")
    print("Install it with: pip install PyMieScatt")

# Plot results
x_axis = 2 * np.pi * r * np.array(freqs)   # size parameter = 2*pi*r/lambda

plt.figure(figsize=(8, 5))
plt.loglog(x_axis, scatt_eff_meep, "bo-", label="Meep", markersize=4)
if has_theory:
    plt.loglog(x_axis, scatt_eff_theory, "r-", label="Mie theory", linewidth=2)
plt.xlabel("Size parameter 2πr/λ")
plt.ylabel("Scattering efficiency Q_sca")
plt.title(f"Mie Scattering: Dielectric Sphere (n={n_sphere}, r={r} μm)")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("uc5_mie_scattering.png", dpi=150)
plt.show()
```

**Expected output**: A log-log plot showing the scattering efficiency as a
function of sphere size parameter. You will see a series of peaks (Mie
resonances) where the sphere scatters much more strongly. The Meep results
should closely match the Mie theory curve, confirming the simulation is correct.

---

### Use Case 6: Metallic Nanoparticle with Predefined Material (2D)

**Physical concept**: Gold and silver nanoparticles have strong optical
resonances called plasmon resonances, where free electrons in the metal
collectively oscillate. These resonances cause intense absorption and
scattering at specific colors (this is why colloidal gold nanoparticles
are red).

**What you will learn**: Using Meep's built-in materials library, dispersive
materials (Drude model), frequency-dependent response.

```
Simulation layout:

    +--PML--+----------------------------+--PML--+
    |///////|                            |///////|
    |/PML///|   plane wave ->            |/PML///|
    |///////|      +------+              |///////|
    |/PML///|      | Gold |              |/PML///|
    |///////|      | disk |              |///////|
    |/PML///|      +------+              |/PML///|
    |///////|   flux monitors around it  |///////|
    +--PML--+----------------------------+--PML--+
```

```python
# use_case_6_gold_nanoparticle.py
# Optical scattering from a gold disk using Meep's materials library.

import meep as mp
import meep.materials as mat   # Meep's built-in material library
import numpy as np
import matplotlib.pyplot as plt

# Gold nanoparticle parameters
radius = 0.05       # 50 nm radius gold disk (in μm)

# Wavelength range: visible spectrum, 0.4-0.7 μm
wvl_min = 0.4
wvl_max = 0.7
frq_min = 1 / wvl_max
frq_max = 1 / wvl_min
frq_cen = 0.5 * (frq_min + frq_max)
dfrq = frq_max - frq_min
nfrq = 50

# High resolution needed: features are 50 nm = 0.05 μm
# resolution=200 gives 200 px/μm = 5 nm/pixel
resolution = 200

dpml = 0.5 * wvl_max
dair = 2 * radius + 0.3    # gap between particle and PML

s = 2 * (dpml + dair)
cell_size = mp.Vector3(s, s, 0)   # 2D simulation

pml_layers = [mp.PML(thickness=dpml)]

# Symmetries: for a point/plane wave along x and Ez polarization:
symmetries = [mp.Mirror(mp.Y, phase=-1)]

# Plane wave source
sources = [
    mp.Source(
        src=mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s, 0),
        component=mp.Ez,
    )
]

# ===================================================================
# RUN 1: No nanoparticle (normalization run).
# ===================================================================

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),
    symmetries=symmetries,
)

# Flux box around where the nanoparticle will go
box_r = radius * 1.5   # flux box size (slightly larger than particle)
mon_x1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(x=-box_r), size=mp.Vector3(0, 2*box_r)))
mon_x2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(x=+box_r), size=mp.Vector3(0, 2*box_r)))
mon_y1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(y=-box_r), size=mp.Vector3(2*box_r, 0)))
mon_y2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(y=+box_r), size=mp.Vector3(2*box_r, 0)))

sim.run(until_after_sources=50)

mon_x1_data = sim.get_flux_data(mon_x1)
mon_x2_data = sim.get_flux_data(mon_x2)
mon_y1_data = sim.get_flux_data(mon_y1)
mon_y2_data = sim.get_flux_data(mon_y2)
freqs = mp.get_flux_freqs(mon_x1)
norm_flux = mp.get_fluxes(mon_x1)

print("Normalization run complete.")
sim.reset_meep()

# ===================================================================
# RUN 2: With gold nanoparticle.
# mat.Au is gold from Meep's materials library.
# It uses a Drude-Lorentz model fitted to experimental optical data.
# ===================================================================

print("\nUsing gold material from Meep materials library.")
print("Gold is valid approximately in the wavelength range 0.5-1.0 μm.")

geometry = [
    mp.Cylinder(
        radius=radius,
        material=mat.Au,          # gold from materials library
        center=mp.Vector3(),
    )
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),
    symmetries=symmetries,
    geometry=geometry,
)

mon_x1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(x=-box_r), size=mp.Vector3(0, 2*box_r)))
mon_x2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(x=+box_r), size=mp.Vector3(0, 2*box_r)))
mon_y1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(y=-box_r), size=mp.Vector3(2*box_r, 0)))
mon_y2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(y=+box_r), size=mp.Vector3(2*box_r, 0)))

sim.load_minus_flux_data(mon_x1, mon_x1_data)
sim.load_minus_flux_data(mon_x2, mon_x2_data)
sim.load_minus_flux_data(mon_y1, mon_y1_data)
sim.load_minus_flux_data(mon_y2, mon_y2_data)

sim.run(until_after_sources=200)

x1_flux = mp.get_fluxes(mon_x1)
x2_flux = mp.get_fluxes(mon_x2)
y1_flux = mp.get_fluxes(mon_y1)
y2_flux = mp.get_fluxes(mon_y2)

# Total scattered power = net outward flux
scatt_flux = (np.array(x1_flux) - np.array(x2_flux)
            + np.array(y1_flux) - np.array(y2_flux))

# Normalize by incident intensity to get scattering cross-section (per unit length in 2D)
incident_intensity = np.array(norm_flux) / (2 * box_r)
scatt_cross_section = -scatt_flux / incident_intensity

# Plot: scattering cross-section vs wavelength
wavelengths = [1 / f for f in freqs]

plt.figure(figsize=(8, 5))
plt.plot(wavelengths, scatt_cross_section, "b-", linewidth=2)
plt.xlabel("Wavelength (μm)")
plt.ylabel("Scattering cross-section (μm)")
plt.title("Gold Nanoparticle (r=50 nm): Scattering Spectrum")
plt.grid(True)
plt.tight_layout()
plt.savefig("uc6_gold_nanoparticle.png", dpi=150)
plt.show()

# Print available materials
print("\nOther predefined materials in meep.materials:")
print("  mat.cSi   - crystalline silicon")
print("  mat.aSi   - amorphous silicon")
print("  mat.SiO2  - silicon dioxide")
print("  mat.Al    - aluminum")
print("  mat.Ag    - silver")
print("  mat.Au    - gold")
print("  mat.Cu    - copper")
print("  mat.TiO2  - titanium dioxide")
```

**Expected output**: A plot of scattering cross-section versus wavelength for
the gold nanoparticle. You should see a peak (plasmonic resonance) in the
visible range. For a 50 nm gold disk in 2D, the plasmon peak appears around
0.5-0.6 μm. The exact position depends on the particle shape and surrounding
medium.

---

### Use Case 7: Photonic Crystal Bandgap (2D)

**Physical concept**: A periodic array of dielectric rods creates a "photonic
bandgap" — a range of frequencies that cannot propagate through the crystal,
similar to how a semiconductor bandgap blocks electrons of certain energies.

**What you will learn**: Periodic boundaries (k_point), Bloch waves, band
structure concepts, `run_k_points`.

```
Simulation layout (one unit cell, repeated periodically):

    +--PML(y)--+--------+--PML(y)--+
    |//////////|        |//////////|
    |//////////|   O    |//////////|  <- dielectric rod (circle)
    |//////////|        |//////////|
    +----------+--------+----------+
    periodic in x (no PML in x)

    Many unit cells in x direction, each with one rod.
```

```python
# use_case_7_photonic_crystal.py
# 2D photonic crystal: find the band structure of a square lattice of rods.

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def compute_bands(num_k_points=20):
    """
    Compute the photonic band structure of a 2D square lattice
    of dielectric rods in air.

    The lattice constant a=1 μm, rod radius r=0.2 μm, rod epsilon=12.
    We scan k-points from Gamma (0,0) to X (0.5,0) to M (0.5,0.5).
    """
    # Lattice constant = 1 (our unit cell is 1x1 μm)
    a = 1.0
    r = 0.2          # rod radius
    eps_rod = 12     # rod permittivity (silicon-like)

    # The unit cell is 1x1 μm in x and y.
    # We use PML in y and periodic (Bloch) in x via k_point.
    sy = 10.0        # cell height (wider than unit cell for better convergence)
    dpml = 1.0       # PML thickness in y

    cell = mp.Vector3(a, sy, 0)   # a=1 in x, sy in y, 2D

    # One rod per unit cell, centered
    geometry = [
        mp.Cylinder(
            radius=r,
            material=mp.Medium(epsilon=eps_rod),
            center=mp.Vector3(0, 0),
        )
    ]

    # Broadband source to excite all modes
    fcen = 0.25     # center frequency (in units of c/a)
    df = 1.5        # very wide bandwidth to excite many bands

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Hz,      # TE polarization for this geometry
            center=mp.Vector3(0.1234, 0),   # slightly off-center to avoid symmetry
        )
    ]

    # Symmetry: the mode we look at is antisymmetric in y
    sym = mp.Mirror(direction=mp.Y, phase=-1)

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        symmetries=[sym],
        boundary_layers=[mp.PML(dpml, direction=mp.Y)],
        resolution=20,
    )

    # k-points along Gamma-X-M-Gamma path (common for square lattice)
    # Gamma = (0,0), X = (0.5,0), M = (0.5,0.5)
    k_points_Gamma_X = mp.interpolate(
        num_k_points,
        [mp.Vector3(0, 0, 0), mp.Vector3(0.5, 0, 0)]
    )

    print("Computing band structure along Gamma -> X...")
    print("This may take a few minutes...")

    # run_k_points runs the simulation at each k-point and uses Harminv
    # to extract the resonant frequencies. Returns a list of lists of freqs.
    all_freqs = sim.run_k_points(
        300,           # run for 300 time units at each k-point
        k_points_Gamma_X
    )

    return k_points_Gamma_X, all_freqs


if __name__ == "__main__":
    k_points, all_freqs = compute_bands(num_k_points=15)

    # Plot the band structure
    plt.figure(figsize=(7, 6))

    kx_vals = [k.x for k in k_points]

    for i, (kx, freqs) in enumerate(zip(kx_vals, all_freqs)):
        for mode in freqs:
            # Each mode is a complex number; real part is frequency,
            # imaginary part indicates decay (leaky vs guided modes)
            if abs(mode.imag) < 0.1 * abs(mode.real):  # only well-defined modes
                plt.plot(kx, mode.real, "b.", markersize=3)

    plt.xlabel("k_x (in units of 2π/a)")
    plt.ylabel("Frequency (in units of c/a)")
    plt.title("Band Structure: 2D Square Lattice of Dielectric Rods\n"
              f"(r=0.2a, epsilon=12) — Gamma to X")
    plt.xlim([0, 0.5])
    plt.ylim([0, 0.8])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("uc7_photonic_crystal_bands.png", dpi=150)
    plt.show()

    print("\nDone. Look for gaps (frequency ranges with no bands) in the plot.")
    print("These are photonic bandgaps where light cannot propagate.")
    print("For the TM gap, try using component=mp.Ez and sym phase=+1.")
```

**Expected output**: A plot of band structure along the Gamma-X direction.
You will see bands (groups of dots) at certain frequencies and gaps between
them where no modes exist. For epsilon=12 rods (silicon-like) in air with
radius 0.2a, there is a large TM bandgap (using Ez polarization) and a
smaller TE bandgap. The plot is cleaner when using MPB, which is specifically
designed for band structure calculations, but Meep FDTD gives a useful first
look.

---

### Use Case 8: Eigenmode Source in a Waveguide (2D)

**Physical concept**: Real waveguides support multiple modes with different
profiles and effective indices. An eigenmode source launches exactly one mode
cleanly, without exciting radiation modes or higher-order modes. This is
essential for precise transmission measurements.

**What you will learn**: EigenModeSource, mode profiles, eig_parity, flux
measurement of a specific mode.

```
Simulation layout:

    +--PML--+------------------------------------+--PML--+
    |///////|                                    |///////|
    |/PML///|  asymmetric waveguide (two slabs)  |/PML///|
    |///////| ============== top slab ==========  |///////|
    |/PML///| ======== main waveguide ===========  |/PML///|
    |///////|                                    |///////|
    |/PML///|   eigenmode source (left)          |/PML///|
    +--PML--+------------------------------------+--PML--+
```

```python
# use_case_8_eigenmode_source.py
# Launch a specific waveguide mode using EigenModeSource.
# Requires MPB to be installed (conda install -c conda-forge mpb).

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# An asymmetric waveguide with two slabs (makes mode selection interesting)
cell = mp.Vector3(16, 8, 0)

# Main waveguide slab
wg_main = mp.Block(
    center=mp.Vector3(0, 0),
    size=mp.Vector3(mp.inf, 1, mp.inf),
    material=mp.Medium(epsilon=12),
)
# A thin slab slightly above, creating asymmetry
wg_top = mp.Block(
    center=mp.Vector3(0, 0.55),   # slightly above center
    size=mp.Vector3(mp.inf, 0.1, mp.inf),
    material=mp.Medium(epsilon=4),  # lower epsilon than core
)

geometry = [wg_main, wg_top]
pml_layers = [mp.PML(1.0)]
resolution = 10

# EigenModeSource automatically computes the mode profile using MPB
# and uses it as the source.
# eig_parity=mp.ODD_Z: select modes with odd symmetry in z (TM-like in 2D)
# eig_band=1: select the fundamental (lowest-frequency) mode
# eig_kpoint: initial guess for the mode wavevector direction (+x = right-going)
sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=0.15, width=10),
        center=mp.Vector3(-5, 0),        # source at x=-5
        size=mp.Vector3(0, 6, 0),        # spans ±3 μm in y (covers waveguide)
        eig_parity=mp.ODD_Z + mp.EVEN_Y, # TM fundamental mode symmetry
        eig_band=1,                       # fundamental mode (1-indexed)
    )
]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution,
    force_complex_fields=True,  # needed for time-averaged flux
)

# Add a flux monitor to measure power in the waveguide
flux_region = mp.FluxRegion(
    center=mp.Vector3(6, 0),
    size=mp.Vector3(0, 6, 0),
)
flux_monitor = sim.add_flux(0.15, 0, 1, flux_region)

sim.run(until=200)

# Get the total power
total_flux = mp.get_fluxes(flux_monitor)[0]
print(f"Total transmitted flux at x=+6: {total_flux:.6f}")

# Visualize the fields
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary",
           origin="lower", extent=[-8, 8, -4, 4])
plt.title("Waveguide geometry (epsilon)")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.colorbar(label="epsilon")

plt.subplot(1, 2, 2)
plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary",
           origin="lower", extent=[-8, 8, -4, 4], alpha=0.3)
plt.imshow(np.real(ez_data).transpose(), interpolation="spline36", cmap="RdBu",
           origin="lower", extent=[-8, 8, -4, 4])
plt.title("Ez field — eigenmode source")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")

plt.tight_layout()
plt.savefig("uc8_eigenmode_source.png", dpi=150)
plt.show()

print("\nNote: EigenModeSource requires MPB. If you get an error,")
print("install MPB with: conda install -c conda-forge mpb")
```

**Expected output**: The field plot should show a very clean guided mode
propagating in the +x direction. Unlike a point source or simple line source,
the eigenmode source produces essentially no radiation modes or reflections at
the source plane. The mode profile in y matches the theoretical waveguide mode
profile.

---

### Use Case 9: Near-to-Far Field Transformation (2D)

**Physical concept**: A radiating structure (an antenna or an aperture) emits
waves that we can only measure near the source in the simulation (the domain
is finite). Near-to-far-field transformation uses Huygens' principle to
mathematically extrapolate the far-field radiation pattern from near-field data.

**What you will learn**: `add_near2far`, `Near2FarRegion`, `get_farfield`,
radiation pattern plotting.

```
Simulation layout:

    +--PML--+---------------------------+--PML--+
    |///////|                           |///////|
    |/PML///|  +-----near2far box-----+ |/PML///|
    |///////|  |                      | |///////|
    |/PML///|  |      *dipole*        | |/PML///|   ---> far field is computed
    |///////|  |       source         | |///////|        outside this box
    |/PML///|  |                      | |/PML///|
    |///////|  +----------------------+ |///////|
    +--PML--+---------------------------+--PML--+
```

```python
# use_case_9_near_to_far.py
# Compute the far-field radiation pattern of a dipole antenna.

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import math

# Simulation parameters
wavelength = 1.0      # wavelength in μm
fcen = 1.0 / wavelength
cell_size_um = 4.0    # interior cell size (before PML)
pml_um = 1.0

sxy = pml_um + cell_size_um + pml_um
cell = mp.Vector3(sxy, sxy, 0)
pml_layers = [mp.PML(pml_um)]

# Point dipole source at the center, Ez polarization
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
        center=mp.Vector3(0, 0),
        component=mp.Ez,
    )
]

# Mirror symmetries (Ez dipole is symmetric in both x and y)
symmetries = [
    mp.Mirror(mp.X, phase=+1),
    mp.Mirror(mp.Y, phase=+1),
]

sim = mp.Simulation(
    resolution=50,
    cell_size=cell,
    boundary_layers=pml_layers,
    sources=sources,
    symmetries=symmetries,
)

# Near-to-far region: a box of 4 line segments surrounding the dipole.
# The box must be inside the PML and away from the source.
# weight=-1 on two sides accounts for the outward normal direction.
n2f_box = sim.add_near2far(
    fcen, 0, 1,   # single frequency (df=0, nfreq=1)
    mp.Near2FarRegion(
        center=mp.Vector3(0, +0.5 * cell_size_um),
        size=mp.Vector3(cell_size_um, 0),
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(0, -0.5 * cell_size_um),
        size=mp.Vector3(cell_size_um, 0),
        weight=-1,   # outward normal points in -y
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(+0.5 * cell_size_um, 0),
        size=mp.Vector3(0, cell_size_um),
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(-0.5 * cell_size_um, 0),
        size=mp.Vector3(0, cell_size_um),
        weight=-1,   # outward normal points in -x
    ),
)

# Run until fields decay
sim.run(until_after_sources=mp.stop_when_dft_decayed())

print("Simulation complete. Computing far fields...")

# Sample the far field at 100 points on a circle of radius 1000 * wavelength.
far_radius = 1000 * wavelength
num_angles = 360
angles = np.linspace(0, 2 * math.pi, num_angles, endpoint=False)

# For each angle, compute the far field at that direction.
# get_farfield returns [Ex, Ey, Ez, Hx, Hy, Hz] at the requested point.
far_Ez = []
for angle in angles:
    pt = mp.Vector3(
        far_radius * math.cos(angle),
        far_radius * math.sin(angle),
        0
    )
    ff = sim.get_farfield(n2f_box, pt)
    # ff is a list: [Ex, Ey, Ez, Hx, Hy, Hz]
    far_Ez.append(abs(ff[2]) ** 2)   # |Ez|^2 is proportional to radiated intensity

far_Ez = np.array(far_Ez)
far_Ez_normalized = far_Ez / max(far_Ez)   # normalize to peak = 1

# Plot polar radiation pattern
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Polar plot
ax1 = plt.subplot(1, 2, 1, projection="polar")
ax1.plot(angles, far_Ez_normalized, "b-", linewidth=1.5, label="Meep")
ax1.set_title("Radiation Pattern (Ez dipole)")
ax1.set_rticks([0, 0.5, 1.0])
ax1.grid(True)

# Theory for Ez dipole: uniform in xy plane (isotropic in 2D)
theory = np.ones_like(angles)
ax1.plot(angles, theory, "r--", linewidth=1.5, label="Theory (uniform)")
ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

# Cartesian plot for comparison
ax2 = plt.subplot(1, 2, 2)
ax2.plot(np.degrees(angles), far_Ez_normalized, "b-", label="Meep |Ez|^2")
ax2.axhline(1.0, color="r", linestyle="--", label="Theory (uniform)")
ax2.set_xlabel("Angle (degrees)")
ax2.set_ylabel("Normalized radiated power")
ax2.set_title("Far Field Intensity vs Angle")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("uc9_radiation_pattern.png", dpi=150)
plt.show()

print("An Ez dipole (out-of-plane) in 2D radiates uniformly in all directions.")
print("This is confirmed by the nearly circular radiation pattern.")
```

**Expected output**: A nearly circular polar radiation pattern (an Ez dipole
in 2D is isotropic — it radiates equally in all directions in the plane).
The Meep result should match the theoretical uniform circle. Small deviations
near the axes may appear due to the finite simulation cell and symmetry.

---

### Use Case 10: Visualization and Plotting (2D)

**Physical concept**: Understanding your simulation requires good visualization.
Meep provides built-in tools for plotting geometry, field snapshots, and
animations.

**What you will learn**: `plot2D`, epsilon visualization, field snapshots,
DFT fields, making animations.

```python
# use_case_10_visualization.py
# Comprehensive visualization techniques for Meep simulations.

import meep as mp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------------------------------------------------
# Set up a simple waveguide simulation to demonstrate visualization.
# -----------------------------------------------------------------------
cell = mp.Vector3(20, 10, 0)
dpml = 1.0
pml_layers = [mp.PML(dpml)]
resolution = 15

geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, 1, mp.inf),
        center=mp.Vector3(0, 0),
        material=mp.Medium(epsilon=12),
    )
]

fcen = 0.15
df = 0.05

sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(-8, 0),
        size=mp.Vector3(0, 1.5),
    )
]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# -----------------------------------------------------------------------
# Technique 1: plot2D — built-in geometry + field visualization.
# Call before running to see just the geometry.
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
sim.plot2D(ax=ax)
ax.set_title("Geometry visualization (before simulation)")
plt.tight_layout()
plt.savefig("uc10_geometry.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------
# Technique 2: Add DFT field monitor (records fields at specific frequency).
# Unlike time-domain snapshots, DFT fields give steady-state at exactly fcen.
# -----------------------------------------------------------------------
dft_mon = sim.add_dft_fields(
    [mp.Ez],          # record Ez field
    fcen,             # at this frequency
    0, 1,             # df=0, nfreq=1 (single frequency)
    center=mp.Vector3(),
    size=cell,
)

# -----------------------------------------------------------------------
# Technique 3: Collect field snapshots during the run for animation.
# -----------------------------------------------------------------------
field_frames = []       # will store Ez arrays at each step

def capture_frame(sim):
    ez = sim.get_array(
        center=mp.Vector3(),
        size=cell,
        component=mp.Ez,
    )
    field_frames.append(ez.copy())

# Run and capture a frame every 1 time unit
sim.run(
    mp.at_every(1.0, capture_frame),
    until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez,
                                                    mp.Vector3(8, 0), 1e-3),
)

print(f"Captured {len(field_frames)} frames.")

# -----------------------------------------------------------------------
# Technique 4: Plot the DFT field (complex, so take real or magnitude).
# -----------------------------------------------------------------------
dft_ez = sim.get_dft_array(dft_mon, mp.Ez, 0)  # index 0 = first frequency
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Geometry
axes[0].imshow(eps_data.transpose(), cmap="binary", origin="lower",
               extent=[-10, 10, -5, 5])
axes[0].set_title("Geometry (epsilon)")
axes[0].set_xlabel("x (μm)")
axes[0].set_ylabel("y (μm)")

# DFT field: real part (phase profile)
im1 = axes[1].imshow(np.real(dft_ez).transpose(), cmap="RdBu",
                     origin="lower", extent=[-10, 10, -5, 5])
axes[1].contour(eps_data.transpose() > 1, extent=[-10, 10, -5, 5],
                colors="k", linewidths=0.5)  # overlay waveguide outline
axes[1].set_title("DFT Ez (real part, steady state)")
axes[1].set_xlabel("x (μm)")
plt.colorbar(im1, ax=axes[1])

# DFT field: magnitude (intensity)
im2 = axes[2].imshow(np.abs(dft_ez).transpose(), cmap="hot",
                     origin="lower", extent=[-10, 10, -5, 5])
axes[2].contour(eps_data.transpose() > 1, extent=[-10, 10, -5, 5],
                colors="w", linewidths=0.5)
axes[2].set_title("|DFT Ez| (field magnitude)")
axes[2].set_xlabel("x (μm)")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("uc10_dft_fields.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------
# Technique 5: Create an animation from the captured frames.
# -----------------------------------------------------------------------
print("Creating animation...")

# Find the color scale from all frames
all_max = max(abs(f).max() for f in field_frames)

fig_anim, ax_anim = plt.subplots(figsize=(10, 5))

# Show geometry as background
ax_anim.imshow(eps_data.transpose(), cmap="binary", origin="lower",
               extent=[-10, 10, -5, 5], alpha=0.3)

# Initial frame
im_anim = ax_anim.imshow(
    field_frames[0].transpose(),
    cmap="RdBu",
    vmin=-all_max, vmax=all_max,
    origin="lower",
    extent=[-10, 10, -5, 5],
    animated=True,
    alpha=0.9,
)
ax_anim.set_title("Ez field evolution")
ax_anim.set_xlabel("x (μm)")
ax_anim.set_ylabel("y (μm)")

def update_frame(frame_idx):
    im_anim.set_array(field_frames[frame_idx].transpose())
    ax_anim.set_title(f"Ez field (t = {frame_idx:.0f} μm/c)")
    return [im_anim]

# Use every 5th frame to keep animation manageable
stride = max(1, len(field_frames) // 60)
frame_indices = range(0, len(field_frames), stride)

anim = animation.FuncAnimation(
    fig_anim,
    update_frame,
    frames=frame_indices,
    interval=50,        # 50 ms between frames
    blit=True,
)

# Save as GIF (requires Pillow: pip install Pillow)
try:
    anim.save("uc10_animation.gif", writer="pillow", fps=20)
    print("Animation saved to uc10_animation.gif")
except Exception as e:
    print(f"Could not save animation: {e}")
    print("Install Pillow with: pip install Pillow")

plt.show()

# -----------------------------------------------------------------------
# Technique 6: plot2D after simulation shows fields.
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
sim.plot2D(
    ax=ax,
    fields=mp.Ez,        # overlay Ez field
    field_parameters={
        "alpha": 0.9,
        "cmap": "RdBu",
    },
    eps_parameters={
        "cmap": "binary",
        "alpha": 0.3,
    },
)
ax.set_title("plot2D: geometry + Ez field")
plt.tight_layout()
plt.savefig("uc10_plot2d.png", dpi=150)
plt.show()
```

**Expected output**: Five output files:
- `uc10_geometry.png`: The simulation cell with waveguide visible as a gray bar.
- `uc10_dft_fields.png`: Three panels showing geometry, the DFT Ez real part
  (a standing wave pattern), and the DFT magnitude (showing the guided mode
  profile cleanly).
- `uc10_animation.gif`: An animated GIF showing the Gaussian pulse traveling
  through the waveguide.
- `uc10_plot2d.png`: The all-in-one plot2D output.

---

## 6. Common Patterns and Recipes

### Normalizing Flux (The Two-Run Technique)

To measure transmittance (fraction of incident power transmitted), you always
need a reference measurement. The standard procedure:

```python
# Run 1: measure incident flux without your structure
sim_reference = mp.Simulation(
    cell_size=cell,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution,
    # NO geometry
)
tran_monitor = sim_reference.add_flux(fcen, df, nfreq, flux_region)
refl_monitor = sim_reference.add_flux(fcen, df, nfreq, refl_region)
sim_reference.run(until_after_sources=mp.stop_when_fields_decayed(50, comp, pt, 1e-3))

# Save incident data for reflection subtraction
incident_refl_data = sim_reference.get_flux_data(refl_monitor)
incident_flux = mp.get_fluxes(tran_monitor)  # this is your denominator

# Run 2: with your structure
sim_structure = mp.Simulation(
    cell_size=cell,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution,
    geometry=geometry,   # now with structure
)
tran_monitor2 = sim_structure.add_flux(fcen, df, nfreq, flux_region)
refl_monitor2 = sim_structure.add_flux(fcen, df, nfreq, refl_region)

# Load negated incident data into reflection monitor to isolate reflected wave
sim_structure.load_minus_flux_data(refl_monitor2, incident_refl_data)
sim_structure.run(until_after_sources=mp.stop_when_fields_decayed(50, comp, pt, 1e-3))

transmitted_flux = mp.get_fluxes(tran_monitor2)
reflected_flux = mp.get_fluxes(refl_monitor2)

# Transmittance and reflectance at each frequency
T = [t / i for t, i in zip(transmitted_flux, incident_flux)]
R = [-r / i for r, i in zip(reflected_flux, incident_flux)]
```

### Sweeping Parameters

To sweep a geometric or material parameter, wrap your simulation in a function:

```python
import meep as mp
import numpy as np

def compute_transmission(waveguide_width):
    """
    Returns transmittance through a waveguide as a function of its width.
    """
    cell = mp.Vector3(20, 10, 0)
    fcen = 0.15
    df = 0.05

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, waveguide_width, mp.inf),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=12),
        )
    ]

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-8, 0),
            size=mp.Vector3(0, waveguide_width * 1.5),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(1.0)],
        geometry=geometry,
        sources=sources,
        resolution=10,
    )

    mon = sim.add_flux(
        fcen, df, 1,
        mp.FluxRegion(center=mp.Vector3(8, 0), size=mp.Vector3(0, waveguide_width * 2))
    )
    sim.run(until_after_sources=100)
    return mp.get_fluxes(mon)[0]

# Sweep waveguide widths
widths = np.linspace(0.5, 3.0, 10)
transmissions = [compute_transmission(w) for w in widths]

import matplotlib.pyplot as plt
plt.plot(widths, transmissions, "bo-")
plt.xlabel("Waveguide width (μm)")
plt.ylabel("Transmitted flux")
plt.title("Transmission vs waveguide width")
plt.grid(True)
plt.savefig("parameter_sweep.png", dpi=150)
plt.show()
```

### Saving and Loading Simulation State

For long simulations, you can checkpoint and resume:

```python
import meep as mp

# Save fields at a certain time
sim.run(until=100)
sim.dump_fields("checkpoint_fields")      # saves field state to files
sim.dump_structure("checkpoint_structure") # saves structure

# Later, restore and continue
sim2 = mp.Simulation(
    cell_size=cell,
    sources=sources,
    boundary_layers=pml_layers,
    geometry=geometry,
    resolution=resolution,
)
sim2.load_structure("checkpoint_structure")
sim2.load_fields("checkpoint_fields")
sim2.run(until=200)   # continue from where you left off
```

### Using Symmetry to Speed Up Simulations

Symmetry can halve (or quarter) your simulation domain:

```python
import meep as mp

# If your structure and source are symmetric about y=0,
# and the Ez field is symmetric (even in y), add:
symmetries = [mp.Mirror(mp.Y)]

# If the Ez field is antisymmetric (odd in y), use:
# symmetries = [mp.Mirror(mp.Y, phase=-1)]

# For a structure symmetric in both x and y:
symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y)]

sim = mp.Simulation(
    ...
    symmetries=symmetries,
)
```

**Rules for symmetry**:
- The geometry must be symmetric about the mirror plane.
- The source must also be symmetric (or antisymmetric) about the mirror plane.
- Use `phase=+1` for fields that are even (symmetric) across the mirror.
- Use `phase=-1` for fields that are odd (antisymmetric) across the mirror.
- For TM (Ez) polarization: Ez is even if the source is symmetric and even.
- Symmetry cuts memory and time roughly in half per dimension you exploit.

### Running in Parallel with MPI

If you installed the MPI version of Meep, you can run simulations on multiple
cores:

```bash
# Run on 4 cores
mpirun -np 4 python my_simulation.py

# On a cluster with SLURM:
# #SBATCH --ntasks=32
mpirun -np 32 python my_simulation.py
```

In your Python script, use `mp.am_master()` to only execute code on the master
process (for output and plotting):

```python
import meep as mp

sim.run(until=200)

if mp.am_master():
    # Only master process does plotting/output
    import matplotlib.pyplot as plt
    data = sim.get_array(...)
    plt.figure()
    plt.imshow(data.transpose())
    plt.savefig("output.png")
```

**Performance tips for parallel runs**:
- Meep parallelizes by splitting the simulation domain across processes.
- Good scaling up to roughly 1 process per 10,000 grid cells.
- For a 3D simulation with 100 x 100 x 100 grid = 1M cells, use up to ~100 cores.
- The `--with-openmp` build also supports threading within a single node.

---

## 7. Troubleshooting

### "My simulation is slow"

**Check resolution**: High resolution is the #1 cause of slow simulations.
Start with resolution=10 to verify your setup is correct, then increase to the
minimum resolution that gives converged results. For 3D simulations, tripling
the resolution multiplies cost by 3^4 = 81x (3 spatial dimensions plus time).

**Use symmetry**: Every mirror symmetry you exploit halves the simulation cost.
A structure symmetric in x and y can run 4x faster with two Mirror symmetries.

**Reduce the cell size**: Add only as much padding as needed (half-wavelength
between your structure and the PML edge is sufficient).

**Use MPI**: If you have multiple cores available, parallel Meep is the easiest
speedup. See Section 6 for details.

**Use 2D when possible**: A 2D approximation of a 3D structure can be 100x-
1000x faster and is often sufficient for qualitative understanding.

### "Fields blow up / simulation goes unstable"

**Check PML placement**: Sources should not be inside or touching PML. Keep
sources at least 1 wavelength away from the PML edge.

**Check material parameters**: Metals (negative epsilon at some frequencies)
and gain media can cause instabilities if not handled carefully. Use
`mp.Medium(epsilon=..., D_conductivity=...)` for lossy materials instead of
manually setting negative epsilon without the corresponding imaginary part.

**Check resolution**: Too-low resolution near sharp material interfaces can
cause instabilities. Try increasing resolution.

**Check Courant condition**: Meep chooses the Courant factor automatically.
If you use a custom `dt` that violates Courant stability, fields will diverge.
Do not set `dt` manually unless you know exactly what you are doing.

**Check for grazing-angle sources**: A plane wave at nearly-parallel incidence
to a PML edge is poorly absorbed. Use `k_point=mp.Vector3()` for normal-
incidence plane waves, and avoid very oblique angles with thick PML regions.

### "Results don't match theory"

**Check units**: The most common beginner mistake. Frequency in Meep is
`1/wavelength_in_microns`. If your wavelength is 1550 nm = 1.55 μm, then
`fcen = 1/1.55 ≈ 0.645`.

**Check resolution convergence**: Run at resolution 10, 20, and 40 and see
if results are converging. FDTD accuracy is O(dx^2), so doubling resolution
halves the error.

**Check PML thickness**: Increase the PML to 1-2 wavelengths if you see
unexpected reflections. For broadband simulations, PML must be thick enough
for all frequencies in the bandwidth.

**Check run time**: For Gaussian pulse simulations, run long enough for the
fields to fully decay. Use `until_after_sources=mp.stop_when_fields_decayed`
with threshold 1e-5 or smaller for precise flux measurements.

**Check monitor placement**: Flux monitors should be at least a few pixels
away from material interfaces where the field interpolation may be inaccurate.

### Common Error Messages

**`ImportError: No module named 'meep'`**
- Your conda environment is not activated. Run `conda activate mp`.
- Or Meep is not installed in the current environment.

**`AttributeError: module 'meep' has no attribute 'materials'`**
- The materials library is a separate module. Import it with
  `import meep.materials as mat` or `from meep import materials as mat`.

**`RuntimeError: MPB not available`**
- `EigenModeSource` requires MPB. Install it with
  `conda install -c conda-forge mpb`.

**`ValueError: PML extends into simulation cell`**
- Your cell is too small. The PML thickness plus the structure size exceeds
  the cell size. Increase `cell_size` or decrease `dpml`.

**`Warning: Courant condition is not satisfied`**
- The time step is too large for your resolution. This usually happens with
  very high-epsilon materials or very small geometry features.
  Try reducing the Courant number: `mp.Simulation(..., Courant=0.4)`.

**Simulation hangs with no output**
- Add `mp.verbosity(1)` at the top of your script to enable progress output.
- Check that `meep_time()` is advancing. If not, the simulation may be
  paused at a callback.

**MPI run fails with `mpirun: command not found`**
- You are not using the MPI version of Meep. Install with:
  `conda create -n mpimpi -c conda-forge pymeep=*=mpi_mpich*`

**`ModuleNotFoundError: No module named 'parameterized'`**
- Some tests require the `parameterized` package which is not included in
  the conda-forge pymeep installation. Fix: `pip install parameterized`.
  This affects 10 test files. See `guides/TEST_REPORT.md` for the full list.

**`AttributeError: module 'numpy' has no attribute 'trapz'`**
- NumPy 2.0 renamed `np.trapz` to `np.trapezoid`. If you see this in
  `antenna-radiation.py`, replace `np.trapz(...)` with `np.trapezoid(...)`.

**`AttributeError: np.complex_ was removed in the NumPy 2.0 release`**
- NumPy 2.0 removed `np.complex_`. Replace with `np.complex128`.
  This affects `solve-cw.py`.

**`FileNotFoundError: No such file or directory: 'h5topng'`**
- The `h5topng` utility converts HDF5 field output to PNG images.
  Install it with: `sudo apt install h5utils` (Ubuntu/Debian) or
  `brew install h5utils` (macOS). Affects `cherenkov-radiation.py`
  and `wvg-src.py`.

---

## 8. Where to Go Next

### Official Documentation

The Meep documentation is comprehensive and includes many worked examples:
https://meep.readthedocs.io

Key sections to read next:
- **Python Tutorials**: Step-by-step tutorials covering waveguides, photonic
  crystals, resonators, and more
- **Python User Interface**: Complete reference for all Python classes and
  functions
- **Exploiting Symmetry**: How to use symmetry correctly
- **Parallel Meep**: How to set up MPI runs effectively

### Examples in This Repository

The `python/examples/` directory in this repository contains many complete,
runnable simulation scripts covering advanced topics:

- `bend-flux.py` — Waveguide bend transmission (basis for Use Case 3)
- `ring.py` — Ring resonator with Harminv (basis for Use Case 4)
- `mie_scattering.py` — Mie scattering with PyMieScatt comparison
- `antenna-radiation.py` — Dipole radiation pattern
- `holey-wvg-bands.py` — Photonic crystal waveguide band structure
- `refl-quartz.py` — Reflection from a dielectric slab (Fabry-Perot)
- `oblique-planewave.py` — Oblique plane wave incidence
- `mode-decomposition.py` — Decomposing fields into waveguide modes
- `gaussian-beam.py` — Focused Gaussian beam simulation
- `multilevel-atom.py` — Laser gain simulation (advanced)

> **Note:** Some examples have known issues with NumPy 2.x or require
> additional packages. Before running examples, install the recommended
> extras: `pip install parameterized` and `sudo apt install h5utils`.
> See [guides/TEST_REPORT.md](TEST_REPORT.md) for a complete pass/fail
> table and per-file notes.

### The Adjoint Solver for Inverse Design

Meep's adjoint solver enables gradient-based optimization of photonic devices.
Instead of manually trying different geometries, you define a performance
objective (e.g., maximize transmission at 1550 nm while minimizing it at
1310 nm) and the adjoint solver computes gradients that guide an optimizer
toward the optimal geometry.

This is used for designing compact wavelength demultiplexers, mode converters,
beam splitters, and other integrated photonic components. The adjoint solver
requires JAX:

```bash
conda install -c conda-forge jax
```

Examples are in `python/examples/adjoint_optimization/`.

### Community and Support

- **GitHub Issues**: https://github.com/NanoComp/meep/issues
  Report bugs and ask questions here.
- **GitHub Discussions**: https://github.com/NanoComp/meep/discussions
  Community Q&A and tips.
- **Meep mailing list**: Search the archives for common questions.

### Learning More About FDTD

If you want to understand the numerical method more deeply:

- **"Computational Electrodynamics: The Finite-Difference Time-Domain Method"**
  by Allen Taflove — the standard reference textbook on FDTD.
- **"Photonic Crystals: Molding the Flow of Light"** by Joannopoulos et al. —
  freely available at http://ab-initio.mit.edu/book/ — covers the photonics
  physics that Meep is designed to simulate.

---

*This guide was written for Meep users who want to run simulations, not develop
the software. For build instructions and developer information, see
`guides/DEVELOPER_GUIDE.md`. For test results and known issues, see
`guides/TEST_REPORT.md`. For the full API reference, see
https://meep.readthedocs.io.*
