# Quick Start Explained: 2D Dielectric Waveguide Simulation

This document walks through the Quick Start example from the README, explaining the underlying physics, how the simulation is configured, and how to interpret the results.

---

## Table of Contents

1. [What Are We Simulating?](#1-what-are-we-simulating)
2. [The Physics Behind It](#2-the-physics-behind-it)
3. [The Simulation Code, Line by Line](#3-the-simulation-code-line-by-line)
4. [What the Output Means](#4-what-the-output-means)
5. [Key Numbers and Why They Matter](#5-key-numbers-and-why-they-matter)
6. [What to Try Next](#6-what-to-try-next)

---

## 1. What Are We Simulating?

We are simulating a **slab dielectric waveguide** in two dimensions. Think of it as a thin strip of glass (or silicon) surrounded by air. When light enters one end, it bounces back and forth inside the strip via total internal reflection and propagates along the strip without escaping. This is the same principle behind optical fibers and on-chip photonic interconnects.

```
              Air (epsilon = 1)
    ┌──────────────────────────────────────┐
    │                                      │
    │  ┌──────────────────────────────────┐│
    │  │  Waveguide (epsilon = 12)        ││  ← Light travels this way →
    │  └──────────────────────────────────┘│
    │                                      │
    └──────────────────────────────────────┘
              Air (epsilon = 1)
```

In our example the "glass" is a material with dielectric constant epsilon = 12 (roughly silicon at infrared wavelengths, refractive index n ~ 3.46). The strip is 1 micrometer wide, centered in a 16 x 8 micrometer simulation cell. A short pulse of light is injected at the left end and we watch it travel to the right.

---

## 2. The Physics Behind It

### 2.1 Maxwell's Equations and FDTD

All electromagnetic wave behavior (light, radio waves, microwaves) is governed by Maxwell's equations. Meep solves these equations numerically using the **Finite-Difference Time-Domain (FDTD)** method:

1. Space is divided into a grid of tiny cells.
2. The electric field (E) and magnetic field (H) are computed at alternating half-steps in time (the "leapfrog" scheme).
3. At each time step, E is updated from the previous H, then H is updated from the new E.
4. This process repeats, and waves naturally emerge, propagate, reflect, and interfere.

The key insight of FDTD is that we do not need to solve anything analytically. We just apply the update rules at every grid point and let the physics unfold in time.

### 2.2 Total Internal Reflection and Waveguiding

When light travels from a high-refractive-index material (n = 3.46) into a low-index material (n = 1.0, air), it can be totally reflected if it strikes the interface at a shallow angle. This is **total internal reflection** (TIR). In our waveguide, the light bounces between the top and bottom surfaces of the strip, but the forward component of its propagation is unimpeded. The net effect: light is confined to the strip and guided along it.

For a slab of width `d = 1 micrometer` and index `n = 3.46`, the number of guided modes depends on the frequency. At our center frequency of 0.15 (wavelength ~ 6.67 micrometer), the waveguide supports one or two modes.

### 2.3 Gaussian Pulse Source

Rather than injecting a single-frequency continuous wave, we use a **Gaussian pulse**: a burst of light whose frequency content is centered at 0.15 (in Meep units) with a frequency width of 0.1. This means the pulse contains a range of frequencies, which is useful for broadband analysis (e.g., computing transmission spectra). The pulse is short in time, so we can watch its leading edge travel through the waveguide.

### 2.4 PML Absorbing Boundaries

In a real experiment, light that reaches the edge of the waveguide would continue propagating into free space forever. In a computer simulation, we must truncate the computational domain. If we simply placed a hard wall at the boundary, light would reflect back and contaminate the simulation.

The solution is a **Perfectly Matched Layer (PML)**: a special absorbing region placed around the edges of the simulation. The PML is designed to absorb outgoing waves with virtually no reflection, regardless of frequency or angle of incidence. It mimics an infinite open space surrounding the simulation.

### 2.5 Meep's Unit System

Meep uses a **scale-invariant unit system** where the speed of light c = 1. This means:

| Quantity | Meep Units | Interpretation |
|---|---|---|
| Length | 1 | 1 micrometer (by convention in this example) |
| Time | 1 | 1 micrometer / c ~ 3.34 femtoseconds |
| Frequency | 0.15 | c / wavelength = 1 / 6.67 micrometer |

You choose the length scale. If you decide 1 Meep unit = 1 micrometer, then a frequency of 0.15 means a free-space wavelength of 1/0.15 = 6.67 micrometers (mid-infrared). All other units follow from c = 1.

---

## 3. The Simulation Code, Line by Line

Here is the complete Quick Start script with detailed annotations.

### 3.1 Imports

```python
import meep as mp
import matplotlib.pyplot as plt
```

`meep` is the simulation engine. `matplotlib` is used to plot the results.

### 3.2 Simulation Cell

```python
cell = mp.Vector3(16, 8)
```

This defines the computational domain: a 16 x 8 rectangle (in Meep length units). Since the z-component is 0 (default), this is a **2D simulation** (infinite and uniform in z). The cell is centered at the origin, so it spans x = [-8, +8] and y = [-4, +4].

### 3.3 PML Boundaries

```python
pml_layers = [mp.PML(1.0)]
```

A 1.0-unit-thick PML is placed on all four sides of the cell. This means the "usable" simulation region is actually 14 x 6 (the inner area after subtracting PML from each side). Any wave reaching the PML is absorbed.

### 3.4 Geometry (The Waveguide)

```python
geometry = [
    mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
             center=mp.Vector3(),
             material=mp.Medium(epsilon=12))
]
```

This creates a rectangular block:
- **size**: infinitely long in x and z, 1 unit tall in y. `mp.inf` means the block extends to the edges of the simulation cell in that direction.
- **center**: at the origin (0, 0, 0), so the strip spans y = [-0.5, +0.5].
- **material**: dielectric with epsilon = 12. Everything outside this block is vacuum (epsilon = 1) by default.

### 3.5 Source

```python
sources = [
    mp.Source(mp.GaussianSource(frequency=0.15, fwidth=0.1),
              component=mp.Hz,
              center=mp.Vector3(-7))
]
```

- **GaussianSource**: A pulse with center frequency 0.15 and frequency width 0.1 (the 1/e half-width of the Gaussian envelope in frequency space).
- **component=mp.Hz**: We excite the Hz (out-of-plane magnetic) field. In 2D, this selects the **TM (transverse magnetic) polarization**, which has field components Hz, Ex, and Ey.
- **center=mp.Vector3(-7)**: The source is a point at x = -7, y = 0 (near the left edge, just inside the PML).

Why Hz? In 2D FDTD, electromagnetic fields split into two independent polarizations: TE (Ez, Hx, Hy) and TM (Hz, Ex, Ey). By exciting Hz, we get a clean TM simulation.

### 3.6 Simulation Object

```python
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=10)
```

- **resolution=10**: 10 grid points per Meep length unit. Since our length unit is 1 micrometer, this gives a grid spacing of 0.1 micrometer. The total grid is 160 x 80 = 12,800 cells.

Resolution rule of thumb: you want at least 8-10 pixels per wavelength *inside the material*. The wavelength inside the waveguide is:

```
lambda_material = lambda_vacuum / n = 6.67 / 3.46 ~ 1.93 micrometer
```

At resolution 10, that is about 19 pixels per wavelength inside the material -- more than sufficient.

### 3.7 Running the Simulation

```python
sim.run(until=100)
```

Run for 100 Meep time units. Since each time step is `dt = 1/(2 * resolution) = 0.05`, this is 2000 time steps. In physical time: 100 x 3.34 fs = 334 femtoseconds (about a third of a picosecond).

During this time, the Gaussian pulse is emitted, travels through the waveguide at the group velocity, and the leading edge reaches approximately x = +7 (the right PML boundary).

### 3.8 Plotting the Result

```python
sim.plot2D(fields=mp.Hz)
plt.savefig("waveguide_hz.png", dpi=150)
print("Field plot saved to waveguide_hz.png")
```

`plot2D` creates a snapshot of the Hz field at the final time step (t = 100). The plot is saved as a PNG image.

---

## 4. What the Output Means

### 4.1 Console Output

When you run the simulation, Meep prints:

```
-----------
Initializing structure...
time for choose_chunkdivision = 0.000377 s
Working in 2D dimensions.
Computational cell is 16 x 8 x 0 with resolution 10
     block, center = (0,0,0)
          size (1e+20,1,1e+20)
          axes (1,0,0), (0,1,0), (0,0,1)
          dielectric constant epsilon diagonal = (12,12,12)
time for set_epsilon = 0.016 s
-----------
run 0 finished at t = 100.0 (2000 timesteps)
Field plot saved to waveguide_hz.png
Elapsed run time = 0.54 s
```

Key information:
- **Working in 2D dimensions**: Confirmed 2D simulation.
- **Computational cell is 16 x 8 x 0 with resolution 10**: 160 x 80 grid.
- **2000 timesteps**: Time step dt = 0.05, run until t = 100.
- **Elapsed run time = 0.54 s**: Half a second on a modern laptop.

### 4.2 The Field Plot

The output image (`waveguide_hz.png`) shows:

```
    ┌─────────────────────────────────────────────┐
    │ ╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲ │ ← PML (green hatching)
    │ ╲                                          ╲ │
    │ ╲           Air (white)                    ╲ │
    │ ╲                                          ╲ │
    │ ╲  ●━━━━━━━━━━━━━━━[🔴🔵🔴🔵🔴]━━━━━━━━━  ╲ │ ← Waveguide (gray bar)
    │ ╲  source        field pattern             ╲ │    with Hz field
    │ ╲                                          ╲ │
    │ ╲           Air (white)                    ╲ │
    │ ╲                                          ╲ │
    │ ╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲ │ ← PML (green hatching)
    └─────────────────────────────────────────────┘
```

**What each element represents:**

| Visual Element | Meaning |
|---|---|
| Green hatched border | PML absorbing boundary layers (1 micrometer thick on each side) |
| Gray horizontal band (y = -0.5 to +0.5) | The dielectric waveguide (epsilon = 12) |
| Red dot at x = -7, y = 0 | The source location |
| Red/blue alternating lobes inside the waveguide | The Hz field at time t = 100 |
| White regions above and below the waveguide | Air (vacuum), with very little field |

**Reading the field pattern:**

- The **red** regions are where Hz is positive (field pointing out of the screen).
- The **blue** regions are where Hz is negative (field pointing into the screen).
- The alternating red-blue pattern is the **oscillating wave** propagating to the right inside the waveguide.
- The field is **concentrated inside the waveguide** (the gray band) with very little leakage into the surrounding air. This confirms that the waveguide is guiding the light.
- The source is at x = -7 but the field pattern is visible mainly from x ~ 0 to x ~ 7. This is because the Gaussian pulse has mostly passed through the left side and its leading edge has reached the right PML.
- The field amplitude is strongest near the right side (leading edge of the pulse) and weaker on the left (trailing edge has partially passed).

### 4.3 Physical Interpretation

The simulation demonstrates:

1. **Light confinement**: The Hz field is almost entirely contained within the 1-micrometer-wide waveguide strip. The high index contrast (n=3.46 vs n=1) provides strong confinement.

2. **Wave propagation**: The pulse travels from left to right. The alternating red/blue lobes represent one complete wavelength of the guided mode. You can estimate the guided wavelength by counting the spacing between consecutive red (or blue) peaks.

3. **PML absorption**: The field near the right boundary is being absorbed by the PML. There is no visible reflection bouncing back from the right edge, confirming the PML is working.

4. **Pulse dispersion**: The pulse shape may appear slightly different from a clean sinusoid because it contains a range of frequencies (broadband Gaussian), and different frequency components travel at slightly different group velocities inside the waveguide (dispersion).

---

## 5. Key Numbers and Why They Matter

| Parameter | Value | Why This Value |
|---|---|---|
| Cell size | 16 x 8 | Large enough to contain the waveguide plus PML, with room for the field to propagate |
| Resolution | 10 | ~19 pixels per wavelength in the material (well above the minimum of ~8) |
| PML thickness | 1.0 | Adequate for the frequency range; thicker PML absorbs better but costs more memory |
| Waveguide width | 1.0 | Supports a small number of guided modes at the source frequency |
| Waveguide epsilon | 12 | Roughly silicon at infrared wavelengths (n ~ 3.46) |
| Source frequency | 0.15 | Free-space wavelength = 6.67 micrometers (mid-infrared) |
| Source fwidth | 0.1 | Broadband pulse covering frequencies 0.05 to 0.25 |
| Run time | 100 | Long enough for the pulse to traverse the cell (~16 micrometers at ~c/3.46) |
| Total grid points | 12,800 | 160 x 80 cells; trivial for a modern computer |
| Time steps | 2,000 | Courant factor dt = 0.05 for stability |
| Wall-clock time | ~0.5 s | Fast enough for interactive exploration |

---

## 6. What to Try Next

Now that the basic simulation works, here are experiments to build intuition:

### 6.1 Change the Waveguide Width

```python
# Narrower waveguide — does the field still stay confined?
mp.Block(size=mp.Vector3(mp.inf, 0.5, mp.inf), ...)

# Wider waveguide — do you see multiple guided modes?
mp.Block(size=mp.Vector3(mp.inf, 3, mp.inf), ...)
```

A very narrow waveguide (below the "cutoff width") will not support any guided mode, and the light will leak out into the air.

### 6.2 Change the Frequency

```python
# Higher frequency (shorter wavelength, more modes)
mp.GaussianSource(frequency=0.5, fwidth=0.2)

# Lower frequency (longer wavelength, fewer modes)
mp.GaussianSource(frequency=0.05, fwidth=0.02)
```

At higher frequencies the waveguide supports more modes and you may see interference patterns.

### 6.3 Increase the Resolution

```python
sim = mp.Simulation(..., resolution=20)  # 20 pixels per unit
```

Higher resolution gives more accurate results but takes longer. Compare the field pattern at resolution 10 vs 20 to assess convergence.

### 6.4 Add a Waveguide Bend

```python
geometry = [
    mp.Block(size=mp.Vector3(10, 1, mp.inf),
             center=mp.Vector3(-3, 0),
             material=mp.Medium(epsilon=12)),
    mp.Block(size=mp.Vector3(1, 10, mp.inf),
             center=mp.Vector3(2, 3),
             material=mp.Medium(epsilon=12)),
]
```

This creates an L-shaped waveguide. Watch how much light makes it around the corner vs how much is radiated away. This is one of the classic Meep tutorials.

### 6.5 Measure Transmission

Instead of just looking at the field, you can measure the power flowing through the waveguide:

```python
# Add a flux monitor before running
tran = sim.add_flux(0.15, 0.1, 50,
                    mp.FluxRegion(center=mp.Vector3(5, 0),
                                  size=mp.Vector3(0, 2)))
sim.run(until=100)

# Print the flux spectrum
freqs = mp.get_flux_freqs(tran)
flux = mp.get_fluxes(tran)
for f, p in zip(freqs, flux):
    print(f"freq={f:.4f}  flux={p:.6f}")
```

This gives you the frequency-resolved power transmission through a cross-section of the waveguide — the starting point for computing transmission spectra, quality factors, and more.

---

## Summary

This Quick Start example demonstrates the complete Meep workflow:

1. **Define the geometry** (what materials, where)
2. **Define the source** (what frequency, what polarization, where)
3. **Set boundary conditions** (PML for open boundaries)
4. **Run the time-stepping loop** (FDTD evolves the fields)
5. **Visualize or analyze the results** (field snapshots, flux spectra, etc.)

Despite being only ~20 lines of Python, this simulation captures real electromagnetic physics: waveguiding by total internal reflection, pulse propagation and dispersion, and absorbing boundary conditions. The same workflow scales to 3D, to complex geometries, to frequency-domain analysis, and to inverse design optimization.

For more tutorials, see the [Meep online manual](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/).
