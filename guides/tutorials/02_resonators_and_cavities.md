# Chapter 2: Resonators and Cavities

This chapter explores optical resonators and electromagnetic cavities using Meep's FDTD engine. We cover the full spectrum of cavity physics: from computing resonant frequencies and quality factors of ring resonators (using both Cartesian and cylindrical coordinate systems), to characterizing photonic crystal defect cavities, to quantifying how a cavity modifies the spontaneous emission rate of an embedded emitter via the Purcell effect and local density of states (LDOS). Far-field radiation patterns, near-to-far-field transformations, and light extraction efficiency from dielectric layers round out the chapter, providing a complete toolkit for designing photonic resonators and quantum-optical emitter-cavity interfaces.

---

### 1. `ring.py` — Ring Resonator Modes in 2D Cartesian Coordinates

**Physics:** Computes the resonant frequencies and quality factors of guided modes in a 2D ring resonator waveguide using harmonic inversion of the time-domain field response.
**Difficulty:** Beginner
**Source:** `python/examples/ring.py`
**Test Status:** PASS (26.5s)

#### Theory

A ring resonator is formed by bending a dielectric waveguide into a closed loop. Guided light circulates around the ring, and resonance occurs when the round-trip accumulated phase is a multiple of 2π. For a ring of mean circumference C with effective modal index n_eff, the resonance condition is:

    m * lambda = n_eff * C,  m = 1, 2, 3, ...

or equivalently in terms of frequency:

    f_m = m * c / (n_eff * C)

Each resonance is characterized by its quality factor Q, which quantifies how long energy is stored before it leaks away. For a resonance at complex frequency f_0 + i*gamma, Q is defined as:

    Q = Re(f_0) / (2 * |Im(f_0)|) = Re(f_0) / (-2 * decay_rate)

High-Q resonances decay slowly in time; in the time domain the field at any fixed monitor point looks like a decaying sinusoid. The FDTD approach to finding resonances exploits this: we excite the cavity with a broadband Gaussian pulse, let the fields ring down, and then use harmonic inversion (Harminv) to extract the complex frequencies of all resonant modes simultaneously.

In the FDTD cell the ring is constructed from two concentric cylinders: a solid dielectric cylinder of radius r+w (the outer wall) with a hollow air cylinder of radius r (the inner air hole). The inner cylinder, defined second, takes precedence in Meep's geometry priority scheme, creating an annular waveguide of width w. The cell is terminated in perfectly matched layers (PML) so that radiation leaking outward is absorbed without reflection.

For a ring with inner radius r = 1 (in Meep units where the lattice constant is 1), outer radius r+w = 2, and refractive index n = 3.4, the lowest whispering-gallery modes occur near a normalized frequency of 0.118, as confirmed by the test suite.

#### Code Walkthrough

The geometry is built using overlapping cylinders, with the inner air cylinder overriding the outer dielectric:

```python
c1 = mp.Cylinder(radius=r + w, material=mp.Medium(index=n))  # dielectric annulus
c2 = mp.Cylinder(radius=r)                                    # air hole (overrides c1)
```

A Gaussian pulse source excites Ez field components. A Y-mirror symmetry halves the computation because the ring is symmetric about the x-axis:

```python
sim = mp.Simulation(
    cell_size=mp.Vector3(sxy, sxy),
    geometry=[c1, c2],
    sources=[src],
    resolution=10,
    symmetries=[mp.Mirror(mp.Y)],
    boundary_layers=[mp.PML(dpml)],
)
```

Resonance frequencies and Q factors are extracted by Harminv, running after the source turns off to avoid contaminating the analysis with the driving pulse:

```python
sim.run(
    mp.at_beginning(mp.output_epsilon),
    mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)),
    until_after_sources=300,
)
```

`mp.Harminv` implements the filter-diagonalization method, fitting the time series at the monitor point to a sum of complex exponentials. Each exponential gives one mode: its frequency, decay rate, amplitude, and phase. After Harminv, the field is output for one optical period at intervals of T/20 to capture a well-resolved snapshot for visualization.

#### Key Takeaways

- The ring geometry is created by geometry list ordering: later objects override earlier ones, so an air cylinder placed after a dielectric cylinder carves a hole in it.
- `mp.Harminv` is the primary tool for resonance spectroscopy in Meep; it runs on the time-domain signal at a monitor point and extracts all modes within the specified bandwidth.
- The Y-mirror symmetry (`mp.Mirror(mp.Y)`) reduces computation by half without affecting the physics for the Ez polarization.
- The PML thickness should be at least half a wavelength at the lowest frequency of interest; `pad + dpml` ensures the waveguide fields decay before hitting the absorber.
- Outputting fields over a full optical period (not just at one instant) avoids accidentally capturing the field near a zero crossing.

---

### 2. `ring-cyl.py` — Ring Resonator in Cylindrical Coordinates

**Physics:** Repeats the ring resonator calculation but in Meep's native cylindrical coordinate system, which reduces a 2D Cartesian problem to a 1D radial problem by exploiting azimuthal symmetry.
**Difficulty:** Beginner
**Source:** `python/examples/ring-cyl.py`
**Test Status:** PASS (24.8s)

#### Theory

Many ring resonators and axially symmetric structures are most efficiently simulated in cylindrical coordinates (r, phi, z). In this coordinate system the fields are expanded as a Fourier series in the azimuthal angle phi:

    F(r, phi, z, t) = F_m(r, z, t) * exp(i * m * phi)

Each azimuthal order m is decoupled from the others, so the 2D (r,z) problem is solved independently for each m. For a 2D ring (infinitely long in z), the cell collapses to a 1D radial slice, making computation extremely fast. The azimuthal mode number m directly gives the number of field oscillations around the ring, and each m value yields a different resonance frequency.

The quality factor extracted in cylindrical coordinates should match the 2D Cartesian result (within discretization errors), which the test suite confirms. The test verifies that for m=3, the resonance frequency is approximately 0.1184 and Q approximately 85.7, consistent with the 2D Cartesian result.

Because the cell extends only from r=0 to r=r_max (there is no negative-r region), the computational domain is half the size of the equivalent 2D Cartesian problem. Meep exploits the implicit r -> -r mirror symmetry automatically in cylindrical mode.

The waveguide in cylindrical coordinates is represented as a Block rather than two cylinders. The radial width of the block defines the waveguide cross-section:

    Block centered at r = r + w/2, with radial width = w

The PML in cylindrical mode is placed at large r to absorb outward-propagating fields; the axis r=0 requires no special boundary condition because the fields are regular there by construction.

#### Code Walkthrough

The simulation is declared with `dimensions=mp.CYLINDRICAL` and the azimuthal mode number m is passed to the `Simulation` object:

```python
sim = mp.Simulation(
    cell_size=cell,          # cell = mp.Vector3(sr, 0, 0)
    geometry=geometry,       # Block at r = r + w/2
    boundary_layers=pml_layers,
    resolution=resolution,
    sources=sources,
    dimensions=mp.CYLINDRICAL,
    m=m,                     # azimuthal order
)
```

The cell size uses `mp.Vector3(sr, 0, 0)` — only the radial extent matters; the phi and z dimensions are set to zero, signaling that those dimensions are handled analytically (phi by the exp(im*phi) expansion, z by infinite periodicity).

Field output uses `mp.in_volume` with a volume that spans `-sr` to `+sr` to reconstruct the apparent 2D extent of the ring for visualization:

```python
sim.run(
    mp.in_volume(
        mp.Volume(center=mp.Vector3(), size=mp.Vector3(2 * sr)),
        mp.to_appended("ez", mp.at_every(1 / fcen / 20, mp.output_efield_z)),
    ),
    until=1 / fcen,
)
```

This produces an r-vs-t data file showing the radial profile evolving over one optical period.

#### Key Takeaways

- Setting `dimensions=mp.CYLINDRICAL` transforms the simulation to cylindrical coordinates, with the cell extent interpreted as the radial dimension.
- The azimuthal mode number m is specified in the `Simulation` constructor; each m must be run as a separate simulation.
- For 2D rings (no z-confinement), the cell has zero extent in phi and z, yielding a 1D radial problem that is dramatically faster than the equivalent 2D Cartesian simulation.
- The waveguide cross-section is specified as a Block at the appropriate radial position; Meep handles the cylindrical geometry internally.
- The r -> -r symmetry is exploited automatically, providing an additional factor-of-2 speed-up.

---

### 3. `ring_gds.py` — Ring Resonator from a GDSII Layout File

**Physics:** Demonstrates importing a photonic integrated circuit layout from an industry-standard GDSII file, running resonance analysis, and extracting Q factors and resonant wavelengths for a silicon ring resonator.
**Difficulty:** Intermediate
**Source:** `python/examples/ring_gds.py`
**Test Status:** FAIL (missing `gdspy` package — install with `pip install gdspy`)

#### Theory

In real photonic integrated circuit (PIC) design, device geometries are defined in GDSII files — the standard interchange format used by semiconductor foundries. Being able to import GDSII geometries directly into Meep eliminates the need to manually re-specify structures that are already defined in a layout tool, and ensures that the simulation geometry exactly matches the fabricated device.

The ring resonator in this example is defined at telecom wavelengths (lambda = 1.55 micrometers) using real material indices: silicon core (n = 3.4) embedded in silicon dioxide cladding (n = 1.4). The physical dimensions — ring radius 2 micrometers, waveguide width 0.5 micrometers — are chosen to support well-confined modes in the near-infrared.

The resonance condition for a ring of radius R and effective index n_eff is:

    m * lambda_0 = 2 * pi * R * n_eff

At lambda_0 = 1.55 micrometers with n_eff ~ 2.5 (a typical value for a 500 nm wide silicon waveguide), the lowest resonant order is m ~ 8-10 for a 2 micrometer radius ring. Harminv extracts all resonances within a bandwidth of 5% around the center frequency.

Two sources with opposite polarity are placed at diametrically opposite points on the ring. This antisymmetric excitation selectively drives one handedness of the circulating mode, which improves mode selectivity and avoids exciting standing-wave superpositions.

#### Code Walkthrough

The GDSII file is created programmatically using `gdspy`, placing the ring annulus, source rectangles, monitor point, and simulation boundary on separate layers:

```python
def create_ring_gds(radius, width):
    ringCell.add(gdspy.Round((0, 0),
        inner_radius=radius - width / 2,
        radius=radius + width / 2,
        layer=RING_LAYER))
    filename = f"ring_r{radius}_w{width}.gds"
    gdspy.write_gds(filename, unit=1.0e-6, precision=1.0e-9)
    return filename
```

Meep reads the geometry and domain directly from the GDSII file using layer-number-based extraction:

```python
geometry = mp.get_GDSII_prisms(Si, filename, RING_LAYER, -100, 100)
cell = mp.GDSII_vol(filename, SIMULATION_LAYER, zmin, zmax)
src_vol0 = mp.GDSII_vol(filename, SOURCE0_LAYER, zmin, zmax)
```

The two antisymmetric sources drive the Hz field component (TE polarization):

```python
src = [
    mp.Source(..., component=mp.Hz, volume=src_vol0),
    mp.Source(..., component=mp.Hz, volume=src_vol1, amplitude=-1),
]
```

After running, resonant wavelengths and Q factors are extracted from the Harminv modes:

```python
wvl = np.array([1 / m.freq for m in h.modes])
Q = np.array([m.Q for m in h.modes])
```

#### Key Takeaways

- `mp.get_GDSII_prisms` converts a GDSII polygon layer into a list of Meep prism geometry objects automatically.
- `mp.GDSII_vol` reads a rectangular bounding box from a GDSII layer, providing the simulation cell, source volumes, and monitor positions directly from the layout.
- Antisymmetric two-source excitation (one source with `amplitude=-1`) preferentially excites one circular mode handedness, improving spectral selectivity.
- The `sim.reset_meep()` call between simulation runs is essential to free C++ memory and allow a fresh initialization.
- The `gdspy` package must be installed separately; it is not bundled with pymeep.

---

### 4. `cavity-farfield.py` — Photonic Crystal Cavity Far-Field Radiation Pattern

**Physics:** Simulates a photonic crystal defect cavity formed by a 1D array of holes in a dielectric waveguide, and computes the far-field radiation pattern by applying the near-to-far-field transformation to DFT monitor data.
**Difficulty:** Intermediate
**Source:** `python/examples/cavity-farfield.py`
**Test Status:** TIMEOUT (CPU-intensive; uses `stop_when_dft_decayed` with a large cell)

#### Theory

A photonic crystal (PhC) cavity is formed by introducing a defect into an otherwise periodic structure. In this example, a dielectric waveguide (epsilon = 13, width w = 1.2) has N=3 air holes of radius r = 0.36 on each side of a central defect region of length d = 1.4. The holes create a photonic bandgap that confines light to the defect region. The resulting mode has a Gaussian-like field profile and leaks radiation primarily into the vertical (y) direction.

Near-to-far-field (N2F) transformation is Meep's primary tool for computing radiation patterns without extending the simulation cell to the far field. The method records the complex DFT of the electric and magnetic fields on a closed surface (the near-field surface) just outside the source region. Then, using the Green's function of free space, the fields at any far-field point are computed as:

    E_far(r) = integral over near-field surface of [ G(r, r') * J_surface(r') ] dr'

where G is the dyadic Green's function and J_surface represents the equivalent surface currents on the near-field monitor. This is exact up to the accuracy of the FDTD near fields, and it avoids the exponential growth of memory needed to store fields at large distances.

In this example, the near-field surface is an L-shaped region that caps the top of the waveguide (the main radiating surface) and the two vertical side walls, with appropriate signs to form a consistent closed surface. The far fields are then evaluated on a horizontal line at distance d2 = 2 micrometers above the near-field surface, and compared directly against actual DFT field data measured at the same location. Agreement between the two validates the N2F transformation.

The symmetry of the structure (Mirror X with phase -1, Mirror Y with phase -1) reduces the computation to one quadrant of the full cell.

#### Code Walkthrough

The photonic crystal is built as a dielectric block with circular air holes:

```python
geometry = [mp.Block(size=mp.Vector3(mp.inf, w, mp.inf),
                     material=mp.Medium(epsilon=eps))]
for i in range(N):
    geometry.append(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)))
    geometry.append(mp.Cylinder(r, center=mp.Vector3(d / -2 - i)))
```

The near-field monitor is a capped box above the waveguide, with the vertical side walls included to close the surface:

```python
nearfield = sim.add_near2far(
    fcen, 0, 1,
    mp.Near2FarRegion(mp.Vector3(0, 0.5*w + d1), size=mp.Vector3(sx - 2*dpml)),
    mp.Near2FarRegion(..., weight=-1.0),   # left wall, negative sign
    mp.Near2FarRegion(...),                # right wall
)
```

The simulation runs until the DFT monitors have converged (fields have decayed to negligible levels):

```python
sim.run(until_after_sources=mp.stop_when_dft_decayed())
```

Far fields are then obtained point by point using `get_farfield`:

```python
for xc in x:
    ff_pt = sim.get_farfield(nearfield, mp.Vector3(xc, y[0]))
    ff.append(ff_pt[5])   # ff_pt[5] is Hz component
```

The result is compared against direct DFT field data from `sim.get_dft_array(mon, mp.Hz, 0)` at the same spatial location.

#### Key Takeaways

- `sim.add_near2far` creates a DFT monitor for near-to-far-field transformation; `mp.Near2FarRegion` objects define the surface segments.
- Surface weights of -1.0 are needed to correctly orient the equivalent surface currents on surfaces where the outward normal points in the negative direction.
- `stop_when_dft_decayed` is the recommended termination criterion for far-field calculations; it monitors convergence of the DFT integrals rather than the instantaneous field amplitude.
- `sim.get_farfield(mon, pt)` computes [Ex, Ey, Ez, Hx, Hy, Hz] at a single far-field point; index 5 is Hz.
- The N2F result can be verified by comparing it against a DFT monitor placed at the far-field evaluation line within the simulation cell.

---

### 5. `cavity_arrayslice.py` — Field Array Slicing of a Photonic Crystal Cavity

**Physics:** Demonstrates how to extract 1D and 2D slices of the FDTD field arrays from a photonic crystal defect cavity at runtime, using the `get_array` API.
**Difficulty:** Beginner
**Source:** `python/examples/cavity_arrayslice.py`
**Test Status:** FAIL (outdated `get_array()` API call — the example uses the old positional argument signature; current API requires a `mp.Volume` object and component as keyword)

#### Theory

During and after an FDTD simulation, it is often necessary to extract field data over a spatial region for visualization, post-processing, or comparison with measurements. Meep provides `sim.get_array` for this purpose: it returns a NumPy array of field values interpolated onto a regular grid within a specified volume.

The photonic crystal cavity geometry is the same as in sections 4 and 13: a dielectric waveguide (epsilon = 13) with three air holes on each side of a defect region. The cavity mode has an Hz field profile that is localized near the defect and decays exponentially in the photonic crystal mirror regions. Array slicing lets us visualize this spatial structure directly.

In the time domain, the field at any point oscillates at the resonant frequency. The `_run_sources_until` method in the example runs only until the sources finish (t = 0 relative to source end), capturing the field while the pulse is still ringing. For a more physically informative snapshot one would typically run longer and catch the field near its peak oscillation.

The example demonstrates both 1D slices (a horizontal line through the center of the cell) and 2D slices (a rectangular region around the defect). The slice dimensions are specified as fractions of the cell size to focus on the defect region.

#### Code Walkthrough

After building the simulation, the source is run until the source pulse ends:

```python
sim._run_sources_until(0, [])
```

A 1D horizontal slice of Hz through the waveguide center is then extracted:

```python
size_1d = mp.Vector3(xMax - xMin)
center_1d = mp.Vector3((xMin + xMax) / 2)
slice1d = sim.get_array(mp.Volume(center_1d, size=size_1d), component=mp.Hz)
```

A 2D slice covering a rectangular region around the defect is extracted similarly:

```python
size_2d = mp.Vector3(xMax - xMin, yMax - yMin)
center_2d = mp.Vector3((xMin + xMax) / 2, (yMin + yMax) / 2)
slice2d = sim.get_array(mp.Volume(center_2d, size=size_2d), component=mp.Hz)
```

The results are plotted: the 1D slice as a line plot and the 2D slice as a filled contour map using `plt.contourf`.

Note: The current correct API signature is `sim.get_array(component, vol)` or `sim.get_array(component, vol, arr=arr)` for writing into a pre-allocated array; the example uses the older positional argument ordering which no longer matches the API.

#### Key Takeaways

- `sim.get_array(component, vol)` extracts field data as a NumPy array over a `mp.Volume`; the array shape matches the grid resolution within the volume.
- The test version (section 14) shows the correct current API: `sim.get_array(mp.Hz, vol)` with `mp.Volume` as the second argument.
- Pre-allocated arrays can be passed via `arr=arr` to avoid memory allocation overhead in repeated calls during a running simulation.
- `cmplx=True` returns complex-valued data (real and imaginary parts of the phasor); without it, only the real part is returned.
- Array slices are invaluable for visualizing localized modes without writing HDF5 output files.

---

### 6. `metal-cavity-ldos.py` — Local Density of States in a Metal Cavity

**Physics:** Computes the local density of states (LDOS) of a point dipole inside a 2D metallic box cavity, comparing Meep's `dft_ldos` measurement against the analytical formula LDOS ~ 2Q/(pi * omega * V).
**Difficulty:** Intermediate
**Source:** `python/examples/metal-cavity-ldos.py`
**Test Status:** TIMEOUT (CPU-intensive; sweeps over multiple cavity aperture widths)

#### Theory

The local density of states (LDOS) describes the number of electromagnetic modes available for a dipole to emit into at a given position and frequency. In free space, the LDOS is uniform, but near a resonant cavity it is strongly modified. The Purcell effect — the enhancement of spontaneous emission inside a cavity — is directly proportional to the LDOS at the emitter location.

For a single-mode cavity at resonance, the LDOS is given by the Purcell formula:

    LDOS(omega) = (2 / pi) * Q / (omega * V)

where Q is the quality factor and V is the mode volume:

    V = integral[ epsilon(r) |E(r)|^2 d^3r ] / max[ epsilon(r) |E(r)|^2 ]

This formula assumes the dipole is at the field maximum and the emission frequency matches the cavity resonance. The Purcell factor F_P = LDOS_cavity / LDOS_vacuum characterizes how much the cavity enhances spontaneous emission relative to free space.

Meep computes the LDOS directly from the power radiated by the dipole source via the optical theorem. The time-averaged power delivered by a harmonic point dipole of amplitude J at frequency omega is:

    P = -Re( J* . E(r_0) ) / 2

where E(r_0) is the electric field at the dipole position due to the dipole itself. In FDTD, Meep computes this integral as:

    LDOS = P / (pi/2 * omega * |J|^2)

The metal cavity is a 2D square box of side a = 1 surrounded by a metallic shell of thickness t = 0.1. A small aperture of variable width w in one wall couples the cavity to the outside, controlling the Q factor. As w decreases, the cavity becomes more isolated, Q increases, and the LDOS near resonance increases proportionally.

#### Code Walkthrough

The geometry is a metal shell with an air box inside and a variable aperture:

```python
geometry = [
    mp.Block(mp.Vector3(a + 2*t, a + 2*t, mp.inf), material=mp.metal),
    mp.Block(mp.Vector3(a, a, mp.inf), material=mp.air),
    mp.Block(center=mp.Vector3(a/2), size=mp.Vector3(2*t, w, mp.inf),
             material=mp.air),   # aperture
]
```

The resonance frequency and Q of the dominant mode (TM_11) are found with Harminv:

```python
h = mp.Harminv(mp.Ez, mp.Vector3(), fcen, df)
sim.run(mp.after_sources(h), until_after_sources=500)
m = h.modes[0]
f = m.freq
Q = m.Q
```

The analytical LDOS prediction:

```python
Vmode = 0.25 * a * a   # approximate mode volume for TM_11
ldos_1 = Q / Vmode / (2 * math.pi * f * math.pi * 0.5)
```

Meep's direct LDOS measurement (the runtime is set to 2Q periods to allow the cavity to ring for the appropriate time):

```python
T = 2 * Q * (1 / f)
sim.run(mp.dft_ldos(f, 0, 1), until_after_sources=T)
ldos_2 = sim.ldos_data[0]
```

#### Key Takeaways

- `mp.dft_ldos(fcen, 0, 1)` is the primary LDOS step function; it computes the DFT of the dipole's self-power at `fcen` (with 0 bandwidth, 1 frequency point).
- The runtime for an LDOS calculation must be long enough for the cavity to fully ring; T = 2*Q/f is a safe choice since the field amplitude decays by exp(-pi) by then.
- `sim.ldos_data[0]` returns the scalar LDOS value at the single requested frequency.
- The Purcell formula LDOS ~ 2Q/(pi*omega*V) requires knowing the mode volume V, which must be estimated analytically or computed separately; the Meep measurement is mode-volume-free.
- `sim.reset_meep()` must be called between the Harminv run and the LDOS run to clear the field state.

---

### 7. `planar_cavity_ldos.py` — Purcell Enhancement in a Planar Metallic Cavity

**Physics:** Computes the Purcell enhancement factor of an in-plane dipole inside a planar dielectric cavity bounded by lossless metallic walls, comparing Meep results in both 3D Cartesian and cylindrical coordinates against the analytical formula of Abram et al. (1998).
**Difficulty:** Advanced
**Source:** `python/examples/planar_cavity_ldos.py`
**Test Status:** TIMEOUT (CPU-intensive; sweeps over 41 cavity thickness values in both 3D and cylindrical coordinates)

#### Theory

A planar cavity — a slab of dielectric material of index n sandwiched between two perfect metal mirrors — supports a series of resonant modes at frequencies:

    f_m = m * c / (2 * n * L),  m = 1, 2, 3, ...

where L is the cavity thickness. The Purcell enhancement factor for an in-plane dipole (polarized parallel to the mirrors) in such a cavity, relative to the bulk dielectric medium, is given analytically by Equation 7 of Abram et al., IEEE J. Quantum Electronics, 34, 71 (1998):

    F_P(c) = (3/4) * floor(c + 0.5) / c + [4 * floor(c + 0.5)^3 - floor(c + 0.5)] / (16 * c^3)

where c = L / (lambda/n) is the cavity thickness in units of the wavelength within the medium, and floor is the nearest-integer function. This expression accounts for the modification of the LDOS by the discrete mode spectrum and oscillates as a function of cavity thickness.

The enhancement factor exceeds 1 when the cavity thickness places the dipole emission frequency near a cavity resonance, and falls below 1 (inhibition) when the cavity spacing pushes all modes away from the dipole frequency.

The simulation computes LDOS_cavity and LDOS_bulk separately. LDOS_bulk is the LDOS of the dipole in an infinite homogeneous medium of the same index — a reference measurement. The Purcell factor is their ratio:

    F_P = LDOS_cavity / LDOS_bulk

In Meep, the metallic walls are implemented by setting the cell boundary condition to use a PEC (perfect electric conductor) wall in the z-direction (no PML on the z-faces), while PML absorbs the fields in the radial/xy directions.

The simulation is run in two coordinate systems: full 3D Cartesian (with an Ex point source and mirror symmetries) and cylindrical coordinates (with an Er source at m = -1, which corresponds to an in-plane dipole). Agreement between the two provides a strong validation.

#### Code Walkthrough

The bulk LDOS reference is computed with PML on all sides:

```python
def ldos_cyl(cavity_um=None):
    if cavity_um is None:
        # bulk: PML everywhere
        pml_layers = [mp.PML(thickness=PML_UM)]
    else:
        # cavity: PML only in r, metallic walls in z
        pml_layers = [mp.PML(thickness=PML_UM, direction=mp.R)]
    ...
    sim.run(
        mp.dft_ldos(frequency, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(
            FIELD_DECAY_PERIOD, mp.Er, src_pt, FIELD_DECAY_TOL),
    )
    return sim.ldos_data[0]
```

The Purcell factor is then the cavity-to-bulk LDOS ratio:

```python
purcell_meep_cyl = ldos_cavity_cyl / ldos_bulk_cyl
```

The theoretical prediction uses `np.fix` (nearest integer toward zero) to implement the floor function from the Abram formula:

```python
purcell_theory = 3 * np.fix(cavity_um + 0.5) / (4 * cavity_um) + \
    (4 * np.power(np.fix(cavity_um + 0.5), 3) - np.fix(cavity_um + 0.5)) \
    / (16 * np.power(cavity_um, 3))
```

#### Key Takeaways

- The Purcell factor is computed as LDOS_cavity / LDOS_bulk; both must be measured with identical source configurations and the same material index.
- Metallic cavity walls are implemented by placing PML only on the lateral faces and relying on the default PEC boundary at the top/bottom of the cell in the z-direction.
- In cylindrical coordinates, an in-plane dipole is represented by an Er source with m = -1 (the phi-dependence exp(-i*phi) corresponds to a dipole oriented in the x-direction).
- `stop_when_fields_decayed` is preferable to a fixed runtime for LDOS calculations, since the required runtime depends strongly on the Q factor of the cavity.
- The resolution of 71 pixels/micrometer (declared in the comment to ensure integer pixel cell sizes) is much higher than in resonance-finding examples because LDOS requires accurate representation of the near-field self-energy.

---

### 8. `extraction_eff_ldos.py` — Dipole Extraction Efficiency Above a Ground Plane

**Physics:** Computes the light extraction efficiency — the fraction of total dipole emission that escapes into the air above a dielectric layer sitting on a metallic ground plane — comparing cylindrical and 3D Cartesian coordinate simulations.
**Difficulty:** Advanced
**Source:** `python/examples/extraction_eff_ldos.py`
**Test Status:** TIMEOUT (CPU-intensive; sweeps over 21 dipole height positions in two coordinate systems)

#### Theory

Light extraction efficiency is a critical metric in LED and quantum dot device design. A dipole embedded in a high-index dielectric layer radiates preferentially into the high-index medium (due to Snell's law and total internal reflection), reducing the fraction of power that exits into air. The extraction efficiency is defined as:

    eta = P_out / P_total

where P_out is the power escaping through the top air surface and P_total is the total power radiated by the dipole.

P_total is computed from the LDOS measurement using the fluctuation-dissipation theorem:

    P_total = -Re( F_dipole . J*_dipole ) * dV / 2

where F_dipole is the electric field at the dipole position (from sim.ldos_Fdata), J_dipole is the source current amplitude (from sim.ldos_Jdata), and dV is the source voxel volume. In cylindrical coordinates:

    dV = 2 * pi * r_src * (delta_r)^2

where r_src is the radial position of the source and delta_r = 1/resolution is the grid spacing. P_out is measured by a flux monitor on a closed surface covering the top and sides of the air region above the dielectric layer.

As the dipole is moved from the ground plane (height = 0) toward the top surface (height = 1, in fraction of layer thickness), the extraction efficiency varies because the dipole's spatial coupling to the guided and radiation modes changes. Near the ground plane the dipole is in a quasi-metallic environment; near the top surface it couples more efficiently to radiation modes.

#### Code Walkthrough

The geometry is a dielectric block (n = 2.4) above a perfect metal ground plane (enforced by the cell boundary) with air padding above:

```python
geometry = [mp.Block(
    material=mp.Medium(index=n),
    center=mp.Vector3(0, 0, -0.5*sz + 0.5*dmat),
    size=mp.Vector3(mp.inf, mp.inf, dmat),
)]
```

PML is applied to the top (air side) and lateral faces only; the bottom metallic boundary is the cell edge:

```python
boundary_layers = [
    mp.PML(dpml, direction=mp.R),
    mp.PML(dpml, direction=mp.Z, side=mp.High),
]
```

Both LDOS and flux are computed in the same run:

```python
sim.run(
    mp.dft_ldos(fcen, 0, 1),
    until_after_sources=mp.stop_when_fields_decayed(20, src_cmpt, src_pt, tol),
)
out_flux = mp.get_fluxes(flux_air)[0]
dV = 2 * np.pi * src_pt.x / (resolution**2)
total_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * dV
ext_eff = out_flux / total_flux
```

The 3D version uses an Ex source with mirror symmetries to reduce computation, and `dV = 1/resolution^3` for the Cartesian volume element.

#### Key Takeaways

- `sim.ldos_Fdata[0]` and `sim.ldos_Jdata[0]` give the complex field amplitude and source current at the dipole location; their product gives the radiated power without needing a full closed-surface flux integral.
- The source voxel volume dV differs between cylindrical (2*pi*r*dr^2) and Cartesian (dx^3) coordinate systems and must match the coordinate system used.
- A metallic ground plane is implemented by placing the dielectric layer at the bottom of the cell and relying on the default PEC boundary condition at the cell edge (no PML on that face).
- PML should only be applied to the faces through which radiation escapes; the ground-plane face has no PML.
- The cylindrical and 3D results should agree within a few percent; larger discrepancies indicate numerical error (insufficient resolution or cell size).

---

### 9. `disc_extraction_efficiency.py` — Extraction Efficiency of a Dipole Collection in a Disc

**Physics:** Computes the total extraction efficiency of a spatially incoherent ensemble of electric dipoles uniformly distributed within a dielectric disc, by summing individual dipole contributions weighted by their radial positions and using a Fourier-series expansion in the azimuthal angle.
**Difficulty:** Advanced
**Source:** `python/examples/disc_extraction_efficiency.py`
**Test Status:** TIMEOUT (CPU-intensive; loops over 11 radial positions, each with multiple m-values)

#### Theory

Real light-emitting devices contain many emitters distributed throughout the active region, not a single dipole. To compute the total extraction efficiency of such a device, one must integrate over the spatial distribution of emitters. For a disc-shaped active region of radius R and thickness D with uniform emitter density, the spatially averaged extraction efficiency is:

    eta_total = integral[ eta(r, z) * r dr dz dphi ] / (pi * R^2 * D)

For a uniform distribution in z (or fixed z = 0.5*D as here) and radial symmetry, this reduces to a weighted radial integral over individual dipole efficiencies.

For a dipole at radius r > 0, the azimuthal symmetry is broken and the fields must be expanded as a Fourier series in phi:

    F(r, phi, z) = sum_m F_m(r, z) * exp(i*m*phi)

Each m-component is an independent cylindrical simulation. The total emission of the dipole at (r, 0) is the sum over all m values:

    P_total(r) = P_0(r) + 2 * sum_{m=1}^{m_max} P_m(r)

The factor of 2 comes from the symmetry P_m = P_{-m}. The sum is truncated when the contribution from order m falls below a threshold (1% of the maximum). For a dipole at r = 0, only m = ±1 contributes (due to the angular selection rule for a linearly polarized dipole).

The total disc extraction efficiency is then computed by a weighted radial sum, with each dipole weighted by its circumferential length r*dr (the annular volume element normalized by pi*R^2):

    eta_disc = sum_j eta_j * r_j * delta_r / (R^2 / 2)

#### Code Walkthrough

Dipoles at r = 0 require only m = ±1 and use a small offset from the axis due to numerical issues with Er at the grid origin:

```python
dipole_rpos_um[0] = 1.5 / RESOLUTION_UM   # small offset from r=0
m = -1
dipole_flux, dipole_radiation_pattern = dipole_in_disc(dipole_height, dipole_rpos_um[0], m)
```

For dipoles at r > 0, a while loop increments m until the flux contribution decays:

```python
m = 0
while True:
    dipole_flux, dipole_radiation_pattern = dipole_in_disc(dipole_height, rpos_um, m)
    dipole_flux_total += dipole_flux * (1 if m == 0 else 2)
    ...
    if m > 0 and (dipole_radiation_pattern_flux / dipole_radiation_pattern_flux_max) \
            < flux_decay_threshold:
        break
    m += 1
```

The extraction efficiency is the ratio of the radiated far-field flux to the total dipole emission:

```python
radiation_pattern_total_flux = radiation_pattern_flux(radiation_pattern_total)
extraction_efficiency = radiation_pattern_total_flux / flux_total
```

#### Key Takeaways

- For emitters at r > 0 in cylindrical coordinates, multiple m values must be summed; the number of terms needed grows with r*k, where k is the wavevector.
- The factor (1 if m == 0 else 2) accounts for the conjugate symmetry P_{-m} = P_m, avoiding double computation.
- A dipole near r = 0 requires a small offset (1.5/resolution) due to the singular behavior of Er at the axis.
- The `force_complex_fields=True` parameter ensures that complex-valued DFT data is computed, which is needed for the near-to-far-field transformation in cylindrical coordinates.
- The disc extraction efficiency integrates individual dipole contributions with area-element weights r*delta_r, correctly accounting for the cylindrical geometry.

---

### 10. `disc_radiation_pattern.py` — Radiation Pattern of a Dielectric Disc in Cylindrical Coordinates

**Physics:** Computes the far-field radiation pattern of a ring current source within a dielectric disc using the near-to-far-field transformation in cylindrical coordinates, and validates the result by comparing the total far-field flux against the directly measured near-field flux.
**Difficulty:** Advanced
**Source:** `python/examples/disc_radiation_pattern.py`
**Test Status:** TIMEOUT (CPU-intensive; high resolution and long runtime for accurate far-field computation)

#### Theory

The radiation pattern of a light source describes how the radiated power is distributed as a function of direction in the far field. For an emitter embedded in a high-index disc, the radiation pattern is strongly shaped by: (1) total internal reflection at the disc boundaries, which redirects light toward the vertical (normal-to-disc) direction; (2) Fabry-Perot resonances within the disc thickness; and (3) diffraction from the disc edge.

In cylindrical coordinates, the far-field radiation pattern is parameterized by the polar angle theta from the disc normal (z-axis) and the azimuthal angle phi. For an Er source with azimuthal mode m, the far field at radius R is:

    E_far(R, theta, phi) = sum_m E_m^far(R, theta) * exp(i*m*phi)

The Poynting vector in the far field gives the angular power distribution:

    dP/dOmega = R^2 * (1/2) * Re(E x H*) . r_hat

The total power is obtained by integrating over the upper hemisphere:

    P_total = 2*pi*R^2 * integral_0^{pi/2} [dP/dOmega] * sin(theta) d(theta)

This is computed as a numerical quadrature using `np.trapezoid`. The radial flux components are assembled from the Cartesian E and H fields returned by `sim.get_farfield`:

```python
flux_x = np.real(np.conj(e_field[:, 1]) * h_field[:, 2]
               - np.conj(e_field[:, 2]) * h_field[:, 1])
flux_z = np.real(np.conj(e_field[:, 0]) * h_field[:, 1]
               - np.conj(e_field[:, 1]) * h_field[:, 0])
flux_r = np.sqrt(flux_x**2 + flux_z**2)
```

The near-field flux (measured by a closed flux surface) and the far-field flux (computed by the N2F transformation and integrated over angles) should agree to within the discretization error of the simulation.

#### Code Walkthrough

The disc geometry sits at the bottom of the cell (against the metallic ground plane) with air padding above:

```python
geometry = [mp.Block(
    material=mp.Medium(index=N_DISC),
    center=mp.Vector3(0.5*DISC_RADIUS_UM, 0, -0.5*sz + 0.5*disc_um),
    size=mp.Vector3(DISC_RADIUS_UM, mp.inf, disc_um),
)]
```

Both a near-field flux monitor and an N2F monitor are registered with the same surface region:

```python
flux_mon = sim.add_flux(frequency, 0, 1, ...)   # for total flux validation
n2f_mon = sim.add_near2far(frequency, 0, 1, ...)  # for far-field computation
```

The radiation pattern at each polar angle is evaluated by calling `get_farfield` on a semicircular arc:

```python
far_field = sim.get_farfield(
    n2f_mon,
    mp.Vector3(FARFIELD_RADIUS_UM * math.sin(polar_rad[i]), 0,
               FARFIELD_RADIUS_UM * math.cos(polar_rad[i])),
    GREENCYL_TOL,
)
```

The `GREENCYL_TOL` parameter controls the accuracy of the cylindrical Green's function summation used internally.

The total far-field flux is then:

```python
flux_far = 2 * math.pi * FARFIELD_RADIUS_UM**2 \
    * np.trapezoid(radial_flux * np.sin(polar_rad), polar_rad)
```

#### Key Takeaways

- In cylindrical coordinates, `sim.get_farfield` accepts a `tol` parameter that controls the convergence of the internal Fourier-Bessel summation; tighter tolerances give more accurate results at the cost of more computation.
- The far-field radius `FARFIELD_RADIUS_UM = 1e6 * WAVELENGTH_UM` (one million wavelengths) ensures the evaluation point is truly in the far field where the 1/R field decay is negligible compared to the oscillatory phase.
- Agreement between near-field and far-field total fluxes validates the N2F transformation; errors larger than a few percent indicate insufficient near-field surface size or inadequate simulation runtime.
- The radiation pattern is plotted both as a 2D polar plot and as a 3D surface for visualization; the 3D surface is obtained by rotating the 2D pattern around the z-axis using phi symmetry.
- `np.trapezoid` (NumPy 2.0 API) is used instead of the deprecated `np.trapz` for the angular integration.

---

### 11. `test_ring.py` — Unit Test for Ring Resonator (Harminv and Pade DFT)

**Physics:** Verifies the resonant frequency, decay rate, and field amplitude of the fundamental ring resonator mode, and cross-validates against the Pade DFT method as an alternative to Harminv for spectral analysis.
**Difficulty:** Intermediate
**Source:** `python/tests/test_ring.py`
**Test Status:** PASS (40.4s)

#### Theory

The test validates two independent spectral analysis methods against the same time-domain data:

1. **Harminv** (filter-diagonalization method): Fits the time series to a sum of decaying sinusoids by solving a generalized eigenvalue problem. It works best when the field contains only a few modes and the run time is much longer than the cavity decay time.

2. **Pade DFT**: Uses Pade rational approximation to analytically continue the DFT of the time-domain signal into the complex frequency plane. Pade approximation can resolve closely spaced resonances with shorter time series than Harminv, but requires careful regularization.

For the ring resonator at resolution 10 with n=3.4, w=1, r=1, the fundamental mode has:
- Resonant frequency: f_0 = 0.11810...
- Decay rate: gamma = -0.000731... (negative because the mode decays)
- Q factor: Q = f_0 / (2*|gamma|) ~ 80.8
- Mode amplitude at monitor point: |A| = 0.00341...

The Q factor of approximately 80 means the mode rings for about 80 optical periods before its energy decreases by a factor of exp(-2*pi) ~ 0.0019.

The test also checks that point queries of epsilon and field values are accurate:

    epsilon at (1, 1): 11.56  (= n^2 = 3.4^2 = 11.56, consistent with dielectric)
    Ez at (1, 1): -0.08186...

#### Code Walkthrough

The test class follows the standard unittest structure. The `init` method sets up the simulation with both Harminv and PadeDFT monitors:

```python
self.h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
self.p = mp.PadeDFT(c=mp.Ez, center=mp.Vector3(r + 0.1), size=mp.Vector3(),
                    sampling_interval=4)
```

`test_harminv` runs and validates the fundamental mode parameters to 4 decimal places:

```python
self.assertAlmostEqual(m1.freq, 0.118101315147, places=4)
self.assertAlmostEqual(m1.decay, -0.000731513241623, places=4)
```

`test_pade` validates that the Pade spectral peak agrees with the Harminv frequency to 3 decimal places:

```python
freq_domain = [self.p.dft(freq) for freq in freqs]
idx = find_peaks(np.abs(freq_domain)**2 / max(np.abs(freq_domain)**2),
                 prominence=1e-4)[0]
self.assertAlmostEqual(freqs[idx[0]], self.h.modes[0].freq, places=3)
```

#### Key Takeaways

- `mp.PadeDFT` is an alternative to Harminv for extracting resonances; it works by calling `self.p.dft(freq)` on a frequency grid after the run, evaluating the Pade rational approximant at each point.
- Harminv mode objects expose `.freq`, `.decay`, `.Q`, `.amp` (complex amplitude), so Q does not need to be computed manually.
- `sim.get_field_point(component, pt)` queries the field at a single grid point; `sim.get_epsilon_point(pt)` queries the dielectric constant.
- `sampling_interval=4` in PadeDFT means field samples are collected every 4 time steps; this must be large enough to avoid aliasing (Nyquist: 2 samples per oscillation).
- Single-precision builds have relaxed tolerances (`places` reduced or `tol` increased) because float32 arithmetic is less accurate than float64.

---

### 12. `test_ring_cyl.py` — Unit Test for Cylindrical Ring Resonator

**Physics:** Verifies that the ring resonator simulation in cylindrical coordinates with azimuthal mode m=3 produces the correct resonant frequency, Q factor, and field amplitude to high precision.
**Difficulty:** Beginner
**Source:** `python/tests/test_ring_cyl.py`
**Test Status:** PASS (7.9s)

#### Theory

This test validates the cylindrical coordinate simulation of section 2. The azimuthal mode m=3 has a ring of three field oscillations around the circumference, corresponding to a higher-order whispering-gallery mode. The resonant frequency of m=3 (approximately 0.1184) is slightly higher than the m-independent Q enhancement would suggest, reflecting the quantization of transverse wavenumber in the ring.

The cylindrical simulation is significantly faster than the 2D Cartesian equivalent (7.9s vs 26.5s for the example) because the phi dimension is handled analytically, reducing the spatial discretization to 1D radial rather than 2D. The convergence tolerance of 1e-7 (double precision) or 1e-6 (single precision) for relative error reflects the high numerical accuracy achievable with the cylindrical coordinate reduction.

The `split_chunks_evenly=False` option tells Meep to use load-balanced chunk partitioning rather than uniform spatial splitting. For a 1D radial domain, this has little effect, but it is good practice in general.

#### Code Walkthrough

The simulation uses a Block for the waveguide (as in the example) and Harminv for resonance extraction:

```python
h = mp.Harminv(mp.Ez, mp.Vector3(self.r + 0.1), self.fcen, self.df)
self.sim.run(mp.after_sources(h), until_after_sources=200)
m = h.modes[0]
res = [m.freq, m.decay, m.Q, abs(m.amp), m.amp.real, m.amp.imag]
```

The expected values are:

```python
expected = [
    0.11835455441250553,   # frequency
    -6.907792691629741e-4, # decay rate
    85.66741917133473,     # Q factor
    0.025701906263451237,  # |amplitude|
    -0.024027038833537524, # Re(amplitude)
    -0.009126302124459489, # Im(amplitude)
]
```

The `ApproxComparisonTestCase.assertClose` helper checks all six values simultaneously with a single relative tolerance.

#### Key Takeaways

- The cylindrical coordinate simulation is approximately 3x faster than the 2D Cartesian equivalent for this geometry, while producing the same physics.
- Each azimuthal mode number m requires a separate simulation; the test covers m=3 specifically.
- The Q factor of ~85.7 for m=3 is larger than the ~80.8 for the Cartesian simulation (m not specified) because the cylindrical simulation's higher resolution and PML placement differ slightly.
- `split_chunks_evenly=False` enables load-balanced MPI decomposition; for serial runs it has no effect.
- The amplitude values (real and imaginary parts) depend on the source position, phase, and normalization, so they must be matched exactly if the geometry is identical.

---

### 13. `test_cavity_farfield.py` — Unit Test for Photonic Crystal Cavity Far Fields

**Physics:** Validates the near-to-far-field transformation for a photonic crystal defect cavity by comparing computed far fields against pre-stored reference data, testing both single-frequency and multi-frequency near-field monitors.
**Difficulty:** Intermediate
**Source:** `python/tests/test_cavity_farfield.py`
**Test Status:** PASS (18.8s)

#### Theory

This test exercises the same photonic crystal cavity geometry as section 4 but with a reduced resolution (10 pixels per unit) and a shorter runtime (200 time units) to make the test tractable. The near-field monitors record complex DFT amplitudes of all six field components (Ex, Ey, Ez, Hx, Hy, Hz) on the near-field surface, and `sim.get_farfields` computes the far fields on a 2D rectangular volume.

The multi-frequency test (`nfreqs=4`) verifies that the near-field monitor correctly accumulates DFT data at multiple frequencies simultaneously. When `nfreqs > 1`, `add_near2far` samples the field at nfreqs frequencies spaced by the bandwidth df/nfreqs around the center frequency fcen. This is useful for computing the spectrally resolved radiation pattern without running the simulation multiple times.

The `decimation_factor=1` parameter means the near-field DFT is updated every time step; higher values reduce accuracy but save memory.

The reference data is stored in HDF5 files (`cavity-farfield.h5` and `cavity-farfield-4-freqs.h5`) in the test data directory. The tolerance for comparison is 1e-7 (double precision) or 1e-5 (single precision), reflecting the round-off limit of the floating-point arithmetic.

#### Code Walkthrough

The near-field monitor uses a near-field surface region with the waveguide top as the main aperture:

```python
nearfield = sim.add_near2far(
    fcen, 0.1, nfreqs,
    mp.Near2FarRegion(mp.Vector3(0, 0.5*w + d1), size=mp.Vector3(2*dpml - sx)),
    mp.Near2FarRegion(..., weight=-1.0),  # left side wall
    mp.Near2FarRegion(...),               # right side wall
    decimation_factor=1,
)
sim.run(until=200)
```

The far fields are computed over a 2D rectangular volume:

```python
vol = mp.Volume(mp.Vector3(0, (0.5*w) + d2 + (0.5*h)),
                size=mp.Vector3(sx - 2*dpml, h))
result = sim.get_farfields(nearfield, resolution, where=vol)
```

`sim.get_farfields` (plural) returns a dictionary with keys 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', each a 2D array of complex values on the specified grid.

Validation reads reference data from the HDF5 file and compares:

```python
ref_ex = mp.complexarray(f["ex.r"][()], f["ex.i"][()])
self.assertClose(ref_ex, result["Ex"], epsilon=tol)
```

#### Key Takeaways

- `sim.get_farfields(mon, resolution, where=vol)` computes far fields on a grid within a Volume; `sim.get_farfield(mon, pt)` computes at a single point.
- Multi-frequency near-field monitors (`nfreqs > 1`) store DFT data at multiple frequencies simultaneously, enabling spectrally resolved radiation patterns in one run.
- `decimation_factor` controls how often the near-field DFT is updated; `decimation_factor=1` updates every timestep for maximum accuracy.
- `mp.complexarray(real, imag)` reconstructs a complex array from separate real/imaginary HDF5 datasets, which is Meep's standard HDF5 storage format.
- Regression testing against stored reference data is crucial for catching subtle numerical regressions in the near-to-far-field machinery.

---

### 14. `test_cavity_arrayslice.py` — Unit Test for Field Array Slicing

**Physics:** Verifies that `sim.get_array` correctly extracts 1D and 2D field slices from a photonic crystal cavity simulation, testing both basic and advanced usage (pre-allocated arrays, complex output, illegal size detection).
**Difficulty:** Beginner
**Source:** `python/tests/test_cavity_arrayslice.py`
**Test Status:** PASS (27.3s)

#### Theory

This test covers the same photonic crystal cavity geometry as sections 4 and 5, but focuses entirely on the field extraction API. The simulation is run only until the sources finish (`until_after_sources=0`), capturing a snapshot of the field while the Gaussian pulse is still active — sufficient for testing the array extraction functionality.

The key physics being validated is that the field interpolation within `get_array` is consistent with the underlying Yee grid. Meep's Yee grid staggers different field components at different spatial positions (Ex is centered at x-face centers, Ey at y-face centers, etc.), so interpolation to a common grid requires care. The test verifies that Hz (which lives at the center of each z-oriented face on the Yee grid) is extracted correctly at a regular grid of points.

The expected array shapes are verified numerically: a 1D slice of length `floor((xMax - xMin) * resolution) + 1 = 126` points and a 2D slice of shape `(126, 38)`. These come from the cell dimensions `sx`, `sy`, and the resolution of 20 pixels per unit.

#### Code Walkthrough

The correct current API passes a Volume object and the field component:

```python
vol = mp.Volume(center=self.center_1d, size=self.size_1d)
hl_slice1d = self.sim.get_array(mp.Hz, vol)
```

For a pre-allocated output array:

```python
arr = np.zeros(126, dtype=np.float64)
vol = mp.Volume(center=self.center_1d, size=self.size_1d)
self.sim.get_array(mp.Hz, vol, arr=arr)
```

Complex output is obtained with the `cmplx=True` flag:

```python
hl_slice1d = self.sim.get_array(mp.Hz, vol, cmplx=True)
# returns complex128 array of shape (126,)
```

Illegal array sizes are detected and raise `ValueError`:

```python
with self.assertRaises(ValueError):
    arr = np.zeros(128)   # wrong size (should be 126)
    self.sim.get_array(mp.Hz, vol, arr=arr)
```

#### Key Takeaways

- The current `sim.get_array` API signature is `sim.get_array(component, vol, arr=None, cmplx=False)` — component first, then Volume.
- Pre-allocated `arr=arr` avoids memory allocation in tight loops; the array must have exactly the right shape determined by `floor(size/resolution) + 1`.
- `cmplx=True` returns a complex-valued array (float64 -> complex128, float32 -> complex64); without it, only the real part is returned as float.
- The example version `cavity_arrayslice.py` uses the old positional argument order and fails; the test version uses the correct API and passes.
- The simulation needs to be run (at least `until_after_sources=0`) before `get_array` can be called; calling it before any run returns zeros.

---

### 15. `test_ldos.py` — Unit Test for Local Density of States

**Physics:** Validates Meep's LDOS computation against the analytical Purcell enhancement formula for a metallic planar cavity, and verifies the extraction efficiency of a dipole above a ground plane in both cylindrical and 3D coordinate systems.
**Difficulty:** Advanced
**Source:** `python/tests/test_ldos.py`
**Test Status:** TIMEOUT (CPU-intensive; requires three separate long simulations at high resolution)

#### Theory

This test performs three independent validations:

**1. Purcell enhancement in cylindrical coordinates (`test_ldos_cyl`):** Computes LDOS_bulk and LDOS_cavity in cylindrical coordinates, forms the ratio, and compares against the Abram et al. formula with a tolerance of 0.1 (10% absolute error). The cavity thickness of 1.63 wavelengths is chosen to avoid a Van Hove singularity (where the formula has a discontinuity) and to fall between two cavity resonances.

**2. Purcell enhancement in 3D coordinates (`test_ldos_3D`):** Repeats the same calculation in full 3D Cartesian geometry at cavity thickness 0.75 wavelengths, validating that both coordinate systems give consistent results. The 3D calculation uses three mirror symmetries to reduce computation by a factor of 8.

**3. Extraction efficiency (`test_ldos_ext_eff`):** Verifies that the extraction efficiency of a dipole at height 0.5*dmat above a ground plane within a dielectric layer of thickness 0.5*lambda/n agrees between cylindrical (m=-1 and m=+1 separately) and 3D Cartesian simulations to within 0.02 (2% absolute error). It also checks that m=-1 and m=+1 give identical results (as required by the m -> -m symmetry of a real-valued dipole source).

The `purcell_enh_theory` method implements Equation 7 of Abram et al.:

    F_P(c) = (3/4) * round(c) / c + [4*round(c)^3 - round(c)] / (16*c^3)

where round uses nearest-half-integer rounding (implemented via `np.fix(c + 0.5)`).

#### Code Walkthrough

The bulk LDOS is measured with PML on all sides (both bulk functions share the same pattern):

```python
sim.run(
    mp.dft_ldos(self.fcen, 0, 1),
    until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Er, mp.Vector3(), self.tol),
)
return sim.ldos_data[0]
```

The cavity LDOS uses PML only radially, with metallic walls in z (no PML in z-direction):

```python
pml_layers = [mp.PML(self.dpml, direction=mp.R)]
```

The extraction efficiency test verifies m symmetry:

```python
ext_eff_cyl = self.ext_eff_cyl(layer_thickness, dipole_height, -1.0)
ext_eff_cyl_m_plus = self.ext_eff_cyl(layer_thickness, dipole_height, +1.0)
self.assertEqual(ext_eff_cyl, ext_eff_cyl_m_plus)
```

Note: `self.assertEqual` (not `assertAlmostEqual`) because m=+1 and m=-1 should give bit-for-bit identical results due to the symmetry of the field equations.

#### Key Takeaways

- The 10% tolerance on the Purcell factor comparison reflects that at resolution=25 pixels/micrometer, the LDOS has a few-percent discretization error that accumulates when forming the ratio.
- Metallic walls in a planar cavity are realized by restricting PML to the radial/lateral directions only; the axial cell boundaries provide the PEC condition automatically.
- m=+1 and m=-1 in cylindrical coordinates give identical LDOS and extraction efficiency for real-valued dipole sources; this is a non-trivial symmetry that the test explicitly verifies.
- `mp.dft_ldos` can also be invoked as `mp.dft_ldos(ldos=mp.Ldos(fcen, 0, 1))` (keyword argument form used in `cavity_ldos_3D`), showing the underlying `mp.Ldos` object that wraps the frequency specification.
- For the extraction efficiency calculation, `sim.ldos_Fdata` and `sim.ldos_Jdata` store the frequency-domain field and source amplitude at the dipole location needed to compute total emitted power.

---

### 16. `test_eigfreq.py` — Unit Test for Cavity Eigenfrequency Solver

**Physics:** Verifies that Meep's iterative eigenfrequency solver `solve_eigfreq` accurately finds the complex resonant frequency of a photonic crystal defect cavity without needing a time-domain Harminv analysis.
**Difficulty:** Advanced
**Source:** `python/tests/test_eigfreq.py`
**Test Status:** TIMEOUT (CPU-intensive; iterative solver requires many FDTD cycles to converge)

#### Theory

Meep's `solve_eigfreq` method implements an FDTD-based iterative eigenvalue solver that directly finds the complex resonant frequency of the dominant mode without requiring a broadband excitation and Harminv post-processing. The algorithm works by:

1. Initializing the fields with a narrowband source near the expected resonance.
2. Running the FDTD simulation for a short time to allow transients to decay.
3. Applying a frequency-shift operator and refining the eigenfrequency estimate iteratively (similar to power iteration in linear algebra).

The result is a complex frequency f = f_real + i*f_imag, where:
- f_real = resonant frequency
- f_imag = -gamma/2 (half the decay rate, negative for a lossy resonance)
- Q = f_real / (2 * |f_imag|)

For the photonic crystal cavity with N=3 holes, spacing d=1.4, hole radius r=0.36, waveguide width w=1.2, and epsilon=13, the test expects:

    f_real = 0.23445413142440263
    f_imag = -0.0003147775697388
    Q ~ 372

This Q factor is significantly higher than the ring resonator examples because the photonic crystal mirror provides stronger confinement through the bandgap mechanism rather than index contrast alone.

The simulation uses `force_complex_fields=True` to enable complex-valued field storage, which is required for the eigenvalue solver (which uses both amplitude and phase information). Two mirror symmetries reduce the computation to one quadrant of the cell.

#### Code Walkthrough

The simulation is initialized with `sim.init_sim()` before calling the eigenfrequency solver:

```python
sim = mp.Simulation(
    cell_size=mp.Vector3(sx, sy),
    force_complex_fields=True,
    geometry=geometry,
    boundary_layers=[mp.PML(1.0)],
    sources=src,
    symmetries=[mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)],
    resolution=20,
)
sim.init_sim()
eigfreq = sim.solve_eigfreq(tol=1e-6)
```

The tolerance `tol=1e-6` specifies the relative convergence criterion for the iterative eigenvalue refinement. Tighter tolerances give more accurate results but require more iterations.

The test validates both real and imaginary parts of the eigenfrequency to 5 decimal places:

```python
self.assertAlmostEqual(eigfreq.real, 0.23445413142440263, places=5)
self.assertAlmostEqual(eigfreq.imag, -0.0003147775697388, places=5)
```

The test is skipped on single-precision builds because the eigenfrequency solver relies on double-precision arithmetic to achieve 5-decimal-place agreement.

#### Key Takeaways

- `sim.solve_eigfreq(tol)` is an alternative to Harminv for finding resonances; it is more accurate for isolated modes but requires an initial frequency estimate (from the source) and `force_complex_fields=True`.
- The result is a Python complex number with `.real` (resonant frequency) and `.imag` (half-decay-rate, negative).
- `@unittest.skipIf(mp.is_single_precision(), ...)` skips tests that require double-precision arithmetic, which is important for CI environments that may use single-precision builds.
- `force_complex_fields=True` stores both real and imaginary parts of all field components at every time step; it doubles the memory requirement but enables phase-sensitive operations like the eigenfrequency solver.
- The photonic crystal cavity Q of ~372 is about 4-5x higher than the ring resonator examples, illustrating the advantage of photonic bandgap confinement over simple total-internal-reflection waveguiding.
