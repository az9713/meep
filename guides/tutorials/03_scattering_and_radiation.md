# Chapter 3: Scattering and Radiation

This chapter covers electromagnetic scattering and radiation simulations using Meep's FDTD engine. The tutorials span Mie scattering from spheres and cylinders, antenna radiation patterns computed via near-to-far-field transforms, dipole radiation in free space and above ground planes, Cherenkov radiation from a moving charge, and diffracted planewave mode decomposition from binary gratings. Together they form a comprehensive treatment of how energy is scattered, redirected, and radiated by structures ranging from simple geometric objects to antenna arrays, with analytic benchmarks drawn from Mie theory, array factor analysis, and classical electrodynamics.

---

### 1. `mie_scattering.py` — Mie Scattering Efficiency of a Dielectric Sphere

**Physics:** Computes the broadband scattering efficiency of a lossless dielectric sphere as a function of the size parameter 2πr/λ, benchmarked against the analytic Mie series solution.
**Difficulty:** Intermediate
**Source:** `python/examples/mie_scattering.py`
**Test Status:** TIMEOUT (CPU-intensive 3D simulation with 100-frequency broadband sweep)

#### Theory

Mie scattering is the exact analytic solution to Maxwell's equations for a plane wave incident on a homogeneous sphere of arbitrary radius. It was derived by Gustav Mie in 1908 and remains the gold standard for validating electromagnetic scattering codes. The scattering efficiency Q_sca is defined as the ratio of the scattering cross section σ_sca to the geometric cross-sectional area πr²:

    Q_sca = σ_sca / (πr²)

The scattering cross section is computed from the net outward flux of scattered power through a closed surface surrounding the sphere, divided by the incident intensity I₀:

    σ_sca = P_sca / I₀

where P_sca is the total scattered power. In Mie theory, Q_sca is expressed as an infinite series over partial waves (Mie coefficients a_n and b_n):

    Q_sca = (2/x²) Σ_{n=1}^{∞} (2n+1)(|a_n|² + |b_n|²)

where x = 2πr/λ = kr is the size parameter and the a_n, b_n coefficients are determined by matching tangential E and H boundary conditions at the sphere surface.

The simulation uses the scattered-field approach: two separate FDTD runs are performed. The first run is in empty space (no sphere) to record the incident field fluxes through a box surrounding where the sphere will be. The second run includes the sphere, and the incident fluxes are subtracted from the total fluxes (via `load_minus_flux_data`) to isolate the scattered power. This subtraction technique is essential because FDTD cannot separately track incident and scattered fields without it.

The scattering efficiency rises from near zero at small size parameters (Rayleigh regime, where σ_sca ∝ λ^{-4}), passes through several Mie resonances, and approaches a geometric-optics limit of approximately 2 at large size parameters (the extinction paradox). For a sphere of refractive index n = 2.0, the resonances are well-separated and clearly visible in a log-log plot.

The PyMieScatt library provides the reference Mie solution computed using its `MieQ` function. Agreement between Meep and theory to within a few percent at resolution = 25 validates both the scattered-field subtraction method and Meep's treatment of curved dielectric interfaces via subpixel averaging.

#### Code Walkthrough

The frequency range is defined as the size parameter range x ∈ [2π/10, 2π/2] converted to Meep's frequency units where c = 1:

```python
r = 1.0  # radius of sphere
wvl_min = 2 * np.pi * r / 10
wvl_max = 2 * np.pi * r / 2
frq_min = 1 / wvl_max
frq_max = 1 / wvl_min
frq_cen = 0.5 * (frq_min + frq_max)
dfrq = frq_max - frq_min
nfrq = 100
```

The planewave source uses `is_integrated=True`, which is required whenever a planewave source extends into the PML region. This ensures the total-field/scattered-field formulation remains self-consistent:

```python
sources = [mp.Source(
    mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
    center=mp.Vector3(-0.5 * s + dpml),
    size=mp.Vector3(0, s, s),
    component=mp.Ez,
)]
```

Mirror symmetries are exploited to reduce the 3D simulation to one quadrant:

```python
symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]
```

Six DFT flux monitors form a closed box around the sphere. The scattered flux is their signed sum, with signs chosen so that outward flux is positive on each face:

```python
scatt_flux = (np.asarray(box_x1_flux) - np.asarray(box_x2_flux)
            + np.asarray(box_y1_flux) - np.asarray(box_y2_flux)
            + np.asarray(box_z1_flux) - np.asarray(box_z2_flux))
```

The scattering efficiency is obtained by dividing by the intensity and normalizing by the geometric cross section:

```python
intensity = np.asarray(box_x1_flux0) / (2 * r) ** 2
scatt_cross_section = np.divide(scatt_flux, intensity)
scatt_eff_meep = scatt_cross_section * -1 / (np.pi * r**2)
```

The minus sign arises because the scattered field subtracts from the forward flux (the scattered field opposes the incident field in the forward hemisphere for a purely scattering object).

The Mie theory benchmark calls PyMieScatt with wavelength and diameter in nanometers (the library's native units):

```python
scatt_eff_theory = [
    ps.MieQ(n_sphere, 1000/f, 2*r*1000, asDict=True)["Qsca"] for f in freqs
]
```

#### Key Takeaways

- The scattered-field subtraction technique (`load_minus_flux_data`) is the standard approach for isolating scattered power from incident power in FDTD.
- A closed surface of six DFT flux monitors with consistent sign conventions computes net scattered power by the optical theorem.
- `is_integrated=True` is mandatory for planewave sources that extend into PML, otherwise the perfectly matched layer does not absorb the source field correctly.
- Mirror symmetry (`mp.Mirror`) can reduce 3D simulation cost by a factor of 4 when the geometry and polarization have appropriate symmetry.
- Subpixel averaging at the sphere surface means that even at resolution = 25 pixels/μm, Mie resonances are well-resolved; convergence improves as roughly 1/resolution².

---

### 2. `gaussian-beam.py` — Focused Gaussian Beam Propagation

**Physics:** Launches a focused Gaussian beam with a specified waist radius, propagation direction, and polarization, demonstrating Meep's `GaussianBeamSource` API.
**Difficulty:** Beginner
**Source:** `python/examples/gaussian-beam.py`
**Test Status:** PASS (68.3s)

#### Theory

A Gaussian beam is the fundamental transverse mode of a laser resonator. In the paraxial approximation, its electric field profile in the plane perpendicular to the propagation axis follows a Gaussian envelope:

    E(r, z) = E₀ (w₀/w(z)) exp(-r²/w(z)²) exp(-ikz - ikr²/2R(z) + iζ(z))

where:
- w₀ is the beam waist (minimum spot radius)
- w(z) = w₀ sqrt(1 + (z/z_R)²) is the beam radius at position z
- z_R = πw₀²/λ is the Rayleigh range
- R(z) = z(1 + (z_R/z)²) is the radius of curvature of the phase front
- ζ(z) = arctan(z/z_R) is the Gouy phase

The Rayleigh range z_R characterizes the distance over which the beam remains well collimated. For the parameters in this example, w₀ = 0.8 μm and λ = 1 μm, giving z_R = π(0.8)²/1 ≈ 2.01 μm. The beam diverges significantly beyond this distance.

Meep implements the Gaussian beam through an amplitude function applied across the source plane. The user specifies the beam waist center `beam_x0` (relative to the source center), the propagation direction `beam_kdir`, the waist radius `beam_w0`, and the polarization vector `beam_E0`. Meep computes the complex amplitude at each point on the source plane corresponding to the field pattern of an ideal Gaussian beam.

The source is placed at the bottom of the computational cell, one pixel above the PML, and the beam is focused to a point 3 μm above the source plane. The `ContinuousSource` drives steady-state field patterns, making it straightforward to visualize the beam shape.

#### Code Walkthrough

The beam parameters define a vertically propagating (along +y) beam focused 3 μm above the source center:

```python
beam_x0 = mp.Vector3(0, 3.0)   # focus 3 um above source
beam_kdir = mp.Vector3(0, 1, 0) # propagate along +y
beam_w0 = 0.8                   # waist radius in um
beam_E0 = mp.Vector3(0, 0, 1)  # z-polarized
```

The `GaussianBeamSource` wraps a `ContinuousSource` with the beam amplitude profile:

```python
sources = [mp.GaussianBeamSource(
    src=mp.ContinuousSource(fcen),
    center=mp.Vector3(0, -0.5*s + dpml + 1.0),
    size=mp.Vector3(s),
    beam_x0=beam_x0,
    beam_kdir=beam_kdir,
    beam_w0=beam_w0,
    beam_E0=beam_E0,
)]
```

After running for 20 meep time units (sufficient for steady state at fcen = 1), the Ez field is plotted over the cell interior (excluding PML regions) using `sim.plot2D`.

The `rot_angle = 0` parameter means the beam propagates straight along +y. By changing this angle and rotating `beam_kdir` about the z-axis, the beam can be tilted to any angle, as demonstrated in the test suite.

#### Key Takeaways

- `GaussianBeamSource` automatically computes the correct amplitude and phase profile for a focused Gaussian beam, including the Gouy phase and curvature.
- The `beam_x0` parameter specifies the focus position relative to the source center, not in global coordinates.
- `beam_kdir` accepts any normalized direction vector; use `.rotate(axis, angle)` for tilted beams.
- `ContinuousSource` gives steady-state fields suitable for visualization; use `GaussianSource` for broadband or time-domain analysis.
- A 2D simulation of a Gaussian beam is a reliable smoke test for source correctness — the beam maximum should appear at the specified focus location.

---

### 3. `antenna-radiation.py` — Radiation Pattern of a Dipole Antenna via Near-to-Far-Field Transform

**Physics:** Computes the 2D radiation pattern of electric dipole antennas polarized along Ex, Ey, and Ez using Meep's near-to-far-field transformation, benchmarked against analytic dipole formulas.
**Difficulty:** Intermediate
**Source:** `python/examples/antenna-radiation.py`
**Test Status:** FAIL (`np.trapz` removed in NumPy 2.x; replace with `np.trapezoid`)

#### Theory

An oscillating electric dipole with moment **p** = p₀ exp(-iωt) radiates an electromagnetic field whose time-averaged Poynting vector in the far field takes the form:

    S_r ∝ sin²θ / r²

where θ is the angle between the observation direction and the dipole axis. This is the characteristic "donut-shaped" radiation pattern with nulls along the dipole axis and a maximum in the equatorial plane perpendicular to the dipole.

In 2D (the xy plane), the radiation pattern of a z-polarized dipole (Ez source) is omnidirectional (constant in φ, the in-plane angle), while an x-polarized dipole (Ex source) has a sin²φ pattern and a y-polarized dipole (Ey source) has a cos²φ pattern. These simple closed-form results, derived in Balanis' "Antenna Theory: Analysis and Design," serve as the analytic benchmarks for this simulation.

The near-to-far-field transformation is based on the Huygens-Fresnel principle: given the tangential E and H fields on a closed surface S surrounding the source, the fields at any exterior point can be computed using the free-space Green's function:

    E_far(r) = -iωμ₀ ∫_S [J_s G - (J_s·∇')∇'G/k²] dS' + ∇×∫_S M_s G dS'

where J_s = n̂ × H and M_s = -n̂ × E are the equivalent surface current densities on S, and G is the free-space Green's function. Meep accumulates the DFT of the tangential near fields during the simulation and then evaluates this integral at user-specified far-field points.

The simulation uses a 2D cell with PML on all sides. A point dipole source at the center excites the near-field box. After the fields decay, the near-to-far transform extrapolates to a circle of radius 1000λ to obtain the far-field radiation pattern. The radial Poynting flux at each angle is computed from the cross product of the far-field E and H vectors.

#### Code Walkthrough

The near-to-far monitor is a closed box of four line segments surrounding the dipole, with sign weights ensuring outward surface normals:

```python
nearfield_box = sim.add_near2far(
    frequency, 0, 1,
    mp.Near2FarRegion(center=mp.Vector3(0,  0.5*cell_um), size=mp.Vector3(cell_um, 0)),
    mp.Near2FarRegion(center=mp.Vector3(0, -0.5*cell_um), size=mp.Vector3(cell_um, 0), weight=-1),
    mp.Near2FarRegion(center=mp.Vector3( 0.5*cell_um, 0), size=mp.Vector3(0, cell_um)),
    mp.Near2FarRegion(center=mp.Vector3(-0.5*cell_um, 0), size=mp.Vector3(0, cell_um), weight=-1),
)
```

The far fields are sampled point-by-point around a large circle. The `get_farfield` call returns [Ex, Ey, Ez, Hx, Hy, Hz] at the requested point:

```python
for i in range(NUM_POLAR):
    far_field = sim.get_farfield(n2f_mon,
        mp.Vector3(FARFIELD_RADIUS_UM * math.cos(polar_rad[i]),
                   FARFIELD_RADIUS_UM * math.sin(polar_rad[i]), 0),
        GREENCYL_TOL)
    e_field[i, :] = [far_field[j] for j in range(3)]
    h_field[i, :] = [far_field[j+3] for j in range(3)]
```

The radial Poynting flux is computed from the real part of E* × H:

```python
flux_x = np.real(np.conj(e_field[:,1])*h_field[:,2] - np.conj(e_field[:,2])*h_field[:,1])
flux_y = np.real(np.conj(e_field[:,2])*h_field[:,0] - np.conj(e_field[:,0])*h_field[:,2])
flux_r = np.sqrt(np.square(flux_x) + np.square(flux_y))
```

The fix for the NumPy 2.x compatibility issue: replace `np.trapz(...)` on line 177 with `np.trapezoid(...)`.

Symmetry is used to reduce computation: for an Ez dipole, both mirror planes have even parity; for Ex and Ey dipoles, one plane has odd parity due to the vector nature of the dipole field.

#### Key Takeaways

- `sim.add_near2far` with a closed contour of `Near2FarRegion` objects enables far-field computation without simulating the enormous far-field domain.
- Weights of +1 and -1 ensure outward normals are consistent — the sign convention follows the right-hand rule with the outward normal.
- `GREENCYL_TOL` controls the tolerance for the cylindrical Green's function used in 2D near-to-far transforms.
- The NumPy 2.x API change removes `np.trapz` in favor of `np.trapezoid`; this is a common portability issue in scientific Python code.
- Dipole radiation patterns are angle-dependent for in-plane polarizations but omnidirectional for the out-of-plane (Ez) component in 2D.

---

### 4. `antenna_pec_ground_plane.py` — Dipole Antenna Above a PEC Ground Plane

**Physics:** Computes the radiation pattern of a dipole antenna above a perfect electric conductor (PEC) ground plane, using the image charge method emulated with a two-source array and odd mirror symmetry.
**Difficulty:** Intermediate
**Source:** `python/examples/antenna_pec_ground_plane.py`
**Test Status:** PASS (128.2s)

#### Theory

A PEC ground plane satisfies the boundary condition E_tangential = 0 at its surface. By the method of images, a dipole at height h above a PEC ground plane is equivalent to a two-element array consisting of the original dipole and its image dipole at height -h on the other side of the plane, with the image having the appropriate sign (opposite for Ez, same for Hz, to satisfy the PEC boundary condition).

The far-field array factor for a two-element array separated by distance 2h along the y-axis, with current ratio -1, gives a radiation pattern proportional to:

    AF(θ) = sin(k_medium · h · cos θ)

where k_medium = 2πn/λ is the wavenumber in the surrounding medium of index n, and θ is measured from the y-axis (the direction perpendicular to the ground plane). The total radiation pattern is the product of the single-element pattern and the array factor squared:

    P(θ) ∝ |AF(θ)|² = sin²(k_medium · h · cos θ)

This analytic result, from Balanis Section 6.2 "Two-Element Array," provides the benchmark for this simulation.

The near-to-far-field transformation cannot accommodate infinite structures like a ground plane — the Huygens surface must enclose all sources and structures in a homogeneous medium. The workaround is elegant: place two dipoles at ±h with opposite amplitudes and use an odd mirror symmetry (phase = -1 for Ez) along the y = 0 plane. This odd symmetry enforces the PEC boundary condition (Ez = 0 at y = 0) without explicitly modeling the conductor. The near-to-far monitor encloses both dipoles.

The simulation uses a background medium of index n = 1.2, wavelength λ = 0.65 μm, and dipole height h = 1.25 μm. The radiation pattern is computed over the quarter circle [0, π/2] corresponding to the upper half-space above the (virtual) ground plane.

#### Code Walkthrough

The two-dipole setup creates an image pair with opposite amplitude for Ez polarization:

```python
sources = [
    mp.Source(src=mp.GaussianSource(frequency, fwidth=0.2*frequency),
              component=dipole_polarization, center=mp.Vector3(0, +ANTENNA_HEIGHT_UM)),
    mp.Source(src=mp.GaussianSource(frequency, fwidth=0.2*frequency),
              component=dipole_polarization, center=mp.Vector3(0, -ANTENNA_HEIGHT_UM),
              amplitude=-1.0 if dipole_polarization == mp.Ez else +1.0),
]
```

The odd mirror symmetry for Ez enforces the PEC condition:

```python
if dipole_polarization == mp.Ez:
    symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y, phase=-1)]
```

Far-field samples are taken on a quarter circle from θ = 0 (along +y, perpendicular to ground plane) to θ = π/2 (along +x, parallel to ground plane):

```python
far_field = sim.get_farfield(n2f_mon,
    mp.Vector3(FARFIELD_RADIUS_UM * math.sin(polar_rad[i]),
               FARFIELD_RADIUS_UM * math.cos(polar_rad[i]), 0),
    GREENCYL_TOL)
```

The analytic array factor benchmark is computed as:

```python
k_free_space = 2 * math.pi / (WAVELENGTH_UM / N_BACKGROUND)
radial_flux_analytic = np.sin(k_free_space * ANTENNA_HEIGHT_UM * np.cos(polar_rad))**2
```

#### Key Takeaways

- A PEC ground plane can be modeled without explicitly placing a metallic boundary by using two image sources with the correct sign and an odd mirror symmetry.
- The near-to-far-field transform only works in homogeneous background media; the image-source technique extends it to half-space geometries.
- The array factor sin²(kh cos θ) predicts lobes in the radiation pattern at angles where k·h·cos θ = π/2 + nπ.
- `mp.stop_when_dft_decayed()` terminates the simulation when the DFT monitors have converged, more efficiently than a fixed run time.
- The background medium index affects the wavenumber k_medium and therefore shifts the lobe positions; using N_BACKGROUND = 1.2 is important for comparing with the analytic formula.

---

### 5. `antenna_pec_ground_plane_1D.py` — Antenna Ground Plane Radiation via Brillouin-Zone Integration

**Physics:** Computes the radiation pattern of a dipole above a PEC ground plane using a 1D Brillouin-zone integration approach, where each angle of the radiation pattern corresponds to one 1D simulation with a specific k-point.
**Difficulty:** Advanced
**Source:** `python/examples/antenna_pec_ground_plane_1D.py`
**Test Status:** TIMEOUT (CPU-intensive: NUM_POLAR = 50 separate simulations)

#### Theory

The Brillouin-zone integration (also called the Fourier modal method or planewave decomposition) provides an alternative to the near-to-far-field transform for computing radiation patterns. The key insight is that the radiation emitted at polar angle θ from a 1D periodic (or here, translationally invariant) structure corresponds exactly to a specific Bloch wavevector k_parallel = k sin θ. By running a separate simulation for each k_parallel and measuring the upward flux, one directly maps out the angular emission spectrum.

For a dipole above a PEC ground plane, the 1D geometry (periodic in x and y, finite in z) allows very fast simulations because the cell has zero transverse extent. Each simulation sets the Bloch wavevector:

    k_x = n_background · f · sin(θ)
    k_z = n_background · f · cos(θ)

where f = 1/λ is the frequency. The simulation measures the upward z-flux at a monitor point above the dipole. The angular flux is then:

    P(θ) = cos(θ) · Φ_z(k_x, k_y, k_z)

where the cos(θ) factor converts flux density to angular power. This technique is exact for structures with continuous translational symmetry and avoids the need for a large far-field radius.

Waves near the light cone (k_parallel → k_total, i.e., θ → 90°) are poorly absorbed by PML because they become nearly parallel to the interface. The code skips wavevectors with k_x > 0.95 · n · f to avoid this numerical artifact.

The simulation uses a 1D cell (`cell_size = mp.Vector3(0, 0, size_z_um)`) with PML on the +z side and a Metallic boundary on the -z side (the ground plane). The dipole is placed at height ANTENNA_HEIGHT_UM above the metallic boundary.

#### Code Walkthrough

The wavevector for angle θ is:

```python
def planewave_wavevector(polar_rad: float):
    kx = N_BACKGROUND * frequency * np.sin(polar_rad)
    ky = 0
    kz = N_BACKGROUND * frequency * np.cos(polar_rad)
    return kx, ky, kz
```

Each 1D simulation assigns this k-point and measures the upward flux:

```python
sim = mp.Simulation(
    resolution=RESOLUTION_UM,
    default_material=mp.Medium(index=N_BACKGROUND),
    cell_size=cell_size,
    sources=sources,
    boundary_layers=pml_layers,
    k_point=mp.Vector3(kx, ky, kz),
)
sim.set_boundary(mp.Low, mp.Z, mp.Metallic)  # PEC ground plane
```

The angular flux is cos(θ)-weighted to account for the solid angle element:

```python
radial_flux_meep[i] = np.cos(polar_rad[i]) * flux_z
```

Normalization uses the analytically computed maximum lobe angle:

```python
polar_rad_max = math.acos(9*math.pi / (2*k_free_space*ANTENNA_HEIGHT_UM))
```

#### Key Takeaways

- Brillouin-zone integration maps each far-field angle to one periodic simulation, enabling accurate angular spectra with 1D unit cells.
- Setting `k_point` in a 1D cell implements the Bloch-periodic boundary conditions corresponding to a specific planewave propagation direction.
- `sim.set_boundary(mp.Low, mp.Z, mp.Metallic)` places a Metallic (PEC) boundary at the bottom of the z domain, modeling the ground plane directly.
- Near-grazing angles (θ → 90°) require special treatment because PML absorbs them inefficiently; the code skips k_x > 0.95 · k as a practical cutoff.
- This approach scales as O(NUM_POLAR) independent simulations but each is fast because the cell is 1D; it complements near-to-far transforms for large computational cells.

---

### 6. `cherenkov-radiation.py` — Cherenkov Radiation from a Moving Charge

**Physics:** Simulates a point charge moving through a dielectric medium at superluminal phase velocity, generating a Cherenkov radiation cone in the Hz field.
**Difficulty:** Beginner
**Source:** `python/examples/cherenkov-radiation.py`
**Test Status:** FAIL (requires `h5topng` to generate PNG images from HDF5 field data; install via `sudo apt install h5utils`)

#### Theory

Cherenkov radiation is electromagnetic radiation emitted when a charged particle moves through a dielectric medium faster than the phase velocity of light in that medium. If a particle moves with velocity v in a medium with refractive index n, the condition for Cherenkov emission is:

    v > c/n  (i.e., β = v/c > 1/n)

The radiation forms a coherent shock wave (analogous to a sonic boom) at a cone half-angle given by:

    cos θ_C = c/(nv) = 1/(nβ)

For the parameters in this simulation, v = 0.7c and n = 1.5, giving β = 0.7 and cos θ_C = 1/(1.5 × 0.7) = 1/1.05 ≈ 0.952, so θ_C ≈ 18°. The wavefront is a Mach cone with the charge at the apex.

The mechanism is constructive interference: the particle drives polarization oscillations along its path, and these oscillations radiate. When the particle moves faster than light in the medium, the radiation from different points along the path arrives in phase on the cone surface, producing a coherent wavefront. In the frequency domain, Cherenkov emission is characterized by a spectrum that rises with frequency until limited by dispersion or absorption, fundamentally different from the thermal spectrum of blackbody radiation.

This simulation is a beautiful demonstration of a dynamic source: the position of the point charge is updated at every time step, and the accumulated Hz field shows the conical wavefront building up as the charge traverses the 60×60 cell.

#### Code Walkthrough

The simulation uses a 2D cell filled with a dielectric of index 1.5, and a mirror symmetry along y to halve the computation:

```python
sim = mp.Simulation(
    resolution=10,
    cell_size=mp.Vector3(sx, sy, 0),
    default_material=mp.Medium(index=1.5),
    symmetries=[mp.Mirror(direction=mp.Y)],
    boundary_layers=pml_layers,
)
```

The key innovation is the `move_source` callback, which updates the source position at every time step:

```python
def move_source(sim):
    sim.change_sources([
        mp.Source(
            mp.ContinuousSource(frequency=1e-10),
            component=mp.Ex,
            center=mp.Vector3(-0.5*sx + dpml + v*sim.meep_time()),
        )
    ])
```

The frequency `1e-10` is effectively zero (DC), making this an impulsive point source. The charge starts at the left edge of the cell and moves at speed v = 0.7 (in Meep's units where c = 1). The simulation runs until the charge exits the right side: `until=sx/v`.

Field snapshots are saved every 2 meep time units as PNG images via the `h5topng` tool, creating a movie of the Cherenkov cone forming.

#### Key Takeaways

- `sim.change_sources()` within a step function enables dynamic source positions, simulating moving charges or antennas.
- A DC-frequency (`frequency=1e-10`) `ContinuousSource` acts as a monopole charge-like excitation.
- The Cherenkov cone angle is determined solely by v and n: cos θ_C = 1/(nβ).
- `mp.output_png` calls the external `h5topng` tool to convert HDF5 field data to images; this requires the `h5utils` package.
- The simulation illustrates how constructive interference of radiation from a supersonic source creates a coherent wavefront, a general wave physics principle applicable to acoustics as well.

---

### 7. `differential_cross_section.py` — Differential Scattering Cross Section of a Sphere

**Physics:** Computes the differential scattering cross section dσ/dΩ of a dielectric sphere for a circularly polarized incident plane wave, using near-to-far-field projection onto a hemisphere, and benchmarks the integrated result against Mie theory.
**Difficulty:** Advanced
**Source:** `python/examples/differential_cross_section.py`
**Test Status:** TIMEOUT (CPU-intensive 3D simulation)

#### Theory

The differential scattering cross section dσ/dΩ describes how much power is scattered into a unit solid angle in a given direction:

    dσ/dΩ = r² (dP/dΩ) / I₀

where I₀ is the incident intensity, r is the observation distance, and dP/dΩ is the scattered power per unit solid angle. The total scattering cross section is obtained by integrating over the full sphere:

    σ_sca = ∫ (dσ/dΩ) dΩ = ∫₀^π ∫₀^{2π} (dσ/dΩ) sin θ dθ dφ

In this simulation with azimuthal symmetry (the sphere is isotropic, the incident beam travels along x), the integral reduces to:

    σ_sca = 2π ∫₀^π (dσ/dΩ)(θ) sin θ dθ

The scattered Poynting flux at a far-field point r on the hemisphere gives dσ/dΩ at that angle:

    dσ/dΩ = r² |S_sca(r)| / I₀ = r² P_r(θ) / I₀

This simulation uses a circularly polarized incident beam (Ez + i·Ey) to excite both polarizations simultaneously. The near-to-far monitor records the near fields on a closed box. After subtracting the incident field (via `load_minus_near2far_data`), the far fields on a semi-circle in the xz plane are computed at 100 points from θ = 0 (forward scattering) to θ = π (backscattering). The resulting dσ/dΩ is numerically integrated to obtain σ_sca.

#### Code Walkthrough

The circularly polarized source uses two coincident sources with a 90° phase shift:

```python
sources = [
    mp.Source(mp.GaussianSource(frq_cen, fwidth=0.2*frq_cen, is_integrated=True),
              center=mp.Vector3(-0.5*s + dpml), size=mp.Vector3(0, s, s), component=mp.Ez),
    mp.Source(mp.GaussianSource(frq_cen, fwidth=0.2*frq_cen, is_integrated=True),
              center=mp.Vector3(-0.5*s + dpml), size=mp.Vector3(0, s, s),
              component=mp.Ey, amplitude=1j),
]
```

After the background and sphere runs, the scattered near fields are stored and the far fields are sampled on a semicircle in the xz plane:

```python
ff = sim.get_farfield(nearfield_box,
    ff_r * mp.Vector3(np.cos(angles[n]), 0, np.sin(angles[n])))
```

The radial Poynting flux at each angle gives the differential cross section:

```python
Pr = np.sqrt(np.square(Px) + np.square(Py) + np.square(Pz))
intensity = input_flux / (4*r)**2
diff_cross_section = ff_r**2 * Pr / intensity
```

The total cross section is numerically integrated using the midpoint rule in sin θ dθ:

```python
scatt_cross_section_meep = (
    2 * np.pi * np.sum(np.multiply(diff_cross_section, np.sin(angles))) * np.pi/npts
)
```

The PyMieScatt benchmark uses `asCrossSection=True` to return σ in μm²:

```python
scatt_cross_section_theory = ps.MieQ(n_sphere, 1000/frq_cen, 2*r*1000,
    asDict=True, asCrossSection=True)["Csca"] * 1e-6  # um^2
```

#### Key Takeaways

- The differential cross section dσ/dΩ = r² |S|/I₀ contains the full angular information about scattering and reduces to σ_sca when integrated over the sphere.
- `load_minus_near2far_data` subtracts the incident near fields from the total near fields, analogous to `load_minus_flux_data` for the scattered field approach.
- Circular polarization excites all Mie modes simultaneously, but the resulting far-field intensity retains azimuthal symmetry for an isotropic sphere.
- Numerical integration on the far-field semicircle requires the sin θ Jacobian factor from the spherical coordinate solid angle element dΩ = sin θ dθ dφ.
- Agreement between Meep and Mie theory validates both the near-to-far transform and the scattered-field subtraction for 3D geometries.

---

### 8. `cyl-ellipsoid.py` — Scattering from a Cylinder with Embedded Ellipsoid

**Physics:** Simulates the time-domain field evolution inside a dielectric cylinder with an embedded ellipsoidal inclusion, driven by a Gaussian source at the center.
**Difficulty:** Beginner
**Source:** `python/examples/cyl-ellipsoid.py`
**Test Status:** PASS (117.8s)

#### Theory

When a wave source is placed inside a dielectric cylinder, the fields undergo multiple reflections and transmissions at the curved cylindrical boundary and at the ellipsoidal inclusion. The fields inside are a superposition of the incident field from the source and all scattered waves. This example is primarily a geometry demonstration showing how Meep handles overlapping objects (a cylinder of radius 3 with index 3.5 and an embedded ellipsoid of semi-axes 0.5 × 1 with unity index), but it also serves as a regression test by comparing the field value at a specific off-center point against a reference value.

The ellipsoid removes material from the interior of the cylinder (since it appears later in the geometry list and uses the default medium). Meep uses the principle that the last object in the geometry list takes precedence at any given point, allowing complex composite geometries to be built up from simple primitives.

Hz symmetry (both mirror planes with phase = -1) reduces the 2D simulation cost by a factor of 4. The source at the origin and the symmetric geometry ensure that both mirror symmetries are respected when src_cmpt = Hz.

The simulation provides a concrete example of field probing at an arbitrary off-axis point using `get_field_point`, which interpolates from the Yee grid to the requested location.

#### Code Walkthrough

The geometry uses object precedence: the ellipsoid is placed second, overriding the cylinder at overlapping points:

```python
c = mp.Cylinder(radius=3, material=mp.Medium(index=3.5))
e = mp.Ellipsoid(size=mp.Vector3(1, 2, mp.inf))
```

Mirror symmetry for Hz requires odd parity (phase = -1) on both axes because Hz is a pseudovector that changes sign under reflection:

```python
symmetries = [mp.Mirror(mp.X, -1), mp.Mirror(mp.Y, -1)]
```

A step function prints the field value at a specific probe point every 0.25 time units:

```python
def print_stuff(sim_obj):
    v = mp.Vector3(4.13, 3.75, 0)
    p = sim.get_field_point(src_cmpt, v)
    print(f"t, Ez: {sim.round_time()} {p.real}+{p.imag}i")
```

At the end of the run, the epsilon distribution and Ez field are saved to HDF5:

```python
sim.run(
    mp.at_beginning(mp.output_epsilon),
    mp.at_every(0.25, print_stuff),
    mp.at_end(mp.output_efield_z),
    until=23,
)
```

#### Key Takeaways

- Later objects in the geometry list take precedence over earlier ones, enabling complex shapes through Boolean-like subtraction.
- `mp.Ellipsoid` uses semi-axis sizes; `mp.Vector3(1, 2, mp.inf)` gives an infinite cylinder with elliptical cross-section (semi-axes 0.5 × 1).
- `sim.get_field_point(component, location)` interpolates the Yee-grid field to an arbitrary point for monitoring and regression testing.
- Hz has odd (pseudovector) symmetry under mirror reflections; using `phase=-1` on both axes is required for an Hz source at the symmetry point.
- `mp.output_epsilon` and `mp.output_efield_z` at specific simulation steps create HDF5 output files suitable for visualization with h5topng or Python.

---

### 9. `cylinder_cross_section.py` — Scattering Cross Section of a Dielectric Cylinder

**Physics:** Computes the broadband scattering cross section of a finite dielectric cylinder using the cylindrical coordinate (CYLINDRICAL) solver, applying the scattered-field subtraction technique with a circularly polarized source.
**Difficulty:** Intermediate
**Source:** `python/examples/cylinder_cross_section.py`
**Test Status:** PASS (94.2s)

#### Theory

The scattering cross section of a finite cylinder (radius r = 0.7, height h = 2.3, index n = 2.0) cannot be expressed in a simple closed-form like Mie theory for spheres; it requires numerical computation. However, the measurement technique is identical: a closed surface of DFT flux monitors surrounds the cylinder, the incident flux is subtracted (scattered-field approach), and the net outward flux divided by the incident intensity gives the scattering cross section.

The key physical distinction from the sphere case is the use of cylindrical coordinates (r, φ, z) with azimuthal mode m = -1. The circularly polarized source (Er - i·Eφ in cylindrical coordinates) efficiently excites the m = ±1 azimuthal modes, which couple to a plane wave. In Meep's CYLINDRICAL solver, the m-quantum number specifies the exp(imφ) angular dependence, and m = -1 corresponds to the left-circularly polarized component of a plane wave.

The use of cylindrical coordinates reduces what would be a 3D problem to an effectively 2D problem in the (r, z) plane, dramatically reducing computational cost. The simulation cell extends from r = 0 to r = r + dair + dpml in the radial direction and from z = -(sz/2) to z = +(sz/2) in the axial direction, with PML on all exterior boundaries. The cylinder is centered at the origin.

The incident intensity is computed from the flux through the bottom face of the monitoring box in the background run. The scattering cross section has units of area (μm²) and represents the effective absorbing area the cylinder presents to the incident plane wave.

#### Code Walkthrough

The CYLINDRICAL solver is specified in the Simulation constructor with the azimuthal mode m = -1:

```python
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution,
    sources=sources,
    dimensions=mp.CYLINDRICAL,
    m=-1,
)
```

The circularly polarized source uses two cylindrical components with a 90° phase difference:

```python
sources = [
    mp.Source(mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
              component=mp.Er, center=..., size=mp.Vector3(sr)),
    mp.Source(mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
              component=mp.Ep, center=..., size=mp.Vector3(sr), amplitude=-1j),
]
```

Three flux monitors form the closed surface: bottom cap, top cap, and radial side wall:

```python
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5*r, 0, -0.5*h), size=mp.Vector3(r)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5*r, 0, +0.5*h), size=mp.Vector3(r)))
box_r  = sim.add_flux(frq_cen, dfrq, nfrq,
    mp.FluxRegion(center=mp.Vector3(r), size=mp.Vector3(z=h)))
```

The signed flux sum gives the net outward scattering:

```python
scatt_flux = np.asarray(box_z1_flux) - np.asarray(box_z2_flux) - np.asarray(box_r_flux)
```

#### Key Takeaways

- `dimensions=mp.CYLINDRICAL` reduces a 3D rotational problem to 2D in the (r, z) plane, with m specifying the azimuthal symmetry.
- m = -1 corresponds to left-circular polarization; the combination of Er and Ep sources with -i amplitude ratio creates a circularly polarized excitation.
- The scattered-field subtraction works identically in cylindrical coordinates: run without the object first, save flux data, then run with the object and subtract.
- In CYLINDRICAL coordinates, flux monitors in the (r, z) plane represent ring integrals over 2π, so the volume factor is automatically included.
- The scattering cross section plotted versus 2πr/λ on a log-log scale reveals resonances analogous to Mie resonances but modified by the finite height of the cylinder.

---

### 10. `dipole_in_vacuum_1D.py` — 3D Dipole Radiation via 1D Brillouin-Zone Integration

**Physics:** Computes the 3D radiation pattern of an electric dipole in vacuum using Brillouin-zone integration: each direction in the hemisphere corresponds to one 1D simulation with a specific Bloch wavevector, and the results are accumulated over both polar and azimuthal angles.
**Difficulty:** Advanced
**Source:** `python/examples/dipole_in_vacuum_1D.py`
**Test Status:** FAIL (requires command-line argument: `python dipole_in_vacuum_1D.py x` or `... y`)

#### Theory

The Brillouin-zone integration method for computing radiation patterns exploits the fact that in a periodic (or homogeneous) medium, the field of a point dipole can be decomposed into planewaves characterized by their transverse wavevector k_parallel. A 1D simulation with Bloch boundary conditions k_point = (k_x, k_y, k_z) selects exactly one Fourier component of the dipole radiation.

For a z-polarized dipole (Ex or Ey), the angular distribution of the Poynting flux measured at a monitor above the dipole directly gives the angular power spectrum. The z-component of flux at wavevector (k_x, k_y, k_z) with k_z = sqrt(k² - k_x² - k_y²) gives the power radiated toward angle (θ, φ) where:

    k_x = (ω/c) sin θ cos φ
    k_y = (ω/c) sin θ sin φ
    k_z = (ω/c) cos θ

The radial flux on the hemisphere is then P_r(θ, φ) = cos θ · Φ_z(k_x, k_y), accounting for the projection of the z-directed flux onto the outward radial direction.

For an x-polarized dipole, the expected radiation pattern (from classical electrodynamics) in the upper hemisphere is:

    U_x(θ, φ) ∝ 1 - sin²θ cos²φ

where the null is along the dipole axis (+x). For a y-polarized dipole:

    U_y(θ, φ) ∝ 1 - sin²θ sin²φ

The script requires NUM_POLAR × NUM_AZIMUTH = 30 × 50 = 1500 separate 1D simulations to map out the full hemisphere, which is computationally intensive. The results are saved to `dipole_radiation_pattern.npz` for subsequent visualization by `plot_radiation_pattern_dipole.py`.

#### Code Walkthrough

The script is invoked with a command-line argument specifying the dipole polarization:

```python
parser = argparse.ArgumentParser()
parser.add_argument("dipole_pol", type=str, choices=["x", "y"])
args = parser.parse_args()
```

Each (θ, φ) combination maps to a specific wavevector:

```python
for i in range(NUM_POLAR):
    for j in range(NUM_AZIMUTH):
        rx, ry, rz = spherical_to_cartesian(polar_rad[i], azimuth_rad[j])
        kx = frequency * rx
        ky = frequency * ry
        kz = frequency * rz
        flux_z = planewave_in_vacuum(args.dipole_pol, kx, ky, kz)
        radial_flux[i, j] = rz * flux_z  # cos(theta) projection
```

Near-grazing angles are skipped to avoid PML inefficiency:

```python
if np.sqrt(kx**2 + ky**2) > (0.95 * frequency):
    continue
```

Each 1D simulation uses a single point source in a 1D cell with bilateral PML:

```python
sim = mp.Simulation(
    resolution=RESOLUTION_UM,
    cell_size=mp.Vector3(0, 0, size_z_um),
    sources=sources,
    boundary_layers=[mp.PML(pml_um, direction=mp.Z)],
    k_point=mp.Vector3(kx, ky, kz),
)
```

#### Key Takeaways

- 3D radiation patterns can be computed from many fast 1D simulations, one per (k_x, k_y) pair, using Brillouin-zone integration.
- The radial flux is the z-flux multiplied by cos θ = k_z/k to convert from flux density to angular power.
- Near-grazing wavevectors (k_parallel → k) are excluded because PML absorption efficiency degrades for nearly parallel waves.
- Results are saved to `.npz` for subsequent visualization, demonstrating good workflow design (compute then plot separately).
- This approach scales as O(N_polar × N_azimuth) simulations but each is negligibly cheap in 1D; the method is exact for homogeneous backgrounds.

---

### 11. `dipole_in_vacuum_cyl_on_axis.py` — On-Axis Dipole Radiation Pattern in Cylindrical Coordinates

**Physics:** Computes the radiation pattern of an on-axis (r = 0) electric dipole in vacuum using Meep's CYLINDRICAL solver, verifying the cos²θ and sin²θ angular distributions for x- and z-polarized dipoles respectively.
**Difficulty:** Advanced
**Source:** `python/examples/dipole_in_vacuum_cyl_on_axis.py`
**Test Status:** FAIL (requires command-line argument: `python dipole_in_vacuum_cyl_on_axis.py x` or `... z`)

#### Theory

A dipole at the origin of cylindrical coordinates (r = 0) has a special property: its fields decompose into a small number of azimuthal modes m. A z-polarized dipole is azimuthally symmetric (m = 0), while an x-polarized dipole excites both m = +1 and m = -1 modes. This decomposition allows each mode to be solved in a 2D (r, z) plane, with the full 3D result obtained by superposition.

For a z-polarized dipole, the far-field radiation pattern in the rz plane is:

    P_z(θ) ∝ sin²θ

with nulls along the z-axis (θ = 0, π) and maximum in the equatorial plane (θ = π/2). The pattern is symmetric around the azimuthal angle φ.

For an x-polarized dipole, the pattern in the xz plane (φ = 0) is:

    P_x(θ) ∝ cos²θ

with maximum along the z-axis and null in the equatorial plane. This corresponds to the radiation emitted in the direction of the dipole axis, which has zero intensity (the dipole radiates perpendicularly to its axis).

The x-polarized dipole in cylindrical coordinates is constructed by superposing left (m = +1) and right (m = -1) circularly polarized dipoles:

    J_x = (J_{+1} + J_{-1}) / 2

where J_m are dipoles with azimuthal quantum number m. Each simulation with a specific m contributes half of the total field. There is a known numerical issue (GitHub issue #2704) where an Er source exactly at r = 0 requires a small offset; the code places it at r = 1.5/resolution.

#### Code Walkthrough

The two simulations for the x-polarized dipole use m = +1 and m = -1 separately, then superpose:

```python
if args.dipole_pol == "x":
    e_field, h_field = dipole_in_vacuum("x", +1)
    e_field_total += 0.5 * e_field * cmath.exp(1j * AZIMUTHAL_RAD)
    e_field, h_field = dipole_in_vacuum("x", -1)
    e_field_total += 0.5 * e_field * cmath.exp(-1j * AZIMUTHAL_RAD)
```

Each CYLINDRICAL simulation specifies the m quantum number:

```python
sim = mp.Simulation(
    resolution=RESOLUTION_UM,
    cell_size=cell_size,
    dimensions=mp.CYLINDRICAL,
    m=m,
    boundary_layers=boundary_layers,
    sources=sources,
    force_complex_fields=True,
)
```

The near-to-far monitor covers a C-shaped surface (top cap, side wall, bottom cap with weight -1):

```python
nearfields_monitor = sim.add_near2far(
    frequency, 0, 1,
    mp.FluxRegion(center=mp.Vector3(0.5*sr, 0, 0.5*sz), size=mp.Vector3(sr, 0, 0)),
    mp.FluxRegion(center=mp.Vector3(sr, 0, 0), size=mp.Vector3(0, 0, sz)),
    mp.FluxRegion(center=mp.Vector3(0.5*sr, 0, -0.5*sz), size=mp.Vector3(sr, 0, 0),
                  weight=-1.0),
)
```

The analytic comparison uses cos²θ for x-polarized and sin²θ for z-polarized dipoles:

```python
if dipole_pol == "x":
    dipole_radial_flux = np.square(np.cos(polar_rad))
else:
    dipole_radial_flux = np.square(np.sin(polar_rad))
```

#### Key Takeaways

- On-axis dipoles (r = 0) in CYLINDRICAL coordinates decompose cleanly into azimuthal modes: m = 0 for z-polarized, m = ±1 for transverse (x or y) polarized.
- The Er source at r = 0 requires a small offset due to a known coordinate singularity; this is a practical numerical detail to be aware of.
- `force_complex_fields=True` is required when using azimuthal modes m ≠ 0, because the exp(imφ) factor makes the fields inherently complex.
- Superposition of left and right circular modes reconstructs the linearly polarized dipole field at azimuthal angle φ = AZIMUTHAL_RAD.
- The radiation pattern relative error compared to the analytic formula serves as a quantitative validation metric, printed by `plot_radiation_pattern`.

---

### 12. `dipole_in_vacuum_cyl_off_axis.py` — Off-Axis Dipole Radiation via Cylindrical Fourier-Series Expansion

**Physics:** Computes the radiation pattern of an off-axis (r > 0) electric dipole in vacuum by expanding the dipole field as a Fourier series in the azimuthal angle φ, summing over m modes until convergence.
**Difficulty:** Advanced
**Source:** `python/examples/dipole_in_vacuum_cyl_off_axis.py`
**Test Status:** FAIL (requires two command-line arguments: `python dipole_in_vacuum_cyl_off_axis.py x 3.5`)

#### Theory

An off-axis dipole breaks the azimuthal symmetry of the cylindrical coordinate system. Unlike an on-axis dipole that requires only m = 0 or m = ±1, an off-axis dipole at radius r₀ > 0 generates a ring current (when integrated over φ) that excites infinitely many azimuthal modes m. The field of a point dipole at position (r₀, φ₀, z₀) can be expanded as:

    E(r, φ, z) = Σ_{m=-∞}^{∞} E_m(r, z) exp(imφ)

Each mode E_m satisfies a 2D problem in the (r, z) plane with the source being a ring current of radius r₀. The convergence of this Fourier series depends on the offset distance r₀; larger offsets require more terms (larger m_max).

The simulation iterates over m = 0, 1, 2, ... until the power in mode m falls below a threshold fraction of the maximum:

    P_m / P_max < POWER_DECAY_THRESHOLD = 1e-4

For each m, two simulations are run: one at frequency +f (positive frequency) and one at -f (negative frequency). The positive and negative frequency components contribute differently to the x and y polarization patterns. This is necessary because the off-axis dipole is modeled as a combination of ring currents at positive and negative frequencies to properly reconstruct the physical x or y polarization.

The expected radiation pattern for an x-polarized dipole observed in the φ = 0 plane has the same cos²θ dependence as the on-axis case. For a y-polarized dipole in the φ = 0 plane, the pattern is constant (independent of θ). This is because, at φ = 0, the y-polarized dipole's radiation is purely transverse-magnetic with uniform amplitude.

#### Code Walkthrough

Each m mode runs two simulations at ±frequency using `force_complex_fields=True`:

```python
sources = [
    mp.Source(src=mp.GaussianSource(frequency, fwidth=0.1*frequency),
              component=src_cmpt, center=mp.Vector3(dipole_pos_r, 0, 0)),
    mp.Source(src=mp.GaussianSource(-frequency, fwidth=0.1*frequency),
              component=src_cmpt, center=mp.Vector3(dipole_pos_r, 0, 0)),
]
```

The convergence loop accumulates fields until power decay:

```python
m = 0
while True:
    (e_field_plus, h_field_plus, e_field_minus, h_field_minus) = \
        dipole_in_vacuum(args.dipole_pol, args.dipole_pos_r, m)
    e_field_total += e_field_plus * cmath.exp(1j * m * AZIMUTHAL_RAD)
    if m > 0:
        e_field_total += np.conj(e_field_minus) * cmath.exp(-1j * m * AZIMUTHAL_RAD)
    power_decay = flux / flux_max
    if m > 0 and power_decay < POWER_DECAY_THRESHOLD:
        break
    m += 1
```

The total flux from far fields uses the spherical surface integral:

```python
flux = 2*math.pi * FARFIELD_RADIUS_UM**2 * np.trapezoid(
    dipole_radiation_pattern * np.sin(polar_rad), polar_rad)
```

Note the use of `np.trapezoid` (NumPy 2.x compatible) rather than the deprecated `np.trapz`.

#### Key Takeaways

- Off-axis dipoles in CYLINDRICAL coordinates require summing over many m modes; convergence rate depends on the ratio r₀/λ (more terms needed for larger offset).
- Positive and negative frequency sources together allow reconstruction of real-valued dipole fields while maintaining the complex field representation.
- The Fourier series approach is exact (given enough modes) and more efficient than full 3D simulation, especially for rotationally near-symmetric structures.
- `np.trapezoid` is the NumPy 2.x replacement for `np.trapz`; this script uses it correctly, unlike `antenna-radiation.py`.
- The radiation pattern independence from dipole position (for a homogeneous medium) is a sanity check: the pattern shape should match the on-axis result regardless of r₀.

---

### 13. `point_dipole_cyl.py` — Dipole Extraction Efficiency in a Dielectric Slab

**Physics:** Computes the light extraction efficiency (fraction of total emitted power that escapes into air) of a dipole emitter inside a dielectric slab, using the CYLINDRICAL solver with Fourier-series decomposition over azimuthal modes m.
**Difficulty:** Advanced
**Source:** `python/examples/point_dipole_cyl.py`
**Test Status:** TIMEOUT (CPU-intensive: multiple dipole positions × multiple m modes per position)

#### Theory

Light extraction efficiency η is a critical figure of merit for solid-state light emitters (LEDs, single-photon sources). For an emitter inside a high-index slab (n = 2.4 here), total internal reflection traps most emitted light: only light emitted within the escape cone (angle θ < arcsin(1/n) ≈ 24.6° for n = 2.4) can escape. The extraction efficiency is:

    η = P_rad / P_total

where P_rad is the power escaping into air above and below the slab, and P_total is the total power emitted by the dipole. P_total is measured via the Local Density of States (LDOS), which accounts for the Purcell effect — the slab modifies the dipole emission rate relative to free space.

The LDOS is accessed through the `dft_ldos` step function, which computes F_ldos · J_ldos* where F is the electric field at the dipole position and J is the dipole current density. The total emitted power is:

    P_total = -Re(F · J*) · V_cell

where V_cell is the volume associated with the point source (a ring in cylindrical coordinates: V = 2πr₀/resolution²).

The radiated power P_rad is measured by two DFT flux monitors: a disk above the slab (collecting upward emission) and a cylindrical side wall (collecting radiation that escapes laterally, though for a sufficiently large cell this is negligible within the padding region).

For a dipole at r = 0 (on-axis), only m = ±1 is needed. For a dipole at r > 0, a Fourier series in m is required with convergence monitored by the ratio of mode m power to maximum power over all modes.

#### Code Walkthrough

The slab geometry is a Block occupying the bottom of the cell:

```python
geometry = [mp.Block(
    material=mp.Medium(index=N_SLAB),
    center=mp.Vector3(0, 0, -0.5*size_z + 0.5*SLAB_THICKNESS_UM),
    size=mp.Vector3(mp.inf, mp.inf, SLAB_THICKNESS_UM),
)]
```

The LDOS is computed during the simulation via a step function:

```python
sim.run(
    mp.dft_ldos(frequency, 0, 1),
    until_after_sources=mp.stop_when_dft_decayed(tol=flux_decay_threshold),
)
```

After the run, the total power is extracted from the LDOS data:

```python
delta_vol = 2 * np.pi * rpos_um / (RESOLUTION_UM**2)
source_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * delta_vol
```

For off-axis dipoles, the Fourier series is summed until convergence:

```python
while True:
    radiated_flux, source_flux = dipole_in_slab(dipole_height, rpos_um, m)
    radiated_flux_total += radiated_flux * (1 if m == 0 else 2)
    source_flux_total   += source_flux   * (1 if m == 0 else 2)
    if m > 0 and (radiated_flux / radiated_flux_max) < flux_decay_threshold:
        break
    m += 1
```

The factor of 2 for m > 0 accounts for the m and -m modes contributing equally to the total.

#### Key Takeaways

- `mp.dft_ldos` computes the Local Density of States, giving the total power emitted by a dipole including Purcell enhancement from the slab.
- The ring-current volume element in cylindrical coordinates is V = 2πr₀/(resolution²), required to convert LDOS flux to physical power.
- Extraction efficiency η = P_rad/P_total is bounded by Snell's law: for index n, η_max ≈ 1 - sqrt(1 - 1/n²)/2 for an isotropic emitter.
- The m = 0 mode is counted once while m > 0 modes are counted twice (for ±m symmetry), a consistent convention throughout the loop.
- `mp.stop_when_dft_decayed` provides an adaptive runtime that terminates when the DFT monitors have converged to within the specified tolerance.

---

### 14. `plot_radiation_pattern_dipole.py` — 3D Radiation Pattern Visualization Tool

**Physics:** Auxiliary visualization script that loads precomputed radiation pattern data from `dipole_in_vacuum_1D.py` and generates both polar plots (for a fixed azimuthal plane) and 2D contour plots of the full hemispheric radiation pattern.
**Difficulty:** Beginner
**Source:** `python/examples/plot_radiation_pattern_dipole.py`
**Test Status:** FAIL (requires `dipole_radiation_pattern.npz` output file from `dipole_in_vacuum_1D.py`)

#### Theory

The radiation pattern of an electric dipole in vacuum is one of the most fundamental results in classical electrodynamics. For an x-polarized dipole, the angular distribution of time-averaged Poynting vector flux on a sphere of radius r is:

    U_x(θ, φ) ∝ 1 - sin²θ cos²φ

In the φ = 0 plane (xz plane), this becomes U_x ∝ cos²θ, with maxima at θ = 0 (along the z-axis, perpendicular to the dipole) and zero at θ = π/2 in the x-direction (along the dipole axis). For a y-polarized dipole, the pattern is:

    U_y(θ, φ) ∝ 1 - sin²θ sin²φ

In the φ = 0 plane, this gives a constant pattern U_y = 1, because at φ = 0 the observation is always perpendicular to the y-axis dipole.

The 3D visualization uses tricontourf in the (x, y) plane of the hemisphere, where the coordinates are:

    x = sin θ cos φ
    y = sin θ sin φ

Each point corresponds to a direction on the hemisphere, with the radial distance from the origin representing the radiation intensity at that angle. The plot allows direct comparison between the Meep numerical result and the analytic pattern.

This script demonstrates good scientific computing practice: separating the expensive computation (dipole_in_vacuum_1D.py) from the visualization (this script), enabling rapid iteration on plot aesthetics without rerunning simulations.

#### Code Walkthrough

The script loads the precomputed data from the NPZ file:

```python
data = np.load("dipole_radiation_pattern.npz")
dipole_pol = data["dipole_pol"]
radial_flux = data["radial_flux"]
```

The 3D contour plot projects the hemisphere onto the (x, y) plane using polar-to-Cartesian conversion:

```python
x = np.sin(polar_rad[:, np.newaxis]) * np.cos(azimuth_rad)
y = np.sin(polar_rad[:, np.newaxis]) * np.sin(azimuth_rad)
normalized_radial_flux = radial_flux / np.max(radial_flux)
```

The analytic pattern for comparison:

```python
if dipole_pol == "x":
    analytic_radial_flux = np.sin(np.arccos(x))**2  # = 1 - x^2 in disguise
```

The φ = 0 slice is extracted by taking the first azimuthal index:

```python
zero_azimuth_idx = 0
normalized_radial_flux = radial_flux[:, zero_azimuth_idx] / np.max(radial_flux[:, zero_azimuth_idx])
```

Relative error between Meep and theory is printed as a quantitative validation metric:

```python
relative_error = np.linalg.norm(normalized_radial_flux - dipole_radial_flux) / \
                 np.linalg.norm(dipole_radial_flux)
```

#### Key Takeaways

- Decoupling computation from visualization (`npz` save/load pattern) enables efficient iteration and reproducibility.
- `tricontourf` handles irregularly spaced data in 2D, making it suitable for the polar-to-Cartesian projection of hemisphere data.
- The analytic formula sin²(arccos(x)) = sqrt(1 - x²) is equivalent to the geometric projection of the dipole pattern onto the hemisphere's top view.
- A relative error metric computed as ||Meep - analytic||/||analytic|| provides a single number summarizing the overall agreement quality.
- Nonzero-masking (`np.nonzero(normalized_radial_flux)`) removes the θ = 0 null from the polar plot, avoiding confusing gaps at the pattern center.

---

### 15. `test_gaussianbeam.py` — Gaussian Beam Focus Validation Test

**Physics:** Unit test verifying that Meep's `GaussianBeamSource` (both 2D and 3D variants) places maximum field intensity at the specified beam focus, for a beam tilted at -40°.
**Difficulty:** Beginner
**Source:** `python/tests/test_gaussianbeam.py`
**Test Status:** PASS (58.2s)

#### Theory

A converging Gaussian beam focused at a point x₀ should have its maximum field intensity at that focus location. The DFT field at the focus frequency fcen = 1 should satisfy:

    |E_z(x_focus)|² / max_{cell}(|E_z|²) > 0.98

This tight tolerance (98% of the maximum) verifies that: (1) the beam focus is correctly positioned, (2) there are no numerical artifacts creating spurious field maxima elsewhere in the cell, and (3) the phase profile of the source correctly focuses the beam.

The test uses a beam rotated -40° about the z-axis, which is more demanding than a straight beam because it exercises the full rotation machinery of `beam_kdir.rotate()` and requires the amplitude function to be evaluated on a source plane that is not aligned with the beam axis. The 2D variant (`GaussianBeam2DSource`) and 3D variant (`GaussianBeam3DSource`) are both tested.

The distinction between 2D and 3D beam sources lies in the transverse profile: the 2D version uses a 1D Gaussian cross-section (appropriate for a cylindrical beam with translational symmetry along z), while the 3D version uses a 2D Gaussian cross-section (appropriate for a fully focused 3D beam). In a 2D simulation, `GaussianBeam2DSource` is the physically correct choice.

#### Code Walkthrough

Both source types are tested with the same rotated beam parameters:

```python
def test_gaussian_beam(self):
    self.gaussian_beam(-40, mp.GaussianBeam2DSource)
    self.gaussian_beam(-40, mp.GaussianBeam3DSource)
```

The beam focus position must also be rotated to match the tilted propagation direction:

```python
beam_kdir = mp.Vector3(0, 1, 0).rotate(mp.Vector3(0, 0, 1), math.radians(rot_angle))
beam_x0 = beam_x0.rotate(mp.Vector3(0, 0, 1), math.radians(rot_angle))
```

After running until transient decays, the DFT Ez field is extracted and the focus location is identified by matching grid coordinates to the expected focus position within tolerance `tol = 0.05`:

```python
Ez_beam_x0 = Ez_cell[np.squeeze(idx_x)[0], np.squeeze(idx_y)[0]]
frac = np.abs(Ez_beam_x0)**2 / np.amax(np.abs(Ez_cell)**2)
self.assertGreater(frac, 0.98)
```

#### Key Takeaways

- The 98% intensity threshold is physically meaningful: a properly focused beam should dominate all other field values inside the computational cell.
- Tilted beams test the full rotation API: both `beam_kdir` and `beam_x0` must be rotated consistently to keep the focus at the correct absolute position.
- `GaussianBeam2DSource` vs `GaussianBeam3DSource` differ in their transverse profile (1D vs 2D Gaussian); always use the one matching your simulation's physical dimensionality.
- `sim.add_dft_fields` combined with `sim.get_dft_array` extracts the single-frequency spatial field pattern, ideal for validating steady-state beam properties.
- `sim.get_array_metadata(dft_cell=...)` returns the grid coordinates for the DFT array, enabling easy index lookups by physical position.

---

### 16. `test_antenna_radiation.py` — Near-to-Far Field Transform Validation Tests

**Physics:** Unit tests verifying (1) Poynting's theorem — that near-field flux, far-field circle flux, and far-field box flux all agree; and (2) that the PEC ground plane radiation pattern matches the analytic two-element array factor formula.
**Difficulty:** Intermediate
**Source:** `python/tests/test_antenna_radiation.py`
**Test Status:** PASS (156.3s)

#### Theory

The test `test_poynting_theorem` verifies energy conservation across different measurement surfaces. By Poynting's theorem, the time-averaged power flowing outward through any closed surface surrounding a source in a lossless medium is the same, independent of the surface shape and size (provided the surface is in the far field for the near-to-far-field transform). The test computes power through:

1. The near-field bounding box (direct DFT flux sum)
2. A far-field circle of radius r = 1000/f (using `get_farfield` point sampling)
3. A far-field square box of side 20/f (using `nearfield_box.flux()` surface integral)

All three should agree to within numerical discretization error. The tolerance here is `places=2` (agreement to 1%).

The test `test_pec_ground_plane` verifies the radiation pattern of a dipole above a PEC ground plane. The analytic result (Section 6.2 of Balanis) for a two-element Ez array with amplitude ratio -1 and separation 2h is:

    P(θ) ∝ P_free_space(θ) · [2 sin(k·h·cos θ)]²

where the factor [2 sin(...)]² is the array factor squared. The test computes both the free-space pattern (single dipole, no ground plane) and the PEC pattern (image dipole method), then verifies their ratio matches the array factor formula to within 2% (tol = 0.02 in L2 norm).

#### Code Walkthrough

The Poynting theorem test measures three flux values and asserts they agree:

```python
near_flux = mp.get_fluxes(flux_box)[0]
Pr = self.radial_flux(sim, nearfield_box, r)
far_flux_circle = 4 * np.sum(Pr) * 0.5*np.pi*r / len(Pr)
far_flux_square = nearfield_box.flux(mp.Y, mp.Volume(...), res_far)[0] - ...
self.assertAlmostEqual(near_flux, far_flux_circle, places=2)
self.assertAlmostEqual(far_flux_circle, far_flux_square, places=2)
```

The PEC ground plane test uses the two-source image method:

```python
sources = [
    mp.Source(..., center=mp.Vector3(0, +self.h), ...),
    mp.Source(..., center=mp.Vector3(0, -self.h), amplitude=-1 if src_cmpt==mp.Ez else +1),
]
```

The analytic array factor is computed and compared:

```python
k = 2*np.pi / (self.wvl / self.n)
for i, ang in enumerate(self.angles):
    Pr_theory[i] = Pr_fsp[i] * 2 * np.sin(k * self.h * np.cos(ang))
Pr_pec_norm   = Pr_pec / np.max(Pr_pec)
Pr_theory_norm = (Pr_theory / max(Pr_theory))**2
self.assertClose(Pr_pec_norm, Pr_theory_norm, epsilon=tol)
```

#### Key Takeaways

- Poynting's theorem provides a fundamental consistency check: near-field and far-field flux integrals must agree for any closed surface in a lossless medium.
- The far-field circle flux is computed by numerical quadrature of `get_farfield` samples; the far-field box uses `nearfield_box.flux()` for direct integration.
- The image dipole method effectively enforces PEC boundary conditions without explicit material modeling, enabling near-to-far-field transforms in half-space geometries.
- The array factor [2 sin(kh cos θ)]² arises from the coherent superposition of two dipole sources; its squared form appears in power (intensity) calculations.
- `assertClose` with `epsilon=0.02` checks L2 norm agreement, which is more tolerant of individual-angle deviations than requiring pointwise agreement.

---

### 17. `test_cyl_ellipsoid.py` — Cylinder-Ellipsoid Scattering Regression Test

**Physics:** Regression test that runs the cylinder-with-ellipsoid-inclusion simulation for both Ez and Hz source polarizations and verifies the field value at an off-axis probe point matches precomputed reference values.
**Difficulty:** Beginner
**Source:** `python/tests/test_cyl_ellipsoid.py`
**Test Status:** PASS (170.2s)

#### Theory

Regression tests in FDTD codes serve a critical role: they verify that changes to the codebase (optimizations, bug fixes, new features) do not alter the physical behavior of existing simulations. The cylinder-ellipsoid geometry involves:

1. Multiple geometric objects with different materials and precedence rules
2. Mirror symmetries with polarization-dependent parity assignments
3. Mixed dielectric interfaces where subpixel averaging matters
4. Time evolution over 23 Meep time units with field probing at a specific location

The reference values are:
- `ref_Ez = -8.29555720049629e-5` (real part of Ez at t = 23, point (4.13, 3.75, 0))
- `ref_Hz = -4.5623185899766e-5` (real part of Hz at same location and time)

These values were established when the simulation was first verified against expected physical behavior and have remained fixed since. The test passes if the computed value differs by less than 5% of the reference:

    |computed - reference| / |reference| < 0.05

The 5% tolerance accommodates minor changes in numerical precision across platforms and compiler versions while still catching any significant regression.

#### Code Walkthrough

The test inherits from `unittest.TestCase` and runs two polarizations:

```python
def test_ez_field(self):
    self.src_cmpt = mp.Ez
    self.init()
    self.run_simulation()

def test_hz_field(self):
    self.src_cmpt = mp.Hz
    self.init()
    self.run_simulation()
```

The comparison uses the actual field value from the Yee grid accessor:

```python
ref_out_field = self.ref_Ez if self.src_cmpt == mp.Ez else self.ref_Hz
out_field = self.sim.fields.get_field(self.src_cmpt, mp.vec(4.13, 3.75)).real
diff = abs(out_field - ref_out_field)
self.assertTrue(abs(diff) <= 0.05 * abs(ref_out_field), "Field output differs")
```

A temporary output directory is used for HDF5 files:

```python
@classmethod
def setUpClass(cls):
    cls.temp_dir = mp.make_output_directory()
@classmethod
def tearDownClass(cls):
    mp.delete_directory(cls.temp_dir)
```

#### Key Takeaways

- Regression tests with fixed reference values detect unintended physics changes across code versions; they are distinct from unit tests that verify analytic formulas.
- A 5% tolerance on field values is typical for FDTD regression tests, accounting for floating-point variation across platforms without masking genuine regressions.
- `sim.fields.get_field(component, mp.vec(...))` accesses the low-level C++ field value directly, bypassing Python-level interpolation.
- Temporary output directories (`mp.make_output_directory()` / `mp.delete_directory()`) prevent test pollution of the working directory.
- Running both Ez and Hz polarizations exercises the symmetry assignment logic and the two distinct field update paths in Meep's core FDTD loop.

---

### 18. `test_diffracted_planewave.py` — Diffracted Planewave Mode Decomposition Test

**Physics:** Verifies that two different methods for identifying diffraction orders in a binary grating — band number enumeration and explicit `DiffractedPlanewave` specification — give identical transmission coefficients, group velocities, and dominant wavevectors.
**Difficulty:** Advanced
**Source:** `python/tests/test_diffracted_planewave.py`
**Test Status:** PASS (134.9s)

#### Theory

A binary grating with period Λ and height h diffracts an incident planewave into discrete diffraction orders. For a wave incident at angle θ_in in a medium of index n_g, the grating equation gives the transverse wavevector of the m-th transmitted order:

    k_y^{(m)} = k_y^{inc} + m/Λ = (n_g f sin θ_in) + m/Λ

Orders are propagating if k_x^{(m)} = sqrt(f² - (k_y^{(m)})²) is real and evanescent otherwise. The number of propagating orders depends on the grating period Λ relative to the wavelength.

For this simulation: grating period Λ = 2.6 μm, height h = 0.4 μm, fill factor 60%, wavelength λ = 0.5 μm, glass substrate n_g = 1.5, tested at both normal incidence (θ = 0°) and oblique incidence (θ = 13.4°). The test verifies:

1. **Transmission coefficient** |α|²: the fraction of incident power in each diffraction order
2. **Group velocity** v_g: verifies dispersion relation consistency
3. **Dominant wavevector** k_dom: the wavevector of the mode at the center frequency

The `DiffractedPlanewave` object provides an explicit, physically intuitive way to specify diffraction orders: you give the (m_x, m_y, m_z) order integers, the grating vector direction, and the polarization. This is compared to the traditional approach of specifying band numbers via MPB eigensolve.

For normal incidence (θ = 0), the test uses `mp.Mirror(direction=mp.Y)` symmetry and `eig_parity = mp.ODD_Z + mp.EVEN_Y`, which restricts the mode search to the correct polarization subspace and eliminates degenerate modes. For oblique incidence, no such simplification is available and all modes are computed without symmetry.

#### Code Walkthrough

The planewave excitation uses an amplitude function to impart the correct phase at the source plane:

```python
def pw_amp(k, x0):
    def _pw_amp(x):
        return cmath.exp(1j * 2 * math.pi * k.dot(x + x0))
    return _pw_amp

sources = [mp.Source(mp.GaussianSource(fcen, fwidth=0.1*fcen),
    component=mp.Ez, center=src_pt, size=mp.Vector3(0, sy, 0),
    amp_func=pw_amp(k, src_pt))]
```

The mode decomposition for each diffraction order uses both methods:

```python
# Method 1: band number
res = sim.get_eigenmode_coefficients(tran_flux, [band+1], eig_parity=eig_parity)

# Method 2: DiffractedPlanewave
res = sim.get_eigenmode_coefficients(tran_flux,
    mp.DiffractedPlanewave((0, order, 0), mp.Vector3(0, 1, 0), 1, 0))
```

The ordering of MPB modes at oblique incidence follows decreasing k_x (closest to normal):

```python
kx = lambda m: np.power(fcen, 2) - np.power(k.y + m/gp, 2)
kxs = [kx(m) for m in ms]
ids = np.flip(np.argsort(kxs))
orders = [ms[d] for d in ids]
```

Agreement is verified at the 1e-4 level:

```python
self.assertAlmostEqual(tran_ref, tran_dp, places=4)
self.assertAlmostEqual(vg_ref, vg_dp, places=4)
```

#### Key Takeaways

- `DiffractedPlanewave` provides a physical, order-based specification for diffraction modes, eliminating the need to manually map band numbers to diffraction orders.
- The amplitude function `pw_amp(k, x0)` applies the planewave phase factor across the source plane, enabling oblique incidence by encoding the phase gradient directly.
- For normal incidence on a symmetric grating, mirror symmetry and parity quantum numbers (`ODD_Z + EVEN_Y`) drastically reduce the mode search space.
- Group velocity `vgrp` from `get_eigenmode_coefficients` provides a consistency check with the dispersion relation; it should match the analytic value k_x/|k| for a homogeneous medium mode.
- At oblique incidence, MPB orders modes by decreasing k_x (proximity to normal propagation), requiring explicit sorting to match physical diffraction order integers m.

---

*Chapter 3 complete. The 18 tutorials cover the full range of scattering and radiation phenomena accessible with Meep: from Mie cross-sections and differential scattering to antenna patterns, Cherenkov radiation, cylindrical coordinate dipole sources, and grating diffraction. Common themes throughout are the scattered-field subtraction technique, the near-to-far-field transform, Brillouin-zone integration as an alternative approach, and systematic validation against analytic benchmarks.*
