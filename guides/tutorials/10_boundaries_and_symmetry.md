# Chapter 10: Boundaries, Symmetry, and Coordinates

This chapter covers the computational infrastructure that makes FDTD simulations
tractable: absorbing boundary layers (PML), periodic and Bloch-periodic boundary
conditions, coordinate-system choices (1D, cylindrical), symmetry exploitation, and
the domain-decomposition machinery that distributes work across MPI processes. Each
tutorial section pairs a concrete test case with the underlying physics and algorithms,
so you understand not only *how* to use each Meep feature but *why* it produces correct
results.

---

### 1. `test_boundaries_1D.py` — Ground-Plane Antenna Radiation Pattern via 1D PML

**Physics:** A dipole source above a perfect electric conductor (PEC) ground plane
radiates into a background medium; its radiation pattern is validated against the
analytic two-element array factor.
**Difficulty:** Intermediate
**Source:** `python/tests/test_boundaries_1D.py`
**Test Status:** PASS (106.4 s)

#### Theory

A 1D Meep simulation collapses the transverse dimensions to zero, retaining only one
spatial axis (here Z). Periodic Bloch boundary conditions in X and Y encode an
in-plane wavevector component k_x and k_y, allowing the simulation to model a
planewave that propagates at an oblique angle through the 1D grid. In Meep units where
c = 1, the dispersion relation in a medium of index n is

    k_x^2 + k_y^2 + k_z^2 = (n * f)^2

where f is the frequency. Fixing (k_x, k_y) via `k_point` and integrating forward in
time effectively samples the radiation pattern of the source at the corresponding
solid angle.

The PEC ground plane at z = -L/2 is enforced with `sim.set_boundary(mp.Low, mp.Z,
mp.Metallic)`. The boundary condition E_tangential = 0 means the ground plane acts as a
perfect mirror. By the method of images, a dipole at height h above a PEC ground plane
is equivalent to a two-element array consisting of the real dipole and its image dipole
at distance 2h below, radiating with opposite phase. The far-field array factor for
this configuration (Balanis, Section 6.2) is

    AF(theta) = sin(k * h * cos(theta))

where k = 2*pi*n/lambda is the wave number in the background medium. The radial power
flux in the far field is proportional to |AF(theta)|^2.

The PML at z = +L/2 absorbs outgoing radiation without reflection. The simulation
sweeps over several polar angles by relaunching with different (k_x, k_y) values.
Because PML absorption degrades for waves traveling nearly parallel to the absorber
interface (glancing-angle incidence), wavevectors with k_x > 0.95 * n * f are skipped.
The radiated flux is measured with a DFT flux monitor just inside the PML.

The test normalizes each measured flux by the flux at the angle of maximum radiation
(computed analytically from the array factor) and checks that the normalized Meep
values match the analytic |AF|^2 to within 1% absolute error.

#### Code Walkthrough

**Setup — 1D cell with PML on one side only:**

```python
size_z_um = self.bulk_um + self.pml_um          # 11 um total
pml_layers = [mp.PML(thickness=self.pml_um, direction=mp.Z, side=mp.High)]
cell_size = mp.Vector3(0, 0, size_z_um)          # 1D along Z
```

Setting `cell_size.x = 0` and `cell_size.y = 0` instructs Meep to use a 1D simulation.
The PML only appears on the `mp.High` (positive-Z) side; the low side is explicitly set
to metallic.

**Encoding angle via Bloch k-vector:**

```python
kx = self.n_background * self.frequency * np.sin(polar_rad)
ky = 0
kz = self.n_background * self.frequency * np.cos(polar_rad)
# ...
sim = mp.Simulation(..., k_point=mp.Vector3(kx, ky, kz))
```

The `k_point` sets the Bloch phase factor e^(i k_transverse . r) applied to the
transverse dimensions. Even though the cell has zero transverse extent, Meep uses this
factor to correctly update the fields with the oblique-incidence phase relationship.

**Applying boundary conditions:**

```python
sim.set_boundary(mp.Low, mp.Z, mp.Metallic)
sim.set_boundary(mp.High, mp.Z, mp.Metallic)
```

`mp.Metallic` enforces E_tangential = 0 (a PEC wall). Although the PML already
attenuates the field on the high side, explicitly setting both boundaries to metallic
ensures no stray fields leak through in edge cases. The PML is placed inside the cell
and handles absorption before the field reaches the metallic wall.

**DFT flux monitor and stopping criterion:**

```python
dft_flux_z = sim.add_flux(
    self.frequency, 0, 1,
    mp.FluxRegion(center=mon_pt, size=mp.Vector3(), direction=mp.Z),
)
sim.run(until_after_sources=mp.stop_when_fields_decayed(
    self.field_decay_period, src_cmpt, mon_pt, self.field_decay_threshold
))
flux_z = mp.get_fluxes(dft_flux_z)[0]
```

`mp.FluxRegion` with `size=mp.Vector3()` measures flux at a single point (dimensionally
consistent with the 1D geometry). The run continues until the field amplitude has
decayed by six orders of magnitude, ensuring that all transient contributions have
radiated away and the DFT has converged.

**Radiation pattern comparison:**

```python
radial_flux_meep = (math.cos(polar_rad) * flux_z) / radial_flux_meep_max
radial_flux_analytic = math.sin(
    k_free_space * self.antenna_height_um * math.cos(polar_rad)
) ** 2
self.assertAlmostEqual(radial_flux_meep, radial_flux_analytic, delta=0.01)
```

The factor `cos(polar_rad)` converts the z-directed Poynting flux to the radial
component in spherical coordinates. Normalization by the reference flux at the
analytically predicted maximum angle removes the dependence on source amplitude.

#### Key Takeaways

- A 1D Meep cell (`cell_size.x = cell_size.y = 0`) combined with `k_point` allows
  simulation of oblique planewaves at negligible computational cost.
- One-sided PML (`direction=mp.Z, side=mp.High`) is the correct choice when the
  opposite boundary is a physical PEC mirror, not an absorber.
- `sim.set_boundary(side, direction, mp.Metallic)` enforces tangential E = 0 and is
  distinct from the default zero-padding boundary; it should be called before
  `sim.run()` but after constructing the simulation object.
- Glancing-angle waves (k_transverse near k_total) are poorly absorbed by PML and
  must be filtered out or handled with a thicker absorber.
- The method of images converts a ground-plane problem into a free-space two-element
  array problem, providing a clean analytic reference for numerical validation.

---

### 2. `test_planewave_1D.py` — Phase Verification of 1D and 3D Planewave DFT Fields

**Physics:** A monochromatic planewave propagating in vacuum acquires a deterministic
complex phase e^(i k.r) between any two spatial positions; this test verifies that
Meep's DFT field arrays and point-query API both reproduce this phase relationship to
within numerical error.
**Difficulty:** Intermediate
**Source:** `python/tests/test_planewave_1D.py`
**Test Status:** TIMEOUT (CPU-intensive; very high resolution 400 pixels/um)

#### Theory

For a time-harmonic planewave with wavevector **k** = (k_x, k_y, k_z), the electric
field at position **r** and frequency f is

    E(**r**, f) = E_0 * exp(i * 2*pi * **k** . **r**)

where in vacuum |**k**| = f (in Meep units). The relative phase between two points r_1
and r_2 along the propagation axis is therefore

    phi = exp(i * 2*pi * k_z * (z_2 - z_1))

This is the simplest possible test of the complex DFT machinery: a single-frequency
source drives a sinusoidal wave, and after the transient dies out, the DFT accumulates
the steady-state complex amplitude at each grid point. Both the ratio of two single-
point DFT values and the corresponding elements of a full-domain DFT array slice should
reproduce the exact analytic phase with high accuracy.

In a 1D simulation (only Z extent), the k_point must be zero or the planewave must be
normally incident (polar_rad = 0). Setting `dimensions=1` reduces the problem to a
1D wave equation, which runs far faster than the equivalent 3D simulation and allows
the extremely high resolution of 400 pixels/um needed to keep dispersion error below
5%. In 3D, oblique incidence is encoded through `k_point = mp.Vector3(k_x, k_y, 0)`
with the z-component of k handled automatically by the source position.

The test also validates that Meep's two field-query mechanisms — `get_dft_array` (which
returns an array over the entire monitor volume) and `get_dft_array` evaluated at a
zero-size monitor (a single-point query) — agree with each other. This cross-check
detects interpolation bugs in the Yee-grid field averaging.

The `yee_grid` parameter controls whether DFT fields are accumulated at the staggered
Yee-grid positions (each component offset by half a grid cell) or at the centered grid
positions (obtained by averaging neighboring Yee cells). Both representations should
give the same phase relationship, though the absolute positions differ by half a pixel.

#### Code Walkthrough

**Choosing simulation dimensionality:**

```python
if cell_dim == 1:
    k_point = False
    dimensions = 1
elif cell_dim == 3:
    k_point = mp.Vector3(kx, ky, 0)
    dimensions = 3
```

In 1D mode, `k_point=False` disables Bloch phases; the wavevector is entirely along Z.
In 3D mode, the transverse components are supplied via `k_point`.

**DFT monitors for phase comparison:**

```python
dft_fields = sim.add_dft_fields(
    [mp.Ex], frequency, 0, 1,
    center=mp.Vector3(0, 0, 0), size=mp.Vector3(0, 0, size_z_um),
    yee_grid=yee_grid,
)
dft_fields_1 = sim.add_dft_fields(
    [mp.Ex], frequency, 0, 1,
    center=mp.Vector3(0, 0, z_pos_1), size=mp.Vector3(0, 0, 0),
    yee_grid=yee_grid,
)
```

The first monitor spans the entire Z extent; the second and third are point monitors
at specific positions. After the simulation runs, `get_dft_array` extracts the complex
amplitude arrays.

**Phase validation logic:**

```python
# From the full array:
z_idx_1 = np.argmin(np.abs(z_pos_1 - z_um))
meep_phase = ex_dft[z_idx_2] / ex_dft[z_idx_1]
expected_phase = np.exp(1j * 2 * np.pi * kz * (z_um[z_idx_2] - z_um[z_idx_1]))
self.assertAlmostEqual(meep_phase, expected_phase, delta=0.05)

# From the point monitors:
meep_phase = ex_dft_pos_2 / ex_dft_pos_1
expected_phase = np.exp(1j * 2 * np.pi * kz * (z_pos_2 - z_pos_1))
self.assertAlmostEqual(meep_phase, expected_phase, delta=0.05)
```

Dividing two complex DFT values cancels the unknown source amplitude, leaving only the
pure phase factor. The 5% tolerance accounts for finite-difference dispersion error at
the chosen resolution.

**Consistency between array and point queries:**

```python
self.assertAlmostEqual(ex_dft[z_idx_1], ex_dft_pos_1, delta=0.08)
self.assertAlmostEqual(ex_dft[z_idx_2], ex_dft_pos_2, delta=0.08)
```

The looser tolerance (8%) here acknowledges that `z_idx_1` is the nearest grid point
to `z_pos_1`, not exactly `z_pos_1`, so a small interpolation error is expected.

#### Key Takeaways

- `add_dft_fields` with `size=mp.Vector3(0, 0, 0)` acts as a single-point DFT
  monitor, equivalent to a continuous-time Fourier transform at one location.
- `force_complex_fields=True` is required to observe complex DFT amplitudes in 1D
  simulations, where the real-valued field representation would discard the imaginary
  part.
- The `yee_grid=True` option preserves the staggered field positions required for
  accurate interpolation in post-processing workflows that expect Yee-aligned data.
- Very high resolution (400 pixels/um at lambda = 1 um) is needed to keep finite-
  difference dispersion below the 5% acceptance threshold; this is why the test times
  out on a standard CI machine.
- The ratio of two DFT complex amplitudes is the most robust way to extract
  phase information because it is independent of the absolute source amplitude and any
  global phase offset introduced by the simulation start time.

---

### 3. `test_pml_cyl.py` — PML Correctness in Cylindrical Coordinates

**Physics:** Electromagnetic waves launched by a source in cylindrical coordinates
(r, phi, z) must be properly absorbed by both the r-direction and z-direction PMLs;
this test verifies that the total radiated flux converges to a constant value long
after the source turns off, confirming that no spurious reflections or field growth
occurs near r = 0 for various azimuthal mode numbers m.
**Difficulty:** Advanced
**Source:** `python/tests/test_pml_cyl.py`
**Test Status:** FAIL (missing `parameterized` package; physics is sound)

#### Theory

Meep's cylindrical coordinate system (selected with `dimensions=mp.CYLINDRICAL`)
exploits the azimuthal symmetry of rotationally symmetric structures by expanding the
fields as e^(i*m*phi), where m is an integer (or half-integer for some spinor
representations). This reduces a 3D problem to a sequence of 2D problems in the (r, z)
plane, each labeled by m. The full 3D field is recovered by summing over m.

The cylindrical FDTD grid has a singular axis at r = 0. Standard Yee-grid field
updates must be modified near the origin to respect the regularity conditions:

- For m = 0: E_phi and H_r must vanish at r = 0; E_z and H_phi satisfy a special
  finite-difference stencil using L'Hopital's rule.
- For |m| = 1: E_r and E_phi are coupled at r = 0 through the angular momentum
  structure of Maxwell's equations.
- For |m| > 1: all tangential fields vanish at r = 0.

The `accurate_fields_near_cylorigin=True` option activates a more expensive but more
accurate field-update algorithm near r = 0 for |m| > 1, and requires a reduced Courant
number to maintain numerical stability:

    Courant = 1 / (|m| + 0.6)

The PML in cylindrical coordinates is implemented as an anisotropic complex coordinate
stretch. In Cartesian coordinates, the standard UPML (Uniaxial PML) modifies
Maxwell's equations by replacing the spatial coordinate x in the absorbing region with
a complex stretched coordinate

    x -> x + (sigma(x) / i*omega) * x

where sigma(x) is a conductivity profile that grows from zero at the PML interface to
a maximum at the outer boundary. In cylindrical coordinates, the r-directed PML applies
an analogous stretch in the r direction, while the z-directed PML stretches along z.
The challenge is that the cylindrical Laplacian contains 1/r factors that must be
handled carefully in the PML region; Meep's implementation correctly handles these
singular terms.

The correctness criterion is energy conservation: after the source has turned off and
all outgoing waves have been absorbed, the total DFT flux summed over a closed surface
(+z face, +r face, -z face) should remain constant. Any spurious reflection from the
PML or field growth at the axis would cause this total to drift. The test checks
stability over multiple simulation times (50.94, 142.15, 214.64, 365.32 time units
after source shutoff) at 7-significant-figure precision.

#### Code Walkthrough

**Cylindrical simulation setup:**

```python
sim = mp.Simulation(
    resolution=self.resolution,
    cell_size=self.cell_size,     # mp.Vector3(s + dpml_r, 0, s + 2*dpml_z)
    dimensions=mp.CYLINDRICAL,
    m=m,                           # azimuthal mode number
    sources=sources,
    boundary_layers=pml_layers,
    accurate_fields_near_cylorigin=accurate_fields_near_cylorigin,
)
```

The `cell_size` has zero Y extent (the phi dimension is not resolved explicitly) and
the X extent corresponds to the radial coordinate r. The PML list includes both an
r-directed absorber (`direction=mp.R`) and a z-directed absorber (`direction=mp.Z`).

**Reduced Courant number for high-m modes:**

```python
if accurate_fields_near_cylorigin and abs(m) > 1:
    sim.Courant = 1 / (abs(m) + 0.6)
```

The default Courant number for 2D FDTD is 0.5. For high azimuthal order m, the
near-origin field updates involve higher-order finite differences that tighten the
Courant-Friedrichs-Lewy (CFL) stability condition.

**Three-surface flux closure:**

```python
flux_plus_z  = sim.add_flux(...)   # +z face (weight=+1)
flux_plus_r  = sim.add_flux(...)   # +r face (weight=+1)
flux_minus_z = sim.add_flux(..., weight=-1.0)  # -z face (inward normal)
```

The total outgoing power is flux_plus_z + flux_plus_r + flux_minus_z. By Poynting's
theorem, once the source is off and all transients have decayed, this sum should
remain constant (no net power is being created or destroyed inside the surface).

**Temporal stability check:**

```python
for t in [142.15, 214.64, 365.32]:
    sim.run(until_after_sources=t)
    cur_flux = [mp.get_fluxes(f)[0] for f in [flux_plus_z, flux_plus_r, flux_minus_z]]
    for i in range(len(cur_flux)):
        self.assertAlmostEqual(prev_flux[i], cur_flux[i], places=7)
```

Each flux value is compared to the previous measurement. A 7-decimal-place tolerance
(or 6 in single-precision mode) means that even tiny numerical reflections or axis
instabilities would cause the test to fail.

#### Key Takeaways

- `dimensions=mp.CYLINDRICAL` with `m=m` converts a 3D rotationally symmetric problem
  into a 2D computation, reducing memory and CPU cost by roughly one order of magnitude.
- The `accurate_fields_near_cylorigin` flag should be enabled for |m| > 1 when
  precise near-axis fields are needed, at the cost of a smaller time step.
- PML in cylindrical coordinates requires specifying `direction=mp.R` for the radial
  absorber, not just a default PML thickness, because the radial and axial absorbers
  have different geometric implementations.
- Testing PML correctness via long-time flux stability (checking that DFT fluxes
  converge to fixed values after source shutoff) is more rigorous than checking field
  amplitudes alone.
- The `parameterized` package enables concise multi-case testing (m = 0, -1, 2, 3)
  without code duplication; its absence causes the test to fail at import time despite
  the underlying Meep physics being correct.

---

### 4. `test_special_kz.py` — Out-of-Plane Wavevector (kz_2d) in 2D Simulations

**Physics:** A 2D simulation can model a planewave propagating in 3D by setting a
fixed out-of-plane wavevector k_z; this test verifies that both the `"complex"` and
`"real/imag"` implementations of this feature reproduce the Fresnel reflectance formula
for P-polarized light at an air-dielectric interface.
**Difficulty:** Advanced
**Source:** `python/tests/test_special_kz.py`
**Test Status:** FAIL (missing `parameterized` package)

#### Theory

A fully 3D simulation of a planewave at oblique incidence is expensive because it
requires a 3D cell with Bloch-periodic boundary conditions in two transverse directions.
Meep's `kz_2d` feature allows a 2D simulation (cell extent only in x and y) to model
a wave with a nonzero z-component of the wavevector by treating k_z as a parameter
that modifies the field update equations.

Concretely, if the full 3D wavevector is **k** = (k_x, k_y, k_z), a 2D simulation
with `kz_2d` handles the k_z dependence analytically. Maxwell's equations are modified
by replacing every occurrence of the z-derivative d/dz with i*k_z (multiplication by
the imaginary unit times k_z), since all fields are assumed to vary as exp(i*k_z*z).
This turns the 3D wave equation into a modified 2D equation with extra terms
proportional to k_z^2.

There are two equivalent implementations selected by `kz_2d`:

- `"complex"`: the fields remain complex-valued and the exp(i*k_z*z) factor is carried
  implicitly. This is conceptually simpler but doubles memory usage compared to a real-
  valued 2D simulation.
- `"real/imag"`: the real and imaginary parts of the exp(i*k_z*z) factor are tracked as
  two separate coupled real-valued field sets. This avoids complex arithmetic in the
  inner time-stepping loop and is typically faster.

The test validates both implementations against the Fresnel reflectance formula for
P-polarization (TM polarization, where E lies in the plane of incidence). For an
incident angle theta_in in medium n1 = 1 (air) onto a dielectric with n2 = 3.5, the
transmitted angle theta_out satisfies Snell's law

    n1 * sin(theta_in) = n2 * sin(theta_out)

and the P-polarized Fresnel reflectance is

    R_P = |( n1*cos(theta_out) - n2*cos(theta_in) )
            / ( n1*cos(theta_out) + n2*cos(theta_in) )|^2

The second parameterized test (`test_eigsrc_kz`) verifies a more subtle property: when
`k_point=mp.Vector3(z=kz)` is set, the fields at any two points differing only in z
should satisfy the exact Bloch phase relationship E(z+d) / E(z) = exp(i*2*pi*kz*d).
This checks that the exp(i*k_z*z) phase is correctly applied throughout the simulation
volume, not just at the source.

#### Code Walkthrough

**Building the k-vector for a given angle:**

```python
k_point = mp.Vector3(1, 0, 0).rotate(mp.Vector3(0, 1, 0), theta).scale(fcen)
```

This creates a unit vector along X, rotates it by angle `theta` about the Y axis (so
it now points at theta from the X axis in the XZ plane), and scales it by the
frequency. The result is a k-vector in the XZ plane with magnitude `fcen`, encoding
incidence angle theta_in.

**Two-pass flux calculation:**

```python
# Empty run (no scatterer):
sim.run(...)
empty_flux = mp.get_fluxes(refl)
empty_data = sim.get_flux_data(refl)
sim.reset_meep()

# Structure run (with dielectric block):
sim = mp.Simulation(..., geometry=[...])
sim.load_minus_flux_data(refl, empty_data)
sim.run(...)
refl_flux = mp.get_fluxes(refl)
return -refl_flux[0] / empty_flux[0]
```

Loading the negative of the empty-run flux data and then summing with the structure-run
flux implements the standard Meep two-simulation technique for extracting the reflected
component. The incident wave cancels in the empty-minus-structure subtraction, leaving
only the scattered (reflected) field contribution.

**kz_2d comparison:**

```python
Rmeep_complex = self.refl_planar(theta, "complex")
Rmeep_real_imag = self.refl_planar(theta, "real/imag")
self.assertAlmostEqual(Rmeep_complex, Rfres, places=2)
self.assertAlmostEqual(Rmeep_real_imag, Rfres, places=2)
```

Both algorithms must agree with the analytic Fresnel result to 1% (2 decimal places).

**Phase verification for EigenModeSource:**

```python
ez1 = sim.get_field_point(mp.Ez, mp.Vector3(2.3, -5.7, 4.8))
ez2 = sim.get_field_point(mp.Ez, mp.Vector3(2.3, -5.7, 4.8 + d))
ratio_ez = ez2 / ez1
phase_diff = cmath.exp(1j * 2 * cmath.pi * kz * d)
self.assertAlmostEqual(ratio_ez.real, phase_diff.real, places=10)
```

The out-of-plane phase must match to 10 decimal places, confirming that the Bloch
factor is applied with floating-point precision throughout the grid.

#### Key Takeaways

- `kz_2d` allows a 2D Meep simulation to faithfully model a 3D planewave with a
  nonzero z-wavevector component, avoiding the cost of a full 3D simulation.
- The `"real/imag"` mode runs two coupled real field sets and is generally faster than
  `"complex"` mode on modern hardware, even though both give identical results.
- The two-simulation flux subtraction technique (load negative empty flux, run with
  structure, measure residual) is the standard Meep idiom for separating incident from
  reflected power.
- `EigenModeSource` with `eig_match_freq=True` finds the exact waveguide mode at the
  specified k_z, not just an approximation; the eigenmode solver accounts for the
  modified dispersion relation induced by k_z.
- Verifying the Bloch phase relationship at ten-decimal-place precision confirms that
  `k_point` is applied correctly not just in the time-stepping but also when
  extracting field values via `get_field_point`.

---

### 5. `test_binary_partition_utils.py` — Binary Partition Tree Utilities

**Physics:** Domain decomposition in parallel FDTD uses a binary space partition (BSP)
tree to divide the simulation cell into rectangular chunks assigned to MPI processes;
this test verifies the utility functions that query and manipulate these BSP trees.
**Difficulty:** Advanced (computational methods)
**Source:** `python/tests/test_binary_partition_utils.py`
**Test Status:** FAIL (missing `parameterized` package)

#### Theory

When Meep runs in MPI mode, the simulation cell must be divided among multiple
processes. Meep uses a binary space partition (BSP) tree to represent this division.
Each internal node of the tree stores:

- `split_dir`: the direction (X, Y, or Z) along which this node splits its region
- `split_pos`: the coordinate value of the dividing plane
- `left`, `right`: child subtrees for the two halves

Each leaf node stores a `proc_id` identifying which MPI process owns that chunk of the
simulation domain.

This representation is both flexible and efficient: the tree can be unbalanced (some
chunks much larger than others), it can contain non-axis-aligned splits (by nesting X
and Y splits), and individual chunks can be owned by the same process (allowing one
process to own multiple disjoint regions).

The utility functions in `meep.binary_partition_utils` operate on these trees:

- `is_leaf_node(partition)`: checks whether a node has no children.
- `enumerate_leaf_nodes(partition)`: depth-first generator over leaf nodes (the actual
  domain chunks).
- `partition_has_duplicate_proc_ids(partition)`: detects whether any process owns more
  than one chunk, which complicates load calculations.
- `get_total_weight(partition, weights)`: sums weights (e.g., timing data) over all
  processes in a subtree.
- `pixel_volume(grid_volume)`: computes the integer number of Yee cells in a chunk.
- `get_total_volume(partition, volumes, owners)`: total pixel count in a subtree.
- `get_box_ranges(partition, volumes, owners)`: the axis-aligned bounding box of a
  subtree, used to compute new split positions.
- `partitions_are_equal(bp1, bp2)`: structural comparison of two trees.

The test uses two fixed reference partitions: one with unique process IDs per leaf
(the normal case) and one with duplicate process IDs. The duplicate case arises when
the BSP tree has more leaves than MPI processes — one process can own multiple chunks.
Several utility functions raise `ValueError` for the duplicate case because the
weight accumulation logic assumes a bijection between leaf nodes and processes.

#### Code Walkthrough

**Defining test partitions:**

```python
PARTITION_NO_DUPLICATE_PROC_ID = mp.BinaryPartition(
    data=[(mp.X, -2.0), 0,
          [(mp.Y, 1.5),
              [(mp.X, 4.0), 1, [(mp.Y, 0.5), 4, 3]],
              2]]
)
```

This nested list syntax builds the BSP tree. Each entry is either an integer (a leaf
process ID) or a tuple `(direction, split_position)` followed by left and right
subtrees. The tree has 5 leaf nodes (processes 0, 1, 4, 3, 2 in depth-first order).

**Leaf enumeration:**

```python
leaf_nodes = list(bpu.enumerate_leaf_nodes(partition))
proc_ids = [node.proc_id for node in bpu.enumerate_leaf_nodes(partition)]
# Expected: [0, 1, 4, 3, 2]  (depth-first)
```

Depth-first order traverses the left subtree completely before the right subtree at
each node.

**Pixel volume computation:**

```python
sim = mp.Simulation(cell_size=mp.Vector3(10.0, 5.0, 0.0),
                    resolution=10, chunk_layout=partition)
sim.init_sim()
chunk_volumes = sim.structure.get_chunk_volumes()
volumes = [bpu.pixel_volume(vol) for vol in chunk_volumes]
# Expected: [1500, 2400, 300, 100, 700]
```

`get_chunk_volumes()` returns `grid_volume` objects. `pixel_volume` extracts `nx * ny`
for 2D or `nx * ny * nz` for 3D. The volumes are in units of Yee cells (pixels), not
physical area.

**Bounding box for a subtree:**

```python
box = bpu.get_box_ranges(partition.right, chunk_volumes, chunk_owners)
# Returns (xmin, xmax, ymin, ymax, zmin, zmax) for the right half
```

This is used by the chunk balancer to compute new split positions: if the right subtree
spans [x_min, x_max] and needs to contribute a fraction f of the total domain, the
new split position is x_min + f*(x_max - x_min).

#### Key Takeaways

- `mp.BinaryPartition(data=[...])` accepts a nested Python list that encodes the BSP
  tree, making it easy to specify custom domain decompositions without calling into C++.
- Duplicate process IDs in the BSP tree mean one MPI rank owns multiple non-contiguous
  chunks; this is valid but complicates weight-balanced operations and requires special
  handling in utility functions.
- `sim.structure.get_chunk_volumes()` and `sim.structure.get_chunk_owners()` are the
  two key methods for inspecting the actual domain decomposition after initialization.
- Depth-first traversal of a BSP tree is the natural order for matching leaf nodes to
  `get_chunk_volumes()` output, since both follow the same left-before-right convention.
- The `partitions_are_equal` function provides a structural comparison of two BSP trees
  (not just reference equality), which is essential for testing the chunk balancer's
  output against expected results.

---

### 6. `test_chunk_balancer.py` — Adaptive Load Balancing via Chunk Rebalancing

**Physics:** In heterogeneous simulations (e.g., photonic crystals where dispersive
material update equations are expensive), different regions of the domain take different
amounts of wall-clock time per FDTD step; adaptive chunk rebalancing adjusts BSP
split positions to equalize per-process simulation times and minimize the overall run
time.
**Difficulty:** Advanced (parallel computing)
**Source:** `python/tests/test_chunk_balancer.py`
**Test Status:** FAIL (missing `parameterized` package)

#### Theory

In an MPI-parallel FDTD simulation, the wall-clock time per step is determined by the
slowest process. If process 0 owns a region containing a dense dispersive material
(requiring many auxiliary field updates) while process 1 owns an empty air region, then
process 0 will finish much later than process 1 on each step, and process 1 will sit
idle waiting for synchronization. This load imbalance reduces parallel efficiency.

The remedy is to give the overloaded process a smaller spatial domain and the
underloaded process a larger one. The `ChunkBalancer` class implements this
adaptively using timing data from the previous simulation iteration.

The balancing algorithm works on the BSP tree, bottom-up. For each internal node that
splits at position `split_pos` along some direction, the algorithm:

1. Computes the total simulation time T_left for all processes in the left subtree
   and T_right for those in the right subtree.
2. Computes the current volumes V_left and V_right of the two subtrees.
3. Infers the compute load per unit volume as T / V for each side.
4. Computes a new split fraction such that the new volumes would equalize the total
   simulation times: if T_left/V_left > T_right/V_right, the left side is more
   expensive per pixel, so shrink it.

The new split position formula is:

    split_frac = (V_left * T_right * n_left) / (V_left * T_right * n_left + V_right * T_left * n_right)
    new_split_pos = d_min + split_frac * (d_max - d_min)

where n_left and n_right are the number of chunks in each subtree (accounting for the
fact that each process must receive a fair share of the total volume). A `sensitivity`
parameter in [0, 1] blends the new and old split positions:

    split_pos = sensitivity * new_split_pos + (1 - sensitivity) * old_split_pos

A sensitivity of 1.0 makes an instantaneous jump to the theoretically optimal split;
0.5 moves halfway each iteration, providing more stable convergence.

The `MockSimulation` class in the test mocks `time_spent_on` to return fake timing
data proportional to chunk volume, simulating a uniform-cost medium. Under this mock,
the optimal split is the equal-volume partition, and the test verifies convergence
toward equal volumes over 25 iterations.

#### Code Walkthrough

**MockSimulation for single-core testing:**

```python
class MockSimulation(mp.Simulation):
    def time_spent_on(self, time_sink):
        # Return times proportional to pixel volume per process
        volumes = [v.nx() * v.ny() for v in chunk_volumes]
        total_volume_by_proc = np.zeros(num_processes)
        for i, v in enumerate(volumes):
            total_volume_by_proc[chunk_owners[i]] += v
        if time_sink == TIMING_MEASUREMENT_IDS["time_stepping"]:
            return 1.0 * total_volume_by_proc
        else:
            return 0.0 * np.ones(num_processes)
```

By returning volume-proportional times, the mock tells the balancer that each Yee cell
is equally expensive to update, so the target outcome is equal-volume chunks.

**Single-iteration improvement test:**

```python
chunk_balancer.adjust_chunk_layout(test_sim, sensitivity=1.0)
# After rebalancing:
self.assertLess(new_max_time, old_max_time)   # worst process improved
self.assertGreater(new_min_time, old_min_time) # best process worsened (more work)
```

After one step with sensitivity=1.0, the maximum per-process time must decrease and
the minimum must increase, confirming that load is being redistributed.

**Convergence test:**

```python
for _ in range(25):
    chunk_balancer.adjust_chunk_layout(test_sim, sensitivity=0.5)

tolerance = 0.05
mean_step_time = np.mean(new_step_times)
self.assertTrue(np.allclose(mean_step_time, new_step_times, rtol=tolerance))
```

After 25 iterations with sensitivity=0.5, all per-process times must be within 5% of
the mean, confirming that the algorithm converges to load balance.

**Direct split-position test with known answers:**

```python
new_chunk_layout = chunk_balancer.compute_new_chunk_layout(
    timing_measurements, chunk_layout, chunk_volumes, chunk_owners, sensitivity=1.0
)
self.assertTrue(bpu.partitions_are_equal(new_chunk_layout, expected_chunk_layout))
```

`TEST_CHUNK_DATA_6` has deliberately unequal timing: process 0 takes 1500 units,
process 1 takes 2400, processes 2-4 take 300-700. The expected output split positions
are computed analytically and checked exactly.

#### Key Takeaways

- `ChunkBalancer.adjust_chunk_layout(sim, sensitivity)` is the single entry point for
  adaptive load rebalancing; it measures timing, computes new BSP split positions,
  resets the simulation, and re-initializes with the new chunk layout.
- The `sensitivity` parameter controls convergence speed vs. stability: use 1.0 for
  maximum correction speed (risks oscillation) or 0.5 for stable convergence over
  multiple iterations.
- The algorithm is purely proportional: it does not require the user to specify which
  regions are computationally expensive, only that they run a timed simulation first.
- `adjust_chunk_layout` calls `sim.reset_meep()` and `sim.init_sim()` internally,
  destroying and reconstructing the C++ data structures; any field state accumulated
  before the rebalancing call is lost.
- Simulations with duplicate process IDs in the BSP tree cannot be rebalanced because
  the per-subtree timing attribution is ambiguous; the validator raises `ValueError` in
  this case.

---

### 7. `test_chunk_layout.py` — Custom BSP Chunk Layout Verification

**Physics:** When a user manually specifies a BSP tree via `chunk_layout`, Meep must
assign simulation chunks to processes such that each chunk's area and owner match the
tree structure; this test verifies that the custom layout is applied correctly and that
Meep's default automatic layout also satisfies the same tree-consistency property.
**Difficulty:** Intermediate
**Source:** `python/tests/test_chunk_layout.py`
**Test Status:** PASS (5.0 s)

#### Theory

Meep normally computes its own domain decomposition automatically, using a simple
bisection algorithm that tries to equalize the number of Yee cells per process. This
default layout is usually adequate for uniform media but may be suboptimal for
geometries where different regions have very different computational costs (e.g., a
region containing dispersive metals near a region of empty air).

The `chunk_layout` parameter of `mp.Simulation` accepts either an existing `Simulation`
object (which copies that simulation's chunk layout) or an `mp.BinaryPartition` object
(the user's custom BSP tree). This allows:

1. **Reproducibility**: running two simulations with identical chunk layouts ensures
   that field accumulations (DFT monitors, flux subtraction) are numerically
   identical regardless of MPI process count.
2. **Custom load balancing**: a user who knows which regions are expensive can manually
   specify a non-uniform decomposition.
3. **Chunk balancer output**: the `ChunkBalancer` class computes an improved layout and
   passes it back to the simulation via this parameter.

The verification logic is implemented by a helper function `traverse_tree` that walks
the BSP tree manually, computes the rectangular area of each leaf chunk from the tree's
split positions and the cell's bounding box, and collects the expected process
assignments. These are then compared against Meep's actual chunk owners and volumes.

The test covers both cases:

1. A user-specified `mp.BinaryPartition` with 5 chunks and a non-trivial layout.
2. Meep's default chunk layout (no `chunk_layout` argument), accessed via
   `sim.chunk_layout` after `init_sim()`.

In both cases the tree traversal must produce chunk areas and owner assignments that
exactly match what `sim.structure.get_chunk_owners()` and `get_chunk_volumes()` return.

#### Code Walkthrough

**Tree traversal helper:**

```python
def traverse_tree(bp, min_corner, max_corner):
    process_ids = []
    chunk_areas = []

    def _traverse_tree(bp, min_corner, max_corner):
        if bp.left is None and bp.right is None:  # leaf
            process_ids.append(bp.proc_id)
            area = (max_corner.x - min_corner.x) * (max_corner.y - min_corner.y)
            chunk_areas.append(area)
        if bp.left is not None:
            new_max = copy.deepcopy(max_corner)
            if bp.split_dir == mp.X: new_max.x = bp.split_pos
            else:                    new_max.y = bp.split_pos
            _traverse_tree(bp.left, min_corner, new_max)
        if bp.right is not None:
            new_min = copy.deepcopy(min_corner)
            if bp.split_dir == mp.X: new_min.x = bp.split_pos
            else:                    new_min.y = bp.split_pos
            _traverse_tree(bp.right, new_min, max_corner)

    _traverse_tree(bp, min_corner, max_corner)
    return process_ids, chunk_areas
```

Each recursive call narrows the bounding box by moving the appropriate corner to the
split position. The depth-first traversal order matches Meep's internal chunk ordering.

**Custom layout test:**

```python
chunk_layout = mp.BinaryPartition(data=[
    (mp.X, -2.0), 0,
    [(mp.Y, 1.5), [(mp.X, 3.0), 1, [(mp.Y, -0.5), 4, 3]], 2],
])
sim = mp.Simulation(cell_size=mp.Vector3(10.0, 5.0, 0), resolution=10,
                    chunk_layout=chunk_layout)
sim.init_sim()

owners = sim.structure.get_chunk_owners()
areas  = [v.surroundings().full_volume() for v in sim.structure.get_chunk_volumes()]

process_ids, chunk_areas = traverse_tree(chunk_layout, -0.5*cell_size, 0.5*cell_size)

self.assertListEqual([int(f) for f in owners],
                     [f % mp.count_processors() for f in process_ids])
self.assertListEqual(areas, chunk_areas)
```

The `% mp.count_processors()` modulo ensures that on a single-core machine (where all
processes map to process 0), the comparison still passes.

**Default layout test:**

```python
sim = mp.Simulation(cell_size=cell_size, resolution=10)
sim.init_sim()
chunk_layout = sim.chunk_layout  # access the auto-generated layout
```

After `init_sim()`, `sim.chunk_layout` returns the BSP tree that Meep computed
automatically, allowing the user to inspect or modify it.

#### Key Takeaways

- `mp.BinaryPartition(data=[...])` provides a concise Python-native syntax for
  describing arbitrarily complex domain decompositions without writing C++ code.
- After `sim.init_sim()`, the `sim.chunk_layout` attribute is always populated (even
  without a user-specified layout), making it possible to inspect and modify the
  decomposition between simulation stages.
- The process owner modulo operation (`% mp.count_processors()`) is a necessary
  normalization when testing on machines with fewer processes than process IDs in the
  BSP tree.
- `v.surroundings().full_volume()` returns the geometric area (or volume) of a chunk's
  bounding box in simulation units, which equals the BSP-tree-computed area for
  rectangular chunks.
- Matching chunk areas (not just owners) between the tree traversal and Meep's internal
  representation ensures that split positions are being applied exactly, with no
  rounding or off-by-one errors at chunk boundaries.

---

### 8. `test_chunks.py` — Flux Conservation Across Chunk Boundaries

**Physics:** A point source in a 2D cell surrounded by PML radiates power that is
collected by a closed flux surface; the total radiated flux must be independent of how
the simulation cell is divided into chunks, verifying that inter-chunk communication
preserves the Poynting vector exactly.
**Difficulty:** Intermediate
**Source:** `python/tests/test_chunks.py`
**Test Status:** PASS (12.6 s)

#### Theory

In a parallel FDTD simulation, adjacent chunks share boundary fields that must be
communicated between processes at every time step. Specifically, the curl operations
in Maxwell's equations

    dB/dt = -curl(E)
    dD/dt = +curl(H)

require each chunk to access the tangential E (or H) fields on its neighbor's
boundary. In Meep, this "ghost zone" communication uses MPI blocking sends and receives
in `mympi.cpp`. If the ghost zone communication is incorrect, the fields will diverge
at chunk boundaries, producing spurious reflections and corrupted DFT flux values.

The test validates inter-chunk communication by measuring the total radiated power from
a central point source. By Poynting's theorem, for a lossless enclosed domain bounded
by PML, the integral of the Poynting flux over any closed surface surrounding the
source must equal the total power emitted by the source. PML absorbs all outgoing
radiation without reflection, so after the source shuts off, the closed-surface DFT
flux should converge to exactly the emitted power.

The test uses a closed surface composed of four `FluxRegion` objects (top, bottom,
right, left), with weights +1 for outward-normal faces and -1 for inward-normal faces.
Summing these gives the net outward flux, which must equal the source power.

The second simulation uses `split_chunks_evenly=False` and `chunk_layout=sim1`,
forcing the second run to use exactly the same chunk decomposition as the first. This
allows the flux from the first run (empty domain) to be subtracted from the second run
(domain with a surrounding dielectric ring) to isolate the scattering contribution.

#### Code Walkthrough

**Four-sided flux closure:**

```python
top = mp.FluxRegion(center=mp.Vector3(0, +0.5*sxy - dpml),
                    size=mp.Vector3(sxy - 2*dpml, 0), weight=+1.0)
bot = mp.FluxRegion(center=mp.Vector3(0, -0.5*sxy + dpml),
                    size=mp.Vector3(sxy - 2*dpml, 0), weight=-1.0)
rgt = mp.FluxRegion(center=mp.Vector3(+0.5*sxy - dpml, 0),
                    size=mp.Vector3(0, sxy - 2*dpml), weight=+1.0)
lft = mp.FluxRegion(center=mp.Vector3(-0.5*sxy + dpml, 0),
                    size=mp.Vector3(0, sxy - 2*dpml), weight=-1.0)
tot_flux = sim.add_flux(fcen, 0, 1, top, bot, rgt, lft, decimation_factor=1)
```

The `weight=-1.0` on the bottom and left faces reverses the sign of their Poynting
flux contribution so that all four faces together give the net outward flux.
`decimation_factor=1` stores every time step in the DFT accumulator (no subsampling).

**Copying chunk layout between simulations:**

```python
sim = mp.Simulation(
    cell_size=cell, geometry=geometry, boundary_layers=pml_layers,
    sources=sources, resolution=resolution,
    chunk_layout=sim1,  # copy layout from first simulation
)
```

Passing a `Simulation` object as `chunk_layout` copies the chunk decomposition exactly.
This is essential for the flux subtraction technique: if the two simulations have
different chunk layouts, the DFT arrays may be accumulated in different MPI process
orderings, making the subtraction produce incorrect results.

**Flux save/load for subtraction:**

```python
sim.save_flux("tot_flux", tot_flux)
# ...
sim.load_minus_flux("tot_flux", tot_flux)
```

`save_flux` writes the DFT flux array to an HDF5 file. `load_minus_flux` reads it
back and subtracts it from the new flux monitor, so the subsequent run accumulates
only the difference (scattered power from the geometry).

**Numerical validation:**

```python
self.assertAlmostEqual(86.90826609300862, mp.get_fluxes(tot_flux)[0], places=7)
```

The expected value is computed from a reference run and hard-coded. Seven-decimal-place
agreement confirms that both inter-chunk communication and DFT accumulation are exact
(up to floating-point precision) regardless of chunk boundaries.

#### Key Takeaways

- `split_chunks_evenly=False` instructs Meep to use a load-aware bisection algorithm
  rather than the default equal-volume split, which can improve performance for
  non-uniform geometries.
- Passing `chunk_layout=sim1` (another `Simulation` object) exactly replicates the
  first simulation's domain decomposition, which is required for the flux subtraction
  technique to work correctly.
- The closed-surface flux test is a fundamental validation of ghost-zone communication:
  any error in inter-chunk boundary exchange will cause the total flux to be wrong.
- `decimation_factor=1` in `add_flux` is important when very precise flux values are
  needed; higher decimation factors reduce memory but introduce a small accumulation
  error.
- The hard-coded reference value (86.908...) makes this a regression test: if the
  FDTD update equations, boundary communication, or DFT accumulation changes in a way
  that affects the physics, this value will change and the test will catch it.

---

### 9. `test_user_defined_material.py` — Symmetry with User-Defined Material Functions

**Physics:** A spatially varying dielectric function described by a user-defined
Python function (either returning a `Medium` object or a scalar epsilon) should produce
identical results to a simulation using equivalent geometric objects, even when mirror
symmetries are exploited to reduce the computational domain.
**Difficulty:** Intermediate
**Source:** `python/tests/test_user_defined_material.py`
**Test Status:** PASS (4.5 s)

#### Theory

Meep supports three equivalent ways to specify a spatially varying permittivity:

1. **Geometric objects**: `mp.Block`, `mp.Cylinder`, `mp.Ellipsoid`, etc. — Meep
   determines epsilon at each grid point by checking point-in-object membership.
2. **material_function**: a Python callable `f(p: Vector3) -> mp.Medium` that returns
   the material at each point p. This is the most general approach and allows arbitrary
   smooth or discontinuous epsilon profiles.
3. **epsilon_func**: a Python callable `f(p: Vector3) -> float` that returns the scalar
   epsilon at each point. Simpler and slightly faster than `material_function` for
   non-dispersive isotropic media.

When using `material_function` or `epsilon_func`, Meep calls the Python function at
every grid point during initialization to fill in the epsilon array. For SWIG-wrapped
C++ code, this means crossing the Python-C++ boundary millions of times for a large
grid, which can be slow. The geometric-object approach avoids this by doing the
point-in-object test in C++.

Mirror symmetry (`mp.Mirror(mp.X)` and `mp.Mirror(mp.Y)`) reduces the simulation
domain to one quarter for a geometry with both X and Y mirror symmetry planes. Meep
enforces the symmetry by evolving only the positive quadrant and mirroring fields to
the other quadrants at each step. The symmetry must be consistent with both the source
and the geometry: the source must be placed on the symmetry axis, and the geometry must
be invariant under the mirror reflections.

The test geometry (ellipsoid inside a cylinder) has both X and Y mirror symmetry
because the ellipsoid axes are aligned with X and Y. The source (Ez at the origin) is
even under both mirrors. This allows `mp.Mirror(mp.X)` and `mp.Mirror(mp.Y)` to be
applied simultaneously, reducing the effective simulation to a 2.5x2.5 cell out of the
full 10x10 cell (a 16x speedup in memory and time).

The test checks that the Ez field at a specific point (x=1) is identical (to 7 decimal
places) when using `material_function` and when using geometric objects, confirming
that the subpixel smoothing applied to both representations gives the same result.

#### Code Walkthrough

**User-defined material function:**

```python
def my_material_func(p):
    x, y = p.x, p.y
    if (x**2 / R1X**2 + y**2 / R1Y**2) < 1.0:     # inside inner ellipsoid
        return mp.Medium(epsilon=1.0)
    elif (x**2 / R2**2 + y**2 / R2**2) < 1.0:       # inside outer cylinder
        return mp.Medium(epsilon=3.5**2)
    else:
        return mp.Medium(epsilon=1.0)
```

The function checks each point against the analytic shapes in order of containment.
`epsilon=nn**2` converts refractive index to permittivity.

**Epsilon function (scalar variant):**

```python
def my_epsilon_func(p):
    if (p.x**2 / R1X**2 + p.y**2 / R1Y**2) < 1.0:
        return 1.0
    elif (p.x**2 / R2**2 + p.y**2 / R2**2) < 1.0:
        return 3.5
    return 1.0
```

This returns the refractive index n directly (Meep treats the return value of
`epsilon_func` as epsilon = n in this context; check documentation carefully).

**Symmetry-exploiting simulation setup:**

```python
sim = mp.Simulation(
    cell_size=self.cell,           # 10x10
    symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y)],
    material_function=my_material_func,
    boundary_layers=[mp.PML(1.0)],
    sources=[mp.Source(mp.GaussianSource(0.2, fwidth=0.1),
                       mp.Ez, mp.Vector3())],
    resolution=10,
)
sim.run(until=200)
fp = sim.get_field_point(mp.Ez, mp.Vector3(x=1))
```

The symmetries halve the cell in each direction internally; the user still specifies
the full cell size and full source/geometry positions.

**Geometric object equivalent:**

```python
geometry = [mp.Cylinder(5, material=my_material_func)]
```

A `Cylinder` can accept a callable as its `material`, making the geometric and
functional approaches even more interchangeable.

#### Key Takeaways

- `material_function=f` and `epsilon_func=f` accept any Python callable, enabling
  arbitrarily complex material distributions including those imported from external
  data files or computed by optimization algorithms.
- Mirror symmetry (`mp.Mirror`) reduces memory and time by a factor of 2 per axis, but
  requires that both the geometry and all sources respect the chosen symmetry planes.
- The Ez field values match at 7 decimal places between `material_function` and
  geometric-object approaches, confirming that subpixel smoothing (which averages
  epsilon over the Yee cell) is applied identically in both cases.
- When a `Cylinder` (or other geometric object) receives a callable as its `material`
  argument, only points inside that object are evaluated by the callable; points outside
  default to the simulation's `default_material`.
- The four test cases (material_function, epsilon_func, geometric object with
  material_function, geometric object with epsilon_func) form a 2x2 matrix that
  verifies all combinations of the two material-specification and two geometry-
  specification mechanisms.

---

### 10. `test_physical.py` — 2D Cylindrical Decay of a Point Source via CW Solver

**Physics:** A 2D point source radiating at a single continuous frequency produces an
outgoing cylindrical wave whose amplitude decays as 1/sqrt(r) in 2D, corresponding to
a power decay of 1/r; this test verifies that the continuous-wave (CW) frequency-
domain solver produces fields consistent with this fundamental 2D Green's function
behavior.
**Difficulty:** Beginner
**Source:** `python/tests/test_physical.py`
**Test Status:** PASS (11.8 s)

#### Theory

In two spatial dimensions, the outgoing solution to the scalar wave equation (Helmholtz
equation) for a point source at the origin is the Hankel function of the first kind:

    H_0^(1)(k*r) ~ sqrt(2 / (pi*k*r)) * exp(i*(k*r - pi/4))   as k*r >> 1

For large r, the amplitude decays as 1/sqrt(r), which means the power (proportional
to |E|^2) decays as 1/r. This 1/r law distinguishes 2D radiation from 3D (where power
decays as 1/r^2) and from 1D (where power is constant, i.e., no geometric spreading).

The test places a ContinuousSource (single-frequency, steady-state) at (-dx, 0) and
measures the field amplitude at the origin (distance dx from the source) and at (+dx,
0) (distance 2*dx from the source). The amplitude ratio should satisfy:

    |E(r=dx)| / |E(r=2*dx)| = sqrt(2*dx / dx) = sqrt(2)

Squaring the amplitude ratio gives

    (|E(dx)| / |E(2*dx)|)^2 = 2

The test checks that this squared ratio lies within ±6% of 2 (i.e., between 1.88 and
2.12).

Rather than running a time-domain simulation to steady state (which requires many
oscillation periods), the test uses Meep's `solve_cw` method, which employs a
convergent variant of the full multigrid (COCG) iterative solver to find the steady-
state solution directly in the frequency domain. This is much faster for problems
where only a single frequency is needed.

The `force_complex_fields=True` flag is required because `solve_cw` operates on
complex-valued fields; without it, the imaginary parts of the oscillating fields would
be discarded, giving incorrect steady-state amplitudes.

The simulation uses `mp.PML(ymax/3)`, placing a thick absorbing layer (1/3 of the cell
width) on all sides. This ensures that the cylindrical wave is absorbed before reaching
the boundary, preventing reflections that would contaminate the 1/sqrt(r) decay
measurement.

#### Code Walkthrough

**Continuous-wave source and CW solver:**

```python
sources = [mp.Source(mp.ContinuousSource(w), mp.Ez,
                     center=mp.Vector3(-dx), size=mp.Vector3())]

sim = mp.Simulation(cell_size=cell_size, resolution=a,
                    boundary_layers=pml_layers, sources=sources,
                    force_complex_fields=True)
sim.init_sim()
sim.solve_cw(tol=1e-6)
```

`ContinuousSource` emits a monochromatic wave at frequency w = 0.30 (in Meep units).
`solve_cw` iterates until the residual drops below 1e-6 (or 1e-5 for single precision),
at which point the fields represent the true steady-state solution. No explicit `run`
call is needed; `solve_cw` replaces the time integration.

**Amplitude ratio measurement:**

```python
amp1 = sim.get_field_point(mp.Ez, mp.Vector3())      # at r = dx
amp2 = sim.get_field_point(mp.Ez, mp.Vector3(dx))    # at r = 2*dx

ratio = abs(amp1) / abs(amp2)
ratio = ratio**2   # 1/sqrt(r) decay -> square to get 1/r
```

`get_field_point` returns a complex number; `abs()` takes the magnitude. Squaring the
amplitude ratio converts the geometric (1/sqrt(r)) amplitude decay into the power (1/r)
decay, which is easier to check against an integer ratio.

**Tolerance check:**

```python
self.assertTrue(ratio <= 2.12 and ratio >= 1.88, fail_msg)
```

The ±6% tolerance accounts for near-field corrections (the Hankel function asymptotic
form is only accurate for k*r >> 1, and k*r = 2*pi*w*dx = 2*pi*0.30*2 ≈ 3.8 is only
moderately large), finite grid resolution, and PML reflection artifacts.

#### Key Takeaways

- `solve_cw(tol)` finds the continuous-wave steady-state solution without time-domain
  integration, using an iterative frequency-domain solver; it is the right choice when
  only a single frequency is needed and the source is time-harmonic.
- `force_complex_fields=True` is mandatory for `solve_cw` because the solver works in
  the complex (phasor) representation of the fields.
- In 2D, point-source amplitude decays as 1/sqrt(r), so the squared amplitude ratio
  between two distances r_1 and r_2 is r_2/r_1; this is a fundamental sanity check
  that the simulation obeys the correct 2D Green's function.
- A thick PML (`dpml = ymax/3`) is important for CW simulations because the CW solver
  requires the PML to fully suppress all reflections at the chosen frequency; a thin
  PML may leave residual standing waves that distort the field amplitude measurements.
- This test is labeled "physical" because it checks a universal physical law (geometric
  spreading in 2D) rather than a specific material or boundary algorithm, making it a
  robust indicator of overall simulation correctness.
