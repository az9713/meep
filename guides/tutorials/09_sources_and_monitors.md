# Chapter 9: Sources, Monitors, and Field Analysis

This chapter covers the core techniques for injecting electromagnetic energy into a Meep simulation and for extracting physically meaningful quantities from the resulting fields. We examine plane-wave sources with spatial phase modulation, the continuous-wave (CW) frequency-domain solver, DFT-based field and energy monitors, perturbation theory for resonant cavities, the Maxwell stress tensor for optical forces, and a range of field-extraction utilities. Together these tools form the measurement layer that sits between the FDTD engine and the physical quantities you actually care about.

---

### 1. `pw-source.py` — Approximate Plane-Wave Source via Phase-Modulated Currents

**Physics:** A true plane wave cannot be injected into a finite computational cell without some form of truncation artifact; this example shows how to build a good approximation by running two line sources whose current amplitudes carry the spatial phase factor `exp(i k·r)` of the desired plane wave, producing an Ez-polarized wave propagating at 45 degrees in vacuum.

**Difficulty:** Intermediate

**Source:** `python/examples/pw-source.py`

**Test Status:** PASS (13.6 s)

#### Theory

A monochromatic plane wave propagating in the xy-plane can be written as

```
E_z(r, t) = E_0 exp(i(k·r - omega t))
```

where `k = (k_x, k_y)` satisfies the dispersion relation `|k| = n omega / c = 2 pi n f`. In free space with `n = 1` and frequency `f`, the wave vector magnitude is `|k| = 2 pi f`.

FDTD sources in Meep are current density sheets `J(r, t)`. A uniform sheet source — one whose amplitude is constant along its length — launches equal power in both the forward and backward directions. To suppress the backward-propagating component and to imprint a transverse phase gradient, we assign the current amplitude function `A(r) = exp(i k·r)` at every point along the source. When the source lies along the left edge of the cell (constant x), the phase gradient in y is exactly `exp(i k_y y)`, which matches the transverse profile of the desired plane wave. A second source on the bottom edge (constant y) with phase gradient `exp(i k_x x)` completes the L-shaped injection region.

This L-shaped geometry is a practical engineering choice: a true plane wave would require sources on all four sides of the cell, but in practice two perpendicular sides create a convincing approximation within the interior of the cell where the interference pattern is coherent. The PML absorbs the outgoing wave on all boundaries.

The test validates the approach by checking that the field ratio at two interior points matches the expected phase difference `exp(i k·(r1 - r2))`, confirming that the injected wave genuinely carries the correct spatial phase.

The source uses `ContinuousSource` rather than a pulsed `GaussianSource` because the phase relationship `A(r) = exp(i k·r)` is only exact at a single frequency; using a broadband pulse would mix contributions from wavenumbers that do not satisfy the dispersion relation at the center frequency, degrading the plane-wave approximation.

#### Code Walkthrough

The wave vector is constructed so its magnitude equals `2 pi f n`:

```python
fcen = 0.8
kdir = mp.Vector3(1, 1)           # propagation direction (diagonal)
n = 1
k = kdir.unit().scale(2 * math.pi * fcen * n)
```

The amplitude function is a closure that captures `k` and an origin offset `x0`. The offset is needed because Meep evaluates the amplitude function relative to the center of the source region, not relative to the global origin:

```python
def pw_amp(k, x0):
    def _pw_amp(x):
        return cmath.exp(1j * k.dot(x + x0))
    return _pw_amp
```

Two sources are placed — one on the left wall and one on the bottom wall. Each has `size` equal to the interior cell length along its respective axis, so together they span the entire L-shaped injection perimeter:

```python
sources = [
    mp.Source(mp.ContinuousSource(fcen, fwidth=df), component=mp.Ez,
              center=mp.Vector3(-0.5 * s, 0), size=mp.Vector3(0, s),
              amp_func=pw_amp(k, mp.Vector3(x=-0.5 * s))),
    mp.Source(mp.ContinuousSource(fcen, fwidth=df), component=mp.Ez,
              center=mp.Vector3(0, -0.5 * s), size=mp.Vector3(s, 0),
              amp_func=pw_amp(k, mp.Vector3(y=-0.5 * s))),
]
```

The simulation runs for 400 time units (well past transient startup) and outputs the Ez field at the end:

```python
sim.run(mp.at_end(mp.output_efield_z), until=400)
```

The test (`test_pw_source.py`) evaluates the field at two points and checks the ratio:

```python
pt1 = self.sim.get_field_point(mp.Ez, v1)
pt2 = self.sim.get_field_point(mp.Ez, v2)
self.assertClose(pt1 / pt2, 27.557668029008262, epsilon=tol)
```

The large ratio (~27.6) reflects that the two points are at different distances from the sources and at different phases, demonstrating consistent phase coherence across the cell.

#### Key Takeaways

- Spatial phase modulation via `amp_func` converts an ordinary current sheet into a directive phase-gradient source.
- The amplitude function receives position relative to the source center; subtract the source center from the global origin to get a globally-phased wave.
- Two perpendicular sources (an L-shape) suffice for a good plane-wave approximation in the interior.
- `ContinuousSource` must be used when the phase relationship `exp(i k·r)` is frequency-specific.
- The plane-wave quality can be verified by checking that field ratios at interior points match the expected phase factor `exp(i k·Delta_r)`.

---

### 2. `solve-cw.py` — Continuous-Wave Frequency-Domain Solver for Ring Resonator Modes

**Physics:** Instead of running the FDTD equations until a continuous-wave steady state is reached (which can take thousands of periods for high-Q resonators), Meep's `solve_cw` method directly solves the frequency-domain linear system `(nabla x nabla x - omega^2 epsilon) E = J` using a quasi-minimal residual (QMR/biCGSTAB-L) iterative solver, yielding the resonant mode fields orders of magnitude faster.

**Difficulty:** Intermediate

**Source:** `python/examples/solve-cw.py`

**Test Status:** FAIL — `np.complex_` removed in NumPy 2.x (use `np.complex128` instead)

#### Theory

For a resonator driven on resonance by a continuous-wave source, the time-domain FDTD fields grow and eventually settle into a steady oscillation at the driving frequency. The convergence time is proportional to Q/f, which for high-Q resonators can be many thousands of optical periods. The CW solver bypasses this by reformulating the problem in the frequency domain: with the time dependence factored out as `exp(-i omega t)`, Maxwell's equations become a sparse linear system whose solution directly yields the steady-state complex field amplitude.

Meep implements the Jacobi-Davidson variant of biCGSTAB-L, an iterative Krylov subspace solver parameterized by a restart length L and a convergence tolerance. The solver is initialised with the current state of the FDTD fields, so it is best to call `sim.init_sim()` before `sim.solve_cw()`.

The example demonstrates convergence: it runs the CW solver five times with decreasing tolerances from `1e-8` down to `1e-12` and plots the relative error (norm of difference against the tightest-tolerance solution) versus tolerance. The expected result is monotonically decreasing error as the tolerance tightens.

For comparison, the script also runs a time-domain simulation with a Gaussian pulse and extracts the resonant mode fields via `add_dft_fields`. Both methods should converge to the same field pattern, providing an independent cross-check.

The bug in this file is on line 80: `dtype=np.complex_` is a NumPy 1.x alias for `complex128` that was removed in NumPy 2.0. The fix is to replace it with `dtype=np.complex128`.

#### Code Walkthrough

Geometry is a 2D ring resonator (dielectric annulus):

```python
geometry = [
    mp.Cylinder(radius=r + w, material=mp.Medium(index=n)),
    mp.Cylinder(radius=r),    # vacuum hole
]
```

Two anti-symmetric sources (amplitude -1 on the second) exploit the known symmetry of the resonant mode:

```python
src = [
    mp.Source(mp.ContinuousSource(fcen), component=mp.Ez,
              center=mp.Vector3(r + 0.1)),
    mp.Source(mp.ContinuousSource(fcen), component=mp.Ez,
              center=mp.Vector3(-(r + 0.1)), amplitude=-1),
]
```

`force_complex_fields=True` is required because the CW solver works with complex-valued fields. The `symmetries` list then halves the memory and CPU cost by exploiting the known mirror symmetry of the mode:

```python
sim = mp.Simulation(..., force_complex_fields=True,
                    symmetries=[mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=+1)])
```

The convergence study loops over tolerances and calls `solve_cw` after re-initialising the simulation:

```python
for i in range(num_tols):
    sim.init_sim()
    sim.solve_cw(tols[i], maxiters, L)
    ez_dat[:, :, i] = sim.get_array(vol=nonpml_vol, component=mp.Ez)
```

The fix for the NumPy 2.x failure is straightforward:

```python
# Old (fails with NumPy 2.x):
ez_dat = np.zeros((...), dtype=np.complex_)
# Fixed:
ez_dat = np.zeros((...), dtype=np.complex128)
```

#### Key Takeaways

- `solve_cw(tol, maxiters, L)` directly finds the frequency-domain steady state, avoiding long FDTD ring-down times for high-Q resonators.
- `force_complex_fields=True` must be set to obtain complex-valued field amplitudes needed by the CW solver.
- Anti-symmetric source placement selectively excites a single symmetry class of modes.
- The `np.complex_` alias was removed in NumPy 2.0; replace it with `np.complex128` for forward-compatible code.
- DFT fields from a pulsed run and CW solver fields provide two independent paths to the same resonant mode profile; comparing them validates both methods.

---

### 3. `perturbation_theory.py` — First-Order Perturbation Theory for Cylindrical Ring Resonators

**Physics:** Perturbation theory predicts how the resonant frequency of a dielectric cavity shifts when the geometry is slightly modified, without requiring a full re-simulation of the perturbed structure. This example computes `dω/dR` — the derivative of the resonant frequency with respect to the inner radius of a cylindrical ring — using two approaches: the perturbation-theory integral over the unperturbed fields, and a direct finite-difference comparison of two FDTD runs.

**Difficulty:** Advanced

**Source:** `python/examples/perturbation_theory.py`

**Test Status:** PASS (66.2 s)

#### Theory

For a photonic resonator with resonant frequency omega_0 and mode fields E, H, the first-order frequency shift due to a small perturbation Delta_epsilon of the dielectric constant is:

```
Delta_omega / omega_0 = - (integral Delta_epsilon |E|^2 dV) / (2 integral epsilon |E|^2 dV)
```

When the perturbation is a shift of a dielectric interface (radius R changes by dR), `Delta_epsilon` is nonzero only in a thin shell at the interface. For a surface moving outward by dR from a region of permittivity `epsilon_1` into a region of permittivity `epsilon_2`, the leading-order formula for `dω/dR` involves the tangential (parallel) and normal (perpendicular) field components at the interface differently:

For a **parallel** field component (E tangential to the interface, continuous across it):
```
Delta_omega / omega_0 = -(Delta_epsilon / 2) * (surface integral |E_parallel|^2 dA) / W
```

For a **perpendicular** field component (D normal to the interface, D_n continuous across it):
```
Delta_omega / omega_0 = +(Delta_epsilon_inv / 2) * (surface integral |D_perp|^2 dA) / W
```

where `Delta_epsilon = epsilon_inside - epsilon_outside`, `Delta_epsilon_inv = 1/epsilon_inside - 1/epsilon_outside`, and W is the total electric energy.

In the cylindrical coordinate system used here (r, phi, z), the ring's interfaces are at constant r, so the `E_z` component is tangential and `D_r` is normal. For the Ez-polarized mode (`src_cmpt = mp.Ez`), only the tangential integral contributes:

```
numerator = Delta_epsilon * 2 pi * (r_inner |E_z(r_inner)|^2 - r_outer |E_z(r_outer)|^2)
```

The denominator is the total electric energy computed by `electric_energy_in_box`.

The cylindrical coordinate system (`dimensions=mp.CYLINDRICAL`) in Meep models the 3D rotationally symmetric structure as an effective 2D problem parameterized by the azimuthal mode number `m`. This is dramatically faster than a full 3D simulation.

#### Code Walkthrough

The simulation uses cylindrical coordinates with `m = 5` (fifth-order azimuthal mode):

```python
dimensions = mp.CYLINDRICAL
cell = mp.Vector3(sr)   # 1D radial cell
sim = mp.Simulation(..., dimensions=dimensions, m=m)
```

Step 1: find the resonant frequency using Harminv with a broadband source:

```python
h = mp.Harminv(src_cmpt, mp.Vector3(r + 0.1), fcen, df)
sim.run(mp.after_sources(h), until_after_sources=100)
frq_unperturbed = h.modes[0].freq
```

Step 2: re-run with a narrowband source centred exactly on the resonance, then extract field values at the interface boundaries:

```python
numerator_integral = (deps * 2 * np.pi *
    (r * abs(sim.get_field_point(mp.Ez, mp.Vector3(r)))**2
   - (r+w) * abs(sim.get_field_point(mp.Ez, mp.Vector3(r+w)))**2))
denominator_integral = sim.electric_energy_in_box(
    center=mp.Vector3(0.5*sr), size=mp.Vector3(sr))
perturb_theory_dw_dR = -frq_unperturbed * numerator_integral / (4 * denominator_integral)
```

Step 3: run the perturbed geometry (inner radius shifted by `dr = 0.01`) and measure the frequency difference:

```python
finite_diff_dw_dR = (frq_perturbed - frq_unperturbed) / dr
```

The `--perpendicular` flag switches to `Hz` polarization where `E_phi` is parallel and `D_r` is perpendicular, activating both the `para_integral` and `perp_integral` branches of the calculation.

#### Key Takeaways

- First-order perturbation theory gives `dω/dR` from a single unperturbed simulation, avoiding the cost of re-simulating every geometric variant.
- Cylindrical coordinates (`dimensions=mp.CYLINDRICAL`) reduce a 3D rotationally symmetric problem to 2D, with the azimuthal mode number `m` as a parameter.
- Field-interface integrals require evaluating `get_field_point` exactly at the dielectric boundary.
- Tangential E and normal D contribute differently to the perturbation integral; both must be included for mixed-polarization modes.
- `electric_energy_in_box` provides the mode normalization (denominator) required by the perturbation formula.

---

### 4. `perturbation_theory_2d.py` — Perturbation Theory in Full 2D Cartesian Geometry

**Physics:** This is the 2D Cartesian counterpart of tutorial 3, computing `dω/dR` for a 2D ring resonator (infinite cylinder cross-section) using mirror symmetries to reduce computational cost. The geometry, source types, and perturbation integrals are adapted for the 2D case where the cylindrical coordinate simplification is unavailable.

**Difficulty:** Advanced

**Source:** `python/examples/perturbation_theory_2d.py`

**Test Status:** TIMEOUT (CPU-intensive; default `res = 30` pixels/um with a full 2D cell is slower than the cylindrical version)

#### Theory

In 2D Cartesian coordinates, the ring resonator is modeled as a dielectric annulus (two concentric cylinders) embedded in a square cell. The resonant mode now has a sinusoidal angular dependence that cannot be factored into a 1D radial equation, so the full 2D field must be stored and integrated.

The perturbation theory formula takes the same form, but now the interface integrals are line integrals around the circumferences at r = r_inner and r = r_outer. In Meep units the factor of `2 pi r` captures the circumference weighting:

```
numerator (Ez mode) = Delta_epsilon * 2 pi *
    (r_inner |E_z(r_inner, 0)|^2 - r_outer |E_z(r_outer, 0)|^2)
```

Because the mode has two-fold mirror symmetry (Mirror X and Mirror Y), Meep can exploit this to reduce the computation to one quadrant. The symmetry phase for the Ez mode is `-1` under Mirror X and `+1` under Mirror Y (for the Hz mode the phases reverse). The script accepts a `--perpendicular` flag to switch between the two polarizations.

The finite-difference verification uses `dr = 0.04` (larger than the cylindrical case's 0.01) because the 2D geometry is more expensive per run, so the perturbation step is larger to keep the finite-difference error clearly above numerical noise.

The denominator integral uses `electric_energy_in_box` over the interior (non-PML) region; for 2D the factor of 8 in the denominator (compared to 4 in the cylindrical case) reflects the difference in the surface area / volume ratio between the 2D and cylindrical coordinate integral normalizations.

#### Code Walkthrough

Geometry uses two cylinders to form the annulus, with `height=mp.inf` to enforce 2D:

```python
geometry = [
    mp.Cylinder(material=mp.Medium(index=n), radius=r+w, height=mp.inf),
    mp.Cylinder(material=mp.vacuum, radius=r, height=mp.inf),
]
```

Two anti-symmetric sources are needed to select the correct symmetry class of the mode (the cylindrical version only needed one because azimuthal number `m` did the selection):

```python
sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=src_cmpt,
              center=mp.Vector3(r + 0.1)),
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=src_cmpt,
              center=mp.Vector3(-(r + 0.1)), amplitude=-1),
]
```

Mirror symmetry cuts the cell in half twice, reducing computation to one quadrant:

```python
symmetries = [
    mp.Mirror(mp.X, phase=+1 if args.perpendicular else -1),
    mp.Mirror(mp.Y, phase=-1 if args.perpendicular else +1),
]
```

The perturbation integral for the `Ez` mode evaluates Ez at the two interface radii along the x-axis (where Mirror Y symmetry guarantees the field is real):

```python
numerator_integral = (deps * 2 * np.pi *
    (r * abs(sim.get_field_point(mp.Ez, mp.Vector3(r)))**2
   - (r+w) * abs(sim.get_field_point(mp.Ez, mp.Vector3(r+w)))**2))
perturb_theory_dw_dR = -frq_unperturbed * numerator_integral / (8 * denominator_integral)
```

#### Key Takeaways

- 2D Cartesian perturbation theory requires the full 2D field whereas the cylindrical version only needs fields at a few radial points; this makes the 2D version significantly more CPU-intensive.
- Mirror symmetries must be assigned carefully: the symmetry phase depends on the field component (Ez vs Hz) and the mirror direction.
- The denominator factor changes between cylindrical (4) and 2D Cartesian (8) due to the difference in coordinate normalizations.
- Two anti-symmetric sources select the correct angular mode in Cartesian geometry, replacing the azimuthal index `m` used in cylindrical coordinates.
- For production use, the cylindrical version is preferred for rotationally symmetric geometries due to its dramatic speed advantage.

---

### 5. `test_pw_source.py` — Unit Test for Phase-Coherent Plane-Wave Source

**Physics:** This test validates that the L-shaped phase-modulated source from tutorial 1 produces a spatially coherent plane wave in the interior of the cell by checking the ratio of complex field amplitudes at two separated points against the analytically expected phase factor.

**Difficulty:** Beginner

**Source:** `python/tests/test_pw_source.py`

**Test Status:** PASS (8.5 s)

#### Theory

If the source correctly launches a plane wave `E_z(r) = E_0 exp(i k·r)`, then at any two interior points r1 and r2 the field ratio should satisfy:

```
E_z(r1) / E_z(r2) = exp(i k·(r1 - r2))
```

The magnitude of this ratio equals one (a pure phase). The test checks the field ratio at two specific points along the propagation direction where the magnitude is not unity (because the fields near the source boundary are not purely plane-wave), providing a sensitive check of both phase coherence and amplitude uniformity.

The setup is identical to `pw-source.py`: a 13x13 interior cell (plus 1-unit PML on each side), `fcen = 0.8`, 45-degree propagation. The test runs for 400 time units to ensure the CW steady state is fully reached before sampling.

The assertion `pt1 / pt2 ≈ 27.557668` is a regression value: the large magnitude (>>1) indicates the two test points are chosen at positions where the amplitude varies significantly — specifically, one point is near the source injection edge and the other is further into the propagation region. The test verifies both that the ratio has the correct magnitude (no spurious amplitude errors) and that it is consistent with the expected phase factor.

#### Code Walkthrough

The test class inherits from `ApproxComparisonTestCase` which provides `assertClose` with a relative tolerance parameter:

```python
class TestPwSource(ApproxComparisonTestCase):
    def setUp(self):
        # ... build sim identically to pw-source.py ...
        self.k = kdir.unit().scale(2 * math.pi * fcen)
```

After running 400 time units, field values are sampled:

```python
def test_pw_source(self):
    self.sim.run(mp.at_end(mp.output_efield_z), until=400)
    v1 = mp.Vector3(0.5 * self.s, 0)
    v2 = mp.Vector3(0.5 * self.s, 0.5 * self.s)
    pt1 = self.sim.get_field_point(mp.Ez, v1)
    pt2 = self.sim.get_field_point(mp.Ez, v2)
    tol = 1e-4 if mp.is_single_precision() else 1e-9
    self.assertClose(pt1 / pt2, 27.557668029008262, epsilon=tol)
```

The precision guard `mp.is_single_precision()` loosens the tolerance for single-precision builds, acknowledging that floating-point accumulation over 400 time units is more significant in single precision.

The analytical phase factor is separately verified:

```python
self.assertAlmostEqual(
    cmath.exp(1j * self.k.dot(v1 - v2)),
    0.7654030066070924 - 0.6435512702783076j,
)
```

This confirms the test geometry is correctly set up, independent of Meep.

#### Key Takeaways

- Regression tests for field values use tight tolerances (`1e-9` for double precision) that detect even subtle numerical regressions.
- `get_field_point` samples the interpolated complex field at an arbitrary position, making it the primary tool for point-wise field verification.
- Separate precision guards via `mp.is_single_precision()` make tests portable between double- and single-precision builds.
- The expected ratio value is derived from a reference run and hard-coded; any change to the source, geometry, or resolution should trigger a deliberate update of this value.
- Verifying the analytical phase factor independently from the simulation ratio ensures the test geometry is self-consistent.

---

### 6. `test_oblique_source.py` — Oblique Eigenmode Source and Flux Measurement

**Physics:** This test validates `EigenModeSource` for launching a waveguide mode at oblique angles (0, 20, and 40 degrees CCW from the x-axis) and verifies that the transmitted power flux is consistent across all angles and agrees with the power computed from eigenmode decomposition coefficients.

**Difficulty:** Intermediate

**Source:** `python/tests/test_oblique_source.py`

**Test Status:** TIMEOUT (CPU-intensive; three separate full-resolution simulations at different angles)

#### Theory

`EigenModeSource` uses MPB (MIT Photonic Bands) to compute the exact eigenmode profile of a waveguide at the specified frequency and wavenumber direction, then uses that profile as the source current distribution. This produces a source that injects a single clean waveguide mode with no contamination from radiation modes, unlike a simple Gaussian beam approximation.

For an oblique waveguide (rotated by angle theta from the x-axis), the eigenmode propagates along the waveguide axis. The key parameters are:

- `direction=mp.NO_DIRECTION`: tells Meep not to use the normal to the source plane as the mode propagation direction.
- `eig_kpoint`: specifies the propagation direction as a unit vector in the rotated coordinate frame.
- `eig_parity`: specifies field symmetry (ODD_Z for the fundamental TE-like mode of a 2D slab waveguide).
- `eig_match_freq=True`: instructs MPB to find the mode at exactly the specified frequency by adjusting |k|.

The flux is measured at a downstream cross-section and compared with the power computed from eigenmode coefficient decomposition: `|alpha|^2 = power`. If the source correctly launches a single mode with unit efficiency, both quantities should agree to the same precision.

For a correctly working rotated source, the total transmitted power should be nearly independent of the rotation angle (within the accuracy of the numerical grid), because the waveguide mode profile and the source geometry are consistently rotated together. The test asserts `fluxes[0] ≈ fluxes[1] ≈ fluxes[2]` to within 0 decimal places (integer-level agreement for ~100 unit fluxes).

#### Code Walkthrough

The waveguide block is rotated by constructing rotated basis vectors `e1` and `e2`:

```python
rot_angle = math.radians(t)
kpoint = mp.Vector3(math.cos(rot_angle), math.sin(rot_angle), 0)
geometry = [
    mp.Block(...,
             e1=mp.Vector3(1).rotate(mp.Vector3(z=1), rot_angle),
             e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), rot_angle),
             material=mp.Medium(index=1.5))
]
```

The source is placed at the left side of the cell and uses `NO_DIRECTION` with an explicit `eig_kpoint`:

```python
sources = [mp.EigenModeSource(
    src=mp.GaussianSource(1.0, fwidth=0.1),
    size=mp.Vector3(y=10),
    center=mp.Vector3(x=-3),
    direction=mp.NO_DIRECTION,
    eig_kpoint=kpoint,
    eig_band=1,
    eig_parity=mp.ODD_Z,
    eig_match_freq=True,
)]
```

After running, eigenmode coefficients are extracted from the transmitted flux:

```python
res = sim.get_eigenmode_coefficients(tran, [1], eig_parity=mp.ODD_Z,
                                     direction=mp.NO_DIRECTION,
                                     kpoint_func=lambda f, n: kpoint)
coeff_fluxes.append(abs(res.alpha[0, 0, 0])**2)
```

The assertion checks that flux and coefficient-derived power agree:

```python
for i in range(3):
    self.assertAlmostEqual(fluxes[i], coeff_fluxes[i], places=0)
```

#### Key Takeaways

- `EigenModeSource` with `direction=mp.NO_DIRECTION` and `eig_kpoint` enables accurate oblique waveguide mode injection.
- Rotating both the geometry (`e1`, `e2` vectors of the Block) and the source (`eig_kpoint`) consistently is essential for correct oblique propagation.
- `get_eigenmode_coefficients` decomposes the transmitted field into modes; `|alpha[0,0,0]|^2` gives the power in the forward-propagating fundamental mode.
- Total transmitted power should be rotationally invariant for a correctly implemented oblique source.
- This test is CPU-intensive because each rotation angle requires a separate full simulation at high resolution.

---

### 7. `test_dft_fields.py` — DFT Field Arrays: Extraction, Output, and Decimation

**Physics:** DFT (Discrete Fourier Transform) monitors accumulate the Fourier transform of the time-domain fields during the simulation, yielding the complex-valued field amplitude at one or more specified frequencies. This test verifies that DFT arrays obtained via `get_dft_array` are numerically identical to the corresponding data written to HDF5 files, and tests the `decimation_factor` parameter that reduces DFT update cost.

**Difficulty:** Intermediate

**Source:** `python/tests/test_dft_fields.py`

**Test Status:** PASS (19.97 s)

#### Theory

At each FDTD time step `n`, the DFT accumulator updates:

```
E_tilde(omega) += E(n * dt) * exp(i omega n dt) * dt
```

After `N_total` steps, `E_tilde(omega)` approximates the continuous Fourier transform of E(t) over the simulation window. The spectral resolution is `1 / (N_total * dt)` and the Nyquist frequency is `1 / (2 dt)`.

For long simulations with many DFT frequencies or large spatial volumes, the DFT update step can become a significant fraction of the total compute time. The `decimation_factor=k` parameter updates the DFT only every k time steps, reducing cost by a factor of k. This introduces a small aliasing error (effectively sampling at `1/(k dt)`) that is negligible as long as the highest DFT frequency is well below `1/(2k dt)`.

The test also verifies the handling of "thin" volumes — DFT regions with zero thickness in one direction (line monitors in 2D). Meep collapses degenerate dimensions so that a line monitor returns a 1D array rather than a 2D array with one singleton dimension. The test checks `ndim == 1` for both x-direction and y-direction line monitors.

The consistency between `get_dft_array` (in-memory Python array) and the data read back from the HDF5 file written by `output_dft` validates the entire DFT data pipeline from accumulation through serialisation.

#### Code Walkthrough

The simulation setup is a 2D ring resonator (same geometry as `solve-cw.py`) with a Gaussian source:

```python
sim = self.init()   # creates ring resonator simulation
sim.init_sim()
dft_fields = sim.add_dft_fields([mp.Ez], self.fcen, 0, 1)
```

The call signature `add_dft_fields(components, fcen, df, nfreq)` adds a DFT monitor for `nfreq` frequencies starting at `fcen - df/2`. With `df=0` and `nfreq=1` this monitors a single frequency `fcen`.

Thin volume monitors are created and checked for correct dimensionality:

```python
thin_x_volume = mp.Volume(center=mp.Vector3(0.35*self.sxy),
                           size=mp.Vector3(y=0.8*self.sxy))   # zero x-extent
thin_x_flux = sim.add_dft_fields([mp.Ez], self.fcen, 0, 1, where=thin_x_volume)
# ...
thin_x_array = sim.get_dft_array(thin_x_flux, mp.Ez, 0)
np.testing.assert_equal(thin_x_array.ndim, 1)
```

The decimation test runs two DFT monitors simultaneously — one at full rate, one at decimation 4 — and checks they agree to 0.1% relative error:

```python
undecimated_field = sim.add_dft_fields([mp.Ez], self.fcen, 0, 1, decimation_factor=1)
decimated_field   = sim.add_dft_fields([mp.Ez], self.fcen, 0, 1, decimation_factor=4)
sim.run(until_after_sources=100)
self.assertClose(expected_dft, actual_dft, epsilon=1e-3)
```

The HDF5 round-trip test reads the `.h5` file using `h5py` and reconstructs the complex array:

```python
with h5py.File("dft-fields.h5", "r") as f:
    exp_fields = mp.complexarray(f["ez_0.r"][()], f["ez_0.i"][()])
self.assertClose(exp_fields, fields_arr, epsilon=1e-6)
```

Meep stores real and imaginary parts as separate HDF5 datasets (`ez_0.r`, `ez_0.i`); `mp.complexarray` combines them into a numpy complex array.

#### Key Takeaways

- `add_dft_fields(components, fcen, df, nfreq)` creates a DFT monitor; `get_dft_array(obj, component, freq_index)` retrieves the complex-valued result.
- Degenerate (zero-thickness) dimensions in DFT volumes are automatically collapsed to produce lower-dimensional arrays.
- `decimation_factor=k` reduces DFT overhead by a factor of k with negligible accuracy loss when the DFT frequencies are well below `1/(2k dt)`.
- `output_dft` writes real and imaginary parts to HDF5 as separate datasets named `component_freqidx.r` and `component_freqidx.i`.
- The `Yee grid` option (`yee_grid=True`) places DFT outputs at staggered Yee grid positions rather than interpolated cell centers, required for some applications.

---

### 8. `test_dft_energy.py` — Group Velocity from DFT Energy and Flux

**Physics:** This test computes the group velocity of a waveguide mode by two independent methods: (1) the ratio of Poynting flux to electric energy density (the group velocity relation `v_g = P / U_E` for a non-dispersive medium), and (2) directly from MPB eigenmode coefficients via `get_eigenmode_coefficients`. Agreement between the two validates both the energy monitor implementation and the eigenmode decomposition.

**Difficulty:** Intermediate

**Source:** `python/tests/test_dft_energy.py`

**Test Status:** PASS (15.2 s)

#### Theory

For a waveguide mode propagating in a non-dispersive medium, the group velocity is related to the time-averaged energy density and Poynting flux by:

```
v_g = S / U
```

where S is the integrated Poynting flux (power, in Meep units) and U is the total electromagnetic energy density integrated over the same cross-sectional volume. Since the electric and magnetic energies are equal for a propagating mode (equipartition), the total energy is `U = 2 U_E`, giving:

```
v_g = S / (2 U_E) = (0.5 S) / U_E
```

This is what the test computes via `ratio_vg = (0.5 * poynting_flux) / e_energy`.

Meep's `add_energy` monitor accumulates DFT versions of the electric and magnetic energy densities:

```
U_E(omega) = (1/2) integral epsilon |E(omega)|^2 dV
U_M(omega) = (1/2) integral mu |H(omega)|^2 dV
U_total = U_E + U_M
```

These are obtained via `mp.get_electric_energy(energy)`, `mp.get_magnetic_energy(energy)`, and `mp.get_total_energy(energy)`.

The MPB method extracts the group velocity directly from the band structure derivative `v_g = d omega / d k`. The test verifies that these two independent methods agree to 3 decimal places, confirming the internal consistency of Meep's energy and flux monitors.

The `decimation_factor` validation compares energy values obtained at full update rate (`decimation_factor=1`) against a decimated run (`decimation_factor=10`). For a waveguide mode with a source frequency far below the Nyquist aliasing limit of the decimated run, the agreement should hold to 1 decimal place.

#### Code Walkthrough

The waveguide is a silicon slab (epsilon=12) in a 10x5 cell with an eigenmode source:

```python
sources = [mp.EigenModeSource(
    src=mp.GaussianSource(frequency=fsrc, fwidth=0.2*fsrc),
    center=mp.Vector3(-3), size=mp.Vector3(y=5),
    eig_band=1, eig_parity=mp.ODD_Z + mp.EVEN_Y,
    eig_match_freq=True,
)]
```

Both flux and energy monitors are placed at the same downstream cross-section:

```python
flux = sim.add_flux(fsrc, 0, 1,
    mp.FluxRegion(center=mp.Vector3(3), size=mp.Vector3(y=5)))
energy = sim.add_energy(fsrc, 0, 1,
    mp.EnergyRegion(center=mp.Vector3(3), size=mp.Vector3(y=5)))
```

After running, both methods are computed and compared:

```python
res = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
mode_vg = res.vgrp[0]                  # MPB group velocity
poynting_flux = mp.get_fluxes(flux)[0]
e_energy = mp.get_electric_energy(energy)[0]
ratio_vg = (0.5 * poynting_flux) / e_energy
self.assertAlmostEqual(ratio_vg, mode_vg, places=3)
```

Energy conservation is verified:

```python
m_energy = mp.get_magnetic_energy(energy)[0]
t_energy = mp.get_total_energy(energy)[0]
self.assertAlmostEqual(m_energy + e_energy, t_energy)
```

#### Key Takeaways

- `add_energy` with `EnergyRegion` creates a DFT energy monitor analogous to `add_flux` with `FluxRegion`.
- Group velocity satisfies `v_g = P / (2 U_E)` for propagating modes in non-dispersive media; this provides an independent check of MPB's eigenmode group velocity.
- `mp.get_electric_energy`, `mp.get_magnetic_energy`, and `mp.get_total_energy` all return lists (one entry per DFT frequency).
- Energy monitors support `decimation_factor` with the same trade-off as DFT field monitors: reduced cost at the price of mild aliasing error.
- The `vgrp` attribute of `get_eigenmode_coefficients` results gives the MPB group velocity for each requested band.

---

### 9. `test_field_functions.py` — Custom Field Integration and Extremum Finding

**Physics:** Meep provides three high-level utilities for computing spatially integrated quantities from the running fields: `integrate_field_function` (volume integral of an arbitrary function of multiple field components), `integrate2_field_function` (overlap integral between two different simulation states), and `max_abs_field_function` (maximum absolute value of a field function). This test validates all three.

**Difficulty:** Intermediate

**Source:** `python/tests/test_field_functions.py`

**Test Status:** PASS (12.3 s)

#### Theory

Physical observables often require integrating nonlinear combinations of field components. For example, the Purcell factor requires the local density of states (proportional to `|E(r_0)|^2`), quality factor calculations need the total electric energy (`integral epsilon |E|^2 dV`), and cross-correlation functions require overlap integrals `integral E_1^*(r) E_2(r) dV`.

Meep's `integrate_field_function(components, function, vol)` evaluates:

```
Result = integral_vol f(r, F1(r), F2(r), ...) dV
```

where `F1, F2, ...` are the field components listed in `components` and `f` is a user-defined Python function that accepts the position vector `r` and the field values as arguments.

`integrate2_field_function(fields2, components1, components2, function)` evaluates the overlap integral between the current simulation state (`self.fields`) and a saved state (`fields2`), enabling cross-correlations and inner products between two different time snapshots or simulations.

`max_abs_field_function(components, function, vol)` scans the volume and returns the maximum `|f(r)|` over all grid points, useful for normalization (e.g., finding `max |E|^2` for mode volume calculations).

The test uses the function:

```python
def f(r, ex, hz, eps):
    return (r.x * r.norm() + ex) - (eps * hz)
```

which combines the position vector, Ex field, Hz field, and dielectric constant into a single scalar. The integral over a symmetric domain (with mirror symmetry in both X and Y) should evaluate to a specific regression value.

#### Code Walkthrough

The simulation setup uses a simple GaussianSource in a 10x10 cell with two mirror symmetries:

```python
symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y)]
sim = mp.Simulation(resolution=20, cell_size=mp.Vector3(10, 10, 0), ...)
sim.run(until=200)
```

Integrating over the full cell versus a restricted volume:

```python
cs = [mp.Ex, mp.Hz, mp.Dielectric]
vol = mp.Volume(size=mp.Vector3(1), center=mp.Vector3())

res1 = sim.integrate_field_function(cs, f)           # full cell
res2 = sim.integrate_field_function(cs, f, vol)      # 1x1 box at center
self.assertAlmostEqual(res1, complex(-6.938893903907228e-18, 0.0))
self.assertAlmostEqual(res2, 0.0j)
```

The full-cell result is essentially machine epsilon (near zero), consistent with the field function having odd symmetry over the symmetric domain. The restricted-volume result is identically zero by symmetry at the center.

The field output function writes a custom-named HDF5 dataset:

```python
sim.output_field_function("weird-function", cs, f)
```

The overlap integral test saves the fields object from one run, resets, runs again, then computes the cross-correlation:

```python
sim.run(until_after_sources=10)
fields2 = sim.fields               # save reference to first-run fields
sim.reset_meep()
sim.run(until_after_sources=10)
res1 = sim.integrate2_field_function(fields2, [mp.Ez], [mp.Ez], f2)
```

#### Key Takeaways

- `integrate_field_function` accepts a Python callable with signature `f(r, *field_values)` where `r` is a `Vector3` and field values are in the order of the `components` list.
- `integrate2_field_function` computes overlap integrals between two field states; the saved `fields` object must be kept alive during the integral call.
- `max_abs_field_function` returns a scalar (the maximum `|f(r)|` over the volume), useful for mode volume normalization.
- `output_field_function` writes the field function to an HDF5 file using the supplied name as the dataset key.
- Mirror symmetries reduce computation but must be consistent with the field function's parity to avoid cancellation artifacts in integrated quantities.

---

### 10. `test_force.py` — Maxwell Stress Tensor Force Computation

**Physics:** Optical forces on dielectric and metallic objects arise from the Maxwell stress tensor. Meep computes these forces by integrating the stress tensor over a closed surface surrounding the object. This test validates the `add_force`/`get_forces` API, the store/load cycle for force data, and the `decimation_factor` consistency for the force monitor.

**Difficulty:** Intermediate

**Source:** `python/tests/test_force.py`

**Test Status:** PASS (11.5 s)

#### Theory

The electromagnetic force on a volume V (and any matter within it) can be computed as a surface integral of the Maxwell stress tensor T over a closed surface S surrounding V:

```
F_i = surface_integral T_ij n_j dA
```

where the stress tensor components are:

```
T_ij = E_i D_j + H_i B_j - (1/2) delta_ij (E·D + H·B)
```

In the frequency domain (using DFT fields), the time-averaged force per unit frequency is:

```
<F_i(omega)> = (1/2) Re[ surface_integral T_ij(omega) n_j dA ]
```

Meep accumulates the DFT of all stress tensor components at the specified frequencies and integrates them over the force surface region. The force region is a flat surface (or closed surface built from multiple flat regions) specified by `ForceRegion`.

For a `ForceRegion` with `direction=mp.Y`, the integration computes the y-component of force on everything enclosed below the surface. The surface size and position determine which objects' forces are captured.

The test uses a simple Gaussian point source in an empty cell (no dielectric objects), so the force on the "enclosed" region is purely the Abraham-Minkowski momentum of the radiation field. The expected force value `-0.11039089113393187` is a regression value from a known-good run.

The `decimation_factor` test verifies that force values computed with `decimation_factor=10` match those from `decimation_factor=1` to 6-7 decimal places (single vs double precision), confirming that force decimation introduces negligible error for source frequencies well below the Nyquist limit.

#### Code Walkthrough

The force monitor is set up before running:

```python
fr = mp.ForceRegion(mp.Vector3(y=1.27), direction=mp.Y, size=mp.Vector3(4.38))
self.myforce = self.sim.add_force(fcen, 0, 1, fr, decimation_factor=1)
self.myforce_decimated = self.sim.add_force(fcen, 0, 1, fr, decimation_factor=10)
```

The simulation runs until the fields have decayed to 1e-6 of their peak value:

```python
self.sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-6))
```

The store/load cycle (analogous to the flux-subtraction trick for reflection measurements) is tested:

```python
fdata = self.sim.get_force_data(self.myforce)
self.sim.load_force_data(self.myforce, fdata)
```

Force values are retrieved and asserted:

```python
self.sim.display_forces(self.myforce)    # prints to stdout
f = mp.get_forces(self.myforce)
self.assertAlmostEqual(f[0], -0.11039089113393187)
```

The decimated force is compared against the undecimated force:

```python
self.assertAlmostEqual(f[0], mp.get_forces(self.myforce_decimated)[0], places=places)
```

#### Key Takeaways

- `add_force(fcen, df, nfreq, ForceRegion)` adds a Maxwell stress tensor surface monitor.
- `ForceRegion(center, direction, size)` specifies the surface: `direction` is the normal to the surface and the component of force computed; `size` specifies the surface extent.
- `mp.get_forces(force_obj)` returns a list of force values, one per DFT frequency.
- `get_force_data`/`load_force_data` enable the subtraction trick (background-subtraction for net force on a scatterer): run background simulation, save force data, add scatterer, load background force, run again — the result is the net force due to scattering.
- Force monitors support `decimation_factor` with the same accuracy trade-off as energy and DFT field monitors.

---

### 11. `test_get_point.py` — Point-Wise Field and Epsilon Sampling with Custom Material Functions

**Physics:** This test validates `get_field_point` and `get_epsilon_point` for a simulation with a spatially varying permittivity defined by a Python function. The material is a sinusoidal function of radial distance, making this an important test for Meep's material function evaluation pipeline.

**Difficulty:** Beginner

**Source:** `python/tests/test_get_point.py`

**Test Status:** PASS (3.7 s)

#### Theory

Meep allows the permittivity at each grid point to be specified via a Python function that maps position to a `Medium` object. This is done by passing a callable as the `material` argument to a geometric object. Internally, Meep evaluates this function at each Yee grid location during `init_sim`, stores the result in the epsilon array, and thereafter treats it as a conventional spatially varying epsilon.

The test uses a sinusoidal radial permittivity:

```python
def sinusoid(p):
    r = (p.x**2 + p.y**2)**0.5
    return mp.Medium(index=1.0 + math.sin(2*math.pi*r)**2)
```

This creates an index that oscillates between 1.0 and 2.0 as the radius changes, forming concentric rings of varying refractive index. The material is applied to a Block covering the entire cell.

`get_field_point(component, position)` returns the interpolated field value at an arbitrary sub-grid position (Meep uses trilinear interpolation between the nearest Yee grid points). `get_epsilon_point(position)` similarly returns the interpolated permittivity.

The test checks 29 field values and 29 epsilon values along two horizontal scan lines at different y positions. The values are hardcoded regression values from a known-good run, verifying both the material function evaluation and the field interpolation pipeline.

#### Code Walkthrough

The sinusoidal material is defined as a Python function and used as the block material:

```python
def sinusoid(p):
    r = (p.x**2 + p.y**2)**0.5
    return mp.Medium(index=1.0 + math.sin(2*math.pi*r)**2)

geometry = [mp.Block(center=mp.Vector3(), size=mp.Vector3(sxy, sxy), material=sinusoid)]
```

The simulation uses Bloch-periodic boundary conditions (`k_point=mp.Vector3()` means Gamma point) and two mirror symmetries to reduce cost:

```python
sim = mp.Simulation(..., k_point=mp.Vector3(),
                    symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y)])
sim.run(until_after_sources=100)
```

The regression check samples points along a scan line:

```python
x = np.linspace(-0.865692, 2.692867, 29)
for j in range(x.size):
    self.assertAlmostEqual(
        np.real(sim.get_field_point(mp.Ez, mp.Vector3(x[j], -0.394862))),
        ez_ref[j], places=places)
    self.assertAlmostEqual(
        sim.get_epsilon_point(mp.Vector3(x[j], 2.967158)),
        eps_ref[j], places=places)
```

The `places` parameter switches between 5 (single precision) and 10 (double precision) significant figures, reflecting the differing floating-point accuracy of the two build modes.

#### Key Takeaways

- Material functions (callables mapping `Vector3` to `Medium`) enable spatially varying permittivities without explicit grid pre-computation.
- `get_field_point(component, position)` returns the trilinearly interpolated complex field at any sub-grid position.
- `get_epsilon_point(position)` returns the interpolated scalar permittivity (or tensor diagonal for anisotropic media).
- Both functions work on arbitrary positions, including non-grid-aligned points, using interpolation.
- Regression tests with hardcoded expected values are an effective way to detect subtle numerical regressions from algorithm or library changes.

---

### 12. `test_get_epsilon_grid.py` — Epsilon Grid Extraction with Complex Geometry and MaterialGrid

**Physics:** This test validates `get_epsilon_grid` — a bulk array extraction function that computes the permittivity on an arbitrary user-specified coordinate grid — against `get_epsilon_point` (single-point extraction) for a complex geometry containing cylinders, blocks, a `MaterialGrid` (topology-optimization array), and a prism with dispersive materials.

**Difficulty:** Advanced

**Source:** `python/tests/test_get_epsilon_grid.py`

**Test Status:** FAIL — requires `parameterized` package (`pip install parameterized`)

#### Theory

`get_epsilon_grid(xv, yv, zv, frequency)` evaluates the permittivity tensor on a user-specified grid of coordinates `(xv, yv, zv)` at the given frequency. This is more efficient than calling `get_epsilon_point` in a loop when many points are needed, as it bypasses the Python-C++ call overhead for each point.

The geometry in this test is deliberately complex:
- A `Cylinder` with a fixed `Medium(index=1.5)`
- A `Block` with `SiN` (silicon nitride, a dispersive material from `meep.materials`)
- A `Block` containing a `MaterialGrid` — a pixelated array of permittivity values used in topology optimization
- A `Prism` with `Co` (cobalt, a dispersive metal)

`MaterialGrid` stores a 2D array of weights interpolated between two materials (here `mp.air` and `mp.Medium(index=3.5)`). The `do_averaging=False` and `beta=0` settings disable threshold projection and spatial averaging, keeping the permittivity distribution as a direct function of the binary weight array.

The frequency-dependent permittivity of `SiN` and `Co` means the result of `get_epsilon_grid` is complex-valued at non-zero frequency. The test verifies that `get_epsilon_grid` and `get_epsilon_point` return the same complex permittivity at four test points spanning different material regions.

The `@parameterized.parameterized.expand(...)` decorator generates four separate test methods from the single `test_get_epsilon_grid` method body, one for each `(point, frequency)` pair. The failure to find the `parameterized` package prevents test discovery entirely.

#### Code Walkthrough

The MaterialGrid is constructed from a binary weight array representing a ring-shaped region:

```python
weights = np.logical_and(
    np.sqrt(xv**2 + yv**2) > rad,
    np.sqrt(xv**2 + yv**2) < rad + w,
)
matgrid = mp.MaterialGrid(mp.Vector3(Nx, Ny), mp.air, mp.Medium(index=3.5),
                          weights=weights, do_averaging=False, beta=0, eta=0.5)
```

The geometry assembles all four object types:

```python
geometry = [
    mp.Cylinder(center=mp.Vector3(0.35, 0.1), radius=0.1, material=mp.Medium(index=1.5)),
    mp.Block(center=mp.Vector3(-0.15, -0.2), size=mp.Vector3(0.2, 0.24), material=SiN),
    mp.Block(center=mp.Vector3(-0.2, 0.2),   size=mp.Vector3(0.4, 0.4),  material=matgrid),
    mp.Prism(vertices=[...], height=0.5, material=Co),
]
```

The parameterized test compares single-point and grid extraction at each test location:

```python
@parameterized.parameterized.expand([
    (mp.Vector3(0.2, 0.2), 1.1),
    (mp.Vector3(-0.2, 0.1), 0.7),
    ...
])
def test_get_epsilon_grid(self, pt, freq):
    eps_grid = self.sim.get_epsilon_grid(
        np.array([pt.x]), np.array([pt.y]), np.array([0]), freq)
    eps_pt = self.sim.get_epsilon_point(pt, freq)
    self.assertAlmostEqual(np.real(eps_grid), np.real(eps_pt), places=6)
    self.assertAlmostEqual(np.imag(eps_grid), np.imag(eps_pt), places=6)
```

To run this test, install the `parameterized` package: `pip install parameterized`.

#### Key Takeaways

- `get_epsilon_grid(xv, yv, zv, freq)` is more efficient than repeated `get_epsilon_point` calls when retrieving permittivity on a dense grid.
- `MaterialGrid` enables topology-optimization parameterizations; the weight array maps linearly (or via threshold projection with `beta > 0`) between two materials.
- Dispersive materials (`SiN`, `Co` from `meep.materials`) produce complex-valued permittivity at finite frequency; `get_epsilon_grid` correctly returns complex values in this case.
- The `parameterized` package (not bundled with conda-forge pymeep) is required for `@parameterized.parameterized.expand`; install separately with `pip install parameterized`.
- `eps_averaging=False` disables subpixel smoothing for cleaner unit testing of permittivity retrieval.

---

### 13. `test_array_metadata.py` — Modal Volume via DFT Arrays and CW Solver Comparison

**Physics:** This test computes the modal volume of a ring resonator's resonant mode using two independent methods — the CW solver with `integrate_field_function` / `max_abs_field_function`, and a pulsed source with `get_dft_array` / `get_array_metadata` — and verifies they agree to within 1-5% depending on precision.

**Difficulty:** Advanced

**Source:** `python/tests/test_array_metadata.py`

**Test Status:** TIMEOUT (CPU-intensive; high-resolution 2D simulation with CW solver plus pulsed DFT run)

#### Theory

The modal (effective mode) volume is a key figure of merit for cavity quantum electrodynamics and nonlinear optics. It quantifies the spatial confinement of the electromagnetic mode:

```
V_mode = (integral epsilon |E|^2 dV) / max(epsilon |E|^2)
```

For many applications a weighted variant is used, replacing the plain volume integral with a function-weighted version:

```
V_eff = [integral epsilon |E|^2 dV / max(epsilon |E|^2)] * integral f(r) dV
```

where `f(r) = x^2 + 2y^2` is an example spatial weighting function. This formulation arises in perturbation theory for Raman or Kerr nonlinear processes.

The test computes this quantity two ways:

**Method 1 (CW solver):** Uses `solve_cw` to obtain steady-state complex fields, then applies `integrate_field_function` and `max_abs_field_function` directly to the running fields.

**Method 2 (DFT / pulse):** Uses a Gaussian source, accumulates DFT fields via `add_dft_fields`, retrieves the DFT array with `get_dft_array`, and obtains the spatial coordinates and integration weights via `get_array_metadata`. The integration is performed in NumPy.

`get_array_metadata(dft_cell=obj)` returns `(X, Y, Z, W)` where X, Y, Z are 1D coordinate arrays and W is a 2D (or 3D) array of integration weights appropriate for the non-uniform Yee grid. This allows accurate numerical integration of DFT arrays using:

```python
integral = np.sum(W * integrand)
```

without having to assume uniform grid spacing.

The 5% tolerance (`tol = 0.05`) for single precision and 1% for double reflects the combined error from finite resolution, finite source bandwidth, and CW solver convergence tolerance.

#### Code Walkthrough

CW method: set `force_complex_fields=True`, call `solve_cw`, then use field integration functions:

```python
sim = mp.Simulation(..., force_complex_fields=True, ...)
sim.init_sim()
sim.solve_cw(1e-6, 1000, 10)

def electric_energy(r, ez, eps):
    return np.real(eps * np.conj(ez) * ez)

electric_energy_total = sim.integrate_field_function(
    [mp.Ez, mp.Dielectric], electric_energy, nonpml_vol)
electric_energy_max = sim.max_abs_field_function(
    [mp.Ez, mp.Dielectric], electric_energy, nonpml_vol)
vec_func_total = sim.integrate_field_function([], vec_func, nonpml_vol)
cw_modal_volume = (electric_energy_total / electric_energy_max) * vec_func_total
```

Pulsed DFT method: use `get_dft_array` and `get_array_metadata` for NumPy-based integration:

```python
dft_obj = sim.add_dft_fields([mp.Ez], fcen, 0, 1, where=nonpml_vol)
sim.run(until_after_sources=100)
Ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
(X, Y, Z, W) = sim.get_array_metadata(dft_cell=dft_obj)
Eps = sim.get_array(vol=nonpml_vol, component=mp.Dielectric)
EpsE2 = np.real(Eps * np.conj(Ez) * Ez)
xm, ym = np.meshgrid(X, Y)
vec_func_sum = np.sum(W * (xm**2 + 2*ym**2))
pulse_modal_volume = np.sum(W * EpsE2) / np.max(EpsE2) * vec_func_sum
```

The comparison:

```python
self.assertClose(cw_modal_volume / pulse_modal_volume, 1.0, epsilon=tol)
```

#### Key Takeaways

- `get_array_metadata(dft_cell=obj)` provides spatial coordinates (X, Y, Z) and quadrature weights (W) that enable accurate NumPy-based integration of DFT arrays on the non-uniform Yee grid.
- Two independent methods (CW solver and pulsed DFT) give the same modal volume, providing mutual validation.
- The integration weights W are not simply the uniform cell area (`dx * dy`); they account for the staggered Yee grid and any half-cell boundary corrections.
- `max_abs_field_function` and `np.max` give the same field maximum only if the integration volume resolution is sufficient; `max_abs_field_function` evaluates at actual Yee grid points while `np.max(EpsE2)` operates on the array returned by `get_dft_array`.
- CW solver convergence (`tol=1e-6`) limits the accuracy of the CW modal volume; the pulsed DFT result is limited instead by source bandwidth and run time.

---

### 14. `test_integrated_source.py` — Integrated Source for Plane Waves Extending into PML

**Physics:** A uniform plane wave requires the source to extend all the way to the PML boundaries. Without the `is_integrated=True` flag, a source that terminates at a sharp edge at the PML interface produces diffraction from the edge discontinuity, corrupting the plane-wave uniformity. The `is_integrated` flag modifies the source amplitude to compensate for this edge effect.

**Difficulty:** Beginner

**Source:** `python/tests/test_integrated_source.py`

**Test Status:** PASS (6.3 s)

#### Theory

In an FDTD simulation with PML boundary conditions, a source current sheet placed at position x = x_0 launches a forward-propagating wave. If the source extends to the PML boundary, the PML absorbs the edges of the source — but this absorption is not instantaneous; the PML is a continuous absorbing medium with a spatially graded conductivity. As a result, the source current at the PML interface effectively "terminates" at a position where the PML conductivity is non-zero, creating a source edge that diffracts.

The `is_integrated=True` parameter tells Meep that the source represents the integral of an underlying current density that has zero divergence when it extends continuously into the PML. Mathematically, this changes the discretization of the source term at the boundary between the active region and the PML, compensating for the discontinuity artifact.

The physical consequence is dramatic: without `is_integrated=True`, a line source in a 2D simulation that extends to the PML produces a non-uniform field with amplitude variations across the downstream cross-section (the wave is not a true plane wave). With `is_integrated=True`, the field is uniform to high precision.

The test quantifies this by measuring the normalized standard deviation of the Ez field along a transverse cross-section far downstream from the source:

```
sigma = std(E_z) / sqrt(mean(E_z^2))
```

For a perfect plane wave this should be zero. The test asserts it is less than `1e-4` (single precision) or `1e-8` (double precision), confirming near-perfect plane-wave uniformity.

The source uses `ContinuousSource` (not `GaussianSource`) because plane-wave uniformity in the steady state is what is being tested, and steady-state uniformity requires a monochromatic source. A Bloch-periodic `k_point=mp.Vector3()` (Gamma point) ensures periodic boundary conditions in the transverse direction, eliminating wrap-around artifacts.

#### Code Walkthrough

The source is a uniform line current with `is_integrated=True`:

```python
sources = [mp.Source(
    mp.ContinuousSource(1, is_integrated=True),
    center=mp.Vector3(-2),
    size=mp.Vector3(y=6),    # extends full cell height (into PML)
    component=mp.Ez,
)]
```

The cell is 6x6 with 1-unit PML, so `size=mp.Vector3(y=6)` ensures the source extends 1 unit into the PML on each side. `k_point=mp.Vector3()` enforces periodic (not PML) transverse boundary conditions in this test:

```python
sim = mp.Simulation(
    resolution=20,
    cell_size=(6, 6),
    boundary_layers=[mp.PML(thickness=1)],
    sources=sources,
    k_point=mp.Vector3(),    # Gamma point: transverse periodicity
)
sim.run(until=30)
```

The field uniformity is quantified:

```python
ez = sim.get_array(mp.Ez, center=mp.Vector3(2), size=mp.Vector3(y=6))
std = np.std(ez) / np.sqrt(np.mean(ez**2))
self.assertAlmostEqual(std, 0.0, places=4 if mp.is_single_precision() else 8)
```

`get_array` with a zero x-extent extracts Ez along the transverse line at x=2.

#### Key Takeaways

- `is_integrated=True` on a `ContinuousSource` (or `GaussianSource`) is required for sources that extend into or through the PML to produce uniform plane waves.
- The `k_point=mp.Vector3()` (Gamma point) activates Bloch-periodic boundary conditions in the transverse direction; for a plane wave propagating purely in x this gives purely periodic (not PML) transverse boundaries.
- `get_array(component, center, size)` extracts a 1D slice of the field when `size` has zero thickness in one direction.
- The normalized standard deviation `std / sqrt(mean(|E|^2))` is a dimensionless measure of plane-wave uniformity; perfect uniformity gives zero.
- This behavior is documented in the Meep manual under "Perfectly Matched Layer — Planewave Sources Extending into PML" and corresponds to regression test for GitHub issue #2043.
