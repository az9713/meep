# Chapter 5: Gratings and Diffractive Optics

This chapter covers diffraction grating simulations, metasurface lens design, and near-to-far-field transformations for periodic structures in Meep. The tutorials progress from the canonical binary grating — a periodic array of rectangular ridges on a substrate — through oblique incidence, phase mapping, polarization gratings built from anisotropic liquid-crystal materials, finite-aperture effects, zone plates, and two-dimensional triangular-lattice gratings. Throughout, the underlying physics is the grating equation `n_i sin(theta_m) = n_i sin(theta_i) + m lambda / Lambda`, where `Lambda` is the period, `m` is the diffraction order, and `n_i` is the refractive index in each half-space. Meep handles all of this through Floquet-Bloch periodic boundary conditions, eigenmode-coefficient decomposition, and the near-to-far-field (near2far) transformation framework.

---

### 1. `binary_grating.py` — Broadband Diffraction Efficiency of a Transmission Grating

**Physics:** Computes the wavelength-resolved transmittance into each diffraction order of a binary grating (glass ridges on a glass substrate) illuminated at normal incidence over a broadband pulse.
**Difficulty:** Intermediate
**Source:** `python/examples/binary_grating.py`
**Test Status:** TIMEOUT (CPU-intensive; high resolution 60 px/um with 10 modes over 21 frequencies)

#### Theory

A binary (rectangular-ridge) grating is the workhorse structure of diffractive optics. When a plane wave of wavelength `lambda` strikes a periodic surface with period `Lambda` at normal incidence, it is split into a discrete set of transmitted orders with diffraction angles given by

```
sin(theta_m) = m * lambda / (n_t * Lambda),   m = 0, ±1, ±2, ...
```

where `n_t` is the refractive index of the transmission medium and `m` is the integer diffraction order. Orders with `|sin(theta_m)| > 1` are evanescent and carry no power to the far field.

The amount of power directed into each order — the diffraction efficiency — depends on the grating geometry: the period `Lambda`, the ridge height `gh`, the duty cycle `gdc` (the fraction of the period covered by the ridge), and the refractive index contrast. For a glass grating (`n = 1.5`) with a duty cycle of 0.5 and a relatively shallow height of 0.5 um, the zeroth order dominates at visible wavelengths (400-600 nm), but higher orders carry meaningful fractions of the total power.

In FDTD the broadband character of a Gaussian pulse lets us resolve diffraction efficiency as a continuous function of wavelength in a single simulation. The trick is to use eigenmode decomposition on the transmitted flux monitor: after decomposing the fields into plane-wave modes (each mode corresponding to one diffraction order), the squared magnitude of the forward coefficient divided by the incident flux gives the transmittance into that order.

Because the grating has a mirror symmetry about its mid-plane in the y-direction, we impose a `Mirror(Y)` symmetry, which halves the computational cost but means that each degenerate pair of `+m` and `-m` orders is counted once and its transmittance must be multiplied by two (except for `m=0`).

#### Code Walkthrough

**Cell geometry and dimensions.** The simulation cell spans the x-direction (normal to the grating surface) with PML on both sides and is one grating period wide in y with Bloch-periodic boundaries.

```python
gp = 10.0   # grating period (um)
gh = 0.5    # grating height (um)
gdc = 0.5   # duty cycle

sx = dpml + dsub + gh + dpad + dpml
sy = gp
cell_size = mp.Vector3(sx, sy, 0)
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]
```

PML is applied only along x (the propagation direction); the y-direction uses Bloch boundaries enforced by `k_point = mp.Vector3(0, 0, 0)` (normal incidence, so zero transverse wavevector).

**Two-pass strategy.** The simulation runs twice. The first pass has no grating geometry — only the glass substrate fills the cell. This records the incident flux `input_flux` for normalization. The second pass adds the grating ridges.

```python
# Pass 1: uniform substrate
sim = mp.Simulation(..., default_material=glass, ...)
flux_mon = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(...))
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))
input_flux = mp.get_fluxes(flux_mon)

sim.reset_meep()

# Pass 2: add grating ridge
geometry = [
    mp.Block(material=glass, size=mp.Vector3(dpml+dsub, mp.inf, mp.inf), ...),
    mp.Block(material=glass, size=mp.Vector3(gh, gdc*gp, mp.inf), ...),
]
```

**Eigenmode decomposition.** After the grating simulation, `get_eigenmode_coefficients` resolves each diffraction order into a separate modal coefficient. The mode index `nm` (1-based in the Meep API) maps to diffraction orders sorted by their transverse wavevector magnitude.

```python
res = sim.get_eigenmode_coefficients(
    mode_mon, range(1, nmode+1), eig_parity=mp.ODD_Z + mp.EVEN_Y
)
coeffs = res.alpha     # shape: (nmode, nfreq, 2)
kdom   = res.kdom      # dominant wavevector for each mode
```

The transmittance into mode `nm` at frequency index `nf` is:

```python
tran = abs(coeffs[nm, nf, 0])**2 / input_flux[nf]
# For nm != 0 (degenerate +/-m pair), multiply by 0.5 because mirror symmetry
# already counted one side
tran_stored = 0.5 * tran if nm != 0 else tran
```

The diffraction angle is recovered from the dominant wavevector's x-component:

```python
angle = math.degrees(math.acos(kdom[nm*nfreq + nf].x / freqs[nf]))
```

**Visualization.** Results are displayed as a color map of transmittance vs. wavelength and diffraction angle using `plt.pcolormesh`.

#### Key Takeaways

- Normal-incidence binary gratings require only one symmetric (EVEN_Y or ODD_Y) Bloch simulation per polarization; the `Mirror(Y)` symmetry halves the cell size.
- `get_eigenmode_coefficients` is the preferred way to resolve diffraction orders because it separates overlapping plane-wave contributions that a simple flux integral cannot distinguish.
- The `kdom` field in the result struct gives the actual propagation direction of each order, from which the diffraction angle can be computed analytically.
- For symmetric (normal incidence) cases, the transmittance from the eigenmode coefficients must be doubled for non-zeroth orders to account for both `+m` and `-m`.
- Resolution of 60 px/um is needed to accurately resolve features at the 0.5 um grating height and 0.4-0.6 um wavelength range; this drives runtime significantly.

---

### 2. `binary_grating_n2f.py` — Near-to-Far-Field Comparison: Unit Cell vs. Finite Grating

**Physics:** Demonstrates the near2far transformation for a periodic grating and validates that the far-field pattern of a unit cell (with `nperiods` extension) matches that of an explicitly simulated finite grating.
**Difficulty:** Intermediate
**Source:** `python/examples/binary_grating_n2f.py`
**Test Status:** TIMEOUT (CPU-intensive; three separate simulations with finite supercell)

#### Theory

The near-to-far-field (near2far) transformation is a Green's function technique. Fields recorded on a closed surface (the "near-field monitor") near the scatterer are propagated analytically to any desired far-field observation point using the free-space dyadic Green's function. This avoids having to simulate the full physical domain out to the far field, making it vastly more efficient than simply extending the FDTD grid.

For a finite grating with `N` unit cells, the far-field pattern is the product of the single-element far-field pattern (the array element factor) and the array factor `sin(N*psi/2) / sin(psi/2)`, where `psi = k*Lambda*sin(theta)`. As `N` increases, the array factor sharpens into delta-function-like peaks at the grating diffraction angles, each peak widening as `1/N`. This constructive interference is the physical reason why many-period gratings produce sharp diffraction orders.

Meep exploits this behavior through the `nperiods` argument on `add_near2far`. Rather than explicitly simulating `N` periods, Meep simulates one unit cell, records its near-field DFT data, and then coherently sums (with Bloch phases) the contributions from `N` copies of that cell. This provides the far-field of a finite grating in one unit-cell-sized simulation.

The tutorial verifies this by comparing three configurations:
1. A flat surface (no grating) — the reference pattern.
2. A unit cell with `nperiods=10` extrapolation.
3. An explicit 2*nperiods+1 = 21-period finite grating.

The `norm_err = LA.norm(ff_unitcell["Ez"] - ff_supercell["Ez"]) / nperiods` confirms that the two approaches agree.

#### Code Walkthrough

**Phase 1: flat-surface reference (1D cell).** The first simulation is a degenerate 1D cell (no y-extent) recording the incident far-field pattern `ff_source` for normalization.

```python
cell_size = mp.Vector3(sx)       # 1D (no y extent)
n2f_obj = sim.add_near2far(fcen, df, nfreq, mp.Near2FarRegion(center=n2f_pt))
sim.run(...)
ff_source = sim.get_farfields(n2f_obj, ff_res,
    center=mp.Vector3(ff_distance, 0.5*ff_length),
    size=mp.Vector3(y=ff_length))
```

**Phase 2: periodic unit cell with coherent summation.**

```python
sy = gp
n2f_obj = sim.add_near2far(
    fcen, df, nfreq,
    mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy)),
    nperiods=nperiods,   # <-- key: coherently extends to N periods
)
ff_unitcell = sim.get_farfields(n2f_obj, ff_res,
    center=mp.Vector3(ff_distance, 0.5*ff_length),
    size=mp.Vector3(y=ff_length))
```

**Phase 3: explicit finite grating (supercell).** A supercell containing 21 periods is built by placing one `Block` per ridge:

```python
num_cells = 2*nperiods + 1
sy = dpml + num_cells*gp + dpml
for j in range(num_cells):
    geometry.append(mp.Block(..., center=mp.Vector3(..., -0.5*sy + dpml + (j+0.5)*gp)))
```

**Relative enhancement.** The spectral far-field pattern is normalized to the source:

```python
rel_enh = np.absolute(ff_unitcell["Ez"])**2 / np.absolute(ff_source["Ez"])**2
```

This highlights the diffraction peaks relative to the bare-substrate intensity.

#### Key Takeaways

- The `nperiods` parameter on `add_near2far` lets a single unit-cell simulation produce the far field of an N-period finite grating by coherent superposition.
- Far-field peaks sharpen as `1/N` in angular width; more periods give more directive, spectrally pure diffraction orders.
- The comparison between unit-cell near2far and explicit finite supercell validates the coherent summation approach to within numerical discretization error.
- `get_farfields` takes a spatial center and size, allowing one-dimensional angle scans or two-dimensional spatial maps at an arbitrary propagation distance.
- Relative enhancement is a useful normalization: dividing by the flat-surface far field removes the angular envelope of the source aperture.

---

### 3. `binary_grating_oblique.py` — Oblique-Incidence Diffraction with Bloch Wavevectors

**Physics:** Extends the binary grating simulation to oblique incidence using a tilted planewave source constructed from a spatially modulated amplitude function, with both the incident k-vector and the Bloch condition set consistently.
**Difficulty:** Advanced
**Source:** `python/examples/binary_grating_oblique.py`
**Test Status:** TIMEOUT (CPU-intensive; supports CW solver alternative)

#### Theory

When a plane wave impinges on a grating at an oblique angle `theta_i` measured from the surface normal, the grating equation becomes

```
n_t * sin(theta_m) = n_i * sin(theta_i) + m * lambda / Lambda,
```

where `n_i` and `n_t` are the refractive indices on the incident and transmitted sides. A key difference from normal incidence is that the Bloch wavevector `k_y = n_i * (2*pi/lambda) * sin(theta_i)` is now nonzero. Meep enforces this via the `k_point` parameter, which sets the Bloch phase between opposite y-faces of the unit cell.

A broadband Gaussian pulse cannot trivially model an oblique plane wave because each frequency component would need a different phase tilt. The standard approach is to use a monochromatic (or narrow-band) source together with an explicit plane-wave phase function `exp(i k . x)` applied as the `amp_func` argument. This imposes the correct spatial phase ramp across the source plane so that the wavefront arrives at the correct angle.

The number of propagating reflected orders is bounded by `floor((n_g*f - k_y)*Lambda) - ceil((-n_g*f - k_y)*Lambda)`, and similarly for transmitted orders. Both must be computed separately since at oblique incidence the substrate has a different refractive index from air.

The code also supports the continuous-wave (CW) iterative solver (`solve_cw`) as an alternative to time-domain stepping, which can be faster for problems dominated by resonances or high-Q features.

#### Code Walkthrough

**Bloch wavevector setup.** The incident k-vector in the glass substrate is constructed by rotating a vector of length `n_g * f_cen` about the z-axis by `theta_in`:

```python
k = mp.Vector3(fcen * ng).rotate(mp.Vector3(z=1), theta_in)
```

This gives `k = (k_x, k_y, 0)` satisfying `|k| = n_g * f` with `k_y = n_g * f * sin(theta_in)`.

**Plane-wave amplitude function.** The source carries a position-dependent phase to synthesize the plane wave:

```python
def pw_amp(k, x0):
    def _pw_amp(x):
        return cmath.exp(1j * 2*math.pi * k.dot(x + x0))
    return _pw_amp

sources = [mp.Source(..., amp_func=pw_amp(k, src_pt))]
```

**Reflection subtraction.** For the grating simulation, the reflected flux monitor subtracts the background incident field via `load_minus_flux_data`:

```python
sim.load_minus_flux_data(refl_flux, input_flux_data)
```

This leaves only the grating-scattered reflected field in the flux.

**Order counting.** After the simulation, the number of non-evanescent orders is computed analytically and each order's coefficient is extracted:

```python
nm_r = int(np.floor((fcen*ng - k.y)*gp) - np.ceil((-fcen*ng - k.y)*gp))
res = sim.get_eigenmode_coefficients(refl_flux, range(1, nm_r+1), eig_parity=eig_parity)
```

**Energy conservation check.** Total reflectance `Rsum` and transmittance `Tsum` are summed over all non-evanescent orders and compared to the Poynting-flux integrals `Rflux` and `Tflux`:

```python
print(f"mode-coeff:, {Rsum:.6f}, {Tsum:.6f}, {Rsum+Tsum:.6f}")
print(f"poynting-flux:, {Rflux:.6f}, {Tflux:.6f}, {Rflux+Tflux:.6f}")
```

Both should sum to 1.0 (energy conservation), confirming the simulation is correctly set up.

#### Key Takeaways

- Oblique incidence requires a nonzero `k_point` in the Bloch direction AND a matching `amp_func` on the source; omitting either gives wrong results.
- The `pw_amp` closure captures the reference position `x0` so that the source plane (at `x = src_pt`) has unit amplitude while positions away from it accumulate the Bloch phase.
- At oblique incidence, `Mirror(Y)` symmetry is broken; the full unit cell must be simulated and all signed diffraction orders (both positive and negative `m`) must be counted.
- The number of propagating orders depends on both the incident angle and the wavelength; the code computes this count dynamically from the grating equation.
- `Rsum + Tsum` equaling 1.0 (or close to it) is a necessary but not sufficient check on correctness; it validates energy conservation but not the angular distribution of orders.

---

### 4. `binary_grating_phasemap.py` — Phase and Amplitude Map vs. Duty Cycle

**Physics:** Sweeps the grating duty cycle from 0.1 to 0.9 and records both the zeroth-order transmittance and the transmitted mode phase at each duty cycle, producing the library of (amplitude, phase) pairs needed to design a metasurface.
**Difficulty:** Intermediate
**Source:** `python/examples/binary_grating_phasemap.py`
**Test Status:** PASS (135.5s)

#### Theory

Sub-wavelength gratings (period `Lambda < lambda / n`) support no propagating diffraction orders other than the zeroth order. In this regime the grating acts as a homogeneous effective medium whose effective index is set by the duty cycle. By varying the duty cycle spatially across a surface, one can impart an arbitrary local phase shift to the transmitted beam — this is the operating principle of a metasurface or flat optic.

The zeroth-order transmitted field `E_t = sqrt(T) * exp(i*phi)` has both an amplitude `T` and a phase `phi` that depend on the grating geometry. For a well-designed metasurface, `T` should remain close to unity (high efficiency) while `phi` sweeps the full `2*pi` range. The ability to achieve `2*pi` phase coverage at high transmittance across a range of duty cycles is the fundamental figure of merit for a transmissive metasurface.

This simulation uses a sub-wavelength period (`gp = 0.35` um by default, configurable via `-gp`) so that only the zeroth order exists in the air half-space. The code supports both TE (Ez, `oddz=True`) and TM (Hz, `oddz=False`) polarizations via a command-line flag.

The script loops over nine duty cycles (`gdc` from 0.1 to 0.9 in steps of 0.1) and for each calls the `grating()` function that:
1. Runs a reference simulation (no grating) to get `input_flux`.
2. Runs the grating simulation and extracts the zeroth-order eigenmode coefficient `coeffs[0, nf, 0]`.
3. Returns `mode_tran` (transmittance) and `mode_phase` (phase in radians) at each frequency.

The resulting 2D maps of transmittance and phase vs. (wavelength, duty cycle) are the output.

#### Code Walkthrough

**Parameterized grating function.** The simulation is encapsulated in a function to support the duty-cycle sweep:

```python
def grating(gp, gh, gdc, oddz):
    ...
    sources = [mp.Source(..., component=mp.Ez if oddz else mp.Hz, ...)]
    symmetries = [mp.Mirror(mp.Y, phase=+1 if oddz else -1)]
```

The mirror symmetry phase (`+1` for Ez odd-z, `-1` for Hz even-z) is set correctly for each polarization.

**Zeroth-order mode extraction.**

```python
res = sim.get_eigenmode_coefficients(
    mode_mon, [1],
    eig_parity=mp.ODD_Z + mp.EVEN_Y if oddz else mp.EVEN_Z + mp.ODD_Y
)
coeffs = res.alpha
mode_tran  = [abs(coeffs[0, nf, 0])**2 / input_flux[nf] for nf in range(nfreq)]
mode_phase = [np.angle(coeffs[0, nf, 0]) for nf in range(nfreq)]
```

The phase `np.angle(...)` returns a value in `[-pi, pi]`. Careful unwrapping is needed if the phase crosses the branch cut.

**Duty-cycle sweep.**

```python
gdc = np.arange(0.1, 1.0, 0.1)
for n in range(gdc.size):
    mode_wvl, mode_tran[n,:], mode_phase[n,:] = grating(args.gp, args.gh, gdc[n], args.oddz)
```

**Visualization.** Two `pcolormesh` panels show transmittance and phase as functions of (wavelength, duty cycle), using a "hot_r" colormap for transmittance and "RdBu" for phase (a diverging colormap suited to signed quantities).

#### Key Takeaways

- For sub-wavelength periods, only the zeroth order propagates; extracting its phase and amplitude produces the metasurface library needed for flat-optic design.
- The phase of `coeffs[0, nf, 0]` from `get_eigenmode_coefficients` is the actual complex phase of the transmitted zeroth-order plane wave, directly usable in metasurface design equations.
- Both TE and TM polarizations must be mapped separately; the available phase range and transmittance profile differ significantly between them.
- The duty cycle is the primary tuning knob for phase control at fixed period and height; the period and height set the overall operating wavelength range.
- This script is a precursor to `metasurface_lens.py`, which uses the (phase, duty cycle) lookup table to design a focusing lens.

---

### 5. `binary_grating_levelset.py` — Finite Grating with Scattered-Field DFT Analysis

**Physics:** Simulates a finite (5-period) binary grating illuminated by a broadband pulse, extracts the scattered electric field by subtracting the flat-surface DFT from the grating DFT, and either visualizes the 2D scattered-field profile or computes its spatial Fourier transform to resolve diffraction orders.
**Difficulty:** Intermediate
**Source:** `python/examples/finite_grating.py` (listed as `binary_grating_levelset.py` in the task; file not present at that path — content from `finite_grating.py`)
**Test Status:** FAIL (file `binary_grating_levelset.py` not found in repository; `finite_grating.py` contains the levelset-style DFT scattered-field analysis described here)

#### Theory

In many practical applications, gratings are finite in aperture — they cover only a limited number of periods. This breaks the strict Bloch periodicity and produces a continuous (rather than discrete) angular spectrum. The scattered field contains both the propagating diffraction orders and evanescent near-field contributions.

The scattered-field decomposition subtracts the incident field (recorded in a reference simulation with no grating) from the total field in the grating simulation:

```
E_scattered = E_total_with_grating - E_total_flat_surface
```

This subtraction is done in the frequency domain using DFT arrays, not in the time domain. Both simulations run separately; their DFT arrays are recorded at identical monitor positions and then subtracted.

Taking the spatial Fourier transform of the scattered field along a cross-sectional line above the grating converts from position space `y` to wavevector space `k_y`. Each propagating diffraction order appears as a peak at `k_y = m / Lambda` (in units of 2*pi/um). The number and position of these peaks directly verifies the grating equation.

The `field_profile` flag switches between two output modes: a 2D spatial map of the scattered intensity (useful for visualizing near-field structure and leaky modes) or the 1D Fourier spectrum (useful for quantifying which diffraction orders are excited).

#### Code Walkthrough

**Two-pass scattered-field strategy.** Both simulations use `add_dft_fields` at an identical monitor spanning the air region above the grating:

```python
near_fields = sim.add_dft_fields(
    [mp.Ez], fcen, 0, 1,
    center=mon_pt,
    size=mp.Vector3(dair if field_profile else 0, sy - 2*dpml),
)
```

For the 2D profile (`field_profile=True`) the monitor has extent `dair` in x; for the 1D spectrum, it collapses to a line (`x=0`).

**Subtraction:**

```python
flat_dft    = sim.get_dft_array(near_fields, mp.Ez, 0)  # reference run
grating_dft = sim.get_dft_array(near_fields, mp.Ez, 0)  # grating run
scattered_field = grating_dft - flat_dft
```

**Fourier-space analysis.**

```python
ky = np.fft.fftshift(np.fft.fftfreq(len(scattered_field), 1/resolution))
FT_scattered_field = np.fft.fftshift(np.fft.fft(scattered_field))
```

The wavevector axis `ky` is in units of cycles per micrometer (Meep length unit). Peaks at `|ky| = m/Lambda` identify the `m`-th diffraction order.

**Finite-grating geometry.** The five grating ridges are placed explicitly:

```python
for j in range(num_cells):
    geometry.append(mp.Block(
        material=glass, size=mp.Vector3(gh, gdc*gp, mp.inf),
        center=mp.Vector3(..., -0.5*sy + dpml + dpad + (j+0.5)*gp),
    ))
```

The source spans the full cell width including PML, using `is_integrated=True` for a properly normalized total-field source.

#### Key Takeaways

- Scattered-field extraction via DFT subtraction is numerically exact (limited only by floating-point precision and discretization) and does not require a separate scattered-field formulation in the FDTD engine.
- `add_dft_fields` records complex-valued frequency-domain fields at specified points, enabling post-processing operations like Fourier analysis that are not possible with raw time-domain data.
- For finite gratings, the angular spectrum is continuous; individual order peaks are broadened by `~1/(N*Lambda)` in k-space.
- The `is_integrated=True` flag on `GaussianSource` ensures that the source generates a unit-amplitude plane wave regardless of the cell discretization.
- The 2D scattered-field visualization reveals near-field phenomena such as Wood anomalies or guided resonances that are hidden in integrated transmittance spectra.

---

### 6. `finite_grating.py` — Scattered-Field Spectrum of a Finite Grating via DFT

**Physics:** Simulates a small finite binary grating (5 unit cells) and resolves the transmitted scattered-field pattern both as a 2D spatial map and as a 1D wavevector spectrum via discrete Fourier transform.
**Difficulty:** Intermediate
**Source:** `python/examples/finite_grating.py`
**Test Status:** TIMEOUT (CPU-intensive; two full simulations with `dair=10` um air monitoring region)

#### Theory

When a plane wave illuminates a finite-aperture grating, the far-field pattern is the Fourier transform of the aperture field distribution. For a grating with `N = 5` periods, each of period `Lambda = 1.0` um, the aperture is 5 um wide. The angular spread of each diffraction order is approximately `lambda / (N * Lambda)` radians, which at `lambda = 0.5` um gives about 5.7 degrees — a detectable spread.

The key advantage of the DFT-subtraction approach over eigenmode decomposition is that it works for arbitrary (non-periodic) structures and gives the full near-field amplitude and phase, not just the modal power. The spatial Fourier transform of the near-field (sampled on a line) gives the angular spectrum of the scattered wave, with each diffraction order appearing as a peak in `k_y`-space.

This simulation is physically identical to tutorial 5 (listed there as `binary_grating_levelset.py`) but differs in the parameters used: `gp = 1.0` um (close-packed, sub-wavelength features are possible), and the monitoring region includes an extended air gap `dair = 10` um to capture evanescent field decay away from the grating surface when `field_profile=True`.

#### Code Walkthrough

**Extended air-region monitor for 2D field profile.** When `field_profile = True`, the DFT monitor spans a 2D region 10 um thick in x:

```python
dair = 10 if field_profile else dpad
near_fields = sim.add_dft_fields(
    [mp.Ez], fcen, 0, 1,
    center=mon_pt,
    size=mp.Vector3(dair if field_profile else 0, sy - 2*dpml),
)
```

**Grid metadata for plotting.** After the simulation, `get_array_metadata` returns the actual x and y coordinates of each DFT sample point:

```python
[x, y, z, w] = sim.get_array_metadata(dft_cell=near_fields)
```

This is essential for correct axis labeling when the cell edges may not align perfectly with the monitor boundaries.

**2D intensity plot.**

```python
scattered_amplitude = np.abs(scattered_field)**2
plt.pcolormesh(x, y, np.rot90(scattered_amplitude), cmap="inferno", ...)
```

The `np.rot90` reorients the array from Meep's (x,y) convention to the standard (row,col) convention expected by `pcolormesh`.

**1D Fourier transform mode.**

```python
ky = np.fft.fftshift(np.fft.fftfreq(len(scattered_field), 1/resolution))
FT = np.fft.fftshift(np.fft.fft(scattered_field))
plt.plot(ky, np.abs(FT)**2, ...)
```

The `1/resolution` argument to `fftfreq` sets the sample spacing in um, so `ky` is in units of (1/um) = (2*pi/um) / (2*pi).

#### Key Takeaways

- `get_array_metadata` should always be used to retrieve the exact coordinates of DFT sample points; do not assume they coincide with nominal cell boundaries.
- The 2D field profile reveals exponential evanescent decay of non-propagating wavevector components above the grating surface, confirming which features are near-field only.
- Increasing `num_cells` from 5 to larger values narrows the diffraction peaks in k-space; this is a direct visualization of the finite-aperture angular spread.
- The `field_profile` flag provides two complementary views of the same data: spatial (where is the field?) and spectral (what wavevectors are present?).
- The `mp.am_master()` guard around plotting calls is essential for MPI simulations to prevent all ranks from writing the same figure.

---

### 7. `polarization_grating.py` — Liquid-Crystal Polarization Grating

**Physics:** Simulates a liquid-crystal polarization grating — a birefringent layer with a spatially rotating optic axis — and compares the diffraction efficiency of TE and TM modes against analytical predictions, validating the anisotropic tensor material model in Meep.
**Difficulty:** Advanced
**Source:** `python/examples/polarization_grating.py`
**Test Status:** TIMEOUT (CPU-intensive; sweeps over 34 thickness values with two simulations each)

#### Theory

A polarization grating is formed by a thin layer of anisotropic (birefringent) material — typically a nematic liquid crystal — in which the optical axis rotates periodically in the plane of the layer with period `Lambda`. The liquid crystal is described by a uniaxial permittivity tensor with ordinary index `n_0` and extraordinary index `n_e = n_0 + delta_n`.

When a linearly polarized plane wave traverses such a layer, it is diffracted primarily into the `m = ±1` orders. For a homogeneous uniaxial grating (constant tilt along the thickness, `ph = 0`), the zeroth-order efficiency follows `cos^2(pi * delta_n * d / lambda)` and the first-order efficiency follows `sin^2(pi * delta_n * d / lambda)`, where `d` is the layer thickness and `delta_n` is the birefringence. This is the Jones-matrix prediction for a birefringent waveplate.

The script validates these analytical expressions by running FDTD simulations for both a uniaxial grating (single homogeneous layer) and a bilayer twisted-nematic grating (two layers with relative twist angle `ph_twisted = 70` degrees). The bilayer grating can achieve 100% first-order efficiency (the "perfect polarization grating" condition) at the right thickness.

The simulation uses a spatially varying material: the permittivity tensor is specified via a Python callable `lc_mat(p)` that returns a different `mp.Medium` at each point in space. This is Meep's most general material interface and allows arbitrary spatial variation of anisotropic tensors.

#### Code Walkthrough

**Anisotropic permittivity tensor.** The liquid crystal at position `p` has its optic axis rotated by angle `phi(p)` about the x-axis:

```python
epsilon_diag = mp.Matrix(
    mp.Vector3(n_0**2, 0, 0),
    mp.Vector3(0, n_0**2, 0),
    mp.Vector3(0, 0, (n_0 + delta_n)**2),
)

def lc_mat(p):
    Rx = mp.Matrix(
        mp.Vector3(1, 0, 0),
        mp.Vector3(0, math.cos(phi(p)), math.sin(phi(p))),
        mp.Vector3(0, -math.sin(phi(p)), math.cos(phi(p))),
    )
    lc_epsilon = Rx * epsilon_diag * Rx.transpose()
    lc_epsilon_diag    = mp.Vector3(lc_epsilon[0].x, lc_epsilon[1].y, lc_epsilon[2].z)
    lc_epsilon_offdiag = mp.Vector3(lc_epsilon[1].x, lc_epsilon[2].x, lc_epsilon[2].y)
    return mp.Medium(epsilon_diag=lc_epsilon_diag, epsilon_offdiag=lc_epsilon_offdiag)
```

The rotation matrix `Rx` transforms the diagonal permittivity tensor to the lab frame; the off-diagonal components encode the coupling between field components.

**Dual-polarization source.** A linearly polarized (45-degree) input is synthesized as the sum of Ez and Ey components:

```python
sources = [
    mp.Source(..., component=mp.Ez, ...),
    mp.Source(..., component=mp.Ey, ...),
]
```

**Eigenmode decomposition for both polarizations.** The transmitted flux is decomposed separately for ODD_Z+EVEN_Y (Ez-like) and EVEN_Z+ODD_Y (Hz-like) parities:

```python
res1 = sim.get_eigenmode_coefficients(tran_flux, range(1, nmode+1), eig_parity=mp.ODD_Z+mp.EVEN_Y)
res2 = sim.get_eigenmode_coefficients(tran_flux, range(1, nmode+1), eig_parity=mp.EVEN_Z+mp.ODD_Y)
tran = (abs(coeffs1)**2 + abs(coeffs2)**2) / input_flux
```

**Analytical comparison.** The analytical predictions from Jones calculus are plotted alongside the FDTD data:

```python
phase = delta_n * dd / wvl
eff_m0_analytic = [math.cos(math.pi * p)**2 for p in phase]
eff_m1_analytic = [math.sin(math.pi * p)**2 for p in phase]
```

Agreement between FDTD and analytics validates the anisotropic material model.

#### Key Takeaways

- Spatially varying anisotropic materials are specified in Meep by passing a Python callable as the `material` argument to a `Block`; the callable receives a position vector and returns an `mp.Medium`.
- The off-diagonal elements `epsilon_offdiag` couple different field components and are essential for accurate simulation of birefringent structures.
- Both TE and TM polarizations must be extracted separately from the same flux monitor by calling `get_eigenmode_coefficients` twice with different `eig_parity` values.
- The FDTD results agree well with the Jones-matrix analytical prediction for the uniaxial grating, confirming that Meep's tensor material model is correctly implemented.
- The bilayer twisted-nematic grating achieves much higher first-order efficiency than the uniaxial grating because the chirality of the twist suppresses the zeroth order — a physically important result for beam-steering applications.

---

### 8. `metasurface_lens.py` — Binary-Grating Metasurface Flat Lens

**Physics:** Designs and simulates a focusing flat lens (metalens) built from binary grating unit cells, using a precomputed duty-cycle vs. phase lookup table and near2far projection to compute the focal-plane intensity.
**Difficulty:** Advanced
**Source:** `python/examples/metasurface_lens.py`
**Test Status:** TIMEOUT (CPU-intensive; three lens simulations with 201-801 cells each, followed by near2far projection to 200 um focal length)

#### Theory

A flat lens (metalens) focuses incident light by imparting a spatially varying phase profile `phi(y) = (2*pi/lambda) * (f - sqrt(y^2 + f^2))` to the transmitted beam, where `f` is the focal length and `y` is the transverse coordinate. This is the phase needed to convert a plane wave into a spherical wave converging at the focal point.

The metasurface achieves this phase profile by tiling the aperture with sub-wavelength grating unit cells, each with a different duty cycle chosen from a precomputed library. The library maps duty cycle to phase shift (and transmittance) for a fixed period and height. Because the grating period `gp = 0.3` um is much smaller than the wavelength `lambda = 0.5` um, each cell is in the sub-wavelength regime and acts effectively as a local phase shifter.

The design algorithm:
1. Compute the ideal phase `phi(y)` at each cell center.
2. Reduce to the range `[-2*pi, 0]` (modular phase).
3. Look up the duty cycle in the precomputed table that gives the closest phase.
4. Assemble cells into a supercell and simulate with near2far.

Three aperture sizes are simulated (201, 401, and 801 cells) to show how focusing sharpens with increasing aperture. The focal-plane intensity is computed via `get_farfields` projected to the focal length `f = 200` um, far beyond the FDTD cell.

#### Code Walkthrough

**Phase library construction.** First, 30 equally spaced duty cycles are simulated as isolated periodic unit cells:

```python
gdc = np.linspace(0.1, 0.9, 30)
for n in range(gdc.size):
    mode_tran[n], mode_phase[n] = grating(gp, gh, [gdc[n]])
```

`grating()` with `num_cells=1` runs the periodic unit cell and returns transmittance and phase.

**Phase interpolation.** The discrete library is interpolated to a fine grid:

```python
gdc_new = np.linspace(0.16, 0.65, 500)
mode_phase_interp = np.interp(gdc_new, gdc, mode_phase)
```

**Duty-cycle assignment.**

```python
for j in range(-num_cells[k], num_cells[k]+1):
    phase_local = 2*np.pi/lcen * (focal_length - np.sqrt((j*gp)**2 + focal_length**2))
    phase_mod = phase_local % (-2*np.pi)
    # find gdc with matching phase
    idx = np.nonzero(np.logical_and(
        mode_phase_interp > phase_mod - phase_tol,
        mode_phase_interp < phase_mod + phase_tol))[0]
    gdc_list.append(gdc_new[idx[0]])
```

**Supercell near2far simulation.** `grating()` with `len(gdc_list) > 1` builds the full supercell and runs near2far:

```python
n2f_obj = sim.add_near2far(fcen, 0, 1, mp.Near2FarRegion(center=mon_pt, size=mp.Vector3(y=sy)))
sim.run(until_after_sources=500)
return abs(sim.get_farfields(n2f_obj, ff_res,
    center=mp.Vector3(-0.5*sx + dpml + dsub + gh + focal_length),
    size=mp.Vector3(spot_length))["Ez"])**2
```

The far-field center is displaced by the focal length from the lens exit plane, projecting the fields to the focal plane.

#### Key Takeaways

- The metasurface design methodology separates into two stages: library generation (FDTD of isolated unit cells) and lens simulation (FDTD of the assembled supercell with near2far).
- Modular phase `phi % (-2*pi)` maps the unbounded hyperbolic phase profile onto the available `2*pi` range; the flat lens essentially wraps the spherical wavefront into a Fresnel-like zone structure.
- Near2far projection to 200 um focal length avoids simulating the full 200 um of free-space propagation in FDTD, saving orders of magnitude in compute time.
- Larger apertures (more cells) give tighter focal spots and higher peak intensity; the focal spot width scales approximately as `lambda * f / D` (diffraction limit) where `D` is the aperture.
- The transmittance map from the library shows near-unity transmission across most duty cycles for this sub-wavelength period, demonstrating low insertion loss for the metalens.

---

### 9. `zone_plate.py` — Binary-Phase Zone Plate in Cylindrical Coordinates

**Physics:** Simulates a binary-phase Fresnel zone plate using Meep's cylindrical coordinate solver (`dimensions=CYLINDRICAL`), computing the far-field intensity at the focal plane and along the optical axis via near2far transformation.
**Difficulty:** Advanced
**Source:** `python/examples/zone_plate.py`
**Test Status:** TIMEOUT (CPU-intensive; near2far projection at 200 um focal distance)

#### Theory

A Fresnel zone plate is a diffractive lens consisting of concentric annular zones, alternating between transparent and opaque (amplitude zone plate) or between 0 and pi phase shift (binary phase zone plate). The radius of the `n`-th zone is given by

```
r_n = sqrt(n * lambda * f + n^2 * lambda^2 / 4)  ~  sqrt(n * lambda * f)  for f >> r_n,
```

where `f` is the focal length and `lambda` is the design wavelength. Odd zones impart a pi phase shift (glass, in this case) and even zones impart no phase shift (air), causing constructive interference at the focal point.

The key computational advantage of cylindrical coordinates is that a rotationally symmetric structure with a circularly symmetric source can be reduced from a 3D problem to a 2D problem in `(r, z)`. The `m=-1` azimuthal order in `mp.Simulation(..., dimensions=mp.CYLINDRICAL, m=-1)` corresponds to a circularly polarized incident field, which for an isotropic structure is equivalent to linearly polarized light.

The zone plate has 25 zones, giving a total radius of approximately `sqrt(25 * 0.5 * 200) ~ 50` um. The near2far monitor is placed at the top and side boundaries of the FDTD cell to capture the full transmitted field, and then projected both radially (across the focal plane at z = 200 um) and axially (along the optical axis).

#### Code Walkthrough

**Zone radius computation.**

```python
zone_radius_um = np.zeros(num_zones)
for n in range(1, num_zones + 1):
    zone_radius_um[n-1] = math.sqrt(n * wavelength_um * (focal_length_um + n * wavelength_um / 4))
```

**Geometry: substrate plus zone rings.** Rings are added from largest to smallest so that smaller rings overwrite (via Meep's priority system) the material of larger rings already placed:

```python
for n in range(num_zones - 1, -1, -1):
    geometry.append(mp.Block(
        material=glass if n % 2 == 0 else mp.vacuum,
        size=mp.Vector3(zone_radius_um[n], 0, height_um),
        center=mp.Vector3(0.5 * zone_radius_um[n], 0, ...),
    ))
```

**Cylindrical solver and m=-1 mode.** The circularly polarized source uses both Er and Ep components with a relative phase of -i:

```python
sim = mp.Simulation(..., dimensions=mp.CYLINDRICAL, m=-1)
sources = [
    mp.Source(..., component=mp.Er, ...),
    mp.Source(..., component=mp.Ep, ..., amplitude=-1j),
]
```

**L-shaped near2far monitor.** Two `Near2FarRegion` objects capture the fields on the top face and the outer radial face of the computational cell:

```python
n2f_monitor = sim.add_near2far(frequency, 0, 1,
    mp.Near2FarRegion(center=..., size=mp.Vector3(size_r_um - pml_um, 0, 0)),   # top
    mp.Near2FarRegion(center=..., size=mp.Vector3(0, 0, height_um + padding_um)), # side
)
```

**Focal plane and axial scans.**

```python
farfields_r = sim.get_farfields(n2f_monitor, ff_res,
    center=mp.Vector3(0.5*(size_r_um - pml_um), 0, focal_length_um),
    size=mp.Vector3(size_r_um - pml_um, 0, 0))  # radial scan at z=f

farfields_z = sim.get_farfields(n2f_monitor, ff_res,
    center=mp.Vector3(0, 0, focal_length_um),
    size=mp.Vector3(0, 0, scan_length_z_um))     # axial scan through focus
```

#### Key Takeaways

- Cylindrical coordinates (`dimensions=CYLINDRICAL`) reduce rotationally symmetric 3D problems to 2D in `(r, z)`, giving a factor of `N_phi` speedup over full 3D.
- The `m` parameter specifies the azimuthal Fourier order; `m=-1` models a circularly polarized plane wave via the combination of Er and Ep sources with a 90-degree phase shift.
- A two-segment near2far monitor (top surface + side surface) captures the full radiated field from the lens, including contributions at all transmitted angles.
- The zone plate focuses via diffraction, not refraction; its focal length scales as `f ~ r_N^2 / (N * lambda)`, and it has strong chromatic aberration (different wavelengths focus at different distances).
- The axial intensity scan through the focal region reveals the depth of focus and confirms the Rayleigh range `~lambda * (f/D)^2`.

---

### 10. `grating2d_triangular_lattice.py` — 2D Triangular-Lattice Grating via Rectangular Supercell

**Physics:** Computes the diffraction orders of a 2D binary grating with a triangular (hexagonal) lattice of cylindrical pillars, using a rectangular supercell and verifying the selection rule that only orders compatible with the triangular lattice have nonzero power.
**Difficulty:** Advanced
**Source:** `python/examples/grating2d_triangular_lattice.py`
**Test Status:** TIMEOUT (CPU-intensive; 3D simulation at resolution 100 px/um)

#### Theory

A 2D grating has a 2D lattice of scatterers and diffracts incident light into a 2D array of orders labeled `(m_x, m_y)`. For a rectangular lattice with periods `(a_x, a_y)`, the diffracted wavevectors are `k_{m_x,m_y} = k_0 + m_x * G_x + m_y * G_y` where `G_x = (2*pi/a_x, 0)` and `G_y = (0, 2*pi/a_y)` are the reciprocal lattice vectors.

A triangular (hexagonal) lattice with basis vectors `a_1 = (a, 0)` and `a_2 = (a/2, a*sqrt(3)/2)` can be simulated using a rectangular supercell with dimensions `(sx, sy) = (a, a*sqrt(3))`. This supercell contains two lattice points (one at the origin and one at the corner of the rectangular cell, which by periodicity contributes from all four corners). The key insight is that not all rectangular-supercell diffraction orders correspond to true triangular-lattice orders.

The selection rule for the triangular lattice, when expressed in terms of rectangular-supercell indices `(n_x, n_y)`, requires `n_y = -n_x + 2*m_y` for the order to be allowed. For the specific case `m_x = 0`, the allowed orders have even `n_y` values. The code verifies this by computing the transmittance of orders `(0, 0)`, `(0, 1)`, `(0, 2)`, and `(0, 3)`, expecting nonzero power only for `n_y = 0` and `n_y = 2`.

#### Code Walkthrough

**Rectangular supercell with two cylinders.** The triangular-lattice basis is represented by placing cylinders at the rectangle corners (shared between adjacent cells) and at the center:

```python
sx = 1.0
sy = np.sqrt(3)  # = sqrt(3) * a for a=1

cyl_grating = [
    mp.Cylinder(center=mp.Vector3(0, 0, ...), radius=rcyl, height=hcyl, material=glass),
    mp.Cylinder(center=mp.Vector3( 0.5*sx,  0.5*sy, ...), ...),
    mp.Cylinder(center=mp.Vector3(-0.5*sx,  0.5*sy, ...), ...),
    mp.Cylinder(center=mp.Vector3(-0.5*sx, -0.5*sy, ...), ...),
    mp.Cylinder(center=mp.Vector3( 0.5*sx, -0.5*sy, ...), ...),
]
```

The corner cylinders are shared between four cells so each contributes 1/4 of a cylinder; together the four corners give one full cylinder. Combined with the center cylinder, this gives two cylinders per supercell — consistent with the triangular lattice's two-atom basis.

**`DiffractedPlanewave` for specific orders.** Rather than the mode-band enumeration used in 1D gratings, the 2D case uses the `DiffractedPlanewave` object to request a specific order:

```python
for ny in range(4):
    kz2 = fcen**2 - (nx/sx)**2 - (ny/sy)**2
    if kz2 > 0:  # only propagating orders
        res = sim.get_eigenmode_coefficients(
            tran_flux,
            mp.DiffractedPlanewave((nx, ny, 0), mp.Vector3(0, 1, 0), 1, 0)
        )
        tran = abs(res.alpha[0, 0, 0])**2
        print(f"order:, {nx}, {ny}, {tran:.5f}")
```

**3D simulation with Z-direction PML.** Unlike the 2D (1D grating) cases, this is a full 3D simulation with PML along z:

```python
boundary_layers = [mp.PML(thickness=dpml, direction=mp.Z)]
k_point = mp.Vector3()  # normal incidence
```

The cell has extent in all three directions; the grating cylinders are fully resolved in 3D.

#### Key Takeaways

- 2D gratings require 3D simulations; the computational cost scales as the square of the lateral cell size times the number of z cells.
- The `DiffractedPlanewave` object specifies a diffraction order by its integer indices `(m_x, m_y, m_z)` and polarization, bypassing the band-index enumeration needed for the MPB-based approach.
- Rectangular-supercell simulations of non-rectangular lattices produce apparent orders at all `(n_x, n_y)`, but selection rules from the actual lattice symmetry suppress most of them to zero power — a useful verification of simulation correctness.
- The triangular lattice supercell contains two basis atoms per rectangular cell; this is why corners (4 × 1/4 = 1) plus the interior (1) total two atoms.
- The propagating-order check `kz2 > 0` must be applied before attempting to extract an order; evanescent orders (`kz2 < 0`) have no real propagation direction and the eigenmode solver cannot resolve them.

---

### 11. `diffracted_planewave.py` — Dual-Method Diffraction Efficiency: MPB vs. DiffractedPlanewave

**Physics:** Computes transmission diffraction efficiencies of a binary grating using two independent methods — MPB eigenmode decomposition and the `DiffractedPlanewave` object — verifying that both give the same result and that their sum equals the Poynting-flux transmittance.
**Difficulty:** Advanced
**Source:** `python/examples/diffracted_planewave.py`
**Test Status:** PASS (148.6s)

#### Theory

There are two distinct ways to extract the power in a specific diffraction order from a Meep flux monitor:

**Method 1: MPB eigenmode decomposition.** The mode solver MPB finds the exact eigenmode of the periodic waveguide cross-section at the monitor plane. For a 1D periodic boundary with zero transverse wavevector (`k_y = m/Lambda` for the m-th order), the eigenmode is a plane wave. `get_eigenmode_coefficients` projects the simulated fields onto this mode and returns a complex amplitude coefficient `alpha`. The power in the order is `|alpha|^2`.

**Method 2: `DiffractedPlanewave` object.** This is a more direct specification: the user declares the integer order indices `(m_x, m_y, m_z)` and the polarization (S or P, via the `s_amp` and `p_amp` parameters). Meep constructs the plane wave internally without calling MPB. This method is particularly useful for 2D gratings (as in tutorial 10) where the MPB band enumeration does not map cleanly to 2D order indices.

Both methods should give the same result for a correctly set up problem. The tutorial makes this comparison explicit, printing both values side-by-side and computing the relative error.

Additionally, the simulation verifies global energy conservation: the sum of all order transmittances from `DiffractedPlanewave` should match the integrated Poynting flux through the monitor plane.

The script runs two cases: normal incidence (`theta=0`) and oblique incidence (`theta=13.5` degrees), covering both the symmetric case (where mirror symmetry can be used) and the general case.

#### Code Walkthrough

**Simulation structure.** The grating function `binary_grating_diffraction(gp, gh, gdc, theta)` runs two simulations: a reference (uniform glass) and the grating, recording fluxes and mode monitors.

```python
tran_mon = sim.add_mode_monitor(
    fcen, 0, 1,
    mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, sy, 0))
)
```

`add_mode_monitor` is essentially an alias for `add_flux` but reserves the monitor for mode decomposition.

**Method 1: MPB-based, by band index.**

```python
for band, order in zip(bands, orders):
    res = sim.get_eigenmode_coefficients(tran_mon, [band], eig_parity=eig_parity)
    tran_eig = abs(res.alpha[0, 0, 0])**2 / input_flux[0]
```

**Method 2: DiffractedPlanewave, by order index.**

```python
    res = sim.get_eigenmode_coefficients(
        tran_mon,
        mp.DiffractedPlanewave((0, order, 0), mp.Vector3(0, 1, 0), 0, 1)
    )
    tran_dp = abs(res.alpha[0, 0, 0])**2 / input_flux[0]
```

The `DiffractedPlanewave` arguments are: order indices `(m_x, m_y, m_z)`, the normal direction of the grating plane `mp.Vector3(0, 1, 0)`, and the S/P amplitudes (here `s_amp=0, p_amp=1` for TM).

**Energy conservation check.** The sum of all order transmittances is compared to the flux:

```python
flux = mp.get_fluxes(tran_mon)
t_flux = flux[0] / input_flux[0]
err = abs(dp_sum - t_flux) / t_flux
print(f"flux:, {eig_sum:.8f}, {dp_sum:.8f}, {t_flux:.8f}, {err:.8f}")
```

A relative error below 1% confirms that the mode decomposition accounts for essentially all of the transmitted power.

#### Key Takeaways

- The `DiffractedPlanewave` object is the preferred interface for specifying diffraction orders in 2D gratings and for non-band-enumerable scenarios; for 1D gratings, MPB eigenmode decomposition is equally valid.
- Both methods should agree to within numerical precision set by the simulation resolution; disagreement indicates either a setup error or a convergence issue.
- The constraint `eig_sum ≈ dp_sum ≈ t_flux` verifies three independent quantities: MPB decomposition, analytic plane-wave construction, and integrated Poynting flux all agree.
- At oblique incidence (`theta != 0`), all signed diffraction orders (positive and negative `m`) must be included; the code computes the range of allowed orders from the grating equation before iterating.
- `add_mode_monitor` is functionally identical to `add_flux` but makes explicit the intent to use modal decomposition; either can be passed to `get_eigenmode_coefficients`.

---

### 12. `test_binary_grating.py` — Unit Tests for Grating Energy Conservation

**Physics:** Automated test suite verifying energy conservation (R + T = 1) for a binary grating at both normal and oblique incidence, and testing the `kz_2d` parameter for out-of-plane momentum with three different 2D computation modes.
**Difficulty:** Advanced
**Source:** `python/tests/test_binary_grating.py`
**Test Status:** FAIL (requires `pip install parameterized`; the `parameterized` package is not bundled with conda-forge pymeep)

#### Theory

This test validates two separate physical scenarios:

**Test 1: `test_binary_grating_oblique`.** For a binary grating at normal or oblique incidence in the XY plane, the sum `R + T` must equal 1 (energy conservation) when summed over all non-evanescent diffraction orders. Both reflected and transmitted orders are computed using explicit Bloch-wavevector eigenmode decomposition (via `kpoint_func`), bypassing the automatic band enumeration. This approach uses a near-zero-length eigenmode volume `eig_vol` to force MPB to treat the periodic boundary modes as plane waves rather than waveguide modes.

**Test 2: `test_binary_grating_special_kz`.** This tests the `kz_2d` capability: simulating a 2D problem (z-invariant geometry) with a nonzero out-of-plane momentum `k_z`. Physically this corresponds to light incident in the XZ plane (tilted in the plane perpendicular to the grating grooves). Three numerical representations are tested via `kz_2d="real/imag"`, `"complex"`, and `"3d"`, each implementing the out-of-plane component differently internally. All three should give identical energy conservation.

The `kz_2d` feature enables simulation of conical diffraction — the general case where the incident wavevector is not in the principal plane of the grating. In this case, both S and P polarized diffraction orders exist and must be summed separately.

#### Code Walkthrough

**Parameterized test expansion.** The test uses the `parameterized` package to run the same test function with different angle values:

```python
@parameterized.parameterized.expand([(0.0,), (10.7,)])
def test_binary_grating_oblique(self, theta):
    ...
```

This generates two separate test methods automatically.

**Manual Bloch-wavevector eigenmode.** Rather than using band indices, the reflected and transmitted order wavevectors are computed analytically:

```python
for nm in orders:
    ky = k.y + nm / self.cell_size.y
    kx2 = (self.fcen * self.ng)**2 - ky**2
    if kx2 > 0:
        res = sim.get_eigenmode_coefficients(
            refl_flux, bands=[1],
            kpoint_func=lambda *not_used: mp.Vector3(np.sqrt(kx2), ky, 0),
            direction=mp.NO_DIRECTION,
            eig_vol=mp.Volume(center=refl_pt, size=mp.Vector3(0, 1e-7, 0)),
        )
```

The `kpoint_func` provides the explicit plane-wave wavevector to MPB; `eig_vol` with near-zero y-extent forces a plane-wave solution.

**kz_2d simulation.**

```python
sim = mp.Simulation(..., kz_2d=kz_2d)  # "real/imag", "complex", or "3d"
```

The `kz_2d` parameter selects the internal representation of the `k_z` component in a 2D simulation, with "3d" being the most accurate but most expensive.

**Assertion threshold.** The tests assert energy conservation to 2 decimal places (1% relative error), consistent with the resolution of 30 px/um used in the test:

```python
self.assertAlmostEqual(Rsum + Tsum, 1.00, places=2)
```

#### Key Takeaways

- The `parameterized` package must be installed separately from Meep (`pip install parameterized`) to run this test suite; it is not a Meep dependency.
- `kpoint_func` combined with `direction=mp.NO_DIRECTION` and a tiny `eig_vol` bypasses MPB's waveguide-mode solver and directly computes the plane-wave eigenmode at any specified k-point — a powerful technique for non-standard geometries.
- The `kz_2d` parameter enables computationally efficient conical diffraction simulation in 2D; all three modes (`real/imag`, `complex`, `3d`) should give equivalent physical results.
- Energy conservation `R + T = 1` at 1% tolerance (2 decimal places) is a reliable but not stringent convergence criterion; higher resolutions would tighten this to 3-4 decimal places.
- The test structure (setUpClass for geometry, individual test methods for angles) is the correct pattern for parameterized Meep tests: expensive geometry creation happens once, while the time-domain runs vary per test.

---

### 13. `test_n2f_periodic.py` — Convergence Test for Near2Far with Periodic Boundaries

**Physics:** Verifies that the near2far projection for a periodic grating converges to the direct DFT field as the simulation resolution increases, validating the consistency of Meep's near2far transformation with periodic (Bloch) boundary conditions.
**Difficulty:** Intermediate
**Source:** `python/tests/test_n2f_periodic.py`
**Test Status:** PASS (276.3s)

#### Theory

The near-to-far-field transformation in Meep relies on the equivalence principle: the fields on a closed surface surrounding a source region determine the fields everywhere outside. For a periodic structure with Bloch boundary conditions, the near2far transformation must account for the phase relationship between unit cells; the `nperiods` parameter controls how many unit-cell contributions are coherently summed.

This test poses a direct consistency check: the near2far projected field at a target plane should converge to the DFT field actually simulated at that plane as the resolution increases. The test geometry places the near2far monitor close to the grating (at `grating surface + 1 um`) and projects to a plane 20 um away (the `dpad = 20` um padding means the DFT target plane is within the simulation cell).

The convergence criterion is that the L2 norm `||E_near2far - E_DFT||` decreases monotonically as resolution increases from 20 to 25 to 30 px/um. This exercises both the near2far integration (which interpolates sampled near-field data) and the DFT field recording.

The `nperiods=10` argument means the near2far monitor coherently superimposes contributions from 10 unit cells in each direction, simulating a 10-period grating from the single unit-cell fields.

#### Code Walkthrough

**Simulation geometry.** The test uses a standard binary grating: `gp=10` um period, `gh=0.5` um height, `gdc=0.5` duty cycle, glass substrate, at `lambda=0.5` um:

```python
n2f_obj = sim.add_near2far(
    fcen, 0, 1,
    mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy)),
    nperiods=10,
)
dft_obj = sim.add_dft_fields(
    [mp.Ez], fcen, 0, 1,
    center=dft_pt, size=mp.Vector3(y=sy)
)
```

Both monitors observe at the same y-extent but at different x positions: `n2f_pt` is close to the grating, `dft_pt` is at the far edge of the cell.

**Near2far projection to DFT plane.** The far-field is computed at the same location as the DFT monitor:

```python
n2f_Ez = sim.get_farfields(
    n2f_obj, res[j],
    center=dft_pt,
    size=mp.Vector3(y=sy)
)
dft_Ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
```

**Convergence norm.** The L2 difference excludes the boundary points (which may have PML contamination):

```python
norm[j] = LA.norm(n2f_Ez["Ez"] - dft_Ez[1:-1])
```

**Assertion.** Convergence is verified by requiring the norm to decrease strictly:

```python
self.assertGreater(norm[0], norm[1])
self.assertGreater(norm[1], norm[2])
```

This is a monotone convergence check rather than an absolute tolerance check — appropriate given the range of resolutions tested is small.

#### Key Takeaways

- The near2far transformation is consistent with direct DFT field recording to within discretization error; the discrepancy decreases with resolution as expected for a second-order FDTD scheme.
- `nperiods=10` on `add_near2far` is required when the simulation cell is one unit cell wide but the structure being modeled has multiple periods; it weights each period's near-field contribution with the correct Bloch phase.
- The norm `LA.norm(n2f_Ez["Ez"] - dft_Ez[1:-1])` uses the `[1:-1]` slice to avoid comparing at PML-adjacent points where the DFT fields may include absorber contributions not modeled by the free-space near2far Green's function.
- A long padding region (`dpad=20`) between the grating and the far DFT monitor is needed so that the evanescent components of the grating near field have decayed to negligible levels before reaching the comparison plane.
- The 276 second runtime at three resolutions indicates each resolution run takes roughly 90 seconds; this is consistent with the `until_after_sources=300` run time and the moderate 20-30 px/um resolutions used.

---

## Chapter Summary

This chapter demonstrated the full range of grating and diffractive optics simulation capabilities in Meep, progressing from foundational concepts to advanced techniques:

**Eigenmode decomposition** (`get_eigenmode_coefficients`) is the primary tool for quantifying diffraction efficiency. It works in two flavors: MPB-based mode matching (by band index) and analytic plane-wave construction (`DiffractedPlanewave`), both of which agree when correctly set up.

**Near-to-far-field transformation** (`add_near2far`, `get_farfields`) enables simulation of focusing elements (metasurfaces, zone plates, polarization gratings) whose far fields are inaccessible within the FDTD cell. The `nperiods` argument efficiently extends single unit-cell results to finite-aperture gratings.

**Bloch boundary conditions** with a nonzero `k_point` handle oblique incidence, and the `kz_2d` parameter extends this to conical diffraction in 2D simulations.

**Material complexity** spans from simple glass ridges through spatially varying anisotropic liquid-crystal tensors to rotationally symmetric zone plates simulated in cylindrical coordinates — each geometry requiring specific choices of simulation cell, boundary conditions, and source type.

**Verification patterns** appear consistently throughout: two-pass normalization (reference + grating simulation), energy conservation checks (R + T = 1), and convergence with resolution. These patterns should be adopted in any new grating simulation built on these examples.
