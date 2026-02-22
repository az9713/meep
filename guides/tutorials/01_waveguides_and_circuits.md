# Chapter 1: Waveguides and Circuits

This chapter covers guided-wave photonics simulations in Meep, progressing from the simplest straight-waveguide visualization through flux normalization, bend transmission, directional couplers, mode decomposition, optical forces, and photonic-crystal band structures. Each tutorial section builds on the core concepts of FDTD simulation—geometry definition, source injection, DFT monitors, and post-processing—while introducing increasingly sophisticated physics. By the end of the chapter a reader should be able to set up, run, and interpret transmission measurements for realistic integrated-photonic components.

---

### 1. `straight-waveguide.py` — Visualizing a Dielectric Waveguide

**Physics:** A dielectric slab waveguide traps light by total internal reflection, producing guided modes whose field profiles are determined by the refractive index contrast and waveguide geometry.
**Difficulty:** Beginner
**Source:** `python/examples/straight-waveguide.py`
**Test Status:** PASS (9.0s)

#### Theory

A dielectric waveguide confines electromagnetic energy laterally by total internal reflection between the high-index core and the lower-index cladding. In 2D (where the simulation is effectively infinite in $z$), the relevant field component for TM polarization is $E_z$, which obeys the scalar Helmholtz equation in each region:

$$\nabla^2 E_z + k_0^2 \varepsilon(\mathbf{r}) E_z = 0$$

For a slab of width $w$ and dielectric constant $\varepsilon$, the dispersion relation for the fundamental guided mode is obtained by matching boundary conditions at each interface. The mode exists only when the free-space wavevector satisfies $k_0 < \beta/\sqrt{\varepsilon}$ (total internal reflection condition), where $\beta$ is the guided propagation constant.

In Meep's natural units (where $c = 1$), frequency $f$ and wavelength $\lambda$ are expressed relative to some length scale—here all distances are in micrometers. The frequency $f = 0.15$ corresponds to a free-space wavelength of $\lambda = 1/f \approx 6.67\,\mu\mathrm{m}$ in a cell measured in micrometers. With $\varepsilon = 12$ (similar to silicon at mid-infrared wavelengths), the core index is $n = \sqrt{12} \approx 3.46$.

The purpose of this simulation is exploratory: run the FDTD to steady state with a continuous-wave source, then extract and overlay the permittivity profile and the $E_z$ field to visually confirm that the field is guided inside the slab.

#### Code Walkthrough

The geometry is a single infinite-length `Block` of width 1 µm centered on the axis:

```python
geometry = [
    mp.Block(
        mp.Vector3(mp.inf, 1, mp.inf),  # infinite in x and z
        center=mp.Vector3(),             # centered at origin
        material=mp.Medium(epsilon=12),
    )
]
```

`mp.inf` tells Meep to extend the object to the cell boundary in that direction, which is the standard way to define a waveguide running along $x$ in a 2D cell.

The source is a continuous-wave (monochromatic) point-like dipole placed at the left side of the waveguide core:

```python
sources = [
    mp.Source(
        mp.ContinuousSource(frequency=0.15), component=mp.Ez,
        center=mp.Vector3(-7, 0)
    )
]
```

`mp.ContinuousSource` emits at a single frequency indefinitely, which is appropriate here because we only want the steady-state field pattern rather than a broadband spectrum. The simulation runs for 200 time units—long enough for the fields to reach a quasi-steady state—with 1 µm thick PML absorbers on all boundaries to prevent reflections from the cell edges.

After running, `sim.get_array` retrieves 2D numpy arrays of the permittivity and $E_z$ field over the entire cell:

```python
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data  = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
```

These are overlaid in matplotlib: the permittivity is shown in grayscale (so the waveguide core is visible), and $E_z$ is shown with a red-blue colormap at 90% opacity so both layers are simultaneously readable.

#### Key Takeaways

- `mp.Medium(epsilon=12)` defines a non-dispersive dielectric; for frequency-dependent materials see Section 18.
- `mp.ContinuousSource` is appropriate when only the spatial field pattern matters; use `mp.GaussianSource` for broadband spectral calculations.
- `mp.inf` in a `Vector3` extends a geometry object to the simulation boundary in that direction.
- `sim.get_array(component=mp.Dielectric)` retrieves the spatially-varying permittivity on the Yee grid, useful for overlaying geometry on field plots.
- PML layers (here `dpml = 1.0` µm) must be thick enough to suppress reflections; a common rule of thumb is at least half a wavelength in the absorbing medium.

---

### 2. `bend-flux.py` — Transmission and Reflection at a 90-Degree Waveguide Bend

**Physics:** A 90-degree bend in an integrated waveguide scatters light: some power is transmitted into the output arm, some is reflected back toward the source, and the remainder is radiated as loss. FDTD flux measurements with a normalization run separate the bend's contribution from the source spectrum.
**Difficulty:** Intermediate
**Source:** `python/examples/bend-flux.py`
**Test Status:** PASS (29.2s)

#### Theory

When a guided mode encounters an abrupt or gradual change in propagation direction, it is partially scattered into radiation modes (loss) and partially reflected. The power budget is:

$$T(\omega) + R(\omega) + L(\omega) = 1$$

where $T$, $R$, and $L$ are the wavelength-dependent transmittance, reflectance, and loss fraction. To extract $T$ and $R$ independently, Meep uses a two-run normalization scheme.

**Run 1 (normalization):** Simulate a straight waveguide with identical source and monitor positions. Record the forward flux $P_\mathrm{fwd}(\omega)$ through the transmission monitor and the flux data at the reflection monitor location. Because there is no bend, the reflected flux is essentially zero (any tiny value is subtracted later).

**Run 2 (signal):** Simulate the bent waveguide, but load the negated normalization fields into the reflection monitor. This cancels the incident field contribution so the monitor sees only the reflected wave:

$$R(\omega) = \frac{-\Phi_\mathrm{refl}^\mathrm{bend}(\omega)}{P_\mathrm{fwd}^\mathrm{straight}(\omega)}, \quad T(\omega) = \frac{\Phi_\mathrm{tran}^\mathrm{bend}(\omega)}{P_\mathrm{fwd}^\mathrm{straight}(\omega)}$$

The time-domain DFT monitors evaluate the Fourier transform of the Poynting vector $\mathbf{S} = \mathbf{E} \times \mathbf{H}$ integrated over a cross-section, providing frequency-resolved power spectra from a single broadband simulation.

#### Code Walkthrough

Both runs share the same `GaussianSource` centered at $f = 0.15$ with $\Delta f = 0.1$, which spans roughly $\lambda = 5$–10 µm:

```python
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(-0.5 * sx + dpml, wvg_ycen, 0),
        size=mp.Vector3(0, w, 0),
    )
]
```

Setting `size=mp.Vector3(0, w, 0)` makes the source span the full waveguide width, which couples more efficiently to the fundamental mode than a point source.

The reflection monitor is placed just inside the PML on the input side, and its data is saved from the straight run:

```python
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
straight_refl_data = sim.get_flux_data(refl)
```

In the bend run, `sim.load_minus_flux_data(refl, straight_refl_data)` pre-loads the negative of these stored fields. When the FDTD propagates forward, the incident contribution cancels, leaving only the reflected wave's contribution to the DFT accumulation.

The simulation terminates adaptively:

```python
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))
```

This waits until the source has finished and then continues until $|E_z|$ at the monitoring point has decayed by three orders of magnitude, ensuring that transient effects have vanished before reading the DFT data.

Post-processing normalizes both spectra against the straight-waveguide transmission and the negative convention for reflected flux (`-bend_refl_flux[i]`, since power flowing backward gives a negative Poynting sign):

```python
Rs = np.append(Rs, -bend_refl_flux[i] / straight_tran_flux[i])
Ts = np.append(Ts, bend_tran_flux[i] / straight_tran_flux[i])
```

The quantity `1 - Rs - Ts` is the radiation loss fraction.

#### Key Takeaways

- The two-run normalization scheme is the standard Meep approach for removing source-spectrum effects from transmission/reflection spectra.
- `sim.load_minus_flux_data` implements the incident-field subtraction for the reflection monitor.
- `mp.stop_when_fields_decayed` provides adaptive termination—far more efficient than a fixed simulation length.
- The sign convention: power flowing in the $+x$ direction gives positive flux; reflected power flowing in $-x$ appears as negative flux, hence the minus sign in the reflectance formula.
- At the design frequency $f = 0.15$, a well-designed 90-degree bend can achieve $T > 0.97$ over a broad bandwidth for silicon-like materials.

---

### 3. `bent-waveguide.py` — Visualizing Fields in a Bent Waveguide

**Physics:** This simulation visualizes the $E_z$ field propagating through a 90-degree L-shaped waveguide bend, demonstrating mode radiation and field redistribution at the corner, and introduces time-series HDF5 field output.
**Difficulty:** Beginner
**Source:** `python/examples/bent-waveguide.py`
**Test Status:** PASS (18.4s)

#### Theory

At the corner of an L-shaped waveguide, the guided mode must negotiate a sharp discontinuity in propagation direction. The local electric field must satisfy boundary conditions simultaneously at both arms of the bend. For a sharp 90-degree corner, a significant fraction of the power is radiated into the cladding because the field cannot adiabatically follow the waveguide axis. This radiation manifests as evanescent waves that leak away from the corner.

The simulation uses a source wavelength $\lambda = 2\sqrt{11}\,\mu\mathrm{m}$ (which evaluates to the wavelength such that $f = c/\lambda$ satisfies a particular mode condition). The `width=20` parameter in the `ContinuousSource` specifies a temporal Gaussian turn-on time in units of $1/f$, which smoothly excites the steady state and avoids the abrupt startup transients that would otherwise fill the cell with broadband radiation.

Outputting the time-evolving field to HDF5 allows post-processing with visualization tools such as `h5topng` or custom Python scripts, enabling animations that show wave propagation and corner scattering.

#### Code Walkthrough

Two `Block` objects form the L-shaped waveguide—a horizontal arm and a vertical arm meeting at a corner:

```python
geometry = [
    mp.Block(mp.Vector3(12, 1, mp.inf), center=mp.Vector3(-2.5, -3.5),
             material=mp.Medium(epsilon=12)),
    mp.Block(mp.Vector3(1, 12, mp.inf), center=mp.Vector3(3.5, 2),
             material=mp.Medium(epsilon=12)),
]
```

The source has a slow turn-on (`width=20`) to ramp up smoothly:

```python
sources = [
    mp.Source(
        mp.ContinuousSource(wavelength=2 * (11**0.5), width=20),
        component=mp.Ez,
        center=mp.Vector3(-7, -3.5),
        size=mp.Vector3(0, 1),
    )
]
```

The run uses callbacks to produce output at multiple times:

```python
sim.run(
    mp.at_beginning(mp.output_epsilon),
    mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
    until=200,
)
```

`mp.at_beginning(mp.output_epsilon)` outputs the dielectric function once at the start. `mp.to_appended("ez", ...)` writes all time-step field snapshots into a single HDF5 file `ez.h5`, appended sequentially. `mp.at_every(0.6, mp.output_efield_z)` samples $E_z$ every 0.6 time units, producing roughly 333 frames over the 200-unit run.

#### Key Takeaways

- `mp.to_appended` creates a single HDF5 file with time-axis data rather than many separate files, which is much more efficient for long runs.
- `mp.at_every(dt, func)` is a step function that fires `func` at regular time intervals during the simulation.
- `mp.at_beginning(func)` fires only on the first step—ideal for recording static geometry data.
- The `width` parameter in `ContinuousSource` controls the temporal turn-on time; larger values give a smoother (slower) startup and produce cleaner steady-state fields at the cost of requiring a longer simulation time.
- Setting `size` on a source to match the waveguide cross-section width preferentially excites the fundamental guided mode.

---

### 4. `coupler.py` — Directional Coupler S-Parameters from a GDSII File

**Physics:** A directional coupler transfers power between two parallel waveguides via evanescent-field overlap; the coupling ratio depends sensitively on the gap, interaction length, and wavelength. This example uses `EigenModeSource` and mode-coefficient extraction to compute S-parameters directly from a GDSII layout file.
**Difficulty:** Advanced
**Source:** `python/examples/coupler.py`
**Test Status:** TIMEOUT (CPU-intensive)

#### Theory

In a symmetric directional coupler, the two waveguides support even (symmetric) and odd (anti-symmetric) supermodes with propagation constants $\beta_e$ and $\beta_o$. Power oscillates between the two waveguides with a spatial period (coupling length):

$$L_c = \frac{\pi}{\beta_e - \beta_o}$$

If light is launched into port 1 (input of waveguide A) and the coupler length is $L$, the power fractions at the output ports are:

$$T_{21} = \cos^2\!\left(\frac{\pi L}{2 L_c}\right), \quad T_{31} = \sin^2\!\left(\frac{\pi L}{2 L_c}\right)$$

For a 3 dB (50/50) splitter, $L = L_c/2$. Real couplers deviate from this ideal due to the finite evanescent-decay length, material dispersion, and fabrication imperfections. FDTD simulation captures all these effects simultaneously.

The example imports waveguide geometry from a GDSII file (the standard layout format in photonic integrated circuit design), allowing direct comparison between design intent and simulated performance. The coupler is simulated in 2D or 3D (with a silicon-on-insulator stack: oxide substrate, silicon layer, and air cladding).

S-parameters are computed as complex mode coefficients: $S_{21} = \alpha_2^{(+)} / \alpha_1^{(+)}$, where the superscript denotes the forward-traveling mode at each port.

#### Code Walkthrough

Geometry is read from the GDSII file using `mp.get_GDSII_prisms`, which converts layer polygons into Meep `Prism` objects:

```python
upper_branch = mp.get_GDSII_prisms(silicon, gdsII_file, UPPER_BRANCH_LAYER, si_zmin, si_zmax)
lower_branch = mp.get_GDSII_prisms(silicon, gdsII_file, LOWER_BRANCH_LAYER, si_zmin, si_zmax)
cell = mp.GDSII_vol(gdsII_file, CELL_LAYER, cell_zmin, cell_zmax)
```

The source is an `EigenModeSource` that automatically excites the fundamental guided mode of waveguide A:

```python
sources = [
    mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        volume=src_vol,
        eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z,
    )
]
```

Mode monitors are placed at all four ports; after the run, `get_eigenmode_coefficients` decomposes the DFT fields into forward and backward mode amplitudes:

```python
p1_coeff = sim.get_eigenmode_coefficients(
    mode1, [1], eig_parity=...
).alpha[0, 0, 0]  # forward mode at port 1
p2_coeff = sim.get_eigenmode_coefficients(
    mode2, [1], eig_parity=...
).alpha[0, 0, 1]  # backward mode at port 2 (reflected into input arm)
```

Transmittance to each output port is $|S_{n1}|^2 = |\alpha_n|^2 / |\alpha_1|^2$:

```python
p3_trans = abs(p3_coeff)**2 / abs(p1_coeff)**2
```

#### Key Takeaways

- `mp.get_GDSII_prisms` and `mp.GDSII_vol` enable direct import of photonic chip layouts without manual geometry re-entry.
- `EigenModeSource` calls the MPB eigenmode solver internally to construct the mode profile, injecting a clean single-mode input.
- `get_eigenmode_coefficients(...).alpha[band, freq, direction]` returns complex mode amplitudes; `direction=0` is forward, `direction=1` is backward.
- `eig_parity=mp.EVEN_Y + mp.ODD_Z` constrains the eigenmode search to the TE-like mode of a horizontal slab waveguide, reducing computation time.
- The branch-separation parameter `d` can be varied from the command line to study coupling vs. gap, enabling design-space exploration.

---

### 5. `wvg-src.py` — Transparent Eigenmode Source in an Asymmetric Waveguide

**Physics:** An `EigenModeSource` acting as a transparent source injects a pure guided mode into an asymmetric waveguide, and `flux_in_box` confirms that the power flows correctly—with negligible flux going left (toward the source) and substantial flux going right (in the guided mode direction).
**Difficulty:** Intermediate
**Source:** `python/examples/wvg-src.py`
**Test Status:** FAIL (missing h5topng)

#### Theory

A standard `Source` in Meep injects fields at a point or line and can scatter in all directions. An `EigenModeSource` is "transparent" in the sense that it adds a current distribution that exactly matches the desired modal profile, so that the guided mode is excited with high efficiency while radiation and back-reflection are strongly suppressed.

For an asymmetric waveguide (one where the cladding refractive indices differ above and below the core), the mode profile is not symmetric. The asymmetry here is introduced by a thin air slit embedded in the slab:

```python
geometry = [
    mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),   material=mp.Medium(epsilon=12)),
    mp.Block(center=mp.Vector3(y=0.3), size=mp.Vector3(mp.inf, 0.1, mp.inf), material=mp.Medium()),
]
```

The second block (air, $\varepsilon = 1$) cuts a 0.1 µm slit into the slab 0.3 µm above center, breaking the up-down symmetry. The eigenmode solver finds the mode numerically by solving the generalized eigenvalue problem on a transverse cross-section.

`force_complex_fields = True` enables complex-valued fields, which allows direct computation of the time-averaged Poynting vector from a single snapshot rather than requiring time averaging.

The left-going flux should be nearly zero (the transparent source does not radiate backward), and the right-going flux should carry essentially all the source power, confirming that the mode coupling is correct.

#### Code Walkthrough

The source specifies `component=mp.Dielectric`—a special sentinel value telling Meep to use the eigenmode profile for all field components automatically:

```python
sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(0.15),
        size=mp.Vector3(y=6),
        center=mp.Vector3(x=-5),
        component=mp.Dielectric,
        eig_parity=mp.ODD_Z,
    )
]
```

After running to steady state ($t = 200$), `flux_in_box` integrates the time-averaged Poynting vector over a volume:

```python
flux1 = sim.flux_in_box(
    mp.X, mp.Volume(center=mp.Vector3(-6.0), size=mp.Vector3(1.8, 6))
)
flux2 = sim.flux_in_box(
    mp.X, mp.Volume(center=mp.Vector3( 6.0), size=mp.Vector3(1.8, 6))
)
```

The first argument `mp.X` specifies the component of the Poynting vector to integrate (the $x$-component). The test in `test_wvg_src.py` asserts that `flux1` is near zero ($-1.78\times10^{-3}$) while `flux2 \approx 7.22$ W/µm, confirming that almost all power flows rightward.

The `rm_h5=False` argument in `mp.output_png` keeps the intermediate HDF5 file; the FAIL status in the example is due to `h5topng` (from the `h5utils` package) not being installed—the physics is correct.

#### Key Takeaways

- `EigenModeSource` with `component=mp.Dielectric` injects the complete multi-component mode profile automatically, rather than exciting a single field component.
- `force_complex_fields=True` is required to compute time-averaged (steady-state) flux from a CW simulation without time averaging.
- `flux_in_box` provides a quick integral of the Poynting flux over a volume, useful for checking power balance.
- `eig_parity=mp.ODD_Z` restricts the eigenmode search to modes odd in $z$, which corresponds to TM polarization in 2D.
- The asymmetric waveguide geometry tests that the eigenmode source handles non-symmetric profiles correctly.

---

### 6. `mode-decomposition.py` — Waveguide Taper Reflectance via Mode Decomposition

**Physics:** A linear waveguide taper that adiabatically converts from a narrow waveguide to a wide one should in principle have zero reflection; this example quantifies how the reflection decreases as the taper becomes longer, comparing two independent measurement methods.
**Difficulty:** Advanced
**Source:** `python/examples/mode-decomposition.py`
**Test Status:** TIMEOUT (CPU-intensive)

#### Theory

A waveguide taper that expands from width $w_1$ to width $w_2$ over length $L_t$ reflects power back toward the source because the local mode profile changes along the taper. In the adiabatic limit ($L_t \to \infty$), the mode follows the local geometry without coupling to radiation or back-reflected modes, so $R \to 0$. For a linear taper, the reflectance scales as:

$$R \propto \frac{1}{L_t^2}$$

This quadratic dependence arises from first-order perturbation theory: the scattering matrix element for back-reflection involves an overlap integral of the mode mismatch across one period, and the total amplitude accumulates coherently over the taper length.

Two independent methods are used to measure $R$:

1. **Mode decomposition:** Project the DFT fields at the reflection monitor onto the eigenmode basis using `get_eigenmode_coefficients`, obtaining the complex amplitude $\alpha^{(-)}$ of the backward-traveling mode. Then $R = |\alpha^{(-)}|^2 / |\alpha^{(+)}|^2$.

2. **Poynting flux subtraction:** Subtract the incident flux (measured in the normalization run) from the total flux at the same monitor; the residual negative flux equals the reflected power.

Agreement between the two methods validates both approaches and confirms that the decomposition correctly accounts for all the reflected power.

#### Code Walkthrough

The simulation iterates over taper lengths $L_t \in \{1, 2, 4, 8\}$ µm. For each, a straight-waveguide normalization run is performed first:

```python
for Lt in Lts:
    # Normalization run (straight waveguide)
    sim = mp.Simulation(
        geometry=[mp.Prism(vertices, height=mp.inf, material=Si)],
        sources=sources, ...
    )
    flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mon_pt, ...))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))
    res = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
    incident_coeffs = res.alpha
    incident_flux_data = sim.get_flux_data(flux)
    sim.reset_meep()
```

The taper geometry is a polygon (Prism) whose vertex list encodes the linear transition:

```python
vertices = [
    mp.Vector3(-0.5*sx - 1,  0.5*w1), mp.Vector3(-0.5*Lt,  0.5*w1),
    mp.Vector3( 0.5*Lt,  0.5*w2),     mp.Vector3( 0.5*sx + 1,  0.5*w2),
    mp.Vector3( 0.5*sx + 1, -0.5*w2), mp.Vector3( 0.5*Lt, -0.5*w2),
    mp.Vector3(-0.5*Lt, -0.5*w1),     mp.Vector3(-0.5*sx - 1, -0.5*w1),
]
```

After subtracting the incident flux data, the residual coefficients give the reflection:

```python
R_coeffs.append(abs(taper_coeffs[0, 0, 1])**2 / abs(incident_coeffs[0, 0, 0])**2)
R_flux.append(-taper_flux[0] / incident_flux[0])
```

The final log-log plot should show both curves falling along a $1/L_t^2$ reference line.

#### Key Takeaways

- Mode decomposition gives the complex reflection coefficient (amplitude and phase), while Poynting flux gives only the total reflected power; both give the same reflectance $R = |r|^2$.
- `mp.Mirror(mp.Y)` symmetry halves the computational cost by exploiting the left-right symmetry of the taper geometry and source.
- `eig_parity=mp.ODD_Z + mp.EVEN_Y` selects TM-polarized even modes, which is the fundamental mode of the taper.
- The convergence of $R \propto L_t^{-2}$ is a fundamental result from coupled-mode theory and adiabatic theorem.
- `sim.reset_meep()` clears the simulation state without reinstantiating the `Simulation` object, preparing for the next loop iteration.

---

### 7. `mode_coeff_phase.py` — Complex Reflection Coefficient of Total Internal Reflection

**Physics:** At a flat interface between two lossless dielectrics, a planewave incident beyond the critical angle undergoes total internal reflection (TIR) with a well-defined phase shift given by the Fresnel equations. This example verifies that Meep's mode-decomposition feature correctly reproduces both the magnitude and the phase of the complex reflection coefficient.
**Difficulty:** Advanced
**Source:** `python/examples/mode_coeff_phase.py`
**Test Status:** TIMEOUT (CPU-intensive)

#### Theory

For S-polarization (electric field perpendicular to the plane of incidence, equivalent to TM-like $E_z$ in 2D), the complex Fresnel reflection coefficient at angle $\theta$ (measured from the interface normal in the denser medium, index $n_1$) is:

$$r_s = \frac{\cos\theta - \sqrt{(n_2/n_1)^2 - \sin^2\theta}}{\cos\theta + \sqrt{(n_2/n_1)^2 - \sin^2\theta}}$$

When $\theta > \theta_c = \arcsin(n_2/n_1)$, the square root becomes imaginary and $|r_s| = 1$—total internal reflection. The phase $\phi_s = \arg(r_s)$ is nonzero and depends on $\theta$ and the index ratio. This phase shift is physically significant: it is the Goos-Hänchen effect and underlies the operation of optical fibers, prism-based delay elements, and interferometric sensors.

The Meep simulation injects an oblique planewave at angle $\theta$ using a phase-modulated line source (`amp_func`). A mode monitor on the incident side captures both the incoming and reflected planewave modes. The trick for extracting the complex $r$ is to:

1. Run without the interface to measure the incident coefficient $\alpha_\mathrm{in}$.
2. Run with the interface (load minus incident flux) to isolate the reflected coefficient $\alpha_\mathrm{refl}$.
3. Form $r = \alpha_\mathrm{refl}/\alpha_\mathrm{in}$ and apply a phase correction for the monitor-to-interface propagation distance $L$.

#### Code Walkthrough

The oblique planewave is injected using a phase-modulated amplitude function:

```python
k = mp.Vector3(n1 * fcen, 0, 0).rotate(mp.Vector3(0, 0, 1), theta)

def pw_amp(k, x0):
    def _pw_amp(x):
        return cmath.exp(1j * 2 * math.pi * k.dot(x + x0))
    return _pw_amp

sources = [mp.Source(..., amp_func=pw_amp(k, src_pt))]
```

The `k_point=k` parameter imposes Bloch-periodic boundary conditions in $y$, which are needed for the oblique planewave to propagate without being reflected at the $y$ boundaries.

After extracting the mode coefficients with a specified `kpoint_func`:

```python
res = sim.get_eigenmode_coefficients(
    mode_mon, bands=[1], eig_parity=eig_parity,
    kpoint_func=lambda *not_used: k,
    direction=mp.NO_DIRECTION,
)
```

The phase correction accounts for the monitor being displaced from the interface by distance $L$:

```python
refl_coeff = refl_mode_coeff / input_mode_coeff
refl_coeff /= cmath.exp(1j * k.x * 2 * math.pi * 2 * L)
```

The factor of 2 accounts for the round-trip propagation (incident + reflected) over distance $L$.

#### Key Takeaways

- The `amp_func` parameter of `mp.Source` enables injection of spatially-modulated sources, including oblique planewaves.
- `k_point` sets the Bloch wavevector for periodic boundary conditions—essential for oblique planewave simulations.
- `direction=mp.NO_DIRECTION` with a custom `kpoint_func` allows mode decomposition along an arbitrary propagation direction, not just the coordinate axes.
- The phase correction $e^{i k_x \cdot 2L}$ must be applied when the monitor is not co-located with the scattering interface.
- Agreement with the Fresnel formula to 4–5 significant figures validates both the mode decomposition and the phase tracking in Meep.

---

### 8. `parallel-wvgs-force.py` — Optical Forces Between Coupled Waveguides (FDTD)

**Physics:** Two parallel dielectric waveguides in close proximity experience an optically induced force due to the gradient of the electromagnetic energy with respect to their separation; symmetric (even) supermodes produce an attractive force, while anti-symmetric (odd) supermodes produce a repulsive force.
**Difficulty:** Advanced
**Source:** `python/examples/parallel-wvgs-force.py`
**Test Status:** TIMEOUT (CPU-intensive)

#### Theory

The optical force per unit length on a dielectric waveguide in a guided mode can be derived from Maxwell's stress tensor. For two parallel waveguides separated by gap $s$, the force is related to the change in mode frequency with separation at fixed photon number (or equivalently, at fixed wavevector):

$$\frac{F}{L} = -\frac{1}{v_g} \frac{\partial \omega}{\partial s}\bigg|_k$$

This formula follows from the Hellmann-Feynman theorem applied to the photonic eigenvalue problem, analogous to the Casimir-Polder force but at a classical level. For the even (symmetric) supermode, the energy is minimized at smaller $s$ (because the mode is pulled into the gap), yielding an attractive force. For the odd (anti-symmetric) supermode, the energy increases as the waveguides approach, yielding a repulsive force.

The force is computed directly in Meep by integrating the Maxwell stress tensor over a closed surface surrounding one waveguide:

$$F_x = \oint \left(T_{xx} n_x + T_{xy} n_y\right) dA$$

where $T_{ij} = \varepsilon E_i E_j + \mu H_i H_j - \frac{1}{2}\delta_{ij}(\varepsilon E^2 + \mu H^2)$ is the Maxwell stress tensor.

#### Code Walkthrough

Two square silicon waveguides ($1 \times 1$ µm, $n = 3.45$) are placed symmetrically on either side of $x = 0$, separated by gap $s$:

```python
geometry = [
    mp.Block(center=mp.Vector3(-0.5*(s + a)), size=mp.Vector3(a, a, mp.inf), material=Si),
    mp.Block(center=mp.Vector3(+0.5*(s + a)), size=mp.Vector3(a, a, mp.inf), material=Si),
]
```

The supermode is excited from an `EigenModeSource` with the appropriate symmetry. Even/odd selection is enforced via `mp.Mirror(mp.X, phase=...)`:

```python
symmetries = [mp.Mirror(mp.X, phase=-1 if xodd else 1), mp.Mirror(mp.Y, phase=-1)]
```

The force is measured using `ForceRegion` objects surrounding one waveguide with opposite-sign weights to form a closed surface integral in 2.5D (infinite in $z$):

```python
force_reg1 = mp.ForceRegion(mp.Vector3(0.49*s), direction=mp.X, weight=+1, size=mp.Vector3(y=sy))
force_reg2 = mp.ForceRegion(mp.Vector3(0.5*s + 1.01*a), direction=mp.X, weight=-1, size=mp.Vector3(y=sy))
wvg_force = sim.add_force(fcen, 0, 1, force_reg1, force_reg2)
```

The normalized force $(F/L)(ac/P)$ is plotted against separation $s/a$, where the normalization by the guided power $P$ and mode group velocity $c$ makes the result geometry-independent.

#### Key Takeaways

- `mp.ForceRegion` with `weight=+1` and `weight=-1` implements the closed-surface Maxwell stress tensor integral.
- `sim.add_force` adds a DFT force monitor; `mp.get_forces` retrieves the frequency-resolved force.
- The force sign changes between even and odd supermodes: even modes attract, odd modes repel—a fundamental result of coupled-mode optics.
- `k_point=mp.Vector3(z=0.5)` sets the longitudinal wavevector (in units of $2\pi/$µm) for a mode propagating in $z$; this enables the waveguide mode simulation in a cell finite in $x$ and $y$.
- `sim.init_sim()` followed by `sim.get_eigenmode(...)` is the low-level API for querying mode properties before running the full FDTD.

---

### 9. `parallel-wvgs-mpb.py` — Optical Forces Between Coupled Waveguides (MPB)

**Physics:** This example computes the same optical forces as Section 8 but uses the MPB photonic band solver rather than FDTD, computing the force from the derivative of the mode frequency with respect to waveguide separation via the Maxwell-Feynman theorem.
**Difficulty:** Advanced
**Source:** `python/examples/parallel-wvgs-mpb.py`
**Test Status:** TIMEOUT (CPU-intensive)

#### Theory

The optical force formula $F/L = -(1/v_g) \partial\omega/\partial s|_k$ can be evaluated efficiently using MPB: compute the mode frequency $\omega(s)$ and group velocity $v_g(s)$ at a fixed wavevector $k$ for a range of separations, then numerically differentiate:

$$\frac{F}{L} = -\frac{1}{v_g} \frac{\partial \omega}{\partial s} \approx -\frac{1}{v_g} \frac{\Delta\omega}{\Delta s}$$

This approach complements the direct FDTD force measurement of Section 8. MPB solves the master equation $\nabla \times \frac{1}{\varepsilon} \nabla \times \mathbf{H} = \left(\frac{\omega}{c}\right)^2 \mathbf{H}$ as a Hermitian eigenvalue problem, yielding exact frequencies and field profiles without time-domain transients. The finite-difference differentiation with respect to $s$ introduces a discretization error $O(\Delta s^2)$, so the step size `ds = 0.05` is chosen to balance accuracy and computational effort.

The group velocity is computed from the eigenmode fields via the expectation value $v_g = d\omega/dk = \langle \mathbf{H} | \partial H/\partial k | \mathbf{H} \rangle / \langle \mathbf{H}|\mathbf{H}\rangle$.

#### Code Walkthrough

The `mpb.ModeSolver` is set up with a 1D periodic unit cell (infinite in $x$, finite in $y$ and $z$):

```python
ms = mpb.ModeSolver(
    resolution=resolution,
    k_points=k_points,
    geometry_lattice=mp.Lattice(size=mp.Vector3(0, syz, syz)),
    geometry=geometry,
    num_bands=1,
    tolerance=1e-9,
)
```

`run_yodd_zodd()` selects the anti-symmetric (odd) mode; `run_yeven_zodd()` selects the symmetric (even) mode:

```python
if yodd:
    ms.run_yodd_zodd()
else:
    ms.run_yeven_zodd()

f = ms.get_freqs()[0]
vg = ms.compute_group_velocity_component(mp.Vector3(1, 0, 0))[0]
```

The force is then computed by numerical differentiation over the array of separations `ss`:

```python
def compute_force(f, vg):
    f_avg = 0.5 * (f[:-1] + f[1:])
    df = f[1:] - f[:-1]
    vg_avg = 0.5 * (vg[:-1] + vg[1:])
    return -1 / f_avg * df / ds * 1 / vg_avg
```

#### Key Takeaways

- MPB computes exact band frequencies without FDTD noise, giving cleaner force curves from numerical differentiation.
- `ms.compute_group_velocity_component` evaluates $v_g$ from the mode field profile analytically (no finite differences needed in $k$ space).
- The FDTD (Section 8) and MPB (Section 9) methods should give consistent forces, serving as a cross-validation of both approaches.
- `ms.run_yodd_zodd()` versus `ms.run_yeven_zodd()` selects the parity symmetry of the mode; combining parity constraints with a small unit cell dramatically reduces memory and CPU requirements at high resolution.
- Both attractive (even) and repulsive (odd) forces are large and practically accessible with guided optical powers in the milliwatt range—this is the basis of optomechanical waveguide actuators.

---

### 10. `ring-mode-overlap.py` — Mode Overlap Integral Between Ring Resonator Modes

**Physics:** A 2D ring resonator supports whispering-gallery modes at discrete resonant frequencies determined by the round-trip phase condition. This example finds two modes (at approximately $\omega$ and $2\omega$) using Harminv, stores the field of the first mode, runs a second simulation to find the second mode, then evaluates their spatial overlap integral.
**Difficulty:** Intermediate
**Source:** `python/examples/ring-mode-overlap.py`
**Test Status:** TIMEOUT (CPU-intensive)

#### Theory

A ring resonator of inner radius $r$ and waveguide width $w$ supports transverse electric (TE) modes with azimuthal mode numbers $m = 1, 2, 3, \ldots$ The resonance condition is approximately:

$$m \lambda \approx 2\pi n_\mathrm{eff}(r + w/2)$$

or equivalently $m / (2\pi n_\mathrm{eff} R_\mathrm{eff}) = f$. For $n = 3.4$, $r = 1$, $w = 1$, the effective optical path length gives resonances at frequencies detectable by Harminv within the pulse bandwidth.

The overlap integral between two modes $E_z^{(1)}$ and $E_z^{(2)}$ is:

$$\mathcal{O} = \int \left[E_z^{(1)}(\mathbf{r})\right]^* E_z^{(2)}(\mathbf{r})\, d^2r$$

For modes with different azimuthal quantum numbers, this integral vanishes by orthogonality. For the same mode, it equals the mode norm squared. The function `sim.integrate2_field_function` computes this overlap by combining fields stored from two separate simulation runs, providing a direct numerical test of mode orthogonality.

This is relevant for nonlinear optics (e.g., second-harmonic generation in rings requires $\mathcal{O}_{m, 2m} \neq 0$, which requires specific phase-matching conditions) and for perturbation theory calculations.

#### Code Walkthrough

The ring is constructed from two overlapping cylinders (inner air, outer high-index):

```python
geometry = [
    mp.Cylinder(radius=r + w, height=mp.inf, material=mp.Medium(index=n)),
    mp.Cylinder(radius=r,     height=mp.inf, material=mp.air),
]
```

Later geometry objects take precedence in Meep, so the inner air cylinder cuts the hole in the outer dielectric cylinder.

`mp.Harminv` is attached as a step function that analyzes the time-domain signal during the simulation:

```python
h1 = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
sim.run(mp.after_sources(h1), until_after_sources=300)
```

`mp.after_sources(h1)` starts the Harminv analysis only after the Gaussian source has decayed, so the mode quality factors are not contaminated by source transients.

After the first mode simulation, the fields object is saved:

```python
fields2 = sim.fields
sim.reset_meep()
```

After the second simulation (different `fcen`), the overlap integral is computed:

```python
def overlap_integral(r, ez1, ez2):
    return ez1.conjugate() * ez2

res = sim.integrate2_field_function(fields2, [mp.Ez], [mp.Ez], overlap_integral)
print(f"overlap integral of mode at w and 2w: {abs(res)}")
```

`integrate2_field_function` evaluates a user-defined function of the fields from two different simulation objects and integrates over the cell.

#### Key Takeaways

- Ring resonator modes are found by `mp.Harminv` (harmonic inversion), which extracts resonant frequencies and quality factors from the time-domain signal—far more accurate than a simple DFT peak search.
- `mp.after_sources(h1)` defers analysis until the source has decayed, ensuring that Harminv analyzes the free-ringing resonator response only.
- `sim.fields` stores a reference to the current field state; `integrate2_field_function` allows cross-correlation of fields from two separate simulations.
- Setting `mp.Mirror(mp.Y)` symmetry halves the cell cost while preserving the modes, since the source is placed off-symmetry (at $x = r + 0.1$, $y = 0$) to couple to both even and odd $y$ modes.
- Mode orthogonality ($\mathcal{O} \approx 0$ for different $m$) or the lack thereof (for modes with accidental degeneracy or same $m$) is directly measurable with this method.

---

### 11. `test_bend_flux.py` — Unit Test for Waveguide Bend Flux Measurement

**Physics:** This test validates the two-run normalization procedure for measuring transmission and reflection at a 90-degree waveguide bend, verifying numerical results against hard-coded expected values and testing that two different DFT decimation factors give identical spectra.
**Difficulty:** Intermediate
**Source:** `python/tests/test_bend_flux.py`
**Test Status:** PASS (61.7s)

#### Theory

The test reproduces the physics of Section 2 in an automated unit-test framework. The key numerical check is that the DFT-monitored Poynting flux spectra are reproducible to high precision (tolerance $10^{-3}$ for the bend run). Two variants are tested: the geometry is specified either as two `Block` objects or (if libGDSII is available) as `Prism` objects imported from a GDSII file, confirming that both geometry representations produce identical physics.

An important numerical feature tested here is the `decimation_factor` parameter of `add_flux`. DFT integration normally accumulates at every FDTD time step, which can be memory-intensive for large cells or long runs. Setting `decimation_factor=5` or `decimation_factor=10` accumulates only every 5th or 10th step. The Nyquist criterion requires sampling at least twice per period of the highest frequency of interest; since the DFT is evaluated at fixed frequencies (not time), decimation is valid as long as the sampling rate exceeds the Nyquist rate for the bandwidth `df`.

#### Code Walkthrough

The test class inherits from `ApproxComparisonTestCase` which provides `self.assertClose` with a relative tolerance:

```python
class TestBendFlux(ApproxComparisonTestCase):
    def run_bend_flux(self, from_gdsii_file):
        # Normalization run
        self.init(no_bend=True, gdsii=from_gdsii_file)
        self.sim.run(until_after_sources=mp.stop_when_energy_decayed(100, 1e-3))
```

`mp.stop_when_energy_decayed(100, 1e-3)` runs for at least 100 time units after the source has peaked, and continues until the total field energy has decayed by three orders of magnitude—an alternative adaptive stopping condition to `stop_when_fields_decayed`.

The test verifies that both standard and decimated flux monitors agree:

```python
tol = 1e-6 if mp.is_single_precision() else 1e-8
self.assertClose(np.array(expected), np.array(res[:20]), epsilon=tol)
self.assertClose(np.array(expected), np.array(res_decimated[:20]), epsilon=tol)
```

A separate check verifies the relationship between the real `flux()` method and the real part of the complex `complexflux()` method:

```python
trans_flux_real = np.array(self.trans.flux())
trans_flux_complex = np.array(self.trans.complexflux())
assert_array_equal(trans_flux_real, np.real(trans_flux_complex))
```

This confirms internal consistency of the flux data structures.

#### Key Takeaways

- The `decimation_factor` in `add_flux` reduces memory usage for DFT accumulation; values up to 10 are safe for typical bandwidths without aliasing.
- `mp.stop_when_energy_decayed(T, tol)` is an alternative to `stop_when_fields_decayed` that integrates the total energy over the cell rather than checking a single point.
- `FluxRegion` can be specified with either a center+size pair or with an explicit array of frequencies (as shown in the `refl` monitor initialization using `np.linspace`).
- Unit tests with hard-coded expected values serve as regression tests: any future change to the Meep core that alters the numerical answer by more than the tolerance will be caught immediately.
- The `gdsii=True` code path validates that `mp.get_GDSII_prisms` produces exactly the same geometry as manually specified `Prism` vertices.

---

### 12. `test_wvg_src.py` — Unit Test for Eigenmode Source Power Balance

**Physics:** This test verifies that an `EigenModeSource` in an asymmetric waveguide launches power exclusively in the forward direction, with negligible back-coupling, by checking the left-going and right-going fluxes against expected values.
**Difficulty:** Beginner
**Source:** `python/tests/test_wvg_src.py`
**Test Status:** PASS (7.0s)

#### Theory

A transparent eigenmode source should act as a one-sided emitter: all the injected power flows in one direction (the direction determined by the mode's phase velocity), with essentially zero power flowing in the opposite direction. In practice, there is a small residual backward flux because the eigenmode calculation is performed on the discretized Yee grid rather than the continuous Maxwell equations—the discrete mode profile does not exactly match the continuous one, leaving a small mismatch that radiates.

The expected values in the test are:
- Left-going flux: $\approx -1.78 \times 10^{-3}$ (the negative sign indicates flow in $-x$)
- Right-going flux: $\approx 7.22$

The ratio of backward-to-forward power is roughly $2.5 \times 10^{-4}$—a directivity of about 36 dB, demonstrating excellent mode selectivity despite the asymmetric waveguide geometry.

#### Code Walkthrough

The test setup is identical to `wvg-src.py` (Section 5). The unit test provides explicit numerical targets:

```python
def test_wvg_src(self):
    self.sim.run(until=200)

    flux1 = self.sim.flux_in_box(mp.X, mp.Volume(center=mp.Vector3(-6.0), size=mp.Vector3(1.8, 6)))
    flux2 = self.sim.flux_in_box(mp.X, mp.Volume(center=mp.Vector3( 6.0), size=mp.Vector3(1.8, 6)))

    self.assertAlmostEqual(flux1, -1.775216564842667e-03)
    places = 5 if mp.is_single_precision() else 7
    self.assertAlmostEqual(flux2, 7.215785537102116, places=places)
```

The `places` variable adjusts numerical tolerance for single-precision builds, where results are accurate to about 6 significant digits rather than the 14–15 of double precision.

#### Key Takeaways

- `flux_in_box` provides a quick integral of the Poynting flux over any rectangular volume, without needing a pre-registered DFT monitor.
- Backward-going flux is negative (by sign convention) when measured in the $+x$ direction.
- The tight numerical tolerance in `assertAlmostEqual` (7 decimal places for double precision) confirms that the eigenmode source is deterministic and reproducible across platforms.
- `mp.is_single_precision()` allows tests to adapt their tolerance to the floating-point precision of the build.
- The asymmetric waveguide (with the embedded air slit) makes this a non-trivial test of the eigenmode solver's ability to handle non-symmetric mode profiles.

---

### 13. `test_holey_wvg_bands.py` — Photonic Band Structure of a Holey Waveguide

**Physics:** A dielectric waveguide with a periodic row of air holes (a photonic crystal waveguide) has a modified band structure with photonic bandgaps and guided resonances. This test computes the band structure by sweeping $k$ over the Brillouin zone and extracts resonant frequencies at a fixed $k_x$.
**Difficulty:** Advanced
**Source:** `python/tests/test_holey_wvg_bands.py`
**Test Status:** PASS (61.3s)

#### Theory

A photonic crystal waveguide consists of a dielectric slab (width 1.2 µm, $\varepsilon = 13$, similar to InP or GaAs at telecom wavelengths) with a periodic row of cylindrical air holes (radius 0.36 µm, period 1 µm). The periodicity creates photonic bandgaps—frequency ranges where no guided modes exist—by Bragg diffraction. Within the bandgap, only modes with special symmetry (due to defects or edges of the bandgap) can propagate.

The band structure is obtained by running `sim.run_k_points`, which performs a separate FDTD simulation at each Bloch wavevector $k_x$ in the list. At each $k_x$, a broadband Gaussian pulse excites all modes, and Harminv extracts the resonant frequencies. The collection of ($k_x$, $\omega$) pairs traces out the photonic band curves.

At a fixed $k_x = 3.5$ (in units of $2\pi$/period), which is well inside the Brillouin zone, the `Hz` Harminv analysis reveals multiple resonances with distinct quality factors—modes with real frequencies are guided or weakly leaky, while those with large imaginary parts decay quickly into radiation modes.

The cell has Bloch-periodic boundary conditions in $x$ (period 1 µm) and PML absorbers in $y$ to damp radiation into the cladding. `mp.Mirror(direction=mp.Y, phase=-1)` exploits the odd-$y$ symmetry of the $H_z$ component of TM-like modes.

#### Code Walkthrough

The unit cell has $x$-dimension 1 (one period) and $y$-dimension 12 (including PML):

```python
cell = mp.Vector3(1, 12)
b = mp.Block(size=mp.Vector3(mp.inf, 1.2, mp.inf), material=mp.Medium(epsilon=13))
c = mp.Cylinder(0.36)
```

The band structure calculation sweeps 20 evenly-spaced $k$ points from 0 to $0.5$ (the Brillouin zone boundary):

```python
all_freqs = self.sim.run_k_points(
    5,  # run each k for 5 time units after sources
    mp.interpolate(19, [mp.Vector3(), mp.Vector3(0.5)])
)
```

`mp.interpolate(19, ...)` generates 21 points (19 interior + 2 endpoints) between $\Gamma$ (k=0) and the zone edge (k=0.5). At each k-point, Harminv extracts resonances; `all_freqs` is a list of lists of complex frequencies.

For the fixed-$k_x$ analysis:

```python
self.sim.k_point = mp.Vector3(3.5)
h = mp.Harminv(mp.Hz, mp.Vector3(0.1234), self.fcen, self.df)
self.sim.run(mp.after_sources(h), until_after_sources=300)
```

The source is placed at $x = 0.1234$ (off-center to break spatial symmetry and excite all modes) and the run length is 300 time units—sufficient for Harminv to resolve modes with $Q \sim 300$.

#### Key Takeaways

- `sim.run_k_points` automates band structure calculations by looping over Bloch $k$ points and collecting Harminv results.
- `mp.interpolate(N, [k1, k2])` generates $N+2$ evenly spaced k-points between `k1` and `k2` along the Brillouin zone path.
- Modes with small imaginary Harminv frequency (decay rate) are long-lived guided modes; large imaginary parts indicate leaky radiation modes.
- The off-symmetry source position ($x = 0.1234$ rather than 0 or 0.5) couples to all modes that have nonzero field amplitude at that point.
- Photonic crystal waveguides exhibit slow-light behavior near the Brillouin zone edge ($k = 0.5$) where $\partial\omega/\partial k \to 0$—detectable as closely-spaced $k$ points that produce nearly the same $\omega$.

---

### 14. `test_holey_wvg_cavity.py` — Resonant Cavity in a Photonic Crystal Waveguide

**Physics:** By introducing a point defect (missing hole or cavity region) in a photonic crystal waveguide, a localized resonant mode is created within the photonic bandgap. This test extracts the resonant frequency and quality factor of the cavity mode, and separately computes the transmission spectrum showing the Fano/Lorentzian resonance feature.
**Difficulty:** Advanced
**Source:** `python/tests/test_holey_wvg_cavity.py`
**Test Status:** PASS (22.6s)

#### Theory

A cavity formed by spacing two mirror arrays of 3 holes each ($N=3$ on each side) with a central defect region of length $d = 1.4$ µm localizes a mode within the bandgap of the photonic crystal. The mode is characterized by its resonant frequency $f_0$ and quality factor $Q$:

$$Q = \frac{f_0}{2 |\mathrm{Im}(f)|}$$

where $\mathrm{Im}(f)$ is the exponential decay rate obtained from Harminv. Here the test expects $Q \approx 372$ for the first resonance near $f = 0.234$.

The transmission spectrum shows a Fano-type resonance: a sharp dip (or peak) at the cavity frequency, with a linewidth $\Delta f = f_0/Q$. The Fano shape arises from interference between the direct transmission pathway through the waveguide and the indirect pathway through the cavity. A pure Lorentzian resonance occurs when only one pathway exists; the asymmetric Fano shape results from the interference of the two.

The test verifies 46 specific $(f, T)$ data points along the transmission spectrum, confirming that the resonance lineshape is reproduced numerically to 10 decimal places of accuracy.

#### Code Walkthrough

The cavity geometry consists of a waveguide slab plus 6 cylinders (3 on each side of center):

```python
geometry.extend(mp.Cylinder(r, center=mp.Vector3(d/2 + i)) for i in range(3))
for i in range(3):
    geometry.append(mp.Cylinder(r, center=mp.Vector3(d/-2 - i)))
```

For the resonant mode test, a point source is placed at the cavity center with both $x$- and $y$-mirror symmetries to select specific mode parities:

```python
self.sim.sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.df), mp.Hz, mp.Vector3())]
self.sim.symmetries = [mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.X, phase=-1)]
h = mp.Harminv(mp.Hz, mp.Vector3(), self.fcen, self.df)
self.sim.run(mp.at_beginning(mp.output_epsilon), mp.after_sources(h), until_after_sources=400)
```

The dual-symmetry constraint ($Y$-antisymmetric, $X$-antisymmetric) selects modes that are odd in both directions—this isolates the cavity mode from background waveguide modes.

For the transmission spectrum, the source is placed at the waveguide input with a `FluxRegion` at the output:

```python
freg = mp.FluxRegion(
    center=mp.Vector3((0.5*self.sx) - self.dpml - 0.5),
    size=mp.Vector3(0, 2*self.w),
)
trans = self.sim.add_flux(self.fcen, self.df, self.nfreq, freg, decimation_factor=1)
```

With `nfreq=500` frequencies sampled over the Gaussian bandwidth, the spectrum has sufficient resolution to resolve the Fano feature.

#### Key Takeaways

- `mp.Harminv` extracts both the resonant frequency (real part) and the decay rate (imaginary part) from which $Q$ is computed.
- Double mirror symmetry ($X$ and $Y$) constrains the simulation to a quarter of the cell, reducing computation by 4× while selecting specific mode parities.
- The transmission spectrum requires a normalization run (without the photonic crystal) to separate the resonance from the source spectrum; this is handled by the different source configurations in the two test methods.
- Quality factor $Q \approx 372$ corresponds to a photon lifetime $\tau = Q/(2\pi f_0) \approx 253$ time units—the run length of 400 units exceeds this by 1.6 lifetimes, ensuring adequate Harminv convergence.
- The expected transmission values span many orders of magnitude across the spectrum, testing numerical precision in both the bandgap (near-zero transmission) and the passband regions.

---

### 15. `test_mode_decomposition.py` — Comprehensive Mode Decomposition Tests

**Physics:** A suite of four tests verifying mode decomposition in different geometries: a 2D linear taper, an oblique waveguide in 2D, a 3D grating with normally-incident light, and a 3D grating with oblique incidence on a triangular lattice.
**Difficulty:** Advanced
**Source:** `python/tests/test_mode_decomposition.py`
**Test Status:** FAIL (missing `parameterized`)

#### Theory

Mode decomposition is the process of projecting the DFT fields at a monitor cross-section onto a set of eigenmodes, yielding complex amplitudes $\alpha_n^{(\pm)}$ for each mode $n$ traveling in the forward ($+$) or backward ($-$) direction. The relationship between mode amplitudes and Poynting flux is:

$$\Phi = \sum_n \left(|\alpha_n^{(+)}|^2 - |\alpha_n^{(-)}|^2\right)$$

This allows decomposition of the total reflected or transmitted flux into contributions from individual diffraction orders, which is essential for analyzing gratings, photonic crystals, and any periodic structure.

The 3D grating test verifies energy conservation: the sum of all reflected diffraction orders $R_\mathrm{sum}$ and transmitted diffraction orders $T_\mathrm{sum}$ must satisfy $R_\mathrm{sum} + T_\mathrm{sum} = 1$ (within numerical accuracy). The diffracted orders are indexed by integers $(m_x, m_y)$ with wavevectors $k_{x,m} = k_x^{(0)} + m_x/\Lambda_x$.

The TIR phase test (parameterized over S and P polarizations) validates the complex coefficient—specifically the phase—against the Fresnel formula. This is the same physics as Section 7 but here embedded in the unit-test framework with `parameterized`.

The FAIL status is due to `import parameterized` failing when the `parameterized` package is not installed (`pip install parameterized` fixes it).

#### Code Walkthrough

The oblique waveguide test checks that a mode propagating at 35 degrees still decomposes correctly with `direction=mp.NO_DIRECTION`:

```python
rot_angle = np.radians(35.0)
kpoint = mp.Vector3(1, 0, 0).rotate(mp.Vector3(0, 0, 1), rot_angle) * -1.0
coeff = sim.get_eigenmode_coefficients(
    mode, [1], direction=mp.NO_DIRECTION,
    kpoint_func=lambda *not_used: kpoint
).alpha[0, 0, 0]
```

The 3D grating test uses `mp.DiffractedPlanewave` to specify individual grating orders:

```python
res = sim.get_eigenmode_coefficients(
    refl_flux,
    mp.DiffractedPlanewave([m_x, m_y, 0], mp.Vector3(1, 0, 0), 1 if S_pol else 0, 0 if S_pol else 1),
)
```

The first argument `[m_x, m_y, 0]` specifies the diffraction order indices; the `Vector3` specifies the reference polarization direction; and the last two arguments encode S ($E_y$-like) or P ($H_y$-like) polarization.

#### Key Takeaways

- `mp.DiffractedPlanewave` enables decomposition of periodic-structure fields into individual Bragg orders indexed by $(m_x, m_y)$.
- The `parameterized` decorator (`@parameterized.parameterized.expand`) generates separate test methods for each parameter set at collection time—without the package, the test module fails to import.
- Energy conservation $R + T = 1$ to 1–2 decimal places at the resolutions used is a fundamental physical check that catches sign errors, missing orders, or monitor placement issues.
- The oblique waveguide test uses `direction=mp.NO_DIRECTION` with an explicit `kpoint_func`—the only way to handle modes that do not propagate along a coordinate axis.
- The triangular lattice test demonstrates that `mp.DiffractedPlanewave` handles non-orthogonal lattice geometries by expressing orders in Cartesian coordinates.

---

### 16. `test_mode_coeffs.py` — Mode Coefficient Extraction and Source Normalization

**Physics:** A suite of tests validating that `EigenModeSource` correctly excites specific waveguide modes (the excited mode has unit coefficient while other modes have near-zero coefficients), that the mode amplitude relates to power as $P = |\alpha|^2$, and that S-parameters satisfy reciprocity ($|S_{21}| = |S_{12}|$).
**Difficulty:** Advanced
**Source:** `python/tests/test_mode_coeffs.py`
**Test Status:** PASS (183.5s)

#### Theory

For a lossless, time-reversal-symmetric waveguide, the scattering matrix $S$ must satisfy $S^\dagger S = I$ (unitarity). For a single-mode two-port system:

$$|S_{11}|^2 + |S_{21}|^2 = 1, \quad S_{12} = S_{21}$$

The second relation is reciprocity. These constraints are not assumed in the FDTD calculation—they emerge as consequences of Maxwell's equations and the geometry. Verifying them numerically (within discretization error) is a non-trivial consistency check.

The mode power normalization test verifies that the eigenmode source injects unit power per unit source amplitude squared at all frequencies within the bandwidth. The expected power spectrum $P_\mathrm{expected}(\omega)$ is computed analytically from `source.eig_power(f)`, which evaluates the mode group velocity and normalization at each frequency. Agreement with the observed flux validates both the source normalization and the group velocity calculation.

The `kdom` field of the mode decomposition result gives the dominant wavevector component, which must satisfy $k_\mathrm{dom,y} = k_{y,0}$ (conservation of $k_y$ in translation-invariant media). This is verified in `test_kdom.py` (Section 17).

#### Code Walkthrough

The main test excites mode 1 or mode 2 and checks that the other mode's coefficient is negligible:

```python
c0 = res.alpha[mode_num - 1, 0, 0]  # forward amplitude of excited mode
for nm in range(1, len(modes_to_check) + 1):
    if nm != mode_num:
        cfrel = np.abs(res.alpha[nm - 1, 0, 0]) / np.abs(c0)
        cbrel = np.abs(res.alpha[nm - 1, 0, 1]) / np.abs(c0)
        if cfrel > TOLERANCE or cbrel > TOLERANCE:
            TestPassed = False
self.assertTrue(TestPassed)
# |alpha|^2 should equal measured power
self.assertAlmostEqual(mode_power / abs(c0**2), 1.0, places=1)
```

The reciprocity test runs the simulation twice—once with the source on the left, once on the right—and compares the transmission and reflection coefficients:

```python
# |S21|^2 = |S12|^2
self.assertAlmostEqual(
    abs(res_fwd.alpha[0, 0, 0])**2 / abs(res_bwd.alpha[0, 0, 1])**2,
    1.00, places=2,
)
```

The `EigenmodeData.amplitude(point, component)` method evaluates the complex mode field amplitude at an arbitrary point, enabling point-by-point comparison of the mode profile:

```python
ex_at_eval_point = emdata.amplitude(eval_point, mp.Ex)
```

#### Key Takeaways

- `res.alpha[band_index, freq_index, direction]` is the primary output of `get_eigenmode_coefficients`; `direction=0` is forward, `direction=1` is backward.
- `res.kpoints[n]` and `res.kdom[n]` give the wavevectors used in the decomposition; `res.cscale[n]` is the normalization factor relating the mode amplitude to field strength.
- Reciprocity ($|S_{21}| = |S_{12}|$) is verified to 2 decimal places at the resolutions used; higher accuracy requires finer discretization.
- `source.eig_power(f)` computes the analytical power injected by the eigenmode source at frequency $f$, accounting for the Gaussian spectral envelope and mode normalization.
- The `yee_grid=False` option for mode monitors places the monitor at the exact requested position rather than snapping to the nearest Yee grid point—important for reproducibility when comparing forward and backward modes.

---

### 17. `test_kdom.py` — Dominant Wavevector Conservation in Oblique Eigenmode Decomposition

**Physics:** When decomposing fields into eigenmodes at oblique incidence ($k_y \neq 0$), the dominant wavevector $k_\mathrm{dom}$ returned by the decomposition must conserve the transverse wavevector component $k_y$ exactly, as required by translational invariance in the $y$ direction.
**Difficulty:** Intermediate
**Source:** `python/tests/test_kdom.py`
**Test Status:** PASS (10.5s)

#### Theory

For a homogeneous medium (no structure varying in $y$), translational invariance requires that a mode at Bloch wavevector $k = (k_x, k_y, 0)$ maintain $k_y$ exactly throughout the cell. When Meep's eigenmode decomposition finds the dominant wavevector `kdom` of the mode at a given cross-section, `kdom.y` must equal the input $k_y$ to machine precision (within the floating-point limits of the computation).

This is verified at two different angles (10.7° and 22.9°) and for different band numbers (6 and 12 respectively). The non-trivial part is that at oblique incidence, the eigenmode solver must find the mode among many degenerate candidates that all have the same $k_y$; the `kdom` should correctly identify the unique one matching the input Bloch wavevector.

The test uses `assertAlmostEqual(..., places=15)`, which is near machine precision for double-precision arithmetic—a stringent test of the eigenmode solver's internal consistency.

#### Code Walkthrough

The simulation is a uniform glass cell ($n = 1.5$) with Bloch-periodic $y$-boundaries:

```python
k = mp.Vector3(math.cos(theta_in), math.sin(theta_in), 0).scale(fcen * ng)

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    k_point=k,
    default_material=glass,
)
sim.init_sim()
```

`sim.init_sim()` initializes the Yee grid and structures without running the time evolution, which is sufficient to call `get_eigenmode`.

The eigenmode is queried at a cross-section:

```python
EigenmodeData = sim.get_eigenmode(
    fcen, mp.X,
    mp.Volume(center=mp.Vector3(0.3*sx, 0, 0), size=mp.Vector3(0, sy, 0)),
    num_band, k, parity=eig_parity,
)
kdom = EigenmodeData.kdom
self.assertAlmostEqual(k.y, kdom.y, places=15)
```

The 15-place tolerance confirms that $k_y$ is conserved at double-precision machine accuracy.

#### Key Takeaways

- `sim.get_eigenmode(freq, direction, volume, band, k)` is the low-level API for retrieving eigenmode data at a specific cross-section and band number.
- `EigenmodeData.kdom` gives the dominant wavevector of the mode, including all components—useful for verifying Bloch vector consistency.
- `sim.init_sim()` sets up the computational structure without time-stepping; useful for querying material properties or eigenmodes before a full run.
- The 15-place precision of the $k_y$ conservation is a consequence of the way MPB implements Bloch boundary conditions; any bug in the Bloch vector handling would appear as a discrepancy here.
- For oblique sources, the `k_point` parameter sets the Bloch wavevector globally for the entire cell—any source or monitor must be consistent with this choice.

---

### 18. `test_dispersive_eigenmode.py` — Eigenmode Features with Dispersive Materials

**Physics:** This test verifies that the eigenmode source, mode decomposition, and `get_epsilon` all correctly handle frequency-dependent (dispersive) materials—including metals (Ag, Au) modeled by Drude-Lorentz fits and crystals (Si, LiNbO$_3$) modeled by multi-oscillator Sellmeier fits.
**Difficulty:** Advanced
**Source:** `python/tests/test_dispersive_eigenmode.py`
**Test Status:** PASS (36.6s)

#### Theory

Real photonic materials have frequency-dependent permittivity $\varepsilon(\omega)$. For silicon (Si), the Sellmeier dispersion formula approximates the permittivity in the near-infrared:

$$\varepsilon(\omega) = \varepsilon_\infty + \sum_j \frac{f_j \omega_j^2}{\omega_j^2 - \omega^2}$$

For noble metals like silver (Ag) and gold (Au), a Drude term (for free electrons) plus Lorentz oscillators (for interband transitions) model the complex permittivity:

$$\varepsilon(\omega) = \varepsilon_\infty - \frac{\omega_p^2}{\omega(\omega + i\gamma)} + \sum_j \frac{f_j \omega_j^2}{\omega_j^2 - \omega^2 - i\gamma_j \omega}$$

Meep implements these via susceptibility objects stored in the `Medium` class. The $\chi_1$ tensor (the linear susceptibility) is the key quantity: the effective refractive index at frequency $\omega$ is $n = \sqrt{\varepsilon(\omega)} = \sqrt{1 + \chi_1}$. The test verifies this at both the minimum and maximum frequencies of each material's `valid_freq_range`.

For anisotropic crystals like lithium niobate (LiNbO$_3$), the permittivity tensor $\varepsilon_{ij}(\omega)$ is non-diagonal in a rotated frame. The test checks that rotating the crystal orientation (via `material.rotate(axis, angle)`) correctly transforms the tensor.

#### Code Walkthrough

`call_chi1` directly tests the C++ `get_chi1inv` routine by computing the inverse susceptibility tensor at a point and comparing to Python's `material.epsilon(frequency)`:

```python
def call_chi1(self, material, frequency):
    sim = mp.Simulation(cell_size=mp.Vector3(1,1,1), default_material=material, resolution=20)
    sim.init_sim()
    v3 = mp.py_v3_to_vec(sim.dimensions, mp.Vector3(0,0,0), sim.is_cylindrical)
    chi1inv = np.zeros((3, 3), dtype=np.complex128)
    for i, com in enumerate([mp.Ex, mp.Ey, mp.Ez]):
        for k, dir in enumerate([mp.X, mp.Y, mp.Z]):
            chi1inv[i, k] = sim.structure.get_chi1inv(com, dir, v3, frequency)
    n = np.real(np.sqrt(np.linalg.inv(chi1inv.astype(np.complex128))))
    n_actual = np.real(np.sqrt(material.epsilon(frequency).astype(np.complex128)))
    self.assertClose(n, n_actual, epsilon=tol)
```

`verify_output_and_slice` checks the consistency between `get_epsilon(frequency=...)` (which returns the field array at a specified evaluation frequency) and the HDF5 file written by `mp.output_epsilon(sim, frequency=...)`:

```python
n_slice = np.sqrt(np.max(sim.get_epsilon(frequency=frequency, snap=True)))
# ...
mp.output_epsilon(sim, frequency=frequency)
with h5py.File(filename, "r") as f:
    n_h5 = np.sqrt(np.max(mp.complexarray(f["eps.r"][()], f["eps.i"][()])))
self.assertAlmostEqual(n, n_h5, places=4)
```

Rotating LiNbO$_3$:

```python
import copy
rotLiNbO3 = copy.deepcopy(LiNbO3)
rotLiNbO3.rotate(mp.Vector3(1, 1, 1), np.radians(34))
self.call_chi1(rotLiNbO3, w0)
```

`deepcopy` is needed because `rotate` modifies the material in-place; the original `LiNbO3` is preserved for the unrotated tests.

#### Key Takeaways

- `mp.Medium.epsilon(frequency)` returns the frequency-dependent permittivity tensor as a $3\times3$ NumPy array; this is the Python-side evaluation of the Drude-Lorentz model.
- `sim.structure.get_chi1inv(component, direction, point, frequency)` is the C++ routine that Meep uses internally during FDTD time-stepping to evaluate $\chi_1^{-1}$ at each grid point—testing it directly confirms the bridge between material model and solver.
- `sim.get_epsilon(frequency=f, snap=True)` returns the permittivity at frequency `f` on the Yee grid; `snap=True` rounds the evaluation point to the nearest grid cell.
- `mp.output_epsilon(sim, frequency=f)` writes the complex permittivity at frequency `f` to HDF5; the `eps.r` and `eps.i` datasets hold the real and imaginary parts.
- `material.rotate(axis, angle)` applies a rotation to all susceptibility tensors, enabling simulation of crystals cut at arbitrary angles—essential for nonlinear optical and electro-optic device modeling.

---

*End of Chapter 1: Waveguides and Circuits*
