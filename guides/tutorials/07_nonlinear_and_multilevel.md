# Chapter 7: Nonlinear Optics and Multi-Level Atoms

This chapter covers nonlinear optical simulations and quantum emitter modeling in Meep, including third-harmonic generation from Kerr media, saturable gain lasers based on multi-level atomic rate equations, stochastic dipole sources for incoherent emission (including reciprocity-based acceleration), and chirped-pulse propagation using user-defined source waveforms. Together these tutorials demonstrate how Meep moves beyond linear, passive media to simulate light-matter interactions where field intensity, population dynamics, and source statistics all play central roles.

---

### 1. `3rd-harm-1d.py` — Third-Harmonic Generation in a Kerr Medium

**Physics:** A plane wave at frequency omega propagates through a medium with a cubic (chi^(3)) nonlinearity, converting a fraction of its power to the third-harmonic frequency 3*omega via a four-wave mixing process.
**Difficulty:** Intermediate
**Source:** `python/examples/3rd-harm-1d.py`
**Test Status:** PASS (35.7 s)

#### Theory

In a medium with an instantaneous Kerr nonlinearity, the electric polarization density acquires a term cubic in the electric field:

    P = epsilon_0 * (chi^(1) * E + chi^(3) * E^3)

The E^3 term is the origin of third-order nonlinear optical effects. When the field is a monochromatic wave at frequency omega, the identity cos^3(omega*t) = (3/4)*cos(omega*t) + (1/4)*cos(3*omega*t) shows that the cubic polarization contains components at both the fundamental omega and at the third harmonic 3*omega. The component at 3*omega acts as a source for a new electromagnetic wave at that frequency — a process called third-harmonic generation (THG).

The efficiency of THG depends critically on phase matching: both the fundamental and third-harmonic waves must travel at the same phase velocity for coherent build-up. In a dispersion-free linear medium (n constant), perfect phase matching holds automatically and the third-harmonic power grows as the square of the propagation length. In practice, material dispersion breaks phase matching, and THG efficiency oscillates with a characteristic coherence length. In the simulation here, the medium is dispersion-free (a flat index n=1 with chi^(3) added), so phase matching is perfect, and we expect the third-harmonic power to grow quadratically with both propagation length and chi^(3).

For weak nonlinearity (the perturbative regime), energy conservation and coupled-mode theory predict that the transmitted power at 3*omega scales as:

    P(3*omega) ~ (chi^(3))^2 * P(omega)^3 * L^2

where L is the medium length. Equivalently, at fixed amplitude the third-harmonic transmittance (normalized to the incident fundamental power) scales quadratically with chi^(3). This quadratic scaling is one of the key checks the example script verifies.

The simulation is one-dimensional: a GaussianSource at the left boundary drives a broadband pulse through a chi^(3) medium that fills the entire cell. PML layers absorb outgoing waves at both ends, and a flux monitor at the right boundary records the transmitted power spectrum. The broadband pulse excites both omega and 3*omega simultaneously, so a single run captures the full picture. A separate narrow-band mode (flux_spectrum=False) measures only the two harmonic frequencies for the scaling study.

#### Code Walkthrough

The simulation domain is a 1D cell of size sz=100 in the z-direction:

```python
dimensions = 1
cell = mp.Vector3(0, 0, sz)
default_material = mp.Medium(index=1, chi3=k)
```

Setting `dimensions=1` tells Meep to use the genuinely one-dimensional solver, which is faster than a 3D run with two no-size directions. The entire cell is filled with the nonlinear medium by passing it as `default_material` — no geometry list is needed.

The source is a Gaussian pulse centered at fcen = 1/3 (in Meep natural units c/a) with fractional bandwidth df/fcen = 1/20:

```python
fcen = 1 / 3.0
df = fcen / 20.0
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ex,
                     center=mp.Vector3(0, 0, -0.5*sz + dpml),
                     amplitude=amp)]
```

Choosing fcen = 1/3 is deliberate: the third harmonic falls at 3*fcen = 1, which is well within the bandwidth of the subsequent flux measurement (fmin = fcen/2, fmax = 4*fcen). In 1D, the only valid field components are Ex (and Hy), so Ex is the natural choice.

Two flux monitors are used depending on the run mode. In the broadband mode, a single monitor covers the full frequency range:

```python
trans = sim.add_flux(0.5*(fmin+fmax), fmax-fmin, nfreq, mp.FluxRegion(mon_pt))
```

In the harmonic-pair mode, two single-frequency monitors isolate omega and 3*omega:

```python
trans1 = sim.add_flux(fcen, 0, 1, mp.FluxRegion(mon_pt))
trans3 = sim.add_flux(3*fcen, 0, 1, mp.FluxRegion(mon_pt))
```

The simulation runs until the fields have decayed by six orders of magnitude relative to their peak, ensuring the DFT accumulators have captured the complete waveform:

```python
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mon_pt, 1e-6))
```

In the `__main__` block, Part 1 sweeps chi^(3) over four decades (0.001 to 1) and plots the transmitted power spectrum on a log-y scale. The third-harmonic peak at frequency 1.0 emerges clearly and grows with chi^(3). Part 2 performs the scaling study: for chi^(3) ranging from 10^-6 to 1, it records the fundamental and third-harmonic transmitted power and overlays a reference line with quadratic slope:

```python
ax.loglog(10**logk, third_order / input_flux, "bo-", label=r"$3\omega$")
ax.loglog(10**logk, (10**logk)**2, "k-", label="quadratic line")
```

Agreement with the quadratic reference line in the perturbative regime (small chi^(3)) and saturation at large chi^(3) are both visible.

#### Key Takeaways

- Setting `chi3=k` in `mp.Medium` enables the instantaneous Kerr nonlinearity; Meep evaluates P = chi^(3) * |E|^2 * E at every grid point every time step.
- Using `dimensions=1` with a `default_material` is the most efficient way to simulate a uniform nonlinear medium: no geometry list, no 2D/3D overhead.
- A GaussianSource with center frequency fcen=1/3 naturally places the third harmonic 3*fcen=1 inside the flux monitor range.
- In the perturbative (weak nonlinearity) regime, third-harmonic power scales as (chi^(3))^2, confirming the coupled-mode theory prediction.
- The `stop_when_fields_decayed` termination criterion ensures the DFT flux is fully converged even for long ring-down tails.

---

### 2. `multilevel-atom.py` — One-Sided Fabry-Perot Laser with a Two-Level Gain Medium

**Physics:** A 1D cavity filled with a two-level gain medium (modeled by Meep's multi-level atom susceptibility) is pumped above threshold and lases, producing a self-consistent FDTD laser simulation based on Maxwell-Bloch equations.
**Difficulty:** Advanced
**Source:** `python/examples/multilevel-atom.py`
**Test Status:** TIMEOUT (CPU-intensive; long time simulation to reach steady-state lasing at t=7000)

#### Theory

The Maxwell-Bloch equations couple Maxwell's equations for the electromagnetic field to the quantum-mechanical Bloch equations for an ensemble of two-level atoms. In the simplest picture, each atom has a ground state |1> and an excited state |2> separated by energy hbar*omega_a. The atomic polarization P drives the field, and the field drives transitions between levels.

The gain medium in Meep is implemented as a generalized multi-level susceptibility. For a two-level system the relevant quantities are:

- omega_a: the transition angular frequency (in units of 2*pi*c/a in Meep)
- gamma_perp: the polarization dephasing rate (HWHM of the Lorentzian linewidth)
- sigma_21 = 2*theta^2*omega_a / hbar: the classical coupling strength, where theta is the off-diagonal dipole matrix element
- rate_21: the spontaneous emission / non-radiative decay rate from level 2 to level 1
- Rp: the external pump rate from level 1 to level 2
- N0: the initial ground-state population density

The steady-state population inversion D_0 = N1 - N2 (in SALT/coupled-mode theory convention) determines whether the medium provides gain or absorption. Above the lasing threshold, D_0 is negative (net inversion) and the cavity field builds up from noise (here, a seed field) to a steady laser oscillation.

The simulation benchmarks against the Steady-state Ab-initio Laser Theory (SALT) of Cerjan et al. (Optics Express 20, 474, 2012), specifically their Fig. 2, which gives the lasing field amplitude as a function of pump rate for a one-sided Fabry-Perot cavity. The Meep approach — fully time-dependent, nonlinear FDTD — can capture transient dynamics, mode competition, and relaxation oscillations that SALT (a frequency-domain steady-state theory) cannot.

An important convention note embedded in the code comments: Meep uses SI-like units where D = eps_0*E + P, whereas many published SALT papers use Gaussian units where D = E + 4*pi*P. This introduces a factor of 4*pi difference in the coupling constant sigma. The code makes this conversion explicitly to allow direct comparison with SALT results.

The cavity geometry is a 1D cell: a dielectric slab of index ncav=1.5 and length Lcav=1 (the gain region) followed by a padding region and a PML on the high-z side only. The low-z boundary is a perfect mirror (no PML), creating the one-sided Fabry-Perot configuration.

#### Code Walkthrough

The atomic frequency and linewidth must be converted from SALT angular-frequency units to Meep's frequency units (cycles per unit time, i.e. 2*pi*c/a):

```python
omega_a = 40                          # angular frequency in SALT units
freq_21 = omega_a / (2 * math.pi)    # Meep frequency = omega / 2pi
gamma_perp = 4                        # HWHM in SALT
gamma_21 = (2 * gamma_perp) / (2 * math.pi)  # FWHM in Meep
```

The coupling strength uses the SALT definition with hbar=1:

```python
theta = 1
sigma_21 = 2 * theta * theta * omega_a
```

The two transitions are defined: one radiative (with pumping, frequency, linewidth, and coupling) and one non-radiative decay:

```python
transitions = [
    mp.Transition(1, 2, pumping_rate=Rp, frequency=freq_21,
                  gamma=gamma_21, sigma_diag=mp.Vector3(sigma_21, 0, 0)),
    mp.Transition(2, 1, transition_rate=rate_21),
]
ml_atom = mp.MultilevelAtom(sigma=1, transitions=transitions,
                             initial_populations=[N0])
two_level = mp.Medium(index=ncav, E_susceptibilities=[ml_atom])
```

`MultilevelAtom` is a subclass of `Susceptibility` and is passed as an element of `E_susceptibilities` in the medium. The `sigma=1` outer parameter scales all transition coupling strengths; the per-transition coupling is specified via `sigma_diag`. Setting `sigma_diag=mp.Vector3(sigma_21, 0, 0)` means only the x-component (Ex) couples to the atoms.

Only the x-polarization of the gain block sees the gain:

```python
geometry = [mp.Block(center=mp.Vector3(z=-0.5*sz + 0.5*Lcav),
                     size=mp.Vector3(mp.inf, mp.inf, Lcav),
                     material=two_level)]
```

The simulation is seeded with a single non-zero pixel of field to break symmetry and allow the laser to start up:

```python
def field_func(p):
    return 1 if p.z == -0.5*sz + 0.5*Lcav else 0

sim.fields.initialize_field(mp.Ex, field_func)
```

The field is then monitored at a point just outside the gain region during the final 250 time units of a 7000-unit run — long enough for the laser to reach steady state. The printed field values can be compared against SALT predictions.

#### Key Takeaways

- `mp.MultilevelAtom` implements the full time-domain Maxwell-Bloch equations; population dynamics, saturation, and field back-action are all self-consistent.
- Transitions are defined as pairs of levels: a `Transition(1, 2, pumping_rate=Rp, ...)` represents stimulated absorption/emission on the 1->2 transition with external pumping, while `Transition(2, 1, transition_rate=rate_21)` is a pure non-radiative decay.
- The gamma parameter in `Transition` is the FWHM linewidth in Meep frequency units — a factor of 2/(2*pi) different from the HWHM angular frequency gamma_perp used in SALT.
- The sigma_21 = 2*theta^2*omega_a coupling follows from the semiclassical dipole approximation; the factor of 4*pi difference between SI and Gaussian units must be handled when comparing with SALT literature.
- Long run times (endt=7000 at resolution=400 means ~5.6 million time steps) are typical for laser simulations that must wait for transient dynamics to settle.

---

### 3. `stochastic_emitter.py` — Incoherent Dipole Emission Above a Silver Substrate

**Physics:** An ensemble of incoherent point dipole emitters inside a high-index semiconductor substrate above a silver mirror are modeled using stochastic random-phase sources, computing the emitted flux spectrum by statistical averaging — a numerical realization of the fluctuation-dissipation theorem.
**Difficulty:** Advanced
**Source:** `python/examples/stochastic_emitter.py`
**Test Status:** TIMEOUT (CPU-intensive; many trials or many individual dipole runs required)

#### Theory

Incoherent light sources — such as LEDs, fluorescent emitters, or thermal radiation — consist of many dipoles oscillating with independent random phases. The total radiated power is the sum of contributions from each dipole, and the spectral density of the total emission is related to the local photonic density of states (LDOS) by the fluctuation-dissipation theorem. No cross-dipole interference terms survive the statistical average.

In principle, the exact way to simulate an ensemble of N incoherent dipoles is to run N separate FDTD simulations (one per dipole at its exact position) and sum the resulting fluxes. This is Method 2 in the code. However, for large N this becomes expensive.

Method 1 exploits a key statistical identity. Consider replacing the N individual dipoles with a single FDTD run in which all dipoles fire simultaneously, but with independent Gaussian random amplitudes at each time step (white noise sources). Because the noise is uncorrelated across dipoles, the time-averaged Poynting flux in such a run equals the sum of individual dipole contributions — but only after averaging over many noise realizations (trials). The number of trials needed for a given signal-to-noise ratio trades off against the number of separate dipole runs.

Formally, if each dipole current is J_n(t) = xi_n(t) * delta(r - r_n), where xi_n(t) are independent standard normal random variables, then:

    <P(omega)> = sum_n P_n(omega)

where P_n(omega) is the flux from dipole n alone and <...> denotes an average over noise realizations. The white-noise source `lambda t: np.random.randn()` implements xi_n(t) in Meep's `CustomSource`.

The physical structure is an LED-like device: a high-index (n=3.45) semiconductor substrate of thickness dsub=5 um, backed by a silver reflector (dAg=0.5 um), with an air gap and PML above. The dipoles are placed in the middle of the substrate along a horizontal line. An optional textured grating (a dielectric rod of width wrod=0.5 um and height hrod=0.7 um) on the substrate surface can enhance light extraction. The flux monitor at the top of the air region measures light escaping upward — the extraction efficiency.

#### Code Walkthrough

The geometry is built from three blocks: the substrate, the silver reflector, and (optionally) the grating rod:

```python
geometry = [
    mp.Block(material=mp.Medium(index=3.45),
             center=mp.Vector3(0, 0.5*sy - dpml - dair - hrod - 0.5*dsub),
             size=mp.Vector3(mp.inf, dsub, mp.inf)),
    mp.Block(material=Ag,
             center=mp.Vector3(0, -0.5*sy + 0.5*dAg),
             size=mp.Vector3(mp.inf, dAg, mp.inf)),
]
```

`Ag` is imported from `meep.materials` — a dispersive Drude-Lorentz fit to the measured silver permittivity.

In Method 1, all ndipole random sources fire in a single simulation run:

```python
sources = [
    mp.Source(mp.CustomSource(src_func=lambda t: np.random.randn()),
              component=mp.Ez,
              center=mp.Vector3(sx*(-0.5 + n/ndipole), -0.5*sy + dAg + 0.5*dsub))
    for n in range(ndipole)
]
```

Each dipole gets an independent `lambda t: np.random.randn()` — Python's closure creates a new lambda object for each iteration (though beware that in a loop all lambdas may share the same `n` unless the list comprehension evaluates immediately, which it does here). The run time is set long enough to resolve the spectral features:

```python
run_time = 2 * nfreq / df
```

This ensures the DFT accumulates for at least nfreq/df time units, giving frequency resolution df/nfreq.

In Method 2, each dipole is simulated individually using a deterministic Gaussian source, and the fluxes are summed:

```python
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(sx*(-0.5 + n/ndipole), ...))]
```

The computed fluxes are saved as NumPy arrays and can be post-processed to compare flat vs. textured substrate extraction efficiencies.

#### Key Takeaways

- Incoherent emission is simulated by summing fluxes from independent dipoles; the fluctuation-dissipation theorem guarantees this equals the stochastic-average result.
- `mp.CustomSource(src_func=lambda t: np.random.randn())` implements a white-noise current source that excites all frequencies uniformly.
- Method 1 (stochastic, one run per trial) vs. Method 2 (deterministic, one run per dipole) trade off computation time against statistical noise; for many dipoles and few trials Method 1 can be faster.
- `k_point=mp.Vector3()` with Bloch-periodic boundary conditions in x enforces the periodic geometry of the unit cell.
- The run time `2*nfreq/df` is a practical rule of thumb: at least two time-bandwidth products to ensure spectral leakage is negligible in the DFT flux.

---

### 4. `stochastic_emitter_line.py` — Line-Source Fourier Decomposition of Incoherent Emission

**Physics:** The same LED-like substrate structure is studied using a Fourier-series decomposition (Method 3): instead of point dipoles, the extended incoherent source is decomposed into cosine-mode line sources, each run separately, dramatically reducing the number of simulations required.
**Difficulty:** Advanced
**Source:** `python/examples/stochastic_emitter_line.py`
**Test Status:** TIMEOUT (CPU-intensive; requires many runs for all Fourier modes or all dipole positions)

#### Theory

A continuous incoherent source layer (e.g., the active region of a planar LED extending across the full width sx of the unit cell) emits from every point independently. The exact computation of the emitted spectrum requires running one FDTD simulation per dipole position — which is O(N) simulations, where N = sx * resolution is the number of grid points across the cell.

Method 3 exploits the fact that in a periodically-repeated unit cell, any line-source amplitude distribution can be expanded in a Fourier cosine series:

    f(x) = a_0 / sqrt(sx)  +  sum_{n=1}^{N} a_n * sqrt(2/sx) * cos(n*pi*(x + sx/2) / sx)

where the basis functions are orthonormal on [-sx/2, sx/2]. The incoherent flux from the full source layer equals the sum of fluxes from each Fourier mode run separately (because the modes are orthogonal, cross terms vanish when integrated across x). This reduces the O(sx*resolution) dipole runs to O(nsrc) mode runs, where nsrc is the number of Fourier terms needed to converge.

The amplitude function for mode n is:

    a_0: 1/sqrt(sx)          (uniform, DC mode)
    a_n: sqrt(2/sx) * cos(n*pi*(x + sx/2)/sx)   for n >= 1

These are implemented via Meep's `amp_func` parameter, which accepts a callable taking a `Vector3` position and returning the source amplitude at that point:

```python
def src_amp_func(n):
    def _src_amp_func(p):
        if n == 0:
            return 1 / np.sqrt(sx)
        else:
            return np.sqrt(2/sx) * np.cos(n*np.pi*(p.x + 0.5*sx)/sx)
    return _src_amp_func
```

The closure captures n correctly. Each mode is a line source (size=mp.Vector3(sx, 0)) at the same vertical position in the substrate.

The total incoherent flux is then reconstructed as the sum of per-mode fluxes. Convergence can be checked by increasing nsrc. Comparison with Method 2 (individual dipoles summed) serves as a cross-validation.

#### Code Walkthrough

The structure geometry is essentially identical to `stochastic_emitter.py`; the key difference is the source construction. For Method 2 (dipole-by-dipole), each source is a point source at a specific x-position:

```python
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(sx*(-0.5 + n/ndipole),
                                       -0.5*sy + dAg + 0.5*dsub))]
```

For Method 3, each source is a line source with a spatially-varying amplitude:

```python
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(0, -0.5*sy + dAg + 0.5*dsub),
                     size=mp.Vector3(sx, 0),
                     amp_func=src_amp_func(n))]
```

The number of dipoles in Method 2 is determined by the grid: `ndipole = int(sx * resolution)`, which equals 75 for sx=1.5 and resolution=50. The number of Fourier modes (default nsrc=15) is much smaller, giving a speedup factor approaching 5x in this configuration — larger in higher-resolution or wider-cell cases.

The simulation loop for Method 3:

```python
fluxes = np.zeros((nfreq, nsrc))
for d in range(nsrc):
    freqs, fluxes[:, d] = compute_flux(3, d)
```

After all modes are computed, `fluxes.sum(axis=1)` gives the total incoherent spectrum as a function of frequency. Results are saved to a .npz file with descriptive naming that encodes all parameters.

#### Key Takeaways

- Fourier decomposition of a spatially-extended incoherent source replaces O(N) dipole runs with O(nsrc) mode runs, where nsrc grows only logarithmically with required accuracy for smooth source distributions.
- `amp_func` accepts any callable `f(p: Vector3) -> complex`, making it possible to implement arbitrary spatial amplitude profiles including Fourier modes, Gaussian beams, or numerically-defined modal profiles.
- The orthonormality of cosine modes on the unit cell guarantees that the sum of per-mode fluxes equals the total incoherent flux from the extended source.
- Combining k_point=mp.Vector3() (periodic boundaries) with a spatially-varying line source is the canonical approach to modeling periodically-tiled LED structures.
- Methods 2 and 3 produce the same physical result but differ in computational cost; cross-comparing them validates both implementations.

---

### 5. `stochastic_emitter_reciprocity.py` — Reciprocity-Based Acceleration of Dipole Emission Calculations

**Physics:** The optical reciprocity theorem is used to compute the emission spectrum of an ensemble of incoherent dipoles in a single backward simulation (a plane wave incident from above), instead of running one forward simulation per dipole — reducing O(N_dipole) runs to O(1).
**Difficulty:** Advanced
**Source:** `python/examples/stochastic_emitter_reciprocity.py`
**Test Status:** TIMEOUT (CPU-intensive; multiple forward runs plus backward run, with high resolution=200)

#### Theory

The reciprocity theorem in electromagnetism states that if source J_a at position r_a produces field E_a(r_b) at position r_b, then an identical source J_b at r_b produces the same field component at r_a: the Green's tensor G(r_a, r_b) = G(r_b, r_a)^T (transposed, in tensorial form). This symmetry has profound computational consequences for problems involving many source positions.

Consider computing the power extracted into a specific radiation channel (e.g., an upward-propagating plane wave) from each of N dipoles embedded at different positions r_n in the substrate. The forward approach requires N FDTD simulations. The reciprocal approach requires only one: inject a plane wave from above (the time-reversed version of the desired collection mode) and record the field amplitude at each dipole position r_n. By reciprocity, this amplitude squared is proportional to the power that dipole n would emit into that channel.

Mathematically, if E_bwd(r_n, omega) is the DFT of the backward (reverse-propagating) field at dipole position r_n, then:

    P_n(omega) = C(omega) * |E_bwd(r_n, omega)|^2

where C(omega) is a known normalization constant. The total emission from N incoherent dipoles is then:

    P_total(omega) = C(omega) * sum_n |E_bwd(r_n, omega)|^2

This is the Wiener-Khintchine theorem applied to spatially-distributed sources: the total power spectrum equals the sum of per-dipole spectral densities, which are proportional to the squared field amplitude of the reciprocal simulation at each dipole location.

The code computes both the forward result (averaging over ndipole=10 individual dipole runs) and the backward result (a single plane-wave run), then compares the normalized flux enhancement (textured / flat) between the two approaches as a validation.

#### Code Walkthrough

The structure is defined through a helper function to avoid code duplication between forward and backward runs:

```python
def substrate_geometry(is_textured: bool):
    geometry = [
        mp.Block(material=mp.Medium(index=3.45), ...),   # substrate
        mp.Block(material=Ag, ...),                       # silver reflector
    ]
    if is_textured:
        geometry.append(mp.Block(...))  # grating rod
    return geometry
```

The forward simulation places a single point dipole source and measures eigenmode flux (the coefficient of the upward-propagating mode):

```python
res = sim.get_eigenmode_coefficients(flux_mon, [1], eig_parity=mp.ODD_Z)
flux = np.abs(res.alpha[0, :, 0])**2
```

Using `get_eigenmode_coefficients` isolates the unidirectional flux into mode 1 (the zeroth-order plane wave in air), avoiding double-counting of forward and backward components.

The backward simulation replaces the dipole with a downward-propagating plane wave (a line source at the top of the air region):

```python
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(0, 0.5*sy - dpml),
                     size=mp.Vector3(sx, 0))]
```

A DFT field monitor records E_z across the dipole layer:

```python
dft_mon = sim.add_dft_fields([mp.Ez], fcen, df, nfreq,
                              center=mp.Vector3(0, -0.5*sy + dAg + 0.5*dsub),
                              size=mp.Vector3(sx))
```

The backward flux proxy is computed as the spatial integral of |E_z|^2:

```python
for nf in range(nfreq):
    dft_ez = sim.get_dft_array(dft_mon, mp.Ez, nf)
    abs_flux[nf] = np.sum(np.abs(dft_ez)**2)
```

The final plot compares the normalized flux ratio (textured / flat) from forward and backward methods. Agreement confirms the reciprocity approach.

#### Key Takeaways

- Optical reciprocity reduces the computation of N-dipole emission patterns from O(N) to O(1) backward simulations — a major speedup when N is large.
- `sim.get_eigenmode_coefficients` in the forward run extracts the directional flux into a specific radiation mode, which is the correct quantity to compare with the backward field amplitude squared.
- `sim.add_dft_fields` with a spatial extent records the full complex E_z profile across the dipole layer at each DFT frequency, enabling the reciprocity integral.
- The normalization factor C(omega) cancels in the ratio (textured / flat), making the comparison robust to absolute calibration.
- The reciprocity approach is especially powerful for LED-like problems where one wants to optimize the structure for a fixed collection geometry but many emitter positions.

---

### 6. `chirped_pulse.py` — Propagation of a Linearly Chirped Pulse

**Physics:** A linearly frequency-chirped Gaussian pulse is injected as a plane wave into a vacuum cell and its spatiotemporal field pattern is recorded, demonstrating the use of Meep's CustomSource for user-defined waveforms.
**Difficulty:** Beginner
**Source:** `python/examples/chirped_pulse.py`
**Test Status:** PASS (106.6 s)

#### Theory

A chirped pulse is one whose instantaneous frequency varies with time. A linearly chirped Gaussian pulse has the analytic form:

    E(t) = exp(-a*(t-t0)^2) * exp(i*(2*pi*v0*(t-t0) + b*(t-t0)^2))

where:
- v0 is the carrier frequency (center frequency at t=t0)
- a is the Gaussian envelope half-width (controls pulse duration)
- b is the chirp rate (rad/time^2)
- t0 is the time of peak amplitude

The instantaneous angular frequency is the time derivative of the phase:

    omega_inst(t) = 2*pi*v0 + 2*b*(t - t0)

When b < 0 (down-chirp), the instantaneous frequency decreases with time: higher frequencies arrive first. When b > 0 (up-chirp), lower frequencies arrive first. The Wigner-Ville time-frequency distribution of this signal is a tilted Gaussian ridge in the (t, omega) plane, a clean representation of the linear chirp.

The Fourier transform of this pulse is also a Gaussian, centered at v0 with an increased bandwidth compared to a transform-limited pulse of the same duration. The time-bandwidth product of a chirped pulse exceeds the minimum value of 1/(4*pi) for a Gaussian.

This simulation is fundamentally a propagation experiment: the chirped pulse travels through vacuum (no material, no dispersion) so its shape does not change during propagation. The educational purpose is to demonstrate: (1) how to implement an arbitrary time-domain waveform using CustomSource, (2) how to use periodic boundaries for a plane-wave source, and (3) how to output snapshots of the field at regular intervals for visualization.

In dispersive media (not present here but a natural extension), a chirped pulse would compress or stretch as different frequency components travel at different phase velocities — a phenomenon central to dispersion compensation in optical fiber communications and ultrafast optics.

#### Code Walkthrough

The chirped waveform is defined as a Python lambda:

```python
v0 = 1.0   # carrier frequency
a = 0.2    # envelope half-width
b = -0.5   # chirp rate (negative = down-chirp)
t0 = 15    # peak time

chirp = lambda t: (np.exp(1j * 2*np.pi*v0*(t-t0)) *
                   np.exp(-a*(t-t0)**2 + 1j*b*(t-t0)**2))
```

The two exponential factors implement the envelope and the phase separately, making the chirp rate b easy to read and modify. The imaginary exponent `1j*b*(t-t0)^2` is the quadratic phase that produces the linear frequency sweep.

The source is a line source spanning the full y-extent of the cell, simulating a plane wave propagating in +x:

```python
sources = [mp.Source(src=mp.CustomSource(src_func=chirp),
                     center=mp.Vector3(-0.5*sx),
                     size=mp.Vector3(y=sy),
                     component=mp.Ez)]
```

Periodic boundary conditions in y (enabled by `k_point=mp.Vector3()`) and a y-mirror symmetry (enabled by `mp.Mirror(mp.Y)`) make this a true plane wave — no spurious diffraction from source edges.

PML layers absorb the wave at the x-boundaries:

```python
dpml = 2
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]
```

The field is output every 2.7 time units within the central non-PML volume:

```python
sim.run(
    mp.in_volume(mp.Volume(center=mp.Vector3(), size=mp.Vector3(sx, sy)),
                 mp.at_every(2.7, mp.output_efield_z)),
    until=t0 + 50,
)
```

`mp.in_volume` restricts output to the specified region, avoiding writing PML data. The 2.7-unit interval is chosen to produce about 24 frames over the simulation, sufficient to show the pulse traveling across the 40-unit cell. The output HDF5 files can be converted to images with `h5topng` or read directly with `h5py`.

#### Key Takeaways

- `mp.CustomSource(src_func=f)` accepts any callable `f(t) -> complex`, making it straightforward to inject chirped pulses, frequency-modulated waveforms, or tabulated experimental data.
- The combination of `k_point=mp.Vector3()` and `mp.Mirror(mp.Y)` efficiently simulates a plane wave in a 2D periodic cell with half the computational cost.
- Separating envelope and phase in the lambda (`exp(-a*(t-t0)^2) * exp(1j*phase(t))`) is a best practice for clarity and ease of modification.
- `mp.at_every(dt, mp.output_efield_z)` combined with `mp.in_volume` is the standard pattern for recording spatiotemporal field movies for visualization.
- The run time `t0 + 50` ensures the pulse has fully traversed the 40-unit cell (travel time = 40 at c=1) with extra time for the tail to clear.

---

### 7. `test_3rd_harm_1d.py` — Regression Test for Third-Harmonic Generation

**Physics:** Automated unit test that verifies the 1D Kerr THG simulation produces numerically exact values for the transmitted flux at both the fundamental and third-harmonic frequencies.
**Difficulty:** Beginner
**Source:** `python/tests/test_3rd_harm_1d.py`
**Test Status:** PASS (4.3 s)

#### Theory

This test validates the correctness of Meep's chi^(3) nonlinearity implementation by comparing simulation output against a fixed reference value derived from a previous validated run. The expected values are:

    k = 0.01         (chi^(3) strength)
    amp = 1.0        (source amplitude)
    flux at omega:   221.89548712071553
    flux at 3*omega: 1.752960413399477

These numbers encode several physical constraints. First, the ratio of the two fluxes (1.75 / 221.9 ~ 0.0079) should be consistent with the perturbative THG scaling P(3*omega) ~ (chi^(3))^2 ~ (0.01)^2 = 10^-4 times the fundamental power — but note that the sz=100 propagation distance, Gaussian pulse, and finite bandwidth all affect the absolute numbers. What matters for the regression test is exact reproducibility to high precision (tolerance 1e-7 for double precision, 3e-5 for single).

The test uses `ApproxComparisonTestCase.assertClose`, a relative comparison that accounts for both double- and single-precision builds. The single-precision tolerance (3e-5) is dramatically wider than the double-precision tolerance (1e-7), reflecting the inherent precision limits of 32-bit floating-point arithmetic in the FDTD update equations.

The test differs from the example script in several ways: it uses a lower resolution (20 vs 25), a decimation_factor=1 in the flux monitors (forcing accumulation at every time step rather than every other step), and it monitors both the broadband spectrum and the two harmonic frequencies in a single simulation run.

#### Code Walkthrough

The test class inherits from `ApproxComparisonTestCase` (from `python/tests/utils.py`) to gain access to precision-aware comparison methods:

```python
class Test3rdHarm1d(ApproxComparisonTestCase):
    def setUp(self):
        ...
        default_material = mp.Medium(index=1, chi3=self.k)
```

Three flux monitors are added to the same simulation: one broadband and two single-frequency:

```python
self.trans  = self.sim.add_flux(0.5*(fmin+fmax), fmax-fmin, nfreq, fr,
                                 decimation_factor=1)
self.trans1 = self.sim.add_flux(fcen,   0, 1, fr, decimation_factor=1)
self.trans3 = self.sim.add_flux(3*fcen, 0, 1, fr, decimation_factor=1)
```

The `decimation_factor=1` ensures the DFT fields are accumulated at every time step — important for high-frequency components (3*fcen = 1.0) where undersampling could introduce aliasing.

The expected reference values are hardcoded:

```python
expected_harmonics = [0.01, 1.0, 221.89548712071553, 1.752960413399477]
```

After the run, the computed values are assembled and compared:

```python
harmonics = [self.k, self.amp,
             mp.get_fluxes(self.trans1)[0],
             mp.get_fluxes(self.trans3)[0]]
tol = 3e-5 if mp.is_single_precision() else 1e-7
self.assertClose(expected_harmonics, harmonics, epsilon=tol)
```

Including `self.k` and `self.amp` in the comparison list is a sanity check that the test parameters themselves have not changed.

#### Key Takeaways

- Regression tests for nonlinear simulations should check absolute flux values, not just qualitative behavior, because subtle bugs in nonlinear update equations can shift amplitudes without changing the qualitative spectrum shape.
- `decimation_factor=1` in `add_flux` is important when monitoring high-frequency components: the default decimation may skip time steps and alias frequencies above the Nyquist limit of the decimated sampling rate.
- `mp.is_single_precision()` allows a single test to work correctly in both single- and double-precision builds with appropriate tolerances.
- The tolerance 1e-7 for double precision reflects that the FDTD solution should be deterministic and reproducible at machine-epsilon precision given identical code, parameters, and hardware (or CPU architecture).
- Using `setUp` to construct the simulation object and `test_*` methods to run it follows the standard Python `unittest` pattern, allowing the test infrastructure to run setUp fresh for each test method.

---

### 8. `test_multilevel_atom.py` — Regression Test for the Multi-Level Atom Laser

**Physics:** Automated unit test that verifies the multi-level atom laser simulation reaches the correct steady-state field amplitude after a long time integration, confirming the Maxwell-Bloch implementation is self-consistent and reproducible.
**Difficulty:** Intermediate
**Source:** `python/tests/test_multilevel_atom.py`
**Test Status:** PASS (33.1 s)

#### Theory

Testing a laser simulation is fundamentally different from testing a passive linear system. In a laser, the steady-state field amplitude is not set by the input (there is no external drive — only a seed), but by a balance between gain, loss, and gain saturation. The steady-state field therefore depends nonlinearly on all parameters: the pump rate Rp, decay rate rate_21, cavity length Lcav, coupling sigma_21, and the cavity Q-factor (determined by the PML).

For a two-level gain medium in a one-sided Fabry-Perot cavity, SALT predicts a specific lasing threshold pump rate Rp_th and a specific output field amplitude for Rp > Rp_th. The test uses N0=28 (reduced from the example's N0=37 to keep the test faster) and Rp=0.0051 — slightly above threshold — where the field settles to a well-defined amplitude.

The expected value fp = -2.7110969214986387 is the Ex field at a specific monitoring point (just outside the gain region) at the end of the simulation (t=7000). The negative sign reflects the phase of the standing wave at that point. This is a double-precision-specific test (skipped for single precision builds), because the long time integration amplifies floating-point differences to the point where single-precision results are not reproducible to the required 10^-10 accuracy of `assertAlmostEqual`.

The test provides mathematical validation that:
1. The Maxwell-Bloch time-stepping is correctly implemented (the laser actually lases at a well-defined amplitude)
2. The result is bit-reproducible across compiler versions, platforms, and Meep updates
3. The multi-level atom infrastructure (population arrays, stimulated emission updates, polarization coupling) all work together correctly

#### Code Walkthrough

The test is skipped automatically for single-precision builds:

```python
@unittest.skipIf(mp.is_single_precision(),
                 "double-precision floating point specific test")
def test_multilevel_atom(self):
```

The gain medium is constructed identically to the example, but with N0=28:

```python
t1 = mp.Transition(1, 2, pumping_rate=Rp, frequency=freq_21,
                   gamma=gamma_21, sigma_diag=mp.Vector3(sigma_21, sigma_21, sigma_21))
t2 = mp.Transition(2, 1, transition_rate=rate_21)
ml_atom = mp.MultilevelAtom(sigma=1, transitions=[t1, t2],
                             initial_populations=[N0])
two_level = mp.Medium(index=ncav, E_susceptibilities=[ml_atom])
```

Note that the test uses `sigma_diag=mp.Vector3(sigma_21, sigma_21, sigma_21)` (isotropic coupling to all three field components) whereas the example uses `mp.Vector3(sigma_21, 0, 0)` (x-only coupling). In a 1D simulation with `dimensions=1`, only Ex is active, so this difference has no effect on the physics.

Field initialization seeds the laser:

```python
sim.init_sim()
sim.initialize_field(mp.Ex, field_func)
```

Note that the test uses the Python-level `sim.initialize_field` method, while the example uses the lower-level `sim.fields.initialize_field`. Both call the same underlying C++ function.

The check is performed at the end of the run:

```python
def check_field(sim):
    fp = sim.get_field_point(mp.Ex,
                             mp.Vector3(z=(-0.5*sz) + Lcav + (0.5*dpad))).real
    self.assertAlmostEqual(fp, -2.7110969214986387)

sim.run(mp.at_end(check_field), until=7000)
```

`mp.at_end(f)` is equivalent to `mp.at_every(inf, f)` — it calls f only at the final time step. `assertAlmostEqual` by default checks to 7 decimal places (roughly single-precision accuracy), which is appropriate even though this is a double-precision test, because 7000 time steps at resolution 40 accumulates rounding error that limits absolute accuracy.

#### Key Takeaways

- Long-duration laser simulations (until=7000) test not just the correctness of a single time step but the long-term stability and accuracy of the nonlinear coupled Maxwell-Bloch integrator.
- The `@unittest.skipIf(mp.is_single_precision(), ...)` decorator is the standard pattern for double-precision-only tests in the Meep test suite; it avoids false failures on single-precision builds without removing the test entirely.
- `mp.at_end(check_field)` is the idiomatic way to check final-state values in a `sim.run()` call, as it avoids storing the full field history.
- The specific expected value (-2.7110969214986387) is a fixed point of the laser dynamics at these parameters; changing N0, Rp, or rate_21 even slightly will shift this value, making the test sensitive to parameter changes.
- The difference in sigma_diag between the example (x-only) and the test (isotropic) highlights that Meep's MultilevelAtom fully supports anisotropic dipole coupling via the per-polarization sigma_diag Vector3 parameter.

---

## Summary

This chapter demonstrated Meep's capabilities for simulating a broad range of light-matter interaction phenomena beyond linear passive optics:

| Tutorial | Key Meep Feature | Physical Phenomenon |
|---|---|---|
| `3rd-harm-1d.py` | `mp.Medium(chi3=k)` | Third-harmonic generation, chi^(3) scaling |
| `multilevel-atom.py` | `mp.MultilevelAtom`, `mp.Transition` | Maxwell-Bloch lasing, gain saturation |
| `stochastic_emitter.py` | `mp.CustomSource(lambda t: randn())` | Incoherent dipole emission, LDOS |
| `stochastic_emitter_line.py` | `Source(amp_func=...)` | Fourier-mode acceleration of LED simulation |
| `stochastic_emitter_reciprocity.py` | `add_dft_fields`, reciprocity | N-dipole emission in O(1) simulations |
| `chirped_pulse.py` | `mp.CustomSource(chirp_lambda)` | Chirped pulse propagation, time-frequency |
| `test_3rd_harm_1d.py` | `decimation_factor`, `assertClose` | Regression testing nonlinear simulations |
| `test_multilevel_atom.py` | `mp.at_end`, `skipIf` | Regression testing laser steady state |

The common threads connecting these tutorials are: (1) moving beyond the `mp.Medium` linear parameters to access nonlinear (chi^(3)) and quantum (MultilevelAtom) material responses; (2) using `CustomSource` for arbitrary time-domain waveforms; and (3) applying statistical and reciprocity arguments to reduce the computational cost of multi-emitter problems from O(N) to O(1) simulations.
