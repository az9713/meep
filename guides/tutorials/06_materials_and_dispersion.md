# Chapter 6: Materials and Dispersion

This chapter covers material modeling in Meep, from the foundational Lorentzian and Drude susceptibility models that capture frequency-dependent dielectric response, to advanced topics including magneto-optical Faraday rotation, conductivity-based loss, and oblique-incidence reflectance. Each tutorial section connects the underlying electromagnetic theory to the specific Meep API calls used to realize that physics in a finite-difference time-domain simulation.

---

### 1. `material-dispersion.py` — Dispersion Relation in a Lorentzian Medium

**Physics:** Computes the photonic dispersion relation omega(k) for a homogeneous dispersive medium by sweeping the Bloch wavevector, revealing the polaritonic gap and absorption features produced by two Lorentzian resonances.
**Difficulty:** Intermediate
**Source:** `python/examples/material-dispersion.py`
**Test Status:** TIMEOUT (CPU-intensive — sweeps 101 k-points each requiring a time-domain run)

#### Theory

A dispersive dielectric medium responds to electric fields not instantaneously but according to a frequency-dependent permittivity epsilon(omega). The Lorentzian oscillator model treats bound electrons as damped harmonic oscillators driven by the electromagnetic field. For a single Lorentzian resonance, the contribution to the susceptibility is:

```
chi(omega) = sigma * omega_0^2 / (omega_0^2 - omega^2 - i*omega*gamma)
```

where `omega_0` is the resonance frequency, `gamma` is the damping rate (half-width at half-maximum of the absorption peak), and `sigma` is a dimensionless strength parameter. The full permittivity including a background (instantaneous) response `epsilon_inf` is:

```
epsilon(omega) = epsilon_inf + sum_n [ sigma_n * omega_n^2 / (omega_n^2 - omega^2 - i*omega*gamma_n) ]
```

The dispersion relation for plane waves in such a medium is `k = omega * sqrt(epsilon(omega)) / c`. When plotted as omega vs. k, the result departs dramatically from the simple linear relationship of free space. Near a strong resonance, the photon couples to the medium polarization to form a phonon-polariton hybrid: the dispersion splits into two branches (upper and lower polariton), separated by a stop band (polaritonic gap) where propagating modes are forbidden.

A weak, lossy resonance (large gamma) does not produce a stop band but introduces absorption, visible as complex frequencies when Meep's harmonic inversion extracts modes. The imaginary part of the frequency is proportional to the material loss at that frequency.

Meep simulates dispersion by introducing auxiliary polarization fields P_n that obey the oscillator equation of motion, updated at each time step alongside the Maxwell equations. The relationship `epsilon(omega) = (ck/omega)^2` lets us recover the permittivity from the numerically computed dispersion data.

The script uses `run_k_points`, which steps through a list of Bloch wavevectors and at each point runs a broadband Gaussian pulse, then applies the Harminv algorithm to extract resonant frequencies. This is identical in spirit to a band structure calculation for a photonic crystal, applied here to a structurally trivial (zero-cell) uniform medium.

#### Code Walkthrough

The medium is defined with two Lorentzian terms. The first is a strong resonance at omega = 1.1 with nearly zero damping — this is the resonance that opens the polaritonic gap. The second is weak but heavily damped, representing an absorption feature near omega = 0.5.

```python
susceptibilities = [
    mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5),
    mp.LorentzianSusceptibility(frequency=0.5, gamma=0.1, sigma=2e-5),
]
default_material = mp.Medium(epsilon=2.25, E_susceptibilities=susceptibilities)
```

The cell size is set to zero (an empty `mp.Vector3()`) because a zero-size cell means there is no spatial structure — it is a purely homogeneous simulation. Meep handles this as a single-point calculation.

```python
cell = mp.Vector3()
```

The wavevector sweep covers k from 0.3 to 2.2 in Meep units. The function `mp.interpolate` generates 101 equally spaced k-points between the two endpoints.

```python
kpts = mp.interpolate(k_interp, [mp.Vector3(kmin), mp.Vector3(kmax)])
all_freqs = sim.run_k_points(200, kpts)
```

The result `all_freqs` is a list of lists of complex frequencies — one list per k-point, potentially multiple modes. The permittivity is recovered by the relation `epsilon = (ck/omega)^2`:

```python
for fs, kx in zip(all_freqs, [v.x for v in kpts]):
    for f in fs:
        print(f"eps:, {f.real:.6g}, {f.imag:.6g}, {(kx / f) ** 2:.6g}")
```

Plotting these values recovers the full complex epsilon(omega), which should match the analytic Lorentzian formula. The real part exhibits the characteristic S-shaped curve near each resonance (normal dispersion below, anomalous dispersion above), and the stop-band region where Re(epsilon) < 0 between the two polariton branches.

#### Key Takeaways

- `mp.LorentzianSusceptibility(frequency, gamma, sigma)` defines a single Lorentzian pole; multiple poles are summed.
- A zero-size cell (`mp.Vector3()`) means a structurally homogeneous simulation — the geometry is all material.
- `run_k_points` efficiently sweeps the Bloch wavevector to trace out the dispersion relation.
- Small gamma (near-zero damping) creates polaritonic stop bands; large gamma smears them into absorption regions.
- Recovering epsilon as `(ck/omega)^2` from dispersion data is a direct validation of the dispersive material implementation.

---

### 2. `eps_fit_lorentzian.py` — Fitting Experimental Data to a Drude-Lorentzian Model

**Physics:** Fits a complex refractive index spectrum (measured data) to a sum of Lorentzian and Drude poles using gradient-based nonlinear optimization, producing a `Medium` object usable in Meep simulations.
**Difficulty:** Advanced
**Source:** `python/examples/eps_fit_lorentzian.py`
**Test Status:** FAIL (requires `mymaterial.csv` data file not included in repository)

#### Theory

Meep requires material dispersion to be represented in the Drude-Lorentzian form because this form is causal (it satisfies the Kramers-Kronig relations) and can be time-stepped efficiently using auxiliary differential equations. However, experimental optical data is typically tabulated as n(lambda) and k(lambda) — real and imaginary parts of the complex refractive index as a function of wavelength.

The complex permittivity is `epsilon = (n + ik)^2`. Subtracting an instantaneous background `epsilon_inf` gives the part that must be fit:

```
epsilon(omega) - epsilon_inf = sum_n [ sigma_n * omega_n^2 / (omega_n^2 - omega^2 - i*omega*gamma_n) ]
```

This is a nonlinear function of the parameters (sigma_n, omega_n, gamma_n) for each Lorentzian term. The fitting problem minimizes the L2 norm of the difference between the model and the data over all sampled frequencies.

The Drude model is a special case of the Lorentzian with resonance frequency omega_0 = 0, describing the free-electron response of metals:

```
chi_Drude(omega) = -sigma * omega_p^2 / (omega^2 + i*omega*gamma)
```

where omega_p is the plasma frequency. The Drude contribution is real and negative at low frequencies, producing high reflectivity in metals.

The Kramers-Kronig relations guarantee that any causal response can be decomposed into a sum of Lorentzian/Drude poles (though possibly requiring many terms for accuracy). Since Meep's dispersive time-stepping is exact for this functional form, a good fit to experimental data produces accurate broadband simulations.

The optimization uses NLopt's MMA (Method of Moving Asymptotes) algorithm with an L-BFGS local optimizer. The fitting is repeated 30 times from random initial conditions to escape local minima in this nonconvex optimization landscape.

#### Code Walkthrough

The fitting objective function evaluates the Lorentzian model for given parameters and computes the L2 error, along with its gradient. The gradient is computed analytically, enabling efficient gradient-based optimization:

```python
def lorentzfunc(p, x):
    N = len(p) // 3
    y = np.zeros(len(x))
    for n in range(N):
        A_n, x_n, g_n = p[3*n], p[3*n+1], p[3*n+2]
        y = y + A_n / (x_n**2 - x**2 - 1j * x * g_n)
    return y
```

The `__main__` block loads tabulated data, converts wavelength to frequency (Meep works in frequency units of c/um), and restricts to a wavelength range of interest:

```python
freqs = 1000 / wl  # units of 1/μm
eps = np.square(n) - eps_inf
```

After optimization, the fitted parameters are used to construct a `Medium` object. If a resonance frequency is zero, the term is treated as Drude; otherwise it is Lorentzian. Note that `sigma` in Meep's convention is related to the fitting amplitude by `sigma = A / omega_0^2`:

```python
if mymaterial_freq == 0:
    E_susceptibilities.append(
        mp.DrudeSusceptibility(frequency=1.0, gamma=mymaterial_gamma, sigma=mymaterial_sigma)
    )
else:
    mymaterial_sigma = ps[idx_opt][3*n+0] / mymaterial_freq**2
    E_susceptibilities.append(
        mp.LorentzianSusceptibility(frequency=mymaterial_freq, gamma=mymaterial_gamma, sigma=mymaterial_sigma)
    )
```

The `Medium.epsilon(f)` method evaluates the fitted permittivity at any frequency, enabling comparison with the original data via matplotlib.

#### Key Takeaways

- Experimental n(lambda) data must be converted to epsilon(omega) = n^2 - eps_inf before fitting.
- `mp.DrudeSusceptibility` is the zero-frequency limit of `mp.LorentzianSusceptibility`, appropriate for free-carrier responses in metals and doped semiconductors.
- `Medium.epsilon(f)` evaluates the Drude-Lorentzian model analytically at any frequency for validation.
- Multiple random restarts are essential for fitting because the objective surface is nonconvex.
- Both `sigma` parameter conventions differ between the fitting function and Meep's API: the conversion is `sigma_meep = A / omega_0^2`.

---

### 3. `phase_in_material.py` — Adiabatic Material Transition (Phase-In)

**Physics:** Demonstrates how to gradually morph the material structure of a running simulation from one geometry to another using the `phase_in_material` method, without restarting the simulation.
**Difficulty:** Beginner
**Source:** `python/examples/phase_in_material.py`
**Test Status:** PASS (11.5s)

#### Theory

In some applications it is useful to continuously modify the dielectric structure while the electromagnetic fields are evolving. For example, when computing adiabatic transformations of photonic structures, slowly deforming geometry, or studying the response of a system as a perturbation is ramped on. Meep supports this through the `fields.phase_in_material` method, which linearly interpolates the permittivity grid from one structure to another over a specified time interval.

The electromagnetic fields respond to the changing permittivity. If the change is slow compared to the optical period (adiabatic condition), the fields track the changing eigenmodes of the structure quasi-statically. If the change is abrupt, it excites transients. The `phase_in_material` function gives control over this timescale.

This is conceptually related to the quantum adiabatic theorem: a system evolves without transitions between modes if the Hamiltonian (or in this case the dielectric structure) changes slowly enough relative to the spectral gap between modes.

#### Code Walkthrough

Two independent simulations are initialized — each with a cylinder of index 3.5, but at different center positions:

```python
sim1 = mp.Simulation(cell_size=cell_size, geometry=geometry1, resolution=20)
sim1.init_sim()

sim2 = mp.Simulation(cell_size=cell_size, geometry=geometry2, resolution=20)
sim2.init_sim()
```

The key call transitions sim1's structure toward sim2's structure over 10 time units:

```python
sim1.fields.phase_in_material(sim2.structure, 10.0)
```

After calling `phase_in_material`, the simulation is run normally. During the first 10 time units, the epsilon grid gradually shifts from having a cylinder at the origin to having one at (1, 1). The field output at each half-unit step shows the epsilon evolving in real time:

```python
sim1.run(
    mp.at_beginning(mp.output_epsilon),
    mp.at_every(0.5, mp.output_epsilon),
    until=10
)
```

This approach avoids the discontinuity that would result from an abrupt geometry change, which can introduce spurious reflections and numerical artifacts.

#### Key Takeaways

- `fields.phase_in_material(structure, time)` linearly interpolates the permittivity grid over the given time period.
- Two separate `Simulation` objects must be initialized to extract the source and target structure objects.
- The transition time controls whether the change is adiabatic (slow, no field transients) or sudden (fast, field redistribution).
- `mp.at_every` combined with `mp.output_epsilon` lets you visualize the time-evolving permittivity grid.
- This technique is useful for adiabatic mode converters, perturbation studies, and sensitivity analysis.

---

### 4. `faraday-rotation.py` — Faraday Rotation in a Gyrotropic Medium

**Physics:** Simulates a linearly polarized plane wave propagating through a gyrotropic Lorentzian medium with a static bias field, demonstrating magneto-optical Faraday rotation where the polarization plane continuously rotates as the wave propagates.
**Difficulty:** Advanced
**Source:** `python/examples/faraday-rotation.py`
**Test Status:** PASS (21.2s)

#### Theory

Faraday rotation is a magneto-optical effect in which a static magnetic bias field B0 along the propagation direction renders the medium optically non-reciprocal. The permittivity tensor becomes asymmetric (off-diagonal in the xy plane), and the medium's eigenmodes are left- and right-circular polarizations rather than linear polarizations. These two circular modes propagate with different phase velocities k+ and k-, leading to a net rotation of the linear polarization by the Faraday rotation angle:

```
theta_F = (k+ - k-) / 2 * L
```

where L is the propagation length. This is distinct from optical activity (chirality), which is also non-reciprocal but arises from structural chirality rather than an external field.

For a gyrotropic Lorentzian medium, the permittivity tensor in the presence of a bias B0 along z has the form:

```
epsilon = [ epsilon_perp   -i*eta    0   ]
          [  i*eta      epsilon_perp  0   ]
          [    0             0     epsilon_z ]
```

where epsilon_perp and eta are frequency-dependent quantities. The analytic formula for the k-vector of the slower circular mode (from which the rotation can be calculated) is:

```
k_gyro = 2*pi*f * sqrt[ 0.5 * (epsilon_perp - sqrt(epsilon_perp^2 - eta^2)) ]
```

The parameters epsilon_perp and eta for the gyrotropic Lorentzian model are:

```
dfsq = f0^2 - i*f*gamma - f^2
epsilon_perp = epsilon_n + sigma*f0^2 * dfsq / (dfsq^2 - (f*b0)^2)
eta = sigma * f0^2 * f * b0 / (dfsq^2 - (f*b0)^2)
```

Meep implements this using `GyrotropicLorentzianSusceptibility` with a bias vector that specifies the direction and magnitude of the static bias field.

#### Code Walkthrough

The material parameters define a gyrotropic Lorentzian medium with a bias field b0 = 0.15 in the z direction (the propagation direction):

```python
susc = [
    mp.GyrotropicLorentzianSusceptibility(
        frequency=f0, gamma=gamma, sigma=sn, bias=mp.Vector3(0, 0, b0)
    )
]
mat = mp.Medium(epsilon=epsn, mu=1, E_susceptibilities=susc)
```

The simulation is a 1D cell in z with PML boundaries. The source is an Ex-polarized ContinuousSource (essential for measuring steady-state phase):

```python
sources = [mp.Source(mp.ContinuousSource(frequency=fsrc), component=mp.Ex,
                     center=mp.Vector3(0, 0, src_z))]
```

After running until steady state (tmax = 100), both Ex and Ey field profiles are extracted along the z axis:

```python
ex_data = sim.get_efield_x().real
ey_data = sim.get_efield_y().real
```

The Faraday rotation is visible as sinusoidal oscillations in both Ex and Ey that are 90 degrees out of phase, with the envelope showing the characteristic helical rotation of the polarization vector as a function of z. The analytic comparison uses the formula above to compute `k_gyro` and plots `Ex_theory = 0.37 * cos(k_gyro * (z - src_z))` as an envelope check.

#### Key Takeaways

- `mp.GyrotropicLorentzianSusceptibility` implements the magneto-optical permittivity tensor with a vector bias field.
- The bias vector direction sets the gyrotropy axis; when it is parallel to the propagation direction, Faraday rotation occurs.
- A `ContinuousSource` is needed for direct phase comparison in Faraday rotation measurements; a pulsed source mixes all frequencies.
- Both Ex and Ey components are excited even though only Ex is injected — the Ey arises entirely from the magneto-optical coupling.
- The rotation rate scales with the off-diagonal susceptibility eta, which in turn depends on the bias magnitude b0.

---

### 5. `absorber-1d.py` — Comparison of PML vs. Absorber Boundary Layers in a Metal

**Physics:** Tests the effectiveness of Meep's pseudo-absorber boundary (a gradual conductivity ramp) as an alternative to PML for absorbing outgoing waves in a 1D simulation filled with a dispersive metal (aluminum).
**Difficulty:** Beginner
**Source:** `python/examples/absorber-1d.py`
**Test Status:** PASS (5.5s)

#### Theory

Perfectly Matched Layer (PML) boundaries are the standard approach for truncating FDTD domains — they absorb outgoing waves without reflection by analytically continuing the wave equation into complex coordinates. However, PML has a known failure mode in dispersive, highly absorbing materials (such as metals at visible frequencies): the coordinate stretching that makes PML work in vacuum becomes ill-conditioned when the material itself has a large imaginary part of epsilon.

Meep provides an alternative: the `Absorber` boundary, which implements a gradual increase in conductivity (both electric D_conductivity and magnetic B_conductivity) near the boundary. This effectively damps the fields before they reach the computational domain edge, acting as an adiabatic absorber. The tradeoff is that the Absorber is generally less reflective than PML for evanescent fields and interfaces, but can outperform PML in highly lossy materials.

For aluminum at visible wavelengths, the permittivity has a large negative real part and a significant imaginary part (from free-carrier absorption). The Drude model for aluminum captures this response. A source in this medium decays rapidly due to the material absorption, and the remaining small-amplitude fields must be absorbed by the boundary without spurious reflection.

The simulation runs until the fields have decayed to 1e-6 of their peak value at the source location, confirming that both the material absorption and the boundary layer have completely damped the excitation.

#### Code Walkthrough

Aluminum is imported from Meep's built-in materials library, which contains Drude-Lorentzian fits to experimental data:

```python
from meep.materials import Al
```

The boundary layer is selected by the command-line argument `--pml` (uses PML) or the default (uses Absorber):

```python
boundary_layers = [
    mp.PML(1, direction=mp.Z) if args.pml else mp.Absorber(1, direction=mp.Z)
]
```

The source frequency corresponds to lambda = 0.803 um (near infrared), a regime where aluminum is highly metallic. The 1D simulation cell is 10 um long with the source at the center:

```python
sim = mp.Simulation(
    cell_size=mp.Vector3(z=10),
    dimensions=1,
    default_material=Al,
    boundary_layers=boundary_layers,
    sources=sources,
    resolution=40,
)
```

The stopping condition `stop_when_fields_decayed` monitors the Ex field at the origin and halts the simulation once it drops below 1e-6 of its peak, ensuring complete field decay:

```python
sim.run(
    mp.at_every(10, print_stuff),
    until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6),
)
```

#### Key Takeaways

- `mp.Absorber` provides an alternative to PML that works by gradually ramping up conductivity near boundaries, particularly effective in highly absorbing or metallic materials.
- `from meep.materials import Al` imports a pre-fitted Drude-Lorentzian model for aluminum covering the near-UV to near-IR range.
- `mp.stop_when_fields_decayed(dt, component, point, decay)` is a clean stopping criterion that avoids fixed-time runs by monitoring actual field decay.
- The `direction` parameter on PML and Absorber restricts the absorbing layer to one axis, useful for 1D simulations.
- For 1D simulations, `dimensions=1` must be set explicitly; the cell only needs a nonzero z extent.

---

### 6. `refl-angular.py` — Fresnel Reflectance at Oblique Incidence

**Physics:** Computes the broadband reflectance of an air-to-dielectric (n=3.5) interface as a function of angle of incidence and wavelength, comparing FDTD results against the Fresnel equations.
**Difficulty:** Intermediate
**Source:** `python/examples/refl-angular.py`
**Test Status:** PASS (19.4s)

#### Theory

When a plane wave strikes a planar interface between two dielectric media at angle of incidence theta, part of the wave is reflected and part is transmitted. The reflectance (fraction of power reflected) is given by the Fresnel equations, which depend on both the angle and the polarization of the incident wave.

For s-polarization (TE, electric field perpendicular to the plane of incidence), the reflectance is:

```
R_s = | (n1*cos(theta_i) - n2*cos(theta_t)) / (n1*cos(theta_i) + n2*cos(theta_t)) |^2
```

For p-polarization (TM, electric field in the plane of incidence), it is:

```
R_p = | (n2*cos(theta_i) - n1*cos(theta_t)) / (n2*cos(theta_i) + n1*cos(theta_t)) |^2
```

where theta_t is the transmission angle given by Snell's law: `n1*sin(theta_i) = n2*sin(theta_t)`.

An important special case is Brewster's angle, where R_p = 0. This occurs when `tan(theta_B) = n2/n1`, giving complete transmission for p-polarization. For n1=1, n2=3.5, Brewster's angle is about 74 degrees.

In Meep, oblique incidence is simulated by setting a nonzero `k_point`, which imposes a Bloch boundary condition corresponding to the in-plane component of the wavevector. The simulation is then effectively 1D (propagation normal to the interface) but with the correct phase relationships for the oblique angle. The entire calculation is done at a fixed `k.x = f_min * sin(theta)`, so different frequencies in the broadband pulse correspond to different angles (because the physical angle theta = arcsin(kx/f) depends on frequency).

The two-run subtraction procedure removes the incident wave from the reflected flux: the first run records the empty-cell flux as a reference, and the second run with the interface loads minus this reference flux so that only the reflected power is counted.

#### Code Walkthrough

The k-point encodes the oblique angle at the minimum frequency. For normal incidence, the simulation collapses to 1D; otherwise it is 3D with the Bloch condition:

```python
theta_r = math.radians(args.theta)
k = mp.Vector3(math.sin(theta_r), 0, math.cos(theta_r)).scale(fmin)
dimensions = 1 if theta_r == 0 else 3
```

The first run (empty cell) captures the incident flux and stores it for subtraction:

```python
refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, -0.25*sz))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
sim.run(until_after_sources=mp.stop_when_fields_decayed(...))
empty_data = sim.get_flux_data(refl)
sim.reset_meep()
```

The second run adds the interface and loads minus the reference flux, so `get_fluxes` returns only the reflected power:

```python
sim.load_minus_flux_data(refl, empty_data)
# ...run...
refl_flux = mp.get_fluxes(refl)
# reflectance:
R = -refl_flux[i] / empty_flux[i]
```

The sign convention: reflected flux flows in the -z direction (toward the source), so it appears negative in Meep's convention. The minus sign in `load_minus_flux_data` and in the final ratio converts this correctly to a positive reflectance.

For each frequency, the actual angle is recovered from the stored k.x:

```python
theta_actual = math.degrees(math.asin(k.x / freqs[i]))
```

#### Key Takeaways

- Setting `k_point = mp.Vector3(kx, 0, kz)` imposes oblique incidence; the Bloch condition handles the transverse phase.
- The two-run subtraction (`load_minus_flux_data`) isolates the reflected flux from the total flux at the monitor.
- At fixed k.x, different frequencies correspond to different physical angles: `theta(f) = arcsin(kx/f)`.
- `mp.reset_meep()` resets all internal state between runs while reusing the Python Simulation object.
- The `nfreq` parameter in `add_flux` controls how many frequency bins the DFT monitor records.

---

### 7. `refl-angular-kz2d.py` — Oblique Reflectance Using the `kz_2d` Approximation

**Physics:** Computes the reflectance at a fixed oblique angle using three different computational modes (`real/imag`, `complex`, and `3d`) enabled by the `kz_2d` parameter, demonstrating that a 1D simulation can exactly replicate a 3D oblique-incidence result.
**Difficulty:** Advanced
**Source:** `python/examples/refl-angular-kz2d.py`
**Test Status:** TIMEOUT (CPU-intensive — runs three separate full reflectance calculations)

#### Theory

For oblique incidence on a structure that is uniform in the transverse directions (y and z for a 1D interface), the full 3D Maxwell equations can be reduced to an effective 1D problem. The in-plane wavevector component kz (in the plane of incidence) acts as a parameter, modifying the effective dispersion relation but not requiring explicit resolution of the transverse dimensions.

Meep implements this through the `kz_2d` parameter, which allows a 2D (or 1D) simulation to incorporate the effect of an out-of-plane wavevector component kz. The three supported modes are:

- `"real/imag"`: Uses the real and imaginary parts of the fields separately to represent the complex Bloch phase, keeping all fields real-valued (most memory efficient, works for any kz).
- `"complex"`: Directly uses complex-valued fields, allowing arbitrary kz (slightly more expensive).
- `"3d"`: Fully 3D simulation with explicit kz in the Bloch condition (most expensive, serves as ground truth).

All three modes should give identical physics for a planar interface with no transverse structure. This example verifies that the `kz_2d` approximation is numerically accurate by comparing all three against the Fresnel analytic result.

The Fresnel reflectance for s-polarization at angle theta from medium n1 to n2 is:

```
R = | (n2*cos(theta_t) - n1*cos(theta_i)) / (n2*cos(theta_t) + n1*cos(theta_i)) |^2
```

where `theta_t = arcsin(n1*sin(theta_i)/n2)` from Snell's law.

#### Code Walkthrough

The key parameter is the `kz_2d` argument passed to the `Simulation` constructor. The k-point contains only a z-component encoding the oblique angle:

```python
k = mp.Vector3(z=math.sin(theta)).scale(fcen)

sim = mp.Simulation(
    cell_size=cell_size,
    k_point=k,
    kz_2d=kz_2d,   # "real/imag", "complex", or "3d"
    ...
)
```

The same two-run subtraction procedure as in `refl-angular.py` is used. The function is called three times with different kz_2d modes:

```python
Rmeep_real_imag = refl_planar(theta_r, "real/imag")
Rmeep_complex   = refl_planar(theta_r, "complex")
Rmeep_3d        = refl_planar(theta_r, "3d")
```

The analytic Fresnel result uses `n1=1, n2=3.5` at `theta_r = 19.4 degrees`:

```python
Rfresnel = lambda theta_in: fabs(
    (n2*cos(theta_out(theta_in)) - n1*cos(theta_in)) /
    (n2*cos(theta_out(theta_in)) + n1*cos(theta_in))
)**2
```

All four values are printed for comparison, demonstrating numerical equivalence to within FDTD discretization error.

#### Key Takeaways

- `kz_2d="real/imag"` or `"complex"` allows a 1D/2D simulation to exactly capture oblique-incidence physics by treating kz as a parameter rather than resolving it spatially.
- The `"3d"` mode confirms the dimensional reduction is exact for planar-uniform structures.
- `kz_2d` reduces computational cost for angle-sweep calculations in structures with translational symmetry.
- The `k_point` in a kz_2d simulation encodes only the oblique wavevector component, not the full propagation vector.
- Comparing all three modes is a standard validation step when first using the kz_2d approximation.

---

### 8. `refl-quartz.py` — Broadband Reflectance of Fused Quartz vs. Sellmeier Formula

**Physics:** Computes the normal-incidence reflectance spectrum of fused quartz (SiO2) from 0.4 to 0.8 um using Meep's built-in Lorentzian material model, and compares with the analytic Fresnel formula using the Sellmeier equation for the refractive index.
**Difficulty:** Intermediate
**Source:** `python/examples/refl-quartz.py`
**Test Status:** PASS (23.3s)

#### Theory

The reflectance at normal incidence from a half-space of permittivity epsilon is:

```
R = | (1 - sqrt(epsilon)) / (1 + sqrt(epsilon)) |^2
```

For a transparent material at normal incidence, epsilon is real and positive, giving `R = ((n-1)/(n+1))^2`.

Fused quartz (amorphous SiO2) is a classic optical glass with a well-characterized dispersion described by the Sellmeier equation, an empirical three-term expression:

```
epsilon(lambda) = 1 + B1*lambda^2/(lambda^2 - C1) + B2*lambda^2/(lambda^2 - C2) + B3*lambda^2/(lambda^2 - C3)
```

where the constants B_i and C_i are fit to experimental data. The Sellmeier form is algebraically equivalent to a sum of Lorentzian poles with zero damping (gamma = 0):

```
epsilon(omega) = epsilon_inf + sum_i [ sigma_i * omega_i^2 / (omega_i^2 - omega^2) ]
```

In Meep's `fused_quartz` material from the materials library, three such lossless Lorentzian terms are used, with resonance wavelengths at 0.0684 um (UV), 0.1162 um (deep UV), and 9.896 um (mid-IR). Because quartz is transparent in the visible, gamma = 0 for all terms.

The broadband FDTD approach is highly efficient: a single Gaussian pulse run produces reflectance data over all 50 frequency bins simultaneously. This leverages the DFT flux monitor's ability to compute the Fourier transform of the time-domain fields at multiple frequencies within a single time-stepping run.

#### Code Walkthrough

The fused quartz material is imported directly from the library:

```python
from meep.materials import fused_quartz
```

The simulation geometry places a semi-infinite quartz block in the right half of the cell:

```python
geometry = [
    mp.Block(
        mp.Vector3(mp.inf, mp.inf, 0.5*sz),
        center=mp.Vector3(z=0.25*sz),
        material=fused_quartz,
    )
]
```

The analytic comparison uses the Sellmeier formula directly in Python:

```python
eps_quartz = lambda l: 1 \
    + 0.6961663 * l**2 / (l**2 - 0.0684043**2) \
    + 0.4079426 * l**2 / (l**2 - 0.1162414**2) \
    + 0.8974794 * l**2 / (l**2 - 9.896161**2)
R_fresnel = lambda l: ((1 - sqrt(eps_quartz(l))) / (1 + sqrt(eps_quartz(l))))**2
```

The meep reflectance is computed using the two-run subtraction at `dimensions=1` (normal incidence):

```python
R_meep = -1 * np.divide(refl_flux, empty_flux)
```

The reflectance of quartz varies slightly with wavelength (from about 0.034 to 0.036 across 0.4-0.8 um) due to the wavelength dependence of its refractive index. The FDTD result agrees with the analytic formula to within < 0.5%.

#### Key Takeaways

- `meep.materials.fused_quartz` provides a lossless three-pole Lorentzian (Sellmeier) model for fused quartz valid from 0.21 to 6.7 um.
- Lossless Lorentzians use `gamma=0`; all poles are purely reactive, contributing only to the real part of epsilon.
- A single broadband FDTD run with DFT monitors produces reflectance vs. wavelength across a continuous spectrum.
- `dimensions=1` reduces the full 3D Maxwell equations to a 1D problem for planar normal-incidence geometries.
- The agreement between FDTD and Fresnel validates that the Lorentzian material model exactly matches the Sellmeier dispersion for quartz.

---

### 9. `oblique-planewave.py` — Launching an Oblique Planewave via EigenModeSource

**Physics:** Demonstrates how to inject a cleanly propagating oblique planewave into a homogeneous 2D medium using an `EigenModeSource` with a rotated k-point, rather than a simple point source.
**Difficulty:** Intermediate
**Source:** `python/examples/oblique-planewave.py`
**Test Status:** PASS (110.0s)

#### Theory

Launching a pure planewave at oblique incidence in FDTD is non-trivial. A simple current source with a fixed amplitude profile launches waves in multiple directions (a finite-width source acts as an aperture, diffracting light in all directions). To inject a clean, single-angle planewave, one approach is to use an extended source (infinite in y) with a Bloch-periodic boundary condition, but this limits the simulation to a single frequency.

The EigenModeSource approach works differently: it uses the MPB eigenmode solver to compute the exact field profile of the planewave mode in the homogeneous medium, then uses that profile as the source. For a homogeneous medium, the planewave at frequency f and angle theta has wavevector:

```
k = n * f * (sin(theta), 0)   [in 2D, CCW from x-axis]
```

The Bloch condition set by `k_point` must match this wavevector for the phase relationship to be consistent. The `eig_kpoint` argument to EigenModeSource tells MPB which direction the eigenmode should propagate, and `eig_band=1` selects the fundamental (planewave) mode.

An important subtlety for oblique incidence: the `direction` parameter must be set to `mp.NO_DIRECTION` (because the source plane normal is not aligned with a principal axis when the mode propagates at an angle), and the `eig_vol` must be specified as a small volume to allow MPB to sample the mode at a specific point rather than integrating over the full source plane.

#### Code Walkthrough

The k-point for the Bloch condition is the full wavevector of the planewave in the medium:

```python
incident_angle = np.radians(40.0)
n_mat = 1.5
k_point = mp.Vector3(n_mat * frequency, 0, 0).rotate(mp.Vector3(0, 0, 1), incident_angle)
```

For oblique incidence, the EigenModeSource requires explicit settings that differ from normal incidence:

```python
sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency),
        center=mp.Vector3(),
        size=mp.Vector3(0, size_um, 0),
        direction=mp.NO_DIRECTION,
        eig_kpoint=k_point,
        eig_band=1,
        eig_parity=mp.ODD_Z,
        eig_vol=mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 1/resolution_um, 0)),
    )
]
```

The `eig_parity=mp.ODD_Z` specifies the TM polarization (Hz is the primary component). The `eig_vol` is a single-pixel volume that tells MPB where to evaluate the mode profile — a single cell in y ensures the mode is sampled locally rather than averaged over the source extent.

PML is applied only in the x-direction (perpendicular to the source plane) to absorb the planewave after it propagates across the cell:

```python
pml_layers = [mp.PML(thickness=pml_um, direction=mp.X)]
```

The field visualization confirms that Ez shows the expected sinusoidal pattern of a planewave propagating at 40 degrees from the x-axis, with constant phase fronts tilted accordingly.

#### Key Takeaways

- `mp.EigenModeSource` with `direction=mp.NO_DIRECTION` and `eig_kpoint` is the correct approach for oblique planewaves in homogeneous media.
- The `k_point` on the Simulation must match the `eig_kpoint` on the source for the Bloch phase to be consistent.
- `eig_parity=mp.ODD_Z` selects TM polarization; `eig_parity=mp.EVEN_Y + mp.ODD_Z` selects TE for normal incidence.
- `eig_vol` set to a single pixel prevents MPB from averaging the mode profile, which would smear the phase for oblique modes.
- PML should be applied only perpendicular to the planewave propagation direction to avoid interfering with the Bloch boundary condition.

---

### 10. `oblique-source.py` — Oblique Eigenmode Source in a Rotated Waveguide

**Physics:** Injects a guided eigenmode at oblique angle into a rotated slab waveguide using an EigenModeSource, demonstrating how to set up mode-matched sources when the waveguide axis is not aligned with the computational grid.
**Difficulty:** Advanced
**Source:** `python/examples/oblique-source.py`
**Test Status:** TIMEOUT (CPU-intensive — full 2D simulation with mode decomposition)

#### Theory

When a photonic structure such as a waveguide is oriented at an angle to the simulation grid, the mode fields are not aligned with any Cartesian direction. The correct way to inject a guided mode into such a rotated structure is to use the EigenModeSource with the `eig_kpoint` pointing along the waveguide axis and `eig_match_freq=True`, so that MPB finds the guided mode at the correct frequency.

For a waveguide tilted by angle `rot_angle` from the x-axis, the propagation direction unit vector is:

```
khat = (cos(rot_angle), sin(rot_angle), 0)
```

The EigenModeSource samples the mode profile across a cross-section perpendicular to this direction and injects it with the correct phase gradient. This is significantly more accurate than a simple current source, which would excite many spurious modes at the waveguide boundaries.

The Bloch condition for the simulation must be zero (no `k_point`) for the rotated waveguide case because the waveguide has a different direction than any simulation boundary. The source is launched at the center of the cell, and the transmission is measured at a flux plane downstream.

Mode decomposition using `get_eigenmode_coefficients` extracts the overlap of the transmitted field with the guided mode, providing the modal power carried by the specific waveguide mode. The ratio of modal flux to total flux gives the mode purity.

#### Code Walkthrough

The waveguide is defined as a Block with rotated basis vectors:

```python
geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, w, mp.inf),
        e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle),
        e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), rot_angle),
        material=mp.Medium(epsilon=12),
    )
]
```

The EigenModeSource uses `direction=mp.NO_DIRECTION` and `eig_kpoint` along the waveguide axis:

```python
kpoint = mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle)
sources = [
    mp.EigenModeSource(
        src=mp.GaussianSource(fsrc, fwidth=0.2*fsrc),
        center=mp.Vector3(),
        size=mp.Vector3(y=3*w),
        direction=mp.NO_DIRECTION,
        eig_kpoint=kpoint,
        eig_band=bnum,
        eig_parity=mp.ODD_Z,
        eig_match_freq=True,
    )
]
```

For flux computation, the transmission is measured and then decomposed into modal contributions:

```python
res = sim.get_eigenmode_coefficients(
    tran, [1],
    eig_parity=mp.ODD_Z,
    direction=mp.NO_DIRECTION,
    kpoint_func=lambda f, n: kpoint,
)
```

The `kpoint_func` tells the decomposition which direction to use for the mode at each frequency, overriding the automatic selection. The modal amplitude `res.alpha[0, 0, 0]` gives the complex coefficient for mode 1.

#### Key Takeaways

- Rotated Block geometry uses `e1` and `e2` basis vectors constructed by rotating unit vectors around the z-axis.
- `eig_match_freq=True` instructs MPB to find the mode frequency that matches the source frequency, essential for dispersion-corrected mode injection.
- `get_eigenmode_coefficients` with `kpoint_func` performs modal decomposition for modes propagating in arbitrary directions.
- For a rotated waveguide without a preferred grid direction, set `symmetries=[]` and `direction=mp.NO_DIRECTION`.
- The ratio `abs(res.alpha[0,0,0])**2 / get_fluxes(tran)[0]` measures the fraction of transmitted power in the fundamental mode.

---

### 11. `test_material_dispersion.py` — Unit Test of Dispersive Material via k-point Sweep

**Physics:** Verifies that the FDTD dispersion relation in a Lorentzian dispersive medium matches expected frequency values extracted by Harminv across a range of k-points, with the medium specified via a user-defined material function.
**Difficulty:** Intermediate
**Source:** `python/tests/test_material_dispersion.py`
**Test Status:** PASS (92.95s)

#### Theory

This test is the automated validation counterpart of `material-dispersion.py`. It confirms that the Lorentzian dispersive material model in Meep produces the correct dispersion relation by comparing computed eigenfrequencies against pre-computed reference values.

The test uses a `material_function` rather than `default_material`. A material function is a Python callable that takes a position vector and returns a `Medium` object. This allows spatially varying materials defined programmatically — for homogeneous media, the function simply returns the same Medium regardless of position.

The expected frequencies in the test represent the Meep-computed dispersion relation at 7 k-points, validated against the analytic result for a Lorentzian medium with `epsilon=2.25` and two susceptibility terms. The first term (strong resonance at omega=1.1) dominates the dispersion at higher k-values; the second (weak at omega=0.5) has negligible effect on the dispersion but adds small imaginary parts to the frequencies.

The test uses only 7 k-points (k_interp=5, producing 7 points total) rather than the 101 of the example, trading resolution for speed in the automated test suite.

#### Code Walkthrough

The material function syntax allows spatially varying media, though here it returns a uniform medium:

```python
def mat_func(p):
    return mp.Medium(epsilon=2.25, E_susceptibilities=susceptibilities)

self.sim = mp.Simulation(
    cell_size=mp.Vector3(),
    material_function=mat_func,
    default_material=mp.air,
    ...
)
```

The test extracts only the real parts of the frequencies for comparison:

```python
all_freqs = self.sim.run_k_points(200, kpts)
res = [f.real for fs in all_freqs for f in fs]

np.testing.assert_allclose(expected, res)
```

The `assert_allclose` comparison uses default tolerances of `rtol=1e-7`, confirming that the dispersion relation is reproduced to high precision.

#### Key Takeaways

- `material_function` accepts a callable `f(position) -> Medium`, enabling arbitrary spatially varying dispersive materials.
- Setting both `material_function` and `default_material=mp.air` is the correct pattern: the material function overrides the default at every gridpoint.
- `run_k_points` returns a list of lists; flattening with a list comprehension extracts individual frequencies.
- The dispersion relation serves as a high-fidelity test of the auxiliary-field dispersive stepping algorithm.
- Expected reference values in unit tests capture the deterministic nature of the FDTD computation, breaking if discretization or material model changes.

---

### 12. `test_faraday_rotation.py` — Quantitative Validation of Gyrotropic Material Models

**Physics:** Validates all three of Meep's gyrotropic material models (GyrotropicLorentzianSusceptibility, GyrotropicDrudeSusceptibility, and GyrotropicSaturatedSusceptibility / Landau-Lifshitz-Gilbert) by measuring the Faraday rotation angle and comparing against analytic formulas.
**Difficulty:** Advanced
**Source:** `python/tests/test_faraday_rotation.py`
**Test Status:** PASS (9.1s)

#### Theory

This test suite validates three distinct physical models for magneto-optical media:

**Gyrotropic Lorentzian model:** Describes bound electrons in a magnetic field, appropriate for magneto-optical dielectrics (e.g., iron garnets, bismuth-substituted ferrites). The permittivity tensor is derived from the equation of motion of a bound electron in a magnetic field, giving off-diagonal components proportional to the bias field.

**Gyrotropic Drude model:** Describes free carriers in a magnetic field (cyclotron resonance), appropriate for semiconductors and metals in a DC magnetic field. The carrier oscillates at the cyclotron frequency `omega_c = e*B/(m*c)`, and the off-diagonal permittivity is enhanced near this frequency.

**Landau-Lifshitz-Gilbert (LLG) model:** Describes the magnetization dynamics of ferromagnetic materials, including ferrites and magnetic garnets. The LLG equation governs the precession of the magnetic moment around the bias field, with Gilbert damping. This is implemented via `GyrotropicSaturatedSusceptibility` with the `alpha` damping parameter.

For each model, the Faraday rotation rate k_gyro is computed analytically by diagonalizing the gyrotropic permittivity tensor. The two eigenvalues correspond to the left- and right-circular polarization wavevectors:

```
k_+- = (omega/c) * sqrt(epsilon_perp +/- eta)
```

The Faraday rotation rate (angle per unit length) is `(k+ - k-)/2`. A linearly polarized wave decomposes into equal left and right circular components, which accumulate a differential phase of `(k+ - k-)*L` after propagation through length L, corresponding to a rotation of `(k+ - k-)*L/2` of the linear polarization direction.

The test measures this rotation by extracting the FFT amplitudes of Ex and Ey at an observation point, computing `arctan2(Ey_amplitude, Ex_amplitude)`, and comparing with the predicted angle `arctan2(sin(k_gyro*(z-z_src)), cos(k_gyro*(z-z_src)))`.

#### Code Walkthrough

The analytic k_gyro formulas are defined for each model:

```python
def kgyro_lorentzian(freq, epsn, f0, gamma, sigma, b0):
    dfsq = f0**2 - 1j * freq * gamma - freq**2
    eperp = epsn + sigma * f0**2 * dfsq / (dfsq**2 - (freq * b0)**2)
    eta = sigma * f0**2 * freq * b0 / (dfsq**2 - (freq * b0)**2)
    return 2 * np.pi * freq * np.sqrt(0.5 * (eperp - np.sqrt(eperp**2 - eta**2)))
```

The Faraday angle is extracted using FFT amplitude comparison to separate the coherent sinusoidal oscillation from noise:

```python
ex_rel = np.amax(abs(np.fft.fft(record_Ex)))
ey_rel = np.amax(abs(np.fft.fft(record_Ey)))
result = np.arctan2(ey_rel, ex_rel) * 180 / np.pi
```

The three material models tested are:

```python
# Lorentzian
mp.GyrotropicLorentzianSusceptibility(frequency=f0, gamma=gamma, sigma=sn, bias=mp.Vector3(0,0,b0))

# Drude
mp.GyrotropicDrudeSusceptibility(frequency=f0, gamma=gamma, sigma=sn, bias=mp.Vector3(0,0,b0))

# Landau-Lifshitz-Gilbert
mp.GyrotropicSaturatedSusceptibility(frequency=f0, gamma=gamma, sigma=sn, alpha=alpha, bias=mp.Vector3(0,0,1.0))
```

The tolerance of 1.5 degrees allows for finite-resolution FDTD discretization errors.

#### Key Takeaways

- Three distinct gyrotropic susceptibility classes cover the main physical regimes: Lorentzian (bound electrons), Drude (free carriers), and LLG (magnetic moment precession).
- The `bias` vector in gyrotropic models sets both the direction and magnitude of the effective static field; the bias should point along the propagation axis for Faraday rotation.
- FFT amplitude extraction (`np.amax(abs(np.fft.fft(...)))`) separates the monochromatic response from transients.
- `mp.after_time` step function records fields only after the initial transient has decayed, ensuring steady-state measurement.
- The LLG model has an additional `alpha` (Gilbert damping) parameter distinct from the dissipation `gamma`.

---

### 13. `test_conductivity.py` — Waveguide Loss via D-Field Conductivity

**Physics:** Validates the D-field conductivity model in Meep by measuring the exponential attenuation of a guided mode in a lossy waveguide and comparing with the expected exponential decay rate derived from the complex refractive index.
**Difficulty:** Intermediate
**Source:** `python/tests/test_conductivity.py`
**Test Status:** PASS (97.6s)

#### Theory

Optical loss in a material is captured by the imaginary part of the permittivity, Im(epsilon). In a plane wave, this imaginary part leads to exponential decay of the field amplitude. The power decays as:

```
P(z) = P(0) * exp(-alpha * z)
```

where the amplitude attenuation coefficient alpha (in units of inverse length) is related to the imaginary part of the refractive index k_ext by:

```
alpha = 4*pi*k_ext / lambda = 4*pi*f * Im(n)
```

In Meep, material loss is represented through the `D_conductivity` parameter. The relationship between the D-field conductivity sigma_D and the complex permittivity is:

```
epsilon_eff = epsilon_r + i * sigma_D / (2*pi*f)
```

Therefore Im(epsilon) = sigma_D / (2*pi*f), and:

```
sigma_D = 2*pi*f * Im(epsilon)
```

Alternatively, in engineering notation with attenuation in dB/cm:

```
Im(n) = (lambda / (4*pi)) * (alpha [1/um]) = (1 / (4*pi*f)) * (att_dB_cm * conversion)
```

The test uses a 2D slab waveguide with `epsilon=12` and a moderate attenuation coefficient of 37.46 dB/cm. Flux monitors at two positions (5 and 10 um downstream from the source) measure the transmitted power. The ratio of powers at these positions gives the attenuation over 5 um, which is compared against the theoretical exponential decay.

The EigenModeSource launches the fundamental TE mode of the waveguide. Note that MPB (which computes the eigenmode for the source) treats the material as lossless (ignores Im(epsilon)), so for small loss values, the launched mode is a good approximation to the true lossy guided mode. For very high losses, the eigenmode itself becomes inaccurate and alternative methods are needed.

#### Code Walkthrough

The effective complex refractive index is computed from the attenuation coefficient, and the D-field conductivity is derived from it:

```python
n_eff = np.sqrt(12.0) + 1j * (1/fsrc) * (dB_cm_to_dB_um * att_coeff) / (4*np.pi)
eps_eff = n_eff * n_eff
sigma_D = 2 * np.pi * fsrc * np.imag(eps_eff) / np.real(eps_eff)
```

The waveguide material is set with `D_conductivity`:

```python
geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, w, mp.inf),
        material=mp.Medium(epsilon=np.real(eps_eff), D_conductivity=sigma_D),
    )
]
```

Two flux monitors measure transmitted power at different downstream positions:

```python
tran1 = sim.add_flux(fsrc, 0, 1, mp.FluxRegion(center=mp.Vector3(x=0.0), size=mp.Vector3(y=10.0)))
tran2 = sim.add_flux(fsrc, 0, 1, mp.FluxRegion(center=mp.Vector3(x=5.0), size=mp.Vector3(y=10.0)))
```

The test checks both that the lossless waveguide transmits unity flux ratio (flux is conserved) and that the lossy waveguide decays at the expected rate:

```python
expected_att = np.exp(-att_coeff * dB_cm_to_dB_um * L)
self.assertAlmostEqual(attenuated_flux / incident_flux, expected_att, places=2)
```

#### Key Takeaways

- `D_conductivity` on `mp.Medium` sets the imaginary part of the permittivity: `sigma_D = 2*pi*f * Im(epsilon)`.
- The conversion from dB/cm attenuation to sigma_D goes through the complex refractive index n + ik.
- EigenModeSource ignores Im(epsilon) when computing the mode profile; for low-loss materials, this approximation is excellent.
- Two-point flux measurement confirms the exponential decay law without needing to measure the absolute field amplitude.
- The `dB_cm_to_dB_um = 1e-4` conversion factor is essential when using Meep's default length unit of micrometers.

---

### 14. `test_absorber_1d.py` — Absorber Boundary Validation in 1D and 2D

**Physics:** Validates the `mp.Absorber` boundary layer (pseudo-absorbing conductivity ramp) in both 1D (with a metallic material) and 2D (with a simple source in free space), confirming that the residual field values after complete decay match expected reference values.
**Difficulty:** Beginner
**Source:** `python/tests/test_absorber_1d.py`
**Test Status:** PASS (17.6s)

#### Theory

The `mp.Absorber` in Meep implements an adiabatic absorber (also called a conductivity-based absorber or stretched-coordinate absorber) by gradually ramping up the electric and magnetic conductivities over a finite transition region at the domain boundary. The conductivity profile follows a smooth polynomial ramp to avoid reflections at the absorber-domain interface.

Unlike PML, which analytically continues the wave equation into complex coordinates and therefore perfectly absorbs all angles of incidence, the Absorber may have residual reflections that decrease with the absorber thickness and the smoothness of the conductivity ramp. However, the Absorber avoids the numerical instabilities that PML can exhibit in highly dispersive or gain media.

For aluminum at near-IR wavelengths, the skin depth is extremely short (tens of nanometers). The field from a source in bulk aluminum decays exponentially with a characteristic length set by `delta = lambda / (2*pi*k_ext)`. By the time the wave reaches the absorber, it has already decayed significantly due to material absorption. The Absorber only needs to handle the small residual amplitude.

The 2D test case uses a simple `Hz` source in free space, where the absorber must work without assistance from material absorption. The large cell (20x20 um) and thick absorber (5 um) ensure adequate absorption.

#### Code Walkthrough

The 1D test sets up aluminum with an Absorber boundary:

```python
boundary_layers = [mp.Absorber(1, direction=mp.Z)]

self.sim = mp.Simulation(
    cell_size=mp.Vector3(z=10),
    dimensions=1,
    default_material=Al,
    boundary_layers=boundary_layers,
    ...
)
```

The field value at the origin after complete decay is compared against a hard-coded reference:

```python
f = self.sim.get_field_point(mp.Ex, mp.Vector3())
self.assertAlmostEqual(f.real, 3.218846961494622e-13, places=6)
```

The 2D test uses a broader absorber (5 um) on all boundaries to handle the 2D spreading of a cylindrical wave:

```python
sim = mp.Simulation(
    cell_size=mp.Vector3(20, 20, 0),
    resolution=10,
    sources=[source],
    boundary_layers=[mp.Absorber(5)],
)
```

The reference value `~-4e-11` is the numerical noise floor after 1000 time units, confirming that the absorber has completely eliminated the wave energy.

#### Key Takeaways

- `mp.Absorber(thickness, direction=mp.Z)` creates a direction-specific absorber; omitting `direction` applies it to all boundaries.
- For metallic materials at optical frequencies, material absorption already handles most of the damping — the Absorber boundary handles only the tail.
- Reference values in regression tests encode the expected numerical behavior, including the floating-point noise floor.
- `stop_when_fields_decayed` is preferred over a fixed `until` time for ensuring complete field decay regardless of material losses.
- The Absorber is less efficient than PML for gradual wavevector angles but more numerically stable for dispersive or active media.

---

### 15. `test_refl_angular.py` — Fresnel Reflectance with BFAST Source

**Physics:** Validates broadband Fresnel reflectance for P-polarization (Ex field) at normal and two oblique angles, testing both the standard fixed-k approach and the BFAST (Broadband Fixed-Angle Source Technique) method that maintains a fixed physical angle across all frequencies.
**Difficulty:** Advanced
**Source:** `python/tests/test_refl_angular.py`
**Test Status:** FAIL (requires `parameterized` package: `pip install parameterized`)

#### Theory

The standard oblique-incidence approach in Meep uses a fixed in-plane wavevector `k.x`, which means that as the broadband pulse sweeps over frequencies, the physical angle of incidence changes as `theta(f) = arcsin(k.x / (n1 * f))`. This is physically correct but mixes angles in the broadband spectrum.

BFAST (Broadband Fixed-Angle Source Technique) keeps the physical angle constant across all frequencies by scaling the Courant number and using a `bfast_scaled_k` parameter. Specifically:

- The simulation uses Bloch boundary conditions with `k = 0` (normal).
- The `bfast_scaled_k = (n1 * sin(theta), 0, 0)` parameter rescales the field components to account for the oblique angle.
- The Courant number is reduced to `(1 - bfast_scaled_k[0]) / sqrt(3)` for numerical stability.

This technique allows the Fresnel reflectance to be computed at a fixed angle as a function of frequency, which is the physically relevant quantity for angular-resolved reflectance spectroscopy.

The Fresnel formula for P-polarization (TM, Ex component incident, plane of incidence = XZ):

```
R_p = | (n1*cos(theta_t) - n2*cos(theta_i)) / (n1*cos(theta_t) + n2*cos(theta_i)) |^2
```

where theta_t is determined by Snell's law. Note the sign convention differs from s-polarization.

The test checks three cases: normal incidence (theta=0), oblique without BFAST (theta=20.6 degrees), and oblique with BFAST (theta=35.7 degrees), with n1=1.4 and n2=3.5.

#### Code Walkthrough

The BFAST path sets `k = mp.Vector3()` and uses `bfast_scaled_k` and a modified Courant number:

```python
if use_bfast:
    bfast_scaled_k = (self.n1 * np.sin(theta_rad), 0, 0)
    Courant = (1 - bfast_scaled_k[0]) / 3**0.5
    k = mp.Vector3()
else:
    bfast_scaled_k = (0, 0, 0)
    Courant = 0.5
    k = mp.Vector3(0,0,1).rotate(mp.Vector3(0,1,0), theta_rad).scale(self.n1 * self.frequency_min)
```

The Fresnel formula for P-polarization checks `n1 * cos(theta_t)` vs `n2 * cos(theta_i)`:

```python
reflectance_fresnel = lambda theta_in: (
    fabs(
        (self.n1 * cos(theta_out(theta_in)) - self.n2 * cos(theta_in))
        / (self.n1 * cos(theta_out(theta_in)) + self.n2 * cos(theta_in))
    )
)**2
```

The tolerance `tol=0.03` (3% relative error) reflects the combined effects of finite resolution, finite cell size, and FDTD discretization.

#### Key Takeaways

- `bfast_scaled_k` enables fixed-angle broadband reflectance measurements, essential for comparing with angle-resolved spectroscopy experiments.
- The Courant number must be reduced when using BFAST to maintain numerical stability at oblique angles.
- P-polarization (TM, Ex source) has a Brewster angle where reflectance vanishes; s-polarization (TE, Ey source) does not.
- The parameterized test framework (`@parameterized.parameterized.expand`) runs the same test logic for multiple (theta, use_bfast) combinations automatically.
- A relative tolerance of 3% is appropriate for oblique-incidence reflectance at this resolution; higher resolution would reduce the error.

---

### 16. `test_materials_library.py` — Validation of Built-in Materials Library

**Physics:** Verifies that the built-in Meep materials library (InP, Ge, Si, LiNbO3, SiO2_aniso, Ag, Cr) returns permittivities that match tabulated experimental data at specific reference wavelengths.
**Difficulty:** Beginner
**Source:** `python/tests/test_materials_library.py`
**Test Status:** PASS (11.4s)

#### Theory

Meep's `meep.materials` module provides pre-fit Drude-Lorentzian material models for a library of common optical and electronic materials. Each material is fit to experimental data from sources such as refractiveindex.info, covering a specified wavelength range. The `Medium.epsilon(f)` method evaluates the permittivity tensor at any frequency (in Meep units of c/um) within the valid range.

The materials tested span several important classes:

**Semiconductors (InP, Ge, Si):** Crystalline semiconductors have refractive indices of 3-4 in the near-IR, modeled with multi-pole Lorentzian fits. Silicon at 1.55 um (n ≈ 3.48) is the canonical integrated photonics platform.

**Electro-optic crystal (LiNbO3):** Lithium niobate is a uniaxial crystal with different ordinary and extraordinary refractive indices. Its anisotropic permittivity tensor is diagonal with different values for the (x,y) components vs. the z-component. In Meep, this is represented with `sigma_diag` to assign different sigma values to different tensor components.

**Anisotropic glass (SiO2_aniso):** A version of fused quartz with an artificially induced birefringence, with slightly different ordinary and extraordinary indices. The off-diagonal permittivity components are zero (no optical activity).

**Metals (Ag, Cr):** Drude-dominated metals with strongly complex permittivity in the visible. Silver at 0.65 um has `epsilon ≈ (0.146 + 3.94i)^2` — large negative real part and significant loss. The test confirms that requesting epsilon outside the valid frequency range raises a `ValueError`.

#### Code Walkthrough

The test checks the permittivity tensor at reference wavelengths by calling `Material.epsilon(f)` where `f = 1/lambda_um`:

```python
from meep.materials import Ag, Cr, Ge, InP, LiNbO3, Si, SiO2_aniso

self.assertAlmostEqual(Si.epsilon(1/1.55)[0][0], (3.4777)**2, places=2)
```

For anisotropic materials, all three diagonal components are checked:

```python
self.assertAlmostEqual(LiNbO3.epsilon(1/1.55)[0][0], (2.2111)**2, places=2)
self.assertAlmostEqual(LiNbO3.epsilon(1/1.55)[2][2], (2.1376)**2, places=2)
```

For off-diagonal elements, the test confirms they are zero (no optical activity):

```python
self.assertEqual(SiO2_aniso.epsilon(1/1.55)[1][0], 0)
```

The boundary check confirms that the valid_freq_range is enforced:

```python
try:
    Ag.epsilon(1/0.2)[0][0]
except ValueError:
    pass  # Expected: Ag is not defined at 0.2 um
```

#### Key Takeaways

- `from meep.materials import Si, Ag, ...` imports pre-fit Drude-Lorentzian material objects for common materials.
- `Medium.epsilon(f)` returns a 3x3 complex permittivity tensor, accessed as `eps[row][col]`.
- Anisotropic materials (LiNbO3, SiO2_aniso) use `sigma_diag` in their Lorentzian terms to assign different pole strengths to each tensor component.
- `valid_freq_range` enforces the frequency range of the experimental data; outside this range, `epsilon()` raises `ValueError`.
- Metals (Ag, Cr) have complex epsilon (from Drude terms), returned as complex numbers; dielectrics have real epsilon.

---

### 17. `test_medium_evaluations.py` — Permittivity Evaluation and Frequency-Range Validation

**Physics:** Exercises the `Medium.epsilon()` method across different argument types (scalars, arrays, edge cases) and verifies permittivity values against reference data from refractiveindex.info for silicon, silver, and lithium niobate.
**Difficulty:** Beginner
**Source:** `python/tests/test_medium_evaluations.py`
**Test Status:** PASS (11.5s)

#### Theory

The `Medium.epsilon(f)` method is the primary interface for evaluating the Drude-Lorentzian permittivity model analytically, outside of a time-domain simulation. This is useful for:

1. **Verification:** Checking that the fitted material model matches the target experimental data.
2. **Plotting:** Generating permittivity dispersion curves over a frequency range.
3. **Preprocessing:** Computing effective indices or group velocities before running expensive FDTD simulations.

The frequency argument can be a scalar (returns a 3x3 tensor), a list (returns an array of 3x3 tensors indexed as `eps[freq_idx, row, col]`), or a numpy array. The valid frequency range is enforced, and out-of-range queries raise `ValueError`.

Key numerical checks in this test include:

- **Silicon** at its frequency range boundaries (0.4 um: n ≈ 3.50, and 1.0 um: n ≈ 3.42).
- **Silver** at UV and visible wavelengths. At very long wavelengths (IR), silver behaves as a Drude metal with n ≈ large imaginary value (dominant Im(n) >> Re(n)). At visible wavelengths, Im(n) is comparable to Re(n) and both contribute to reflectivity.
- **Lithium niobate** at both ordinary (x, y) and extraordinary (z) polarization components across its frequency range, confirming the anisotropy is modeled correctly.

#### Code Walkthrough

Scalar evaluation and the valid_freq_range property are checked together:

```python
w0 = LiNbO3.valid_freq_range.min
eps = LiNbO3.epsilon(w0)
self.assertAlmostEqual(np.real(np.sqrt(eps[0, 0])), 2.0508, places=4)
```

Array input is tested to verify that numpy array evaluation does not raise an exception (the test is just a smoke test for array support):

```python
eps = Si.epsilon(np.linspace(w0, w1, 100))
```

The distinction between real and complex permittivity is verified using `np.iscomplex`:

```python
self.assertTrue(np.iscomplex(Ag.epsilon(1.0)[0, 0]))
self.assertFalse(np.iscomplex(fused_quartz.epsilon(1.0)[0, 0]))
```

This reflects that silver's Drude poles contribute a complex-valued susceptibility even away from any resonance, while fused quartz's lossless Lorentzians (gamma=0) produce purely real permittivity values.

For array input returning multiple frequency results, indexing uses `eps[freq_idx, row, col]`:

```python
eps = Si.epsilon([w0, w1])
self.assertAlmostEqual(np.real(np.sqrt(eps[0, 0, 0])), 3.4175, places=4)
self.assertAlmostEqual(np.real(np.sqrt(eps[1, 0, 0])), 3.4971, places=4)
```

#### Key Takeaways

- `Medium.epsilon(f)` accepts a scalar, list, or numpy array of frequencies and returns the 3x3 permittivity tensor at each.
- The indexing convention for array input is `eps[freq_idx, row, col]`; for scalar input it is `eps[row][col]`.
- `valid_freq_range.min` and `.max` expose the frequency limits of the material fit.
- `np.iscomplex()` distinguishes lossy (complex eps) from lossless (real eps) materials; lossless Lorentzians with gamma=0 return real permittivities.
- `Medium.epsilon()` evaluates the model analytically (not via FDTD) and is fast — it is safe to call in tight loops for dispersion plotting or preprocessing.

---

## Summary

This chapter has presented material modeling in Meep across the full spectrum of complexity, from simple isotropic Lorentzian media through to gyrotropic magneto-optical systems and fitted experimental data. The key themes that connect all 17 tutorials are:

**Drude-Lorentzian universality.** Every dispersive material in Meep is ultimately represented as a sum of Lorentzian and Drude poles. This causal form enables exact time-stepping via auxiliary polarization fields and guarantees Kramers-Kronig compliance. The parameters (sigma, omega_0, gamma) map directly to physical resonance properties.

**The two-run subtraction pattern.** Computing reflectance or any relative optical quantity requires first a reference run (empty cell) and then a structure run, with the reference subtracted from the flux monitor. This eliminates the incident wave contribution and isolates the scattered or reflected signal.

**k-point and Bloch boundaries for oblique incidence.** Setting `k_point` imposes a transverse wavevector component that implements oblique incidence without needing to simulate a large transverse extent. BFAST extends this to fixed-angle broadband calculations. The `kz_2d` parameter further reduces full 3D oblique problems to 1D or 2D simulations.

**Validation against analytic benchmarks.** Every quantitative result in this chapter is compared against an analytic formula: Fresnel equations, Lorentzian dispersion relations, Faraday rotation rates, exponential attenuation. This disciplined comparison between simulation and theory is the foundation of confident FDTD modeling.

**Materials library as a starting point.** For common optical materials (Si, Ge, Ag, fused quartz, LiNbO3), the built-in library provides immediately usable models. For custom materials, `eps_fit_lorentzian.py` demonstrates how to fit measured n(lambda) data to the Drude-Lorentzian form required by Meep.
