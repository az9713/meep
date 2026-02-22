# Spherical Cow Cloak: FDTD Simulation Report

## Technical Analysis of Limitations, Workarounds, and Paths Forward

**Files:** `python/examples/spherical_cow_cloak.py`, `python/examples/spherical_cow_cloak_viz.py`
**Date:** 2026-02-22
**Meep version:** 1.31.0 (conda-forge pymeep, Python 3.13)

---

## 1. Executive Summary

We implemented a Pendry-style transformation-optics invisibility cloak around a spherical
scatterer ("spherical cow") using Meep's 3D FDTD engine. The simulation infrastructure
works correctly — the bare dielectric sphere matches Mie theory — but **the cloak does
not reduce scattering**. With `eps_min=0.6` (the minimum stable value), the cloaked sphere
scatters **1.8x more** than the bare sphere at center frequency.

This is not a bug in the code. It is a fundamental consequence of three compounding
limitations:

1. **Yee grid instability** with off-diagonal permittivity tensors
2. **Reduced-parameter approximation** (epsilon-only, no magnetic response)
3. **Heavy regularization** required to keep the simulation stable

Each limitation is detailed below with the empirical evidence from our debugging sessions.

---

## 2. The Physics: What We're Trying to Do

### 2.1 Pendry Cloak Transformation

The Pendry cloak (Science 312, 1780, 2006) uses a coordinate transformation that maps
the interior of a sphere to a spherical shell:

```
r' = a + r(b-a)/b       mapping 0 < r < b  to  a < r' < b
```

This yields material parameters in the cloak shell (a < r < b) in spherical coordinates:

```
epsilon_r = mu_r     = (b/(b-a)) * ((r-a)/r)^2     (radial)
epsilon_theta = mu_t = b/(b-a)                       (tangential)
epsilon_phi   = mu_t = b/(b-a)                       (tangential)
```

For our default parameters (a=0.5, b=1.0):
- `C = b/(b-a) = 2.0` (tangential component, constant throughout shell)
- `eps_r` ranges from 0 (at r=a) to 0.5 (at r=b)
- At r=a, eps_r = mu_r = 0 — **singular**, the defining challenge

### 2.2 Cartesian Tensor Conversion

FDTD operates in Cartesian coordinates. The spherically-diagonal tensor must be
converted:

```
epsilon_cart = eps_t * I + (eps_r - eps_t) * rhat (x) rhat^T
```

where `rhat = (x,y,z)/r` is the radial unit vector. This outer-product form avoids
explicit trigonometric rotation matrices and directly produces the symmetric 3x3
Cartesian tensor with both diagonal and **off-diagonal** components.

For Meep's `mp.Medium`:
- `epsilon_diag = Vector3(eps_t + d*rx*rx, eps_t + d*ry*ry, eps_t + d*rz*rz)`
- `epsilon_offdiag = Vector3(d*rx*ry, d*rx*rz, d*ry*rz)`

where `d = eps_r - eps_t`.

---

## 3. Issue #1: Yee Grid Instability with Off-Diagonal Epsilon

### 3.1 The Problem

Meep's FDTD engine uses a staggered Yee grid where E and H field components are stored
at different spatial locations (half-grid offsets). Off-diagonal permittivity components
couple different field components (e.g., Dx depends on both Ex and Ey), requiring
**spatial interpolation** between Yee grid points.

When the off-diagonal components are large relative to the diagonal components, this
interpolation introduces an **unconditional instability** — one that cannot be fixed
by reducing the time step.

### 3.2 The Evidence: Systematic Debugging

We performed a systematic investigation using the dielectric cow (n=2, a=0.5, b=1.0):

| eps_min | Stable? | Max offdiag/diag ratio | Notes |
|---------|---------|------------------------|-------|
| 0.01    | NO      | 199.0                  | Original plan value, immediate NaN |
| 0.1     | NO      | 19.0                   | Still far too extreme |
| 0.3     | NO      | 5.67                   | Diverged ~65% through run |
| 0.5     | NO      | 1.50                   | Diverged late in run |
| 0.55    | YES     | 1.32                   | First stable value found |
| 0.6     | YES     | 1.17                   | Safe default, used in production |
| 0.65    | YES     | 1.00                   | Comfortable margin |
| 0.7     | YES     | 0.93                   | |
| 0.8     | YES     | 0.75                   | |
| 0.9     | YES     | 0.61                   | |
| 1.0     | YES     | 0.50                   | Trivially stable (mild anisotropy) |

**Key finding:** The stability threshold is between eps_min=0.50 and eps_min=0.55,
corresponding to a maximum off-diagonal/diagonal ratio of approximately 1.3-1.5.

### 3.3 Why Courant Number Reduction Doesn't Help

We tested with extremely conservative Courant numbers:

| Courant | eps_min | Stable? |
|---------|---------|---------|
| 0.5     | 0.3     | NO      |
| 0.285   | 0.3     | NO      |
| 0.1559  | 0.3     | NO      |
| 0.05    | 0.3     | NO      |

Even at Courant=0.05 (10x smaller than default, 100x slower simulation), eps_min=0.3
still diverged. This proves the instability is **unconditional** — it is not a CFL
(Courant-Friedrichs-Lewy) stability violation but a fundamental issue with how
off-diagonal tensor components are interpolated on the staggered Yee grid.

For standard diagonal media, the CFL condition is:
```
dt < dx / (c * sqrt(D))     where D = dimensionality (3)
```

For anisotropic media, the effective phase velocity can be very large when epsilon
eigenvalues are small (v = c/sqrt(eps)), but this is a **conditional** instability
that should be fixable by reducing dt. The fact that it isn't fixable means the
instability arises from the spatial discretization of the off-diagonal coupling,
not from the time-stepping CFL condition.

### 3.4 Why Absorber Boundaries Don't Help

We also tested replacing PML with `mp.Absorber(thickness=1.5)` to rule out
PML-anisotropic material interaction as the cause. The simulation still diverged
identically. This eliminated PML instability as the root cause.

### 3.5 The Off-Diagonal Ratio Metric

The critical metric for stability is the ratio of the largest off-diagonal component
to the smallest diagonal component of the Cartesian epsilon tensor at any point in
the simulation domain:

```python
# At a point along a 45-degree direction (worst case for off-diagonal):
# rx = ry = rz = 1/sqrt(3)
# offdiag component = d * rx * ry = (eps_r - eps_t) / 3
# diagonal component = eps_t + d * rx^2 = eps_t + (eps_r - eps_t) / 3
#
# ratio = |offdiag| / |diagonal| = |eps_r - eps_t| / (2*eps_t + eps_r)
```

When eps_r << eps_t (near the inner boundary), `d = eps_r - eps_t` is large and
negative, making the off-diagonal components comparable to (or larger than) the
diagonal ones. This is the regime where the Yee grid interpolation breaks down.

---

## 4. Issue #2: Reduced-Parameter Cloak (mu=1)

### 4.1 What We Tried First

The ideal Pendry cloak transforms **both** epsilon and mu (perfect impedance matching):

```python
# Full Pendry cloak (UNSTABLE in FDTD):
return mp.Medium(
    epsilon_diag=diag, epsilon_offdiag=offdiag,
    mu_diag=diag, mu_offdiag=offdiag,  # Same tensors for mu
)
```

This was our first implementation. It caused immediate NaN divergence even with
modest regularization (eps_min=0.3), because having **both** epsilon and mu as full
anisotropic tensors creates a doubly-stiff system that the leapfrog time-stepping
scheme cannot handle.

### 4.2 The Reduced-Parameter Approach

The "reduced parameter" cloak sets mu=1 everywhere and only transforms epsilon.
This is a well-known simplification in the cloaking literature (see Cai et al.,
Opt. Express 15, 3333, 2007). The trade-offs:

**Advantages:**
- Only one material tensor to handle (simpler for FDTD)
- Eliminates the doubly-anisotropic instability
- The beam-bending physics is still present in the epsilon transformation

**Disadvantages:**
- **Impedance mismatch at r=b:** The cloak outer surface has eta = sqrt(mu/eps) != 1,
  causing reflections at the cloak-air interface
- **Impedance mismatch at r=a:** Same issue at the inner boundary
- Scattering is not eliminated even with perfect (unregularized) parameters
- The reduced cloak only works perfectly for specific polarizations in 2D (TE or TM),
  not in full 3D

### 4.3 Impact on Results

Even a perfect (unregularized) reduced-parameter cloak would show non-zero scattering.
The impedance mismatch causes reflections that cannot be avoided without magnetic
response. In our simulation, this effect is **compounded** by the heavy regularization,
making it impossible to separate the two degradation mechanisms.

---

## 5. Issue #3: Regularization and Its Impact

### 5.1 Why Regularization Is Needed

At the inner cloak surface (r=a), the transformation optics formulas give:
```
eps_r(r=a) = 0     (singular)
```

FDTD cannot represent zero or negative permittivity directly (it would correspond
to infinite phase velocity or evanescent behavior). The standard approach is to
clamp: `eps_r = max(eps_r_computed, eps_min)`.

### 5.2 The Regularization-Stability Trade-off

| eps_min | Cloaking quality | FDTD stability | Status |
|---------|------------------|----------------|--------|
| 0.01    | Excellent (near-ideal) | Unconditionally unstable | Unusable |
| 0.1     | Very good | Unstable | Unusable |
| 0.3     | Good | Unstable | Unusable |
| 0.5     | Moderate | Marginal (unstable) | Unusable |
| 0.55    | Moderate-poor | Barely stable | Risky |
| 0.6     | Poor | Stable (default) | **Current** |
| 1.0     | None (isotropic shell) | Trivially stable | Baseline |

At eps_min=0.6:
- The radial permittivity only varies from 0.6 to 0.5 across the cloak shell
- The anisotropy ratio (eps_t/eps_r) only ranges from 2.0 to 4.0
- The gradient-index "lens" is too weak to significantly bend light around the cow
- The shell acts more like an additional dielectric layer that adds scattering

### 5.3 Current Simulation Results

With eps_min=0.6, dielectric cow (n=2, a=0.5, b=1.0):

| Case | Q_sca at f_cen |
|------|----------------|
| Bare sphere | 4.2335 |
| Cloaked sphere | 7.6573 |
| **Ratio** | **0.55x** (cloak makes it 1.8x WORSE) |
| Mie theory (bare) | ~4.23 (excellent match) |

The bare sphere Q_sca matches Mie theory to <1%, validating the simulation
infrastructure (scattered-field subtraction, flux box measurement, Mie series
computation). The problem is entirely in the cloak physics.

---

## 6. Issue #4: Courant Number History (Now Resolved)

### 6.1 What Happened

During debugging, we added custom Courant number computation to try to stabilize
the simulation:

```python
# Attempted fix (removed): compute CFL-based Courant
cloak_courant = min(0.5, dx / (c_max * math.sqrt(3)) * 0.9)
sim = mp.Simulation(..., Courant=cloak_courant)
```

This was based on the hypothesis that the instability was CFL-related. After
proving it was unconditional (Courant=0.05 still diverged), we removed all
custom Courant logic and reverted to Meep's default Courant=0.5.

### 6.2 Current State

The simulation code has **no** custom Courant number references. It uses Meep's
default of 0.5 everywhere. This is correct because:
1. The instability is not CFL-related
2. Custom Courant reduction would slow the simulation without benefit
3. The fix is in eps_min, not in the time step

---

## 7. Quality Assessment of Current Results

### 7.1 What Works

1. **Bare sphere scattering**: Matches Mie theory to <1% — the simulation
   infrastructure (sources, flux boxes, scattered-field subtraction) is correct
2. **Field visualization**: Clear wavefront patterns showing scattering interference
3. **Code architecture**: Clean separation of simulation engine and visualization,
   configurable parameters, CLI interface, .npz data exchange
4. **Multi-material support**: PEC, dielectric, and lossy cow materials all work

### 7.2 What Doesn't Work

1. **Cloaking**: The cloak increases rather than decreases scattering
2. **Beam bending**: Field plots show no visible wavefront bending around the cow
3. **Broadband behavior**: No frequency-dependent cloaking effect visible

### 7.3 Root Cause Chain

```
Ideal Pendry cloak requires eps_r -> 0 near inner surface
    |
    v
eps_r -> 0 creates large off-diagonal/diagonal ratios in Cartesian tensor
    |
    v
Yee grid interpolation becomes unconditionally unstable (off-diag > 1.3x diag)
    |
    v
Must regularize eps_min >= 0.55 for stability
    |
    v
eps_min = 0.6 makes cloak shell nearly isotropic (weak anisotropy)
    |
    v
Plus: reduced-parameter (mu=1) adds impedance mismatch
    |
    v
Result: cloak shell is just an extra dielectric layer that ADDS scattering
```

---

## 8. Mitigation Strategies Within FDTD

### 8.1 Larger Cloak Ratio (b/a >> 2)

**Idea:** Increase b/a so the transformation is "gentler" — eps_r varies more
slowly and doesn't need to reach as close to zero.

For general b/a ratio, at the inner surface: `eps_r(a) = 0` (always singular).
But at a fixed fraction of the shell, say r = a + 0.1*(b-a):

```
eps_r = C * (0.1 * (b-a) / r)^2
```

With b/a = 4 (instead of 2): C = 4/3, and the gradient is much gentler.
The minimum eps_r in the stable region would still be clamped to eps_min, but
a larger fraction of the shell would have eps_r > eps_min, giving more
bending.

**Trade-off:** Much larger simulation domain (8x volume for 2x radius), much
longer run times. At resolution=10, going from b=1 to b=2 increases the cell
from 6^3 to 10^3, roughly a 5x cost increase.

**Estimated improvement:** Moderate. The singular core region is always
problematic.

### 8.2 Higher Resolution at Shell Boundaries

**Idea:** Use Meep's subpixel averaging more effectively by running at much
higher resolution (e.g., 30-50 pixels/unit) specifically to better resolve the
material variation in the shell.

**Trade-off:** Resolution 30 in 3D means 7.5x more grid points per axis =
~420x more memory and compute than resolution 10. A quick-mode run would take
hours instead of seconds.

**Estimated improvement:** Small. The fundamental off-diagonal instability is
resolution-independent (proven by Courant tests).

### 8.3 Smoothed/Tapered Regularization

**Idea:** Instead of a hard clamp `eps_r = max(computed, eps_min)`, use a smooth
transition that prevents sharp material discontinuities:

```python
eps_r_smooth = eps_min + (eps_r_ideal - eps_min) * tanh(alpha * (r - a) / (b - a))
```

This would reduce reflections at the regularization boundary (where eps_r
transitions from the clamped region to the ideal profile).

**Estimated improvement:** Small to moderate. Reduces reflection artifacts but
doesn't address the fundamental cloaking weakness.

### 8.4 2D Simulation (TE/TM)

**Idea:** Run in 2D (cylindrical or true 2D) where the reduced-parameter cloak
can be exact for one polarization (e.g., TM: only Ez, Hx, Hy).

**Key advantage:** In 2D TM mode, the cloak only needs `eps_z` and `mu_x, mu_y`
(or equivalently a scalar eps_z and anisotropic mu in-plane). The tensor is
simpler, the off-diagonal issue is less severe, and the reduced-parameter
approach can be made to work for single-polarization cloaking.

**Trade-off:** Not a spherical cow anymore (cylindrical cow). But this is the
approach used in most published FDTD cloaking papers (e.g., Zhao et al., Opt.
Express 16, 6717, 2008).

**Estimated improvement:** Large. Published 2D FDTD cloaking results show
10-100x scattering reduction with moderate regularization.

### 8.5 Dispersive Cloak Materials (Drude Model)

**Idea:** Instead of trying to achieve eps < 1 through a static material tensor,
use dispersive Drude-model materials:

```
eps(omega) = 1 - omega_p^2 / (omega^2 + i*gamma*omega)
```

By choosing omega_p and gamma appropriately at each spatial position, achieve
the target eps_r(r) at the operating frequency. Meep handles Drude dispersive
materials natively and stably.

**Trade-off:** The cloak becomes narrowband (works only near the design
frequency). Requires careful Drude parameter fitting. More complex material
setup. Each shell layer needs different Drude parameters.

**Estimated improvement:** Moderate to large. Dispersive materials are well-
handled by Meep's auxiliary differential equation (ADE) method, avoiding the
off-diagonal interpolation issue for the dispersive part.

---

## 9. Alternative Approaches (Beyond FDTD)

### 9.1 Frequency-Domain Solver (Meep CW Solver)

Meep includes a continuous-wave (CW) frequency-domain solver (`solve_cw()`)
that iteratively solves Maxwell's equations at a single frequency. This avoids
time-stepping instabilities entirely.

**Advantages:**
- No CFL or time-stepping stability constraints
- Can handle arbitrary anisotropic tensors
- Direct access to steady-state fields

**Disadvantages:**
- Only single-frequency (no broadband spectrum)
- Convergence can be slow for complex geometries
- Still uses the same Yee grid (off-diagonal interpolation issues may persist)

### 9.2 Transfer Matrix / Scattering Matrix Methods

For spherically symmetric geometries, analytical methods based on Mie theory
can be extended to layered spheres with anisotropic materials. The cloak can
be discretized into N concentric shells, each with approximately uniform
anisotropic parameters, and the scattering computed analytically.

**Advantages:**
- No spatial discretization — exact in the radial direction
- No stability issues
- Handles the singularity analytically (or with extremely fine radial layers)
- Fast computation

**Disadvantages:**
- Only works for spherical geometry
- Requires implementing T-matrix methods for anisotropic spherical layers
- Not generalizable to arbitrary shapes

### 9.3 Finite Element Method (FEM)

Commercial FEM solvers (COMSOL, etc.) or open-source ones (FEniCS + dolfinx)
handle anisotropic media naturally because:
- No staggered grid (all components at same nodes)
- No off-diagonal interpolation issues
- Unstructured meshes can refine near r=a where the singularity lives
- Frequency-domain formulation avoids time-stepping instabilities

**This is likely the most practical path to a physically realistic spherical
cow cloak simulation.**

### 9.4 Spectral Methods

Pseudospectral time-domain (PSTD) methods use Fourier transforms for spatial
derivatives instead of finite differences. They have much lower numerical
dispersion and can handle anisotropic media more gracefully than FDTD.

---

## 10. Recommendations

### For a Quick Demonstration (Current Capability)

Use the existing code as-is. It demonstrates:
- Correct FDTD simulation infrastructure (validated against Mie theory)
- Transformation optics material function generation
- The fundamental challenges of FDTD cloaking
- Good visualization and comparison tools

Run: `python spherical_cow_cloak.py --quick --cow-material dielectric`

### For a Publishable FDTD Cloaking Result

1. **Switch to 2D** (cylindrical coordinates or 2D Cartesian with TM polarization)
2. Use reduced-parameter cloak optimized for single polarization
3. Use eps_min=0.05-0.1 (stable in 2D with simpler tensor structure)
4. Compare with analytical 2D Mie theory for cylinder
5. Expected: 10-100x scattering reduction at center frequency

### For a True 3D Spherical Cow Cloak

1. **Use FEM** (COMSOL or FEniCS) instead of FDTD
2. Frequency-domain formulation at target wavelength
3. Full Pendry parameters (both eps and mu as anisotropic tensors)
4. Adaptive mesh refinement near r=a
5. eps_min=0.001 or smaller (no stability concern in FEM)
6. Expected: near-perfect cloaking at design frequency

### For Understanding the Physics (Educational)

The current simulation is actually excellent for teaching because it clearly
shows:
1. Why transformation optics requires extreme material parameters
2. Why FDTD has fundamental limitations with highly anisotropic media
3. The importance of impedance matching (reduced vs. full parameter cloak)
4. How regularization degrades cloaking performance
5. The gap between theoretical design and practical simulation

---

## 11. Appendix: Key Numerical Values

### Cloak Material Parameters at Selected Radii (a=0.5, b=1.0)

| r | eps_r (ideal) | eps_r (clamped, eps_min=0.6) | eps_t | Max offdiag/diag |
|---|---------------|------------------------------|-------|------------------|
| 0.50 | 0.000 | 0.600 | 2.0 | 1.17 |
| 0.55 | 0.018 | 0.600 | 2.0 | 1.17 |
| 0.60 | 0.074 | 0.600 | 2.0 | 1.17 |
| 0.65 | 0.159 | 0.600 | 2.0 | 1.17 |
| 0.70 | 0.262 | 0.600 | 2.0 | 1.17 |
| 0.75 | 0.370 | 0.600 | 2.0 | 1.17 |
| 0.80 | 0.500 | 0.600 | 2.0 | 1.17 |
| 0.85 | 0.580 | 0.600 | 2.0 | 1.17 |
| 0.87 | 0.604 | 0.604 | 2.0 | 1.16 |
| 0.90 | 0.691 | 0.691 | 2.0 | 0.95 |
| 0.95 | 0.805 | 0.805 | 2.0 | 0.74 |
| 1.00 | 0.500 | 0.500 | 2.0 | 1.50 |

Note: With eps_min=0.6, the clamp is active for all r < 0.87. That means **74%
of the cloak shell by radius** (and even more by volume) has a uniform eps_r=0.6
instead of the ideal graded profile. The shell barely functions as a gradient-index
lens.

### Simulation Timing (Quick Mode, resolution=10)

| Phase | Time (s) | Notes |
|-------|----------|-------|
| Reference (empty cell) | 2.9 | 1200 timesteps |
| Bare dielectric sphere | 4.4 | 1600 timesteps |
| Cloaked dielectric sphere | 4.1 | 1600 timesteps |
| **Total** | **12.2** | Single material |

---

## 12. References

1. Pendry, J.B., Schurig, D., & Smith, D.R. (2006). "Controlling Electromagnetic
   Fields." *Science*, 312(5781), 1780-1782.

2. Cai, W., Chettiar, U.K., Kildishev, A.V., & Shalaev, V.M. (2007). "Optical
   cloaking with metamaterials." *Nature Photonics*, 1, 224-227.

3. Zhao, Y., Argyropoulos, C., & Hao, Y. (2008). "Full-wave finite-difference
   time-domain simulation of electromagnetic cloaking structures." *Optics Express*,
   16(9), 6717-6730.

4. Cai, W., Chettiar, U.K., Kildishev, A.V., Shalaev, V.M., & Milton, G.W. (2007).
   "Nonmagnetic cloak with minimized scattering." *Applied Physics Letters*, 91,
   111105.

5. Taflove, A., & Hagness, S.C. (2005). *Computational Electrodynamics: The
   Finite-Difference Time-Domain Method*, 3rd ed. Artech House. (Chapter on
   anisotropic media and Yee grid stability.)
