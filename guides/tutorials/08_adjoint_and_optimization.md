# Chapter 8: Adjoint Optimization and Inverse Design

This chapter covers topology optimization and inverse design techniques in Meep using the adjoint method, including waveguide device optimization, multilayer film design, and JAX-based automatic differentiation. The adjoint approach makes gradient-based optimization of electromagnetic structures computationally tractable by reducing the cost of a full gradient computation — over thousands of design parameters — to just two FDTD simulations regardless of parameter count. These tutorials progress from verifying the mathematical foundations of the adjoint gradient through complete inverse design workflows for photonic devices.

---

### 1. `waveguide_crossing.py` — Topology and Shape Optimization of a Waveguide Crossing

**Physics:** Finds the optimal dielectric structure inside a square design region that maximizes transmission of the fundamental waveguide mode across a 90-degree crossing, comparing shape optimization (starting from a physical cross) against full topology optimization (starting from a uniform grayscale).
**Difficulty:** Advanced
**Source:** `python/examples/waveguide_crossing.py`
**Test Status:** FAIL (amplitude TypeError — `EigenModeSource` amplitude type mismatch in current environment)

#### Theory

A waveguide crossing is one of the canonical benchmark problems in photonic inverse design. Two waveguides intersecting at 90 degrees couple light from the horizontal into the vertical guide, causing insertion loss and crosstalk. Without optimization, the simple overlap region scatters power into all four ports and into radiation modes. The design challenge is to find a permittivity distribution epsilon(x, y) inside a square design region that routes light from one horizontal port to the other with minimum loss, while simultaneously maintaining C4 symmetry so that the device is reciprocal under 90-degree rotations.

The problem is posed as minimizing the figure of merit (FOM):

```
J = 1 - |S_21|^2 = 1 - |output_mode_coefficient / input_mode_coefficient|^2
```

where S_21 is the complex transmission coefficient from input to output, computed by eigenmode decomposition on both sides of the design region. A value of J = 0 means perfect transmission; J = 1 means total blockage. The adjoint method allows the gradient dJ/d_rho — where rho is the vector of N_x * N_y density weights — to be computed from exactly one forward simulation and one adjoint (backward-propagating) simulation, making gradient-based optimization tractable even when the design region contains tens of thousands of pixels.

The density-to-permittivity map is not applied directly. Instead, a two-stage filter-and-project pipeline converts the raw design weights rho into a physically realizable permittivity field. First, a conic spatial filter with radius r_f smooths the raw design to enforce a minimum length scale: features smaller than the filter radius are suppressed. The filter radius is related to the minimum feature size L_min and the erosion threshold eta_e by:

```
r_f = get_conic_radius_from_eta_e(L_min, eta_e)
```

Second, a tanh-based projection function maps the smoothed field toward a binary 0-1 distribution:

```
rho_projected = tanh(beta * (rho_filtered - eta)) / tanh(beta * eta)
```

As the projection strength beta grows from 0 toward infinity, the design transitions from a smooth grayscale (beta = 0) to a hard binary mask (beta -> inf). Topology optimization exploits this by beginning with a small beta and progressively increasing it across multiple epochs — a continuation strategy. The key difficulty with large beta is that the gradient of the tanh projection becomes exponentially small far from the threshold eta, making the optimization ill-conditioned. The smoothed projection function `mpa.smoothed_projection` addresses this by replacing the standard tanh with a filtered version whose gradient remains well-conditioned for all values of beta. The code demonstrates both approaches and compares the gradient norm as a function of beta.

Two optimization strategies are demonstrated. Shape optimization starts from a physically meaningful initial guess — a simple cross matching the waveguide geometry — and uses a high fixed beta to immediately drive toward a near-binary design. Topology optimization starts from a uniform gray initial guess (rho = 0.5 everywhere) and uses a beta-continuation schedule, evolving the design through grayscale stages before binarizing. The latter approach explores a broader design space and typically discovers non-intuitive structures, but requires more simulation time.

#### Code Walkthrough

The build function creates the core simulation geometry. Silicon waveguides are represented by two overlapping `mp.Block` objects (one horizontal, one vertical), and the design region is a `mp.MaterialGrid` object embedded in a third block:

```python
matgrid = mp.MaterialGrid(
    mp.Vector3(Nx, Ny),
    mp.air,
    silicon,
    weights=np.ones((Nx, Ny)),
    beta=0,            # disable internal smoothing — handled externally
    do_averaging=False,
    damping=damping,
)
matgrid_region = mpa.DesignRegion(
    matgrid,
    volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(dx, dy, 0)),
)
```

The design resolution is set at twice the simulation resolution (`design_region_resolution = int(2 * resolution)`) so that the design grid is finer than the Yee grid, enabling subpixel-accurate feature representation.

The objective is registered with two `mpa.EigenmodeCoefficient` monitors: one measuring the backward-propagating mode amplitude at the input port (which captures any reflected power) and one measuring the forward-propagating mode amplitude at the output port. The objective function itself is straightforward:

```python
def J(input, output):
    return 1 - npa.power(npa.abs(output / input), 2)
```

The `mpa.OptimizationProblem` object wraps the simulation and objectives:

```python
opt = mpa.OptimizationProblem(
    simulation=sim,
    maximum_run_time=500,
    objective_functions=J,
    objective_arguments=obj_list,
    design_regions=[matgrid_region],
    frequencies=frequencies,
)
```

The mapping function applies the full filter-project-symmetrize pipeline. C4 symmetry is enforced analytically by averaging the design with its three 90-degree rotations before projection:

```python
def mapping(x):
    x = x.reshape(Nx, Ny)
    x = mpa.conic_filter(x, filter_radius, dx, dy, design_region_resolution)
    x = (x + npa.rot90(x) + npa.rot90(x, 2) + npa.rot90(x, 3)) / 4
    x = mpa.smoothed_projection(x, beta=beta, eta=eta,
                                 resolution=design_region_resolution)
    return x.flatten()
```

The NLopt optimization loop calls the FOM wrapper which computes both the objective value and the gradient via backpropagation through the mapping function:

```python
f0, dJ_du = opt([mapping(x)])
backprop_gradient = tensor_jacobian_product(mapping, 0)(x, dJ_du)
```

Here `tensor_jacobian_product` (from the `autograd` library) computes the vector-Jacobian product (VJP) of `mapping` with respect to its first argument, evaluated at `(x, dJ_du)`. This is exactly the chain-rule step that backpropagates the adjoint gradient from the mapped design space back to the raw parameter space.

Topology optimization wraps the above in a beta-continuation loop:

```python
for beta in beta_evolution:
    opt, mapping = build_optimization_problem(resolution, beta, ...)
    x0 = solver.optimize(x0)
```

Each epoch starts from the best design of the previous epoch, with a higher beta sharpening the design toward binary.

The `analyze_gradient_convergence` function at the bottom of the script runs a controlled experiment showing how gradient norm varies with beta for both the standard tanh projection and the smoothed projection. With standard projection, the gradient norm collapses to zero as beta increases (gradient vanishing), while with smoothed projection, the gradient remains bounded.

#### Key Takeaways

- The adjoint gradient requires only two FDTD simulations (forward + adjoint) regardless of how many design parameters are used, making it far more efficient than finite differences for large designs.
- The filter-project pipeline (conic filter then tanh projection) is essential for manufacturing feasibility: the filter enforces minimum feature size and the projection drives the design toward binary silicon/air.
- Symmetry constraints (C4 here) can be enforced analytically inside the mapping function rather than as optimization constraints, reducing degrees of freedom and improving convergence.
- Beta continuation — progressively increasing the projection strength beta across multiple epochs — is a standard technique for avoiding poor local minima in topology optimization.
- The smoothed projection function maintains well-conditioned gradients for all values of beta, enabling direct optimization at high beta without the gradient vanishing that afflicts standard tanh projection.

---

### 2. `mode_converter.py` — Worst-Case Broadband Waveguide Mode Converter

**Physics:** Designs a 2D silicon/SiO2 waveguide structure that converts the fundamental (TE1) mode into the second-order (TE2) mode with low reflectance and high transmittance across a 30 nm wavelength band from 1.26–1.30 µm, using worst-case (minimax) epigraph optimization.
**Difficulty:** Advanced
**Source:** `python/examples/adjoint_optimization/mode_converter.py`
**Test Status:** TIMEOUT (CPU-intensive — multiple epochs of 50 px/µm resolution topology optimization)

#### Theory

Mode converters are essential building blocks in photonic integrated circuits, enabling signals encoded in one spatial mode to be transferred to another. A TE1-to-TE2 converter must simultaneously minimize reflectance of the TE1 input mode (preventing back-reflections) and maximize transmittance into the TE2 output mode (maximizing the conversion). Achieving this over a 30 nm bandwidth requires the design to work well at six discrete design wavelengths: 1.265, 1.270, 1.275, 1.285, 1.290, and 1.295 µm.

Broadband optimization creates a multi-objective problem: the figure of merit has 2 * 6 = 12 scalar values (reflectance R and 1 - transmittance T at each wavelength). Rather than minimizing a weighted sum of these objectives — which requires manual weight tuning — the code uses the epigraph formulation, a standard technique from convex optimization for minimax problems. An auxiliary variable t (the epigraph variable) is introduced, and the optimization becomes:

```
minimize   t
subject to R(lambda_i) <= t   for all i
           1 - T(lambda_i) <= t  for all i
```

This is equivalent to minimizing the worst-case value over all wavelengths. The epigraph variable t is added to the design vector as its first element, and the 12 spectral constraints are passed to NLopt as inequality constraints. Because the constraints are differentiable with respect to the design weights (via the adjoint method) and the constraint function computes its own gradient in-place, NLopt's gradient-based CCSA (Conservative Convex Separable Approximation) algorithm can handle this formulation directly.

The minimum feature size constraint is added during the final optimization epoch. The `mpa.constraint_solid` and `mpa.constraint_void` functions compute measures of the solid (silicon) and void (SiO2) feature sizes respectively, and their gradients are computed via `autograd`'s `grad` function. These constraints ensure that the final optimized structure can be fabricated using standard lithographic processes with a minimum feature size of 150 nm.

Subpixel smoothing (`do_averaging=True` in `MaterialGrid`) is activated at high values of the sigmoid bias (beta >= 64) to reduce staircasing artifacts at material interfaces. At lower beta values, the design remains grayscale enough that subpixel averaging would introduce errors, so it is disabled during the early epochs.

The material damping feature of `MaterialGrid` is enabled throughout (`use_damping=True`). A small imaginary part is added to the permittivity of intermediate (grayscale) pixels proportional to the gradient of the sigmoid function:

```
damping = 0.02 * 2 * pi * frequency_center
```

This penalizes gray regions during optimization, progressively driving the design toward fully binary silicon/SiO2 as the optimization proceeds.

#### Code Walkthrough

The cell geometry places the 1.6 µm x 1.6 µm design region between two 3.0 µm long waveguide arms, all embedded in SiO2 cladding:

```python
cell_um = mp.Vector3(
    PML_UM + WAVEGUIDE_UM.x + DESIGN_REGION_UM.x + WAVEGUIDE_UM.x + PML_UM,
    PML_UM + PADDING_UM + DESIGN_REGION_UM.y + PADDING_UM + PML_UM,
    0,
)
```

Before optimization begins, a normalization run simulates a straight waveguide (no design region) to measure the incident mode flux at all six frequencies. These fluxes are stored and used to normalize the reflectance and transmittance during optimization:

```python
input_flux, input_flux_data = straight_waveguide()
```

The `input_flux_data` (a DFT fields snapshot) is also subtracted from the reflected fields during optimization, isolating the scattered reflection from the geometric reflection.

The two objective functions J1 and J2 are:

```python
def J1(refl_mon, tran_mon):
    return npa.power(npa.abs(refl_mon), 2) / input_flux   # reflectance R

def J2(refl_mon, tran_mon):
    return 1 - npa.power(npa.abs(tran_mon), 2) / input_flux  # 1 - transmittance
```

The epigraph constraint function calls `opt(...)` to run the forward and adjoint simulations, then backpropagates the gradient through the filter-and-project mapping:

```python
for k in range(2 * num_wavelengths):
    grad[:, k] = tensor_jacobian_product(filter_and_project, 0)(
        weights, sigmoid_threshold, sigmoid_bias, grad[:, k]
    )
gradient[:, 0] = -1     # d(constraint)/d(epigraph) = -1
gradient[:, 1:] = grad.T
result[:] = np.real(obj_val_merged) - epigraph
```

The beta-continuation schedule runs six epochs with sigmoid_bias values of [8, 16, 32, 64, 128, 256]. At each epoch, a forward run initializes the epigraph variable to slightly above the current worst-case objective value, ensuring the constraints start satisfied:

```python
epigraph_and_weights[0] = np.max(epigraph_initial)
```

Border masks enforce that the design is pinned to silicon at the waveguide port entries and to SiO2 at the non-port boundaries, preventing artifacts at the design region edges:

```python
silicon_mask = left_waveguide_port | right_waveguide_port
silicon_dioxide_mask = border_mask.copy()
silicon_dioxide_mask[silicon_mask] = False
```

Results at each epoch are saved as both a PNG bitmap and a CSV file of unmapped design weights, enabling post-processing and design verification.

#### Key Takeaways

- The epigraph (minimax) formulation is the correct approach for broadband optimization — it directly minimizes the worst-case spectral performance without requiring manual objective weighting.
- Reflectance normalization via a preliminary straight-waveguide run is essential for accurate S-parameter extraction; the `subtracted_dft_fields` mechanism removes incident-field contamination from reflection monitors.
- Material damping in `MaterialGrid` acts as a soft penalty on grayscale regions, effectively regularizing the optimization toward binary designs.
- The border mask technique locks pixels at the design region periphery to known materials (silicon at waveguide ports, SiO2 elsewhere), ensuring smooth connectivity between the optimized device and the access waveguides.
- At high beta, subpixel smoothing (`do_averaging=True`) should be enabled to accurately compute the permittivity of partially-filled pixels and avoid staircase-induced gradient errors.

---

### 3. `multilayer_opt.py` — Shape Optimization of a 1D Multilayer Stack

**Physics:** Optimizes the thicknesses of a nine-layer alternating dielectric stack to minimize the maximum integrated electric field intensity inside the stack over two design wavelengths (0.95 and 1.05 µm), using a levelset-based shape parameterization and the adjoint method.
**Difficulty:** Intermediate
**Source:** `python/examples/adjoint_optimization/multilayer_opt.py`
**Test Status:** TIMEOUT (CPU-intensive — repeated random restarts with 800 px/µm 1D simulations)

#### Theory

A multilayer stack consists of alternating layers of two dielectric materials with refractive indices n1 = 1.0 and n2 = 1.3. For a stack with quarter-wavelength layers, the round-trip phase condition produces destructive interference for reflected waves (antireflection coating behavior) or constructive interference (high-reflectance mirror behavior) depending on the layer ordering. Optimizing the layer thicknesses can achieve specific broadband spectral targets that quarter-wavelength layers alone cannot satisfy.

The design parameters are the N = 9 layer thicknesses [t_1, t_2, ..., t_9], which are real-valued scalars rather than pixel arrays. This is a shape optimization problem rather than a topology optimization problem: the topology (the number of layers and their material assignments) is fixed, and only the geometric extent of each layer varies. The objective function measures how much electric field energy is stored inside the stack, computed by integrating the squared magnitude of the DFT electric field Ex over the entire design volume:

```
J(lambda) = log( integral |Ex(r, lambda)|^2 dr )
```

The logarithm is included to normalize the dynamic range across the two wavelengths. The worst-case formulation uses the epigraph variable t to minimize the maximum of J(lambda_1) and J(lambda_2) simultaneously.

The key challenge in this problem is the parameterization. The design parameters are layer thicknesses, not pixel weights, so the Meep adjoint solver — which computes gradients with respect to MaterialGrid weights — cannot be applied directly. Instead, a custom `@primitive` function `levelset_and_smoothing` converts layer thicknesses to a 1D density weight vector on the design grid. This function is registered with `autograd` via `defvjp`, which allows the chain rule to pass gradients from the adjoint solver (in density weight space) back to the layer thickness space:

```python
@primitive
def levelset_and_smoothing(layer_thickness_um):
    # Convert thicknesses to 1D density weights on design grid
    ...
    return smoothed_weights.flatten()

defvjp(levelset_and_smoothing, levelset_and_smoothing_vjp)
```

The vector-Jacobian product function `levelset_and_smoothing_vjp` computes the Jacobian of the levelset mapping numerically: it perturbs each layer thickness by a small amount (`LAYER_PERTURBATION_UM = 1/RESOLUTION_UM`) and measures how the output density vector changes. This finite-difference Jacobian is then used to construct the VJP via `np.tensordot`.

Because the design space is low-dimensional (only 9 parameters) and the optimization is non-convex, the script runs 10 random restarts (`NUM_OPT_REPEAT = 10`), each starting from a random set of layer thicknesses within physically reasonable bounds, and keeps the best result. Each restart runs up to 30 optimization iterations.

#### Code Walkthrough

The simulation is 1D (`dimensions=1`), which makes it exceptionally fast while still using the full FDTD engine. The design region extends over the z-axis and is filled by a `MaterialGrid` with `NZ_SIM_GRID` pixels in the z direction:

```python
sim = mp.Simulation(
    resolution=RESOLUTION_UM,
    cell_size=cell_size,
    dimensions=1,
    boundary_layers=pml_layers,
    sources=sources,
    geometry=geometry,
)
```

The optimization problem uses a `mpa.FourierFields` monitor rather than an eigenmode coefficient monitor. This captures the full complex DFT electric field array inside the design region, which is needed to compute the integrated intensity:

```python
obj_args = [
    mpa.FourierFields(sim, volume=matgrid_region.volume, component=mp.Ex, yee_grid=True)
]

def obj_func(dft_ex):
    return npa.log(npa.sum(npa.absolute(dft_ex) ** 2, axis=1))
```

The epigraph formulation is structured identically to the mode converter example. The epigraph constraint function calls the adjoint solver, backpropagates through the levelset function, and fills in the gradient matrix in place:

```python
grad_backpropagate = np.zeros((NUM_LAYERS, num_wavelengths))
defvjp(levelset_and_smoothing, levelset_and_smoothing_vjp)
for k in range(num_wavelengths):
    grad_backpropagate[:, k] = tensor_jacobian_product(
        levelset_and_smoothing, 0
    )(layer_thickness_um, grad[:, k])

gradient[:, 0] = -1           # d(constraint)/d(t)
gradient[:, 1:] = grad_backpropagate.T
```

The bounds on layer thicknesses are set to a narrow 5% band around the quarter-wavelength thickness at the mean design wavelength. This physically motivated prior prevents degenerate solutions (zero-thickness layers) while still allowing the optimizer flexibility to tune the interference conditions.

```python
fraction_thickness = 0.05
layer_thickness_um_lower_bound[0::2] = (1 - fraction_thickness) * mean_layer_thickness_um[1]
layer_thickness_um_upper_bound[0::2] = (1 + fraction_thickness) * mean_layer_thickness_um[1]
```

All results (convergence histories, optimal layer thicknesses, objective values) are saved to `optimal_design.npz` for post-processing.

#### Key Takeaways

- Shape optimization (varying geometric extents of fixed-topology features) can be combined with Meep's adjoint solver by registering a custom `@primitive` levelset function that maps shape parameters to density weights.
- The `defvjp` mechanism in `autograd` allows custom non-differentiable functions (here, a piecewise-constant levelset) to participate in gradient backpropagation by supplying their VJP explicitly.
- `mpa.FourierFields` provides access to the full spatial DFT field array, enabling objective functions that integrate field intensities over volumes rather than just measuring modal coefficients at ports.
- Multiple random restarts are essential for non-convex shape optimization with few parameters, since the optimization landscape has many local minima corresponding to different resonance conditions.
- 1D simulations are a powerful development and testing tool: they run in seconds at high resolution and expose the same adjoint API as full 3D simulations.

---

### 4. `absorbed_power_density.py` — Mapping Absorbed Power Density in a Lossy Cylinder

**Physics:** Computes the spatial distribution of absorbed electromagnetic power inside a silica (SiO2) cylinder illuminated by a plane wave, using DFT field monitors to evaluate the local Joule dissipation rate, and verifies the result against a closed-surface flux integration.
**Difficulty:** Beginner
**Source:** `python/examples/absorbed_power_density.py`
**Test Status:** TIMEOUT (CPU-intensive — resolution 100 px/µm full 2D simulation)

#### Theory

When a lossy dielectric material (with complex permittivity epsilon = epsilon' + i*epsilon'') is exposed to an electromagnetic field, it absorbs energy from the field through Joule heating. The local absorbed power density (power dissipated per unit volume) at a point r inside the material is given by:

```
Q(r) = (omega/2) * Im[epsilon(r)] * |E(r)|^2
     = (omega/2) * Im[E*(r) . D(r)]
```

where the second form uses the displacement field D = epsilon * E and is numerically more convenient because Meep stores D and E on different subgrids of the Yee lattice. Computing this quantity pointwise from DFT (frequency-domain) fields gives a spatial map of where the material is most strongly absorbing — a useful diagnostic for understanding absorption cross-sections, hot-spot formation, and near-field energy concentration.

SiO2 has a small but nonzero absorption at optical wavelengths. The complex permittivity of SiO2 at lambda = 1 µm has a small positive imaginary part, corresponding to an absorption length of L_abs = lambda / Im[sqrt(epsilon)] on the order of kilometers. For a micron-scale cylinder, the fractional absorption is tiny, but it can be accurately computed from the DFT fields.

The verification technique used in the code compares two independent calculations of the total absorbed power:

1. Volume integral of absorbed power density: sum over all pixels inside the cylinder of Q(r) * dA, where dA = (1/resolution)^2 is the area element.
2. Flux through a closed box: the net Poynting flux into the closed box surrounding the cylinder, measured by four flux monitors (left, right, top, bottom faces).

By the Poynting theorem, these two quantities must be equal in steady state. Any discrepancy quantifies the numerical error. The code prints the error as a fractional difference:

```
err = |absorbed_power - absorbed_flux| / absorbed_flux
```

The simulation uses a mirror symmetry in Y (`mp.Mirror(mp.Y)`) to halve the computational domain, since both the plane wave source and the cylinder geometry are symmetric about y = 0.

#### Code Walkthrough

The simulation is a straightforward 2D FDTD run with PML boundaries. A Gaussian plane wave source is placed at the left edge of the cell and propagates in the +x direction:

```python
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=0.1 * fcen, is_integrated=True),
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s),
        component=mp.Ez,
    )
]
```

The `is_integrated=True` flag is required for plane-wave sources that span the full transverse width including PML regions — without it, the source amplitude is not normalized correctly.

After running, the DFT field arrays for Dz and Ez are extracted over a box enclosing the cylinder:

```python
Dz = sim.get_dft_array(dft_fields, mp.Dz, 0)
Ez = sim.get_dft_array(dft_fields, mp.Ez, 0)
absorbed_power_density = 2 * np.pi * fcen * np.imag(np.conj(Ez) * Dz)
```

The factor 2*pi*fcen converts from Meep's internal angular frequency to a physically meaningful absorbed power density. The closed-surface flux box uses weighted `mp.FluxRegion` objects: inward-pointing faces have `weight=+1` and outward-pointing faces have `weight=-1`, so the net flux into the cylinder is computed automatically:

```python
flux_box = sim.add_flux(
    fcen, 0, 1,
    mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2 * r), weight=+1),
    mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2 * r), weight=-1),
    mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2 * r, 0), weight=-1),
    mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r, 0), weight=+1),
)
```

The power density map is visualized with a `pcolormesh` plot using the `inferno_r` colormap, which renders high-absorption regions in light colors against a dark background. The plot title includes the computed absorption length L_abs of SiO2 at the design wavelength, extracted from the imaginary part of the square root of the complex permittivity:

```python
wvl / np.imag(np.sqrt(SiO2.epsilon(fcen)[0][0]))
```

This calls Meep's built-in dispersive material model for SiO2 at the specified frequency.

#### Key Takeaways

- The formula `Q = omega * Im[conj(E) * D] / 2` computes absorbed power density directly from DFT field arrays, avoiding the need to integrate over closed surfaces.
- The Poynting theorem provides a rigorous cross-check: agreement between the volume integral of Q and the closed-surface net flux validates the simulation setup.
- `is_integrated=True` is required for plane-wave sources extending into PML regions and should never be omitted for such configurations.
- The `SiO2.epsilon(fcen)` call demonstrates how Meep's dispersive material library returns frequency-dependent complex permittivity tensors, enabling post-processing calculations of derived optical properties.
- Mirror symmetries should always be exploited to reduce memory and compute time; a single `mp.Mirror(mp.Y)` halves the 2D cell size for this symmetric geometry.

---

### 5. `test_adjoint_solver.py` — Verification of Adjoint Gradients Across Multiple Monitor Types

**Physics:** Systematically verifies that adjoint-computed gradients match finite-difference approximations for a suite of objective function types: eigenmode coefficients, DFT field integrals, local density of states (LDoS), complex fields with Bloch periodicity, material damping, anisotropic materials, and multiple simultaneous objective functions.
**Difficulty:** Advanced
**Source:** `python/tests/test_adjoint_solver.py`
**Test Status:** TIMEOUT (CPU-intensive — each test runs two FDTD simulations at resolution 30 px/µm)

#### Theory

The correctness of any adjoint gradient solver rests on a single mathematical identity: the directional derivative of the objective function in a direction dp must equal the inner product of the adjoint gradient with dp. Formally, for a small perturbation dp to the design parameters p:

```
J(p + dp) - J(p) = dp^T * (dJ/dp) + O(|dp|^2)
```

This can be tested by choosing a random direction dp (the perturbation vector) and comparing:
- The adjoint directional derivative: `dp^T * grad_adjoint`
- The finite-difference directional derivative: `J(p + dp) - J(p)`

Agreement to relative tolerance on the order of `|dp|` (here `deps = 1e-5`) confirms that the adjoint gradient is correct. The test suite uses this strategy for every monitor type supported by `mpa.OptimizationProblem`.

The adjoint method works by solving a "reverse" problem. If the forward FDTD solve computes fields E satisfying curl H - dD/dt = J_src, the adjoint solve computes a set of adjoint fields E_adj satisfying the time-reversed version of Maxwell's equations with sources derived from the gradient of the objective with respect to the DFT fields. The sensitivity of the objective to the permittivity at each point in the design region is then:

```
dJ/d_epsilon(r) = -2 * omega * Im[ E_adj(r) * E_fwd(r) ]
```

This coupling between forward and adjoint fields at each spatial point in the design region is the fundamental operation that the `OptimizationProblem` class automates.

The test file exercises several non-trivial extensions of this basic formula:

**Multifrequency monitors.** When the objective depends on fields at multiple frequencies simultaneously, the adjoint simulation must carry contributions from all frequency components. The test verifies that the adjoint gradient for a multi-frequency simulation matches what would be obtained by running a separate single-frequency adjoint simulation for each frequency.

**Gradient backpropagation.** In realistic optimization, the adjoint gradient is computed with respect to the mapped design (after filter and projection). To optimize with respect to the raw design parameters, the gradient must be backpropagated through the differentiable mapping using the chain rule. The test verifies this backpropagation by comparing the backpropagated adjoint gradient to a finite difference computed in raw parameter space.

**Complex fields with Bloch periodicity.** When a `k_point` is specified, Meep evolves complex-valued fields. The adjoint formula requires an additional conjugation in certain terms. The test verifies that this case is handled correctly.

**Anisotropic materials.** For materials with non-diagonal permittivity tensors (like sapphire, which is uniaxial with off-diagonal elements in the epsilon tensor), the sensitivity formula involves tensor contractions. The test uses a sapphire-like medium with off-diagonal epsilon components.

**Material damping.** When `MaterialGrid` is initialized with a nonzero `damping` parameter, an imaginary part proportional to `omega * damping` is added to the permittivity of each pixel proportional to the derivative of the weight function. This makes the gradient formula more complex, and the test verifies its correctness.

#### Code Walkthrough

The test class inherits from `ApproxComparisonTestCase`, which provides the `assertClose` method for comparing arrays with a tolerance. All simulation parameters are set up in `setUpClass` to be reused across tests. A random but reproducible design `p` and perturbation `dp` are generated using a seeded RNG:

```python
rng = np.random.RandomState(9861548)
cls.p = 0.5 * rng.rand(cls.Nx * cls.Ny)
cls.dp = 1e-5 * rng.rand(cls.Nx * cls.Ny)
```

The core `adjoint_solver` method is generic over the monitor type (EIGENMODE, DFT, LDOS):

```python
def adjoint_solver(self, design_params, mon_type, frequencies, mat2=None, need_gradient=True):
    matgrid = mp.MaterialGrid(mp.Vector3(self.Nx, self.Ny), mp.air, self.silicon,
                               weights=np.ones((self.Nx, self.Ny)))
    ...
    f, dJ_du = opt([design_params], need_gradient=need_gradient)
    return f, dJ_du
```

The eigenmode test verifies the gradient for both the reflectance (S11) and transmittance (S21) monitors simultaneously:

```python
def J(refl_mon, tran_mon):
    return -npa.power(npa.abs(refl_mon), 2) + npa.power(npa.abs(tran_mon), 2)
```

The finite-difference comparison computes a directional derivative:

```python
adj_dd = (self.dp[None, :] @ unperturbed_grad).flatten()
fnd_dd = perturbed_val - unperturbed_val
self.assertClose(adj_dd, fnd_dd, epsilon=tol)
```

The gradient backpropagation test applies a conic filter plus tanh projection as the mapping function, then backpropagates through it using `tensor_jacobian_product`:

```python
unperturbed_grad_backprop = tensor_jacobian_product(self.mapping, 0)(
    self.p, filter_radius, eta, beta, unperturbed_grad
)
```

The periodic design test checks that the length-scale constraint functions (`mpa.constraint_solid` and `mpa.constraint_void`) are translation-invariant under periodic boundary conditions — an important mathematical property required for their use in periodic device optimization.

#### Key Takeaways

- The finite-difference gradient check `(J(p+dp) - J(p)) ~ dp^T * grad` is the gold-standard test for adjoint gradient correctness and should always be run when implementing a new objective function.
- Tolerances differ by monitor type: eigenmode coefficients achieve ~1e-5 relative error at double precision, while LDoS and DFT-field objectives achieve ~1e-3, reflecting differences in the conditioning of each monitor's adjoint source.
- For multi-frequency simulations, the adjoint solver returns a 2D gradient array of shape `(N_design_params, N_frequencies)`; each column is the gradient at a single frequency.
- The `subtracted_dft_fields` mechanism for reflectance measurement subtracts the incident field DFT data captured from a separate straight-waveguide run, yielding the pure scattered-field S11 coefficient.
- The `mpa.constraint_solid` and `mpa.constraint_void` length-scale constraints are analytically differentiable via `autograd` and can be added directly as NLopt inequality constraints, enabling hard fabrication constraints in the optimization.

---

### 6. `test_adjoint_jax.py` — JAX-Based Automatic Differentiation with `MeepJaxWrapper`

**Physics:** Validates the `MeepJaxWrapper` interface that makes Meep FDTD simulations differentiable within the JAX automatic differentiation framework, enabling loss functions expressed in JAX to receive correct gradients through a Meep simulation.
**Difficulty:** Advanced
**Source:** `python/tests/test_adjoint_jax.py`
**Test Status:** FAIL (missing `jax` package — install with `pip install jax`)

#### Theory

The `autograd`-based adjoint interface used in the previous examples requires the user to manually call `opt([design_params])` to trigger the forward and adjoint simulations, then manually backpropagate gradients through pre-processing functions via `tensor_jacobian_product`. While powerful, this workflow does not integrate natively with modern deep learning frameworks that define entire computational graphs automatically.

The `MeepJaxWrapper` provides an alternative interface where the Meep simulation is treated as a node in a JAX computation graph. When JAX's automatic differentiation system encounters the `wrapped_meep([x])` call during reverse-mode differentiation (`jax.value_and_grad`), it triggers the appropriate adjoint simulation and returns the gradient with respect to the design input `x`. The user never calls the adjoint solver directly — JAX handles the entire backward pass:

```python
def loss_fn(x, excite_port_idx=0):
    wrapped_meep = mpa.MeepJaxWrapper(simulation, [sources[excite_port_idx]],
                                       monitors, design_regions, frequencies)
    monitor_values = wrapped_meep([x])
    s1p, s1m, s2p, s2m = monitor_values
    t = s2p / s1p if excite_port_idx == 0 else s1m / s2m
    return jnp.mean(jnp.square(jnp.abs(t)))

value, adjoint_grad = jax.value_and_grad(loss_fn)(x, excite_port_idx=excite_port_idx)
```

The wrapper unpacks monitor values as S-parameters (forward and backward modal amplitudes at each port), which are natural outputs for waveguide optimization. From these, any scattering matrix element can be constructed using standard JAX operations — complex arithmetic, absolute values, averages — all of which are automatically differentiable.

The test evaluates correctness by computing the gradient via `jax.value_and_grad` and then projecting along 5 random directions in design space, comparing against the corresponding finite-difference directional derivatives:

```
dp . grad_JAX  ~  loss(x + dp) - loss(x)
```

This projection test is more robust than comparing individual gradient components because it integrates over the entire design space and catches cancellation errors that might hide in any single component.

JAX requires double precision for accurate finite-difference validation. The `jax.config.update("jax_enable_x64", True)` call at the top of the file enables 64-bit floating point throughout, overriding JAX's default behavior of using 32-bit everywhere for GPU efficiency.

#### Code Walkthrough

The simulation setup builds a straight waveguide with a design region segment. Two EigenModeSource objects are created — one propagating in the +x direction (exciting port 1) and one in the -x direction (exciting port 2). During a test, only the selected source is active:

```python
sources = [
    mp.EigenModeSource(..., eig_kpoint=mp.Vector3(1, 0, 0),  ...),  # port 1
    mp.EigenModeSource(..., eig_kpoint=mp.Vector3(-1, 0, 0), ...),  # port 2
]
```

Four `mpa.EigenmodeCoefficient` monitors are created — forward and backward at each of the two monitor planes — giving the full S-parameter set {S1+, S1-, S2+, S2-}:

```python
monitors = [
    mpa.EigenmodeCoefficient(simulation, mp.Volume(center=center, size=monitor_size),
                              mode=1, forward=forward)
    for center in monitor_centers
    for forward in [True, False]
]
```

The `MeepJaxWrapper` accepts the simulation object, active sources, monitors, design regions, and frequencies. Calling it returns the monitor values as a JAX array:

```python
wrapped_meep = mpa.MeepJaxWrapper(
    simulation, [sources[excite_port_idx]], monitors, design_regions, frequencies
)
monitor_values = wrapped_meep([x])
```

The loss function computes transmittance as a power ratio. For port 1 excitation:

```python
t = s2p / s1p   # T = |S21|^2 when squared
return jnp.mean(jnp.square(jnp.abs(t)))
```

The parameterized test matrix exercises six combinations of frequency bandwidth and port index, ensuring the wrapper is correct for both excitation directions and across different spectral ranges.

#### Key Takeaways

- `MeepJaxWrapper` enables native JAX automatic differentiation through FDTD simulations, allowing Meep to participate in JAX-based machine learning and inverse design workflows without explicit adjoint bookkeeping.
- S-parameter outputs from `MeepJaxWrapper` are complex JAX arrays; arbitrary differentiable post-processing (complex arithmetic, norms, sums, means) can be applied before calling `jax.value_and_grad`.
- `jax.config.update("jax_enable_x64", True)` is mandatory for gradient verification — JAX's default 32-bit mode has insufficient precision for the finite-difference check at step size 1e-4.
- The directional-derivative projection test (`dp . grad ~ loss(x+dp) - loss(x)`) is a numerically robust gradient validation strategy that should be applied whenever integrating new AD frameworks with physical simulators.
- The wrapper internally manages two FDTD runs (forward and adjoint) per `jax.grad` call; users should structure their JAX programs to minimize unnecessary re-evaluation of `jax.value_and_grad`.

---

### 7. `test_adjoint_utils.py` — Verification of Spatial Filter Zero-Phase Property

**Physics:** Verifies that the spatial filters used in the adjoint optimization pipeline (conic, cylindrical, and Gaussian) are zero-phase filters — meaning they do not introduce any spatial phase shift when applied to a symmetric input field.
**Difficulty:** Beginner
**Source:** `python/tests/test_adjoint_utils.py`
**Test Status:** FAIL (missing `parameterized` package — install with `pip install parameterized`)

#### Theory

The filter-project pipeline in topology optimization requires the spatial filter to have a specific mathematical property: zero phase shift. A zero-phase filter is one whose output is symmetric whenever its input is symmetric. More precisely, if x is symmetric under left-right reflection (x = fliplr(x)) and up-down reflection (x = flipud(x)), then the filtered output y = filter(x) must also be symmetric.

This property is essential for two reasons. First, it ensures that symmetric design initial conditions remain symmetric throughout the optimization, which is important for designing symmetric devices (like the C4-symmetric waveguide crossing). If the filter introduced a phase shift, an initially symmetric design would break symmetry on the first optimization step. Second, for filters based on convolution, zero-phase is equivalent to having a real-valued frequency response — meaning the filter does not rotate the phase of any Fourier mode. This is the case for all three filter types in `meep.adjoint`:

- **Conic filter:** a cone-shaped convolution kernel in real space, equivalent to multiplication by a (sin(k*r)/(k*r))^2 envelope in Fourier space — real, non-negative everywhere.
- **Cylindrical filter:** a disk-shaped (top-hat) kernel in real space, equivalent to a Bessel function envelope in Fourier space — real, oscillating but zero-phase.
- **Gaussian filter:** a Gaussian kernel in real space, equivalent to a Gaussian envelope in Fourier space — real, non-negative, maximally smooth.

All three kernels are centro-symmetric (point-symmetric about the origin), which is the sufficient condition for zero-phase behavior.

#### Code Walkthrough

The test constructs a random input array x and symmetrizes it by adding with its reflections:

```python
x = np.random.rand(Nx, Ny)
x = x + np.fliplr(x)   # enforce left-right symmetry
x = x + np.flipud(x)   # enforce up-down symmetry
```

After this double symmetrization, x is exactly symmetric in both directions. The test then applies each filter and checks that the output y is also symmetric to within numerical precision:

```python
y = filter_func(x, radius, Lx, Ly, resolution)
self.assertClose(y, np.fliplr(y), epsilon=_TOL)
self.assertClose(y, np.flipud(y), epsilon=_TOL)
```

The tolerance is set to `_TOL = 1e-14` for double precision — essentially machine epsilon, confirming that the filters are exactly zero-phase up to floating-point rounding. For single-precision builds, the tolerance is relaxed to `1e-6`.

The parameterized test matrix covers several combinations of domain aspect ratios (Lx, Ly), resolutions (20 and 23 pixels per unit length, chosen to be coprime to test non-power-of-2 grid sizes), and filter types:

```python
@parameterized.parameterized.expand([
    ("1.0_1.0_20_conic",    1.0, 1.0, 20, 0.24, mpa.conic_filter),
    ("0.887_1.56_gaussian", 0.887, 1.56, 20, 0.24, mpa.gaussian_filter),
    ("0.887_1.56_cylindrical", 0.887, 1.56, 20, 0.24, mpa.cylindrical_filter),
    ...
])
def test_filter_offset(self, test_name, Lx, Ly, resolution, radius, filter_func):
    ...
```

The non-square aspect ratios (0.887 x 1.56) test that the filters correctly handle rectangular design regions, where the convolution kernel must be normalized appropriately in each direction.

#### Key Takeaways

- Zero-phase spatial filters are a mathematical prerequisite for symmetric inverse design workflows; all three filters in `meep.adjoint` satisfy this requirement to machine precision.
- The test strategy of applying the filter to a known-symmetric input and checking symmetry of the output is simpler and more reliable than directly analyzing the filter's frequency response.
- Non-power-of-2 grid sizes (like 23 pixels per unit) must be explicitly tested because FFT-based convolutions can behave differently for such sizes in some implementations.
- The `parameterized.parameterized.expand` decorator from the `parameterized` package generates separate named test cases for each parameter combination, making test failures easy to trace to specific configurations.
- Filter radius selection directly controls minimum feature size in the optimized design: `r_f = mpa.get_conic_radius_from_eta_e(L_min, eta_e)` converts a target minimum length scale L_min and erosion threshold eta_e into the appropriate convolution kernel radius.

---

### 8. `test_adjoint_cyl.py` — Adjoint Optimization in Cylindrical Coordinates with Near-to-Far Field Objectives

**Physics:** Verifies the adjoint gradient for an objective function based on near-to-far field transformation in cylindrical coordinates (r, phi, z), testing multiple azimuthal mode orders m and far-field observation points to confirm correct sensitivity computation for rotationally symmetric designs.
**Difficulty:** Advanced
**Source:** `python/tests/test_adjoint_cyl.py`
**Test Status:** FAIL (missing `parameterized` package — install with `pip install parameterized`)

#### Theory

Cylindrical coordinates (r, phi, z) with azimuthal mode number m allow 3D rotationally symmetric problems to be solved as effectively 2D simulations. In Meep's cylindrical coordinate mode, fields are decomposed as E(r,z) * exp(i*m*phi), where m is an integer (or half-integer for some geometries). A simulation at a single m value captures the full 3D physics for that particular azimuthal harmonic, at a computational cost comparable to a 2D Cartesian simulation.

This is particularly powerful for rotationally symmetric photonic devices such as lenses, axicons, metalens structures, and cylindrical Mie resonators. The design region in this test occupies a disk of radius `design_r = 5 µm` and thickness `design_z = 2 µm`, parameterized by a 2D grid of (Nr, Nz) = (201, 81) design variables representing the local permittivity interpolated between SiO2 (index 1.44) and Si (index 3.4).

The objective function is based on `mpa.Near2FarFields`, which computes the electromagnetic field at a specified far-field observation point by applying the near-to-far field transformation theorem (the Kirchhoff-Huygens integral) to DFT fields on a near-field surface. The quantity optimized is `|Er(far_x)|^2`, the radial electric field intensity at a specific far-field point. This is a natural objective for beam shaping, focusing, or far-field pattern control.

The adjoint method for near-to-far field objectives requires special treatment. The adjoint source for a far-field objective is not located at the observation point (which may be far outside the simulation cell) but rather on the near-field surface itself. Meep's adjoint solver handles this automatically by converting the far-field objective gradient into equivalent near-field sources before running the adjoint simulation.

The test exercises four parameter combinations: azimuthal mode number m = 0, m = -1 (backward circular polarization), m = 1.2 (a non-integer m value, relevant for structures with partial rotational symmetry), and two different far-field observation points. For |m| > 1, the source size excludes the region near r = 0 to avoid the coordinate singularity in cylindrical coordinates.

The gradient validation uses the same finite-difference strategy as the Cartesian adjoint tests: compare the adjoint directional derivative `dp^T * grad_adjoint` against the finite-difference directional derivative `|Er|^2(p+dp) - |Er|^2(p)`. The tolerance is deliberately more generous (0.3–0.6) than for Cartesian tests, reflecting the additional numerical complexity of the near-to-far field transform and cylindrical geometry.

#### Code Walkthrough

The simulation is initialized with `dimensions=mp.CYLINDRICAL` and the azimuthal mode number `m`:

```python
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=boundary_layers,
    sources=get_source(m),
    geometry=geometry,
    dimensions=dimensions,
    m=m,
)
```

The design region `MaterialGrid` is a 2D (r, z) grid with dimensions `(Nr, 0, Nz)` — the middle component zero in the Vector3 indicates no phi variation, consistent with the cylindrical symmetry assumption:

```python
design_variables = mp.MaterialGrid(mp.Vector3(Nr, 0, Nz), SiO2, Si, do_averaging=True)
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=mp.Vector3(design_r / 2, 0, 0),
                     size=mp.Vector3(design_r, 0, design_z)),
)
```

The near-to-far field objective uses a `mpa.Near2FarFields` monitor that wraps Meep's built-in near2far capabilities:

```python
FarFields = mpa.Near2FarFields(sim, NearRegions, far_x)
ob_list = [FarFields]

def J(alpha):
    return npa.abs(alpha[0, 0, 0]) ** 2
```

The `alpha` array has dimensions `(num_far_points, num_frequencies, num_field_components)`. Selecting `alpha[0, 0, 0]` extracts the first field component (Er) at the first far-field point and first frequency.

The forward simulation computes the reference value using a traditional (non-adjoint) approach: running a full simulation and calling `sim.get_farfield(mode, far_x)` directly. This provides an independent check on the adjoint solver's objective value:

```python
S12_unperturbed = forward_simulation(p, m, far_x)
self.assertClose(adjsol_obj, S12_unperturbed, epsilon=1e-3)
```

The parameterized test matrix tests four cases:

```python
@parameterized.parameterized.expand([
    (0,   [mp.Vector3(5, 0, 20)]),
    (0,   [mp.Vector3(4, 0, 28)]),
    (-1,  [mp.Vector3(5, 0, 20)]),
    (1.2, [mp.Vector3(5, 0, 20)]),
])
def test_adjoint_solver_cyl_n2f_fields(self, m, far_x):
    ...
```

The negative m = -1 tests backward circular polarization coupling; m = 1.2 is a non-integer mode that arises when the source or geometry lacks full rotational symmetry but still benefits from the cylindrical coordinate decomposition.

#### Key Takeaways

- Cylindrical coordinate simulations in Meep reduce 3D rotationally symmetric inverse design problems to effective 2D computations, delivering a massive speedup for designing lenses, axicons, and ring resonators.
- `mpa.Near2FarFields` enables objectives based on far-field radiation patterns, which are the natural figures of merit for antenna, beam-shaping, and far-field focusing applications.
- The adjoint source for near-to-far field objectives is placed on the near-field surface, not at the distant observation point; Meep's adjoint solver handles this transformation internally.
- Non-integer azimuthal mode numbers (like m = 1.2) are supported in Meep and arise physically for structures with partial rotational symmetry or for certain types of angular momentum sources.
- The gradient tolerance for near2far objectives in cylindrical coordinates is substantially larger (0.3–0.6) than for eigenmode objectives in Cartesian coordinates (< 0.01), reflecting the additional approximation inherent in the Kirchhoff integral discretization at modest simulation resolutions.
