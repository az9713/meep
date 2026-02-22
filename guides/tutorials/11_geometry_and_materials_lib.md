# Chapter 11: Geometry and Material Grids

This chapter covers the foundational building blocks of every Meep simulation: how geometric objects are constructed, validated, and tiled into periodic structures; how polygonal prisms handle non-convex fabrication-realistic shapes; how material grids encode spatially varying permittivity for topology optimization; and how sources, coordinate utilities, and simulation bookkeeping are tested and verified. Together these six tutorial sections form a complete picture of the Python-level infrastructure that sits between the user's physical intent and the C++ FDTD engine.

---

### 1. `test_geom.py` — Geometric Objects, Linear Algebra, and Material Validation

**Physics:** Validates the complete geometry and linear-algebra layer of Meep's Python API, covering point-containment queries, periodic tiling of objects, anisotropic media, and material frequency-range warnings.
**Difficulty:** Beginner
**Source:** `python/tests/test_geom.py`
**Test Status:** PASS (7.7 s)

#### Theory

Every FDTD simulation begins with a geometry: a list of objects placed inside a computational cell, each carrying a material (a permittivity tensor and, optionally, dispersive susceptibilities). Meep inherits the libctl geometry model in which objects are defined analytically by their type and parameters rather than by an explicit mesh. This analytic representation is crucial for subpixel smoothing: at every Yee grid voxel that straddles a material interface, Meep computes a locally averaged permittivity tensor using the exact object boundary, not a pixelated approximation. The test suite in `test_geom.py` validates that this analytic layer is self-consistent before any FDTD stepping occurs.

The six primitive object types — `Sphere`, `Cylinder`, `Cone`, `Wedge`, `Block`, `Ellipsoid`, and `Prism` — are all subclasses of `GeometricObject`. Each stores a center, a material, and type-specific parameters. Point-containment (`__contains__`) is tested directly, which exercises the same C++ geometry routines that are called during the `set_epsilon` phase of `init_sim`. If these containment tests fail, the dielectric profile would be incorrect even before stepping begins.

Periodic crystal photonics requires duplicating unit-cell objects across a Bravais lattice. The test covers `geometric_object_duplicates`, `geometric_objects_duplicates`, and `geometric_objects_lattice_duplicates`. The last function accepts an explicit `Lattice` object whose non-orthogonal basis vectors describe the primitive cell. The hexagonal lattice test uses basis vectors $\mathbf{a}_1 = (\sqrt{3}/2, 1/2)$ and $\mathbf{a}_2 = (\sqrt{3}/2, -1/2)$, the standard 2D triangular lattice. After duplication, each cylinder center should lie at an integer linear combination of these basis vectors, and the test verifies this precisely.

Anisotropic and gyrotropic materials require coordinate transformations of the permittivity tensor. If a medium defined in a crystal's principal-axis frame is rotated by an angle $\theta$, the new permittivity tensor becomes $\epsilon' = R \epsilon R^T$, where $R$ is the rotation matrix. The `Medium.transform` test applies a 23.9° rotation about $z$ and checks both diagonal and off-diagonal components of the resulting tensor, as well as verifying that all attached `E_susceptibilities` undergo the identical transformation. This is essential for modeling obliquely cut crystals.

The `test_check_material_frequencies` test enforces that Meep warns the user when a source's frequency bandwidth overlaps with the region outside a material's declared valid frequency range (`valid_freq_range`). Dispersive material fits are only accurate within the fitted band, so attempting to drive a Lorentzian-fit material far outside its design band can produce unphysical negative permittivities or numerical blow-up.

#### Code Walkthrough

**Object construction and containment** is the simplest layer:

```python
s = gm.Sphere(center=zeros(), radius=2.0)
point = ones()
self.assertTrue(point in s)          # calls s.__contains__(point)
self.assertFalse(gm.Vector3(10, 10, 10) in s)
```

The `in` operator calls `__contains__`, which maps directly to the libctl `point_in_object` routine. For a sphere of radius $r$ centered at the origin, the condition is simply $|\mathbf{x}|^2 \leq r^2$.

**Sphere shifting** tests the `+` operator overload and in-place `+=`:

```python
s = gm.Sphere(center=zeros(), radius=2.0)
new_sphere = s + mp.Vector3(5, 5)
self.assertEqual(new_sphere.center, mp.Vector3(5, 5))
self.assertEqual(s.center, zeros())  # original unchanged
```

Both `v + s` and `s + v` work via `__add__` and `__radd__`, allowing natural list-comprehension idioms like `[s + offset for offset in lattice_vectors]`.

**Lattice duplication** demonstrates how periodic PhC structures are built without manual loops:

```python
geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 7),
    basis1=mp.Vector3(math.sqrt(3)/2, 0.5),
    basis2=mp.Vector3(math.sqrt(3)/2, -0.5),
)
geometry = mp.geometric_objects_lattice_duplicates(geometry_lattice, geometry)
```

Internally, `geometric_objects_lattice_duplicates` enumerates all integer combinations $(n_1, n_2)$ such that $n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2$ lies within the cell, then places a copy of each object at that offset. The test checks that seven cylinders appear at $y \in \{-3, -2, -1, 0, 1, 2, 3\}$.

**Material transformation** checks tensor rotation:

```python
rot_angle = math.radians(23.9)
rot_matrix = mp.Matrix(
    mp.Vector3(math.cos(rot_angle), math.sin(rot_angle), 0),
    mp.Vector3(-math.sin(rot_angle), math.cos(rot_angle), 0),
    mp.Vector3(0, 0, 1),
)
mat.transform(rot_matrix)
self.assertTrue(mat.epsilon_diag.close(expected_diag, tol=4))
```

All attached susceptibilities are transformed simultaneously, ensuring that the frequency-domain response of the rotated material is physically correct.

**Frequency-range warnings** use Python's `warnings` module to intercept Meep's built-in advisory:

```python
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sim.run(until=5)
    self.assertEqual(len(w), 1)
    self.assertIn("material", str(w[-1].message))
```

The warning fires for sources whose bandwidth extends outside `FreqRange(min=10, max=20)`, covering cases such as center frequencies too low or too high, or bandwidths so large they straddle the range boundary.

**Vector3 arithmetic** is tested exhaustively including `cross`, `cdot` (conjugate dot product), `rotate`, and `norm` on complex-valued vectors (GitHub issue #722):

```python
v = mp.Vector3(1, 1j, 0)
self.assertAlmostEqual(v.norm(), math.sqrt(2))
```

For a complex vector $\mathbf{v}$, Meep defines the norm as $\sqrt{\text{Re}(\mathbf{v} \cdot \mathbf{v}^*)}$, which here gives $\sqrt{1^2 + 1^2} = \sqrt{2}$.

#### Key Takeaways

- Meep's analytic geometry layer determines subpixel-smoothed permittivities; correctness of containment tests directly determines simulation accuracy at interfaces.
- The `geometric_objects_lattice_duplicates` helper handles non-orthogonal Bravais lattices, enabling compact specification of photonic crystal unit cells.
- `Medium.transform` propagates rotation to all susceptibilities atomically, so anisotropic dispersive materials can be oriented arbitrarily without manually recomputing tensor components.
- Declaring `valid_freq_range` on custom-fit dispersive materials lets Meep warn users automatically when source bandwidths stray outside the reliable fitting range.
- `Vector3` supports both real and complex components and interoperates with NumPy arrays via the standard `__array__` protocol.

---

### 2. `test_prism.py` — Non-Convex Prisms, Marching-Squares Vertices, and GDSII Import

**Physics:** Verifies that polygonal prism objects with non-convex outlines and large vertex counts faithfully represent complex fabricated structures, and demonstrates convergence of integrated dielectric volume with increasing vertex density.
**Difficulty:** Intermediate
**Source:** `python/tests/test_prism.py`
**Test Status:** PASS (183.9 s)

#### Theory

Real photonic devices are almost never composed of ideal spheres and cylinders; they are etched from lithographically patterned wafers whose cross-sections are arbitrary polygons captured in GDSII layout files. Meep's `Prism` object accepts an ordered list of 2D vertices and extrudes them along the $z$-axis to a specified height, creating a generalized polyhedron. Unlike `Block` or `Cylinder`, `Prism` supports non-convex polygons, which are required to model structures like spirals, L-shaped waveguides, or grating teeth with re-entrant angles.

The critical challenge for non-convex prisms is accurate subpixel smoothing. At every voxel straddling the boundary, Meep must determine what fraction of the voxel lies inside the prism, and must compute the correct normal-direction-weighted average of the permittivity tensor components. For convex objects, the inside/outside test is a simple half-plane intersection. For non-convex polygons, the algorithm uses a ray-casting or winding-number approach that remains correct even for self-intersecting or highly indented outlines.

The marching-squares algorithm (`skimage.measure.find_contours`) is the standard method for extracting iso-contour polygons from raster images. It produces a dense, ordered list of (x, y) coordinates tracing the boundary of any binary mask at sub-pixel precision. This is exactly the workflow used in practice to import fabricated device outlines from SEM images or from level-set topology optimization results: the optimizer produces a density field, and the final manufacturing intent is extracted as a polygon. The test checks that as the number of vertices $N$ extracted from an analytic "blob" shape increases, the integrated dielectric volume $\int \epsilon(\mathbf{r}) \, d^2r$ converges to a reference value computed at $N \to \infty$.

The convex case tests use a circle of radius $r$ approximated by $N$ equally spaced vertices. The exact dielectric volume is $\epsilon \cdot \pi r^2$; the polygon approximates this with area $\frac{N}{2} r^2 \sin(2\pi/N)$, which converges as $O(N^{-2})$ before subpixel smoothing and faster afterward. The test compares the prism integral against a native `Cylinder` integral at the same resolution, confirming that both representations converge to the same value.

The GDSII import path (`mp.get_GDSII_prisms`) reads a spiral inductor layout and returns a list of `Prism` objects, one per polygon in the specified GDSII layer. This workflow is critical for co-simulation of photonic integrated circuits: designers can export their layout tool output directly and simulate it without any intermediate geometry description.

Convergence is verified by a monotonicity assertion: if $d_a$ and $d_b$ are the relative errors at vertex counts $N_a < N_b$, then $d_b < d_a$ must hold. The active test (`test_prism`) focuses on non-convex blob #3 with 164 and 336 vertices, requiring the higher-vertex result to be both closer to the reference and within 2% relative error.

#### Code Walkthrough

**Non-convex prism from marching-squares vertices:**

```python
vertices_data = vertices_obj[f"N{npts}"]
vertices = [mp.Vector3(v[0], v[1], 0) for v in vertices_data]

geometry = [mp.Prism(vertices, height=mp.inf, material=mp.Medium(epsilon=12))]

sim = mp.Simulation(cell_size=cell, geometry=geometry, resolution=resolution)
sim.init_sim()

prism_eps = sim.integrate_field_function([mp.Dielectric], lambda r, eps: eps)
```

The `height=mp.inf` makes the prism infinite in $z$, equivalent to a 2D simulation cross-section. `integrate_field_function` performs a weighted quadrature over the grid with the specified field component as the integrand; here `mp.Dielectric` gives $\epsilon(x,y)$ at each grid point and the lambda simply returns it, so the integral is $\int \epsilon \, dA$.

**Convex prism vs. cylinder comparison:**

```python
angles = 2 * np.pi / npts * np.arange(npts)
vertices = [mp.Vector3(r * np.cos(ang), r * np.sin(ang)) for ang in angles]
geometry = [mp.Prism(vertices, height=mp.inf, material=mp.Medium(epsilon=12))]
# ... then compare to:
geometry = [mp.Cylinder(radius=r, height=mp.inf, material=mp.Medium(epsilon=12))]

return abs((prism_eps - cyl_eps) / cyl_eps)
```

Symmetry exploitation is tested explicitly: `mp.Mirror(direction=mp.X)` and `mp.Mirror(direction=mp.Y)` are applied to both the prism and cylinder simulations, and the relative error must be identical to three decimal places, confirming that symmetry reduction does not alter the subpixel averaging logic.

**GDSII import:**

```python
geometry = mp.get_GDSII_prisms(mp.Medium(index=3.5), gdsii_file, 0, 0, mp.inf)
```

Arguments are: material, GDSII filename, layer number, datatype, height. The function reads the GDSII database, extracts all polygons on the specified layer/datatype combination, and returns one `Prism` per polygon with the given material.

**Convergence assertion:**

```python
self.assertLess(abs((d3_b - d3_ref) / d3_ref), abs((d3_a - d3_ref) / d3_ref))
self.assertLess(abs((d3_b - d3_ref) / d3_ref), 0.02)
```

`d3_a` uses 164 vertices; `d3_b` uses 336. The 2% absolute tolerance combined with the monotonicity requirement ensures that the prism implementation is both accurate and consistent.

#### Key Takeaways

- `mp.Prism` supports arbitrary non-convex polygons, enabling direct import of lithographic layouts without any geometric simplification.
- The marching-squares workflow (raster mask → contour polygon → `Prism`) is the standard pipeline connecting image-based or optimization-based design to FDTD simulation.
- `mp.get_GDSII_prisms` provides one-line GDSII layer import, returning simulation-ready `Prism` objects with user-specified materials.
- Convergence of `integrate_field_function` with vertex count serves as a proxy for the accuracy of subpixel smoothing at curved boundaries approximated by polygons.
- Mirror symmetries (`mp.Mirror`) are compatible with `Prism` geometries and must not alter the integrated dielectric volume, which is verified explicitly.

---

### 3. `test_material_grid.py` — Material Grids, Subpixel Smoothing, and Topology Optimization

**Physics:** Demonstrates that `MaterialGrid` — a spatially varying permittivity specified on a dense rectangular grid — supports subpixel smoothing and converges with increasing FDTD resolution at better-than-first-order rates comparable to analytic geometric objects.
**Difficulty:** Advanced
**Source:** `python/tests/test_material_grid.py`
**Test Status:** TIMEOUT (CPU-intensive; high-resolution 2D and 3D resonant-mode calculations)

#### Theory

Topology optimization in photonics requires that the permittivity field $\epsilon(\mathbf{r})$ be a continuous, differentiable function of design variables. The standard approach represents $\epsilon$ as a weighted interpolation between two materials:

$$\epsilon(\mathbf{r}; \mathbf{w}) = (1 - \tilde{p}(\mathbf{r})) \, \epsilon_{\min} + \tilde{p}(\mathbf{r}) \, \epsilon_{\max}$$

where $\mathbf{w} \in [0, 1]^{N_x \times N_y}$ is the weight field (the design variables), and $\tilde{p}$ is a projected, filtered version of $\mathbf{w}$. The projection uses a hyperbolic tangent thresholding:

$$\tilde{p} = \frac{\tanh(\beta \eta) + \tanh(\beta (w - \eta))}{\tanh(\beta \eta) + \tanh(\beta (1 - \eta))}$$

with $\beta$ controlling sharpness and $\eta$ the threshold level. As $\beta \to \infty$, the field approaches a binary (0 or 1) distribution. The test uses $\beta = 1000$, $\eta = 0.5$, which effectively binarizes the weight field.

`MaterialGrid` stores this weight array on a grid that is independent of the FDTD Yee grid. When computing the permittivity at a Yee grid point, Meep bilinearly interpolates the weight field. When `do_averaging=True`, Meep goes further: for each Yee voxel that straddles a transition region in the weight field (where the projected weights change between 0 and 1), it applies the same subpixel averaging procedure used for analytic geometric objects. This is the key capability that makes material grids useful in practice — without subpixel averaging, the convergence with FDTD resolution would be only first-order (the dielectric staircase error), causing gradient calculations in the adjoint optimizer to be unreliable.

The `test_subpixel_smoothing` test checks this claim quantitatively. It constructs a 2D `MaterialGrid` encoding a disk of radius 0.301943 in a $1 \times 1$ unit cell (using a Gaussian-filtered weight field to produce smooth transitions), then computes the resonant frequency of the whispering-gallery mode using `Harminv` at two resolutions (25 and 50). The convergence ratio must exceed linear: the condition

$$|f_{50} - f_\text{ref}| \cdot \frac{50}{25} < |f_{25} - f_\text{ref}|$$

is equivalent to requiring the error to drop by more than a factor of 2 when the resolution doubles — i.e., better than first-order convergence, which confirms subpixel smoothing is active.

The 3D test (`test_matgrid_3d`) verifies that a `MaterialGrid` encoding a silicon sphere in silicon dioxide matches the resonant frequency obtained from an analytic `Sphere` object to two decimal places in frequency units. This confirms that the grid interpolation and subpixel averaging produce the same effective dielectric profile as the analytic boundary description.

The symmetry test (`test_symmetry`) checks that when a `MaterialGrid` is declared to have left-right symmetry (by supplying `weights = 0.5*(w + np.fliplr(w))`) and an additional `Block` with `e2=mp.Vector3(y=-1)` (which maps $y \to -y$), the computed transmittance through an eigenmode monitor agrees with the unsymmetrized case to five decimal places. This validates that the symmetry-enforcement mechanism does not alter the physics.

The `grid_type` parameter controls how multiple overlapping material grid contributions are combined. `"U_MEAN"` takes the arithmetic mean of all overlapping weights; other options include `"U_MIN"`, `"U_MAX"`, and `"U_DEFAULT"` (last-object-wins). The transmittance test uses `"U_MEAN"` without subpixel averaging (`do_averaging=False`), isolating the symmetry mechanism from the smoothing.

#### Code Walkthrough

**Setting up a binarized material grid for 2D resonant mode:**

```python
Nx, Ny = int(matgrid_size.x * matgrid_resolution), int(matgrid_size.y * matgrid_resolution)
x = np.linspace(-0.5, 0.5, Nx)
xv, yv = np.meshgrid(x, y)
weights = np.sqrt(np.square(xv) + np.square(yv)) < rad
filtered_weights = gaussian_filter(weights, sigma=3.0, output=np.double)

matgrid = mp.MaterialGrid(
    mp.Vector3(Nx, Ny),
    mp.air,
    mp.Medium(index=3.5),
    weights=filtered_weights,
    do_averaging=True,
    beta=1000,
    eta=0.5,
)
```

The Gaussian filter (`sigma=3.0` pixels) creates a smooth transition zone around the disk boundary. The large $\beta=1000$ then re-binarizes the filtered weights, producing a sharp interface with a thin smooth transition layer that is exactly what subpixel smoothing needs.

**Embedding the material grid in a geometric object:**

```python
geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(matgrid_size.x, matgrid_size.y, 0),
        material=matgrid,
    )
]
```

The `MaterialGrid` is used as the `material` of a `Block`. This means the grid's permittivity values only apply within the block's bounding box; outside the block the background (or other geometry objects) takes precedence.

**Using material grid as `default_material`:**

```python
sim = mp.Simulation(
    resolution=res,
    cell_size=cell_size,
    default_material=matgrid if default_mat else mp.Medium(),
    geometry=[] if default_mat else geometry,
    ...
)
```

When the grid spans the entire cell, it can be passed as `default_material` directly, bypassing the `Block` wrapper entirely. The test verifies that both formulations give identical resonant frequencies.

**3D silicon sphere via material grid:**

```python
coord = np.linspace(-0.5*s, 0.5*s, N)
xv, yv, zv = np.meshgrid(coord, coord, coord)
weights = np.sqrt(np.square(xv) + np.square(yv) + np.square(zv)) < rad
filtered_weights = gaussian_filter(weights, sigma=4/resolution, output=np.double)

matgrid = mp.MaterialGrid(mp.Vector3(N, N, N), SiO2, Si,
                          weights=filtered_weights, do_averaging=True,
                          beta=1000, eta=0.5)
```

The Gaussian sigma is scaled with resolution (`4/resolution` pixels) to keep the physical transition width constant as resolution changes.

**Transmittance with symmetry enforcement:**

```python
weights = 0.5 * (w + np.fliplr(w))   # enforce left-right symmetry

matgrid = mp.MaterialGrid(..., weights=weights, grid_type="U_MEAN")

geometry.append(mp.Block(..., material=matgrid, e2=mp.Vector3(y=-1)))
```

The mirrored block with `e2=mp.Vector3(y=-1)` tells Meep to look up the material grid with $y$ reflected, effectively overlaying a mirror-image copy. Combined with `grid_type="U_MEAN"`, this averages the original and reflected grids, giving a symmetric permittivity map.

#### Key Takeaways

- `MaterialGrid` enables continuous, gradient-friendly permittivity fields required for adjoint-based topology optimization.
- With `do_averaging=True`, `MaterialGrid` achieves super-linear convergence with FDTD resolution, comparable to analytic geometric objects, because Meep applies the same subpixel smoothing algorithm at material transitions.
- The projection parameters `beta` and `eta` implement hyperbolic-tangent thresholding to drive weights toward binary values, with `beta=1000` and `eta=0.5` effectively binarizing the field.
- The `grid_type` parameter controls how overlapping material grids are combined; `"U_MEAN"` enables symmetry enforcement by averaging a grid with its mirror image.
- A `MaterialGrid` can serve as both a geometry object's material and as `default_material`, providing flexibility for full-cell or sub-region optimization domains.

---

### 4. `test_source.py` — Source Types, Amplitude Functions, and EigenMode Sources

**Physics:** Exercises all source injection mechanisms in Meep, including Gaussian and continuous sources with scalar and spatially varying amplitudes, custom time-domain waveforms, and eigenmode sources with chirped time profiles.
**Difficulty:** Intermediate
**Source:** `python/tests/test_source.py`
**Test Status:** PASS (18.9 s)

#### Theory

In an FDTD simulation, electromagnetic sources inject energy into the computational domain by adding a prescribed current $\mathbf{J}(\mathbf{r}, t)$ to the curl equations at each time step. The spatial profile of the source (which field component is excited, over what volume, with what amplitude weighting) and its temporal profile (the time envelope) together determine the modes excited and the bandwidth covered.

Meep provides four source time profiles: `GaussianSource`, `ContinuousSource`, `CustomSource`, and (implicitly) `EigenModeSource`'s internal MPB-computed mode profile. A `GaussianSource` with center frequency $f_0$ and fractional bandwidth `fwidth` has the envelope

$$g(t) = e^{-(t - t_0)^2 / (2\sigma^2)} \cos(2\pi f_0 t)$$

where $\sigma = 1/(2\pi \cdot \text{fwidth} \cdot f_0)$. The broadband content makes it ideal for computing transmission spectra via DFT monitors. A `ContinuousSource` is a sinusoid turned on gradually via a smooth Gaussian turn-on, useful for steady-state field profiles or resonance excitation.

The spatial amplitude of a source can be specified in three equivalent ways: (1) a Python callable `amp_func(r)` evaluated at each source point, (2) an HDF5 file containing a pre-computed amplitude array on a regular grid (`amp_func_file`), or (3) a NumPy array passed directly via `amp_data`. The test `test_amp_file_func` verifies that all three produce identical field values to four decimal places. This equivalence is important because the HDF5 path enables storing large amplitude datasets without Python-level per-point function calls, which would be prohibitively slow for high-resolution 3D simulations.

`EigenModeSource` computes the mode profile numerically using MPB at simulation initialization time. Given a cross-section (the source region), it launches the eigenmode of the waveguide formed by the geometry intersecting that plane. The `eig_parity`, `eig_band`, and `eig_kpoint` parameters select which mode to launch. This is the standard method for exciting a single guided mode in waveguide simulations, far superior to a simple dipole source which would excite all modes simultaneously.

The chirped eigenmode source test combines `EigenModeSource` with `CustomSource(src_func=chirp)`, where the chirp has both a Gaussian amplitude envelope and a quadratic phase:

$$c(t) = e^{-a(t-t_0)^2} \cdot e^{i[2\pi f_0 (t-t_0) + b(t-t_0)^2]}$$

with $b < 0$ for a down-chirp. The `center_frequency=v0` argument ensures that MPB computes the eigenmode at frequency $v_0$, even though the actual temporal waveform is not purely sinusoidal at that frequency. The test only checks that the simulation runs without the fields diverging — confirming that this combination of source types is numerically stable.

The low-level SWIG typemap test (`test_typemap_swig` vs `test_typemap_py`) verifies that the Python-layer `GaussianSource` object is interchangeable with the C++-layer `mp.gaussian_src_time` object when passed to `fields.add_volume_source`. This is a SWIG interface consistency check that prevents silent type errors when mixing high-level and low-level API calls.

#### Code Walkthrough

**Source frequency specified by wavelength vs. frequency:**

```python
g_src = GaussianSource(wavelength=10)
self.assertAlmostEqual(1.0 / 10.0, g_src.frequency)

g_src = GaussianSource(10)       # positional: frequency
self.assertEqual(10, g_src.frequency)
```

Both `frequency` and `wavelength` keyword arguments are accepted; Meep stores frequency internally and converts wavelength via $f = 1/\lambda$ (in natural units where $c = 1$).

**Custom bump-function source with `Harminv` mode extraction:**

```python
def my_src_func(t):
    return math.exp(-1/(1 - ((t-1)**2))) if 0 < t < 2 else 0j

sources = [mp.Source(
    src=mp.CustomSource(src_func=my_src_func, end_time=100),
    component=mp.Ez, center=mp.Vector3(r + 0.1),
)]
h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
sim.run(mp.after_sources(h), until_after_sources=200)
fp = sim.get_field_point(mp.Ez, mp.Vector3(1))
self.assertAlmostEqual(fp, -0.021997617628500023 + 0j, 7)
```

The bump function is $C^\infty$ with compact support on $(0, 2)$, so it excites all frequencies in the band without the Gaussian's infinite-support tails. `end_time=100` tells Meep when to stop calling the function. `Harminv` extracts the resonant mode from the decaying time-domain signal after the source turns off.

**Amplitude function, file, and array equivalence:**

```python
amp_fun = lambda p: p.x + 2*p.y

sources = [mp.Source(..., amp_func=amp_fun)]       # Python callable
sources = [mp.Source(..., amp_func_file="file:dataset")]  # HDF5 file
sources = [mp.Source(..., amp_data=self.amp_data)]  # NumPy array

self.assertAlmostEqual(field_point_amp_file, field_point_amp_func, places=4)
self.assertAlmostEqual(field_point_amp_arr,  field_point_amp_func, places=4)
```

**Change sources mid-simulation:**

```python
sim.restart_fields()
sim.clear_dft_monitors()
sim.change_sources(default_lattice)
sim.run(until=1)

sim.reset_meep()
sim.change_sources(amp_source)
sim.run(until=1)
self.assertTrue(sim.sources[0].amp_func is ampfunc)
```

`change_sources` replaces the source list without reconstructing the entire simulation, enabling parameter sweeps that reuse the same structure. After `reset_meep()` and switching back, the `amp_func` reference is preserved on the Python `Source` object.

**Chirped eigenmode source:**

```python
chirp = lambda t: np.exp(1j*2*np.pi*v0*(t-t0)) * np.exp(-a*(t-t0)**2 + 1j*b*(t-t0)**2)

sources = [mp.EigenModeSource(
    src=mp.CustomSource(src_func=chirp, center_frequency=v0),
    eig_kpoint=mp.Vector3(kx),
    eig_band=1,
    eig_parity=mp.EVEN_Y + mp.ODD_Z,
    eig_match_freq=True,
)]
```

`center_frequency=v0` tells MPB which frequency to use when solving for the eigenmode cross-section profile, decoupled from the custom time waveform.

#### Key Takeaways

- Meep accepts `frequency` or `wavelength` as constructor arguments for all source time profiles; internally everything is stored in frequency units ($c = 1$).
- Amplitude weighting via `amp_func`, `amp_func_file`, and `amp_data` are numerically equivalent; the file and array forms avoid per-timestep Python callbacks and are preferable for large 3D sources.
- `EigenModeSource` solves for the waveguide mode profile at simulation initialization using MPB; it accepts any `src` time profile, including `CustomSource` with chirps or arbitrary waveforms.
- `change_sources` enables efficient parameter sweeps by replacing the source list without rebuilding the structure; Python-level attributes (including `amp_func` references) are preserved.
- The `test_typemap` tests confirm that Python-level source objects and their C++-level SWIG equivalents are interchangeable, preventing silent type mismatch bugs.

---

### 5. `test_simulation.py` — Simulation Infrastructure, Output, and Field Diagnostics

**Physics:** Validates the orchestration layer of the `Simulation` class: output file management, interpolation utilities, field array extraction, dimension inference, geometry updates mid-run, timing profiling, and modal volume calculations.
**Difficulty:** Intermediate
**Source:** `python/tests/test_simulation.py`
**Test Status:** PASS (33.3 s)

#### Theory

The `Simulation` class in `python/simulation.py` is the primary user-facing object in Meep's Python API. It manages the lifecycle of a simulation: constructing the C++ `structure` and `fields` objects, applying sources and boundary conditions, driving the time-stepping loop, and routing output to HDF5 files or NumPy arrays. The tests in `test_simulation.py` cover this orchestration layer in depth, verifying that the various callback mechanisms, output functions, and diagnostic utilities work correctly.

The Meep time-stepping loop accepts step functions — Python callables or built-in C++ functions — that are invoked at specified simulation times or events. Built-in combinators include `mp.at_time(t, fn)`, `mp.at_end(fn)`, `mp.after_sources(fn)`, and `mp.after_sources_and_time(dt, fn)`. These are composable: `mp.with_prefix("pfx-", fn)` wraps any output function to prepend a string to its HDF5 filename. The test `test_with_prefix` checks that the resulting file exists at the expected path. This design allows users to collect multiple diagnostic snapshots in a single `sim.run(...)` call without writing any loops.

The dimension-inference logic (`test_infer_dimensions` and `test_require_dimensions`) demonstrates that Meep determines simulation dimensionality from the cell size: if `cell_size.z == 0`, the simulation is 2D. Before `init_sim` is called, `sim.dimensions` returns 3 (the default). After `_init_structure()`, it is re-examined and set to 2. The `require_dimensions()` method forces this inference early, which is necessary when other setup logic (such as symmetry specification) depends on knowing the dimensionality.

The `test_set_materials` test shows how geometry can change mid-simulation using `sim.set_materials(geometry=geom)`. This re-runs the `set_epsilon` routine at the current time step, constructing a new dielectric profile. The geometry depends on the current meep time via `fn = t * 0.02`, so the cylinder and ellipsoid move as the simulation progresses. The test verifies that the epsilon array extracted at time 50 differs from the array at time 200, confirming that the update took effect.

Field diagnostics are validated in `test_get_array_output` by cross-checking that `sim.get_epsilon()` and `sim.get_efield_z()` return arrays numerically identical to those written to HDF5 files by `mp.output_epsilon(sim)` and `mp.output_efield_z(sim)`. This round-trip test is important because the in-memory and on-disk paths go through different code (SWIG typemaps vs. HDF5 write routines), and small discrepancies would indicate a bug in either path.

The `test_timing_data` test exercises `sim.get_timing_data()`, which returns a dictionary mapping each computational phase (`mp.Stepping`, `mp.FieldUpdateB`, `mp.Boundaries`, etc.) to a list of wall-clock times, one per MPI rank. The test verifies both that the expected phases appear in the dictionary and that the time hierarchy is consistent: the total stepping time must be at least the sum of its component updates and DFT transformations.

The `test_epsilon_input_file` test loads a pre-computed permittivity grid from an HDF5 file (`cyl-ellipsoid-eps-ref.h5`) and uses it as `sim.epsilon_input_file`. This enables simulating structures whose dielectric profile was computed by an external tool or stored from a previous simulation. The `test_numpy_epsilon` variant passes the same data as a NumPy array in `sim.default_material`, verifying that the two input paths produce identical field values.

#### Code Walkthrough

**Standard simulation factory used across all tests:**

```python
def init_simple_simulation(self, **kwargs):
    return mp.Simulation(
        resolution=20, cell_size=mp.Vector3(10, 10),
        boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=1.0),
                           center=mp.Vector3(), component=mp.Ez)],
        symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y)],
        **kwargs,
    )
```

Mirror symmetries halve the computational domain twice, reducing memory and time by a factor of 4 for this 2D geometry.

**Step-function combinators and HDF5 output:**

```python
sim.run(mp.at_time(100, mp.output_efield_z), until=200)
fname = f"{sim.get_filename_prefix()}-ez-000100.00.h5"
self.assertTrue(os.path.exists(os.path.join(output_dir, fname)))
```

`mp.at_time(100, fn)` fires `fn` when meep-time first reaches 100. The HDF5 filename encodes the field component (`ez`) and the simulation time (`000100.00`).

**Mid-run geometry update:**

```python
def change_geom(sim):
    fn = sim.meep_time() * 0.02
    geom = [mp.Cylinder(radius=3, material=mp.Medium(index=3.5),
                         center=mp.Vector3(fn, fn)),
            mp.Ellipsoid(size=mp.Vector3(1, 2, mp.inf),
                          center=mp.Vector3(fn, fn))]
    sim.set_materials(geometry=geom)

sim.run(mp.at_time(100, change_geom), until=200)
```

`sim.set_materials` triggers a full rebuild of the C++ `structure` object from the new geometry list, including fresh subpixel smoothing over the updated object boundaries.

**Dimension inference:**

```python
sim = self.init_simple_simulation()
self.assertEqual(sim.dimensions, 3)   # before init
sim._init_structure()
self.assertEqual(sim.dimensions, 2)   # cell_size.z == 0 => 2D
```

**Round-trip field array verification:**

```python
eps_arr = sim.get_epsilon(snap=True)
mp.output_epsilon(sim)  # writes HDF5

with h5py.File(fname_fmt.format("eps"), "r") as f:
    eps = f["eps"][()]

np.testing.assert_allclose(eps, eps_arr)
```

`snap=True` forces time-synchronization of E and H fields (which are staggered by half a timestep in the Yee scheme) before returning the array.

**Timing data hierarchy verification:**

```python
timing_data = sim.get_timing_data()
self.assertGreaterEqual(
    sum(timing_data[mp.Stepping]),
    sum(timing_data[mp.FieldUpdateB])
    + sum(timing_data[mp.FieldUpdateH])
    + sum(timing_data[mp.FieldUpdateD])
    + sum(timing_data[mp.FieldUpdateE])
    + sum(timing_data[mp.FourierTransforming]),
)
```

**Geometry center offset:**

```python
sim = mp.Simulation(..., geometry_center=mp.Vector3(2, -1))
```

`geometry_center` shifts the coordinate origin of all geometry objects without changing the computational cell, enabling simulations where the physical structure is not centered at the origin. The test verifies the field at `mp.Vector3(2, -1)` matches an expected value computed with the offset geometry.

#### Key Takeaways

- Meep's step-function combinators (`at_time`, `at_end`, `after_sources`, `with_prefix`) compose cleanly, enabling complex diagnostic schedules in a single `sim.run()` call.
- Dimension inference from `cell_size.z == 0` is automatic but can be forced early via `require_dimensions()` for setup code that depends on dimensionality.
- `sim.set_materials(geometry=geom)` allows dynamic geometry updates mid-simulation, re-triggering full subpixel smoothing; useful for moving-object or time-varying structure simulations.
- `sim.get_epsilon()` and the HDF5-based `mp.output_epsilon()` produce numerically identical arrays, confirming consistency between the in-memory and on-disk output paths.
- `sim.get_timing_data()` returns per-rank timing for all FDTD computational phases, useful for diagnosing load imbalance in MPI runs and for identifying bottlenecks such as DFT computation or boundary communication.

---

### 6. `test_get_epsilon_grid.py` — Epsilon Grid Queries over Mixed Geometry and Material Grids

**Physics:** Verifies that `get_epsilon_grid`, which evaluates the permittivity tensor at arbitrary user-specified spatial coordinates, agrees with `get_epsilon_point` for mixed geometries combining cylinders, blocks with dispersion, prisms, and material grids.
**Difficulty:** Intermediate
**Source:** `python/tests/test_get_epsilon_grid.py`
**Test Status:** FAIL (missing `parameterized` package; install with `pip install parameterized`)

#### Theory

After `init_sim` completes, Meep has assigned an effective permittivity to every Yee grid voxel through the subpixel smoothing algorithm. Users often need to query the permittivity at specific points for postprocessing, visualization, or to verify that the dielectric profile matches the intended design. Meep provides two query interfaces: `get_epsilon_point(pt, freq)` returns a scalar at a single point, while `get_epsilon_grid(xarr, yarr, zarr, freq)` evaluates the permittivity over a batch of coordinates, more efficiently for large arrays.

The test constructs a deliberately heterogeneous geometry combining four different object types and material sources:

1. A `Cylinder` with a simple isotropic dielectric (`index=1.5`, $\epsilon = 2.25$).
2. A `Block` using `SiN` — a real dispersive material from `meep.materials` with a frequency-dependent refractive index.
3. A `Block` whose material is a `MaterialGrid` encoding a ring structure (annular region where the inner circle has one material and the outer another), with `do_averaging=False` and `beta=0`, producing a raw, unsmoothed binary field.
4. A `Prism` using `Co` (cobalt), a gyrotropic metal from `meep.materials` with off-diagonal permittivity tensor components.

The key correctness property being verified is that `get_epsilon_grid` and `get_epsilon_point` return numerically identical results (to six decimal places) at each of four test points, one in each material region. This matters because the two functions use different code paths: `get_epsilon_point` queries the C++ `fields` object directly at a single coordinate, while `get_epsilon_grid` may use a different interpolation or averaging strategy. Disagreement would indicate a subtle interface mismatch.

The test is parameterized over four (point, frequency) pairs:
- $(0.2, 0.2)$ at $f = 1.1$ — inside the `MaterialGrid` block.
- $(-0.2, 0.1)$ at $f = 0.7$ — inside the `MaterialGrid` block but at a different frequency (SiN's dispersion is relevant here).
- $(-0.2, -0.25)$ at $f = 0.55$ — inside the `SiN` block.
- $(0.4, 0.1)$ at $f = 0$ — outside all objects (in the background `mp.air`); $f = 0$ means evaluate the DC (zero-frequency, non-dispersive) permittivity.

The frequency argument matters for dispersive materials: SiN and Co have frequency-dependent $\epsilon(\omega)$ computed from Lorentzian fits. At $f = 0$, only the DC contribution (the static permittivity) is returned. At finite frequency, the susceptibility contributions $\sum_k \sigma_k f_k^2 / (f_k^2 - f^2 - if\gamma_k)$ add to the background epsilon.

The `eps_averaging=False` flag in the `Simulation` constructor disables subpixel smoothing globally. Without subpixel averaging, each Yee grid point takes the permittivity of whichever object it lies inside, with no boundary blending. This makes the test independent of the smoothing algorithm and focuses purely on the query consistency between the two getter functions.

#### Code Walkthrough

**Heterogeneous geometry construction:**

```python
geometry = [
    mp.Cylinder(center=mp.Vector3(0.35, 0.1), radius=0.1,
                height=mp.inf, material=mp.Medium(index=1.5)),
    mp.Block(center=mp.Vector3(-0.15, -0.2), size=mp.Vector3(0.2, 0.24, mp.inf),
             material=SiN),
    mp.Block(center=mp.Vector3(-0.2, 0.2), size=mp.Vector3(0.4, 0.4, mp.inf),
             material=matgrid),
    mp.Prism(vertices=[mp.Vector3(0.05, 0.45), mp.Vector3(0.32, 0.22),
                        mp.Vector3(0.15, 0.10)],
             height=0.5, material=Co),
]
```

This geometry fits inside a $1 \times 1$ cell. Objects are listed in priority order — later objects override earlier ones where they overlap. The `Prism` has finite height (0.5) while the cylinder and blocks have `height=mp.inf`, but since this is a 2D simulation the $z$-extent is irrelevant for the epsilon grid query.

**Ring-shaped material grid:**

```python
weights = np.logical_and(
    np.sqrt(np.square(xv) + np.square(yv)) > rad,      # outside inner circle
    np.sqrt(np.square(xv) + np.square(yv)) < rad + w,  # inside outer circle
)
matgrid = mp.MaterialGrid(
    mp.Vector3(Nx, Ny), mp.air, mp.Medium(index=3.5),
    weights=weights, do_averaging=False, beta=0, eta=0.5,
)
```

With `beta=0` the projection is linear (no thresholding), and `do_averaging=False` means no subpixel blending. The weight array is Boolean (True/False), so after linear projection the grid is simply $\epsilon = 1$ (air) inside the inner radius and outside the outer radius, and $\epsilon = 12.25$ in the annular region.

**Parameterized test executing the dual query:**

```python
@parameterized.parameterized.expand([
    (mp.Vector3(0.2, 0.2),   1.1),
    (mp.Vector3(-0.2, 0.1),  0.7),
    (mp.Vector3(-0.2, -0.25), 0.55),
    (mp.Vector3(0.4, 0.1),   0),
])
def test_get_epsilon_grid(self, pt, freq):
    eps_grid = self.sim.get_epsilon_grid(
        np.array([pt.x]), np.array([pt.y]), np.array([0]), freq
    )
    eps_pt = self.sim.get_epsilon_point(pt, freq)
    self.assertAlmostEqual(np.real(eps_grid), np.real(eps_pt), places=6)
    self.assertAlmostEqual(np.imag(eps_grid), np.imag(eps_pt), places=6)
```

`get_epsilon_grid` accepts coordinate arrays (NumPy arrays of x, y, z values). Passing `np.array([0])` for $z$ specifies the $z = 0$ plane, appropriate for 2D simulations. The result is a complex number (real part is the effective dielectric constant, imaginary part reflects material losses or conductivities).

**Fixing the missing `parameterized` dependency:**

The test fails because `parameterized` is not bundled with conda-forge's pymeep package. Install it separately:

```bash
pip install parameterized
```

After installation, run the test with:

```bash
python python/tests/test_get_epsilon_grid.py
```

The `@parameterized.parameterized.expand` decorator generates four separate test methods from the list of `(pt, freq)` pairs, which is why `parameterized` must be available at import time; without it, the module fails to import entirely rather than skipping individual tests.

#### Key Takeaways

- `get_epsilon_grid` and `get_epsilon_point` must return numerically identical results; any discrepancy indicates a bug in the batch-query or single-point interpolation path.
- Heterogeneous geometry combining `Cylinder`, `Block`, `MaterialGrid`, and `Prism` with different dispersive materials tests the full epsilon assembly pipeline.
- Dispersive materials (`SiN`, `Co`) return complex, frequency-dependent $\epsilon(\omega)$; both real and imaginary parts must agree between the two query functions.
- `eps_averaging=False` isolates query correctness from subpixel smoothing, allowing focused testing of the epsilon retrieval interface.
- The test fails due to a missing `parameterized` package — a packaging gap in conda-forge pymeep; `pip install parameterized` resolves it without any code changes.

---

## Chapter Summary

This chapter has surveyed the geometry and material infrastructure of Meep from six complementary angles:

| Section | File | Core Concept | Status |
|---------|------|--------------|--------|
| 1 | `test_geom.py` | Primitive objects, lattice tiling, tensor transforms, frequency warnings | PASS |
| 2 | `test_prism.py` | Non-convex prisms, marching-squares vertices, GDSII import, convergence | PASS |
| 3 | `test_material_grid.py` | Spatially-varying epsilon, subpixel smoothing, topology optimization | TIMEOUT |
| 4 | `test_source.py` | Source types, amplitude functions, eigenmode sources, chirp | PASS |
| 5 | `test_simulation.py` | Simulation lifecycle, output, mid-run updates, timing, field queries | PASS |
| 6 | `test_get_epsilon_grid.py` | Batch epsilon queries over mixed geometry and material grids | FAIL (missing `parameterized`) |

A few cross-cutting themes emerge from reading these tests together. First, subpixel smoothing is the central algorithmic ingredient that connects geometric accuracy to FDTD convergence — it appears in discussions of prisms, material grids, and analytic objects alike. Second, Meep's geometry system is object-priority based: objects later in the `geometry` list override earlier ones, and materials can be any of the supported types (`Medium`, `MaterialGrid`, Python callable, or NumPy array). Third, the Python API is designed for composability: source types, step-function callbacks, output functions, and geometry operations all follow consistent conventions that allow mixing and matching without unexpected interactions.

Practitioners building on these foundations should note that `MaterialGrid` with `do_averaging=True` is essential for adjoint-gradient-based optimization workflows — the super-linear convergence it provides ensures that finite-difference gradient checks pass and that the optimizer converges smoothly. For fabrication-realistic device simulation, the `Prism` + GDSII import workflow eliminates the need for any intermediate geometry format. And for diagnostic postprocessing, the equivalence of `get_epsilon_grid` and `get_epsilon_point` guarantees that batch epsilon queries can be safely used in place of point queries without loss of accuracy.
