# Chapter 4: Photonic Crystals and MPB

This chapter covers photonic crystal band structure calculations and waveguide simulations using MPB (MIT Photonic Bands) integrated with Meep. MPB solves Maxwell's equations as an eigenvalue problem in the frequency domain to compute the exact band structure — the relationship between frequency and wavevector — for periodic dielectric structures. Topics include band gaps, Bloch modes, and photonic crystal waveguides across a range of lattice geometries (square, triangular, honeycomb, diamond, and one-dimensional Bragg stacks), as well as the coupling between MPB band structure results and Meep time-domain simulations.

---

### 1. `mpb_tutorial.py` — Comprehensive MPB Tutorial: Band Structures, Defect States, and Optimization

**Physics:** Computes TE and TM band structures for square and triangular lattices, optimizes the TM band gap radius, finds a point-defect cavity mode in a supercell, and tunes the defect epsilon to hit a target frequency.
**Difficulty:** Intermediate
**Source:** `python/examples/mpb_tutorial.py`
**Test Status:** TIMEOUT (CPU-intensive: runs multiple full band-structure solves plus optimization loops over a supercell)

#### Theory

Photonic crystals are periodic dielectric structures whose periodicity couples electromagnetic waves at wavelengths comparable to the lattice constant. When the dielectric contrast is sufficient, Bragg-like interference opens photonic band gaps — frequency ranges in which no propagating modes exist. The phenomenon is the electromagnetic analog of electronic band gaps in semiconductors, governed by the same Bloch theorem.

For a periodic dielectric function epsilon(**r**) = epsilon(**r** + **R**) (where **R** is any lattice vector), Bloch's theorem states that the electromagnetic eigenmodes take the form:

    H_nk(r) = e^(i k . r) * u_nk(r)

where u_nk(**r**) has the same periodicity as the lattice. The eigenvalue problem that MPB solves is the curl-of-curl equation:

    (1/sqrt(eps)) * curl * curl * (1/sqrt(eps)) * H = (omega/c)^2 * H

This is a Hermitian operator, guaranteeing real eigenvalues omega_n(**k**). For each wavevector **k** in the first Brillouin zone there is a discrete spectrum of bands indexed by n. Plotting omega_n(**k**) along high-symmetry paths through the Brillouin zone gives the photonic band structure.

For a square lattice of dielectric rods (epsilon = 12) in air, the relevant Brillouin zone has high-symmetry points Gamma (k = 0), X (k = pi/a along a lattice axis), and M (k = (pi/a, pi/a) at the zone corner). TM polarization (electric field along the rod axis) typically opens a band gap between bands 1 and 2 because the TM field prefers to concentrate inside the high-epsilon rods. TE polarization (magnetic field along the rod axis, electric field in the plane) requires a different geometry optimization.

A complete photonic band gap — a gap in all polarizations and all directions simultaneously — is the holy grail for applications such as lossless optical confinement and zero-threshold lasing. The tutorial also demonstrates defect engineering: removing a single rod from the periodic lattice creates a donor-like point defect whose eigenfrequency pulls down from the upper band edge into the gap. The defect mode is exponentially localized because no propagating bulk modes exist at gap frequencies.

Band-gap optimization is performed by varying the rod radius r and using `scipy.optimize.minimize_scalar` to maximize `ms.retrieve_gap(1)`, which returns the gap-to-midgap ratio of the first gap. The targeted solver (`ms.target_freq`) converges much faster than a full diagonalization when only a single defect mode near a known frequency is needed.

#### Code Walkthrough

Setting up the square-lattice band structure:

```python
k_points = mp.interpolate(4, [mp.Vector3(),      # Gamma
                               mp.Vector3(0.5),   # X
                               mp.Vector3(0.5,0.5), # M
                               mp.Vector3()])      # Gamma
geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))]
geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
ms = mpb.ModeSolver(num_bands=8, k_points=k_points,
                    geometry=geometry,
                    geometry_lattice=geometry_lattice,
                    resolution=32)
ms.run_te()
ms.run_tm()
```

`mp.interpolate(4, points)` inserts 4 evenly spaced intermediate k-points between each listed high-symmetry point, producing a smooth band diagram. `run_te()` enforces in-plane electric fields; `run_tm()` enforces out-of-plane electric fields.

Maximizing the TM gap:

```python
def first_tm_gap(r):
    ms.geometry = [mp.Cylinder(r, material=mp.Medium(epsilon=12))]
    ms.run_tm()
    return -1 * ms.retrieve_gap(1)   # negative because we minimize

result = minimize_scalar(first_tm_gap, method='bounded',
                         bounds=[0.1, 0.5], options={'xatol': 0.1})
```

`retrieve_gap(1)` returns the gap-to-midgap ratio (in percent) of the first band gap. Negating it converts the maximization into a minimization problem.

Constructing the 5x5 supercell point defect:

```python
ms.geometry_lattice = mp.Lattice(size=mp.Vector3(5, 5))
ms.geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))]
ms.geometry = mp.geometric_objects_lattice_duplicates(ms.geometry_lattice, ms.geometry)
ms.geometry.append(mp.Cylinder(0.2, material=mp.air))  # erase one rod
ms.k_points = [mp.Vector3(0.5, 0.5)]
ms.num_bands = 50
ms.run_tm()
```

`geometric_objects_lattice_duplicates` tiles the unit-cell geometry over the 5x5 supercell. The appended air cylinder overwrites the central rod, forming the defect. The calculation requires 50 bands because the supercell has 25 times the unit-cell area, folding the bulk bands 25-fold.

Using the targeted solver to find only the defect mode:

```python
ms.num_bands = 1
ms.target_freq = (0.2812 + 0.4174) / 2   # midgap frequency
ms.tolerance = 1e-8
ms.run_tm()
```

The targeted solver uses a shift-and-invert technique to find eigenvalues near `target_freq`, avoiding the cost of computing all bands below the gap.

Tuning the defect frequency with Brent's root-finding algorithm:

```python
def rootfun(eps):
    ms.geometry = old_geometry + [mp.Cylinder(0.2, material=mp.Medium(epsilon=eps))]
    ms.run_tm()
    return ms.get_freqs()[0] - 0.314159

rooteps = ridder(rootfun, 1, 12)
```

Each call to `rootfun` runs a full targeted MPB solve. `ridder` from scipy finds the epsilon value that places the defect mode exactly at omega*a/2*pi = 0.314159.

#### Key Takeaways

- `ms.run_te()` and `ms.run_tm()` solve for modes with in-plane and out-of-plane electric fields respectively; `ms.run()` finds all modes without polarization restriction.
- `ms.retrieve_gap(n)` returns the gap-to-midgap ratio in percent for the gap between bands n and n+1; a positive value confirms a genuine gap.
- The targeted solver (`ms.target_freq`) dramatically accelerates defect calculations by focusing the eigensolver near a known frequency rather than diagonalizing the full spectrum.
- `geometric_objects_lattice_duplicates` tiles unit-cell geometry over a supercell, enabling point- and line-defect calculations within a periodic framework.
- Band gap size depends strongly on rod radius; optimization reveals that for TM modes in a square lattice of epsilon=12 rods, the optimal radius is near r = 0.2a.

---

### 2. `mpb_sq_rods.py` — Square Lattice of Dielectric Rods: TE and TM Band Structure

**Physics:** Computes the full TE and TM photonic band structure for a square lattice of GaAs rods (epsilon = 11.56) in air, timing both calculations.
**Difficulty:** Beginner
**Source:** `python/examples/mpb_sq_rods.py`
**Test Status:** PASS (83.1s)

#### Theory

The square lattice of circular rods is the canonical two-dimensional photonic crystal. Its first Brillouin zone is a square, and the irreducible Brillouin zone is the triangle Gamma-X-M-Gamma. For TM polarization (E field parallel to rod axes), the electric field concentrates inside the high-epsilon rods at low frequencies. The strong dielectric contrast between the rods (epsilon = 11.56, corresponding to GaAs near the bandgap) and the air background opens a TM band gap between bands 1 and 2.

The polarization degeneracy argument explains why TM gaps are easier to open in connected rod arrays while TE gaps prefer connected slab-like geometries. For TM modes, the field energy concentrates inside the isolated rods; for TE modes, the field energy spreads through connected regions, requiring the "air-cylinder in dielectric" geometry for an effective gap.

The normalized frequency omega*a/2*pi*c (also written a/lambda) is the dimensionless figure of merit for photonic crystals. All MPB outputs use this convention, so the results are scale-invariant: a gap at frequency 0.3 means that light with wavelength lambda = a/0.3 falls in the gap.

GaAs (epsilon = 11.56, n = 3.4) is a common semiconductor for photonic crystal fabrication in the telecom and mid-IR wavelength ranges. Its high refractive index provides sufficient dielectric contrast to open wide band gaps with modest rod filling fractions.

The `display_eigensolver_stats()` call at the end reports the number of conjugate-gradient iterations, FLOP counts, and solver convergence metrics, which are useful for benchmarking resolution and mesh-size tradeoffs.

#### Code Walkthrough

Material and geometry setup:

```python
r = 0.2           # rod radius in units of the lattice constant
eps = 11.56       # GaAs dielectric constant
GaAs = mp.Medium(epsilon=eps)
geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))   # unit square
geometry = [mp.Cylinder(r, material=GaAs)]
```

The lattice `size` is always given in units of the primitive lattice vectors. For a square lattice with `size=mp.Vector3(1, 1)`, one cylinder of radius 0.2 occupies a filling fraction of pi * 0.2^2 / 1 = 12.6% of the unit cell.

k-point path through the irreducible Brillouin zone:

```python
Gamma = mp.Vector3()
X = mp.Vector3(0.5, 0)
M = mp.Vector3(0.5, 0.5)
k_points = mp.interpolate(k_interp, [Gamma, X, M, Gamma])
```

In MPB, k-vectors are given in units of the reciprocal lattice vectors. For a square lattice with a=1, X=(0.5,0) corresponds to k=(pi/a, 0) and M=(0.5,0.5) to k=(pi/a, pi/a).

Solving and reporting timing:

```python
t0 = time.time()
ms.run_te()
ms.run_tm()
print(f"total time: {time.time() - t0:.2f} seconds")
ms.display_eigensolver_stats()
```

Running both polarizations separately is essential for identifying whether a gap is polarization-specific or complete (overlapping in both TE and TM).

#### Key Takeaways

- The square lattice of rods has a TM band gap (no TE gap) for epsilon = 11.56 and r = 0.2a because the TM field concentrates efficiently in the isolated rods.
- Normalized frequency a/lambda makes band structure results wavelength-independent; scale the lattice constant to target any operating wavelength.
- `display_eigensolver_stats()` provides solver diagnostics useful for resolution convergence studies.
- Separating `run_te()` and `run_tm()` is required to identify which gap type exists; running both is the standard workflow for 2D photonic crystals.
- With `resolution=32`, the 83-second runtime reflects the cost of solving an 8-band eigenvalue problem at 16 k-points for two polarizations.

---

### 3. `mpb_tri_rods.py` — Triangular Lattice of Dielectric Rods: TM Band Structure and Field Visualization

**Physics:** Computes TM and TE band structures for a triangular lattice of dielectric rods, outputting the Ez field at the K point for data analysis.
**Difficulty:** Beginner
**Source:** `python/examples/mpb_tri_rods.py`
**Test Status:** PASS (99.7s)

#### Theory

The triangular (hexagonal) lattice has a smaller irreducible Brillouin zone than the square lattice because of its higher rotational symmetry (6-fold vs 4-fold). Its first Brillouin zone is a regular hexagon, and the irreducible zone is the triangle Gamma-M-K-Gamma. The higher symmetry generally produces wider band gaps because it reduces the anisotropy of the photonic dispersion at zone boundaries.

For a triangular lattice the basis vectors are not orthogonal. In MPB they are defined as:

    basis1 = (sqrt(3)/2,  1/2)
    basis2 = (sqrt(3)/2, -1/2)

These are the primitive lattice vectors for a triangular lattice with lattice constant a=1 (rod-to-rod distance = 1). The k-space high-symmetry points in lattice coordinates are Gamma=(0,0), M=(0,0.5), and K=(-1/3,1/3).

The triangular lattice of rods has a TM band gap between bands 1 and 2 for epsilon=12 and r=0.2. Unlike the square lattice, the triangular lattice does not have a TE band gap for this geometry; a triangular lattice of air holes in dielectric is needed for a TE gap. This script is the reference structure for the data analysis tutorial in section 12, which uses the field data generated here.

The `fix_efield_phase` call at the K point normalizes the complex phase of the field, making the visualization reproducible (without this, the eigensolver can return any global phase). Field output at a single k-point avoids storing data for the entire dispersion.

#### Code Walkthrough

Defining the triangular lattice:

```python
geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1),
    basis1=mp.Vector3(math.sqrt(3) / 2,  0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)
```

The k-point path in lattice (reciprocal) coordinates:

```python
k_points = [
    mp.Vector3(),            # Gamma
    mp.Vector3(y=0.5),       # M
    mp.Vector3(1/-3, 1/3),   # K
    mp.Vector3(),            # Gamma
]
```

Running with field output at K only:

```python
ms.run_tm(
    mpb.output_at_kpoint(
        mp.Vector3(1/-3, 1/3),
        mpb.fix_efield_phase,
        mpb.output_efield_z
    )
)
```

`output_at_kpoint` is a band function that runs the provided callbacks only when the current k-point matches the specified one. `fix_efield_phase` then `output_efield_z` are called in sequence: the phase is fixed first, then the field is written to an HDF5 file.

#### Key Takeaways

- The triangular lattice has 6-fold rotational symmetry and generally wider band gaps than the square lattice for the same dielectric contrast.
- Basis vectors must be explicitly specified for non-square lattices; k-points are given in the basis of the reciprocal lattice vectors.
- `mpb.output_at_kpoint(k, func1, func2, ...)` chains multiple band functions, applied only at the specified k-point — an efficient way to save field data at a single high-symmetry point.
- `mpb.fix_efield_phase` normalizes the arbitrary global phase of the eigenmode field for reproducible visualization.
- This file is the source for the data analysis workflow in `mpb_data_analysis.py`, where its Ez fields are post-processed with `MPBData`.

---

### 4. `mpb_tri_holes.py` — Triangular Lattice of Air Holes: Complete Photonic Band Gap

**Physics:** Computes TE and TM band structures for a triangular lattice of large air holes (r = 0.45a) in a high-epsilon dielectric background, demonstrating a complete photonic band gap for both polarizations simultaneously.
**Difficulty:** Beginner
**Source:** `python/examples/mpb_tri_holes.py`
**Test Status:** PASS (72.2s)

#### Theory

Inverting the triangular rod structure — replacing dielectric rods in air with air holes in dielectric — dramatically changes which polarization develops a band gap. The TE modes (in-plane electric field) now see a connected high-epsilon background, which acts like a dielectric waveguide network. This connectivity is precisely what TE modes need to open a gap: the field energy concentrates in the continuous dielectric regions, and a sufficiently large hole radius creates the dielectric contrast needed for a gap between bands 2 and 3.

Simultaneously, the TM modes also develop a gap because the large hole radius (r = 0.45a) substantially reduces the filling fraction of dielectric (only 1 - pi*0.45^2/A_cell remains), pushing TM bands apart by dielectric contrast. The remarkable result — first rigorously demonstrated by Joannopoulos et al. in the book "Photonic Crystals" — is that at r = 0.45a and epsilon = 12, the TE and TM gaps overlap in frequency, creating a complete photonic band gap.

A complete band gap means that no electromagnetic wave of any polarization, propagating in any in-plane direction, can exist in the bulk crystal at gap frequencies. This is the prerequisite for three-dimensional optical confinement in 2D photonic crystal slabs (as treated in `mpb_hole_slab.py`).

The script also handles out-of-plane propagation (kz != 0), where the modes are no longer purely TE or TM. In that case, `ms.run()` is used instead of separate polarized runs. For kz != 0, the "light cone" — the region above omega = c*kz/sqrt(eps) where radiation modes exist — becomes relevant for slab applications.

#### Code Walkthrough

Geometry: air holes in dielectric background:

```python
eps = 12
r = 0.45              # large hole radius for complete gap
default_material = mp.Medium(epsilon=eps)   # dielectric background
geometry = [mp.Cylinder(r, material=mp.air)]  # air hole
```

Handling in-plane vs out-of-plane:

```python
kz = 0   # set non-zero for off-axis propagation

k_points = [
    mp.Vector3(z=kz),         # Gamma
    mp.Vector3(0, 0.5, kz),   # M
    mp.Vector3(1/-3, 1/3, kz), # K
    mp.Vector3(z=kz),         # Gamma
]

if kz == 0:
    ms.run_te()
    ms.run_tm()
else:
    ms.run()  # no pure TE/TM for kz != 0
```

The default_material is set on the `ModeSolver` instance; everything not explicitly placed in `geometry` takes this material. By placing only one air cylinder per unit cell, the rest of the cell fills with the default dielectric.

#### Key Takeaways

- Inverting the geometry (holes instead of rods) switches which polarization develops the primary band gap: holes favor TE gaps, rods favor TM gaps.
- At r = 0.45a and epsilon = 12, the triangular hole lattice achieves a complete band gap (both TE and TM gaps overlap) — the workhorse geometry for photonic crystal slabs.
- `default_material` fills all space not occupied by explicit geometry objects, making it straightforward to define a dielectric background with air holes.
- Out-of-plane propagation (kz != 0) breaks the TE/TM symmetry; use `ms.run()` and inspect the band structure for the light cone boundary.
- The complete gap enables 2D confinement in slab geometries without requiring a full 3D photonic crystal.

---

### 5. `mpb_honey_rods.py` — Honeycomb Lattice of Dielectric Rods: Complete Band Gap via Multi-Atom Unit Cell

**Physics:** Computes TM and TE band structures for a honeycomb lattice of small dielectric rods (r = 0.14a, epsilon = 12), demonstrating a complete overlapping TE/TM band gap achievable with a two-atom basis.
**Difficulty:** Intermediate
**Source:** `python/examples/mpb_honey_rods.py`
**Test Status:** PASS (103.6s)

#### Theory

The honeycomb lattice is the 2D photonic crystal analog of graphene. It consists of a triangular Bravais lattice with a two-rod basis: one rod at position (1/6, 1/6) and one at (-1/6, -1/6) in lattice coordinates, arranged with inversion symmetry. This two-point basis doubles the number of bands per unit cell compared to the simple triangular lattice.

The honeycomb geometry achieves something the triangular rod lattice cannot: a complete photonic band gap that exists for both TE and TM polarizations. For small rods (r = 0.14a), the structure is mostly air, and the bands near the gap resemble the Dirac cone dispersion of graphene (linear touching at the K point for the lower bands). As the dielectric contrast is increased or the rod radius tuned, this Dirac point opens into a gap.

The comment in the source code notes that the true complete gap is between bands 12 and 13. The apparent gap between bands 2 and 3 is a false gap that disappears as k-point sampling is refined — a warning about aliasing in coarsely sampled band structures. False gaps arise from incomplete Brillouin zone coverage; increasing `k_interp` resolves them.

The Brillouin zone of the honeycomb lattice uses the same triangular lattice k-points (Gamma-M-K-Gamma), because the Bravais lattice is triangular. The two-atom basis contributes to the structure factor, modifying the band structure while preserving the Brillouin zone shape.

Inversion symmetry (the two rods are related by r -> -r) is important for MPB performance. The solver can exploit time-reversal and inversion symmetry to reduce the computational domain and guarantee real-valued eigenvectors.

#### Code Walkthrough

Defining the two-atom honeycomb basis:

```python
r = 0.14
eps = 12
geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1),
    basis1=mp.Vector3(math.sqrt(3)/2,  0.5),
    basis2=mp.Vector3(math.sqrt(3)/2, -0.5),
)

geometry = [
    mp.Cylinder(r, center=mp.Vector3( 1/6,  1/6), height=mp.inf,
                material=mp.Medium(epsilon=eps)),
    mp.Cylinder(r, center=mp.Vector3(-1/6, -1/6), height=mp.inf,
                material=mp.Medium(epsilon=eps)),
]
```

The two rods are placed at fractional lattice coordinates that put them at the correct positions for a honeycomb lattice: each rod sits at a vertex of a regular hexagon with the unit cell.

Running both polarizations in sequence:

```python
ms.run_tm()
ms.run_te()
# Note: ms.run() would work too, and show the complete gap between bands 12 and 13
```

#### Key Takeaways

- The honeycomb lattice requires a two-atom basis within a triangular Bravais lattice unit cell; this doubles the number of bands and enables new gap topologies.
- False gaps between bands 2 and 3 disappear with finer k-point sampling; always increase `k_interp` to verify gap reality.
- The true complete gap (both polarizations) lies between bands 12 and 13 for r = 0.14a and epsilon = 12.
- Inversion symmetry of the two-atom basis (r -> -r maps one atom to the other) allows MPB to use real arithmetic, improving performance.
- Photonic analogs of graphene physics (Dirac cones, topological effects) are accessible through honeycomb lattice calculations.

---

### 6. `mpb_hole_slab.py` — Photonic Crystal Slab: Guided Modes Above a Substrate

**Physics:** Computes even and odd guided modes of a finite-thickness triangular-lattice hole slab on an optional substrate, capturing the photonic band gap projected onto the 2D Brillouin zone.
**Difficulty:** Advanced
**Source:** `python/examples/mpb_hole_slab.py`
**Test Status:** TIMEOUT (CPU-intensive: 3D calculation with large supercell and anisotropic resolution)

#### Theory

Realistic photonic crystal devices are fabricated as thin dielectric slabs, not infinite 2D structures. Vertical confinement comes from total internal reflection (as in a planar waveguide), while lateral confinement and band gaps arise from the in-plane periodic hole pattern. The combined structure is called a photonic crystal slab.

To model a slab with MPB, a supercell technique is used: the vertical direction (z) is treated as a third periodic dimension, but with a supercell height large enough that the modes of adjacent slab replicas do not couple. Here, `supercell_h = 4` means the computational cell is 4 lattice constants tall, with the slab occupying only h = 0.5 of the central region.

The fundamental analysis of this structure appears in Johnson, Fan, Villeneuve, Joannopoulos, and Kolodziejski, PRB 60, 5751 (1999), referenced in the source file. That paper established the concept of projected band gaps for slab photonic crystals and the notion of guided modes below the light cone.

A mode in the slab is guided if its frequency falls below the light line, omega < c*|k_parallel|/n_substrate, where n_substrate is the refractive index of the substrate. Guided modes decay exponentially in the vertical direction and are truly confined. Modes above the light line couple to radiation and become leaky (resonances with finite Q).

Because the slab has mirror symmetry about its midplane (z=0) when loweps = 1.0 (no substrate or symmetric cladding), the modes split into even (zeven, Hz-like) and odd (zodd, Ez-like) parities. These are the slab analogs of TE and TM. `run_zeven` and `run_zodd` enforce these parities and compute only the relevant modes. When a substrate is present (loweps != 1.0), the mirror symmetry is broken and all modes must be computed together.

The anisotropic resolution `mp.Vector3(32, 32, 16)` uses finer sampling in the in-plane directions (where the hole features require resolution) and coarser sampling in z (where the fields vary more slowly).

#### Code Walkthrough

The 3D supercell geometry — slab plus optional substrate plus holes:

```python
geometry = [
    mp.Block(material=mp.Medium(epsilon=loweps),
             center=mp.Vector3(z=0.25*supercell_h),
             size=mp.Vector3(mp.inf, mp.inf, 0.5*supercell_h)),  # upper half-space
    mp.Block(material=mp.Medium(epsilon=eps),
             size=mp.Vector3(mp.inf, mp.inf, h)),                 # slab
    mp.Cylinder(r, material=mp.air, height=supercell_h),          # holes through slab
]
```

The Block filling the upper half-space sets the substrate or cladding. The slab Block (centered at z=0) is placed on top. The Cylinder punches holes through the entire supercell height, creating air columns that penetrate through and above the slab.

Symmetry-resolved runs with field output at K:

```python
if loweps == 1.0:
    ms.run_zeven(mpb.output_at_kpoint(K, mpb.output_hfield_z))
    ms.run_zodd(mpb.output_at_kpoint(K, mpb.output_dfield_z))
else:
    ms.run(mpb.output_at_kpoint(K, mpb.output_hfield_z),
           mpb.display_zparities)
```

`display_zparities` prints the parity of each mode at each k-point, useful for mode classification even when symmetry is broken.

#### Key Takeaways

- Slab photonic crystals use a 3D supercell calculation with periodic boundary conditions in z; the supercell must be large enough to prevent interaction between slab images.
- Guided modes lie below the light line (omega < c*k_parallel/n_substrate); only these modes are truly confined in the slab.
- Even/odd symmetry with respect to the slab midplane (`run_zeven`/`run_zodd`) replaces TE/TM for slab calculations when the structure is vertically symmetric.
- Anisotropic resolution (`mp.Vector3(nx, ny, nz)`) allows different grid spacings in different directions, essential for slabs where vertical and lateral length scales differ.
- The supercell height must be large enough (typically 4-6 times the slab thickness) that evanescent fields decay to negligible amplitude at the supercell boundary.

---

### 7. `mpb_diamond.py` — Diamond Lattice of Dielectric Spheres: 3D Photonic Band Gap

**Physics:** Computes the 3D photonic band structure for a face-centered cubic (FCC) diamond lattice of dielectric spheres, the first structure demonstrated to have a complete 3D photonic band gap.
**Difficulty:** Advanced
**Source:** `python/examples/mpb_diamond.py`
**Test Status:** PASS (61.0s)

#### Theory

A complete three-dimensional photonic band gap requires a periodic structure that blocks light propagation in all directions and all polarizations simultaneously. The diamond lattice of dielectric spheres was one of the first structures predicted to achieve this, by Ho, Chan, and Soukoulis (1990). The diamond structure has face-centered cubic (FCC) Bravais lattice symmetry with a two-sphere basis, giving it tetrahedral coordination with near-spherical Brillouin zone — the key to isotropy.

The FCC lattice primitive vectors and their reciprocal lattice define the body-centered cubic (BCC) first Brillouin zone, whose irreducible portion has high-symmetry points X, U, L, Gamma, W, and K. The code traces the k-path X-U-L-Gamma-X-W-K, which is the canonical path for FCC band structures (matching the NIST AFLOW convention).

The FCC primitive lattice vectors are:

    a1 = (a/2)(0, 1, 1)
    a2 = (a/2)(1, 0, 1)
    a3 = (a/2)(1, 1, 0)

In MPB, `basis_size = sqrt(1/2)` sets the length of each primitive vector (= a/sqrt(2) for a conventional FCC cube of side a=1), and the `basis1`, `basis2`, `basis3` vectors give the directions.

The diamond structure has two spheres per FCC unit cell, positioned at (1/8, 1/8, 1/8) and (-1/8, -1/8, -1/8) in lattice coordinates (the 1/4-body-diagonal displacement). The key dimensionless ratio controlling the gap is the sphere radius in units of a/4 — values near 0.25 (as used here, r=0.25 with basis_size=sqrt(1/2)) are near optimal.

For 3D calculations, resolution=16 means a 16x16x16 grid per unit cell; the actual number of grid points scales as (resolution)^3 times the number of unit cells. The mesh_size=5 parameter controls the number of sub-grid points for averaging the dielectric function over each pixel, critical for accurate representation of curved sphere surfaces.

The calculation outputs the electric energy density (dpwr = |D|^2/epsilon) at the U point (k = (0, 0.625, 0.375)). The U point is where the band gap edges are typically located in diamond FCC structures.

#### Code Walkthrough

Defining the FCC diamond lattice:

```python
sqrt_half = math.sqrt(0.5)
geometry_lattice = mp.Lattice(
    basis_size=mp.Vector3(sqrt_half, sqrt_half, sqrt_half),
    basis1=mp.Vector3(0, 1, 1),
    basis2=mp.Vector3(1, 0, 1),
    basis3=mp.Vector3(1, 1, 0),
)
```

Placing the two-sphere diamond basis:

```python
geometry = [
    mp.Sphere(r, center=mp.Vector3( 0.125,  0.125,  0.125), material=diel),
    mp.Sphere(r, center=mp.Vector3(-0.125, -0.125, -0.125), material=diel),
]
```

The canonical FCC Brillouin zone path:

```python
vlist = [
    mp.Vector3(0, 0.5, 0.5),       # X
    mp.Vector3(0, 0.625, 0.375),   # U
    mp.Vector3(0, 0.5, 0),         # L
    mp.Vector3(0, 0, 0),           # Gamma
    mp.Vector3(0, 0.5, 0.5),       # X
    mp.Vector3(0.25, 0.75, 0.5),   # W
    mp.Vector3(0.375, 0.75, 0.375),# K
]
```

Running and collecting energy density at U:

```python
ms.run(mpb.output_at_kpoint(mp.Vector3(0, 0.625, 0.375), mpb.output_dpwr))
```

#### Key Takeaways

- Three-dimensional photonic band gaps require a fully 3D calculation with 3D primitive lattice vectors; the FCC diamond lattice is the canonical example.
- `basis_size` sets the lengths of the primitive vectors (each being sqrt(1/2) * a for the FCC lattice with conventional cube side a=1).
- The two-sphere diamond basis at positions +-( 1/8, 1/8, 1/8) is essential; an FCC lattice with only one sphere would give different (BCC) symmetry.
- `mesh_size=5` improves accuracy for curved surfaces by averaging the dielectric over sub-pixel regions; larger values are more accurate but slower.
- The U point is the critical k-point for the diamond lattice gap edges; `output_at_kpoint` limits field output to this point, avoiding large data volumes across all k-points.

---

### 8. `mpb_bragg.py` — Quarter-Wave Bragg Mirror: 1D Band Gap at the X Point

**Physics:** Computes the band structure at the Brillouin zone edge (kx = 0.5) for a 1D quarter-wave stack, the simplest possible photonic crystal with a well-known analytical gap formula.
**Difficulty:** Beginner
**Source:** `python/examples/mpb_bragg.py`
**Test Status:** PASS (5.4s)

#### Theory

A 1D photonic crystal — the Bragg mirror or dielectric stack — is the oldest and most analytically tractable photonic band gap structure. It consists of alternating layers of high and low refractive indices (n_hi = 3.0, n_lo = 1.0 here) with thicknesses chosen to satisfy the quarter-wave condition:

    n_hi * d_hi = n_lo * d_lo = lambda/4

where lambda is the design wavelength and d_hi, d_lo are the layer thicknesses. For a quarter-wave stack, the reflectances from each interface add constructively in reflection, and the stop band (band gap) is maximized.

For a unit cell of total length a=1, the quarter-wave condition gives d_hi = n_lo/(n_hi + n_lo) = 1/4 and d_lo = n_hi/(n_hi + n_lo) = 3/4. In the code, `w_hi = n_lo/(n_hi + n_lo)` computes this ratio.

The analytical gap-to-midgap ratio for a 1D quarter-wave stack is:

    delta_omega / omega_0 = (4/pi) * arcsin((n_hi - n_lo)/(n_hi + n_lo))

For n_hi = 3 and n_lo = 1, this gives approximately 40%. The MPB calculation at the X point (kx = 0.5) finds the band edges on either side of the gap, validating this analytical result.

Because the 1D problem has infinite translational symmetry in the y and z directions, TM and TE polarizations are degenerate. The code runs `run_tm()` and notes this explicitly. The fast run time (5.4s) reflects the small 1D calculation with only a single k-point.

The geometry uses an infinite cylinder (`radius=mp.inf`) to fill an infinite slab of thickness `w_hi` within the unit cell. This is MPB's idiomatic way to create 1D layers — a cylinder of infinite radius in 1D is just a slab.

#### Code Walkthrough

1D cell and quarter-wave geometry:

```python
n_lo, n_hi = 1.0, 3.0
w_hi = n_lo / (n_hi + n_lo)   # quarter-wave thickness of high-index layer

geometry_lattice = mp.Lattice(size=mp.Vector3(1))   # 1D cell along x

geometry = mp.Cylinder(
    material=mp.Medium(index=n_hi),
    axis=mp.Vector3(1),          # cylinder axis along x
    radius=mp.inf,               # infinite radius = slab
    height=w_hi,                 # thickness of slab
)
default_material = mp.Medium(index=n_lo)
```

Solving at the zone-edge X point only:

```python
kx = 0.5
k_points = [mp.Vector3(kx)]
ms.run_tm(mpb.output_hfield_y)   # TM=TE due to 1D degeneracy
```

#### Key Takeaways

- A 1D photonic crystal (Bragg mirror) is specified with `geometry_lattice = mp.Lattice(size=mp.Vector3(1))` and `mp.Cylinder(radius=mp.inf)` as slab layers.
- The quarter-wave condition w_hi = n_lo/(n_hi + n_lo) maximizes the stop-band width for a two-layer unit cell.
- In 1D, TE and TM are degenerate — only one polarization calculation is needed.
- Restricting k-points to just the zone edge (kx = 0.5) is sufficient to locate the band gap edges for a Bragg mirror.
- The 5.4s runtime demonstrates the efficiency advantage of 1D calculations; the same physics takes much longer in 2D or 3D.

---

### 9. `mpb_bragg_sine.py` — Sinusoidally-Varying 1D Bragg Mirror: Custom Material Functions

**Physics:** Computes the band structure of a 1D photonic crystal with a sinusoidally-varying dielectric index, demonstrating MPB's support for custom material-function callables.
**Difficulty:** Beginner
**Source:** `python/examples/mpb_bragg_sine.py`
**Test Status:** PASS (5.6s)

#### Theory

Not all photonic crystal designs use step-discontinuity index profiles. Graded-index structures with smooth, continuously-varying dielectric profiles can exhibit photonic band gaps while avoiding the scattering losses from abrupt interfaces. The sinusoidal profile is the simplest smooth periodic function and has an exact analytical treatment via Mathieu's equation.

For a dielectric function that varies as:

    epsilon(x) = (n_min + 0.5*(n_max - n_min)*(1 + cos(2*pi*x)))^2

the Mathieu equation governs the band structure. Near the zone boundary (kx = pi/a), a gap opens even for small dielectric modulation, with a gap size proportional to the Fourier amplitude of the dielectric function. Unlike the step-index Bragg mirror, the sinusoidal profile has only a single Fourier component, so higher-order gaps are exponentially smaller.

The code computes the dispersion across the entire Brillouin zone (from Gamma to X) using 10 k-points (`mp.interpolate(9, [Gamma, X])`), in contrast to the step-index example which used only the X point. This shows the full dispersion relation and makes the gap at kx = 0.5 visible in context.

MPB supports arbitrary position-dependent dielectric functions through Python callables. The function `eps_func(p)` receives a position vector `p` in lattice coordinates and must return a `mp.Medium` object. Setting `default_material = eps_func` makes this function fill all space. This feature allows modeling any smooth spatially-varying dielectric without discretizing the profile by hand.

#### Code Walkthrough

Defining the sinusoidal material function:

```python
index_min, index_max = 1, 3

def eps_func(p):
    return mp.Medium(
        index=index_min + 0.5*(index_max - index_min)*(1 + math.cos(2*math.pi*p.x))
    )
```

Setting it as the default material (fills all space):

```python
geometry_lattice = mp.Lattice(size=mp.Vector3(1))    # 1D
default_material = eps_func                          # no geometry needed
k_points = mp.interpolate(9, [mp.Vector3(), mp.Vector3(x=0.5)])
ms.run_tm()
```

No explicit geometry objects are needed; the callable fills the entire unit cell with the spatially-varying index profile.

#### Key Takeaways

- MPB accepts Python callables as `default_material`, enabling arbitrary spatially-varying dielectric profiles without geometric discretization.
- The callable receives position `p` in lattice coordinates and must return `mp.Medium`; the `p.x`, `p.y`, `p.z` components are in [0,1] within the unit cell.
- A sinusoidal dielectric variation produces band gaps at zone boundaries analogous to a Bragg mirror, but with exponentially smaller higher-order gaps.
- Computing the full Gamma-to-X dispersion (versus the single X-point calculation in `mpb_bragg.py`) reveals how bands flatten near the zone boundary as the gap opens.
- Custom material functions make it straightforward to explore graded-index photonic crystals, chirped mirrors, and aperiodic structures.

---

### 10. `mpb_line_defect.py` — Line-Defect Waveguide in a Triangular Rod Crystal

**Physics:** Computes the guided band of a line-defect photonic crystal waveguide formed by a missing row of rods in a triangular lattice, demonstrating supercell band folding and the extraction of the guided mode within the bulk band gap.
**Difficulty:** Intermediate
**Source:** `python/examples/mpb_line_defect.py`
**Test Status:** FAIL (API change: `ms.fix_efield_phase` should be `mpb.fix_field_phase`)

#### Theory

A photonic crystal waveguide is created by introducing a line defect into a periodic crystal — here, by removing an entire row of dielectric rods along one direction. Within the photonic band gap frequency range, no bulk modes exist in the surrounding crystal, and any mode that exists must be confined to the defect. The defect acts as a channel waveguide with the photonic crystal providing "perfect mirror" sidewalls: there is no leakage into the bulk crystal at gap frequencies, only into the ends of the waveguide.

To compute the waveguide band structure with MPB, a supercell is constructed with `supercell_y = 7` periods in the transverse direction. The supercell is large enough that the guided mode's evanescent tail decays to negligible amplitude before reaching the supercell boundary. The longitudinal direction (along the waveguide) remains periodic with period equal to the original lattice constant.

The supercell has 7 times the unit-cell area in the transverse direction, so the bulk crystal bands are "folded" seven times. This means that with N_bulk bands in the primitive cell, the supercell calculation has 7*N_bulk bands, most of which correspond to bulk crystal states. The guided mode appears as an isolated band that crosses the otherwise-avoided bulk bands within the gap.

Finding the guided mode requires computing enough bands to see above the bulk crystal gap. The code uses `num_bands = supercell_y + extra_bands = 7 + 5 = 12` — enough to see several bulk bands below the gap and a few above. The guided mode in the gap is identified as the band with a significantly different group velocity (dispersion slope) from the bulk bands.

The failing API call `ms.fix_efield_phase` should be the module-level function `mpb.fix_field_phase`. This rename occurred between MPB versions. The correct call pattern is:

```python
ms.run_tm(
    mpb.output_at_kpoint(k_points[len(k_points)//2]),
    mpb.fix_field_phase,    # correct name
    mpb.output_efield_z,
)
```

#### Code Walkthrough

Supercell construction with a missing row:

```python
supercell_y = 7
geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, supercell_y),
    basis1=mp.Vector3(math.sqrt(3)/2,  0.5),
    basis2=mp.Vector3(math.sqrt(3)/2, -0.5),
)

geometry = [mp.Cylinder(r, material=mp.Medium(epsilon=eps))]
geometry = mp.geometric_objects_lattice_duplicates(geometry_lattice, geometry)
geometry += [mp.Cylinder(r, material=mp.air)]   # erase central row
```

The last air cylinder overwrites the central rod, creating the missing-row defect. `geometric_objects_lattice_duplicates` tiles the unit-cell rod over the full 7-period supercell first.

Setting up k-points along the waveguide direction:

```python
Gamma = mp.Vector3()
K_prime = mp.lattice_to_reciprocal(mp.Vector3(0.5), geometry_lattice)
k_points = mp.interpolate(4, [Gamma, K_prime])
```

`mp.lattice_to_reciprocal` converts from lattice to Cartesian reciprocal coordinates, correctly accounting for the non-square triangular basis. `K_prime` is the zone edge in the waveguide direction.

Running TM with field output at the midpoint k-vector:

```python
ms.run_tm(
    mpb.output_at_kpoint(k_points[len(k_points)//2]),
    ms.fix_efield_phase,    # BUG: should be mpb.fix_field_phase
    mpb.output_efield_z,
)
```

#### Key Takeaways

- Line-defect photonic crystal waveguides require a supercell in the transverse direction; larger supercells better isolate the guided mode but fold more bulk bands.
- The number of required bands scales with the supercell size: `num_bands = supercell_y + extra_bands` ensures coverage of the gap region.
- The API function `ms.fix_efield_phase` has been renamed `mpb.fix_field_phase`; this is the source of the FAIL status for this example.
- `mp.lattice_to_reciprocal` is needed for non-square lattices to correctly compute the Brillouin zone edge in the waveguide direction.
- The guided mode appears as a distinctive dispersive band crossing the folded bulk bands within the bulk band gap frequency range.

---

### 11. `mpb_strip.py` — Silicon Strip Waveguide: Dispersion and find_k for Fixed Wavelength

**Physics:** Computes the guided-mode dispersion and group velocity of a silicon-on-insulator strip waveguide (Si on SiO2 substrate), then uses `find_k` to solve the inverse problem: finding beta at a fixed telecommunications wavelength (1.55 um).
**Difficulty:** Intermediate
**Source:** `python/examples/mpb_strip.py`
**Test Status:** PASS (58.6s)

#### Theory

Silicon photonic wire waveguides (strip waveguides) are the foundational component of silicon photonics — the platform for on-chip optical interconnects and sensing at the 1.55 micrometer telecommunications wavelength. A typical strip waveguide consists of a high-index silicon core (n_Si = 3.45) on a silicon dioxide substrate (n_SiO2 = 1.45), with air cladding above.

The waveguide supports guided modes below the light line of the substrate. A mode is guided if its propagation constant beta = |k| satisfies:

    beta > omega * n_substrate / c

Above this threshold (the light line), modes couple to radiation and become leaky. Only modes below the light line in the (omega, k) diagram are truly guided with zero radiation loss.

MPB solves for guided modes naturally because it computes the exact eigenfrequencies at each k-point. In the strip waveguide context, k plays the role of the propagation constant beta (in units of 2*pi/um). The guided modes appear as bands that stay below the light line omega = c*k/n_SiO2.

The `find_k` function inverts the usual calculation: instead of computing omega(k), it finds k(omega) for a specified frequency. This is useful for device design where the operating wavelength is fixed (e.g., lambda = 1.55 um, omega = 1/1.55 in units of c/um). `find_k` uses a bisection algorithm to find the k value that gives the specified frequency for each band.

The cell lattice `size=mp.Vector3(0, sc_y, sc_z)` sets the propagation direction as x (size 0 = non-periodic, just a k-vector direction). The waveguide cross-section is in the yz plane with a 2x2 um supercell large enough that the mode field decays to zero at the boundaries.

Group velocity `v_g = d_omega/d_k` is output by `display_group_velocities`. This is the speed of electromagnetic energy propagation (signal velocity) in the waveguide, which determines the group delay and chromatic dispersion — critical parameters for data transmission.

#### Code Walkthrough

Geometry: Si strip on SiO2 substrate:

```python
w = 0.3    # Si width (um)
h = 0.25   # Si height (um)
Si  = mp.Medium(index=3.45)
SiO2 = mp.Medium(index=1.45)

sc_y, sc_z = 2, 2     # supercell cross-section (um)
geometry_lattice = mp.Lattice(size=mp.Vector3(0, sc_y, sc_z))

geometry = [
    mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5*(sc_z-h)),
             center=mp.Vector3(z=0.25*(sc_z+h)),
             material=SiO2),           # substrate half-space
    mp.Block(size=mp.Vector3(mp.inf, w, h), material=Si),  # strip core
]
```

Forward solve: omega(k) dispersion with parity display:

```python
k_points = mp.interpolate(num_k, [mp.Vector3(k_min), mp.Vector3(k_max)])
ms.run(mpb.display_yparities, mpb.display_zparities)
```

Inverse solve: k(omega) at lambda = 1.55 um:

```python
omega = 1 / 1.55   # frequency in units of c/um
ms.find_k(
    mp.NO_PARITY,       # no parity constraint
    omega,              # target frequency
    1, num_bands,       # band range
    mp.Vector3(1),      # propagation direction
    1e-3,              # fractional tolerance
    omega * 3.45,       # k_guess (Si light line)
    omega * 0.1,        # k_min bound
    omega * 4,          # k_max bound
    mpb.output_poynting_x,
    mpb.display_yparities,
    mpb.display_group_velocities,
)
```

The k_guess near the silicon light line (omega * n_Si) is a good starting point for the bisection since guided modes in a Si waveguide on SiO2 have beta between the SiO2 and Si light lines.

#### Key Takeaways

- Strip waveguide guided modes fall below the substrate light line omega = c*k/n_SiO2; modes above this line are leaky and should be ignored.
- `ms.find_k(parity, omega, band_min, band_max, k_direction, tol, k_guess, k_min, k_max, ...)` solves the inverse dispersion problem for device design at a fixed wavelength.
- `display_group_velocities` outputs d_omega/d_k, the group velocity — essential for dispersion engineering in photonic wire waveguides.
- `display_yparities` and `display_zparities` classify modes by their mirror symmetry, distinguishing TE-like (horizontal E) from TM-like (vertical E) modes.
- Using physical units (micrometers) for the geometry means output frequencies are in um/lambda; the telecom wavelength 1.55 um corresponds to frequency 1/1.55 = 0.6452.

---

### 12. `mpb_data_analysis.py` — Post-Processing and Visualization of MPB Field Data

**Physics:** Post-processes the Ez field data from the triangular rod crystal (from `mpb_tri_rods.py`) and the electric energy density from the diamond lattice (from `mpb_diamond.py`), demonstrating coordinate transformation, field rectification, and matplotlib visualization.
**Difficulty:** Intermediate
**Source:** `python/examples/mpb_data_analysis.py`
**Test Status:** PASS (123.9s)

#### Theory

MPB computes eigenmodes defined on a potentially non-orthogonal lattice with periodic (Bloch-form) boundary conditions. To visualize and analyze these fields in familiar Cartesian coordinates, two transformations are needed:

1. **Rectification**: Converting the non-orthogonal lattice grid to a rectangular Cartesian grid by interpolation.
2. **Periodic unfolding**: Tiling the unit cell over multiple periods to show the spatial extent of the mode.

The `MPBData` class provides both operations. The `rectify=True` flag performs the coordinate transformation, and `periods=3` tiles the result over a 3x3 supercell. The output resolution is set independently from the calculation resolution, allowing high-quality visualization even from a coarser computation.

The complex-valued eigenmodes have an arbitrary global phase. The `fix_efield_phase` function (applied during the run via `output_at_kpoint`) normalizes this phase so that the field is predominantly real, making the visualization meaningful. Without phase fixing, the real part of the field is an arbitrary linear combination of the physical field and its phase-shifted version.

For the triangular rod example, the script collects Ez fields for all 8 bands at the K point by appending them in a list via a custom band function `get_efields`. The MPBData transformation then converts each field from lattice coordinates to a rectangular grid, suitable for `plt.imshow`. The epsilon distribution (from `tr_ms.get_epsilon()`) is overlaid as a contour to show the rod positions.

For the 3D diamond lattice example, the script collects the electric energy density `dpwr = |D|^2/epsilon` at the U point across all 5 bands. Converting these to rectangular coordinates and plotting would show how each band's energy concentrates in different spatial regions relative to the dielectric spheres — a fundamental visualization for understanding photonic band structure physics.

#### Code Walkthrough

Collecting fields during the MPB run using a custom band function:

```python
efields = []

def get_efields(tr_ms, band):
    efields.append(tr_ms.get_efield(band))

tr_ms.run_tm(
    mpb.output_at_kpoint(
        mp.Vector3(1/-3, 1/3),
        mpb.fix_efield_phase,
        get_efields
    )
)
```

Any callable `f(ms, band)` can be passed as a band function; here it captures the field array into a Python list for later processing.

Converting non-rectangular lattice data to a Cartesian grid:

```python
md = mpb.MPBData(rectify=True, resolution=32, periods=3)

converted = []
for f in efields:
    f = f[..., 0, 2]          # extract z-component: shape (..., xyz)[..., z]
    converted.append(md.convert(f))
```

The indexing `f[..., 0, 2]` selects the first k-point index (index 0, since fields were collected at a single k-point) and the z-component (index 2) of the vector field.

Visualizing the epsilon distribution:

```python
eps = tr_ms.get_epsilon()
plt.imshow(eps.T, interpolation='spline36', cmap='binary')

md = mpb.MPBData(rectify=True, resolution=32, periods=3)
rectangular_data = md.convert(eps)
plt.imshow(rectangular_data.T, ...)
```

Overlaying fields with the dielectric structure:

```python
for i, f in enumerate(converted):
    plt.subplot(331 + i)
    plt.contour(rectangular_data.T, cmap='binary')    # epsilon contour
    plt.imshow(np.real(f).T, cmap='RdBu', alpha=0.9)  # Ez field
    plt.axis('off')
```

#### Key Takeaways

- `mpb.MPBData(rectify=True, periods=N, resolution=R)` converts MPB field data from a possibly non-orthogonal lattice coordinate system to a rectangular Cartesian grid tiled N times with resolution R.
- Custom band functions `f(ms, band)` can collect field data during a run by appending to an external list, enabling batch post-processing after all k-points are computed.
- The field array shape from `get_efield(band)` is `(nx, ny, nz, nk, 3)` where the last axis indexes x, y, z components; `f[..., 0, 2]` extracts the z-component at the first k-point.
- Phase normalization (`fix_efield_phase`) must be applied before collecting fields for visualization to ensure reproducible results.
- The `get_epsilon()` method retrieves the dielectric function on the MPB grid, useful for overlaying on field plots to show the relationship between mode profiles and the dielectric structure.

---

### 13. `holey-wvg-bands.py` — Photonic Crystal Waveguide Bands via Meep k-point Sweep

**Physics:** Computes the photonic band structure of a holey waveguide (dielectric waveguide with periodic air holes) using Meep's time-domain `run_k_points` method, extracting band frequencies via Harminv harmonic inversion.
**Difficulty:** Intermediate
**Source:** `python/examples/holey-wvg-bands.py`
**Test Status:** TIMEOUT (CPU-intensive: 20 k-points, each requiring a full FDTD run with Harminv)

#### Theory

This example demonstrates an alternative approach to computing photonic band structure using Meep's FDTD engine rather than MPB's eigenvalue solver. The two approaches are complementary: MPB is more efficient for clean periodic systems, while Meep's `run_k_points` is useful when the structure involves dissipative materials, nonlinear effects, or when the user wants to compute the band structure of a system that is already set up for FDTD transmission/reflection analysis.

The structure is a 2D photonic crystal waveguide: an infinite dielectric waveguide (eps = 13, width w = 1.2) with a periodic array of air holes (radius r = 0.36) punched through it. The periodicity is 1 (in dimensionless units), and the 2D cell is one period long in x with absorbing boundaries (PML) in y to simulate an infinite transverse extent.

Meep's `run_k_points` method computes band structure by:
1. Setting the Bloch k-point.
2. Exciting the structure with a broadband Gaussian pulse.
3. Running the simulation and applying Harminv (harmonic inversion) to the time-domain signal.
4. Extracting the resonant frequencies from the Harminv output.

Harminv decomposes the recorded time-domain field into a sum of decaying sinusoids, identifying each as a frequency + Q factor. For a perfectly periodic structure, the Q values should be infinite (no decay); finite Q indicates coupling to radiation or numerical dissipation.

The y-mirror symmetry of the waveguide is exploited: `mp.Mirror(direction=mp.Y, phase=-1)` halves the computation by enforcing the anti-symmetric (Hz-like) modes. The phase=-1 selects the mode with Hz(-y) = -Hz(y), which corresponds to the guided mode polarization of interest.

The `kx=False` branch (commented out) shows an alternative: running at a specific kx and using Harminv to find the frequency, then explicitly outputting the field for one period. The main `kx=False` branch sweeps 20 k-points automatically.

#### Code Walkthrough

Setting up the holey waveguide unit cell:

```python
eps = 13; w = 1.2; r = 0.36
cell = mp.Vector3(1, sy)              # 1 period in x, sy in y
b = mp.Block(size=mp.Vector3(mp.inf, w, mp.inf),
             material=mp.Medium(epsilon=eps))
c = mp.Cylinder(radius=r)             # air hole
```

Broadband pulse source at an off-axis position:

```python
s = mp.Source(
    src=mp.GaussianSource(fcen=0.25, fwidth=1.5),
    component=mp.Hz,
    center=mp.Vector3(0.1234),       # slightly off-center for symmetry breaking
)
```

The off-center source position (x = 0.1234, not 0.0) ensures coupling to modes that have odd symmetry in x, which would be zero at the center. The broadband `fwidth=1.5` excites a wide range of frequencies in a single run.

Running the k-point sweep:

```python
sim.run_k_points(300,
    mp.interpolate(k_interp, [mp.Vector3(), mp.Vector3(0.5)]))
```

The first argument (300) is the maximum run time after sources. The k-points sweep from Gamma (0,0) to X (0.5, 0) — the Brillouin zone boundary for a 1D periodic structure.

#### Key Takeaways

- `sim.run_k_points(T, k_list)` automates the Meep band structure workflow: it loops over k-points, sets the Bloch phase, runs the simulation with Harminv, and collects frequencies.
- Meep's FDTD band structure is complementary to MPB: slower per k-point but supports dispersive materials, nonlinear effects, and direct integration with transmission/flux calculations.
- The Harminv monitor must be placed at a location where the modes of interest have non-zero amplitude; a slightly off-center position excites both even and odd modes.
- Mirror symmetry (`mp.Mirror`) halves the computation and filters modes by polarization; `phase=-1` selects anti-symmetric Hz modes.
- The `run_k_points` TIMEOUT for this test reflects the cost of 20 independent FDTD runs plus Harminv analysis, each requiring the field to decay significantly.

---

### 14. `holey-wvg-cavity.py` — Photonic Crystal Cavity: Resonant Modes and Transmission Spectrum

**Physics:** Computes either the resonant mode frequencies (via Harminv) or the transmission spectrum through a photonic crystal cavity formed by a local defect in a holey waveguide, demonstrating both modes of Meep analysis for the same structure.
**Difficulty:** Intermediate
**Source:** `python/examples/holey-wvg-cavity.py`
**Test Status:** PASS (35.8s)

#### Theory

This example creates a photonic crystal cavity by introducing a structural defect into the periodic holey waveguide from the previous section. The defect is a larger spacing `d = 1.4` between the central pair of holes (compared to the regular spacing of 1). This local perturbation breaks the translational symmetry and creates a localized mode within the photonic band gap.

The physical mechanism is the same as for the point defect in MPB (section 1): the local modification shifts a mode from the bulk band edge into the gap, where it becomes exponentially localized. However, in a 1D photonic crystal waveguide, the localization is along the waveguide direction (x), and the mode decays exponentially away from the defect center on both sides into the periodic mirror sections.

The cavity Q factor measures how long the mode stores energy relative to a field oscillation period. For N holes on each side, the Q factor scales as Q ~ exp(2*N*kappa), where kappa is the field decay constant per unit cell in the bulk crystal at the resonance frequency. More holes = higher Q = narrower linewidth = longer photon storage time.

The script supports two operating modes via the `--resonant_modes` flag:

**Resonant mode calculation**: A point source at the cavity center (x=0) excites the cavity, Harminv extracts the resonant frequency and Q, and the field is then output for visualization. Dual mirror symmetry (`mp.Mirror(mp.Y, phase=-1)` and `mp.Mirror(mp.X, phase=-1)`) halves the computation twice by requiring even x-symmetry and odd y-symmetry.

**Transmission spectrum**: An extended source on the left side launches a waveguide mode, and a flux plane on the right side measures the transmitted power spectrum. The transmission spectrum shows a sharp peak at the cavity resonant frequency (where the cavity temporarily stores the light before re-emitting it forward) and a stop band (the photonic band gap) where transmission is zero.

The `stop_when_fields_decayed` termination criterion automatically ends the simulation once the fields have decreased by 1e-3 relative to their peak, avoiding unnecessary computation after the transient has passed.

#### Code Walkthrough

Structure: N holes on each side of a widened central spacing:

```python
d = 1.4   # defect spacing (regular = 1)
N = args.N  # number of holes per side (default=3)
sx = 2*(pad + dpml + N) + d - 1

for i in range(N):
    geometry.append(mp.Cylinder(r, center=mp.Vector3( d/2 + i)))
    geometry.append(mp.Cylinder(r, center=mp.Vector3(-(d/2 + i))))
```

The holes are placed symmetrically: the innermost holes are at +-d/2 from center, with subsequent holes 1 unit apart.

Resonant mode calculation with dual symmetry:

```python
sim.sources.append(mp.Source(mp.GaussianSource(fcen, fwidth=df),
                             component=mp.Hz,
                             center=mp.Vector3()))    # at cavity center
sim.symmetries = [mp.Mirror(mp.Y, phase=-1),
                  mp.Mirror(mp.X, phase=-1)]

sim.run(mp.at_beginning(mp.output_epsilon),
        mp.after_sources(mp.Harminv(mp.Hz, mp.Vector3(), fcen, df)),
        until_after_sources=400)
```

Transmission spectrum with flux measurement:

```python
freg = mp.FluxRegion(center=mp.Vector3(0.5*sx - dpml - 0.5),
                     size=mp.Vector3(0, 2*w))
trans = sim.add_flux(fcen, df, nfreq=500, freg)

sim.run(until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ey, mp.Vector3(0.5*sx-dpml-0.5), 1e-3))
sim.display_fluxes(trans)
```

#### Key Takeaways

- A photonic crystal cavity is formed by a local perturbation (here, increased hole spacing d=1.4 vs regular d=1) that creates a localized mode within the photonic band gap.
- The `--resonant_modes` flag switches between Harminv resonance analysis and broadband flux-based transmission spectrum measurement — two complementary characterization methods for the same structure.
- Dual mirror symmetry (`mp.Mirror(mp.X)` and `mp.Mirror(mp.Y)`) reduces a 2D problem to a quarter-cell computation, available only when source and geometry both respect both symmetries.
- `mp.stop_when_fields_decayed(dt, component, point, threshold)` is the recommended simulation termination criterion: it ends the run once fields have decayed by `threshold` from their peak at the monitor point.
- Cavity Q factors increase exponentially with the number of mirror holes N; the transmission peak narrows correspondingly as N increases.

---

### 15. `test_mpb.py` — MPB Test Suite: Verification and Regression Testing

**Physics:** Comprehensive unit tests validating MPB eigenfrequencies, field values, gap lists, and solver outputs against reference data, ensuring numerical correctness of the MPB Python bindings.
**Difficulty:** Advanced
**Source:** `python/tests/test_mpb.py`
**Test Status:** TIMEOUT (CPU-intensive: runs dozens of full MPB band structure calculations with tight tolerance 1e-12)

#### Theory

Numerical eigenvalue solvers must be validated against analytically known results and previous-version outputs to catch regression bugs introduced by code changes. The MPB test suite does this through:

1. **Analytical benchmarks**: For homogeneous media (no geometry), the photonic bands are exact plane waves with omega_n(**k**) = c*|**k** + **G**_n| / sqrt(eps), where **G**_n are reciprocal lattice vectors. The test `test_run_te_no_geometry` verifies that MPB matches these exact values to 3 decimal places.

2. **Reference H5 comparisons**: Field arrays computed by the current code are compared against pre-computed reference HDF5 files in `python/tests/data/`. The function `check_fields_against_h5` reads the reference fields (stored as separate real and imaginary parts per component) and verifies that the computed fields match within `epsilon=1e-4`.

3. **Band range data**: `check_band_range_data` verifies that the (minimum frequency, k-point) and (maximum frequency, k-point) for each band match expected values. This tests the entire dispersion relation, not just specific k-points.

4. **Gap list**: `check_gap_list` verifies that detected photonic band gaps (frequency ranges where no bulk modes exist) match expected values.

5. **Attribute accessors**: `test_attribute_accessors` verifies the complete set of ModeSolver configuration parameters: `num_bands`, `deterministic`, `tolerance`, `mesh_size`, `target_freq`, `dimensions`, `verbose`, `ensure_periodicity`, `eigensolver_flops`, `negative_epsilon_ok`, `epsilon_input_file`, `mu_input_file`, `force_mu`, `use_simple_preconditioner`, `eigensolver_nwork`, and `eigensolver_block_size`.

The `deterministic=True` flag forces MPB to use a fixed random seed for the initial eigenvector guess, making results reproducible across runs. Without this, the random initial guess can lead to slightly different convergence paths, and occasionally different sign conventions for eigenvectors.

The tight tolerance `1e-12` (compared to the default `1e-7`) in the test setup ensures that the eigensolver is converged to near machine precision. This is important for validating field values: at lower tolerance, the eigenvectors may be accurate enough for eigenvalues but not for detailed field comparisons.

The test class inherits from `ApproxComparisonTestCase` (defined in `python/tests/utils.py`), which provides `assertClose` for comparing arrays within an epsilon tolerance. Standard `assertAlmostEqual` is used for scalar comparisons.

#### Code Walkthrough

Standard test solver initialization (square lattice of rods):

```python
def init_solver(self, geom=True):
    num_bands = 8
    k_points = mp.interpolate(4, [mp.Vector3(), mp.Vector3(0.5),
                                  mp.Vector3(0.5, 0.5), mp.Vector3()])
    geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))] if geom else []
    return mpb.ModeSolver(
        num_bands=num_bands, k_points=k_points,
        geometry=geometry,
        geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1)),
        resolution=32,
        filename_prefix=self.filename_prefix,
        deterministic=True,
        tolerance=1e-12,
    )
```

The `filename_prefix` redirects all file output (HDF5 field files, etc.) to a temporary directory that is cleaned up after the test class completes.

Verifying a TE band structure against expected values:

```python
def test_run_te(self):
    expected_freqs = [0.0, 0.5527092320101986, ...]
    expected_brd = [
        ((0.0, mp.Vector3(0,0,0)), (0.496, mp.Vector3(0.5,0.5,0))),
        ...
    ]
    ms = self.init_solver()
    ms.run_te()
    self.check_band_range_data(expected_brd, ms.band_range_data)
    for e, r in zip(expected_freqs, ms.all_freqs[-1]):
        self.assertAlmostEqual(e, r, places=3)
```

`ms.all_freqs[-1]` retrieves the frequencies at the last k-point (Gamma, returning from M). `ms.band_range_data` is a list of `((min_freq, min_k), (max_freq, max_k))` tuples for each band across all computed k-points.

Field verification against HDF5 reference:

```python
def check_fields_against_h5(self, ref_path, field, suffix=""):
    with h5py.File(ref_path, 'r') as ref:
        ref_x = mp.complexarray(ref['x.r'][()], ref['x.i'][()])
        ref_y = mp.complexarray(ref['y.r'][()], ref['y.i'][()])
        ref_z = mp.complexarray(ref['z.r'][()], ref['z.i'][()])
        ref_arr = np.zeros(np.prod(field.shape), dtype=np.complex128)
        ref_arr[0::3] = ref_x.ravel()
        ref_arr[1::3] = ref_y.ravel()
        ref_arr[2::3] = ref_z.ravel()
        self.assertClose(ref_arr, field, epsilon=1e-4)
```

The HDF5 format stores each vector field component's real and imaginary parts separately (e.g., `x.r`, `x.i`, `y.r`, `y.i`, `z.r`, `z.i`). `mp.complexarray` reconstructs the complex array.

#### Key Takeaways

- `deterministic=True` and `tolerance=1e-12` are essential for reproducible regression testing; without them, random initial conditions and early convergence can give slightly different results.
- `ms.all_freqs[-1]` retrieves band frequencies at the last k-point; `ms.all_freqs[i]` retrieves them at the i-th k-point. `ms.band_range_data` stores the global min/max across all k-points for each band.
- Reference HDF5 files in `python/tests/data/` store field arrays split into real/imaginary parts per component; `mp.complexarray(real, imag)` reconstructs complex NumPy arrays.
- The `MEEP_SKIP_LARGE_TESTS` environment variable allows skipping this entire test class in CI environments where runtime is constrained.
- `ms.display_eigensolver_stats()` prints convergence diagnostics; in tests, the tight tolerance=1e-12 ensures that the eigensolver is fully converged before fields are extracted for comparison.

---

## Summary

This chapter covered the complete range of photonic crystal band structure calculations available through MPB and Meep:

**1D structures** (sections 8-9): Bragg mirrors and sinusoidally-varying stacks demonstrate the simplest photonic band gaps with analytical benchmarks and custom material functions.

**2D structures** (sections 2-5, 10-11): Square and triangular lattices of rods and holes illustrate how lattice symmetry, geometry (rods vs holes), and filling fraction control which polarizations develop band gaps. The honeycomb lattice achieves a complete gap. Line-defect waveguides (section 10) and strip waveguides (section 11) extend the framework to guided modes.

**3D structures** (section 7): The diamond FCC lattice demonstrates the additional complexity of 3D band gap calculations with non-orthogonal lattice vectors and higher computational cost.

**Defects and devices** (sections 1, 10, 13-14): Point defects (cavities) and line defects (waveguides) show how band gaps enable optical confinement. The holey waveguide examples bridge MPB band structure analysis and Meep transmission/resonance calculations.

**Advanced techniques** (sections 1, 12, 15): Gap optimization, field post-processing with MPBData, and the full test suite validate and extend the core calculations.

The key APIs introduced in this chapter are:

| API | Purpose |
|-----|---------|
| `mpb.ModeSolver(...)` | Configure and run MPB band structure calculations |
| `ms.run_te()`, `ms.run_tm()`, `ms.run()` | Polarization-specific or combined band solves |
| `ms.run_zeven()`, `ms.run_zodd()` | Vertical-parity band solves for slab structures |
| `ms.find_k(parity, omega, ...)` | Inverse solve: find k given omega |
| `ms.retrieve_gap(n)` | Gap-to-midgap ratio for band gap n |
| `ms.target_freq` | Targeted eigensolver for defect modes |
| `mpb.output_at_kpoint(k, ...)` | Conditional field output at one k-point |
| `mpb.fix_efield_phase` / `mpb.fix_field_phase` | Phase normalization for reproducible visualization |
| `mpb.MPBData(rectify, periods, resolution)` | Coordinate transform for non-rectangular lattices |
| `mp.geometric_objects_lattice_duplicates(lat, geom)` | Tile unit-cell geometry over a supercell |
| `sim.run_k_points(T, k_list)` | Meep FDTD band structure via Harminv sweep |
