# Chapter 12: Simulation Infrastructure

This chapter covers Meep's simulation management infrastructure: the tools and subsystems that sit beneath the physics and make production-grade simulations reliable, scalable, and debuggable. Topics include checkpoint/restart via HDF5 dump and load, MPI process subdivision for parameter sweeps, pre-run fragment statistics and load-balance analysis, performance timing instrumentation, verbosity control, 2D visualization and animation, and field array slicing for data extraction.

---

### 1. `test_dump_load.py` — Checkpoint and Restart of Simulation State

**Physics:** Infrastructure — serialize and deserialize the complete state of an FDTD simulation (dielectric structure plus time-domain fields) to HDF5 files so that long runs can be interrupted and resumed without loss of accuracy.
**Difficulty:** Advanced
**Source:** `python/tests/test_dump_load.py`
**Test Status:** TIMEOUT (CPU-intensive; high-resolution 2D and 3D runs with resolution 50 and 15 respectively produce large field arrays that are slow to write and re-read)

#### Theory

An FDTD simulation advances the electromagnetic field through a leapfrog time-stepping algorithm: given the field state at time step $n$, it computes the state at step $n+1$ using finite differences of Maxwell's curl equations. Because each step depends only on the immediately preceding state, the simulation is fully defined by two objects: the dielectric structure (the $\epsilon(\mathbf{r})$, $\mu(\mathbf{r})$, and material-dispersion auxiliary fields on the Yee grid) and the current field arrays (E, H, D, B, and any polarization currents). Checkpointing therefore requires serializing both of these objects faithfully.

The significance of checkpoint/restart is immense in practice. A photonic crystal optimization run might take days on a cluster. System administrators impose wall-time limits (commonly 24 or 48 hours). Without checkpoint/restart, any job that exceeds the time limit must start over from scratch. With it, the simulation can be dumped at any convenient point and continued on the next allocation. This is not merely a convenience — it is often the only practical way to complete large 3D calculations.

Meep separates the checkpoint into two independently controllable parts. The structure dump captures the material geometry: the $\epsilon$ and $\mu$ tensors, susceptibility coefficients (Lorentzian, Drude), and conductivity values on every grid cell. The fields dump captures the instantaneous field state: every E, H, D, B array and the polarization state $\mathbf{P}^{(n)}$ for dispersive materials. Crucially, the current implementation raises an error if a fields dump is attempted when there is a non-null polarization state, because dispersive materials require extended auxiliary state that is not yet fully serialized. Users working with materials like aluminum (which has Drude-Lorentz poles) must dump only the structure when dispersive dynamics are active.

In an MPI-parallel run, the field data is distributed across multiple ranks. Meep provides two strategies for writing this distributed data. The `single_parallel_file=True` mode (the default) collects all data to rank 0 and writes a single HDF5 file. This is simple to manage but creates a bottleneck at rank 0 for very large problems. The `single_parallel_file=False` (sharded) mode writes one HDF5 file per MPI rank, named with a rank suffix. Sharded output is faster because all ranks write simultaneously, but the files must be reassembled if you want to inspect them with standard HDF5 tools.

The test also validates chunk-layout preservation. Meep divides its simulation domain into spatial chunks, one per MPI rank (or more). When reloading, the new simulation must use the same chunk layout as the dumped one, otherwise the field arrays on disk would not align with the new partition. Meep supports three ways to specify this: passing a chunk layout HDF5 file (written by `sim.dump_chunk_layout()`), passing the original simulation object directly as `chunk_layout=sim1`, or when using two MPI processes, constructing a `BinaryPartition` object explicitly. These three modes are all exercised by the test.

#### Code Walkthrough

The test class defines two major logical patterns: structure-only dumps and structure-plus-fields dumps, each tested in 2D and 3D. Consider the 2D structure dump pattern in `_load_dump_structure_2d`.

The reference simulation is constructed and run first, collecting field values at a sample point every 5 time units until $t=50$:

```python
sim1 = mp.Simulation(
    resolution=resolution,
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    symmetries=symmetries,
    sources=[sources],
)

ref_field_points = []
def get_ref_field_point(sim):
    p = sim.get_field_point(mp.Ez, sample_point)
    ref_field_points.append(p.real)

sim1.run(mp.at_every(5, get_ref_field_point), until=50)
```

After the first run completes, the structure is dumped. The `dump_structure=True, dump_fields=False` combination writes only the epsilon/mu grid, not the time-domain fields:

```python
sim1.dump(
    dump_dirname,
    dump_structure=True,
    dump_fields=False,
    single_parallel_file=single_parallel_file,
)
```

Optionally, the chunk layout is saved separately so it can be passed to the new simulation. This is important because the second simulation has no geometry — it loads the geometry from the dump file — so it would otherwise not know how to partition the domain among MPI ranks:

```python
if chunk_file:
    sim1.dump_chunk_layout(dump_chunk_fname)
    chunk_layout = dump_chunk_fname
if chunk_sim:
    chunk_layout = sim1  # pass the original sim object directly
```

The second simulation is created without geometry (no `geometry=` argument), then loads the structure from disk before running:

```python
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell,
    boundary_layers=pml_layers,
    sources=[sources],
    symmetries=symmetries,
    chunk_layout=chunk_layout,
)
sim.load(dump_dirname, load_structure=True, load_fields=False,
         single_parallel_file=single_parallel_file)
sim.run(mp.at_every(5, get_field_point), until=50)
```

The test then asserts that every sampled field point from the restored simulation matches the reference to within Python's default `assertAlmostEqual` tolerance (7 decimal places). Because the structure is identical down to the last grid cell, the time-stepping update equations produce identical floating-point results.

The fields dump test, `_load_dump_fields_2d`, is more sophisticated. It exercises the scenario where you want to resume a run from a mid-simulation checkpoint:

```python
sim1.run(mp.at_every(1, get_ref_field_point), until=15)
sim1.dump(dump_dirname, dump_structure=True, dump_fields=True, ...)

# Continue reference to t=20
sim1.run(mp.at_every(1, get_ref_field_point), until=5)
```

The second simulation loads structure first, then loads fields (restoring the simulation state to $t=15$) and continues for 5 more time units to reach $t=20$. DFT monitor arrays accumulated before the checkpoint are also verified to match after reload using `sim.get_dft_array()`.

The test `test_dump_fails_for_non_null_polarization_state` explicitly confirms that attempting to dump fields when a dispersive material (aluminum, which has Drude poles) has been time-stepped raises a `RuntimeError`. This test is skipped in MPI builds because MPI typically calls `meep::abort` which does not translate cleanly into a Python exception.

#### Key Takeaways

- Use `sim.dump(dir, dump_structure=True, dump_fields=True)` and `sim.load(dir, ...)` to checkpoint and resume long simulations; the two flags are independent so you can update one without the other.
- The `chunk_layout` parameter on `mp.Simulation` must match the layout used when the dump was created; pass the original sim object, a layout HDF5 file, or a `BinaryPartition` to guarantee consistency.
- Field dumps are not yet supported for simulations containing dispersive materials with active polarization currents (Drude/Lorentz); dump structure only in those cases and restart from $t=0$ with the material geometry reloaded.
- The `single_parallel_file=False` (sharded) mode writes one file per MPI rank and is faster for large distributed runs, but requires careful file management.
- DFT monitor state is included in the structure dump, so accumulated Fourier data is preserved across a checkpoint/restart cycle when both dumps are used together.

---

### 2. `test_divide_mpi_processes.py` — MPI Process Subdivision for Concurrent Parameter Sweeps

**Physics:** Infrastructure — split a set of MPI processes into independent subgroups, each running a different simulation in parallel, and then gather results across all subgroups.
**Difficulty:** Intermediate
**Source:** `python/tests/test_divide_mpi_processes.py`
**Test Status:** PASS (7.0s; skipped when fewer than 2 MPI processes are available)

#### Theory

A common workflow in computational photonics is the parametric sweep: run the same geometry at many different frequencies, geometric parameters, or material values, and collect a scalar result (flux, transmission, Q factor) from each. In a naive implementation, each parameter value is run sequentially. This is embarrassingly parallel — the runs do not communicate with each other — so the ideal approach is to run all of them simultaneously.

Meep supports this through the `mp.divide_parallel_processes(k)` function, which subdivides the available MPI communicator into $k$ equal-sized subgroups. Each subgroup independently runs a full Meep simulation using only the ranks assigned to it. Within each subgroup, the usual domain-decomposition parallelism still applies, so a subgroup with multiple ranks decomposes the simulation cell spatially. The two levels of parallelism — across subgroups (parameter sweep dimension) and within subgroups (spatial decomposition) — are completely orthogonal.

The function `divide_parallel_processes(k)` returns an integer `n` that uniquely identifies the subgroup, ranging from 0 to $k-1$. Each rank within subgroup $n$ receives the same value of `n`. The test uses this to set the center frequency of a Gaussian source: subgroup 0 uses $f_\text{cen} = 1/(0+1) = 1.0$ and subgroup 1 uses $f_\text{cen} = 1/(1+1) = 0.5$.

After all subgroups have completed their respective simulations, results are gathered using `mp.merge_subgroup_data(value)`. This collective operation assembles one value from each subgroup (the value on rank 0 of each subgroup, by convention) into a Python list. The gathered list has exactly $k$ elements, one per subgroup, in subgroup-index order.

The communication architecture here reflects a common design in HPC frameworks: a single MPI job is submitted to the scheduler requesting $N$ total cores; the job then subdivides those cores among $k$ concurrent tasks at runtime. This avoids the overhead of launching $k$ separate MPI jobs through the scheduler and keeps all inter-subgroup coordination within a single process group.

#### Code Walkthrough

The test begins by verifying the environment and dividing processes:

```python
@unittest.skipIf(mp.count_processors() < 2, "MPI specific test")
class TestDivideParallelProcesses(unittest.TestCase):
    def test_divide_parallel_processes(self):
        n = mp.divide_parallel_processes(2)
        fcen = 1.0 / (n + 1)
```

The call to `divide_parallel_processes(2)` splits all available MPI ranks into two subgroups and returns the subgroup index `n`. The center frequency `fcen` is computed from `n`, so each subgroup runs at a different frequency. From this point on, all Meep operations (grid construction, time-stepping, flux computation) occur independently within each subgroup.

The simulation itself is straightforward: a square 2D cell with PML, a Gaussian source, and four flux regions forming a box around the source to measure total radiated power:

```python
flux_box = self.sim.add_flux(
    fcen, 0, 1,
    mp.FluxRegion(mp.Vector3(y=0.5*sxy), size=mp.Vector3(sxy)),
    mp.FluxRegion(mp.Vector3(y=-0.5*sxy), size=mp.Vector3(sxy), weight=-1),
    mp.FluxRegion(mp.Vector3(0.5*sxy), size=mp.Vector3(y=sxy)),
    mp.FluxRegion(mp.Vector3(-0.5*sxy), size=mp.Vector3(y=sxy), weight=-1),
    decimation_factor=1,
)
self.sim.run(until_after_sources=30)
tot_flux = mp.get_fluxes(flux_box)[0]
```

The `weight=-1` on two of the four flux regions ensures that the fluxes on opposite sides of the box subtract correctly to give the net outward power. After the run, each subgroup holds its own `tot_flux` value.

The merge step gathers data across subgroups:

```python
tot_fluxes = mp.merge_subgroup_data(tot_flux)
fcens = mp.merge_subgroup_data(fcen)

self.assertEqual(fcens[0], 1)
self.assertEqual(fcens[1], 0.5)
self.assertAlmostEqual(tot_fluxes[0], 9.8628728533, places=4)
self.assertAlmostEqual(tot_fluxes[1], 19.6537275387, places=4)
```

The two merged lists contain results indexed by subgroup number. Subgroup 0, which ran at $f=1.0$, yields a lower total flux than subgroup 1 at $f=0.5$, consistent with the frequency dependence of radiation from a point source in 2D. The 4-decimal-place tolerance accommodates single-precision builds.

#### Key Takeaways

- `mp.divide_parallel_processes(k)` returns an integer subgroup index 0 to $k-1$ and splits all MPI ranks into $k$ independent communicators; use it at the top of your script before any `mp.Simulation` construction.
- Each subgroup runs a fully independent simulation; there is no communication between subgroups during time-stepping, only at the final `merge_subgroup_data` call.
- `mp.merge_subgroup_data(scalar)` gathers one scalar per subgroup into a list; use it after the simulation to collect results from all parameter values simultaneously.
- The total MPI rank count must be divisible by $k$; if it is not, the behavior is undefined. Plan your job submission accordingly.
- This pattern scales to arbitrary numbers of parameters: a sweep over 16 frequencies on 64 ranks would call `divide_parallel_processes(16)` and assign 4 ranks per frequency for spatial decomposition.

---

### 3. `test_fragment_stats.py` — Fragment Statistics and Load-Balance Analysis

**Physics:** Infrastructure — compute per-fragment complexity statistics (number of anisotropic, nonlinear, dispersive, and PML pixels) to enable load-balanced domain decomposition before a simulation begins.
**Difficulty:** Advanced
**Source:** `python/tests/test_fragment_stats.py`
**Test Status:** PASS (6.1s)

#### Theory

When Meep decomposes a simulation domain across multiple MPI ranks, it must decide how to partition the spatial domain. A naive partition divides the domain into equal-volume slices, but this ignores the fact that different regions of the domain have very different computational costs. A cell occupied by an isotropic, non-dispersive dielectric requires only a few floating-point operations per time step: the standard E and H curl updates. A cell occupied by an anisotropic material with off-diagonal $\epsilon$ tensor requires additional tensor-vector multiplications. A cell with Drude or Lorentzian susceptibilities requires integrating one or more auxiliary polarization ODEs per time step. A PML cell requires additional damping terms that vary in dimensionality depending on how many PML faces overlap at that cell.

Meep's load-balance framework addresses this by dividing the domain into fragments (sub-blocks of the computational grid), computing a complexity score for each fragment, and then partitioning fragments across MPI ranks such that each rank receives approximately equal total complexity. The fragment statistics computed are:

- **num_anisotropic_eps_pixels**: cells where $\epsilon$ has off-diagonal components, requiring full 3x3 tensor arithmetic.
- **num_anisotropic_mu_pixels**: same for $\mu$.
- **num_nonlinear_pixels**: cells with $\chi^{(2)}$ or $\chi^{(3)}$ nonlinearity, requiring field-dependent polarization evaluation.
- **num_susceptibility_pixels**: cells with Lorentzian or Drude poles, requiring auxiliary ODE integration per time step.
- **num_nonzero_conductivity_pixels**: cells with finite conductivity $\sigma$, which adds a damping term to the E or H update.
- **num_dft_pixels**: cells participating in DFT monitor accumulation (flux, near-to-far, force, field DFT), scaled by the number of frequency points.
- **num_1d_pml_pixels**, **num_2d_pml_pixels**, **num_3d_pml_pixels**: PML cells categorized by how many PML faces they overlap (edges and corners are more expensive than faces).

The categorization of PML pixels deserves special explanation. In 1D, a PML slab modifies only one field component update. In 2D, a corner cell where two PML slabs overlap requires two separate damping terms and is roughly twice as expensive. In 3D, an edge where two PML slabs meet is a 2D PML, and a corner where all three PML slabs meet is a 3D PML, each progressively more expensive.

The `TestPMLToVolList` class tests the lower-level geometry that computes these regions: `sim._boundary_layers_to_vol_list()` returns three lists of volumes corresponding to the 1D, 2D, and 3D PML regions. The test verifies exact bounding boxes for these regions in 1D, 2D, 3D, and cylindrical geometries with various combinations of PML thickness and side specification.

The `TestChunkCommunicationArea` class tests a related quantity: the area of the shared face between adjacent MPI chunks. Communication volume between ranks scales with this area because field values on shared boundaries must be exchanged every time step. The test verifies that `get_max_chunk_communication_area()` and `get_avg_chunk_communication_area()` return correct values for 2D and 3D cells with and without periodic boundary conditions.

#### Code Walkthrough

The helper `get_fragment_stats` builds a simulation with a deliberately complex material containing every type of special physics, then calls `sim._compute_fragment_stats(gv)`:

```python
mat = mp.Medium(
    epsilon=12,
    epsilon_offdiag=mp.Vector3(z=1),    # anisotropic eps
    mu_offdiag=mp.Vector3(x=20),        # anisotropic mu
    E_chi2_diag=mp.Vector3(1, 1),       # chi2 nonlinearity
    H_chi3_diag=mp.Vector3(z=1),        # chi3 nonlinearity
    E_susceptibilities=[
        mp.LorentzianSusceptibility(),
        mp.NoisyLorentzianSusceptibility(),
    ],
    H_susceptibilities=[mp.DrudeSusceptibility()],
    D_conductivity_diag=mp.Vector3(y=1),
    B_conductivity_diag=mp.Vector3(x=1, z=1),
)
```

Note that `_create_grid_volume(False)` is called to instantiate the Yee grid without actually running the simulation. This allows fragment statistics to be computed as a pre-run analysis step.

A representative 1D test with PML and symmetry illustrates how the pixel counts scale:

```python
def _test_1d(self, sym, pml=[]):
    fs = self.get_fragment_stats(
        mp.Vector3(z=10), mp.Vector3(z=30), 1, dft_vecs=dft_vecs, sym=sym, pml=pml
    )
    sym_factor = 2 if sym else 1
    self.check_stats(
        fs,
        a_eps=300 / sym_factor,
        a_mu=300 / sym_factor,
        nonlin=300 / sym_factor,
        susc=300 / sym_factor,
        cond=300 / sym_factor,
    )
    self.assertEqual(fs.num_dft_pixels, 40800)
```

The 1D cell has length 30 at resolution 10, giving 300 grid points. The material block covers 10 units (100 points) at the center. The anisotropic and nonlinear counts differ, reflecting how many cells contain the block material. With Mirror symmetry applied, the effective domain is halved, so all pixel counts divide by 2. The DFT pixel count of 40800 comes from the three DFT monitors (flux, near2far, force), each covering 100 grid points times multiple frequency bins and field components.

In 2D, symmetry reduces by a factor of 4 (two Mirror symmetries), and in 3D by 8. The anisotropic epsilon count in 3D without symmetry is 27,000,000 for a 10x10x10 block at resolution 10, while the nonlinear count is only 3,000,000 — illustrating that only chi2 and chi3 (and not epsilon anisotropy) are counted as nonlinear.

The PML volume list test is verified geometrically:

```python
def test_2d_all_directions_all_sides(self):
    sim = make_sim(mp.Vector3(10, 10), 10, [mp.PML(1)], 2)
    v1, v2, v3 = sim._boundary_layers_to_vol_list(sim.boundary_layers)
    # 4 face regions (1D PML), 4 corner regions (2D PML)
    self.assertEqual(len(v1), 4)
    self.assertEqual(len(v2), 4)
```

In 2D with PML on all four sides, there are 4 pure face regions (1D PML) and 4 corner regions where two PML faces overlap (2D PML). The test verifies the exact bounding-box coordinates for each region.

#### Key Takeaways

- Fragment statistics are computed before any time-stepping via `sim._compute_fragment_stats(gv)` and can be used to predict and optimize load balance before submitting a long job.
- Anisotropic epsilon and mu counts are based on all cells containing material with off-diagonal tensor components; nonlinear and susceptibility counts reflect only cells with those specific material properties.
- PML complexity is categorized into 1D (face), 2D (edge), and 3D (corner) regions; the Absorber boundary condition contributes to conductivity pixels instead of PML pixel counts.
- Mirror symmetry reduces effective pixel counts by a factor of 2 per symmetry axis, reflecting the actual computational saving from exploiting the symmetry.
- `get_max_chunk_communication_area()` and `get_avg_chunk_communication_area()` quantify inter-rank communication surface area, which is a key driver of parallel efficiency for large 3D simulations.

---

### 4. `test_timing_measurements.py` — Performance Profiling and Timing Instrumentation

**Physics:** Infrastructure — instrument a running simulation to collect wall-time breakdowns across all computational phases, and compute MPI communication efficiency metrics.
**Difficulty:** Beginner
**Source:** `python/tests/test_timing_measurements.py`
**Test Status:** PASS (4.3s)

#### Theory

Understanding where simulation time is spent is essential for performance optimization. An FDTD code has several distinct computational phases that contribute to total runtime:

**Field update operations** update the six field components in each Yee cell. In the leapfrog algorithm, H is updated at half-integer time steps and E at integer steps. Separate timers track updates to B (magnetic flux density), H (magnetic field, which requires applying the inverse $\mu$ tensor), D (electric flux density), and E (electric field, requiring inverse $\epsilon$). Boundary stepping timers track the analogous updates applied within PML regions for each component.

**Fourier transform accumulation** occurs at every time step for each DFT monitor. The cost scales as: (number of DFT pixels) × (number of frequency points) × (number of time steps). For large frequency sweeps over many monitors, this can dominate runtime.

**MPI communication** occurs in two forms. One-to-one (point-to-point) communication transfers field boundary data between adjacent ranks, happening every time step with volume proportional to the shared surface area. All-to-all communication is used for collective operations such as computing global flux values or synchronizing field norms. The ratio of communication time to computation time defines the communication efficiency — a dimensionless metric that indicates how well the parallelism is scaling.

The communication efficiency is defined as:

$$\eta_\text{comm} = \frac{\langle t_\text{mpi\_all} + t_\text{mpi\_one} \rangle}{\langle t_\text{step} \rangle + \langle t_\text{DFT} \rangle}$$

where angle brackets denote averages over MPI ranks. If $\eta_\text{comm}$ approaches 1, the simulation spends nearly as much time communicating as computing, indicating poor scalability. An efficient distributed simulation keeps $\eta_\text{comm}$ well below 0.1.

The `timing_measurements` module defines the `TIMING_MEASUREMENT_IDS` dictionary, which maps human-readable names like `"time_stepping"` and `"mpi_all_to_all"` to Meep's internal integer timing codes (e.g., `mp.Stepping`, `mp.MpiAllTime`). After a simulation runs, `sim.time_spent_on(code)` returns a NumPy array of per-rank timings for that phase.

The `MeepTimingMeasurements` class wraps this raw data into a structured container with computed properties:

- `measurements`: dict mapping timer name to per-rank timing list.
- `elapsed_time`: total wall time for the simulation.
- `num_time_steps`: total number of leapfrog steps taken (`sim.fields.t`).
- `comm_efficiency`: combined MPI communication fraction.
- `comm_efficiency_one_to_one`: fraction due to point-to-point transfers.
- `comm_efficiency_all_to_all`: fraction due to collective operations.

#### Code Walkthrough

The test creates a simple 3D simulation and runs it for a fixed number of time units:

```python
sim = mp.Simulation(
    cell_size=mp.Vector3(2, 2, 2),
    resolution=20,
)
time_start = time.time()
sim.run(until=5)
timing_measurements = timing.MeepTimingMeasurements.new_from_simulation(sim)
```

`new_from_simulation` is the factory class method. It iterates over the `TIMING_MEASUREMENT_IDS` dictionary and calls `sim.time_spent_on(id)` for each:

```python
measurements = {
    name: sim.time_spent_on(timing_id).tolist()
    for name, timing_id in TIMING_MEASUREMENT_IDS.items()
}
```

The full set of 22 timer names corresponds to fine-grained phases of the FDTD update loop, covering both the global update categories and the boundary-region stepping for each component pair (B/WH/PH/H for the magnetic cycle, D/WE/PE/E for the electric cycle).

The test then verifies the expected structure of the result:

```python
self.assertSetEqual(
    set(timing_measurements.measurement_names),
    set(timing.TIMING_MEASUREMENT_IDS.keys()),
)
self.assertTrue(
    timing_measurements.elapsed_time > 0
    or timing_measurements.elapsed_time == -1
)
self.assertGreater(timing_measurements.num_time_steps, 0)
self.assertGreaterEqual(timing_measurements.comm_efficiency, 0)
```

The `elapsed_time` check allows -1 because the factory method defaults `elapsed_time=-1` when the caller does not provide a measured wall time. The communication efficiency checks use `assertGreaterEqual(0)` because in a single-process run, MPI communication time is zero and the efficiency is exactly 0.

For production use, you would capture the elapsed time explicitly:

```python
import time
start = time.time()
sim.run(until=T)
measurements = timing.MeepTimingMeasurements.new_from_simulation(
    sim, elapsed_time=time.time() - start
)
print(f"Field update: {measurements.measurements['time_stepping']} s")
print(f"DFT: {measurements.measurements['fourier_transform']} s")
print(f"Comm efficiency: {measurements.comm_efficiency:.3f}")
```

#### Key Takeaways

- Import `meep.timing_measurements` and use `MeepTimingMeasurements.new_from_simulation(sim)` after any `sim.run()` call to get a complete per-phase timing breakdown.
- The 22 timing categories cover all phases of the FDTD loop; the most important for optimization are `time_stepping`, `fourier_transform`, `mpi_one_to_one`, and `mpi_all_to_all`.
- Communication efficiency $\eta_\text{comm}$ quantifies MPI overhead; values below 0.1 are generally acceptable, while values above 0.3 indicate that adding more MPI ranks is unlikely to improve performance.
- `sim.time_spent_on(mp.Stepping)` can be called directly for quick access to a single timer; it returns a NumPy array with one element per MPI rank.
- The `num_time_steps` attribute equals `sim.fields.t`, the internal leapfrog counter, which counts individual half-steps in the leapfrog; the number of full time-unit intervals is `num_time_steps * sim.Courant / resolution`.

---

### 5. `test_verbosity_mgr.py` — Verbosity Control and the Singleton Pattern

**Physics:** Infrastructure — manage the global output verbosity level across all Meep C++ library components (core Meep and optionally MPB) through a unified Python singleton object.
**Difficulty:** Beginner
**Source:** `python/tests/test_verbosity_mgr.py`
**Test Status:** PASS (3.6s)

#### Theory

Large-scale simulations running on HPC clusters can generate enormous amounts of console output if left unchecked. A single time-step in a 3D simulation with many DFT monitors might print dozens of diagnostic lines. Over millions of time steps, this produces gigabytes of stdout that the batch scheduler captures to log files — files that are slow to write, slow to read, and in most cases contain nothing useful for the end user.

Meep provides a four-level verbosity system to control output quantity:

- **Level 0**: minimal — only critical warnings and errors are printed.
- **Level 1** (default): a moderate amount of diagnostic information — initialization messages, timing summaries, convergence notices.
- **Level 2**: verbose — additional per-step or per-region information useful for debugging.
- **Level 3**: debugging — exhaustive output intended for Meep developers.

The verbosity level is stored as an integer flag in each C++ shared library that Meep uses. When the Python `meep` package is imported, the C-level `cvar.verbosity` variable is registered. When `meep.mpb` is imported later, the MPB library's verbosity flag is also registered. The `Verbosity` singleton manages all registered flags simultaneously, so a single call to `meep.verbosity(0)` silences both Meep and MPB output at once.

The singleton pattern ensures that no matter how many times `Verbosity()` is instantiated (which happens once per library import), the same underlying object is always returned. This is implemented via `__new__` override: if `cls._instance is None`, a real object is created and stored; otherwise the stored instance is returned. Each time a new library's verbosity flag is added, a Python property is dynamically added to the class via `make_property()`, which creates a getter/setter pair wrapping the C-level attribute access.

The design prioritizes usability: `verbosity(2)` sets all flags, `verbosity.meep = 2` sets only Meep's flag while leaving MPB at its current level, `int(verbosity)` returns the current level, and comparison operators (`==`, `<`, `>`) work naturally against integers. This allows code like `if mp.verbosity > 1: print(...)` to gate expensive diagnostic output.

#### Code Walkthrough

The test creates a test-specific subclass with its own `_instance = None` to avoid interfering with the real Meep singleton:

```python
class VerbosityForTest(Verbosity):
    """Allows for testing of Verbosity without interfering with the singleton."""
    _instance = None
```

The singleton identity test confirms that multiple constructor calls return the same object:

```python
def test_identity(self):
    v1, v2 = self.v1, self.v2
    self.assertTrue(v1 is v2)
    self.assertEqual(id(v1), id(v2))
    self.assertEqual(v1.get_all(), [1, 1])
```

Both `v1` and `v2` are the same Python object (same `id()`), and `get_all()` returns a list of the verbosity values for every registered cvar, both initialized to 1.

The property test shows how individual library flags can be set independently:

```python
def test_properties(self):
    v1.foo = 2
    v2.bar = 3
    self.assertEqual(v2.foo, 2)  # shared singleton, both sides see it
    self.assertEqual(v2.bar, 3)
```

Setting `v1.foo` updates the underlying cvar named `"foo"`, and because `v1` and `v2` are the same singleton, `v2.foo` reflects the same value.

The callable interface is verified:

```python
def test_operators(self):
    v1(3)                    # sets all flags to 3
    self.assertTrue(v2 == 3)
```

Range validation ensures levels stay within 0-3:

```python
def test_out_of_range(self):
    with self.assertRaises(ValueError):
        v1.set(5)
    with self.assertRaises(ValueError):
        v1.set(-5)
```

In real simulation scripts, verbosity is typically managed like this:

```python
import meep as mp

mp.verbosity(0)        # silence all output for batch runs
# ... run simulation ...
mp.verbosity(1)        # restore normal output
```

Or for selective control:

```python
mp.verbosity.meep = 2  # verbose Meep output
mp.verbosity.mpb = 0   # silence MPB eigenmode solver
```

#### Key Takeaways

- `mp.verbosity` is a singleton object shared across the entire Python process; any assignment to it propagates immediately to the underlying C++ library.
- Call `mp.verbosity(0)` to suppress all diagnostic output in batch simulations; call `mp.verbosity(1)` to restore the default.
- Individual library verbosity levels can be set independently via named properties: `mp.verbosity.meep = 2` and `mp.verbosity.mpb = 0`.
- Comparison operators (`==`, `<`, `>`) work against plain integers, enabling conditional diagnostic output with `if mp.verbosity > 1`.
- The `reset()` class method drops the singleton for testing purposes; do not call it in production code.

---

### 6. `test_visualization.py` — 2D Plotting and Animation of Simulation Geometry and Fields

**Physics:** Infrastructure — render the simulation geometry, source positions, monitor regions, PML boundaries, and time-domain field data as 2D matplotlib figures and video animations.
**Difficulty:** Beginner
**Source:** `python/tests/test_visualization.py`
**Test Status:** PASS (8.3s)

#### Theory

Visualization plays multiple roles in electromagnetic simulation workflows. Before running an expensive calculation, plotting the geometry and source/monitor positions allows the user to catch setup errors — misplaced sources, overlapping objects, monitors placed inside PML regions — without spending compute time. During development, plotting field distributions after a short run validates that the physics is behaving as expected (wave propagation direction, mode shape, field symmetry). After a production run, field plots provide intuition for interpreting numerical results.

Meep's `plot2D` function extracts the information needed for a 2D visualization from the simulation object and renders it using matplotlib. The rendering pipeline works as follows. The dielectric structure (epsilon distribution) is obtained by calling the internal `get_array` mechanism over the full cell, producing a 2D or 3D NumPy array that is then displayed as a pseudocolor image using `imshow`. Sources are plotted as colored markers or rectangles based on their center and size. Flux and DFT monitors appear as colored overlays. PML regions appear as shaded boundary strips. When the simulation has been time-stepped, field data can be overlaid by passing a field component like `fields=mp.Ez`.

For 3D simulations, `plot2D` requires an `output_plane` argument specifying which 2D cross-section to visualize. An `mp.Volume` with one of its three dimensions set to zero defines a plane parallel to two coordinate axes. The origin of the plane is the center of the volume. Meep interpolates field values onto the slice plane and renders the result. This allows inspection of XY, XZ, and YZ slices at arbitrary offsets, which is essential for understanding 3D field distributions.

The `Animate2D` class extends `plot2D` into a time-series visualization. An `Animate2D` object is used as a step function callback: at each invocation, it calls `plot2D`, captures the rendered image into an in-memory buffer, and appends it to an internal list of frames. After the simulation completes, the frame list can be exported to:

- **MP4 video** via `to_mp4(fps, filename)` — requires ffmpeg.
- **GIF animation** via `to_gif(fps, filename)` — requires Pillow.
- **JavaScript HTML** via `to_jshtml(fps)` — returns a self-contained HTML string suitable for embedding in Jupyter notebooks.

The `normalize=True` option to `Animate2D` normalizes field data to the maximum field value seen so far, which is useful when field amplitudes grow during source injection. The `normalize=False` option uses raw field values, which can show the field growing in amplitude if a source is still active.

The `hash_figure` utility in the test captures the pixel content of a matplotlib figure for approximate comparison. It renders the figure to a raw byte buffer using `fig.savefig(buf, format="raw")`, interprets the result as a uint8 array, and computes a simple hash. The actual assertions comparing hash values are commented out because they would be platform-dependent (rendering varies slightly across matplotlib backends and operating systems), but the infrastructure for comparison is in place.

#### Code Walkthrough

The `setup_sim` function builds a simulation with deliberately complex source and PML configurations to exercise all visualization elements:

```python
pml_layers = [
    mp.PML(2.0, mp.X),
    mp.PML(1.0, mp.Y, mp.Low),
    mp.PML(1.5, mp.Y, mp.High),
]
```

Three distinct PML layers with different thicknesses on different sides allow the visualization test to verify that PML rendering handles asymmetric boundary conditions correctly. Eight sources of different types (point, line, plane) at various positions exercise the source visualization code paths.

The basic 2D plot test has three stages:

```python
# Stage 1: geometry only (before run)
sim = setup_sim()
ax = sim.plot2D(ax=ax)

# Stage 2: geometry + fields (after run)
sim.run(until=200)
ax = sim.plot2D(ax=ax, fields=mp.Ez)

# Stage 3: cropped subdomain
vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(2, 2))
ax = sim.plot2D(ax=ax, fields=mp.Ez, output_plane=vol)
```

Stage 1 renders only the dielectric structure and simulation setup without any field data. Stage 2 adds the Ez field as a pseudocolor overlay on top of the geometry. Stage 3 restricts the plot to a small 2x2 region at the center, demonstrating the `output_plane` argument.

The animation test constructs two `Animate2D` objects to run concurrently:

```python
Animate = mp.Animate2D(sim=sim, fields=mp.Ez, realtime=False, normalize=False)
Animate_norm = mp.Animate2D(sim=sim, fields=mp.Ez, realtime=False, normalize=True)

sim.run(mp.at_every(1, Animate), mp.at_every(1, Animate_norm), until=5)
```

The `mp.at_every(1, callback)` step function causes both animation objects to be invoked at every time unit throughout the 5-unit run, capturing 5 frames each. The `realtime=False` argument suppresses display during the simulation (required for batch/headless environments). Output then tests all three export formats:

```python
Animate.to_mp4(5, os.path.join(self.temp_dir, "test_2D.mp4"))
Animate.to_gif(150, os.path.join(self.temp_dir, "test_2D.gif"))
Animate.to_jshtml(10)
```

For 3D animations, the test combines `mp.in_volume` with `Animate2D` to capture two orthogonal slice planes simultaneously:

```python
sim.run(
    mp.at_every(1, mp.in_volume(
        mp.Volume(center=mp.Vector3(),
                  size=mp.Vector3(sim.cell_size.x, sim.cell_size.y)),
        Animate_xy,
    )),
    mp.at_every(1, mp.in_volume(
        mp.Volume(center=mp.Vector3(),
                  size=mp.Vector3(sim.cell_size.x, 0, sim.cell_size.z)),
        Animate_xz,
    )),
    until=5,
)
```

The `mp.in_volume` wrapper restricts field output to the specified plane, so `Animate_xy` captures the XY midplane while `Animate_xz` captures the XZ midplane from the same run.

#### Key Takeaways

- Call `sim.plot2D()` before running to verify geometry placement, source positions, and monitor/PML configuration without incurring simulation cost.
- Pass `fields=mp.Ez` (or any field component) to overlay time-domain field data on the geometry plot; the simulation must have been run first.
- Use `output_plane=mp.Volume(center=..., size=...)` with one zero dimension to visualize a cross-section of a 3D simulation.
- `mp.Animate2D` captures frames during `sim.run()` via step-function callbacks and exports to MP4, GIF, or JSHTML; normalize with `normalize=True` when field amplitudes change significantly during the run.
- Wrap animation callbacks in `mp.in_volume(plane, callback)` to animate specific cross-sections of 3D simulations without running separate simulations for each plane.

---

### 7. `test_cavity_arrayslice.py` — Field Array Extraction from a Photonic Crystal Cavity

**Physics:** Photonic crystal cavity resonance in a slab waveguide with hole arrays creating a defect cavity; field extraction demonstrates the array slice infrastructure for capturing field distributions.
**Difficulty:** Intermediate
**Source:** `python/tests/test_cavity_arrayslice.py`
**Test Status:** PASS (27.3s)

#### Theory

The photonic crystal cavity in this test consists of a dielectric slab (epsilon=13, representing a semiconductor) with a row of cylindrical air holes. The geometry creates a photonic bandgap: the periodic hole array forbids propagation of certain frequencies within the slab plane. A defect — here the gap $d$ between the two innermost holes — breaks the periodicity and creates a localized resonant mode at a frequency within the bandgap. The Gaussian source at the origin excites this cavity mode, and the field distribution at the moment the source switches off (`until_after_sources=0`) shows the spatial structure of the mode still ringing in the cavity.

This test is placed in the Simulation Infrastructure chapter not because the physics is infrastructural, but because its primary purpose is to validate the `get_array` API — the mechanism by which field data is extracted from Meep's internal distributed arrays and returned as a NumPy array to Python. The physics of the cavity provides a meaningful, non-trivial field distribution that makes any errors in the array extraction visible.

The `get_array` mechanism works by specifying a spatial volume — center and size — and a field component. Meep's C++ layer iterates over the Yee grid cells within the specified volume, interpolates field values onto a uniform Cartesian grid, and copies the results into either a newly allocated NumPy array or a user-provided pre-allocated buffer. In MPI-parallel runs, each rank contributes the portion of the volume that falls within its chunk, and the results are assembled on rank 0.

Key design choices of the array extraction API include:

- **Real vs. complex output**: by default, `get_array` returns real-valued arrays (the instantaneous real part of the field). Passing `cmplx=True` returns complex arrays; this requires that `force_complex_fields=True` was set on the simulation.
- **User-provided buffers**: the `arr=` parameter allows the caller to pre-allocate a NumPy array of the correct shape and dtype, avoiding a memory allocation on each call. This is important for performance in loops that extract fields at every time step.
- **Shape validation**: if a user-provided array has the wrong size or wrong dtype (e.g., real array when `cmplx=True` was requested), `get_array` raises `ValueError` with a descriptive message.
- **Yee grid vs. interpolated**: the `get_array` call without special options returns values interpolated to the center of each output pixel. The `yee_grid=True` option (available on DFT monitors) returns values at the actual Yee staggered positions.

The test validates output against pre-computed reference arrays stored in `python/tests/data/cavity_arrayslice_1d.npy` and `cavity_arrayslice_2d.npy`. These references were generated by running the test geometry and saving the expected output. The numerical tolerance is $10^{-8}$ for double precision and $10^{-5}$ for single precision.

#### Code Walkthrough

The simulation geometry is a 2D photonic crystal slab cavity:

```python
r = 0.36    # hole radius
d = 1.4     # defect spacing (distance between innermost holes)
sy = 6      # cell height (transverse)
pad = 2     # padding between holes and PML
dpml = 1    # PML thickness

blk = mp.Block(size=mp.Vector3(mp.inf, 1.2, mp.inf),
               material=mp.Medium(epsilon=13))

geometry = [blk]
geometry.extend(mp.Cylinder(r, center=mp.Vector3(d/2 + i)) for i in range(3))
geometry.extend(mp.Cylinder(r, center=mp.Vector3(d/-2 - i)) for i in range(3))
```

Six air cylinders (three on each side, extending from $\pm d/2$ outward) interrupt the slab and form the mirror sections of the cavity. The defect at the center — the gap of width $d$ between the two innermost holes — localizes the resonant mode.

The source excites the Hz component (in-plane magnetic field) at the cavity center with a Gaussian pulse:

```python
sources = [mp.Source(mp.GaussianSource(0.25, fwidth=0.2), mp.Hz, mp.Vector3())]
```

Running `until_after_sources=0` means the simulation stops exactly when the source pulse has effectively decayed, at which point the dominant field is the cavity mode ringing down.

The 1D slice extracts a horizontal line through the cavity at y=0:

```python
vol = mp.Volume(center=self.center_1d, size=self.size_1d)
hl_slice1d = self.sim.get_array(mp.Hz, vol)
```

The volume `size_1d` has zero y and z components (`mp.Vector3(self.x_max - self.x_min)`), making it a 1D array along x covering the central quarter of the cell (from -sx/4 to +sx/4). The returned array has 126 elements (quarter of the 504-point x-grid).

The 2D slice extracts a rectangle of width x_max-x_min and height y_max-y_min:

```python
vol = mp.Volume(center=self.center_2d, size=self.size_2d)
hl_slice2d = self.sim.get_array(mp.Hz, vol)
```

The size `mp.Vector3(x_max-x_min, y_max-y_min)` with nonzero x and y produces a (126, 38) array.

The user-array tests allocate the output buffer explicitly before calling `get_array`:

```python
arr = np.zeros(126, dtype=np.float64)
self.sim.get_array(mp.Hz, vol, arr=arr)
```

The results match the auto-allocated version identically. The illegal-array tests confirm that passing a buffer of the wrong size or requesting complex output with a real-typed buffer raises `ValueError`:

```python
with self.assertRaises(ValueError):
    arr = np.zeros(128)              # wrong size (128 instead of 126)
    self.sim.get_array(mp.Hz, vol, arr=arr)

with self.assertRaises(ValueError):
    arr = np.zeros((126, 38))        # real buffer for complex request
    self.sim.get_array(mp.Hz, vol, cmplx=True, arr=arr)
```

#### Key Takeaways

- Use `sim.get_array(component, volume)` to extract time-domain field values as a NumPy array from any spatial subregion defined by an `mp.Volume`; the volume's zero-length dimensions determine whether the output is 1D, 2D, or 3D.
- Pre-allocate output buffers with the correct size and dtype and pass them via `arr=` to avoid repeated memory allocation in tight loops; the shape must exactly match the grid points within the volume.
- Pass `cmplx=True` (and set `force_complex_fields=True` on the simulation) to extract complex field arrays suitable for phase analysis or mode overlap calculations.
- Array shape validation is strict: incorrect size or dtype mismatch raises `ValueError` with a diagnostic message explaining the expected dimensions.
- The default output is real-valued (instantaneous field), interpolated to regular Cartesian positions; for Yee-grid-aligned output, use `yee_grid=True` on DFT monitor extraction instead.

---

### 8. `test_wvg_src.py` — Eigenmode Source Excitation and Flux Measurement Infrastructure

**Physics:** Waveguide mode excitation using an EigenModeSource in a 2D dielectric strip waveguide; flux measurements before and after the source validate that the EigenModeSource injects a pure waveguide mode propagating in one direction.
**Difficulty:** Intermediate
**Source:** `python/tests/test_wvg_src.py`
**Test Status:** PASS (7.0s)

#### Theory

Coupling light efficiently into a waveguide mode is one of the most common tasks in integrated photonics simulation. A point or plane-wave source injects into all modes simultaneously, including unguided radiation modes. For waveguide transmission calculations, reflectance measurements, and filter analysis, it is essential to inject only the desired waveguide mode with unit amplitude and defined propagation direction. Meep's `EigenModeSource` accomplishes this by calling the MPB eigenmode solver at initialization time to compute the transverse field profile of the requested mode and then uses that profile as the source amplitude function.

The `EigenModeSource` takes several parameters that control mode selection:

- `eig_parity`: restricts the search to modes of a given parity. `mp.ODD_Z` selects modes where Ez is odd under mirror reflection (TE-like modes in 2D), which excludes TM-like modes.
- `component=mp.Dielectric`: instructs the source to use the MPB-computed amplitude profile for the dielectric constant field as a proxy for the mode profile, which properly handles continuity across dielectric interfaces.
- `eig_band`: selects the mode band index (default 1 for the fundamental mode).

The geometry in this test is slightly more complex than a simple waveguide. A second narrow block of air (vacuum, `mp.Medium()`) is embedded inside the dielectric waveguide block, displacing the mode profile slightly. This ensures the test exercises mode solving in a non-trivial geometry rather than a uniform-material waveguide, where any reasonable source would accidentally excite the mode.

The flux measurement infrastructure validates the source by computing the power flux in boxes on either side of the source. The expected result is that essentially no power flows backward (toward negative x), while the power flowing forward (toward positive x) represents nearly all the injected power, confirming that the EigenModeSource successfully injects a unidirectional mode.

`flux_in_box` is a convenience method that integrates the Poynting vector over a rectangular region. It calls the same DFT machinery internally but returns a real number rather than a frequency-resolved spectrum. The result at a single CW frequency (ContinuousSource with no bandwidth) represents the steady-state time-averaged flux.

The test uses `force_complex_fields=True` on the simulation. This forces all field arrays to be stored as complex numbers even though the source is real-valued ContinuousSource. Complex fields are necessary here because the EigenModeSource injects a complex-amplitude mode profile, and the resulting steady-state fields are complex-valued. Without `force_complex_fields=True`, only the real part would be stored, losing phase information that the flux calculation depends on.

#### Code Walkthrough

The simulation geometry defines a dielectric waveguide with an internal air slot:

```python
geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, 1, mp.inf),
        material=mp.Medium(epsilon=12),
    ),
    mp.Block(
        center=mp.Vector3(y=0.3),
        size=mp.Vector3(mp.inf, 0.1, mp.inf),
        material=mp.Medium(),       # air slot
    ),
]
```

The air slot at y=0.3 with height 0.1 modifies the mode profile compared to a uniform slab, making it a more demanding test of the eigenmode solver.

The EigenModeSource is placed at x=-5, spanning the full waveguide cross-section in y:

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

The source size `mp.Vector3(y=6)` spans the entire 8-unit cell height (minus PML) in the y direction, ensuring the mode profile is accurately represented across the full cross-section. After running until `t=200` (long enough for transients to decay and steady state to establish):

```python
self.sim.run(until=200)

flux1 = self.sim.flux_in_box(
    mp.X,
    mp.Volume(center=mp.Vector3(-6.0), size=mp.Vector3(1.8, 6))
)
flux2 = self.sim.flux_in_box(
    mp.X,
    mp.Volume(center=mp.Vector3(6.0), size=mp.Vector3(1.8, 6))
)

self.assertAlmostEqual(flux1, -1.775216564842667e-03)
self.assertAlmostEqual(flux2, 7.215785537102116, places=5)
```

`flux_in_box(mp.X, vol)` integrates the x-component of the Poynting vector over the specified volume. A negative value for `flux1` at x=-6 means power flows in the -x direction at that location, which is the backward-propagating reflection from the source. The magnitude $|flux1| \approx 0.00178$ is tiny compared to the forward flux $flux2 \approx 7.22$, confirming better than 99.97% forward injection efficiency. The slight backward flux is a known artifact of the EigenModeSource implementation rather than a physical reflection.

The `places=5` tolerance for `flux2` (rather than the default 7) accommodates small numerical differences between single-precision and double-precision builds, since the eigenmode solve itself involves iterative numerics.

#### Key Takeaways

- `mp.EigenModeSource` calls MPB at simulation initialization to compute the waveguide mode profile; it requires MPB to be installed and the geometry to support bound modes at the requested frequency.
- Set `eig_parity=mp.ODD_Z` (or `mp.EVEN_Z`, `mp.ODD_Y`, etc.) to restrict mode selection to the desired symmetry class and avoid accidentally injecting the wrong mode.
- Always set `force_complex_fields=True` when using `EigenModeSource` with a ContinuousSource, because the mode profile and steady-state fields are inherently complex-valued.
- `sim.flux_in_box(direction, volume)` returns the steady-state time-averaged power flux in the specified direction through the specified volume; use it as a quick check of source injection quality by comparing backward flux to forward flux.
- A well-configured EigenModeSource should show backward flux several orders of magnitude smaller than forward flux; significant backward flux indicates a mode-matching problem (wrong parity, incorrect geometry, or frequency outside the guided band).

---

*End of Chapter 12: Simulation Infrastructure*
