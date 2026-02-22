# Meep Architecture Reference

Meep (MIT Electromagnetic Equation Propagation) is an open-source FDTD
(finite-difference time-domain) electromagnetics simulation engine developed at
MIT. It propagates electromagnetic fields through space and time on a
Yee-staggered grid, solving Maxwell's equations numerically. This document
describes the full architecture for developers who are new to the codebase.

Project home: https://github.com/NanoComp/meep

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [C++ Core Architecture](#2-c-core-architecture)
3. [Python Interface Architecture](#3-python-interface-architecture)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [Communication Flow Diagrams](#5-communication-flow-diagrams)
6. [Material System Architecture](#6-material-system-architecture)
7. [Build System Architecture](#7-build-system-architecture)
8. [Key File Reference](#8-key-file-reference)

---

## 1. System Overview

Meep has three distinct layers a user can interact with, all converging on a
single C++ core engine that performs the actual computation.

```
+------------------------------------------------------------------+
|                        USER API LAYER                            |
|                                                                  |
|   +------------------+  +------------------+  +--------------+  |
|   |   Python API     |  |   Scheme API     |  |  C++ API     |  |
|   |  simulation.py   |  |  scheme/*.scm    |  |  meep.hpp    |  |
|   |  geom.py         |  |  via libctl/     |  |  (direct)    |  |
|   |  source.py       |  |  Guile           |  |              |  |
|   |  materials.py    |  |                  |  |              |  |
|   |  adjoint/        |  |                  |  |              |  |
|   +--------+---------+  +--------+---------+  +------+-------+  |
|            |                     |                    |         |
+------------------------------------------------------------------+
             |                     |                    |
+------------------------------------------------------------------+
|                     BINDING LAYER                                |
|                                                                  |
|   +---------------------+        +---------------------------+  |
|   |    SWIG Bindings     |        |   libctl / ctlgeom        |  |
|   |  python/meep.i       |        |   (for Scheme interface)  |  |
|   |  python/numpy.i      |        |                           |  |
|   |  python/vec.i        |        |                           |  |
|   |  --> meep-python.cxx |        |                           |  |
|   |  --> _meep.so        |        |                           |  |
|   +----------+----------+        +-----------+---------------+  |
|              |                               |                   |
+------------------------------------------------------------------+
               |                               |
+------------------------------------------------------------------+
|                     C++ CORE ENGINE                              |
|                                                                  |
|  +-------------------+    +----------------------------------+   |
|  |   fields class     |    |   structure class                |   |
|  |   (time-stepper)   |    |   (material/geometry data)       |   |
|  +-------------------+    +----------------------------------+   |
|                                                                  |
|  +------------------+     +----------------------------------+   |
|  |  DFT monitors     |    |  susceptibility hierarchy         |   |
|  |  (flux, energy,   |    |  (Lorentzian, Drude, etc.)        |   |
|  |   force, near2far)|    |                                  |   |
|  +------------------+     +----------------------------------+   |
|                                                                  |
|  +------------------+     +----------------------------------+   |
|  |  Source injection |    |  Domain decomposition            |   |
|  |  (point, volume,  |    |  (structure_chunk per MPI rank,  |   |
|  |   eigenmode)      |    |   binary_partition tree)         |   |
|  +------------------+     +----------------------------------+   |
|                                                                  |
+------------------------------------------------------------------+
               |
+------------------------------------------------------------------+
|                     EXTERNAL LIBRARIES                           |
|                                                                  |
|  +--------+  +------+  +-------+  +------+  +-----+  +------+  |
|  | FFTW3  |  | GSL  |  | LAPACK|  | HDF5 |  | MPI |  | MPB  |  |
|  | (FFTs) |  |(math)|  |(linalg|  |(I/O) |  |(par)|  |(eig) |  |
|  +--------+  +------+  +-------+  +------+  +-----+  +------+  |
|                                                                  |
|  +----------+  +---------+  +----------+                        |
|  | harminv  |  | libctl  |  | libGDSII |                        |
|  | (filter  |  |(scheme  |  |(GDS file |                        |
|  |  diag.)  |  | interp) |  | import)  |                        |
|  +----------+  +---------+  +----------+                        |
|                                                                  |
+------------------------------------------------------------------+
```

### Units

Meep uses dimensionless units where the speed of light c = 1. The user picks a
unit length `a` (e.g., 1 micrometer). Frequencies are in units of c/a, time in
a/c, and the grid resolution is specified in grid points per unit length.

---

## 2. C++ Core Architecture

### 2.1 FDTD Time-Stepping Pipeline

The FDTD algorithm advances the electromagnetic field in discrete time steps.
Each step, Meep updates B/D (flux densities) first via curl operations, then
converts to H/E (field intensities) via the constitutive relations, and accounts
for dispersive polarization currents. The sequence within a single call to
`fields::step()` is:

```
fields::step()   [src/step.cpp, line 35]
  |
  |-- phase_material()          // handle mid-simulation material changes
  |-- update_condinv()          // refresh cached conductivity-inverse arrays
  |
  |-- [B-field half step]
  |     |-- calc_sources(t)         // evaluate source currents at time t
  |     |-- step_db(B_stuff)        // curl-H --> update B on each chunk
  |     |     |
  |     |     +--> [per chunk] step_generic()   [step_generic.cpp]
  |     |               Implements: dB/dt = curl(H) - conductivity*B
  |     |               With PML sigma stretching if in absorbing layer
  |     |
  |     |-- step_source(B_stuff)    // inject magnetic current sources into B
  |     |-- step_boundaries(B_stuff)// communicate ghost cells across MPI chunks
  |     |-- calc_sources(t + dt/2)
  |     |-- update_eh(H_stuff)      // H = chi1inv * (B - polarization_B)
  |     |     |                     [update_eh.cpp]
  |     |     +--> per chunk, per component: H[c] = chi1inv[c][d] * B[c]
  |     |
  |     |-- step_boundaries(WH, PH, H)  // sync auxiliary & H fields
  |     |-- update_pols(H_stuff)    // advance polarization P_H state
  |                                 // [update_pols.cpp / susceptibility.cpp]
  |
  |-- [D-field half step]
  |     |-- calc_sources(t + dt/2)
  |     |-- step_db(D_stuff)        // curl-E (previous) --> update D
  |     |-- step_source(D_stuff)    // inject electric current sources
  |     |-- step_boundaries(D_stuff)
  |     |-- calc_sources(t + dt)
  |     |-- update_eh(E_stuff)      // E = chi1inv * (D - polarization_E)
  |     |-- step_boundaries(WE, PE, E)
  |     |-- update_pols(E_stuff)
  |
  |-- t += 1                        // advance integer timestep counter
  |-- update_dfts()                 // accumulate DFT monitor data
  |-- NaN check on field energy
```

The actual numerical loop over grid points lives in `step_generic.cpp`.  The
function signature is:

```cpp
// Implements: f += dt * curl(g) with optional PML sigma stretching
// and optional conductivity damping.
void step_curl(realnum *f, component fc, const realnum *g1, const realnum *g2,
               ptrdiff_t s1, ptrdiff_t s2, const grid_volume &gv, double dtdx,
               direction dsig, const realnum *sig, const realnum *siginv, ...)
```

A second file, `step_generic_stride1.cpp`, is auto-generated from
`step_generic.cpp` by a `sed` transform at build time. It specializes the inner
loop for unit stride (stride-1) memory access, enabling the compiler to
auto-vectorize with SIMD instructions. Do not edit this file by hand.

### 2.2 Core Class Hierarchy

```
meep namespace
|
+-- fields                              [src/fields.cpp, src/meep.hpp:~1450]
|     The master object for a running simulation. Owns:
|     |
|     +-- fields_chunk* chunks[]        One chunk per decomposed sub-volume
|     |     |-- realnum* f[c][cmp]      Field arrays: E,H,D,B,P,W per component
|     |     |-- realnum* f_backup[c][cmp] Backup for CW solver
|     |     |-- dft_chunk* dft_chunks   Linked list of DFT accumulation regions
|     |     |-- src_vol* sources        Linked list of active sources on chunk
|     |     +-- structure_chunk* s      Material data for this chunk (owned by structure)
|     |
|     +-- structure* s                  Material/geometry description
|     +-- src_time* sources             Linked list of time dependences
|     +-- flux_vol* fluxes              Instantaneous Poynting flux monitors
|     +-- symmetry S                    Spatial symmetry being exploited
|     +-- int t                         Integer timestep counter
|     +-- double dt                     Timestep (Courant/a)
|     +-- int num_chunks                Total number of chunks
|     +-- comms_sequence[]              Pre-computed MPI communication schedule
|
+-- structure                           [src/structure.cpp, src/meep.hpp:~809]
|     Describes the material and geometry. Immutable during a run (unless
|     material phasing is used). Owns:
|     |
|     +-- structure_chunk* chunks[]     One per MPI rank sub-volume
|     |     |-- realnum* chi1inv[c][d]  1/(epsilon) tensor (inverse permittivity)
|     |     |-- realnum* chi3[c]        Kerr nonlinearity chi_3
|     |     |-- realnum* chi2[c]        Second-order nonlinearity chi_2
|     |     |-- realnum* conductivity   Ohmic conductivity
|     |     |-- realnum* condinv        Cached 1/(1 + sigma*dt/2)
|     |     |-- realnum* sig[6]         PML sigma arrays (one per direction-side)
|     |     |-- realnum* kap[6]         PML kappa stretching coordinate
|     |     |-- susceptibility* chiP[]  Linked list of dispersive terms for E/H
|     |     +-- grid_volume gv          Integer grid volume of this chunk
|     |
|     +-- binary_partition* bp          The chunk partitioning tree
|     +-- grid_volume gv                Full simulation volume
|     +-- symmetry S
|
+-- susceptibility  (abstract base)     [src/susceptibility.cpp, src/meep.hpp:~79]
|     Represents a polarizability contribution. Forms a linked list:
|     chiP[E_stuff] -> sus1 -> sus2 -> NULL
|     |
|     +-- lorentzian_susceptibility     chi(w) = sigma*w0^2/(w0^2 - w^2 - i*gamma*w)
|     |     (no_omega_0_denominator=true => Drude model)
|     |
|     +-- noisy_lorentzian_susceptibility  Lorentzian + white noise (for LDOS)
|     |
|     +-- gyrotropic_susceptibility     Off-diagonal chi tensor (magnetics/plasma)
|     |     Models: GYROTROPIC_LORENTZIAN, GYROTROPIC_DRUDE, GYROTROPIC_SATURATED
|     |
|     +-- multilevel_susceptibility     Full rate-equation laser gain medium
|           L levels, T optical transitions, population dynamics
|
+-- src_time  (abstract base)           [src/meep.hpp:~934]
|     Time dependence of a current source.
|     |
|     +-- gaussian_src_time            Gaussian envelope: exp(-(t-t0)^2/2w^2)*cos(2pi*f*t)
|     +-- continuous_src_time          CW: cos(2pi*f*t) with optional slow turn-on
|     +-- custom_src_time              Arbitrary C function pointer
|     +-- custom_py_src_time           Python callable (python/meep-python.hpp)
|
+-- DFT monitor classes                 [src/dft.cpp, src/meep.hpp:~1124]
|     |
|     +-- dft_chunk                    Per-chunk DFT accumulator
|     |     complex<realnum>* dft      N_spatial x N_freq array of accumulated values
|     |     vector<double> omega       Angular frequencies being accumulated
|     |     void update_dft(double t)  Called every timestep from update_dfts()
|     |
|     +-- dft_flux                     Poynting flux integral: Re[E x H*] . n dA
|     +-- dft_energy                   Field energy integral: integral(E.D + H.B)
|     +-- dft_force                    Maxwell stress tensor integral
|     +-- dft_near2far                 Near-to-far-field transformation
|     +-- dft_fields                   Raw field values at specified frequencies
|     +-- dft_ldos                     Local density of states: P(w) = Re[E.J*]
|
+-- boundary_region                     [src/meep.hpp:~650]
|     Describes a PML absorbing boundary layer. Linked list of regions, one
|     per direction/side combination.
|
+-- material_function  (abstract)       [src/meep.hpp:~481]
      User-supplied callback for position-dependent material properties.
      Methods: eps(r), mu(r), conductivity(c,r), sigma_row(c,sigrow,r),
               chi2(c,r), chi3(c,r), eff_chi1inv_row(c,chi1inv_row,v,tol,maxeval)
```

### 2.3 Domain Decomposition

Meep divides the simulation volume into rectangular sub-volumes called chunks.
Each MPI rank owns one or more chunks. The decomposition is stored as a binary
tree (a k-d tree in disguise).

```
Full grid_volume gv
|
+-- binary_partition tree               [src/structure.cpp]
      Recursively bisects the volume along X, Y, or Z axes.
      Leaf nodes correspond to individual chunks.

      Example (2 ranks, split along X):
      +-------------------------------+
      |              |                |
      |  chunk 0     |   chunk 1      |
      |  (rank 0)    |   (rank 1)     |
      |              |                |
      +-------------------------------+
                     ^
                  bisection plane

      Example (4 ranks):
      +-------------------------------+
      |       |        |       |      |
      | c0    |  c1    |  c2   |  c3  |
      | r0    |  r1    |  r2   |  r3  |
      +-------------------------------+
```

Each `fields_chunk` overlaps its neighbors by one grid cell (the "ghost" or
"not-owned" region). Boundary communication copies field values from owned
regions of neighboring chunks into the ghost cells before each sub-step.

```
Chunk 0                   Chunk 1
+--------------------+    +--------------------+
| owned  | ghost     |    | ghost   | owned    |
| cells  | (copy of  |    | (copy   | cells    |
|        | chunk 1's |    |  of c0) |          |
|        | boundary) |    |         |          |
+--------------------+    +--------------------+
         <-- MPI send ------------>
         <-------------- MPI send -->
```

The inter-chunk communication is pre-computed once in `connect_chunks()` and
stored as a `comms_sequence` (ordered list of send/receive operations) for each
field type. During each timestep, `step_boundaries(field_type)` executes this
pre-computed sequence using asynchronous MPI sends and receives managed by the
`comms_manager` abstraction.

---

## 3. Python Interface Architecture

### 3.1 SWIG Binding Chain

SWIG (Simplified Wrapper and Interface Generator) reads annotated C++ headers
and generates a C extension module that Python can import.

```
Source inputs:
  python/meep.i        (~65K lines)  -- SWIG interface file; declares all
  |                                     wrapped classes/functions; contains
  |                                     %typemaps for Python<->C++ conversions
  |-- python/numpy.i                 -- NumPy array typemap helpers
  |-- python/vec.i                   -- vec/ivec/volume typemap helpers
  |-- src/meep.hpp                   -- C++ class definitions
  |-- src/meep/vec.hpp               -- Component enums, grid_volume
  |-- src/meepgeom.hpp               -- Geometry structures
  +-- libpympb/pympb.hpp             -- MPB eigenmode solver wrapper

SWIG processes these to generate:
  python/meep-python.cxx             -- C++ wrapper code (~200K lines)
  python/meep.py                     -- Python shadow classes

Compiled with the C++ core to produce:
  python/_meep.so                    -- Loadable Python extension module

Python import chain:
  import meep as mp
  |-- python/__init__.py             -- re-exports public API
  |-- python/meep.py                 -- SWIG shadow module (auto-generated)
  +-- python/_meep.so                -- C extension (contains real C++ objects)
```

The file `python/meep-python.hpp` contains C++ helper classes and functions
that the SWIG interface uses but that are not part of the public `meep.hpp` API.
For example, `custom_py_src_time` (a subclass of `src_time` that calls a Python
callable) is defined there so that Python functions can be passed as source time
dependences directly.

### 3.2 High-Level Python Modules

```
python/
|
+-- simulation.py          -- Simulation class (the main user-facing object)
|     class Simulation:
|       __init__(cell_size, resolution, geometry, sources, ...)
|       init_sim()         -- builds C++ structure + fields objects
|       run(*step_funcs, until=T)     -- main time integration loop
|       add_flux(...)      -- create DFT flux monitor (returns DftFlux)
|       add_force(...)     -- create DFT force monitor
|       add_near2far(...)  -- create DFT near2far monitor
|       get_array_slice()  -- extract field slice as NumPy array
|       get_dft_array()    -- extract DFT data as NumPy array
|
+-- geom.py                -- Geometry and material objects
|     class Vector3         -- 3-component vector with arithmetic
|     class Medium          -- Material specification
|     |  epsilon_diag       -- diagonal permittivity tensor
|     |  mu_diag            -- diagonal permeability tensor
|     |  E_susceptibilities -- list of Susceptibility objects
|     |  H_susceptibilities
|     |  E_chi2_diag / E_chi3_diag  -- nonlinear coefficients
|     class LorentzianSusceptibility
|     class DrudeSusceptibility      (Lorentzian with no_omega_0_denominator)
|     class NoisyLorentzianSusceptibility
|     class GyrotropicLorentzianSusceptibility
|     class MultilevelAtom
|     class GeometricObject (abstract)
|       +-- Sphere, Cylinder, Cone, Block, Ellipsoid, Prism, ...
|
+-- source.py              -- Source class definitions
|     class Source          -- base class; wraps C++ add_volume_source
|     class EigenModeSource -- calls MPB to compute waveguide modes
|     class GaussianBeamSource -- Gaussian beam injection
|
+-- materials.py           -- Predefined material library
|     cSi, aSi, SiO2, Au, Ag, Al, Cu, ... (broadband Lorentzian fits)
|     Each defined as: mp.Medium(epsilon=..., E_susceptibilities=[...])
|
+-- visualization.py       -- matplotlib helpers
|     plot2D(), plot_eps(), animate_simulation(), ...
|
+-- adjoint/               -- Adjoint solver for inverse design
|     optimization_problem.py -- OptimizationProblem class
|     filters.py           -- spatial filter operations
|     utils.py             -- adjoint helper utilities
|     wrapper.py           -- MeepJaxWrapper: JAX-differentiable callable
```

### 3.3 Adjoint Solver Module

The adjoint module enables gradient-based inverse design by computing the
gradient of an objective function with respect to design parameters. It wraps
ordinary Meep simulations to make them differentiable via JAX.

```
python/adjoint/
|
+-- optimization_problem.py
|     class OptimizationProblem:
|       __call__(x)  -- evaluates objective f(x) and gradient df/dx
|       _run_forward_simulation()   -- standard Meep run, records DFT fields
|       _run_adjoint_simulation()   -- re-run with adjoint sources derived from
|                                      grad of objective w.r.t. monitor values
|
+-- wrapper.py
|     class MeepJaxWrapper:
|       -- Wraps OptimizationProblem into a JAX-compatible function
|       -- Implements custom_vjp so JAX can call forward+adjoint
|       __call__(designs) -> monitor_values  (shape: [n_monitors, n_freqs])
|       -- design variables are JAX arrays (density maps for material grids)
|
+-- utils.py
|     create_adjoint_sources()  -- convert objective gradient -> source currents
|     register_monitors()       -- set up mode overlap monitors
|     install_design_region_monitors()  -- DFT fields in design regions
|     gather_monitor_values()   -- collect DFT data after forward run
|
+-- filters.py
      conic_filter(), cylindrical_filter(), etc.  -- density filter operations
      tanh_projection()   -- threshold projection of density
```

---

## 4. Data Flow Diagrams

### 4.1 Simulation Lifecycle: Python to Output

```
User code
  |
  | sim = mp.Simulation(cell_size=..., resolution=...,
  |         geometry=[mp.Block(...)], sources=[mp.Source(...)])
  v
Simulation.__init__()                  [python/simulation.py:1226]
  Stores all arguments as Python attributes.
  Does NOT call into C++ yet.
  |
  | sim.run(mp.at_every(1, mp.output_efield_z), until=200)
  v
Simulation.run()                       [python/simulation.py]
  |-- if not initialized: self.init_sim()
  |                           |
  v                           v
Simulation.init_sim()          [python/simulation.py:2453]
  |
  |-- _init_structure(k_point)
  |     |-- build grid_volume gv from cell_size + resolution
  |     |-- build boundary_region br from boundary_layers (PML specs)
  |     |-- build symmetry sym
  |     |-- mp.create_structure(gv, br, sym, geometry, ...)
  |     |         |
  |     |         v
  |     |   meep_geom::set_materials()        [src/meepgeom.cpp]
  |     |     -- Iterates over geometric objects in Python list
  |     |     -- For each grid point, finds the highest-priority object
  |     |     -- Calls eff_chi1inv_row() for subpixel averaging
  |     |     -- Populates structure_chunk::chi1inv arrays
  |     |     -- Adds susceptibilities to structure_chunk::chiP
  |     |
  |     +-> self.structure  (C++ meep::structure object, wrapped by SWIG)
  |
  |-- self.fields = mp.fields(self.structure, ...)
  |         Creates meep::fields, allocates field arrays per chunk
  |
  |-- add_sources()
  |     -- For each Source in self.sources:
  |          if EigenModeSource:
  |            call MPB via fields::get_eigenmode()   [src/mpb.cpp]
  |            then fields::add_eigenmode_source()
  |          else:
  |            fields::add_volume_source() or add_point_source()
  |
  +-> self.fields and self.structure initialized
  |
  v
Simulation.run() continues:
  |-- while time < T:
  |       for func in step_functions:    // e.g. at_every(1, output_efield_z)
  |           func(sim)                  // Python callback, may output HDF5
  |       self.fields.step()            // single FDTD timestep (C++)
  |
  v
Output
  -- HDF5 files via fields::output_hdf5()     [src/h5file.cpp]
  -- NumPy arrays via fields::get_array_slice() [src/array_slice.cpp]
  -- DFT data via dft_flux::flux() etc.
```

### 4.2 DFT Monitor Data Collection

DFT (discrete Fourier transform) monitors accumulate frequency-domain field
data in-situ during the time-stepping loop, without storing the full
time-domain field history.

```
Setup (before run):
  sim.add_flux(fcen, df, nfreq, FluxRegion(center=..., size=...))
      |
      v
  fields::add_dft_flux()               [src/dft.cpp]
      -- Calls fields::add_dft() for each field component (Ex, Hy, etc.)
      -- fields::add_dft() calls loop_in_chunks() to create one
         dft_chunk per fields_chunk that overlaps the monitor region
      -- Each dft_chunk allocates: complex<realnum>* dft[N_spatial x N_freq]
      -- Precomputes: complex<realnum>* dft_phase[N_freq]
                      = exp(i * omega * dt) for each frequency

During each timestep (inside fields::step()):
  fields::update_dfts()                [src/step.cpp:126]
      |
      +-- for each dft_chunk dc in all chunks:
            dc.update_dft(time)        [src/dft.cpp]
              |
              for each frequency iw:
                phase = dft_phase[iw]  // exp(i*omega*t), precomputed scale
                for each spatial point ix:
                  dft[ix*Nfreq + iw] += phase * f[component][ix]
                  // f[component][ix] is the live field array in fields_chunk

After run:
  flux = sim.get_flux_data()
      |
      v
  dft_flux::flux()                     [src/dft.cpp]
      -- Iterates dft_chunk E and H linked lists
      -- Computes: flux[iw] = Re( integral(E_dft[iw] x H_dft[iw]*) . n dA )
      -- MPI reduction: sum over all ranks
      -- Returns Python list of floats (one per frequency)
```

The "decimation factor" feature lets DFT monitors skip timesteps (accumulate
every N steps instead of every step) for high-frequency signals, trading
accuracy for speed.

### 4.3 Eigenmode Source Flow (MPB Integration)

Eigenmode sources launch a specific waveguide mode into the simulation.

```
Python:
  src = mp.EigenModeSource(src=mp.GaussianSource(fcen=1.0, fwidth=0.2),
                           center=..., size=..., eig_band=1)
  sim = mp.Simulation(..., sources=[src])
  sim.init_sim()
      |
      v
simulation.py: _add_eigenmode_source()
      |
      v
fields::add_eigenmode_source()         [src/mpb.cpp]
      |
      |-- fields::get_eigenmode()
      |       -- Instantiates mode_solver from libpympb [libpympb/pympb.cpp]
      |       -- Builds an MPB grid matching the cross-section of the
      |          MEEP structure at the source plane
      |       -- Runs MPB iterative eigensolver (ARPACK/LAPACK/FFTW)
      |       -- Returns eigenvector data: mode profile E(r), H(r)
      |          and eigenvalue kz (propagation constant)
      |
      |-- Interpolates MPB mode fields onto Meep Yee grid
      |   (different grid spacings, staggered vs. collocated)
      |
      +-- fields::add_volume_source() for each field component
            -- Amplitude function A(r) = mode_profile(r) * amplitude
            -- Source injects mode-matched currents to launch the eigenmode
               and simultaneously suppress the backward-propagating mode

During time-stepping:
  The injected currents create a forward-propagating mode matched to MPB's
  eigenmode solution. The total-field/scattered-field technique is NOT used;
  instead a current sheet (Huygens surface) excites the mode directly.
```

### 4.4 Adjoint Solver Forward + Adjoint Passes

The adjoint method computes gradients at the cost of two simulations
(one forward, one adjoint) regardless of the number of design parameters.

```
User objective:  f(x) = |monitor_value(x)|^2
                 where x is the material density in a design region

Forward pass:
  MeepJaxWrapper._run_fwd_simulation(x)   [python/adjoint/wrapper.py:140]
      |
      |-- Set material in design region:
      |   structure_chunk::chi1inv arrays updated from x via MaterialGrid
      |
      |-- Standard Meep run:
      |   sim.run(until=stop_when_dft_decayed(...))
      |
      |-- Record DFT fields in design region:
      |   fwd_fields[r,iw] = E_dft(r, omega_iw)  (complex arrays)
      |
      +-- Record monitor values:
          S_fwd = eigenmode_overlap(dft_flux, mode_data)

Gradient computation:
  JAX calls backward through MeepJaxWrapper via custom_vjp:
      |
      |-- Compute adjoint source amplitudes from output gradient:
      |   dL/dS_fwd determines amplitude at each monitor frequency
      |   J_adj(r, w) = dL/dS_fwd * mode_field(r, w)  [utils.py]
      |
      v
Adjoint pass:
  MeepJaxWrapper._run_adjoint_simulation()  [python/adjoint/wrapper.py:166]
      |
      |-- New simulation with adjoint sources J_adj (at monitor locations)
      |-- Standard Meep run (same structure, reversed time is NOT needed
      |   for linear problems due to reciprocity)
      |
      |-- Record adjoint DFT fields in design region:
      |   adj_fields[r,iw] = E_adj_dft(r, omega_iw)
      |
      v
Gradient assembly:
  dL/dx[r] = -2 * Re( sum_iw [ adj_fields[r,iw] . fwd_fields[r,iw]
                                * d(chi1inv)/d(x) ] )
  |
  -- This is the sensitivity of the objective to the material density
     at each pixel in the design region.
  |
  +-> Returned as a JAX array, enabling JAX optimizer (optax, scipy, etc.)
```

---

## 5. Communication Flow Diagrams

### 5.1 MPI Parallel Communication Between Chunks

```
  Rank 0                                   Rank 1
  +----------------------------+            +----------------------------+
  | fields_chunk 0             |            | fields_chunk 1             |
  | owns: x in [0, Lx/2]      |            | owns: x in [Lx/2, Lx]     |
  |                            |            |                            |
  | f[Ex][owned] [ghost]       |            | [ghost] f[Ex][owned]       |
  |    [........][.....]       |            | [.....][................]   |
  |                   |        |            |    ^                        |
  |                   |        |            |    |                        |
  +----------------------------+            +----------------------------+
                      |                         |
                      |  MPI_Isend (async)       |
                      +------------------------->|  copy into ghost cells
                      |<-------------------------+
                      |  MPI_Isend (async)       |

The communication schedule is:
  1. Before step_db(B_stuff):
       fields::step_boundaries(B_stuff)
         -- For each comms_operation in comms_sequence_for_field[B_stuff]:
              if send: pack comm_blocks[ft][pair] and MPI_Isend
              if recv: MPI_Irecv, on completion copy into f_plus_notowned
         -- comms_manager destructor: MPI_Waitall

  2. Same pattern for D_stuff, H_stuff, E_stuff, WE_stuff, WH_stuff,
     PE_stuff, PH_stuff (polarization auxiliary fields).

The communication schedule (comms_sequence) is computed once in
fields::connect_chunks() and reused every timestep. It sorts sends by
decreasing payload size (largest first) to maximize overlap with computation.

For OpenMP builds:
  -- The CHUNK_OPENMP macro parallelizes the per-chunk loops
  -- Each chunk is an independent OpenMP task
  -- No intra-process MPI needed between OpenMP threads on same rank
```

### 5.2 Python to C++ Data Flow Through SWIG

```
Python call:
  sim.get_array_slice(vol, mp.Ez)
         |
         | (via SWIG-generated wrapper in meep.py)
         v
meep::fields::get_array_slice()        [src/array_slice.cpp]
         |
         |-- Calls loop_in_chunks() with a callback that:
         |     -- On each chunk owned by this rank:
         |          interpolates f[Ez][ix] to requested coordinates
         |          writes into a raw C++ realnum[] buffer
         |     -- On each chunk NOT owned by this rank:
         |          MPI_Reduce / MPI_Allreduce collects partial results
         |
         |-- Returns realnum* (C array)
         |
         v
SWIG typemap in meep.i:
  %typemap(out) realnum* get_array_slice
      -- wraps the returned pointer as a NumPy array
      -- sets np.float32 or np.float64 dtype depending on MEEP_SINGLE
      -- uses PyArray_SimpleNewFromData (zero-copy if possible)
      -- sets base object so NumPy holds a reference keeping C buffer alive
         |
         v
Python receives: np.ndarray (2D or 3D, dtype=float64)

Reverse direction (Python array -> C++):
  fields::add_volume_source(component, src, vol, complex_arr, dim1, dim2, dim3, amp)
         |
         | SWIG typemap converts np.ndarray -> complex<double>* with dims
         v
  src/sources.cpp: copies amplitude array into src_vol structure,
  which is stored per-chunk and used during step_source() calls.
```

### 5.3 Field Data Flow from C++ to NumPy

```
C++ fields_chunk (on MPI rank 0):         C++ fields_chunk (on MPI rank 1):
  realnum* f[Ez][0]                          realnum* f[Ez][0]
  [e0, e1, e2, e3, e4, e5, e6, e7]          [e8, e9, e10, e11, ...]
      |                                          |
      | (MPI_Reduce or loop_in_chunks)           |
      +-------------------------------------------+
                        |
                        v
            Master rank: raw C realnum[] buffer
            [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, ...]
                        |
                        | SWIG %typemap(out) or numpy.i conversion
                        v
            Python: numpy.ndarray (shape=(Nx, Ny, Nz), dtype=float64)
            -- Memory is shared (zero-copy) when buffer is already contiguous
            -- A copy is made when interpolation or reordering is needed

For DFT arrays (complex-valued):
  complex<realnum>* dft[N_spatial * N_freq]
      |
      | MPI_Allreduce sums contributions from all ranks (each chunk knows
      | only its own spatial points)
      |
      v
  get_dft_array() returns complex<realnum>*
      |
      | SWIG typemap wraps as numpy.ndarray(dtype=complex128 or complex64)
      v
  Python: array shape (N_freq, Nx, Ny, Nz) or (Nx, Ny, N_freq) depending
          on monitor type
```

---

## 6. Material System Architecture

### 6.1 The Susceptibility Class Hierarchy

Meep models dispersive materials using a sum of polarizability terms. Each term
is a subclass of `susceptibility` and lives in a singly-linked list
(`chiP[E_stuff]` or `chiP[H_stuff]`) hanging off each `structure_chunk`.

```
susceptibility  (abstract)             [src/meep.hpp:79, src/susceptibility.cpp]
|
|  Key interface methods:
|    virtual void update_P(W, W_prev, dt, gv, P_data)
|      -- Advance the polarization P by one timestep given the field W (E or H)
|    virtual void subtract_P(ft, f_minus_p, P_data)
|      -- Compute D - P (used to update E = chi1inv * (D - P_E))
|    virtual void* new_internal_data(W, gv)
|      -- Allocate per-chunk storage for P and any auxiliary state
|    virtual bool needs_W_notowned(c, W)
|      -- If true, ghost cells of W must be communicated before update_P
|
+-- lorentzian_susceptibility          [src/meep.hpp:246]
|     chi(w) = sigma * w0^2 / (w0^2 - w^2 - i*gamma*w)
|     Auxiliary variable method: stores P[n] and P[n-1] per grid point.
|     Update equation (discretized ADE):
|       P[n+1] = (2 - w0^2*dt^2) * P[n] - P[n-1] + sigma*w0^2*dt^2 * E[n]
|                  all divided by (1 + gamma*dt/2)
|     Parameters: omega_0 (resonance freq), gamma (damping), sigma (strength)
|     If no_omega_0_denominator==true: Drude model (free electrons)
|       chi(w) = sigma * w0^2 / (-w^2 - i*gamma*w)
|
+-- noisy_lorentzian_susceptibility    [src/meep.hpp:282]
|     Same as Lorentzian but adds white Gaussian noise to P at each step.
|     Used to model thermal fluctuations for LDOS calculations.
|     Extra parameter: noise_amp
|
+-- gyrotropic_susceptibility          [src/meep.hpp:304]
|     For magnetically biased materials (ferrites, magnetized plasma).
|     chi is an antisymmetric (off-diagonal) tensor, set by a bias vec B0.
|     Models:
|       GYROTROPIC_LORENTZIAN  -- gyromagnetic resonance
|       GYROTROPIC_DRUDE       -- magnetized free electrons
|       GYROTROPIC_SATURATED   -- saturated ferrite (Polder tensor)
|     Needs W_notowned because the vector cross-product mixes components
|     that may span chunk boundaries.
|
+-- multilevel_susceptibility          [src/meep.hpp:340]
      Full population-inversion laser gain model.
      L levels (e.g., 3 or 4 for standard laser systems).
      T optical transitions between level pairs.
      State variables per grid point:
        N[i]  : population of level i  (L values)
        P[t]  : polarization for transition t  (T complex values)
      Rate equations:
        dN_i/dt = sum_j(Gamma_ji * N_j) - sum_j(Gamma_ij * N_i)
                  + sum_t(alpha_{it} * omega_t * Im[E . dP_t/dt])
        dP_t/dt = -gamma_t * P_t - omega_t^2 * sigma_t * E * N_t_upper
      Has_nonlinearities() returns true; requires special treatment in
      update_pols() for the nonlinear E.J coupling.
```

### 6.2 How Python Materials Map to C++ Susceptibilities

```
Python (geom.py / materials.py):
  mp.Medium(
      epsilon=11.7,                    -- background epsilon (chi1 = epsilon - 1)
      E_susceptibilities=[
          mp.LorentzianSusceptibility(
              frequency=3.64,          -- omega_0 / (2*pi)  in units c/a
              gamma=0.0,               -- damping rate
              sigma=8.0,               -- dimensionless strength
          ),
          mp.DrudeSusceptibility(
              frequency=1e-10,         -- effectively 0 for Drude
              gamma=0.002,
              sigma=1.0,
          ),
      ]
  )
  |
  | When sim.init_sim() calls mp.create_structure() / mp._set_materials():
  v
meepgeom.cpp: meep_geom::make_material_data()
  -- Converts mp.Medium to a meep_geom::material_data C struct
  -- Iterates geometry objects; for each grid point finds the enclosing
     object's material
  -- Calls structure::add_susceptibility() for each E/H susceptibility term

structure::add_susceptibility()        [src/structure.cpp]
  -- Calls structure_chunk::add_susceptibility() on each relevant chunk
  -- The susceptibility prototype is cloned and appended to chiP[E_stuff]
     or chiP[H_stuff] linked list on each chunk

materials.py example (crystalline silicon):
  cSi_susc = [
      mp.LorentzianSusceptibility(frequency=3.64, gamma=0,    sigma=8),
      mp.LorentzianSusceptibility(frequency=2.76, gamma=0.126, sigma=2.85),
      mp.LorentzianSusceptibility(frequency=1.73, gamma=5.0,  sigma=-0.107),
  ]
  cSi = mp.Medium(epsilon=1.0, E_susceptibilities=cSi_susc)
  -- Sum of 3 Lorentzians fitted to experimental n(lambda) data
  -- The background epsilon=1.0 means the Lorentzians carry all of n^2
```

### 6.3 PML Absorbing Boundaries

PML (Perfectly Matched Layer) is implemented as a complex coordinate stretch:

```
Standard curl equations:      dB/dt = curl(H)
PML-modified equations:       dB/dt = (1/kap) * curl(H) - (sigma/kap) * B
                                      + auxiliary field u satisfying
                                      du/dt = curl(H) - sigma_u * u

The sigma and kappa arrays (realnum* sig[6] and kap[6] in structure_chunk)
store the PML parameters. sig[2*d + side] gives the profile for direction d,
boundary side (Low=1, High=0).

Default profile (quadratic):
  sigma(x) = sigma_max * (x/d_pml)^2
  where d_pml is the PML thickness and sigma_max is chosen such that the
  theoretical reflectance at the far edge is Rasymptotic (default 1e-15).

The step_generic() inner loop checks if dsig != NO_DIRECTION to activate
the PML update path versus the standard curl update.

PML setup path:
  Python: boundary_layers=[mp.PML(thickness=1.0)]
      |
      v
  simulation.py: _init_structure() builds boundary_region br
  boundary_region::apply(structure*)   [src/boundaries.cpp]
  structure_chunk::use_pml()           [src/structure.cpp]
    -- Fills sig[], kap[], siginv[] arrays with the quadratic profile
    -- PML regions are created as additional structure_chunks that overlap
       the physical domain boundary
```

---

## 7. Build System Architecture

### 7.1 Autotools Overview

```
configure.ac                           [root: configure.ac]
  |-- AC_INIT(meep, version)
  |-- Checks for: CC, CXX, MPI, SWIG, HDF5, FFTW3, GSL, LAPACK,
  |               harminv, libctl, MPB, libGDSII
  |-- AM_CONDITIONALS: HAVE_MPI, WITH_OPENMP, WITH_SCHEME, etc.
  |-- Generates: config.h, Makefile (from Makefile.am)
  |
  +-> sh autogen.sh   (run once to generate ./configure from configure.ac)
  +-> ./configure ... (run to configure for specific system)

Makefile.am hierarchy:
  Makefile.am  (root)
  |-- src/Makefile.am            -- builds libmeep.la
  |-- python/Makefile.am         -- builds _meep.so, runs Python tests
  |-- libpympb/Makefile.am       -- builds libpympb.la (MPB Python wrapper)
  |-- tests/Makefile.am          -- builds C++ test executables
  |-- scheme/Makefile.am         -- builds Scheme interface (if --with-libctl)
  +-- doc/Makefile.am

Build products:
  src/.libs/libmeep.so.*         -- Shared library (SHARED_VERSION_INFO=36:0:1)
  src/.libs/libmeep.la           -- Libtool wrapper
  python/_meep.so                -- Python C extension
  python/meep-python.cxx         -- SWIG-generated C++ (only in maintainer mode)
  tests/bench                    -- C++ benchmark executable
  tests/harmonics, tests/cyl_1d, tests/symmetry, ...  (test executables)
```

### 7.2 What Gets Built and How

```
Step 1: SWIG code generation (maintainer mode only)
  SWIG python/meep.i + python/numpy.i + python/vec.i
    --> python/meep-python.cxx   (C++ SWIG wrapper)
    --> python/meep.py           (Python shadow module)
  Note: Release tarballs ship pre-generated meep-python.cxx.
        Developers need SWIG installed; users building from tarball do not.

Step 2: C++ core library
  Compile: src/*.cpp + libpympb/pympb.cpp
    src/step.cpp
    src/step_generic.cpp
    src/step_generic_stride1.cpp   (auto-generated from step_generic.cpp via sed)
    src/step_db.cpp
    src/update_eh.cpp
    src/update_pols.cpp
    src/boundaries.cpp
    src/fields.cpp, fields_dump.cpp
    src/structure.cpp, structure_dump.cpp
    src/dft.cpp
    src/sources.cpp
    src/susceptibility.cpp
    src/meepgeom.cpp
    src/near2far.cpp
    src/array_slice.cpp
    src/mympi.cpp
    src/h5file.cpp
    src/cw_fields.cpp
    ... (20+ files total)
  Link: -lfftw3 -lgsl -lhdf5 -lharminv -llapack -lmpb (optional) -lmpi (optional)
  Output: src/.libs/libmeep.so (versioned shared library)

Step 3: Python extension module
  Compile: python/meep-python.cxx
    CXXFLAGS include -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)
  Link: _meep.so against libmeep.so + libpympb.so
  Output: python/_meep.so

Step 4: Tests
  make check
    -- C++ tests: make in tests/, run each executable
    -- Python tests: cd python && python -m pytest tests/test_*.py
       (or equivalent make check target)
  Parallelism:
    -- MPI tests prepend: mpirun -np 2
    -- OpenMP tests prepend: env OMP_NUM_THREADS=2
    -- Controlled by RUNCODE variable in tests/Makefile.am

Step 5: Distcheck (CI)
  make distcheck DISTCHECK_CONFIGURE_FLAGS="--with-libctl=..."
    -- Creates a distribution tarball
    -- Unpacks into a clean directory
    -- Configures, builds, and runs tests
    -- Verifies no files were missed in the tarball
```

### 7.3 Key Configure Flags and Their Effects

```
Flag                     What it does
--enable-maintainer-mode Regenerates meep-python.cxx via SWIG (needs swig installed)
--with-mpi               Enables MPI parallelism (sets CXX=$MPICXX)
--enable-single          Uses float instead of double for field arrays (2x faster)
--with-openmp            Enables OpenMP chunk-level parallelism
--enable-debug           Adds -g -O0, enables extra assertions
--with-libctl=DIR        Enables Scheme interface (requires Guile + libctl)
--with-coverage          Enables gcov for Python test coverage measurement
--prefix=DIR             Installation prefix (headers, libs, python module)
```

---

## 8. Key File Reference

### 8.1 C++ Core (`src/`)

| File | Lines | Purpose |
|------|-------|---------|
| `meep.hpp` | ~2500 | Main public API header. All core class declarations: `fields`, `structure`, `susceptibility` hierarchy, DFT monitors, `src_time` hierarchy, `boundary_region`, `material_function`. The single source of truth for the C++ API. |
| `meep/vec.hpp` | ~600 | Vector math: `vec`, `ivec`, `grid_volume`, `volume`. Component enum (`Ex`, `Ey`, ..., `Bz`), direction enum (`X`, `Y`, `Z`, `R`, `P`), field_type enum (`E_stuff`, `H_stuff`, etc.). |
| `meep_internals.hpp` | ~300 | Implementation-only declarations not exposed to users: internal step function signatures, loop macros, timing infrastructure. |
| `meepgeom.hpp` | ~200 | C++ interface between meep geometry (geometric_object_list from libctl) and meep materials. `dft_data`, `fragment_stats` structs. |
| `material_data.hpp` | ~100 | `material_data` struct used by meepgeom layer to pass material parameters from Python through SWIG into C++. |
| `step.cpp` | ~300 | `fields::step()` orchestrates a single FDTD timestep in the exact sequence described in Section 2.1. Also: `phase_material()` for mid-run material changes. |
| `step_generic.cpp` | ~400 | The inner numerical loop. `step_curl()` updates one field component via finite-difference curl of another. Handles PML sigma stretching and conductivity damping. The most performance-critical file. |
| `step_db.cpp` | ~200 | `fields::step_db(field_type)` dispatches the curl update to the correct component pairs (B from curl-H, D from curl-E) for each chunk via `step_generic`. |
| `update_eh.cpp` | ~200 | `fields::update_eh(field_type)` computes H = chi1inv * B and E = chi1inv * D after the curl step, looping over all owned grid points. |
| `update_pols.cpp` | ~100 | `fields::update_pols(field_type)` advances the polarization state for all susceptibility terms by calling `susceptibility::update_P()`. |
| `fields.cpp` | ~600 | `fields` class constructor/destructor, `fields_chunk` allocation, `connect_chunks()`, `sync_chunk_connections()`. Also time-domain field query: `get_field()`. |
| `fields_dump.cpp` | ~400 | Checkpoint/restart: `fields::dump()` and `fields::load()` write/read all field arrays to HDF5. |
| `structure.cpp` | ~500 | `structure` constructor: builds grid, calls `choose_chunkdivision()` (binary_partition), distributes chunks to MPI ranks. Also `set_materials()`, `add_susceptibility()`. |
| `structure_dump.cpp` | ~300 | `structure::dump()` / `structure::load()` for saving/restoring material arrays. |
| `boundaries.cpp` | ~800 | `fields::step_boundaries()`: executes pre-computed MPI communication. Also `connect_chunks()` logic for setting up ghost cell copy operations. `boundary_region::apply()` sets up PML arrays. |
| `dft.cpp` | ~1200 | All DFT monitor logic: `dft_chunk::update_dft()`, `fields::add_dft()` (creates dft_chunks via `loop_in_chunks()`), `dft_flux::flux()`, `fields::update_dfts()`. |
| `sources.cpp` | ~600 | `fields::add_volume_source()`, `fields::add_point_source()`, `fields::step_source()`. Converts Python Source objects to per-chunk `src_vol` objects with spatial amplitude arrays. |
| `susceptibility.cpp` | ~500 | `lorentzian_susceptibility::update_P()` (the ADE update), `gyrotropic_susceptibility::update_P()`, `multilevel_susceptibility::update_P()` (rate equations). |
| `meepgeom.cpp` | ~3000 | Geometry-to-grid conversion. `set_materials()` iterates all grid points, finds enclosing geometric object, computes subpixel-averaged chi1inv via numerical quadrature (Gaussian quadrature over Yee cell). The slowest part of `init_sim()`. |
| `near2far.cpp` | ~600 | `dft_near2far` implementation: accumulates tangential E,H on a closed surface, then evaluates Green's function integral to extrapolate to far-field points. |
| `array_slice.cpp` | ~700 | `fields::get_array_slice()`: extracts a subvolume of field data, interpolating from the staggered Yee grid to a regular grid, with MPI reduction. |
| `cw_fields.cpp` | ~300 | Continuous-wave frequency-domain solver: iterative (Bi-CGSTAB) solver for `(curl curl - omega^2 eps) E = -i omega J`. |
| `mympi.cpp` | ~600 | MPI wrappers: `or_to_all()`, `sum_to_all()`, `broadcast()`, `send()`, `receive()`. Also OpenMP and timing utilities. Abstracts MPI/non-MPI builds. |
| `mpb.cpp` | ~400 | Interfaces with libpympb for eigenmode computation. `fields::get_eigenmode()` runs MPB and returns mode data. `fields::add_eigenmode_source()` converts mode data to current sources. |
| `h5file.cpp` | ~600 | HDF5 I/O wrapper with parallel (collective MPI I/O) and serial modes. Chunked write for large arrays. |
| `monitor.cpp` | ~200 | `get_chi1inv()`, `get_field()` at arbitrary (sub-grid) points via trilinear interpolation. |
| `stress.cpp` | ~300 | Maxwell stress tensor DFT (`dft_force`): optical force calculation via frequency-domain stress tensor. |

### 8.2 Python Interface (`python/`)

| File | Purpose |
|------|---------|
| `meep.i` | SWIG interface (~65K lines). Declares `%module meep`. Contains `%typemap` directives converting NumPy arrays to/from C++ pointer+dims. Wraps all public meep.hpp classes. |
| `meep-python.hpp` | C++ helper code visible only to the Python interface layer: `custom_py_src_time` (Python callable as source), array conversion utilities, `meep_geom` material helpers. |
| `simulation.py` | `Simulation` class (the top-level user object). Manages the lifecycle: `init_sim()` -> `run()` -> outputs. Contains step function combinators: `at_every()`, `at_beginning()`, `at_end()`, `until()`. |
| `geom.py` | `Vector3`, `Medium`, all `GeometricObject` subclasses, all `Susceptibility` subclasses. These are pure Python classes that are converted to C++ equivalents during `init_sim()`. |
| `source.py` | `Source`, `EigenModeSource`, `GaussianBeamSource`. Python-side wrappers that are converted to C++ source calls during `init_sim()`. |
| `materials.py` | ~50 predefined materials (cSi, aSi, SiO2, Au, Ag, Al, Cu, GaAs, InGaAsP, ...) as `mp.Medium` objects with Lorentzian fit parameters from literature. |
| `visualization.py` | `plot2D()`, `plot_eps()`, `plot_boundaries()`, `animate_simulation()`. Pure Python using matplotlib. Calls `sim.get_array_slice()` internally. |
| `adjoint/wrapper.py` | `MeepJaxWrapper`: makes a Meep simulation differentiable via JAX's `custom_vjp`. |
| `adjoint/optimization_problem.py` | `OptimizationProblem`: higher-level adjoint API with objective/gradient calls. |
| `adjoint/filters.py` | Spatial filters for design variables: conic filter (cone-shaped), cylindrical filter, tanh projection. Applied to density maps before material assignment. |
| `adjoint/utils.py` | `create_adjoint_sources()`, `gather_monitor_values()`, etc. Core adjoint bookkeeping. |

### 8.3 MPB Integration (`libpympb/`)

| File | Purpose |
|------|---------|
| `pympb.hpp` | Declares `mode_solver` struct wrapping MPB's global state. Methods: `init()`, `run_te()`, `run_tm()`, `run_zeven()`, `run_zodd()`, `get_eigenvalues()`. |
| `pympb.cpp` | Implements `mode_solver`. Calls into MPB's C API (`maxwell_create_target_data()`, `eigensolver()`, etc.). Also implements `map_data()` for interpolating MPB fields onto Meep grids (different resolutions, different coordinate systems). |

### 8.4 Build System Files

| File | Purpose |
|------|---------|
| `configure.ac` | Autoconf input. Checks for all dependencies. Generates `config.h` and `Makefile`. Current shared library version: `SHARED_VERSION_INFO="36:0:1"`. |
| `Makefile.am` | Root Automake input. Defines `SUBDIRS`. |
| `src/Makefile.am` | Builds `libmeep.la`. Lists all `.cpp` sources. Defines compiler flags. Handles `step_generic_stride1.cpp` auto-generation via sed. |
| `python/Makefile.am` | Builds `_meep.so` via SWIG + compilation. Defines Python test targets. Sets `RUNCODE` for MPI/OpenMP test execution. |
| `.clang-format` | LLVM style, 2-space indent, 100-char lines, Cpp03 mode. Run `clang-format -i src/*.cpp` to format. |
| `.pre-commit-config.yaml` | Pre-commit hooks: clang-format (C++), black + pyupgrade + bandit (Python). Run `pre-commit run -a` to check all files. |

---

## Appendix A: Field Component Reference

Meep uses the Yee staggered grid. Different field components live at different
positions within each unit cell.

```
Component enum (src/meep/vec.hpp):
  Electric:   Ex, Ey, Er, Ep, Ez    (Cartesian + cylindrical)
  Magnetic:   Hx, Hy, Hr, Hp, Hz
  D-field:    Dx, Dy, Dr, Dp, Dz    (electric flux density)
  B-field:    Bx, By, Br, Bp, Bz    (magnetic flux density)
  Special:    Dielectric (= Centered), Permeability, NO_COMPONENT

Derived components (computed from above):
  Sx, Sy, Sr, Sp, Sz    (Poynting vector components)
  EnergyDensity, D_EnergyDensity, H_EnergyDensity

Field type groupings:
  E_stuff=0   electric field (E)
  H_stuff=1   magnetic field (H)
  D_stuff=2   electric flux density (D)
  B_stuff=3   magnetic flux density (B)
  PE_stuff=4  electric polarization auxiliary W
  PH_stuff=5  magnetic polarization auxiliary W
  WE_stuff=6  electric polarization P
  WH_stuff=7  magnetic polarization P
```

## Appendix B: FDTD Background for C++/Java Developers

The FDTD method discretizes Maxwell's curl equations on a staggered grid
(Yee grid) and advances them in time using a leapfrog scheme:

```
Maxwell's equations (time domain):
  dB/dt = -curl(E)           (Faraday's law)
  dD/dt = +curl(H) - J       (Ampere's law with current source J)
  D = epsilon * E            (constitutive relation)
  B = mu * H

Yee grid (2D example, Ez/Hx/Hy components):
  Ez lives at (i, j) integer grid points
  Hx lives at (i, j+1/2) half-integer in y
  Hy lives at (i+1/2, j) half-integer in x
  Staggered so that the curl operator is a simple finite difference.

Leapfrog time scheme:
  B^{n+1/2} = B^{n-1/2} - dt * curl(E^n)
  E^{n+1}   = E^n + dt/eps * (curl(H^{n+1/2}) - J^{n+1/2})
  where H = B/mu (or more generally H = chi1inv * B - polarization)

Stability: the Courant condition requires
  dt <= Courant / (a * sqrt(dim))
  where a is the spatial resolution (grid points per unit length).
  Meep defaults to Courant = 0.5 which satisfies this for all dimensions.

PML absorbing boundaries:
  Conceptually, the PML is a lossy material whose impedance matches the
  interior exactly for all angles and frequencies. In practice it is
  implemented as a complex coordinate stretch sigma(x) that damps
  outgoing waves exponentially while reflecting nothing at the interface.
```

---

## See Also

- [148 Physics Tutorials](tutorials/00_index.md) — Deep-dive theory and code walkthroughs for every Python example and test
- [Developer Guide](DEVELOPER_GUIDE.md) — Building from source, testing, and contributing
- [User Guide](USER_GUIDE.md) — Installation, tutorials, and 10 worked use cases
- [Quick Start Explained](QUICKSTART_EXPLAINED.md) — Detailed walkthrough of the Quick Start example
- [Test Report](TEST_REPORT.md) — Results from running all 148 Python examples and tests
- [Online Manual](https://meep.readthedocs.io/en/latest) — Full reference documentation

---

*This document was written for Meep version 1.33.0-beta (build date 2026-02-21).*
*Source: https://github.com/NanoComp/meep*
