# Meep Physics Tutorials — Master Index

**148 deep-dive tutorials** covering every Python example and test in the Meep repository. Each tutorial explains the physics theory, walks through the code, and connects electromagnetic concepts to Meep API usage.

---

## At a Glance

| File | Lines | Tutorials |
|------|------:|----------:|
| [`01_waveguides_and_circuits.md`](01_waveguides_and_circuits.md) | 1,152 | 18 |
| [`02_resonators_and_cavities.md`](02_resonators_and_cavities.md) | 1,078 | 16 |
| [`03_scattering_and_radiation.md`](03_scattering_and_radiation.md) | 1,326 | 18 |
| [`04_photonic_crystals_and_mpb.md`](04_photonic_crystals_and_mpb.md) | 1,144 | 15 |
| [`05_gratings_and_diffractive.md`](05_gratings_and_diffractive.md) | 998 | 13 |
| [`06_materials_and_dispersion.md`](06_materials_and_dispersion.md) | 1,223 | 17 |
| [`07_nonlinear_and_multilevel.md`](07_nonlinear_and_multilevel.md) | 680 | 8 |
| [`08_adjoint_and_optimization.md`](08_adjoint_and_optimization.md) | 747 | 8 |
| [`09_sources_and_monitors.md`](09_sources_and_monitors.md) | 1,143 | 14 |
| [`10_boundaries_and_symmetry.md`](10_boundaries_and_symmetry.md) | 1,264 | 10 |
| [`11_geometry_and_materials_lib.md`](11_geometry_and_materials_lib.md) | 675 | 6 |
| [`12_simulation_infrastructure.md`](12_simulation_infrastructure.md) | 782 | 8 |
| **Total** | **12,212** | **148 unique** |

Each tutorial contains physics theory with governing equations, a code walkthrough with key snippets from the actual source files, key takeaways connecting electromagnetic concepts to Meep API usage, and test status (PASS/FAIL/TIMEOUT) from the [Test Report](../TEST_REPORT.md).

---

## How to Use This Guide

Each tutorial follows a consistent structure:

- **Physics** — One-line summary of the electromagnetic phenomenon
- **Difficulty** — Beginner / Intermediate / Advanced
- **Theory** — Governing equations, physical intuition, and analytical benchmarks
- **Code Walkthrough** — Key code snippets with API explanations
- **Key Takeaways** — Bullet points connecting physics to Meep usage

Tutorials are organized into 12 thematic chapters. Use the table below to find any specific file, or browse chapters by topic.

---

## Chapters

| # | Chapter | Tutorials | File |
|---|---------|-----------|------|
| 1 | [Waveguides and Circuits](01_waveguides_and_circuits.md) | 18 | `01_waveguides_and_circuits.md` |
| 2 | [Resonators and Cavities](02_resonators_and_cavities.md) | 16 | `02_resonators_and_cavities.md` |
| 3 | [Scattering and Radiation](03_scattering_and_radiation.md) | 18 | `03_scattering_and_radiation.md` |
| 4 | [Photonic Crystals and MPB](04_photonic_crystals_and_mpb.md) | 15 | `04_photonic_crystals_and_mpb.md` |
| 5 | [Gratings and Diffractive Optics](05_gratings_and_diffractive.md) | 13 | `05_gratings_and_diffractive.md` |
| 6 | [Materials and Dispersion](06_materials_and_dispersion.md) | 17 | `06_materials_and_dispersion.md` |
| 7 | [Nonlinear Optics and Multi-Level Atoms](07_nonlinear_and_multilevel.md) | 8 | `07_nonlinear_and_multilevel.md` |
| 8 | [Adjoint Optimization and Inverse Design](08_adjoint_and_optimization.md) | 8 | `08_adjoint_and_optimization.md` |
| 9 | [Sources, Monitors, and Field Analysis](09_sources_and_monitors.md) | 14 | `09_sources_and_monitors.md` |
| 10 | [Boundaries, Symmetry, and Coordinates](10_boundaries_and_symmetry.md) | 10 | `10_boundaries_and_symmetry.md` |
| 11 | [Geometry and Material Grids](11_geometry_and_materials_lib.md) | 6 | `11_geometry_and_materials_lib.md` |
| 12 | [Simulation Infrastructure](12_simulation_infrastructure.md) | 8 | `12_simulation_infrastructure.md` |

**Total: 148 unique tutorials** (3 files appear in two chapters for cross-reference)

---

## Complete File Index

Every Python file from `python/examples/` and `python/tests/` with its chapter location.

### Examples (`python/examples/`) — 83 files

| File | Chapter | Topic |
|------|---------|-------|
| `3rd-harm-1d.py` | [Ch 7](07_nonlinear_and_multilevel.md) | Third-harmonic generation via chi(3) |
| `absorbed_power_density.py` | [Ch 8](08_adjoint_and_optimization.md) | Absorbed power density mapping |
| `absorber-1d.py` | [Ch 6](06_materials_and_dispersion.md) | PML vs Absorber boundary comparison |
| `antenna-radiation.py` | [Ch 3](03_scattering_and_radiation.md) | Dipole antenna radiation pattern |
| `antenna_pec_ground_plane.py` | [Ch 3](03_scattering_and_radiation.md) | Antenna above PEC ground plane |
| `antenna_pec_ground_plane_1D.py` | [Ch 3](03_scattering_and_radiation.md) | 1D Brillouin-zone integration for antenna |
| `bend-flux.py` | [Ch 1](01_waveguides_and_circuits.md) | Waveguide bend transmission/reflection |
| `bent-waveguide.py` | [Ch 1](01_waveguides_and_circuits.md) | Bent waveguide field visualization |
| `binary_grating.py` | [Ch 5](05_gratings_and_diffractive.md) | Binary grating diffraction efficiency |
| `binary_grating_levelset.py` | [Ch 5](05_gratings_and_diffractive.md) | Level-set binary grating |
| `binary_grating_n2f.py` | [Ch 5](05_gratings_and_diffractive.md) | Near-to-far field for periodic grating |
| `binary_grating_oblique.py` | [Ch 5](05_gratings_and_diffractive.md) | Oblique incidence on binary grating |
| `binary_grating_phasemap.py` | [Ch 5](05_gratings_and_diffractive.md) | Phase map for metasurface design |
| `cavity-farfield.py` | [Ch 2](02_resonators_and_cavities.md) | Cavity far-field radiation pattern |
| `cavity_arrayslice.py` | [Ch 2](02_resonators_and_cavities.md) | Cavity field array slicing |
| `cherenkov-radiation.py` | [Ch 3](03_scattering_and_radiation.md) | Cherenkov radiation cone |
| `chirped_pulse.py` | [Ch 7](07_nonlinear_and_multilevel.md) | Chirped pulse propagation |
| `coupler.py` | [Ch 1](01_waveguides_and_circuits.md) | Directional coupler S-parameters |
| `cyl-ellipsoid.py` | [Ch 3](03_scattering_and_radiation.md) | Cylinder-ellipsoid scattering |
| `cylinder_cross_section.py` | [Ch 3](03_scattering_and_radiation.md) | Cylinder scattering cross section |
| `differential_cross_section.py` | [Ch 3](03_scattering_and_radiation.md) | Differential scattering cross section |
| `diffracted_planewave.py` | [Ch 5](05_gratings_and_diffractive.md) | Diffracted planewave decomposition |
| `dipole_in_vacuum_1D.py` | [Ch 3](03_scattering_and_radiation.md) | 1D dipole radiation pattern |
| `dipole_in_vacuum_cyl_off_axis.py` | [Ch 3](03_scattering_and_radiation.md) | Off-axis dipole in cylindrical coords |
| `dipole_in_vacuum_cyl_on_axis.py` | [Ch 3](03_scattering_and_radiation.md) | On-axis dipole in cylindrical coords |
| `disc_extraction_efficiency.py` | [Ch 2](02_resonators_and_cavities.md) | Disc extraction efficiency |
| `disc_radiation_pattern.py` | [Ch 2](02_resonators_and_cavities.md) | Disc radiation pattern |
| `eps_fit_lorentzian.py` | [Ch 6](06_materials_and_dispersion.md) | Lorentzian fit to material data |
| `extraction_eff_ldos.py` | [Ch 2](02_resonators_and_cavities.md) | Extraction efficiency via LDOS |
| `faraday-rotation.py` | [Ch 6](06_materials_and_dispersion.md) | Faraday rotation in gyrotropic media |
| `finite_grating.py` | [Ch 5](05_gratings_and_diffractive.md) | Finite-length grating scattering |
| `gaussian-beam.py` | [Ch 3](03_scattering_and_radiation.md) | Gaussian beam propagation |
| `grating2d_triangular_lattice.py` | [Ch 5](05_gratings_and_diffractive.md) | 2D triangular lattice grating |
| `holey-wvg-bands.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Photonic crystal waveguide bands |
| `holey-wvg-cavity.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Photonic crystal waveguide cavity |
| `material-dispersion.py` | [Ch 6](06_materials_and_dispersion.md) | Dispersive material simulation |
| `metal-cavity-ldos.py` | [Ch 2](02_resonators_and_cavities.md) | Metallic cavity LDOS |
| `metasurface_lens.py` | [Ch 5](05_gratings_and_diffractive.md) | Metasurface lens design |
| `mie_scattering.py` | [Ch 3](03_scattering_and_radiation.md) | Mie scattering from sphere |
| `mode-decomposition.py` | [Ch 1](01_waveguides_and_circuits.md) | Waveguide mode decomposition |
| `mode_coeff_phase.py` | [Ch 1](01_waveguides_and_circuits.md) | Mode coefficient phase extraction |
| `mode_converter.py` | [Ch 8](08_adjoint_and_optimization.md) | Adjoint-optimized mode converter |
| `mpb_bragg.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | 1D Bragg stack band structure |
| `mpb_bragg_sine.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Sinusoidal Bragg grating |
| `mpb_data_analysis.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | MPB data analysis and visualization |
| `mpb_diamond.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Diamond lattice photonic crystal |
| `mpb_hole_slab.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Photonic crystal slab with holes |
| `mpb_honey_rods.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Honeycomb lattice rods |
| `mpb_line_defect.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Line defect waveguide |
| `mpb_sq_rods.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Square lattice rods |
| `mpb_strip.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Strip waveguide dispersion |
| `mpb_tri_holes.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Triangular lattice holes |
| `mpb_tri_rods.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | Triangular lattice rods |
| `mpb_tutorial.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | MPB introductory tutorial |
| `multilayer_opt.py` | [Ch 8](08_adjoint_and_optimization.md) | Multilayer film optimization |
| `multilevel-atom.py` | [Ch 7](07_nonlinear_and_multilevel.md) | Multi-level atomic medium |
| `oblique-planewave.py` | [Ch 6](06_materials_and_dispersion.md) | Oblique planewave injection |
| `oblique-source.py` | [Ch 6](06_materials_and_dispersion.md) | Oblique EigenModeSource |
| `parallel-wvgs-force.py` | [Ch 1](01_waveguides_and_circuits.md) | Optical forces between waveguides |
| `parallel-wvgs-mpb.py` | [Ch 1](01_waveguides_and_circuits.md) | Parallel waveguide MPB analysis |
| `perturbation_theory.py` | [Ch 9](09_sources_and_monitors.md) | Perturbation theory (cylindrical) |
| `perturbation_theory_2d.py` | [Ch 9](09_sources_and_monitors.md) | Perturbation theory (2D Cartesian) |
| `phase_in_material.py` | [Ch 6](06_materials_and_dispersion.md) | Adiabatic material phase-in |
| `planar_cavity_ldos.py` | [Ch 2](02_resonators_and_cavities.md) | Planar cavity Purcell enhancement |
| `plot_radiation_pattern_dipole.py` | [Ch 3](03_scattering_and_radiation.md) | Radiation pattern visualization |
| `point_dipole_cyl.py` | [Ch 3](03_scattering_and_radiation.md) | Point dipole in cylindrical coords |
| `polarization_grating.py` | [Ch 5](05_gratings_and_diffractive.md) | Polarization grating |
| `pw-source.py` | [Ch 9](09_sources_and_monitors.md) | Plane wave source construction |
| `refl-angular-kz2d.py` | [Ch 6](06_materials_and_dispersion.md) | Angular reflectance with kz_2d |
| `refl-angular.py` | [Ch 6](06_materials_and_dispersion.md) | Angular reflectance (Fresnel) |
| `refl-quartz.py` | [Ch 6](06_materials_and_dispersion.md) | Quartz reflectance spectrum |
| `ring-cyl.py` | [Ch 2](02_resonators_and_cavities.md) | Ring resonator (cylindrical) |
| `ring-mode-overlap.py` | [Ch 1](01_waveguides_and_circuits.md) | Ring resonator mode overlap |
| `ring.py` | [Ch 2](02_resonators_and_cavities.md) | Ring resonator (2D Cartesian) |
| `ring_gds.py` | [Ch 2](02_resonators_and_cavities.md) | Ring resonator from GDSII |
| `solve-cw.py` | [Ch 9](09_sources_and_monitors.md) | Continuous-wave frequency solver |
| `stochastic_emitter.py` | [Ch 7](07_nonlinear_and_multilevel.md) | Stochastic dipole emitter |
| `stochastic_emitter_line.py` | [Ch 7](07_nonlinear_and_multilevel.md) | Stochastic line source |
| `stochastic_emitter_reciprocity.py` | [Ch 7](07_nonlinear_and_multilevel.md) | Reciprocity-based emitter |
| `straight-waveguide.py` | [Ch 1](01_waveguides_and_circuits.md) | Straight waveguide basics |
| `waveguide_crossing.py` | [Ch 8](08_adjoint_and_optimization.md) | Waveguide crossing optimization |
| `wvg-src.py` | [Ch 1](01_waveguides_and_circuits.md) | Waveguide eigenmode source |
| `zone_plate.py` | [Ch 5](05_gratings_and_diffractive.md) | Fresnel zone plate |

### Tests (`python/tests/`) — 65 files

| File | Chapter | Topic |
|------|---------|-------|
| `test_3rd_harm_1d.py` | [Ch 7](07_nonlinear_and_multilevel.md) | THG regression test |
| `test_absorber_1d.py` | [Ch 6](06_materials_and_dispersion.md) | Absorber boundary test |
| `test_adjoint_cyl.py` | [Ch 8](08_adjoint_and_optimization.md) | Cylindrical adjoint gradients |
| `test_adjoint_jax.py` | [Ch 8](08_adjoint_and_optimization.md) | JAX adjoint wrapper |
| `test_adjoint_solver.py` | [Ch 8](08_adjoint_and_optimization.md) | Adjoint gradient verification |
| `test_adjoint_utils.py` | [Ch 8](08_adjoint_and_optimization.md) | Spatial filter utilities |
| `test_antenna_radiation.py` | [Ch 3](03_scattering_and_radiation.md) | Antenna pattern validation |
| `test_array_metadata.py` | [Ch 9](09_sources_and_monitors.md) | Array metadata and modal volume |
| `test_bend_flux.py` | [Ch 1](01_waveguides_and_circuits.md) | Bend flux regression |
| `test_binary_grating.py` | [Ch 5](05_gratings_and_diffractive.md) | Grating diffraction test |
| `test_binary_partition_utils.py` | [Ch 10](10_boundaries_and_symmetry.md) | BSP tree partitioning |
| `test_boundaries_1D.py` | [Ch 10](10_boundaries_and_symmetry.md) | 1D boundary conditions |
| `test_cavity_arrayslice.py` | [Ch 2](02_resonators_and_cavities.md) | Cavity array slicing |
| `test_cavity_farfield.py` | [Ch 2](02_resonators_and_cavities.md) | Cavity far-field test |
| `test_chunk_balancer.py` | [Ch 10](10_boundaries_and_symmetry.md) | Adaptive chunk balancing |
| `test_chunk_layout.py` | [Ch 10](10_boundaries_and_symmetry.md) | Custom chunk layout |
| `test_chunks.py` | [Ch 10](10_boundaries_and_symmetry.md) | Chunk boundary flux conservation |
| `test_conductivity.py` | [Ch 6](06_materials_and_dispersion.md) | Material conductivity |
| `test_cyl_ellipsoid.py` | [Ch 3](03_scattering_and_radiation.md) | Cylinder-ellipsoid regression |
| `test_dft_energy.py` | [Ch 9](09_sources_and_monitors.md) | DFT energy monitors |
| `test_dft_fields.py` | [Ch 9](09_sources_and_monitors.md) | DFT field accumulation |
| `test_diffracted_planewave.py` | [Ch 3](03_scattering_and_radiation.md) | Diffracted planewave test |
| `test_dispersive_eigenmode.py` | [Ch 1](01_waveguides_and_circuits.md) | Dispersive material eigenmodes |
| `test_divide_mpi_processes.py` | [Ch 12](12_simulation_infrastructure.md) | MPI process division |
| `test_dump_load.py` | [Ch 12](12_simulation_infrastructure.md) | Checkpoint/restart |
| `test_eigfreq.py` | [Ch 2](02_resonators_and_cavities.md) | Eigenfrequency solver |
| `test_faraday_rotation.py` | [Ch 6](06_materials_and_dispersion.md) | Faraday rotation test |
| `test_field_functions.py` | [Ch 9](09_sources_and_monitors.md) | Field function integration |
| `test_force.py` | [Ch 9](09_sources_and_monitors.md) | Maxwell stress tensor force |
| `test_fragment_stats.py` | [Ch 12](12_simulation_infrastructure.md) | Fragment statistics |
| `test_gaussianbeam.py` | [Ch 3](03_scattering_and_radiation.md) | Gaussian beam test |
| `test_geom.py` | [Ch 11](11_geometry_and_materials_lib.md) | Geometric objects |
| `test_get_epsilon_grid.py` | [Ch 9](09_sources_and_monitors.md) | Epsilon grid queries |
| `test_get_point.py` | [Ch 9](09_sources_and_monitors.md) | Field point interpolation |
| `test_holey_wvg_bands.py` | [Ch 1](01_waveguides_and_circuits.md) | Holey waveguide bands |
| `test_holey_wvg_cavity.py` | [Ch 1](01_waveguides_and_circuits.md) | Holey waveguide cavity |
| `test_integrated_source.py` | [Ch 9](09_sources_and_monitors.md) | Integrated source flag |
| `test_kdom.py` | [Ch 1](01_waveguides_and_circuits.md) | Bloch wavevector test |
| `test_ldos.py` | [Ch 2](02_resonators_and_cavities.md) | LDOS calculation |
| `test_material_dispersion.py` | [Ch 6](06_materials_and_dispersion.md) | Material dispersion test |
| `test_material_grid.py` | [Ch 11](11_geometry_and_materials_lib.md) | Material grid topology opt |
| `test_materials_library.py` | [Ch 6](06_materials_and_dispersion.md) | Materials library validation |
| `test_medium_evaluations.py` | [Ch 6](06_materials_and_dispersion.md) | Medium epsilon evaluation |
| `test_mode_coeffs.py` | [Ch 1](01_waveguides_and_circuits.md) | Mode coefficient extraction |
| `test_mode_decomposition.py` | [Ch 1](01_waveguides_and_circuits.md) | Mode decomposition test |
| `test_mpb.py` | [Ch 4](04_photonic_crystals_and_mpb.md) | MPB eigenmode solver test |
| `test_multilevel_atom.py` | [Ch 7](07_nonlinear_and_multilevel.md) | Multi-level atom test |
| `test_n2f_periodic.py` | [Ch 5](05_gratings_and_diffractive.md) | Periodic near-to-far |
| `test_oblique_source.py` | [Ch 9](09_sources_and_monitors.md) | Oblique source injection |
| `test_physical.py` | [Ch 10](10_boundaries_and_symmetry.md) | CW solver physical validation |
| `test_planewave_1D.py` | [Ch 10](10_boundaries_and_symmetry.md) | 1D planewave DFT phase |
| `test_pml_cyl.py` | [Ch 10](10_boundaries_and_symmetry.md) | Cylindrical PML test |
| `test_prism.py` | [Ch 11](11_geometry_and_materials_lib.md) | Prism geometry test |
| `test_pw_source.py` | [Ch 9](09_sources_and_monitors.md) | Plane wave source test |
| `test_refl_angular.py` | [Ch 6](06_materials_and_dispersion.md) | Angular reflectance test |
| `test_ring.py` | [Ch 2](02_resonators_and_cavities.md) | Ring resonator test |
| `test_ring_cyl.py` | [Ch 2](02_resonators_and_cavities.md) | Cylindrical ring test |
| `test_simulation.py` | [Ch 11](11_geometry_and_materials_lib.md) | Simulation infrastructure |
| `test_source.py` | [Ch 11](11_geometry_and_materials_lib.md) | Source configuration test |
| `test_special_kz.py` | [Ch 10](10_boundaries_and_symmetry.md) | Out-of-plane kz modes |
| `test_timing_measurements.py` | [Ch 12](12_simulation_infrastructure.md) | Performance timing |
| `test_user_defined_material.py` | [Ch 10](10_boundaries_and_symmetry.md) | User-defined materials |
| `test_verbosity_mgr.py` | [Ch 12](12_simulation_infrastructure.md) | Verbosity control |
| `test_visualization.py` | [Ch 12](12_simulation_infrastructure.md) | Visualization and plotting |
| `test_wvg_src.py` | [Ch 1](01_waveguides_and_circuits.md) | Waveguide source test |

---

## By Difficulty

### Beginner (recommended starting points)
- `straight-waveguide.py` — [Ch 1](01_waveguides_and_circuits.md) — Your first Meep simulation
- `ring.py` — [Ch 2](02_resonators_and_cavities.md) — Ring resonator fundamentals
- `bend-flux.py` — [Ch 1](01_waveguides_and_circuits.md) — Transmission and reflection measurement
- `pw-source.py` — [Ch 9](09_sources_and_monitors.md) — Plane wave source construction
- `absorber-1d.py` — [Ch 6](06_materials_and_dispersion.md) — Boundary condition basics
- `mpb_bragg.py` — [Ch 4](04_photonic_crystals_and_mpb.md) — 1D photonic band gap

### Intermediate (core techniques)
- `binary_grating.py` — [Ch 5](05_gratings_and_diffractive.md) — Diffraction efficiency calculation
- `mie_scattering.py` — [Ch 3](03_scattering_and_radiation.md) — Scattered-field method
- `refl-angular.py` — [Ch 6](06_materials_and_dispersion.md) — Fresnel equation validation
- `3rd-harm-1d.py` — [Ch 7](07_nonlinear_and_multilevel.md) — Nonlinear optics basics
- `perturbation_theory.py` — [Ch 9](09_sources_and_monitors.md) — Cavity perturbation analysis

### Advanced (specialized topics)
- `waveguide_crossing.py` — [Ch 8](08_adjoint_and_optimization.md) — Topology optimization
- `stochastic_emitter_reciprocity.py` — [Ch 7](07_nonlinear_and_multilevel.md) — Reciprocity-based emission
- `metasurface_lens.py` — [Ch 5](05_gratings_and_diffractive.md) — Metalens design workflow
- `multilevel-atom.py` — [Ch 7](07_nonlinear_and_multilevel.md) — Gain media and lasing

---

## By Physics Topic

### Maxwell's Equations and FDTD
- Yee grid basics: `straight-waveguide.py`, `test_planewave_1D.py`
- Time stepping: `test_simulation.py`, `test_timing_measurements.py`
- Boundary conditions: `test_boundaries_1D.py`, `test_pml_cyl.py`
- CW solver: `solve-cw.py`, `test_physical.py`

### Guided Waves
- Slab/strip waveguides: `straight-waveguide.py`, `mpb_strip.py`
- Bends and couplers: `bend-flux.py`, `coupler.py`
- Mode analysis: `mode-decomposition.py`, `test_mode_coeffs.py`
- Photonic crystal waveguides: `holey-wvg-bands.py`, `mpb_line_defect.py`

### Resonators
- Ring resonators: `ring.py`, `ring-cyl.py`, `ring_gds.py`
- Photonic crystal cavities: `holey-wvg-cavity.py`, `cavity-farfield.py`
- LDOS and Purcell: `metal-cavity-ldos.py`, `extraction_eff_ldos.py`
- Q-factor extraction: `test_eigfreq.py`, `test_ldos.py`

### Scattering
- Mie theory: `mie_scattering.py`, `cylinder_cross_section.py`
- Near-to-far: `antenna-radiation.py`, `test_antenna_radiation.py`
- Dipole radiation: `point_dipole_cyl.py`, `dipole_in_vacuum_1D.py`

### Periodic Structures
- Band structures: `mpb_sq_rods.py`, `mpb_tri_rods.py`, `mpb_diamond.py`
- Diffraction: `binary_grating.py`, `diffracted_planewave.py`
- Metasurfaces: `binary_grating_phasemap.py`, `metasurface_lens.py`

### Materials
- Dispersion: `material-dispersion.py`, `eps_fit_lorentzian.py`
- Magneto-optics: `faraday-rotation.py`, `test_faraday_rotation.py`
- Reflectance: `refl-angular.py`, `refl-quartz.py`
- Conductivity: `test_conductivity.py`

### Nonlinear and Quantum
- Chi(3): `3rd-harm-1d.py`
- Multi-level atoms: `multilevel-atom.py`
- Stochastic sources: `stochastic_emitter.py`, `stochastic_emitter_reciprocity.py`

### Optimization
- Adjoint method: `waveguide_crossing.py`, `mode_converter.py`
- Material grids: `test_material_grid.py`
- JAX integration: `test_adjoint_jax.py`

---

*Generated from the Meep repository at [github.com/NanoComp/meep](https://github.com/NanoComp/meep). See also the [Test Report](../TEST_REPORT.md) for execution results.*
