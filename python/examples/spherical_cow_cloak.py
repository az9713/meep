#!/usr/bin/env python3
"""
Spherical Cow Cloak: Transformation-Optics Invisibility Cloak
=============================================================

A 2D FDTD simulation demonstrating electromagnetic cloaking of a
"spherical cow" (the classic physics oversimplification) using Pendry's
transformation-optics invisibility cloak.

The cloak is a cylindrical shell (2D cross-section of a spherical shell)
with anisotropic, inhomogeneous permittivity (epsilon) and permeability
(mu) tensors derived from the coordinate transformation:

    r' = R1 + r * (R2 - R1) / R2

which maps the interior of a circle of radius R2 to the annular shell
R1 < r' < R2, rendering the interior invisible.

Three simulations compare:
  1. Empty space (reference for scattered-field subtraction)
  2. Bare spherical cow (dielectric cylinder, no cloak)
  3. Cloaked spherical cow (cow + transformation-optics shell)

The scattering cross section is computed via a closed-surface flux-box
technique (scattered-field subtraction) following the approach in
mie_scattering.py.

Reference:
  J.B. Pendry, D. Schurig, D.R. Smith, "Controlling Electromagnetic
  Fields," Science 312, 1780-1782 (2006).
"""

import math
import sys

import numpy as np

import meep as mp

# =====================================================================
# Physical and simulation parameters
# =====================================================================

R1 = 1.0  # Inner cloak radius = cow radius
R2 = 2.0  # Outer cloak radius
n_cow = 1.5  # Refractive index of the spherical cow

fcen = 0.3  # Center frequency (wavelength = 3.33)
df = 0.1  # Source bandwidth
nfreq = 1  # Single frequency for simplicity

resolution = 16  # Pixels per unit length
dpml = 1.5  # PML thickness
pad = 3.0  # Padding around cloak

# Regularization to smooth the singularity at r = R1 where
# eps_r, mu_r -> 0 and eps_theta, mu_theta -> infinity.
delta = 0.05

# Derived quantities
s = 2 * (R2 + pad + dpml)  # Cell size
cell_size = mp.Vector3(s, s, 0)
pml_layers = [mp.PML(thickness=dpml)]

# Half-size of the flux monitoring box (just outside the cloak)
box_half = R2 + 0.5

# =====================================================================
# Pendry cloak material function
# =====================================================================


def pendry_cloak_material(p):
    """
    Compute the anisotropic epsilon and mu tensors for the Pendry cloak
    at position p using transformation optics.

    In cylindrical coordinates (r, theta, z), the cloak parameters are:
        eps_r = mu_r = (r - R1) / r
        eps_t = mu_t = r / (r - R1)
        eps_z = mu_z = [R2 / (R2 - R1)]^2 * (r - R1) / r

    These are then rotated to Cartesian (x, y, z) coordinates.
    """
    x, y = p.x, p.y
    r = math.sqrt(x * x + y * y)

    if r <= R1 or r >= R2:
        return mp.air

    # Regularized radial distance from inner boundary
    rp = max(r - R1, delta)
    R_ratio_sq = (R2 / (R2 - R1)) ** 2

    # Cylindrical tensor components (eps = mu for impedance matching)
    comp_r = rp / r  # Radial component
    comp_t = r / rp  # Tangential (azimuthal) component
    comp_z = R_ratio_sq * rp / r  # Axial component

    # Rotation from cylindrical to Cartesian coordinates
    theta = math.atan2(y, x)
    ct = math.cos(theta)
    st = math.sin(theta)
    ct2 = ct * ct
    st2 = st * st
    ctst = ct * st

    xx = comp_r * ct2 + comp_t * st2
    yy = comp_r * st2 + comp_t * ct2
    xy = (comp_r - comp_t) * ctst

    return mp.Medium(
        epsilon_diag=mp.Vector3(xx, yy, comp_z),
        epsilon_offdiag=mp.Vector3(xy, 0, 0),
        mu_diag=mp.Vector3(xx, yy, comp_z),
        mu_offdiag=mp.Vector3(xy, 0, 0),
    )


# =====================================================================
# Plane wave source (Ez polarization, propagating in +x)
# =====================================================================

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s),
    )
]

# Representative magnetic material for extra_materials, so Meep
# allocates storage for mu tensors when using the material function.
mu_representative = R2 / delta
extra_materials_list = [mp.Medium(mu=mu_representative)]

# =====================================================================
# Helper: set up flux monitors (4-face box around origin)
# =====================================================================


def add_flux_box(sim):
    """Add 4 flux monitors forming a closed box of half-size box_half."""
    regions = [
        mp.FluxRegion(
            center=mp.Vector3(x=-box_half), size=mp.Vector3(0, 2 * box_half)
        ),
        mp.FluxRegion(
            center=mp.Vector3(x=+box_half), size=mp.Vector3(0, 2 * box_half)
        ),
        mp.FluxRegion(
            center=mp.Vector3(y=-box_half), size=mp.Vector3(2 * box_half, 0)
        ),
        mp.FluxRegion(
            center=mp.Vector3(y=+box_half), size=mp.Vector3(2 * box_half, 0)
        ),
    ]
    return [sim.add_flux(fcen, df, nfreq, fr) for fr in regions]


# =====================================================================
# Helper: DFT field region (for steady-state field visualization)
# =====================================================================

vis_half = R2 + pad
dft_center = mp.Vector3()
dft_size = mp.Vector3(2 * vis_half, 2 * vis_half)


def add_dft_monitor(sim):
    """Add a DFT field monitor over the visible region."""
    return sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=dft_center, size=dft_size)


# =====================================================================
# Simulation runner
# =====================================================================


def run_simulation(geometry=None, extra_materials=None, minus_flux_data=None, label=""):
    """
    Run a simulation and return (fluxes, flux_data, ez_dft_array).

    Parameters
    ----------
    geometry : list of GeometricObject, optional
    extra_materials : list of Medium, optional
    minus_flux_data : list of flux data from empty run, optional
    label : str, description for log messages
    """
    if mp.am_master():
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=mp.Vector3(),
        geometry=geometry or [],
        extra_materials=extra_materials or [],
    )

    # Flux monitors
    flux_monitors = add_flux_box(sim)

    # Subtract incident flux if reference data provided
    if minus_flux_data is not None:
        for mon, data in zip(flux_monitors, minus_flux_data):
            sim.load_minus_flux_data(mon, data)

    # DFT field monitor
    dft_mon = add_dft_monitor(sim)

    # Run until fields decay
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ez, mp.Vector3(R2 + 1, 0), 1e-6
        )
    )

    # Collect flux results
    fluxes = [mp.get_fluxes(m) for m in flux_monitors]
    flux_data = [sim.get_flux_data(m) for m in flux_monitors]

    # Collect DFT field
    ez_dft = sim.get_dft_array(dft_mon, mp.Ez, 0)

    sim.reset_meep()
    return fluxes, flux_data, ez_dft


# =====================================================================
# Run 1: Empty reference (no scatterer)
# =====================================================================

empty_fluxes, empty_flux_data, ez_empty = run_simulation(label="Empty reference")

# Incident flux for normalization (flux through the x1 face in +x direction)
incident_flux = empty_fluxes[0][0]

# =====================================================================
# Run 2: Bare spherical cow (no cloak)
# =====================================================================

geom_bare = [
    mp.Cylinder(radius=R1, material=mp.Medium(index=n_cow)),
]

bare_fluxes, _, ez_bare = run_simulation(
    geometry=geom_bare,
    minus_flux_data=empty_flux_data,
    label="Bare spherical cow (n={})".format(n_cow),
)

# =====================================================================
# Run 3: Cloaked spherical cow
# =====================================================================

geom_cloak = [
    mp.Cylinder(radius=R2, material=pendry_cloak_material),
    mp.Cylinder(radius=R1, material=mp.Medium(index=n_cow)),
]

cloak_fluxes, _, ez_cloak = run_simulation(
    geometry=geom_cloak,
    extra_materials=extra_materials_list,
    minus_flux_data=empty_flux_data,
    label="Cloaked spherical cow",
)

# =====================================================================
# Compute scattering cross sections
# =====================================================================

# Net scattered power through closed box (following mie_scattering.py convention)
# P_scat = flux_x1 - flux_x2 + flux_y1 - flux_y2
# Then sigma_scat = -P_scat / I_incident where I_incident = incident_flux / (2*box_half)

intensity = abs(incident_flux) / (2 * box_half)


def scattering_cross_section(fluxes):
    """Compute scattering cross section from the 4-face flux data."""
    raw = (
        np.asarray(fluxes[0])
        - np.asarray(fluxes[1])
        + np.asarray(fluxes[2])
        - np.asarray(fluxes[3])
    )
    return -raw / intensity


sigma_bare = scattering_cross_section(bare_fluxes)
sigma_cloak = scattering_cross_section(cloak_fluxes)

# Scattering efficiency Q = sigma / geometric_cross_section
# In 2D the geometric cross section is the diameter = 2*R1
geom_xs = 2 * R1
Q_bare = sigma_bare / geom_xs
Q_cloak = sigma_cloak / geom_xs

# =====================================================================
# Print results
# =====================================================================

if mp.am_master():
    print("\n")
    print("=" * 60)
    print("    SPHERICAL COW CLOAK — RESULTS")
    print("=" * 60)
    print(f"  Cow radius (R1):         {R1:.2f}")
    print(f"  Cloak outer radius (R2): {R2:.2f}")
    print(f"  Cow refractive index:    {n_cow:.2f}")
    print(f"  Wavelength:              {1/fcen:.2f}")
    print(f"  Resolution:              {resolution} px/unit")
    print(f"  Regularization (delta):  {delta:.3f}")
    print("-" * 60)
    print(f"  {'Case':<25s} {'sigma_scat':>12s} {'Q_scat':>10s}")
    print("-" * 60)
    print(f"  {'Bare cow':<25s} {sigma_bare[0]:12.4f} {Q_bare[0]:10.4f}")
    print(f"  {'Cloaked cow':<25s} {sigma_cloak[0]:12.4f} {Q_cloak[0]:10.4f}")
    print("-" * 60)

    if abs(Q_bare[0]) > 1e-10:
        reduction = (1 - abs(Q_cloak[0]) / abs(Q_bare[0])) * 100
        print(f"  Scattering reduction:    {reduction:.1f}%")
    else:
        print("  (Bare cow scattering too small to compute reduction)")

    print("=" * 60)

# =====================================================================
# Save field data and plot (if matplotlib is available)
# =====================================================================

if mp.am_master():
    # Save raw field data
    np.savez(
        "spherical_cow_cloak_fields.npz",
        ez_empty=ez_empty,
        ez_bare=ez_bare,
        ez_cloak=ez_cloak,
        x=np.linspace(-vis_half, vis_half, ez_empty.shape[0]),
        y=np.linspace(-vis_half, vis_half, ez_empty.shape[1]),
        params={
            "R1": R1,
            "R2": R2,
            "n_cow": n_cow,
            "fcen": fcen,
            "resolution": resolution,
        },
    )
    print("\nField data saved to spherical_cow_cloak_fields.npz")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        extent = [-vis_half, vis_half, -vis_half, vis_half]
        vmax = max(
            np.max(np.abs(np.real(ez_empty))),
            np.max(np.abs(np.real(ez_bare))),
            np.max(np.abs(np.real(ez_cloak))),
        )

        for ax, data, title in zip(
            axes,
            [ez_empty, ez_bare, ez_cloak],
            [
                "Empty (reference)",
                "Bare Spherical Cow",
                "Cloaked Spherical Cow",
            ],
        ):
            im = ax.imshow(
                np.real(data).T,
                cmap="RdBu_r",
                origin="lower",
                extent=extent,
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # Draw cow outline
            cow_circle = plt.Circle(
                (0, 0), R1, fill=False, color="black", linewidth=1.5, linestyle="--"
            )
            ax.add_patch(cow_circle)

            # Draw cloak boundary (for cloaked case)
            if "Cloak" in title:
                cloak_circle = plt.Circle(
                    (0, 0),
                    R2,
                    fill=False,
                    color="black",
                    linewidth=1.5,
                    linestyle=":",
                )
                ax.add_patch(cloak_circle)

        fig.colorbar(im, ax=axes, label="Re(Ez)", shrink=0.8)
        fig.suptitle(
            "Spherical Cow Cloak — Transformation Optics\n"
            f"λ = {1/fcen:.2f}, R₁ = {R1}, R₂ = {R2}, "
            f"n_cow = {n_cow}, Q_bare = {Q_bare[0]:.3f}, "
            f"Q_cloak = {Q_cloak[0]:.3f}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig("spherical_cow_cloak.png", dpi=150, bbox_inches="tight")
        print("Plot saved to spherical_cow_cloak.png")

    except ImportError:
        print("matplotlib not available; skipping plot generation.")
        print("Field data can be plotted from spherical_cow_cloak_fields.npz")
