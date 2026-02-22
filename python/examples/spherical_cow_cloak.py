#!/usr/bin/env python3
"""
Spherical Cow Cloak: 3D Transformation-Optics Invisibility Cloak
================================================================

A full 3D FDTD simulation demonstrating electromagnetic cloaking of a
"spherical cow" (the classic physics oversimplification) using Pendry's
transformation-optics invisibility cloak.

The cloak is a spherical shell with anisotropic, inhomogeneous
permittivity (epsilon) and permeability (mu) tensors derived from the
coordinate transformation:

    r' = R1 + r * (R2 - R1) / R2

which maps the interior of a sphere of radius R2 to the spherical shell
R1 < r' < R2, rendering the interior invisible.

In spherical coordinates, the cloak material parameters are:
    eps_r = mu_r = [R2 / (R2 - R1)] * [(r - R1) / r]^2
    eps_t = mu_t = R2 / (R2 - R1)     (tangential, same for theta & phi)

These are converted to full 3x3 Cartesian tensors using:
    eps_ij = eps_t * delta_ij + (eps_r - eps_t) * r_i * r_j / r^2

Three simulations compare:
  1. Empty space (reference for scattered-field subtraction)
  2. Bare spherical cow (dielectric sphere, no cloak)
  3. Cloaked spherical cow (cow + transformation-optics shell)

The scattering cross section is computed via a 6-face closed-surface
flux-box technique (scattered-field subtraction), following the same
approach as mie_scattering.py.

Mirror symmetries [Mirror(Y), Mirror(Z, phase=-1)] reduce the 3D
computational volume by 4x.

Reference:
  J.B. Pendry, D. Schurig, D.R. Smith, "Controlling Electromagnetic
  Fields," Science 312, 1780-1782 (2006).
"""

import math

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

resolution = 10  # Pixels per unit length (3D is expensive)
dpml = 1.0  # PML thickness
pad = 2.0  # Padding around cloak

# Regularization to smooth the singularity at r = R1 where
# eps_r, mu_r -> 0 and eps_theta, mu_theta -> infinity.
delta = 0.1

# Derived quantities
s = 2 * (R2 + pad + dpml)  # Cell size
cell_size = mp.Vector3(s, s, s)
pml_layers = [mp.PML(thickness=dpml)]

# Half-size of the flux monitoring box (just outside the cloak)
box_half = R2 + 0.5

# Mirror symmetries for Ez-polarized plane wave propagating in +x
# (same as mie_scattering.py, validated against Mie theory)
symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

# =====================================================================
# 3D Pendry cloak material function
# =====================================================================


def pendry_cloak_material(p):
    """
    Compute the anisotropic epsilon and mu tensors for the 3D Pendry
    spherical cloak at position p using transformation optics.

    In spherical coordinates (r, theta, phi), the cloak parameters are:
        eps_r = mu_r = [R2/(R2-R1)] * [(r-R1)/r]^2
        eps_t = mu_t = R2/(R2-R1)      (tangential, isotropic)

    Since the tangential components are equal, the Cartesian tensor is:
        eps_ij = eps_t * delta_ij + (eps_r - eps_t) * r_i*r_j / r^2
    """
    x, y, z = p.x, p.y, p.z
    r = math.sqrt(x * x + y * y + z * z)

    if r <= R1 or r >= R2:
        return mp.air

    # Regularized radial distance from inner boundary
    rp = max(r - R1, delta)
    R_ratio = R2 / (R2 - R1)

    # Spherical tensor components (eps = mu for impedance matching)
    comp_r = R_ratio * (rp / r) ** 2  # Radial
    comp_t = R_ratio  # Tangential (theta and phi, constant)

    # Convert to Cartesian: eps_ij = comp_t*I + (comp_r - comp_t)*rhat_i*rhat_j
    r2 = r * r
    dr = comp_r - comp_t  # Radial - tangential difference

    e_xx = comp_t + dr * x * x / r2
    e_yy = comp_t + dr * y * y / r2
    e_zz = comp_t + dr * z * z / r2
    e_xy = dr * x * y / r2
    e_xz = dr * x * z / r2
    e_yz = dr * y * z / r2

    return mp.Medium(
        epsilon_diag=mp.Vector3(e_xx, e_yy, e_zz),
        epsilon_offdiag=mp.Vector3(e_xy, e_xz, e_yz),
        mu_diag=mp.Vector3(e_xx, e_yy, e_zz),
        mu_offdiag=mp.Vector3(e_xy, e_xz, e_yz),
    )


# =====================================================================
# Plane wave source (Ez polarization, propagating in +x)
# =====================================================================

# is_integrated=True is required for planewave sources extending into PML
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s, s),
    )
]

# Representative magnetic material so Meep allocates mu tensor storage
# when the material function returns magnetic media.
extra_materials_list = [mp.Medium(mu=R2 / (R2 - R1))]

# =====================================================================
# Helper: set up 6-face flux box monitors (3D closed surface)
# =====================================================================


def add_flux_box(sim):
    """Add 6 flux monitors forming a closed box of half-size box_half."""
    bh = box_half
    regions = [
        mp.FluxRegion(center=mp.Vector3(x=-bh), size=mp.Vector3(0, 2 * bh, 2 * bh)),
        mp.FluxRegion(center=mp.Vector3(x=+bh), size=mp.Vector3(0, 2 * bh, 2 * bh)),
        mp.FluxRegion(center=mp.Vector3(y=-bh), size=mp.Vector3(2 * bh, 0, 2 * bh)),
        mp.FluxRegion(center=mp.Vector3(y=+bh), size=mp.Vector3(2 * bh, 0, 2 * bh)),
        mp.FluxRegion(center=mp.Vector3(z=-bh), size=mp.Vector3(2 * bh, 2 * bh, 0)),
        mp.FluxRegion(center=mp.Vector3(z=+bh), size=mp.Vector3(2 * bh, 2 * bh, 0)),
    ]
    return [sim.add_flux(fcen, df, nfreq, fr) for fr in regions]


# =====================================================================
# Helper: DFT field monitor (2D slice at z=0 for visualization)
# =====================================================================

vis_half = R2 + pad
dft_center = mp.Vector3()
dft_size = mp.Vector3(2 * vis_half, 2 * vis_half, 0)  # z=0 cross-section


def add_dft_monitor(sim):
    """Add a DFT field monitor for the z=0 cross-section."""
    return sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=dft_center, size=dft_size)


# =====================================================================
# Simulation runner
# =====================================================================


def run_simulation(geometry=None, extra_materials=None, minus_flux_data=None, label=""):
    """
    Run a 3D simulation and return (fluxes, flux_data, ez_dft_array).

    Parameters
    ----------
    geometry : list of GeometricObject, optional
    extra_materials : list of Medium, optional
    minus_flux_data : list of flux data from empty run for subtraction
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
        symmetries=symmetries,
        geometry=geometry or [],
        extra_materials=extra_materials or [],
    )

    # 6-face flux monitors
    flux_monitors = add_flux_box(sim)

    # Subtract incident flux if reference data provided
    if minus_flux_data is not None:
        for mon, data in zip(flux_monitors, minus_flux_data):
            sim.load_minus_flux_data(mon, data)

    # DFT field monitor (z=0 slice)
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

    # Collect DFT field (z=0 cross-section)
    ez_dft = sim.get_dft_array(dft_mon, mp.Ez, 0)

    sim.reset_meep()
    return fluxes, flux_data, ez_dft


# =====================================================================
# Run 1: Empty reference (no scatterer)
# =====================================================================

empty_fluxes, empty_flux_data, ez_empty = run_simulation(label="Empty reference")

# Incident flux for normalization (flux through the x1 face)
incident_flux = empty_fluxes[0][0]

# =====================================================================
# Run 2: Bare spherical cow (no cloak)
# =====================================================================

geom_bare = [
    mp.Sphere(radius=R1, material=mp.Medium(index=n_cow)),
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
    mp.Sphere(radius=R2, material=pendry_cloak_material),
    mp.Sphere(radius=R1, material=mp.Medium(index=n_cow)),
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

# Net scattered power through 6-face closed box
# (following mie_scattering.py sign convention)
# P_raw = x1 - x2 + y1 - y2 + z1 - z2
# sigma_scat = -P_raw / I_incident

intensity = abs(incident_flux) / (2 * box_half) ** 2


def scattering_cross_section(fluxes):
    """Compute scattering cross section from the 6-face flux data."""
    raw = (
        np.asarray(fluxes[0])
        - np.asarray(fluxes[1])
        + np.asarray(fluxes[2])
        - np.asarray(fluxes[3])
        + np.asarray(fluxes[4])
        - np.asarray(fluxes[5])
    )
    return -raw / intensity


sigma_bare = scattering_cross_section(bare_fluxes)
sigma_cloak = scattering_cross_section(cloak_fluxes)

# Scattering efficiency Q = sigma / geometric_cross_section
# In 3D the geometric cross section is pi*R1^2
geom_xs = np.pi * R1**2
Q_bare = sigma_bare / geom_xs
Q_cloak = sigma_cloak / geom_xs

# =====================================================================
# Print results
# =====================================================================

if mp.am_master():
    print("\n")
    print("=" * 60)
    print("    SPHERICAL COW CLOAK (3D) \u2014 RESULTS")
    print("=" * 60)
    print(f"  Cow radius (R1):         {R1:.2f}")
    print(f"  Cloak outer radius (R2): {R2:.2f}")
    print(f"  Cow refractive index:    {n_cow:.2f}")
    print(f"  Wavelength:              {1/fcen:.2f}")
    print(f"  Resolution:              {resolution} px/unit")
    print(f"  Regularization (delta):  {delta:.3f}")
    print(f"  Cell size:               {s:.1f} x {s:.1f} x {s:.1f}")
    print(f"  Symmetries:              Mirror(Y), Mirror(Z, phase=-1)")
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
    # Save raw field data (z=0 cross-section)
    np.savez(
        "spherical_cow_cloak_fields.npz",
        ez_empty=ez_empty,
        ez_bare=ez_bare,
        ez_cloak=ez_cloak,
        x=np.linspace(-vis_half, vis_half, ez_empty.shape[0]),
        y=np.linspace(-vis_half, vis_half, ez_empty.shape[1]),
        sigma_bare=sigma_bare,
        sigma_cloak=sigma_cloak,
        Q_bare=Q_bare,
        Q_cloak=Q_cloak,
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
            "Spherical Cow Cloak (3D) \u2014 z=0 Cross-Section\n"
            f"\u03bb = {1/fcen:.2f}, R\u2081 = {R1}, R\u2082 = {R2}, "
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
