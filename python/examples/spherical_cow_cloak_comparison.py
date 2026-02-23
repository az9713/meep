#!/usr/bin/env python3
"""
Four-Approach Cloaking Comparison for the Spherical Cow
=======================================================

Compares four electromagnetic cloaking strategies, demonstrating which
approaches are compatible with FDTD simulation in Meep:

  1. Scattering-Cancellation (Alu-Engheta) -- TRUE 3D SPHERE
     Uses an isotropic sub-unity-epsilon shell to cancel dipole scattering.
     No anisotropy or singularities: fully Yee-grid compatible.

  2. Reduced-Parameter 2D Pendry Cloak (cylinder)
     Transformation-optics with reduced-parameter mu tensor (Cai et al.,
     2007). Avoids the divergent tangential component that breaks FDTD.

  3. Multi-Shell Discrete Cloak (cylinder)
     Staircase approximation of the reduced Pendry profile using N
     concentric shells. Natural regularization via discretization.

  4. Carpet (Ground-Plane) Cloak
     Hides a triangular bump on a PEC mirror using quasi-conformal mapping.
     Nearly isotropic material: excellent FDTD compatibility.

Usage:
  python spherical_cow_cloak_comparison.py --method all --quick
  python spherical_cow_cloak_comparison.py --method cancellation
  python spherical_cow_cloak_comparison.py --method all --production

References:
  [1] A. Alu, N. Engheta, "Achieving transparency with plasmonic and
      metamaterial coatings," Phys. Rev. E 72, 016623 (2005).
  [2] J.B. Pendry et al., "Controlling Electromagnetic Fields,"
      Science 312, 1780-1782 (2006).
  [3] J. Li, J.B. Pendry, "Hiding under the Carpet: A New Strategy
      for Cloaking," Phys. Rev. Lett. 101, 203901 (2008).
"""

import argparse
import math
import sys
import time

import numpy as np
from scipy import special

import meep as mp


# =====================================================================
# CLI
# =====================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Four-approach cloaking comparison for the spherical cow"
    )
    parser.add_argument(
        "--method",
        choices=["cancellation", "pendry2d", "multishell", "carpet", "all"],
        default="all",
        help="Which cloaking approach to run (default: all)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (res=16, nfreq=1)"
    )
    parser.add_argument(
        "--production", action="store_true", help="Production mode (res=32, nfreq=21)"
    )
    parser.add_argument(
        "--resolution", type=int, default=None, help="Override resolution"
    )
    parser.add_argument(
        "--n-cow", type=float, default=1.5, help="Cow refractive index (default: 1.5)"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plot generation"
    )
    return parser.parse_args()


# =====================================================================
# Simulation parameters
# =====================================================================


class SimParams:
    """Container for all simulation parameters."""

    def __init__(self, args):
        self.n_cow = args.n_cow
        self.eps_cow = self.n_cow ** 2
        self.R1 = 1.0  # Cow radius
        self.R2 = 2.0  # Cloak outer radius
        self.fcen = 0.3
        self.df = 0.1

        if args.production:
            self.resolution = 32
            self.dpml = 2.0
            self.nfreq = 21
            self.pad = 4.0
            self.n_shells = 20
        elif args.quick:
            self.resolution = 16
            self.dpml = 1.5
            self.nfreq = 1
            self.pad = 3.0
            self.n_shells = 8
        else:
            self.resolution = 20
            self.dpml = 1.5
            self.nfreq = 1
            self.pad = 3.0
            self.n_shells = 12

        if args.resolution is not None:
            self.resolution = args.resolution

        # Carpet cloak parameters
        self.h_bump = 0.3
        self.w_bump = 2.0
        self.H_cloak = 1.5


# =====================================================================
# Inline 3D Mie Theory
# =====================================================================


def mie_3d_sphere_qsca(radius, n_sphere, freqs):
    """
    Compute 3D Mie scattering efficiency Q_sca for a dielectric sphere.

    Uses Riccati-Bessel functions via scipy.special.spherical_jn/yn.

    Parameters
    ----------
    radius : float
        Sphere radius (Meep units).
    n_sphere : float
        Refractive index of the sphere.
    freqs : array_like
        Frequencies (Meep units, c=1).

    Returns
    -------
    Q_sca : ndarray
        Scattering efficiency at each frequency.
    """
    freqs = np.atleast_1d(freqs)
    Q_sca = np.zeros(len(freqs))

    for fi, f in enumerate(freqs):
        k = 2 * np.pi * f
        x = k * radius
        mx = n_sphere * x

        if x < 1e-10:
            continue

        l_max = int(x + 4 * x ** (1.0 / 3.0) + 2)
        l_max = max(l_max, 10)
        qsca = 0.0

        for l in range(1, l_max + 1):
            jl_x = special.spherical_jn(l, x)
            jl_mx = special.spherical_jn(l, mx)
            yl_x = special.spherical_yn(l, x)
            djl_x = special.spherical_jn(l, x, derivative=True)
            djl_mx = special.spherical_jn(l, mx, derivative=True)
            dyl_x = special.spherical_yn(l, x, derivative=True)

            # Riccati-Bessel functions
            psi_x = x * jl_x
            psi_mx = mx * jl_mx
            dpsi_x = jl_x + x * djl_x
            dpsi_mx = jl_mx + mx * djl_mx

            hl_x = jl_x + 1j * yl_x
            dhl_x = djl_x + 1j * dyl_x
            xi_x = x * hl_x
            dxi_x = hl_x + x * dhl_x

            m = n_sphere
            a_l = (m * psi_mx * dpsi_x - psi_x * dpsi_mx) / (
                m * psi_mx * dxi_x - xi_x * dpsi_mx
            )
            b_l = (psi_mx * dpsi_x - m * psi_x * dpsi_mx) / (
                psi_mx * dxi_x - m * xi_x * dpsi_mx
            )
            qsca += (2 * l + 1) * (abs(a_l) ** 2 + abs(b_l) ** 2)

        Q_sca[fi] = (2.0 / x ** 2) * qsca

    return Q_sca


# =====================================================================
# Inline 2D Mie Theory (TM polarization, infinite cylinder)
# =====================================================================


def mie_2d_cylinder_qsca(radius, n_cylinder, freqs):
    """
    Compute 2D Mie scattering efficiency Q_sca for a dielectric cylinder
    (TM polarization: Ez, Hx, Hy).

    Parameters
    ----------
    radius : float
        Cylinder radius (Meep units).
    n_cylinder : float
        Refractive index.
    freqs : array_like
        Frequencies (Meep units).

    Returns
    -------
    Q_sca : ndarray
        Scattering efficiency at each frequency.
    """
    freqs = np.atleast_1d(freqs)
    Q_sca = np.zeros(len(freqs))

    for fi, f in enumerate(freqs):
        k = 2 * np.pi * f
        x = k * radius
        mx = n_cylinder * x

        if x < 1e-10:
            continue

        m_max = int(x + 4 * x ** (1.0 / 3.0) + 10)
        m_max = max(m_max, 15)

        qsca = 0.0
        m = n_cylinder

        for n in range(0, m_max + 1):
            eps_n = 1.0 if n == 0 else 2.0

            Jn_x = special.jv(n, x)
            Jn_mx = special.jv(n, mx)
            dJn_x = special.jvp(n, x)
            dJn_mx = special.jvp(n, mx)
            Hn_x = special.hankel1(n, x)
            dHn_x = special.h1vp(n, x)

            num = m * Jn_mx * dJn_x - Jn_x * dJn_mx
            den = m * Jn_mx * dHn_x - Hn_x * dJn_mx
            b_n = num / den
            qsca += eps_n * abs(b_n) ** 2

        Q_sca[fi] = (2.0 / x) * qsca

    return Q_sca


# =====================================================================
# Shared: flux box helpers
# =====================================================================


def add_flux_box_2d(sim, fcen, df, nfreq, box_half):
    """Add 4-face flux box for 2D scattering measurement."""
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


def add_flux_box_3d(sim, fcen, df, nfreq, box_half):
    """Add 6-face flux box for 3D scattering measurement."""
    bh = box_half
    regions = [
        mp.FluxRegion(
            center=mp.Vector3(x=-bh), size=mp.Vector3(0, 2 * bh, 2 * bh)
        ),
        mp.FluxRegion(
            center=mp.Vector3(x=+bh), size=mp.Vector3(0, 2 * bh, 2 * bh)
        ),
        mp.FluxRegion(
            center=mp.Vector3(y=-bh), size=mp.Vector3(2 * bh, 0, 2 * bh)
        ),
        mp.FluxRegion(
            center=mp.Vector3(y=+bh), size=mp.Vector3(2 * bh, 0, 2 * bh)
        ),
        mp.FluxRegion(
            center=mp.Vector3(z=-bh), size=mp.Vector3(2 * bh, 2 * bh, 0)
        ),
        mp.FluxRegion(
            center=mp.Vector3(z=+bh), size=mp.Vector3(2 * bh, 2 * bh, 0)
        ),
    ]
    return [sim.add_flux(fcen, df, nfreq, fr) for fr in regions]


def compute_scattering_cs_2d(fluxes, incident_flux, box_half):
    """Compute scattering cross section from 4-face 2D flux box."""
    incident_flux = np.atleast_1d(incident_flux)
    intensity = np.abs(incident_flux) / (2 * box_half)
    raw = (
        np.asarray(fluxes[0])
        - np.asarray(fluxes[1])
        + np.asarray(fluxes[2])
        - np.asarray(fluxes[3])
    )
    return -raw / intensity


def compute_scattering_cs_3d(fluxes, incident_flux, box_half):
    """Compute scattering cross section from 6-face 3D flux box."""
    incident_flux = np.atleast_1d(incident_flux)
    intensity = np.abs(incident_flux) / (2 * box_half) ** 2
    raw = (
        np.asarray(fluxes[0])
        - np.asarray(fluxes[1])
        + np.asarray(fluxes[2])
        - np.asarray(fluxes[3])
        + np.asarray(fluxes[4])
        - np.asarray(fluxes[5])
    )
    return -raw / intensity


# =====================================================================
# Generic simulation runners
# =====================================================================


def run_sim_2d(params, geometry=None, extra_materials=None,
               minus_flux_data=None, label=""):
    """
    Run a 2D simulation and return (fluxes, flux_data, ez_dft).

    Uses a plane-wave Ez source propagating in +x with a closed 4-face
    flux box and a DFT field monitor.
    """
    s = 2 * (params.R2 + params.pad + params.dpml)
    cell_size = mp.Vector3(s, s, 0)
    pml_layers = [mp.PML(thickness=params.dpml)]
    box_half = params.R2 + 0.5
    vis_half = params.R2 + params.pad

    sources = [
        mp.Source(
            mp.GaussianSource(params.fcen, fwidth=params.df, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(-0.5 * s + params.dpml),
            size=mp.Vector3(0, s),
        )
    ]

    if mp.am_master():
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

    sim = mp.Simulation(
        resolution=params.resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=mp.Vector3(),
        geometry=geometry or [],
        extra_materials=extra_materials or [],
    )

    flux_monitors = add_flux_box_2d(
        sim, params.fcen, params.df, params.nfreq, box_half
    )

    if minus_flux_data is not None:
        for mon, data in zip(flux_monitors, minus_flux_data):
            sim.load_minus_flux_data(mon, data)

    dft_mon = sim.add_dft_fields(
        [mp.Ez], params.fcen, 0, 1,
        center=mp.Vector3(), size=mp.Vector3(2 * vis_half, 2 * vis_half),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ez, mp.Vector3(params.R2 + 1, 0), 1e-6
        )
    )

    fluxes = [mp.get_fluxes(m) for m in flux_monitors]
    flux_data = [sim.get_flux_data(m) for m in flux_monitors]
    ez_dft = sim.get_dft_array(dft_mon, mp.Ez, 0)

    sim.reset_meep()
    return fluxes, flux_data, ez_dft


def run_sim_3d(params, geometry=None, extra_materials=None,
               minus_flux_data=None, label=""):
    """
    Run a 3D simulation and return (fluxes, flux_data, ez_dft_xy).

    Uses Mirror(Y) and Mirror(Z, phase=-1) symmetries for 4x speedup.
    DFT monitor captures the XY cross-section at z=0.
    """
    s = 2 * (params.R2 + params.pad + params.dpml)
    cell_size = mp.Vector3(s, s, s)
    pml_layers = [mp.PML(thickness=params.dpml)]
    box_half = params.R2 + 0.5
    vis_half = params.R2 + params.pad

    sources = [
        mp.Source(
            mp.GaussianSource(params.fcen, fwidth=params.df, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(-0.5 * s + params.dpml),
            size=mp.Vector3(0, s, s),
        )
    ]

    symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

    if mp.am_master():
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

    sim = mp.Simulation(
        resolution=params.resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=mp.Vector3(),
        symmetries=symmetries,
        geometry=geometry or [],
        extra_materials=extra_materials or [],
    )

    flux_monitors = add_flux_box_3d(
        sim, params.fcen, params.df, params.nfreq, box_half
    )

    if minus_flux_data is not None:
        for mon, data in zip(flux_monitors, minus_flux_data):
            sim.load_minus_flux_data(mon, data)

    # 2D cross-section at z=0
    dft_mon = sim.add_dft_fields(
        [mp.Ez], params.fcen, 0, 1,
        center=mp.Vector3(), size=mp.Vector3(2 * vis_half, 2 * vis_half, 0),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ez, mp.Vector3(params.R2 + 1, 0, 0), 1e-6
        )
    )

    fluxes = [mp.get_fluxes(m) for m in flux_monitors]
    flux_data = [sim.get_flux_data(m) for m in flux_monitors]
    ez_dft = sim.get_dft_array(dft_mon, mp.Ez, 0)

    sim.reset_meep()
    return fluxes, flux_data, ez_dft


# =====================================================================
# Shared 2D reference runs (used by Pendry 2D and Multi-Shell)
# =====================================================================


def run_empty_and_bare_2d(params):
    """Run empty and bare-cylinder 2D simulations, returning reference data."""
    # Empty reference
    empty_fluxes, empty_flux_data, ez_empty = run_sim_2d(
        params, label="2D Empty reference"
    )
    incident_flux = np.asarray(empty_fluxes[0])  # array for broadband support

    # Bare cylinder (cow only)
    geom_bare = [
        mp.Cylinder(
            radius=params.R1, material=mp.Medium(epsilon=params.eps_cow)
        )
    ]
    bare_fluxes, _, ez_bare = run_sim_2d(
        params,
        geometry=geom_bare,
        minus_flux_data=empty_flux_data,
        label=f"2D Bare cylinder (n={params.n_cow})",
    )

    box_half = params.R2 + 0.5
    sigma_bare = compute_scattering_cs_2d(bare_fluxes, incident_flux, box_half)
    geom_xs = 2 * params.R1
    Q_bare = sigma_bare[0] / geom_xs

    # Mie theory validation
    Q_mie = mie_2d_cylinder_qsca(params.R1, params.n_cow, [params.fcen])[0]

    if mp.am_master():
        print(f"\n  2D Bare Q_sca = {Q_bare:.4f}")
        print(f"  2D Mie theory = {Q_mie:.4f}")
        err = abs(Q_bare - Q_mie) / max(abs(Q_mie), 1e-10) * 100
        print(f"  Agreement:      {err:.1f}%")

    return {
        "empty_flux_data": empty_flux_data,
        "ez_empty": ez_empty,
        "ez_bare": ez_bare,
        "Q_bare": float(Q_bare),
        "Q_mie": float(Q_mie),
        "incident_flux": incident_flux,
    }


# =====================================================================
# Approach 1: Scattering-Cancellation (3D Sphere)
# =====================================================================


def cancellation_eps_shell_3d(eps_cow, ratio_ab):
    """
    Solve the 3D dipole (l=1) cancellation equation for eps_shell.

    The condition for zero dipole scattering from a core-shell sphere:
      (e-1)(E+2e) + g*(E-e)(e+2) = 0
    where e = eps_shell, E = eps_cow, g = (a/b)^3.

    Expanding to a quadratic:
      (2-g)*e^2 + (1+g)*(E-2)*e + E*(2g-1) = 0

    Returns the root with 0 < eps_shell < 1 (sub-unity for cloaking).
    """
    E = eps_cow
    g = ratio_ab ** 3

    a_coeff = 2.0 - g
    b_coeff = (1.0 + g) * (E - 2.0)
    c_coeff = E * (2.0 * g - 1.0)

    disc = b_coeff ** 2 - 4 * a_coeff * c_coeff
    if disc < 0:
        raise ValueError("No real solution for cancellation condition")

    e1 = (-b_coeff + math.sqrt(disc)) / (2 * a_coeff)
    e2 = (-b_coeff - math.sqrt(disc)) / (2 * a_coeff)

    # Prefer the sub-unity root (physical for cloaking)
    for e in [e1, e2]:
        if 0 < e < 1:
            return e

    # Fallback: positive root
    return e1 if e1 > 0 else e2


def run_cancellation_3d(params):
    """
    Run the 3D scattering-cancellation cloak simulation.

    This is the headline result: a TRUE spherical cow, cloaked using an
    isotropic sub-unity-epsilon shell that cancels dipole scattering.
    No anisotropy, no singularities, no Yee grid instability.
    """
    t0 = time.time()

    eps_shell = cancellation_eps_shell_3d(params.eps_cow, params.R1 / params.R2)
    if mp.am_master():
        print(f"\n  Scattering-Cancellation Design:")
        print(f"    eps_cow   = {params.eps_cow:.3f}")
        print(f"    eps_shell = {eps_shell:.4f}")
        print(f"    R1/R2     = {params.R1/params.R2:.3f}")

    # Run 1: Empty 3D reference
    empty_fluxes, empty_flux_data, ez_empty = run_sim_3d(
        params, label="3D Empty reference"
    )
    incident_flux = np.asarray(empty_fluxes[0])  # array for broadband support

    # Run 2: Bare sphere (cow only)
    geom_bare = [
        mp.Sphere(
            radius=params.R1, material=mp.Medium(epsilon=params.eps_cow)
        )
    ]
    bare_fluxes, _, ez_bare = run_sim_3d(
        params,
        geometry=geom_bare,
        minus_flux_data=empty_flux_data,
        label=f"3D Bare sphere (n={params.n_cow})",
    )

    # Run 3: Cloaked sphere (isotropic shell + cow)
    shell_material = mp.Medium(epsilon=eps_shell)
    geom_cloak = [
        mp.Sphere(radius=params.R2, material=shell_material),
        mp.Sphere(
            radius=params.R1, material=mp.Medium(epsilon=params.eps_cow)
        ),
    ]
    cloak_fluxes, _, ez_cloak = run_sim_3d(
        params,
        geometry=geom_cloak,
        minus_flux_data=empty_flux_data,
        label="3D Scattering-cancellation cloak",
    )

    # Compute scattering
    box_half = params.R2 + 0.5
    sigma_bare = compute_scattering_cs_3d(bare_fluxes, incident_flux, box_half)
    sigma_cloak = compute_scattering_cs_3d(
        cloak_fluxes, incident_flux, box_half
    )
    geom_xs = np.pi * params.R1 ** 2  # 3D: disk area
    Q_bare = sigma_bare / geom_xs
    Q_cloak = sigma_cloak / geom_xs

    # Mie theory reference for bare sphere
    Q_mie = mie_3d_sphere_qsca(params.R1, params.n_cow, [params.fcen])

    elapsed = time.time() - t0
    vis_half = params.R2 + params.pad

    result = {
        "name": "Scattering-Cancellation (3D)",
        "dim": "3D",
        "object": "sphere",
        "Q_bare": float(Q_bare[0]),
        "Q_cloak": float(Q_cloak[0]),
        "Q_mie": float(Q_mie[0]),
        "eps_shell": eps_shell,
        "ez_empty": ez_empty,
        "ez_bare": ez_bare,
        "ez_cloak": ez_cloak,
        "extent": [-vis_half, vis_half, -vis_half, vis_half],
        "R1": params.R1,
        "R2": params.R2,
        "time": elapsed,
    }

    if mp.am_master():
        _print_approach_result(result)

    return result


# =====================================================================
# Approach 2: Enhanced 2D Pendry Cloak
# =====================================================================


def make_pendry_material_func(R1, R2, delta):
    """
    Create a reduced-parameter Pendry cloak material function for TM.

    Uses the "reduced parameters" approach (Cai et al., 2007):
    for TM polarization (Ez, Hx, Hy), the relevant parameters are
    mu_rr, mu_tt (in-plane) and eps_zz (out-of-plane).

    Full Pendry:
        mu_r = (r-R1)/r -> 0          (bounded)
        mu_t = r/(r-R1) -> infinity    (DIVERGENT, breaks FDTD)
        eps_z = (R2/(R2-R1))^2*(r-R1)/r

    Reduced (TM):
        mu_r = ((r-R1)/r)^2           (bounded in [0,1])
        mu_t = 1                       (constant, no divergence)
        eps_z = (R2/(R2-R1))^2        (constant)

    This sacrifices impedance matching (causing some reflections)
    but preserves the ray-bending trajectory and avoids the
    catastrophic singularity that makes full Pendry unstable in FDTD.
    """

    R_ratio_sq = (R2 / (R2 - R1)) ** 2  # constant eps_z
    delta_sq = delta * delta

    def material_func(p):
        x, y = p.x, p.y
        r = math.sqrt(x * x + y * y)

        if r <= R1 or r >= R2:
            return mp.air

        rp = max(r - R1, delta)

        # Reduced TM parameters: bounded, no divergence
        mu_r = max((rp / r) ** 2, delta_sq)  # in (0, 1]
        mu_t = 1.0  # constant, avoids infinity

        # Rotate mu from cylindrical to Cartesian
        theta = math.atan2(y, x)
        ct = math.cos(theta)
        st = math.sin(theta)
        ct2 = ct * ct
        st2 = st * st
        ctst = ct * st

        mu_xx = mu_r * ct2 + mu_t * st2
        mu_yy = mu_r * st2 + mu_t * ct2
        mu_xy = (mu_r - mu_t) * ctst

        return mp.Medium(
            epsilon_diag=mp.Vector3(1, 1, R_ratio_sq),
            mu_diag=mp.Vector3(mu_xx, mu_yy, 1),
            mu_offdiag=mp.Vector3(mu_xy, 0, 0),
        )

    return material_func


def run_pendry2d(params, ref_2d=None):
    """
    Run the 2D Pendry cloak simulation with small regularization.

    Tries delta=0.02 first; falls back to larger values if unstable.
    """
    t0 = time.time()

    if ref_2d is None:
        ref_2d = run_empty_and_bare_2d(params)

    empty_flux_data = ref_2d["empty_flux_data"]
    ez_empty = ref_2d["ez_empty"]
    ez_bare = ref_2d["ez_bare"]
    Q_bare = ref_2d["Q_bare"]
    incident_flux = ref_2d["incident_flux"]
    Q_mie = ref_2d["Q_mie"]

    # Try progressively larger delta until stable
    ez_cloak = None
    cloak_fluxes = None
    used_delta = None

    for delta in [0.30, 0.40, 0.50]:
        if mp.am_master():
            print(f"\n  Pendry 2D: trying delta = {delta}")

        mat_func = make_pendry_material_func(params.R1, params.R2, delta)
        # Reduced parameters: max mu component is 1.0, but we need
        # extra_materials to tell Meep to allocate mu storage
        extra_mats = [mp.Medium(mu=2)]

        geom_cloak = [
            mp.Cylinder(radius=params.R2, material=mat_func),
            mp.Cylinder(
                radius=params.R1,
                material=mp.Medium(epsilon=params.eps_cow),
            ),
        ]

        try:
            cf, _, ec = run_sim_2d(
                params,
                geometry=geom_cloak,
                extra_materials=extra_mats,
                minus_flux_data=empty_flux_data,
                label=f"2D Pendry cloak (delta={delta})",
            )

            # Check for divergence
            if np.any(np.isnan(ec)) or np.max(np.abs(ec)) > 1e6:
                if mp.am_master():
                    print(f"  delta={delta} diverged, trying larger")
                continue

            cloak_fluxes = cf
            ez_cloak = ec
            used_delta = delta
            break

        except Exception as e:
            if mp.am_master():
                print(f"  delta={delta} failed: {e}")
            continue

    if ez_cloak is None:
        if mp.am_master():
            print("  WARNING: All Pendry 2D attempts failed")
        return None

    box_half = params.R2 + 0.5
    sigma_cloak = compute_scattering_cs_2d(
        cloak_fluxes, incident_flux, box_half
    )
    geom_xs = 2 * params.R1
    Q_cloak = sigma_cloak / geom_xs

    vis_half = params.R2 + params.pad
    elapsed = time.time() - t0

    result = {
        "name": "Pendry 2D (reduced)",
        "dim": "2D",
        "object": "cylinder",
        "Q_bare": float(Q_bare),
        "Q_cloak": float(Q_cloak[0]),
        "Q_mie": float(Q_mie),
        "delta": used_delta,
        "ez_empty": ez_empty,
        "ez_bare": ez_bare,
        "ez_cloak": ez_cloak,
        "extent": [-vis_half, vis_half, -vis_half, vis_half],
        "R1": params.R1,
        "R2": params.R2,
        "time": elapsed,
    }

    if mp.am_master():
        _print_approach_result(result)

    return result


# =====================================================================
# Approach 3: Multi-Shell Discrete Cloak
# =====================================================================


def make_multishell_material_func(R1, R2, n_shells, eps_min=0.30):
    """
    Create a material function for N-shell discrete reduced-parameter cloak.

    Uses the same reduced-parameter approach as the continuous Pendry:
    mu_r = ((r-R1)/r)^2 at each shell midpoint, mu_t = 1, eps_z = constant.
    This avoids the divergent tangential component while preserving
    the staircase approximation of the radial gradient.
    """
    dr = (R2 - R1) / n_shells
    R_ratio_sq = (R2 / (R2 - R1)) ** 2
    eps_min_sq = eps_min * eps_min

    shell_data = []
    for i in range(n_shells):
        r_mid = R1 + (i + 0.5) * dr
        rp = r_mid - R1
        # Reduced parameters: mu_r = (rp/r)^2, mu_t = 1
        mu_r = max((rp / r_mid) ** 2, eps_min_sq)
        r_inner = R1 + i * dr
        r_outer = R1 + (i + 1) * dr
        shell_data.append((r_inner, r_outer, mu_r))

    def material_func(p):
        x, y = p.x, p.y
        r = math.sqrt(x * x + y * y)

        if r <= R1 or r >= R2:
            return mp.air

        # Determine shell index
        idx = int((r - R1) / dr)
        idx = min(idx, n_shells - 1)
        _, _, mu_r = shell_data[idx]
        mu_t = 1.0

        # Rotate mu from cylindrical to Cartesian
        theta = math.atan2(y, x)
        ct = math.cos(theta)
        st = math.sin(theta)
        ct2 = ct * ct
        st2 = st * st
        ctst = ct * st

        mu_xx = mu_r * ct2 + mu_t * st2
        mu_yy = mu_r * st2 + mu_t * ct2
        mu_xy = (mu_r - mu_t) * ctst

        return mp.Medium(
            epsilon_diag=mp.Vector3(1, 1, R_ratio_sq),
            mu_diag=mp.Vector3(mu_xx, mu_yy, 1),
            mu_offdiag=mp.Vector3(mu_xy, 0, 0),
        )

    return material_func


def run_multishell(params, ref_2d=None):
    """Run the multi-shell discrete cloak simulation."""
    t0 = time.time()

    if ref_2d is None:
        ref_2d = run_empty_and_bare_2d(params)

    empty_flux_data = ref_2d["empty_flux_data"]
    ez_empty = ref_2d["ez_empty"]
    ez_bare = ref_2d["ez_bare"]
    Q_bare = ref_2d["Q_bare"]
    incident_flux = ref_2d["incident_flux"]
    Q_mie = ref_2d["Q_mie"]

    n_shells = params.n_shells
    mat_func = make_multishell_material_func(
        params.R1, params.R2, n_shells
    )

    # Reduced parameters: max mu component is 1.0, but we need
    # extra_materials to tell Meep to allocate mu storage
    extra_mats = [mp.Medium(mu=2)]

    geom_cloak = [
        mp.Cylinder(radius=params.R2, material=mat_func),
        mp.Cylinder(
            radius=params.R1,
            material=mp.Medium(epsilon=params.eps_cow),
        ),
    ]

    try:
        cloak_fluxes, _, ez_cloak = run_sim_2d(
            params,
            geometry=geom_cloak,
            extra_materials=extra_mats,
            minus_flux_data=empty_flux_data,
            label=f"2D Multi-shell cloak (N={n_shells})",
        )
    except Exception as e:
        if mp.am_master():
            print(f"  WARNING: Multi-shell simulation failed: {e}")
        return None

    # Check for divergence
    if np.any(np.isnan(ez_cloak)) or np.max(np.abs(ez_cloak)) > 1e6:
        if mp.am_master():
            print("  WARNING: Multi-shell simulation diverged")
        return None

    box_half = params.R2 + 0.5
    sigma_cloak = compute_scattering_cs_2d(
        cloak_fluxes, incident_flux, box_half
    )
    geom_xs = 2 * params.R1
    Q_cloak = sigma_cloak / geom_xs

    vis_half = params.R2 + params.pad
    elapsed = time.time() - t0

    result = {
        "name": f"Multi-Shell (N={n_shells})",
        "dim": "2D",
        "object": "cylinder",
        "Q_bare": float(Q_bare),
        "Q_cloak": float(Q_cloak[0]),
        "Q_mie": float(Q_mie),
        "n_shells": n_shells,
        "ez_empty": ez_empty,
        "ez_bare": ez_bare,
        "ez_cloak": ez_cloak,
        "extent": [-vis_half, vis_half, -vis_half, vis_half],
        "R1": params.R1,
        "R2": params.R2,
        "time": elapsed,
    }

    if mp.am_master():
        _print_approach_result(result)

    return result


# =====================================================================
# Approach 4: Carpet (Ground-Plane) Cloak
# =====================================================================


def make_carpet_material_func(h_bump, w_bump, H_cloak, y_ground):
    """
    Create a material function for the quasi-conformal carpet cloak.

    Uses the quasi-conformal approximation from Li & Pendry (2008):
      eps_z = 1/det(J) = H / (H - g(x))
      mu = 1 (quasi-conformal: the mapping is nearly conformal so mu ~ 1)
    where g(x) is the triangular bump profile and H is the cloak height.
    """

    def material_func(p):
        x, y = p.x, p.y

        # Triangular bump profile
        if abs(x) < w_bump / 2:
            g = h_bump * (1.0 - 2.0 * abs(x) / w_bump)
        else:
            g = 0.0

        # Only apply in cloak region above bump, below H_cloak
        if g < 1e-10:
            return mp.air
        if y < y_ground + g or y > y_ground + H_cloak:
            return mp.air

        eps_val = H_cloak / (H_cloak - g)
        return mp.Medium(epsilon=eps_val)

    return material_func


def run_carpet(params):
    """
    Run the carpet cloak simulation.

    Three sub-simulations compare:
      1. Flat PEC ground (reference reflection)
      2. PEC ground + triangular bump (distorted reflection)
      3. PEC ground + bump + carpet cloak (restored reflection)

    Uses a plane wave from the top propagating downward (-y) to
    reflect off the PEC ground -- the standard carpet cloak geometry.

    Reports fidelity instead of Q_sca:
      fidelity = 1 - ||E_cloak - E_flat||^2 / ||E_bump - E_flat||^2
    """
    t0 = time.time()

    h_bump = params.h_bump
    w_bump = params.w_bump
    H_cloak = params.H_cloak
    dpml = params.dpml
    pad = params.pad

    # Cell geometry -- taller to accommodate downward reflection
    margin_below = 0.5
    reflect_space = 3.0  # space above cloak for reflected wave to develop
    sx = w_bump + 2 * pad + 2 * dpml
    sy = margin_below + dpml + H_cloak + reflect_space + dpml

    # Ground position in cell coordinates
    y_ground = -sy / 2 + dpml + margin_below

    cell_size = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(thickness=dpml)]

    # Source: plane wave from the top, propagating in -y (downward)
    # Placed at the inner PML boundary at the top of the cell
    source_y = sy / 2 - dpml
    sources = [
        mp.Source(
            mp.GaussianSource(
                params.fcen, fwidth=params.df, is_integrated=True
            ),
            component=mp.Ez,
            center=mp.Vector3(0, source_y),
            size=mp.Vector3(sx, 0),
        )
    ]

    # Monitor region: from ground up to just below source
    vis_x = sx / 2 - dpml
    vis_y_lo = y_ground
    vis_y_hi = source_y - 0.5

    def _run_carpet_sim(geometry, label):
        """Run a single carpet sub-simulation, return DFT field."""
        if mp.am_master():
            print(f"\n{'='*60}")
            print(f"Running: {label}")
            print(f"{'='*60}")

        sim = mp.Simulation(
            resolution=params.resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            sources=sources,
            k_point=mp.Vector3(),
            geometry=geometry,
        )

        mon_center = mp.Vector3(0, (vis_y_lo + vis_y_hi) / 2)
        mon_size = mp.Vector3(2 * vis_x, vis_y_hi - vis_y_lo)
        dft_mon = sim.add_dft_fields(
            [mp.Ez], params.fcen, 0, 1,
            center=mon_center, size=mon_size,
        )

        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                20, mp.Ez, mp.Vector3(0, y_ground + H_cloak + 0.5), 1e-6
            )
        )

        ez_dft = sim.get_dft_array(dft_mon, mp.Ez, 0)
        sim.reset_meep()
        return ez_dft

    # Ground PEC: fills from bottom of cell to y_ground
    ground_block = mp.Block(
        center=mp.Vector3(0, (y_ground + (-sy / 2)) / 2),
        size=mp.Vector3(mp.inf, y_ground - (-sy / 2)),
        material=mp.perfect_electric_conductor,
    )

    # Bump PEC: triangle on the ground
    bump_prism = mp.Prism(
        vertices=[
            mp.Vector3(-w_bump / 2, y_ground),
            mp.Vector3(0, y_ground + h_bump),
            mp.Vector3(w_bump / 2, y_ground),
        ],
        height=mp.inf,
        material=mp.perfect_electric_conductor,
    )

    # Carpet cloak material region (below bump PEC priority)
    cloak_block = mp.Block(
        center=mp.Vector3(0, y_ground + H_cloak / 2),
        size=mp.Vector3(w_bump + 1.0, H_cloak),
        material=make_carpet_material_func(
            h_bump, w_bump, H_cloak, y_ground
        ),
    )

    # Run 1: Flat ground (reference)
    ez_flat = _run_carpet_sim([ground_block], "Carpet: flat ground")

    # Run 2: Ground + bump
    ez_bump = _run_carpet_sim(
        [ground_block, bump_prism], "Carpet: bare bump"
    )

    # Run 3: Cloak + ground + bump (cloak has lower priority than PEC)
    ez_cloaked = _run_carpet_sim(
        [cloak_block, ground_block, bump_prism],
        "Carpet: cloaked bump",
    )

    # Compute fidelity
    diff_cloak = np.real(ez_cloaked) - np.real(ez_flat)
    diff_bump = np.real(ez_bump) - np.real(ez_flat)
    denom = np.sum(diff_bump ** 2)
    if denom > 1e-20:
        fidelity = 1.0 - np.sum(diff_cloak ** 2) / denom
    else:
        fidelity = 1.0

    elapsed = time.time() - t0
    extent = [-vis_x, vis_x, vis_y_lo, vis_y_hi]

    result = {
        "name": "Carpet Cloak",
        "dim": "2D",
        "object": "ground bump",
        "Q_bare": None,
        "Q_cloak": None,
        "Q_mie": None,
        "fidelity": fidelity,
        "ez_empty": ez_flat,
        "ez_bare": ez_bump,
        "ez_cloak": ez_cloaked,
        "extent": extent,
        "R1": None,
        "R2": None,
        "h_bump": h_bump,
        "w_bump": w_bump,
        "H_cloak": H_cloak,
        "y_ground": y_ground,
        "time": elapsed,
    }

    if mp.am_master():
        _print_carpet_result(result)

    return result


# =====================================================================
# Result printing
# =====================================================================


def _print_approach_result(result):
    """Print results for a scattering-based approach."""
    Q_b = abs(result["Q_bare"])
    Q_c = abs(result["Q_cloak"])
    ratio = Q_b / max(Q_c, 1e-10)
    reduction_pct = (1 - Q_c / max(Q_b, 1e-10)) * 100

    print(f"\n  {'='*50}")
    print(f"  {result['name']}")
    print(f"  {'='*50}")
    print(f"  Q_bare  (FDTD):  {result['Q_bare']:.4f}")
    print(f"  Q_cloak (FDTD):  {result['Q_cloak']:.4f}")
    if result.get("Q_mie") is not None:
        print(f"  Q_bare  (Mie):   {result['Q_mie']:.4f}")
    print(f"  Reduction:       {reduction_pct:.1f}% ({ratio:.1f}x)")
    print(f"  Time:            {result['time']:.1f}s")


def _print_carpet_result(result):
    """Print results for the carpet cloak."""
    print(f"\n  {'='*50}")
    print(f"  {result['name']}")
    print(f"  {'='*50}")
    print(f"  Fidelity:        {result['fidelity']:.4f}")
    print(f"  Time:            {result['time']:.1f}s")


def print_comparison_table(results):
    """Print a summary comparison table of all approaches."""
    if not mp.am_master():
        return

    print(f"\n{'='*72}")
    print(f"  SPHERICAL COW CLOAKING COMPARISON -- SUMMARY")
    print(f"{'='*72}")
    print(
        f"  {'Method':<30s} {'Dim':>4s} {'Q_bare':>8s} "
        f"{'Q_cloak':>8s} {'Reduce':>8s} {'Time':>6s}"
    )
    print(f"  {'-'*70}")

    for key, r in results.items():
        if r is None:
            continue
        if r.get("Q_bare") is not None and r.get("Q_cloak") is not None:
            Q_b = r["Q_bare"]
            Q_c = r["Q_cloak"]
            if abs(Q_b) > 1e-10:
                red = f"{abs(Q_b)/max(abs(Q_c),1e-10):.1f}x"
            else:
                red = "N/A"
            print(
                f"  {r['name']:<30s} {r['dim']:>4s} "
                f"{Q_b:8.4f} {Q_c:8.4f} {red:>8s} {r['time']:5.0f}s"
            )
        elif r.get("fidelity") is not None:
            fid = r["fidelity"]
            print(
                f"  {r['name']:<30s} {r['dim']:>4s} "
                f"{'N/A':>8s} {'N/A':>8s} "
                f"{'F='+f'{fid:.2f}':>8s} {r['time']:5.0f}s"
            )

    print(f"{'='*72}")


# =====================================================================
# Plotting
# =====================================================================


def plot_approach(result, filename, params):
    """Generate a 3-panel field comparison plot for one approach."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ez_empty = result["ez_empty"]
    ez_bare = result["ez_bare"]
    ez_cloak = result["ez_cloak"]
    extent = result["extent"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(
        np.max(np.abs(np.real(ez_empty))),
        np.max(np.abs(np.real(ez_bare))),
        np.max(np.abs(np.real(ez_cloak))),
    )
    if vmax < 1e-20:
        vmax = 1.0

    if result["name"] == "Carpet Cloak":
        titles = ["Flat Ground (ref)", "Bare Bump", "Cloaked Bump"]
    else:
        obj = result["object"]
        titles = ["Empty (reference)", f"Bare {obj}", f"Cloaked {obj}"]

    for ax, data, title in zip(
        axes, [ez_empty, ez_bare, ez_cloak], titles
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

        # Draw object outlines for cylindrical/spherical approaches
        if result.get("R1") is not None:
            cow_circle = plt.Circle(
                (0, 0),
                result["R1"],
                fill=False,
                color="black",
                linewidth=1.5,
                linestyle="--",
            )
            ax.add_patch(cow_circle)

        if result.get("R2") is not None and "Cloak" in title:
            cloak_circle = plt.Circle(
                (0, 0),
                result["R2"],
                fill=False,
                color="black",
                linewidth=1.5,
                linestyle=":",
            )
            ax.add_patch(cloak_circle)

        # Draw shell boundaries for multi-shell
        if "Multi-Shell" in result["name"] and "Cloak" in title:
            ns = result.get("n_shells", 0)
            if ns > 0:
                dr = (result["R2"] - result["R1"]) / ns
                for i in range(1, ns):
                    ri = result["R1"] + i * dr
                    c = plt.Circle(
                        (0, 0),
                        ri,
                        fill=False,
                        color="gray",
                        linewidth=0.5,
                        linestyle=":",
                        alpha=0.5,
                    )
                    ax.add_patch(c)

        # Draw ground/bump outlines for carpet cloak
        if result.get("y_ground") is not None:
            y_g = result["y_ground"]
            ax.axhline(y=y_g, color="gray", linewidth=2)
            if "Bump" in title or "Cloak" in title:
                w = result["w_bump"]
                h = result["h_bump"]
                bump_x = [-w / 2, 0, w / 2]
                bump_y = [y_g, y_g + h, y_g]
                ax.plot(bump_x, bump_y, "k--", linewidth=1.5)

    fig.colorbar(im, ax=axes, label="Re(Ez)", shrink=0.8)

    if result.get("Q_bare") is not None:
        Q_b = result["Q_bare"]
        Q_c = result["Q_cloak"]
        ratio = abs(Q_b) / max(abs(Q_c), 1e-10)
        suptitle = (
            f"{result['name']}\n"
            f"Q_bare = {Q_b:.3f}, Q_cloak = {Q_c:.3f} "
            f"({ratio:.1f}x reduction)"
        )
    elif result.get("fidelity") is not None:
        suptitle = (
            f"{result['name']}\n"
            f"Fidelity = {result['fidelity']:.3f}"
        )
    else:
        suptitle = result["name"]

    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {filename}")


def plot_summary(results, filename):
    """Generate a summary bar chart comparing all approaches."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = []
    values = []
    colors = []
    annotations = []

    for key, r in results.items():
        if r is None:
            continue
        names.append(r["name"])

        if r.get("Q_bare") is not None and r.get("Q_cloak") is not None:
            Q_b = abs(r["Q_bare"])
            Q_c = abs(r["Q_cloak"])
            ratio = Q_b / max(Q_c, 1e-10)
            values.append(ratio)
            colors.append("forestgreen" if ratio > 1.5 else "orange")
            annotations.append(f"{ratio:.1f}x")
        elif r.get("fidelity") is not None:
            fid = r["fidelity"]
            # Scale fidelity to comparable range for visualization
            values.append(max(fid * 5, 0.1))
            colors.append("steelblue")
            annotations.append(f"F={fid:.2f}")

    if not names:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="black")

    for bar, label in zip(bars, annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            label,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Scattering Reduction Factor")
    ax.set_title(
        "Spherical Cow Cloaking: Method Comparison\n"
        "(green = scattering reduction, blue = fidelity metric)"
    )
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No reduction")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary plot saved: {filename}")


# =====================================================================
# Data saving
# =====================================================================


def save_data(results, params):
    """Save all results to a .npz file."""
    if not mp.am_master():
        return

    save_dict = {
        "R1": params.R1,
        "R2": params.R2,
        "n_cow": params.n_cow,
        "fcen": params.fcen,
        "resolution": params.resolution,
    }

    for key, r in results.items():
        if r is None:
            continue
        prefix = key
        save_dict[f"{prefix}_ez_empty"] = r["ez_empty"]
        save_dict[f"{prefix}_ez_bare"] = r["ez_bare"]
        save_dict[f"{prefix}_ez_cloak"] = r["ez_cloak"]
        if r.get("Q_bare") is not None:
            save_dict[f"{prefix}_Q_bare"] = r["Q_bare"]
        if r.get("Q_cloak") is not None:
            save_dict[f"{prefix}_Q_cloak"] = r["Q_cloak"]
        if r.get("fidelity") is not None:
            save_dict[f"{prefix}_fidelity"] = r["fidelity"]

    np.savez("spherical_cow_cloak_comparison.npz", **save_dict)
    print("\nData saved to spherical_cow_cloak_comparison.npz")


# =====================================================================
# Main
# =====================================================================


def main():
    args = parse_args()
    params = SimParams(args)

    if mp.am_master():
        print("=" * 72)
        print("  SPHERICAL COW CLOAKING: FOUR-APPROACH COMPARISON")
        print("=" * 72)
        print(f"  Method:     {args.method}")
        print(f"  Resolution: {params.resolution}")
        print(f"  R1 (cow):   {params.R1}")
        print(f"  R2 (cloak): {params.R2}")
        print(f"  n_cow:      {params.n_cow}")
        print(f"  fcen:       {params.fcen}")
        print("=" * 72)

    results = {}

    # Approach 1: Scattering-Cancellation (3D sphere)
    if args.method in ("cancellation", "all"):
        results["cancellation"] = run_cancellation_3d(params)

    # Shared 2D reference (empty + bare cylinder) for Pendry and multi-shell
    ref_2d = None
    if args.method in ("pendry2d", "multishell", "all"):
        ref_2d = run_empty_and_bare_2d(params)

    # Approach 2: Pendry 2D (cylinder)
    if args.method in ("pendry2d", "all"):
        results["pendry2d"] = run_pendry2d(params, ref_2d)

    # Approach 3: Multi-Shell (cylinder)
    if args.method in ("multishell", "all"):
        results["multishell"] = run_multishell(params, ref_2d)

    # Approach 4: Carpet Cloak (ground plane)
    if args.method in ("carpet", "all"):
        results["carpet"] = run_carpet(params)

    # Summary table
    print_comparison_table(results)

    # Plotting
    if not args.no_plot and mp.am_master():
        for key, r in results.items():
            if r is not None:
                plot_approach(r, f"cloak_comparison_{key}.png", params)
        if len(results) > 1:
            plot_summary(results, "cloak_comparison_summary.png")

    # Save data
    save_data(results, params)

    if mp.am_master():
        print("\nDone.")


if __name__ == "__main__":
    main()
