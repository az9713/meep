#!/usr/bin/env python3
"""
Visualization suite for Spherical Cow Cloak simulation results.

Loads data from .npz files produced by spherical_cow_cloak.py and generates:
  1. 2D cross-section field comparison plots (matplotlib)
  2. Scattering cross-section spectrum (matplotlib)
  3. Interactive 3D field visualization (plotly, if installed)
  4. Time-domain animation as GIF (matplotlib)

Usage:
    python spherical_cow_cloak_viz.py                          # auto-detect data
    python spherical_cow_cloak_viz.py cloak_results_quick.npz  # specific file
    python spherical_cow_cloak_viz.py --no-3d                  # skip plotly
    python spherical_cow_cloak_viz.py --no-animation           # skip GIF
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =========================================================================
# Data Loading
# =========================================================================

def load_data(filepath):
    """Load simulation results from .npz file."""
    raw = dict(np.load(filepath, allow_pickle=True))

    # Convert 0-d arrays to scalars / lists
    for key in ["a", "b", "fcen", "df", "resolution"]:
        if key in raw and raw[key].ndim == 0:
            raw[key] = float(raw[key])

    if "materials" in raw:
        raw["materials"] = [str(x) for x in raw["materials"]]
    if "mode" in raw:
        raw["mode"] = str(raw["mode"])
    if "field_extent" in raw:
        raw["field_extent"] = raw["field_extent"].astype(float)

    return raw


# =========================================================================
# 1. 2D Cross-Section Field Comparison
# =========================================================================

def plot_field_comparison(data, output_dir):
    """2D cross-section comparison: bare vs cloaked for each cow material."""
    materials = data["materials"]
    a, b = data["a"], data["b"]
    ext = data["field_extent"]
    extent = [ext[0], ext[1], ext[0], ext[1]]

    for mat in materials:
        bare_xy = data.get(f"{mat}_bare_ez_xy")
        cloak_xy = data.get(f"{mat}_cloak_ez_xy")
        bare_xz = data.get(f"{mat}_bare_ez_xz")
        cloak_xz = data.get(f"{mat}_cloak_ez_xz")

        if bare_xy is None or cloak_xy is None:
            print(f"  Skipping {mat}: no field data")
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Spherical Cow Cloak \u2014 {mat.upper()} cow",
            fontsize=16,
            fontweight="bold",
        )

        # Columns 0-1: Re(Ez) for bare / cloaked
        for col, (case, label, ez_xy, ez_xz) in enumerate(
            [
                ("bare", "Bare Cow", bare_xy, bare_xz),
                ("cloak", "Cloaked Cow", cloak_xy, cloak_xz),
            ]
        ):
            # Row 0: XY plane
            ax = axes[0, col]
            vmax = max(np.abs(ez_xy).max(), 1e-10)
            im = ax.imshow(
                np.real(ez_xy).T,
                origin="lower",
                cmap="RdBu_r",
                extent=extent,
                vmin=-vmax,
                vmax=vmax,
                aspect="equal",
            )
            ax.add_patch(Circle((0, 0), a, fill=False, ec="black", lw=2))
            ax.add_patch(Circle((0, 0), b, fill=False, ec="black", lw=1, ls="--"))
            ax.set_title(f"{label}: Re(Ez), XY plane", fontsize=12)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Row 1: XZ plane
            if ez_xz is not None:
                ax = axes[1, col]
                vmax = max(np.abs(ez_xz).max(), 1e-10)
                im = ax.imshow(
                    np.real(ez_xz).T,
                    origin="lower",
                    cmap="RdBu_r",
                    extent=extent,
                    vmin=-vmax,
                    vmax=vmax,
                    aspect="equal",
                )
                ax.add_patch(Circle((0, 0), a, fill=False, ec="black", lw=2))
                ax.add_patch(
                    Circle((0, 0), b, fill=False, ec="black", lw=1, ls="--")
                )
                ax.set_title(f"{label}: Re(Ez), XZ plane", fontsize=12)
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                plt.colorbar(im, ax=ax, shrink=0.8)

        # Column 2: |Ez|^2 ratio (cloaked / bare)
        for row, (plane, ylabel) in enumerate([("xy", "y"), ("xz", "z")]):
            bare = data.get(f"{mat}_bare_ez_{plane}")
            cloak = data.get(f"{mat}_cloak_ez_{plane}")
            if bare is None or cloak is None:
                continue

            ax = axes[row, 2]
            ratio = np.abs(cloak) ** 2 / np.maximum(np.abs(bare) ** 2, 1e-20)
            im = ax.imshow(
                np.log10(ratio + 1e-10).T,
                origin="lower",
                cmap="coolwarm",
                extent=extent,
                vmin=-2,
                vmax=2,
                aspect="equal",
            )
            ax.add_patch(Circle((0, 0), a, fill=False, ec="black", lw=2))
            ax.add_patch(Circle((0, 0), b, fill=False, ec="black", lw=1, ls="--"))
            plane_label = plane.upper()
            ax.set_title(
                f"log$_{{10}}$(|Ez$_{{cloak}}$/Ez$_{{bare}}$|$^2$), {plane_label}",
                fontsize=11,
            )
            ax.set_xlabel("x")
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax, shrink=0.8, label="log$_{10}$ ratio")

        plt.tight_layout()
        out = os.path.join(output_dir, f"cloak_fields_{mat}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.close()


# =========================================================================
# 2. Scattering Cross-Section Spectrum
# =========================================================================

def plot_spectrum(data, output_dir):
    """Scattering cross-section spectrum for all cow materials."""
    materials = data["materials"]
    freqs = data["freqs"]
    a = data["a"]

    n_mat = len(materials)
    fig, axes = plt.subplots(1, n_mat, figsize=(7 * n_mat, 5), squeeze=False)
    axes = axes[0]

    colors = {"pec": "C0", "dielectric": "C1", "lossy": "C2"}

    for i, mat in enumerate(materials):
        ax = axes[i]
        bare_q = np.abs(data.get(f"{mat}_bare_qsca", np.zeros_like(freqs)))
        cloak_q = np.abs(data.get(f"{mat}_cloak_qsca", np.zeros_like(freqs)))

        c = colors.get(mat, "C0")
        ax.semilogy(freqs, bare_q, "-", color=c, linewidth=2, label="Bare")
        ax.semilogy(freqs, cloak_q, "--", color=c, linewidth=2, label="Cloaked")

        if mat == "dielectric" and "mie_qsca" in data:
            ax.semilogy(
                freqs, data["mie_qsca"], "k:", linewidth=1.5, label="Mie theory"
            )

        # Annotate reduction factor at center frequency
        mid = len(freqs) // 2
        if cloak_q[mid] > 1e-20:
            ratio = bare_q[mid] / cloak_q[mid]
            ax.axvline(freqs[mid], color="gray", ls=":", alpha=0.5)
            ax.text(
                0.05,
                0.95,
                f"Reduction at f\u2080: {ratio:.1f}x",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel("Frequency (meep units)")
        ax.set_ylabel("Scattering Efficiency Q$_{sca}$")
        ax.set_title(f"{mat.upper()} Cow")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Scattering Cross-Section: Bare vs Cloaked",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "cloak_spectrum.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# =========================================================================
# 3. Interactive 3D Visualization (plotly)
# =========================================================================

def plot_3d_interactive(data, output_dir):
    """Interactive 3D field visualization using plotly."""
    if not HAS_PLOTLY:
        print("  plotly not installed, skipping 3D visualization")
        print("  Install with: pip install plotly")
        return

    materials = data["materials"]
    a, b = data["a"], data["b"]
    ext = data["field_extent"]

    fig = go.Figure()
    buttons = []
    trace_idx = 0

    for mat in materials:
        for case in ["bare", "cloak"]:
            ez_xy = data.get(f"{mat}_{case}_ez_xy")
            ez_xz = data.get(f"{mat}_{case}_ez_xz")
            if ez_xy is None or ez_xz is None:
                continue

            start_idx = trace_idx

            # XY plane slice (z=0)
            nx, ny = ez_xy.shape
            x = np.linspace(ext[0], ext[1], nx)
            y = np.linspace(ext[0], ext[1], ny)
            X_xy, Y_xy = np.meshgrid(x, y, indexing="ij")
            Z_xy = np.zeros_like(X_xy)

            fig.add_trace(
                go.Surface(
                    x=X_xy, y=Y_xy, z=Z_xy,
                    surfacecolor=np.real(ez_xy),
                    colorscale="RdBu_r",
                    opacity=0.85,
                    showscale=True,
                    visible=False,
                    name=f"{mat} {case} XY",
                    colorbar=dict(title="Re(Ez)", x=1.05),
                )
            )
            trace_idx += 1

            # XZ plane slice (y=0)
            nx2, nz = ez_xz.shape
            x2 = np.linspace(ext[0], ext[1], nx2)
            z = np.linspace(ext[0], ext[1], nz)
            X_xz, Z_xz = np.meshgrid(x2, z, indexing="ij")
            Y_xz = np.zeros_like(X_xz)

            fig.add_trace(
                go.Surface(
                    x=X_xz, y=Y_xz, z=Z_xz,
                    surfacecolor=np.real(ez_xz),
                    colorscale="RdBu_r",
                    opacity=0.85,
                    showscale=False,
                    visible=False,
                    name=f"{mat} {case} XZ",
                )
            )
            trace_idx += 1

            # Sphere wireframes for cow and cloak boundaries
            phi = np.linspace(0, 2 * np.pi, 30)
            th = np.linspace(0, np.pi, 20)
            PHI, TH = np.meshgrid(phi, th)

            for radius, name_r, color in [
                (a, "cow", "rgba(50,50,50,0.15)"),
                (b, "cloak", "rgba(150,150,150,0.1)"),
            ]:
                xs = radius * np.sin(TH) * np.cos(PHI)
                ys = radius * np.sin(TH) * np.sin(PHI)
                zs = radius * np.cos(TH)
                fig.add_trace(
                    go.Surface(
                        x=xs, y=ys, z=zs,
                        opacity=0.12,
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        visible=False,
                        name=f"{name_r} boundary",
                    )
                )
                trace_idx += 1

            # Button for this case
            n_traces = trace_idx - start_idx
            visibility = [False] * len(fig.data)
            for j in range(start_idx, trace_idx):
                visibility[j] = True

            case_label = "Bare" if case == "bare" else "Cloaked"
            buttons.append(
                dict(
                    label=f"{mat} - {case_label}",
                    method="update",
                    args=[{"visible": visibility}],
                )
            )

    # Show first case by default
    if buttons:
        first_vis = buttons[0]["args"][0]["visible"]
        for i, v in enumerate(first_vis):
            fig.data[i].visible = v

    fig.update_layout(
        title="Spherical Cow Cloak: 3D Field Visualization",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.1,
                y=1.15,
                showactive=True,
                buttons=buttons,
            )
        ]
        if buttons
        else [],
        width=900,
        height=700,
    )

    out = os.path.join(output_dir, "cloak_3d_interactive.html")
    fig.write_html(out)
    print(f"  Saved: {out}")


# =========================================================================
# 4. Time-Domain Animation (phase rotation of DFT fields)
# =========================================================================

def make_animation(data, output_dir, mat=None, nframes=60):
    """
    Create a GIF animation showing the wavefront passing through the cloak.

    Uses phase rotation of the single-frequency DFT data to reconstruct
    the steady-state time-domain field oscillation:
        E(t) = Re(E_dft * exp(i * phase))
    """
    materials = data["materials"]
    if mat is None:
        mat = materials[0]
    if mat not in materials:
        print(f"  Material '{mat}' not found, skipping animation")
        return

    a, b = data["a"], data["b"]
    ext = data["field_extent"]
    extent = [ext[0], ext[1], ext[0], ext[1]]

    bare_ez = data.get(f"{mat}_bare_ez_xy")
    cloak_ez = data.get(f"{mat}_cloak_ez_xy")
    if bare_ez is None or cloak_ez is None:
        print(f"  No field data for {mat}, skipping animation")
        return

    vmax = max(np.abs(bare_ez).max(), np.abs(cloak_ez).max(), 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Spherical Cow Cloak: {mat.upper()} \u2014 Ez field (XY plane)",
        fontsize=14,
        fontweight="bold",
    )

    im1 = ax1.imshow(
        np.real(bare_ez).T,
        origin="lower",
        cmap="RdBu_r",
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    im2 = ax2.imshow(
        np.real(cloak_ez).T,
        origin="lower",
        cmap="RdBu_r",
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )

    for ax, title in [(ax1, "Bare Cow"), (ax2, "Cloaked Cow")]:
        ax.add_patch(Circle((0, 0), a, fill=False, ec="black", lw=2))
        ax.add_patch(Circle((0, 0), b, fill=False, ec="black", lw=1, ls="--"))
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.colorbar(im1, ax=ax1, shrink=0.8, label="Re(Ez)")
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="Re(Ez)")
    plt.tight_layout()

    def update(frame):
        phase = 2 * np.pi * frame / nframes
        im1.set_data(np.real(bare_ez * np.exp(1j * phase)).T)
        im2.set_data(np.real(cloak_ez * np.exp(1j * phase)).T)
        return [im1, im2]

    anim = animation.FuncAnimation(
        fig, update, frames=nframes, interval=50, blit=False
    )

    out = os.path.join(output_dir, f"cloak_animation_{mat}.gif")
    try:
        anim.save(out, writer="pillow", fps=20)
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  Animation save failed ({e}), trying imagemagick...")
        try:
            anim.save(out, writer="imagemagick", fps=20)
            print(f"  Saved: {out}")
        except Exception as e2:
            print(f"  Could not save animation: {e2}")
            print("  Install pillow (pip install pillow) for GIF export")

    plt.close()


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Spherical Cow Cloak simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python spherical_cow_cloak_viz.py                          # auto-detect
  python spherical_cow_cloak_viz.py cloak_results_quick.npz  # specific file
  python spherical_cow_cloak_viz.py --no-3d --no-animation   # plots only
""",
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        default=None,
        help="Path to .npz results file (auto-detects if not given)",
    )
    parser.add_argument(
        "--no-3d", action="store_true", help="Skip 3D plotly visualization"
    )
    parser.add_argument(
        "--no-animation", action="store_true", help="Skip GIF animation"
    )
    parser.add_argument(
        "--animate-material",
        default=None,
        help="Material for animation (default: first available)",
    )
    args = parser.parse_args()

    # Find data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_file:
        data_path = args.data_file
    else:
        for name in ["cloak_results_production.npz", "cloak_results_quick.npz"]:
            candidate = os.path.join(script_dir, name)
            if os.path.exists(candidate):
                data_path = candidate
                break
        else:
            print("No data file found. Run spherical_cow_cloak.py first.")
            sys.exit(1)

    print(f"Loading data from: {data_path}")
    data = load_data(data_path)
    print(f"  Mode:        {data.get('mode', 'unknown')}")
    print(f"  Materials:   {data['materials']}")
    print(f"  Frequencies: {len(data['freqs'])} points")
    print(f"  Cow radius:  {data['a']},  Cloak radius: {data['b']}")

    output_dir = os.path.dirname(os.path.abspath(data_path))

    # Generate all visualizations
    print("\n" + "=" * 50)
    print("1. Field comparison plots (2D cross-sections)")
    print("=" * 50)
    plot_field_comparison(data, output_dir)

    print("\n" + "=" * 50)
    print("2. Scattering cross-section spectrum")
    print("=" * 50)
    plot_spectrum(data, output_dir)

    if not args.no_3d:
        print("\n" + "=" * 50)
        print("3. Interactive 3D visualization")
        print("=" * 50)
        plot_3d_interactive(data, output_dir)

    if not args.no_animation:
        print("\n" + "=" * 50)
        print("4. Time-domain animation (GIF)")
        print("=" * 50)
        make_animation(data, output_dir, mat=args.animate_material)

    print("\n" + "=" * 50)
    print("All visualizations complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
