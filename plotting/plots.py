from .core import AXL_figure

def plot_field(Ey, grid, style=None, title=None, n_map=None):
    fig, axes, _ = AXL_figure(style=style, title=title if title is not None else "Field")
    ax = axes[0, 0]
    
    im = ax.imshow(Ey, origin="lower", extent=grid.extent, aspect="equal")
    fig.colorbar(im, ax=ax)

    ax.set_xlabel(r"$x \ \left[\mathrm{\mu m}\right]$")
    ax.set_ylabel(r"$z \ \left[\mathrm{\mu m}\right]$")
    if n_map is not None:
        ax.contour(
            n_map,
            levels=[1.25],
            extent=grid.extent
        )
    return fig, ax


def plot_index_map(n_map, grid, style=None, title=None):
    fig, axes, _ = AXL_figure(style=style, title=title if title is not None else "Refractive index map")
    ax = axes[0, 0]
    im = ax.imshow(n_map, origin="lower", cmap="coolwarm", extent=grid.extent, aspect="equal")
    fig.colorbar(im, ax=ax)
    
    ax.set_xlabel(r"$x \ \left[\mathrm{\mu m}\right]$")
    ax.set_ylabel(r"$z \ \left[\mathrm{\mu m}\right]$")
    return fig, ax


def plot_dose(dose, grid, style=None, title=None):
    fig, axes, _ = AXL_figure(style=style, title=title if title is not None else r"Dose ($\int\left|E\right|^2 dt$)")
    ax = axes[0, 0]
    im = ax.imshow(dose, origin="lower", cmap="magma", extent=grid.extent, aspect="equal")
    fig.colorbar(im, ax=ax)

    ax.set_title()
    ax.set_xlabel(r"$x \ \left[\mathrm{\mu m}\right]$")
    ax.set_ylabel(r"$z \ \left[\mathrm{\mu m}\right]$")
    return fig, ax