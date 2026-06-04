from .core import AXL_figure

def plot_field(Ey, grid, style=None):
    fig, axes, _ = AXL_figure(style=style)
    ax = axes[0, 0]
    
    im = ax.imshow(Ey, origin="lower", extent=grid.extent, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_title("Field (Ey)")
    return fig, ax


def plot_index_map(n_map, grid, style=None):
    fig, axes, _ = AXL_figure(style=style)
    ax = axes[0, 0]
    im = ax.imshow(n_map, origin="lower", cmap="coolwarm", extent=grid.extent, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_title("Refractive index map")
    return fig, ax


def plot_dose(dose, grid, style=None):
    fig, axes, _ = AXL_figure(style=style)
    ax = axes[0, 0]
    im = ax.imshow(dose, origin="lower", cmap="magma", extent=grid.extent, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_title("Dose (∫|E|² dt)")
    return fig, ax