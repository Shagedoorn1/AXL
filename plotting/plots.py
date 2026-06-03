from .core import AXL_figure

def plot_field(Ey, style=None, cmap="viridis"):
    fig, ax = AXL_figure(style=style)

    im = ax.imshow(Ey, origin="lower", cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_title("Field (Ey)")
    return fig, ax


def plot_index_map(n_map, x=None, z=None, style=None):
    fig, ax = AXL_figure(style=style)

    im = ax.imshow(n_map, origin="lower", cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_title("Refractive index map")
    return fig, ax


def plot_dose(dose, style=None):
    fig, ax = AXL_figure(style=style)

    im = ax.imshow(dose, origin="lower", cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_title("Dose (∫|E|² dt)")
    return fig, ax