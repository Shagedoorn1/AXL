import matplotlib.pyplot as plt
import numpy as np

def AXL_figure(layout=(1, 1), figsize=(10, 6), title=None, gridspec_kw=None, style=None):
    """
    Standardized Owly-style figure builder.
    """

    # 1. create figure first
    fig = plt.figure(figsize=figsize)

    # 2. apply style (must operate on fig)
    if style is not None:
        style.apply(fig)

    # 3. title
    if title:
        fig.suptitle(title, fontsize=13)


    # 5. layout
    gs = fig.add_gridspec(*layout, **(gridspec_kw or {}))

    axes = np.empty(layout, dtype=object)

    for r in range(layout[0]):
        for c in range(layout[1]):
            axes[r, c] = fig.add_subplot(gs[r, c])

    return fig, axes, gs