import numpy as np
import matplotlib as mpl
    
class Style:
    def __init__(self, dark=True, font_size=9, cmap="viridis", grid=True, watermark=None):
        
        self.dark = dark
        self.font_size = font_size
        self.cmap = cmap
        self.grid = grid
        self.watermark = watermark
        
        # precomputed themes
        if self.dark:
            self.bg = "#0b0d12"
            self.axes_bg = "#0f1117"
            self.text = "#e6edf3"
            self.grid_color = "#2f3440"
            
        else:
            self.bg = "#fffff0"
            self.axes_bg = "#ffffff"
            self.text = "#000000"
            self.grid_color = "#cccccc"
            
    def apply(self, fig):
        mpl.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": self.font_size,

            "axes.facecolor": self.axes_bg,
            "figure.facecolor": self.bg,

            "text.color": self.text,
            "axes.labelcolor": self.text,
            "axes.titlecolor": self.text,

            "xtick.color": self.text,
            "ytick.color": self.text,

            "grid.color": self.grid_color,
            "grid.alpha": 0.5,
        })
        
        if self.watermark is not None:
            fig.text(
                0.995, 0.005,
                self.watermark,
                ha="right",
                va="bottom",
                fontsize=16,
                family="monospace",
                color="#6aa9ff",
                alpha=0.7
            )
        
        fig.tight_layout()