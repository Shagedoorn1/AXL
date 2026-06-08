import numpy as np
import os
import shutil
import subprocess

import matplotlib.pyplot as plt

class BaseRecorder:
    def start(self, solver):
        pass
    
    def capture(self, Ey, step):
        raise NotImplementedError
    
    def finalize(self):
        pass
    
class MemmapRecorder(BaseRecorder):
    def __init__(self, stride=50, filename="frames.dat", dtype=np.float32, mode="intensity"):
        self.stride = stride
        self.filename = filename
        self.dtype = dtype
        self.mode = mode
        
        self.frame_idx = 0
        self.recorder = None
        
    def start(self, solver):
        self.Nz = solver.grid.Nz
        self.Nx = solver.grid.Nx
        
        self.n_frames = (solver.n_steps // self.stride) + 1
        
        self.recorder = np.memmap(
            self.filename,
            dtype=self.dtype,
            mode="w+",
            shape=(self.n_frames, self.Nz, self.Nx)
        )
    
    def capture(self, Ey, step):
        if step % self.stride != 0:
            return
        
        frame = Ey
        
        self.recorder[self.frame_idx] = frame.astype(self.dtype)
        self.frame_idx += 1
        
    def finalize(self):
        if self.recorder is not None:
            self.recorder.flush()
            
    def render_frames(self, frames, n_map):
        if os.path.exists("frames"):
            shutil.rmtree("frames")

        os.makedirs("frames")
        
        vmax = np.max(frames)
        
        for i in range(self.frame_idx):
            
            plt.figure(figsize=(6,4))

            plt.imshow(
                frames[i],
                origin="lower",
                cmap="RdBu",
                vmin=0,
                vmax=vmax
            )
            plt.contour(
                n_map,
                levels=[1.25],
                colors="cyan",
                linewidths=1
            )

            plt.savefig(
                f"frames/frame_{i:05d}.png",
                dpi=100,
                bbox_inches="tight"
            )
            
            plt.close()
            
        subprocess.run([
            "ffmpeg",
            "-y",
            "-framerate", "30",
            "-i", "frames/frame_%05d.png",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "fdtd.mp4"
        ])