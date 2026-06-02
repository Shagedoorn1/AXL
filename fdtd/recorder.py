import numpy as np

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
        self.Nz = solver.Nz
        self.Nx = solver.Nx
        
        self.n_frames = 10_000
        
        self.recorder = np.memmap(
            self.filename,
            dtype=self.dtype,
            mode="w+",
            shape=(self.n_frames, self.Nz, self.Nx)
        )
    
    def capture(self, Ey, step):
        if step % self.stride != 0:
            return
        
        frame = Ey * Ey
        
        self.recorder[self.frame_idx] = frame.astype(self.dtype)
        self.frame_idx += 1
        
    def finalize(self):
        if self.recorder is not None:
            self.recorder.flush()