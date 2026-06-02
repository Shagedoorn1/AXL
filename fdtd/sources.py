import numpy as np


SOURCE_REGISTRY = {}
def register_source(name):
    def wrapper(cls):
        SOURCE_REGISTRY[name] = cls
        return cls
    return wrapper

class Source:
    def apply(self, Ey, step, dt):
        raise NotImplementedError

@register_source("continuous_wave")
class ContinuousWaveSource(Source):
    def __init__(self, x, src_j, w_in, amplitude, omega):
        self.src_j = src_j
        self.profile = np.exp(-(x / w_in) ** 2)
        self.amplitude = amplitude
        self.omega = omega

    def waveform(self, step, dt):
        t = step * dt
        ramp = 1.0 - np.exp(-(step / 100.0) ** 2)
        return ramp * np.sin(self.omega * t)

    def apply(self, Ey, step, dt):
        Ey[self.src_j, :] += (
            self.amplitude * self.waveform(step, dt) * self.profile
        )

@register_source("gaussian_pulse")
class GaussianPulseSource(Source):
    def __init__(self, x, src_j, w_in, t0, spread, amplitude=1.0):
        self.src_j = src_j
        self.profile = np.exp(-(x / w_in) ** 2)
        self.t0 = t0
        self.spread = spread
        self.amplitude = amplitude

    def waveform(self, step, dt):
        t = step * dt
        return self.amplitude * np.exp(
            -((t - self.t0) ** 2) / (2 * self.spread ** 2)
        )

    def apply(self, Ey, step, dt):
        Ey[self.src_j, :] += self.waveform(step, dt) * self.profile

@register_source("functional")
class FunctionalSource(Source):
    def __init__(self, src_j, profile, func):
        self.src_j = src_j
        self.profile = profile
        self.func = func

    def apply(self, Ey, step, dt):
        Ey[self.src_j, :] += self.func(step, dt) * self.profile
     
def get_source(name, **kwargs):
    if name not in SOURCE_REGISTRY:
        raise ValueError(f"Unknown source: {name}")
    return SOURCE_REGISTRY[name](**kwargs)