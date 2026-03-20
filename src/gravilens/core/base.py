from dataclasses import dataclass
import numpy as np
import pandas as pd

def brinkmann_to_minkowski(u, v):
    t = 0.5 * (u + v)
    z = 0.5 * (v - u)
    return t, z


def minkowski_to_brinkmann(t, z):
    u = t - z
    v = t + z
    return u, v

@dataclass(frozen=True)
class InitialData:
    u0: float
    v0: float
    x0: float
    y0: float
    x0_dot: float
    y0_dot: float
    lam: float


@dataclass
class GeodesicSolution:
    u: np.ndarray
    v: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    z: np.ndarray
    x_dot: np.ndarray | None = None
    y_dot: np.ndarray | None = None
    v_dot: np.ndarray | None = None
    lam: float | None = None

    def to_frame(self) -> pd.DataFrame:
        data = {
            "u": self.u,
            "v": self.v,
            "x": self.x,
            "y": self.y,
            "t": self.t,
            "z": self.z,
        }
        if self.x_dot is not None:
            data["x_dot"] = self.x_dot
        if self.y_dot is not None:
            data["y_dot"] = self.y_dot
        if self.v_dot is not None:
            data["v_dot"] = self.v_dot
        return pd.DataFrame(data)