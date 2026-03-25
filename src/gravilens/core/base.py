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

@dataclass(frozen=True)
class BrinkmannVector:
    du: float
    dv: float
    dx: float
    dy: float

@dataclass(frozen=True)
class GeodesicEvent:
    u: float
    v: float
    x: float
    y: float
    x_dot: float | None = None
    y_dot: float | None = None
    v_dot: float | None = None
    i: int | None = None

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

    def event_at_index(self, i: int) -> GeodesicEvent:
        return GeodesicEvent(
            u=float(self.u[i]),
            v=float(self.v[i]),
            x=float(self.x[i]),
            y=float(self.y[i]),
            x_dot=None if self.x_dot is None else float(self.x_dot[i]),
            y_dot=None if self.y_dot is None else float(self.y_dot[i]),
            v_dot=None if self.v_dot is None else float(self.v_dot[i]),
            i=int(i),
        )

    def event_at_u(self, u_value: float) -> GeodesicEvent:
        u_arr = np.asarray(self.u)

        if len(u_arr) == 0:
            raise ValueError("Cannot query an event from an empty geodesic solution.")

        if len(u_arr) == 1:
            return self.event_at_index(0)

        if u_arr[0] <= u_arr[-1]:
            i = int(np.searchsorted(u_arr, u_value))
            if i == 0:
                return self.event_at_index(0)
            if i >= len(u_arr):
                return self.event_at_index(len(u_arr) - 1)

            left = i - 1
            right = i
        else:
            u_rev = u_arr[::-1]
            j = int(np.searchsorted(u_rev, u_value))
            if j == 0:
                return self.event_at_index(len(u_arr) - 1)
            if j >= len(u_arr):
                return self.event_at_index(0)

            left = len(u_arr) - j
            right = len(u_arr) - j - 1

        if abs(u_arr[left] - u_value) <= abs(u_arr[right] - u_value):
            return self.event_at_index(left)
        return self.event_at_index(right)