from dataclasses import dataclass
import numpy as np
from gravilens.core.base import InitialData, GeodesicSolution, brinkmann_to_minkowski, minkowski_to_brinkmann

@dataclass
class PlaneWaveModel:
    h_plus: float = 0.0
    h_p: float = 0.0
    u0: float = 0.0
    v0: float = 0.0
    eps: float = 1e-14

    @property
    def h1(self) -> float:
        return self.h_plus - self.h_p

    @property
    def h2(self) -> float:
        return -self.h_plus - self.h_p

    @property
    def h_mat(self) -> np.ndarray:
        return np.array([
            [self.h1, 0.0],
            [0.0, self.h2],
        ])

    def H(self, x: float, y: float) -> float:
        X = np.array([x, y])
        return X @ self.h_mat @ X

    def A(self, u, u0=None):
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self.eps
        if h1 > E:
            A11 = np.cosh(np.sqrt(h1)*(u-u0))
            A22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        elif abs(h1) <= E:
            A11 = 1.0; A22 = 1.0
        else:
            A11 = np.cos(np.sqrt(abs(h1))*(u-u0))
            A22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        return np.array([[A11, 0.0], [0.0, A22]])

    def B(self, u,u0=None):
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self.eps
        if h1 > E:
            B11 = (1/np.sqrt(h1))*np.sinh(np.sqrt(h1)*(u-u0))
            B22 = (1/np.sqrt(abs(h2)))*np.sin(np.sqrt(abs(h2))*(u-u0))
        elif abs(h1) <= E:
            B11 = (u-u0); B22 = (u-u0)
        else:
            B11 = (1/np.sqrt(abs(h1)))*np.sin(np.sqrt(abs(h1))*(u-u0))
            B22 = (1/np.sqrt(abs(h2)))*np.sin(np.sqrt(abs(h2))*(u-u0))
        return np.array([[B11, 0.0], [0.0, B22]])

    def A_dot(self, u,u0=None):
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self.eps
        if h1 > E:
            A11 = np.sqrt(h1)*np.sinh(np.sqrt(h1)*(u-u0))
            A22 = -np.sqrt(abs(h2))*np.sin(np.sqrt(abs(h2))*(u-u0))
        elif abs(h1) <= E:
            A11 = 0.0; A22 = 0.0
        else:
            A11 = -np.sqrt(abs(h1))*np.sin(np.sqrt(abs(h1))*(u-u0))
            A22 = -np.sqrt(abs(h2))*np.sin(np.sqrt(abs(h2))*(u-u0))
        return np.array([[A11, 0.0], [0.0, A22]])

    def B_dot(self, u,u0=None):
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self.eps
        if h1 >= E:
            B11 = np.cosh(np.sqrt(h1)*(u-u0))
            B22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        elif abs(h1) <= E:
            B11 = 1.0; B22 = 1.0
        else:
            B11 = np.cos(np.sqrt(abs(h1))*(u-u0))
            B22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        return np.array([[B11, 0.0], [0.0, B22]])

    def transverse_geodesic(self, u, X0, X0_dot,u0=None):
        return self.A(u,u0) @ X0 + self.B(u,u0) @ X0_dot

    def transverse_geodesic_dot(self, u, X0, X0_dot, u0 = None):
        return self.A_dot(u,u0) @ X0 + self.B_dot(u,u0) @ X0_dot

    def v_geodesic(self, u, X, X_dot, X0, X0_dot, v0=0.0, lam=0.0, u0=None):
        u0 = self.u0 if u0 is None else u0
        return v0 + 0.5 * (-lam*(u-u0) + X_dot @ X - X0_dot @ X0)

    @staticmethod
    def alpha_for_phi(phi_deg):
        """phi in Grad -> alpha in Radiant."""
        phi = np.deg2rad(phi_deg)
        c = np.cos(phi)
        if np.isclose(c, 0.0):
            return np.pi / 4
        return np.arctan(1.0 / c) - np.pi / 4

    def x0_dot_from_alpha(self, alpha, x0, y0, y0_dot):
        x0_dot_sq = 2 * np.tan(alpha) - self.H(x0, y0) - y0_dot ** 2
        if x0_dot_sq < 0:
            x0_dot_sq = 0.0
        return np.sqrt(x0_dot_sq)

    def lam_from_initial(self, X0, X0_dot, v0_dot):
        X0 = np.asarray(X0, float);
        X0_dot = np.asarray(X0_dot, float)
        H0 = self.H(X0[0], X0[1])
        return -2.0 * float(v0_dot) + H0 + float(X0_dot @ X0_dot)

    def v0_dot_for_lam(self, lam_target, X0, X0_dot):
        X0 = np.asarray(X0, float);
        X0_dot = np.asarray(X0_dot, float)
        H0 = self.H(X0[0], X0[1])
        return 0.5 * (-float(lam_target) + H0 + float(X0_dot @ X0_dot))


    def solve_geodesic(self, initial: InitialData, u_grid: np.ndarray) -> GeodesicSolution:
        u_grid = np.asarray(u_grid, dtype=float)

        X0 = np.array([initial.x0, initial.y0], dtype=float)
        X0_dot = np.array([initial.x0_dot, initial.y0_dot], dtype=float)

        X_list = []
        X_dot_list = []
        v_list = []
        for u in u_grid:
            X = self.transverse_geodesic(u, X0, X0_dot, u0=initial.u0)
            X_dot = self.transverse_geodesic_dot(u, X0, X0_dot, u0=initial.u0)
            v = self.v_geodesic(
                u, X, X_dot, X0, X0_dot,
                v0=initial.v0,
                lam=initial.lam,
                u0=initial.u0,
            )
            X_list.append(X)
            X_dot_list.append(X_dot)
            v_list.append(v)

        X_arr = np.asarray(X_list)
        X_dot_arr = np.asarray(X_dot_list)
        v_arr = np.asarray(v_list)

        x = X_arr[:, 0]
        y = X_arr[:, 1]
        x_dot = X_dot_arr[:, 0]
        y_dot = X_dot_arr[:, 1]

        t, z = brinkmann_to_minkowski(u_grid, v_arr)

        return GeodesicSolution(
            u=u_grid,
            v=v_arr,
            x=x,
            y=y,
            t=t,
            z=z,
            x_dot=x_dot,
            y_dot=y_dot,
            lam=initial.lam,
        )