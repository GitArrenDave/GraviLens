import numpy as np
import gravilens.models.plane_wave as pw
import gravilens.core.base as bs
if __name__ == '__main__':
    model = pw.PlaneWaveModel(h_plus=0.1, h_p=0.0)

    initial = bs.InitialData(
        u0=0.0,
        v0=0.0,
        x0=0.0,
        y0=0.0,
        x0_dot=1.0,
        y0_dot=0.0,
        lam=0.0,
    )

    u = np.linspace(0.0, 1.0, 100)
    sol = model.solve_geodesic(initial, u)

    df = sol.to_frame()
    print(df.head())

