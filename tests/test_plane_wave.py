import numpy as np

from gravilens.models.plane_wave import PlaneWaveModel
from gravilens.core.base import InitialData

def test_H_quadratic_form():
    model = PlaneWaveModel(h1=2.0, h2=-1.0)
    assert np.isclose(model.H(3.0, 4.0), 2.0 * 3.0**2 - 1.0 * 4.0**2)

def test_first_conjugate_u_negative_profile():
    model = PlaneWaveModel(h1=-1.0, h2=4.0, u0=0.0)
    assert np.isclose(model.first_conjugate_u(), np.pi)


def test_first_conjugate_u_none_for_nonoscillatory_case():
    model = PlaneWaveModel(h1=1.0, h2=4.0)
    assert model.first_conjugate_u() is None

def test_lam_and_v0_dot_are_inverse():
    model = PlaneWaveModel(h1=-0.1, h2=-0.2)
    X0 = np.array([1.0, -2.0])
    X0_dot = np.array([0.3, 0.4])
    lam_target = -1.5

    v0_dot = model.v0_dot_for_lam(lam_target, X0, X0_dot)
    lam = model.lam_from_initial(X0, X0_dot, v0_dot)

    assert np.isclose(lam, lam_target)

def test_solve_geodesic_minkowski_case():
    model = PlaneWaveModel(h1=0.0, h2=0.0)

    initial = InitialData(
        u0=0.0,
        v0=0.0,
        x0=1.0,
        y0=-2.0,
        x0_dot=2.0,
        y0_dot=0.5,
        lam=0.0,
    )

    u_grid = np.array([0.0, 1.0, 2.0])
    sol = model.solve_geodesic(initial, u_grid)

    assert np.allclose(sol.x, [1.0, 3.0, 5.0])
    assert np.allclose(sol.y, [-2.0, -1.5, -1.0])
    assert np.allclose(sol.x_dot, [2.0, 2.0, 2.0])
    assert np.allclose(sol.y_dot, [0.5, 0.5, 0.5])

