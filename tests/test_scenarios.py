import numpy as np
import pytest
import gravilens.models.plane_wave as pw
import gravilens.core.base as bs
from gravilens.scenarios import (
    lightcone,
    timelike_geodesic_through_event,
    comoving_geodesic_through_event,
)

def build_comoving_setup(
    *,
    h1=-0.1,
    h2=-0.1,
    obs_solution_index=2,
    obs_u=2.8,
    lam_timelike=-1.0,
    X_obs_dot_timelike=(0.15, 0.0),
    target_event=None,
    n_u=120,
    n_back=120,
    n_fwd=120,
):
    model = pw.PlaneWaveModel(h1=h1, h2=h2)

    fig, ax, solutions, cutpoints, coeff = lightcone(
        model,
        n_u=n_u,
        angles_deg=(0, 20, 40, 60, 80),
        x0=0.0,
        y0=0.0,
        y0_dot=0.0,
        lam=0.0,
        L=6.0,
        show=False,
        draw_u_planes=False,
    )

    obs_null_geo = solutions[obs_solution_index]
    obs_event = obs_null_geo.event_at_u(obs_u)

    obs_past, obs_future = timelike_geodesic_through_event(
        model,
        obs_event=obs_event,
        lam_timelike=lam_timelike,
        X_obs_dot_timelike=X_obs_dot_timelike,
        u_start=model.u0,
        u_end=model.first_conjugate_u(),
        n_back=n_back,
        n_fwd=n_fwd,
    )

    if target_event is None:
        target_event = bs.GeodesicEvent(u=0.0, v=0.0, x=0.0, y=0.0)

    ref_geo = obs_future if obs_future is not None else obs_past

    src_past, src_future, src_initial = comoving_geodesic_through_event(
        model,
        ref_geo=ref_geo,
        target_event=target_event,
        u_match=obs_event.u,
        u_start=model.u0,
        u_end=model.first_conjugate_u(),
        n_back=n_back,
        n_fwd=n_fwd,
    )

    return {
        "model": model,
        "fig": fig,
        "ax": ax,
        "solutions": solutions,
        "cutpoints": cutpoints,
        "coeff": coeff,
        "obs_null_geo": obs_null_geo,
        "obs_event": obs_event,
        "obs_past": obs_past,
        "obs_future": obs_future,
        "target_event": target_event,
        "src_past": src_past,
        "src_future": src_future,
        "src_initial": src_initial,
        "ref_geo": ref_geo,
    }



def test_comoving_geodesic_hits_target_event():
    model = pw.PlaneWaveModel(h1=-0.1, h2=-0.1)

    fig, ax, solutions, cutpoints, coeff = lightcone(
        model,
        n_u=120,
        angles_deg=(0, 20, 40, 60, 80),
        x0=0.0,
        y0=0.0,
        y0_dot=0.0,
        lam=0.0,
        L=6.0,
        show=False,
        draw_u_planes=False,
    )

    obs_null_geo = solutions[2]
    obs_event = obs_null_geo.event_at_u(2.8)

    obs_past, obs_future = timelike_geodesic_through_event(
        model,
        obs_event=obs_event,
        lam_timelike=-1.0,
        X_obs_dot_timelike=(0.15, 0.0),
        u_start=model.u0,
        u_end=model.first_conjugate_u(),
        n_back=120,
        n_fwd=120,
    )

    target_event = bs.GeodesicEvent(
        u=0.0,
        v=0.0,
        x=0.0,
        y=0.0,
    )

    src_past, src_future, src_initial = comoving_geodesic_through_event(
        model,
        ref_geo=obs_future if obs_future is not None else obs_past,
        target_event=target_event,
        u_match=obs_event.u,
        u_start=model.u0,
        u_end=model.first_conjugate_u(),
        n_back=120,
        n_fwd=120,
    )

    assert src_past is not None

    ev0 = src_past.event_at_u(target_event.u)

    assert np.isclose(ev0.u, target_event.u)
    assert np.isclose(ev0.v, target_event.v, atol=1e-8)
    assert np.isclose(ev0.x, target_event.x, atol=1e-8)
    assert np.isclose(ev0.y, target_event.y, atol=1e-8)


def test_comoving_frcy_shift():
    setup = build_comoving_setup(h1=0.1, h2=-0.1)

    model = setup["model"]
    obs_geo = setup["ref_geo"]
    src_geo = setup["src_past"]
    obs_null_geo = setup["obs_null_geo"]

    u_o = setup["obs_event"].u
    u_e = setup["target_event"].u

    # Events
    obs_ev = obs_geo.event_at_u(u_o)
    src_ev_at_uo = src_geo.event_at_u(u_o)
    src_ev_at_ue = src_geo.event_at_u(u_e)

    null_obs_ev = obs_null_geo.event_at_u(u_o)
    null_emit_ev = obs_null_geo.event_at_u(u_e)

    # delta x_o
    delta = np.array(
        [obs_ev.x - src_ev_at_uo.x, obs_ev.y - src_ev_at_uo.y],
        dtype=float,
    )

    # B^{-T} delta x_o
    B = model.B(u_o, u_e)
    BinvT_delta = np.linalg.solve(B.T, delta)

    # thesis formula adapted to code convention lam < 0
    h = model.h_mat
    lam_s = src_geo.lam  # negative in the current code conventions

    num = delta @ h @ delta
    den = -lam_s + BinvT_delta @ BinvT_delta

    z_theory = 1.0 + num / den

    # numeric frequency shift from the implemented API
    Xdot_obs_o = np.array([obs_ev.x_dot, obs_ev.y_dot], dtype=float)
    Xdot_src_e = np.array([src_ev_at_ue.x_dot, src_ev_at_ue.y_dot], dtype=float)



    r_o = bs.BrinkmannVector(
        du=0.0,
        dv=0.0,
        dx=-float(null_obs_ev.x_dot),
        dy=-float(null_obs_ev.y_dot),
    )
    r_e = bs.BrinkmannVector(
        du=0.0,
        dv=0.0,
        dx=float(null_emit_ev.x_dot),
        dy=float(null_emit_ev.y_dot),
    )

    psi_deg = model.angle_lightray_dv(Xdot_obs_o, r_o, obs_geo.lam)
    psi = np.radians(psi_deg)

    z_deg_theory = -1/np.sin(psi/2)**2 * lam_s/den

    z_numeric = model.frequency_shift(
        lambda_e=src_geo.lam,
        lambda_o=obs_geo.lam,
        r_o=r_o,
        r_e=r_e,
        Xdot_obs_o=Xdot_obs_o,
        Xdot_src_e=Xdot_src_e,
    )
    print("z_theory:",z_theory)
    print("z_numeric:", z_numeric)
    print("z_deg_theory:", z_deg_theory)
    assert z_theory == pytest.approx(z_deg_theory, rel=2e-1, abs=1e-3)
    assert z_numeric == pytest.approx(z_theory, rel=2e-1, abs=1e-3)