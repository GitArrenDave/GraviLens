import matplotlib.pyplot as plt
import gravilens.models.plane_wave as pw
import gravilens.core.base as bs
import gravilens.plotting as gp
from gravilens.scenarios import lightcone, timelike_geodesic_through_event, comoving_geodesic_through_event
if __name__ == '__main__':
    model = pw.PlaneWaveModel(h1=-0.1, h2=-0.1)

    fig, ax, solutions, cutpoints, coeff = lightcone(
        model,
        n_u=300,
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
    t_obs, z_obs = bs.brinkmann_to_minkowski(obs_event.u, obs_event.v)
    ax.scatter(z_obs, obs_event.x, t_obs, color="red", s=20, depthshade=False)

    obs_past, obs_future = timelike_geodesic_through_event(
        model,
        obs_event=obs_event,
        lam_timelike=-1.0,
        X_obs_dot_timelike=(0.15, 0.0),
        u_start=model.u0,
        u_end=model.first_conjugate_u(),
        n_back=400,
        n_fwd=400,
    )
    gp.plot_solution(ax, obs_past, color="black", lw=2.0)
    if obs_future is not None:
        gp.plot_solution(ax, obs_future, color="black", lw=2.0)

    target_event = bs.GeodesicEvent(
        u=0.0,
        v=0.0,
        x=0.0,
        y=0.0,
    )
    t_tar, z_tar = bs.brinkmann_to_minkowski(target_event.u, target_event.v)
    ax.scatter(z_tar, target_event.x, t_tar, color="magenta", s=20, depthshade=False)

    src_past, src_future, src_initial = comoving_geodesic_through_event(
        model,
        ref_geo=obs_future if obs_future is not None else obs_past,
        target_event=target_event,
        u_match=obs_event.u,
        u_start=model.u0,
        u_end=model.first_conjugate_u(),
        n_back=400,
        n_fwd=400,
    )

    if src_past is not None:
        gp.plot_solution(ax, src_past, color="darkorange", lw=2.0)
    if src_future is not None:
        gp.plot_solution(ax, src_future, color="darkorange", lw=2.0)

    if src_past is not None:
        ev0 = src_past.event_at_u(target_event.u)
        print("Target event:")
        print(target_event)
        print("Comoving geodesic at target u:")
        print(ev0)

        gp.draw_brinkmann_vector(
            ax,
            u=obs_ref.u,
            v=obs_ref.v,
            x=obs_ref.x,
            y=obs_ref.y,
            vec_u=0.0,
            vec_v=0.0,
            vec_x=obs_ref.x_dot,
            vec_y=obs_ref.y_dot,
            color="blue",
        )

        gp.draw_brinkmann_vector(
            ax,
            u=src_ref.u,
            v=src_ref.v,
            x=src_ref.x,
            y=src_ref.y,
            vec_u=0.0,
            vec_v=0.0,
            vec_x=src_ref.x_dot,
            vec_y=src_ref.y_dot,
            color="green",
        )

        # --- hübscher Plot ---
    ax.set_title("Lightcone + observer geodesic + comoving source")
    plt.tight_layout()
    plt.show()

