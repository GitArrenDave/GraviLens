import numpy as np
import matplotlib.pyplot as plt
import gravilens.plotting as gp
from gravilens.core.base import InitialData

def lightcone(
        model,
        u_end=None,
        n_u=1200,
        angles_deg=(0, 30, 60, 90, 120),
        X0_dot_list=None,
        u0=None,
        v0=None,
        x0=0.0,
        y0=0.0,
        y0_dot=0.0,
        L=20.0,
        figsize=(7, 6),
        view_elev=9,
        view_azim=-105,
        lam=0.0,
        ax=None,
        show=True,
        draw_u_planes=True
):
    r"""Construct and optionally plot a bundle of null geodesics.

    This helper builds a family of null geodesics emitted from a common initial
    event in Brinkmann coordinates and integrates them on a uniform
    :math:`u`-grid. The resulting curves can be plotted as a sampled
    light cone.

    Initial transverse velocities can be prescribed explicitly through
    ``X0_dot_list``. If they are not given, they are generated from the angle
    list ``angles_deg`` via :meth:`PlaneWaveModel.alpha_for_phi` and
    :meth:`PlaneWaveModel.x0_dot_from_alpha`, using both signs of the
    corresponding :math:`x`-velocity.

    Parameters
    ----------
    model : PlaneWaveModel
        Plane-wave background model.
    u_end : float or None, optional
        Final value of :math:`u`. If omitted, the first finite conjugate value
        returned by :meth:`PlaneWaveModel.first_conjugate_u` is used.
    n_u : int, default=1200
        Number of grid points in the :math:`u`-integration grid.
    angles_deg : sequence of float, optional
        List of azimuthal angles in degrees in the x,z-plane used to generate initial null
        directions when ``X0_dot_list`` is not provided.
    X0_dot_list : sequence of array-like or None, optional
        Explicit list of transverse initial velocities
        :math:`\dot X_0 = (\dot x_0, \dot y_0)`. If given, ``angles_deg`` is
        ignored.
    u0, v0 : float or None, optional
        Initial Brinkmann coordinates. Defaults to ``model.u0`` and
        ``model.v0``.
    x0, y0 : float, default=0.0
        Initial transverse position.
    y0_dot : float, default=0.0
        Initial :math:`y`-velocity used when directions are generated from
        ``angles_deg``.
    L : float, default=20.0
        Plotting scale passed to :func:`gravilens.plotting.setup_scene`.
    figsize : tuple, default=(7, 6)
        Figure size used when a new figure is created.
    view_elev, view_azim : float, optional
        Matplotlib 3D viewing angles.
    lam : float, default=0.0
        Normalization constant passed to the generated initial data. The
        default corresponds to null geodesics in the conventions of this
        package.
    ax : matplotlib.axes.Axes or None, optional
        Existing 3D axis. If omitted, a new figure and axis are created.
    show : bool, default=True
        If ``True`` and a new figure is created internally, display the figure.
    draw_u_planes : bool, default=True
        Forwarded to :func:`gravilens.plotting.setup_scene`.

    Returns
    -------
    tuple
        Tuple ``(fig, ax, solutions, cutpoints, coeff)`` where

        - ``fig`` is the Matplotlib figure,
        - ``ax`` is the 3D axis,
        - ``solutions`` is the list of computed geodesic solutions,
        - ``cutpoints`` contains the endpoints :math:`(z, x, t)` at ``u_end``,
        - ``coeff`` contains quadratic fit coefficients for the cut set if at
          least three cut points are available, otherwise ``None``.
    """
    if u0 is None:
        u0 = model.u0
    if v0 is None:
        v0 = model.v0

    if u_end is None:
        u_end = model.first_conjugate_u(u0=u0)
        if u_end is None:
            raise ValueError("No finite first conjugate point; pass u_end explicitly.")

    u_grid = np.linspace(u0, u_end, n_u)

    created_here = False
    if ax is None:
        fig, ax = gp.setup_scene(model, L=L, u0=u0 ,u_end=u_end, figsize=figsize,
                                       view_elev=view_elev, view_azim=view_azim,draw_u_planes=draw_u_planes)
        created_here = True
    else:
        fig = ax.figure

    initials = []

    if X0_dot_list is not None:
        for vec in X0_dot_list:
            initials.append(
                InitialData(
                    u0=u0,
                    v0=v0,
                    x0=x0,
                    y0=y0,
                    x0_dot=float(vec[0]),
                    y0_dot=float(vec[1]),
                    lam=lam,
                )
            )
    else:
        for phi in angles_deg:
            alpha = model.alpha_for_phi(np.deg2rad(phi))
            xdot_mag = model.x0_dot_from_alpha(alpha, x0, y0, y0_dot)
            for sgn in (-1.0, 1.0):
                initials.append(
                    InitialData(
                        u0=u0,
                        v0=v0,
                        x0=x0,
                        y0=y0,
                        x0_dot=sgn * xdot_mag,
                        y0_dot=y0_dot,
                        lam=lam,
                    )
                )

    cutpoints = []
    solutions = []

    for initial in initials:
        sol, _ = gp.add_geodesic(ax, model, initial, u_grid)
        solutions.append(sol)
        ax.scatter(sol.z[-1], sol.x[-1], sol.t[-1], color="k", s=8, depthshade=False)
        cutpoints.append((sol.z[-1], sol.x[-1], sol.t[-1]))

    if len(cutpoints) >= 3:
        cp = np.asarray(cutpoints, dtype=float)
        coeff = np.polyfit(cp[:, 1], cp[:, 0], 2)
        x_fit = np.linspace(cp[:, 1].min(), cp[:, 1].max(), 400)
        z_fit = np.polyval(coeff, x_fit)
        t_fit = z_fit + np.sqrt(2.0) * u_end
        ax.plot(z_fit, x_fit, t_fit, "k--", lw=0.9, label="Parabel-Fit")
    else:
        coeff = None

    if created_here and show:
        plt.tight_layout()
        plt.show()

    return fig, ax, solutions, cutpoints, coeff


def timelike_geodesic_through_event(
    model,
    obs_event,
    lam_timelike,
    X_obs_dot_timelike,
    u_start=None,
    u_end=None,
    n_back=600,
    n_fwd=1000,
):
    r"""Construct a timelike geodesic through a prescribed event.

    The geodesic is specified by an event in Brinkmann coordinates together
    with a transverse velocity and normalization constant
    :math:`\lambda < 0`. The routine then integrates the geodesic backward
    and forward in :math:`u`, whenever the corresponding bounds are provided.

    Parameters
    ----------
    model : PlaneWaveModel
        Plane-wave background model.
    obs_event : GeodesicEvent
        Event-like object with attributes ``u``, ``v``, ``x``, and ``y``.
    lam_timelike : float
        Normalization constant of the timelike geodesic.
    X_obs_dot_timelike : array-like, shape (2,)
        Transverse velocity :math:`(\dot x, \dot y)` at the prescribed event.
    u_start : float or None, optional
        Initial value of :math:`u` for backward integration. Defaults to
        ``model.u0``.
    u_end : float or None, optional
        Final value of :math:`u` for forward integration. If omitted, the first
        finite conjugate value returned by
        :meth:`PlaneWaveModel.first_conjugate_u` is used.
    n_back : int, default=600
        Number of grid points for backward integration.
    n_fwd : int, default=1000
        Number of grid points for forward integration.

    Returns
    -------
    tuple
        Tuple ``(past, future)`` of :class:`GeodesicSolution` objects. Each
        entry is ``None`` if the corresponding integration interval is absent.

    Notes
    -----
    The event itself is used as initial data,

    .. math::

       (u_0, v_0, x_0, y_0) = (u_{\mathrm{obs}}, v_{\mathrm{obs}},
       x_{\mathrm{obs}}, y_{\mathrm{obs}}),

    together with the transverse velocity ``X_obs_dot_timelike`` and the
    prescribed value of :math:`\lambda`.
    """
    if u_start is None:
        u_start = model.u0
    if u_end is None:
        u_end = model.first_conjugate_u(u0=u_start)

    X_obs_dot_timelike = np.asarray(X_obs_dot_timelike, dtype=float)

    initial = InitialData(
        u0=float(obs_event.u),
        v0=float(obs_event.v),
        x0=float(obs_event.x),
        y0=float(obs_event.y),
        x0_dot=float(X_obs_dot_timelike[0]),
        y0_dot=float(X_obs_dot_timelike[1]),
        lam=float(lam_timelike),
    )

    past = None
    future = None

    if u_start is not None:
        u_grid_back = np.linspace(obs_event.u, u_start, n_back)
        past = model.solve_geodesic(initial, u_grid_back)

    if u_end is not None:
        u_grid_fwd = np.linspace(obs_event.u, u_end, n_fwd)
        future = model.solve_geodesic(initial, u_grid_fwd)

    return past, future

def comoving_geodesic_through_event(
    model,
    ref_geo,
    target_event,
    u_match,
    u_start=None,
    u_end=None,
    n_back=600,
    n_fwd=600,
):
    r"""Construct a geodesic comoving with a reference geodesic at ``u_match``.

    This routine determines a new geodesic that shares the transverse
    velocity of a reference geodesic at the matching parameter value
    :math:`u_{\mathrm{match}}`, while passing through a prescribed target
    event at another value of :math:`u`.

    The construction uses the transverse fundamental matrices
    :math:`A(u,u_0)` and :math:`B(u,u_0)` to recover the unknown initial
    transverse position of the new geodesic from the condition that it reaches
    the target event.

    Parameters
    ----------
    model : PlaneWaveModel
        Plane-wave background model.
    ref_geo : GeodesicSolution
        Reference geodesic providing the matching velocity and normalization
        constant.
    target_event : GeodesicEvent
        Event-like object with attributes ``u``, ``v``, ``x``, and ``y`` that
        must lie on the constructed geodesic.
    u_match : float
        Value of :math:`u` at which the new geodesic is required to be
        comoving with ``ref_geo``.
    u_start : float or None, optional
        Initial value of :math:`u` for backward integration.
    u_end : float or None, optional
        Final value of :math:`u` for forward integration.
    n_back, n_fwd : int, default=600
        Number of grid points for backward and forward integration.

    Returns
    -------
    tuple
        Tuple ``(past, future, initial_src_o)`` where ``past`` and ``future``
        are :class:`GeodesicSolution` objects or ``None``, and
        ``initial_src_o`` is the reconstructed :class:`InitialData` at
        :math:`u=u_{\mathrm{match}}`.

    Raises
    ------
    ValueError
        If the reference geodesic does not provide transverse derivatives or a
        value of ``lam``.

    Notes
    -----
    Let :math:`u_o = u_{\mathrm{match}}` and :math:`u_t` be the
    :math:`u`-coordinate of the target event. If :math:`X_t` denotes the
    transverse target position and :math:`\dot X_o` the transverse velocity of
    the reference geodesic at :math:`u_o`, then the unknown initial
    transverse position :math:`X_{\mathrm{src}}(u_o)` is obtained from

    .. math::

       X_t = A(u_t,u_o)\,X_{\mathrm{src}}(u_o) + B(u_t,u_o)\,\dot X_o.

    Hence

    .. math::

       X_{\mathrm{src}}(u_o)
       = A(u_t,u_o)^{-1}\left(X_t - B(u_t,u_o)\dot X_o\right).

    The corresponding initial Brinkmann coordinate :math:`v(u_o)` is then
    reconstructed from the closed-form expression for :math:`v(u)`.
    """
    match_event = ref_geo.event_at_u(u_match)

    if match_event.x_dot is None or match_event.y_dot is None:
        raise ValueError("Reference geodesic needs transverse derivatives for comoving construction.")
    if ref_geo.lam is None:
        raise ValueError("Reference geodesic needs lam for comoving construction.")

    u_o = float(match_event.u)
    Xdot_o = np.array([match_event.x_dot, match_event.y_dot], dtype=float)
    lam = float(ref_geo.lam)

    u_t = float(target_event.u)
    X_t = np.array([target_event.x, target_event.y], dtype=float)

    A_to = model.A(u_t, u_o)
    B_to = model.B(u_t, u_o)

    X_src_o = np.linalg.solve(A_to, X_t - B_to @ Xdot_o)

    Xdot_src_t = model.transverse_geodesic_dot(u_t, X_src_o, Xdot_o, u0=u_o)

    v_src_o = float(target_event.v) - 0.5 * (
        -lam * (u_t - u_o)
        + Xdot_src_t @ X_t
        - Xdot_o @ X_src_o
    )

    initial_src_o = InitialData(
        u0=u_o,
        v0=v_src_o,
        x0=float(X_src_o[0]),
        y0=float(X_src_o[1]),
        x0_dot=float(Xdot_o[0]),
        y0_dot=float(Xdot_o[1]),
        lam=lam,
    )

    past = None
    future = None

    if u_start is not None:
        u_back = np.linspace(u_o, u_start, n_back)
        past = model.solve_geodesic(initial_src_o, u_back)

    if u_end is not None:
        u_fwd = np.linspace(u_o, u_end, n_fwd)
        future = model.solve_geodesic(initial_src_o, u_fwd)

    return past, future, initial_src_o