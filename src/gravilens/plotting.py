import numpy as np
import matplotlib.pyplot as plt

from gravilens.core.base import brinkmann_to_minkowski


def draw_u_plane(ax, u_c, xlim, zlim, n=80,
                 face_color='gray', face_alpha=0.28,
                 grid=True, grid_step=20, grid_color='k', grid_alpha=0.25, grid_lw=0.5):
    r"""Draw the null hyperplane of constant :math:`u` in Minkowski coordinates.

    In Brinkmann and Minkowski coordinates one has

    .. math::

       u = \frac{t-z}{\sqrt{2}},
       \qquad
       t = z + \sqrt{2}\,u.

    Hence a surface of constant :math:`u=u_c` is represented in the
    :math:`(z,x,t)` plot by a lightlike plane

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Three-dimensional axis on which the surface is drawn.
    u_c : float
        Constant value of the Brinkmann coordinate :math:`u`.
    xlim, zlim : float or tuple of float
        Plotting range in :math:`x` and :math:`z`. If a scalar is given, a
        symmetric interval is used.
    n : int, default=80
        Number of sampling points per coordinate direction.
    face_color : str, default='gray'
        Surface color.
    face_alpha : float, default=0.28
        Surface transparency.
    grid : bool, default=True
        If ``True``, overlay a wireframe grid on the surface.
    grid_step : int, default=20
        Step size used to subsample the wireframe grid.
    grid_color : str, default='k'
        Wireframe color.
    grid_alpha : float, default=0.25
        Wireframe transparency.
    grid_lw : float, default=0.5
        Wireframe line width.

    Returns
    -------
    None
        The surface is drawn in-place on ``ax``.
    """
    if np.isscalar(xlim):
        xlim = (-xlim, xlim)
    if np.isscalar(zlim):
        zlim = (-zlim, zlim)

    z = np.linspace(zlim[0], zlim[1], n)
    x = np.linspace(xlim[0], xlim[1], n)
    Z, X = np.meshgrid(z, x, indexing='ij')
    T = Z + np.sqrt(2.0) * u_c

    ax.plot_surface(Z, X, T, color=face_color, alpha=face_alpha, linewidth=0)

    if grid:
        s = max(1, int(grid_step))
        r_idx = np.unique(np.r_[0:Z.shape[0]:s, Z.shape[0] - 1])
        c_idx = np.unique(np.r_[0:X.shape[1]:s, X.shape[1] - 1])
        Zw = Z[np.ix_(r_idx, c_idx)]
        Xw = X[np.ix_(r_idx, c_idx)]
        Tw = T[np.ix_(r_idx, c_idx)]
        ax.plot_wireframe(Zw, Xw, Tw, color=grid_color, linewidth=grid_lw, alpha=grid_alpha)



def setup_scene(model, L=20.0, u0=None, u_end=None, figsize=(7, 6), view_elev=9, view_azim=-105,draw_u_planes=True):
    r"""Create a standard 3D Minkowski scene for geodesic visualization.

    The scene uses coordinates :math:`(z,x,t)` and is intended for plotting
    plane-wave geodesics after conversion from Brinkmann coordinates. By
    default, the initial null hyperplane :math:`u=u_0` and the final plane
    :math:`u=u_{\mathrm{end}}` are drawn as reference surfaces.

    Parameters
    ----------
    model : PlaneWaveModel
        Plane-wave background model providing defaults such as ``u0`` and the
        first conjugate value.
    L : float, default=20.0
        Overall spatial plotting scale.
    u0 : float or None, optional
        Initial value of :math:`u`. Defaults to ``model.u0``.
    u_end : float or None, optional
        Final value of :math:`u`. Defaults to
        ``model.first_conjugate_u(u0=u0)``.
    figsize : tuple, default=(7, 6)
        Figure size passed to Matplotlib.
    view_elev, view_azim : float, optional
        Elevation and azimuth of the 3D camera.
    draw_u_planes : bool, default=True
        If ``True``, draw the reference planes :math:`u=u_0` and
        :math:`u=u_{\mathrm{end}}`.

    Returns
    -------
    tuple
        Tuple ``(fig, ax)`` consisting of the Matplotlib figure and the 3D axis.

    Notes
    -----
    This function prepares only the scene geometry and axis styling. Geodesics
    can subsequently be added with :func:`add_geodesic` or
    :func:`plot_solution`.
    """
    if u0 is None:
        u0 = model.u0
    if u_end is None:
        u_end = model.first_conjugate_u(u0=u0)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(0, 0, 0, color='k', s=8, depthshade=False)
    ax.grid(False)
    ax.set_zlim(-L/2-10, 1.5*L+10)
    ax.set_ylim(-L-10, L+10)
    ax.set_xlim(-1.5*L-10, L/2+10)

    if draw_u_planes:
        draw_u_plane(
            ax, model.u0,
            xlim=(-1.5*L+2, 1.5*L-2),
            zlim=(-0.75*L+2.5, 1.0*L-7.5),
            n=80
        )

        draw_u_plane(
            ax, u_end,
            xlim=(-1.5*L+2, 1.5*L-2),
            zlim=(-0.75*L+2.5, 1.0*L-7.5),
            n=80
        )

    ax.set_xlabel(r"$z$", fontsize=16, labelpad=6)
    ax.set_ylabel(r"$x$", fontsize=16, labelpad=6)
    ax.set_zlabel(r"$t$", fontsize=16, labelpad=6)
    ax.view_init(elev=view_elev, azim=view_azim)
    return fig, ax


def add_geodesic(ax, model, initial, u_grid, **plt_kwargs):
    r"""Solve and plot a geodesic on a given :math:`u`-grid.

    This is a convenience wrapper around
    :meth:`PlaneWaveModel.solve_geodesic`. The resulting trajectory is plotted
    in Minkowski coordinates :math:`(z,x,t)`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Three-dimensional axis on which the geodesic is plotted.
    model : PlaneWaveModel
        Plane-wave background model.
    initial : InitialData
        Initial Brinkmann data at :math:`u=u_0`.
    u_grid : numpy.ndarray
        One-dimensional grid of :math:`u` values used for integration.
    **plt_kwargs
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    tuple
        Tuple ``(sol, line)`` where ``sol`` is the computed
        :class:`GeodesicSolution` and ``line`` is the corresponding Matplotlib
        line object.
    """
    sol = model.solve_geodesic(initial, u_grid)
    defaults = {"lw": 1.0, "color": "darkcyan"}
    defaults.update(plt_kwargs)
    line = ax.plot(sol.z, sol.x, sol.t, **defaults)[0]
    return sol, line

def draw_partial_v(ax, u, v, x=0.0, y=0.0,
               length=3.0, color="crimson",
               arrow_length_ratio=0.2, lw=1.5):
    t0, z0 = brinkmann_to_minkowski(u, v)
    x0 = float(x)
    dz =  length / np.sqrt(2.0)
    dx =  0.0
    dt =  length / np.sqrt(2.0)
    ax.quiver(z0, x0, t0, dz, dx, dt,
            length=1.0, normalize=False,
            color=color, linewidth=lw,
            arrow_length_ratio=arrow_length_ratio)

def draw_brinkmann_vector(ax, u, v, x=0.0, y=0.0,
                      vec_u=0.0, vec_v=0.0, vec_x=0.0, vec_y=0.0,
                      color="orange", lw=1.6, arrow_length_ratio=0.2):
    t0, z0 = brinkmann_to_minkowski(u, v)
    x0 = float(x)
    y0 = float(y)
    du = float(vec_u); dv = float(vec_v); dx = float(vec_x); dy = float(vec_y)
    dt, dz = brinkmann_to_minkowski(du,dv)
    return ax.quiver(z0, x0, t0, dz, dx, dt,
                    color=color, linewidth=lw,
                    arrow_length_ratio=arrow_length_ratio)


def plot_solution(ax, sol, **plt_kwargs):
    defaults = {"lw": 1.0, "color": "darkcyan"}
    defaults.update(plt_kwargs)
    return ax.plot(sol.z, sol.x, sol.t, **defaults)[0]