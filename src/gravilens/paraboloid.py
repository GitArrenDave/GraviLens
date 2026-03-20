import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
BASE_FONTSIZE = 28

mpl.rcParams.update({
    # Grundschrift
    "font.size": BASE_FONTSIZE,

    # Achsenlabels (z, t, x)
    "axes.labelsize": BASE_FONTSIZE,

    # Tick-Labels
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,

    # Legende
    "legend.fontsize": BASE_FONTSIZE * 0.8,

    # Titel (falls du welche nutzt)
    "axes.titlesize": BASE_FONTSIZE,

    # Math-Schrift (LaTeX-ähnlich)
    "mathtext.fontset": "cm",
})
def Hx(x,y,alpha):
    r2 = x*x + y*y
    return -2*alpha*x
def Hy(x,y,alpha):
    r2 = x*x + y*y
    return -2*alpha*y
def H(x,y,alpha):
    r2 = x*x + y*y
    return -alpha*r2


def plot_zt_cut_monochrome(
    gp,
    Z, T,
    *,
    v0,
    u_min,
    alpha_wave,
    b,
    epsilon=0.8,
    eta=2.5,
    # plot window
    zlim=(-60, 60),
    tlim=(-60, 30),
    # geodesics
    geo_color="k",
    geo_lw=1.3,
    geo_alpha=0.95,
    # cauchy
    show_cauchy=True,
    cauchy_style=dict(color="0.5", lw=1.6, alpha=0.9, ls=(0,(3,2))),
    u_span_for_cauchy=None,   # optional (u_lo, u_hi)
    # helper from your code
    segments_inside_box_zt=None,
    figsize=None,
    highlight_indices=None, highlight_colors=None,
):
    """
    Standalone 2D plot (z,t): monochrome geodesics + (optional) Cauchy surface.
    Expects Z,T arrays of shape (n_geodesics, n_points).
    """

    if segments_inside_box_zt is None:
        raise ValueError("Bitte segments_inside_box_zt=... übergeben (aus deinem Code).")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_xlim(*zlim)
    ax.set_ylim(*tlim)
    
    # --- Cauchy surface ---
    if show_cauchy:
        if u_span_for_cauchy is None:
            # default: breit genug, damit im unteren Fenster was sichtbar wird
            u_lo = u_min - 120
            u_hi = u_min + 220
        else:
            u_lo, u_hi = u_span_for_cauchy

        u_vals = np.linspace(u_lo, u_hi, 900)
        v_sigma = (
            v0
            - epsilon*(u_vals - u_min)
            - eta*(1/24)*(alpha_wave**2 / b**2)*(u_vals - u_min)**3
        )
        t_sigma, z_sigma = gp.brinkmann_to_minkowski(u_vals, v_sigma)

        segs_sigma = segments_inside_box_zt(z_sigma, t_sigma, zlim=ax.get_xlim(), tlim=ax.get_ylim())
        for zseg, tseg in segs_sigma:
            ax.plot(zseg, tseg, **cauchy_style)
        if highlight_indices is None:
            highlight_indices = []
        if np.isscalar(highlight_indices):
            highlight_indices = [int(highlight_indices)]
        highlight_indices = [int(i) for i in highlight_indices]

        if highlight_colors is None:
            highlight_colors = ["crimson", "royalblue"]  # default 2 Farben
        # Map: index -> (color, lw, zorder)
        hl_map = {}
        for k, idx in enumerate(highlight_indices):
            col = highlight_colors[k % len(highlight_colors)]
            hl_map[idx] = dict(color=col, lw=1.8, zorder=6)  # zorder hoch!

    for i in range(Z.shape[0]):
        style = dict(color=geo_color, lw=geo_lw, zorder=3)
        if i in hl_map:
            style.update(hl_map[i])

        segs = segments_inside_box_zt(Z[i], T[i], zlim=ax.get_xlim(), tlim=ax.get_ylim())
        for zseg, tseg in segs:
            ax.plot(zseg, tseg, alpha=geo_alpha, **style)

        segs = segments_inside_box_zt(
            Z[i], T[i],
            zlim=ax.get_xlim(),
            tlim=ax.get_ylim()
        )

    # --- u/v arrows: anchored at bottom edge, on the z-axis (z=0) ---
    zmin, zmax = ax.get_xlim()
    tmin, tmax = ax.get_ylim()

    # Anchor point exactly on bottom border
    z0 = 0.5*(zmin + zmax)
    t0 = tmin

    # Choose scale relative to plot width/height so it behaves consistently
    Lz = (zmax - zmin)
    Lt = (tmax - tmin)
    uv = 0.18 * min(Lz, Lt)/2          # arrow length scale
    text_off = 0.04 * min(Lz, Lt)/2    # small text offset

    # v points up-right in (z,t), u points up-left
    ax.quiver(z0, t0, +uv, +uv, angles='xy', scale_units='xy', scale=1,
            color='black', width=0.004, zorder=20, clip_on=False)
    ax.quiver(z0, t0, -uv, +uv, angles='xy', scale_units='xy', scale=1,
            color='black', width=0.004, zorder=20, clip_on=False)

    ax.text(z0 + 0.7*uv + text_off, t0 + 0.7*uv - text_off, r'$\vec{v}$',
            color='black', va='center', ha='left', zorder=21, clip_on=False, fontsize=BASE_FONTSIZE)
    ax.text(z0 - 0.7*uv - text_off, t0 + 0.7*uv - text_off+0.02, r'$\vec{u}$',
            color='black', va='center', ha='right', zorder=21, clip_on=False, fontsize=BASE_FONTSIZE)
    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$z$", fontsize=BASE_FONTSIZE)
    ax.set_ylabel(r"$t$", fontsize=BASE_FONTSIZE)
    ax.set_xticks([]); ax.set_yticks([])
    # fig.tight_layout()
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="crimson", lw=2),
        Line2D([0], [0], color="darkorange", lw=2),
    ]

    legend_labels = [
        r"$C_{\mathrm{crimson}}$",
        r"$C_{\mathrm{darkorange}}$",
    ]

    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        fontsize=0.8*BASE_FONTSIZE,
        title=r"$C_{\mathrm{crimson}} < C_{\mathrm{darkorange}}$"
    )
    return fig, ax


def f_cauchy_surface(u, r, alpha, b, eta, epsilon, u0=0.0, v0=0.0):
    """
    Cauchy hypersurface function v = f(u, r)

    Parameters
    ----------
    u : float or ndarray
        Coordinate u
    r : float or ndarray
        Radial coordinate r
    alpha : float
        Wave parameter alpha
    b : float
        Paraboloid parameter b
    eta : float
        Cubic scaling parameter (eta >= 1)
    epsilon : float
        Positive constant ensuring spacelike character
    u0 : float
        Emission event u-coordinate
    v0 : float
        Emission event v-coordinate

    Returns
    -------
    v : float or ndarray
        Value of the hypersurface function
    """

    du = u - u0
    H = -alpha * r**2

    return (
        v0
        + 0.5 * du * H
        - epsilon * du
        - eta * (1.0 / 24.0) * (alpha**2 / b**2) * du**3
    )

def add_cauchy_gridlines_3d_y0(
    ax, gp,
    *,
    alpha, b, eta, epsilon,
    u0=0.0, v0=0.0,
    u_span, x_span,
    Nu_lines=18, Nx_lines=14,
    n_samples=250,
    color="0.25", lw=0.7, a=0.9
):
    """
    Draws coordinate grid lines on the y=0 slice of the Cauchy surface:
    - lines of constant u (vary x)
    - lines of constant x (vary u)
    """

    u_lo, u_hi = u_span
    x_lo, x_hi = x_span

    # --- lines: constant u (vary x) ---
    u_lines = np.linspace(u_lo, u_hi, Nu_lines)
    x_samp = np.linspace(x_lo, x_hi, n_samples)

    for uu in u_lines:
        U = np.full_like(x_samp, uu)
        X = x_samp
        R = np.abs(X)
        V = f_cauchy_surface(U, R, alpha=alpha, b=b, eta=eta, epsilon=epsilon, u0=u0, v0=v0)
        T, Z = gp.brinkmann_to_minkowski(U, V)

        Zc, Xc, Tc = gp.clip_line3d(Z, X, T,
                                   xlim=ax.get_ylim(), zlim=ax.get_xlim(), tlim=ax.get_zlim(), pad=0.0)
        ax.plot(Zc, Xc, Tc, color=color, lw=lw, alpha=a)

    # --- lines: constant x (vary u) ---
    x_lines = np.linspace(x_lo, x_hi, Nx_lines)
    u_samp = np.linspace(u_lo, u_hi, n_samples)

    for xx in x_lines:
        X = np.full_like(u_samp, xx)
        U = u_samp
        R = np.abs(X)
        V = f_cauchy_surface(U, R, alpha=alpha, b=b, eta=eta, epsilon=epsilon, u0=u0, v0=v0)
        T, Z = gp.brinkmann_to_minkowski(U, V)

        Zc, Xc, Tc = gp.clip_line3d(Z, X, T,
                                   xlim=ax.get_ylim(), zlim=ax.get_xlim(), tlim=ax.get_zlim(), pad=0.0)
        ax.plot(Zc, Xc, Tc, color=color, lw=lw, alpha=a)




def add_cauchy_surface_3d_slice_y0(
    ax, gp,
    *,
    alpha, b, eta, epsilon,
    u0=0.0, v0=0.0,
    u_span=None, x_span=None,
    Nu=160, Nx=160,
    color="0.6", alpha_face=0.22,
    edgecolor="none",
    # neu:
    pad=20.02,                 # kleiner Randabstand gegen „Fransen“
    draw_grid=True,
    grid_kwargs=None
):
    # --- ranges ---
    if x_span is None:
        x_lo, x_hi = ax.get_ylim()
        x_span_eff = (x_lo, x_hi)
    else:
        x_span_eff = tuple(x_span)
        x_lo, x_hi = x_span_eff

    if u_span is None:
        u_lo = gp.u_min - 20.0
        u_hi = gp.u_max + 40.0
        u_span_eff = (u_lo, u_hi)
    else:
        u_span_eff = tuple(u_span)
        u_lo, u_hi = u_span_eff

    # --- optional: gridlines first (so surface stays behind) ---
    if draw_grid:
        if grid_kwargs is None:
            grid_kwargs = dict(Nu_lines=18, Nx_lines=14, n_samples=260,
                               color="0.15", lw=0.7, a=0.85)
        add_cauchy_gridlines_3d_y0(
            ax, gp,
            alpha=alpha, b=b, eta=eta, epsilon=epsilon,
            u0=u0, v0=v0,
            u_span=u_span_eff,
            x_span=x_span_eff,
            **grid_kwargs
        )

    # --- mesh ---
    u = np.linspace(u_lo, u_hi, Nu)
    x = np.linspace(x_lo, x_hi, Nx)
    U, X = np.meshgrid(u, x, indexing="ij")
    R = np.abs(X)

    V = f_cauchy_surface(U, R, alpha=alpha, b=b, eta=eta, epsilon=epsilon, u0=u0, v0=v0)
    T, Z = gp.brinkmann_to_minkowski(U, V)

    # --- robust clipping: use a small pad so triangulation at boundary doesn't fray ---
    zmin, zmax = ax.get_xlim()
    xmin, xmax = ax.get_ylim()
    tmin, tmax = ax.get_zlim()

    inside = (
        (Z >= zmin + pad) & (Z <= zmax - pad) &
        (X >= xmin + pad) & (X <= xmax - pad) &
        (T >= tmin + pad) & (T <= tmax - pad)
    )

    Zp = np.where(inside, Z, np.nan)
    Xp = np.where(inside, X, np.nan)
    Tp = np.where(inside, T, np.nan)

    surf = ax.plot_surface(
        Zp, Xp, Tp,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,   # wichtiger gegen Fransen!
        shade=False,
        color=color,
        alpha=alpha_face,
        edgecolor=edgecolor,
    )
    return surf

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import colors as mcolors

def add_depth_cued_geodesic(ax, Z, X, T, *,
                            color="k",
                            lw_front=1.2, lw_back=0.8,
                            alpha_front=1.0, alpha_back=0.25,
                            depth_percentiles=(5, 95),
                            gamma=0.9,
                            sort_back_to_front=False):   # <<< default AUS!
    Z = np.asarray(Z, float)
    X = np.asarray(X, float)
    T = np.asarray(T, float)

    finite = np.isfinite(Z) & np.isfinite(X) & np.isfinite(T)
    if finite.sum() < 2:
        return None

    M = ax.get_proj()

    def proj_depth(z, x, t):
        P = np.c_[z, x, t, np.ones_like(z)]
        Q = P @ M.T
        return Q[:, 2] / Q[:, 3]   # "optische Tiefe" im NDC-Raum

    # Segmente (ohne Reordering)
    idx = np.where(finite)[0]
    runs = []
    start = idx[0]; prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev))
            start = prev = k
    runs.append((start, prev))

    segs_all, d_all = [], []
    for a, b in runs:
        z = Z[a:b+1]; x = X[a:b+1]; t = T[a:b+1]
        if len(z) < 2:
            continue

        pts0 = np.c_[z[:-1], x[:-1], t[:-1]]
        pts1 = np.c_[z[1:],  x[1:],  t[1:]]
        segs = np.stack([pts0, pts1], axis=1)

        zm = 0.5*(z[:-1] + z[1:])
        xm = 0.5*(x[:-1] + x[1:])
        tm = 0.5*(t[:-1] + t[1:])
        d  = proj_depth(zm, xm, tm)

        segs_all.append(segs)
        d_all.append(d)

    if not segs_all:
        return None

    segs = np.concatenate(segs_all, axis=0)
    d    = np.concatenate(d_all, axis=0)

    # robuste Normierung
    p_lo, p_hi = np.percentile(d[np.isfinite(d)], depth_percentiles)
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        s = np.zeros_like(d)
    else:
        s = (d - p_lo) / (p_hi - p_lo)
        s = np.clip(s, 0.0, 1.0)

    # "vorne" = w groß (du kannst bei Bedarf w = s statt 1-s setzen)
    w = 1.0 - s
    w = w**gamma

    alphas = alpha_back + (alpha_front - alpha_back)*w
    lws    = lw_back    + (lw_front   - lw_back)   *w

    rgb = mcolors.to_rgb(color)
    cols = np.c_[np.full_like(alphas, rgb[0]),
                 np.full_like(alphas, rgb[1]),
                 np.full_like(alphas, rgb[2]),
                 alphas]

    if sort_back_to_front:
        order = np.argsort(d)   # optional
        segs, cols, lws = segs[order], cols[order], lws[order]

    lc = Line3DCollection(segs, colors=cols, linewidths=lws)
    lc.set_zorder(20)
    ax.add_collection3d(lc)
    return lc



def add_cauchy_surface_3d_slice_y0_clipped(
    ax, gp,
    *,
    alpha, b, eta, epsilon,
    u0=0.0, v0=0.0,
    u_span=None, x_span=None,
    Nu=140, Nx=140,
    color="0.55", alpha_face=0.18,
    edgecolor="none",
    max_face_span=None,
    max_face_area=None,
):
    # spans
    if x_span is None:
        xlim = ax.get_ylim()
        x_lo, x_hi = xlim
    else:
        x_lo, x_hi = x_span
        xlim = (x_lo, x_hi)

    if u_span is None:
        u_lo = gp.u_min - 20.0
        u_hi = gp.u_max + 40.0
    else:
        u_lo, u_hi = u_span

    zlim = ax.get_xlim()
    tlim = ax.get_zlim()

    u = np.linspace(u_lo, u_hi, Nu)
    x = np.linspace(x_lo, x_hi, Nx)

    # precompute grid in Brinkmann, then Minkowski
    U, X = np.meshgrid(u, x, indexing="ij")
    R = np.abs(X)
    V = f_cauchy_surface(U, R, alpha=alpha, b=b, eta=eta, epsilon=epsilon, u0=u0, v0=v0)
    T, Z = gp.brinkmann_to_minkowski(U, V)

    faces = []

    # build quads and clip polygonally (like your lightcone surface)
    for i in range(Nu - 1):
        for j in range(Nx - 1):
            quad = [
                (Z[i, j],     X[i, j],     T[i, j]),
                (Z[i, j+1],   X[i, j+1],   T[i, j+1]),
                (Z[i+1, j+1], X[i+1, j+1], T[i+1, j+1]),
                (Z[i+1, j],   X[i+1, j],   T[i+1, j]),
            ]

            poly = gp._clip_poly_against_box3d(quad, xlim=xlim, zlim=zlim, tlim=tlim)
            if len(poly) < 3:
                continue

            if (max_face_span is not None) and (gp._poly_max_span(poly) > max_face_span):
                continue
            if (max_face_area is not None) and (gp._poly_area3d(poly) > max_face_area):
                continue

            faces.append(poly)

    coll = Poly3DCollection(
        faces,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=0.0,
        alpha=float(alpha_face),
    )
    coll.set_zsort('min')
    coll.set_zorder(1)
    ax.add_collection3d(coll)
    return coll




def add_colored_geodesic_zt(ax, zseg, tseg, xseg, *, cmap="viridis",
                            vmin=None, vmax=None, alpha_front=0.9, alpha_back=0.05,
                            xfade=None, use_abs=True, lw=1.2, zorder=1,norm=None):
    """
    Zeichnet eine (z,t)-Kurve als LineCollection.
    Farbe kodiert x (oder |x|).
    Zusätzlich: Alpha lokal als Funktion von 'Tiefe' (|x| oder Abstand zu x=0).
    """
    zseg = np.asarray(zseg, float)
    tseg = np.asarray(tseg, float)
    xseg = np.asarray(xseg, float)

    if len(zseg) < 2:
        return None

    # kleine Liniensegmente erzeugen
    pts = np.column_stack([zseg, tseg])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    # Skalarfeld für Farbe (lokal pro Mini-Segment)
    xmid = 0.5 * (xseg[:-1] + xseg[1:])
    sval = np.abs(xmid) if use_abs else xmid


    if norm is None:
        if vmin is None or vmax is None:
            finite = np.isfinite(sval)
            if not np.any(finite):
                return None
            # robust
            lo, hi = np.nanpercentile(sval[finite], [2, 98])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(sval[finite])), float(np.nanmax(sval[finite]) + 1e-12)

        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap)

    rgba = cmap(norm(sval))

    # lokales Fading (Alpha pro Mini-Segment)
    if xfade is None:
        # Default: halbe x-Box (so wie du xlim2 hast)
        xfade = 0.5 * (np.nanmax(np.abs(xseg)) + 1e-12)

    depth = np.abs(xmid)  # "hinten" = größer |x|
    a = alpha_front * (1.0 - depth / xfade)
    a = np.clip(a, alpha_back, alpha_front)
    rgba[:, 3] = a

    lc = LineCollection(segs, colors=rgba, linewidths=lw, zorder=zorder)
    ax.add_collection(lc)
    return lc, norm


def mask_after_first_sign_change(x, *arrs, eps=1e-12,):
    x = np.asarray(x)
    sx = np.sign(x)
    sx[np.abs(x) < eps] = 0.0

    idx = None
    for i in range(len(x)-1):
        if sx[i] != 0.0 and sx[i+1] != 0.0 and sx[i] != sx[i+1]:
            idx = i+3  # ab hier maskieren
            break

    if idx is None:
        return (x, *arrs)

    xm = x.copy()
    xm[idx:] = np.nan
    out = [xm]
    for a in arrs:
        am = np.asarray(a).copy()
        am[idx:] = np.nan
        out.append(am)
    return tuple(out)

def mask_after_first_event(
    x, z, t,
    *,
    xlim, zlim, tlim,
    eps=1e-12):
    """
    Setzt ab dem ersten Ereignis NaN:
      (A) erster Vorzeichenwechsel von x
      (B) erstes Verlassen der Box (xlim,zlim,tlim)
    """

    x = np.asarray(x)
    z = np.asarray(z)
    t = np.asarray(t)

    # -------- Signwechsel von x --------
    sx = np.sign(x)
    sx[np.abs(x) < eps] = 0.0

    idx_sign = None
    for i in range(len(x) - 1):
        if sx[i] != 0.0 and sx[i+1] != 0.0 and sx[i] != sx[i+1]:
            idx_sign = i + 1
            break

    # -------- Box verlassen --------
    xmin, xmax = xlim
    zmin, zmax = zlim
    tmin, tmax = tlim

    inside = (
        (x >= xmin) & (x <= xmax) &
        (z >= zmin) & (z <= zmax) &
        (t >= tmin) & (t <= tmax)
    )

    idx_box = None
    was_inside = inside[0]
    for i in range(1, len(x)):
        if was_inside and not inside[i]:
            idx_box = i
            break
        was_inside = was_inside or inside[i]

    # -------- frühestes Ereignis --------
    candidates = [i for i in (idx_sign, idx_box) if i is not None]
    if not candidates:
        return x, z, t

    idx = min(candidates)

    # -------- maskieren --------
    xm = x.copy()
    zm = z.copy()
    tm = t.copy()

    xm[idx:] = np.nan
    zm[idx:] = np.nan
    tm[idx:] = np.nan

    return xm, zm, tm



def mask_after_box_exit(x, z, t, *, xlim, zlim, tlim):
    """
    Setzt ab dem ersten Verlassen der Box (xlim,zlim,tlim) alles auf NaN.
    """

    x = np.asarray(x)
    z = np.asarray(z)
    t = np.asarray(t)

    xmin, xmax = xlim
    zmin, zmax = zlim
    tmin, tmax = tlim

    inside = (
        (x >= xmin) & (x <= xmax) &
        (z >= zmin) & (z <= zmax) &
        (t >= tmin) & (t <= tmax)
    )

    idx = None
    was_inside = inside[0]

    for i in range(1, len(x)):
        if was_inside and not inside[i]:
            idx = i
            break
        was_inside = was_inside or inside[i]

    if idx is None:
        return x, z, t

    xm = x.copy()
    zm = z.copy()
    tm = t.copy()

    xm[idx:] = np.nan
    zm[idx:] = np.nan
    tm[idx:] = np.nan

    return xm, zm, tm


# def Hx(x,y,alpha):
#     r2 = x*x + y*y
#     return -2*alpha*x*(8*b**2*r2+1)
# def Hy(x,y,alpha):
#     r2 = x*x + y*y
#     return -2*alpha*y*(8*b**2*r2+1)
# def H(x,y,alpha):
#     r2 = x*x + y*y
#     return -alpha*r2*(4*b**2*r2+1)
# def Hx(x,y,alpha):
#     r2 = x*x + y*y
#     return -2*alpha*x*(12*b**2*r2**2+2*r2)
# def Hy(x,y,alpha):
#     r2 = x*x + y*y
#     return -2*alpha*y*(12*b**2*r2**2+2*r2)
# def H(x,y,alpha):
#     r2 = x*x + y*y
#     return -alpha*r2**2*(4*b**2*r2+1)


def mask_outside_zt(z, t, *, zlim, tlim, pad=1e-2):
    z = np.asarray(z); t = np.asarray(t)
    zmin, zmax = zlim; tmin, tmax = tlim
    zmin -= pad; zmax += pad
    tmin -= pad; tmax += pad

    inside = (z >= zmin) & (z <= zmax) & (t >= tmin) & (t <= tmax)

    zm = z.copy(); tm = t.copy()
    zm[~inside] = np.nan
    tm[~inside] = np.nan
    return zm, tm

def segments_inside_box_zt_clip_3arrays(z, t, x, *, zlim, tlim, eps=1e-12):
    """
    Wie segments_inside_box_zt (mit Interpolation an Boxkanten),
    aber trägt x linear mit und gibt (z_seg, t_seg, x_seg) zurück.
    KEIN x-Halbraum, nur (z,t)-Box.
    """
    z = np.asarray(z, float)
    t = np.asarray(t, float)
    x = np.asarray(x, float)

    zmin, zmax = zlim
    tmin, tmax = tlim

    def clip_segment_to_box_zt(z0, t0, z1, t1):
        dz = z1 - z0
        dt = t1 - t0
        p = [-dz, dz, -dt, dt]
        q = [z0 - zmin, zmax - z0, t0 - tmin, tmax - t0]
        s0, s1 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if abs(pi) < eps:
                if qi < 0:
                    return None
            else:
                r = qi / pi
                if pi < 0:
                    s0 = max(s0, r)
                else:
                    s1 = min(s1, r)
                if s0 - s1 > 1e-15:
                    return None
        return max(0.0, s0), min(1.0, s1)

    segs = []
    cur_z, cur_t, cur_x = [], [], []

    for i in range(len(z) - 1):
        if not (np.isfinite(z[i]) and np.isfinite(t[i]) and np.isfinite(x[i]) and
                np.isfinite(z[i+1]) and np.isfinite(t[i+1]) and np.isfinite(x[i+1])):
            if cur_z:
                segs.append((np.array(cur_z), np.array(cur_t), np.array(cur_x)))
                cur_z, cur_t, cur_x = [], [], []
            continue

        z0,t0,x0 = z[i], t[i], x[i]
        z1,t1,x1 = z[i+1], t[i+1], x[i+1]

        c = clip_segment_to_box_zt(z0,t0,z1,t1)
        if c is None:
            if cur_z:
                segs.append((np.array(cur_z), np.array(cur_t), np.array(cur_x)))
                cur_z, cur_t, cur_x = [], [], []
            continue

        s_enter, s_exit = c
        if s_exit < s_enter + 1e-14:
            continue

        # Eintritt/Austritt interpoliert in allen 3
        ze = z0 + s_enter*(z1 - z0)
        te = t0 + s_enter*(t1 - t0)
        xe = x0 + s_enter*(x1 - x0)

        zx = z0 + s_exit*(z1 - z0)
        tx = t0 + s_exit*(t1 - t0)
        xx = x0 + s_exit*(x1 - x0)

        if not cur_z:
            cur_z.append(ze); cur_t.append(te); cur_x.append(xe)

        cur_z.append(zx); cur_t.append(tx); cur_x.append(xx)

        if s_exit < 1.0 - 1e-12:
            segs.append((np.array(cur_z), np.array(cur_t), np.array(cur_x)))
            cur_z, cur_t, cur_x = [], [], []

    if cur_z:
        segs.append((np.array(cur_z), np.array(cur_t), np.array(cur_x)))

    return segs


def segments_inside_box_zt(z, t, *, zlim, tlim, eps=1e-12):
    """
    Zerlegt eine (z,t)-Kurve in Segmente, die innerhalb der Box liegen.
    Bei jedem Eintritt/Austritt wird der Schnittpunkt mit der Boxkante interpoliert.
    
    Rückgabe: Liste von (z_seg, t_seg) Arrays.
    """
    z = np.asarray(z, float)
    t = np.asarray(t, float)

    zmin, zmax = zlim
    tmin, tmax = tlim

    def inside(zz, tt):
        return (zmin - eps <= zz <= zmax + eps) and (tmin - eps <= tt <= tmax + eps)

    def clip_segment_to_box(p0, p1):
        """
        Clippt das Segment p(s)=p0 + s*(p1-p0) gegen Achsen-parallele Box.
        Liefert (s_enter, s_exit) im Intervall [0,1] oder None falls kein Schnitt.
        Liang-Barsky in 2D.
        """
        z0, t0 = p0
        dz = p1[0] - z0
        dt = p1[1] - t0

        p = [-dz, dz, -dt, dt]
        q = [z0 - zmin, zmax - z0, t0 - tmin, tmax - t0]

        s0, s1 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if abs(pi) < eps:
                if qi < 0:
                    return None
            else:
                r = qi / pi
                if pi < 0:
                    s0 = max(s0, r)
                else:
                    s1 = min(s1, r)
                if s0 - s1 > 1e-15:
                    return None
        return max(0.0, s0), min(1.0, s1)

    segs = []
    current_z = []
    current_t = []

    for i in range(len(z) - 1):
        p0 = (z[i], t[i])
        p1 = (z[i+1], t[i+1])

        # NaNs sauber behandeln
        if not (np.isfinite(p0[0]) and np.isfinite(p0[1]) and
                np.isfinite(p1[0]) and np.isfinite(p1[1])):
            if current_z:
                segs.append((np.array(current_z), np.array(current_t)))
                current_z, current_t = [], []
            continue

        c = clip_segment_to_box(p0, p1)

        if c is None:
            if current_z:
                segs.append((np.array(current_z), np.array(current_t)))
                current_z, current_t = [], []
            continue

        s_enter, s_exit = c

        # Eintrittspunkt
        z_enter = p0[0] + s_enter*(p1[0]-p0[0])
        t_enter = p0[1] + s_enter*(p1[1]-p0[1])

        # Austrittspunkt
        z_exit  = p0[0] + s_exit *(p1[0]-p0[0])
        t_exit  = p0[1] + s_exit *(p1[1]-p0[1])

        if not current_z:
            current_z.append(z_enter)
            current_t.append(t_enter)

        current_z.append(z_exit)
        current_t.append(t_exit)

        # Wenn Segment vorzeitig endet → schließen
        if s_exit < 1.0 - 1e-12:
            segs.append((np.array(current_z), np.array(current_t)))
            current_z, current_t = [], []

    if current_z:
        segs.append((np.array(current_z), np.array(current_t)))

    return segs

def all_zero_crossings(u, x, *, u0=0.0, eps=1e-12, max_n=None, treat_touch_as_event=True):
    u = np.asarray(u, float)
    x = np.asarray(x, float)

    # starte erst ab u0
    i0 = np.searchsorted(u, u0 + eps)
    if i0 >= len(u) - 1:
        return []

    def sgn(val):
        if abs(val) < eps:
            return 0.0
        return 1.0 if val > 0 else -1.0

    zeros = []
    s_prev = sgn(x[i0])

    for i in range(i0 + 1, len(u)):
        s_curr = sgn(x[i])

        # "Touch" zählt?
        if treat_touch_as_event and s_curr == 0.0 and s_prev != 0.0:
            zeros.append(float(u[i]))
        # echter Vorzeichenwechsel
        elif (s_prev != 0.0) and (s_curr != 0.0) and (s_prev != s_curr):
            u1, u2 = u[i-1], u[i]
            x1, x2 = x[i-1], x[i]
            denom = (x2 - x1)
            tau = 1.0 if abs(denom) < eps else float(np.clip(-x1 / denom, 0.0, 1.0))
            zeros.append(float(u1 + tau * (u2 - u1)))

        s_prev = s_curr

        if max_n is not None and len(zeros) >= max_n:
            break

    return zeros




def cut_geodesic_first_event(
    u, x, z, t,
    *,
    xlim, zlim, tlim,
    eps=1e-12,
    interp=True,
    treat_touch_as_event=True,
    stop_on_x_zero=True
):
    """
    Schneidet/maskiert eine Geodäte beim ersten Ereignis:
      (A) erstes Nulldurchgang-Ereignis von x(u) (Vorzeichenwechsel oder |x|<eps)
      (B) erstes Verlassen der Box in (z,x,t)

    Rückgabe:
      (u_m, x_m, z_m, t_m, info)

    wobei u_m etc. gleiche Länge behalten und ab Cut NaN gesetzt werden.
    info ist ein dict mit:
      - event: "x_zero" oder "box_exit"
      - u_event, x_event, z_event, t_event  (interpoliert, wenn interp=True)
      - idx_event (Index im Originalarray, an dem das Event "passiert")
    """
    u = np.asarray(u, float)
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    t = np.asarray(t, float)

    n = len(u)
    if n < 2:
        return u, x, z, t, {"event": None}

    xmin, xmax = xlim
    zmin, zmax = zlim
    tmin, tmax = tlim

    # ---------- Hilfsfunktionen ----------
    def sgn(val):
        if abs(val) < eps:
            return 0.0
        return 1.0 if val > 0 else -1.0

    def inside_point(xx, zz, tt):
        return (xmin <= xx <= xmax) and (zmin <= zz <= zmax) and (tmin <= tt <= tmax)

    # ---------- Event A: x=0 (oder Vorzeichenwechsel) ----------
    idx_x = None
    tau_x = None  # Interpolationsanteil zwischen idx_x-1 und idx_x

    
    if stop_on_x_zero:
        sx0 = sgn(x[0])
        for i in range(1, n):
            s_prev = sgn(x[i-1])
            s_curr = sgn(x[i])

            # "touch" zählt als Event?
            if treat_touch_as_event:
                if s_curr == 0.0 and s_prev != 0.0:
                    idx_x = i
                    tau_x = 1.0
                    break
                if s_prev == 0.0 and s_curr != 0.0:
                    # Startpunkt lag numerisch auf 0 -> ignoriere das, weiter
                    continue

            # echter Vorzeichenwechsel
            if (s_prev != 0.0) and (s_curr != 0.0) and (s_prev != s_curr):
                idx_x = i
                # lineare Interpolation x=0 zwischen i-1 und i
                denom = (x[i] - x[i-1])
                if abs(denom) < eps:
                    tau_x = 1.0
                else:
                    tau_x = float(np.clip(-x[i-1] / denom, 0.0, 1.0))
                break

    # ---------- Event B: Box verlassen ----------
    idx_box = None
    tau_box = None

    inside0 = inside_point(x[0], z[0], t[0])
    was_inside = inside0

    for i in range(1, n):
        inside_i = inside_point(x[i], z[i], t[i])
        if was_inside and (not inside_i):
            idx_box = i
            # Interpolation: Schnitt mit einer der 6 Box-Ebenen,
            # nimm den kleinsten tau in [0,1], der eine Grenze erreicht.
            if interp:
                i0 = i - 1
                i1 = i

                taus = []

                def add_tau(a0, a1, lo, hi):
                    da = a1 - a0
                    if abs(da) < eps:
                        return
                    # Schnitt lo
                    tl = (lo - a0) / da
                    if 0.0 <= tl <= 1.0:
                        taus.append(tl)
                    # Schnitt hi
                    th = (hi - a0) / da
                    if 0.0 <= th <= 1.0:
                        taus.append(th)

                add_tau(x[i0], x[i1], xmin, xmax)
                add_tau(z[i0], z[i1], zmin, zmax)
                add_tau(t[i0], t[i1], tmin, tmax)

                tau_box = float(min(taus)) if taus else 1.0
            else:
                tau_box = 1.0
            break

        was_inside = was_inside or inside_i

    # ---------- frühestes Event auswählen ----------
    candidates = []
    if idx_x is not None:
        candidates.append(("x_zero", idx_x, tau_x))
    if idx_box is not None:
        candidates.append(("box_exit", idx_box, tau_box))

    if not candidates:
        return u, x, z, t, {"event": None}

    event, idx, tau = min(candidates, key=lambda it: it[1])

    # ---------- Ereignispunkt bestimmen ----------
    # Event liegt zwischen idx-1 und idx (oder genau auf idx, wenn tau=1)
    i0 = max(0, idx - 1)
    i1 = idx

    if interp and (i1 > i0):
        u_ev = u[i0] + tau * (u[i1] - u[i0])
        x_ev = x[i0] + tau * (x[i1] - x[i0])
        z_ev = z[i0] + tau * (z[i1] - z[i0])
        t_ev = t[i0] + tau * (t[i1] - t[i0])

        # Bei x_zero wollen wir x_ev exakt auf 0 setzen (numerisch sauber)
        if event == "x_zero":
            x_ev = 0.0
    else:
        u_ev, x_ev, z_ev, t_ev = u[i1], x[i1], z[i1], t[i1]
        if event == "x_zero":
            x_ev = 0.0

    # ---------- maskieren: ab idx alles NaN ----------
    u_m = u.copy()
    x_m = x.copy()
    z_m = z.copy()
    t_m = t.copy()

    x_m[idx:] = np.nan
    z_m[idx:] = np.nan
    t_m[idx:] = np.nan
    # u kann man lassen; oder auch NaN setzen, ist egal fürs plotten:
    # u_m[idx:] = np.nan

    # Für schönes Ende: überschreibe den letzten sichtbaren Punkt (idx-1) mit Eventpunkt
    if idx - 1 >= 0:
        u_m[idx - 1] = u_ev
        x_m[idx - 1] = x_ev
        z_m[idx - 1] = z_ev
        t_m[idx - 1] = t_ev

    info = {
        "event": event,
        "idx_event": idx,
        "u_event": float(u_ev),
        "x_event": float(x_ev),
        "z_event": float(z_ev),
        "t_event": float(t_ev),
    }
    return u_m, x_m, z_m, t_m, info

class SandwichParabolWavePlotter:
    def __init__(self, Hx, Hy, H, u_min=0.0, u_max=1.0):
        self.Hx = Hx
        self.Hy = Hy
        self.H  = H
        self.u_min = float(u_min)
        self.u_max = float(u_max)
    def brinkmann_to_minkowski(self,u, v):
        t = (v + u) / np.sqrt(2)
        z = (v - u) / np.sqrt(2)
        return t, z

    def minkowski_to_brinkmann(self,t, z):
        v = (t + z) / np.sqrt(2)
        u = (t - z) / np.sqrt(2)
        return u, v
    
    def dgl(self, u, Z, lam):
        x, xd, y, yd, v = Z
        r2 = x*x + y*y
        Hx  = self.Hx(x, y, alpha)
        Hy  = self.Hy(x, y, alpha)

        xdd = -4*b**2*(xd*xd + yd*yd)/(4*b**2*r2+1)*x + 1/(2*(4*b**2*r2+1))*((4*b**2*y**2+1)*Hx-4*b**2*x*y*Hy)
        ydd = -4*b**2*(xd*xd + yd*yd)/(4*b**2*r2+1)*y + 1/(2*(4*b**2*r2+1))*((4*b**2*x**2+1)*Hy-4*b**2*x*y*Hx)
        vd = 0.5*(-lam + self.H(x,y,alpha) + (1+4*b**2*x**2)*xd*xd + (1+4*b**2*y**2)*yd*yd + 8*b**2*x*y*xd*yd)
        return [xd, xdd, yd, ydd, vd]
    
    def integrate_geodesic(self, u_span, u0, x0, x0_dot, y0, y0_dot, v0, lam,
                           rtol=1e-9, atol=1e-12, n_eval=800):
        t_eval = np.linspace(u_span[0], u_span[1], n_eval)
        Z0 = [x0, x0_dot, y0, y0_dot, v0]
        sol = solve_ivp(lambda uu, ZZ: self.dgl(uu, ZZ, lam),
                        u_span, Z0, t_eval=t_eval, rtol=rtol, atol=atol)
        u = sol.t
        x, xd, y, yd, v = sol.y
        t, z = self.brinkmann_to_minkowski(u, v)
        return {"u":u, "x":x, "x_dot":xd, "y":y, "y_dot":yd, "v":v, "t":t, "z":z, "lam":lam}
    
    @staticmethod
    def alpha_for_phi(phi_deg):
        phi = np.deg2rad(phi_deg)
        c = np.cos(phi)
        return (np.pi/4 if np.isclose(c, 0.0) else np.arctan(1.0/c) - np.pi/4)

    def x0_dot_from_alpha(self, alpha_rad, u0, x0, y0, y0_dot):
        # Nullbedingung: (4b^2r^2+1)r_dot^2 + H(u0,x0,y0) = 2 tan(alpha_deg)
        r02 = x0*x0+y0*y0
        rhs = (2*np.tan(alpha_rad) - self.H(x0, y0,alpha))/(4*b**2*r02+1) - y0_dot**2
        return np.sqrt(max(0.0, rhs))
    
    
    def integrate_v_line(self, u, v_start, v_end, x=0.0, y=0.0, n=200):
        """
        Exakte Nullgeodäte entlang ∂_v bei festem (u,x,y).
        Gibt Arrays in Brinkmann+Minkowski zurück.
        """
        v = np.linspace(v_start, v_end, int(max(2, n)))
        u_arr = np.full_like(v, float(u))
        x_arr = np.full_like(v, float(x))
        y_arr = np.full_like(v, float(y))
        t, z = self.brinkmann_to_minkowski(u_arr, v)
        return {"u":u_arr, "v":v, "x":x_arr, "y":y_arr, "t":t, "z":z}
    
    def plot_v_line(self, ax, u, v_start=None, v_end=None, x=0.0, y=0.0,
                n=400, color="crimson", lw=1.6, alpha=0.9, clip=True, pad=0.0):
        # Box-Intervall für v (bei festem u)
        zmin,zmax = ax.get_xlim(); xmin,xmax = ax.get_ylim(); tmin,tmax = ax.get_zlim()
        Iv_t = (np.sqrt(2)*tmin - u, np.sqrt(2)*tmax - u)
        Iv_z = (np.sqrt(2)*zmin + u, np.sqrt(2)*zmax + u)
        v_lo, v_hi = max(Iv_t[0], Iv_z[0]), min(Iv_t[1], Iv_z[1])

        # Nur fehlende Grenze automatisch setzen
        if v_start is None and v_end is None:
            v_start, v_end = v_lo, v_hi
        elif v_start is None:
            v_start = v_lo
        elif v_end is None:
            v_end = v_hi

        # Optional: NICHT einklemmen – so bleibt z.B. v_start=0 erhalten
        # v_start = max(v_start, v_lo); v_end = min(v_end, v_hi)

        if not (v_end > v_start):
            return None, None

        data = self.integrate_v_line(u=u, v_start=v_start, v_end=v_end, x=x, y=y, n=n)
        Z, X, T = data["z"], data["x"], data["t"]
        if clip:
            Z, X, T = self.clip_line3d(Z, X, T,
                                    xlim=ax.get_ylim(), zlim=ax.get_xlim(), tlim=ax.get_zlim(), pad=pad)
        ln, = ax.plot(Z, X, T, color=color, lw=lw, alpha=float(np.clip(alpha, 0.0, 1.0)))
        return ln, data
    #Versuch Lichtgeodäten zu normieren
    def initial_transverse_velocity_normalized(self, theta, u0, x0, y0, lam=0.0):
        # Normierung am Scheitelpunkt (x0=y0=0 empfohlen):
        # choose -k·U = 1 with U=∂t, and param u so k^u=1
        kv = np.sqrt(2) - 1.0                 # from -k·U=1
        vperp2 = 2.0*kv - (self.H(x0,y0,alpha) - lam)  # generalizing if needed
        vperp = np.sqrt(max(0.0, vperp2))
        xd0 = vperp*np.cos(theta)
        yd0 = vperp*np.sin(theta)
        return xd0, yd0
    
    
    
    
    def lightcone_mesh_crosswrap(self, u0=0.0, v0=0.0, x0=0.0, y0=0.0, y0_dot=0.0,
                                lam=0.0, phis=None, u_end=None, n_eval=600):
        # φ in [0,180) – ohne 180°, damit u-Parametrisierung nicht kollabiert
        if phis is None:
            phis = np.linspace(0.0, 180.0, 31, endpoint=False)
        else:
            phis = np.asarray(phis, float)
            phis = phis[(np.mod(phis,360.0) >= 0) & (np.mod(phis,360.0) < 180.0)]
            phis = np.unique(phis)

        if u_end is None:
            width = max(1.0, self.u_max - self.u_min)
            u_end = self.u_max + 3*width
        u_span = (u0, u_end)

        rowsZ, rowsX, rowsT = [], [], []
        rowsU, rowsV = [], []
        rowsC = []  
        # +Äste aufsteigend
        for phi in phis:
            a  = self.alpha_for_phi(phi)
            xD = self.x0_dot_from_alpha(a, u0, x0, y0, y0_dot)
            g  = self.integrate_geodesic(u_span, u0, x0, +xD, y0, y0_dot, v0, lam, n_eval=n_eval)
            rowsZ.append(g["z"]); rowsX.append(g["x"]); rowsT.append(g["t"]); rowsU.append(g["u"]); rowsV.append(g["v"])
            _x0 = g["x"][0]; _y0 = g["y"][0]
            _xd0 = g["x_dot"][0]; _yd0 = g["y_dot"][0]
            C0 = (1 + 4*b**2*(_x0*_x0 + _y0*_y0))*(_xd0*_xd0 + _yd0*_yd0) - self.H(_x0, _y0, alpha)
            rowsC.append(C0)
        # −Äste absteigend
        for phi in phis[::-1]:
            a  = self.alpha_for_phi(phi)
            xD = self.x0_dot_from_alpha(a, u0, x0, y0, y0_dot)
            g  = self.integrate_geodesic(u_span, u0, x0, -xD, y0, y0_dot, v0, lam, n_eval=n_eval)
            rowsZ.append(g["z"]); rowsX.append(g["x"]); rowsT.append(g["t"]); rowsU.append(g["u"]); rowsV.append(g["v"])
            _x0 = g["x"][0]; _y0 = g["y"][0]
            _xd0 = g["x_dot"][0]; _yd0 = g["y_dot"][0]
            C0 = (1 + 4*b**2*(_x0*_x0 + _y0*_y0))*(_xd0*_xd0 + _yd0*_yd0) - self.H(_x0, _y0, alpha)
            rowsC.append(C0)
        Z = np.vstack(rowsZ); X = np.vstack(rowsX); T = np.vstack(rowsT)  # (2*Nφ, Nu)
        U = np.vstack(rowsU); V = np.vstack(rowsV)
        C = np.array(rowsC, float) 
        return Z, X, T,U,V,C
    
    
    def lightcone_2d_crosswrap(self, u0=0.0, v0=0.0, x0=0.0, y0=0.0, y0_dot=0.0,
                                lam=0.0, phis=None, u_end=None, n_eval=600,split=False):
        # φ in [0,180) – ohne 180°, damit u-Parametrisierung nicht kollabiert
        rowsZp, rowsXp, rowsTp, rowsUp, rowsVp = [], [], [], [], []
        rowsZm, rowsXm, rowsTm, rowsUm, rowsVm = [], [], [], [], []
        if phis is None:
            phis = np.linspace(0.0, 180.0, 31, endpoint=False)
        else:
            phis = np.asarray(phis, float)
            phis = phis[(np.mod(phis,360.0) >= 0) & (np.mod(phis,360.0) < 180.0)]
            phis = np.unique(phis)

        if u_end is None:
            width = max(1.0, self.u_max - self.u_min)
            u_end = self.u_max + 3*width
        u_span = (u0, u_end)

        rowsZ, rowsX, rowsT = [], [], []
        rowsU, rowsV = [], []

        # +Äste aufsteigend
        for phi in phis:
            a  = self.alpha_for_phi(phi)
            xD = self.x0_dot_from_alpha(a, u0, x0, y0, y0_dot)
            g  = self.integrate_geodesic(u_span, u0, x0, +xD, y0, y0_dot, v0, lam, n_eval=n_eval)
            rowsZp.append(g["z"]); rowsXp.append(g["x"]); rowsTp.append(g["t"])
            rowsUp.append(g["u"]); rowsVp.append(g["v"])

        # −Äste absteigend
        for phi in phis[::-1]:
            a  = self.alpha_for_phi(phi)
            xD = self.x0_dot_from_alpha(a, u0, x0, y0, y0_dot)
            g  = self.integrate_geodesic(u_span, u0, x0, -xD, y0, y0_dot, v0, lam, n_eval=n_eval)
            rowsZm.append(g["z"]); rowsXm.append(g["x"]); rowsTm.append(g["t"])
            rowsUm.append(g["u"]); rowsVm.append(g["v"])

        Zp = np.vstack(rowsZp); Xp = np.vstack(rowsXp); Tp = np.vstack(rowsTp)
        Up = np.vstack(rowsUp); Vp = np.vstack(rowsVp)

        Zm = np.vstack(rowsZm); Xm = np.vstack(rowsXm); Tm = np.vstack(rowsTm)
        Um = np.vstack(rowsUm); Vm = np.vstack(rowsVm)

        if split:
            return (Zp, Xp, Tp, Up, Vp), (Zm, Xm, Tm, Um, Vm)

        # wie bisher: für Flächen “crosswrap”
        rowsZ = rowsZp + rowsZm[::-1]
        rowsX = rowsXp + rowsXm[::-1]
        rowsT = rowsTp + rowsTm[::-1]
        rowsU = rowsUp + rowsUm[::-1]
        rowsV = rowsVp + rowsVm[::-1]
        Z = np.vstack(rowsZ); X = np.vstack(rowsX); T = np.vstack(rowsT)
        U = np.vstack(rowsU); V = np.vstack(rowsV)
        return Z, X, T, U, V
    
    
    def clip_line3d(self, Z, X, T, *, xlim, zlim, tlim, pad=0.0):
        zmin, zmax = zlim[0]-pad, zlim[1]+pad
        xmin, xmax = xlim[0]-pad, xlim[1]+pad
        tmin, tmax = tlim[0]-pad, tlim[1]+pad
        keep = (Z>=zmin)&(Z<=zmax)&(X>=xmin)&(X<=xmax)&(T>=tmin)&(T<=tmax)
        
        Zc, Xc, Tc = Z.copy(), X.copy(), T.copy()
        Zc[~keep] = np.nan; Xc[~keep] = np.nan; Tc[~keep] = np.nan
        return Zc, Xc, Tc
    
    def _clip_polygons_against_box3d(self, poly, *, xlim, zlim, tlim):
        """
        Sutherland–Hodgman-Clip gegen 6 Achsen-parallele Ebenen.
        poly: Liste von (Z,X,T)-Punkten im Uhr-/GgsUhrzeigersinn.
        Rückgabe: Liste von Punkten des geclippten Polygons (oder []).
        """

        def clip_against_plane(pts, axis, value, keep_greater_equal):
            if not pts: return []
            out = []
            def inside(P):
                return (P[axis] >= value) if keep_greater_equal else (P[axis] <= value)
            def intersect(P,Q):
                # P->Q Schnitt mit Ebene axis=value
                t = (value - P[axis]) / (Q[axis] - P[axis])
                return P + t*(Q - P)

            for i in range(len(pts)):
                P = np.asarray(pts[i], float)
                Q = np.asarray(pts[(i+1)%len(pts)], float)
                Pin, Qin = inside(P), inside(Q)
                if Pin and Qin:
                    out.append(Q)
                elif Pin and not Qin:
                    out.append(intersect(P,Q))
                elif (not Pin) and Qin:
                    out.append(intersect(P,Q)); out.append(Q)
                # else: beide draußen → nichts
            return out

        ZMIN,ZMAX = zlim; XMIN,XMAX = xlim; TMIN,TMAX = tlim
        P = [np.asarray(p,float) for p in poly]
        # Reihenfolge: z>=ZMIN, z<=ZMAX, x>=XMIN, x<=XMAX, t>=TMIN, t<=TMAX
        P = clip_against_plane(P, axis=0, value=ZMIN, keep_greater_equal=True)
        P = clip_against_plane(P, axis=0, value=ZMAX, keep_greater_equal=False)
        P = clip_against_plane(P, axis=1, value=XMIN, keep_greater_equal=True)
        P = clip_against_plane(P, axis=1, value=XMAX, keep_greater_equal=False)
        P = clip_against_plane(P, axis=2, value=TMIN, keep_greater_equal=True)
        P = clip_against_plane(P, axis=2, value=TMAX, keep_greater_equal=False)
        return [tuple(p) for p in P]
    
    def _poly_area3d(self, poly):
        """3D-Polygonfläche (Triangulation ab erstem Punkt)."""
        import numpy as np
        A = np.asarray(poly, float)
        if len(A) < 3: 
            return 0.0
        area = 0.0
        p0 = A[0]
        for i in range(1, len(A)-1):
            v1 = A[i]   - p0
            v2 = A[i+1] - p0
            area += 0.5 * np.linalg.norm(np.cross(v1, v2))
        return float(area)
    
    
    
    def plot_lightcone_ribbons(self, ax, Z, X, T, step_phi=1, alpha=0.25, cmap="viridis",
                        clip_to_axes_box=True, edge_lw=0.0, edge_color=None):
        C = cm.get_cmap(cmap)
        n_rows, n_cols = Z.shape
        if clip_to_axes_box:
            xlim = ax.get_ylim(); zlim = ax.get_xlim(); tlim = ax.get_zlim()

        for i in range(0, n_rows - step_phi, step_phi):
            Zi, Xi, Ti = Z[i], X[i], T[i]
            Zj, Xj, Tj = Z[i+step_phi], X[i+step_phi], T[i+step_phi]

            verts_all = []
            for k in range(n_cols-1):
                quad = [
                    (Zi[k],  Xi[k],  Ti[k]),
                    (Zi[k+1],Xi[k+1],Ti[k+1]),
                    (Zj[k+1],Xj[k+1],Tj[k+1]),
                    (Zj[k],  Xj[k],  Tj[k]),
                ]
                if clip_to_axes_box:
                    quad = self._clip_polygons_against_box3d(quad, xlim=xlim, zlim=zlim, tlim=tlim)
                if len(quad) >= 3:
                    verts_all.append(quad)

            if not verts_all:
                continue
            color = C(i / max(1, n_rows-1))
            poly = Poly3DCollection(verts_all, facecolor=color, edgecolor=(edge_color or 'none'),
                                    linewidth=edge_lw, alpha=np.clip(alpha,0,1))
            ax.add_collection3d(poly)
    
    def _clip_poly_against_box3d(self, poly, *, xlim, zlim, tlim):
        # poly: Liste von (Z,X,T), Rückgabe: Liste von (Z,X,T) nach Clipping (oder [])
        import numpy as np

        def clip_plane(pts, axis, value, keep_ge):
            if not pts: return []
            out = []
            def inside(P):
                return (P[axis] >= value) if keep_ge else (P[axis] <= value)
            def intersect(P, Q):
                t = (value - P[axis]) / (Q[axis] - P[axis])
                return P + t*(Q - P)

            for i in range(len(pts)):
                P = np.asarray(pts[i], float); Q = np.asarray(pts[(i+1) % len(pts)], float)
                Pin, Qin = inside(P), inside(Q)
                if Pin and Qin:
                    out.append(Q)
                elif Pin and not Qin:
                    out.append(intersect(P, Q))
                elif not Pin and Qin:
                    out.append(intersect(P, Q)); out.append(Q)
            return out

        ZMIN,ZMAX = zlim; XMIN,XMAX = xlim; TMIN,TMAX = tlim
        P = [np.asarray(p, float) for p in poly]
        # Reihenfolge: z>=ZMIN, z<=ZMAX, x>=XMIN, x<=XMAX, t>=TMIN, t<=TMAX
        P = clip_plane(P, 0, ZMIN, True)
        P = clip_plane(P, 0, ZMAX, False)
        P = clip_plane(P, 1, XMIN, True)
        P = clip_plane(P, 1, XMAX, False)
        P = clip_plane(P, 2, TMIN, True)
        P = clip_plane(P, 2, TMAX, False)
        return [tuple(p) for p in P]
    
    def _poly_max_span(self, poly):
        """Maximale 3D-Spannweite (Box-Diagonale) eines Polygons in (Z,X,T)."""
        import numpy as np
        A = np.asarray(poly, float)
        dz = A[:,0].max() - A[:,0].min()
        dx = A[:,1].max() - A[:,1].min()
        dt = A[:,2].max() - A[:,2].min()
        return float(np.sqrt(dz*dz + dx*dx + dt*dt))
    
    
    
    def _surface_from_wireframe(self, ax, Z, X, T, *,
                                Crows=None,
                                wrap=False, step_phi=1,
                                cmap="viridis", color=None, alpha=0.9,
                                edge_lw=0.0, edge_color='none',
                                max_face_span=None, max_face_area=None,
                                norm=None,
                                zorder=3,
                                color_by="t",
                                norm_mode="linear",
                                gamma=0.8,
                                n_blend=1):
        import numpy as np, matplotlib as mpl
        from matplotlib.colors import Normalize, PowerNorm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        nr, nc = Z.shape
        xlim = ax.get_ylim(); zlim = ax.get_xlim(); tlim = ax.get_zlim()

        keep = ((X >= xlim[0]) & (X <= xlim[1]) &
                (Z >= zlim[0]) & (Z <= zlim[1]) &
                (T >= tlim[0]) & (T <= tlim[1]))

        # ---------- Norm global bestimmen ----------
        if norm is None:
            if color_by == "C":
                if Crows is None:
                    raise ValueError("color_by='C' requires Crows array of shape (n_rows,)")
                vals = np.asarray(Crows, float)
                vals = vals[np.isfinite(vals)]
                vmin, vmax = np.nanpercentile(vals, [20, 90])
                # norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
                norm = PowerNorm(gamma=0.60, vmin=float(vmin), vmax=float(vmax), clip=True)

            elif color_by in ("t", "u", "v"):
                if color_by == "t":
                    S = T
                elif color_by == "u":
                    S = (T - Z) / np.sqrt(2)
                else:  # "v"
                    S = (T + Z) / np.sqrt(2)

                vals = S[keep] if np.any(keep) else S.ravel()
                vals = vals[np.isfinite(vals)]
                vmin, vmax = np.nanpercentile(vals, [2, 98])
                if norm_mode == "power":
                    norm = PowerNorm(gamma=gamma, vmin=float(vmin), vmax=float(vmax), clip=True)
                else:
                    norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
            else:
                raise ValueError("color_by must be 't','u','v','C'")

        Cmap = mpl.colormaps.get_cmap(cmap)
        faces, cols = [], []

        nrows_to = nr if wrap else (nr - 1)
        for i in range(0, nrows_to, step_phi):
            i2 = (i + 1) % nr
            if (not wrap) and (i == nr - 1):
                continue

            Zi, Xi, Ti = Z[i],  X[i],  T[i]
            Zj, Xj, Tj = Z[i2], X[i2], T[i2]

            # für u/v einmal pro i vorbereiten (effizient + sauber)
            if color_by == "u":
                Ui = (Ti - Zi) / np.sqrt(2)
                Uj = (Tj - Zj) / np.sqrt(2)
            elif color_by == "v":
                Vi = (Ti + Zi) / np.sqrt(2)
                Vj = (Tj + Zj) / np.sqrt(2)

            for k in range(nc - 1):
                # ungeclipptes Quad (4 Ecken)
                            # --- Eckpunkte des (ungeclippten) Quads ---
                P00 = np.array([Zi[k],   Xi[k],   Ti[k]],   float)
                P01 = np.array([Zi[k+1], Xi[k+1], Ti[k+1]], float)
                P11 = np.array([Zj[k+1], Xj[k+1], Tj[k+1]], float)
                P10 = np.array([Zj[k],   Xj[k],   Tj[k]],   float)

                # --- Skalarwerte an den zwei Geodäten-Rändern (nur t/v smoothen) ---
                if color_by == "t":
                    Si0, Si1 = Ti[k],  Ti[k+1]
                    Sj0, Sj1 = Tj[k],  Tj[k+1]
                    do_blend = True
                elif color_by == "v":
                    # v = (T+Z)/sqrt(2)
                    Si0, Si1 = (Ti[k]  + Zi[k])  / np.sqrt(2), (Ti[k+1]  + Zi[k+1])  / np.sqrt(2)
                    Sj0, Sj1 = (Tj[k]  + Zj[k])  / np.sqrt(2), (Tj[k+1]  + Zj[k+1])  / np.sqrt(2)
                    do_blend = True
                elif color_by == "u":
                    # u lässt du wie bisher
                    Si0, Si1 = Ui[k], Ui[k+1]
                    Sj0, Sj1 = Uj[k], Uj[k+1]
                    do_blend = False
                elif color_by == "C":
                    Sm0 = 0.5 * (Crows[i] + Crows[i2])
                    do_blend = False
                else:
                    raise ValueError("color_by must be 't','u','v','C'")

                # --- Für u/C: altes Verhalten (ein Face) ---
                if (not do_blend) or (n_blend <= 1):
                    quad = [tuple(P00), tuple(P01), tuple(P11), tuple(P10)]

                    # Sm0 wie gehabt:
                    if color_by == "C":
                        pass  # Sm0 schon gesetzt
                    elif color_by == "u":
                        Sm0 = 0.25 * (Ui[k] + Ui[k+1] + Uj[k+1] + Uj[k])
                    else:
                        # fallback
                        Sm0 = 0.25 * (Si0 + Si1 + Sj1 + Sj0)

                    poly = self._clip_poly_against_box3d(quad, xlim=xlim, zlim=zlim, tlim=tlim)
                    if len(poly) < 3:
                        continue

                    if (max_face_span is not None) and (self._poly_max_span(poly) > max_face_span):
                        continue
                    if (max_face_area is not None) and (self._poly_area3d(poly) > max_face_area):
                        continue

                    faces.append(poly)
                    if color is None:
                        c = Cmap(norm(Sm0))
                        cols.append((c[0], c[1], c[2], float(np.clip(alpha, 0, 1))))
                    else:
                        cols.append((*mpl.colors.to_rgb(color), float(np.clip(alpha, 0, 1))))
                    continue

                # --- Für t/v: Blend zwischen zwei Geodäten durch Streifen ---
                # Mittelwerte entlang u-Kante (k..k+1) pro Rand-Geodäte
                Si_mid = 0.5 * (Si0 + Si1)
                Sj_mid = 0.5 * (Sj0 + Sj1)

                for s in range(int(n_blend)):
                    a0 = s / float(n_blend)
                    a1 = (s + 1) / float(n_blend)
                    am = 0.5 * (a0 + a1)

                    # Geometrie linear zwischen Geodäten
                    Q00 = (1 - a0) * P00 + a0 * P10
                    Q01 = (1 - a0) * P01 + a0 * P11
                    Q11 = (1 - a1) * P01 + a1 * P11
                    Q10 = (1 - a1) * P00 + a1 * P10

                    quad_s = [tuple(Q00), tuple(Q01), tuple(Q11), tuple(Q10)]

                    poly = self._clip_poly_against_box3d(quad_s, xlim=xlim, zlim=zlim, tlim=tlim)
                    if len(poly) < 3:
                        continue

                    if (max_face_span is not None) and (self._poly_max_span(poly) > max_face_span):
                        continue
                    if (max_face_area is not None) and (self._poly_area3d(poly) > max_face_area):
                        continue

                    # Skalarwert: linearer Mix zwischen den beiden Rand-Geodäten
                    Sm = (1 - am) * Si_mid + am * Sj_mid

                    faces.append(poly)
                    if color is None:
                        c = Cmap(norm(Sm))
                        cols.append((c[0], c[1], c[2], float(np.clip(alpha, 0, 1))))
                    else:
                        cols.append((*mpl.colors.to_rgb(color), float(np.clip(alpha, 0, 1))))

        coll = Poly3DCollection(faces, facecolors=cols,
                                edgecolor=edge_color, linewidth=edge_lw)
        # coll.set_zsort('average')
        coll.set_zsort('max')
        coll.set_alpha(None)
        coll.set_zorder(zorder)
        ax.add_collection3d(coll)
        return coll, norm
    
    def plot_lightcone_surface(self, ax,
                        u0=0.0, v0=0.0, x0=0.0, y0=0.0, y0_dot=0.0,
                        lam=0.0, phis=tuple(range(0,181,6)),
                        u_end=None, n_eval=600,
                        mode="surface",              # "surface" | "wireframe" | "ribbons"
                        color=None, cmap="viridis",
                        alpha=0.6, rstride=1, cstride=1, linewidth=1.6,
                        show_geodesics=True, geodesic_color="k", geodesic_lw=0.6, geodesic_alpha=0.5,
                        geodesic_color_by="fixed",
                        clip=True, pad=0.0,
                        show_crosslines=True,
                        cross_stride_u=12,
                        cross_color=None, cross_lw=1.6, cross_alpha=0.4,
                        add_v_lines=False,
                        v_lines_u=None,            
                        v_line_x=0.0, v_line_y=0.0,
                        v_line_color="k", v_line_lw=0.7, v_line_alpha=0.9,
                        v_line_n=400,
                        max_face_span=None,   
                        max_face_area=None,    
                        color_by="t",          
                        norm_mode="linear",    
                        gamma=0.8,
                        surface_color_by=None,  # None -> benutze altes color_by
                        surface_cmap=None,      # None -> benutze cmap
                        show_surface_colorbar=False,
                        surface_cbar_label=None,
                        edge_lw=0.2,
                        edge_color="none",    
                        geodesic_stride=1,          # 1 = alle, 2 = jede 2., 3 = jede 3., ...
                        geodesic_phase=0,           # Offset: 0..stride-1
                        geodesic_keep_indices=None, # Liste/Set von Indizes, die IMMER gezeigt werden          
                        ):
        # 1) Mesh für beide Äste
        Z, X, T,U,V,C= self.lightcone_mesh_crosswrap(
                        u0=u0, v0=v0, x0=x0, y0=y0, y0_dot=y0_dot,
                        lam=lam, phis=phis, u_end=u_end, n_eval=n_eval
                    )

        
        if surface_color_by is None:
            surface_color_by = color_by  # Rückwärtskompatibel

        if surface_cmap is None:
            surface_cmap = cmap
        artists = []
        vals = T[np.isfinite(T)]
        vmin, vmax = np.percentile(vals, [20, 55])
        norm = PowerNorm(gamma=0.8, vmin=float(vmin), vmax=float(vmax), clip=True)
        # tmin, tmax = ax.get_zlim()
        # norm = mpl.colors.Normalize(vmin=float(tmin), vmax=float(tmax), clip=True)
        # 3) Rendern nach Modus (getrennt je Ast)
        if mode == "surface":
            coll, surf_norm = self._surface_from_wireframe(
                ax, Z, X, T,
                Crows=C,
                wrap=False,
                step_phi=1,
                cmap=surface_cmap, color=color, alpha=alpha,
                edge_lw=edge_lw, edge_color=edge_color,
                max_face_span=max_face_span, max_face_area=max_face_area,
                color_by=surface_color_by,
                norm_mode=norm_mode,
                gamma=gamma,
                n_blend=(10 if surface_color_by in ("t","v") else 1),
            )
            artists.append(coll)
        elif mode == "wireframe":
            # Limits nur falls du clippen willst
            if clip:
                xlim = ax.get_ylim(); zlim = ax.get_xlim(); tlim = ax.get_zlim()

            # Längslinien: jede Zeile des Rings
            for i in range(Z.shape[0]):
                Zi, Xi, Ti = Z[i], X[i], T[i]
                if clip:
                    Zi, Xi, Ti = self.clip_line3d(Zi, Xi, Ti, xlim=xlim, zlim=zlim, tlim=tlim, pad=pad)
                ln, = ax.plot(Zi, Xi, Ti, color=(color or "k"), lw=linewidth, alpha=float(np.clip(alpha, 0.0, 1.0)))
                artists.append(ln)

            # Querlinien (fixes u-Sample → Spalten)
            if show_crosslines:
                n_rows, n_cols = Z.shape
                step = max(1, int(cross_stride_u))
                for j in range(0, n_cols, step):
                    Zc, Xc, Tc = Z[:, j], X[:, j], T[:, j]
                    if clip:
                        Zc, Xc, Tc = self.clip_line3d(Zc, Xc, Tc, xlim=xlim, zlim=zlim, tlim=tlim, pad=pad)
                    ln2, = ax.plot(Zc, Xc, Tc,
                                color=(cross_color or color or "k"),
                                lw=cross_lw, alpha=float(np.clip(cross_alpha, 0.0, 1.0)))
                    artists.append(ln2)
                
                
                
        elif mode == "ribbons":
            self.plot_lightcone_ribbons(ax, Z, X, T, step_phi=1, alpha=alpha, cmap=cmap)
        else:
            raise ValueError("mode must be 'surface', 'wireframe', or 'ribbons'")
        if show_surface_colorbar:
            sm = mpl.cm.ScalarMappable(norm=surf_norm, cmap=mpl.colormaps.get_cmap(surface_cmap))
            sm.set_array([])

            if surface_cbar_label is not None:
                lab = surface_cbar_label
            else:
                lab = {"t": r"$t$", "u": r"$u$", "v": r"$v$", "C": r"$C$"}.get(surface_color_by, "")

            ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label=lab)
        if add_v_lines:
            # Limits müssen schon stehen, weil plot_v_line daraus v_start/v_end ableitet
            if v_lines_u is None:
                u_lo = u0
                u_hi = u_end if u_end is not None else (self.u_max + 3*max(1.0, self.u_max - self.u_min))
                v_lines_u = [0.5*(u_lo + u_hi)]
            for uu in v_lines_u:
                if uu == 0:
                    self.plot_v_line(ax, v_start=0,u=uu, x=v_line_x, y=v_line_y,
                                n=v_line_n, color=v_line_color,
                                lw=v_line_lw, alpha=v_line_alpha,
                                clip=True, pad=0.0)
                else:
                    self.plot_v_line(ax, u=uu, x=v_line_x, y=v_line_y,
                                    n=v_line_n, color=v_line_color,
                                    lw=v_line_lw, alpha=v_line_alpha,
                                    clip=True, pad=0.0)


        # 4) Geodäten-Overlay
        if show_geodesics:

            if clip:
                xlim = ax.get_ylim()
                zlim = ax.get_xlim()
                tlim = ax.get_zlim()

            # falls nach C gefärbt wird: Norm+Colormap
            if geodesic_color_by == "C":
                C_arr = np.asarray(C, float)
                Cmin, Cmax = np.nanpercentile(C_arr, [2, 98])
                normC = mpl.colors.Normalize(vmin=float(Cmin), vmax=float(Cmax), clip=True)
                cmapC = mpl.colormaps.get_cmap(cmap)

            N = Z.shape[0]
            stride = max(1, int(geodesic_stride))
            phase  = int(geodesic_phase) % stride

            # --- 1) Basis-Auswahl durch stride/phase ---
            selected = set()
            if stride == 1:
                selected = set(range(N))
            else:
                selected = set(i for i in range(N) if ((i - phase) % stride == 0))

            # --- 2) "keep indices" immer drin ---
            if geodesic_keep_indices is not None:
                selected |= set(int(k) for k in geodesic_keep_indices)

            # --- 3) Spiegelpartner hinzufügen: i -> N-1-i ---
            # (damit immer beide Seiten gezeigt werden)
            selected_with_mirror = set(selected)
            for i in list(selected):
                j = (N - 1 - i)
                selected_with_mirror.add(j)
            selected = selected_with_mirror


            N = Z.shape[0]
            if N % 2 != 0:
                raise ValueError("Expected 2*Nphi rows in Z (crosswrap).")

            Nphi = N // 2

            stride = int(geodesic_stride) if geodesic_stride is not None else 1
            phase  = int(geodesic_phase)  if geodesic_phase  is not None else 0
            stride = max(1, stride)
            phase %= stride

            sel_rows = set()

            # gleichmäßig über den +Ast (0..Nphi-1)
            for k in range(phase, Nphi, stride):
                sel_rows.add(k)                  # +Ast
                sel_rows.add(2*Nphi - 1 - k)     # Spiegel im −Ast

            # "immer zeigen" (Row-Indizes) + optional Spiegel
            if geodesic_keep_indices is not None:
                for idx in geodesic_keep_indices:
                    idx = int(idx)
                    if 0 <= idx < N:
                        sel_rows.add(idx)
                        sel_rows.add(2*Nphi - 1 - idx)

            sel_rows = sorted(sel_rows)
            # --- 4) Plotten nur für selected ---
            for i in sel_rows:
                Zi, Xi, Ti = Z[i], X[i], T[i]
                if clip:
                    Zi, Xi, Ti = self.clip_line3d(Zi, Xi, Ti, xlim=xlim, zlim=zlim, tlim=tlim, pad=pad)

                if geodesic_color_by == "C":
                    col = cmapC(normC(C_arr[i]))
                else:
                    col = geodesic_color

                ln, = ax.plot(Zi, Xi, Ti, color=col, lw=geodesic_lw, alpha=geodesic_alpha)
                ln.set_zorder(4)
                try:
                    ln.set_depthshade(True)
                except Exception:
                    pass
    # ----- 3D-Szene + Sandwich-Flächen zeichnen
    def setup_scene(self, L=12, figsize=(12,10), view_elev=6, view_azim=-75):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$x$"); ax.set_zlabel(r"$t$")
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.tick_params(axis='x', labelsize=BASE_FONTSIZE)
        ax.tick_params(axis='y', labelsize=BASE_FONTSIZE)
        ax.tick_params(axis='z', labelsize=BASE_FONTSIZE)
        return fig, ax
    
    



if __name__ == "__main__":
    
    b=0.5
    alpha = 1/9
    u_e = 10*np.pi/np.sqrt(alpha) 
    u_min = 0
    u_max = u_e
    #cauchy surface
    epsilon = 0.8
    eta     = 2.5
    
    gp = SandwichParabolWavePlotter(
        Hx=Hx,
        Hy=Hy,
        H=H,
        u_min=u_min,
        u_max=u_max
    )
    
    FIGSIZE_2D = (6, 8)
    # FIGSIZE_3D = (8, 6) 
    
    
    fig, ax = gp.setup_scene(L=12, view_elev=7, view_azim=-150)
    box_radius = 52
    ax.set_xlim(-box_radius, box_radius)   # für phys. z
    ax.set_ylim(-box_radius, box_radius)   # für phys. x
    ax.set_zlim(-box_radius, box_radius)   # für phys. t
    ax.set_proj_type('ortho')
    box_diag = np.sqrt((2*box_radius)**2 + (2*box_radius)**2 + (2*box_radius)**2)
    
    
  # Fläche (hinten)
    add_cauchy_surface_3d_slice_y0_clipped(
        ax, gp,
        alpha=alpha, b=b, eta=eta, epsilon=epsilon,
        u0=0.0, v0=0.0,
        Nu=180, Nx=180,
        color="0.55", alpha_face=0.3,
        max_face_span=0.8*box_diag,
        max_face_area=0.3*box_diag**2,
    )

    # Grid (vorne)
    # add_cauchy_gridlines_3d_y0(
    #     ax, gp,
    #     alpha=alpha, b=b, eta=eta, epsilon=epsilon,
    #     u0=0.0, v0=0.0,
    #     u_span=(gp.u_min - 20.0, gp.u_max + 40.0),
    #     x_span=ax.get_ylim(),
    #     Nu_lines=20, Nx_lines=0,
    #     n_samples=260,
    #     color="0.10", lw=0.6, a=0.6
    # )
    
    gp.plot_lightcone_surface(
        ax,
        u0=0.0, v0=0.0, x0=0.0, y0=0.0, y0_dot=0.0,
        lam=0.0, phis=tuple(range(0,180,10)),
        u_end=gp.u_max+40, n_eval=800,
        mode="surface",
        cmap="plasma_r", alpha=0.99,
        show_geodesics=True,
        geodesic_color="0.2",
        geodesic_color_by="fixed", #"C" um nach C zu färeben oder "fixed" für farbe in  geodesic_color
        linewidth=5.2,
        geodesic_lw=1.0,
        add_v_lines=True, v_lines_u=[0],
        max_face_span=0.8*box_diag,      
        max_face_area=0.3*box_diag**2,
        color_by="C",
        show_surface_colorbar=True,  
        edge_lw=0,
        edge_color="k",   
        geodesic_alpha=0.7,
        geodesic_stride=6,
        geodesic_phase=0,              # optional: mal 0/1/2 probieren
        geodesic_keep_indices=[0],   # immer zeigen
    )    
    
    # gp.plot_lightcone_surface(
    #     ax,
    #     u0=0.0, v0=0.0, x0=0.0, y0=0.0, y0_dot=0.0,
    #     lam=0.0, phis=tuple(range(0,180,5)),
    #     u_end=gp.u_max+40, n_eval=800,
    #     mode="wireframe",
    #     cmap="viridis", alpha=1,
    #     show_geodesics=True,
    #     add_v_lines=False, v_lines_u=[0],
    #     max_face_span=80.0,      
    #     max_face_area=80.0,
    #     geodesic_lw=1.6,
    #     geodesic_alpha=1.0    
    # )    
    
    v_0=20.0
    (pZ, pX, pT, pU, pV), (mZ, mX, mT, mU, mV)= gp.lightcone_2d_crosswrap(
    u0=0.0, v0=v_0,
    x0=0.0, y0=0.0, y0_dot=0.0,
    lam=0.0,
    phis=[0,15,30,45,60,75,90,105,120,135,150,165,179],
    u_end=gp.u_max+40,
    n_eval=400,split=True
    )

    Z, X, T, U, V = mZ, mX, mT, mU, mV
    
    
    
    
    
    #geodäten overlay in 3d
#     Z_top, X_top, T_top, U_top, V_top, C_top = gp.lightcone_mesh_crosswrap(
#     u0=0.0, v0=0.0, x0=0.0, y0=0.0, y0_dot=0.0,
#     lam=0.0, phis=tuple(range(0,180,6)),
#     u_end=gp.u_max+40, n_eval=800
# )

#     for i in range(Z.shape[0]):
#         Zi, Xi, Ti = Z_top[i], X_top[i], T_top[i]

#         # optional: clippen wie vorher
#         Zi, Xi, Ti = gp.clip_line3d(Zi, Xi, Ti,
#                                     xlim=ax.get_ylim(),
#                                     zlim=ax.get_xlim(),
#                                     tlim=ax.get_zlim(),
#                                     pad=0.0)

#         # ln, = ax.plot(Zi, Xi, Ti, color="k", lw=1.2, alpha=1.0, zorder=20)
#         add_depth_cued_geodesic(
#                         ax, Zi, Xi, Ti,
#                         color="k",
#                         lw_front=1.3, lw_back=0.6,
#                         alpha_front=1.0, alpha_back=0.9,
#                         depth_percentiles=(5, 95),
#                         gamma=0.9,
#                         sort_back_to_front=True
#                     )
    
    # fig.savefig("/Users/davidjohann/Documents/Uni/Thesis/masterproject/plots/plot.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.tight_layout()
    plt.show()
    
    #2D-PLOT
    # 2D-PLOT


fig2, ax2 = plt.subplots(figsize=FIGSIZE_2D, constrained_layout=False)
AXPOS = (0.12, 0.10, 0.78, 0.86)
ax2.set_position(AXPOS)
box2d_radius=40
ax2.set_xlim(-1.5*box2d_radius, 1.5*box2d_radius)
ax2.set_ylim(0, 2.2*box2d_radius)

# >>> WICHTIG: Box-Grenzen jetzt aus dem 2D-Plot nehmen (für z,t)
zlim2 = ax2.get_xlim()
tlim2 = ax2.get_ylim()


# x-Grenzen: entweder aus 3D (physikalische Box) oder explizit setzen
xlim2 = ax.get_ylim()      # (-15,15) aus 3D
# oder: xlim2 = (-15, 15)




    
    # ---- Cauchy-Hypersurface v_Sigma(u) ----

u_vals = np.linspace(u_min-30, gp.u_max+40, 800)

v_sigma = (
    v_0
    - epsilon*(u_vals - u_min)
    - eta*(1/24)*(alpha**2 / b**2)*(u_vals - u_min)**3
)

# nach (t,z) transformieren
t_sigma, z_sigma = gp.brinkmann_to_minkowski(u_vals, v_sigma)
segs_sigma = segments_inside_box_zt(
    z_sigma, t_sigma,
    zlim=zlim2,
    tlim=tlim2
)
# for zseg, tseg in segs_sigma:
#     ax2.plot(
#         zseg, tseg,
#         color="0.5",      # schönes neutrales Grau
#         lw=1.5,
#         alpha=0.8,
#         zorder=2,
#         ls=(0, (3, 2))
#     )


max_order = 3

# colors = plt.cm.tab10(np.linspace(0, 1, max_order))  # 1..max_order
colors = ["black", "crimson", "royalblue"]
# optional: pro Ordnung sammeln (für 1x scatter je Ordnung -> schöner/performanter)
pts_by_order = [[] for _ in range(max_order)]  # Liste von [(z,t),...]

side = "lt"   # x<=0 (nur eine Hälfte), oder "gt"
xfade = 0.5*(abs(xlim2[0]) + abs(xlim2[1]))   # z.B. aus deiner x-Box
# cmap = "PRGn"
# cmap = LinearSegmentedColormap.from_list(
#     "blue_gray_red",
#     ["#2b6cb0",  # blau (x<0)
#      "#404040",  # grau bei x=0 (nicht weiß!)
#      "#c53030"]  # rot (x>0)
# )
# cmap = "turbo"      # sehr kontrastreich
# cmap = "plasma"

# --- deine Skala ---
# --- Breite der grauen Null-Zone in der Colormap (relativ zu [0,1]) ---
mid = 0.5
w = 0.03   # <<<<<< 3% graue Zone (probier 0.01 ... 0.06)

cmap = LinearSegmentedColormap.from_list(
    "div_strong",
    [
        (0.00, "#41b6c4"),   # sehr dunkelblau
        (0.25, "#225ea8"),   # blau
        (0.45, "#183585"),   # cyan (mehr Verlauf auf x<0)
        (mid-w, "#081d58"),
        (mid,   "#505050"),  # schmal grau bei 0
        (mid+w, "#7f0000"),  # gelblich direkt nach 0
        (0.55, "#cc4c02"),   # orange
        (0.75, "#fe9929"),   # dunkelorange/rot
        (1.00, "#fec44f"),   # dunkelrot
    ]
)

#for colorcode
use_abs = False
scale = 0.2  # kleiner => stärkerer Farbkontrast
scale_max = 1
if use_abs:
    norm = Normalize(vmin=0.0, vmax=scale*xfade, clip=True)
    cbar_label = r"$|x|$"
else:
    # norm = Normalize(vmin=-scale*xfade, vmax=scale_max*scale*xfade+6, clip=True)
    
    vmax = scale * xfade
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cbar_label = r"$x$"
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])








all_svals = []  # optional, um globale vmin/vmax zu bestimmen
# (wenn du global willst: erst sammeln, dann plotten)
for i in range(Z.shape[0]):
    segs = segments_inside_box_zt_clip_3arrays(Z[i], T[i], -X[i], zlim=zlim2, tlim=tlim2)
    for zseg, tseg, xseg in segs:
        # lokal gefärbt + gefadet
        add_colored_geodesic_zt(
            ax2, zseg, tseg, xseg=xseg,
            cmap=cmap,
            alpha_front=1, alpha_back=1,
            xfade=xfade,
            use_abs=use_abs,   # Farbe nach |x| (oder False für Vorzeichen)
            lw=1.2, zorder=1,
            norm=norm
        )

    u_zeros = all_zero_crossings(U[i], X[i], u0=0.0, eps=1e-12, max_n=max_order)
    
    for n, u_star in enumerate(u_zeros, start=1):
        v_star = np.interp(u_star, U[i], V[i])
        t_star, z_star = gp.brinkmann_to_minkowski(u_star, v_star)

        # nur wenn innerhalb der 2D-Box:
        zmin2, zmax2 = ax2.get_xlim()
        tmin2, tmax2 = ax2.get_ylim()
        if (zmin2 <= z_star <= zmax2) and (tmin2 <= t_star <= tmax2):
            pts_by_order[n-1].append((z_star, t_star))

# nach der Schleife: plotte jede Ordnung in eigener Farbe
for n in range(1, max_order+1):
    
    pts = np.array(pts_by_order[n-1], float)
    if pts.size == 0:
        continue
    
    ax2.scatter(
        pts[:,0], pts[:,1],
        marker="x", s=33, linewidths=1.0,
        color=colors[n-1], alpha=0.8,
        label=fr"order n = {n}"
    )


#colorbar legende
cax = inset_axes(
    ax2,
    width="3.5%",     # Breite der Colorbar relativ zur Achse
    height="100%",    # exakt gleiche Höhe wie ax2
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0, 1, 1),  # rechts neben ax2
    bbox_transform=ax2.transAxes,
    borderpad=0
)



cbar = fig2.colorbar(sm, cax=cax)
cbar.set_label(cbar_label, fontsize=BASE_FONTSIZE)
cbar.ax.tick_params(labelsize=BASE_FONTSIZE)
ticks = np.linspace(norm.vmin, norm.vmax, 5)
cbar.set_ticks(ticks)

# Labels vorbereiten
labels = []
for val in ticks:
    if np.isclose(val, 0.0):
        labels.append("0")
    else:
        labels.append("")   # leer lassen

cbar.set_ticklabels(labels)
cbar.set_label(cbar_label)
    
    
uv_scale=0.19*box2d_radius
offset = 0.09 * box2d_radius
ax2.quiver(0, tmin2, +uv_scale, +uv_scale, angles='xy', scale_units='xy', scale=1,
                    color='black',width=0.004)
ax2.quiver(0, tmin2, -uv_scale, +uv_scale, angles='xy', scale_units='xy', scale=1,
        color='black',width=0.004)
ax2.text(0.7*uv_scale, tmin2+0.7*uv_scale-offset+0.02, r'$\vec{v}$', color='black', fontsize=BASE_FONTSIZE)
ax2.text(-0.7*uv_scale-offset, tmin2+0.7*uv_scale-offset, r'$\vec{u}$', color='black', fontsize=BASE_FONTSIZE)


# Startpunkt in (t,z)

t0, z0 = gp.brinkmann_to_minkowski(0, v_0)
ax2.plot(
        [z0], [t0],
        marker="o",
        markersize=4,
        color="black",
        zorder=10
    )

zmin, zmax = ax2.get_xlim()
zz = np.linspace(zmin, zmax, 400)

# ax2.plot(zz,  zz, ls="--", lw=1.0, color="gray", alpha=0.6)  
# ax2.plot(zz, -zz, ls="--", lw=1.0, color="gray", alpha=0.6)  
ax2.legend(loc="lower right", fontsize=0.8*BASE_FONTSIZE)
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel(r"$z$", fontsize=BASE_FONTSIZE)
ax2.set_ylabel(r"$t$", fontsize=BASE_FONTSIZE)
ax2.set_xticks([]); ax2.set_yticks([])
# ax2.grid(True, alpha=0.2)



# ------------------------------------------------------------
# Zweiter, separater 2D-Plot: anderer Ausschnitt (t nach unten),
# Geodäten einfarbig + Cauchy-Fläche sichtbar
# ------------------------------------------------------------
offset=30

fig_low, ax_low = plot_zt_cut_monochrome(
    gp,
    Z=mZ, T=mT,              # oder Z=Z, T=T je nachdem welchen Ast du willst
    v0=v_0,
    u_min=u_min,
    alpha_wave=alpha,
    b=b,
    epsilon=epsilon,
    eta=eta,
    zlim=(-1.5*box2d_radius-offset, 1.5*box2d_radius-offset),
    tlim=(-offset, 2.2*box2d_radius-offset),   # << tiefer nach unten
    geo_color="darkcyan",
    geo_lw=1.2,
    geo_alpha=0.95,
    show_cauchy=True,
    segments_inside_box_zt=segments_inside_box_zt,
    figsize=FIGSIZE_2D,
    highlight_indices=[4, 7],                 # << zwei!
    highlight_colors=["crimson", "darkorange"] # << zwei Farben!
)

ax_low.set_position(AXPOS)
u0 = 0.0
tE, zE = gp.brinkmann_to_minkowski(u0, v_0)

# optional: nur plotten, wenn im Sichtfenster
zmin, zmax = ax_low.get_xlim()
tmin, tmax = ax_low.get_ylim()
if (zmin <= zE <= zmax) and (tmin <= tE <= tmax):
    ax_low.plot([zE], [tE],
                marker="o", markersize=5,
                color="black", zorder=10)


# --- Kreuze bei den "Maxima" (halber u-Weg bis zur ersten x=0-Nullstelle) ---
u0 = 0.0  # Emission (bei dir 0)
zmin, zmax = ax_low.get_xlim()
tmin, tmax = ax_low.get_ylim()

pts_max = []

for i in range(mZ.shape[0]):   # oder Z.shape[0], je nachdem wie du es unten nennst
    u_zeros = all_zero_crossings(mU[i], mX[i], u0=u0, eps=1e-12, max_n=1)  # nur erste Nullstelle
    if len(u_zeros) == 0:
        continue

    u1 = u_zeros[0]
    u_max = 0.5 * (u0 + u1)  # bei u0=0 -> u1/2

    v_max = np.interp(u_max, mU[i], mV[i])
    t_max, z_max = gp.brinkmann_to_minkowski(u_max, v_max)

    if (zmin <= z_max <= zmax) and (tmin <= t_max <= tmax):
        pts_max.append((z_max, t_max))

if pts_max:
    pts_max = np.array(pts_max, float)
    ax_low.scatter(
        pts_max[:,0], pts_max[:,1],
        marker="x", s=40, linewidths=1.2,
        color="k", alpha=0.9, zorder=10,
        label=r"$v(u_{\text{turn}})$"
    )
    # ax_low.legend(loc="lower right")
    from matplotlib.lines import Line2D

    # vorhandene Legendeneinträge (z.B. von "v(u_turn)" Scatter)
    handles, labels = ax_low.get_legend_handles_labels()

    # Dummy-Lines für die 2 Highlight-Geodäten
    h1 = Line2D([0], [0], color="crimson", lw=2)
    h2 = Line2D([0], [0], color="darkorange", lw=2)

    # falls du die Highlights auch als Einträge willst:
    handles = [h1, h2] + handles
    labels  = [r"$C_2$", r"$C_1$"] + labels

    ax_low.legend(
        handles, labels,
        loc="lower right",
        fontsize=0.8*BASE_FONTSIZE,
    )

plt.tight_layout()
plt.show()