from dataclasses import dataclass
import numpy as np
from gravilens.core.base import InitialData, GeodesicSolution, brinkmann_to_minkowski, BrinkmannVector

"""
Plane-wave background models in Brinkmann coordinates.

This module provides a simple diagonal plane-wave model together with
closed-form expressions for transverse geodesic motion and the associated
``v``-component in Brinkmann coordinates.

The model supports two equivalent parametrizations of the transverse profile:

- directly via the diagonal entries ``h1`` and ``h2``,
- or via the derived parameters ``h_plus`` and ``h_p``. 
With :math:`h1 = -(h_p-h_+)` and :math:`h2 = -(h_p+h_+)`

For a transverse position vector :math:`X = (x, y)^T`, the profile function is

.. math::

   H(x, y) = X^T h X,

with

.. math::

   h =
   \\begin{pmatrix}
   h_1 & 0 \\\\
   0 & h_2
   \\end{pmatrix}.

The transverse geodesic is written in terms of matrix-valued fundamental
solutions :math:`A(u,u_0)` and :math:`B(u,u_0)` as

.. math::

   X(u) = A(u,u_0) X_0 + B(u,u_0) \\dot{X}_0.

The corresponding longitudinal Brinkmann coordinate :math:`v(u)` is then
recovered from the normalization parameter :math:`\\lambda`.

Notes
-----
This implementation currently assumes a diagonal transverse profile matrix.
"""


@dataclass
class PlaneWaveModel:
    r"""
    Diagonal plane-wave spacetime model in Brinkmann coordinates.

    Parameters
    ----------
    h1, h2 : float or None, optional
        Diagonal entries of the transverse profile matrix. If both are given,
        they are used directly.
    h_plus, h_p : float or None, optional
        Alternative parametrization for superpositions of pure gravitational and electromagnetic plane waves.
    u0 : float, default=0.0
        Default initial Brinkmann coordinate :math:`u_0`.
    v0 : float, default=0.0
        Default initial Brinkmann coordinate :math:`v_0`.
    eps : float, default=1e-14
        Numerical threshold used to distinguish positive, zero, and negative
        profile eigenvalues.

    Notes
    -----
    If ``h1`` and ``h2`` are not provided, they are computed from
    ``h_plus`` and ``h_p`` via

    .. math::

       h_1 = h_+ - h_p,
       \\qquad
       h_2 = -h_+ - h_p.

    where :math:`h1 \geq 0` corresponds to gravity dominated waves and :math:`h1 < 0` to matter dominated waves.

    Examples
    --------
    Construct a model directly from ``h1`` and ``h2``:

    >>> model = PlaneWaveModel(h1=-1.0, h2=-1.0)
    >>> model.h1, model.h2
    (-1.0, -1.0)

    Construct the same kind of model from ``h_plus`` and ``h_p``:

    >>> model = PlaneWaveModel(h_plus=0.0, h_p=1.0)
    >>> model.h1
    -1.0
    >>> model.h2
    -1.0
    """
    h1: float | None = None
    h2: float | None = None

    h_plus: float | None = None
    h_p: float | None = None
    u0: float = 0.0
    v0: float = 0.0
    eps: float = 1e-14

    def __post_init__(self):
        if self.h1 is None or self.h2 is None:
            if self.h_plus is None or self.h_p is None:
                raise ValueError("Either (h1,h2) or (h_plus,h_p) must be given")

            self.h1 = self.h_plus - self.h_p
            self.h2 = -self.h_plus - self.h_p

    @property
    def h_mat(self) -> np.ndarray:
        return np.array([
            [self.h1, 0.0],
            [0.0, self.h2],
        ])

    def first_conjugate_u(self, u0=None) -> float | None:
        r"""
        Return the first finite conjugate value of ``u``.

        A finite conjugate point occurs when one of the oscillatory transverse
        directions reaches the first zero of the corresponding ``B`` component.

        Parameters
        ----------
        u0 : float or None, optional
            Initial value of :math:`u`. If omitted, ``self.u0`` is used.

        Returns
        -------
        float or None
            The first finite conjugate value of :math:`u`, or ``None`` if no
            oscillatory direction is present.

        Notes
        -----
        For each negative profile entry :math:`h < 0`, the first conjugate value is

        .. math::

           u = u_0 + \\frac{\\pi}{\\sqrt{|h|}}.

        Examples
        --------
        >>> model = PlaneWaveModel(h1=-1.0, h2=4.0, u0=0.0)
        >>> np.isclose(model.first_conjugate_u(), np.pi)
        np.True_

        >>> model = PlaneWaveModel(h1=1.0, h2=4.0)
        >>> model.first_conjugate_u() is None
        True
        """
        u0 = self.u0 if u0 is None else u0
        times = []

        for h in (self.h1, self.h2):
            if h < -self.eps:
                times.append(u0 + np.pi / np.sqrt(abs(h)))

        if not times:
            return None

        return min(times)


    def H(self, x: float, y: float) -> float:
        r"""Evaluate the Brinkmann profile :math:`H(x,y)`.

        Parameters
        ----------
        x, y : float
            Transverse coordinates.

        Returns
        -------
        float
            The quadratic form

            .. math::

               H(x,y) = X^T h X = h_1 x^2 + h_2 y^2.

        Examples
        --------
        >>> model = PlaneWaveModel(h1=2.0, h2=-1.0)
        >>> model.H(3.0, 4.0)
        2.0
        """
        X = np.array([x, y],dtype=float)
        return float(X @ self.h_mat @ X)

    def A(self, u, u0=None):
        r"""Return the fundamental matrix :math:`A(u,u_0)`
        defined by the transverse solution formula

        .. math::

           X(u) = A(u,u_0) X_0 + B(u,u_0) \dot X_0,

        with initial conditions

        .. math::

           A(u_0,u_0) = I,
           \qquad
           \partial_u A(u,u_0)|_{u=u_0} = 0.

        Parameters
        ----------
        u : float
            Evaluation point.
        u0 : float or None, optional
            Initial value :math:`u_0`. Defaults to ``self.u0``.

        Returns
        -------
        numpy.ndarray
            Diagonal :math:`2\times 2` matrix :math:`A(u,u_0)`.
        """
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self.eps
        if h1 > E:
            A11 = np.cosh(np.sqrt(h1)*(u-u0))
        elif h1 < -E:
            A11 = np.cos(np.sqrt(abs(h1))*(u-u0))
        else:
            A11 = 1.0

        if h2 > E:
            A22 =  np.cosh(np.sqrt(h2)*(u-u0))
        elif h2 < -E:
            A22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        else:
            A22 = 1.0
        return np.array([[A11, 0.0], [0.0, A22]])

    def B(self, u,u0=None):
        r"""Return the fundamental matrix :math:`B(u,u_0)` defined by

        .. math::

           X(u) = A(u,u_0) X_0 + B(u,u_0) \dot X_0,

        with initial conditions

        .. math::

           B(u_0,u_0) = 0,
           \qquad
           \partial_u B(u,u_0)|_{u=u_0} = I.

        For negative profile entries one obtains sine terms, for vanishing
        entries linear terms, and for positive entries hyperbolic sine terms.

        Parameters
        ----------
        u : float
            Evaluation point.
        u0 : float or None, optional
            Initial value :math:`u_0`. Defaults to ``self.u0``.

        Returns
        -------
        numpy.ndarray
            Diagonal :math:`2\times 2` matrix :math:`B(u,u_0)`.
        """
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self.eps
        if h1 > E:
            B11 = (1/np.sqrt(h1))*np.sinh(np.sqrt(h1)*(u-u0))
        elif h1 < -E:
            B11 = (1 / np.sqrt(abs(h1))) * np.sin(np.sqrt(abs(h1)) * (u - u0))
        else:
            B11 = (u - u0)

        if h2 > E:
            B22 = (1/np.sqrt(h2))*np.sinh(np.sqrt(h2)*(u-u0))
        elif h2 < -E:
            B22 = (1/np.sqrt(abs(h2)))*np.sin(np.sqrt(abs(h2))*(u-u0))
        else:
            B22 = (u - u0)
        return np.array([[B11, 0.0], [0.0, B22]])

    def A_dot(self, u,u0=None):
        r"""Return :math:`\partial_u A(u,u_0)`.

        Parameters
        ----------
        u : float
            Evaluation point.
        u0 : float or None, optional
            Initial value :math:`u_0`. Defaults to ``self.u0``.

        Returns
        -------
        numpy.ndarray
            Derivative of :meth:`A` with respect to :math:`u`.
        """
        u0 = self.u0 if u0 is None else u0
        s = u - u0
        h1, h2, E = self.h1, self.h2, self.eps

        if h1 > E:
            A11_dot = np.sqrt(h1) * np.sinh(np.sqrt(h1) * s)
        elif h1 < -E:
            A11_dot = -np.sqrt(-h1) * np.sin(np.sqrt(-h1) * s)
        else:
            A11_dot = 0.0

        if h2 > E:
            A22_dot = np.sqrt(h2) * np.sinh(np.sqrt(h2) * s)
        elif h2 < -E:
            A22_dot = -np.sqrt(-h2) * np.sin(np.sqrt(-h2) * s)
        else:
            A22_dot = 0.0

        return np.array([[A11_dot, 0.0], [0.0, A22_dot]])

    def B_dot(self, u,u0=None):
        r"""Return :math:`\partial_u B(u,u_0)`.

        Parameters
        ----------
        u : float
            Evaluation point.
        u0 : float or None, optional
            Initial value :math:`u_0`. Defaults to ``self.u0``.

        Returns
        -------
        numpy.ndarray
            Derivative of :meth:`B` with respect to :math:`u`.

        Examples
        --------
        >>> model = PlaneWaveModel(h1=-1.0, h2=-1.0)
        >>> np.allclose(model.B_dot(model.u0), np.eye(2))
        True
        """
        u0 = self.u0 if u0 is None else u0
        s = u - u0
        h1, h2, E = self.h1, self.h2, self.eps

        if h1 > E:
            B11_dot = np.cosh(np.sqrt(h1) * s)
        elif h1 < -E:
            B11_dot = np.cos(np.sqrt(-h1) * s)
        else:
            B11_dot = 1.0

        if h2 > E:
            B22_dot = np.cosh(np.sqrt(h2) * s)
        elif h2 < -E:
            B22_dot = np.cos(np.sqrt(-h2) * s)
        else:
            B22_dot = 1.0

        return np.array([[B11_dot, 0.0], [0.0, B22_dot]])

    def transverse_geodesic(self, u, X0, X0_dot,u0=None):
        r"""Evaluate the transverse geodesic :math:`X(u)`.

        Parameters
        ----------
        u : float
            Evaluation point.
        X0 : array-like, shape (2,)
            Initial transverse position :math:`X_0`.
        X0_dot : array-like, shape (2,)
            Initial transverse velocity :math:`\dot X_0`.
        u0 : float or None, optional
            Initial value :math:`u_0`. Defaults to ``self.u0``.

        Returns
        -------
        numpy.ndarray
            The vector :math:`X(u)`.
        """
        return self.A(u,u0) @ X0 + self.B(u,u0) @ X0_dot

    def transverse_geodesic_dot(self, u, X0, X0_dot, u0 = None):
        r"""Evaluate :math:`\dot X(u)` for the transverse geodesic.

        Parameters
        ----------
        u : float
            Evaluation point.
        X0 : array-like, shape (2,)
            Initial transverse position :math:`X_0`.
        X0_dot : array-like, shape (2,)
            Initial transverse velocity :math:`\dot X_0`.
        u0 : float or None, optional
            Initial value :math:`u_0`. Defaults to ``self.u0``.

        Returns
        -------
        numpy.ndarray
            The transverse derivative :math:`\dot X(u)`.
        """
        return self.A_dot(u,u0) @ X0 + self.B_dot(u,u0) @ X0_dot

    def v_geodesic(self, u, X, X_dot, X0, X0_dot, v0=0.0, lam=0.0, u0=None):
        r"""Evaluate the Brinkmann coordinate :math:`v(u)`.
        .. math::

            v(u1) = v(u0)+ \frac{1}{2} \int\limits_{u0}^{u1} |\dot X(u)|^2 + H(u,X) - \lambda du
        Parameters
        ----------
        u : float
            Evaluation point.
        X, X_dot : array-like, shape (2,)
            Transverse position and velocity at :math:`u`.
        X0, X0_dot : array-like, shape (2,)
            Initial transverse position and velocity at :math:`u_0`.
        v0 : float, default=0.0
            Initial value :math:`v(u_0)`.
        lam : float, default=0.0
            Normalization constant :math:`\lambda = g(\dot\gamma, \dot\gamma)` in
            the conventions used by this package.
        u0 : float or None, optional
            Initial value :math:`u_0`. Defaults to ``self.u0``.

        Returns
        -------
        float
            The longitudinal Brinkmann coordinate :math:`v(u)`.
        """
        u0 = self.u0 if u0 is None else u0
        return v0 + 0.5 * (-lam*(u-u0) + X_dot @ X - X0_dot @ X0)

    @staticmethod
    def alpha_for_phi(phi):
        r"""Convert azimuthal angle ``phi`` in the x,z plane
        to polar angle ``alpha`` between the v-component of a null geodesic and
        the affine u-axis.

        .. math::

           \alpha = \arctan\!\left(\frac{1}{\cos\phi}\right) - \frac{\pi}{4},

        with the limiting value :math:`\alpha=\pi/4` for :math:`\cos\phi = 0`.

        Parameters
        ----------
        phi : float
            Angle in radians.

        Returns
        -------
        float
            The corresponding angle in radians.
        """
        c = np.cos(phi)
        if np.isclose(c, 0.0):
            return np.pi / 4
        return np.arctan(1.0 / c) - np.pi / 4

    def x0_dot_from_alpha(self, alpha, x0, y0, y0_dot):
        r"""Gives ``x0_dot`` from v0_dot via ``alpha`` and the null constraint.

        .. math::

           \dot x_0^2 = 2\tan\alpha - H(x_0,y_0) - \dot y_0^2.

        Parameters
        ----------
        alpha : float
            Angle parameter in radians.
        x0, y0 : float
            Initial transverse position.
        y0_dot : float
            Initial :math:`y`-velocity.

        Returns
        -------
        float
            Non-negative value of :math:`\dot x_0`.
        """
        x0_dot_sq = 2 * np.tan(alpha) - self.H(x0, y0) - y0_dot ** 2
        if x0_dot_sq < 0:
            x0_dot_sq = 0.0
        return np.sqrt(x0_dot_sq)

    def lam_from_initial(self, X0, X0_dot, v0_dot):
        r"""Compute :math:`\lambda` from initial data.

        Parameters
        ----------
        X0 : array-like, shape (2,)
            Initial transverse position.
        X0_dot : array-like, shape (2,)
            Initial transverse velocity.
        v0_dot : float
            Initial derivative :math:`\dot v_0`.

        Returns
        -------
        float
            The normalization constant

            .. math::

               \lambda = -2\dot v_0 + H(X_0) + \dot X_0\cdot\dot X_0.
        """
        X0 = np.asarray(X0, float)
        X0_dot = np.asarray(X0_dot, float)
        H0 = self.H(X0[0], X0[1])
        return -2.0 * float(v0_dot) + H0 + float(X0_dot @ X0_dot)

    def v0_dot_for_lam(self, lam_target, X0, X0_dot):
        r"""Solve for :math:`\dot v_0` from a target value of :math:`\lambda`.

        Parameters
        ----------
        lam_target : float
            Desired normalization constant.
        X0 : array-like, shape (2,)
            Initial transverse position.
        X0_dot : array-like, shape (2,)
            Initial transverse velocity.

        Returns
        -------
        float
            The compatible initial value :math:`\dot v_0`.
        """
        X0 = np.asarray(X0, float)
        X0_dot = np.asarray(X0_dot, float)
        H0 = self.H(X0[0], X0[1])
        return 0.5 * (-float(lam_target) + H0 + float(X0_dot @ X0_dot))


    def solve_geodesic(self, initial: InitialData, u_grid: np.ndarray) -> GeodesicSolution:
        r"""Solve a geodesic on a prescribed :math:`u`-grid.

        Parameters
        ----------
        initial : InitialData
            Initial Brinkmann data at :math:`u=u_0`.
        u_grid : numpy.ndarray
            One-dimensional grid of :math:`u` values.

        Returns
        -------
        GeodesicSolution
            Object containing the Brinkmann coordinates ``(u, v, x, y)``, their
            Minkowski counterparts ``(t, z)``, and transverse derivatives.

        Examples
        --------
        >>> model = PlaneWaveModel(h1=0.0, h2=0.0)
        >>> initial = InitialData(u0=0.0, v0=0.0, x0=1.0, y0=0.0,
        ...                       x0_dot=2.0, y0_dot=0.0, lam=0.0)
        >>> sol = model.solve_geodesic(initial, np.array([0.0, 1.0]))
        >>> list(sol.x)
        [1.0, 3.0]

        Convert to pandas DataFrame

        >>> df = sol.to_frame()
        >>> list(df.columns)
        ['u', 'v', 'x', 'y', 't', 'z']

        >>> float(df["x"].iloc[1])
        3.0
        """
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

    def frequency_shift(
            self,
            lambda_e: float,
            lambda_o: float,
            r_o: BrinkmannVector,
            r_e: BrinkmannVector,
            Xdot_obs_o,
            Xdot_src_e,
    ):
        r"""Return the frequency-shift factor between emitter and observer.

        Parameters
        ----------
        lambda_e, lambda_o : float
            Normalization constants for emitter and observer worldlines.
            In the conventions used here, timelike geodesics have
            :math:`\lambda < 0`.

        r_o, r_e : BrinkmannVector
            Tangent vectors of the connecting null ray evaluated at the
            observation and emission events, respectively.

            Only the transverse components ``dx`` and ``dy`` enter the
            frequency-shift formula.

        Xdot_obs_o, Xdot_src_e : array-like, shape (2,)
            Transverse velocities of observer and source at observation
            and emission.

        Returns
        -------
        float
            Ratio

            .. math::

                \frac{\omega_o}{\omega_e}

            between observed and emitted frequency in the conventions used
            in this module.

        Notes
        -----
        The formula implemented here is

        .. math::

            \omega =
            \frac{-\lambda + |\dot X + r_\perp|^2}{\sqrt{|\lambda|}},

        evaluated at emission and observation.

        Examples
        --------
        >>> model = PlaneWaveModel(h1=0.0, h2=0.0)
        >>> r_o = BrinkmannVector(0, 0, 0.0, 0.0)
        >>> r_e = BrinkmannVector(0, 0, 0.0, 0.0)
        >>> model.frequency_shift(
        ...     lambda_e=-1.0,
        ...     lambda_o=-1.0,
        ...     r_o=r_o,
        ...     r_e=r_e,
        ...     Xdot_obs_o=[0.0, 0.0],
        ...     Xdot_src_e=[0.0, 0.0],
        ... )
        1.0
        """
        r_o_trans = np.array([r_o.dx, r_o.dy], dtype=float)
        r_e_trans = np.array([r_e.dx, r_e.dy], dtype=float)
        v_obs = Xdot_obs_o + r_o_trans
        v_src = Xdot_src_e + r_e_trans
        w_o = (-lambda_o+v_obs@v_obs)/np.sqrt(abs(lambda_o))
        w_e = (-lambda_e+v_src@v_src)/np.sqrt(abs(lambda_e))
        freq_shift = w_o/w_e
        return freq_shift

    def angle_lightray_dv(self, Xdot_o, r_o, lam_obs):
        r"""Return the observation angle between an incoming null ray
        and the wave propagation direction :math:`\partial_v`.

        Parameters.
        ----------
        Xdot_o : array-like, shape (2,)
            Transverse observer velocity.
        r_o : BrinkmannVector
            Tangent of the incoming null ray.
            Must provide components ``dx`` and ``dy``.
        lam_obs : float
            Observer normalization constant.

        Returns
        -------
        float
            Angle :math:`\psi` in degrees, computed from

            .. math::

               \cos\psi = 1 + \frac{2\lambda}{-\lambda + |-\sqrt{\lambda}\,\dot X_o + r_o|^2}.
        """
        r_o_trans = np.array(
            [r_o.dx, r_o.dy],
            dtype=float,
        )
        Xdot_o = np.asarray(Xdot_o, dtype=float)

        abs_vec = Xdot_o + r_o_trans

        cos_psi = 1.0 + (2.0 * lam_obs) / (-lam_obs + abs_vec @ abs_vec)
        psi = np.arccos(cos_psi)

        return np.degrees(psi)