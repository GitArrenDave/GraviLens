import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utilities (Brinkmann <-> Minkowski)
# -----------------------------
def brinkmann_to_minkowski(u, v):
    t = (v + u) / np.sqrt(2)
    z = (v - u) / np.sqrt(2)
    return t, z

def minkowski_to_brinkmann(t, z):
    v = (t + z) / np.sqrt(2)
    u = (t - z) / np.sqrt(2)
    return u, v
# -----------------------------
# Hauptklasse
# -----------------------------
class GeodesicPlaneWave:
    """
    Löst transversale Geodäten in einer (algebraischen) ebenen Gravitationswelle
    in Brinkmann-Koordinaten und plottet sie als Kurven in (z, x, t) mit u=const-Flächen.

    H(x, y) = h1 x^2 + h2 y^2, h1 = h_plus - w^2, h2 = -h_plus - w^2
    """
    _EPS = 1e-14

    def __init__(self, h_plus=0.0, h_p=0.0, u0=None, v0=None,lam =0):
        # self.w = float(w)
        self.h_plus = float(h_plus) 
        self.h_p = float(h_p)
        self.h1 = self.h_plus - self.h_p
        self.h2 = -self.h_plus - self.h_p
        self.lam = float(lam)
        if u0 is None or v0 is None:
            u0_def, v0_def = minkowski_to_brinkmann(0.0, 0.0)
            if u0 is None: u0 = u0_def
            if v0 is None: v0 = v0_def
        self.u0 = float(u0)
        self.v0 = float(v0)

    # -------- Feldfunktion --------
    def H(self, x, y):
        return self.h1*x**2 + self.h2*y**2

    # -------- Fundamentallösungen A, B und Ableitungen --------
    def A(self, u, u0=None):
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self._EPS
        if h1 > E:
            A11 = np.cosh(np.sqrt(h1)*(u-u0))
            A22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        elif abs(h1) <= E:
            A11 = 1.0; A22 = 1.0
        else:
            A11 = np.cos(np.sqrt(abs(h1))*(u-u0))
            A22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        return np.array([[A11, 0.0], [0.0, A22]])

    def B(self, u,u0):
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self._EPS
        if h1 > E:
            B11 = (1/np.sqrt(h1))*np.sinh(np.sqrt(h1)*(u-u0))
            B22 = (1/np.sqrt(abs(h2)))*np.sin(np.sqrt(abs(h2))*(u-u0))
        elif abs(h1) <= E:
            B11 = (u-u0); B22 = (u-u0)
        else:
            B11 = (1/np.sqrt(abs(h1)))*np.sin(np.sqrt(abs(h1))*(u-u0))
            B22 = (1/np.sqrt(abs(h2)))*np.sin(np.sqrt(abs(h2))*(u-u0))
        return np.array([[B11, 0.0], [0.0, B22]])

    def A_dot(self, u,u0):
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self._EPS
        if h1 > E:
            A11 = np.sqrt(h1)*np.sinh(np.sqrt(h1)*(u-u0))
            A22 = -np.sqrt(abs(h2))*np.sin(np.sqrt(abs(h2))*(u-u0))
        elif abs(h1) <= E:
            A11 = 0.0; A22 = 0.0
        else:
            A11 = -np.sqrt(abs(h1))*np.sin(np.sqrt(abs(h1))*(u-u0))
            A22 = -np.sqrt(abs(h2))*np.sin(np.sqrt(abs(h2))*(u-u0))
        return np.array([[A11, 0.0], [0.0, A22]])

    def B_dot(self, u,u0):
        u0 = self.u0 if u0 is None else u0
        h1, h2, E = self.h1, self.h2, self._EPS
        if h1 >= E:
            B11 = np.cosh(np.sqrt(h1)*(u-u0))
            B22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        elif abs(h1) <= E:
            B11 = 1.0; B22 = 1.0
        else:
            B11 = np.cos(np.sqrt(abs(h1))*(u-u0))
            B22 = np.cos(np.sqrt(abs(h2))*(u-u0))
        return np.array([[B11, 0.0], [0.0, B22]])
    
    @property
    def h_mat(self):
    # 2x2-Matrix für das quadratische H = h1 x^2 + h2 y^2
        return np.array([[self.h1, 0.0],
                        [0.0,     self.h2]], dtype=float)

    # -------- Geodäten (quer + v) --------
    def transverse_geodesic(self, u, X0, X0_dot,u0=None):
        return self.A(u,u0) @ X0 + self.B(u,u0) @ X0_dot

    def transverse_geodesic_dot(self, u, X0, X0_dot, u0 = None):
        return self.A_dot(u,u0) @ X0 + self.B_dot(u,u0) @ X0_dot

    def v_geodesic(self, u, X, X_dot, X0, X0_dot, v0=0.0, lam=0.0, u0=None):
        u0 = self.u0 if u0 is None else u0
        return v0 + 0.5 * (-lam*(u-u0) + X_dot @ X - X0_dot @ X0)

    # -------- Hilfsfunktionen für Startwerte --------
    @staticmethod
    def alpha_for_phi(phi_deg):
        """phi in Grad -> alpha in Radiant."""
        phi = np.deg2rad(phi_deg)
        c = np.cos(phi)
        if np.isclose(c, 0.0):
            return np.pi/4
        return np.arctan(1.0/c) - np.pi/4

    def x0_dot_from_alpha(self, alpha, x0, y0, y0_dot):
        x0_dot_sq = 2*np.tan(alpha) - self.H(x0, y0) - y0_dot**2
        if x0_dot_sq < 0:
            # Stabil: auf 0 clippen, wie in deinem Code
            x0_dot_sq = 0.0
        return np.sqrt(x0_dot_sq)

    def lam_from_initial(self, X0, X0_dot, v0_dot):
        X0 = np.asarray(X0, float); X0_dot = np.asarray(X0_dot, float)
        H0 = self.H(X0[0], X0[1])
        return -2.0*float(v0_dot) + H0 + float(X0_dot @ X0_dot)
    
    def v0_dot_for_lam(self,lam_target, X0, X0_dot):
        X0 = np.asarray(X0, float); X0_dot = np.asarray(X0_dot, float)
        H0 = self.H(X0[0], X0[1])
        return 0.5 * ( -float(lam_target) + H0 + float(X0_dot @ X0_dot) )

    # -------- gesamte Bahn auf u-Gitter berechnen --------
    def solve(self, u_grid, X0, X0_dot, v0=0.0, lam=None, u0=None, v0_dot=None):
        ueff = self.u0 if u0 is None else float(u0)
        if v0_dot is not None:
            lam = self.lam_from_initial(X0,X0_dot,v0_dot)
        elif lam is None:
            lam = self.lam
        u_grid = np.asarray(u_grid)
        X_list, Xd_list, v_list, vd_list= [], [], [], []
        for u in u_grid:
            X  = self.transverse_geodesic(u, X0, X0_dot,u0=ueff)
            Xd = self.transverse_geodesic_dot(u, X0, X0_dot, u0=ueff)
            v  = self.v_geodesic(u, X, Xd, X0, X0_dot, v0=v0, lam=lam,u0=ueff)
            vd = self.v0_dot_for_lam(lam,X,Xd)
            X_list.append(X); Xd_list.append(Xd); v_list.append(v); vd_list.append(vd)
        X  = np.vstack(X_list)
        Xd = np.vstack(Xd_list)
        v  = np.array(v_list)
        vd  = np.array(vd_list)
        t, z = brinkmann_to_minkowski(u_grid, v)

        return {
            "u": u_grid,
            "x": X[:, 0], "y": X[:, 1],
            "x_dot": Xd[:, 0], "y_dot": Xd[:, 1],
            "v": v,"v_dot":vd, "t": t, "z": z,"lam": lam,
        }
        
    def sigma_same_geodesic(self, sol, i0, i1):
        """
        Synge function zwischen zwei Punkten (Indices i0,i1) auf EINER bereits gelösten Geodäte.
        Nutzt die in sol gespeicherte lam (falls vorhanden), sonst rekonstruiert lam numerisch.
        """
        u = np.asarray(sol["u"])
        du = float(u[i1] - u[i0])

        if "lam" in sol:
            lam = float(sol["lam"])
        else:
            # numerisch aus Daten: g = 2 v' + H + |X'|^2  => lam = -g
            vprime = np.gradient(sol["v"], u)  # robust
            Hvals  = self.H(sol["x"], sol["y"])
            gnorm  = -2.0*vprime + Hvals + sol["x_dot"]**2 + sol["y_dot"]**2
            lam    = float(gnorm[i0])  # konstant; nimm einen Punkt

        return 0.5 * lam * du * du
    
    def sigma_grad_from_sol(self, p_e, sol, null_g, u_o, u_e):
        """
        ∇_p σ(p,q) am Startpunkt p=sol[i0] für q=sol[i1] (gleiche Geodäte).
        Rückgabe: dict mit 'u','v','x','y' Komponenten der Ableitung.
        """
        
        i_e_null = np.searchsorted(null_g["u"],u_e)
        i_o_null = np.searchsorted(null_g["u"],u_o)
        # u_o = sol["u"][i_o]
       
        # # v_o = sol["v"][i_o]
        # # v_o_dot = self.v0_dot_for_lam(lam,X_o,X_o_dot)
        
        # xd_o, yd_o = float(sol["x_dot"][i_o]), float(sol["y_dot"][i_o])
        # X_o_dot = np.asarray([xd_o,yd_o],float)
        # #source emmittion point
        # u_e= p_e["u"][i_e]
        # gamma_e = np.asarray([p_e["x"][i_e],p_e["y"][i_e]],float)
        # gamma_e_dot = np.asarray([p_e["x_dot"][i_e],p_e["y_dot"][i_e]],float)
        # # v_e = p_e["v"][i_e]
        # v_e_dot = self.v0_dot_for_lam(lam, gamma_e,gamma_e_dot)
        # u_o = null_g["u"][i_o]
        # u_e= null_g["u"][i_e]
        Xdot_null_e = np.asarray([null_g["x_dot"][i_e_null],null_g["y_dot"][i_e_null]],float)
        X_null_e = np.asarray([null_g["x"][i_e_null],null_g["y"][i_e_null]],float)
        # H_e = float(self.H(X_null_e[0],X_null_e[1]))
        v_e = null_g["v"][i_e_null]
        # vdot_e = null_g["v_dot"][i_e]
        X_null_o = np.asarray([null_g["x"][i_o_null],null_g["y"][i_o_null]],float)
        Xdot_null_o = np.asarray([null_g["x_dot"][i_o_null],null_g["y_dot"][i_o_null]],float)
        v_o = null_g["v"][i_o_null]
        vdot_o = null_g["v_dot"][i_o_null]
        H_o = float(self.H(X_null_o[0],X_null_o[1]))
        # partial_u_sig = 0.5*(u_e-u_o)*( -1*Xdot_null_o @ Xdot_null_o - H_o +2*vdot_o ) -0.5*(X_null_e @ Xdot_null_e - X_null_o @ Xdot_null_o-2*v_e+2*v_o) 
        # partial_u_sig = 0
        # partial_v_sig = (u_e-u_o)
        partial_u_r_o= -1
        partial_v_r_o = -null_g["v_dot"][i_o_null]
        # partial_u_r_o = -partial_u_sig/(u_o-u_e)
        # partial_v_r_o = -partial_v_sig/(u_o-u_e)
        
        #calculate \partial_A r_o
        A_eo = self.A(u_e,u_o)
        B_eo = self.B(u_e,u_o)
        i_o_gamma = np.searchsorted(p_e["u"],u_o)
        gamma_o = (p_e["x"][i_o_gamma],p_e["y"][i_o_gamma])
        gamma_o_dot = (p_e["x_dot"][i_o_gamma],p_e["y_dot"][i_o_gamma])
        i_o_obs = np.searchsorted(sol["u"],u_o)
        x_o, y_o = float(sol["x"][i_o_obs]), float(sol["y"][i_o_obs])
        X_o= np.asarray([x_o,y_o],float)
        delta_x_o = X_o-gamma_o
        r_o = np.linalg.solve(B_eo, A_eo @ delta_x_o) - gamma_o_dot

        #calculate r_e
        Bdot_eo = self.B_dot(u_e,u_o)
        Adot_eo = self.A_dot(u_e,u_o)
        r_e_trans = Bdot_eo @ r_o - Adot_eo @ X_o
        parial_u_r_e = 1
        partial_v_r_e = null_g["v_dot"][i_e_null]
        r_e = {"du": parial_u_r_e, "dv": partial_v_r_e, "dx": r_e_trans[0], "dy": r_e_trans[1]}
        grad = {
            "du": partial_u_r_o,
            "dv": partial_v_r_o,
            "dx": r_o[0],
            "dy": r_o[1],
        }
        return grad, r_e
    
        
        
    def angle_lightray_dv(self, Xdot_o, r_o, lam_obs):
        r_o_trans = np.asarray([r_o["dx"],r_o["dy"]],float)
        kappa = -1*lam_obs
        Xdot_o_norm = np.sqrt(kappa)*Xdot_o
        abs_vec = Xdot_o_norm+r_o_trans
        # abs_vec = np.abs(abs_vec)
        cos_psi = 1-(2*kappa)/(kappa+abs_vec@abs_vec)
        psi = np.arccos(cos_psi)
        
        return np.degrees(psi)
    
    def angle_lightray_dv_control(self, obs_g,src_g,u_e,u_o):
        A_eo = self.A(u_e,u_o)
        B_eo = self.B(u_e,u_o)
        kappa = -1*src_g["lam"]
        i_o_gamma = np.searchsorted(src_g["u"],u_o)
        gamma_o = (src_g["x"][i_o_gamma],src_g["y"][i_o_gamma])
        i_o_obs = np.searchsorted(obs_g["u"],u_o)
        x_o, y_o = float(obs_g["x"][i_o_obs]), float(obs_g["y"][i_o_obs])
        X_o= np.asarray([x_o,y_o],float)
        delta_x_o = X_o-gamma_o
        vec = np.linalg.solve(B_eo, A_eo @ delta_x_o)
        cos_psi = 1-(2*kappa)/(kappa+vec@vec)
        psi = np.arccos(cos_psi)
        #calc frequency shift
        vec_freq = np.linalg.solve(B_eo,delta_x_o)
        sin_m2 = 1/np.sin(psi/2)**2
        freq_shift_angle = sin_m2*kappa/(kappa+vec_freq@vec_freq)
        #calc frequency shift with h
        num_vec = self.h_mat @ delta_x_o  
        freq_shift_h = 1+(delta_x_o@num_vec)/(kappa+ vec_freq @ vec_freq)
        
        
        return np.degrees(psi), freq_shift_angle, freq_shift_h
    
    def frequence_shift(self, lambda_e, lambda_o, r_o, r_e, Xdot_obs_o, Xdot_src_e):
        """_summary_
            omega_o/omega_e = 
        Returns:
            _type_: _description_
        """
        kappa_e = -1*lambda_e
        kappa_o = -1*lambda_o
        r_o_trans = np.asarray([r_o["dx"],r_o["dy"]],float)
        r_e_trans = np.asarray([r_e["dx"],r_e["dy"]],float)
        v_obs = Xdot_obs_o + r_o_trans
        v_src = Xdot_src_e + r_e_trans
        w_o = (kappa_o+v_obs@v_obs)/np.sqrt(kappa_o)
        w_e = (kappa_e+v_src@v_src)/np.sqrt(kappa_e)
        freq_shift = w_o/w_e
        return freq_shift
    
    # def frequence_control_angle(self, ):
    
    
    # -------- u=const-Fläche zeichnen --------
    @staticmethod
    def draw_u_plane(ax, u_c, xlim, zlim, n=80,
                     face_color='gray', face_alpha=0.28,
                     grid=True, grid_step=20, grid_color='k', grid_alpha=0.25, grid_lw=0.5):
        if np.isscalar(xlim): xlim = (-xlim, xlim)
        if np.isscalar(zlim): zlim = (-zlim, zlim)

        z = np.linspace(zlim[0], zlim[1], n)
        x = np.linspace(xlim[0], xlim[1], n)
        Z, X = np.meshgrid(z, x, indexing='ij')
        T = Z + np.sqrt(2.0) * u_c

        ax.plot_surface(Z, X, T, color=face_color, alpha=face_alpha, linewidth=0)

        if grid:
            s = max(1, int(grid_step))
            r_idx = np.unique(np.r_[0:Z.shape[0]:s, Z.shape[0]-1])
            c_idx = np.unique(np.r_[0:X.shape[1]:s, X.shape[1]-1])
            Zw = Z[np.ix_(r_idx, c_idx)]
            Xw = X[np.ix_(r_idx, c_idx)]
            Tw = T[np.ix_(r_idx, c_idx)]
            ax.plot_wireframe(Zw, Xw, Tw, color=grid_color, linewidth=grid_lw, alpha=grid_alpha)
            
    # ---- NEU: Szene einmalig vorbereiten (gibt fig, ax zurück)
    def setup_scene(self, L=20.0, u_end=None, figsize=(7,6), view_elev=9, view_azim=-105,align_u2_to_straight=False,n_u_align=1200):
        if u_end is None:
            u_end = np.pi/np.sqrt(abs(self.h2))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(0, 0, 0, color='k', s=8, depthshade=False)
        ax.grid(False)
        ax.set_zlim(-L/2-10, 1.5*L+10)
        ax.set_ylim(-L-10, L+10)
        ax.set_xlim(-1.5*L-10, L/2+10)

        # u=0 Ebene
        self.draw_u_plane(ax, self.u0,
                          xlim=(-1.5*L+2, 1.5*L-2),
                          zlim=(-0.75*L+2.5, 1.0*L-7.5),
                          n=80)
        if align_u2_to_straight:
            u_grid_align = np.linspace(self.u0, u_end, n_u_align)
            X0 = np.array([0.0, 0.0])
            X0_dot_zero = np.array([0.0, 0.0])
            out_straight = self.solve(u_grid_align, X0, X0_dot_zero, v0=0.0, lam=self.lam)
            # Index des u-Endes (falls numerisch minimal versetzt)
            i = np.searchsorted(out_straight["u"], u_end)
            i = min(i, len(out_straight["u"]) - 1)
            z_star = out_straight["z"][i]
            zlim_u2 = (z_star - 0.75*L + 2.5, z_star + 1.0*L - 7.5)
        else:
            zlim_u2 = (-0.75*L+2.5, 1.0*L-7.5)

        # u=u_end Ebene (ohne z-Verschiebung; optional später feinjustieren)
        self.draw_u_plane(ax, u_end,
                          xlim=(-1.5*L+2, 1.5*L-2),
                          zlim=(-0.75*L+2.5, 1.0*L-7.5),
                          n=80)

        # ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(r"$z$", fontsize=16, labelpad=6)
        ax.set_ylabel(r"$x$", fontsize=16, labelpad=6)
        ax.set_zlabel(r"$t$", fontsize=16, labelpad=6)
        ax.view_init(elev=view_elev, azim=view_azim)
        return fig, ax
    # ---- eine einzelne Geodäte in bestehenden Ax hinzufügen
    def add_geodesic(self, ax, u_grid,
                 X0=(0.0, 0.0),
                 X0_dot=(0.0, 0.0),
                 v0=0.0, lam=None,
                 u0=None, v0_dot = None,
                 align_u_grid_to_u0=True,
                 **plt_kwargs):
        if u0 is None:
            u0 = self.u0
        if v0 is None:
            v0 = self.v0
        u_grid = np.asarray(u_grid)
        if align_u_grid_to_u0:
            shift = float(u0) - float(u_grid[0])
            u_grid_eff = u_grid + shift
        else:
            u_grid_eff = u_grid 
        out = self.solve(u_grid_eff, np.array(X0), np.array(X0_dot),
                     v0=v0, lam=lam,u0=u0,v0_dot=v0_dot)
        line = ax.plot(out["z"], out["x"], out["t"], **({"lw":1.0, "color":"darkcyan"}|plt_kwargs))[0]
        return out, line
    
    def timelike_geo_intersect(self, null_geo, cutpoint_at_u, lam_timelike, X_obs_dot_timelike):
        #fix observation event
        u_start = null_geo["u"][0]
        u_end = np.pi/np.sqrt(abs(self.h2))
        u_grid = np.linspace(u_start, u_end, 1000)
        i_cut = np.searchsorted(null_geo["u"],cutpoint_at_u)
        u_obs = null_geo["u"][i_cut]
        v_obs = null_geo["v"][i_cut]
        x_obs = null_geo["x"][i_cut]
        # x_dot_obs = null_geo["x_dot"][i_cut]
        # y_dot_obs = null_geo["y_dot"][i_cut]
        y_obs = null_geo["y"][i_cut]
        # v_dot_obs = null_observed["v_dot"][i_cut]
        #timelike geodesic from the observation point up
        v0dot_future_timelike = self.v0_dot_for_lam(lam_target=lam_timelike, X0=np.asarray([x_obs,y_obs],float),X0_dot=np.asarray([X_obs_dot_timelike[0],X_obs_dot_timelike[1]],float))
        # obs_future = self.solve(u_grid,X0=(x_obs,y_obs), X0_dot=(X_obs_dot_timelike[0], X_obs_dot_timelike[1]), v0_dot=v0dot_future_timelike, u0=u_obs, v0=v_obs)
        #past of the timelike observer
        u_grid_back = np.linspace(u_obs, 0, 600)
        obs_hist = self.solve(u_grid_back,X0=np.asarray([x_obs,y_obs],float),X0_dot=np.asarray([X_obs_dot_timelike[0], X_obs_dot_timelike[1]],float),
        v0_dot=v0dot_future_timelike,
        u0=u_obs,
        v0=v_obs)
        u_grid_ges = np.linspace(u_start, u_end, 1600)
        obs_ges, obs_ges_line = self.add_geodesic(ax, u_grid_ges, X0=(obs_hist["x"][-1],obs_hist["y"][-1]), X0_dot=(obs_hist["x_dot"][-1], obs_hist["y_dot"][-1]), v0_dot=obs_hist["v_dot"][-1], u0=u_start, v0=obs_hist["v"][-1], color="k")
        return obs_ges, obs_ges_line
    
    def comoving_geo_at_observation_event(self, timelike_geo, u_observ, X_at_u_observ):
        i_obs_obs = np.searchsorted(timelike_geo["u"],u_observ)
        i_em_obs = np.searchsorted(timelike_geo["u"],0)
        Xdot_o = np.asarray([timelike_geo["x_dot"][i_obs_obs],timelike_geo["y_dot"][i_obs_obs]],float)
        lam = timelike_geo["lam"]
        delta_x = timelike_geo["x"][i_em_obs]-timelike_geo["x"][i_obs_obs]
        delta_y = timelike_geo["y"][i_em_obs]-timelike_geo["y"][i_obs_obs]
        X_o = np.asarray([delta_x,delta_y],float)
        v0dot_timelike = self.v0_dot_for_lam(lam_target=lam, X0=X_at_u_observ,X0_dot=Xdot_o)
        u_grid_hist = np.linspace(u_observ,0,len(timelike_geo["u"]))
        geo_past,geo_past_line = self.add_geodesic(ax,u_grid_hist,X0=X_at_u_observ, X0_dot=Xdot_o, v0=timelike_geo["v"][i_obs_obs],v0_dot=v0dot_timelike, u0=u_obs)
        i_em_src = np.searchsorted(geo_past["u"],0)
        u=geo_past["u"][i_em_src]
        v=geo_past["v"][i_em_src]
        x=geo_past["x"][i_em_src]
        y=geo_past["y"][i_em_src]
        print(f"geopast(u,v,x,y)_em:({u},{v},{x},{y})")
        return geo_past,geo_past_line
    
    def comoving_geo_through_origin(self, obs_geo, u_obs, n_back=600, n_fwd=600, u_end=None):
        """
        Baut die Geodäte der Quelle, die
        (i) am Beobachtungsereignis u=u_obs instantan mit obs_geo comoving ist und
        (ii) durch den Nullpunkt (u,v,x,y)=(0,0,0,0) geht.
        Gibt Past-Teil (bis u=0) und optional Future-Teil zurück.
        """
        # Index am Beobachtungs-Event
        i = int(np.searchsorted(obs_geo["u"], u_obs))
        lam = float(obs_geo["lam"])
        Xdot_o = np.array([obs_geo["x_dot"][i], obs_geo["y_dot"][i]], dtype=float)

        # Fundamentallösungen von u_obs nach 0
        A_0o = self.A(0.0, u_obs)
        B_0o = self.B(0.0, u_obs)

        # X_o so wählen, dass X(0)=0
        # (A kann bei Fokussierungen singulär werden -> kleiner Offset als Fallback)
        # try:
        X_o = -np.linalg.solve(A_0o, B_0o @ Xdot_o)
        # except np.linalg.LinAlgError:
        #     eps = 1e-10
        #     A_eps = self.A(eps, u_obs)
        #     B_eps = self.B(eps, u_obs)
        #     X_o = -np.linalg.solve(A_eps, B_eps @ Xdot_o)

        # v0 so wählen, dass v(0)=0
        v0 = 0.5 * ( Xdot_o @ X_o - lam * u_obs )

        # v0_dot passend zur gleichen Norm (Komoving)
        v0_dot = self.v0_dot_for_lam(lam_target=lam, X0=X_o, X0_dot=Xdot_o)

        # Past-Anteil bis u=0 integrieren
        u_back = np.linspace(u_obs, 0.0, n_back)
        src_past = self.solve(u_back, X0=X_o, X0_dot=Xdot_o, v0=v0, lam=lam, u0=u_obs)
        # src_past, src_past_line = self.add_geodesic(ax,)
        # Optional: Future-Anteil ab u_obs
        src_future = None
        if u_end is not None:
            u_fwd = np.linspace(u_obs, u_end, n_fwd)
            src_future = self.solve(u_fwd, X0=X_o, X0_dot=Xdot_o, v0=v0, lam=lam, u0=u_obs)
            u_grid_ges = np.linspace(0,u_end, n_fwd+n_back)
        obs_ges, obs_ges_line = self.add_geodesic(ax, u_grid_ges, X0=(src_past["x"][-1],src_past["y"][-1]), X0_dot=(src_past["x_dot"][-1], src_past["y_dot"][-1]), v0_dot=src_past["v_dot"][-1], u0=0, v0=src_past["v"][-1], color="k")
        return obs_ges, obs_ges_line

    
    def draw_partial_v(self, ax, u, v, x=0.0, y=0.0,
                   length=3.0, color="crimson",
                   arrow_length_ratio=0.2, lw=1.5):
        """
        Zeichne den Basisvektor ∂_v als Pfeil im (z,x,t)-Plot
        mit Fußpunkt bei (u,v,x,y).
        length: Skalenfaktor in Plot-Einheiten (nicht normiert).
        """
        # Fußpunkt ins (z,x,t)-Koordinatensystem
        t0, z0 = brinkmann_to_minkowski(u, v)
        x0 = float(x)

        # ∂_v -> (dz, dx, dt) = (1/√2, 0, 1/√2); skaliert
        dz =  length / np.sqrt(2.0)
        dx =  0.0
        dt =  length / np.sqrt(2.0)

        # 3D-Quiver: ax.quiver(x, y, z, U, V, W, ...)
        ax.quiver(z0, x0, t0, dz, dx, dt,
                length=1.0, normalize=False,
                color=color, linewidth=lw,
                arrow_length_ratio=arrow_length_ratio)

    def draw_brinkmann_vector(self, ax, u, v, x=0.0, y=0.0,
                          vec_u=0.0, vec_v=0.0, vec_x=0.0, vec_y=0.0,
                          color="orange", lw=1.6, arrow_length_ratio=0.2):
        """
        Zeichne einen Pfeil mit Brinkmann-Komponenten (δu,δv,δx,δy) am Event (u,v,x,y).
        Im (z,x,t)-Plot wird dy ignoriert (dein Plot zeigt nur z,x,t).
        """
        # Fußpunkt (u,v) -> (t0,z0)
        t0, z0 = brinkmann_to_minkowski(u, v)
        x0 = float(x)
        y0 = float(y)
        # Vektorabbildung (δu,δv,δx) -> (Δz,Δx,Δt)
        du = float(vec_u); dv = float(vec_v); dx = float(vec_x); dy = float(vec_y)
        dt, dz = brinkmann_to_minkowski(du,dv)
        return ax.quiver(z0, x0, t0, dz, dx, dt,
                        color=color, linewidth=lw,
                        arrow_length_ratio=arrow_length_ratio)    
        
     # ---- plot: akzeptiert optional ax (dann alles in denselben Plot)
    def lightcone(self, u_end=None, n_u=1200, angles_deg=(0,30,60,90,120),X0_dot_list=None, u0=None,v0=None,
             x0=0.0, y0=0.0, ydot0=0.0, L=20.0,
             figsize=(7,6), view_elev=9, view_azim=-105,
             lam=None, ax=None, show=True):
        lam = self.lam if lam is None else lam
        if u_end is None:
            u_end = np.pi/np.sqrt(abs(self.h2))
        if u0 is None:
            u0 = self.u0
        if v0 is None:
            v0 = self.v0
        u_grid = np.linspace(u0, u_end, n_u)
        X0 = np.array([x0, y0], dtype=float)

        created_here = False
        if ax is None:
            fig, ax = self.setup_scene(L=L, u_end=u_end, figsize=figsize,
                                       view_elev=view_elev, view_azim=view_azim)
            created_here = True
        else:
            fig = ax.figure


        cutpoints = []
        # --- Modus A: direkte Startvektoren (kein φ/α)
        if X0_dot_list is not None:
            for vec in X0_dot_list:
                X0_dot = np.array(vec, dtype=float)
                out, _ = self.add_geodesic(ax, u_grid, X0, X0_dot, v0=v0, u0=u0, lam=lam,align_u_grid_to_u0=False)
                ax.scatter(out["z"][-1], out["x"][-1], out["t"][-1], color='k', s=8, depthshade=False)
                cutpoints.append((out["z"][-1], out["x"][-1], out["t"][-1]))

        # --- Modus B: Komfort für nullartige Geodäten via φ/α (wie bisher)
        else:
                        # kleine Helper für alpha(phi)
            def alpha_for_phi(phi_deg):
                phi = np.deg2rad(phi_deg)
                c = np.cos(phi)
                return (np.pi/4 if np.isclose(c,0.0) else np.arctan(1.0/c) - np.pi/4)

            def x0_dot_from_alpha(alpha, x0, y0, y0_dot):
                x0_dot_sq = 2*np.tan(alpha) - self.H(x0, y0) - y0_dot**2
                return np.sqrt(max(0.0, x0_dot_sq))

            for phi in angles_deg:
                alpha = alpha_for_phi(phi)
                xdot_mag = x0_dot_from_alpha(alpha, x0, y0, ydot0)
                for sgn in (-1.0, +1.0):
                    X0_dot = np.array([sgn*xdot_mag, ydot0], dtype=float)
                    out, _ = self.add_geodesic(ax, u_grid, X0, X0_dot, v0=v0, u0=u0, lam=lam,align_u_grid_to_u0=False)
                    ax.scatter(out["z"][-1], out["x"][-1], out["t"][-1], color='k', s=8, depthshade=False)
                    cutpoints.append((out["z"][-1], out["x"][-1], out["t"][-1]))

        # Parabel-Fit in u=u_end-Ebene
        cp = np.array(cutpoints)
        coeff = np.polyfit(cp[:,1], cp[:,0], 2)  # z(x)
        x_fit = np.linspace(cp[:,1].min(), cp[:,1].max(), 400)
        z_fit = np.polyval(coeff, x_fit)
        t_fit = z_fit + np.sqrt(2.0) * u_end
        ax.plot(z_fit, x_fit, t_fit, 'k--', lw=0.9, label='Parabel-Fit')

        if created_here and show:
            plt.tight_layout()
            plt.show()
        return fig, ax


class LensingTools:
    
    def __init__(self):
        
        pass



# -----------------------------

# -----------------------------
if __name__ == "__main__":
    w=1/9
    gp = GeodesicPlaneWave(h_plus=0.0,h_p=w**2, lam=0)  # z.B. Nullgeodäten
    fig, ax = gp.setup_scene(L=20,align_u2_to_straight=True)

    u_end = np.pi/np.sqrt(abs(gp.h2))
    u_grid = np.linspace(gp.u0, u_end, 1000)
    # X0 = np.array([0.0, 0.0])

    #lightcone emission event at 0,0,0,0
    gp.lightcone(u_end=u_end,angles_deg=(0,30,60,90,120),ax=ax,show=False)
    
    #calculate the values of the light for phi=90
    phi = 120
    alpha = gp.alpha_for_phi(phi)
    x0, y0 = 0.0, 0.0
    ydot0   = 0.0
    x0dot_obs = gp.x0_dot_from_alpha(alpha,x0,y0,ydot0)
    X0_null      = np.array([x0, y0])
    X0_dot_null  = np.array([+x0dot_obs, ydot0])
    null_observed = gp.solve(u_grid, X0=X0_null, X0_dot=X0_dot_null, v0=0.0, lam=0, u0=gp.u0)
    v0_dot_obs = gp.v0_dot_for_lam(lam_target=0,X0=(x0,y0),X0_dot=(x0dot_obs,ydot0))
    #second lightcone at (0,10,0,0) in minkowski
    # u_start_sec,v_start_sec =  minkowski_to_brinkmann(0,10)
    # u_end_sec = u_start_sec+np.pi/np.sqrt(abs(gp.h2))
    # u_grid_sec = np.linspace(u_start_sec,u_end_sec,1000)
    # gp.lightcone(u_end=u_end_sec,angles_deg=(0,30,60,90,120),ax=ax,u0=u_start_sec,v0=v_start_sec,x0=10,show=False)
    #set the observationevent 
    u_obs = 22
    obs, obs_line = gp.timelike_geo_intersect(null_geo=null_observed,cutpoint_at_u=u_obs,lam_timelike=-1,X_obs_dot_timelike=np.asarray([-1.0, 0],float))
    target_lam = obs["lam"]
    
    #extract data from observer geodesic at the emission phase for the source
    srs_x0dot= obs["x_dot"][0]
    srs_y0dot = obs["y_dot"][0]
    srs_v0 = obs["v"][0]
    srs_v0dot = gp.v0_dot_for_lam(target_lam, X0=(0,0),X0_dot=np.asarray([srs_x0dot,srs_y0dot],float)) 
    #plot source worldine analog to observer
    # print(srs["lam"])
    #draw r_o
    
    #nach dem index der source geodesic suchen an observation phase
    i_obs = np.searchsorted(null_observed["u"],u_obs)
    v_obs = null_observed["v"][i_obs]
    x_obs = null_observed["x"][i_obs]
    y_obs = null_observed["y"][i_obs]
    i_obs_obs = np.searchsorted(obs["u"],u_obs)
    xdot_obs = obs["x_dot"][i_obs_obs]
    ydot_obs = obs["y_dot"][i_obs_obs]
    Xdot_o_vec = np.asarray([xdot_obs, ydot_obs],float)


    # gp.comoving_geo_at_observation_event(timelike_geo=obs,u_observ=u_obs,X_at_u_observ=np.asarray([srs["x"][i_obs_srs],srs["y"][i_obs_srs]],float))
    src, src_line = gp.comoving_geo_through_origin(obs_geo=obs,u_obs=u_obs, n_back=800, u_end=np.pi/np.sqrt(abs(gp.h2)))
    
    # ax.plot(src["past"]["z"], src["past"]["x"], src["past"]["t"], color="crimson", lw=1.4, label="comoving source (past)")
    # if src["future"] is not None:
    #     ax.plot(src["future"]["z"], src["future"]["x"], src["future"]["t"], color="crimson", lw=1.0, ls="--")
    # i0 = -1
    i_obs_src = int(np.argmin(np.abs(src["u"] - u_obs)))
    
    i_em_src = np.searchsorted(src["u"],0)
    
    src_xdot_e = src["x_dot"][i_em_src]
    src_ydot_e = src["y_dot"][i_em_src]
    Xdot_src_e = np.asarray([src_xdot_e,src_ydot_e],float)
    
    calc_grad = gp.sigma_grad_from_sol(p_e=src,sol=obs,null_g=null_observed, u_e=0,u_o=u_obs)
    r_o_vec = calc_grad[0]
    r_e_vec = calc_grad[1]
    gp.draw_brinkmann_vector(ax,u=u_obs,v=v_obs,x=x_obs,y=y_obs,vec_u=r_o_vec["du"],vec_v=r_o_vec["dv"],vec_x=r_o_vec["dx"],vec_y=r_o_vec["dy"])
    gp.draw_brinkmann_vector(ax,u=0,v=0,x=0,y=0,vec_u=r_e_vec["du"],vec_v=r_e_vec["dv"],vec_x=-r_e_vec["dx"],vec_y=-r_e_vec["dy"])
    gp.draw_partial_v(ax,u_obs,v_obs,x_obs,y_obs)
    psi = gp.angle_lightray_dv(lam_obs=target_lam,Xdot_o=Xdot_o_vec,r_o=r_o_vec)
    psi2,freq_shift_control_angle,freq_shift_control_h = gp.angle_lightray_dv_control(obs_g=obs,src_g=src,u_e=0,u_o=u_obs)
    print(psi)
    print(psi2)
    fre_shift = gp.frequence_shift(lambda_e=target_lam, lambda_o=target_lam,r_o=r_o_vec, r_e=r_e_vec,Xdot_obs_o=Xdot_o_vec,Xdot_src_e=Xdot_src_e)
    delta_x_o= obs["x"][i_obs_obs]-src["x"][i_obs_src]
    delta_y_o = obs["y"][i_obs_obs]-src["y"][i_obs_src]
    print("u at i_obs_src =", src["u"][i_obs_src])
    print("source at u=u_obs:", src["x_dot"][i_obs_src], src["y_dot"][i_obs_src], src["lam"])
    print("obs at u=u_obs:", xdot_obs,ydot_obs,obs["lam"] )
    print("frwquency shift", fre_shift)
    print("frwquency shift control angle", freq_shift_control_angle)
    print("frwquency shift control h", freq_shift_control_h)
    print(f"h1*delta_x:{gp.h1}*{delta_x_o}", f"h2*delta_y:{gp.h2}*{delta_y_o}")
    
    # r_o_u =r_o["du"]
    # ro_v = r_o["dv"]
    # ro_x = r_o["dx"]
    # ro_y = r_o["dy"]
    # vdot = null_observed["v_dot"][i_obs]
    # xdot = null_observed["x_dot"][i_obs]
    # ydot = null_observed["y_dot"][i_obs]
    # gp.draw_brinkmann_vector(ax,u=u_obs,v=v_obs,x=x_obs,y=y_obs,vec_u=-1,vec_v=-vdot,vec_x=-xdot,vec_y=-ydot,color="blue")
    # gp.draw_brinkmann_vector(ax,u=0,v=0,x=0,y=0,vec_u=1,vec_v=v0_dot_obs,vec_x=x0dot_obs,vec_y=0)
    
    # print(f"tangent at the observation event: vec_v:{v_dot_obs},vec_x:{x_dot_obs},vec_y:{y_dot_obs}")
    # print(f"tangent at the emission event: vec_v:{v0_dot_obs},vec_x:{x0dot_obs},vec_y:{ydot0}")
    #deltaX=x_obs-x_srs berechnen
    # delta_x = x_obs - srs["x"][i_obs]
    # delta_y = y_obs - srs["y"][i_obs]
    #calculate angle psi between incoming lightray and \partial_v
    # psi = 1-2/(1+abs())
    plt.tight_layout(); plt.show()
