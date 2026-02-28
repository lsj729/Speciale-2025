import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass
import numba as nb
import household_problem
import steady_state
import blocks

class HANKModelClass(EconModelClass, GEModelClass):

    #########
    # setup #
    #########

    def settings(self):
        """Fundamental settings."""

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']

        # b. household
        self.grids_hh      = ['a']              # grids
        self.pols_hh       = ['a']              # policy functions
        self.inputs_hh     = ['Z','ra','chi']   # direct inputs
        self.inputs_hh_z   = []                 # transition matrix inputs
        self.outputs_hh    = ['a','c']          # outputs
        self.intertemps_hh = ['vbeg_a']         # intertemporal variables

        # (note) we will NOT add new keys to ss to respect EconModel schema.

        # c. GE
        self.shocks   = ['eps_i']
        self.unknowns = ['pi_w','L']
        self.targets  = ['NKWC_res','clearing_A']

        # d. all variables
        self.blocks = [
            'blocks.central_bank',
            'blocks.production',
            'blocks.mutual_fund',
            'blocks.government',
            'hh',
            'blocks.NKWC',
            'blocks.market_clearing'
        ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """Set baseline parameters."""
        par = self.par

        # a. preferences and HH_type
        par.Nfix   = 1
        par.beta   = 0.987
        par.varphi = np.nan
        par.HH_type = 'HANK'  # 'HANK' or 'RANK'

        par.sigma = 1.0  # IES^-1
        par.nu    = 1.0  # Frisch^-1

        # c. income parameters
        par.rho_z     = 0.95
        par.sigma_psi = 0.10
        par.Nz        = 9

        # d. price setting
        par.kappa = 0.1
        par.mu    = np.nan

        # Government
        par.do_B = False
        par.omega = 0.05

        # e. firms
        par.Gamma = np.nan

        # f. CB
        par.phi_pi      = 1.25
        par.r_target_ss = 0.02/4

        # g. grids
        par.a_min = 0.0
        par.a_max = 150.0
        par.Na    = 300

        # h. shocks
        par.jump_eps_i = 0.005
        par.rho_eps_i  = 0.80
        par.std_eps_i  = 0.00

        # EXP (kept as in your code)
        par.g_lambda = 1.0

        # misc.
        par.T = 300

        par.max_iter_solve    = 50_000
        par.max_iter_simulate = 50_000
        par.max_iter_broyden  = 100

        par.tol_ss      = 1e-12
        par.tol_solve   = 1e-12
        par.tol_simulate= 1e-12
        par.tol_broyden = 1e-10

        par.py_hh    = True
        par.py_blocks= True

    def allocate(self):
        """Allocate model containers."""
        par = self.par
        self.allocate_GE()

    # expose SS helpers from steady_state.py
    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss       = steady_state.find_ss

    # -------- NEW: on-demand exposures (no new ss keys) -----------------------
    def get_exposures(self):
        """
        Compute stationary exposure objects from existing, schema-approved arrays.

        Returns
        -------
        mu_za  : (Nz, Na) joint stationary distribution over (z, a), aggregated over i_fix
        mass_z : (Nz,)     population mass per productivity state
        Abar_z : (Nz,)     average assets by z (level, not share)
        eff_z  : (Nz,)     efficiency/wage proxy (uses par.z_grid)
        """
        ss, par = self.ss, self.par

        # ss.D has shape (Nfix, Nz, Na). Aggregate across i_fix:
        mu_za  = np.sum(ss.D, axis=0)                  # (Nz, Na)
        mass_z = np.sum(mu_za, axis=1)                 # (Nz,)
        Abar_z = mu_za @ par.a_grid                    # (Nz,)
        eff_z  = par.z_grid.copy()                     # (Nz,)

        return mu_za, mass_z, Abar_z, eff_z
    # --------------------------------------------------------------------------

    def calc_MPC(self):
        """MPC diagnostics (unchanged)."""
        par = self.par
        ss  = self.ss
        MPC = np.sum(
            ss.D[:,:,:-1]*(ss.c[:,:,1:]-ss.c[:,:,:-1]) /
            ((1+ss.ra)*(par.a_grid[1:]-par.a_grid[:-1]))
        )
        iMPC = -self.jac_hh[('C_hh','chi')]
        annual_MPC = np.sum(iMPC[:4,0])
        print(f'{MPC = :.2f}, {iMPC[0,0] = :.2f}')
        print(f'{annual_MPC = :.2f}')

    def get_RA_J(self):
        """Return simple RA Jacobians for (C_hh, A_hh) wrt (Z, ra)."""
        par, ss = self.par, self.ss
        T = par.T
        M_RA = {
            'C_hh': {'Z': np.zeros((T,T)), 'ra': np.zeros((T,T))},
            'A_hh': {'Z': np.zeros((T,T)), 'ra': np.zeros((T,T))},
        }

        h  = 1e-4
        Z  = np.zeros(T) + ss.Z
        ra = np.zeros(T) + ss.ra

        for s in range(T):
            # Z shock
            Z_ = Z.copy(); Z_[s] += h
            C,A = RA_block(ra, Z_, par.sigma, ss.ra, ss.C_hh, ss.A_hh, T)
            M_RA['C_hh']['Z'][:,s] = (C - ss.C_hh)/h
            M_RA['A_hh']['Z'][:,s] = (A - ss.A_hh)/h

            # ra shock
            ra_ = ra.copy(); ra_[s] += h
            C,A = RA_block(ra_, Z, par.sigma, ss.ra, ss.C_hh, ss.A_hh, T)
            M_RA['C_hh']['ra'][:,s] = (C - ss.C_hh)/h
            M_RA['A_hh']['ra'][:,s] = (A - ss.A_hh)/h

        return M_RA


@nb.njit
def RA_block(ra, Z, sigma, ss_ra, ss_C, ss_A, T):
    C_hh = np.zeros(T)
    A_hh = np.zeros(T)

    # Euler (very stylized)
    beta_RA = 1/(1+ss_ra)
    for s in range(T):
        t = T - 1 - s  # backward time if needed
        if t == T-1:
            C_hh[t] = ss_C
            A_hh[t] = ss_A
        else:
            # placeholder linearization
            C_hh[t] = ss_C + (Z[t]-ss_C)*0.0 - sigma*(ra[t]-ss_ra)
            A_hh[t] = ss_A + (Z[t]-ss_C)*0.0 + (ra[t]-ss_ra)*0.0

    return C_hh, A_hh
