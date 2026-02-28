# find steady state

import time
import numpy as np
from scipy import optimize
from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def aggregate_to_annual(quarterly_data):
    # Ensure the quarterly data length is a multiple of 4 (4 quarters per year)
    if len(quarterly_data) % 4 != 0:
        raise ValueError("The length of the quarterly data must be a multiple of 4.")
    quarterly_data = np.array(quarterly_data)
    reshaped_data = quarterly_data.reshape(-1, 4)
    annual_data = reshaped_data.sum(axis=1)
    return annual_data


def prepare_hh_ss(model):
    """Prepare the household block for finding the steady state."""

    par = model.par
    ss = model.ss

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # a: assets
    par.a_grid[:] = equilogspace(par.a_min, par.a_max, par.Na)

    # z: idiosyncratic productivity
    par.z_grid[:], ss.z_trans[:,:,:], e_ergodic, _, _ = log_rouwenhorst(
        par.rho_z, par.sigma_psi, n=par.Nz
    )

    ###########################
    # 2. initial distribution #
    ###########################

    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,0]  = e_ergodic / par.Nfix
        ss.Dbeg[i_fix,:,1:] = 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix, par.Nz, par.Na))
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            z = par.z_grid[i_z]
            income = ss.w*ss.L*z + ss.chi

            c = (1+ss.ra)*par.a_grid + income
            v_a[i_fix,i_z,:] = c**(-par.sigma)

        ss.vbeg_a[i_fix] = ss.z_trans[i_fix] @ v_a[i_fix]


def evaluate_ss(model, do_print=False):
    """Evaluate steady state (no schema-breaking extra keys)."""

    par = model.par
    ss = model.ss

    # a. fixed nominal/real anchors
    ss.chi = 0.0
    ss.L = 1.0
    ss.pi = ss.pi_w = 0.0
    ss.eps_i = 0.0

    # b. monetary policy
    ss.ra = ss.i = ss.r = par.r_target_ss

    # c. firms / production
    par.Gamma = 1.0
    ss.Y = par.Gamma * ss.L
    ss.w = par.Gamma / par.mu
    ss.Z = ss.w * ss.L

    # d. government
    ss.chi = ss.r * ss.B

    # e. household blocks
    model.solve_hh_ss(do_print=False)
    model.simulate_hh_ss(do_print=False)

    # f. market clearing / dividend pricing
    ss.Div = ss.Y - ss.w * ss.L
    ss.pD = ss.Div / ss.r

    ss.clearing_A = ss.A_hh - ss.pD - ss.B
    ss.clearing_Y = ss.Y - ss.C_hh

    # g. NK wage curve (pins varphi)
    par.varphi = (1/par.mu * ss.w * ss.C_hh**(-par.sigma)) / ss.L**par.nu
    ss.NKWC_res = 0.0  # used to derive par.varphi

def obj_ss(x, model, do_print=False):
    """Objective function for finding the steady state."""

    par = model.par
    ss  = model.ss

    if par.do_B:
        # Solve for markup, debt, and beta
        par.mu   = x[0]
        ss.B     = x[1]
        par.beta = x[2]
    else:
        # Solve for mu and beta only, debt = 0
        par.mu   = x[0]
        ss.B     = 0.0
        par.beta = x[1]

    # Evaluate steady state
    evaluate_ss(model, do_print=do_print)

    # Compute MPC Jacobian
    model._compute_jac_hh(inputs_hh_all=['chi'])
    ann_mpcs = aggregate_to_annual(-model.jac_hh[('C_hh','chi')][:,0])

    # Residuals
    residuals = []

    # 1. Asset market clearing
    residuals.append(ss.clearing_A)

    # 2. MPC normalization (e.g. target = 0.5)
    residuals.append(ann_mpcs[0] - 0.5)

    # 3. Goods market clearing
    residuals.append(ss.clearing_Y)

    return np.array(residuals)


def find_ss(model, do_print=False):
    """Find the steady state."""

    par = model.par
    ss  = model.ss

    # Initial guess
    if par.do_B:
        par.mu   = 1.05     # initial guess for markup
        ss.B     = 0.7      # initial guess for debt
        par.beta = 0.987    # initial guess for beta
        x0 = np.array([par.mu, ss.B, par.beta])
    else:
        par.mu   = 1.00778
        par.beta = 0.988
        x0 = np.array([par.mu, par.beta])

    # Solve using root-finding
    sol = optimize.root(obj_ss, x0, method='hybr', args=(model, do_print))

    if not sol.success:
        raise RuntimeError(f"Steady state solver failed: {sol.message}")

    # Optional print
    if do_print:
        print(f' Y     = {ss.Y:12.6f}')
        print(f' C     = {ss.C_hh:12.6f}')
        print(f' A     = {ss.A_hh:12.6f}')
        print(f' B     = {ss.B:12.6f}')
        print(f' pD    = {ss.pD:12.6f}')
        print(f' mu    = {par.mu:12.6f}')
        print(f' beta  = {par.beta:12.8f}')
        print(f' r     = {ss.r:12.6f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')