
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass
import numba as nb 
import matplotlib.pyplot as plt   



def hetero_expectations_IRFs(model, E_by_z):
    """
    Build HH policy IRFs using per-z beliefs and store in model.IRF['pols'][(pol, shock)].
    Requires: model.compute_jacs() already run (populates model.dpols).
    """
    par, ss = model.par, model.ss
    T, Nz, Nfix = par.T, par.Nz, par.Nfix

    # Build input IRFs (skip household part) and reuse G_U already built
    model.prepare_simulate(skip_hh=True, reuse_G_U=True, do_print=False)
    if 'pols' not in model.IRF: model.IRF['pols'] = {}

    for shockname in model.shocks:
        for polname in model.pols_hh:  # e.g., ['a']
            IRF_pols = np.zeros((*ss.D.shape, T))  # (Nfix, Nz, Na, T)
            for inputname in model.inputs_hh_all:  # e.g., 'Z','ra'
                base_IRF = model.IRF[(inputname, shockname)]  # (T,)
                if inputname in E_by_z:
                    IRF_per_z = np.vstack([E_by_z[inputname][iz] @ base_IRF for iz in range(Nz)])  # (Nz,T)
                else:
                    IRF_per_z = np.tile(base_IRF, (Nz,1))
                dpols = model.dpols[(polname, inputname)]  # (T, Nfix, Nz, Na)

                # by-z convolution
                for i_fix in range(Nfix):
                    for iz in range(Nz):
                        irf_z = IRF_per_z[iz]
                        for t in range(T):
                            IRF_pols[i_fix, iz, :, t] += dpols[:T-t, i_fix, iz, :].T @ irf_z[t:]
            model.IRF['pols'][(polname, shockname)] = IRF_pols

def hetero_expectations_IRFs_from_paths(model, E_by_z, base_paths_by_input, shockname):
    """
    Build HH policy IRFs using *provided* input paths (arrays length T) and per-z beliefs.
    - DOES NOT call prepare_simulate() and DOES NOT read IRF[(input,shock)].
    - Writes model.IRF['pols'][(pol, shockname)] only.
    """
    par, ss = model.par, model.ss
    T, Nz, Nfix = par.T, par.Nz, par.Nfix

    if 'pols' not in model.IRF: model.IRF['pols'] = {}

    for polname in model.pols_hh:                       # e.g. 'a'
        IRF_pols = np.zeros((*ss.D.shape, T))          # (Nfix, Nz, Na, T)

        for inputname in model.inputs_hh_all:          # e.g. 'Z','ra','chi'
            # use the provided path (default zeros if not supplied)
            base_path = base_paths_by_input.get(inputname, np.zeros(T))

            # per-z perceived path via E_z, or same for all z
            if inputname in E_by_z:
                IRF_per_z = np.vstack([E_by_z[inputname][iz] @ base_path for iz in range(Nz)])  # (Nz,T)
            else:
                IRF_per_z = np.tile(base_path, (Nz,1))  # (Nz,T)

            dpols = model.dpols[(polname, inputname)]  # (T, Nfix, Nz, Na)

            # time convolution
            for i_fix in range(Nfix):
                for iz in range(Nz):
                    pz = IRF_per_z[iz]
                    for t in range(T):
                        IRF_pols[i_fix, iz, :, t] += dpols[:T-t, i_fix, iz, :].T @ pz[t:]

        model.IRF['pols'][(polname, shockname)] = IRF_pols

def rebuild_jac_hh_hetero(model, E_by_z, hh_outputs=('C_hh','A_hh')):
    """
    Rebuild HH Jacobians jac_hh[(o,i)] with per-z beliefs E_by_z by columns.
    DOES NOT rely on IRF[(input,shock)] to carry the basis; avoids overwriting.
    After running, call: model.compute_jacs(skip_hh=True); model.find_IRFs(..., reuse_G_U=False).
    """
    par, ss = model.par, model.ss
    T, Nz = par.T, par.Nz
    jac_hh_alt = {}
    hh_inputs = tuple(model.inputs_hh_all)              # e.g. ('Z','ra','chi', ...)

    # Make sure the model has a shock name and IRF containers initialized
    if not hasattr(model, 'shocks') or len(model.shocks) == 0:
        raise RuntimeError("model.shocks is empty; run find_IRFs once to initialize.")
    shockname = model.shocks[0]

    # Initialize IRF dicts if needed
    if 'pols' not in model.IRF: model.IRF['pols'] = {}

    for i_name in hh_inputs:
        for s in range(T):
            # basis path for this input: e_s
            basis = np.zeros(T); basis[s] = 1.0
            base_paths = {i_name: basis}               # other inputs default to zeros

            # 1) Build policy IRFs directly from provided basis paths (per-z beliefs applied here)
            hetero_expectations_IRFs_from_paths(model, E_by_z, base_paths, shockname)

            # 2) Aggregate to HH outputs using existing 'pols' (do NOT rebuild HH)
            model.prepare_simulate(skip_hh=True, reuse_G_U=True, do_print=False)

            # 3) Read column s for each requested HH output
            for o_name in hh_outputs:
                col = model.IRF.get((o_name, shockname), None)
                if col is None:
                    raise RuntimeError(f"Missing IRF for {o_name}")
                if (o_name, i_name) not in jac_hh_alt:
                    jac_hh_alt[(o_name, i_name)] = np.zeros((T, T))
                jac_hh_alt[(o_name, i_name)][:, s] = col

    # Install on the model
    model.jac_hh.update(jac_hh_alt)
    return jac_hh_alt

def make_forward_endo(shockname: str):
    """
    Return a function forward_endo(IRF_pols, ss, par) that maps the policy IRFs
    into ΔDbeg_plus at each horizon using ONLY arrays and the correct shapes.
    """
    def forward_endo(IRF_pols: dict, ss, par):
        T = par.T
        out = []

        # IMPORTANT: keep the leading policy-dimension intact (no [0] slicing)
        i = ss.pol_indices     # shape (1, Nfix, Nz, Na) for 1D endog
        w = ss.pol_weights     # shape (1, Nfix, Nz, Na)

        # Use the asset policy 'a' to drive transitions (shape Nfix×Nz×Na×T)
        A = IRF_pols[('a', shockname)]

        # Ensure C-contiguous arrays (Numba is picky)
        i = np.ascontiguousarray(i)
        w = np.ascontiguousarray(w)

        for t in range(T):
            Dbeg_plus_t = np.ascontiguousarray(A[..., t])   # (Nfix, Nz, Na)
            dD_t = simulate_hh_forwards_endo_transpose(Dbeg_plus_t, i, w)
            out.append(dD_t)
        return out
    return forward_endo

def forward_exo(dD_endo_T, ss, par):
    """
    Push each ΔD_endo_t through the exogenous z transition (array-only).
    """
    T = par.T
    out = []
    zT = np.ascontiguousarray(ss.z_trans)  # (Nz,Nz) or (Nfix,Nz,Nz), both supported
    for t in range(T):
        Dbeg_t = np.ascontiguousarray(dD_endo_T[t])
        out.append(simulate_hh_forwards_exo_transpose(Dbeg_t, zT))
    return out

def rebuild_jac_hh_hetero_fakenews(
    model,
    E_by_z: dict,
    hh_outputs=('C_hh','A_hh'),
    hh_inputs=None,
    shockname=None,
    # OPTIONAL: pass array-only forwarders to include the distribution term
    # forward_endo(IRF_pols, ss, par) -> list/array of ΔD_t (length T), each shaped like ss.D
    # forward_exo(dD_endo_T, ss, par) -> list/array of ΔD_t (same shape)
    forward_endo=None,
    forward_exo=None,
):
    """
    Build T×T HH Jacobians jac_hh[(O,i)] under per-z expectations (E_by_z).

    For each input i and each column s:
      1) Build per-z perceived basis path e_s^perc = E_z @ e_s.
      2) Convolve with dpols[(pol,i)] to get policy IRFs IRF['pols'][(pol,shock)] over all t.
      3) IMPACT (row vector for this column): y0_vec[t] = <IRF_pol[...,t], D_ss>.
         Store the full vector into column s of F (no cumulation here).
      4) If forward_endo/exo are provided, compute ΔD path and add distribution rows:
         F[τ,s] += <level_O, ΔD[τ-1]> for τ=1..T-1 (O in {C_hh,A_hh} uses level c_ss or a_ss).
    After the column loop:
      - If no forwarders: J = F (impact-only).
      - Else: J = fake-news cumulation across columns:
            J[:,0] = F[:,0];
            J[:,s] = F[:,s] + [0 ; J[:-1,s-1]]  for s>=1.
    Installs into model.jac_hh and returns the dict of updated blocks.
    """
    par, ss = model.par, model.ss
    T, Nz = par.T, par.Nz

    # shock key to stash pol-IRFs under
    if shockname is None:
        if not getattr(model, 'shocks', None):
            raise RuntimeError("model.shocks is empty; run find_IRFs() once to initialize IRF store.")
        shockname = model.shocks[0]

    # which HH inputs to build Jacobians for
    inputs = list(hh_inputs) if hh_inputs is not None else list(model.inputs_hh_all)

    # keep your model’s exact HH-output names
    def _resolve_hh_outkey(want):
        want_low = want.lower()
        for (o,_i) in model.jac_hh.keys():
            if isinstance(o,str) and o.lower() == want_low: return o
        for o in getattr(model,'outputs_hh',[]):
            if isinstance(o,str) and o.lower() == want_low: return o
        return want

    OUTKEY = {O: _resolve_hh_outkey(O) for O in hh_outputs}

    # aggregate chosen output via the matching HH 'policy' (these should be in model.outputs_hh)
    pol_for_out = {'C_hh': 'c', 'A_hh': 'a'}

    # steady-state pieces
    Dss = ss.D                       # (Nfix, Nz, Na)
    c_ss = getattr(ss, 'c', None)    # (Nfix, Nz, Na) or None
    a_ss = getattr(ss, 'a', None)

    # --- helper: build IRF['pols'] for ALL HH outputs from a single perceived path of input 'i_name'
    def _build_pols_perceived_paths(model, E_by_z, i_name, path_1d, shockname):
        if 'pols' not in model.IRF:
            model.IRF['pols'] = {}

        # zero-init for this shock and for each HH output
        for pol in model.outputs_hh:
            model.IRF['pols'][(pol, shockname)] = np.zeros((*Dss.shape, T))

        # perceived path per z
        if i_name in E_by_z:
            Pz = np.vstack([E_by_z[i_name][iz] @ path_1d for iz in range(Nz)])  # (Nz, T)
        else:
            Pz = np.tile(path_1d, (Nz, 1))

        # convolve perceived path with dpols kernels
        for pol in model.outputs_hh:
            d = model.dpols[(pol, i_name)]                 # (T, Nfix, Nz, Na)
            IRFpol = model.IRF['pols'][(pol, shockname)]   # (Nfix, Nz, Na, T)
            for i_fix in range(par.Nfix):
                for iz in range(Nz):
                    pz = Pz[iz]
                    for t in range(T):
                        # (Na x (T-t)) @ ((T-t),) → (Na,)
                        IRFpol[i_fix, iz, :, t] = d[:T-t, i_fix, iz, :].T @ pz[t:]

    # inner product helper
    def _agg_level(level_arr, dD):
        if level_arr is None: return 0.0
        return float(np.sum(level_arr * dD))

    jac_alt = {}  # to collect and return

    # ===== main: loop inputs, build columns, then assemble J for each output =====
    for i_name in inputs:
        # F_by_O will accumulate full columns (impact + optional distribution) BEFORE any cumulation
        F_by_O = {O: np.zeros((T, T)) for O in hh_outputs}

        for s in range(T):
            # basis for this column
            e_s = np.zeros(T); e_s[s] = 1.0

            # 1) per-z perceived policies for this column
            _build_pols_perceived_paths(model, E_by_z, i_name, e_s, shockname)

            # 2) impact: FULL vector y0_vec over t, then store as column s
            for O in hh_outputs:
                pol = pol_for_out[O]
                IRFpol = model.IRF['pols'][(pol, shockname)]             # (Nfix, Nz, Na, T)
                y0_vec = np.sum(IRFpol * Dss[..., None], axis=(0, 1, 2)) # (T,)
                F_by_O[O][:, s] = y0_vec

            # 3) optional distribution rows: add <level_O, ΔD[τ-1]> for τ>=1
            if (forward_endo is not None) and (forward_exo is not None):
                dD_endo_T = forward_endo(model.IRF['pols'], ss, par)  # length T list/array
                dD_T      = forward_exo(dD_endo_T, ss, par)           # same shapes as Dss
                for O in hh_outputs:
                    level = c_ss if O == 'C_hh' else a_ss if O == 'A_hh' else None
                    if level is not None:
                        for tau in range(1, T):
                            F_by_O[O][tau, s] += _agg_level(level, dD_T[tau-1])
                    # if level is None, nothing to add

        # 4) assemble J per output: impact-only -> J=F; with distribution -> fake-news cumulation
        for O in hh_outputs:
            F = F_by_O[O]
            if (forward_endo is None) or (forward_exo is None):
                # impact-only Jacobian: EXACTLY the impact matrix, no cumulation
                J = F.copy()
            else:
                # full fake-news cumulation across columns
                J = np.zeros_like(F)
                J[:, 0] = F[:, 0]
                for s in range(1, T):
                    J[:, s]  = F[:, s]
                    J[1:, s] += J[:-1, s-1]

            jac_alt[(OUTKEY[O], i_name)] = J

    # install and return
    model.jac_hh.update(jac_alt)
    return jac_alt


def E_hybrid(T, lam, theta=0.95):
    E = np.zeros((T,T))
    for s in range(T):
        for t in range(T):
            if t < s:      # before the realization
                E[t,s] = lam**((s-t)/10)
            else:          # after the realization
                E[t,s] = theta**(t-s)
    return E

import numpy as np
import matplotlib.pyplot as plt

def decomp(model, model_het, plot_test=False,modellabel='Baseline model',hetlabel='Sticky expectations model'):
    """
    Compare decomposition of C_hh into contributions from Z, ra, and chi
    for baseline (model) vs sticky/hetero expectations (model_het).
    """

    # ---------- BASELINE HANK ----------
    Js_HA = model.jac_hh
    ssC_HA = model.ss.C_hh

    dC_dZ_HA   = Js_HA[('C_hh', 'Z')]   @ model.IRF['Z']   * 100 / ssC_HA
    dC_dra_HA  = Js_HA[('C_hh', 'ra')]  @ model.IRF['ra']  * 100 / ssC_HA
    dC_dchi_HA = Js_HA[('C_hh', 'chi')] @ model.IRF['chi'] * 100 / ssC_HA

    dC_test_HA = dC_dZ_HA + dC_dra_HA + dC_dchi_HA
    dC_tot_HA  = model.IRF['C_hh'] * 100 / ssC_HA

    # ---------- HET / STICKY EXPECTATIONS ----------
    Js_RA = model_het.jac_hh
    ssC_RA = model_het.ss.C_hh

    dC_dZ_RA   = Js_RA[('C_hh', 'Z')]   @ model_het.IRF['Z']   * 100 / ssC_RA
    dC_dra_RA  = Js_RA[('C_hh', 'ra')]  @ model_het.IRF['ra']  * 100 / ssC_RA
    dC_dchi_RA = Js_RA[('C_hh', 'chi')] @ model_het.IRF['chi'] * 100 / ssC_RA

    dC_test_RA = dC_dZ_RA + dC_dra_RA + dC_dchi_RA
    dC_tot_RA  = model_het.IRF['C_hh'] * 100 / ssC_RA

    # ---------- PLOT ----------
    lw = 2.5
    colors = {
        "tot": "navy",
        "Z": "firebrick",
        "ra": "forestgreen",
        "chi": "purple",
        "test": "orange"
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Plot HET/STICKY
    axes[0].plot(np.zeros(21), color='black')
    axes[0].plot(dC_tot_RA[:21], label='Total', linewidth=lw, color=colors["tot"])
    axes[0].plot(dC_dZ_RA[:21], label='Z', linewidth=lw, color=colors["Z"])
    axes[0].plot(dC_dra_RA[:21], label='ra', linewidth=lw, color=colors["ra"])
    axes[0].plot(dC_dchi_RA[:21], label='chi', linewidth=lw, color=colors["chi"])
    if plot_test:
        axes[0].plot(dC_test_RA[:21], label='Z + ra + chi', linestyle='--', color=colors["test"])
    axes[0].set_title(modellabel)
    axes[0].set_xlabel('Quarters')
    axes[0].set_ylabel('% change in C')

    # Plot BASELINE
    axes[1].plot(np.zeros(21), color='black')
    axes[1].plot(dC_tot_HA[:21], label='Total', linewidth=lw, color=colors["tot"])
    axes[1].plot(dC_dZ_HA[:21], label='Z', linewidth=lw, color=colors["Z"])
    axes[1].plot(dC_dra_HA[:21], label='ra', linewidth=lw, color=colors["ra"])
    axes[1].plot(dC_dchi_HA[:21], label='chi', linewidth=lw, color=colors["chi"])
    if plot_test:
        axes[1].plot(dC_test_HA[:21], label='Z + ra + chi', linestyle='--', color=colors["test"])
    axes[1].set_title(hetlabel)
    axes[1].set_xlabel('Quarters')
    axes[1].set_ylabel('% change in C')

    axes[1].legend()
    plt.tight_layout()
    plt.show()


def decomp_single(model, title='Model decomposition', plot_test=False, T=21):
    """
    Plot consumption decomposition for a single model into contributions from Z, ra, and chi.
    """

    Js = model.jac_hh
    ssC = model.ss.C_hh

    dC_dZ   = Js[('C_hh', 'Z')]   @ model.IRF['Z']   * 100 / ssC
    dC_dra  = Js[('C_hh', 'ra')]  @ model.IRF['ra']  * 100 / ssC
    dC_dchi = Js[('C_hh', 'chi')] @ model.IRF['chi'] * 100 / ssC

    dC_test = dC_dZ + dC_dra + dC_dchi
    dC_tot  = model.IRF['C_hh'] * 100 / ssC

    lw = 2.5
    colors = {
        "tot": "navy",
        "Z": "firebrick",
        "ra": "forestgreen",
        "chi": "purple",
        "test": "orange"
    }

    plt.figure(figsize=(5.5, 4))
    plt.plot(np.zeros(T), color='black')
    plt.plot(dC_tot[:T], label='Total', linewidth=lw, color=colors["tot"])
    plt.plot(dC_dZ[:T], label='Z', linewidth=lw, color=colors["Z"])
    plt.plot(dC_dra[:T], label='ra', linewidth=lw, color=colors["ra"])
    plt.plot(dC_dchi[:T], label='chi', linewidth=lw, color=colors["chi"])

    if plot_test:
        plt.plot(dC_test[:T], label='Z + ra + chi', linestyle='--', color=colors["test"])

    plt.title(title)
    plt.xlabel('Quarters')
    plt.ylabel('% change in C')
    plt.legend()
    plt.tight_layout()
    plt.show()


def decomp_single_A(model, title='Model decomposition', plot_test=False, T=21):
    """
    Plot consumption decomposition for a single model.
    """

    Js = model.jac_hh
    ssA = model.ss.A_hh

    dA_dZ = Js[('A_hh', 'Z')] @ model.IRF['Z'] * 100 / ssA
    dA_dra = Js[('A_hh', 'ra')] @ model.IRF['ra'] * 100 / ssA
    dA_test = dA_dZ + dA_dra
    dA_tot = model.IRF['A_hh'] * 100 / ssA

    lw = 2.5
    colors = ['navy', 'firebrick', 'forestgreen', 'orange']

    plt.figure(figsize=(5.5, 4))
    plt.plot(np.zeros(T), color='black')
    plt.plot(dA_tot[:T], label='Total', linewidth=lw, color=colors[0])
    plt.plot(dA_dZ[:T], label='Z', linewidth=lw, color=colors[1])
    plt.plot(dA_dra[:T], label='ra', linewidth=lw, color=colors[2])

    if plot_test:
        plt.plot(dA_test[:T], label='Z + ra', linestyle='--', color=colors[3])

    plt.title(title)
    plt.xlabel('Quarters')
    plt.ylabel('% change in A')
    plt.legend()
    plt.tight_layout()
    plt.show()

def rebuild_jac_hh_hetero_fakenews_with_z(
    model,
    E_by_z: dict,
    hh_outputs=('C_hh','A_hh'),
    hh_inputs=None,
    shockname=None,
    forward_endo=None,   # optional: forward_endo(IRF_pols, ss, par) -> list of ΔD_t (Nfix,Nz,Na)
    forward_exo=None,    # optional: forward_exo(dD_endo_T, ss, par) -> list of ΔD_t (Nfix,Nz,Na)
):
    """
    Rebuild T×T household Jacobians under per-z beliefs, and ALSO store
    per-productivity Jacobians.

    Writes:
      - model.jac_hh[(OUT, IN)]                     -> (T,T) aggregate
      - model.jac_hh_by_z[((OUT, IN), iz)]          -> (T,T) for each z
      - model.jac_hh_by_z_stack[(OUT, IN)]          -> (Nz,T,T) stacked by ascending z_grid
      - model.jac_hh_meta['z_order']                -> sorting index for z_grid

    Notes
    -----
    • If forward_endo/exo are provided, the “fake-news” distribution term is added
      before cumulation across columns (same logic as your fakenews builder).
    • Keys for IN must exist in model.dpols[(pol, IN)] for each output’s policy pol.
    """

    par, ss = model.par, model.ss
    T, Nz, Nfix = par.T, par.Nz, par.Nfix
    Dss = ss.D                        # (Nfix, Nz, Na)
    c_ss = getattr(ss, 'c', None)     # (Nfix, Nz, Na) or None
    a_ss = getattr(ss, 'a', None)

    # shock name
    if shockname is None:
        if not getattr(model, 'shocks', None):
            raise RuntimeError("model.shocks is empty; run find_IRFs() once to initialize IRF store.")
        shockname = model.shocks[0]

    # inputs to build for
    inputs = list(hh_inputs) if hh_inputs is not None else list(model.inputs_hh_all)

    # map HH output name -> policy symbol used in dpols / IRF['pols']
    pol_for_out = {'C_hh': 'c', 'A_hh': 'a'}

    # resolve OUT keys to match model storage (case-insensitive safety)
    def _resolve_hh_outkey(want):
        want_low = want.lower()
        for (o,_i) in getattr(model, 'jac_hh', {}).keys():
            if isinstance(o,str) and o.lower() == want_low: return o
        for o in getattr(model,'outputs_hh',[]):
            if isinstance(o,str) and o.lower() == want_low: return o
        return want
    OUTKEY = {O: _resolve_hh_outkey(O) for O in hh_outputs}

    # ensure IRF['pols'] exists
    if 'pols' not in model.IRF:
        model.IRF['pols'] = {}

    # helper: build policy IRFs for ONE input path (1D) applying per-z beliefs
    def _build_pols_perceived_paths(i_name, path_1d):
        # zero-init for this shock and for each HH policy
        for pol in model.outputs_hh:
            model.IRF['pols'][(pol, shockname)] = np.zeros((*Dss.shape, T))
        # perceived path per z
        if i_name in E_by_z:
            Pz = np.vstack([E_by_z[i_name][iz] @ path_1d for iz in range(Nz)])  # (Nz, T)
        else:
            Pz = np.tile(path_1d, (Nz, 1))
        # convolve with dpols
        for pol in model.outputs_hh:
            key = (pol, i_name)
            if key not in model.dpols:
                raise KeyError(f"Missing kernel dpols{key}")
            d = model.dpols[key]                                 # (T, Nfix, Nz, Na)
            IRFpol = model.IRF['pols'][(pol, shockname)]         # (Nfix, Nz, Na, T)
            for i_fix in range(Nfix):
                for iz in range(Nz):
                    pz = Pz[iz]
                    for t in range(T):
                        IRFpol[i_fix, iz, :, t] = d[:T-t, i_fix, iz, :].T @ pz[t:]

    # inner products for distribution term
    def _agg_level(level_arr, dD):
        if level_arr is None: return 0.0
        return float(np.sum(level_arr * dD))

    # per-z inner product
    def _agg_level_by_z(level_arr, dD):
        # returns (Nz,) with <level[:,:,iz,:], dD[:,:,iz,:]>
        if level_arr is None:
            return np.zeros(Nz)
        return np.sum(level_arr * dD, axis=(0,2))  # sum over (i_fix, a)

    jac_alt_agg = {}                   # aggregate J’s to (also) install on model.jac_hh
    jac_hh_by_z = {}                   # per-z dict: key ((OUT, IN), iz) -> (T,T)
    jac_hh_by_z_stack = {}             # per OUT,IN -> stacked (Nz,T,T)

    # ---------- main: loop inputs ----------
    for i_name in inputs:

        # F matrices BEFORE fake-news cumulation (impact + optional distribution rows)
        F_by_O_agg = {O: np.zeros((T, T)) for O in hh_outputs}
        F_by_O_z   = {O: np.zeros((Nz, T, T)) for O in hh_outputs}  # [iz, t, s]

        for s in range(T):
            # basis column
            e_s = np.zeros(T); e_s[s] = 1.0

            # policies (perceived) for this input + column s
            _build_pols_perceived_paths(i_name, e_s)

            # impact pieces (aggregate and by-z)
            for O in hh_outputs:
                pol = pol_for_out[O]
                IRFpol = model.IRF['pols'][(pol, shockname)]  # (Nfix, Nz, Na, T)

                # aggregate impact column (t = 0..T-1)
                y0_vec = np.sum(IRFpol * Dss[..., None], axis=(0,1,2))    # (T,)
                F_by_O_agg[O][:, s] = y0_vec

                # per-z impact columns
                # y0_vec_z[iz,t] = sum_{i_fix,a} IRFpol[i_fix,iz,:,t] * Dss[i_fix,iz,:]
                y0_z = np.sum(IRFpol * Dss[..., None], axis=(0,2))        # (Nz, T)
                F_by_O_z[O][:, :, s] = y0_z                               # (Nz,T) -> (Nz,T,s)

            # distribution term (optional)
            if (forward_endo is not None) and (forward_exo is not None):
                dD_endo_T = forward_endo(model.IRF['pols'], ss, par)  # list length T, each (Nfix,Nz,Na)
                dD_T      = forward_exo(dD_endo_T, ss, par)           # same

                for O in hh_outputs:
                    level = c_ss if O == 'C_hh' else a_ss if O == 'A_hh' else None
                    if level is None: 
                        continue
                    # aggregate rows τ>=1
                    for tau in range(1, T):
                        F_by_O_agg[O][tau, s] += _agg_level(level, dD_T[tau-1])
                    # per-z rows τ>=1
                    for tau in range(1, T):
                        add_z = _agg_level_by_z(level, dD_T[tau-1])   # (Nz,)
                        F_by_O_z[O][:, tau, s] += add_z

        # fake-news cumulation across columns → final Jacobians
        for O in hh_outputs:
            # aggregate
            F = F_by_O_agg[O]
            if (forward_endo is None) or (forward_exo is None):
                Jagg = F.copy()
            else:
                Jagg = np.zeros_like(F)
                Jagg[:, 0] = F[:, 0]
                for s in range(1, T):
                    Jagg[:, s]  = F[:, s]
                    Jagg[1:, s] += Jagg[:-1, s-1]
            jac_alt_agg[(OUTKEY[O], i_name)] = Jagg

            # per-z (do the SAME cumulation independently for each iz)
            Fz = F_by_O_z[O]                      # (Nz,T,T)
            Jz_stack = np.zeros_like(Fz)          # (Nz,T,T)
            if (forward_endo is None) or (forward_exo is None):
                Jz_stack[:] = Fz
            else:
                for iz in range(Nz):
                    Jz_stack[iz,:,0] = Fz[iz,:,0]
                    for s in range(1, T):
                        Jz_stack[iz,:,s]  = Fz[iz,:,s]
                        Jz_stack[iz,1:,s] += Jz_stack[iz,:-1,s-1]

            # store by individual z and as a stacked, z-sorted array
            z_order = np.argsort(par.z_grid)
            Jz_sorted = Jz_stack[z_order,:,:]
            for rank, iz in enumerate(z_order):
                jac_hh_by_z[((OUTKEY[O], i_name), iz)] = Jz_stack[iz].copy()
            jac_hh_by_z_stack[(OUTKEY[O], i_name)] = Jz_sorted

    # install on the model
    if not hasattr(model, 'jac_hh'): model.jac_hh = {}
    model.jac_hh.update(jac_alt_agg)
    model.jac_hh_by_z = jac_hh_by_z
    model.jac_hh_by_z_stack = jac_hh_by_z_stack
    # small meta bag
    if not hasattr(model, 'jac_hh_meta'): model.jac_hh_meta = {}
    model.jac_hh_meta['z_order'] = np.argsort(par.z_grid)

    return jac_alt_agg, jac_hh_by_z, jac_hh_by_z_stack


def _compute_policy_brackets_from_ss(ss, par):
    """Return (pol_indices, pol_weights) from the steady-state policy a_ss."""
    import numpy as np
    a_grid = np.asarray(par.a_grid, dtype=float)
    a_ss   = np.asarray(ss.a, dtype=float)   # (Nfix,Nz,Na)

    Nfix, Nz, Na = a_ss.shape
    i = np.empty((1, Nfix, Nz, Na), dtype=np.int64)
    w = np.empty((1, Nfix, Nz, Na), dtype=np.float64)

    for f in range(Nfix):
        for z in range(Nz):
            ap  = a_ss[f, z, :]
            j   = np.searchsorted(a_grid, ap, side='right') - 1
            j   = np.clip(j, 0, Na-2)
            aL  = a_grid[j]; aU = a_grid[j+1]
            lam = (ap - aL) / (aU - aL)
            i[0, f, z, :] = j.astype(np.int64)
            w[0, f, z, :] = lam
    return i, w


def make_forward_endo(shockname: str):
    """
    Endogenous step: convert policy IRFs Δa′ into ΔD^{endo}_t by
    Δλ = Δa′ / gap and moving *steady-state mass* between bracket nodes.
    """
    def forward_endo(IRF_pols: dict, ss, par):
        import numpy as np
        Dss = np.asarray(ss.D, dtype=float)       # (Nfix,Nz,Na)
        Nfix, Nz, Na = Dss.shape
        T = int(par.T)

        # ensure brackets on ss
        if not (hasattr(ss, 'pol_indices') and hasattr(ss, 'pol_weights')):
            i, w = _compute_policy_brackets_from_ss(ss, par)
            ss.pol_indices, ss.pol_weights = i, w
        i = np.asarray(ss.pol_indices, dtype=np.int64)   # (1,Nfix,Nz,Na)

        # local gaps from grid via lower index j
        a_grid  = np.asarray(par.a_grid, dtype=float)
        gap_vec = a_grid[1:] - a_grid[:-1]               # (Na-1,)
        g0      = gap_vec[i][0]                          # (Nfix,Nz,Na)
        # guard against zeros/NaNs
        bad = ~np.isfinite(g0) | (g0 <= 0)
        if np.any(bad):
            g0 = g0.copy()
            g0[bad] = 1.0

        # policy IRF Δa′
        Da = IRF_pols.get(('a', shockname))
        if Da is None:
            raise KeyError(f"IRF_pols missing ('a','{shockname}')")

        out = []
        for t in range(T):
            Da_t = np.asarray(Da[..., t], dtype=float)
            Da_t[~np.isfinite(Da_t)] = 0.0
            dlam = Da_t / g0                              # Δλ per source node

            dD = np.zeros_like(Dss)
            for f in range(Nfix):
                for z in range(Nz):
                    j   = i[0, f, z, :]
                    dl  = dlam[f, z, :]
                    mass= Dss[f, z, :]
                    np.add.at(dD[f, z, :], j,   -mass*dl)
                    np.add.at(dD[f, z, :], j+1,  mass*dl)
            out.append(dD)
        return out
    return forward_endo


def forward_exo(dD_endo_T, ss, par):
    """Exogenous step: push ΔD^{endo}_t through z-transition to get ΔD_t."""
    import numpy as np
    zP = np.asarray(ss.z_trans, dtype=float)
    out = []
    if zP.ndim == 2:
        for dD in dD_endo_T:
            out.append(np.einsum('ij,fja->fia', zP.T, dD))
    else:  # per-fix
        for dD in dD_endo_T:
            res = np.empty_like(dD)
            for f in range(dD.shape[0]):
                res[f] = zP[f].T @ dD[f]
            out.append(res)
    return out

import numpy as np

# --- your original homogeneous operator (unchanged) ---
def create_alt_M(M, E):
    T, m = M.shape
    assert T == m and E.shape == (T, T)
    M_beh = np.empty_like(M)
    for t in range(T):
        for s in range(T):
            summand = 0.0
            for tau in range(min(s, t) + 1):
                if tau > 0:
                    summand += (E[tau, s] - E[tau - 1, s]) * M[t - tau, s - tau]
                else:
                    summand += E[tau, s] * M[t - tau, s - tau]
            M_beh[t, s] = summand
    return M_beh

# --- EXACT sticky expectations (your function; unchanged) ---
def E_sticky_exp(theta, T=300):
    col = 1 - theta**(1 + np.arange(T))       # cumulative adoption by tau
    E = np.tile(col[:, np.newaxis], (1, T))
    E = np.triu(E, +1) + np.tril(np.ones((T, T)))  # ones at/after s
    return E

# --- NEW: per-z cumulative matrix from lambda_z (heterogeneity enters pre-s) ---
# Intuition: lam_z is the per-period adoption rate BEFORE s.
# Cumulative adoption up to lag tau: 1 - (1 - lam_z)^(tau+1).
# At/after s, agents fully see the realization (lower triangle = 1), just like sticky.
def E_from_lambda_cumulative(T, lam_z):
    """
    Per-z expectations matrix where pre-s entries are *cumulative sums* of geometric
    powers: ΔE[tau,s] = E[tau,s] - E[tau-1,s] = lam_z**(tau+1).
    Post-s (t >= s) entries are 1.

    Example for lam_z=0.2 and column s:
      [0.2, 0.2+0.2^2, 0.2+0.2^2+0.2^3, ...] above the diagonal,
      1 on and below the diagonal.
    """
    import numpy as np
    E = np.zeros((T, T))
    for s in range(T):
        # pre-s cumulative geometric sum
        for tau in range(s + 1):  # tau = s - t
            if lam_z == 1.0:
                E[tau, s] = tau + 1.0
            else:
                E[tau, s] = lam_z * (1.0 - lam_z**(tau + 1)) / (1.0 - lam_z)
        # on/after s: full information
        E[s:, s] = 1.0
    return E

def E_hybrid_cumulative(T, lam, theta_post=None):
    """
    Cumulative E for create_alt_M.

    Pre-s (t < s): ΔE[tau,s] = lam**(tau+1)  ⇒  E[tau,s] = sum_{k=1}^{tau+1} lam**k
    Post-s (t >= s):
        - if theta_post is None: full revelation ⇒ E[t,s] = 1
        - else: ΔE[s+k,s] = theta_post**k  ⇒  E[s+k,s] = sum_{j=0}^{k} theta_post**j
    """
    import numpy as np
    E = np.zeros((T, T))
    # pre-s cumulative
    for s in range(T):
        for tau in range(s + 1):
            if lam == 1.0:
                E[tau, s] = tau + 1.0
            else:
                E[tau, s] = lam * (1.0 - lam**(tau + 1)) / (1.0 - lam)

        # post-s cumulative
        if theta_post is None:
            E[s:, s] = 1.0
        else:
            run = 0.0
            for k in range(T - s):
                run += theta_post**k  # cumulative of post-shock arrivals
                E[s + k, s] = run
    return E


# --- weights & Markov helpers (unchanged logic) ---
def _z_weights_from_ss(model):
    D = model.ss.D                   # (Nfix, Nz, Na)
    w = np.sum(D, axis=(0, 2))       # (Nz,)
    total = np.sum(w)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid steady-state mass; cannot compute z-weights.")
    return w / total

def _find_markov_P(model, Nz):
    for name in ['Pi','P','pi_z','Pz','Pi_z','PiZ','P_Z']:
        if hasattr(model.par, name):
            P = np.asarray(getattr(model.par, name))
            if P.ndim == 2 and P.shape == (Nz, Nz):
                if np.allclose(P.sum(axis=1), 1.0, atol=1e-10) and np.all(P >= -1e-12):
                    return P
    return None

def _markov_weight_history(w0, P, T):
    w_hist = np.zeros((T, w0.size))
    w = w0.copy()
    for t in range(T):
        w_hist[t] = w
        w = w @ P
    return w_hist

# build an effective cumulative matrix by (possibly) time-varying z-weights
def _create_E_eff_markov_weighted(E_list, w0, P, T):
    Nz = len(E_list)
    for iz, Ez in enumerate(E_list):
        Ez = np.asarray(Ez)
        if Ez.shape != (T, T):
            raise ValueError(f"E_list[{iz}] has shape {Ez.shape}, expected {(T,T)}")
    w_hist = _markov_weight_history(w0, P, T) if P is not None else None
    E_eff = np.zeros((T, T))
    for s in range(T):
        for tau in range(s + 1):        # only tau<=s is used by the transform
            w = w_hist[s - tau] if w_hist is not None else w0
            E_eff[tau, s] = sum(w[iz] * E_list[iz][tau, s] for iz in range(Nz))
    return E_eff

# --- single-entry solver: homogeneous (pass E) OR per-z (pass dict) ---
def solve_alt_exp(model, E_or_Ebyz, sticky_vars=('Z','ra')):
    T  = int(model.par.T)
    Nz = int(model.par.Nz)
    model_sticky = model.copy()

    hetero = isinstance(E_or_Ebyz, dict)
    if hetero:
        w0 = _z_weights_from_ss(model_sticky)
        P  = _find_markov_P(model_sticky, Nz)

    n_updated = 0
    for out_name in ['C_hh','A_hh']:
        for in_name in sticky_vars:
            key = (out_name, in_name)
            if key not in model_sticky.jac_hh:
                continue
            M = model_sticky.jac_hh[key].copy()  # (T,T)

            if hetero and (in_name in E_or_Ebyz):
                E_list = E_or_Ebyz[in_name]      # list of Nz cumulative matrices
                if len(E_list) != Nz:
                    raise ValueError(f"{in_name}: need {Nz} matrices, got {len(E_list)}")
                E_eff = _create_E_eff_markov_weighted(E_list, w0, P, T)
                J = create_alt_M(M, E_eff)
                model_sticky.jac_hh[key] = J
                n_updated += 1
            elif not hetero:
                E = np.asarray(E_or_Ebyz)        # single cumulative matrix
                if E.shape != (T, T):
                    raise ValueError(f"Homogeneous E must be {(T,T)}, got {E.shape}")
                J = create_alt_M(M, E)
                model_sticky.jac_hh[key] = J
                n_updated += 1
            # else: hetero but no matrix for this input -> leave unchanged

    mode_txt = "heterogeneous (per-z cumulative)" if hetero else "homogeneous"
    print(f"Applied alternative expectations for {tuple(sticky_vars)} — {mode_txt}. "
          f"Updated {n_updated} HH Jacobians.")
    model_sticky.compute_jacs(skip_hh=True, skip_shocks=False)
    model_sticky.find_IRFs(shocks=['eps_i'], do_print=False)
    return model_sticky

import numpy as np

def E_hybrid_cumulative_single(T, lam, theta_post=None):
    """
    Cumulative E for create_alt_M.

    Pre-s (t < s): ΔE[tau,s] = lam**(tau+1)  ⇒  E[tau,s] = sum_{k=1}^{tau+1} lam**k
    Post-s (t >= s):
        - if theta_post is None: full revelation ⇒ E[t,s] = 1
        - else: ΔE[s+k,s] = theta_post**k  ⇒  E[s+k,s] = sum_{j=0}^{k} theta_post**j
    """
    E = np.zeros((T, T))
    # pre-s cumulative
    for s in range(T):
        for tau in range(s + 1):
            if lam == 1.0:
                E[tau, s] = tau + 1.0
            else:
                E[tau, s] = lam * (1.0 - lam**(tau + 1)) / (1.0 - lam)
        # post-s cumulative
        if theta_post is None:
            E[s:, s] = 1.0
        else:
            run = 0.0
            for k in range(T - s):
                run += (theta_post ** k)
                E[s + k, s] = run
    return E

def build_E_by_z_from_lambda(T, lambda_by_z, inputs=('Z','ra'), theta_post=None):
    """
    Turn a lambda vector (length Nz) into the per-z expectation matrices dict:
      {'Z':[E_z0,...], 'ra':[E_z0,...], ...}

    theta_post can be:
      - None (full info after s),
      - a scalar (same post-shock decay for all z),
      - a vector of length Nz (different post-shock decay per z).
    """
    lambda_by_z = np.asarray(lambda_by_z, dtype=float)
    Nz = lambda_by_z.size

    # broadcast theta_post
    if theta_post is None or np.isscalar(theta_post):
        theta_vec = np.full(Nz, 1.0 if theta_post is None else float(theta_post))
        theta_vec[theta_post is None] = 1.0  # not used, just explicit
    else:
        theta_vec = np.asarray(theta_post, dtype=float)
        if theta_vec.size != Nz:
            raise ValueError(f"theta_post length {theta_vec.size} must equal Nz={Nz}")

    # build one E per z
    E_list = [E_hybrid_cumulative_single(T, lam=float(lambda_by_z[iz]),
                                         theta_post=None if theta_post is None else float(theta_vec[iz]))
              for iz in range(Nz)]

    # same set for every input you want to treat as expectation-driven
    return {i: [E.copy() for E in E_list] for i in inputs}

    import numpy as np

# --- your original homogeneous operator (unchanged) ---
def create_alt_M(M, E):
    T, m = M.shape
    assert T == m and E.shape == (T, T)
    M_beh = np.empty_like(M)
    for t in range(T):
        for s in range(T):
            summand = 0.0
            for tau in range(min(s, t) + 1):
                if tau > 0:
                    summand += (E[tau, s] - E[tau - 1, s]) * M[t - tau, s - tau]
                else:
                    summand += E[tau, s] * M[t - tau, s - tau]
            M_beh[t, s] = summand
    return M_beh

# --- EXACT sticky expectations (your function; unchanged) ---
def E_sticky_exp(theta, T=300):
    col = 1 - theta**(1 + np.arange(T))       # cumulative adoption by tau
    E = np.tile(col[:, np.newaxis], (1, T))
    E = np.triu(E, +1) + np.tril(np.ones((T, T)))  # ones at/after s
    return E

# --- NEW: per-z cumulative matrix from lambda_z (heterogeneity enters pre-s) ---
# Intuition: lam_z is the per-period adoption rate BEFORE s.
# Cumulative adoption up to lag tau: 1 - (1 - lam_z)^(tau+1).
# At/after s, agents fully see the realization (lower triangle = 1), just like sticky.
def E_from_lambda_cumulative(T, lam_z):
    """
    Per-z expectations matrix where pre-s entries are *cumulative sums* of geometric
    powers: ΔE[tau,s] = E[tau,s] - E[tau-1,s] = lam_z**(tau+1).
    Post-s (t >= s) entries are 1.

    Example for lam_z=0.2 and column s:
      [0.2, 0.2+0.2^2, 0.2+0.2^2+0.2^3, ...] above the diagonal,
      1 on and below the diagonal.
    """
    import numpy as np
    E = np.zeros((T, T))
    for s in range(T):
        # pre-s cumulative geometric sum
        for tau in range(s + 1):  # tau = s - t
            if lam_z == 1.0:
                E[tau, s] = tau + 1.0
            else:
                E[tau, s] = lam_z * (1.0 - lam_z**(tau + 1)) / (1.0 - lam_z)
        # on/after s: full information
        E[s:, s] = 1.0
    return E

def E_hybrid_cumulative(T, lam, theta_post=None):
    """
    Cumulative E for create_alt_M.

    Pre-s (t < s): ΔE[tau,s] = lam**(tau+1)  ⇒  E[tau,s] = sum_{k=1}^{tau+1} lam**k
    Post-s (t >= s):
        - if theta_post is None: full revelation ⇒ E[t,s] = 1
        - else: ΔE[s+k,s] = theta_post**k  ⇒  E[s+k,s] = sum_{j=0}^{k} theta_post**j
    """
    import numpy as np
    E = np.zeros((T, T))
    # pre-s cumulative
    for s in range(T):
        for tau in range(s + 1):
            if lam == 1.0:
                E[tau, s] = tau + 1.0
            else:
                E[tau, s] = lam * (1.0 - lam**(tau + 1)) / (1.0 - lam)

        # post-s cumulative
        if theta_post is None:
            E[s:, s] = 1.0
        else:
            run = 0.0
            for k in range(T - s):
                run += theta_post**k  # cumulative of post-shock arrivals
                E[s + k, s] = run
    return E


# --- weights & Markov helpers (unchanged logic) ---
def _z_weights_from_ss(model):
    D = model.ss.D                   # (Nfix, Nz, Na)
    w = np.sum(D, axis=(0, 2))       # (Nz,)
    total = np.sum(w)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid steady-state mass; cannot compute z-weights.")
    return w / total

def _find_markov_P(model, Nz):
    for name in ['Pi','P','pi_z','Pz','Pi_z','PiZ','P_Z']:
        if hasattr(model.par, name):
            P = np.asarray(getattr(model.par, name))
            if P.ndim == 2 and P.shape == (Nz, Nz):
                if np.allclose(P.sum(axis=1), 1.0, atol=1e-10) and np.all(P >= -1e-12):
                    return P
    return None

def _markov_weight_history(w0, P, T):
    w_hist = np.zeros((T, w0.size))
    w = w0.copy()
    for t in range(T):
        w_hist[t] = w
        w = w @ P
    return w_hist

# build an effective cumulative matrix by (possibly) time-varying z-weights
def _create_E_eff_markov_weighted(E_list, w0, P, T):
    Nz = len(E_list)
    for iz, Ez in enumerate(E_list):
        Ez = np.asarray(Ez)
        if Ez.shape != (T, T):
            raise ValueError(f"E_list[{iz}] has shape {Ez.shape}, expected {(T,T)}")
    w_hist = _markov_weight_history(w0, P, T) if P is not None else None
    E_eff = np.zeros((T, T))
    for s in range(T):
        for tau in range(s + 1):        # only tau<=s is used by the transform
            w = w_hist[s - tau] if w_hist is not None else w0
            E_eff[tau, s] = sum(w[iz] * E_list[iz][tau, s] for iz in range(Nz))
    return E_eff

# --- single-entry solver: homogeneous (pass E) OR per-z (pass dict) ---
def solve_alt_exp(model, E_or_Ebyz, sticky_vars=('Z','ra')):
    T  = int(model.par.T)
    Nz = int(model.par.Nz)
    model_sticky = model.copy()

    hetero = isinstance(E_or_Ebyz, dict)
    if hetero:
        w0 = _z_weights_from_ss(model_sticky)
        P  = _find_markov_P(model_sticky, Nz)

    n_updated = 0
    for out_name in ['C_hh','A_hh']:
        for in_name in sticky_vars:
            key = (out_name, in_name)
            if key not in model_sticky.jac_hh:
                continue
            M = model_sticky.jac_hh[key].copy()  # (T,T)

            if hetero and (in_name in E_or_Ebyz):
                E_list = E_or_Ebyz[in_name]      # list of Nz cumulative matrices
                if len(E_list) != Nz:
                    raise ValueError(f"{in_name}: need {Nz} matrices, got {len(E_list)}")
                E_eff = _create_E_eff_markov_weighted(E_list, w0, P, T)
                J = create_alt_M(M, E_eff)
                model_sticky.jac_hh[key] = J
                n_updated += 1
            elif not hetero:
                E = np.asarray(E_or_Ebyz)        # single cumulative matrix
                if E.shape != (T, T):
                    raise ValueError(f"Homogeneous E must be {(T,T)}, got {E.shape}")
                J = create_alt_M(M, E)
                model_sticky.jac_hh[key] = J
                n_updated += 1
            # else: hetero but no matrix for this input -> leave unchanged

    mode_txt = "heterogeneous (per-z cumulative)" if hetero else "homogeneous"
    print(f"Applied alternative expectations for {tuple(sticky_vars)} — {mode_txt}. "
          f"Updated {n_updated} HH Jacobians.")
    model_sticky.compute_jacs(skip_hh=True, skip_shocks=False)
    model_sticky.find_IRFs(shocks=['eps_i'], do_print=False)
    return model_sticky


import numpy as np

def E_hybrid_cumulative_single(T, lam, theta_post=None):
    """
    Cumulative E for create_alt_M.

    Pre-s (t < s): ΔE[tau,s] = lam**(tau+1)  ⇒  E[tau,s] = sum_{k=1}^{tau+1} lam**k
    Post-s (t >= s):
        - if theta_post is None: full revelation ⇒ E[t,s] = 1
        - else: ΔE[s+k,s] = theta_post**k  ⇒  E[s+k,s] = sum_{j=0}^{k} theta_post**j
    """
    E = np.zeros((T, T))
    # pre-s cumulative
    for s in range(T):
        for tau in range(s + 1):
            if lam == 1.0:
                E[tau, s] = tau + 1.0
            else:
                E[tau, s] = lam * (1.0 - lam**(tau + 1)) / (1.0 - lam)
        # post-s cumulative
        if theta_post is None:
            E[s:, s] = 1.0
        else:
            run = 0.0
            for k in range(T - s):
                run += (theta_post ** k)
                E[s + k, s] = run
    return E

def build_E_by_z_from_lambda(T, lambda_by_z, inputs=('Z','ra'), theta_post=None):
    """
    Turn a lambda vector (length Nz) into the per-z expectation matrices dict:
      {'Z':[E_z0,...], 'ra':[E_z0,...], ...}

    theta_post can be:
      - None (full info after s),
      - a scalar (same post-shock decay for all z),
      - a vector of length Nz (different post-shock decay per z).
    """
    lambda_by_z = np.asarray(lambda_by_z, dtype=float)
    Nz = lambda_by_z.size

    # broadcast theta_post
    if theta_post is None or np.isscalar(theta_post):
        theta_vec = np.full(Nz, 1.0 if theta_post is None else float(theta_post))
        theta_vec[theta_post is None] = 1.0  # not used, just explicit
    else:
        theta_vec = np.asarray(theta_post, dtype=float)
        if theta_vec.size != Nz:
            raise ValueError(f"theta_post length {theta_vec.size} must equal Nz={Nz}")

    # build one E per z
    E_list = [E_hybrid_cumulative_single(T, lam=float(lambda_by_z[iz]),
                                         theta_post=None if theta_post is None else float(theta_vec[iz]))
              for iz in range(Nz)]

    # same set for every input you want to treat as expectation-driven
    return {i: [E.copy() for E in E_list] for i in inputs}

import numpy as np

def E_per_z_sticky_style(T, lam):
    """
    Pre-s: ΔE[tau,s] = lam**(tau+1)  (so E[tau,s] = sum_{k=1}^{tau+1} lam**k)
    Post-s: E[t,s] = 1  (full information at/after realization, like E_sticky_exp)
    """
    E = np.zeros((T, T))
    if lam == 1.0:
        pre = np.arange(1, T+1, dtype=float)
    else:
        tau = np.arange(T, dtype=float)
        pre = lam * (1.0 - lam**(tau + 1)) / (1.0 - lam)  # cumulative geometric sum
    for s in range(T):
        E[:s+1, s] = pre[:s+1]  # cumulative pre-s
        E[s:,   s] = 1.0        # full info post-s (match E_sticky)
    return E

def build_E_by_z_from_lambda_sticky_post(T, lambda_by_z, inputs=('Z','ra')):
    """
    Build per-z expectation matrices with sticky-style post-s (ones).
    """
    mats = [E_per_z_sticky_style(T, float(lam)) for lam in lambda_by_z]
    return {i: [M.copy() for M in mats] for i in inputs}


import numpy as np

# --- core: sanitize ONE cumulative E (T,T) ---
def _sanitize_cumulative_E_single(E):
    """
    Input E: cumulative expectations matrix (T,T) that may violate [0,1] / monotonicity.
    Output:   E_san of the same shape with:
              - for each column s: E_san[t,s] is nondecreasing in t, in [0,1], with E_san[s,s]=1
              - pre-entries t<s kept (clipped to [0,1])
    """
    E = np.asarray(E, dtype=float)
    T = E.shape[0]
    E_san = np.zeros_like(E)

    for s in range(T):
        col = E[:, s].copy()

        # 1) handle pre-announcement part (t < s)
        pre = np.clip(col[:s], 0.0, 1.0)  # keep and clip to [0,1]
        E_san[:s, s] = pre

        # 2) ensure diagonal is 1 (fully known at t=s)
        base = max(0.0, min(1.0, col[s])) if s < T else 1.0
        if base == 0.0:
            base = 1.0
        E_san[s, s] = 1.0

        # 3) incremental "news weights" for t >= s:
        #    raw increments ΔE_raw[t] = E[t,s] - E[t-1,s]; then clip negatives; renormalize to sum=1
        #    finally re-cumulate to enforce bounded monotone cumulative ≤ 1
        # construct raw increments on t>=s
        inc = np.zeros(T - s)
        inc[0] = 1.0                             # force E_san[s,s] = 1
        if s + 1 < T:
            raw = np.diff(col[s:s+1+(T-s-1)], prepend=col[s])  # zeros, but explicit
            # Better: compute from original (we already set inc[0]=1). For t>s:
            raw = col[s:(T)] - np.r_[col[s], col[s:(T-1)]]
            raw[0] = 1.0  # override to ensure diag=1
            # clip negatives
            raw = np.maximum(raw, 0.0)
            # if sum(raw) == 0, fall back to a point mass at t=s already (we set raw[0]=1)
            if raw.sum() == 0.0:
                inc[:] = 0.0
                inc[0] = 1.0
            else:
                # renormalize to sum 1
                inc = raw / raw.sum()
        # re-cumulate to get a proper bounded cumulative on t>=s
        post_cum = np.cumsum(inc)
        post_cum = np.clip(post_cum, 0.0, 1.0)

        # write back (ensuring t=s matches 1.0)
        E_san[s:, s] = post_cum

    # final clip for numerical safety
    np.clip(E_san, 0.0, 1.0, out=E_san)
    return E_san

# --- helpers to accept many shapes for per-z collections ---
def _ensure_TxT(E, T):
    E = np.asarray(E)
    if E.ndim != 2: raise ValueError(f"E must be 2D, got {E.shape}")
    if E.shape[0] < T or E.shape[1] < T: raise ValueError(f"E too small: {E.shape} < ({T},{T})")
    if E.shape != (T, T): E = E[:T, :T]
    return E

def _iter_perz(perz, Nz, T):
    """Yield per-z matrices from many input formats, returning a list of (T,T)."""
    out = []
    if isinstance(perz, dict):
        if len(perz) != Nz:
            raise ValueError(f"per-z dict has {len(perz)} entries but Nz={Nz}.")
        for k in perz.keys():
            out.append(_ensure_TxT(perz[k], T))
        return out
    arr = np.asarray(perz, dtype=object)
    if arr.ndim == 3:
        if arr.shape[0] == Nz:
            return [_ensure_TxT(arr[z], T) for z in range(Nz)]
        if arr.shape[-1] == Nz:
            return [_ensure_TxT(arr[:, :, z], T) for z in range(Nz)]
        raise ValueError(f"3D per-z has incompatible shape {arr.shape}.")
    if arr.ndim == 2:
        E = _ensure_TxT(arr, T)
        return [E for _ in range(Nz)]
    if arr.ndim == 1:
        # If you passed lambdas (length Nz), build bounded cumulative form <= 1
        if arr.size != Nz:
            raise ValueError(f"lambda vector length {arr.size} != Nz={Nz}.")
        mats = []
        for lam in arr:
            lam = float(lam)
            # bounded sticky-like cumulative for t>=s:
            # pre-announcement set to 0 here; change if you want leakage
            E = np.zeros((T, T))
            for s in range(T):
                E[s, s] = 1.0
                for t in range(s+1, T):
                    h = t - s
                    E[t, s] = 1.0 - (1.0 - lam) ** (h+1)
            mats.append(E)
        return mats
    raise ValueError(f"Unsupported per-z spec with shape {arr.shape}.")

# --- top-level: sanitize E_by_z in any of the forms we used earlier ---
def sanitize_E_by_z(E_by_z, T, Nz):
    """
    Accepts:
      - dict {'Z': perzZ, 'ra': perzR}, where perzZ/perzR can be dict/list/array of (T_full,T_full) or lambdas
      - OR a single per-z collection (applied to both channels upstream if you reuse)
    Returns a structure with the same top-level form, but with each per-z matrix sanitized to be a proper cumulative CDF.
    """
    # If it's the two-channel dict, sanitize each channel separately
    if isinstance(E_by_z, dict) and set(E_by_z.keys()) >= {'Z', 'ra'}:
        out = {}
        for key in ['Z', 'ra']:
            perz = _iter_perz(E_by_z[key], Nz, T)  # list of (T,T)
            out[key] = [ _sanitize_cumulative_E_single(Ez) for Ez in perz ]
        return out
    # Otherwise treat as a single per-z collection
    perz = _iter_perz(E_by_z, Nz, T)
    return [ _sanitize_cumulative_E_single(Ez) for Ez in perz ]


import numpy as np
from copy import deepcopy
import time

def _calc_jac_hh_direct_with_z(model, jac_hh, jac_hh_z, inputname, dx=1e-4, do_print=False, s_list=None):
    """
    Direct household Jacobian for a single input, extended with per-z Jacobians.
    
    Fills:
        jac_hh[(OUTPUT_hh, inputname)]   -> (T, T)
        jac_hh_z[(OUTPUT_hh, inputname)] -> (Nz, T, T)
    """

    par  = model.par
    ss   = model.ss
    path = model.path

    if s_list is None:
        s_list = list(range(par.T))

    if do_print:
        t0 = time.time()
        print(f'finding Jacobian wrt. {inputname:15s}:', end='')

    # a. allocate
    for outputname in model.outputs_hh:
        key = (f'{outputname.upper()}_hh', inputname)
        jac_hh[key]   = np.zeros((par.T, par.T))
        jac_hh_z[key] = np.zeros((par.Nz, par.T, par.T))

    # b. solve with shock in last period
    model._set_inputs_hh_all_ss()

    if inputname != 'ghost':
        shockarray = getattr(model.path, inputname)
        shockarray[-1, 0] += dx

    model.solve_hh_path()

    # c. baseline shocked path storage
    path_shock = deepcopy(model.path)

    # Precompute steady-state aggregates by z for each output
    base_agg = {}
    base_z   = {}
    for outputname in model.outputs_hh:
        pol_ss = ss.__dict__[outputname]  # shape (Nfix, Nz, Na)
        D_ss   = ss.D                     # shape (Nfix, Nz, Na)

        base_agg[outputname] = np.sum(pol_ss * D_ss)

        base_z_out = np.zeros(par.Nz)
        for i_z in range(par.Nz):
            base_z_out[i_z] = np.sum(pol_ss[:, i_z, :] * D_ss[:, i_z, :])
        base_z[outputname] = base_z_out

    # d. loop over s (time of shock)
    for s in s_list:

        if do_print:
            print(f' {s}', end='')

        # i. before shock: time to shock matters
        path.z_trans[:s+1]     = path_shock.z_trans[par.T-(s+1):]
        path.pol_indices[:s+1] = path_shock.pol_indices[par.T-(s+1):]
        path.pol_weights[:s+1] = path_shock.pol_weights[par.T-(s+1):]

        for outputname in model.outputs_hh:
            path.__dict__[outputname][:s+1] = path_shock.__dict__[outputname][par.T-(s+1):]

        # ii. after shock: solution is steady state
        path.pol_indices[s+1:] = ss.pol_indices
        path.pol_weights[s+1:] = ss.pol_weights
        path.z_trans[s+1:]     = ss.z_trans

        for outputname in model.outputs_hh:
            path.__dict__[outputname][s+1:] = ss.__dict__[outputname]

        # iii. simulate full path
        model.simulate_hh_path()

        # iv. compute Jacobians (aggregate + per z)
        for outputname in model.outputs_hh:

            key     = (f'{outputname.upper()}_hh', inputname)
            J_agg   = jac_hh[key]          # (T, T)
            J_z_out = jac_hh_z[key]        # (Nz, T, T)

            pol_ss = ss.__dict__[outputname]   # (Nfix, Nz, Na)
            D_ss   = ss.D                      # (Nfix, Nz, Na)

            for t in range(par.T):

                pol_t = path.__dict__[outputname][t]  # (Nfix, Nz, Na)
                D_t   = path.D[t]                     # (Nfix, Nz, Na)

                # aggregate
                shock_agg = np.sum(pol_t * D_t)
                J_agg[t, s] = (shock_agg - base_agg[outputname]) / dx

                # per-z
                for i_z in range(par.Nz):
                    shock_z = np.sum(pol_t[:, i_z, :] * D_t[:, i_z, :])
                    J_z_out[i_z, t, s] = (shock_z - base_z[outputname][i_z]) / dx

    if do_print:
        print(' done')


def compute_jac_hh_z(model, dx=1e-4, inputs_hh_all=None, do_print=False, s_list=None):
    """
    Compute household Jacobians per productivity state z using the *direct* method.

    After calling this, you get:
        model.jac_hh_z[(OUTPUT_hh, INPUT)]  -> array with shape (Nz, T, T)

    Returns:
        jac_hh_z (same dict as model.jac_hh_z)
    """

    if inputs_hh_all is None:
        inputs_hh_all = model.inputs_hh_all

    par = model.par

    jac_hh        = {}
    jac_hh_ghost  = {}
    jac_hh_z      = {}
    jac_hh_z_ghost = {}

    path_original = model.path
    model.path    = deepcopy(model.path)

    t0 = time.time()
    if do_print:
        print('computing per-z household Jacobians (direct method)...')

    # i. ghost run
    _calc_jac_hh_direct_with_z(model, jac_hh_ghost, jac_hh_z_ghost,
                               'ghost', dx=dx, do_print=do_print, s_list=s_list)

    # ii. each actual input
    for inputname in inputs_hh_all:
        _calc_jac_hh_direct_with_z(model, jac_hh, jac_hh_z,
                                   inputname, dx=dx, do_print=do_print, s_list=s_list)

    # iii. ghost-corrected result
    jac_hh_z_final = {}

    for outputname in model.outputs_hh:
        for inputname in model.inputs_hh_all:

            key       = (f'{outputname.upper()}_hh', inputname)
            key_ghost = (f'{outputname.upper()}_hh', 'ghost')

            if inputname not in inputs_hh_all:
                jac_hh_z_final[key] = np.zeros((par.Nz, par.T, par.T))
            else:
                jac_hh_z_final[key] = jac_hh_z[key] - jac_hh_z_ghost[key_ghost]

    model.jac_hh_z = jac_hh_z_final

    if do_print:
        elapsed = time.time() - t0
        print(f'per-z household Jacobians computed in {elapsed:5.2f} seconds')

    # reset path
    model.path = path_original

    return jac_hh_z_final

import numpy as np
from copy import deepcopy
import time

def _calc_jac_hh_direct_with_z(model, jac_hh, jac_hh_z, inputname, dx=1e-4, do_print=False, s_list=None):
    """
    Direct household Jacobian for a single input, extended with per-z Jacobians.

    Fills:
        jac_hh[(OUTPUT_hh, inputname)]   -> (T, T)
        jac_hh_z[(OUTPUT_hh, inputname)] -> (Nz, T, T)
    """

    par  = model.par
    ss   = model.ss
    path = model.path

    if s_list is None:
        s_list = list(range(par.T))

    t0 = time.time()
    if do_print:
        print(f'finding Jacobian wrt. {inputname:15s}:', end='')

    # a. allocate aggregate + per-z
    for outputname in model.outputs_hh:
        key = (f'{outputname.upper()}_hh', inputname)
        jac_hh[key]   = np.zeros((par.T, par.T))
        jac_hh_z[key] = np.zeros((par.Nz, par.T, par.T))

    # b. set baseline inputs (steady state)
    model._set_inputs_hh_all_ss()

    # if we are not in the ghost run, apply a tiny shock in last period
    if inputname != 'ghost':
        shockarray = getattr(model.path, inputname)
        shockarray[-1, 0] += dx

    # solve household problem along the shocked path
    model.solve_hh_path()

    # store full shocked path
    path_shock = deepcopy(model.path)

    # steady state policies & distribution
    base_agg = {}
    base_z   = {}
    for outputname in model.outputs_hh:
        pol_ss = ss.__dict__[outputname]   # (Nfix, Nz, Na)
        D_ss   = ss.D                      # (Nfix, Nz, Na)

        base_agg[outputname] = np.sum(pol_ss * D_ss)

        bz = np.zeros(par.Nz)
        for i_z in range(par.Nz):
            bz[i_z] = np.sum(pol_ss[:, i_z, :] * D_ss[:, i_z, :])
        base_z[outputname] = bz

    # c. loop over shock timing s
    for s in s_list:

        if do_print:
            print(f' {s}', end='')

        # i. before shock: time to shock matters
        path.z_trans[:s+1]     = path_shock.z_trans[par.T-(s+1):]
        path.pol_indices[:s+1] = path_shock.pol_indices[par.T-(s+1):]
        path.pol_weights[:s+1] = path_shock.pol_weights[par.T-(s+1):]

        for outputname in model.outputs_hh:
            path.__dict__[outputname][:s+1] = path_shock.__dict__[outputname][par.T-(s+1):]

        # ii. after shock: back to steady state
        path.pol_indices[s+1:] = ss.pol_indices
        path.pol_weights[s+1:] = ss.pol_weights
        path.z_trans[s+1:]     = ss.z_trans

        for outputname in model.outputs_hh:
            path.__dict__[outputname][s+1:] = ss.__dict__[outputname]

        # iii. simulate the full distribution path
        model.simulate_hh_path()

        # iv. compute Jacobians
        for outputname in model.outputs_hh:

            key     = (f'{outputname.upper()}_hh', inputname)
            J_agg   = jac_hh[key]        # (T, T)
            J_z_out = jac_hh_z[key]      # (Nz, T, T)

            for t in range(par.T):

                pol_t = path.__dict__[outputname][t]  # (Nfix, Nz, Na)
                D_t   = path.D[t]                     # (Nfix, Nz, Na)

                # aggregate value at t
                shock_agg = np.sum(pol_t * D_t)
                J_agg[t, s] = (shock_agg - base_agg[outputname]) / dx

                # per-z values at t
                for i_z in range(par.Nz):
                    shock_z = np.sum(pol_t[:, i_z, :] * D_t[:, i_z, :])
                    J_z_out[i_z, t, s] = (shock_z - base_z[outputname][i_z]) / dx

    if do_print:
        print(' done')


def compute_jac_hh_z(model, dx=1e-4, inputs_hh_all=None, do_print=False, s_list=None, overwrite_model_jac_hh=False):
    """
    Compute household Jacobians per productivity state z using the *direct* method.

    After calling this, you get:
        model.jac_hh_direct[(OUTPUT_hh, INPUT)]  -> (T, T)   [direct method]
        model.jac_hh_z[(OUTPUT_hh, INPUT)]       -> (Nz, T, T)

    If overwrite_model_jac_hh=True, model.jac_hh will also be replaced
    by the direct aggregate.
    """

    if inputs_hh_all is None:
        inputs_hh_all = model.inputs_hh_all

    par = model.par

    jac_hh         = {}
    jac_hh_ghost   = {}
    jac_hh_z       = {}
    jac_hh_z_ghost = {}

    path_original = model.path
    model.path    = deepcopy(model.path)

    t0 = time.time()
    if do_print:
        print('computing per-z household Jacobians (direct method)...')

    # i. ghost run
    _calc_jac_hh_direct_with_z(model, jac_hh_ghost, jac_hh_z_ghost,
                               'ghost', dx=dx, do_print=do_print, s_list=s_list)

    # ii. actual inputs
    for inputname in inputs_hh_all:
        _calc_jac_hh_direct_with_z(model, jac_hh, jac_hh_z,
                                   inputname, dx=dx, do_print=do_print, s_list=s_list)

    # iii. ghost-corrected aggregates and per-z
    jac_hh_direct = {}
    jac_hh_z_final = {}

    for outputname in model.outputs_hh:
        for inputname in model.inputs_hh_all:

            key       = (f'{outputname.upper()}_hh', inputname)
            key_ghost = (f'{outputname.upper()}_hh', 'ghost')

            if inputname not in inputs_hh_all:
                jac_hh_direct[key]   = np.zeros((par.T, par.T))
                jac_hh_z_final[key]  = np.zeros((par.Nz, par.T, par.T))
            else:
                jac_hh_direct[key]  = jac_hh[key]   - jac_hh_ghost[key_ghost]
                jac_hh_z_final[key] = jac_hh_z[key] - jac_hh_z_ghost[key_ghost]

    model.jac_hh_direct = jac_hh_direct   # direct aggregate
    model.jac_hh_z      = jac_hh_z_final  # per-z

    if overwrite_model_jac_hh:
        model.jac_hh = jac_hh_direct

    if do_print:
        elapsed = time.time() - t0
        print(f'per-z household Jacobians computed in {elapsed:5.2f} seconds')

    # reset path
    model.path = path_original

    return jac_hh_z_final


def replace_aggregate_with_sum_over_z(model, source='jac_hh_z', target='jac_hh'):
    """
    Replace model.<target> with the aggregate obtained by summing over z
    from model.<source>.

    Parameters
    ----------
    model : GEModel (or similar)
        The model that holds jac_hh and jac_hh_z.
    source : str
        Attribute name containing per-z Jacobians (default: 'jac_hh_z').
    target : str
        Attribute name where the aggregate should be stored (default: 'jac_hh').

    After running:
        model.jac_hh[(OUTPUT_hh, INPUT)]  <-- sum_z model.jac_hh_z[(OUTPUT, INPUT)]
    """

    jac_z = getattr(model, source)

    jac_new = {}

    for key, Jz in jac_z.items():
        # Jz has shape (Nz, T, T)
        Jagg = Jz.sum(axis=0)  # (T, T)
        jac_new[key] = Jagg

    setattr(model, target, jac_new)

    return jac_new


import numpy as np
import numba as nb
from copy import deepcopy
import collections.abc as cabc


# ------------------------------------------------------------
# 1. Numba-accelerated expectation-transformation operator
#    (same structure as you already use)
# ------------------------------------------------------------

@nb.njit
def create_alt_M(M, E):
    """Transform FIRE Jacobian M using expectation matrix E."""
    T, m = M.shape
    assert T == m
    assert E.shape == (T, T)
    
    M_beh = np.empty_like(M)
    for t in range(T):
        for s in range(T):
            summand = 0.0
            upper = min(s, t)
            for tau in range(upper + 1):
                if tau > 0:
                    summand += (E[tau, s] - E[tau-1, s]) * M[t - tau, s - tau]
                else:
                    summand += E[tau, s] * M[t - tau, s - tau]
            M_beh[t, s] = summand
    return M_beh


# ------------------------------------------------------------
# 2. Helper: get E_z for one productivity state, given a container
#    that holds all z-states for ONE input (e.g. for 'Z' or 'ra').
# ------------------------------------------------------------

def _get_E_for_z(E_container, i_z):
    """
    Flexible accessor for E_z given i_z, where E_container holds the
    per-z expectation matrices for ONE input (e.g. for 'Z' or 'ra').

    Supports:
    - 2D array: same E for all z
    - 3D array: E_container[i_z, :, :]
    - list/tuple: E_container[i_z]
    - dict:
        * if int keys 0..Nz-1: use E_container[i_z]
        * else: use insertion-order and take the i_z'th key
    """

    # numpy array
    if isinstance(E_container, np.ndarray):
        if E_container.ndim == 2:
            # same E for all z
            return E_container
        elif E_container.ndim == 3:
            return E_container[i_z, :, :]
        else:
            raise ValueError("E_container ndarray must be 2D or 3D.")

    # list or tuple
    if isinstance(E_container, (list, tuple)):
        return E_container[i_z]

    # mapping / dict-like
    if isinstance(E_container, cabc.Mapping):
        # if integer key exists, use it directly
        if i_z in E_container:
            return E_container[i_z]

        # otherwise, fall back to 'i_z-th element in insertion order'
        keys = list(E_container.keys())
        if i_z >= len(keys):
            raise IndexError(
                f"i_z={i_z} but E_container has only {len(keys)} keys."
            )
        k = keys[i_z]
        return E_container[k]

    raise TypeError(
        "E_container must be ndarray, list, tuple or dict-like. "
        f"Got type {type(E_container)}."
    )


# ------------------------------------------------------------
# 3. Main helper: apply heterogeneous expectations by z
#    to per-z household Jacobians, and (optionally) recompute
#    aggregate jac_hh as sum_z of the transformed per-z jacobians.
# ------------------------------------------------------------

def apply_heterogeneous_expectations_by_z(
    model_fire,
    model_beliefs,
    E_by_z_sane,
    sticky_vars=('Z', 'ra', 'chi'),   # <-- include chi
    outputs=('C_hh', 'A_hh'),
    overwrite_aggregate=True
):
    """
    Take FIRE per-z Jacobians from model_fire.jac_hh_z
    and transform them with heterogeneous expectations E_by_z_sane,
    writing the result into model_beliefs.jac_hh_z.

    Expected structure of E_by_z_sane:
        E_by_z_sane[input_name][i_z] -> (T,T) expectation matrix

    Example:
        E_by_z_sane['Z'][0].shape   == (T, T)
        E_by_z_sane['ra'][3].shape  == (T, T)
        E_by_z_sane['chi'][2].shape == (T, T)

    Parameters
    ----------
    model_fire : model with FIRE (rational) expectations
        Must already have model_fire.jac_hh_z[(OUTPUT_hh, INPUT)] with shape (Nz, T, T).
    model_beliefs : model that will hold the heterogeneous-beliefs Jacobians
        Typically a deepcopy(model_fire) or model_fire itself.
    E_by_z_sane : dict
        Outer keys: input names, e.g. 'Z', 'ra', 'chi'.
        Values: containers (list/tuple/dict/ndarray) indexed by z-state.
    sticky_vars : iterable of str
        Inputs for which expectations are deformed (e.g. 'Z', 'ra', 'chi').
    outputs : iterable of str
        Household outputs to transform (e.g. 'C_hh', 'A_hh').
    overwrite_aggregate : bool
        If True, sets model_beliefs.jac_hh to sum_z model_beliefs.jac_hh_z.

    Returns
    -------
    model_beliefs : same object, mutated in-place.
    """

    # 1. Check that FIRE per-z Jacobians exist
    if not hasattr(model_fire, 'jac_hh_z'):
        raise AttributeError(
            "model_fire.jac_hh_z not found. "
            "You need to run compute_jac_hh_z(model_fire, ...) first."
        )

    # 2. Start from FIRE per-z Jacobians
    #    (non-sticky inputs stay as FIRE)
    model_beliefs.jac_hh_z = {
        key: val.copy() for key, val in model_fire.jac_hh_z.items()
    }

    # 3. Apply heterogeneous expectations per z for sticky vars
    for o in outputs:
        for i in sticky_vars:

            key = (o, i)
            if key not in model_fire.jac_hh_z:
                continue  # skip if combination does not exist

            if i not in E_by_z_sane:
                raise KeyError(
                    f"E_by_z_sane has no entry for input '{i}'. "
                    f"Available keys: {list(E_by_z_sane.keys())}"
                )

            # container with the per-z expectations for this INPUT (e.g. 'Z', 'ra', 'chi')
            E_container = E_by_z_sane[i]

            J_fire_z = model_fire.jac_hh_z[key]  # shape (Nz, T, T)
            Nz, T, _ = J_fire_z.shape

            J_beh_z = np.empty_like(J_fire_z)

            for i_z in range(Nz):
                # Get expectations matrix for this z-type and this input
                E_z = _get_E_for_z(E_container, i_z)
                if E_z.shape != (T, T):
                    raise ValueError(
                        f"E_z for input '{i}', z-state {i_z} has shape {E_z.shape}, "
                        f"expected {(T, T)}."
                    )
                # Transform FIRE Jacobian for this z-type
                J_beh_z[i_z, :, :] = create_alt_M(J_fire_z[i_z].copy(), E_z)

            model_beliefs.jac_hh_z[key] = J_beh_z

    # 4. Optionally recompute aggregate jac_hh as sum over z
    if overwrite_aggregate:
        jac_new = {}
        for key, Jz in model_beliefs.jac_hh_z.items():
            # sum over z (axis 0) -> (T, T)
            jac_new[key] = Jz.sum(axis=0)
        model_beliefs.jac_hh = jac_new

    return model_beliefs


import numpy as np
from copy import deepcopy

def make_identity_E_by_z_like(E_by_z_sane):
    """
    Build an E_by_z_identity object that has the SAME structure as E_by_z_sane,
    but with FIRE expectations (identity / perfect foresight) for each z and input.
    """
    E_id = {}

    for inp, container in E_by_z_sane.items():
        # Example: container is something like {0: E0, 1: E1, ...} or a list
        # Peek at one matrix to get T
        if isinstance(container, dict):
            first_key = next(iter(container.keys()))
            T = container[first_key].shape[0]
        else:
            T = container[0].shape[0]

        # build identity expectation matrix
        E_base = np.eye(T)

        # replicate structure
        if isinstance(container, dict):
            E_id[inp] = {k: E_base.copy() for k in container.keys()}
        else:
            E_id[inp] = [E_base.copy() for _ in range(len(container))]

    return E_id

import numpy as np
from copy import deepcopy

def align_jac_hh_z_with_aggregate(model, keys=None, eps=1e-12, do_print=True):
    """
    Rescale per-z household Jacobians model.jac_hh_z so that for each key:
        sum_z jac_hh_z[key][z, :, :] == model.jac_hh[key]

    Uses model.jac_hh_direct[key] as the pre-alignment aggregate corresponding
    to the raw per-z jacobians.

    After running this, the identity-expectations pipeline should reproduce the
    original FIRE aggregate jac_hh and IRFs (up to small numerical noise).
    """

    if not hasattr(model, 'jac_hh_z'):
        raise AttributeError("model.jac_hh_z not found. Run compute_jac_hh_z(...) first.")
    if not hasattr(model, 'jac_hh_direct'):
        raise AttributeError("model.jac_hh_direct not found. compute_jac_hh_z must set it.")
    if not hasattr(model, 'jac_hh'):
        raise AttributeError("model.jac_hh (baseline aggregate) not found. Run compute_jacs(...) first on the original model before copying.")

    if keys is None:
        keys = list(model.jac_hh_z.keys())

    for key in keys:
        if key not in model.jac_hh_z:
            continue

        Jz      = model.jac_hh_z[key]          # (Nz, T, T) raw per-z (direct)
        J_dir   = model.jac_hh_direct.get(key, None)  # (T, T) direct aggregate
        J_base  = model.jac_hh.get(key, None)         # (T, T) baseline (fake-news) aggregate

        if J_dir is None or J_base is None:
            continue

        # scaling matrix S(t,s) such that:
        #   S(t,s) * J_dir(t,s) = J_base(t,s)
        # we then multiply ALL z-level entries by S(t,s),
        # so that sum_z Jz_scaled = J_base.
        S = np.ones_like(J_base)
        mask = np.abs(J_dir) > eps
        S[mask] = J_base[mask] / J_dir[mask]

        # apply scaling to each z
        model.jac_hh_z[key] = Jz * S[None, :, :]

        if do_print:
            J_new_agg = model.jac_hh_z[key].sum(axis=0)
            max_abs = np.max(np.abs(J_new_agg - J_base))
            print(f"{key}: alignment max|Σ_z Jz - J_base| = {max_abs:.3e}")

    return model


def E_sticky_exp(theta, T=300):
    col = 1 - theta**(1 + np.arange(T))
    E = np.tile(col[:, np.newaxis], (1, T))
    E = np.triu(E, +1) + np.tril(np.ones((T, T)))
    return E 

    import numpy as np

def show_E_by_z(E_by_z, preview=6, max_items=6):
    # Handles dict -> (list/dict/array), or a single matrix/array
    print("E_by_z type:", type(E_by_z))
    if hasattr(E_by_z, "items"):
        print("keys:", list(E_by_z.keys()))
        for key, mats in E_by_z.items():
            print(f"\nE_by_z['{key}']:")
            # normalize to iterable of matrices
            if isinstance(mats, dict):
                items = list(mats.items())
            else:
                try:
                    items = list(enumerate(mats))
                except TypeError:
                    items = [(0, mats)]
            for j, (idx, M) in enumerate(items):
                A = np.asarray(M)
                print(f"  [{idx}] shape={A.shape}, dtype={A.dtype}", end="")
                if A.ndim == 2:
                    finite = np.isfinite(A).all()
                    print(f", finite={finite}")
                    r = min(preview, A.shape[0]); c = min(preview, A.shape[1])
                    print(A[:r, :c])
                else:
                    print("")
                if j+1 >= max_items:
                    print("  ... (truncated)")
                    break
    else:
        A = np.asarray(E_by_z)
        print("Single value shape:", A.shape, "dtype:", A.dtype)
        if A.ndim == 2:
            r = min(preview, A.shape[0]); c = min(preview, A.shape[1])
            print(A[:r, :c])


import numpy as np

def E_sticky_exp(theta, T=300, decay=None):
    """
    Standard sticky expectations matrix with optional decay of past realizations.

    theta in [0,1] : Calvo-style non-update probability
    decay in (0,1] : 'forgetting' factor for the effect of old shocks.
                     decay = 1.0 reproduces the original E_sticky_exp.
    """

    # --- original sticky structure (cumulative) ---
    col = 1.0 - theta**(1 + np.arange(T))   # shape (T,)
    E = np.tile(col[:, np.newaxis], (1, T)) # T x T
    E = np.triu(E, +1) + np.tril(np.ones((T, T)))

    # --- optional decay of past realizations ---
    if decay is not None and decay < 1.0:
        # d[t,s] = max(t-s, 0): how far in the past shock s is at time t
        h = np.arange(T)
        D = np.maximum(h[:, None] - h[None, :], 0)   # T x T, 0 on & above diag, 1,2,... below
        E = E * (decay ** D)

    return E

def build_E_by_z_sticky(theta_vec_Z,
                        theta_vec_ra=None,
                        theta_vec_chi=None,
                        decay_vec_Z=None,
                        decay_vec_ra=None,
                        decay_vec_chi=None,
                        T=300):
    """
    Build E_by_z_sane-like dict of sticky expectations matrices, now with
    OPTIONAL decay of past realizations.

    theta_vec_Z   : array-like (Nz,)  sticky parameter for Z
    theta_vec_ra  : same for ra. If None, reuse theta_vec_Z
    theta_vec_chi : same for chi. If None, reuse theta_vec_Z

    decay_vec_Z   : array-like (Nz,) decay in (0,1]. If None, no decay (1.0)
    decay_vec_ra  : same for ra. If None, reuse decay_vec_Z
    decay_vec_chi : same for chi. If None, reuse decay_vec_Z

    T             : horizon

    Returns
    -------
    E_by_z : dict
        E_by_z['Z'][i_z]   -> (T,T) matrix for Z in state z
        E_by_z['ra'][i_z]  -> (T,T) matrix for ra in state z
        E_by_z['chi'][i_z] -> (T,T) matrix for chi in state z
    """

    theta_vec_Z = np.asarray(theta_vec_Z, dtype=float)
    Nz = len(theta_vec_Z)

    # --- theta handling ---
    if theta_vec_ra is None:
        theta_vec_ra = theta_vec_Z
    else:
        theta_vec_ra = np.asarray(theta_vec_ra, dtype=float)
        assert len(theta_vec_ra) == Nz

    if theta_vec_chi is None:
        theta_vec_chi = theta_vec_Z
    else:
        theta_vec_chi = np.asarray(theta_vec_chi, dtype=float)
        assert len(theta_vec_chi) == Nz

    # --- decay handling ---
    if decay_vec_Z is None:
        decay_vec_Z = np.ones(Nz)
    else:
        decay_vec_Z = np.asarray(decay_vec_Z, dtype=float)
        assert len(decay_vec_Z) == Nz

    if decay_vec_ra is None:
        decay_vec_ra = decay_vec_Z
    else:
        decay_vec_ra = np.asarray(decay_vec_ra, dtype=float)
        assert len(decay_vec_ra) == Nz

    if decay_vec_chi is None:
        decay_vec_chi = decay_vec_Z
    else:
        decay_vec_chi = np.asarray(decay_vec_chi, dtype=float)
        assert len(decay_vec_chi) == Nz

    # --- build matrices ---
    E_by_z = {'Z': [], 'ra': [], 'chi': []}

    for iz in range(Nz):
        thZ  = theta_vec_Z[iz]
        thR  = theta_vec_ra[iz]
        thC  = theta_vec_chi[iz]

        dZ = decay_vec_Z[iz]
        dR = decay_vec_ra[iz]
        dC = decay_vec_chi[iz]

        E_Z_z   = E_sticky_exp(thZ, T=T, decay=dZ)
        E_ra_z  = E_sticky_exp(thR, T=T, decay=dR)
        E_chi_z = E_sticky_exp(thC, T=T, decay=dC)

        E_by_z['Z'].append(E_Z_z)
        E_by_z['ra'].append(E_ra_z)
        E_by_z['chi'].append(E_chi_z)

    return E_by_z


import numpy as np
from copy import deepcopy

# ------------------------------------------------------
# 1. Align per-z Jacobians with baseline aggregate
# ------------------------------------------------------

def align_jac_hh_z_with_aggregate(model, keys=None, eps=1e-12, do_print=True):
    """
    Rescale per-z household Jacobians model.jac_hh_z so that for each key:
        sum_z jac_hh_z[key][z, :, :] == model.jac_hh[key]

    Uses model.jac_hh_direct[key] as the aggregate corresponding to the
    raw per-z jacobians (both come from compute_jac_hh_z).

    After running this, the per-z decomposition is consistent with the
    baseline aggregate jac_hh.

    Assumes the model has:
        - model.jac_hh          (baseline FIRE / fake-news aggregate)
        - model.jac_hh_direct   (direct aggregate from compute_jac_hh_z)
        - model.jac_hh_z        (per-z direct jacobians)
    """

    if not hasattr(model, 'jac_hh_z'):
        raise AttributeError("model.jac_hh_z not found. Run compute_jac_hh_z(...) first.")
    if not hasattr(model, 'jac_hh_direct'):
        raise AttributeError("model.jac_hh_direct not found. compute_jac_hh_z must set it.")
    if not hasattr(model, 'jac_hh'):
        raise AttributeError("model.jac_hh (baseline aggregate) not found. "
                             "Run model.compute_jacs(...) on the original FIRE model before copying.")

    if keys is None:
        keys = list(model.jac_hh_z.keys())

    for key in keys:
        if key not in model.jac_hh_z:
            continue

        Jz     = model.jac_hh_z[key]                 # (Nz, T, T) raw per-z (direct)
        J_dir  = model.jac_hh_direct.get(key, None)  # (T, T) direct aggregate
        J_base = model.jac_hh.get(key, None)         # (T, T) baseline (fake-news) aggregate

        if J_dir is None or J_base is None:
            continue

        # scaling matrix S(t,s) such that:
        #   S(t,s) * J_dir(t,s) = J_base(t,s)
        S = np.ones_like(J_base)
        mask = np.abs(J_dir) > eps
        S[mask] = J_base[mask] / J_dir[mask]

        # apply scaling to each z-level slice
        model.jac_hh_z[key] = Jz * S[None, :, :]

        if do_print:
            J_new_agg = model.jac_hh_z[key].sum(axis=0)
            max_abs = np.max(np.abs(J_new_agg - J_base))
            print(f"{key}: alignment max|Σ_z Jz - J_base| = {max_abs:.3e}")

    return model


# ------------------------------------------------------
# 2. Diagnostics: goods-market residual and IRF comparison
# ------------------------------------------------------

def goods_market_residual(model, T=40, var_Y='Y', var_C='C_hh'):
    """Max |Y - C_hh| over first T periods."""
    Y = model.IRF[var_Y][:T]
    C = model.IRF[var_C][:T]
    return float(np.max(np.abs(Y - C)))

def compare_irfs(model_a, model_b, vars=('Y','C_hh','Z','ra'), T=40, label_a='A', label_b='B'):
    """Print max |Δ| for selected variables between two models."""
    print(f"\nIRF differences: {label_a} vs {label_b}")
    for v in vars:
        if v in model_a.IRF and v in model_b.IRF:
            diff = model_b.IRF[v][:T] - model_a.IRF[v][:T]
            print(f"  {v}: max|Δ| = {float(np.max(np.abs(diff))):.3e}")
        else:
            print(f"  {v}: (missing in one model)")


# ------------------------------------------------------
# 3. Optional: aggregate sticky expectations helper
# ------------------------------------------------------

def apply_aggregate_sticky_expectations(model_fire, E, sticky_vars=('Z','ra'), outputs=('C_hh','A_hh')):
    """
    Build a 'sticky expectations' model at the AGGREGATE level using create_alt_M
    directly on model_fire.jac_hh.

    E is the expectation matrix used in your standard sticky case (shape (T,T)).
    """
    model_sticky = deepcopy(model_fire)

    for o in outputs:
        for i in sticky_vars:
            key = (o, i)
            if key not in model_fire.jac_hh:
                continue
            J_fire = model_fire.jac_hh[key]
            J_sticky = create_alt_M(J_fire.copy(), E)
            model_sticky.jac_hh[key] = J_sticky

    # rebuild GE jacobians, but keep HH block
    model_sticky.compute_jacs(skip_hh=True, skip_shocks=False, do_print=False)
    return model_sticky


# ------------------------------------------------------
# 4. Full pipeline with intermediate diagnostics
# ------------------------------------------------------

def run_full_beliefs_diagnostics(model, E, E_by_z_sane,
                                 sticky_vars=('Z','ra'),
                                 outputs=('C_hh','A_hh'),
                                 T_test=40, dx=1e-4):
    """
    Run a multi-stage pipeline and print diagnostics at each step:

    Stage 0: Baseline FIRE
    Stage 1: Aggregate sticky expectations (create_alt_M on model.jac_hh)
    Stage 2: Per-z decomposition + alignment (no expectations modification)
    Stage 3: Heterogeneous expectations by z (E_by_z_sane)
    """

    print("========== STAGE 0: BASELINE FIRE ==========")
    # assume model already has jac_hh and IRFs (but ensure)
    model.compute_jacs(do_print=False)
    model.find_IRFs(shocks=['eps_i'], do_print=False)
    gm_fire = goods_market_residual(model, T=T_test)
    print(f"Max |Y - C_hh| (FIRE) = {gm_fire:.6e}")

    # --------------------------------------------------
    print("\n========== STAGE 1: AGGREGATE STICKY (create_alt_M) ==========")
    model_sticky = apply_aggregate_sticky_expectations(
        model_fire = model,
        E          = E,
        sticky_vars= sticky_vars,
        outputs    = outputs
    )
    model_sticky.find_IRFs(shocks=['eps_i'], do_print=False)
    gm_sticky = goods_market_residual(model_sticky, T=T_test)
    print(f"Max |Y - C_hh| (agg sticky) = {gm_sticky:.6e}")
    compare_irfs(model, model_sticky, T=T_test, label_a='FIRE', label_b='AGG-STICKY')

    # --------------------------------------------------
    print("\n========== STAGE 2: PER-z DECOMP + ALIGNMENT (no beliefs change) ==========")
    model_z = deepcopy(model)   # start from FIRE
    # compute per-z HH jacobians (direct)
    compute_jac_hh_z(model_z, dx=dx, do_print=False)
    # align per-z to baseline aggregate (fake-news)
    align_jac_hh_z_with_aggregate(model_z, do_print=True)
    # rebuild GE jacobians with aligned HH block
    model_z.compute_jacs(skip_hh=True, skip_shocks=False, do_print=False)
    model_z.find_IRFs(shocks=['eps_i'], do_print=False)

    gm_z = goods_market_residual(model_z, T=T_test)
    print(f"Max |Y - C_hh| (per-z aligned) = {gm_z:.6e}")
    compare_irfs(model, model_z, T=T_test, label_a='FIRE', label_b='PER-z-ALIGNED')

    # --------------------------------------------------
    print("\n========== STAGE 3: HETEROGENEOUS BELIEFS BY z (create_alt_M + E_by_z_sane) ==========")
    model_het = deepcopy(model_z)  # start from aligned per-z + baseline aggregate

    # apply heterogeneous expectations by z using your existing helper
    apply_heterogeneous_expectations_by_z(
        model_fire    = model_het,
        model_beliefs = model_het,
        E_by_z_sane   = E_by_z_sane,
        sticky_vars   = sticky_vars,
        outputs       = outputs,
        overwrite_aggregate = True
    )

    model_het.compute_jacs(skip_hh=True, skip_shocks=False, do_print=False)
    model_het.find_IRFs(shocks=['eps_i'], do_print=False)

    gm_het = goods_market_residual(model_het, T=T_test)
    print(f"Max |Y - C_hh| (HET by z) = {gm_het:.6e}")
    compare_irfs(model,      model_het, T=T_test, label_a='FIRE',       label_b='HET-z')
    compare_irfs(model_z,    model_het, T=T_test, label_a='PER-z-AL',   label_b='HET-z')
    compare_irfs(model_sticky, model_het, T=T_test, label_a='AGG-STICKY', label_b='HET-z')

    return {
        'model_fire':   model,
        'model_sticky': model_sticky,
        'model_z':      model_z,
        'model_het':    model_het
    }


import numpy as np

def check_hh_jacobian_consistency(model_fire, model_het,
                                  outputs=('C_hh','A_hh'),
                                  inputs=('Z','ra'),
                                  rtol=1e-6, atol=1e-8):
    """
    1) Check that for BOTH models, aggregate hh Jacobians equal sum over z.
    2) Compare FIRE vs HET aggregate Jacobians.
    """

    print("=== 1. Aggregate vs sum over z (FIRE model) ===")
    for o in outputs:
        for i in inputs:
            key = (o,i)
            if not hasattr(model_fire, 'jac_hh_z') or key not in model_fire.jac_hh_z:
                print(f"[SKIP FIRE] {key} not in jac_hh_z")
                continue
            J_z = model_fire.jac_hh_z[key]         # (Nz,T,T)
            J_agg = model_fire.jac_hh[key]         # (T,T) (FIRE aggregate)
            ok = np.allclose(J_agg, J_z.sum(axis=0), rtol=rtol, atol=atol)
            print(f"FIRE {key}: agg == sum_z? {ok}")

    print("\n=== 2. Aggregate vs sum over z (HET model) ===")
    for o in outputs:
        for i in inputs:
            key = (o,i)
            if not hasattr(model_het, 'jac_hh_z') or key not in model_het.jac_hh_z:
                print(f"[SKIP HET] {key} not in jac_hh_z")
                continue
            J_z = model_het.jac_hh_z[key]
            J_agg = model_het.jac_hh[key]
            ok = np.allclose(J_agg, J_z.sum(axis=0), rtol=rtol, atol=atol)
            max_abs = np.max(np.abs(J_agg - J_z.sum(axis=0)))
            print(f"HET  {key}: agg == sum_z? {ok}, max|Δ|={max_abs:.3e}")

    print("\n=== 3. FIRE vs HET aggregate differences ===")
    for o in outputs:
        for i in inputs:
            key = (o,i)
            if key not in model_fire.jac_hh or key not in model_het.jac_hh:
                print(f"[SKIP] {key}: missing in one of the models")
                continue
            J_fire = model_fire.jac_hh[key]
            J_het  = model_het.jac_hh[key]
            diff   = J_het - J_fire
            max_abs = np.max(np.abs(diff))
            denom   = np.maximum(np.abs(J_fire), atol)
            max_rel = np.max(np.abs(diff) / denom)
            print(f"{key}: max|Δ|={max_abs:.3e}, max rel Δ={max_rel:.3e}")

import numpy as np

def summarize_jacobian_differences(
    model_fire,
    model_het,
    outputs=('C_hh', 'A_hh'),
    inputs=('Z', 'ra'),
    rtol=1e-8,
    atol=1e-10
):
    """
    Compare aggregate household Jacobians between:
        model_fire.jac_hh  (FIRE)
        model_het.jac_hh   (heterogeneous beliefs)

    Prints max abs and relative differences for each (output,input) pair,
    and returns a dict with the stats.
    """

    stats = {}

    print("=== Comparing aggregate household Jacobians (FIRE vs hetero beliefs) ===")
    for o in outputs:
        for i in inputs:
            key = (o, i)
            if key not in model_fire.jac_hh or key not in model_het.jac_hh:
                print(f"[SKIP] {key}: missing in one of the models")
                continue

            J_fire = model_fire.jac_hh[key]
            J_het  = model_het.jac_hh[key]

            diff = J_het - J_fire
            max_abs = np.max(np.abs(diff))
            denom = np.maximum(np.abs(J_fire), atol)
            max_rel = np.max(np.abs(diff) / denom)

            close = np.allclose(J_fire, J_het, rtol=rtol, atol=atol)

            stats[key] = {
                'max_abs_diff': float(max_abs),
                'max_rel_diff': float(max_rel),
                'allclose': bool(close),
            }

            print(f"{key}: max|Δ|={max_abs:.3e}, max rel Δ={max_rel:.3e}, allclose={close}")

    return stats


def z_moments_from_D_and_c(D, c, Nz):
    """
    Returns:
      mu_z      : mass by z, shape (Nz,)
      cbar_z_ss : mean consumption conditional on z, shape (Nz,)
    Works when D has shape like (1,Nz,Na) and c like (1,Nz,Na) or broadcastable.
    """
    D = np.asarray(D)
    c = np.asarray(c)

    # find z axis (the one whose length is Nz)
    z_axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
    if len(z_axes) != 1:
        raise ValueError(f"Ambiguous z axis. D.shape={D.shape}, Nz={Nz}")
    z_ax = z_axes[0]

    # mu_z = sum of D over all non-z axes
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)
    mu_z = D.sum(axis=sum_axes)
    mu_z = np.maximum(mu_z, 0)
    mu_z = mu_z / mu_z.sum()

    # total consumption in each z: sum D * c over non-z axes
    Dc = (D * c).sum(axis=sum_axes)

    # conditional mean cbar_z_ss
    cbar_z_ss = Dc / np.maximum(mu_z, 1e-16)
    return mu_z, cbar_z_ss


def stationary_dist_from_P(P, tol=1e-14, maxiter=100_000):
    """
    Stationary distribution mu such that mu = mu P.
    """
    P = np.asarray(P)
    Nz = P.shape[0]
    mu = np.ones(Nz) / Nz
    for _ in range(maxiter):
        mu_new = mu @ P
        if np.max(np.abs(mu_new - mu)) < tol:
            mu = mu_new
            break
        mu = mu_new
    mu = np.maximum(mu, 0)
    mu /= mu.sum()
    return mu

import numpy as np

def dist_z_from_D(D, Nz):
    """
    Extract mass by z from joint stationary distribution D, for any layout.
    Finds an axis of D with length Nz, sums over all other axes.
    """
    D = np.asarray(D)
    if D.ndim < 2:
        raise ValueError("D must be at least 2D (joint distribution over states).")

    # find axes matching Nz
    axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
    if len(axes) == 0:
        raise ValueError(f"No axis of D matches Nz={Nz}. D.shape={D.shape}")
    if len(axes) > 1:
        # if ambiguous, pick the last matching axis (usually z is later than assets)
        z_axis = axes[-1]
    else:
        z_axis = axes[0]

    # sum over all other axes
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_axis)
    mu = D.sum(axis=sum_axes)

    mu = np.maximum(mu, 0)
    s = mu.sum()
    if s <= 0:
        raise ValueError("Computed dist_z has non-positive sum; check D.")
    mu = mu / s
    return mu



def _project_to_simplex(v):
    """Project v onto the probability simplex {w>=0, sum w = 1}."""
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    if len(rho) == 0:
        # fallback: uniform
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def _infer_dist_z_from_agg(model, output, inputs, Nz, T_jac, prefer_inp=None):
    """
    Infer weights mu_z by matching aggregate contribution:
        sum_z mu_z (Jz[z] @ x)  ≈  (J_agg @ x)
    using least squares + simplex projection.
    """
    # pick an input that exists in both jac_hh_z and jac_hh
    candidates = list(inputs)
    if prefer_inp is not None:
        candidates = [prefer_inp] + [k for k in candidates if k != prefer_inp]

    chosen = None
    for inp in candidates:
        if (output, inp) in getattr(model, "jac_hh", {}) and (output, inp) in getattr(model, "jac_hh_z", {}):
            if inp in model.IRF:
                chosen = inp
                break
    if chosen is None:
        raise RuntimeError("Cannot infer weights: no common (output, inp) in jac_hh and jac_hh_z with IRF available.")

    J_agg = model.jac_hh[(output, chosen)]        # (T,T)
    Jz    = model.jac_hh_z[(output, chosen)]      # (Nz,T,T)
    x     = model.IRF[chosen][:T_jac]             # (T,)

    # Build matrix A of shape (T, Nz): column z is (Jz[z] @ x)
    A = np.zeros((T_jac, Nz))
    for iz in range(Nz):
        A[:, iz] = Jz[iz] @ x

    y = J_agg @ x  # (T,)

    # Solve min ||A w - y||^2
    w_ls, *_ = np.linalg.lstsq(A, y, rcond=None)

    # Project to simplex
    w = _project_to_simplex(w_ls)

    # Quick diagnostics
    y_fit = A @ w
    err = np.max(np.abs(y_fit - y))
    return w, chosen, err


import numpy as np
import matplotlib.pyplot as plt


def _dist_z_from_D_anyaxis(D, Nz):
    """Find axis of D with size Nz; sum over other axes to get mu_z."""
    D = np.asarray(D)
    z_axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
    if len(z_axes) != 1:
        raise ValueError(f"Ambiguous z-axis: D.shape={D.shape}, Nz={Nz}")
    z_ax = z_axes[0]
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)
    mu = D.sum(axis=sum_axes)
    mu = np.maximum(mu, 0)
    mu = mu / mu.sum()
    return mu, z_ax


def _cbar_z_from_D_and_c(D, c, Nz):
    """Compute conditional mean c by z: cbar_z = E[c | z]."""
    D = np.asarray(D)
    c = np.asarray(c)
    mu_z, z_ax = _dist_z_from_D_anyaxis(D, Nz)
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)
    Dc = (D * c).sum(axis=sum_axes)
    cbar = Dc / np.maximum(mu_z, 1e-16)
    return mu_z, cbar

def compute_C_hh_z_ss(model):
    """
    Compute per-agent steady-state consumption by productivity group z.

    Returns
    -------
    C_hh_z : (Nz,) array
        Mean consumption per agent in productivity group z
    mass_z : (Nz,) array
        Population mass of each z
    """

    ss = model.ss
    par = model.par

    # joint stationary distribution over (z, a)
    # mu_za: (Nz, Na)
    mu_za, mass_z, _, _ = model.get_exposures()

    # --- extract consumption policy ---
    # ss.C or ss.c typically has shape (Nfix, Nz, Na)
    if hasattr(ss, "C"):
        c_pol_full = ss.C
    elif hasattr(ss, "c"):
        c_pol_full = ss.c
    else:
        raise AttributeError("Could not find consumption policy in ss (C or c).")

    # aggregate over i_fix if present
    if c_pol_full.ndim == 3:
        # (Nfix, Nz, Na) → (Nz, Na)
        c_pol = np.mean(c_pol_full, axis=0)
    elif c_pol_full.ndim == 2:
        # already (Nz, Na)
        c_pol = c_pol_full
    else:
        raise ValueError("Unexpected shape for consumption policy.")

    # sanity check
    if c_pol.shape != mu_za.shape:
        raise ValueError(
            f"Shape mismatch: c_pol {c_pol.shape}, mu_za {mu_za.shape}"
        )

    # conditional distribution a | z
    mu_a_given_z = mu_za / mass_z[:, None]

    # per-agent mean consumption by z
    C_hh_z = np.sum(c_pol * mu_a_given_z, axis=1)

    return C_hh_z, mass_z


def stationary_dist(P, tol=1e-14, maxit=10_000):
    Nz = P.shape[0]
    mu = np.ones(Nz) / Nz
    for _ in range(maxit):
        mu_new = mu @ P
        if np.max(np.abs(mu_new - mu)) < tol:
            return mu_new
        mu = mu_new
    raise RuntimeError("Stationary distribution did not converge")

import numpy as np

def decompose_by_z_and_input_norm(
    model,
    inputs=("Z", "ra", "chi"),
    output="C_hh",
    T=None,
    check_consistency=True,
    agg_mode="auto",   # "auto" | "sum" | "mu_sum" | "mean" | "mu_cbar"
    include_distribution_residual=True,
    dist_irf_key_candidates=("D", "Dbeg"),
):
    """
    Decompose output IRFs by productivity z and input using per-z Jacobians, and (optionally)
    compute the missing distribution/composition component as a residual.

    Works for output="C_hh" OR output="A_hh" (or any output in jac_hh_z).

    Returns
    -------
    Y_z_total            : (Nz, T)
    Y_z_contrib          : dict[input -> (Nz, T)]
    Y_agg_total          : (T,)
    Y_agg_contrib        : dict[input -> (T,)]
    agg_mode_used        : str
    Y_agg_residual       : (T,) or None
    Y_agg_dist_term      : (T,) or None
    dist_irf_key_used    : str or None
    """

    if isinstance(inputs, str):
        inputs = (inputs,)

    # ---- infer dimensions ----
    example_key = None
    for inp in inputs:
        key = (output, inp)
        if hasattr(model, "jac_hh_z") and key in model.jac_hh_z:
            example_key = key
            break
    if example_key is None:
        raise KeyError(f"No per-z Jacobians found for output={output} and inputs={inputs}")

    J_example = model.jac_hh_z[example_key]
    Nz, T_jac, _ = J_example.shape
    if T is None or T > T_jac:
        T = T_jac

    # ---- get mu_z and (optional) ybar_z_ss from ss.D and ss.<micro policy> ----
    # We always need mu_z for some aggregation modes.
    Dss = np.asarray(model.ss.D)
    # find z-axis
    z_axes = [ax for ax, n in enumerate(Dss.shape) if n == Nz]
    if len(z_axes) != 1:
        raise ValueError(f"Ambiguous z-axis in ss.D: D.shape={Dss.shape}, Nz={Nz}")
    z_ax = z_axes[0]
    sum_axes = tuple(ax for ax in range(Dss.ndim) if ax != z_ax)

    mu_z = Dss.sum(axis=sum_axes)
    mu_z = np.maximum(mu_z, 0)
    mu_z = mu_z / np.maximum(mu_z.sum(), 1e-32)

    # For "mu_cbar" we need a conditional mean of the OUTPUT by z.
    # For C_hh we can use ss.c. For A_hh we can use ss.a (if it exists).
    ybar_z_ss = None
    if agg_mode in ("mu_cbar", "auto"):
        micro_map = None
        # common mapping for your model:
        if output == "C_hh" and hasattr(model.ss, "c"):
            micro_map = np.asarray(model.ss.c)
        elif output == "A_hh" and hasattr(model.ss, "a"):
            micro_map = np.asarray(model.ss.a)
        # if unknown output, we cannot build mu_cbar meaningfully
        if micro_map is not None:
            Dy = (Dss * micro_map).sum(axis=sum_axes)
            ybar_z_ss = Dy / np.maximum(mu_z, 1e-32)

    # ---- build per-z contributions (levels) ----
    Y_z_contrib = {}
    for inp in inputs:
        key = (output, inp)
        if key not in model.jac_hh_z:
            continue
        if inp not in model.IRF:
            raise KeyError(f"model.IRF has no entry for '{inp}'")

        Jz = model.jac_hh_z[key]       # (Nz,T,T)
        x  = np.asarray(model.IRF[inp])[:T_jac]   # (T,)

        Yz = np.zeros((Nz, T_jac))
        for iz in range(Nz):
            Yz[iz] = Jz[iz] @ x

        Y_z_contrib[inp] = Yz[:, :T]

    if len(Y_z_contrib) == 0:
        raise RuntimeError("No contributions computed (check inputs and jac_hh_z keys).")

    Y_z_total = sum(Y_z_contrib.values())

    # ---- candidate aggregation rules ----
    def agg_apply(Yz, mode):
        if mode == "sum":
            return Yz.sum(axis=0)
        if mode == "mu_sum":
            return (mu_z[:, None] * Yz).sum(axis=0)
        if mode == "mean":
            return Yz.mean(axis=0)
        if mode == "mu_cbar":
            if ybar_z_ss is None:
                raise ValueError("agg_mode='mu_cbar' requires mapping from output to a micro policy (ss.c or ss.a).")
            # interpret Yz as %/log change in mean-by-z; convert to levels with mu_z*ybar_z_ss
            return (mu_z[:, None] * (ybar_z_ss[:, None] * Yz)).sum(axis=0)
        raise ValueError(f"Unknown agg_mode '{mode}'")

    # ---- choose agg_mode ----
    if agg_mode == "auto":
        if output not in model.IRF:
            raise KeyError(f"Need model.IRF['{output}'] for agg_mode='auto'")
        y_true = np.asarray(model.IRF[output])[:T]  # levels
        modes = ["sum", "mu_sum", "mean"]
        if ybar_z_ss is not None:
            modes.append("mu_cbar")
        errs = {}
        for m in modes:
            try:
                errs[m] = np.max(np.abs(agg_apply(Y_z_total, m) - y_true))
            except Exception:
                continue
        agg_mode_used = min(errs, key=errs.get)
        print(f"[decomp] agg_mode auto-chosen: {agg_mode_used}  max|diff|={errs[agg_mode_used]:.2e}")
    else:
        agg_mode_used = agg_mode

    # ---- aggregate contributions and totals (policy-only) ----
    Y_agg_contrib = {inp: agg_apply(Yz, agg_mode_used) for inp, Yz in Y_z_contrib.items()}
    Y_agg_total = sum(Y_agg_contrib.values())

    # ---- residual ----
    Y_agg_residual = None
    if include_distribution_residual and output in model.IRF:
        Y_agg_residual = np.asarray(model.IRF[output])[:T] - Y_agg_total

    # ---- theoretical distribution term (if D IRF exists) ----
    Y_agg_dist_term = None
    dist_irf_key_used = None
    if include_distribution_residual:
        for k in dist_irf_key_candidates:
            if k in model.IRF:
                dist_irf_key_used = k
                break

        # only compute if we have a micro mapping for the output (c or a)
        micro_map = None
        if output == "C_hh" and hasattr(model.ss, "c"):
            micro_map = np.asarray(model.ss.c)
        elif output == "A_hh" and hasattr(model.ss, "a"):
            micro_map = np.asarray(model.ss.a)

        if dist_irf_key_used is not None and micro_map is not None:
            dD = np.asarray(model.IRF[dist_irf_key_used])

            if dD.shape[0] == T_jac:
                dD_t = dD[:T]
            elif dD.shape[-1] == T_jac:
                dD_t = np.moveaxis(dD, -1, 0)[:T]
            else:
                dD_t = None

            if dD_t is not None:
                # sum_s micro_ss(s) * dD_t(s)
                Y_agg_dist_term = np.tensordot(
                    dD_t, micro_map,
                    axes=(tuple(range(1, dD_t.ndim)), tuple(range(micro_map.ndim)))
                )

    # ---- checks ----
    if check_consistency and output in model.IRF:
        diff = Y_agg_total - np.asarray(model.IRF[output])[:T]
        print(f"[decomp] max|agg(from z, policy-only) − IRF| = {np.max(np.abs(diff)):.2e}")
        if Y_agg_residual is not None:
            print(f"[decomp] residual range: [{Y_agg_residual.min():.3e}, {Y_agg_residual.max():.3e}]")

    return (Y_z_total, Y_z_contrib, Y_agg_total, Y_agg_contrib,
            agg_mode_used, Y_agg_residual, Y_agg_dist_term, dist_irf_key_used)

import numpy as np
import matplotlib.pyplot as plt

def plot_by_z_with_input_decomp_norm(
    model,
    inputs=("Z", "ra", "chi"),
    output="C_hh",
    T=None,
    norm_mode="none",           # "none" | "own_ss" | "elasticity"
    shock_var=None,             # required for elasticity
    title_prefix=None,
    plot_aggregate=True,
    overlay_true_agg_irf=True,
    add_distribution_residual=True,
    agg_mode="auto",
    # --- NEW: baseline overlay ---
    overlay_baseline_total_by_z=False,
    baseline_model=None,
    baseline_label="Baseline total (per z)",
    common_yaxis=True,
):
    """
    Per-z decomposition grid + aggregate decomposition in % of SS.

    NEW:
      - Set output="A_hh" to get the per-z A decomposition.
      - Set overlay_baseline_total_by_z=True and pass baseline_model to overlay baseline per-z TOTAL only.
    """

    (Y_z, Y_z_contrib, Y_agg, Y_agg_contrib,
     agg_mode_used, Y_resid, Y_dist_term, dist_key) = decompose_by_z_and_input_norm(
        model=model,
        inputs=inputs,
        output=output,
        T=T,
        check_consistency=True,
        agg_mode=agg_mode,
        include_distribution_residual=True,
    )

    Nz, T_eff = Y_z.shape
    t = np.arange(T_eff)

    # ---- steady states ----
    # For own_ss normalization, we want mean(output micro) by z
    if norm_mode == "own_ss":
        # Use ss.D with micro mapping
        D = np.asarray(model.ss.D)
        if output == "C_hh":
            micro = np.asarray(model.ss.c)
        elif output == "A_hh":
            micro = np.asarray(model.ss.a)
        else:
            raise ValueError("own_ss normalization currently supported for output in {'C_hh','A_hh'} only.")

        # infer z axis
        z_axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
        if len(z_axes) != 1:
            raise ValueError(f"Ambiguous z-axis in ss.D: D.shape={D.shape}, Nz={Nz}")
        z_ax = z_axes[0]
        sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)

        mass_z = D.sum(axis=sum_axes)
        Y_z_ss = (D * micro).sum(axis=sum_axes) / np.maximum(mass_z, 1e-32)  # mean within z

    Y_agg_ss = float(getattr(model.ss, output))

    # ---- elasticity denominator ----
    if norm_mode == "elasticity":
        if shock_var is None:
            raise ValueError("elasticity normalization requires shock_var")
        shock_impact = float(np.asarray(model.IRF[shock_var]).ravel()[0])
        if abs(shock_impact) < 1e-12:
            raise ValueError("Shock impact is numerically zero")

    def normalize_perz(arr, z=None, ss_by_z=None):
        if norm_mode == "none":
            return arr
        elif norm_mode == "own_ss":
            return 100.0 * arr / np.maximum(ss_by_z[z], 1e-32)
        elif norm_mode == "elasticity":
            return arr / shock_impact
        else:
            raise ValueError(f"Unknown norm_mode '{norm_mode}'")

    # ---- baseline per-z total overlay (optional) ----
    Y_base_z = None
    if overlay_baseline_total_by_z:
        if baseline_model is None:
            raise ValueError("overlay_baseline_total_by_z=True requires baseline_model.")
        # get baseline per-z TOTAL response using its own jac_hh_z + its own IRF paths
        (Yb_z, _, _, _, _, _, _, _) = decompose_by_z_and_input_norm(
            model=baseline_model,
            inputs=inputs,
            output=output,
            T=T_eff,
            check_consistency=False,
            agg_mode="sum",  # doesn't matter for per-z
            include_distribution_residual=False,
        )
        Y_base_z = Yb_z  # levels

        # for own_ss normalization, compute baseline's Y_ss(z)
        if norm_mode == "own_ss":
            Db = np.asarray(baseline_model.ss.D)
            if output == "C_hh":
                microb = np.asarray(baseline_model.ss.c)
            elif output == "A_hh":
                microb = np.asarray(baseline_model.ss.a)

            z_axes_b = [ax for ax, n in enumerate(Db.shape) if n == Nz]
            if len(z_axes_b) != 1:
                raise ValueError(f"Ambiguous z-axis in baseline ss.D: D.shape={Db.shape}, Nz={Nz}")
            z_ax_b = z_axes_b[0]
            sum_axes_b = tuple(ax for ax in range(Db.ndim) if ax != z_ax_b)

            mass_z_b = Db.sum(axis=sum_axes_b)
            Y_base_z_ss = (Db * microb).sum(axis=sum_axes_b) / np.maximum(mass_z_b, 1e-32)
        else:
            Y_base_z_ss = None

    # ---- per-z grid ----
    ncols = int(np.ceil(np.sqrt(Nz)))
    nrows = int(np.ceil(Nz / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=True,
        sharey=common_yaxis
    )
    axes = np.atleast_1d(axes).flatten()

    for iz in range(Nz):
        ax = axes[iz]
        ax.plot(t, normalize_perz(Y_z[iz], z=iz, ss_by_z=Y_z_ss if norm_mode=="own_ss" else None),
                lw=2.0, label="total")
        for inp in inputs:
            if inp in Y_z_contrib:
                ax.plot(t, normalize_perz(Y_z_contrib[inp][iz], z=iz, ss_by_z=Y_z_ss if norm_mode=="own_ss" else None),
                        "--", lw=1.4, label=f"from {inp}")

        # baseline overlay (total only)
        if overlay_baseline_total_by_z and (Y_base_z is not None):
            ax.plot(
                t,
                normalize_perz(Y_base_z[iz], z=iz, ss_by_z=Y_base_z_ss if norm_mode=="own_ss" else None),
                lw=2.0,
                color="gray",
                alpha=0.9,
                label=baseline_label if iz == 0 else None,
            )

        ax.axhline(0, ls=":", lw=0.8)
        ax.set_title(f"z = {iz}", fontsize=9)
        ax.grid(alpha=0.25)
        if iz == 0:
            ax.legend(frameon=False, fontsize=8)

    for j in range(Nz, len(axes)):
        fig.delaxes(axes[j])

    ylabel = {
        "none": "level deviation",
        "own_ss": "% deviation from own steady state",
        "elasticity": "response per unit shock impact",
    }[norm_mode]

    if title_prefix is None:
        title_prefix = f"{output} IRF decomposition"

    fig.supxlabel("t")
    fig.supylabel(ylabel)
    fig.suptitle(f"{title_prefix} ({norm_mode})", y=1.02)
    plt.tight_layout()
    plt.show()

    # ---- aggregate plot (% of aggregate SS) ----
    if plot_aggregate:
        true_levels = np.asarray(model.IRF[output])[:T_eff] if output in model.IRF else None

        fig2, ax2 = plt.subplots(figsize=(7.2, 4.2))

        ax2.plot(t, 100.0 * Y_agg / np.maximum(Y_agg_ss, 1e-32),
                 lw=2.8, label=f"Policy-only total (agg_mode={agg_mode_used})")

        for inp in inputs:
            if inp in Y_agg_contrib:
                ax2.plot(t, 100.0 * Y_agg_contrib[inp] / np.maximum(Y_agg_ss, 1e-32),
                         "--", lw=2.0, label=inp)

        if add_distribution_residual and (Y_resid is not None):
            ax2.plot(t, 100.0 * Y_resid / np.maximum(Y_agg_ss, 1e-32),
                     "-.", lw=2.6, color="black", label="Distribution/composition (residual)")

        if add_distribution_residual and (Y_dist_term is not None):
            ax2.plot(t, 100.0 * Y_dist_term / np.maximum(Y_agg_ss, 1e-32),
                     ":", lw=2.4, color="gray", label=f"Implied dist term (ss·d{dist_key})")

        if overlay_true_agg_irf and (true_levels is not None):
            ax2.plot(t, 100.0 * true_levels / np.maximum(Y_agg_ss, 1e-32),
                     lw=3.0, color="gray", label="TRUE IRF")

        ax2.axhline(0, ls=":", lw=0.9)
        ax2.set_xlabel("Quarters")
        ax2.set_ylabel(f"% change in {output}")
        ax2.set_title(f"{output} decomposition: Aggregate (% of SS)")
        ax2.grid(alpha=0.25)
        ax2.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        if add_distribution_residual and (true_levels is not None) and (Y_resid is not None):
            closed = Y_agg + Y_resid
            print("[agg check] max| (policy + residual) − TRUE | =",
                  f"{np.max(np.abs(closed - true_levels)):.2e}")

    return (Y_z, Y_z_contrib, Y_agg, Y_agg_contrib,
            agg_mode_used, Y_resid, Y_dist_term, dist_key)

import numpy as np
import matplotlib.pyplot as plt

def diagnose_perz_vs_agg(model, output='C_hh', inputs=('Z','ra','chi'), T=21):
    ssC = model.ss.C_hh

    # True aggregate total (%)
    C_true_tot = model.IRF[output][:T] * 100 / ssC

    print("=== Check: does model.IRF[C_hh] have a hump? ===")
    print("C_true_tot[0:6] =", np.round(C_true_tot[:6], 4))

    # z-mass from D (works for your D.shape=(1,9,300))
    Nz = model.jac_hh_z[(output, inputs[0])].shape[0]
    mu = dist_z_from_D(model.ss.D, Nz)

    # Helper: compute per-z contribution series for an input under a given Jz transform
    def perz_series(inp, transpose=False):
        Jz = model.jac_hh_z[(output, inp)]
        if transpose:
            Jz = Jz.transpose(0,2,1)
        x = model.IRF[inp][:Jz.shape[1]]
        Cz = np.array([Jz[iz] @ x for iz in range(Nz)])  # (Nz,Tjac)
        return Cz[:, :T]  # (Nz,T)

    # Compare, input-by-input, the implied aggregate contribution to the trusted one
    print("\n=== Input-by-input: compare jac_hh @ IRF to aggregates built from jac_hh_z ===")
    results = {}

    for inp in inputs:
        if (output, inp) not in model.jac_hh or (output, inp) not in model.jac_hh_z:
            print(f"skip {inp} (missing jacobians)")
            continue

        # trusted contribution
        y_true = (model.jac_hh[(output, inp)] @ model.IRF[inp])[:T] * 100 / ssC

        # candidates for per-z -> aggregate
        Cz      = perz_series(inp, transpose=False)          # (Nz,T)
        Cz_T    = perz_series(inp, transpose=True)

        cand = {
            "sum":            Cz.sum(axis=0)       * 100 / ssC,
            "mu_sum":         (mu[:,None]*Cz).sum(axis=0)   * 100 / ssC,
            "sum_T":          Cz_T.sum(axis=0)     * 100 / ssC,
            "mu_sum_T":       (mu[:,None]*Cz_T).sum(axis=0) * 100 / ssC,
        }

        errs = {k: np.max(np.abs(v - y_true)) for k,v in cand.items()}
        best = min(errs, key=errs.get)
        results[inp] = (best, errs[best], errs)

        print(f"{inp:>4s} best={best:8s}  max|diff|={errs[best]:.3e}   all={ {k:float(f'{e:.2e}') for k,e in errs.items()} }")

        # plot for visual sanity
        plt.figure(figsize=(6.2,3.2))
        plt.plot(y_true, lw=2.6, label=f"TRUE (jac_hh @ IRF) [{inp}]")
        plt.plot(cand[best], lw=2.0, ls="--", label=f"FROM jac_hh_z [{best}]")
        plt.axhline(0, color='k', lw=0.7)
        plt.title(f"Contribution comparison: {inp}")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    # Now compare totals
    print("\n=== Total: compare sum of TRUE contributions vs sum of best FROM-Z contributions ===")
    true_sum = np.zeros(T)
    fromz_sum = np.zeros(T)

    for inp, (best, _, _) in results.items():
        true_sum += (model.jac_hh[(output, inp)] @ model.IRF[inp])[:T] * 100 / ssC

        Cz = perz_series(inp, transpose=("T" in best))
        if "mu" in best:
            fromz_sum += (mu[:,None]*Cz).sum(axis=0) * 100 / ssC
        else:
            fromz_sum += Cz.sum(axis=0) * 100 / ssC

    plt.figure(figsize=(6.6,3.6))
    plt.plot(C_true_tot, lw=3.0, label="TRUE total = IRF[C_hh]")
    plt.plot(true_sum, lw=2.6, label="Sum TRUE inputs (jac_hh)")
    plt.plot(fromz_sum, lw=2.2, ls="--", label="Sum FROM-Z best rules")
    plt.axhline(0, color='k', lw=0.7)
    plt.title("Totals: what has the hump and what doesn't?")
    plt.ylabel("% change in C")
    plt.xlabel("quarters")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    print("max| (Sum TRUE inputs) - IRF[C_hh] | =", np.max(np.abs(true_sum - C_true_tot)))
    print("max| (Sum FROM-Z best) - (Sum TRUE inputs) | =", np.max(np.abs(fromz_sum - true_sum)))

    return results

import numpy as np
import pandas as pd

def table_rel_diff_total_vs_baseline_by_z(
    model,
    baseline_model,
    output="C_hh",
    inputs=("Z","ra","chi"),
    T=21,
    window=(0, 8),     # window for trough/avg metrics
    eps=1e-12,
):
    """
    Build a table comparing per-z TOTAL response in 'model' vs 'baseline_model'.

    Uses the same machinery as your plot function:
      - Computes per-z TOTAL in levels from jac_hh_z @ IRF inputs
      - Converts to % deviation from own steady state: 100*ΔY(z,t)/Y_ss(z)
      - Reports impact, trough-in-window, average-in-window
      - Reports pp gap and "relative gap" scaled by |baseline|.

    Returns a DataFrame.
    """

    # --- compute per-z TOTAL (levels) for model and baseline ---
    Yz, _, _, _, _, _, _, _ = decompose_by_z_and_input_norm(
        model=model, inputs=inputs, output=output, T=T,
        check_consistency=False, agg_mode="sum",
        include_distribution_residual=False
    )
    Yb, _, _, _, _, _, _, _ = decompose_by_z_and_input_norm(
        model=baseline_model, inputs=inputs, output=output, T=T,
        check_consistency=False, agg_mode="sum",
        include_distribution_residual=False
    )

    Nz, T_eff = Yz.shape
    T_eff = min(T_eff, Yb.shape[1])
    Yz = Yz[:, :T_eff]
    Yb = Yb[:, :T_eff]

    # --- compute own-ss by z for output (mean within z) ---
    def ss_by_z(m):
        D = np.asarray(m.ss.D)
        if output == "C_hh":
            micro = np.asarray(m.ss.c)
        elif output == "A_hh":
            micro = np.asarray(m.ss.a)
        else:
            raise ValueError("ss_by_z supports output in {'C_hh','A_hh'} unless you add mapping.")

        # find z axis
        z_axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
        if len(z_axes) != 1:
            raise ValueError(f"Ambiguous z axis in ss.D: D.shape={D.shape}, Nz={Nz}")
        z_ax = z_axes[0]
        sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)

        mass_z = D.sum(axis=sum_axes)
        ybar_z = (D * micro).sum(axis=sum_axes) / np.maximum(mass_z, eps)
        return ybar_z, mass_z

    Yss_z, mass_z = ss_by_z(model)
    Yss_b, mass_b = ss_by_z(baseline_model)

    # --- normalize to % deviations from own SS (like norm_mode="own_ss") ---
    Pz = 100.0 * Yz / np.maximum(Yss_z[:, None], eps)
    Pb = 100.0 * Yb / np.maximum(Yss_b[:, None], eps)

    # --- metrics ---
    a, b = window
    a = max(0, int(a))
    b = min(T_eff-1, int(b))

    impact_model = Pz[:, 0]
    impact_base  = Pb[:, 0]
    gap_pp_impact = impact_model - impact_base
    gap_rel_impact = gap_pp_impact / np.maximum(np.abs(impact_base), eps)

    trough_model = Pz[:, a:b+1].min(axis=1)
    trough_base  = Pb[:, a:b+1].min(axis=1)
    gap_pp_trough = trough_model - trough_base
    gap_rel_trough = gap_pp_trough / np.maximum(np.abs(trough_base), eps)

    avg_model = Pz[:, a:b+1].mean(axis=1)
    avg_base  = Pb[:, a:b+1].mean(axis=1)
    gap_pp_avg = avg_model - avg_base
    gap_rel_avg = gap_pp_avg / np.maximum(np.abs(avg_base), eps)

    df = pd.DataFrame({
        "z": np.arange(Nz),
        "mass(z)_model": mass_z,
        "mass(z)_base": mass_b,
        "%Δ_total_model_t0": impact_model,
        "%Δ_total_base_t0": impact_base,
        "gap_pp_t0": gap_pp_impact,
        "gap_rel_t0": gap_rel_impact,

        f"min_%Δ_model_t{a}-{b}": trough_model,
        f"min_%Δ_base_t{a}-{b}": trough_base,
        f"gap_pp_trough_t{a}-{b}": gap_pp_trough,
        f"gap_rel_trough_t{a}-{b}": gap_rel_trough,

        f"avg_%Δ_model_t{a}-{b}": avg_model,
        f"avg_%Δ_base_t{a}-{b}": avg_base,
        f"gap_pp_avg_t{a}-{b}": gap_pp_avg,
        f"gap_rel_avg_t{a}-{b}": gap_rel_avg,
    })

    # optional: rank who deviates most from baseline at trough (absolute pp)
    df["rank_abs_gap_trough_pp"] = df[f"gap_pp_trough_t{a}-{b}"].abs().rank(
        ascending=False, method="dense"
    ).astype(int)

    return df

import numpy as np
import pandas as pd

def table_peakgap_total_vs_baseline_by_z(
    model,
    baseline_model,
    output="C_hh",
    inputs=("Z","ra","chi"),
    T=21,
    window=(0, 8),
    stat_mode="auto",   # "auto" | "min" | "max"
    eps=1e-12,
    keep_cols="compact",  # "compact" or "ultra"
):
    """
    Build a *narrow* table comparing per-z TOTAL response in 'model' vs 'baseline_model'
    over a window.

    IMPORTANT:
      - For C_hh we typically want the trough (most negative) => min
      - For A_hh we typically want the peak increase           => max

    Returns a DataFrame suitable for LaTeX.
    """

    # ---------- compute per-z TOTAL in LEVELS ----------
    Yz_model, *_ = decompose_by_z_and_input_norm(
        model=model, inputs=inputs, output=output, T=T,
        check_consistency=False, agg_mode="sum",
        include_distribution_residual=False
    )
    Yz_base, *_ = decompose_by_z_and_input_norm(
        model=baseline_model, inputs=inputs, output=output, T=T,
        check_consistency=False, agg_mode="sum",
        include_distribution_residual=False
    )

    Nz, T_eff = Yz_model.shape
    T_eff = min(T_eff, Yz_base.shape[1])
    Yz_model = Yz_model[:, :T_eff]
    Yz_base  = Yz_base[:, :T_eff]

    # ---------- compute steady-state mean-by-z for normalization (own SS) ----------
    def ss_by_z(m):
        D = np.asarray(m.ss.D)

        if output == "C_hh":
            micro = np.asarray(m.ss.c)
        elif output == "A_hh":
            micro = np.asarray(m.ss.a)
        else:
            raise ValueError("own-SS normalization supported for output in {'C_hh','A_hh'} unless you add mapping.")

        z_axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
        if len(z_axes) != 1:
            raise ValueError(f"Ambiguous z-axis in ss.D: D.shape={D.shape}, Nz={Nz}")
        z_ax = z_axes[0]
        sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)

        mass_z = D.sum(axis=sum_axes)
        ybar_z = (D * micro).sum(axis=sum_axes) / np.maximum(mass_z, eps)
        return ybar_z, mass_z

    Yss_model, mass_model = ss_by_z(model)
    Yss_base,  mass_base  = ss_by_z(baseline_model)

    # percent deviations from own SS (like norm_mode="own_ss")
    P_model = 100.0 * Yz_model / np.maximum(Yss_model[:, None], eps)
    P_base  = 100.0 * Yz_base  / np.maximum(Yss_base[:, None], eps)

    # ---------- choose window and statistic ----------
    a, b = window
    a = max(0, int(a))
    b = min(T_eff - 1, int(b))

    if stat_mode == "auto":
        # default: trough for C, peak for A
        use_max = (output == "A_hh")
    elif stat_mode == "max":
        use_max = True
    elif stat_mode == "min":
        use_max = False
    else:
        raise ValueError("stat_mode must be 'auto', 'min', or 'max'")

    if use_max:
        stat_model = P_model[:, a:b+1].max(axis=1)
        stat_base  = P_base[:,  a:b+1].max(axis=1)
        stat_label = "max %Δ"
    else:
        stat_model = P_model[:, a:b+1].min(axis=1)
        stat_base  = P_base[:,  a:b+1].min(axis=1)
        stat_label = "min %Δ"

    gap_pp   = stat_model - stat_base
    gap_rel  = gap_pp / np.maximum(np.abs(stat_base), eps)

    # rank by absolute pp gap
    rank = np.abs(gap_pp).rank if isinstance(gap_pp, pd.Series) else None

    df = pd.DataFrame({
        "z": np.arange(Nz),
        "mass(z)": mass_model,   # use model mass; typically same as baseline
        f"{stat_label} (model) [{a},{b}]": stat_model,
        f"{stat_label} (base) [{a},{b}]":  stat_base,
        "gap (pp)": gap_pp,
        "gap / |base|": gap_rel,
    })

    df["rank"] = df["gap (pp)"].abs().rank(ascending=False, method="dense").astype(int)

    # ---------- keep table narrow ----------
    if keep_cols == "ultra":
        out = df[["z","mass(z)","gap (pp)","gap / |base|","rank"]].copy()
    else:
        out = df[["z","mass(z)",
                  f"{stat_label} (model) [{a},{b}]",
                  f"{stat_label} (base) [{a},{b}]",
                  "gap (pp)","gap / |base|","rank"]].copy()

    return out

import numpy as np

def dist_contribution_from_dD(model, T=21, dD_key_candidates=("D","Dbeg"), output_ss_key="C_hh"):
    """
    Computes dist term: sum_s c_ss(s) * dD_t(s), in levels and in % of agg SS.
    Returns (dist_term_levels, dist_term_percent, key_used).
    """
    css = np.asarray(model.ss.c)      # same shape as ss.D (usually)
    ssC = getattr(model.ss, output_ss_key)

    key_used = None
    for k in dD_key_candidates:
        if k in model.IRF:
            key_used = k
            break
    if key_used is None:
        raise KeyError(f"No distribution IRF found. Tried keys {dD_key_candidates}")

    dD = np.asarray(model.IRF[key_used])

    # detect time axis: either (T, ...) or (..., T)
    if dD.shape[0] >= T:
        dD_t = dD[:T]
    elif dD.shape[-1] >= T:
        dD_t = np.moveaxis(dD, -1, 0)[:T]
    else:
        raise ValueError(f"Cannot find time axis in dD with shape {dD.shape}")

    # dist term levels: for each t, sum over all non-time axes of dD_t * c_ss
    # tensordot over state axes
    dist_levels = np.tensordot(dD_t, css, axes=(tuple(range(1, dD_t.ndim)), tuple(range(css.ndim))))
    dist_percent = 100 * dist_levels / ssC
    return dist_levels, dist_percent, key_used

import numpy as np
import pandas as pd

def welfare_loss_by_z(model, output="C_hh", inputs=("Z","ra","chi"), T=21, eps=1e-32):
    """
    Returns:
      dYz_level : (Nz,T) per-z level deviation in output (policy-only via jac_hh_z)
      Yss_z     : (Nz,)  steady-state mean output within z
      mass_z    : (Nz,)  mass within z
    """

    dYz_level, *_ = decompose_by_z_and_input_norm(
        model=model, inputs=inputs, output=output, T=T,
        check_consistency=False, agg_mode="sum",
        include_distribution_residual=False
    )
    dYz_level = np.asarray(dYz_level)[:, :T]
    Nz = dYz_level.shape[0]

    D = np.asarray(model.ss.D)

    if output == "C_hh":
        micro = np.asarray(model.ss.c)
    elif output == "A_hh":
        micro = np.asarray(model.ss.a)
    else:
        raise ValueError("Add mapping for output -> steady-state micro object.")

    # find the z axis in D
    z_axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
    if len(z_axes) != 1:
        raise ValueError(f"Ambiguous z axis in D.shape={D.shape}, Nz={Nz}")
    z_ax = z_axes[0]
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)

    mass_z = D.sum(axis=sum_axes)
    Yss_z  = (D * micro).sum(axis=sum_axes) / np.maximum(mass_z, eps)

    # ensure 1D
    mass_z = np.asarray(mass_z).reshape(Nz)
    Yss_z  = np.asarray(Yss_z).reshape(Nz)

    return dYz_level, Yss_z, mass_z


def compare_welfare_burden_by_z(
    model_hetero,
    model_homo,
    inputs=("Z","ra","chi"),
    T=21,
    eps=1e-32,
):
    """
    Welfare-weighted loss index by z using u'(c)=c^{-sigma} and discounting beta^t.

    L(z) = sum_t beta^t * c_ss(z)^(-sigma) * dC_t(z)
    ΔL(z) = L_hetero(z) - L_homo(z)
    """

    par = model_hetero.par
    beta  = float(par.beta)
    sigma = float(par.sigma)

    dC_h, Css_h, mass_h = welfare_loss_by_z(model_hetero, output="C_hh", inputs=inputs, T=T, eps=eps)
    dC_b, Css_b, mass_b = welfare_loss_by_z(model_homo,  output="C_hh", inputs=inputs, T=T, eps=eps)

    disc = beta ** np.arange(T)              # (T,)

    mu_h = Css_h ** (-sigma)                 # (Nz,)
    mu_b = Css_b ** (-sigma)                 # (Nz,)

    # discounted sum over time -> (Nz,)
    sum_h = (dC_h[:, :T] * disc[None, :]).sum(axis=1)   # (Nz,)
    sum_b = (dC_b[:, :T] * disc[None, :]).sum(axis=1)   # (Nz,)

    # L(z) must be 1D
    L_h = mu_h * sum_h
    L_b = mu_b * sum_b
    dL  = L_h - L_b

    # force 1D (robust)
    Nz = dL.size
    L_h = np.asarray(L_h).reshape(Nz)
    L_b = np.asarray(L_b).reshape(Nz)
    dL  = np.asarray(dL).reshape(Nz)

    df = pd.DataFrame({
        "z": np.arange(Nz),
        "mass(z)": mass_h,
        "C_ss(z)_hetero": Css_h,
        "C_ss(z)_homo": Css_b,
        "LossIndex_hetero": L_h,
        "LossIndex_homo": L_b,
        "ΔLossIndex (hetero-homo)": dL,
    })

    df["rank_abs_ΔLoss"] = df["ΔLossIndex (hetero-homo)"].abs().rank(
        ascending=False, method="dense"
    ).astype(int)

    return df

import re, numpy as np
from collections import Counter

def _shape(x):
    return getattr(x, "shape", None)

def _is_sq_T(x, T=None):
    sh = _shape(x)
    return isinstance(sh, tuple) and len(sh)==2 and sh[0]==sh[1] and (T is None or sh[0]==T)

def _to_int_index(x):
    """Try to parse a z-index from ints or strings like 'z0', 'iz=3', 'state_5'."""
    if isinstance(x, (int, np.integer)): 
        return int(x)
    if isinstance(x, str):
        m = re.search(r'(-?\d+)', x)
        if m: 
            return int(m.group(1))
    return None

def _short(items, n=8):
    items = list(items)
    return items[:n] + (["..."] if len(items)>n else [])

def print_model_io_overview(model):
    # Pull T, Nz, and the input labels from your HH Jacobians
    if not hasattr(model, "jac_hh") or not model.jac_hh:
        print("model.jac_hh is missing or empty.")
        return None, None, []
    any_J = next(iter(model.jac_hh.values()))
    T = any_J.shape[0]
    Nz = int(getattr(model.par, "Nz", 0))
    inputs = sorted({k[1] for k in model.jac_hh.keys() if k[0] in ("C_hh","A_hh")})
    print(f"Model says: T={T}, Nz={Nz}, inputs={inputs}")
    return T, Nz, inputs

def peek_E_top(E):
    print(f"\nE_by_z_sane top-level type: {type(E).__name__}")
    if hasattr(E, "shape"):
        print(f"  ndarray shape: {E.shape}")
    elif isinstance(E, dict):
        keys = list(E.keys())
        print(f"  dict with {len(keys)} top-level keys. Sample: {_short(keys)}")
    elif isinstance(E, (list, tuple)):
        print(f"  {type(E).__name__} of length {len(E)}")

def analyze_E_dict(E, T=None, Nz=None, inputs=None, max_show=10):
    print("\n— Top-level dict analysis —")
    keys = list(E.keys())
    # 1) What kinds of keys?
    kinds = Counter(type(k).__name__ for k in keys)
    print("  key types:", dict(kinds))

    # 2) Tuple keys like (input, iz)
    tuple_keys = [k for k in keys if isinstance(k, tuple) and len(k)==2]
    if tuple_keys:
        print(f"  Found {len(tuple_keys)} tuple keys (input, iz). Sample:", _short(tuple_keys))
        # Parse tuples
        inputs_seen = []
        iz_seen = []
        bad_slices = 0
        for k in tuple_keys[:max_show]:
            I, z = k
            iz = _to_int_index(z)
            inputs_seen.append(str(I))
            iz_seen.append(iz)
            v = E[k]
            print(f"    {k!r}: value type={type(v).__name__}, shape={_shape(v)}")
            if T is not None and not _is_sq_T(v, T):
                bad_slices += 1
        print("  tuple inputs seen:", sorted(set(inputs_seen)))
        print("  tuple z parsed:", sorted(set(i for i in iz_seen if i is not None)))
        if bad_slices:
            print(f"  WARNING: {bad_slices} tuple entries are not (T,T)")

    # 3) String keys that might be input names
    str_keys = [k for k in keys if isinstance(k, str)]
    if str_keys:
        print(f"\n  Found {len(str_keys)} string keys. Sample:", _short(str_keys))
        # For each string key, look at value shape/type
        for k in str_keys[:max_show]:
            v = E[k]
            print(f"    '{k}': type={type(v).__name__}, shape={_shape(v)}")
            if isinstance(v, dict):
                subk = list(v.keys())
                print(f"      nested dict with {len(subk)} keys. Sample:", _short(subk))
                # Try parse iz from nested keys
                izs = [_to_int_index(sk) for sk in subk]
                izs = [i for i in izs if i is not None]
                if izs:
                    print(f"      parsed nested z-indices:", sorted(set(izs)))
                # Check shapes of first few
                for sk in _short(subk, n=5):
                    vv = v[sk]
                    print(f"        subkey {sk!r}: type={type(vv).__name__}, shape={_shape(vv)}")
            elif hasattr(v, "shape") and len(v.shape)==3:
                print("      3D array under this key → likely per-z block (Nz,T,T) or (T,T,Nz).")
            elif isinstance(v, (list, tuple)):
                print(f"      list/tuple len={len(v)}; first item shape={_shape(v[0]) if v else None}")

    # 4) Int-like keys (top-level dict[iz] case)
    intlike = [k for k in keys if _to_int_index(k) is not None]
    if intlike:
        print(f"\n  Found {len(intlike)} int-like keys at top-level (e.g., per-z). Sample:", _short(intlike))
        for k in intlike[:max_show]:
            v = E[k]
            print(f"    key {k!r}: type={type(v).__name__}, shape={_shape(v)}")
            if isinstance(v, dict):
                print("      nested dict under a z-key (unusual).")
            if hasattr(v, "shape") and len(v.shape)==3:
                print("      3D array under a z-key → likely per-z block holding many z.")

def analyze_E_array(A, T=None, Nz=None):
    sh = getattr(A, "shape", None)
    print("\n— Array analysis —")
    print("  shape:", sh)
    if sh is None:
        print("  Not an array.")
        return
    if len(sh)==2:
        print("  Looks like a single (T,T) operator for all z and inputs.")
    elif len(sh)==3:
        if Nz is not None and sh[0]==Nz and sh[1]==sh[2]:
            print("  Looks like (Nz, T, T): per-z operator for all inputs.")
        elif Nz is not None and sh[-1]==Nz and sh[0]==sh[1]:
            print("  Looks like (T, T, Nz): per-z operator for all inputs.")
        else:
            print("  3D but cannot infer axes from Nz/T—will need explicit mapping.")
    else:
        print("  Unexpected ndim; please share.")

def overview_E_by_z_sane(E_by_z_sane, model):
    print("=== MODEL IO OVERVIEW ===")
    T, Nz, inputs = print_model_io_overview(model)
    print("\n=== E_by_z_sane OVERVIEW ===")
    peek_E_top(E_by_z_sane)
    if isinstance(E_by_z_sane, dict):
        analyze_E_dict(E_by_z_sane, T=T, Nz=Nz, inputs=inputs)
    elif hasattr(E_by_z_sane, "shape"):
        analyze_E_array(E_by_z_sane, T=T, Nz=Nz)
    elif isinstance(E_by_z_sane, (list, tuple)):
        print("\n— Top-level list/tuple —")
        L = E_by_z_sane
        print("  len:", len(L))
        if len(L)>0:
            print("  first item type:", type(L[0]).__name__, "shape:", _shape(L[0]))
            if hasattr(L[0], "shape") and len(L[0].shape)==2:
                print("  Looks like per-z list/tuple of (T,T).")
            elif hasattr(L[0], "shape") and len(L[0].shape)==3:
                print("  Nested 3D under list—please share.")
    else:
        print("E_by_z_sane is of unsupported type; please share its structure.")


import numpy as np
import matplotlib.pyplot as plt

def plot_asset_distribution_by_z(model, bins=40, density=False):
    """
    Plot asset distributions separately for each productivity level z.

    Parameters
    ----------
    model : solved model
        Requires:
            model.ss.D       : stationary distribution (Nfix, Nz, Na)
            model.par.a_grid : asset grid (Na,)
            model.par.Nz     : number of productivity states
    bins : int
        Number of bins.
    density : bool
        If True, normalize each z-distribution to integrate to 1.
    """

    a_grid = np.asarray(model.par.a_grid)        # (Na,)
    D = np.asarray(model.ss.D)                    # (Nfix, Nz, Na)
    Nz = model.par.Nz

    fig, axes = plt.subplots(Nz, 1, figsize=(7, 2.2 * Nz), sharex=True)
    if Nz == 1:
        axes = [axes]

    for z in range(Nz):
        # Sum over fixed types only
        mass_a_z = D[:, z, :].sum(axis=0)         # (Na,)

        if density:
            mass_a_z = mass_a_z / mass_a_z.sum()

        axes[z].bar(
            a_grid,
            mass_a_z,
            width=np.diff(a_grid, prepend=a_grid[0]),
            alpha=0.8
        )

        axes[z].set_ylabel(f"z = {z}")
        axes[z].grid(alpha=0.3)

    axes[-1].set_xlabel("Asset holdings a")
    fig.suptitle("Asset holdings by productivity level", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_asset_distribution_overlay_by_z(model, bins=40):
    a_grid = np.asarray(model.par.a_grid)
    D = np.asarray(model.ss.D)
    Nz = model.par.Nz

    plt.figure(figsize=(7, 4))

    for z in range(Nz):
        mass_a_z = D[:, z, :].sum(axis=0)
        mass_a_z /= mass_a_z.sum()  # normalize within z

        plt.plot(a_grid, mass_a_z, linewidth=2, label=f"z={z}")

    plt.xlabel("Asset holdings a")
    plt.ylabel("Density")
    plt.title("Asset distributions by productivity")
    plt.xlim(-1,20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_asset_cdf_by_z(model):
    a_grid = np.asarray(model.par.a_grid)
    D = np.asarray(model.ss.D)
    Nz = model.par.Nz

    plt.figure(figsize=(7,4))

    for z in range(Nz):
        mass_a_z = D[:, z, :].sum(axis=0)
        cdf_z = np.cumsum(mass_a_z)
        cdf_z /= cdf_z[-1]

        plt.plot(a_grid, cdf_z, label=f"z={z}")

    plt.xlabel("Asset holdings a")
    plt.ylabel("Cumulative share of agents")
    plt.title("CDF of asset holdings by productivity")
    plt.xlim(-1,20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt



def fire_benchmark_given_equilibrium_paths(
    model_fire,
    model_eq,
    outputs=('C_hh', 'A_hh'),
    inputs=('Z', 'ra', 'chi'),
    T=None,
    use_per_z=True,
    return_percent=False
):
    """
    Compute "FIRE benchmark" household responses to equilibrium paths from model_eq
    using FIRE household Jacobians from model_fire.

    Robust to mismatched horizons between jacobians and IRFs by using
    T_use = min(T, T_jac, len(IRF[input])).
    """

    # --- choose T if not given ---
    if T is None:
        T = min(len(model_eq.IRF[inp]) for inp in inputs if inp in model_eq.IRF)

    # --- input paths from equilibrium economy ---
    x_paths = {}
    for inp in inputs:
        if inp not in model_eq.IRF:
            raise KeyError(f"model_eq.IRF has no '{inp}'. Available: {list(model_eq.IRF.keys())}")
        x_paths[inp] = np.asarray(model_eq.IRF[inp], dtype=float)

    out = {'agg': {}, 'agg_contrib': {}}

    # ---------- aggregate FIRE benchmark ----------
    for o in outputs:
        contrib = {}
        total = np.zeros(T)

        for inp in inputs:
            key = (o, inp)
            if not hasattr(model_fire, 'jac_hh') or key not in model_fire.jac_hh:
                continue

            J = model_fire.jac_hh[key]          # (T_jac, T_jac)
            T_jac = J.shape[0]
            x_full = x_paths[inp]               # (T_irf,)

            T_use = min(T, T_jac, len(x_full))
            if T_use <= 0:
                continue

            y = J[:T_use, :T_use] @ x_full[:T_use]   # (T_use,)
            y_out = np.zeros(T)
            y_out[:T_use] = y

            contrib[inp] = y_out
            total += y_out

        out['agg'][o] = total
        out['agg_contrib'][o] = contrib

    # ---------- per-z FIRE benchmark (optional) ----------
    if use_per_z and hasattr(model_fire, 'jac_hh_z'):
        out['z'] = {}
        out['z_contrib'] = {}

        # infer Nz and T_jac_z from any available key
        example_key = None
        for o in outputs:
            for inp in inputs:
                k = (o, inp)
                if k in model_fire.jac_hh_z:
                    example_key = k
                    break
            if example_key is not None:
                break

        if example_key is not None:
            Nz, T_jac_z, _ = model_fire.jac_hh_z[example_key].shape

            for o in outputs:
                contrib_z = {}
                total_z = np.zeros((Nz, T))

                for inp in inputs:
                    key = (o, inp)
                    if key not in model_fire.jac_hh_z:
                        continue

                    Jz = model_fire.jac_hh_z[key]    # (Nz, T_jac_z, T_jac_z)
                    x_full = x_paths[inp]

                    T_use = min(T, T_jac_z, len(x_full))
                    if T_use <= 0:
                        continue

                    yz_out = np.zeros((Nz, T))
                    for iz in range(Nz):
                        yz = Jz[iz, :T_use, :T_use] @ x_full[:T_use]
                        yz_out[iz, :T_use] = yz

                    contrib_z[inp] = yz_out
                    total_z += yz_out

                out['z'][o] = total_z
                out['z_contrib'][o] = contrib_z

    # ---------- percent deviations (optional) ----------
    if return_percent:
        out['agg_pct'] = {}
        out['agg_contrib_pct'] = {}
        for o in outputs:
            ss_val = getattr(model_fire.ss, o)
            if ss_val == 0:
                raise ValueError(f"Steady state {o} is zero, cannot form percent deviations.")
            out['agg_pct'][o] = 100 * out['agg'][o] / ss_val
            out['agg_contrib_pct'][o] = {k: 100*v/ss_val for k, v in out['agg_contrib'][o].items()}

        if 'z' in out:
            out['z_pct'] = {}
            out['z_contrib_pct'] = {}
            for o in outputs:
                ss_val = getattr(model_fire.ss, o)
                out['z_pct'][o] = 100 * out['z'][o] / ss_val
                out['z_contrib_pct'][o] = {k: 100*v/ss_val for k, v in out['z_contrib'][o].items()}

    return out



def plot_eq_vs_fire_benchmark(
    model_eq,
    fire_out,
    output='C_hh',
    T=40,
    percent=True,
    title=None
):
    """
    Plot equilibrium IRF vs FIRE benchmark for one output.
    """
    t = np.arange(T)

    if percent:
        y_eq = 100 * model_eq.IRF[output][:T] / getattr(model_eq.ss, output)
        y_fire = fire_out['agg_pct'][output][:T]
        ylab = "% deviation from steady state"
    else:
        y_eq = model_eq.IRF[output][:T]
        y_fire = fire_out['agg'][output][:T]
        ylab = "level deviation"

    plt.figure(figsize=(7, 4))
    plt.plot(t, y_eq, lw=2.5, label='Equilibrium (non-FIRE)')
    plt.plot(t, y_fire, lw=2.5, ls='--', label='FIRE benchmark (price-taking)')
    plt.axhline(0, ls=':', lw=0.8)
    plt.grid(alpha=0.3)
    plt.xlabel("t")
    plt.ylabel(ylab)
    plt.title(title if title is not None else f"{output}: equilibrium vs FIRE benchmark")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import math

def plot_eq_vs_fire_by_z_and_aggregate(
    model_eq,
    model_fire,
    fire_out,
    output='C_hh',
    inputs=('Z','ra','chi'),
    T=40,
    percent=True,
    max_cols=3,
    title_prefix=None,
):
    """
    Per-z: plot EQ(z) vs FIRE(z) (both computed from jac_hh_z using the SAME input paths)
    Aggregate: plot EQ(agg) vs FIRE(agg)

    Requirements:
      - model_eq.IRF[inp] exists for each inp in inputs
      - model_eq.jac_hh_z[(output, inp)] exists (to compute EQ(z))
      - fire_out contains per-z FIRE benchmark: fire_out['z'][output] or fire_out['z_pct'][output]
        and aggregate FIRE: fire_out['agg'][output] or fire_out['agg_pct'][output]
    """

    # ---- horizon ----
    if output not in model_eq.IRF:
        raise KeyError(f"model_eq.IRF has no '{output}'")

    # Build input paths
    x_paths = {}
    for inp in inputs:
        if inp not in model_eq.IRF:
            raise KeyError(f"model_eq.IRF has no input '{inp}'. Available: {list(model_eq.IRF.keys())}")
        x_paths[inp] = np.asarray(model_eq.IRF[inp], dtype=float)

    # Helper: compute per-z implied IRF from a model's jac_hh_z
    def compute_per_z_from_jac(model, T_target):
        # find an example key to infer (Nz, T_jac)
        example_key = None
        for inp in inputs:
            k = (output, inp)
            if hasattr(model, 'jac_hh_z') and k in model.jac_hh_z:
                example_key = k
                break
        if example_key is None:
            raise KeyError(f"{model} has no jac_hh_z for {(output, inputs)}")

        J_ex = model.jac_hh_z[example_key]
        Nz, T_jac, _ = J_ex.shape

        T_use = min(T_target, T_jac, *(len(x_paths[inp]) for inp in inputs))
        Yz = np.zeros((Nz, T_target))

        for inp in inputs:
            k = (output, inp)
            if k not in model.jac_hh_z:
                continue
            Jz = model.jac_hh_z[k]              # (Nz, T_jac, T_jac)
            x = x_paths[inp][:T_use]            # (T_use,)

            for iz in range(Nz):
                Yz[iz, :T_use] += Jz[iz, :T_use, :T_use] @ x

        return Yz, T_use

    # ---- compute equilibrium per-z from heterogeneous-beliefs Jacobians ----
    Y_eq_z, T_use_eq = compute_per_z_from_jac(model_eq, T)

    # ---- FIRE per-z from fire_out (already computed from FIRE jac_hh_z) ----
    if percent:
        if 'z_pct' not in fire_out or output not in fire_out['z_pct']:
            raise KeyError("fire_out missing z_pct for this output. Recompute with return_percent=True.")
        Y_fire_z = np.asarray(fire_out['z_pct'][output])
    else:
        if 'z' not in fire_out or output not in fire_out['z']:
            raise KeyError("fire_out missing z for this output.")
        Y_fire_z = np.asarray(fire_out['z'][output])

    # Align horizons
    T_use = min(T, Y_eq_z.shape[1], Y_fire_z.shape[1])
    t = np.arange(T_use)

    # Convert EQ(z) to percent if requested
    if percent:
        ss_val = getattr(model_eq.ss, output)
        if ss_val == 0:
            raise ValueError(f"Steady state {output} is zero.")
        Y_eq_z = 100 * Y_eq_z / ss_val
        y_eq_agg = 100 * np.asarray(model_eq.IRF[output][:T_use]) / ss_val
        y_fire_agg = np.asarray(fire_out['agg_pct'][output][:T_use])
        ylab = "% deviation from steady state"
    else:
        y_eq_agg = np.asarray(model_eq.IRF[output][:T_use])
        y_fire_agg = np.asarray(fire_out['agg'][output][:T_use])
        ylab = "level deviation"

    # ---- per-z grid ----
    Nz = Y_eq_z.shape[0]
    ncols = min(max_cols, Nz)
    nrows = math.ceil(Nz / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 2.9 * nrows),
        sharex=True,
        sharey=False
    )
    axes = np.atleast_1d(axes).flatten()

    for iz in range(Nz):
        ax = axes[iz]
        ax.plot(t, Y_eq_z[iz, :T_use], lw=2.3, label="EQ (heterogeneous beliefs)")
        ax.plot(t, Y_fire_z[iz, :T_use], lw=2.3, ls="--", label="FIRE benchmark")

        ax.axhline(0, ls=":", lw=0.8)
        ax.set_title(f"z = {iz}", fontsize=10)
        ax.grid(alpha=0.3)

        if iz == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")

    for j in range(Nz, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("t")
    fig.supylabel(ylab)
    if title_prefix is None:
        title_prefix = f"{output}: EQ vs FIRE benchmark by productivity"
    fig.suptitle(title_prefix, y=1.02)
    plt.tight_layout()
    plt.show()

    # ---- aggregate plot ----
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(t, y_eq_agg, lw=2.6, label="Aggregate EQ (heterogeneous beliefs)")
    plt.plot(t, y_fire_agg, lw=2.6, ls="--", label="Aggregate FIRE benchmark")
    plt.axhline(0, ls=":", lw=0.8)
    plt.grid(alpha=0.3)
    plt.xlabel("t")
    plt.ylabel(ylab)
    plt.title(f"{output}: aggregate EQ vs FIRE benchmark")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return Y_eq_z[:, :T_use], Y_fire_z[:, :T_use]


import numpy as np
import matplotlib.pyplot as plt
import math

def plot_eq_vs_fire_by_z_and_aggregate(
    model_eq,
    model_fire,
    output="C_hh",
    inputs=("Z","ra","chi"),
    T=40,
    percent=True,
    max_cols=3,
    title_prefix=None,
    use_same_input_paths_from="eq",   # "eq" or "fire" (default "eq" for clean counterfactual)
    agg_from="jac",                  # "jac" or "irf" (see note below)
):
    """
    Compare EQ (sticky/hetero expectations) vs FIRE benchmark.

    Key idea:
      - Use THE SAME input paths (default from model_eq.IRF) for both models
      - Apply each model's per-z Jacobians jac_hh_z to those paths.

    This isolates differences in household decision rules (Jacobians) holding the environment fixed.

    Parameters
    ----------
    use_same_input_paths_from : "eq" (recommended) or "fire"
        If "eq": uses model_eq.IRF[inp] as the input paths for both EQ and FIRE.
        If "fire": uses model_fire.IRF[inp] as the input paths for both (less common).
    agg_from : "jac" or "irf"
        If "jac": aggregate lines are computed by aggregating per-z Jacobian-implied responses (consistent).
        If "irf": aggregate EQ uses model_eq.IRF[output], aggregate FIRE uses model_fire.IRF[output]
                  (this becomes GE vs GE, but the *inputs* may differ across models, so interpret carefully).
    """

    if isinstance(inputs, str):
        inputs = (inputs,)

    # -----------------------
    # Choose input paths
    # -----------------------
    src = model_eq if use_same_input_paths_from == "eq" else model_fire
    if not hasattr(src, "IRF"):
        raise RuntimeError("Chosen source model has no IRF dict.")

    x_paths = {}
    for inp in inputs:
        if inp not in src.IRF:
            raise KeyError(f"Input '{inp}' not in {use_same_input_paths_from}.IRF")
        x_paths[inp] = np.asarray(src.IRF[inp], dtype=float)

    # -----------------------
    # Helper: per-z response from jac_hh_z
    # -----------------------
    def compute_per_z_from_jac(model, T_target):
        # infer dims
        example_key = None
        for inp in inputs:
            k = (output, inp)
            if hasattr(model, "jac_hh_z") and k in model.jac_hh_z:
                example_key = k
                break
        if example_key is None:
            raise KeyError(f"Model missing jac_hh_z for {(output, inputs)}")

        J_ex = model.jac_hh_z[example_key]
        Nz, T_jac, _ = J_ex.shape

        T_use = min(T_target, T_jac, *(len(x_paths[inp]) for inp in inputs))
        Yz = np.zeros((Nz, T_use))

        for inp in inputs:
            k = (output, inp)
            if k not in model.jac_hh_z:
                continue
            Jz = model.jac_hh_z[k]              # (Nz,T,T)
            x = x_paths[inp][:T_use]            # (T,)

            # IMPORTANT: use the same truncation on both dimensions for the convolution
            for iz in range(Nz):
                Yz[iz, :] += (Jz[iz, :T_use, :T_use] @ x)

        return Yz, T_use

    # Compute per-z series
    Y_eq_z, T_use1 = compute_per_z_from_jac(model_eq, T)
    Y_fire_z, T_use2 = compute_per_z_from_jac(model_fire, T)
    T_use = min(T_use1, T_use2)

    Y_eq_z = Y_eq_z[:, :T_use]
    Y_fire_z = Y_fire_z[:, :T_use]
    t = np.arange(T_use)

    # -----------------------
    # Convert to percent if requested
    # -----------------------
    if percent:
        ss_eq = float(getattr(model_eq.ss, output))
        ss_fire = float(getattr(model_fire.ss, output))

        # For comparability, you have two choices:
        # (i) normalize both by *their own* SS (default below)
        # (ii) normalize both by eq SS (commented alternative)
        Y_eq_z_pct = 100 * Y_eq_z / ss_eq
        Y_fire_z_pct = 100 * Y_fire_z / ss_fire

        Y_eq_plot = Y_eq_z_pct
        Y_fire_plot = Y_fire_z_pct
        ylab = "% deviation from steady state"

    else:
        Y_eq_plot = Y_eq_z
        Y_fire_plot = Y_fire_z
        ylab = "level deviation"

    # -----------------------
    # Per-z grid plot
    # -----------------------
    Nz = Y_eq_plot.shape[0]
    ncols = min(max_cols, Nz)
    nrows = math.ceil(Nz / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 2.9 * nrows),
        sharex=True,
        sharey=False
    )
    axes = np.atleast_1d(axes).flatten()

    for iz in range(Nz):
        ax = axes[iz]
        ax.plot(t, Y_eq_plot[iz], lw=2.3, label="EQ (sticky expectations)")
        ax.plot(t, Y_fire_plot[iz], lw=2.3, ls="--", label="FIRE benchmark")

        ax.axhline(0, ls=":", lw=0.8)
        ax.set_title(f"z = {iz}", fontsize=10)
        ax.grid(alpha=0.3)
        if iz == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")

    for j in range(Nz, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("t")
    fig.supylabel(ylab)
    if title_prefix is None:
        title_prefix = f"{output}: EQ vs FIRE benchmark (same input paths from {use_same_input_paths_from})"
    fig.suptitle(title_prefix, y=1.02)
    plt.tight_layout()
    plt.show()

    # -----------------------
    # Aggregate plot
    # -----------------------
    plt.figure(figsize=(7.2, 4.0))

    if agg_from == "jac":
        # Aggregate implied by summing across z.
        # IMPORTANT: whether sum or mu-weighted depends on how your jac_hh_z is stored.
        # In your setup you previously found sum over z matches aggregate jac_hh, so use sum:
        y_eq_agg = Y_eq_plot.sum(axis=0)
        y_fire_agg = Y_fire_plot.sum(axis=0)
        plt.plot(t, y_eq_agg, lw=2.6, label="Aggregate EQ (from per-z jac)")
        plt.plot(t, y_fire_agg, lw=2.6, ls="--", label="Aggregate FIRE (from per-z jac)")

    elif agg_from == "irf":
        # GE IRFs from each model (not necessarily same input paths!)
        if percent:
            ss_eq = float(getattr(model_eq.ss, output))
            ss_fire = float(getattr(model_fire.ss, output))
            y_eq_agg = 100 * np.asarray(model_eq.IRF[output][:T_use]) / ss_eq
            y_fire_agg = 100 * np.asarray(model_fire.IRF[output][:T_use]) / ss_fire
        else:
            y_eq_agg = np.asarray(model_eq.IRF[output][:T_use])
            y_fire_agg = np.asarray(model_fire.IRF[output][:T_use])
        plt.plot(t, y_eq_agg, lw=2.6, label="Aggregate EQ (GE IRF)")
        plt.plot(t, y_fire_agg, lw=2.6, ls="--", label="Aggregate FIRE (GE IRF)")
    else:
        raise ValueError("agg_from must be 'jac' or 'irf'")

    plt.axhline(0, ls=":", lw=0.8)
    plt.grid(alpha=0.3)
    plt.xlabel("t")
    plt.ylabel(ylab)
    plt.title(f"{output}: aggregate EQ vs FIRE benchmark")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return Y_eq_z, Y_fire_z

import matplotlib.pyplot as plt
import numpy as np

def plot_gap_paths(gap_metric, zs=(2,3,5,7), T=21):
    t = np.arange(T)
    plt.figure(figsize=(7,4))
    for z in zs:
        plt.plot(t, gap_metric[z,:T], lw=2, label=f"z={z}")
    plt.axhline(0,color="black",lw=1)
    plt.title("EQ − FIRE gap paths (the object being summed)")
    plt.xlabel("t")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

def plot_eq_vs_fire_by_z_and_aggregate_debug(
    model_eq,
    model_fire,
    output="C_hh",
    inputs=("Z","ra","chi"),
    T=40,
    percent=True,
    max_cols=3,
    title_prefix=None,

    # NEW: which GE paths to feed into both models' Jacobians
    input_source="eq",     # "eq" or "fire"

    # NEW: inspection options
    inspect_z=(0,1,2),      # low-z groups you care about
    inspect_show_channels=True,
    print_hump_diagnostics=True,

    # NEW: plotting toggles
    plot_grid=True,
    plot_aggregate=True,
):
    """
    EQ vs FIRE benchmark using jac_hh_z with the SAME input paths.
    Also saves and inspects the exact series used for the plot, including
    per-channel contributions by z.

    Requirements:
      - model_eq.jac_hh_z[(output, inp)] exists for each inp used
      - model_fire.jac_hh_z[(output, inp)] exists for each inp used
      - model_{source}.IRF[inp] exists for each inp used
      - model_eq.IRF[output] exists if you want aggregate eq IRF overlay

    Returns
    -------
    debug : dict with x_paths, per-z totals and per-z contributions (levels and % if requested)
    """

    if isinstance(inputs, str):
        inputs = (inputs,)

    # -----------------------
    # Pick the input paths
    # -----------------------
    if input_source == "eq":
        src = model_eq
        src_name = "eq"
    elif input_source == "fire":
        src = model_fire
        src_name = "fire"
    else:
        raise ValueError("input_source must be 'eq' or 'fire'")

    x_paths = {}
    for inp in inputs:
        if not hasattr(src, "IRF") or inp not in src.IRF:
            raise KeyError(f"input_source='{src_name}' has no IRF['{inp}']")
        x_paths[inp] = np.asarray(src.IRF[inp], dtype=float).ravel()

    # -----------------------
    # Helper: compute per-z totals + contributions
    # -----------------------
    def per_z_from_jac(model, T_target):
        # infer Nz,T_jac
        example_key = None
        for inp in inputs:
            k = (output, inp)
            if hasattr(model, "jac_hh_z") and k in model.jac_hh_z:
                example_key = k
                break
        if example_key is None:
            raise KeyError(f"Model has no jac_hh_z for output={output} and inputs={inputs}")

        J_ex = model.jac_hh_z[example_key]
        Nz, T_jac, _ = J_ex.shape

        T_use = min(T_target, T_jac, *(len(x_paths[inp]) for inp in inputs))

        contrib = {inp: np.zeros((Nz, T_use)) for inp in inputs}
        total   = np.zeros((Nz, T_use))

        for inp in inputs:
            k = (output, inp)
            if k not in model.jac_hh_z:
                continue
            Jz = np.asarray(model.jac_hh_z[k])  # (Nz,T,T)
            x  = x_paths[inp][:T_use]

            # compute for each z: y_z = Jz[z,:,:] @ x
            for iz in range(Nz):
                contrib[inp][iz, :] = Jz[iz, :T_use, :T_use] @ x
                total[iz, :] += contrib[inp][iz, :]

        return total, contrib, Nz, T_use

    # Compute EQ and FIRE objects (LEVELS)
    Y_eq_z,  Y_eq_contrib,  Nz, T_use = per_z_from_jac(model_eq,  T)
    Y_fi_z,  Y_fi_contrib,  Nz2, T_use2 = per_z_from_jac(model_fire, T)

    if Nz2 != Nz:
        raise ValueError(f"Nz mismatch eq={Nz} fire={Nz2}")
    if T_use2 != T_use:
        T_use = min(T_use, T_use2)
        # truncate all
        Y_eq_z = Y_eq_z[:, :T_use]
        Y_fi_z = Y_fi_z[:, :T_use]
        for inp in inputs:
            Y_eq_contrib[inp] = Y_eq_contrib[inp][:, :T_use]
            Y_fi_contrib[inp] = Y_fi_contrib[inp][:, :T_use]

    t = np.arange(T_use)

    # -----------------------
    # Convert to percent?
    # -----------------------
    if percent:
        ss_val = float(getattr(model_eq.ss, output))
        if abs(ss_val) < 1e-16:
            raise ValueError(f"Steady state {output} is ~0.")
        Y_eq_plot = 100 * Y_eq_z / ss_val
        Y_fi_plot = 100 * Y_fi_z / ss_val
        Y_eq_contrib_plot = {inp: 100 * Y_eq_contrib[inp] / ss_val for inp in inputs}
        Y_fi_contrib_plot = {inp: 100 * Y_fi_contrib[inp] / ss_val for inp in inputs}
        ylab = "% deviation from steady state"
    else:
        Y_eq_plot = Y_eq_z
        Y_fi_plot = Y_fi_z
        Y_eq_contrib_plot = Y_eq_contrib
        Y_fi_contrib_plot = Y_fi_contrib
        ylab = "level deviation"

    # -----------------------
    # Diagnostics: hump / trough timing
    # -----------------------
    def trough_info(series):
        idx = int(np.argmin(series))
        return idx, float(series[idx])

    if print_hump_diagnostics:
        print(f"\n=== Hump diagnostics (input_source='{src_name}', output='{output}', T_use={T_use}) ===")
        for zz in inspect_z:
            if zz < 0 or zz >= Nz:
                continue
            i_eq, v_eq = trough_info(Y_eq_plot[zz])
            i_fi, v_fi = trough_info(Y_fi_plot[zz])
            print(f"z={zz:2d}: EQ trough at t={i_eq:2d} val={v_eq: .4g} | FIRE trough at t={i_fi:2d} val={v_fi: .4g}")

            if inspect_show_channels:
                # which channel drives the FIRE trough? (largest negative contrib at that t)
                tt = i_fi
                vals = {inp: float(Y_fi_contrib_plot[inp][zz, tt]) for inp in inputs}
                # sort by most negative
                ordered = sorted(vals.items(), key=lambda kv: kv[1])
                print("     FIRE contribs at FIRE trough:", ", ".join([f"{k}={v: .4g}" for k,v in ordered]))

    # -----------------------
    # Plot 1: grid EQ vs FIRE totals
    # -----------------------
    if plot_grid:
        ncols = min(max_cols, Nz)
        nrows = math.ceil(Nz / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.2 * ncols, 2.9 * nrows),
            sharex=True, sharey=False
        )
        axes = np.atleast_1d(axes).flatten()

        for iz in range(Nz):
            ax = axes[iz]
            ax.plot(t, Y_eq_plot[iz], lw=2.2, label="EQ")
            ax.plot(t, Y_fi_plot[iz], lw=2.2, ls="--", label="FIRE")
            ax.axhline(0, ls=":", lw=0.8)
            ax.set_title(f"z = {iz}", fontsize=10)
            ax.grid(alpha=0.3)
            if iz == 0:
                ax.legend(frameon=False, fontsize=8)

        for j in range(Nz, len(axes)):
            fig.delaxes(axes[j])

        fig.supxlabel("t")
        fig.supylabel(ylab)
        if title_prefix is None:
            title_prefix = f"{output}: EQ vs FIRE (same input paths from {src_name})"
        fig.suptitle(title_prefix, y=1.02)
        plt.tight_layout()
        plt.show()

    # -----------------------
    # Plot 2: INSPECTION plots for selected low z:
    #   show total + channels for EQ and FIRE
    # -----------------------
    if inspect_z is not None and len(inspect_z) > 0:
        ncols_i = min(len(inspect_z), 3)
        nrows_i = math.ceil(len(inspect_z) / ncols_i)
        fig2, axes2 = plt.subplots(
            nrows_i, ncols_i,
            figsize=(5.2 * ncols_i, 3.6 * nrows_i),
            sharex=True, sharey=False
        )
        axes2 = np.atleast_1d(axes2).flatten()

        for k, zz in enumerate(inspect_z):
            ax = axes2[k]
            ax.plot(t, Y_eq_plot[zz], lw=2.6, color="navy", label="EQ total")
            ax.plot(t, Y_fi_plot[zz], lw=2.6, color="darkorange", ls="--", label="FIRE total")

            if inspect_show_channels:
                # FIRE channels (dotted)
                for inp in inputs:
                    ax.plot(t, Y_fi_contrib_plot[inp][zz], lw=1.8, ls=":", label=f"FIRE from {inp}")
                # EQ channels (dash-dot)
                for inp in inputs:
                    ax.plot(t, Y_eq_contrib_plot[inp][zz], lw=1.6, ls="-.", label=f"EQ from {inp}")

            ax.axhline(0, color="black", lw=1)
            ax.set_title(f"Inspection: z={zz} ({src_name} input paths)")
            ax.grid(alpha=0.25)
            if k == 0:
                ax.legend(frameon=False, fontsize=8, ncol=2)

        for j in range(len(inspect_z), len(axes2)):
            fig2.delaxes(axes2[j])

        fig2.supxlabel("t")
        fig2.supylabel(ylab)
        fig2.suptitle(f"{output}: per-z inspection (totals + channels)", y=1.02)
        plt.tight_layout()
        plt.show()

    # -----------------------
    # Plot 3: aggregate EQ IRF vs aggregate FIRE-implied (optional)
    # Note: FIRE-implied aggregate here is NOT an equilibrium IRF unless you built it that way.
    # We'll just plot mean across z as a visual check unless you provide weights.
    # -----------------------
    if plot_aggregate:
        fig3, ax3 = plt.subplots(figsize=(7.2, 4.0))

        # If you want the true aggregate EQ IRF:
        if hasattr(model_eq, "IRF") and output in model_eq.IRF:
            y_eq_agg_true = np.asarray(model_eq.IRF[output]).ravel()[:T_use]
            if percent:
                y_eq_agg_true = 100 * y_eq_agg_true / float(getattr(model_eq.ss, output))
            ax3.plot(t, y_eq_agg_true, lw=3.0, color="gray", label="EQ aggregate IRF (true)")

        # A simple unweighted average across z (not necessarily meaningful as an aggregate)
        ax3.plot(t, np.mean(Y_eq_plot, axis=0), lw=2.4, color="navy", label="EQ avg across z")
        ax3.plot(t, np.mean(Y_fi_plot, axis=0), lw=2.4, color="darkorange", ls="--", label="FIRE avg across z")

        ax3.axhline(0, ls=":", lw=0.8)
        ax3.grid(alpha=0.3)
        ax3.set_xlabel("t")
        ax3.set_ylabel(ylab)
        ax3.set_title(f"{output}: aggregate-style comparison (diagnostic)")
        ax3.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    # -----------------------
    # Save everything for external inspection
    # -----------------------
    debug = {
        "input_source": src_name,
        "inputs": inputs,
        "output": output,
        "t": t.copy(),
        "x_paths": {k: x_paths[k][:T_use].copy() for k in inputs},  # the exact paths used
        "Y_eq_levels": Y_eq_z[:, :T_use].copy(),
        "Y_fire_levels": Y_fi_z[:, :T_use].copy(),
        "Y_eq_contrib_levels": {inp: Y_eq_contrib[inp][:, :T_use].copy() for inp in inputs},
        "Y_fire_contrib_levels": {inp: Y_fi_contrib[inp][:, :T_use].copy() for inp in inputs},
        "Y_eq_plot": Y_eq_plot[:, :T_use].copy(),
        "Y_fire_plot": Y_fi_plot[:, :T_use].copy(),
        "Y_eq_contrib_plot": {inp: Y_eq_contrib_plot[inp][:, :T_use].copy() for inp in inputs},
        "Y_fire_contrib_plot": {inp: Y_fi_contrib_plot[inp][:, :T_use].copy() for inp in inputs},
    }

    return debug

import numpy as np

def inspect_discounted_gap(gap_metric, beta=0.99, early=4, mid=12):
    """
    gap_metric: array (Nz,T) in the units you are summing (could be ratio or pp)
    """
    gap_metric = np.asarray(gap_metric)
    Nz, T = gap_metric.shape
    disc = beta ** np.arange(T)

    contrib = np.abs(gap_metric) * disc[None, :]
    total = contrib.sum(axis=1)

    early_sum = contrib[:, :early].sum(axis=1)
    mid_sum   = contrib[:, :mid].sum(axis=1)

    print("z | total Σ | early share | first-12 share | abs gap t=0..3")
    for z in range(Nz):
        es = early_sum[z] / total[z] if total[z] > 0 else np.nan
        ms = mid_sum[z] / total[z] if total[z] > 0 else np.nan
        print(f"{z:1d} | {total[z]:8.4f} | {es:10.3f} | {ms:12.3f} | {np.abs(gap_metric[z,:4])}")

    return total, early_sum, mid_sum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _infer_Nz_Tjac_from_jac_hh_z(model, output, inputs):
    example_key = None
    for inp in inputs:
        k = (output, inp)
        if hasattr(model, "jac_hh_z") and k in model.jac_hh_z:
            example_key = k
            break
    if example_key is None:
        raise KeyError(f"No jac_hh_z found for output={output} with inputs={inputs}")
    J = model.jac_hh_z[example_key]
    Nz, T_jac, _ = J.shape
    return Nz, T_jac


def _perz_from_jac_hh_z(model, output, inputs, x_paths, T):
    Nz, T_jac = _infer_Nz_Tjac_from_jac_hh_z(model, output, inputs)
    T_use = min(T, T_jac, *(len(x_paths[inp]) for inp in inputs))

    Yz = np.zeros((Nz, T_use))
    for inp in inputs:
        k = (output, inp)
        if k not in model.jac_hh_z:
            continue
        Jz = model.jac_hh_z[k]  # (Nz,T,T)
        x  = np.asarray(x_paths[inp])[:T_use]
        for iz in range(Nz):
            Yz[iz] += Jz[iz, :T_use, :T_use] @ x
    return Yz, T_use


def _z_moments_from_ss(model):
    par, ss = model.par, model.ss
    D = np.asarray(ss.D)      # (Nfix,Nz,Na)
    a = np.asarray(ss.a)
    c = np.asarray(ss.c)
    a_grid = np.asarray(par.a_grid)

    mass_z = D.sum(axis=(0,2))
    mass_z = np.maximum(mass_z, 1e-32)

    Abar_z = (a * D).sum(axis=(0,2)) / mass_z
    Css_z  = (c * D).sum(axis=(0,2)) / mass_z
    share_a0_z = D[:,:,0].sum(axis=0) / mass_z

    da = (a_grid[1:] - a_grid[:-1])[None,None,:]
    mpc_micro = (c[:,:,1:] - c[:,:,:-1]) / np.maximum((1.0 + ss.ra) * da, 1e-32)
    w = D[:,:,:-1]
    MPC_z = (mpc_micro * w).sum(axis=(0,2)) / np.maximum(w.sum(axis=(0,2)), 1e-32)

    return mass_z, Abar_z, MPC_z, share_a0_z, Css_z


def diagnose_eq_vs_fire_gap_by_z(
    model_eq,
    model_fire,
    output="C_hh",
    inputs=("Z","ra","chi"),
    T=40,
    percent_own_ss=True,
    gap_window=(0, 20),
    discount_beta=None,
    gap_metric="rel_to_fire",   # "pp" | "rel_to_fire" | "rel_to_eq" | "rel_to_scale_fire"
    eps_denom=1e-9,
    make_plot=True,
    # NEW plotting controls
    separate_figures=True,      # True -> two separate figures
    ylabel_short=True,          # True -> "Gap metric" instead of long string
    left_margin=0.16,           # extra space for y-label
    ylab_fontsize=11,
    title_fontsize=13,
):
    """
    Compute EQ vs FIRE wedge by z under SAME GE input paths (from model_eq).

    gap_metric:
      - "pp":         gap in percentage points
      - "rel_to_fire": (EQ-FIRE)/(|FIRE|+eps)  [time-varying denom]
      - "rel_to_eq":   (EQ-FIRE)/(|EQ|+eps)    [time-varying denom]
      - "rel_to_scale_fire": (EQ-FIRE)/scale_z where scale_z = |min_t FIRE(z,t)| on the window
                             (stable denom, recommended for plotting)
    """

    # --- build x paths from EQ model ---
    x_paths = {inp: np.asarray(model_eq.IRF[inp]).ravel() for inp in inputs if inp in model_eq.IRF}
    missing = [inp for inp in inputs if inp not in x_paths]
    if len(missing) > 0:
        raise KeyError(f"model_eq.IRF missing inputs: {missing}")

    # --- implied per-z responses from jac_hh_z (LEVELS) ---
    Y_eq_z_levels, T_use = _perz_from_jac_hh_z(model_eq, output, inputs, x_paths, T)
    Y_fire_z_levels, _   = _perz_from_jac_hh_z(model_fire, output, inputs, x_paths, T_use)

    # --- put both into % deviation from SS (either own-SS or agg-SS) ---
    if percent_own_ss:
        _, _, _, _, Css_z = _z_moments_from_ss(model_eq)
        Y_eq_z = 100.0 * (Y_eq_z_levels / np.maximum(Css_z[:,None], 1e-32))
        Y_fire_z = 100.0 * (Y_fire_z_levels / np.maximum(Css_z[:,None], 1e-32))
        base_units = "percent (own-SS)"
    else:
        ss_val = float(getattr(model_eq.ss, output))
        Y_eq_z = 100.0 * (Y_eq_z_levels / np.maximum(ss_val, 1e-32))
        Y_fire_z = 100.0 * (Y_fire_z_levels / np.maximum(ss_val, 1e-32))
        base_units = "percent (agg-SS)"

    # --- window for statistics ---
    t0, t1 = gap_window
    t0 = int(max(0, t0))
    t1 = int(min(T_use-1, t1))
    sl = slice(t0, t1+1)

    # --- compute GAP series according to requested metric ---
    gap_pp = (Y_eq_z - Y_fire_z)  # in percentage points

    if gap_metric == "pp":
        gap = gap_pp
        gap_units = "pp"
    elif gap_metric == "rel_to_fire":
        denom = np.abs(Y_fire_z) + eps_denom
        gap = gap_pp / denom
        gap_units = "(EQ−FIRE)/|FIRE|"
    elif gap_metric == "rel_to_eq":
        denom = np.abs(Y_eq_z) + eps_denom
        gap = gap_pp / denom
        gap_units = "(EQ−FIRE)/|EQ|"
    elif gap_metric == "rel_to_scale_fire":
        scale_z = np.abs(np.minimum(Y_fire_z[:, sl].min(axis=1), 0.0))  # |min| (nonnegative)
        scale_z = np.maximum(scale_z, eps_denom)
        gap = gap_pp / scale_z[:, None]
        gap_units = "(EQ−FIRE)/|min FIRE| (window)"
    else:
        raise ValueError(f"Unknown gap_metric='{gap_metric}'")

    # --- discounting for integrated measures (applied to |gap|) ---
    if discount_beta is None:
        discount_beta = getattr(getattr(model_eq, "par", object()), "beta", None)
    if discount_beta is None:
        disc = np.ones(t1-t0+1)
    else:
        disc = np.array([discount_beta**k for k in range(t1-t0+1)], dtype=float)

    trough_gap = gap[:, sl].min(axis=1)
    peak_gap   = gap[:, sl].max(axis=1)
    int_abs_gap = (np.abs(gap[:, sl]) * disc[None,:]).sum(axis=1)
    int_signed_gap = (gap[:, sl] * disc[None,:]).sum(axis=1)

    # --- z moments for “why” panel ---
    mass_z, Abar_z, MPC_z, share_a0_z, Css_z = _z_moments_from_ss(model_eq)

    df = pd.DataFrame({
        "z": np.arange(gap.shape[0]),
        "mass(z)": mass_z,
        "Abar(z)": Abar_z,
        "MPC(z)": MPC_z,
        "share_a0(z)": share_a0_z,
        "trough_gap": trough_gap,
        "peak_gap": peak_gap,
        "int_abs_gap": int_abs_gap,
        "int_signed_gap": int_signed_gap,
    })
    df["rank_int_abs"] = df["int_abs_gap"].rank(ascending=False, method="dense").astype(int)
    df["rank_abs_trough"] = df["trough_gap"].abs().rank(ascending=False, method="dense").astype(int)

    # =========================
    # PLOTS (SEPARATE FIGURES)
    # =========================
    if make_plot:
        z = df["z"].to_numpy()

        # -------- Figure 1: gap measures --------
        fig1, ax = plt.subplots(figsize=(7.6, 4.6))

        ax.plot(z, df["trough_gap"], marker="o", lw=2.3, color="black",
                label=f"Trough gap [{t0},{t1}]")
        ax.plot(z, df["int_abs_gap"], marker="s", lw=2.3, color="#1f77b4",
                label=f"Σ|gap| (disc) [{t0},{t1}]")

        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(z)
        ax.set_xlabel("Productivity state z")

        # ✅ shorter y-label + no clipping
        if ylabel_short:
            ax.set_ylabel("Gap metric", fontsize=ylab_fontsize)
        else:
            ax.set_ylabel(f"Gap metric: {gap_units}", fontsize=ylab_fontsize)

        ax.set_title(f"EQ–FIRE deviation by z  (base series in {base_units})", fontsize=title_fontsize)
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)

        # reserve space for label
        fig1.tight_layout()
        fig1.subplots_adjust(left=left_margin)
        plt.show()

        # -------- Figure 2: “why” moments --------
        fig2, ax2 = plt.subplots(figsize=(7.6, 4.6))

        ax2.plot(z, df["Abar(z)"], marker="o", lw=2.3, color="forestgreen", label="Mean assets A(z)")
        ax2b = ax2.twinx()
        ax2b.plot(z, df["MPC(z)"], marker="s", lw=2.3, color="#1f77b4", label="MPC(z)")
        ax2b.plot(z, df["share_a0(z)"], marker="^", lw=2.3, ls="--", color="firebrick",
                  label="P(a=0|z)")

        ax2.set_xticks(z)
        ax2.set_xlabel("Productivity state z")
        ax2.set_ylabel("Assets", fontsize=ylab_fontsize)
        ax2b.set_ylabel("Liquidity / cash-flow sensitivity", fontsize=ylab_fontsize)
        ax2.set_title("Where is the forward-looking margin large?", fontsize=title_fontsize)

        L1, lab1 = ax2.get_legend_handles_labels()
        L2, lab2 = ax2b.get_legend_handles_labels()
        ax2b.legend(L1+L2, lab1+lab2, frameon=True, loc="best")

        ax2.grid(alpha=0.25)
        fig2.tight_layout()
        fig2.subplots_adjust(left=left_margin)
        plt.show()

    out = {
        "gap": gap,
        "gap_pp": gap_pp,
        "Y_eq_z_percent": Y_eq_z,
        "Y_fire_z_percent": Y_fire_z,
        "T_use": T_use,
        "gap_units": gap_units,
        "base_units": base_units,
        "window": (t0, t1),
    }
    return df, out


import numpy as np
import pandas as pd

def table_gap_short_mid_total_3models(
    model_base,
    model_homo,
    model_hetero,
    model_fire,                  # FIRE benchmark model (usually = model_base)
    output="C_hh",
    inputs=("Z","ra","chi"),
    T=40,
    percent_own_ss=True,
    full_window=(0, 20),         # (t_start, t_end)
    split_t=6,                   # short = [t0, split_t], mid = [split_t+1, t1]
    discount_beta=None,
    gap_metric="rel_to_scale_fire",
    eps_denom=1e-9,
    rank_by="total",             # "short"|"mid"|"total"
):
    """
    Builds a per-z table of discounted |EQ-FIRE| gaps for three models:
      baseline, homo sticky, hetero sticky.

    The gap is computed per model using that model's GE input paths (IRF paths),
    but the same FIRE jacobians (model_fire) for the FIRE benchmark.

    Requires helper functions already in your notebook:
      - _perz_from_jac_hh_z(model, output, inputs, x_paths, T) -> (Nz,Tuse), Tuse
      - _z_moments_from_ss(model) -> (mass_z, Abar_z, MPC_z, share_a0_z, Css_z)
    """

    def _get_beta(m):
        b = discount_beta
        if b is None:
            b = getattr(getattr(m, "par", object()), "beta", None)
        return b

    def _make_disc(beta, t0, t1):
        if beta is None:
            return np.ones(t1 - t0 + 1)
        return np.array([beta**k for k in range(t1 - t0 + 1)], dtype=float)

    def _compute_gap_measures(model_eq, label):
        # build paths from this EQ model
        x_paths = {inp: np.asarray(model_eq.IRF[inp]).ravel() for inp in inputs}
        # implied per-z in levels
        Y_eq_levels, T_use = _perz_from_jac_hh_z(model_eq, output, inputs, x_paths, T)
        Y_fire_levels, _   = _perz_from_jac_hh_z(model_fire, output, inputs, x_paths, T_use)

        # percent conversion
        if percent_own_ss:
            _, _, _, _, Css_z = _z_moments_from_ss(model_eq)
            Y_eq = 100.0 * (Y_eq_levels / np.maximum(Css_z[:, None], 1e-32))
            Y_fire = 100.0 * (Y_fire_levels / np.maximum(Css_z[:, None], 1e-32))
        else:
            ss_val = float(getattr(model_eq.ss, output))
            Y_eq = 100.0 * (Y_eq_levels / np.maximum(ss_val, 1e-32))
            Y_fire = 100.0 * (Y_fire_levels / np.maximum(ss_val, 1e-32))

        # window
        t0, t1 = full_window
        t0 = int(max(0, t0))
        t1 = int(min(T_use - 1, t1))
        if t1 < t0:
            raise ValueError("full_window invalid after clipping to available T_use")

        # split
        split = int(split_t)
        split = min(max(split, t0), t1)
        sl_short = slice(t0, split + 1)
        sl_mid   = slice(split + 1, t1 + 1) if split + 1 <= t1 else slice(t1, t1)  # empty safe

        # gap
        gap_pp = (Y_eq - Y_fire)

        if gap_metric == "pp":
            gap = gap_pp
        elif gap_metric == "rel_to_fire":
            gap = gap_pp / (np.abs(Y_fire) + eps_denom)
        elif gap_metric == "rel_to_eq":
            gap = gap_pp / (np.abs(Y_eq) + eps_denom)
        elif gap_metric == "rel_to_scale_fire":
            # stable denom: |min FIRE| on full window
            scale_z = np.abs(np.minimum(Y_fire[:, t0:t1+1].min(axis=1), 0.0))
            scale_z = np.maximum(scale_z, eps_denom)
            gap = gap_pp / scale_z[:, None]
        else:
            raise ValueError(f"Unknown gap_metric={gap_metric}")

        # discount
        beta = _get_beta(model_eq)
        disc_full  = _make_disc(beta, t0, t1)
        disc_short = _make_disc(beta, t0, split)
        disc_mid   = _make_disc(beta, split + 1, t1) if split + 1 <= t1 else np.array([])

        # measures (discounted sum of abs gap)
        short_val = (np.abs(gap[:, sl_short]) * disc_short[None, :]).sum(axis=1)
        if disc_mid.size == 0:
            mid_val = np.zeros(gap.shape[0])
        else:
            mid_val = (np.abs(gap[:, sl_mid]) * disc_mid[None, :]).sum(axis=1)
        total_val = (np.abs(gap[:, t0:t1+1]) * disc_full[None, :]).sum(axis=1)

        return {
            "label": label,
            "T_use": T_use,
            "t0": t0, "t1": t1, "split": split,
            "short": short_val,
            "mid": mid_val,
            "total": total_val,
        }

    # compute for each model
    res_base   = _compute_gap_measures(model_base,   "baseline")
    res_homo   = _compute_gap_measures(model_homo,   "homo_sticky")
    res_hetero = _compute_gap_measures(model_hetero, "hetero_sticky")

    # use z-moments from baseline for mass(z) etc (or from hetero; just pick one consistently)
    mass_z, _, _, _, _ = _z_moments_from_ss(model_base)
    Nz = mass_z.size
    z = np.arange(Nz)

    df = pd.DataFrame({"z": z, "mass(z)": mass_z})

    # attach columns
    def add_cols(prefix, res):
        df[f"{prefix}_short"] = res["short"]
        df[f"{prefix}_mid"]   = res["mid"]
        df[f"{prefix}_total"] = res["total"]

    add_cols("base",   res_base)
    add_cols("homo",   res_homo)
    add_cols("hetero", res_hetero)

    # ranks (by chosen metric, on hetero-vs-FIRE by default; change if you want)
    rank_col = f"hetero_{rank_by}"
    df["rank_abs"] = df[rank_col].abs().rank(ascending=False, method="dense").astype(int)

    meta = {
        "window": (res_base["t0"], res_base["t1"]),
        "split_t": res_base["split"],
        "gap_metric": gap_metric,
        "percent_own_ss": percent_own_ss,
        "rank_by": rank_by,
    }

    return df, meta


def print_gap_table_latex_compact(df, meta, floatfmt="%.3f"):
    """
    Compact LaTeX print helper.
    """
    t0, t1 = meta["window"]
    split = meta["split_t"]
    cap = (f"Discounted |EQ−FIRE| gap by productivity state z "
           f"(short: [{t0},{split}], mid: [{split+1},{t1}], total: [{t0},{t1}]).")
    lab = "tab:eq_fire_gap_short_mid_total"

    cols = [
        "z","mass(z)",
        "base_short","base_mid","base_total",
        "homo_short","homo_mid","homo_total",
        "hetero_short","hetero_mid","hetero_total",
        "rank_abs"
    ]
    out = df[cols].copy()

    print(out.to_latex(index=False, float_format=floatfmt,
                       caption=cap, label=lab))
    return out

import pandas as pd
import numpy as np

def rotate_gap_table_z_columns(
    df_gap,
    include_mass_row=True,
    model_prefixes=("base", "homo", "hetero"),
    parts=("short", "mid", "total"),
    z_col="z",
    mass_col="mass(z)",
    float_format="%.3f",
    caption="Discounted |EQ−FIRE| gap by z (z as columns).",
    label="tab:eq_fire_gap_zcols",
    print_latex=True,
):
    """
    Convert df with columns like base_short, base_mid, base_total, ... into a table with:
      - columns: z
      - rows: (optional) mass(z), then base_short, base_mid, base_total, homo_short, ...
    """

    df = df_gap.copy().sort_values(z_col)
    z_vals = df[z_col].to_numpy().astype(int)

    # Build row order
    row_keys = []
    if include_mass_row:
        row_keys.append(mass_col)

    for m in model_prefixes:
        for p in parts:
            row_keys.append(f"{m}_{p}")

    # Create matrix: rows x z
    mat = []
    row_names = []
    for key in row_keys:
        if key == mass_col:
            values = df[mass_col].to_numpy()
            row_names.append("mass(z)")
        else:
            if key not in df.columns:
                raise KeyError(f"Missing column '{key}' in df_gap. Available: {list(df.columns)}")
            values = df[key].to_numpy()
            # nicer row label
            m, p = key.split("_", 1)
            pretty = {
                "base": "baseline",
                "homo": "homo sticky",
                "hetero": "hetero sticky",
            }.get(m, m)
            row_names.append(f"{pretty}: {p}")

        mat.append(values)

    out = pd.DataFrame(mat, index=row_names, columns=[f"z={z}" for z in z_vals])

    # Print LaTeX if requested
    if print_latex:
        print(out.to_latex(
            index=True,
            float_format=float_format,
            caption=caption,
            label=label
        ))

    return out

import numpy as np
import matplotlib.pyplot as plt

def show_inputs(model, Tshow=40):
    for k in ["ra","Z","chi","C_hh"]:
        if k in model.IRF:
            x = np.asarray(model.IRF[k]).ravel()
            print(k, x[:10], " ...  min@",
                  int(np.argmin(x[:Tshow])), "max@",
                  int(np.argmax(x[:Tshow])))
            plt.plot(x[:Tshow], label=k)
    plt.axhline(0,color="k",lw=1)
    plt.legend()
    plt.title("Input/Output IRFs (first Tshow)")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def assets_by_z_steady_state(model):
    """
    Compute steady-state asset holdings by productivity state z.

    Returns
    -------
    mass_z : (Nz,)         population mass in each z
    Abar_z : (Nz,)         mean assets conditional on z: E[a | z]
    Alevel_z : (Nz,)       level contribution to aggregate assets from each z: mass_z * Abar_z
    share_z : (Nz,)        share of aggregate assets held by each z: Alevel_z / A_hh
    """
    ss, par = model.ss, model.par
    D = ss.D  # (Nfix, Nz, Na)
    a = ss.a  # (Nfix, Nz, Na) policy/level assets on grid

    # mass by z
    mass_z = D.sum(axis=(0, 2))  # sum over i_fix and a

    # level assets held by each z (sum a * D over i_fix and a)
    Alevel_z = (a * D).sum(axis=(0, 2))  # (Nz,)

    # mean assets conditional on z
    Abar_z = Alevel_z / np.maximum(mass_z, 1e-32)

    # shares in aggregate assets
    A_hh = getattr(ss, "A_hh", Alevel_z.sum())
    share_z = Alevel_z / np.maximum(A_hh, 1e-32)

    return mass_z, Abar_z, Alevel_z, share_z


def plot_assets_by_z_ss(model):
    mass_z, Abar_z, Alevel_z, share_z = assets_by_z_steady_state(model)

    Nz = model.par.Nz
    zgrid = np.arange(Nz)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(zgrid, mass_z)
    axes[0].set_title("Mass by z")
    axes[0].set_xlabel("z index")
    axes[0].set_ylabel("mass")

    axes[1].bar(zgrid, Abar_z)
    axes[1].set_title("Mean assets E[a|z] in SS")
    axes[1].set_xlabel("z index")
    axes[1].set_ylabel("assets (level)")

    axes[2].bar(zgrid, 100 * share_z)
    axes[2].set_title("Share of aggregate assets by z (SS)")
    axes[2].set_xlabel("z index")
    axes[2].set_ylabel("percent of A_hh")

    plt.tight_layout()
    plt.show()


def assets_by_z_irf_from_policy_irf(model, shockname, polname="a"):
    """
    Compute z-specific asset IRFs (impact/within-z component) from stored policy IRFs.

    Requires:
      model.IRF['pols'][(polname, shockname)] exists with shape (Nfix, Nz, Na, T).

    Returns
    -------
    dAbar_z_t : (Nz, T)    mean asset change in each z (within-z, holding D fixed at SS)
    dAlevel_z_t : (Nz, T)  level contribution to aggregate ΔA from each z: sum_a,i_fix dpol * D_ss
    """
    ss, par = model.ss, model.par
    Dss = ss.D  # (Nfix, Nz, Na)
    mass_z = Dss.sum(axis=(0, 2))  # (Nz,)

    IRFpol = model.IRF["pols"][(polname, shockname)]  # (Nfix, Nz, Na, T)
    assert IRFpol.shape[:3] == Dss.shape, f"Shape mismatch: IRFpol {IRFpol.shape}, Dss {Dss.shape}"

    # level contribution by z over time: sum_{i_fix,a} d a(i,z,a,t) * Dss(i,z,a)
    dAlevel_z_t = (IRFpol * Dss[..., None]).sum(axis=(0, 2))  # (Nz, T)

    # conditional mean change by z: divide by mass_z
    dAbar_z_t = dAlevel_z_t / np.maximum(mass_z[:, None], 1e-32)

    return dAbar_z_t, dAlevel_z_t


def plot_assets_by_z_contribution(model, shockname, T=40):
    """
    Plot which z-groups contribute to aggregate ΔA (within-z/impact component).
    """
    dAbar_z_t, dAlevel_z_t = assets_by_z_irf_from_policy_irf(model, shockname, polname="a")
    Nz = model.par.Nz
    tgrid = np.arange(min(T, model.par.T))

    # Plot level contributions (these add up to the within-z part of aggregate ΔA)
    fig, ax = plt.subplots(figsize=(8, 4))
    for z in range(Nz):
        ax.plot(tgrid, dAlevel_z_t[z, :len(tgrid)], label=f"z={z}", lw=1.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_title(f"ΔA level contribution by z (within-z impact), shock={shockname}")
    ax.set_xlabel("t")
    ax.set_ylabel("ΔA (level units)")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Also show percent contribution shares at each t (optional)
    total = dAlevel_z_t[:, :len(tgrid)].sum(axis=0)  # (T,)
    share = dAlevel_z_t[:, :len(tgrid)] / np.maximum(total[None, :], 1e-32)

    fig, ax = plt.subplots(figsize=(8, 4))
    for z in range(Nz):
        ax.plot(tgrid, 100 * share[z], label=f"z={z}", lw=1.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_title(f"Percent share of ΔA by z (within-z impact), shock={shockname}")
    ax.set_xlabel("t")
    ax.set_ylabel("share of total ΔA (%)")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd

def table_C_impact_by_z_from_perz_decomp(
    model,
    inputs=("Z", "ra", "chi"),
    output="C_hh",
    T=None,
    per_100bp=True,
    agg_mode="auto",
    ):
    """
    Impact (t=0) consumption response by current z-group using the SAME objects
    as your per-z IRF plots:
        - per-z contributions from decompose_C_by_z_and_input_norm
        - normalization by own steady state via compute_C_hh_z_ss

    Produces a df compatible with your downstream diagnostics:
      - includes mass(z), C_ss(z) as conditional means within z,
      - includes dC0_total in LEVELS (mean within z),
      - includes %dC0_total and channel % contributions.

    Notes
    -----
    - 'mass(z)' is computed from ss.D (stationary distribution).
    - 'C_ss(z)' is conditional mean consumption within z using ss.D and ss.c.
      This is the right denominator for your contrib_to_agg_C_impact_by_z().
    - '%dC0_*' uses the SAME C_ss(z) denominator, so it's consistent with your table style.
    """

    par, ss = model.par, model.ss
    D = np.asarray(ss.D)      # (Nfix, Nz, Na) in your model
    c_ss = np.asarray(ss.c)

    # --- get Nz and T ---
    Nz = D.shape[1]  # assumes (Nfix, Nz, Na)
    if T is None:
        # use model horizon if present, else infer from IRF length
        T = getattr(par, "T", len(model.IRF[output]))

    # --- mass(z) and C_ss(z) as conditional means within z ---
    mass_z = D.sum(axis=(0, 2))  # (Nz,)
    Css_z = (c_ss * D).sum(axis=(0, 2)) / np.maximum(mass_z, 1e-32)  # conditional mean within z

    # --- per-z IRF decomposition in LEVELS ---
    (C_z, C_z_contrib, C_agg, C_agg_contrib,
     agg_mode_used, C_resid, C_dist_term, dist_key) = decompose_C_by_z_and_input_norm(
        model=model,
        inputs=inputs,
        output=output,
        T=T,
        check_consistency=False,
        agg_mode=agg_mode,
        include_distribution_residual=False,  # impact table: focus on policy-only pieces
    )

    # C_z is (Nz, T_eff) in levels; take impact
    T_eff = C_z.shape[1]
    if T_eff < 1:
        raise RuntimeError("Empty per-z IRF array returned by decomposition.")

    dC0_tot_z = C_z[:, 0].copy()

    # channels (allow missing chi)
    dC0_ra_z  = C_z_contrib.get("ra",  np.zeros_like(C_z))[:, 0]
    dC0_Z_z   = C_z_contrib.get("Z",   np.zeros_like(C_z))[:, 0]
    dC0_chi_z = C_z_contrib.get("chi", np.zeros_like(C_z))[:, 0]

    # percent impact relative to conditional mean C_ss(z)
    def pct(x):
        return 100.0 * x / np.maximum(Css_z, 1e-32)

    df = pd.DataFrame({
        "z": np.arange(Nz),
        "mass(z)": mass_z,
        "C_ss(z)": Css_z,
        "dC0_total": dC0_tot_z,
        "%dC0_total": pct(dC0_tot_z),
        "%dC0_from_ra": pct(dC0_ra_z),
        "%dC0_from_Z": pct(dC0_Z_z),
        "%dC0_from_chi": pct(dC0_chi_z),
    })

    # per 100bp scaling (using impact ra[0] from model.IRF)
    if per_100bp:
        dra = np.asarray(model.IRF.get("ra", np.zeros(T))).ravel()
        dra0 = float(dra[0]) if dra.size > 0 else 0.0
        if abs(dra0) > 1e-12:
            scale = 0.01 / dra0
            df["%dC0_total per 100bp ra"] = df["%dC0_total"] * scale
        else:
            df["%dC0_total per 100bp ra"] = np.nan

    # rank by absolute impact percent
    df["rank_abs"] = df["%dC0_total"].abs().rank(ascending=False, method="dense").astype(int)

    return df, Css_z

import numpy as np
import matplotlib.pyplot as plt

def plot_key_irfs_clean(model, T=40, extra=("pi","Y","L","i","r"),
                       title="Model IRFs",
                       ylabel="Deviation from steady state",
                       ncols=3,
                       figsize_per_col=4.2,
                       figsize_per_row=2.8):
    """
    Clean, robust IRF grid plot with ONE shared y-label that never overlaps.

    Key idea: do NOT use suptitle/supylabel/tight_layout magic.
    Instead, reserve margins explicitly with subplots_adjust and place text with fig.text.

    Parameters
    ----------
    model : object with model.IRF dict
    T : int, horizon
    extra : tuple, extra variables to try
    title : str, figure title
    ylabel : str, shared y-axis label
    ncols : int, number of columns
    """

    wanted = ["ra", "Z", "B", "chi"] + list(extra)
    available = [
        v for v in wanted
        if hasattr(model, "IRF") and v in model.IRF and np.any(np.isfinite(np.asarray(model.IRF[v])))
    ]
    if len(available) == 0:
        raise RuntimeError("No requested IRF series found in model.IRF.")

    n = len(available)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        squeeze=False
    )

    # ---- HARD-CODED MARGINS (robust) ----
    # left space for shared y-label, top space for title, reasonable gaps between plots
    left, right, bottom, top = 0.12, 0.98, 0.10, 0.88
    wspace, hspace = 0.30, 0.55
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)

    # Title and shared y-label placed in the reserved margins
    fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=14)
    fig.text(0.04, 0.5, ylabel, ha="center", va="center", rotation="vertical", fontsize=12)

    # Plot panels
    for idx, var in enumerate(available):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        series = np.asarray(model.IRF[var]).ravel()[:T]
        ax.plot(np.arange(series.size), series, lw=2.3, color="navy")
        ax.axhline(0, color="black", lw=1)
        ax.set_title(var, fontsize=12)
        ax.set_xlabel("quarters")
        ax.grid(alpha=0.25)

        # no per-panel y-label
        ax.set_ylabel("")

    # Turn off unused axes
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.show()
    return available


import numpy as np
import matplotlib.pyplot as plt

def plot_key_irfs_clean_two_models(
    model_hetero,
    model_homo,
    model_base,
    T=40,
    extra=("pi","Y","i","r"),
    title="IRFs: heterogeneous vs homogeneous sticky expectations",
    ylabel="Deviation from steady state",
    ncols=3,
    figsize_per_col=4.2,
    figsize_per_row=2.8,
    colors=("forestgreen", "navy", "red"),
    labels=("hetero sticky", "homo sticky", "base"),
    ls=("-", "--", "-."),
    force_union=True,   # if True: plot union of available vars across the two models
):
    """
    Same layout as plot_key_irfs_clean, but overlays TWO models in each subplot.

    Parameters
    ----------
    model_hetero, model_homo : objects with .IRF dict
    force_union : if True, plot variables available in either model
                  if False, plot only variables available in BOTH models
    """

    wanted = ["C_hh","A_hh", "ra", "Z", "chi"] + list(extra)

    def _avail(m):
        out = []
        if not hasattr(m, "IRF"):
            return out
        for v in wanted:
            if v in m.IRF:
                arr = np.asarray(m.IRF[v])
                if np.any(np.isfinite(arr)):
                    out.append(v)
        return out

    avail1 = set(_avail(model_hetero))
    avail2 = set(_avail(model_homo))
    avail3 = set(_avail(model_base))

    if force_union:
        available = [v for v in wanted if (v in avail1 or v in avail2 or v in avail3)]
    else:
        available = [v for v in wanted if (v in avail1 and v in avail2 and v in avail3)]

    if len(available) == 0:
        raise RuntimeError("No requested IRF series found in either model.IRF.")

    n = len(available)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        squeeze=False
    )

    # ---- hard-coded margins (robust) ----
    left, right, bottom, top = 0.12, 0.98, 0.10, 0.88
    wspace, hspace = 0.30, 0.55
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)

    fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=14)
    fig.text(0.04, 0.5, ylabel, ha="center", va="center", rotation="vertical", fontsize=12)

    def _get_series(m, var):
        if (not hasattr(m, "IRF")) or (var not in m.IRF):
            return None
        s = np.asarray(m.IRF[var]).ravel()
        if s.size == 0 or not np.any(np.isfinite(s)):
            return None
        return s[:T]

    # plot panels
    for idx, var in enumerate(available):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        s1 = _get_series(model_hetero, var)
        s2 = _get_series(model_homo, var)
        s3 = _get_series(model_base, var)

        # x-axis length chosen by max available length
        maxT = 0
        if s1 is not None: maxT = max(maxT, s1.size)
        if s2 is not None: maxT = max(maxT, s2.size)
        if s3 is not None: maxT = max(maxT, s3.size)
        x = np.arange(maxT)

        if s1 is not None:
            ax.plot(np.arange(s1.size), s1, lw=2.3, color=colors[0], ls=ls[0], label=labels[0])
        if s2 is not None:
            ax.plot(np.arange(s2.size), s2, lw=2.3, color=colors[1], ls=ls[1], label=labels[1])
        if s3 is not None:
            ax.plot(np.arange(s3.size), s3, lw=2.3, color=colors[2], ls=ls[2], label=labels[2])

        ax.axhline(0, color="black", lw=1)
        ax.set_title(var, fontsize=12)
        ax.set_xlabel("quarters")
        ax.grid(alpha=0.25)
        ax.set_ylabel("")

        # legend only once (first panel)
        if idx == 0:
            ax.legend(frameon=True, fontsize=9)

    # turn off unused axes
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.show()
    return available


import numpy as np
import matplotlib.pyplot as plt

def plot_key_irfs_clean_up_to_three_models(
    model1,
    model2=None,
    model3=None,
    T=40,
    extra=("pi","Y","i","r"),
    title="IRFs",
    ylabel="Deviation from steady state",
    ncols=3,
    figsize_per_col=4.2,
    figsize_per_row=2.8,
    colors=("forestgreen", "navy", "red"),
    labels=("model 1", "model 2", "model 3"),
    ls=("-", "--", "-."),
    force_union=True,   # union of available vars across provided models
):
    """
    Plot key IRFs for 1, 2, or 3 models (overlayed in each subplot).

    Parameters
    ----------
    model1 : required, must have .IRF dict
    model2, model3 : optional, can be None
    force_union : if True, plot variables available in ANY provided model
                  if False, plot only variables available in ALL provided models
    """

    wanted = ["C_hh","A_hh","ra","Z","chi"] + list(extra)

    models = [m for m in (model1, model2, model3) if m is not None]
    if len(models) == 0:
        raise ValueError("At least one model must be provided.")

    # clip styles to number of models
    colors = list(colors)[:len(models)]
    labels = list(labels)[:len(models)]
    ls     = list(ls)[:len(models)]

    def _avail(m):
        out = []
        if not hasattr(m, "IRF"):
            return out
        for v in wanted:
            if v in m.IRF:
                arr = np.asarray(m.IRF[v])
                if np.any(np.isfinite(arr)):
                    out.append(v)
        return out

    avails = [set(_avail(m)) for m in models]

    if force_union:
        available = [v for v in wanted if any(v in a for a in avails)]
    else:
        available = [v for v in wanted if all(v in a for a in avails)]

    if len(available) == 0:
        raise RuntimeError("No requested IRF series found in provided model.IRF dicts.")

    n = len(available)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        squeeze=False
    )

    # ---- margins ----
    left, right, bottom, top = 0.12, 0.98, 0.10, 0.88
    wspace, hspace = 0.30, 0.55
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)

    fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=14)
    fig.text(0.04, 0.5, ylabel, ha="center", va="center", rotation="vertical", fontsize=12)

    def _get_series(m, var):
        if (not hasattr(m, "IRF")) or (var not in m.IRF):
            return None
        s = np.asarray(m.IRF[var]).ravel()
        if s.size == 0 or not np.any(np.isfinite(s)):
            return None
        return s[:T]

    # plot panels
    for idx, var in enumerate(available):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        series_list = [_get_series(m, var) for m in models]

        # plot each provided model if series exists
        for j, s in enumerate(series_list):
            if s is None:
                continue
            ax.plot(np.arange(s.size), s, lw=2.3, color=colors[j], ls=ls[j], label=labels[j])

        ax.axhline(0, color="black", lw=1)
        ax.set_title(var, fontsize=12)
        ax.set_xlabel("quarters")
        ax.grid(alpha=0.25)
        ax.set_ylabel("")

        # legend only once (first panel)
        if idx == 0 and len(models) > 1:
            ax.legend(frameon=True, fontsize=9)

    # turn off unused axes
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.show()
    return available


import numpy as np
import matplotlib.pyplot as plt

def plot_sensitive_and_aggregate_separately(
    df_mpc,
    df_impact,     # <-- from table_C_impact_by_z_from_perz_decomp(...)
    df_contrib,    # <-- from contrib_to_agg_C_impact_by_z(df_impact)
    highlight_z=(4,5),
    figsize_top=(10,4.5),
    figsize_bottom=(10,3.2)
):
    # ---- column checks ----
    req_mpc = {"z","share_a0(z)","MPC_quarterly(z)"}
    req_imp = {"z","%dC0_total"}
    req_con = {"z","contrib_pp_to_agg_%dC0"}

    missing = (req_mpc - set(df_mpc.columns)) | (req_imp - set(df_impact.columns)) | (req_con - set(df_contrib.columns))
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ---- merge ----
    use1 = df_mpc[["z","share_a0(z)","MPC_quarterly(z)"]].copy()
    use2 = df_impact[["z","%dC0_total"]].copy()
    use3 = df_contrib[["z","contrib_pp_to_agg_%dC0"]].copy()

    df = use1.merge(use2, on="z").merge(use3, on="z").sort_values("z").reset_index(drop=True)

    z = df["z"].to_numpy()
    pctC = df["%dC0_total"].to_numpy()
    mpc = df["MPC_quarterly(z)"].to_numpy()
    a0  = df["share_a0(z)"].to_numpy()
    contrib = df["contrib_pp_to_agg_%dC0"].to_numpy()

    # ---------------- TOP FIGURE ----------------
    fig1, ax = plt.subplots(figsize=figsize_top)

    # highlight
    if highlight_z is not None and len(highlight_z) > 0:
        for zz in highlight_z:
            ax.axvspan(zz-0.35, zz+0.35, color="gold", alpha=0.15, lw=0)

    ax.plot(z, pctC, marker="o", lw=2.6, color="black", label=r"$\%\Delta C_0(z)$")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel(r"Impact response: $\%\Delta C_0(z)$", labelpad=12)
    ax.set_xlabel("Productivity state z")
    ax.set_xticks(z)
    ax.grid(alpha=0.25)

    axr = ax.twinx()
    axr.plot(z, mpc, marker="s", lw=2.2, color="#1f77b4", label="MPC (quarterly)")
    axr.plot(z, a0, marker="^", lw=2.2, color="firebrick", ls="--", label=r"$P(a=0\mid z)$")
    axr.set_ylabel("Liquidity / cash-flow sensitivity", labelpad=12)

    # combined legend
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = axr.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right", frameon=True)

    ax.set_title("Sensitivity per z (impact, t=0)")
    fig1.tight_layout()
    plt.show()

    # ---------------- BOTTOM FIGURE ----------------
    fig2, ax2 = plt.subplots(figsize=figsize_bottom)

    if highlight_z is not None and len(highlight_z) > 0:
        for zz in highlight_z:
            ax2.axvspan(zz-0.35, zz+0.35, color="gold", alpha=0.15, lw=0)

    colors = ["firebrick" if v < 0 else "#1f77b4" for v in contrib]
    ax2.bar(z, contrib, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_ylabel(r"Contribution to aggregate $\Delta C_0$ (pp)", labelpad=12)
    ax2.set_xlabel("Productivity state z")
    ax2.set_xticks(z)
    ax2.grid(alpha=0.25)

    ax2.set_title("Who matters for aggregate impact consumption?")
    fig2.tight_layout()
    plt.show()

    return df


# ============================
# READY-TO-RUN “MOST IMPORTANT” BASELINE HANK DIAGNOSTICS
# (1) MPC + constraint share by z (table)
# (2) Who drives aggregate C on impact? (contribution bar plot + table)
# (3) Key IRFs panel: ra, Z, B, chi (plus optional pi, Y, L if available)
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- (1) MPC + constraint share by z ----------

def mpc_by_z_table(model, annualize=True):
    """
    Compute (i) mean MPC by z (local derivative along a-grid),
            (ii) share at borrowing constraint a=a_min by z,
            (iii) mean assets by z.

    Notes:
    - MPC here is the *local* MPC along the asset grid (finite difference).
    - If annualize=True: approximate annual MPC as sum of first 4 quarters of iMPC
      using the existing household Jacobian to chi (your model already uses this).
    """
    par, ss = model.par, model.ss
    D = ss.D            # (Nfix, Nz, Na)
    a_pol = ss.a        # (Nfix, Nz, Na)
    c_pol = ss.c        # (Nfix, Nz, Na)
    a_grid = par.a_grid

    Nz, Na = par.Nz, par.Na

    # mass by z
    mass_z = D.sum(axis=(0, 2))  # (Nz,)

    # share at constraint a==a_min -> index 0
    share_a0_z = D[:, :, 0].sum(axis=0) / np.maximum(mass_z, 1e-32)

    # mean assets by z (levels)
    Abar_z = (a_pol * D).sum(axis=(0, 2)) / np.maximum(mass_z, 1e-32)

    # local MPC along grid: dc/dm where m = (1+ra)a + income, but we approximate via a-grid step:
    # mpc ≈ (c(a_{j+1}) - c(a_j)) / ((1+ra)*(a_{j+1}-a_j))
    da = (a_grid[1:] - a_grid[:-1])[None, None, :]  # (1,1,Na-1)
    mpc_micro = (c_pol[:, :, 1:] - c_pol[:, :, :-1]) / np.maximum((1.0 + ss.ra) * da, 1e-32)  # (Nfix,Nz,Na-1)

    # weight MPC by D on left nodes
    w = D[:, :, :-1]
    mpc_z = (mpc_micro * w).sum(axis=(0, 2)) / np.maximum(w.sum(axis=(0, 2)), 1e-32)

    # optional annual MPC from iMPC = -jac_hh[(C_hh, chi)] columns at t=0
    ann_mpc = np.full(Nz, np.nan)
    if annualize and hasattr(model, "jac_hh") and ("C_hh", "chi") in model.jac_hh:
        # aggregate iMPC, not by z (your jac_hh is aggregate). We'll report it as a scalar in table footer.
        pass

    df = pd.DataFrame({
        "z": np.arange(Nz),
        "mass(z)": mass_z,
        "share_a0(z)": share_a0_z,
        "Abar(z)": Abar_z,
        "MPC_quarterly(z)": mpc_z,
    })

    with pd.option_context("display.max_rows", None, "display.width", 160,
                           "display.float_format", "{:,.6g}".format):
        print(df)

    # print aggregate annual MPC (your target) if available
    if annualize and hasattr(model, "jac_hh") and ("C_hh", "chi") in model.jac_hh:
        iMPC = -model.jac_hh[("C_hh", "chi")]
        annual_MPC = float(np.sum(iMPC[:4, 0]))
        print(f"\nAggregate annual MPC (sum first 4 quarters of iMPC): {annual_MPC:.4f}")

    return df


# ---------- (2) Who drives aggregate C on impact? ----------

def contrib_to_agg_C_impact_by_z(df_chainrule):
    """
    Using your chain-rule output table df (from table_C_impact_by_z_chainrule),
    compute each z-group's contribution to aggregate impact ΔC in levels and percent points.

    Contribution in levels:
        contrib_level(z) = mass(z) * dC0_total(z)   where dC0_total(z) is mean within z

    Contribution in % points of aggregate C:
        contrib_pp(z) = 100 * [mass(z)*dC0_total(z)] / C_agg_ss
    """
    # Expect these columns exist:
    # 'mass(z)', 'C_ss(z)', 'dC0_total', '%dC0_total'
    mass = df_chainrule["mass(z)"].to_numpy()
    dCz  = df_chainrule["dC0_total"].to_numpy()
    Cssz = df_chainrule["C_ss(z)"].to_numpy()

    Cagg_ss = float(np.sum(mass * Cssz))
    contrib_level = mass * dCz
    contrib_pp = 100.0 * contrib_level / np.maximum(Cagg_ss, 1e-32)

    out = df_chainrule.copy()
    out["contrib_level_to_agg_dC0"] = contrib_level
    out["contrib_pp_to_agg_%dC0"] = contrib_pp

    # rank by absolute contribution to aggregate impact change
    out["rank_abs_contrib_pp"] = out["contrib_pp_to_agg_%dC0"].abs().rank(ascending=False, method="dense").astype(int)

    with pd.option_context("display.max_rows", None, "display.width", 180,
                           "display.float_format", "{:,.6g}".format):
        print(out[["z","mass(z)","%dC0_total","contrib_pp_to_agg_%dC0","rank_abs_contrib_pp"]])

    return out


def plot_contrib_bar(out_contrib, title="Contributions to aggregate impact ΔC (percentage points)"):
    z = out_contrib["z"].to_numpy()
    contrib_pp = out_contrib["contrib_pp_to_agg_%dC0"].to_numpy()

    fig, ax = plt.subplots(figsize=(8,4))
    colors = ["firebrick" if v < 0 else "steelblue" for v in contrib_pp]
    ax.bar(z, contrib_pp, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Productivity state z")
    ax.set_ylabel("Contribution (pp of aggregate C)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------- (3) Key IRFs panel: ra, Z, B, chi (+ optional extras) ----------

import numpy as np
import matplotlib.pyplot as plt

def plot_key_irfs(model, T=40, extra=("pi","Y","L","i","r"), title="Model IRFs"):

    wanted = ["ra", "Z", "B", "chi"] + list(extra)
    available = [
        v for v in wanted
        if hasattr(model, "IRF") and v in model.IRF and np.any(np.isfinite(model.IRF[v]))
    ]
    if len(available) == 0:
        raise RuntimeError("No requested IRF series found in model.IRF.")

    n = len(available)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for k, var in enumerate(available):
        ax = axes[k]
        series = np.asarray(model.IRF[var]).ravel()[:T]
        ax.plot(series, lw=2.3, color="navy")
        ax.axhline(0, color="black", lw=1)
        ax.set_title(var)
        ax.set_xlabel("quarters")
        ax.grid(alpha=0.25)

    for k in range(len(available), len(axes)):
        axes[k].axis("off")

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.text(0.02, 0.5, "Deviation from steady state", va="center", rotation="vertical")

    # Reserve space for y-label and title explicitly:
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.08, wspace=0.25, hspace=0.40)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_ra_effect_on_C(model_baseline, model_sticky, model_het, T=40,
                        label_base="baseline", label_sticky="sticky expectations", label_het="heterogeneous sticky",
                        title="Return channel: contribution of $r^a$ to consumption"):
    """
    Plot the *ra contribution* to aggregate consumption for two models:
        dC_from_ra(t) = J_{C,ra} @ IRF_ra    (scaled to % of SS C)

    Requirements:
      - model.jac_hh[('C_hh','ra')] exists
      - model.IRF['ra'] exists
      - model.ss.C_hh exists
    """

    def ra_contrib_pct(model, T):
        J = model.jac_hh[('C_hh', 'ra')]
        ra_path = np.asarray(model.IRF['ra']).ravel()
        ssC = float(model.ss.C_hh)

        # ensure length T
        if ra_path.size < T:
            tmp = np.zeros(T); tmp[:ra_path.size] = ra_path
            ra_path = tmp
        else:
            ra_path = ra_path[:T]

        # contribution in percent of SS C
        dC_ra = (J[:T, :T] @ ra_path) * 100.0 / ssC
        return dC_ra

    dC_ra_base = ra_contrib_pct(model_baseline, T)
    dC_ra_sticky = ra_contrib_pct(model_sticky, T)
    dC_ra_het = ra_contrib_pct(model_het, T)

    t = np.arange(T)

    plt.figure(figsize=(7.5, 4.5))
    plt.axhline(0, color="black", lw=1)
    plt.plot(t, dC_ra_base, lw=2.6, color="navy", label=label_base)
    plt.plot(t, dC_ra_sticky, lw=2.6, color="firebrick", ls="--", label=label_sticky)
    plt.plot(t, dC_ra_het, lw=2.6, color="darkgreen", ls="-.", label=label_het)

    plt.title(title)
    plt.xlabel("Quarters")
    plt.ylabel(r"Contribution to $\%\Delta C$ (via $r^a$)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return dC_ra_base, dC_ra_sticky, dC_ra_het


import numpy as np
import matplotlib.pyplot as plt

def plot_input_effect_on_C(
    model1,
    model2,
    model3=None,              # <-- optional now
    inp="ra",
    output="C_hh",
    T=40,
    label1="baseline",
    label2="sticky expectations",
    label3="heterogeneous sticky",
    title=None,
    scale="pct",              # "pct" or "level"
    ls2="--",
    ls3="-.",                 # <-- added for consistency
    color1="navy",
    color2="firebrick",
    color3="darkgreen"
):
    """
    Plot contribution of a chosen input (ra, Z, chi, ...) to aggregate consumption.

        dC_from_inp(t) = J_{C,inp}[0:T,0:T] @ IRF_inp[0:T]

    Supports:
      - two models:  plot model1 vs model2
      - three models: plot model1 vs model2 vs model3 (if provided)

    Returns
    -------
    If model3 is None:
        (d1, d2)
    else:
        (d1, d2, d3)
    """

    def contrib(model, inp, output, T, scale):
        key = (output, inp)
        if not hasattr(model, "jac_hh") or key not in model.jac_hh:
            raise KeyError(f"Missing Jacobian key {key} in model.jac_hh")
        if not hasattr(model, "IRF") or inp not in model.IRF:
            raise KeyError(f"Missing IRF path for '{inp}' in model.IRF")

        J = np.asarray(model.jac_hh[key])
        x = np.asarray(model.IRF[inp]).ravel()

        # pad/trim x to length T
        if x.size < T:
            x_pad = np.zeros(T)
            x_pad[:x.size] = x
            x = x_pad
        else:
            x = x[:T]

        # ensure Jacobian is at least TxT
        JT = J[:T, :T]
        d = JT @ x  # level contribution

        if scale == "pct":
            ssC = float(getattr(model.ss, output))
            d = 100.0 * d / ssC
        elif scale == "level":
            pass
        else:
            raise ValueError("scale must be 'pct' or 'level'")

        return d

    # contributions
    d1 = contrib(model1, inp, output, T, scale)
    d2 = contrib(model2, inp, output, T, scale)
    d3 = contrib(model3, inp, output, T, scale) if model3 is not None else None

    t = np.arange(T)

    # title
    if title is None:
        if scale == "pct":
            title = rf"Contribution of {inp} to $\%\Delta {output}$"
        else:
            title = f"Contribution of {inp} to {output} (levels)"

    # plot
    plt.figure(figsize=(7.5, 4.5))
    plt.axhline(0, color="black", lw=1)
    plt.plot(t, d1, lw=2.6, color=color1, label=label1)
    plt.plot(t, d2, lw=2.6, color=color2, ls=ls2, label=label2)
    if d3 is not None:
        plt.plot(t, d3, lw=2.6, color=color3, ls=ls3, label=label3)

    plt.title(title)
    plt.xlabel("Quarters")
    plt.ylabel(r"Contribution to $\%\Delta C$" if scale == "pct" else "Contribution (level)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (d1, d2) if d3 is None else (d1, d2, d3)

import numpy as np
import pandas as pd

def calibration_table(model):
    """
    Builds a compact calibration table: symbol, numeric value, and 'Set vs solved' note.

    Assumes your steady state logic follows steady_state.py:
      - beta and (mu or B) are solved in find_ss via root targets:
          ss.clearing_A = 0 and annual MPC = 0.5  (see steady_state.py)  :contentReference[oaicite:4]{index=4}
      - varphi is implied from NK wage curve in SS  :contentReference[oaicite:5]{index=5}
      - z-grid and z-transition are constructed with Rouwenhorst  :contentReference[oaicite:6]{index=6}
      - a-grid is equilogspace  :contentReference[oaicite:7]{index=7}
    """
    par, ss = model.par, model.ss

    # Helper to safely read attributes
    def get(obj, name, default=np.nan):
        return getattr(obj, name) if hasattr(obj, name) else default

    do_B = bool(get(par, "do_B", False))

    rows = []

    # --- Preferences / technology / policy "parameters" ---
    rows += [
        (r"$\sigma$",   get(par, "sigma"),      "Set (preference curvature)"),
        (r"$\beta$",    get(par, "beta"),       "Solved in SS root-finding to hit targets"),
        (r"$\nu$",      get(par, "nu"),         "Set (inverse Frisch elasticity)"),
        (r"$\varphi$",  get(par, "varphi"),     "Implied in SS (pinned by NK wage curve)"),
        (r"$\mu$",      get(par, "mu"),         "Solved in SS if do_B=False; fixed μ=1 if do_B=True"),
        (r"$r^{ss}$",   get(par, "r_target_ss"),"Set (steady-state real rate target)"),
    ]

    # Government bond supply: parameter vs solved depending on do_B
    rows += [
        (r"$B^{ss}$",   get(ss, "B"),           "Solved in SS if do_B=True; else set to 0"),
        (r"$\chi^{ss}$",get(ss, "chi"),         r"Implied by budget in SS: $\chi^{ss}=r^{ss}B^{ss}$"),
    ]

    # Idiosyncratic income process parameters
    rows += [
        (r"$\rho_z$",       get(par, "rho_z"),      "Set (idiosyncratic productivity persistence)"),
        (r"$\sigma_\psi$",  get(par, "sigma_psi"),  "Set (idiosyncratic shock s.d.)"),
    ]

    # Numerical/grids
    rows += [
        (r"$a_{\min}$", get(par, "a_min"), "Set (numerical: borrowing constraint)"),
        (r"$a_{\max}$", get(par, "a_max"), "Set (numerical: grid upper bound)"),
        (r"$N_a$",      get(par, "Na"),    "Set (asset grid size)"),
        (r"$N_z$",      get(par, "Nz"),    "Set (productivity states)"),
        (r"$T$",        get(par, "T"),     "Set (IRF/Jacobian horizon)"),
        (r"$N_{fix}$",  get(par, "Nfix"),  "Set (fixed types, if used)"),
    ]

    df = pd.DataFrame(rows, columns=["Symbol", "Value", "How determined"])

    # Make numbers prettier
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.6g}"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return str(x)

    df["Value"] = df["Value"].map(fmt)

    # Add “targets” rows explicitly (these are *not* parameters, but document calibration)
    targets = pd.DataFrame([
        ("Target: asset mkt clearing", "0", r"Solved: $\texttt{ss.clearing\_A}=0$"),
        ("Target: annual MPC", "0.5", r"Solved: annual MPC $=0.5$"),
    ], columns=["Symbol", "Value", "How determined"])

    df_out = pd.concat([df, targets], ignore_index=True)

    print(df_out.to_string(index=False))

    latex = df_out.to_latex(
        index=False,
        escape=False,
        caption="Calibration and steady-state determination",
        label="tab:calibration",
        column_format="lll",
    )
    return df_out, latex



import numpy as np
import pandas as pd

def calibration_table_three_sections(
    model,
    external_keys_extra=("phi_pi", "phi_y", "rho_eps_i", "sigma_eps_i", "rho_i"),
    set_keys_extra=("Na","Nz","T","Nfix","a_min","a_max"),
    floatfmt="%.6g",
):
    """
    Build a calibration table with three sections:
      (i) Set parameters (numerical/normalizations)
      (ii) Externally calibrated parameters (chosen from literature/targets)
      (iii) Internally calibrated/implied (solved in SS or pinned by SS identities)

    This is tailored to your workflow:
      - beta is solved in SS root-finding
      - mu is solved if do_B==False; else fixed to 1
      - B is solved if do_B==True; else set to 0
      - varphi pinned from the NK wage curve in SS
      - chi_ss implied from govt budget chi = r*B
      - monetary policy rule parameters like phi_pi and shock persistence rho_eps_i
        are “external” (unless you explicitly solve them, which you don’t)

    Returns:
      df, latex_string
    """

    par, ss = model.par, model.ss

    def has(obj, name): 
        return hasattr(obj, name)

    def get(obj, name, default=None):
        return getattr(obj, name) if hasattr(obj, name) else default

    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, (float, np.floating)):
            return floatfmt % float(x)
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return str(x)

    do_B = bool(get(par, "do_B", False))

    # ---- classify symbols ----
    rows = []

    # (1) Set parameters: numerical + normalizations you pick
    set_params = [
        ("Set", r"$a_{\min}$", "a_min", "Borrowing constraint / grid lower bound"),
        ("Set", r"$a_{\max}$", "a_max", "Asset grid upper bound"),
        ("Set", r"$N_a$", "Na", "Number of asset grid points"),
        ("Set", r"$N_z$", "Nz", "Number of productivity states"),
        ("Set", r"$T$", "T", "IRF/Jacobian horizon"),
        ("Set", r"$N_{\mathrm{fix}}$", "Nfix", "Number of fixed types (if used)"),
    ]
    # allow user additions
    for k in set_keys_extra:
        pass  # already included common ones; keep hook in case you want to extend later

    # (2) Externally calibrated: preference/technology/process/policy-rule/shock persistence
    external_params = [
        ("External", r"$\sigma$", "sigma", "CRRA coefficient (literature)"),
        ("External", r"$\nu$", "nu", "Inverse Frisch elasticity (literature)"),
        ("External", r"$\rho_z$", "rho_z", "Idiosyncratic productivity persistence"),
        ("External", r"$\sigma_{\psi}$", "sigma_psi", "Idiosyncratic shock s.d."),
        ("External", r"$r^{ss}$", "r_target_ss", "Steady-state real rate target"),
        # monetary policy rule / shock process (names may differ in your par)
        ("External", r"$\phi_{\pi}$", "phi_pi", "Taylor-rule inflation coefficient"),
        ("External", r"$\phi_{y}$", "phi_y", "Taylor-rule output coefficient (if used)"),
        ("External", r"$\rho_{\varepsilon_i}$", "rho_eps_i", "Monetary policy shock persistence"),
        ("External", r"$\sigma_{\varepsilon_i}$", "sigma_eps_i", "Monetary policy shock s.d. (if used)"),
        ("External", r"$\rho_{i}$", "rho_i", "Interest-rate smoothing (if used)"),
    ]

    # also include any extra keys you asked for even if not in list above
    for k in external_keys_extra:
        # we already cover common ones; this hook is mainly to ensure presence
        pass

    # (3) Internally calibrated / implied in SS
    internal_params = [
        ("Internal", r"$\beta$", "beta", "Solved in SS (hits targets via root-finding)"),
        ("Internal", r"$\mu$", "mu",
         "Solved in SS if do\\_B=False; fixed (often $\\mu=1$) if do\\_B=True"),
        ("Internal", r"$B^{ss}$", "__ss_B__", "Solved in SS if do\\_B=True; else set to 0"),
        ("Internal", r"$\varphi$", "varphi", "Implied/pinned by NK wage Phillips curve in SS"),
        ("Internal", r"$\chi^{ss}$", "__ss_chi__", r"Implied by govt budget: $\chi^{ss}=r^{ss}B^{ss}$"),
    ]

    # Build rows
    def add_block(block):
        for cat, sym, key, desc in block:
            if key == "__ss_B__":
                val = get(ss, "B", None)
            elif key == "__ss_chi__":
                val = get(ss, "chi", None)
            else:
                val = get(par, key, None)
            if val is None and (key not in ("__ss_B__", "__ss_chi__")):
                # some things might live in ss instead of par
                val = get(ss, key, None)

            # skip truly missing entries
            if val is None:
                continue

            rows.append({
                "Category": cat,
                "Symbol": sym,
                "Value": fmt(val),
                "Description / how determined": desc
            })

    add_block(set_params)
    add_block(external_params)
    add_block(internal_params)

    # Targets (always include)
    targets = [
        {"Category":"Target","Symbol":"Asset mkt clearing","Value":"0",
         "Description / how determined":r"$\texttt{ss.clearing\_A}=0$"},
        {"Category":"Target","Symbol":"Annual MPC","Value":"0.5",
         "Description / how determined":r"Annual MPC $=0.5$ (SS target)"},
    ]
    rows.extend(targets)

    df = pd.DataFrame(rows)

    # Order categories nicely
    cat_order = {"Set":0, "External":1, "Internal":2, "Target":3}
    df["_ord"] = df["Category"].map(cat_order).fillna(99).astype(int)
    df = df.sort_values(["_ord","Symbol"]).drop(columns="_ord").reset_index(drop=True)

    # LaTeX
    latex = df.to_latex(
        index=False,
        escape=False,
        caption="Calibration and steady-state determination",
        label="tab:calibration",
        column_format="llll",
    )

    return df, latex

import numpy as np
import pandas as pd

def calibration_table_two_models(
    model_equity,
    model_bonds,
    label_equity="Baseline (equity)",
    label_bonds="Baseline (bonds)",
    floatfmt="%.6g",
    include_targets=True,
):
    """
    Side-by-side calibration table for two models.

    Output columns:
      Category | Symbol | Value (equity) | Value (bonds) | Description/how determined
    """

    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, (float, np.floating)):
            return floatfmt % float(x)
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return str(x)

    def get_any(m, key):
        # Try par then ss
        if hasattr(m, "par") and hasattr(m.par, key):
            return getattr(m.par, key)
        if hasattr(m, "ss") and hasattr(m.ss, key):
            return getattr(m.ss, key)
        return None

    # --- Row specification: (Category, Symbol, key, description) ---
    # Special keys: "__ss_B__", "__ss_chi__", "__ss_i__", "__annual_mpc__"
    rowspec = []

    # Set / numerical
    rowspec += [
        ("Set", r"$a_{\min}$", "a_min", "Borrowing constraint / grid lower bound"),
        ("Set", r"$a_{\max}$", "a_max", "Asset grid upper bound"),
        ("Set", r"$N_a$", "Na", "Number of asset grid points"),
        ("Set", r"$N_z$", "Nz", "Number of productivity states"),
        ("Set", r"$T$", "T", "IRF/Jacobian horizon"),
        ("Set", r"$N_{\mathrm{fix}}$", "Nfix", "Number of fixed types (if used)"),
    ]

    # External calibration / policy / processes
    rowspec += [
        ("External", r"$\sigma$", "sigma", "CRRA coefficient"),
        ("External", r"$\nu$", "nu", "Inverse Frisch elasticity"),
        ("External", r"$\rho_z$", "rho_z", "Idiosyncratic productivity persistence"),
        ("External", r"$\sigma_{\psi}$", "sigma_psi", "Idiosyncratic shock s.d."),
        ("External", r"$r^{ss}$", "r_target_ss", "Steady-state real rate target"),
        ("External", r"$\phi_{\pi}$", "phi_pi", "Taylor-rule inflation coefficient"),
        ("External", r"$\phi_y$", "phi_y", "Taylor-rule output coefficient (if used)"),
        ("External", r"$\rho_i$", "rho_i", "Interest-rate smoothing (if used)"),
        ("External", r"$\rho_{\varepsilon_i}$", "rho_eps_i", "Monetary policy shock persistence"),
        ("External", r"$\sigma_{\varepsilon_i}$", "sigma_eps_i", "Monetary policy shock s.d. (if used)"),
    ]

    # Internal SS calibrated / implied
    rowspec += [
        ("Internal", r"$\beta$", "beta", "Solved in SS to hit targets"),
        ("Internal", r"$\mu$", "mu", "Solved in SS (or fixed depending on closure)"),
        ("Internal", r"$B^{ss}$", "__ss_B__", "SS bond supply (solved or set by closure)"),
        ("Internal", r"$i^{ss}$", "__ss_i__", "SS nominal rate (typically implied by Fisher relation)"),
        ("Internal", r"$\varphi$", "varphi", "Pinned/implied by NK wage Phillips curve in SS"),
        ("Internal", r"$\chi^{ss}$", "__ss_chi__", r"Implied by govt budget: $\chi^{ss}=r^{ss}B^{ss}$"),
    ]

    # Targets block
    targets = [
        ("Target", "Asset mkt clearing", "clearing_A", r"$\texttt{ss.clearing\_A}=0$"),
        ("Target", "Annual MPC", "__annual_mpc__", r"Targeted in SS routine (e.g. 0.5)"),
    ]

    def annual_mpc(m):
        if hasattr(m, "jac_hh") and ("C_hh", "chi") in m.jac_hh:
            iMPC = -m.jac_hh[("C_hh", "chi")]
            return float(np.sum(iMPC[:4, 0]))
        return None

    def value_for_model(m, key):
        if key == "__ss_B__":
            return getattr(m.ss, "B", None) if hasattr(m, "ss") else None
        if key == "__ss_chi__":
            return getattr(m.ss, "chi", None) if hasattr(m, "ss") else None
        if key == "__ss_i__":
            # Prefer ss.i, else par.ss_i if you store it there
            if hasattr(m, "ss") and hasattr(m.ss, "i"):
                return getattr(m.ss, "i")
            if hasattr(m, "par") and hasattr(m.par, "ss_i"):
                return getattr(m.par, "ss_i")
            return None
        if key == "__annual_mpc__":
            return annual_mpc(m)
        return get_any(m, key)

    out_rows = []
    for cat, sym, key, desc in rowspec:
        v1 = value_for_model(model_equity, key)
        v2 = value_for_model(model_bonds, key)
        if (v1 is None) and (v2 is None):
            continue
        out_rows.append({
            "Category": cat,
            "Symbol": sym,
            label_equity: fmt(v1),
            label_bonds: fmt(v2),
            "Description / how determined": desc
        })

    if include_targets:
        for cat, sym, key, desc in targets:
            v1 = value_for_model(model_equity, key)
            v2 = value_for_model(model_bonds, key)
            if key != "__annual_mpc__" and (v1 is None) and (v2 is None):
                continue
            out_rows.append({
                "Category": cat,
                "Symbol": sym,
                label_equity: fmt(v1),
                label_bonds: fmt(v2),
                "Description / how determined": desc
            })

    df = pd.DataFrame(out_rows)

    # Order categories
    cat_order = {"Set": 0, "External": 1, "Internal": 2, "Target": 3}
    df["_ord"] = df["Category"].map(cat_order).fillna(99).astype(int)
    df = df.sort_values(["_ord", "Symbol"]).drop(columns="_ord").reset_index(drop=True)

    latex = df.to_latex(
        index=False,
        escape=False,
        caption="Calibration: baseline with equity vs baseline with bonds",
        label="tab:calibration_two_models",
        column_format="l l r r p{6.2cm}",
    )

    return df, latex


import numpy as np

import numpy as np

import numpy as np

def z_moments_from_D_and_c(D, c, Nz):
    """
    Returns:
      mu_z      : mass by z, shape (Nz,)
      cbar_z_ss : mean consumption conditional on z, shape (Nz,)
    Works when D has shape like (1,Nz,Na) and c like (1,Nz,Na) or broadcastable.
    """
    D = np.asarray(D)
    c = np.asarray(c)

    # find z axis (the one whose length is Nz)
    z_axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
    if len(z_axes) != 1:
        raise ValueError(f"Ambiguous z axis. D.shape={D.shape}, Nz={Nz}")
    z_ax = z_axes[0]

    # mu_z = sum of D over all non-z axes
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)
    mu_z = D.sum(axis=sum_axes)
    mu_z = np.maximum(mu_z, 0)
    mu_z = mu_z / mu_z.sum()

    # total consumption in each z: sum D * c over non-z axes
    Dc = (D * c).sum(axis=sum_axes)

    # conditional mean cbar_z_ss
    cbar_z_ss = Dc / np.maximum(mu_z, 1e-16)
    return mu_z, cbar_z_ss


def stationary_dist_from_P(P, tol=1e-14, maxiter=100_000):
    """
    Stationary distribution mu such that mu = mu P.
    """
    P = np.asarray(P)
    Nz = P.shape[0]
    mu = np.ones(Nz) / Nz
    for _ in range(maxiter):
        mu_new = mu @ P
        if np.max(np.abs(mu_new - mu)) < tol:
            mu = mu_new
            break
        mu = mu_new
    mu = np.maximum(mu, 0)
    mu /= mu.sum()
    return mu

import numpy as np

def dist_z_from_D(D, Nz):
    """
    Extract mass by z from joint stationary distribution D, for any layout.
    Finds an axis of D with length Nz, sums over all other axes.
    """
    D = np.asarray(D)
    if D.ndim < 2:
        raise ValueError("D must be at least 2D (joint distribution over states).")

    # find axes matching Nz
    axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
    if len(axes) == 0:
        raise ValueError(f"No axis of D matches Nz={Nz}. D.shape={D.shape}")
    if len(axes) > 1:
        # if ambiguous, pick the last matching axis (usually z is later than assets)
        z_axis = axes[-1]
    else:
        z_axis = axes[0]

    # sum over all other axes
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_axis)
    mu = D.sum(axis=sum_axes)

    mu = np.maximum(mu, 0)
    s = mu.sum()
    if s <= 0:
        raise ValueError("Computed dist_z has non-positive sum; check D.")
    mu = mu / s
    return mu



def _project_to_simplex(v):
    """Project v onto the probability simplex {w>=0, sum w = 1}."""
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    if len(rho) == 0:
        # fallback: uniform
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def _infer_dist_z_from_agg(model, output, inputs, Nz, T_jac, prefer_inp=None):
    """
    Infer weights mu_z by matching aggregate contribution:
        sum_z mu_z (Jz[z] @ x)  ≈  (J_agg @ x)
    using least squares + simplex projection.
    """
    # pick an input that exists in both jac_hh_z and jac_hh
    candidates = list(inputs)
    if prefer_inp is not None:
        candidates = [prefer_inp] + [k for k in candidates if k != prefer_inp]

    chosen = None
    for inp in candidates:
        if (output, inp) in getattr(model, "jac_hh", {}) and (output, inp) in getattr(model, "jac_hh_z", {}):
            if inp in model.IRF:
                chosen = inp
                break
    if chosen is None:
        raise RuntimeError("Cannot infer weights: no common (output, inp) in jac_hh and jac_hh_z with IRF available.")

    J_agg = model.jac_hh[(output, chosen)]        # (T,T)
    Jz    = model.jac_hh_z[(output, chosen)]      # (Nz,T,T)
    x     = model.IRF[chosen][:T_jac]             # (T,)

    # Build matrix A of shape (T, Nz): column z is (Jz[z] @ x)
    A = np.zeros((T_jac, Nz))
    for iz in range(Nz):
        A[:, iz] = Jz[iz] @ x

    y = J_agg @ x  # (T,)

    # Solve min ||A w - y||^2
    w_ls, *_ = np.linalg.lstsq(A, y, rcond=None)

    # Project to simplex
    w = _project_to_simplex(w_ls)

    # Quick diagnostics
    y_fit = A @ w
    err = np.max(np.abs(y_fit - y))
    return w, chosen, err


import numpy as np
import matplotlib.pyplot as plt


def _dist_z_from_D_anyaxis(D, Nz):
    """Find axis of D with size Nz; sum over other axes to get mu_z."""
    D = np.asarray(D)
    z_axes = [ax for ax, n in enumerate(D.shape) if n == Nz]
    if len(z_axes) != 1:
        raise ValueError(f"Ambiguous z-axis: D.shape={D.shape}, Nz={Nz}")
    z_ax = z_axes[0]
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)
    mu = D.sum(axis=sum_axes)
    mu = np.maximum(mu, 0)
    mu = mu / mu.sum()
    return mu, z_ax


def _cbar_z_from_D_and_c(D, c, Nz):
    """Compute conditional mean c by z: cbar_z = E[c | z]."""
    D = np.asarray(D)
    c = np.asarray(c)
    mu_z, z_ax = _dist_z_from_D_anyaxis(D, Nz)
    sum_axes = tuple(ax for ax in range(D.ndim) if ax != z_ax)
    Dc = (D * c).sum(axis=sum_axes)
    cbar = Dc / np.maximum(mu_z, 1e-16)
    return mu_z, cbar


def decompose_C_by_z_and_input_norm(
    model,
    inputs=("Z", "ra", "chi"),
    output="C_hh",
    T=None,
    check_consistency=True,
    agg_mode="auto",   # "auto" | "sum" | "mu_sum" | "mean" | "mu_cbar"
    include_distribution_residual=True,
    dist_irf_key_candidates=("D", "Dbeg"),   # will use the first that exists in model.IRF
):
    """
    Decompose output IRFs by productivity z and input using per-z Jacobians, and (optionally)
    compute the missing distribution/composition component as a residual.

    Returns
    -------
    C_z_total            : (Nz, T)
    C_z_contrib          : dict[input -> (Nz, T)]
    C_agg_total          : (T,)
    C_agg_contrib        : dict[input -> (T,)]
    agg_mode_used        : str
    C_agg_residual        : (T,) or None   # TRUE IRF - sum of contributions (levels)
    C_agg_dist_term       : (T,) or None   # if D IRF exists: sum_s c_ss(s) * dD_t(s)
    dist_irf_key_used     : str or None
    """

    if isinstance(inputs, str):
        inputs = (inputs,)

    # ---- infer dimensions ----
    example_key = None
    for inp in inputs:
        key = (output, inp)
        if key in model.jac_hh_z:
            example_key = key
            break
    if example_key is None:
        raise KeyError(f"No per-z Jacobians found for {output} and inputs={inputs}")

    J_example = model.jac_hh_z[example_key]
    Nz, T_jac, _ = J_example.shape
    if T is None or T > T_jac:
        T = T_jac

    # ---- get mu_z and cbar_z_ss from ss.D and ss.c ----
    Dss = np.asarray(model.ss.D)
    css = np.asarray(model.ss.c)

    # find z-axis (axis whose length equals Nz)
    z_axes = [ax for ax, n in enumerate(Dss.shape) if n == Nz]
    if len(z_axes) != 1:
        raise ValueError(f"Ambiguous z-axis in ss.D: D.shape={Dss.shape}, Nz={Nz}")
    z_ax = z_axes[0]
    sum_axes = tuple(ax for ax in range(Dss.ndim) if ax != z_ax)

    mu_z = Dss.sum(axis=sum_axes)
    mu_z = np.maximum(mu_z, 0)
    mu_z = mu_z / mu_z.sum()

    Dc = (Dss * css).sum(axis=sum_axes)
    cbar_z_ss = Dc / np.maximum(mu_z, 1e-16)

    # ---- build per-z contributions (as produced by jac_hh_z) ----
    C_z_contrib = {}
    for inp in inputs:
        key = (output, inp)
        if key not in model.jac_hh_z:
            continue
        if inp not in model.IRF:
            raise KeyError(f"model.IRF has no entry for '{inp}'")

        Jz = model.jac_hh_z[key]       # (Nz,T,T)
        x  = model.IRF[inp][:T_jac]    # (T,)

        Cz = np.zeros((Nz, T_jac))
        for iz in range(Nz):
            Cz[iz] = Jz[iz] @ x

        C_z_contrib[inp] = Cz[:, :T]

    if len(C_z_contrib) == 0:
        raise RuntimeError("No contributions computed")

    C_z_total = sum(C_z_contrib.values())

    # ---- candidate aggregation rules for mapping per-z object to aggregate ----
    def agg_apply(Cz, mode):
        if mode == "sum":
            return Cz.sum(axis=0)
        if mode == "mu_sum":
            return (mu_z[:, None] * Cz).sum(axis=0)
        if mode == "mean":
            return Cz.mean(axis=0)
        if mode == "mu_cbar":
            # interpret Cz as %/log change in mean c by z; convert to levels using mu_z*cbar_z_ss
            return (mu_z[:, None] * (cbar_z_ss[:, None] * Cz)).sum(axis=0)
        raise ValueError(f"Unknown agg_mode '{mode}'")

    # choose agg_mode
    if agg_mode == "auto":
        if output not in model.IRF:
            raise KeyError(f"Need model.IRF['{output}'] for agg_mode='auto'")
        y_true = model.IRF[output][:T]  # levels
        modes = ["sum", "mu_sum", "mean", "mu_cbar"]
        errs = {m: np.max(np.abs(agg_apply(C_z_total, m) - y_true)) for m in modes}
        agg_mode_used = min(errs, key=errs.get)
        print(f"[decomp] agg_mode auto-chosen: {agg_mode_used}  max|diff|={errs[agg_mode_used]:.2e}")
    else:
        agg_mode_used = agg_mode

    # aggregate
    C_agg_contrib = {inp: agg_apply(Cz, agg_mode_used) for inp, Cz in C_z_contrib.items()}
    C_agg_total = sum(C_agg_contrib.values())

    # ---- residual (distribution/composition + anything missing) ----
    C_agg_residual = None
    if include_distribution_residual and output in model.IRF:
        C_agg_residual = model.IRF[output][:T] - C_agg_total

    # ---- compute theoretical distribution term if D IRF exists ----
    C_agg_dist_term = None
    dist_irf_key_used = None
    if include_distribution_residual:
        for k in dist_irf_key_candidates:
            if k in model.IRF:
                dist_irf_key_used = k
                break

        if dist_irf_key_used is not None:
            dD = np.asarray(model.IRF[dist_irf_key_used])  # should be same shape as ss.D over time
            # common layouts: (T, ...state dims...) or (...state dims..., T)
            # We handle both by detecting the time axis.
            if dD.shape[0] == T_jac:
                dD_t = dD[:T]
            elif dD.shape[-1] == T_jac:
                dD_t = np.moveaxis(dD, -1, 0)[:T]
            else:
                # can't interpret time axis safely
                dD_t = None

            if dD_t is not None:
                # distribution term: sum_s c_ss(s) * dD_t(s)
                # broadcast css to match dD_t[time, ...]
                C_agg_dist_term = np.tensordot(dD_t, css, axes=(tuple(range(1, dD_t.ndim)), tuple(range(css.ndim))))
                # C_agg_dist_term shape (T,)

    # ---- checks ----
    if check_consistency and output in model.IRF:
        diff = C_agg_total - model.IRF[output][:T]
        print(f"[decomp] max|agg(from z, policy-only) − IRF| = {np.max(np.abs(diff)):.2e}")
        if C_agg_residual is not None:
            print(f"[decomp] residual range: [{C_agg_residual.min():.3e}, {C_agg_residual.max():.3e}]")

    return (C_z_total, C_z_contrib, C_agg_total, C_agg_contrib,
            agg_mode_used, C_agg_residual, C_agg_dist_term, dist_irf_key_used)

import numpy as np
import matplotlib.pyplot as plt

def plot_C_by_z_with_input_decomp_norm(
    model,
    inputs=("Z", "ra", "chi"),
    output="C_hh",
    T=None,
    norm_mode="none",           # "none" | "own_ss" | "elasticity"
    shock_var=None,             # required for elasticity
    title_prefix="C_hh IRF decomposition",
    plot_aggregate=True,
    overlay_true_agg_irf=True,
    add_distribution_residual=True,   # Option 1
    agg_mode="auto",                  # passed into decompose
):
    """
    Per-z decomposition grid + aggregate decomposition in % of SS.
    Compatible with decompose_C_by_z_and_input_norm returning 8 objects:
      (C_z_total, C_z_contrib, C_agg_total, C_agg_contrib,
       agg_mode_used, C_agg_residual, C_agg_dist_term, dist_irf_key_used)

    Option 1: show "Distribution/composition term" = TRUE IRF - policy-only total.
    """

    (C_z, C_z_contrib, C_agg, C_agg_contrib,
     agg_mode_used, C_resid, C_dist_term, dist_key) = decompose_C_by_z_and_input_norm(
        model=model,
        inputs=inputs,
        output=output,
        T=T,
        check_consistency=True,
        agg_mode=agg_mode,
        include_distribution_residual=True,
    )

    Nz, T_eff = C_z.shape
    t = np.arange(T_eff)

    # ---- per-z normalization helpers ----
    if norm_mode == "own_ss":
        C_z_ss, _ = compute_C_hh_z_ss(model)  # you already have this helper

    if norm_mode == "elasticity":
        if shock_var is None:
            raise ValueError("elasticity normalization requires shock_var")
        shock_impact = model.IRF[shock_var][0]
        if abs(shock_impact) < 1e-12:
            raise ValueError("Shock impact is numerically zero")

    def normalize_perz(arr, z=None):
        if norm_mode == "none":
            return arr
        elif norm_mode == "own_ss":
            return 100 * arr / C_z_ss[z]
        elif norm_mode == "elasticity":
            return arr / shock_impact
        else:
            raise ValueError(f"Unknown norm_mode '{norm_mode}'")

    # ---- per-z grid ----
    ncols = int(np.ceil(np.sqrt(Nz)))
    nrows = int(np.ceil(Nz / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=True,
        sharey=True
    )
    axes = np.atleast_1d(axes).flatten()

    for iz in range(Nz):
        ax = axes[iz]
        ax.plot(t, normalize_perz(C_z[iz], z=iz), lw=2.0, label="total")
        for inp in inputs:
            if inp in C_z_contrib:
                ax.plot(
                    t,
                    normalize_perz(C_z_contrib[inp][iz], z=iz),
                    "--",
                    lw=1.4,
                    label=f"from {inp}",
                )
        ax.axhline(0, ls=":", lw=0.8)
        ax.set_title(f"z = {iz}", fontsize=9)
        ax.grid(alpha=0.25)
        if iz == 0:
            ax.legend(frameon=False, fontsize=8)

    for j in range(Nz, len(axes)):
        fig.delaxes(axes[j])

    ylabel = {
        "none": "level deviation",
        "own_ss": "% deviation from own steady state",
        "elasticity": "elasticity (per unit shock impact)",
    }[norm_mode]

    fig.supxlabel("t")
    fig.supylabel(ylabel)
    fig.suptitle(f"{title_prefix} ({norm_mode})", y=1.02)
    plt.tight_layout()
    plt.show()

    # ---- aggregate plot (% of aggregate SS) ----
    if plot_aggregate:
        ssC = getattr(model.ss, output)
        true_levels = model.IRF[output][:T_eff] if output in model.IRF else None

        fig2, ax2 = plt.subplots(figsize=(7.2, 4.2))

        # Policy-only total (from Z+ra+chi as built from jac_hh_z)
        ax2.plot(
            t,
            100 * C_agg / ssC,
            lw=2.8,
            label=f"Policy-only total (agg_mode={agg_mode_used})"
        )

        # Contributions
        for inp in inputs:
            if inp in C_agg_contrib:
                ax2.plot(
                    t,
                    100 * C_agg_contrib[inp] / ssC,
                    "--",
                    lw=2.0,
                    label=inp
                )

        # Option 1: residual = TRUE - policy
        if add_distribution_residual and (C_resid is not None):
            ax2.plot(
                t,
                100 * C_resid / ssC,
                "-.",
                lw=2.6,
                color="black",
                label="Distribution/composition term (residual)"
            )

        # If (rarely) you actually had an explicit dist term computed
        if add_distribution_residual and (C_dist_term is not None):
            ax2.plot(
                t,
                100 * C_dist_term / ssC,
                ":",
                lw=2.4,
                color="gray",
                label=f"Implied dist term (c_ss·d{dist_key})"
            )

        # Overlay TRUE IRF
        if overlay_true_agg_irf and (true_levels is not None):
            ax2.plot(
                t,
                100 * true_levels / ssC,
                lw=3.0,
                color="gray",
                label="TRUE IRF"
            )

        ax2.axhline(0, ls=":", lw=0.9)
        ax2.set_xlabel("Quarters")
        ax2.set_ylabel("% change in C")
        ax2.set_title(f"{output} decomposition: Aggregate (% of SS)")
        ax2.grid(alpha=0.25)
        ax2.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        # Closure check
        if add_distribution_residual and (true_levels is not None) and (C_resid is not None):
            closed = C_agg + C_resid
            print("[agg check] max| (policy + residual) − TRUE | =",
                  f"{np.max(np.abs(closed - true_levels)):.2e}")

    return (C_z, C_z_contrib, C_agg, C_agg_contrib,
            agg_mode_used, C_resid, C_dist_term, dist_key)

import numpy as np
import matplotlib.pyplot as plt

def plot_key_irfs_clean_up_to_three_models(
    model1,
    model2=None,
    model3=None,
    T=40,
    extra=("pi","Y","i","r"),
    title="IRFs",
    ylabel="Deviation from steady state",
    ncols=3,
    figsize_per_col=4.2,
    figsize_per_row=2.8,
    colors=("forestgreen", "navy", "red"),
    labels=("model 1", "model 2", "model 3"),
    ls=("-", "--", "-."),
    force_union=True,   # union of available vars across provided models
):
    """
    Plot key IRFs for 1, 2, or 3 models (overlayed in each subplot).

    Parameters
    ----------
    model1 : required, must have .IRF dict
    model2, model3 : optional, can be None
    force_union : if True, plot variables available in ANY provided model
                  if False, plot only variables available in ALL provided models
    """

    wanted = ["C_hh","A_hh","ra","Z","chi"] + list(extra)

    models = [m for m in (model1, model2, model3) if m is not None]
    if len(models) == 0:
        raise ValueError("At least one model must be provided.")

    # clip styles to number of models
    colors = list(colors)[:len(models)]
    labels = list(labels)[:len(models)]
    ls     = list(ls)[:len(models)]

    def _avail(m):
        out = []
        if not hasattr(m, "IRF"):
            return out
        for v in wanted:
            if v in m.IRF:
                arr = np.asarray(m.IRF[v])
                if np.any(np.isfinite(arr)):
                    out.append(v)
        return out

    avails = [set(_avail(m)) for m in models]

    if force_union:
        available = [v for v in wanted if any(v in a for a in avails)]
    else:
        available = [v for v in wanted if all(v in a for a in avails)]

    if len(available) == 0:
        raise RuntimeError("No requested IRF series found in provided model.IRF dicts.")

    n = len(available)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        squeeze=False
    )

    # ---- margins ----
    left, right, bottom, top = 0.12, 0.98, 0.10, 0.88
    wspace, hspace = 0.30, 0.55
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)

    fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=14)
    fig.text(0.04, 0.5, ylabel, ha="center", va="center", rotation="vertical", fontsize=12)

    def _get_series(m, var):
        if (not hasattr(m, "IRF")) or (var not in m.IRF):
            return None
        s = np.asarray(m.IRF[var]).ravel()
        if s.size == 0 or not np.any(np.isfinite(s)):
            return None
        return s[:T]

    # plot panels
    for idx, var in enumerate(available):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        series_list = [_get_series(m, var) for m in models]

        # plot each provided model if series exists
        for j, s in enumerate(series_list):
            if s is None:
                continue
            ax.plot(np.arange(s.size), s, lw=2.3, color=colors[j], ls=ls[j], label=labels[j])

        ax.axhline(0, color="black", lw=1)
        ax.set_title(var, fontsize=12)
        ax.set_xlabel("quarters")
        ax.grid(alpha=0.25)
        ax.set_ylabel("")

        # legend only once (first panel)
        if idx == 0 and len(models) > 1:
            ax.legend(frameon=True, fontsize=9)

    # turn off unused axes
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.show()
    return available