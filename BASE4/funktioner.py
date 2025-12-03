
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

def decomp(model,model_het, plot_test=False):
    
    # HANK 
    Js_HA = model.jac_hh
    dC_dZ_HA = Js_HA[('C_hh', 'Z')] @ model.IRF['Z']*100/model.ss.C_hh
    dC_dra_HA = Js_HA[('C_hh', 'ra')] @ model.IRF['ra']*100/model.ss.C_hh
    dC_test_HA = dC_dZ_HA + dC_dra_HA
    dC_tot_HA = model.IRF['C_hh']/model.ss.C_hh*100

    # RANK 
    Js_RA = model_het.jac_hh
    dC_dZ_RA = Js_RA[('C_hh', 'Z')] @ model_het.IRF['Z']*100/model_het.ss.C_hh
    dC_dra_RA = Js_RA[('C_hh', 'ra')] @ model_het.IRF['ra']*100/model_het.ss.C_hh
    dC_test_RA = dC_dZ_RA + dC_dra_RA
    dC_tot_RA = model_het.IRF['C_hh']/model_het.ss.C_hh*100

    # PLOT  
    lw = 2.5
    colors = ['navy', 'firebrick', 'forestgreen', 'orange']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Plot RANK 
    axes[0].plot(np.zeros(21), color='black')
    axes[0].plot(dC_tot_RA[:21], linewidth=lw, color=colors[0])
    axes[0].plot(dC_dZ_RA[:21], label='Z', linewidth=lw, color=colors[1])
    axes[0].plot(dC_dra_RA[:21], label='ra', linewidth=lw, color=colors[2])
    if plot_test: axes[0].plot(dC_test_RA[:21], label='Test', linestyle='--', color='orange')
    axes[0].set_title('HET_E_HANK')
    axes[0].set_xlabel('Quarters')
    axes[0].set_ylabel('% change in C') 

    # Plot HANK
    axes[1].plot(np.zeros(21), color='black')
    axes[1].plot(dC_tot_HA[:21], label='Total', linewidth=lw, color=colors[0])
    if plot_test: axes[1].plot(dC_test_HA[:21], label='Test', linestyle='--', color='orange')
    axes[1].plot(dC_dZ_HA[:21], label='Z', linewidth=lw, color=colors[1])
    axes[1].plot(dC_dra_HA[:21], label='ra', linewidth=lw, color=colors[2])
    axes[1].set_title('BASELINE_HANK')
    axes[1].set_xlabel('Quarters')
    axes[1].set_ylabel('% change in C') 

    
    axes[1].legend()
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
