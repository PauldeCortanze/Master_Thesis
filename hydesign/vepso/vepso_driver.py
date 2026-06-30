"""
VESPO – Vector-based Evolutionary Swarm Particle Optimiser
Multi-objective PSO driver for OpenMDAO.

Architecture (2-swarm case, generalises to M swarms for M objectives):
  • One swarm per objective  →  swarm s optimises objective s
  • Each particle has its own personal best (PBest) for its own objective
  • Velocity update (eq. 32):
        v_k^s(t+1) = ω · v_k^s(t)
                   + c1 · r1 · (PBest_k^s  - p_k^s(t))     ← own best
                   + c2 · r2 · (GBest^s'   - p_k^s(t))     ← best from OTHER swarm

  where GBest^s' is the position from the *other* swarm that yielded the
  best value of objective s' (the objective the other swarm is chasing),
  selected via crowding-distance leader selection from the shared Pareto archive.

  The shared Pareto archive collects non-dominated solutions from ALL swarms
  and is used to build the final Pareto front.
"""

import numpy as np
import openmdao.api as om
from openmdao.core.driver import Driver


# ── helpers ───────────────────────────────────────────────────────────────────

def _dominates(a, b):
    """True if vector a dominates b (minimisation)."""
    return np.all(a <= b) and np.any(a < b)


def _crowding_distance(objs):
    """Crowding distance for N×M objective array."""
    N, M = objs.shape
    dist = np.zeros(N)
    if N <= 2:
        dist[:] = np.inf
        return dist
    for m in range(M):
        idx = np.argsort(objs[:, m])
        dist[idx[0]] = dist[idx[-1]] = np.inf
        rng = objs[idx[-1], m] - objs[idx[0], m]
        if rng == 0:
            continue
        for i in range(1, N - 1):
            dist[idx[i]] += (objs[idx[i + 1], m] - objs[idx[i - 1], m]) / rng
    return dist


def _update_archive(arc_pos, arc_obj, new_pos, new_obj, max_size):
    """Insert new solution into shared Pareto archive if non-dominated."""
    if len(arc_pos) == 0:
        return [new_pos.copy()], [new_obj.copy()]

    arr = np.array(arc_obj)
    for ao in arr:
        if _dominates(ao, new_obj):
            return arc_pos, arc_obj          # dominated → reject

    keep    = [not _dominates(new_obj, ao) for ao in arr]
    arc_pos = [p for p, k in zip(arc_pos, keep) if k]
    arc_obj = [o for o, k in zip(arc_obj, keep) if k]
    arc_pos.append(new_pos.copy())
    arc_obj.append(new_obj.copy())

    while len(arc_pos) > max_size:          # prune most-crowded
        cd     = _crowding_distance(np.array(arc_obj))
        victim = int(np.argmin(cd))
        arc_pos.pop(victim)
        arc_obj.pop(victim)

    return arc_pos, arc_obj


def _select_leader(arc_pos, arc_obj, n_cands=5):
    """
    Pick a leader from the archive via crowding-distance tournament.
    Returns the position of the least-crowded candidate (most unexplored region).
    """
    N = len(arc_pos)
    if N == 1:
        return arc_pos[0]
    cd    = _crowding_distance(np.array(arc_obj))
    cands = np.random.choice(N, size=min(n_cands, N), replace=False)
    best  = cands[np.argmax(cd[cands])]
    return arc_pos[best]


# ── driver ────────────────────────────────────────────────────────────────────

class VESPODriver(Driver):
    """
    True VESPO driver: one swarm per objective, cross-swarm GBest influence,
    shared Pareto archive.

    Options (set via driver.options[...]):
        swarm_size     : particles per swarm            (default 20)
        max_iter       : iterations                     (default 100)
        omega          : initial inertia weight         (default 0.7)
        c1             : cognitive acceleration         (default 1.5)
        c2             : social / cross-swarm accel.    (default 1.5)
        omega_damp     : inertia decay per iteration    (default 0.99)
        v_max_frac     : v_max as fraction of DV range  (default 0.2)
        archive_size   : max Pareto archive size        (default 100)
        seed           : random seed (None = random)    (default None)
        constraint_tol : feasibility threshold          (default 1e-6)
        verbose        : print progress                 (default True)
        history_csv    : path for swarm history CSV     (default "swarm_history.csv")

    After run():
        driver.pareto_pos  – list of DV dicts on the Pareto front
        driver.pareto_obj  – list of objective dicts on the Pareto front
        driver.history     – list of dicts, one row per (iter, swarm, particle)
    """

    def __init__(self):
        super().__init__()
        self.pareto_pos = []
        self.pareto_obj = []
        self.history    = []
        self._vespo_set_dv = (
            self._set_design_var
            if hasattr(self, '_set_design_var')
            else self.set_design_var
        )

    def _declare_options(self):
        self.options.declare("swarm_size",     default=20,   lower=2)
        self.options.declare("max_iter",       default=100,  lower=1)
        self.options.declare("omega",          default=0.7,  lower=0.0)
        self.options.declare("c1",             default=1.5,  lower=0.0)
        self.options.declare("c2",             default=1.5,  lower=0.0)
        self.options.declare("omega_damp",     default=0.99)
        self.options.declare("v_max_frac",     default=0.2,  lower=0.0)
        self.options.declare("archive_size",   default=100,  lower=1)
        self.options.declare("seed",           default=None, allow_none=True)
        self.options.declare("constraint_tol", default=1e-6)
        self.options.declare("verbose",        default=True)
        self.options.declare("history_csv",    default="swarm_history.csv",
                             allow_none=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _read_design_vars(self):
        """Return (dv_names, dims, lb_all, ub_all) as flat numpy arrays."""
        desvars = self.get_design_var_values()
        dv_meta = self._problem().model.get_design_vars()
        names   = list(desvars.keys())
        lbs, ubs, dims = [], [], []
        for name in names:
            meta = dv_meta[name]
            lb   = np.atleast_1d(meta.get("lower", -1e30)).astype(float)
            ub   = np.atleast_1d(meta.get("upper",  1e30)).astype(float)
            lb   = np.where(np.isinf(lb), -1e6, lb)
            ub   = np.where(np.isinf(ub),  1e6, ub)
            lbs.append(lb); ubs.append(ub); dims.append(len(lb))
        return names, dims, np.concatenate(lbs), np.concatenate(ubs)

    def _flat_to_dict(self, flat, dv_names, dims):
        d, off = {}, 0
        for name, dim in zip(dv_names, dims):
            d[name] = flat[off:off + dim].copy()
            off += dim
        return d

    def _evaluate(self, flat, dv_names, dims, obj_names, con_names, con_meta, ctol):
        """Set DVs, run model, return (obj_vector, violation_scalar)."""
        dv = self._flat_to_dict(flat, dv_names, dims)
        for name, val in dv.items():
            self._vespo_set_dv(name, val)
        self._problem().run_model()

        objs = np.array([
            float(np.atleast_1d(self.get_objective_values()[n]).ravel()[0])
            for n in obj_names
        ])

        viol = 0.0
        for cname in con_names:
            cval = np.atleast_1d(self.get_constraint_values()[cname])
            cm   = con_meta[cname]
            lo   = cm.get("lower", -np.inf)
            hi   = cm.get("upper",  np.inf)
            if lo is not None and not np.isneginf(lo):
                viol += float(np.sum(np.maximum(0.0, lo - cval)))
            if hi is not None and not np.isposinf(hi):
                viol += float(np.sum(np.maximum(0.0, cval - hi)))

        return objs, viol

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self):
        opt      = self.options
        prob     = self._problem()

        if opt["seed"] is not None:
            np.random.seed(opt["seed"])

        n_part   = opt["swarm_size"]
        max_it   = opt["max_iter"]
        omega    = opt["omega"]
        c1, c2   = opt["c1"], opt["c2"]
        damp     = opt["omega_damp"]
        vf       = opt["v_max_frac"]
        verbose  = opt["verbose"]
        ctol     = opt["constraint_tol"]
        hist_csv = opt["history_csv"]

        dv_names, dims, lb_all, ub_all = self._read_design_vars()
        n_dim  = len(lb_all)
        rng    = ub_all - lb_all
        v_max  = vf * rng

        obj_names = list(self.get_objective_values().keys())
        con_names = list(self.get_constraint_values().keys())
        con_meta  = prob.model.get_constraints()
        n_obj     = len(obj_names)

        if n_obj < 2:
            raise ValueError("VESPO requires at least 2 objectives.")

        if verbose:
            print(f"\n{'='*65}")
            print(f"VESPO  |  {n_obj} swarms × {n_part} particles  |  {max_it} iterations")
            print(f"Objectives  : {obj_names}")
            print(f"Constraints : {con_names}")
            print(f"DV dimension: {n_dim}")
            print(f"{'='*65}")

        # ── initialise one swarm per objective ────────────────────────────
        # pos[s]      : (n_part, n_dim)  positions  of swarm s
        # vel[s]      : (n_part, n_dim)  velocities of swarm s
        # pbest_pos[s]: (n_part, n_dim)  personal best positions
        # pbest_obj[s]: (n_part,)        personal best value of objective s
        pos      = [lb_all + np.random.rand(n_part, n_dim) * rng   for _ in range(n_obj)]
        vel      = [-v_max + np.random.rand(n_part, n_dim) * 2 * v_max for _ in range(n_obj)]
        pbest_pos = [p.copy() for p in pos]
        pbest_val = [np.full(n_part, np.inf) for _ in range(n_obj)]  # scalar per particle

        # shared Pareto archive
        arc_pos: list = []
        arc_obj: list = []

        history: list = []

        # ── main loop ─────────────────────────────────────────────────────
        for it in range(max_it):

            # For each swarm s, compute its best position (GBest^s) to share
            # with the *other* swarms.  We use the archive member that is best
            # for objective s (minimum stored value of obj s), falling back to
            # the personal-best champion of swarm s before the archive fills.
            def _gbest_for_obj(s):
                """Position of the archive leader that best optimises obj s."""
                if len(arc_pos) == 0:
                    # archive empty: use the current personal-best champion of swarm s
                    best_p = int(np.argmin(pbest_val[s]))
                    return pbest_pos[s][best_p]
                arr = np.array(arc_obj)          # shape (N, n_obj)
                best_arc = int(np.argmin(arr[:, s]))
                return arc_pos[best_arc]

            for s in range(n_obj):               # iterate over swarms
                for k in range(n_part):          # iterate over particles

                    objs_k, viol_k = self._evaluate(
                        pos[s][k], dv_names, dims,
                        obj_names, con_names, con_meta, ctol
                    )
                    feasible_k = viol_k <= ctol

                    # ── history row ───────────────────────────────────────
                    row = {
                        "iteration": it + 1,
                        "swarm":     s,
                        "particle":  k,
                        "feasible":  int(feasible_k),
                        "violation": float(viol_k),
                    }
                    for n, v in zip(obj_names, objs_k):
                        row[n] = float(v)
                    for dv_n, dv_v in self._flat_to_dict(pos[s][k], dv_names, dims).items():
                        row[dv_n] = float(np.atleast_1d(dv_v)[0])
                    history.append(row)

                    if feasible_k:
                        # ── personal best (own objective only) ───────────
                        own_obj = float(objs_k[s])
                        if own_obj < pbest_val[s][k]:
                            pbest_val[s][k]  = own_obj
                            pbest_pos[s][k]  = pos[s][k].copy()

                        # ── shared Pareto archive ─────────────────────────
                        arc_pos, arc_obj = _update_archive(
                            arc_pos, arc_obj,
                            pos[s][k], objs_k, opt["archive_size"]
                        )

                    # ── velocity update (eq. 32) ──────────────────────────
                    # GBest^s' = best position from the OTHER swarm
                    # (using round-robin: swarm s gets GBest from swarm s+1)
                    other_s = (s + 1) % n_obj
                    gbest   = _gbest_for_obj(other_s)

                    r1 = np.random.rand(n_dim)
                    r2 = np.random.rand(n_dim)
                    vel[s][k] = (
                        omega * vel[s][k]
                        + c1 * r1 * (pbest_pos[s][k] - pos[s][k])   # cognitive
                        + c2 * r2 * (gbest            - pos[s][k])   # cross-swarm
                    )

                    vel[s][k] = np.clip(vel[s][k], -v_max, v_max)
                    pos[s][k] = np.clip(pos[s][k] + vel[s][k], lb_all, ub_all)

            omega *= damp

            if verbose:
                print(f"  iter {it+1:>4d}/{max_it}"
                      f"  |  archive: {len(arc_pos):>4d}"
                      f"  |  ω={omega:.4f}"
                      f"  |  evals: {(it+1)*n_obj*n_part}")

        # ── write history CSV ─────────────────────────────────────────────
        self.history = history
        if hist_csv and history:
            import csv as _csv
            with open(hist_csv, "w", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(history[0].keys()))
                _w.writeheader()
                _w.writerows(history)
            if verbose:
                print(f"  History → {hist_csv}  ({len(history)} rows)")

        # ── store Pareto front ────────────────────────────────────────────
        self.pareto_pos = [self._flat_to_dict(p, dv_names, dims) for p in arc_pos]
        self.pareto_obj = [{n: v for n, v in zip(obj_names, o)} for o in arc_obj]

        # set model to knee point
        if arc_pos:
            arr      = np.array(arc_obj)
            norm     = (arr - arr.min(0)) / (np.ptp(arr, axis=0) + 1e-30)
            knee     = int(np.argmin(np.linalg.norm(norm, axis=1)))
            for name, val in self._flat_to_dict(arc_pos[knee], dv_names, dims).items():
                self._vespo_set_dv(name, val)
            prob.run_model()

        if verbose:
            print(f"\nVESPO done. Pareto front: {len(self.pareto_pos)} solutions")
            print("="*65)

        return False

    def supports(self, feature, raises=False):
        supported = {
            "inequality_constraints":  True,
            "equality_constraints":    False,
            "linear_constraints":      False,
            "two_sided_constraints":   True,
            "multiple_objectives":     True,
            "gradients":               False,
            "active_set":              False,
            "integer_design_vars":     False,
            "distributed_design_vars": False,
        }
        if feature in supported:
            return supported[feature]
        if raises:
            raise RuntimeError(f"VESPODriver: unsupported feature '{feature}'")
        return False
