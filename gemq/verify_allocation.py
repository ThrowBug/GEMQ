"""
Re-solve the bit-allocation ILP and check it against the committed configs.

The pickles under `configs/` were produced by Gurobi, so they double as a
regression baseline for any change to the solver. Run this after touching
`gemq/allocation/ilp_solvers.py`.

The ILP optimum is not necessarily unique -- two different allocations can both be
optimal -- so the hard criterion is the *objective value*, not the per-expert
assignment. Differing assignments at an equal objective are reported for
inspection but do not fail the check.
"""
import argparse
import json
import os.path as osp
import pickle

from gemq.allocate_bits import auto_parse_filename, compute_total_bits
from gemq.allocation.ilp_solvers import AVAILABLE_BACKENDS, GEMQSolver
from gemq.utils.model_utils import get_model_info

# objective values are sums of ~1e3 float64 terms; anything above this is a real change
REL_TOL = 1e-9


def reference_path(model_name, layer_re_path, bpe, bit_cands, extra_constr):
    """Mirror the auto-generated save path used by gemq.allocate_bits."""
    bc_str = ",".join(map(str, bit_cands))
    calib_str, model_str = auto_parse_filename(layer_re_path)
    const_str = "" if extra_constr == "none" else f"_{extra_constr}"
    return f"configs/{model_name}/GEMQ/{calib_str}_E{bpe:.1f}_B{bc_str}{const_str}{model_str}.pkl"


def check_constraints(opt_set, bit_cands, total_bits, extra_constr):
    """Return a list of human-readable constraint violations (empty means clean)."""
    problems = []

    used = sum(b for experts in opt_set.values() for b in experts.values())
    if used > total_bits + 1e-6:
        problems.append(f"budget exceeded ({used} > {total_bits:g})")

    for i, experts in opt_set.items():
        bad = sorted({b for b in experts.values() if b not in bit_cands})
        if bad:
            problems.append(f"layer {i}: bit-widths outside x_space: {bad}")

    if extra_constr == "c2c3":
        for i, experts in opt_set.items():
            vals = list(experts.values())
            for k in sorted(bit_cands, reverse=True)[:2]:
                if vals.count(k) < 1:
                    problems.append(f"layer {i}: no expert at {k} bits (c2c3)")

    return problems


def count_differences(a, b):
    """(differing, total) over (layer, expert) pairs; (None, None) if shapes differ."""
    if set(a) != set(b):
        return None, None
    diff = total = 0
    for i in a:
        if set(a[i]) != set(b[i]):
            return None, None
        for j in a[i]:
            total += 1
            diff += int(a[i][j] != b[i][j])
    return diff, total


def verify_one(solver, args, bit_cands, bpe):
    """Solve for one budget and compare against the reference. Returns a row dict."""
    total_bits = compute_total_bits(args.model_name, bpe, bit_cands)
    opt_set = solver.solve_all(total_bits=total_bits)

    row = {
        "bpe": bpe,
        "obj_new": solver.last_objective,
        "used": sum(b for e in opt_set.values() for b in e.values()),
        "budget": total_bits,
        "obj_ref": None,
        "rel": None,
        "diff": "--",
    }

    problems = check_constraints(opt_set, bit_cands, total_bits, args.extra_constr)

    ref_path = args.reference or reference_path(
        args.model_name, args.layer_re_path, bpe, bit_cands, args.extra_constr
    )
    if not osp.exists(ref_path):
        row["verdict"] = f"NO REFERENCE ({ref_path})"
        row["failed"] = bool(problems)
        if problems:
            row["verdict"] = "FAIL (" + "; ".join(problems[:2]) + ")"
        return row

    with open(ref_path, "rb") as f:
        opt_ref = pickle.load(f)

    obj_ref = solver.compute_objective(opt_ref)
    n_diff, n_total = count_differences(opt_set, opt_ref)

    row["obj_ref"] = obj_ref
    row["rel"] = abs(row["obj_new"] - obj_ref) / max(abs(obj_ref), 1e-30)
    row["diff"] = "--" if n_diff is None else f"{n_diff}/{n_total}"

    if problems:
        row["verdict"] = "FAIL (" + "; ".join(problems[:2]) + ")"
    elif row["obj_new"] > obj_ref * (1 + REL_TOL):
        row["verdict"] = "FAIL (worse than reference)"
    elif row["obj_new"] < obj_ref * (1 - REL_TOL):
        row["verdict"] = "BETTER than reference -- inspect"
    elif n_diff is None:
        row["verdict"] = "OK (obj matches, shape differs)"
    elif n_diff:
        row["verdict"] = "OK (tied optimum, allocation differs)"
    else:
        row["verdict"] = "OK (identical)"

    row["failed"] = row["verdict"].startswith("FAIL")
    return row


def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    bit_cands = list(map(int, args.bit_candidates.split(",")))
    budgets = [float(x) for x in args.bit_budgets.split(",")]

    # the LayerRE table is the same for every budget, so load and parse it once
    solver = GEMQSolver(
        layer_re_path=args.layer_re_path,
        x_space=bit_cands,
        extra_constr=args.extra_constr,
        start_layer_idx=get_model_info(args.model_name).first_k_dense_layers,
        backend=args.ilp_backend,
    )

    rows = [verify_one(solver, args, bit_cands, bpe) for bpe in budgets]

    print(f"\nmodel: {args.model_name}   backend: {args.ilp_backend}")
    header = (f"{'bpe':>5}  {'obj(new)':>14}  {'obj(ref)':>14}  {'rel.diff':>9}  "
              f"{'bits/budget':>15}  {'differing':>11}  verdict")
    print(header)
    print("-" * len(header))
    for r in rows:
        obj_ref = "--" if r["obj_ref"] is None else f"{r['obj_ref']:.7e}"
        rel = "--" if r["rel"] is None else f"{r['rel']:.1e}"
        usage = f"{r['used']:.0f}/{r['budget']:.0f}"
        print(f"{r['bpe']:>5.1f}  {r['obj_new']:>14.7e}  {obj_ref:>14}  {rel:>9}  "
              f"{usage:>15}  {r['diff']:>11}  {r['verdict']}")

    failed = [r["bpe"] for r in rows if r["failed"]]
    print()
    if failed:
        print(f"FAILED for budget(s): {failed}")
        raise SystemExit(1)
    print("All budgets passed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the bit-allocation ILP against the committed configs."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--layer_re_path", type=str, required=True,
        help="Path to the pre-computed weighted layer reconstruction errors",
    )
    parser.add_argument(
        "--bit_budgets", type=str, default="1.5,2.0,2.5",
        help="Comma-separated bits-per-expert budgets to check",
    )
    parser.add_argument("--bit_candidates", type=str, default="1,2,3")
    parser.add_argument("--extra_constr", type=str, default="c2c3")
    parser.add_argument(
        "--ilp_backend", type=str, default="highs", choices=list(AVAILABLE_BACKENDS),
    )
    parser.add_argument(
        "--reference", type=str, default="",
        help="Reference pickle to compare against (default: auto-derive from configs/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
