import numpy as np
from collections import defaultdict
from ortools.linear_solver import pywraplp
from data import make_customers, make_candidates, haversine_matrix

PROFIT_PER_CUST = 150    # Rs per customer per month
COST_PER_KM     = 8      # Rs per km delivered
FIXED_COST      = 60000  # Rs per open store per month
CAPACITY        = 1500   # customers per store
CELL_KM         = 0.5    # grid cell size for aggregating customers

def aggregate(lat, lon, cell_km=CELL_KM):
    dlat = cell_km / 111.0
    dlon = cell_km / (111.0 * 0.97)
    key = np.stack([np.floor(lat / dlat), np.floor(lon / dlon)], axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    return (np.bincount(inv, weights=lat) / counts,
            np.bincount(inv, weights=lon) / counts,
            counts.astype(float))

def solve(n_cust, n_cand, max_dist, max_stores, seed=42, time_limit=60):
    c_lat, c_lon, _ = make_customers(n_cust, seed)
    s_lat, s_lon, _ = make_candidates(n_cand, seed + 1)
    g_lat, g_lon, w = aggregate(c_lat, c_lon)
    D = haversine_matrix(g_lat, g_lon, s_lat, s_lon)      # cells x stores

    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SetTimeLimit(time_limit * 1000)

    y = [solver.BoolVar(f"y{i}") for i in range(n_cand)]

    # x_ci = fraction of cell c's demand served by store i; exists only if d_ci <= max_dist
    x = {}
    by_cell, by_store = defaultdict(list), defaultdict(list)
    cs, ss = np.where(D <= max_dist)                      # DISTANCE CONSTRAINT
    for c, i in zip(cs.tolist(), ss.tolist()):
        v = solver.NumVar(0, 1, "")
        x[(c, i)] = v
        by_cell[c].append(v)
        by_store[i].append((float(w[c]), v))
        solver.Add(v <= y[i])                             # serve only from open stores

    for c, vs in by_cell.items():                         # at most 100% of a cell's demand
        solver.Add(solver.Sum(vs) <= 1)
    for i, items in by_store.items():                     # capacity
        solver.Add(solver.Sum([wc * v for wc, v in items]) <= CAPACITY * y[i])
    solver.Add(solver.Sum(y) <= max_stores)               # store limit K

    solver.Maximize(
        solver.Sum([float(w[c]) * (PROFIT_PER_CUST - COST_PER_KM * float(D[c, i])) * v
                    for (c, i), v in x.items()])
        - FIXED_COST * solver.Sum(y)
    )

    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(params.RELATIVE_MIP_GAP, 0.01)
    status = solver.Solve(params)
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None

    opened = [i for i in range(n_cand) if y[i].solution_value() > 0.5]
    flows = []                                            # (cell, store, customers served)
    for (c, i), v in x.items():
        f = v.solution_value()
        if f > 1e-6:
            flows.append((c, i, f * float(w[c])))

    served = sum(f for _, _, f in flows)
    dist_w = sum(f * float(D[c, i]) for c, i, f in flows)
    max_d = max((float(D[c, i]) for c, i, _ in flows), default=0.0)
    revenue = PROFIT_PER_CUST * served
    delivery = COST_PER_KM * dist_w
    fixed = FIXED_COST * len(opened)

    kpis = {
        "Stores opened": len(opened),
        "Customers served": round(served),
        "% customers served": round(100 * served / n_cust, 1),
        "Gross margin (Rs)": round(revenue),
        "Delivery cost (Rs)": round(delivery),
        "Fixed cost (Rs)": fixed,
        "Net profit (Rs)": round(revenue - delivery - fixed),
        "Avg distance (km)": round(dist_w / served, 2) if served else 0.0,
        "Max distance (km)": round(max_d, 2),
        "Avg store utilisation (%)": round(100 * served / (CAPACITY * max(len(opened), 1)), 1),
    }
    return dict(kpis=kpis, opened=opened, flows=flows,
                g_lat=g_lat, g_lon=g_lon, g_w=w, s_lat=s_lat, s_lon=s_lon,
                status="OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE")
