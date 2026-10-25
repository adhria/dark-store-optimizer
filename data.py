import numpy as np

# (name, lat, lon, demand weight)
AREAS = [
    ("T Nagar",     13.0418, 80.2341, 1.3),
    ("Adyar",       13.0012, 80.2565, 1.1),
    ("Anna Nagar",  13.0850, 80.2101, 1.3),
    ("Velachery",   12.9815, 80.2180, 1.2),
    ("Guindy",      13.0067, 80.2206, 0.9),
    ("Tambaram",    12.9249, 80.1000, 1.0),
    ("Porur",       13.0382, 80.1565, 1.0),
    ("Mylapore",    13.0368, 80.2676, 0.9),
    ("OMR",         12.9010, 80.2279, 1.2),
    ("Perambur",    13.1186, 80.2330, 0.9),
]

KM_PER_DEG = 111.0

def _sample(n, sigma_km, rng):
    w = np.array([a[3] for a in AREAS]); w = w / w.sum()
    idx = rng.choice(len(AREAS), size=n, p=w)
    lat0 = np.array([AREAS[i][1] for i in idx])
    lon0 = np.array([AREAS[i][2] for i in idx])
    lat = lat0 + rng.normal(0, sigma_km / KM_PER_DEG, n)
    lon = lon0 + rng.normal(0, sigma_km / (KM_PER_DEG * 0.97), n)
    return lat, lon, [AREAS[i][0] for i in idx]

def make_customers(n, seed=42):
    rng = np.random.default_rng(seed)
    return _sample(n, sigma_km=2.0, rng=rng)

def make_candidates(m, seed=7):
    rng = np.random.default_rng(seed)
    return _sample(m, sigma_km=3.0, rng=rng)

def haversine_matrix(lat1, lon1, lat2, lon2):
    """Distance in km, shape (len(1), len(2))."""
    R = 6371.0
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dphi = p2 - p1
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
