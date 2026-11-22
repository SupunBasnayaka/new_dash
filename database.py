"""
Builds paddy_data.db from the real Department of Census & Statistics Excel files
(kept in the PADDY_DATA_DIR folder next to this file).

Tables
------
paddy_stats           district/scheme/national rows, per season, year and irrigation type
national_paddy_stats  national season totals 1951-2026 (from the PaddyExtent files)

Rules: '-' / blank cells in the source stay NULL. Nothing is estimated or computed.
"""
import os
import re
import sqlite3
import openpyxl

import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PADDY_DATA_DIR", os.path.join(BASE_DIR, "PADDY_DATA_DIR"))
DB_PATH = os.environ.get("PADDY_DB_PATH", os.path.join(BASE_DIR, "paddy_data.db"))

FILES = {
    ("sown", "Maha"): "Sown_Extent-Maha*.xlsx",
    ("sown", "Yala"): "Sown_Extent-Yala*.xlsx",
    ("harvested", "Maha"): "Harvested_Extent-Maha*.xlsx",
    ("harvested", "Yala"): "Harvested_Extent-Yala*.xlsx",
    ("yield", "Maha"): "Average_Yeild-Maha*.xlsx",
    ("yield", "Yala"): "Average_Yeild-Yala*.xlsx",
    ("production", "Maha"): "Production-Maha*.xlsx",
    ("production", "Yala"): "Production-Yala*.xlsx",
}
NATIONAL_FILES = {
    "Maha": "PaddyExtent_Maha*.xlsx",
    "Yala": "PaddyExtent_Yala*.xlsx",
}


def find_file(pattern):
    """Resolve a filename pattern inside DATA_DIR (tolerates ' (1)' / '__1_' suffixes)."""
    hits = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern!r} in {DATA_DIR}")
    return hits[0]

# District -> province (Sri Lanka's 9 provinces)
PROVINCE = {
    "Colombo": "Western", "Gampaha": "Western", "Kalutara": "Western",
    "Kandy": "Central", "Matale": "Central", "Nuwara Eliya": "Central",
    "Galle": "Southern", "Matara": "Southern", "Hambantota": "Southern",
    "Jaffna": "Northern", "Mannar": "Northern", "Vavuniya": "Northern",
    "Mullativu": "Northern", "Kilinochchi": "Northern",
    "Batticaloa": "Eastern", "Ampara": "Eastern", "Trincomalee": "Eastern",
    "Kurunegala": "North Western", "Puttalam": "North Western",
    "Anuradhapura": "North Central", "Polonnaruwa": "North Central",
    "Badulla": "Uva", "Monaragala": "Uva",
    "Ratnapura": "Sabaragamuwa", "Kegalle": "Sabaragamuwa",
}
# Non-district rows in the source files
SCHEMES = {"Udawalawe", "Mahaweli H", "Highland Paddy"}
NATIONAL = "Sri Lanka"

IRRIGATION = ["Major", "Minor", "Rainfed", "Total"]


def norm_name(raw):
    s = re.sub(r"[\s']+", " ", str(raw)).strip().lower()
    s = s.replace("mahaweli h", "mahaweli h")
    if s.startswith("kilinochch") or s.startswith("killinochch"):
        return "Kilinochchi"
    if s.startswith("mahaweli"):
        return "Mahaweli H"
    return s.title()


def area_type(name):
    if name == NATIONAL:
        return "national"
    if name in SCHEMES:
        return "scheme"
    return "district"


def num(v):
    """Numeric value or None. '-' and blanks stay missing."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s in ("", "-", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def season_year(label):
    """Maha '1978/1979' -> 1979 (ends in the following calendar year); Yala 1979 -> 1979."""
    s = str(label).strip()
    m = re.match(r"^(\d{4})\s*/\s*(\d{2,4})$", s)
    if m:
        a, b = m.group(1), m.group(2)
        if len(b) == 4:
            return int(b)
        century = int(a) // 100 * 100
        y = century + int(b)
        return y if y > int(a) else y + 100      # '1999/00' -> 2000
    return int(float(s))


def canon_label(season, year):
    """Canonical season label derived from the year, which also repairs header typos
    (e.g. Maha sown file has '2000/2021' where 2020/2021 belongs)."""
    return f"{year - 1}/{year}" if season == "Maha" else str(year)


def find_row(ws, pred, start=1, end=12):
    for r in range(start, end + 1):
        if pred(ws.cell(r, 2).value, ws.cell(r, 1).value):
            return r
    return None


def read_wide(path, season):
    """Wide files: per year 4 columns (Major, Minor, Rainfed, Total).
    Returns {(district, label, year, irrigation): value}."""
    ws = openpyxl.load_workbook(path, data_only=True).active
    hdr = find_row(ws, lambda b, a: str(b).strip().lower() == "district")
    sub = next(r for r in range(hdr, hdr + 4)
               if str(ws.cell(r, 3).value).strip() == "Major")
    cols = {}
    current = None
    for c in range(3, ws.max_column + 1):
        y = ws.cell(hdr, c).value
        if y is not None:
            current = y
        t = ws.cell(sub, c).value
        if t is None or current is None:
            continue
        t = str(t).strip().capitalize()      # source has 'major'/'Major' mixed
        if t in IRRIGATION:
            cols[c] = (str(current), t)
    out = {}
    for r in range(sub + 1, ws.max_row + 1):
        raw = ws.cell(r, 2).value
        if raw is None or str(raw).strip().lower().startswith("after the"):
            continue
        name = norm_name(raw)
        for c, (label, irr) in cols.items():
            y = season_year(label)
            out[(name, canon_label(season, y), y, irr)] = num(ws.cell(r, c).value)
    return out


def read_production(path, season):
    """Production files: one column per year, total only, unit = '000 Mt.
    Returns {(district, label, year): value_in_mt}."""
    ws = openpyxl.load_workbook(path, data_only=True).active
    hdr = find_row(ws, lambda b, a: str(a).strip().lower() == "district"
                   or str(b).strip().lower() == "district")
    first_col = 1 if str(ws.cell(hdr, 1).value).strip().lower() == "district" else 2
    labels = []
    for c in range(first_col + 1, ws.max_column + 1):
        v = ws.cell(hdr, c).value
        if v is not None:
            labels.append((c, v))
    # Fix source typo: Yala production header has '198' where 1980 belongs.
    if season == "Yala":
        start = season_year(labels[0][1])
        labels = [(c, start + i) for i, (c, _) in enumerate(labels)]
    out = {}
    for r in range(hdr + 1, ws.max_row + 1):
        raw = ws.cell(r, first_col).value
        if raw is None or str(raw).strip().lower().startswith("after the"):
            continue
        name = norm_name(raw)
        for c, label in labels:
            v = num(ws.cell(r, c).value)
            y = season_year(label)
            out[(name, canon_label(season, y), y)] = None if v is None else round(v * 1000, 1)
    return out


def read_national(path, season):
    ws = openpyxl.load_workbook(path, data_only=True).active
    rows = []
    for r in range(1, ws.max_row + 1):
        label = ws.cell(r, 2).value
        if label is None or not re.match(r"^\d{4}(/\d{2})?$", str(label).strip()):
            continue
        vals = [num(ws.cell(r, c).value) for c in range(3, 11)]
        rows.append((season, str(label).strip(), season_year(label), *vals))
    return rows


def ensure_province_column():
    """Make sure paddy_stats has a `province` column and that every district row has its
    province filled in. Safe to run any number of times (adds / back-fills only what is missing)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(paddy_stats)")}
        if "province" not in cols:
            conn.execute("ALTER TABLE paddy_stats ADD COLUMN province TEXT")
        conn.executemany(
            "UPDATE paddy_stats SET province = ? "
            "WHERE district = ? AND area_type = 'district' AND (province IS NULL OR province != ?)",
            [(prov, dist, prov) for dist, prov in PROVINCE.items()],
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_province ON paddy_stats (province, year)")
        conn.commit()
    finally:
        conn.close()


def database_ready():
    """True when paddy_data.db already holds real data."""
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM paddy_stats").fetchone()[0]
        cols = {r[1] for r in conn.execute("PRAGMA table_info(paddy_stats)")}
        conn.close()
        return n > 0 and "area_type" in cols       # old synthetic DB has no area_type
    except sqlite3.Error:
        return False


def initialize_database(force=False):
    """Build the database from the Excel files unless a real-data DB already exists."""
    if not force and database_ready():
        ensure_province_column()        # existing DB: add / back-fill province if needed
        return
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE paddy_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            district TEXT NOT NULL,          -- district, scheme or 'Sri Lanka' (see area_type)
            area_type TEXT NOT NULL,         -- 'district' | 'scheme' | 'national'
            province TEXT,                   -- NULL for schemes / national
            season TEXT NOT NULL,            -- 'Maha' | 'Yala'
            season_label TEXT NOT NULL,      -- as in source: '2023/2024' (Maha) or '2024' (Yala)
            year INTEGER NOT NULL,           -- calendar year the season ends in
            irrigation_type TEXT NOT NULL,   -- 'Major' | 'Minor' | 'Rainfed' | 'Total'
            sown_extent_ha REAL,
            harvested_extent_ha REAL,
            average_yield_kg_ha REAL,
            production_mt REAL,              -- only available for irrigation_type = 'Total'
            fertilizer_type TEXT,            -- not in source data; always NULL
            UNIQUE (district, season, year, irrigation_type)
        )
    """)
    cur.execute("""
        CREATE TABLE national_paddy_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT NOT NULL,
            season_label TEXT NOT NULL,
            year INTEGER NOT NULL,
            sown_000_acres REAL,
            sown_000_ha REAL,
            harvested_000_acres REAL,
            harvested_000_ha REAL,
            avg_yield_bushels_per_acre REAL,
            avg_yield_kg_ha REAL,
            production_000_bushels REAL,
            production_000_mt REAL,
            UNIQUE (season, year)
        )
    """)

    rows = {}
    for season in ("Maha", "Yala"):
        sown = read_wide(find_file(FILES[("sown", season)]), season)
        harv = read_wide(find_file(FILES[("harvested", season)]), season)
        yld = read_wide(find_file(FILES[("yield", season)]), season)
        prod = read_production(find_file(FILES[("production", season)]), season)

        keys = set(sown) | set(harv) | set(yld)
        for (name, label, year, irr) in keys:
            rows[(name, season, label, year, irr)] = [
                sown.get((name, label, year, irr)),
                harv.get((name, label, year, irr)),
                yld.get((name, label, year, irr)),
                None,
            ]
        for (name, label, year), v in prod.items():
            k = (name, season, label, year, "Total")
            rows.setdefault(k, [None, None, None, None])[3] = v

    inserts, dropped = [], 0
    for (name, season, label, year, irr), vals in sorted(
            rows.items(), key=lambda kv: (kv[0][3], kv[0][1], kv[0][0], IRRIGATION.index(kv[0][4]))):
        if all(v is None for v in vals):
            dropped += 1          # nothing known for this combination
            continue
        at = area_type(name)
        prov = PROVINCE.get(name)
        inserts.append((name, at, prov, season, label, year, irr, *vals, None))

    cur.executemany("""
        INSERT INTO paddy_stats (district, area_type, province, season, season_label, year,
            irrigation_type, sown_extent_ha, harvested_extent_ha, average_yield_kg_ha,
            production_mt, fertilizer_type)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, inserts)

    nat = []
    for season, fn in NATIONAL_FILES.items():
        nat += read_national(find_file(fn), season)
    cur.executemany("""
        INSERT INTO national_paddy_stats (season, season_label, year, sown_000_acres, sown_000_ha,
            harvested_000_acres, harvested_000_ha, avg_yield_bushels_per_acre, avg_yield_kg_ha,
            production_000_bushels, production_000_mt)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, nat)

    cur.execute("CREATE INDEX idx_stats_lookup ON paddy_stats (district, season, year)")
    cur.execute("CREATE INDEX idx_stats_province ON paddy_stats (province, year)")
    conn.commit()
    conn.close()
    print(f"paddy_stats: {len(inserts)} rows ({dropped} all-empty combinations skipped)")
    print(f"national_paddy_stats: {len(nat)} rows")


if __name__ == "__main__":
    initialize_database(force=True)
    print("Database initialized with real data.")
