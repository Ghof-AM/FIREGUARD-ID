"""
FIREGUARD ID — Self-Contained Decision-Support Leaflet Map Generator (Phase 10).

Generates 8 interactive HTML maps (plus updated legacy maps `peta_titik_api_riau.html`
and `peta_kerawanan_grid_riau.html`) under `outputs/maps/` and writes a machine-readable
manifest at `reports/visualization_manifest.json`.

Every map enforces FIREGUARD ID methodological guardrails:
  - Explicit CRS note (EPSG:4326) and observation period (2021-01-01 to 2025-12-31)
  - Visible NASA FIRMS VIIRS N20 + GADM 4.1 (IDN.24_1) attribution
  - Explicit separation of RAW OBSERVATIONS vs DERIVED INDICATORS vs
    GAP ANALYSIS vs SYNTHETIC OPTIMIZATION SCENARIOS (is_synthetic_scenario=True)
  - Clear legends, physical units, and QA caveats (zero detections != zero fire risk;
    hotspots != unique fire incidents; cluster labels != risk scores)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.geospatial.geometry import haversine_km
from src.optimization.base_location import evaluate_base_selection
from src.optimization.common import select_candidate_sites
from src.utils import ensure_dir, get_logger, get_repo_root, load_yaml, resolve_path, sha256_file

logger = get_logger("fireguard.visualization.maps")

LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
BASEMAP_DARK = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
    "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
BASEMAP_OSM = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
PEKANBARU_COORDS = (0.5071, 101.4478)


def _clean_for_json(obj: Any) -> Any:
    """Recursively converts NaN/Inf/numpy scalars to JSON-safe Python primitives."""
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if pd.isna(obj):
        return None
    return str(obj)


def _json_dumps(obj: Any) -> str:
    """Serializes Python structures with strict valid JSON (no NaN/Infinity literals)."""
    return json.dumps(_clean_for_json(obj), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _round_or_none(val: Any, digits: int = 4) -> float | None:
    if val is None or pd.isna(val):
        return None
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_html_page(
    *,
    title: str,
    subtitle: str,
    category_badge: str,
    badge_bg: str,
    badge_fg: str,
    banner_html: str,
    legend_html: str,
    controls_html: str,
    stats_html: str,
    notes_html: str,
    extra_body_html: str,
    extra_css: str,
    map_script_js: str,
    period_text: str,
    attribution_text: str,
) -> str:
    """Builds a self-contained dark-mode Leaflet HTML page with standard FIREGUARD chrome."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="{LEAFLET_CSS}" />
  <script src="{LEAFLET_JS}"></script>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; padding: 0; width: 100%; height: 100%;
      background: #0f172a; color: #e2e8f0;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      overflow: hidden;
    }}
    #map {{ width: 100%; height: 100%; background: #0f172a; }}
    .top-banner {{
      position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
      z-index: 1100; max-width: 720px; width: calc(100% - 420px);
      padding: 8px 14px; border-radius: 8px; font-size: 12px; line-height: 1.35;
      text-align: center; font-weight: 600; backdrop-filter: blur(8px);
      box-shadow: 0 4px 18px rgba(0,0,0,0.45);
    }}
    .banner-synthetic {{
      background: rgba(153, 27, 27, 0.92); color: #fef2f2;
      border: 1px solid #f87171;
    }}
    .banner-gap {{
      background: rgba(146, 64, 14, 0.92); color: #fef3c7;
      border: 1px solid #fbbf24;
    }}
    .banner-derived {{
      background: rgba(15, 23, 42, 0.88); color: #cbd5e1;
      border: 1px solid rgba(56, 189, 248, 0.4);
    }}
    .banner-raw {{
      background: rgba(30, 41, 59, 0.90); color: #fed7aa;
      border: 1px solid rgba(251, 146, 60, 0.55);
    }}
    .info-panel {{
      position: absolute; top: 14px; right: 14px; z-index: 1050;
      width: 345px; max-height: calc(100vh - 54px); overflow-y: auto;
      background: rgba(15, 23, 42, 0.92); color: #e2e8f0;
      padding: 14px 16px; border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.25);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.55);
      backdrop-filter: blur(10px);
    }}
    .info-panel h2 {{ margin: 0 0 4px 0; font-size: 16px; color: #f97316; letter-spacing: 0.2px; }}
    .subtitle {{ font-size: 11.5px; color: #94a3b8; margin-bottom: 8px; }}
    .cat-badge {{
      display: inline-block; font-size: 10.5px; font-weight: 700;
      letter-spacing: 0.5px; text-transform: uppercase;
      padding: 3px 8px; border-radius: 5px; margin-bottom: 8px;
      background: {badge_bg}; color: {badge_fg};
    }}
    .meta-row {{ font-size: 11.5px; color: #cbd5e1; margin: 3px 0; line-height: 1.35; }}
    .meta-row strong {{ color: #f8fafc; }}
    .section-box {{
      margin-top: 10px; padding-top: 9px;
      border-top: 1px solid rgba(148, 163, 184, 0.2);
      font-size: 11.5px;
    }}
    .section-title {{
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.45px; color: #38bdf8; margin-bottom: 6px;
    }}
    .legend-item {{ display: flex; align-items: center; margin: 4px 0; font-size: 11.5px; color: #e2e8f0; }}
    .swatch {{
      width: 14px; height: 12px; border-radius: 3px; margin-right: 8px;
      flex-shrink: 0; display: inline-block; border: 1px solid rgba(255,255,255,0.25);
    }}
    .dot {{
      width: 11px; height: 11px; border-radius: 50%; margin-right: 8px;
      flex-shrink: 0; display: inline-block; border: 1px solid rgba(255,255,255,0.35);
    }}
    .swatch-dashed {{ border-style: dashed !important; border-color: #cbd5e1 !important; }}
    .ctrl-group {{ margin: 5px 0; }}
    .ctrl-group label {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; cursor: pointer; font-size: 11.5px; }}
    .ctrl-select {{
      width: 100%; padding: 5px 8px; border-radius: 6px; margin-top: 3px;
      background: #1e293b; color: #f8fafc; border: 1px solid #475569; font-size: 11.5px;
    }}
    .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 4px; }}
    .stat-card {{
      background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 6px; padding: 6px 8px;
    }}
    .stat-val {{ font-size: 13px; font-weight: 700; color: #38bdf8; }}
    .stat-lbl {{ font-size: 10px; color: #94a3b8; }}
    .qa-note {{
      margin-top: 8px; padding: 7px 9px; border-radius: 6px;
      background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35);
      color: #fde68a; font-size: 11px; line-height: 1.38;
    }}
    .leaflet-popup-content-wrapper {{
      background: #0f172a; color: #e2e8f0;
      border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 8px;
    }}
    .leaflet-popup-tip {{ background: #0f172a; }}
    .leaflet-popup-content {{ font-size: 12px; line-height: 1.45; margin: 10px 12px; }}
    .leaflet-control-attribution {{
      background: rgba(15, 23, 42, 0.85) !important;
      color: #cbd5e1 !important; font-size: 10.5px !important;
      padding: 3px 8px !important;
    }}
    .leaflet-control-attribution a {{ color: #38bdf8 !important; }}
    {extra_css}
  </style>
</head>
<body>
  <div id="map"></div>
  {banner_html}
  <div class="info-panel">
    <div class="cat-badge">{category_badge}</div>
    <h2>{title}</h2>
    <div class="subtitle">{subtitle}</div>
    <div class="meta-row"><strong>Region:</strong> Riau Province, Indonesia (GADM 4.1 <code>IDN.24_1</code>)</div>
    <div class="meta-row"><strong>CRS:</strong> EPSG:4326 (WGS 84 Geographic)</div>
    <div class="meta-row"><strong>Period:</strong> {period_text}</div>
    <div class="meta-row"><strong>Attribution:</strong> {attribution_text}</div>

    {f'<div class="section-box">{stats_html}</div>' if stats_html else ''}
    {f'<div class="section-box">{controls_html}</div>' if controls_html else ''}
    {f'<div class="section-box">{legend_html}</div>' if legend_html else ''}
    {f'<div class="qa-note">{notes_html}</div>' if notes_html else ''}
  </div>
  {extra_body_html}

  <script>
    var map = L.map('map', {{ preferCanvas: true, zoomControl: true }}).setView([0.55, 101.65], 8);
    var darkLayer = L.tileLayer('{BASEMAP_DARK}', {{
      maxZoom: 16,
      attribution: 'NASA FIRMS VIIRS N20 (2021-2025) | GADM 4.1 IDN.24_1 | CRS EPSG:4326 | Basemap: Esri'
    }}).addTo(map);
    var osmLayer = L.tileLayer('{BASEMAP_OSM}', {{
      maxZoom: 18,
      attribution: 'NASA FIRMS VIIRS N20 | GADM 4.1 | &copy; OpenStreetMap contributors'
    }});
    L.control.layers({{ 'Esri Dark Gray': darkLayer, 'OpenStreetMap': osmLayer }}, null, {{ position: 'topleft' }}).addTo(map);
    L.control.scale({{ imperial: false, position: 'bottomleft' }}).addTo(map);

    function addRiauBoundary(geo) {{
      return L.geoJSON(geo, {{
        style: {{
          color: '#38bdf8', weight: 1.8, opacity: 0.9,
          fillColor: '#0ea5e9', fillOpacity: 0.03
        }}
      }}).addTo(map);
    }}

    {map_script_js}
  </script>
</body>
</html>
"""


def _prepare_grid_records(df: pd.DataFrame, extra_cols: list[str]) -> list[dict[str, Any]]:
    """Extracts compact, JSON-safe cell geometry and requested attributes from a grid DataFrame."""
    base_cols = [
        "grid_id",
        "lat_min",
        "lon_min",
        "lat_max",
        "lon_max",
        "land_rep_lat",
        "land_rep_lon",
        "spatial_block",
    ]
    cols = base_cols + [c for c in extra_cols if c not in base_cols]
    records: list[dict[str, Any]] = []
    for _, row in df[cols].iterrows():
        rec: dict[str, Any] = {}
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                rec[c] = _round_or_none(val, 4)
            elif isinstance(val, (bool, np.bool_)):
                rec[c] = bool(val)
            elif isinstance(val, (int, np.integer)):
                rec[c] = int(val)
            else:
                rec[c] = None if pd.isna(val) else val
        records.append(rec)
    return records


def _build_map_01_historical_hotspots(
    hotspots_df: pd.DataFrame,
    boundary_geo: dict[str, Any],
) -> str:
    """Map 1: Raw VIIRS N20 historical hotspot observations (2021-2025)."""
    points = [
        [
            round(float(r.latitude), 5),
            round(float(r.longitude), 5),
            round(float(r.frp), 2) if pd.notnull(r.frp) else 0.0,
            str(r.acq_date),
            str(r.confidence).lower(),
            str(r.daynight),
        ]
        for r in hotspots_df[["latitude", "longitude", "frp", "acq_date", "confidence", "daynight"]].itertuples(index=False)
    ]
    conf_counts = hotspots_df["confidence"].astype(str).str.lower().value_counts().to_dict()
    n_high = int(conf_counts.get("h", 0))
    n_nom = int(conf_counts.get("n", 0))
    n_low = int(conf_counts.get("l", 0))
    years = sorted(hotspots_df["acq_date"].astype(str).str.slice(0, 4).unique().tolist())

    year_options = '<option value="all">All Years (2021-2025)</option>' + "".join(
        f'<option value="{y}">{y}</option>' for y in years
    )
    controls_html = f"""
    <div class="section-title">Interactive Filters</div>
    <div class="ctrl-group">
      <label for="yearFilter" style="display:block;font-size:11px;color:#94a3b8;">Acquisition Year</label>
      <select id="yearFilter" class="ctrl-select">{year_options}</select>
    </div>
    <div class="ctrl-group">
      <label for="confFilter" style="display:block;font-size:11px;color:#94a3b8;">QA Confidence Level</label>
      <select id="confFilter" class="ctrl-select">
        <option value="all">All Confidence Levels ({len(points):,})</option>
        <option value="h">High QA ('h') ({n_high:,})</option>
        <option value="n">Nominal QA ('n') ({n_nom:,})</option>
        <option value="l">Low QA ('l') ({n_low:,})</option>
      </select>
    </div>
    """
    stats_html = f"""
    <div class="section-title">Observed Detections Summary</div>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-val" id="shownCount">{len(points):,}</div><div class="stat-lbl">Displayed Detections</div></div>
      <div class="stat-card"><div class="stat-val">{len(points):,}</div><div class="stat-lbl">Total Clipped (595,676 raw IDN)</div></div>
      <div class="stat-card"><div class="stat-val">{hotspots_df['frp'].median():.2f} MW</div><div class="stat-lbl">Median FRP (Max {hotspots_df['frp'].max():.1f} MW)</div></div>
      <div class="stat-card"><div class="stat-val">{n_high:,} / {n_nom:,} / {n_low:,}</div><div class="stat-lbl">QA Conf (h / n / l)</div></div>
    </div>
    """
    legend_html = """
    <div class="section-title">Legend (QA Confidence &amp; FRP Size)</div>
    <div class="legend-item"><span class="dot" style="background:#ef4444;"></span> High Confidence ('h') — QA metadata</div>
    <div class="legend-item"><span class="dot" style="background:#f97316;"></span> Nominal Confidence ('n') — QA metadata</div>
    <div class="legend-item"><span class="dot" style="background:#facc15;"></span> Low Confidence ('l') — QA metadata</div>
    <div class="meta-row" style="margin-top:5px;color:#94a3b8;">Marker radius scales with Fire Radiative Power (FRP, units: MW).</div>
    """
    notes_html = (
        "<strong>Methodological Guardrail:</strong> Raw VIIRS N20 thermal anomaly detections are "
        "<strong>NOT</strong> unique fire incidents, burned areas, or complete fire perimeters. "
        "Detection confidence (<code>l/n/h</code>) is QA metadata only (not a risk/vulnerability driver). "
        "Cloud cover, canopy, and swath gaps affect detection completeness."
    )
    js = f"""
    var riauBoundary = {_json_dumps(boundary_geo)};
    addRiauBoundary(riauBoundary);
    var rawPoints = {_json_dumps(points)};
    var canvasRenderer = L.canvas({{ padding: 0.5 }});
    var pointsLayer = L.layerGroup().addTo(map);
    var confColor = {{ h: '#ef4444', n: '#f97316', l: '#facc15' }};

    function renderPoints() {{
      pointsLayer.clearLayers();
      var yr = document.getElementById('yearFilter').value;
      var cf = document.getElementById('confFilter').value;
      var shown = 0;
      for (var i = 0; i < rawPoints.length; i++) {{
        var p = rawPoints[i];
        if (yr !== 'all' && p[3].slice(0, 4) !== yr) continue;
        if (cf !== 'all' && p[4] !== cf) continue;
        shown++;
        var c = confColor[p[4]] || '#94a3b8';
        var rad = Math.min(Math.max(Math.sqrt(Math.max(p[2], 0.2)) * 1.15, 2.5), 10);
        L.circleMarker([p[0], p[1]], {{
          renderer: canvasRenderer,
          radius: rad,
          fillColor: c,
          color: c,
          weight: 0.6,
          opacity: 0.85,
          fillOpacity: 0.62
        }}).bindPopup(
          '<strong>VIIRS N20 Hotspot Detection (Raw Observation)</strong><br>' +
          'Date: <b>' + p[3] + '</b> (' + (p[5] === 'N' ? 'Night' : 'Day') + ')<br>' +
          'FRP: <b>' + p[2].toFixed(2) + ' MW</b><br>' +
          'QA Confidence: <b>' + p[4].toUpperCase() + '</b><br>' +
          'Coordinates (EPSG:4326): ' + p[0].toFixed(4) + ', ' + p[1].toFixed(4) + '<br>' +
          '<em>Note: Single thermal detection, not a unique fire incident.</em>'
        ).addTo(pointsLayer);
      }}
      document.getElementById('shownCount').textContent = shown.toLocaleString();
    }}
    document.getElementById('yearFilter').addEventListener('change', renderPoints);
    document.getElementById('confFilter').addEventListener('change', renderPoints);
    renderPoints();
    """
    return _build_html_page(
        title="FIREGUARD ID — 01 Historical VIIRS N20 Hotspots",
        subtitle="Polygon-clipped thermal anomaly detections across Riau Province",
        category_badge="RAW OBSERVATIONS",
        badge_bg="rgba(234, 88, 12, 0.25)",
        badge_fg="#fdba74",
        banner_html='<div class="top-banner banner-raw">RAW OBSERVATIONS — NASA FIRMS VIIRS N20 Detections (2021-2025). Hotspots are thermal anomalies, NOT unique fire incidents or burned area.</div>',
        legend_html=legend_html,
        controls_html=controls_html,
        stats_html=stats_html,
        notes_html=notes_html,
        extra_body_html="",
        extra_css="",
        map_script_js=js,
        period_text="2021-01-01 to 2025-12-31 (detections: 2021-01-05 to 2025-12-29)",
        attribution_text="NASA FIRMS VIIRS N20 &amp; GADM v4.1 (IDN.24_1, 89,874.50 km&sup2;)",
    )


def _build_map_02_spatial_grid(
    grid_df: pd.DataFrame,
    boundary_geo: dict[str, Any],
) -> str:
    """Map 2: 853 intersecting 0.1 deg grid cells with density/km2 and zero-detection flag."""
    recs = _prepare_grid_records(
        grid_df,
        [
            "hotspot_density_km2",
            "frp_total_density_mw_km2",
            "hotspot_count",
            "fire_days",
            "active_years_count",
            "riau_intersection_area_km2",
            "riau_area_fraction",
            "has_zero_detections",
            "temporal_completeness_flag",
        ],
    )
    n_zero = int(grid_df["has_zero_detections"].sum())
    max_dens = float(grid_df["hotspot_density_km2"].max())
    max_frp_dens = float(grid_df["frp_total_density_mw_km2"].max())

    stats_html = f"""
    <div class="section-title">0.1&deg; Intersecting Grid Metrics</div>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-val">{len(recs)}</div><div class="stat-lbl">Intersecting Grid Cells</div></div>
      <div class="stat-card"><div class="stat-val">{n_zero} ({100*n_zero/len(recs):.1f}%)</div><div class="stat-lbl">Zero-Detection Cells</div></div>
      <div class="stat-card"><div class="stat-val">{max_dens:.2f} /km&sup2;</div><div class="stat-lbl">Max Hotspot Density</div></div>
      <div class="stat-card"><div class="stat-val">{max_frp_dens:.2f} MW/km&sup2;</div><div class="stat-lbl">Max FRP Density</div></div>
    </div>
    """
    controls_html = """
    <div class="section-title">Display Metric &amp; QA Highlight</div>
    <div class="ctrl-group">
      <label><input type="radio" name="gridMetric" value="hotspot" checked /> Hotspot Density (detections / km&sup2;)</label>
      <label><input type="radio" name="gridMetric" value="frp" /> FRP Total Density (MW / km&sup2;)</label>
      <label style="margin-top:6px;"><input type="checkbox" id="highlightZero" checked /> Emphasize Zero-Detection Cells (dashed outline)</label>
    </div>
    """
    legend_html = """
    <div class="section-title" id="gridLegendTitle">Hotspot Density (detections / km&sup2;)</div>
    <div id="gridLegendItems">
      <div class="legend-item"><span class="swatch swatch-dashed" style="background:#334155;"></span> Zero detections observed (0.00, n=249)</div>
      <div class="legend-item"><span class="swatch" style="background:#fef08a;"></span> &gt;0.00 – 0.05 det/km&sup2;</div>
      <div class="legend-item"><span class="swatch" style="background:#facc15;"></span> &gt;0.05 – 0.20 det/km&sup2;</div>
      <div class="legend-item"><span class="swatch" style="background:#f97316;"></span> &gt;0.20 – 0.50 det/km&sup2;</div>
      <div class="legend-item"><span class="swatch" style="background:#dc2626;"></span> &gt;0.50 det/km&sup2;</div>
    </div>
    """
    notes_html = (
        "<strong>Observation Coverage Caveat:</strong> <code>has_zero_detections=True</code> (249 cells) "
        "indicates zero VIIRS N20 detections during 2021–2025, which does <strong>NOT</strong> prove zero "
        "fire risk due to cloud cover, peat/forest canopy obstruction, and satellite overpass timing. "
        "Density uses exact geodesic Riau intersection area (km&sup2;)."
    )
    js = f"""
    var riauBoundary = {_json_dumps(boundary_geo)};
    var gridData = {_json_dumps(recs)};
    var gridLayer = L.layerGroup().addTo(map);
    addRiauBoundary(riauBoundary);

    function colorHotspot(d, isZero) {{
      if (isZero || d <= 0) return '#334155';
      if (d <= 0.05) return '#fef08a';
      if (d <= 0.20) return '#facc15';
      if (d <= 0.50) return '#f97316';
      return '#dc2626';
    }}
    function colorFrp(d, isZero) {{
      if (isZero || d <= 0) return '#334155';
      if (d <= 0.25) return '#bae6fd';
      if (d <= 1.00) return '#38bdf8';
      if (d <= 3.00) return '#f59e0b';
      return '#dc2626';
    }}
    function renderGrid() {{
      gridLayer.clearLayers();
      var metric = document.querySelector('input[name="gridMetric"]:checked').value;
      var hlZero = document.getElementById('highlightZero').checked;
      document.getElementById('gridLegendTitle').textContent =
        metric === 'hotspot' ? 'Hotspot Density (detections / km\\u00b2)' : 'FRP Total Density (MW / km\\u00b2)';
      document.getElementById('gridLegendItems').innerHTML = metric === 'hotspot'
        ? '<div class="legend-item"><span class="swatch swatch-dashed" style="background:#334155;"></span> Zero detections observed (0.00, n=249)</div>' +
          '<div class="legend-item"><span class="swatch" style="background:#fef08a;"></span> &gt;0.00 – 0.05 det/km&sup2;</div>' +
          '<div class="legend-item"><span class="swatch" style="background:#facc15;"></span> &gt;0.05 – 0.20 det/km&sup2;</div>' +
          '<div class="legend-item"><span class="swatch" style="background:#f97316;"></span> &gt;0.20 – 0.50 det/km&sup2;</div>' +
          '<div class="legend-item"><span class="swatch" style="background:#dc2626;"></span> &gt;0.50 det/km&sup2;</div>'
        : '<div class="legend-item"><span class="swatch swatch-dashed" style="background:#334155;"></span> Zero detections observed (0.00 MW/km&sup2;, n=249)</div>' +
          '<div class="legend-item"><span class="swatch" style="background:#bae6fd;"></span> &gt;0.00 – 0.25 MW/km&sup2;</div>' +
          '<div class="legend-item"><span class="swatch" style="background:#38bdf8;"></span> &gt;0.25 – 1.00 MW/km&sup2;</div>' +
          '<div class="legend-item"><span class="swatch" style="background:#f59e0b;"></span> &gt;1.00 – 3.00 MW/km&sup2;</div>' +
          '<div class="legend-item"><span class="swatch" style="background:#dc2626;"></span> &gt;3.00 MW/km&sup2;</div>';

      for (var i = 0; i < gridData.length; i++) {{
        var g = gridData[i];
        var val = metric === 'hotspot' ? g.hotspot_density_km2 : g.frp_total_density_mw_km2;
        var fill = metric === 'hotspot' ? colorHotspot(val, g.has_zero_detections) : colorFrp(val, g.has_zero_detections);
        var isZero = g.has_zero_detections;
        L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
          color: (isZero && hlZero) ? '#94a3b8' : '#1e293b',
          weight: (isZero && hlZero) ? 1.1 : 0.6,
          dashArray: (isZero && hlZero) ? '3,3' : null,
          fillColor: fill,
          fillOpacity: isZero ? 0.42 : 0.68
        }}).bindPopup(
          '<strong>' + g.grid_id + '</strong> (Block: ' + g.spatial_block + ')<br>' +
          'Hotspot Density: <b>' + g.hotspot_density_km2.toFixed(4) + ' det/km&sup2;</b><br>' +
          'FRP Density: <b>' + g.frp_total_density_mw_km2.toFixed(4) + ' MW/km&sup2;</b><br>' +
          'Hotspot Count: ' + g.hotspot_count + ' | Fire Days: ' + g.fire_days + ' | Active Years: ' + g.active_years_count + '<br>' +
          'Riau Intersection Area: ' + g.riau_intersection_area_km2.toFixed(2) + ' km&sup2; (' + (g.riau_area_fraction*100).toFixed(1) + '%)<br>' +
          'Zero-Detection Flag: <b>' + (isZero ? 'TRUE (No observed detections)' : 'FALSE') + '</b>'
        ).addTo(gridLayer);
      }}
    }}
    document.querySelectorAll('input[name="gridMetric"]').forEach(function(el) {{ el.addEventListener('change', renderGrid); }});
    document.getElementById('highlightZero').addEventListener('change', renderGrid);
    renderGrid();
    """
    return _build_html_page(
        title="FIREGUARD ID — 02 Spatial Analytical Grid (0.1&deg;)",
        subtitle="853 intersecting cells with geodesic area-normalized density & zero-detection QA flags",
        category_badge="DERIVED INDICATORS",
        badge_bg="rgba(14, 165, 233, 0.22)",
        badge_fg="#7dd3fc",
        banner_html='<div class="top-banner banner-derived">DERIVED INDICATORS — 0.1&deg; Intersecting Grid (853 cells, 15,510 detections). Zero detections do NOT prove zero fire risk.</div>',
        legend_html=legend_html,
        controls_html=controls_html,
        stats_html=stats_html,
        notes_html=notes_html,
        extra_body_html="",
        extra_css="",
        map_script_js=js,
        period_text="2021-01-01 to 2025-12-31 (1,826 days)",
        attribution_text="NASA FIRMS VIIRS N20 &amp; GADM v4.1 (IDN.24_1)",
    )


def _build_map_03_clustering_comparison(
    clustered_df: pd.DataFrame,
    comparison_json: dict[str, Any],
    boundary_geo: dict[str, Any],
) -> str:
    """Map 3: Layer switcher comparing K-Means (k=3), HDBSCAN (k=8 + noise=-1), and DEC (k=3)."""
    recs = _prepare_grid_records(
        clustered_df,
        [
            "cluster_kmeans",
            "cluster_hdbscan",
            "hdbscan_membership_prob",
            "is_hdbscan_noise",
            "cluster_dec",
            "dec_assignment_conf",
            "hotspot_density_km2",
            "fire_days",
            "active_years_count",
            "frp_total_density_mw_km2",
        ],
    )
    comp_rows = comparison_json.get("comparison_table", [])
    metrics_by_key = {}
    for r in comp_rows:
        mname = str(r.get("model", "")).lower()
        if "k-means" in mname:
            metrics_by_key["kmeans"] = r
        elif "hdbscan" in mname:
            metrics_by_key["hdbscan"] = r
        elif "dec" in mname:
            metrics_by_key["dec"] = r

    controls_html = """
    <div class="section-title">Unsupervised Model Layer Switcher</div>
    <div class="ctrl-group">
      <label><input type="radio" name="clustModel" value="kmeans" checked /> K-Means Baseline (k=3, 100% assigned)</label>
      <label><input type="radio" name="clustModel" value="hdbscan" /> HDBSCAN Density (k=8 + Noise=-1)</label>
      <label><input type="radio" name="clustModel" value="dec" /> DEC Deep Embedded (k=3, latent=4D)</label>
    </div>
    """
    stats_html = """
    <div class="section-title">Active Model Diagnostics</div>
    <div class="stat-grid" id="clustStatsGrid"></div>
    """
    legend_html = """
    <div class="section-title" id="clustLegendTitle">Cluster IDs (Arbitrary Identifiers)</div>
    <div id="clustLegendItems"></div>
    """
    notes_html = (
        "<strong>Guardrail — Cluster Labels Are Not Risk Scores:</strong> Raw cluster IDs "
        "(<code>0, 1, 2...</code> or HDBSCAN noise <code>-1</code>) are arbitrary unsupervised labels, "
        "NEVER direct risk tiers. All three models share the identical 4 locked features from "
        "<code>config/features.yaml</code> (no confidence or facility inputs)."
    )
    js = f"""
    var riauBoundary = {_json_dumps(boundary_geo)};
    var clustData = {_json_dumps(recs)};
    var modelMeta = {_json_dumps(metrics_by_key)};
    var clustLayer = L.layerGroup().addTo(map);
    addRiauBoundary(riauBoundary);

    var kmeansColors = {{ 0: '#f59e0b', 1: '#38bdf8', 2: '#ef4444' }};
    var decColors = {{ 0: '#38bdf8', 1: '#f59e0b', 2: '#ef4444' }};
    var hdbscanColors = {{
      '-1': '#475569',
      '0': '#1e3a8a', '1': '#10b981', '2': '#22d3ee', '3': '#a3e635',
      '4': '#facc15', '5': '#fb923c', '6': '#f43f5e', '7': '#c084fc'
    }};

    function countLabels(field) {{
      var counts = {{}};
      for (var i = 0; i < clustData.length; i++) {{
        var k = String(clustData[i][field]);
        counts[k] = (counts[k] || 0) + 1;
      }}
      return counts;
    }}

    function renderClustering() {{
      clustLayer.clearLayers();
      var model = document.querySelector('input[name="clustModel"]:checked').value;
      var meta = modelMeta[model] || {{}};
      var field = model === 'kmeans' ? 'cluster_kmeans' : (model === 'hdbscan' ? 'cluster_hdbscan' : 'cluster_dec');
      var palette = model === 'kmeans' ? kmeansColors : (model === 'hdbscan' ? hdbscanColors : decColors);
      var counts = countLabels(field);

      var sil = meta.silhouette_feature != null ? Number(meta.silhouette_feature).toFixed(4) : 'N/A';
      var dbi = meta.davies_bouldin_feature != null ? Number(meta.davies_bouldin_feature).toFixed(4) : 'N/A';
      var seedAri = meta.multi_seed_mean_ari != null ? Number(meta.multi_seed_mean_ari).toFixed(4) : 'N/A';
      var spAri = meta.spatial_fold_mean_ari != null ? Number(meta.spatial_fold_mean_ari).toFixed(4) : 'N/A';
      var extraLabel = model === 'hdbscan' ? 'Noise Ratio (-1)' : (model === 'dec' ? 'Latent Sil (4D)' : 'Calinski-Harabasz');
      var extraVal = model === 'hdbscan' ? ((meta.noise_ratio * 100).toFixed(2) + '%')
        : (model === 'dec' ? '0.8777' : Number(meta.calinski_harabasz_feature || 0).toFixed(1));

      document.getElementById('clustStatsGrid').innerHTML =
        '<div class="stat-card"><div class="stat-val">' + sil + '</div><div class="stat-lbl">Feature Silhouette</div></div>' +
        '<div class="stat-card"><div class="stat-val">' + dbi + '</div><div class="stat-lbl">Davies-Bouldin</div></div>' +
        '<div class="stat-card"><div class="stat-val">' + seedAri + ' / ' + spAri + '</div><div class="stat-lbl">Seed / Spatial ARI</div></div>' +
        '<div class="stat-card"><div class="stat-val">' + extraVal + '</div><div class="stat-lbl">' + extraLabel + '</div></div>';

      var keys = Object.keys(counts).sort(function(a, b) {{ return Number(a) - Number(b); }});
      var legHtml = '';
      for (var kIdx = 0; kIdx < keys.length; kIdx++) {{
        var key = keys[kIdx];
        var col = palette[key] || '#94a3b8';
        var labelTxt = (model === 'hdbscan' && key === '-1')
          ? 'Noise / Outlier (-1): ' + counts[key] + ' cells (22.4%)'
          : 'Cluster ID ' + key + ': ' + counts[key] + ' cells';
        var dashCls = (model === 'hdbscan' && key === '-1') ? ' swatch-dashed' : '';
        legHtml += '<div class="legend-item"><span class="swatch' + dashCls + '" style="background:' + col + ';"></span> ' + labelTxt + '</div>';
      }}
      document.getElementById('clustLegendItems').innerHTML = legHtml;

      for (var i = 0; i < clustData.length; i++) {{
        var g = clustData[i];
        var lbl = String(g[field]);
        var isNoise = (model === 'hdbscan' && g.cluster_hdbscan === -1);
        var col = palette[lbl] || '#94a3b8';
        L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
          color: isNoise ? '#cbd5e1' : '#0f172a',
          weight: isNoise ? 1.0 : 0.55,
          dashArray: isNoise ? '3,3' : null,
          fillColor: col,
          fillOpacity: isNoise ? 0.45 : 0.72
        }}).bindPopup(
          '<strong>' + g.grid_id + '</strong> (Block: ' + g.spatial_block + ')<br>' +
          'K-Means ID: <b>' + g.cluster_kmeans + '</b><br>' +
          'HDBSCAN ID: <b>' + g.cluster_hdbscan + '</b> (prob=' + Number(g.hdbscan_membership_prob).toFixed(3) + ')<br>' +
          'DEC ID: <b>' + g.cluster_dec + '</b> (conf=' + Number(g.dec_assignment_conf).toFixed(3) + ')<br>' +
          '<hr style="border-color:#334155;margin:5px 0;">' +
          'Hotspot Density: ' + g.hotspot_density_km2.toFixed(4) + ' det/km&sup2;<br>' +
          'Fire Days: ' + g.fire_days + ' days | Active Years: ' + g.active_years_count + ' yrs<br>' +
          'FRP Density: ' + g.frp_total_density_mw_km2.toFixed(4) + ' MW/km&sup2;'
        ).addTo(clustLayer);
      }}
    }}
    document.querySelectorAll('input[name="clustModel"]').forEach(function(el) {{ el.addEventListener('change', renderClustering); }});
    renderClustering();
    """
    return _build_html_page(
        title="FIREGUARD ID — 03 Unsupervised Clustering Comparison",
        subtitle="Interactive layer switcher: K-Means (k=3) vs HDBSCAN (k=8 + noise=-1) vs DEC (k=3)",
        category_badge="DERIVED INDICATORS",
        badge_bg="rgba(14, 165, 233, 0.22)",
        badge_fg="#7dd3fc",
        banner_html='<div class="top-banner banner-derived">DERIVED INDICATORS — Unsupervised Clustering on 4 Locked Features. Cluster IDs are NOT direct risk scores.</div>',
        legend_html=legend_html,
        controls_html=controls_html,
        stats_html=stats_html,
        notes_html=notes_html,
        extra_body_html="",
        extra_css="",
        map_script_js=js,
        period_text="2021-01-01 to 2025-12-31 (853 grid cells)",
        attribution_text="NASA FIRMS VIIRS N20 &amp; GADM v4.1 (IDN.24_1)",
    )


def _build_map_04_vulnerability_assessment(
    vuln_df: pd.DataFrame,
    vuln_summary: dict[str, Any],
    boundary_geo: dict[str, Any],
) -> str:
    """Map 4: Multi-attribute cluster profiling & vulnerability classes with QA caveats."""
    recs = _prepare_grid_records(
        vuln_df,
        [
            "vulnerability_class",
            "vulnerability_score",
            "cluster_profile_tier",
            "dec_profile_tier",
            "kmeans_dec_tier_agree",
            "is_hdbscan_noise",
            "high_conf_share",
            "dry_season_share",
            "night_detection_share",
            "hotspot_density_km2",
            "fire_days",
            "active_years_count",
            "qa_interpretation_caveat",
        ],
    )
    counts = vuln_summary.get("vulnerability_class_counts", {})
    agree_pct = float(vuln_summary.get("kmeans_dec_tier_agreement_pct", 87.34))
    n_noise = int(vuln_df["is_hdbscan_noise"].sum())
    n_disagree = int((~vuln_df["kmeans_dec_tier_agree"]).sum())

    stats_html = f"""
    <div class="section-title">Vulnerability Profile Summary</div>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-val">{agree_pct:.2f}%</div><div class="stat-lbl">K-Means / DEC Tier Agreement</div></div>
      <div class="stat-card"><div class="stat-val">{counts.get('High / Chronic-Intense', 78)}</div><div class="stat-lbl">High / Chronic-Intense Cells</div></div>
      <div class="stat-card"><div class="stat-val">{counts.get('No Detections Observed (QA Caveat)', 249)}</div><div class="stat-lbl">Zero-Detection QA Caveat</div></div>
      <div class="stat-card"><div class="stat-val">{n_noise}</div><div class="stat-lbl">HDBSCAN Noise/Borderline</div></div>
    </div>
    """
    controls_html = f"""
    <div class="section-title">QA Caveat Overlays</div>
    <div class="ctrl-group">
      <label><input type="checkbox" id="showNoiseQa" checked /> Outline HDBSCAN Noise/Borderline ({n_noise} cells)</label>
      <label><input type="checkbox" id="showDisagreeQa" /> Outline K-Means / DEC Tier Disagreement ({n_disagree} cells)</label>
    </div>
    """
    legend_html = f"""
    <div class="section-title">Vulnerability Class (Derived Profile)</div>
    <div class="legend-item"><span class="swatch" style="background:#dc2626;"></span> High / Chronic-Intense ({counts.get('High / Chronic-Intense', 78)})</div>
    <div class="legend-item"><span class="swatch" style="background:#f97316;"></span> Moderate / Recurrent ({counts.get('Moderate / Recurrent', 329)})</div>
    <div class="legend-item"><span class="swatch" style="background:#eab308;"></span> Low / Sporadic Observed ({counts.get('Low / Sporadic Observed', 197)})</div>
    <div class="legend-item"><span class="swatch swatch-dashed" style="background:#334155;"></span> No Detections Observed (QA Caveat) ({counts.get('No Detections Observed (QA Caveat)', 249)})</div>
    <div class="legend-item"><span class="swatch" style="background:transparent;border:2px solid #a855f7;"></span> HDBSCAN Density Anomaly / Borderline</div>
    """
    notes_html = (
        "<strong>QA Caveats:</strong> Derived via multi-attribute cluster profiling + normalized "
        "0–1 vulnerability index (not raw cluster labels). "
        "<em>No Detections Observed</em> does not prove zero fire risk. "
        "High-confidence share is reported strictly as QA metadata."
    )
    js = f"""
    var riauBoundary = {_json_dumps(boundary_geo)};
    var vulnData = {_json_dumps(recs)};
    var vulnLayer = L.layerGroup().addTo(map);
    addRiauBoundary(riauBoundary);

    var vulnColors = {{
      'High / Chronic-Intense': '#dc2626',
      'Moderate / Recurrent': '#f97316',
      'Low / Sporadic Observed': '#eab308',
      'No Detections Observed (QA Caveat)': '#334155'
    }};

    function renderVuln() {{
      vulnLayer.clearLayers();
      var showNoise = document.getElementById('showNoiseQa').checked;
      var showDisagree = document.getElementById('showDisagreeQa').checked;
      for (var i = 0; i < vulnData.length; i++) {{
        var g = vulnData[i];
        var cls = g.vulnerability_class;
        var isZero = cls.indexOf('No Detections') === 0;
        var col = vulnColors[cls] || '#64748b';
        var borderCol = '#0f172a';
        var weight = 0.55;
        var dash = isZero ? '3,3' : null;
        if (showDisagree && !g.kmeans_dec_tier_agree) {{
          borderCol = '#22d3ee'; weight = 1.6;
        }} else if (showNoise && g.is_hdbscan_noise) {{
          borderCol = '#c084fc'; weight = 1.4;
        }} else if (isZero) {{
          borderCol = '#94a3b8'; weight = 0.9;
        }}
        var hc = g.high_conf_share != null ? (g.high_conf_share * 100).toFixed(1) + '%' : 'N/A (0 det)';
        var dry = g.dry_season_share != null ? (g.dry_season_share * 100).toFixed(1) + '%' : 'N/A';
        var night = g.night_detection_share != null ? (g.night_detection_share * 100).toFixed(1) + '%' : 'N/A';
        L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
          color: borderCol, weight: weight, dashArray: dash,
          fillColor: col, fillOpacity: isZero ? 0.45 : 0.72
        }}).bindPopup(
          '<strong>' + g.grid_id + '</strong> (' + g.spatial_block + ')<br>' +
          'Class: <b>' + cls + '</b><br>' +
          'Vulnerability Score (0–1): <b>' + g.vulnerability_score.toFixed(4) + '</b><br>' +
          'K-Means Tier: ' + g.cluster_profile_tier + ' | DEC Tier: ' + g.dec_profile_tier + '<br>' +
          'K-Means/DEC Agree: <b>' + g.kmeans_dec_tier_agree + '</b> | HDBSCAN Noise: <b>' + g.is_hdbscan_noise + '</b><br>' +
          'QA Metadata — High Conf: ' + hc + ' | Dry Season: ' + dry + ' | Night: ' + night + '<br>' +
          '<div style="margin-top:5px;color:#fde68a;font-size:11px;"><strong>QA Caveat:</strong> ' + g.qa_interpretation_caveat + '</div>'
        ).addTo(vulnLayer);
      }}
    }}
    document.getElementById('showNoiseQa').addEventListener('change', renderVuln);
    document.getElementById('showDisagreeQa').addEventListener('change', renderVuln);
    renderVuln();
    """
    return _build_html_page(
        title="FIREGUARD ID — 04 Vulnerability Assessment &amp; QA Caveats",
        subtitle="Multi-attribute cluster profiling (87.34% K-Means/DEC concordance)",
        category_badge="DERIVED INDICATORS",
        badge_bg="rgba(14, 165, 233, 0.22)",
        badge_fg="#7dd3fc",
        banner_html='<div class="top-banner banner-derived">DERIVED INDICATORS — Vulnerability Assessment from Multi-Attribute Profiling (with Zero-Detection &amp; HDBSCAN QA Caveats).</div>',
        legend_html=legend_html,
        controls_html=controls_html,
        stats_html=stats_html,
        notes_html=notes_html,
        extra_body_html="",
        extra_css="",
        map_script_js=js,
        period_text="2021-01-01 to 2025-12-31 (853 grid cells)",
        attribution_text="NASA FIRMS VIIRS N20 &amp; GADM v4.1 (IDN.24_1)",
    )


def _build_map_05_spatial_prioritization(
    prio_df: pd.DataFrame,
    prio_summary: dict[str, Any],
    boundary_geo: dict[str, Any],
) -> str:
    """Map 5: Spatial prioritization Tiers 1-4, rank, and 4 normalized composite factors."""
    recs = _prepare_grid_records(
        prio_df,
        [
            "priority_class",
            "priority_rank",
            "priority_score",
            "factor_vulnerability",
            "factor_persistence",
            "factor_spatial_corridor",
            "factor_remoteness_proxy",
            "dist_to_pekanbaru_km",
            "vulnerability_class",
        ],
    )
    counts = prio_summary.get("priority_class_counts", {})
    w = prio_summary.get("baseline_weights", {"vuln": 0.45, "persistence": 0.25, "corridor": 0.20, "remoteness": 0.10})
    rho = float(prio_summary.get("vulnerability_vs_priority_spearman", 0.9738))

    stats_html = f"""
    <div class="section-title">Composite Priority &amp; Sensitivity</div>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-val">{counts.get('Tier 1 (Critical Priority - Top 10%)', 85)}</div><div class="stat-lbl">Tier 1 Critical (Top 10%)</div></div>
      <div class="stat-card"><div class="stat-val">{prio_df['priority_score'].sum():.2f}</div><div class="stat-lbl">Total Priority Demand</div></div>
      <div class="stat-card"><div class="stat-val">{rho:.4f}</div><div class="stat-lbl">Vuln vs Priority Spearman</div></div>
      <div class="stat-card"><div class="stat-val">0.717–0.809</div><div class="stat-lbl">Top-85 Weight Jaccard</div></div>
    </div>
    """
    controls_html = """
    <div class="section-title">Tier Filter</div>
    <div class="ctrl-group">
      <label><input type="checkbox" class="tierFilter" value="Tier 1" checked /> Tier 1 (Critical - Top 10%, n=85)</label>
      <label><input type="checkbox" class="tierFilter" value="Tier 2" checked /> Tier 2 (High - Top 10–25%, n=128)</label>
      <label><input type="checkbox" class="tierFilter" value="Tier 3" checked /> Tier 3 (Elevated - Top 25–50%, n=213)</label>
      <label><input type="checkbox" class="tierFilter" value="Tier 4" checked /> Tier 4 (Baseline - Bottom 50%, n=427)</label>
    </div>
    """
    legend_html = f"""
    <div class="section-title">Priority Tiers &amp; Factor Weights</div>
    <div class="legend-item"><span class="swatch" style="background:#dc2626;"></span> Tier 1 Critical (Ranks 1–85)</div>
    <div class="legend-item"><span class="swatch" style="background:#f97316;"></span> Tier 2 High (Ranks 86–213)</div>
    <div class="legend-item"><span class="swatch" style="background:#eab308;"></span> Tier 3 Elevated (Ranks 214–426)</div>
    <div class="legend-item"><span class="swatch" style="background:#1e3a5f;"></span> Tier 4 Baseline (Ranks 427–853)</div>
    <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> Pekanbaru Regional Reference Hub</div>
    <div class="meta-row" style="margin-top:5px;color:#cbd5e1;">
      Weights: Vulnerability ({w.get('vuln')}), Persistence ({w.get('persistence')}),
      Spatial Corridor ({w.get('corridor')}), Remoteness Proxy ({w.get('remoteness')}).
    </div>
    """
    notes_html = (
        "<strong>Units &amp; Interpretation:</strong> <code>priority_score</code> and all 4 factors are "
        "normalized 0–1 indices. <code>dist_to_pekanbaru_km</code> is WGS84 geodesic distance (km) used "
        "as a remoteness proxy (not road travel time)."
    )
    js = f"""
    var riauBoundary = {_json_dumps(boundary_geo)};
    var prioData = {_json_dumps(recs)};
    var prioLayer = L.layerGroup().addTo(map);
    addRiauBoundary(riauBoundary);
    L.circleMarker([{PEKANBARU_COORDS[0]}, {PEKANBARU_COORDS[1]}], {{
      radius: 7, color: '#ffffff', weight: 2, fillColor: '#38bdf8', fillOpacity: 0.95
    }}).bindPopup('<strong>Pekanbaru Regional Reference Hub</strong><br>0.5071, 101.4478 (Remoteness anchor)').addTo(map);

    function tierColor(cls) {{
      if (cls.indexOf('Tier 1') === 0) return '#dc2626';
      if (cls.indexOf('Tier 2') === 0) return '#f97316';
      if (cls.indexOf('Tier 3') === 0) return '#eab308';
      return '#1e3a5f';
    }}
    function renderPrio() {{
      prioLayer.clearLayers();
      var active = {{}};
      document.querySelectorAll('.tierFilter:checked').forEach(function(el) {{ active[el.value] = true; }});
      for (var i = 0; i < prioData.length; i++) {{
        var g = prioData[i];
        var prefix = g.priority_class.slice(0, 6);
        if (!active[prefix]) continue;
        var isT1 = prefix === 'Tier 1';
        L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
          color: isT1 ? '#fecaca' : '#0f172a',
          weight: isT1 ? 1.0 : 0.5,
          fillColor: tierColor(g.priority_class),
          fillOpacity: prefix === 'Tier 4' ? 0.50 : 0.78
        }}).bindPopup(
          '<strong>' + g.grid_id + '</strong> — Rank <b>#' + g.priority_rank + ' / 853</b><br>' +
          'Class: <b>' + g.priority_class + '</b><br>' +
          'Composite Priority Score (0–1): <b>' + g.priority_score.toFixed(4) + '</b><br>' +
          '<hr style="border-color:#334155;margin:5px 0;">' +
          'Factor 1 — Vulnerability (w=0.45): <b>' + g.factor_vulnerability.toFixed(4) + '</b><br>' +
          'Factor 2 — Persistence (w=0.25): <b>' + g.factor_persistence.toFixed(4) + '</b><br>' +
          'Factor 3 — Spatial Corridor (w=0.20): <b>' + g.factor_spatial_corridor.toFixed(4) + '</b><br>' +
          'Factor 4 — Remoteness Proxy (w=0.10): <b>' + g.factor_remoteness_proxy.toFixed(4) + '</b><br>' +
          'Geodesic Dist to Pekanbaru: ' + g.dist_to_pekanbaru_km.toFixed(2) + ' km'
        ).addTo(prioLayer);
      }}
    }}
    document.querySelectorAll('.tierFilter').forEach(function(el) {{ el.addEventListener('change', renderPrio); }});
    renderPrio();
    """
    return _build_html_page(
        title="FIREGUARD ID — 05 Spatial Prioritization (Tiers 1–4)",
        subtitle="Ranked intervention priority combining vulnerability, persistence, corridor &amp; remoteness",
        category_badge="DERIVED INDICATORS",
        badge_bg="rgba(14, 165, 233, 0.22)",
        badge_fg="#7dd3fc",
        banner_html='<div class="top-banner banner-derived">DERIVED INDICATORS — 4-Factor Composite Spatial Prioritization (85 Tier-1 Critical Cells).</div>',
        legend_html=legend_html,
        controls_html=controls_html,
        stats_html=stats_html,
        notes_html=notes_html,
        extra_body_html="",
        extra_css="",
        map_script_js=js,
        period_text="2021-01-01 to 2025-12-31 (853 grid cells)",
        attribution_text="NASA FIRMS VIIRS N20 &amp; GADM v4.1 (IDN.24_1)",
    )


def _build_map_06_infrastructure_gaps(
    prio_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    bases_df: pd.DataFrame,
    opt_cfg: dict[str, Any],
    boundary_geo: dict[str, Any],
) -> str:
    """Map 6: Documents real inventory absence & Tier-1 coverage gaps beyond 60-min proxy."""
    merged = prio_df.merge(
        cov_df[
            [
                "grid_id",
                "assigned_base_id",
                "nearest_geodesic_km",
                "nearest_effective_road_km",
                "nearest_travel_time_min",
                "is_covered_radius",
                "is_covered_time_limit",
            ]
        ],
        on="grid_id",
        how="left",
    )
    recs = _prepare_grid_records(
        merged,
        [
            "priority_class",
            "priority_rank",
            "priority_score",
            "assigned_base_id",
            "nearest_geodesic_km",
            "nearest_effective_road_km",
            "nearest_travel_time_min",
            "is_covered_radius",
            "is_covered_time_limit",
        ],
    )
    t1_mask = merged["priority_class"].astype(str).str.startswith("Tier 1")
    t1_total = int(t1_mask.sum())
    t1_gap_60 = int((t1_mask & (~merged["is_covered_time_limit"])).sum())
    t1_cov_60 = int((t1_mask & merged["is_covered_time_limit"]).sum())
    all_gap_60 = int((~merged["is_covered_time_limit"]).sum())

    dist_cfg = opt_cfg["distance_model"]
    detour = float(dist_cfg["road_detour_factor"])
    speed = float(dist_cfg["avg_response_speed_kmh"])
    max_min = float(dist_cfg["max_response_time_min"])
    eff_60_km = speed * (max_min / 60.0)
    geo_60_m = (eff_60_km / detour) * 1000.0

    bases_pts = [
        {
            "base_id": str(r["base_id"]),
            "grid_id": str(r["grid_id"]),
            "lat": round(float(r["land_rep_lat"]), 5),
            "lon": round(float(r["land_rep_lon"]), 5),
            "priority_rank": int(r["priority_rank"]),
        }
        for _, r in bases_df.iterrows()
    ]

    stats_html = f"""
    <div class="section-title">60-Min Proxy Gap Diagnostics</div>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-val" style="color:#fb7185;">{t1_gap_60} / {t1_total}</div><div class="stat-lbl">Tier-1 Gaps (&gt;60 min proxy)</div></div>
      <div class="stat-card"><div class="stat-val" style="color:#34d399;">{t1_cov_60} / {t1_total}</div><div class="stat-lbl">Tier-1 Covered (&le;60 min)</div></div>
      <div class="stat-card"><div class="stat-val">{all_gap_60} / {len(recs)}</div><div class="stat-lbl">All Cells &gt;60 min Proxy</div></div>
      <div class="stat-card"><div class="stat-val">{merged['nearest_travel_time_min'].mean():.1f} min</div><div class="stat-lbl">Mean Proxy Travel Time</div></div>
    </div>
    """
    controls_html = """
    <div class="section-title">Gap Diagnostic Layers</div>
    <div class="ctrl-group">
      <label><input type="checkbox" id="showAllTravel" checked /> All Cells Travel Time Gradient (min)</label>
      <label><input type="checkbox" id="showTier1Gaps" checked /> Highlight Tier-1 Gaps (&gt;60 min, n=41)</label>
      <label><input type="checkbox" id="showTier1Cov" checked /> Highlight Tier-1 Covered (&le;60 min, n=44)</label>
      <label><input type="checkbox" id="show60Circles" checked /> Synthetic 60-Min Proxy Circles (~33.3 km geodesic)</label>
    </div>
    """
    legend_html = """
    <div class="section-title">Response Time &amp; Tier-1 Gap Legend</div>
    <div class="legend-item"><span class="swatch" style="background:#065f46;"></span> &le;30 min effective road proxy</div>
    <div class="legend-item"><span class="swatch" style="background:#0ea5e9;"></span> 30–60 min effective road proxy</div>
    <div class="legend-item"><span class="swatch" style="background:#f59e0b;"></span> 60–90 min (beyond 60-min threshold)</div>
    <div class="legend-item"><span class="swatch" style="background:#7f1d1d;"></span> &gt;90 min (severe access gap)</div>
    <div class="legend-item"><span class="swatch" style="background:transparent;border:2px solid #f43f5e;"></span> Tier-1 Critical Cell &gt;60 min Gap</div>
    <div class="legend-item"><span class="swatch" style="background:transparent;border:2px solid #10b981;"></span> Tier-1 Critical Cell &le;60 min Covered</div>
    <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> Synthetic Reference Base (M1, n=10)</div>
    """
    notes_html = (
        "<strong>REAL INVENTORY ABSENCE DOCUMENTED:</strong> Verified local BPBD / Manggala Agni "
        "station coordinates, fleet inventories, and OSM road speeds are <strong>NOT</strong> present "
        "in <code>data/external/</code>. Gaps are therefore measured against the synthetic 10-base MCLP "
        f"reference network using Haversine &times; {detour} detour @ {speed:.0f} km/h "
        f"(60 min = {eff_60_km:.1f} km effective road / {geo_60_m/1000:.2f} km geodesic)."
    )
    js = f"""
    var riauBoundary = {_json_dumps(boundary_geo)};
    var gapData = {_json_dumps(recs)};
    var refBases = {_json_dumps(bases_pts)};
    var geo60m = {geo_60_m:.2f};
    addRiauBoundary(riauBoundary);

    var bgLayer = L.layerGroup().addTo(map);
    var t1GapLayer = L.layerGroup().addTo(map);
    var t1CovLayer = L.layerGroup().addTo(map);
    var circleLayer = L.layerGroup().addTo(map);
    var baseLayer = L.layerGroup().addTo(map);

    function travelColor(m) {{
      if (m <= 30) return '#065f46';
      if (m <= 60) return '#0ea5e9';
      if (m <= 90) return '#f59e0b';
      return '#7f1d1d';
    }}

    for (var i = 0; i < gapData.length; i++) {{
      var g = gapData[i];
      var isT1 = g.priority_class.indexOf('Tier 1') === 0;
      var popup = '<strong>' + g.grid_id + '</strong> (Rank #' + g.priority_rank + ')<br>' +
        'Priority Class: <b>' + g.priority_class + '</b><br>' +
        'Nearest Synthetic Base: <b>' + g.assigned_base_id + '</b><br>' +
        'Geodesic Dist: ' + g.nearest_geodesic_km.toFixed(2) + ' km<br>' +
        'Effective Road Proxy (1.35x): ' + g.nearest_effective_road_km.toFixed(2) + ' km<br>' +
        'Travel Time Proxy (@45 km/h): <b>' + g.nearest_travel_time_min.toFixed(1) + ' min</b><br>' +
        '60-Min Status: <b>' + (g.is_covered_time_limit ? 'COVERED (<=60 min)' : 'GAP (>60 min proxy)') + '</b>';

      L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
        color: '#0f172a', weight: 0.45,
        fillColor: travelColor(g.nearest_travel_time_min), fillOpacity: 0.56
      }}).bindPopup(popup).addTo(bgLayer);

      if (isT1 && !g.is_covered_time_limit) {{
        L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
          color: '#f43f5e', weight: 2.2, fillColor: '#f43f5e', fillOpacity: 0.18
        }}).bindPopup(popup).addTo(t1GapLayer);
      }} else if (isT1 && g.is_covered_time_limit) {{
        L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
          color: '#10b981', weight: 2.0, fillColor: '#10b981', fillOpacity: 0.12
        }}).bindPopup(popup).addTo(t1CovLayer);
      }}
    }}

    for (var b = 0; b < refBases.length; b++) {{
      var bs = refBases[b];
      L.circle([bs.lat, bs.lon], {{
        radius: geo60m, color: '#38bdf8', weight: 1.2, dashArray: '5,4',
        fillColor: '#38bdf8', fillOpacity: 0.05
      }}).addTo(circleLayer);
      L.circleMarker([bs.lat, bs.lon], {{
        radius: 6, color: '#ffffff', weight: 1.8, fillColor: '#38bdf8', fillOpacity: 0.95
      }}).bindPopup('<strong>Synthetic Reference ' + bs.base_id + '</strong> (' + bs.grid_id + ', Rank #' + bs.priority_rank + ')<br><em>Synthetic planning proxy (real inventory absent)</em>').addTo(baseLayer);
    }}

    L.circleMarker([{PEKANBARU_COORDS[0]}, {PEKANBARU_COORDS[1]}], {{
      radius: 5, color: '#fde047', weight: 2, fillColor: '#0f172a', fillOpacity: 1
    }}).bindPopup('<strong>Pekanbaru Regional Hub Reference</strong>').addTo(baseLayer);

    function syncLayers() {{
      document.getElementById('showAllTravel').checked ? map.addLayer(bgLayer) : map.removeLayer(bgLayer);
      document.getElementById('showTier1Gaps').checked ? map.addLayer(t1GapLayer) : map.removeLayer(t1GapLayer);
      document.getElementById('showTier1Cov').checked ? map.addLayer(t1CovLayer) : map.removeLayer(t1CovLayer);
      document.getElementById('show60Circles').checked ? map.addLayer(circleLayer) : map.removeLayer(circleLayer);
    }}
    ['showAllTravel', 'showTier1Gaps', 'showTier1Cov', 'show60Circles'].forEach(function(id) {{
      document.getElementById(id).addEventListener('change', syncLayers);
    }});
    """
    return _build_html_page(
        title="FIREGUARD ID — 06 Infrastructure Coverage Gaps",
        subtitle="Real inventory absence &amp; Tier-1 critical gaps beyond 60-min proxy",
        category_badge="GAP ANALYSIS (REAL INVENTORY ABSENT)",
        badge_bg="rgba(245, 158, 11, 0.25)",
        badge_fg="#fde68a",
        banner_html='<div class="top-banner banner-gap">GAP DIAGNOSTIC — Real BPBD/Manggala Agni Inventory Absent Locally; Uses Synthetic 60-Min Road Proxy (41/85 Tier-1 Cells Uncovered)</div>',
        legend_html=legend_html,
        controls_html=controls_html,
        stats_html=stats_html,
        notes_html=notes_html,
        extra_body_html="",
        extra_css="",
        map_script_js=js,
        period_text="2021-01-01 to 2025-12-31 (853 grid cells)",
        attribution_text="NASA FIRMS VIIRS N20 &amp; GADM v4.1 (IDN.24_1)",
    )


def _build_map_07_optimized_infrastructure(
    prio_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    bases_m1: pd.DataFrame,
    cap_m2: pd.DataFrame,
    fleet_m3: pd.DataFrame,
    pers_m4: pd.DataFrame,
    relays_m5: pd.DataFrame,
    base_comms_m5: pd.DataFrame,
    opt_cfg: dict[str, Any],
    m1_summary: dict[str, Any],
    m2_summary: dict[str, Any],
    m3_summary: dict[str, Any],
    m4_summary: dict[str, Any],
    m5_summary: dict[str, Any],
    boundary_geo: dict[str, Any],
) -> str:
    """Map 7: Integrated synthetic scenario M1 bases, M2 tiers, M3 fleet, M4 crew, M5 relays & links."""
    grid_merged = prio_df.merge(
        cov_df[["grid_id", "assigned_base_id", "nearest_effective_road_km", "nearest_travel_time_min", "is_covered_radius"]],
        on="grid_id",
        how="left",
    )
    grid_recs = _prepare_grid_records(
        grid_merged,
        ["priority_class", "priority_rank", "priority_score", "assigned_base_id", "is_covered_radius", "nearest_travel_time_min"],
    )

    bases_joined = (
        bases_m1[["base_id", "grid_id", "land_rep_lat", "land_rep_lon", "priority_rank", "assigned_priority_demand", "assigned_cell_count", "covered_cell_count_within_radius"]]
        .merge(cap_m2[["base_id", "selected_tier", "tier_max_demand", "capex_million_idr", "storage_units", "utilization_pct"]], on="base_id", how="left")
        .merge(fleet_m3[["base_id", "trucks", "tankers", "pumps", "deployed_capacity", "demand_coverage_pct"]], on="base_id", how="left")
        .merge(pers_m4[["base_id", "total_crew_assigned", "truck_crew_total", "tanker_crew_total", "pump_specialists_total", "staffed_coverage_pct"]], on="base_id", how="left")
        .merge(base_comms_m5[["base_id", "is_comms_covered", "nearest_relay_haversine_km"]], on="base_id", how="left")
    )
    bases_recs = _clean_for_json(bases_joined.to_dict(orient="records"))

    relays_recs = [
        {
            "relay_cand_id": str(r["relay_cand_id"]),
            "grid_id": str(r["grid_id"]),
            "lat": round(float(r["land_rep_lat"]), 5),
            "lon": round(float(r["land_rep_lon"]), 5),
            "is_primary_hub": bool(r["is_primary_hub"]),
            "dist_to_pekanbaru_km": round(float(r["dist_to_pekanbaru_km"]), 2),
            "relay_coverage_radius_km": float(r["relay_coverage_radius_km"]),
            "backbone_link_radius_km": float(r["backbone_link_radius_km"]),
        }
        for _, r in relays_m5.iterrows()
    ]

    # Compute M5 inter-relay backbone links (<= backbone_link_radius_km) and base-to-nearest-relay links
    backbone_links: list[dict[str, Any]] = []
    for i in range(len(relays_recs)):
        for j in range(i + 1, len(relays_recs)):
            d = float(haversine_km(relays_recs[i]["lat"], relays_recs[i]["lon"], relays_recs[j]["lat"], relays_recs[j]["lon"]))
            if d <= relays_recs[i]["backbone_link_radius_km"]:
                backbone_links.append({
                    "from": relays_recs[i]["relay_cand_id"],
                    "to": relays_recs[j]["relay_cand_id"],
                    "lat1": relays_recs[i]["lat"], "lon1": relays_recs[i]["lon"],
                    "lat2": relays_recs[j]["lat"], "lon2": relays_recs[j]["lon"],
                    "dist_km": round(d, 2),
                })

    base_relay_links: list[dict[str, Any]] = []
    for b in bases_recs:
        best_r = None
        best_d = 1e9
        for r in relays_recs:
            d = float(haversine_km(float(b["land_rep_lat"]), float(b["land_rep_lon"]), r["lat"], r["lon"]))
            if d < best_d:
                best_d = d
                best_r = r
        if best_r is not None:
            base_relay_links.append({
                "base_id": b["base_id"],
                "relay_id": best_r["relay_cand_id"],
                "lat1": round(float(b["land_rep_lat"]), 5), "lon1": round(float(b["land_rep_lon"]), 5),
                "lat2": best_r["lat"], "lon2": best_r["lon"],
                "dist_km": round(best_d, 2),
            })

    dist_cfg = opt_cfg["distance_model"]
    geo_35_m = (float(dist_cfg["coverage_radius_km"]) / float(dist_cfg["road_detour_factor"])) * 1000.0
    relay_rad_m = float(opt_cfg["module5_comms"]["relay_coverage_radius_km"]) * 1000.0

    m1_def = m1_summary.get("default_solution", {})
    m2_def = m2_summary.get("default_solution", {})
    m3_def = m3_summary.get("nominal_summary", {})
    m4_def = m4_summary.get("nominal_summary", {})
    m5_def = m5_summary.get("default_solution", {}).get("metrics", {})

    stats_html = f"""
    <div class="section-title">Synthetic Scenario Summary (M1–M5)</div>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-val">{m1_def.get('n_bases', 10)} Bases</div><div class="stat-lbl">M1: {m1_def.get('covered_priority_pct', 33.59)}% @ 35km</div></div>
      <div class="stat-card"><div class="stat-val">{m2_def.get('total_capex_million_idr', 31800)/1000:.1f}B IDR</div><div class="stat-lbl">M2: 1S / 6M / 3L Tiers</div></div>
      <div class="stat-card"><div class="stat-val">{m3_def.get('trucks_assigned',24)}T/{m3_def.get('tankers_assigned',18)}Tk/{m3_def.get('pumps_assigned',60)}P</div><div class="stat-lbl">M3: 100% Demand Fleet</div></div>
      <div class="stat-card"><div class="stat-val">{m4_def.get('total_personnel_assigned',298)} Crew | {m5_def.get('n_relays',8)} Relays</div><div class="stat-lbl">M4 0 Viol. | M5 {m5_def.get('priority_coverage_pct',64.77)}%</div></div>
    </div>
    """
    controls_html = """
    <div class="section-title">Integrated Module Layers</div>
    <div class="ctrl-group">
      <label><input type="checkbox" id="lyrBases" checked /> M1–M4 Emergency Bases (size/color = M2 tier)</label>
      <label><input type="checkbox" id="lyrBaseCov" checked /> M1 Rapid Coverage Circles (35 km road &asymp; 25.9 km geo)</label>
      <label><input type="checkbox" id="lyrRelays" checked /> M5 Comms Relays &amp; Hub (n=8)</label>
      <label><input type="checkbox" id="lyrRelayCov" /> M5 Relay 55 km Isotropic Radius Proxy</label>
      <label><input type="checkbox" id="lyrLinks" checked /> M5 Backbone (110 km) &amp; Base-Relay Links</label>
      <label><input type="checkbox" id="lyrSpider" /> M1 Cell-to-Base Assignment Spider Lines</label>
    </div>
    """
    legend_html = """
    <div class="section-title">M2 Base Tiers &amp; M5 Comms Legend</div>
    <div class="legend-item"><span class="dot" style="background:#ef4444;width:14px;height:14px;"></span> Large Tier Base (cap=160, 4,500M IDR, n=3)</div>
    <div class="legend-item"><span class="dot" style="background:#f59e0b;width:12px;height:12px;"></span> Medium Tier Base (cap=90, 2,800M IDR, n=6)</div>
    <div class="legend-item"><span class="dot" style="background:#22c55e;width:9px;height:9px;"></span> Small Tier Base (cap=40, 1,500M IDR, n=1)</div>
    <div class="legend-item"><span class="dot" style="background:#a855f7;"></span> M5 Primary Comms Hub (RELAY_CAND_26)</div>
    <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> M5 Relay Repeater (n=7)</div>
    <div class="legend-item"><span class="swatch" style="background:#22d3ee;height:3px;"></span> M5 Backbone Backhaul Link (&le;110 km)</div>
    """
    notes_html = (
        "<strong>SYNTHETIC SCENARIO DISCLAIMER:</strong> <code>is_synthetic_scenario=True</code>. "
        "No local BPBD/Manggala Agni inventory exists. M1 travel uses Haversine &times; 1.35 detour @ 45 km/h; "
        "M5 comms uses 55 km isotropic haversine LOS proxy &amp; 110 km directional backhaul (no DEM/canopy path loss)."
    )
    js = f"""
    var riauBoundary = {_json_dumps(boundary_geo)};
    var gridData = {_json_dumps(grid_recs)};
    var basesData = {_json_dumps(bases_recs)};
    var relaysData = {_json_dumps(relays_recs)};
    var backboneLinks = {_json_dumps(backbone_links)};
    var baseRelayLinks = {_json_dumps(base_relay_links)};
    var geo35m = {geo_35_m:.2f};
    var relayRadm = {relay_rad_m:.2f};

    addRiauBoundary(riauBoundary);
    var gridBgLayer = L.layerGroup().addTo(map);
    var spiderLayer = L.layerGroup();
    var baseCovLayer = L.layerGroup().addTo(map);
    var relayCovLayer = L.layerGroup();
    var linksLayer = L.layerGroup().addTo(map);
    var relaysLayer = L.layerGroup().addTo(map);
    var basesLayer = L.layerGroup().addTo(map);

    var baseCoordMap = {{}};
    basesData.forEach(function(b) {{ baseCoordMap[b.base_id] = [b.land_rep_lat, b.land_rep_lon]; }});

    function tierBg(cls) {{
      if (cls.indexOf('Tier 1') === 0) return '#b91c1c';
      if (cls.indexOf('Tier 2') === 0) return '#c2410c';
      if (cls.indexOf('Tier 3') === 0) return '#a16207';
      return '#1e293b';
    }}
    for (var i = 0; i < gridData.length; i++) {{
      var g = gridData[i];
      L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
        color: '#0f172a', weight: 0.4,
        fillColor: tierBg(g.priority_class), fillOpacity: g.priority_class.indexOf('Tier 4') === 0 ? 0.25 : 0.45
      }}).bindPopup(
        '<strong>' + g.grid_id + '</strong> (Rank #' + g.priority_rank + ')<br>' +
        'Assigned Base: <b>' + g.assigned_base_id + '</b><br>' +
        '35km Rapid Covered: <b>' + g.is_covered_radius + '</b> | Est Travel: ' + g.nearest_travel_time_min.toFixed(1) + ' min'
      ).addTo(gridBgLayer);

      var bc = baseCoordMap[g.assigned_base_id];
      if (bc) {{
        L.polyline([[g.land_rep_lat, g.lon_rep_lon], bc], {{
          color: g.is_covered_radius ? '#34d399' : '#64748b',
          weight: g.is_covered_radius ? 0.9 : 0.45,
          opacity: g.is_covered_radius ? 0.55 : 0.25
        }}).addTo(spiderLayer);
      }}
    }}

    for (var k = 0; k < backboneLinks.length; k++) {{
      var lk = backboneLinks[k];
      L.polyline([[lk.lat1, lk.lon1], [lk.lat2, lk.lon2]], {{
        color: '#22d3ee', weight: 2.2, opacity: 0.85
      }}).bindPopup('M5 Backbone Link: ' + lk.from + ' &harr; ' + lk.to + ' (' + lk.dist_km + ' km)').addTo(linksLayer);
    }}
    for (var m = 0; m < baseRelayLinks.length; m++) {{
      var br = baseRelayLinks[m];
      L.polyline([[br.lat1, br.lon1], [br.lat2, br.lon2]], {{
        color: '#a855f7', weight: 1.3, dashArray: '4,4', opacity: 0.75
      }}).bindPopup('Base-Relay Link: ' + br.base_id + ' &rarr; ' + br.relay_id + ' (' + br.dist_km + ' km)').addTo(linksLayer);
    }}

    for (var rIdx = 0; rIdx < relaysData.length; rIdx++) {{
      var rl = relaysData[rIdx];
      L.circle([rl.lat, rl.lon], {{
        radius: relayRadm, color: '#38bdf8', weight: 1.0, dashArray: '4,4',
        fillColor: '#38bdf8', fillOpacity: 0.06
      }}).addTo(relayCovLayer);

      L.circleMarker([rl.lat, rl.lon], {{
        radius: rl.is_primary_hub ? 8 : 6,
        color: '#ffffff', weight: 1.8,
        fillColor: rl.is_primary_hub ? '#a855f7' : '#38bdf8', fillOpacity: 0.95
      }}).bindPopup(
        '<strong>' + rl.relay_cand_id + (rl.is_primary_hub ? ' (PRIMARY HUB)' : '') + '</strong><br>' +
        'Grid: ' + rl.grid_id + ' | Dist Pekanbaru: ' + rl.dist_to_pekanbaru_km + ' km<br>' +
        'Coverage Radius Proxy: ' + rl.relay_coverage_radius_km + ' km (haversine)<br>' +
        '<em>Synthetic M5 Comms Scenario</em>'
      ).addTo(relaysLayer);
    }}

    var tierColors = {{ small: '#22c55e', medium: '#f59e0b', large: '#ef4444' }};
    var tierRadii = {{ small: 7, medium: 9.5, large: 12 }};
    for (var bIdx = 0; bIdx < basesData.length; bIdx++) {{
      var bs = basesData[bIdx];
      var col = tierColors[bs.selected_tier] || '#f59e0b';
      var rad = tierRadii[bs.selected_tier] || 9;
      L.circle([bs.land_rep_lat, bs.land_rep_lon], {{
        radius: geo35m, color: col, weight: 1.5, dashArray: '5,4',
        fillColor: col, fillOpacity: 0.08
      }}).addTo(baseCovLayer);

      L.circleMarker([bs.land_rep_lat, bs.land_rep_lon], {{
        radius: rad, color: '#ffffff', weight: 2.0, fillColor: col, fillOpacity: 0.95
      }}).bindPopup(
        '<strong>' + bs.base_id + '</strong> (' + bs.grid_id + ', Priority Rank #' + bs.priority_rank + ')<br>' +
        '<span style="color:#fca5a5;font-size:11px;">[SYNTHETIC SCENARIO]</span><br>' +
        '<b>M1 Coverage:</b> Assigned Demand=' + Number(bs.assigned_priority_demand).toFixed(2) + ' (' + bs.assigned_cell_count + ' cells; ' + bs.covered_cell_count_within_radius + ' within 35km)<br>' +
        '<b>M2 Capacity:</b> Tier=<b>' + String(bs.selected_tier).toUpperCase() + '</b> (max=' + bs.tier_max_demand + ', CAPEX=' + bs.capex_million_idr + 'M IDR, ' + bs.storage_units + ' units, util=' + bs.utilization_pct + '%)<br>' +
        '<b>M3 Fleet:</b> ' + bs.trucks + ' Fire Trucks, ' + bs.tankers + ' Tankers, ' + bs.pumps + ' Pumps (cap=' + bs.deployed_capacity + ', cov=' + bs.demand_coverage_pct + '%)<br>' +
        '<b>M4 Personnel:</b> <b>' + bs.total_crew_assigned + ' crew</b> (2 shifts: ' + bs.truck_crew_total + ' truck, ' + bs.tanker_crew_total + ' tanker, ' + bs.pump_specialists_total + ' pump spec.)<br>' +
        '<b>M5 Comms:</b> Covered=' + bs.is_comms_covered + ' (nearest relay ' + Number(bs.nearest_relay_haversine_km).toFixed(1) + ' km)'
      ).addTo(basesLayer);
    }}

    var toggles = [
      ['lyrBases', basesLayer],
      ['lyrBaseCov', baseCovLayer],
      ['lyrRelays', relaysLayer],
      ['lyrRelayCov', relayCovLayer],
      ['lyrLinks', linksLayer],
      ['lyrSpider', spiderLayer]
    ];
    toggles.forEach(function(pair) {{
      document.getElementById(pair[0]).addEventListener('change', function() {{
        this.checked ? map.addLayer(pair[1]) : map.removeLayer(pair[1]);
      }});
    }});
    """
    return _build_html_page(
        title="FIREGUARD ID — 07 Optimized Infrastructure (Synthetic M1–M5)",
        subtitle="Integrated bases, capacity tiers, fleet, 2-shift crew &amp; comms relay backbone",
        category_badge="SYNTHETIC OPTIMIZATION SCENARIO",
        badge_bg="rgba(220, 38, 38, 0.28)",
        badge_fg="#fca5a5",
        banner_html='<div class="top-banner banner-synthetic">SYNTHETIC OPERATIONAL SCENARIO (is_synthetic_scenario=True) — Real BPBD/Manggala Agni Inventory Unavailable; Not a Physical Deployment Directive</div>',
        legend_html=legend_html,
        controls_html=controls_html,
        stats_html=stats_html,
        notes_html=notes_html,
        extra_body_html="",
        extra_css="",
        map_script_js=js,
        period_text="2021-01-01 to 2025-12-31 (anchored to Riau priority grid)",
        attribution_text="NASA FIRMS VIIRS N20 &amp; GADM v4.1 (IDN.24_1)",
    )


def _build_map_08_coverage_comparison(
    prio_df: pd.DataFrame,
    bases_m1: pd.DataFrame,
    cov_m1: pd.DataFrame,
    opt_cfg: dict[str, Any],
    m1_summary: dict[str, Any],
    m6_pareto: pd.DataFrame,
    m6_bases_all: pd.DataFrame,
    boundary_geo: dict[str, Any],
) -> str:
    """Map 8: Naive vs MCLP 35km/60min spatial coverage comparison + M6 Pareto table."""
    dist_cfg = opt_cfg["distance_model"]
    m1_cfg = opt_cfg["module1_base_location"]
    detour = float(dist_cfg["road_detour_factor"])
    speed = float(dist_cfg["avg_response_speed_kmh"])
    cov_rad = float(dist_cfg["coverage_radius_km"])
    max_time = float(dist_cfg["max_response_time_min"])
    p_bases = int(m1_cfg["default_p_bases"])

    # Compute exact cell-level coverage for Naive Top-P and Spatial-Block Uniform using Module 1 evaluator
    cands = select_candidate_sites(prio_df, n_candidates=int(m1_cfg["n_candidates"]))
    top_p_df = prio_df.sort_values("priority_rank", ascending=True).head(p_bases).copy().reset_index(drop=True)
    m_top, bases_top, cov_top = evaluate_base_selection(
        prio_df, top_p_df, coverage_radius_km=cov_rad, max_response_time_min=max_time, detour_factor=detour, speed_kmh=speed
    )

    ordered_cands = cands.sort_values("priority_rank", ascending=True).reset_index(drop=True)
    blocks = sorted(ordered_cands["spatial_block"].astype(str).unique().tolist())
    block_queues = {
        b: ordered_cands.loc[ordered_cands["spatial_block"].astype(str) == b].index.tolist()
        for b in blocks
    }
    chosen_idx: list[int] = []
    while len(chosen_idx) < p_bases:
        progressed = False
        for b in blocks:
            if len(chosen_idx) >= p_bases:
                break
            if block_queues[b]:
                chosen_idx.append(block_queues[b].pop(0))
                progressed = True
        if not progressed:
            break
    unif_df = ordered_cands.loc[chosen_idx].copy().reset_index(drop=True)
    m_unif, bases_unif, cov_unif = evaluate_base_selection(
        prio_df, unif_df, coverage_radius_km=cov_rad, max_response_time_min=max_time, detour_factor=detour, speed_kmh=speed
    )

    # Also evaluate active bases for each M6 budget scenario
    scenarios: dict[str, dict[str, Any]] = {}
    m1_def = m1_summary.get("default_solution", {})

    def _pack_scenario(
        key: str,
        label: str,
        metrics: dict[str, Any],
        b_df: pd.DataFrame,
        c_df: pd.DataFrame,
    ) -> None:
        cov_map = c_df.set_index("grid_id")
        cov35_ids = [gid for gid in prio_df["grid_id"] if bool(cov_map.loc[gid, "is_covered_radius"])]
        cov60_ids = [gid for gid in prio_df["grid_id"] if bool(cov_map.loc[gid, "is_covered_time_limit"])]
        b_list = [
            {
                "base_id": str(r.get("base_id", f"BASE_{idx:02d}")),
                "grid_id": str(r["grid_id"]),
                "lat": round(float(r["land_rep_lat"]), 5),
                "lon": round(float(r["land_rep_lon"]), 5),
                "tier": str(r.get("capacity_tier", "MCLP")),
            }
            for idx, r in b_df.iterrows()
        ]
        scenarios[key] = {
            "label": label,
            "metrics": metrics,
            "bases": b_list,
            "cov35": cov35_ids,
            "cov60": cov60_ids,
        }

    _pack_scenario("mclp_10", "MCLP Optimal (10 Bases)", m1_def, bases_m1, cov_m1)
    _pack_scenario("naive_top10", "Baseline: Naive Top-P Greedy (10 Bases)", m_top, bases_top, cov_top)
    _pack_scenario("uniform_block", "Baseline: Spatial-Block Uniform (10 Bases)", m_unif, bases_unif, cov_unif)

    for _, prow in m6_pareto.iterrows():
        sname = str(prow["scenario_name"])
        sub_b = m6_bases_all[(m6_bases_all["scenario_name"] == sname) & (m6_bases_all["is_active"])].copy().reset_index(drop=True)
        if not sub_b.empty:
            m_b, _, cov_b = evaluate_base_selection(
                prio_df, sub_b, coverage_radius_km=cov_rad, max_response_time_min=max_time, detour_factor=detour, speed_kmh=speed
            )
            _pack_scenario(
                sname,
                f"M6 {sname} ({int(prow['active_bases'])} bases, {prow['total_cost_million_idr']:,.0f}M IDR)",
                m_b,
                sub_b,
                cov_b,
            )

    grid_geom = _prepare_grid_records(prio_df, ["priority_class", "priority_rank", "priority_score"])
    geo_35_m = (cov_rad / detour) * 1000.0

    # Build M6 Pareto HTML table
    pareto_rows_html = ""
    for _, r in m6_pareto.iterrows():
        pareto_rows_html += (
            f"<tr>"
            f"<td><b>{r['scenario_name']}</b></td>"
            f"<td>{r['budget_million_idr']:,.0f}</td>"
            f"<td>{r['total_cost_million_idr']:,.0f}</td>"
            f"<td>{r['capex_million_idr']:,.0f} / {r['opex_million_idr']:,.0f}</td>"
            f"<td>{int(r['active_bases'])} ({int(r['tier_small'])}S/{int(r['tier_medium'])}M/{int(r['tier_large'])}L)</td>"
            f"<td>{int(r['trucks'])}/{int(r['tankers'])}/{int(r['pumps'])}</td>"
            f"<td>{int(r['personnel'])}</td>"
            f"<td>{int(r['relays'])}</td>"
            f"<td><b>{r['regional_demand_served_pct']:.2f}%</b></td>"
            f"<td>{r['rapid_35km_coverage_pct']:.2f}%</td>"
            f"<td>{r['baseline_regional_demand_served_pct']:.2f}%</td>"
            f"<td style='color:#34d399;'>+{r['gain_vs_baseline_pct_pts']:.2f} pp</td>"
            f"<td>{r['marginal_pct_per_billion_idr']:.3f}</td>"
            f"</tr>"
        )

    extra_body_html = f"""
    <div class="pareto-drawer" id="paretoDrawer">
      <div class="pareto-header">
        <span><strong>Module 6 Budget Pareto Tradeoff (Synthetic Scenarios, IDR Million)</strong> — Total Regional Demand = 362.6335</span>
        <button id="toggleParetoBtn" class="pareto-btn">Hide Table</button>
      </div>
      <div class="pareto-scroll" id="paretoScroll">
        <table class="pareto-table">
          <thead>
            <tr>
              <th>Scenario</th><th>Budget (M)</th><th>Total Cost (M)</th><th>CAPEX / OPEX (M)</th>
              <th>Bases (S/M/L)</th><th>Fleet (T/Tk/P)</th><th>Crew</th><th>Relays</th>
              <th>Demand Served</th><th>Rapid 35km Cov</th><th>Baseline</th><th>Gain vs Base</th><th>Marginal %/B</th>
            </tr>
          </thead>
          <tbody>{pareto_rows_html}</tbody>
        </table>
      </div>
    </div>
    """
    extra_css = """
    .pareto-drawer {
      position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
      z-index: 1040; max-width: 980px; width: calc(100% - 390px);
      background: rgba(15, 23, 42, 0.94); border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 10px; padding: 8px 12px; backdrop-filter: blur(10px);
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.6);
    }
    .pareto-header {
      display: flex; justify-content: space-between; align-items: center;
      font-size: 11.5px; color: #f8fafc; margin-bottom: 6px;
    }
    .pareto-btn {
      background: #1e293b; color: #38bdf8; border: 1px solid #475569;
      border-radius: 5px; padding: 2px 8px; font-size: 11px; cursor: pointer;
    }
    .pareto-scroll { overflow-x: auto; max-height: 170px; }
    .pareto-table { width: 100%; border-collapse: collapse; font-size: 10.8px; color: #e2e8f0; }
    .pareto-table th, .pareto-table td {
      padding: 4px 6px; border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      text-align: right; white-space: nowrap;
    }
    .pareto-table th:first-child, .pareto-table td:first-child { text-align: left; }
    .pareto-table th { color: #38bdf8; font-weight: 700; background: rgba(30, 41, 59, 0.9); position: sticky; top: 0; }
    """

    scenario_options = "".join(
        f'<option value="{k}">{v["label"]}</option>' for k, v in scenarios.items()
    )
    controls_html = f"""
    <div class="section-title">Scenario Coverage Switcher</div>
    <div class="ctrl-group">
      <select id="scenSelect" class="ctrl-select">{scenario_options}</select>
    </div>
    """
    stats_html = """
    <div class="section-title">Active Scenario Coverage KPIs</div>
    <div class="stat-grid" id="scenStatsGrid"></div>
    """
    legend_html = """
    <div class="section-title">Coverage Status (35 km / 60 min Proxy)</div>
    <div class="legend-item"><span class="swatch" style="background:#10b981;"></span> Covered within 35 km road (&le;46.7 min)</div>
    <div class="legend-item"><span class="swatch" style="background:#38bdf8;"></span> Covered 35 km–60 min (&le;45 km road)</div>
    <div class="legend-item"><span class="swatch" style="background:#334155;"></span> Uncovered (&gt;60 min proxy)</div>
    <div class="legend-item"><span class="dot" style="background:#f97316;"></span> Active Base + 35 km Effective Radius</div>
    """
    notes_html = (
        "<strong>SYNTHETIC SCENARIO COMPARISON:</strong> MCLP (33.59% priority @ 35km; 41.00% @ 60min; "
        "mean 92.8 min) outperforms Naive Top-P (19.95% @ 35km; 23.73% @ 60min; mean 124.1 min) by "
        "<strong>+13.64 pp</strong> at 35 km and <strong>+17.27 pp</strong> at 60 min."
    )
    js = f"""
    var riauBoundary = {_json_dumps(boundary_geo)};
    var gridGeom = {_json_dumps(grid_geom)};
    var scenarios = {_json_dumps(scenarios)};
    var geo35m = {geo_35_m:.2f};

    addRiauBoundary(riauBoundary);
    var covLayer = L.layerGroup().addTo(map);
    var baseLayer = L.layerGroup().addTo(map);

    function renderScenario() {{
      covLayer.clearLayers();
      baseLayer.clearLayers();
      var key = document.getElementById('scenSelect').value;
      var sc = scenarios[key];
      var m = sc.metrics || {{}};
      var set35 = new Set(sc.cov35 || []);
      var set60 = new Set(sc.cov60 || []);

      document.getElementById('scenStatsGrid').innerHTML =
        '<div class="stat-card"><div class="stat-val">' + Number(m.covered_priority_pct || 0).toFixed(2) + '%</div><div class="stat-lbl">Priority Covered @ 35km (' + (m.covered_cells_radius || 0) + ' cells)</div></div>' +
        '<div class="stat-card"><div class="stat-val">' + Number(m.covered_priority_time_limit_pct || 0).toFixed(2) + '%</div><div class="stat-lbl">Priority Covered @ 60min (' + (m.covered_cells_time_limit || 0) + ' cells)</div></div>' +
        '<div class="stat-card"><div class="stat-val">' + (m.tier1_cells_covered_radius || 0) + '/' + (m.tier1_cells_total || 85) + '</div><div class="stat-lbl">Tier-1 Covered @ 35km (' + (m.tier1_cells_covered_time_limit || 0) + ' @ 60m)</div></div>' +
        '<div class="stat-card"><div class="stat-val">' + Number(m.mean_travel_time_min || 0).toFixed(1) + ' min</div><div class="stat-lbl">Mean Travel (P95 ' + Number(m.p95_travel_time_min || 0).toFixed(1) + 'm)</div></div>';

      for (var i = 0; i < gridGeom.length; i++) {{
        var g = gridGeom[i];
        var in35 = set35.has(g.grid_id);
        var in60 = set60.has(g.grid_id);
        var col = in35 ? '#10b981' : (in60 ? '#38bdf8' : '#334155');
        var status = in35 ? 'Rapid Covered (<=35 km)' : (in60 ? '60-Min Covered (35-45 km)' : 'Uncovered (>60 min)');
        L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {{
          color: '#0f172a', weight: 0.45,
          fillColor: col, fillOpacity: in35 ? 0.72 : (in60 ? 0.58 : 0.36)
        }}).bindPopup(
          '<strong>' + g.grid_id + '</strong> (Rank #' + g.priority_rank + ')<br>' +
          'Scenario: <b>' + sc.label + '</b><br>' +
          'Status: <b>' + status + '</b><br>' +
          'Priority Score: ' + g.priority_score.toFixed(4)
        ).addTo(covLayer);
      }}

      (sc.bases || []).forEach(function(b) {{
        L.circle([b.lat, b.lon], {{
          radius: geo35m, color: '#fb923c', weight: 1.5, dashArray: '5,4',
          fillColor: '#f97316', fillOpacity: 0.08
        }}).addTo(baseLayer);
        L.circleMarker([b.lat, b.lon], {{
          radius: 7, color: '#ffffff', weight: 1.8, fillColor: '#f97316', fillOpacity: 0.95
        }}).bindPopup('<strong>' + b.base_id + '</strong> (' + b.grid_id + ')<br>Tier/Type: ' + b.tier + '<br><em>' + sc.label + '</em>').addTo(baseLayer);
      }});
    }}

    document.getElementById('scenSelect').addEventListener('change', renderScenario);
    document.getElementById('toggleParetoBtn').addEventListener('click', function() {{
      var sc = document.getElementById('paretoScroll');
      if (sc.style.display === 'none') {{
        sc.style.display = 'block'; this.textContent = 'Hide Table';
      }} else {{
        sc.style.display = 'none'; this.textContent = 'Show Table';
      }}
    }});
    renderScenario();
    """
    return _build_html_page(
        title="FIREGUARD ID — 08 Coverage Comparison &amp; M6 Pareto",
        subtitle="Naive baselines vs MCLP optimal 35km/60min coverage + integrated budget tradeoff",
        category_badge="SYNTHETIC OPTIMIZATION SCENARIO",
        badge_bg="rgba(220, 38, 38, 0.28)",
        badge_fg="#fca5a5",
        banner_html='<div class="top-banner banner-synthetic">SYNTHETIC OPERATIONAL SCENARIO (is_synthetic_scenario=True) — Naive vs MCLP Coverage (+13.64 pp @ 35km) &amp; M6 Budget Pareto</div>',
        legend_html=legend_html,
        controls_html=controls_html,
        stats_html=stats_html,
        notes_html=notes_html,
        extra_body_html=extra_body_html,
        extra_css=extra_css,
        map_script_js=js,
        period_text="2021-01-01 to 2025-12-31 (853 grid cells)",
        attribution_text="NASA FIRMS VIIRS N20 &amp; GADM v4.1 (IDN.24_1)",
    )


def generate_all_maps(repo_root: str | Path | None = None) -> dict[str, Any]:
    """
    Generates all 8 self-contained Leaflet HTML maps (plus updated legacy maps
    `peta_titik_api_riau.html` and `peta_kerawanan_grid_riau.html`) and writes
    `reports/visualization_manifest.json`.
    """
    root = Path(repo_root).resolve() if repo_root is not None else get_repo_root()
    proj_cfg = load_yaml("config/project.yaml", repo_root=root)
    opt_cfg = load_yaml("config/optimization.yaml", repo_root=root)["optimization"]

    processed_dir = resolve_path(proj_cfg["paths"]["processed_dir"], repo_root=root)
    outputs_dir = resolve_path(proj_cfg["paths"]["outputs_dir"], repo_root=root)
    maps_dir = ensure_dir(resolve_path(proj_cfg["paths"]["maps_dir"], repo_root=root))
    reports_dir = ensure_dir(resolve_path(proj_cfg["paths"]["reports_dir"], repo_root=root))

    # Load verified geospatial & analytical inputs
    with open(processed_dir / "riau_boundary.geojson", "r", encoding="utf-8") as fh:
        boundary_geo = json.load(fh)

    hotspots_df = pd.read_parquet(processed_dir / "firms_viirs_riau_polygon_clipped.parquet")
    grid_df = pd.read_parquet(processed_dir / "riau_grid_analytical.parquet")
    clustered_df = pd.read_parquet(processed_dir / "riau_grid_clustered.parquet")
    vuln_df = pd.read_parquet(processed_dir / "riau_vulnerability_profile.parquet")
    prio_df = pd.read_parquet(processed_dir / "riau_spatial_priorities.parquet")

    comparison_json = _load_json_if_exists(reports_dir / "clustering" / "model_comparison.json")
    vuln_summary = _load_json_if_exists(reports_dir / "vulnerability" / "vulnerability_summary.json")
    prio_summary = _load_json_if_exists(reports_dir / "prioritization" / "prioritization_summary.json")

    # Load Phase 8 optimization outputs
    m1_dir = outputs_dir / "optimization" / "m1_m2"
    m3_dir = outputs_dir / "optimization" / "m3_m4"
    m5_dir = outputs_dir / "optimization" / "m5_m6"

    bases_m1 = pd.read_parquet(m1_dir / "selected_bases_m1.parquet")
    cov_m1 = pd.read_parquet(m1_dir / "m1_cell_coverage.parquet")
    cap_m2 = pd.read_parquet(m1_dir / "base_capacity_plan_m2.parquet")
    fleet_m3 = pd.read_parquet(m3_dir / "fleet_allocation.parquet")
    pers_m4 = pd.read_parquet(m3_dir / "personnel_allocation.parquet")
    relays_m5 = pd.read_parquet(m5_dir / "m5_selected_relays.parquet")
    base_comms_m5 = pd.read_parquet(m5_dir / "m5_base_comms_status.parquet")
    m6_pareto = pd.read_parquet(m5_dir / "m6_pareto_tradeoff.parquet")
    m6_bases_all = pd.read_parquet(m5_dir / "m6_base_allocations_all_scenarios.parquet")

    m1_summary = _load_json_if_exists(m1_dir / "m1_summary.json")
    m2_summary = _load_json_if_exists(m1_dir / "m2_summary.json")
    m3_summary = _load_json_if_exists(m3_dir / "module3_summary.json")
    m4_summary = _load_json_if_exists(m3_dir / "module4_summary.json")
    m5_summary = _load_json_if_exists(m5_dir / "m5_summary.json")

    specs: list[tuple[str, str, str, list[str]]] = [
        (
            "01_historical_hotspots.html",
            _build_map_01_historical_hotspots(hotspots_df, boundary_geo),
            "raw_observations",
            ["peta_titik_api_riau.html"],
        ),
        (
            "02_spatial_grid.html",
            _build_map_02_spatial_grid(grid_df, boundary_geo),
            "derived_indicators",
            [],
        ),
        (
            "03_clustering_comparison.html",
            _build_map_03_clustering_comparison(clustered_df, comparison_json, boundary_geo),
            "derived_indicators",
            [],
        ),
        (
            "04_vulnerability_assessment.html",
            _build_map_04_vulnerability_assessment(vuln_df, vuln_summary, boundary_geo),
            "derived_indicators",
            ["peta_kerawanan_grid_riau.html"],
        ),
        (
            "05_spatial_prioritization.html",
            _build_map_05_spatial_prioritization(prio_df, prio_summary, boundary_geo),
            "derived_indicators",
            [],
        ),
        (
            "06_infrastructure_gaps.html",
            _build_map_06_infrastructure_gaps(prio_df, cov_m1, bases_m1, opt_cfg, boundary_geo),
            "gap_analysis_real_inventory_absent_with_synthetic_proxy",
            [],
        ),
        (
            "07_optimized_infrastructure.html",
            _build_map_07_optimized_infrastructure(
                prio_df,
                cov_m1,
                bases_m1,
                cap_m2,
                fleet_m3,
                pers_m4,
                relays_m5,
                base_comms_m5,
                opt_cfg,
                m1_summary,
                m2_summary,
                m3_summary,
                m4_summary,
                m5_summary,
                boundary_geo,
            ),
            "synthetic_optimization_scenario",
            [],
        ),
        (
            "08_coverage_comparison.html",
            _build_map_08_coverage_comparison(
                prio_df,
                bases_m1,
                cov_m1,
                opt_cfg,
                m1_summary,
                m6_pareto,
                m6_bases_all,
                boundary_geo,
            ),
            "synthetic_optimization_scenario",
            [],
        ),
    ]

    generated_maps: list[dict[str, Any]] = []
    for filename, html, classification, legacy_aliases in specs:
        out_path = maps_dir / filename
        out_path.write_text(html, encoding="utf-8")
        size_bytes = out_path.stat().st_size
        entry = {
            "filename": filename,
            "path": str(out_path.relative_to(root)).replace("\\", "/"),
            "size_bytes": int(size_bytes),
            "sha256": sha256_file(out_path),
            "classification": classification,
            "is_synthetic_scenario": classification == "synthetic_optimization_scenario",
            "crs": proj_cfg["project"]["crs"],
            "period": f"{proj_cfg['pipeline']['observation_start']} to {proj_cfg['pipeline']['observation_end']}",
            "attribution": "NASA FIRMS VIIRS N20 & GADM 4.1 (IDN.24_1)",
        }
        generated_maps.append(entry)
        logger.info("Generated %s (%s bytes, %s)", entry["path"], f"{size_bytes:,}", classification)

        for alias in legacy_aliases:
            alias_path = maps_dir / alias
            alias_path.write_text(html, encoding="utf-8")
            alias_size = alias_path.stat().st_size
            generated_maps.append(
                {
                    **entry,
                    "filename": alias,
                    "path": str(alias_path.relative_to(root)).replace("\\", "/"),
                    "size_bytes": int(alias_size),
                    "sha256": sha256_file(alias_path),
                    "legacy_alias_of": filename,
                }
            )
            logger.info("Updated legacy map %s (%s bytes)", alias_path.relative_to(root), f"{alias_size:,}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": proj_cfg["project"]["name"],
        "version": proj_cfg["project"]["version"],
        "crs": proj_cfg["project"]["crs"],
        "period": f"{proj_cfg['pipeline']['observation_start']} to {proj_cfg['pipeline']['observation_end']}",
        "attribution": "NASA FIRMS VIIRS N20 (2021-2025) & GADM 4.1 (IDN.24_1)",
        "n_primary_maps": 8,
        "n_total_files": len(generated_maps),
        "maps": generated_maps,
    }
    manifest_path = reports_dir / "visualization_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Visualization manifest written to %s", manifest_path)

    return {
        "n_primary_maps": 8,
        "n_total_files": len(generated_maps),
        "manifest_path": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "maps": [{k: m[k] for k in ("filename", "path", "size_bytes", "classification")} for m in generated_maps],
    }


def main() -> None:
    """CLI entrypoint for `python -m src.visualization.maps`."""
    res = generate_all_maps()
    print(f"Generated {res['n_total_files']} HTML maps ({res['n_primary_maps']} primary). Manifest: {res['manifest_path']}")


if __name__ == "__main__":
    main()
