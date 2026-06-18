import json
import os
import urllib.request
from typing import List, Optional, Tuple

import streamlit as st
import pydeck as pdk

import data_pipeline
import forecaster
import heuristics
import feedback_store
import predictions
import geo

st.set_page_config(page_title="Gridlock Intelligence System", layout="wide")

# --- Routing / map constants ------------------------------------------------
ROUTE_HALF_LEN_M = 3500
OSRM_BASE = "https://router.project-osrm.org/route/v1/driving/"
MAX_ZONE_RADIUS_M = 3000

BARRICADE_COLOR = [250, 204, 21]   # yellow — a closed/barricaded road segment
POST_COLOR = [79, 70, 229, 200]    # indigo — a police posting

# Approximate incident anchor per corridor (lat, lon).
CORRIDOR_ANCHOR = {
    "Mysore Road":    (12.9450, 77.5300),
    "Bellary Road 1": (13.0300, 77.5900),
    "Tumkur Road":    (13.0200, 77.5300),
    "Hosur Road":     (12.9000, 77.6200),
    "Non-corridor":   (12.9716, 77.5946),
}
# Compass bearing (deg from North) of each corridor's nominal flow.
CORRIDOR_BEARING = {
    "Mysore Road": 110, "Bellary Road 1": 10, "Tumkur Road": 300, "Hosur Road": 150,
}
# Two clean side bypasses (one each side). Each is loop-corrected at build time by
# trying several perpendicular offsets and keeping the route with the best quality score.
BYPASS_SIDES = [("W", -90), ("E", +90)]
BYPASS_CANDIDATE_MULTS = (1.0, 1.5)
VENUE_CANDIDATE_OFFSETS = (350.0, 650.0)
# Congestion bullseye: red core -> green clearing. (base_radius_m, RGBA)
CONGESTION_BANDS = [
    (2200, [34, 197, 94, 55]),    # green   — peripheral
    (1200, [250, 204, 21, 85]),   # yellow  — moderate
    (600,  [234, 88, 12, 115]),   # orange  — severe
    (250,  [220, 38, 38, 145]),   # red     — core
]


def _zone_scale(forecast_p50: float) -> float:
    return max(0.5, min(forecast_p50 / 60.0, 3.0))


def _cardinal(bearing_deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((bearing_deg % 360) / 45.0 + 0.5) % 8]


def congestion_zones(lat: float, lon: float, forecast_p50: float) -> List[dict]:
    """Concentric severity zones, drawn worst-last so the red core sits on top."""
    scale = _zone_scale(forecast_p50)
    return [{"position": [lon, lat], "radius": min(r * scale, MAX_ZONE_RADIUS_M), "color": c}
            for r, c in CONGESTION_BANDS]


# --- OSRM road routing (stdlib only) ----------------------------------------
def _osrm_route(points: List[List[float]]) -> dict:
    """Road-following route through `points` ([lon, lat]); straight-line fallback on failure."""
    fallback = {"coordinates": points, "duration_s": None, "distance_m": None, "fallback": True}
    if os.environ.get("GRIDLOCK_OFFLINE") == "1":
        return fallback
    coord_str = ";".join(f"{lon:.5f},{lat:.5f}" for lon, lat in points)
    url = f"{OSRM_BASE}{coord_str}?overview=full&geometries=geojson"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gridlock/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        if data.get("routes"):
            r = data["routes"][0]
            return {"coordinates": r["geometry"]["coordinates"], "duration_s": r.get("duration"),
                    "distance_m": r.get("distance"), "fallback": False}
    except Exception:
        pass
    return fallback


def _clean_bypass(origin: List[float], dest: List[float], lat: float, lon: float,
                  side_bearing: float, clearance: float) -> Tuple[List[float], dict]:
    """Pick the least-loopy road route around the core; clean geometric bow as fallback."""
    best_wp, best = None, None
    for mult in BYPASS_CANDIDATE_MULTS:
        wp = geo.offset(lat, lon, side_bearing, clearance * mult)
        route = _osrm_route([origin, wp, dest])
        if not route or not route.get("coordinates"):
            continue
        if best is None or (geo.route_score(route["coordinates"], origin, dest)
                            < geo.route_score(best["coordinates"], origin, dest)):
            best_wp, best = wp, route
    if best is None or geo.is_loopy(best["coordinates"], origin, dest):
        best_wp = geo.offset(lat, lon, side_bearing, clearance * 1.2)
        coords = [origin, best_wp, dest]
        best = {"coordinates": coords, "duration_s": None,
                "distance_m": geo.path_length_m(coords), "fallback": True}
    return best_wp, best


@st.cache_data(show_spinner="Fetching road routes…")
def build_routes(corridor: str, lat: float, lon: float, forecast_p50: float) -> Optional[dict]:
    """Blocked corridor route + two loop-corrected side bypasses (shared FROM/TO)."""
    bearing = CORRIDOR_BEARING.get(corridor)
    if bearing is None:
        return None
    origin = geo.offset(lat, lon, (bearing + 180.0) % 360.0, ROUTE_HALF_LEN_M)
    dest = geo.offset(lat, lon, bearing, ROUTE_HALF_LEN_M)
    main = _osrm_route([origin, [lon, lat], dest])

    clearance = min(CONGESTION_BANDS[-1][0] * _zone_scale(forecast_p50), MAX_ZONE_RADIUS_M) + 350.0
    diversions = []
    for _, perp in BYPASS_SIDES:
        side_bearing = (bearing + perp) % 360.0
        wp, route = _clean_bypass(origin, dest, lat, lon, side_bearing, clearance)
        diversions.append({"name": f"{_cardinal(side_bearing)} bypass", "anchor": wp, "route": route})
    return {"origin": origin, "dest": dest, "main": main, "diversions": diversions}


@st.cache_data(show_spinner=False)
def venue_reroutes(gate_tuples: Tuple[Tuple[float, float, float, float], ...]) -> List[List[dict]]:
    """Per gate, two loop-corrected dispersal routes sharing the gate (start) and dispersal (end)."""
    out: List[List[dict]] = []
    for glat, glon, dlat, dlon in gate_tuples:
        brg = geo.bearing(glat, glon, dlat, dlon)
        mlat, mlon = (glat + dlat) / 2.0, (glon + dlon) / 2.0
        start, end = [glon, glat], [dlon, dlat]
        alts = []
        for side in (1, -1):
            best = None
            for off in VENUE_CANDIDATE_OFFSETS:
                wp = geo.offset(mlat, mlon, (brg + 90 * side) % 360.0, off)
                r = _osrm_route([start, wp, end])
                if not r or not r.get("coordinates"):
                    continue
                if best is None or (geo.route_score(r["coordinates"], start, end)
                                    < geo.route_score(best["coordinates"], start, end)):
                    best = r
            if best is None or geo.is_loopy(best["coordinates"], start, end):
                wp = geo.offset(mlat, mlon, (brg + 90 * side) % 360.0, 450.0)
                coords = [start, wp, end]
                best = {"coordinates": coords, "duration_s": None,
                        "distance_m": geo.path_length_m(coords), "fallback": True}
            alts.append(best)
        out.append(alts)
    return out


# --- Cached engine ----------------------------------------------------------
@st.cache_resource
def get_forecaster() -> forecaster.Forecaster:
    df = data_pipeline.load_clearance_dataset()
    if not df.empty and "start_datetime" in df.columns:
        df = df.copy()
        df["hour_of_day"] = df["start_datetime"].dt.hour
    return forecaster.Forecaster(df)


@st.cache_data
def get_metrics():
    df = data_pipeline.load_clearance_dataset()
    if df.empty or "start_datetime" not in df.columns:
        return None
    df = df.copy()
    df["hour_of_day"] = df["start_datetime"].dt.hour
    return forecaster.evaluate_forecaster(df)


def get_feedback_stats(corridor: str, cause: str, priority: str,
                       hour_bucket: str) -> Tuple[Optional[float], int]:
    fb_df = feedback_store.get_feedback()
    if fb_df.empty:
        return None, 0
    match = fb_df[(fb_df["corridor"] == corridor) & (fb_df["event_cause"] == cause)
                  & (fb_df["priority"] == priority) & (fb_df["hour_bucket"] == hour_bucket)]
    if len(match) == 0:
        return None, 0
    return float(match["actual_clearance_min"].mean()), len(match)


# --- Map layers -------------------------------------------------------------
def _path_layer(coords: List[List[float]], color: List[int], width_px: int, name: str) -> pdk.Layer:
    return pdk.Layer("PathLayer", [{"path": coords, "name": name}], get_path="path",
                     get_color=color, get_width=4, width_min_pixels=width_px, pickable=True)


def _barricade_layers(segments: List[List[List[float]]]) -> List[pdk.Layer]:
    """Barricades: a yellow line with a dark casing so it stays visible on the yellow
    congestion band, plus a 🚧 marker."""
    if not segments:
        return []
    paths = [{"path": seg, "name": "Barricade — channelised / closed"} for seg in segments]
    mids = [{"position": [(seg[0][0] + seg[-1][0]) / 2, (seg[0][1] + seg[-1][1]) / 2],
             "label": "🚧"} for seg in segments]
    return [
        pdk.Layer("PathLayer", paths, get_path="path", get_color=[17, 24, 39],
                  get_width=9, width_min_pixels=7, pickable=False),   # dark casing
        pdk.Layer("PathLayer", paths, get_path="path", get_color=BARRICADE_COLOR,
                  get_width=5, width_min_pixels=4, pickable=True),     # yellow core
        pdk.Layer("TextLayer", mids, get_position="position", get_text="label", get_size=22),
    ]


def _post_layers(posts: List[heuristics.Post]) -> List[pdk.Layer]:
    """Indigo markers (sized by headcount) showing where police are posted."""
    if not posts:
        return []
    data = [{"position": [p.lon, p.lat], "label": f"👮{p.count}", "radius": 90 + 8 * p.count,
             "name": f"{p.label}: {p.count} police — {p.purpose}"} for p in posts]
    return [
        pdk.Layer("ScatterplotLayer", data, get_position="position", get_radius="radius",
                  get_fill_color=POST_COLOR, get_line_color=[255, 255, 255],
                  line_width_min_pixels=2, stroked=True, pickable=True),
        pdk.Layer("TextLayer", data, get_position="position", get_text="label",
                  get_size=13, get_color=[255, 255, 255], get_alignment_baseline="'center'"),
    ]


def render_map(selection: dict, routes: Optional[dict], active_idx: int, forecast_p50: float,
               use_dark: bool, posts: Optional[List[heuristics.Post]] = None,
               barricades: Optional[List[List[List[float]]]] = None) -> pdk.Deck:
    lat, lon = selection["lat"], selection["lon"]
    layers: List[pdk.Layer] = [pdk.Layer(
        "ScatterplotLayer", congestion_zones(lat, lon, forecast_p50),
        get_position="position", get_radius="radius", get_fill_color="color",
        stroked=False, pickable=False)]

    if routes:
        for i, d in enumerate(routes["diversions"]):
            if i != active_idx and d["route"]:
                layers.append(_path_layer(d["route"]["coordinates"], [100, 116, 139], 4, d["name"]))
        if routes["main"]:
            layers.append(_path_layer(routes["main"]["coordinates"], [220, 38, 38], 6, "Blocked corridor"))
        if routes["diversions"] and 0 <= active_idx < len(routes["diversions"]):
            d = routes["diversions"][active_idx]
            if d["route"]:
                layers.append(_path_layer(d["route"]["coordinates"], [22, 163, 74], 8, f"Reroute → {d['name']}"))
        endpoints = [{"position": routes["origin"], "label": "FROM", "name": "FROM (start)", "color": [37, 99, 235]},
                     {"position": routes["dest"], "label": "TO", "name": "TO (end)", "color": [217, 119, 6]}]
        layers.append(pdk.Layer("ScatterplotLayer", endpoints, get_position="position", get_radius=120,
                                get_fill_color="color", get_line_color=[255, 255, 255],
                                line_width_min_pixels=2, stroked=True, pickable=True))
        layers.append(pdk.Layer("TextLayer", endpoints, get_position="position", get_text="label",
                                get_size=15, get_color=[255, 255, 255],
                                get_alignment_baseline="'bottom'", get_pixel_offset=[0, -14]))

    layers += _barricade_layers(barricades or [])
    layers.append(pdk.Layer("ScatterplotLayer",
                            [{"position": [lon, lat], "name": f"Incident @ {lat:.4f}, {lon:.4f}"}],
                            get_position="position", get_radius=80, get_fill_color=[17, 24, 39],
                            get_line_color=[255, 255, 255], line_width_min_pixels=2,
                            stroked=True, pickable=True))
    layers += _post_layers(posts or [])

    style = pdk.map_styles.CARTO_DARK if use_dark else pdk.map_styles.CARTO_LIGHT
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12.2, pitch=0),
                    map_style=style, tooltip={"text": "{name}"})


def render_venue_map(venue: predictions.Venue, choices: List[int], use_dark: bool) -> pdk.Deck:
    routes = venue_reroutes(tuple((g.lat, g.lon, g.disp_lat, g.disp_lon) for g in venue.gates))
    layers: List[pdk.Layer] = list(_barricade_layers(venue.closed_segments))
    for i, alts in enumerate(routes):
        sel = choices[i] if i < len(choices) else 0
        for j, r in enumerate(alts):
            if not r:
                continue
            color, width = ([22, 163, 74], 7) if j == sel else ([148, 163, 184], 3)
            layers.append(_path_layer(r["coordinates"], color, width, f"{venue.gates[i].label} · route {chr(65 + j)}"))

    gate_pts = [{"position": [g.lon, g.lat], "name": g.label} for g in venue.gates]
    disp_pts = [{"position": [g.disp_lon, g.disp_lat], "name": f"{g.label} dispersal"} for g in venue.gates]
    layers.append(pdk.Layer("ScatterplotLayer", gate_pts, get_position="position", get_radius=110,
                            get_fill_color=[217, 119, 6], get_line_color=[255, 255, 255],
                            line_width_min_pixels=2, stroked=True, pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer", disp_pts, get_position="position", get_radius=85,
                            get_fill_color=[37, 99, 235], get_line_color=[255, 255, 255],
                            line_width_min_pixels=1, stroked=True, pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer", [{"position": [venue.lon, venue.lat], "name": venue.name}],
                            get_position="position", get_radius=150, get_fill_color=[17, 24, 39],
                            get_line_color=[255, 255, 255], line_width_min_pixels=2, stroked=True, pickable=True))
    style = pdk.map_styles.CARTO_DARK if use_dark else pdk.map_styles.CARTO_LIGHT
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=venue.lat, longitude=venue.lon, zoom=13.2, pitch=0),
                    map_style=style, tooltip={"text": "{name}"})


# --- Helpers ----------------------------------------------------------------
def _fmt_min(seconds: Optional[float]) -> str:
    return f"{seconds / 60:.0f} min" if seconds else "—"


def _fmt_km(metres: Optional[float]) -> str:
    return f"{metres / 1000:.1f} km" if metres else "—"


def build_anchors(selection: dict, routes: Optional[dict]) -> List[Tuple[str, str, float, float]]:
    """Map locations available to post personnel: incident, reroute points, upstream."""
    anchors = [("incident", "Incident junction", selection["lat"], selection["lon"])]
    if routes:
        for d in routes["diversions"][:2]:
            lon, lat = d["anchor"]
            anchors.append(("diversion", f"{d['name'].title()} pt", lat, lon))
        o_lon, o_lat = routes["origin"]
        anchors.append(("upstream", "Upstream warning", o_lat, o_lon))
    return anchors


ICONS = {"police": "👮", "wardens": "🦺", "barricades": "🚧", "medical": "🚑", "tow_crew": "🚜",
         "pump_crew": "💧", "clearing_crew": "🪚", "repair_crew": "🛠️", "crowd_mgmt": "📣", "escort": "🚓"}


# --- UI sections ------------------------------------------------------------
def sidebar_controls() -> dict:
    st.header("📋 Event Selector")
    st.radio("Mode", ["Historical", "Sim"], horizontal=True, index=1)
    cause = st.selectbox("Cause", list(heuristics.CAUSE_BASE.keys()))
    corridor = st.selectbox("Corridor", list(CORRIDOR_ANCHOR.keys()))
    priority = st.selectbox("Priority", ["High", "Low"])
    hour = st.slider("Hour", 0, 23, 8)
    road_closure = st.toggle("Road closure", value=False)
    lat, lon = CORRIDOR_ANCHOR.get(corridor, CORRIDOR_ANCHOR["Non-corridor"])
    return {"event_id": "sim-event", "event_type": "unplanned", "cause": cause, "corridor": corridor,
            "priority": priority, "hour": hour, "road_closure": road_closure, "lat": lat, "lon": lon}


def render_feedback_form(selection: dict) -> None:
    st.header("📝 Post-Event Feedback")
    hour_bucket = forecaster.get_hour_bucket(selection["hour"])
    with st.form("feedback_form"):
        actual_clearance = st.number_input("Actual clearance (min)", min_value=0.0, value=0.0)
        actual_manpower = st.number_input("Actual manpower", min_value=0, value=0)
        actual_barricades = st.number_input("Actual barricades", min_value=0, value=0)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Submit Feedback")
    if submitted and actual_clearance > 0:
        feedback_store.save_feedback(
            event_id=selection["event_id"], corridor=selection["corridor"], event_cause=selection["cause"],
            priority=selection["priority"], hour_bucket=hour_bucket, event_type=selection["event_type"],
            actual_clearance_min=float(actual_clearance), actual_manpower=int(actual_manpower) or None,
            actual_barricades=int(actual_barricades) or None, notes=notes or None)
        st.success("Feedback saved — forecast will re-blend on next run.")
        st.rerun()
    elif submitted:
        st.warning("Enter an actual clearance > 0 to save feedback.")


def render_deployment_hero(dep: heuristics.Deployment, posts: List[heuristics.Post]) -> None:
    """The product centrepiece: why each count, and where each unit goes."""
    with st.container(border=True):
        st.subheader("🎯 Deployment Recommendation")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total personnel", dep.total_personnel)
        m2.metric("Situational multiplier", f"×{dep.total_multiplier}")
        m3.metric("Confidence", dep.confidence)
        if dep.factors:
            st.caption("Factors applied: " + "  ·  ".join(f"{f.label} (×{f.multiplier})" for f in dep.factors))
        else:
            st.caption("Factors applied: none — base deployment for this cause.")

        for col, line in zip(st.columns(len(dep.lines)), dep.lines):
            with col:
                st.markdown(f"### {ICONS.get(line.key, '•')} {line.count}")
                st.markdown(f"**{line.label}**")
                st.caption(line.purpose)
                if line.scaled and dep.factors:
                    calc = " × ".join([str(line.base)] + [f"{f.label} ({f.multiplier})" for f in dep.factors])
                    st.caption(f"{calc} → **{line.count}**")
                else:
                    st.caption(f"Base {line.base} (not situation-scaled)")

        st.caption(f"⚠️ {dep.consequence.note} Under-resourcing drifts toward the P95 tail "
                   f"(~{dep.consequence.tail_min:.0f} min).")
        if posts:
            st.markdown("**📍 Posting plan:**  " + "  ·  ".join(f"{p.count} — {p.label}" for p in posts))


def _render_venue_block(venue: predictions.Venue, use_dark: bool) -> None:
    st.markdown(f"**{venue.name} — multi-gate egress plan**  ·  yellow = barricaded approach; "
                "toggle each gate's reroute below.")
    choices = []
    for i, (col, g) in enumerate(zip(st.columns(len(venue.gates)), venue.gates)):
        with col:
            choice = st.radio(g.label, ["Route A", "Route B"], key=f"venue_{venue.name}_{i}", horizontal=True)
            choices.append(0 if choice == "Route A" else 1)
    st.pydeck_chart(render_venue_map(venue, choices, use_dark))
    st.caption("Each gate's reroutes share the same start (the gate) and end (its dispersal hub).  "
               "🟠 gate · 🔵 dispersal · 🚧 barricaded approach.")


def render_predictions(f: forecaster.Forecaster, use_dark: bool) -> None:
    with st.expander("🔮 Upcoming congestion forecast & pre-deployment plan", expanded=False):
        st.caption("Forward-looking — the thing a navigation app can't do: quantify a *future* event "
                   "and pre-stage manpower, barricading and reroutes before it happens.")
        for p in sorted(predictions.upcoming(), key=lambda x: -x.certainty):
            fc = f.get_forecast(p.corridor, p.cause, p.priority, p.hour)
            dep = heuristics.recommend(p.cause, p.priority, p.road_closure, p.hour, fc["p50"],
                                       p.corridor, fc["p95"], fc["lookup_level"], fc["n_lookup"])
            badge = "🟢" if p.certainty >= 85 else ("🟡" if p.certainty >= 65 else "🟠")
            st.markdown(f"#### {badge} {p.name}")
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.caption(f"**{p.category}**  ·  {p.when}  ·  {p.corridor}")
            c2.metric("Certainty", f"{p.certainty}%")
            c3.metric("Est. personnel", dep.total_personnel)
            st.caption(f"_Why predictable:_ {p.reasoning}")
            st.caption(f"Est. deployment → {ICONS['police']} {dep.counts['police']} police · "
                       f"{ICONS['wardens']} {dep.counts['wardens']} wardens · "
                       f"{ICONS['barricades']} {dep.counts['barricades']} barricades · "
                       f"clears ~{fc['p50']:.0f} min")
            if p.venue:
                _render_venue_block(p.venue, use_dark)
            st.divider()


def main() -> None:
    st.title("Gridlock Intelligence System")
    st.caption("Operations decision-support for traffic control rooms — **not** a driver navigation app. "
               "We forecast event impact, recommend *who* to deploy and *why*, and learn from outcomes.")

    with st.sidebar:
        selection = sidebar_controls()

    f = get_forecaster()
    hour_bucket = forecaster.get_hour_bucket(selection["hour"])
    fb_mean, fb_n = get_feedback_stats(selection["corridor"], selection["cause"],
                                       selection["priority"], hour_bucket)
    forecast = f.get_forecast(corridor=selection["corridor"], event_cause=selection["cause"],
                              priority=selection["priority"], hour=selection["hour"],
                              feedback_mean=fb_mean, feedback_n=fb_n)
    deployment = heuristics.recommend(
        selection["cause"], selection["priority"], selection["road_closure"], selection["hour"],
        forecast["p50"], selection["corridor"], forecast["p95"], forecast["lookup_level"], forecast["n_lookup"])
    routes = build_routes(selection["corridor"], selection["lat"], selection["lon"], forecast["p50"])
    posts = heuristics.assign_posts(deployment.counts["police"], build_anchors(selection, routes))

    # Sidebar transparency + feedback.
    with st.sidebar:
        st.divider()
        st.header("📊 Forecast Transparency")
        st.text(f"Lookup level: {forecast['lookup_level']} of 5")
        st.text(f"Sample count: {forecast['n_lookup']} events")
        metrics = get_metrics()
        if metrics:
            note = "≤ baseline ✓" if metrics["mae_model"] <= metrics["mae_baseline"] else "above baseline"
            st.text(f"Holdout MAE: {metrics['mae_model']:.1f} min ({note})")
            st.text(f"vs Baseline MAE: {metrics['mae_baseline']:.1f} min")
        st.divider()
        render_feedback_form(selection)
        st.text(f"Feedback count: {feedback_store.get_feedback().shape[0]} entries")

    # HERO: the deployment recommendation.
    render_deployment_hero(deployment, posts)

    # Map.
    st.markdown("### Map View")
    top_l, top_r = st.columns([1, 1])
    with top_l:
        use_dark = st.radio("Map style", ["Street", "Satellite"], horizontal=True) == "Satellite"
    div_names = [d["name"] for d in routes["diversions"]] if routes else []
    active_idx = 0
    with top_r:
        if div_names:
            active_idx = div_names.index(st.radio("Highlight reroute", div_names, horizontal=True, key="reroute"))
        else:
            st.caption("No corridor reroutes for this selection.")

    # Barricades render whenever the engine recommends any (not only on full closure);
    # a wider span signals a full road closure.
    barricades = []
    if deployment.counts.get("barricades", 0) > 0:
        brg = CORRIDOR_BEARING.get(selection["corridor"], 0.0)
        half = 280 if selection["road_closure"] else 150
        barricades = [[geo.offset(selection["lat"], selection["lon"], brg, half),
                       geo.offset(selection["lat"], selection["lon"], (brg + 180) % 360, half)]]

    st.caption(f"🔴 Incident @ {selection['lat']:.4f}, {selection['lon']:.4f} · {selection['corridor']}  ·  "
               "red core → green clearing  ·  🔵 FROM → 🟠 TO  ·  🟢 reroute  ·  "
               "👮 police posts  ·  🚧 barricade")
    st.pydeck_chart(render_map(selection, routes, active_idx, forecast["p50"], use_dark, posts, barricades))

    if routes and ((routes["main"] and routes["main"]["fallback"])
                   or any(d["route"] and d["route"]["fallback"] for d in routes["diversions"])):
        st.warning("⚠️ Road-routing service unreachable — showing straight-line approximations.")

    # Secondary panels.
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Clearance Forecast")
        for label in ("p25", "p50", "p75", "p95"):
            st.markdown(f"**{label.upper()}:** {forecast[label]:.0f} min")
        st.caption(f"n = {forecast['n_lookup']} events"
                   + (f" · blended with {fb_n} feedback" if forecast["adjusted_by_feedback"] else ""))
    with c2:
        st.subheader("Diversion Plan")
        if routes and routes["diversions"]:
            active = selection["road_closure"] or forecast["p50"] > 60
            st.markdown(f"_Rerouting {'ACTIVE' if active else 'standby'}_")
            main_dur = routes["main"]["duration_s"] if routes["main"] else None
            for i, d in enumerate(routes["diversions"]):
                r = d["route"]
                marker = "🟢" if i == active_idx else "⚪"
                dur, dist = (r["duration_s"], r["distance_m"]) if r else (None, None)
                delta = f" · {'+' if (dur - main_dur) >= 0 else ''}{(dur - main_dur) / 60:.0f} min vs corridor" \
                    if (dur and main_dur) else ""
                st.markdown(f"{marker} **{d['name']}** — {_fmt_km(dist)}, {_fmt_min(dur)}{delta}")
        else:
            st.markdown("No diversion needed.")
    with c3:
        st.subheader("Learning Loop")
        st.markdown(f"**Feedback entries:** {feedback_store.get_feedback().shape[0]}")
        st.markdown(f"**This cell:** {fb_n} matching")

    # Forward-looking prediction section (collapsed by default).
    render_predictions(f, use_dark)


if __name__ == "__main__":
    main()
