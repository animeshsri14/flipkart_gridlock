"""End-to-end smoke tests. Run offline so OSRM is never hit:  GRIDLOCK_OFFLINE=1."""
import os

os.environ["GRIDLOCK_OFFLINE"] = "1"  # routing falls back to straight lines — no network

import pytest

import geo
import heuristics
import predictions


def test_geo_offset_and_bearing():
    lon, lat = geo.offset(12.97, 77.59, 90.0, 1000.0)  # 1 km due east
    assert lon > 77.59 and abs(lat - 12.97) < 1e-4
    assert abs(geo.bearing(12.97, 77.59, lat, lon) - 90.0) < 1.0


def test_path_length_and_backtrack():
    south, north = [77.59, 12.95], [77.59, 13.00]  # ~5.56 km due north
    assert 5000 < geo.path_length_m([south, north]) < 6000
    assert geo.backtrack_m([south, north, south], south, north) > 4000          # out-and-back
    assert geo.backtrack_m([south, [77.60, 12.975], north], south, north) < 200  # monotonic bow


def test_clean_route_not_loopy_but_hook_is():
    start, end = [77.59, 12.95], [77.59, 13.00]
    clean = [start, [77.60, 12.975], end]                                  # gentle bow
    hook = [start, [77.64, 12.97], [77.55, 12.96], [77.63, 12.99], end]    # zigzag loop
    assert not geo.is_loopy(clean, start, end)
    assert geo.is_loopy(hook, start, end)
    assert geo.route_score(clean, start, end) < geo.route_score(hook, start, end)


@pytest.mark.parametrize("coords,expected", [
    ([[77.59, 12.95], [77.59, 13.00]], False),                   # straight
    ([[77.59, 12.95], [77.59, 13.00], [77.59, 12.96]], True),    # overshoot then back
], ids=["straight", "overshoot-back"])
def test_is_loopy_cases(coords, expected):
    assert geo.is_loopy(coords, [77.59, 12.95], [77.59, 13.00]) is expected


def test_build_routes_emits_non_loopy_diversions_offline():
    import app  # GRIDLOCK_OFFLINE is already set at this module's import
    routes = app.build_routes("Mysore Road", 12.945, 77.53, 60.0)
    assert routes and len(routes["diversions"]) == 2
    for d in routes["diversions"]:
        assert not geo.is_loopy(d["route"]["coordinates"], routes["origin"], routes["dest"])


def test_recommend_matches_legacy_counts():
    args = ("accident", "High", True, 8, 60.0, "Tumkur Road")
    assert heuristics.recommend(*args).counts == heuristics.recommend_resources(*args)


def test_recommend_factors_and_scaling():
    dep = heuristics.recommend("accident", "High", False, 8, 60.0, "Tumkur Road")
    labels = [f.label for f in dep.factors]
    assert {"High priority", "Peak hour", "Major corridor"} <= set(labels)
    assert "Road closure" not in labels
    police = next(l for l in dep.lines if l.key == "police")
    barricades = next(l for l in dep.lines if l.key == "barricades")
    assert police.scaled and police.count >= police.base
    assert not barricades.scaled and barricades.count == barricades.base


def test_every_cause_has_purpose_and_personnel():
    for cause in heuristics.CAUSE_BASE:
        dep = heuristics.recommend(cause, "Low", False, 12, 30.0, "Non-corridor")
        assert all(line.purpose for line in dep.lines)
        assert dep.total_personnel >= 1


def test_confidence_thresholds():
    assert heuristics.confidence_label(1, 50) == "High"
    assert heuristics.confidence_label(5, 0) == "Low (generic base)"
    assert heuristics.confidence_label(3, 10) == "Moderate"
    assert heuristics.confidence_label(None, None) == "Moderate"


@pytest.mark.parametrize("police", list(range(0, 31)))
def test_assign_posts_sum_invariant(police):
    anchors = [("incident", "Incident junction", 12.97, 77.59),
               ("diversion", "A", 12.98, 77.60), ("diversion", "B", 12.96, 77.58),
               ("upstream", "Up", 12.99, 77.61)]
    posts = heuristics.assign_posts(police, anchors)
    assert sum(p.count for p in posts) == police
    assert all(p.count >= 1 for p in posts)
    if police > 0:
        assert posts[0].label == "Incident junction"


def test_assign_posts_handles_single_anchor():
    posts = heuristics.assign_posts(10, [("incident", "Incident junction", 12.97, 77.59)])
    assert len(posts) == 1 and posts[0].count == 10


def test_predictions_well_formed():
    preds = predictions.upcoming()
    assert len(preds) >= 3
    for p in preds:
        assert 0 <= p.certainty <= 100 and p.reasoning and p.cause in heuristics.CAUSE_BASE
        assert heuristics.recommend(p.cause, p.priority, p.road_closure, p.hour, 60.0, p.corridor).total_personnel >= 1
    venues = [p.venue for p in preds if p.venue]
    assert venues, "expected at least one venue scenario"
    for v in venues:
        assert len(v.gates) >= 2 and len(v.closed_segments) == len(v.gates)


def test_app_runs_end_to_end():
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "app.py"), default_timeout=90)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
