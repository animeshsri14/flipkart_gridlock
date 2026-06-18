"""Shared city-scale geometry helpers.

Equirectangular (flat-earth) approximation — accurate to well under a metre over
a few km at Bengaluru's latitude, and far simpler than spherical trig. All
functions are pure (math only) so every module can share them without cycles.
"""
import math
from typing import List

M_PER_DEG = 111_320.0  # metres per degree of latitude


def offset(lat: float, lon: float, bearing_deg: float, dist_m: float) -> List[float]:
    """Return [lon, lat] reached by moving dist_m metres at bearing_deg from (lat, lon).

    Bearing is degrees clockwise from North. Returns [lon, lat] (pydeck order).
    """
    theta = math.radians(bearing_deg)
    dlat = dist_m * math.cos(theta) / M_PER_DEG
    cos_lat = math.cos(math.radians(lat))
    dlon = dist_m * math.sin(theta) / (M_PER_DEG * cos_lat) if cos_lat > 1e-9 else 0.0
    return [lon + dlon, lat + dlat]


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing in degrees (clockwise from North) from point 1 to point 2."""
    east = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2.0))
    north = lat2 - lat1
    return math.degrees(math.atan2(east, north)) % 360.0


# --- Route-quality scoring (used to reject "idiotic"/looping reroutes) -------
def _seg_m(a: List[float], b: List[float]) -> float:
    """Metres between two [lon, lat] points (equirectangular)."""
    lat_mean = math.radians((a[1] + b[1]) / 2.0)
    east = (b[0] - a[0]) * M_PER_DEG * math.cos(lat_mean)
    north = (b[1] - a[1]) * M_PER_DEG
    return math.hypot(east, north)


def path_length_m(coords: List[List[float]]) -> float:
    """Total length of a [lon, lat] polyline, in metres."""
    return sum(_seg_m(a, b) for a, b in zip(coords, coords[1:]))


def _axis_unit(start: List[float], end: List[float]) -> tuple:
    """Unit vector (east, north) pointing start -> end; (0, 0) if degenerate."""
    lat_mean = math.radians((start[1] + end[1]) / 2.0)
    east = (end[0] - start[0]) * M_PER_DEG * math.cos(lat_mean)
    north = (end[1] - start[1]) * M_PER_DEG
    n = math.hypot(east, north)
    return (east / n, north / n) if n > 1e-9 else (0.0, 0.0)


def backtrack_m(coords: List[List[float]], start: List[float], end: List[float]) -> float:
    """Total distance the path travels *backwards* along the start->end axis.

    A clean bow stays monotonic (~0). A hook/loop reverses direction, producing
    a large value — the signal we use to reject idiotic routes.
    """
    ax, ay = _axis_unit(start, end)
    if (ax, ay) == (0.0, 0.0):
        return 0.0
    projs = []
    for p in coords:
        lat_mean = math.radians((p[1] + start[1]) / 2.0)
        east = (p[0] - start[0]) * M_PER_DEG * math.cos(lat_mean)
        north = (p[1] - start[1]) * M_PER_DEG
        projs.append(east * ax + north * ay)
    return sum(max(0.0, projs[i] - projs[i + 1]) for i in range(len(projs) - 1))


def route_quality(coords: List[List[float]], start: List[float], end: List[float]) -> tuple:
    """Return (detour_ratio, backtrack_ratio); both relative to straight-line distance."""
    if len(coords) < 2:
        return (1.0, 0.0)
    straight = _seg_m(start, end)
    if straight < 1.0:
        return (1.0, 0.0)
    return (path_length_m(coords) / straight, backtrack_m(coords, start, end) / straight)


def route_score(coords: List[List[float]], start: List[float], end: List[float]) -> float:
    """Lower is better. Backtracking is weighted heavily as the loop signal."""
    detour, backtrack = route_quality(coords, start, end)
    return detour + 3.0 * backtrack


def is_loopy(coords: List[List[float]], start: List[float], end: List[float],
             max_detour: float = 2.6, max_backtrack: float = 0.4) -> bool:
    """True if a route doubles back or detours absurdly relative to its endpoints."""
    detour, backtrack = route_quality(coords, start, end)
    return detour > max_detour or backtrack > max_backtrack
