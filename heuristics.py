"""Resource-deployment recommendation engine.

Cause-based additive deployment x situational multipliers, plus a self-explaining
breakdown (why each count) and a rule-based spatial posting plan. Pure module — no
Streamlit, no I/O — so it stays testable.
"""
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

CAUSE_BASE = {
    'vehicle_breakdown':  {'police': 2, 'wardens': 1, 'barricades': 4,  'tow_crew': 1},
    'accident':           {'police': 4, 'wardens': 2, 'barricades': 8,  'medical': 1},
    'water_logging':      {'police': 3, 'wardens': 2, 'barricades': 6,  'pump_crew': 1},
    'tree_fall':          {'police': 3, 'wardens': 1, 'barricades': 6,  'clearing_crew': 1},
    'construction':       {'police': 2, 'wardens': 2, 'barricades': 10, 'tow_crew': 0},
    'pot_holes':          {'police': 1, 'wardens': 1, 'barricades': 4,  'repair_crew': 1},
    'public_event':       {'police': 6, 'wardens': 4, 'barricades': 12, 'crowd_mgmt': 2},
    'procession':         {'police': 8, 'wardens': 4, 'barricades': 16, 'crowd_mgmt': 3},
    'vip_movement':       {'police': 10,'wardens': 6, 'barricades': 20, 'escort': 2},
    'protest':            {'police': 12,'wardens': 6, 'barricades': 20, 'crowd_mgmt': 4},
    'congestion':         {'police': 2, 'wardens': 2, 'barricades': 2,  'tow_crew': 0},
    'road_conditions':    {'police': 2, 'wardens': 1, 'barricades': 6,  'repair_crew': 1},
    'others':             {'police': 2, 'wardens': 1, 'barricades': 4,  'tow_crew': 0},
}

# What each resource is actually for — turns a bare count into a justification.
ROLE_PURPOSE = {
    'police':        "Junction control & diversion enforcement",
    'wardens':       "On-ground lane management & public guidance",
    'barricades':    "Physical lane channelisation / closure",
    'tow_crew':      "Clear disabled / abandoned vehicles",
    'medical':       "On-site casualty care & ambulance access",
    'pump_crew':     "De-water the carriageway",
    'clearing_crew': "Remove fallen tree / debris",
    'repair_crew':   "Patch road surface / potholes",
    'crowd_mgmt':    "Manage gathering / procession crowd",
    'escort':        "VIP convoy escort & corridor sterilisation",
}

# Only personnel that direct live traffic scale with situational pressure.
SCALED_ROLES = ('police', 'wardens')
LONG_EVENT_MIN = 120


@dataclass(frozen=True)
class Factor:
    """One situational multiplier that fired (e.g. 'High priority', 1.5)."""
    label: str
    multiplier: float


@dataclass(frozen=True)
class ResourceLine:
    key: str
    label: str
    count: int
    base: int
    purpose: str
    scaled: bool


@dataclass(frozen=True)
class Consequence:
    target_min: float
    tail_min: float
    note: str


@dataclass(frozen=True)
class Post:
    """A single on-ground posting of personnel at a map location."""
    label: str
    role: str
    count: int
    purpose: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Deployment:
    lines: List[ResourceLine]
    factors: List[Factor]
    total_multiplier: float
    total_personnel: int
    consequence: Consequence
    confidence: str
    counts: Dict[str, int]


def is_peak_hour(hour: int) -> bool:
    """Peak buckets: 06:00-10:00 and 17:00-22:00."""
    return (6 <= hour < 10) or (17 <= hour < 22)


def _active_factors(priority: str, requires_road_closure: bool, hour: int,
                    forecast_p50: float, corridor: str) -> List[Factor]:
    """Return only the multipliers that actually apply, in display order."""
    factors: List[Factor] = []
    if priority == 'High':
        factors.append(Factor("High priority", 1.5))
    if requires_road_closure:
        factors.append(Factor("Road closure", 1.5))
    if is_peak_hour(hour):
        factors.append(Factor("Peak hour", 1.3))
    if forecast_p50 > LONG_EVENT_MIN:
        factors.append(Factor("Long clearance (>2h)", 1.2))
    if corridor != 'Non-corridor':
        factors.append(Factor("Major corridor", 1.2))
    return factors


def confidence_label(lookup_level: Optional[int], n_lookup: Optional[int]) -> str:
    """Deployment confidence inherits from the forecast's data support."""
    if lookup_level is None or n_lookup is None:
        return "Moderate"
    if lookup_level <= 2 and n_lookup >= 20:
        return "High"
    if lookup_level >= 4 or n_lookup < 5:
        return "Low (generic base)"
    return "Moderate"


def recommend(event_cause: str, priority: str, requires_road_closure: bool, hour: int,
              forecast_p50: float, corridor: str, forecast_p95: Optional[float] = None,
              lookup_level: Optional[int] = None, n_lookup: Optional[int] = None) -> Deployment:
    """Full, self-explaining deployment recommendation."""
    base = CAUSE_BASE.get(event_cause, CAUSE_BASE['others']).copy()
    factors = _active_factors(priority, requires_road_closure, hour, forecast_p50, corridor)
    mult = 1.0
    for f in factors:
        mult *= f.multiplier

    lines: List[ResourceLine] = []
    counts: Dict[str, int] = {}
    for key, val in base.items():
        scaled = key in SCALED_ROLES
        count = math.ceil(val * mult) if scaled else val
        counts[key] = count
        lines.append(ResourceLine(
            key=key, label=key.replace('_', ' ').title(), count=count, base=val,
            purpose=ROLE_PURPOSE.get(key, "Specialist response unit"), scaled=scaled,
        ))

    total_personnel = sum(c for k, c in counts.items() if k != 'barricades')
    tail = forecast_p95 if forecast_p95 is not None else forecast_p50
    consequence = Consequence(
        target_min=forecast_p50, tail_min=tail,
        note=("Calibrated to clear within the central estimate "
              f"(P50 ~ {forecast_p50:.0f} min). Operational doctrine, not a "
              "staffing-vs-time curve — the dataset records no manpower."),
    )
    return Deployment(
        lines=lines, factors=factors, total_multiplier=round(mult, 2),
        total_personnel=total_personnel, consequence=consequence,
        confidence=confidence_label(lookup_level, n_lookup), counts=counts,
    )


def assign_posts(police: int, anchors: List[Tuple[str, str, float, float]]) -> List[Post]:
    """Distribute `police` across map anchors as a concrete posting plan.

    `anchors` items are (kind, label, lat, lon) with kind in
    {'incident', 'diversion', 'upstream'}. Guarantees the post counts sum to
    `police` exactly, with the incident point always staffed first.
    """
    if police <= 0 or not anchors:
        return []
    incident = next((a for a in anchors if a[0] == 'incident'), anchors[0])
    diversions = [a for a in anchors if a[0] == 'diversion'][:2]
    upstream = next((a for a in anchors if a[0] == 'upstream'), None)

    up_count = 1 if (upstream is not None and police >= 4) else 0
    div_target = round(0.3 * police) if diversions else 0
    div_counts = [max(1, div_target // len(diversions)) for _ in diversions] if diversions else []

    # Never starve the incident point: it must keep >= 2 (or all of a tiny force).
    floor = min(2, police)
    while div_counts and sum(div_counts) + up_count > police - floor:
        if max(div_counts) > 1:
            div_counts[div_counts.index(max(div_counts))] -= 1
        else:
            div_counts.pop()
    diversions = diversions[:len(div_counts)]
    incident_count = police - up_count - sum(div_counts)

    posts = [Post("Incident junction", "police", incident_count,
                  "Core traffic control at the incident point", incident[2], incident[3])]
    for (kind, label, lat, lon), c in zip(diversions, div_counts):
        posts.append(Post(label, "police", c, "Channel traffic onto the reroute", lat, lon))
    if up_count and upstream is not None:
        posts.append(Post(upstream[1], "police", up_count,
                          "Advance warning & speed control upstream", upstream[2], upstream[3]))
    return posts


def recommend_resources(event_cause: str, priority: str, requires_road_closure: bool,
                        hour: int, forecasted_clearance_min: float,
                        corridor: str) -> Dict[str, int]:
    """Backward-compatible flat-count view (used by older callers / tests)."""
    return recommend(event_cause, priority, requires_road_closure, hour,
                     forecasted_clearance_min, corridor).counts


def compare_deployment(recommended: Dict[str, int], actual_police: int,
                       actual_barricades: int) -> Dict[str, Any]:
    """Compare recommended vs actual for the feedback loop."""
    return {
        'police': {
            'recommended': recommended.get('police', 0),
            'actual': actual_police,
            'delta': actual_police - recommended.get('police', 0),
        },
        'barricades': {
            'recommended': recommended.get('barricades', 0),
            'actual': actual_barricades,
            'delta': actual_barricades - recommended.get('barricades', 0),
        },
    }
