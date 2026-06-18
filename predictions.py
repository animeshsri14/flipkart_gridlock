"""Forward-looking congestion forecast (demo data).

Google Maps reacts to congestion that already exists; a control room needs to
pre-position for congestion that is *coming*. This module enumerates the three
classes of predictable event grounded in the dataset's own columns:

  * Planned   — event_type='planned' with a known start/end (matches, VIP convoys).
  * Weather   — 'unplanned' but probability spikes with rainfall (water_logging).
  * Hotspot   — spatially predictable from corridor/junction history (breakdowns,
                still-active potholes).

Dates/venues here are illustrative demo values, as requested. Geometry is derived
from real Bengaluru anchor coordinates so the venue maps route on real roads.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import geo

VENUE_GATE_M = 350.0       # how far a gate sits from the venue centre
VENUE_DISPERSAL_M = 1500.0  # where rerouted traffic is sent to disperse


@dataclass(frozen=True)
class Gate:
    """A venue entrance/exit road that becomes a reroute point on event days."""
    label: str
    lat: float
    lon: float
    disp_lat: float   # shared dispersal endpoint for this gate's reroutes
    disp_lon: float


@dataclass(frozen=True)
class Venue:
    name: str
    lat: float
    lon: float
    gates: List[Gate]
    closed_segments: List[List[List[float]]]  # yellow barricade polylines, [[lon,lat], ...]


@dataclass(frozen=True)
class Prediction:
    name: str
    category: str          # 'Planned' | 'Weather' | 'Hotspot'
    when: str
    corridor: str
    cause: str
    priority: str
    road_closure: bool
    hour: int
    lat: float
    lon: float
    certainty: int         # 0-100, our confidence the congestion will occur
    reasoning: str
    venue: Optional[Venue] = None


def _ring_venue(name: str, lat: float, lon: float,
                gate_specs: List[tuple]) -> Venue:
    """Build a venue with gates + barricaded approach roads from compass bearings.

    gate_specs: list of (label, bearing_deg). Each approach road (venue -> gate) is
    drawn as a closed/barricaded segment; each gate disperses outward along its bearing.
    """
    gates: List[Gate] = []
    segments: List[List[List[float]]] = []
    for label, brg in gate_specs:
        g = geo.offset(lat, lon, brg, VENUE_GATE_M)
        disp = geo.offset(lat, lon, brg, VENUE_DISPERSAL_M)
        gates.append(Gate(label, g[1], g[0], disp[1], disp[0]))
        segments.append([[lon, lat], [g[0], g[1]]])  # barricaded approach road
    return Venue(name, lat, lon, gates, segments)


def upcoming() -> List[Prediction]:
    """Demo set of upcoming predictable events, richest signal first."""
    chinnaswamy = _ring_venue(
        "M. Chinnaswamy Stadium", 12.9788, 77.5996,
        [("MG Road gate", 90), ("Cubbon Road gate", 210), ("Queens Road gate", 330)],
    )
    return [
        Prediction(
            "IPL fixture — Chinnaswamy Stadium", "Planned", "Sat 21 Jun 2026, 19:00",
            "Cubbon Road", "public_event", "High", True, 19, 12.9788, 77.5996, 95,
            "Scheduled fixture (~40k capacity). Multi-gate egress floods MG / Cubbon / "
            "Queens roads 21:30-23:00 every match night — known recurring pattern.",
            chinnaswamy,
        ),
        Prediction(
            "VIP movement — Tumkur Road", "Planned", "Mon 23 Jun 2026, 09:00",
            "Tumkur Road", "vip_movement", "High", True, 9, 13.0200, 77.5300, 90,
            "Pre-approved convoy. Rolling closures through the AM peak; corridor is "
            "sterilised ~20 min ahead of the movement.",
        ),
        Prediction(
            "Active pothole hotspot — Hosur Road", "Hotspot", "Ongoing (until repaired)",
            "Hosur Road", "pot_holes", "Low", False, 18, 12.9000, 77.6200, 88,
            "Open pothole ticket, status = active. Static hazard — lane-drop slowdowns "
            "persist at this exact point until a repair crew closes the ticket.",
        ),
        Prediction(
            "Monsoon water-logging — KR Circle underpass", "Weather", "Sun 22 Jun 2026, AM",
            "Mysore Road", "water_logging", "High", False, 9, 12.9650, 77.5730, 70,
            "IMD heavy-rain warning for 22 Jun. This underpass logged repeated "
            "water-logging in past monsoons — a low-lying drainage black-spot.",
        ),
        Prediction(
            "Breakdown hotspot — ORR heavy-vehicle incline", "Hotspot", "Recurring (peak hours)",
            "Non-corridor", "vehicle_breakdown", "Low", False, 18, 12.9100, 77.6650, 60,
            "Historical cluster of heavy-vehicle stalls on the incline. Spatially "
            "predictable, timing uncertain — pre-stage a tow crew during peak windows.",
        ),
    ]
