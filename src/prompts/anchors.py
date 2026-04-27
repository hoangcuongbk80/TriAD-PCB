"""PCB semantic anchors and prompt templates."""
from __future__ import annotations

PCB_ANCHORS = [
    "solder joint",
    "component pad",
    "trace",
    "via",
    "connector",
    "hole",
    "solder mask",
    "silkscreen",
]

NORMAL_TEMPLATES = [
    "a clean {anchor}",
    "a normal {anchor} on a printed circuit board",
    "a well-formed {anchor}",
    "a high-quality {anchor} component",
]

ABNORMAL_TEMPLATES = [
    "a broken {anchor}",
    "a cracked {anchor}",
    "a missing {anchor}",
    "a contaminated {anchor}",
    "a defective {anchor} with surface damage",
]
