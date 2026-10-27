"""
Safety layer deterministik (bersama untuk keempat kategori).

- Menggabungkan (union) aturan BLOKIR dari semua kondisi user (multi-kondisi).
- Konflik alergi / pantangan / istilah bebas dari parser regex selalu BLOKIR.
- BATASI tidak membuang makanan, tetapi dicatat (fitur model + reason codes).
Fungsi yang sama dipakai untuk membuat fitur H saat latih dan untuk memfilter
kandidat saat inferensi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from safety.allergy_parser import UserRestrictions, check_conflicts, food_groups

HAZARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hazards")


@lru_cache(maxsize=None)
def load_hazards(category: str) -> dict:
    path = os.path.join(HAZARD_DIR, f"{category}.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class SafetyResult:
    eligible: bool
    n_conflict_allergen: int
    n_conflict_pantangan: int
    n_conflict_free: int
    n_block_category: int
    n_limit_category: int
    block_reasons: list[str] = field(default_factory=list)
    limit_reasons: list[str] = field(default_factory=list)


def evaluate_food(user: UserRestrictions, food: dict, categories: list[str],
                  portion_mult: float = 1.0) -> SafetyResult:
    groups = food_groups(food, portion_mult)
    conflict = check_conflicts(user, food, groups=groups)
    block, limit = [], []
    for cat in categories:
        hz = load_hazards(cat)
        for rule in hz.get("blokir", []) or []:
            if any(g in groups for g in rule["groups"]):
                block.append(f"BLOKIR:{rule['code']}")
        for rule in hz.get("batasi", []) or []:
            if any(g in groups for g in rule["groups"]):
                limit.append(f"BATASI:{rule['code']}")
    block = sorted(set(block))
    limit = sorted(set(limit))
    reasons = conflict.reasons + block
    return SafetyResult(
        eligible=not reasons,
        n_conflict_allergen=len(conflict.allergen),
        n_conflict_pantangan=len(conflict.pantangan),
        n_conflict_free=len(conflict.free_terms),
        n_block_category=len(block),
        n_limit_category=len(limit),
        block_reasons=reasons,
        limit_reasons=limit,
    )
