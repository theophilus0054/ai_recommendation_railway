"""
Parser regex alergi & pantangan (dipakai bersama oleh keempat model).

Satu kosakata dipakai di dua sisi:
  - sisi user  : teks bebas alergi / pantangan -> grup alergen, grup pantangan,
                 istilah bebas yang tidak dikenal.
  - sisi makanan: nama, ingredients, tags, field allergens, metode masak, flag
                 dan ambang gizi per porsi -> himpunan grup yang dikandung makanan.

Semua pencocokan memakai batas kata, sehingga "terong" tidak dianggap "telur",
"santan" tidak dianggap "susu", dan "minyak zaitun" tidak dianggap alkohol.

Hasil pengecekan konflik dipakai sebagai fitur model (kelompok H) dan sebagai
filter deterministik saat inferensi.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable

# ==========================================================
# KOSAKATA ALERGEN (grup baku -> pola regex, Indonesia + Inggris)
# ==========================================================
ALLERGEN_PATTERNS: dict[str, list[str]] = {
    "ikan": [
        r"ikan", r"fish", r"salmon", r"tuna", r"tongkol", r"cakalang", r"lele",
        r"nila", r"tilapia", r"kembung", r"bandeng", r"teri", r"gurame", r"gurami", r"kakap",
        r"patin", r"sarden", r"sardine", r"mackerel", r"tenggiri", r"cod", r"anchov\w*",
        r"otak[\s-]?otak", r"pempek", r"empek[\s-]?empek", r"siomay", r"batagor", r"bawal",
    ],
    "udang": [r"udang", r"shrimps?", r"prawns?", r"ebi", r"rebon", r"terasi", r"lobster"],
    "kepiting": [r"kepiting", r"rajungan", r"crabs?"],
    "kerang": [r"kerang", r"clams?", r"oysters?", r"tiram", r"scallops?", r"remis", r"mussels?", r"simping"],
    "cumi": [r"cumi(?:[\s-]?cumi)?", r"sotong", r"squid", r"calamari", r"gurita", r"octopus"],
    "telur": [r"telur", r"telor", r"eggs?", r"mayones", r"mayonnaise", r"mayo"],
    "susu": [
        r"susu(?!\s*(?:kedelai|kedele|soya?|almond|oat|beras|kelapa))", r"milk(?<!soy milk)(?<!almond milk)(?<!oat milk)(?<!coconut milk)",
        r"dairy", r"keju", r"cheese", r"yogh?urt", r"mentega", r"butter(?<!peanut butter)",
        r"krim", r"cream", r"whey", r"laktosa", r"lactose", r"kental\s+manis", r"skm",
    ],
    "kacang": [
        r"kacang(?!\s*(?:hijau|merah|panjang|kedelai|kedele|polong|tolo|kapri|buncis|koro|hitam|kedelai))",
        r"kacang\s+tanah", r"peanuts?", r"almonds?", r"mete", r"mede", r"cashews?", r"walnuts?",
        r"hazelnuts?", r"pecans?", r"pistachios?", r"tree\s+nuts?", r"nuts?", r"bumbu\s+pecel",
        r"selai\s+kacang", r"peanut\s+butter",
    ],
    "kedelai": [
        r"kedelai", r"kedele", r"soy", r"soya", r"tahu", r"tempe", r"tofu", r"tempeh", r"edamame",
        r"kecap", r"tauco", r"miso", r"susu\s+kedelai", r"soy\s+milk",
    ],
    "gluten": [
        r"gluten", r"gandum", r"wheat", r"terigu", r"tepung\s+terigu", r"roti", r"bread",
        r"mie", r"mi(?=\s+(?:goreng|instan|ayam|rebus|kuah))", r"pasta", r"spaghetti", r"barley",
        r"malt", r"biskuit", r"biscuits?", r"crackers?", r"donat", r"donut", r"martabak",
        r"bakpao", r"panir", r"tepung\s+roti", r"oat(?:meal|s)?", r"pangsit", r"tepung(?!\s+(?:tapioka|beras|ketan|maizena|jagung|sagu|kanji))", r"kulit\s+lumpia",
    ],
    "wijen": [r"wijen", r"sesame", r"tahini"],
}

# Field "allergens" produk Supabase memakai bahasa Inggris
ENGLISH_ALLERGEN_FIELD = {
    "fish": ["ikan"], "milk": ["susu"], "dairy": ["susu"], "soy": ["kedelai"], "egg": ["telur"],
    "eggs": ["telur"], "wheat": ["gluten"], "gluten": ["gluten"], "peanuts": ["kacang"],
    "peanut": ["kacang"], "tree nuts": ["kacang"], "nuts": ["kacang"],
    "shellfish": ["udang", "kepiting", "kerang"], "crustacean": ["udang", "kepiting"],
    "sesame": ["wijen"], "shrimp": ["udang"], "squid": ["cumi"],
}

# Kata payung yang diperluas ke beberapa grup
UMBRELLA_TERMS: dict[str, list[str]] = {
    r"seafood|sea\s*food|makanan\s+laut|hasil\s+laut|hewan\s+laut": ["ikan", "udang", "kepiting", "kerang", "cumi"],
    r"shellfish|kerang[\s-]?kerangan|crustacea\w*|krustasea": ["udang", "kepiting", "kerang"],
}

# ==========================================================
# KOSAKATA PANTANGAN (sisi user). Sisi makanan dihitung dari flag/ambang.
# ==========================================================
PANTANGAN_PATTERNS: dict[str, list[str]] = {
    "pedas": [r"pedas", r"spicy", r"chil+i(?:es)?", r"sambal", r"cabai", r"cabe", r"balado", r"rica[\s-]?rica"],
    "berminyak": [r"berminyak", r"gorengan", r"goreng(?:an)?", r"fried", r"oily", r"berlemak", r"lemak\s+tinggi", r"tinggi\s+lemak"],
    "mentah": [r"mentah", r"raw", r"setengah\s+matang", r"undercooked", r"sushi", r"sashimi"],
    "kafein": [r"kafein", r"caffeine", r"kopi", r"coffee", r"energy\s+drink", r"minuman\s+berenergi"],
    "tinggi_natrium": [
        r"tinggi\s+natrium", r"natrium\s+tinggi", r"tinggi\s+garam", r"garam", r"asin", r"sodium",
        r"high\s+sodium", r"hipertensi", r"darah\s+tinggi", r"rendah\s+garam",
    ],
    "tinggi_gula": [
        r"tinggi\s+gula", r"gula\s+tinggi", r"gula", r"manis(?:an)?", r"sugar", r"sugary",
        r"diabetes", r"kencing\s+manis", r"dm",
    ],
    "santan": [r"santan", r"coconut\s+milk", r"bersantan"],
    "alkohol": [r"alkohol", r"alcohol", r"beer", r"wine", r"bir", r"tuak", r"arak", r"ciu", r"miras"],
    "daging_merah": [r"daging\s+merah", r"red\s+meat", r"daging\s+sapi", r"(?<!susu\s)sapi", r"kambing", r"beef", r"mutton"],
    "babi": [r"babi", r"pork", r"lard", r"bacon", r"ham", r"non[\s-]?halal"],
    "olahan": [r"makanan\s+olahan", r"olahan", r"processed", r"instan", r"instant", r"sosis", r"nugget", r"kornet"],
    "vegetarian": [r"vegetarian", r"vegan", r"tidak\s+makan\s+daging", r"no\s+meat"],
}

# Frasa yang berarti "tidak ada" -> diabaikan
NEGATION_PHRASES = [
    r"tidak\s+ada(?:\s+alergi|\s+pantangan)?", r"tdk\s+ada", r"gak\s+ada", r"nggak\s+ada", r"ga\s+ada",
    r"tidak\s+punya(?:\s+alergi)?", r"no\s+allerg\w*", r"none", r"nothing", r"n/?a", r"nihil",
    r"tidak\s+tahu", r"gak\s+tahu", r"ga\s+tau", r"belum\s+tahu", r"normal", r"aman",
]

STOPWORDS = {
    "alergi", "allergy", "allergic", "to", "dan", "and", "atau", "or", "makanan", "food", "pantangan",
    "saya", "aku", "terhadap", "sama", "dengan", "yang", "tidak", "boleh", "jangan", "hindari",
    "avoid", "no", "minum", "makan", "semua", "jenis", "segala", "macam", "sedikit", "banyak",
}

# Ambang gizi per porsi untuk menandai sisi makanan
FOOD_THRESHOLDS = {
    "tinggi_natrium_mg": 600.0,
    "tinggi_gula_g": 15.0,
    "berminyak_lemak_g": 20.0,
    "pedas_level": 2,
}

HEWANI_GROUPS = {"ikan", "udang", "kepiting", "kerang", "cumi"}
HEWANI_FOODTYPES = {"lauk_hewani"}
MEAT_WORDS = [r"ayam", r"chicken", r"daging", r"sapi", r"beef", r"kambing", r"bebek", r"duck", r"hati", r"ati", r"babi", r"pork", r"sosis", r"bakso", r"kornet", r"abon"]
RED_MEAT_WORDS = [r"daging\s+sapi", r"(?<!susu\s)sapi", r"beef", r"kambing", r"mutton", r"rendang", r"kornet", r"abon\s+sapi", r"bakso\s+sapi", r"iga"]
PORK_WORDS = [r"babi", r"pork", r"lard", r"bacon", r"ham"]
SANTAN_WORDS = [r"santan", r"coconut\s+milk", r"gulai", r"opor", r"lodeh", r"rendang", r"kari", r"curry", r"nasi\s+uduk"]


def _compile(patterns: Iterable[str]) -> re.Pattern:
    return re.compile(r"(?<![\w])(?:" + "|".join(patterns) + r")(?![\w])", re.IGNORECASE)


_ALLERGEN_RE = {g: _compile(p) for g, p in ALLERGEN_PATTERNS.items()}
_PANTANGAN_RE = {g: _compile(p) for g, p in PANTANGAN_PATTERNS.items()}
_UMBRELLA_RE = [(_compile([p]), groups) for p, groups in UMBRELLA_TERMS.items()]
_NEGATION_RE = re.compile(r"^\s*(?:" + "|".join(NEGATION_PHRASES) + r")\s*[.!]*\s*$", re.IGNORECASE)
_MEAT_RE = _compile(MEAT_WORDS)
_RED_MEAT_RE = _compile(RED_MEAT_WORDS)
_PORK_RE = _compile(PORK_WORDS)
_SANTAN_RE = _compile(SANTAN_WORDS)
_SPLIT_RE = re.compile(r"[,;/&\n+]|\s+(?:dan|and|atau|or|serta)\s+", re.IGNORECASE)

# Kunci untuk koreksi salah ketik (udnag -> udang, kacng -> kacang)
_TYPO_KEYS: dict[str, tuple[str, str]] = {}
for _g, _pats in ALLERGEN_PATTERNS.items():
    for _p in _pats:
        if re.fullmatch(r"[a-z]+", _p):
            _TYPO_KEYS[_p] = ("allergen", _g)
for _g in ALLERGEN_PATTERNS:
    _TYPO_KEYS[_g] = ("allergen", _g)
for _g, _pats in PANTANGAN_PATTERNS.items():
    for _p in _pats:
        if re.fullmatch(r"[a-z]+", _p):
            _TYPO_KEYS.setdefault(_p, ("pantangan", _g))


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).lower().replace("_", " ")
    text = re.sub(r"[()\[\]\"'`]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class UserRestrictions:
    allergens: set[str] = field(default_factory=set)
    pantangan: set[str] = field(default_factory=set)
    free_terms: set[str] = field(default_factory=set)

    @property
    def n_pantangan(self) -> int:
        return len(self.pantangan) + len(self.free_terms)

    def union(self, other: "UserRestrictions") -> "UserRestrictions":
        return UserRestrictions(
            self.allergens | other.allergens,
            self.pantangan | other.pantangan,
            self.free_terms | other.free_terms,
        )


def _parse_chunk(chunk: str, out: UserRestrictions, prefer: str) -> None:
    if _NEGATION_RE.match(chunk.strip(" .-")):
        return
    chunk = re.sub(r"^\s*(?:alergi|allergic\s+to|allergy|pantangan|hindari|avoid|tidak\s+boleh|no)\s+", "", chunk.strip())
    chunk = chunk.strip(" .-")
    if not chunk or _NEGATION_RE.match(chunk):
        return
    allergens: set[str] = set()
    for rx, groups in _UMBRELLA_RE:
        if rx.search(chunk):
            allergens.update(groups)
    for g, rx in _ALLERGEN_RE.items():
        if rx.search(chunk):
            allergens.add(g)
    pantangan = {g for g, rx in _PANTANGAN_RE.items() if rx.search(chunk)}
    # Teks alergi: grup alergen didahulukan; teks pantangan: grup pantangan didahulukan.
    # Contoh: pantangan "mie instan" -> olahan (bukan alergi gluten).
    if prefer == "allergen" and allergens:
        out.allergens |= allergens
        return
    if prefer == "pantangan" and pantangan:
        out.pantangan |= pantangan
        return
    if allergens or pantangan:
        out.allergens |= allergens
        out.pantangan |= pantangan
        return
    words = [w for w in re.findall(r"[a-z]+", chunk) if w not in STOPWORDS]
    if not words:
        return
    # Koreksi salah ketik satu kata (udnag -> udang, kacng -> kacang)
    if len(words) == 1:
        close = difflib.get_close_matches(words[0], list(_TYPO_KEYS), n=1, cutoff=0.8)
        if close:
            kind, g = _TYPO_KEYS[close[0]]
            (out.allergens if kind == "allergen" else out.pantangan).add(g)
            return
    out.free_terms.add(" ".join(words))


def parse_user(alergi: str | None = None, pantangan: str | None = None) -> UserRestrictions:
    """Ubah teks bebas alergi dan pantangan user menjadi grup baku + istilah bebas."""
    out = UserRestrictions()
    for raw, prefer in ((alergi, "allergen"), (pantangan, "pantangan")):
        text = normalize_text(raw)
        if not text or text in {"-", "--", "0"}:
            continue
        for chunk in _SPLIT_RE.split(text):
            _parse_chunk(chunk, out, prefer)
    return out


def parse_user_text(*texts: str | None) -> UserRestrictions:
    """Kompatibilitas: teks campuran tanpa pembedaan alergi/pantangan."""
    out = UserRestrictions()
    for raw in texts:
        text = normalize_text(raw)
        if not text or text in {"-", "--", "0"}:
            continue
        for chunk in _SPLIT_RE.split(text):
            _parse_chunk(chunk, out, "none")
    return out


def _as_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, float) and val != val:  # NaN
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(v) for v in val]
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    return [p.strip() for p in re.split(r"[|,;]", s) if p.strip()]


def _num(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def _flag(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "ya", "yes", "y"}
    try:
        return bool(val) and val == val
    except Exception:
        return False


def food_text(food: dict) -> str:
    parts = [food.get("nama") or food.get("name") or ""]
    parts += _as_list(food.get("ingredients"))
    parts += _as_list(food.get("tags"))
    return normalize_text(" | ".join(parts))


def food_groups(food: dict, portion_mult: float = 1.0) -> set[str]:
    """
    Himpunan grup yang dikandung sebuah makanan: 'ALERGEN:<g>', 'PANTANGAN:<g>',
    dan flag 'FLAG:<nama>'. `portion_mult` dipakai untuk ambang gizi per porsi.
    """
    text = food_text(food)
    groups: set[str] = set()

    for g, rx in _ALLERGEN_RE.items():
        if rx.search(text):
            groups.add(f"ALERGEN:{g}")
    for a in _as_list(food.get("allergens")):
        a_norm = normalize_text(a)
        for g in ENGLISH_ALLERGEN_FIELD.get(a_norm, []):
            groups.add(f"ALERGEN:{g}")
        for g, rx in _ALLERGEN_RE.items():
            if rx.search(a_norm):
                groups.add(f"ALERGEN:{g}")

    method = normalize_text(food.get("metode_masak") or food.get("cookingMethod") or "")
    spicy = _num(food.get("spicy_level", food.get("spicyLevel")), 0)
    fat = _num(food.get("lemak", food.get("fat"))) * portion_mult
    na = _num(food.get("natrium", food.get("sodium"))) * portion_mult
    sugar = _num(food.get("gula", food.get("sugars"))) * portion_mult

    if spicy >= FOOD_THRESHOLDS["pedas_level"] or _PANTANGAN_RE["pedas"].search(text):
        groups.add("PANTANGAN:pedas")
    if method in {"goreng", "fried", "deep fried", "deep-fried"} or fat > FOOD_THRESHOLDS["berminyak_lemak_g"] \
            or re.search(r"(?<!\w)(goreng|gorengan|fried|crispy|geprek|keripik|kerupuk)(?!\w)", text):
        groups.add("PANTANGAN:berminyak")
    if _flag(food.get("mentah")) or method in {"mentah", "raw"} or _PANTANGAN_RE["mentah"].search(text):
        groups.add("PANTANGAN:mentah")
        groups.add("FLAG:mentah")
    if _flag(food.get("berkafein")) or _PANTANGAN_RE["kafein"].search(text) or re.search(r"(?<!\w)(teh|tea|cokelat|coklat|chocolate)(?!\w)", text):
        if _flag(food.get("berkafein")) or not re.search(r"(?<!\w)(tanpa\s+kafein|decaf|herbal)(?!\w)", text):
            groups.add("PANTANGAN:kafein")
            groups.add("FLAG:berkafein")
    if na > FOOD_THRESHOLDS["tinggi_natrium_mg"]:
        groups.add("PANTANGAN:tinggi_natrium")
    if sugar > FOOD_THRESHOLDS["tinggi_gula_g"]:
        groups.add("PANTANGAN:tinggi_gula")
    if _SANTAN_RE.search(text) or method in {"gulai", "santan", "opor"}:
        groups.add("PANTANGAN:santan")
    if _flag(food.get("beralkohol")) or _PANTANGAN_RE["alkohol"].search(text):
        groups.add("PANTANGAN:alkohol")
        groups.add("FLAG:beralkohol")
    if _RED_MEAT_RE.search(text):
        groups.add("PANTANGAN:daging_merah")
    if _PORK_RE.search(text):
        groups.add("PANTANGAN:babi")
    if _flag(food.get("olahan")) or _PANTANGAN_RE["olahan"].search(text):
        groups.add("PANTANGAN:olahan")
        groups.add("FLAG:olahan")
    is_hewani = (food.get("foodType") in HEWANI_FOODTYPES) or _MEAT_RE.search(text) \
        or any(f"ALERGEN:{g}" in groups for g in HEWANI_GROUPS)
    if is_hewani:
        groups.add("PANTANGAN:vegetarian")
    if _flag(food.get("risiko_tersedak")):
        groups.add("FLAG:risiko_tersedak")
    return groups


@dataclass
class ConflictResult:
    allergen: list[str]
    pantangan: list[str]
    free_terms: list[str]

    @property
    def reasons(self) -> list[str]:
        return ([f"ALERGEN:{g}" for g in self.allergen]
                + [f"PANTANGAN:{g}" for g in self.pantangan]
                + [f"ISTILAH:{t}" for t in self.free_terms])

    @property
    def any(self) -> bool:
        return bool(self.allergen or self.pantangan or self.free_terms)


def check_conflicts(user: UserRestrictions, food: dict, groups: set[str] | None = None,
                    portion_mult: float = 1.0) -> ConflictResult:
    groups = groups if groups is not None else food_groups(food, portion_mult)
    allergen = sorted(g for g in user.allergens if f"ALERGEN:{g}" in groups)
    pantangan = sorted(g for g in user.pantangan if f"PANTANGAN:{g}" in groups)
    text = food_text(food)
    free = sorted(t for t in user.free_terms if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text))
    return ConflictResult(allergen, pantangan, free)
