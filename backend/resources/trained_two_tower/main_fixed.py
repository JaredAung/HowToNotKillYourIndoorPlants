"""
main_fixed.py

A cleaned, runnable version of your two-tower model script that:
- Fixes undefined-name / import issues
- Moves vocab building + config creation AFTER Mongo plant loading
- Adds missing preprocessing + batching utilities for plant tower
- Fixes the broken UserTower forward() string
- Removes the duplicate example config that overwrote the real config

You can keep your training loop elsewhere and import:
  - preprocess_plant, collate_plants
  - build_all_vocabs, build_config
  - TwoTowerMatchingModel
"""

from __future__ import annotations

import os
import re
import warnings
import random
from typing import Any, Dict, List, Tuple, Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np

# Torch will warn if NumPy isn't installed; we don't depend on it, so silence that warning.
warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy",
    module="torch._subclasses.functional_tensor",
)

# Optional: only needed if you actually use Voyage to embed text at runtime
try:
    import voyageai  
except Exception:
    voyageai = None  # type: ignore

# Mongo / dotenv are optional at import time; required if you call connect_to_mongo/load_plants_from_mongo
try:
    from dotenv import load_dotenv
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
    from pymongo.database import Database
except Exception:
    load_dotenv = None  # type: ignore
    MongoClient = None  # type: ignore
    ServerApi = None  # type: ignore
    Database = None  # type: ignore


# -----------------------------
# Constants
# -----------------------------
PAD = "<PAD>"
UNK = "<UNK>"


# -----------------------------
# 1) Vocab utilities
# -----------------------------
def build_vocab(values: List[Any]) -> Dict[str, int]:
    """
    values: iterable of strings (may include None)
    returns: dict str->int, with PAD=0, UNK=1
    """
    vocab: Dict[str, int] = {PAD: 0, UNK: 1}
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if s not in vocab:
            vocab[s] = len(vocab)
    return vocab


def vocab_lookup(vocab: Dict[str, int], key: Any) -> int:
    if key is None:
        return vocab[UNK]
    s = str(key).strip()
    if not s:
        return vocab[UNK]
    return vocab.get(s, vocab[UNK])


def pad_tag_batch(list_of_tag_id_lists: List[List[int]], pad_id: int = 0) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    list_of_tag_id_lists: List[List[int]]
    returns:
      tags_padded: LongTensor [B, T]
      mask: FloatTensor [B, T] with 1.0 for real, 0.0 for pad
    """
    B = len(list_of_tag_id_lists)
    T = max((len(x) for x in list_of_tag_id_lists), default=1)
    tags = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.float32)
    for i, ids in enumerate(list_of_tag_id_lists):
        if not ids:
            continue
        n = len(ids)
        tags[i, :n] = torch.tensor(ids, dtype=torch.long)
        mask[i, :n] = 1.0
    return tags, mask


# -----------------------------
# 2) Feature normalization + tag derivation
# -----------------------------
def normalize_light(raw: Any) -> str:
    """Collapse many light strings into canonical buckets."""
    if raw is None:
        return "medium"
    s = str(raw).strip().lower()
    if "direct" in s or "full sun" in s:
        return "direct_sun"
    if "bright" in s:
        if "indirect" in s:
            return "bright_indirect"
        return "bright_indirect"
    if "diffus" in s or "filtered" in s or "indirect" in s:
        return "bright_indirect"
    if "low" in s or "shade" in s:
        return "low"
    if "medium" in s or "moderate" in s:
        return "medium"
    return "medium"


def watering_to_tags(watering_text: Any) -> List[str]:
    """
    Convert watering free-text into consistent tags.
    Outputs e.g.:
      - watering_level:low|medium|high
      - drought_tolerant:true|false
      - watering_pattern:dry_between|keep_moist
    """
    if not watering_text:
        return ["watering_level:medium", "drought_tolerant:false"]

    s = str(watering_text).strip().lower()
    tags: List[str] = []

    keep_moist = ("keep moist" in s) or ("must not be dry" in s)
    dry_between = ("dry between" in s) or ("allow to dry" in s)

    if keep_moist:
        tags += ["watering_pattern:keep_moist", "watering_level:high", "drought_tolerant:false"]
    elif dry_between:
        tags += ["watering_pattern:dry_between", "watering_level:low", "drought_tolerant:true"]
    else:
        tags.append("watering_level:medium")
        tags.append("drought_tolerant:true" if ("drought" in s or "tolerant" in s) else "drought_tolerant:false")

    return tags


def humidity_to_tags(humidity_detail: Any) -> List[str]:
    if not humidity_detail:
        return []
    s = str(humidity_detail).strip().lower()
    if "high" in s:
        return ["humidity_level:high"]
    if "low" in s:
        return ["humidity_level:low"]
    return ["humidity_level:medium"]


def safe_celsius(temp_obj: Any, default: Optional[float] = None) -> Optional[float]:
    """temp_obj like {"celsius": 28, "fahrenheit": 82.4}"""
    if isinstance(temp_obj, dict) and temp_obj.get("celsius") is not None:
        try:
            return float(temp_obj["celsius"])
        except Exception:
            return default
    return default


def normalize_temp_celsius(c: Optional[float], lo: float = 0.0, hi: float = 40.0) -> float:
    """Min-max scale into [0,1] with clipping."""
    if c is None:
        c = 20.0
    x = (c - lo) / (hi - lo)
    return float(max(0.0, min(1.0, x)))


# -----------------------------
# 3) Plant preprocessing (schema-aware)
# -----------------------------
def preprocess_plant(plant_raw: Dict[str, Any], vocabs: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """
    plant_raw: MongoDB record (dict), like your schema.
    vocabs: dict of vocab dicts for categoricals and tags.
    """
    # Light
    ideal_light = normalize_light(plant_raw.get("ideallight"))
    tol_light = normalize_light(plant_raw.get("toleratedlight"))

    # Water tags
    watering_tags = watering_to_tags(plant_raw.get("watering"))

    # Humidity tags (optional)
    humidity_detail = (plant_raw.get("care_guidelines") or {}).get("humidity_detail")
    humidity_tags = humidity_to_tags(humidity_detail)

    # Temperature derived numeric
    tmin = safe_celsius(plant_raw.get("tempmin"), default=15.0)
    tmax = safe_celsius(plant_raw.get("tempmax"), default=28.0)
    tmin_n = normalize_temp_celsius(tmin)
    tmax_n = normalize_temp_celsius(tmax)
    t_mid = (tmin_n + tmax_n) / 2.0
    t_rng = abs(tmax_n - tmin_n)

    # Size bucket only (numeric size removed)
    size_bucket = plant_raw.get("size_bucket") or "unknown"

    # Categoricals
    climate = plant_raw.get("climate") or "unknown"
    care_level = plant_raw.get("care_level") or "unknown"
    category = plant_raw.get("category") or "unknown"
    family = plant_raw.get("family") or "unknown"
    origin = plant_raw.get("origin") or "unknown"
    toxic = 1 if plant_raw.get("toxic_to_pets") else 0

    # Multi-label fields (soil removed)
    use_list = plant_raw.get("use") or []

    use_ids = [vocab_lookup(vocabs["use_vocab"], s) for s in use_list if s]
    water_ids = [vocab_lookup(vocabs["water_vocab"], t) for t in watering_tags]
    humid_ids = [vocab_lookup(vocabs["humidity_vocab"], t) for t in humidity_tags]

    features: Dict[str, Any] = {
        # categoricals -> indices
        "ideal_light": vocab_lookup(vocabs["light_vocab"], ideal_light),
        "tolerated_light": vocab_lookup(vocabs["light_vocab"], tol_light),
        "climate": vocab_lookup(vocabs["climate_vocab"], climate),
        "care_level": vocab_lookup(vocabs["care_level_vocab"], care_level),
        "category": vocab_lookup(vocabs["category_vocab"], category),
        "family": vocab_lookup(vocabs["family_vocab"], family),
        "origin": vocab_lookup(vocabs["origin_vocab"], origin),
        "size_bucket": vocab_lookup(vocabs["size_bucket_vocab"], size_bucket),
        "toxic_to_pets": toxic,

        # numeric -> floats
        "tempmin_n": tmin_n,
        "tempmax_n": tmax_n,
        "temp_mid": t_mid,
        "temp_range": t_rng,

        # multilabel -> list[int]
        "use_ids": use_ids,
        "water_ids": water_ids,
        "humidity_ids": humid_ids,
    }

    # Description embedding is always a plant feature (from build_description_text + Voyage).
    features["description_embedding"] = plant_raw.get("description_embedding")

    return features


def collate_plants(feature_dicts: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Batch plant features into tensors for PlantTower."""
    out: Dict[str, torch.Tensor] = {
        "ideal_light": torch.tensor([f["ideal_light"] for f in feature_dicts], dtype=torch.long),
        "tolerated_light": torch.tensor([f["tolerated_light"] for f in feature_dicts], dtype=torch.long),
        "climate": torch.tensor([f["climate"] for f in feature_dicts], dtype=torch.long),
        "care_level": torch.tensor([f["care_level"] for f in feature_dicts], dtype=torch.long),
        "category": torch.tensor([f["category"] for f in feature_dicts], dtype=torch.long),
        "family": torch.tensor([f["family"] for f in feature_dicts], dtype=torch.long),
        "origin": torch.tensor([f["origin"] for f in feature_dicts], dtype=torch.long),
        "size_bucket": torch.tensor([f["size_bucket"] for f in feature_dicts], dtype=torch.long),
        "toxic_to_pets": torch.tensor([f["toxic_to_pets"] for f in feature_dicts], dtype=torch.long),

        "tempmin_n": torch.tensor([f["tempmin_n"] for f in feature_dicts], dtype=torch.float32),
        "tempmax_n": torch.tensor([f["tempmax_n"] for f in feature_dicts], dtype=torch.float32),
        "temp_mid": torch.tensor([f["temp_mid"] for f in feature_dicts], dtype=torch.float32),
        "temp_range": torch.tensor([f["temp_range"] for f in feature_dicts], dtype=torch.float32),
    }

    use_tags, use_mask = pad_tag_batch([f["use_ids"] for f in feature_dicts], pad_id=0)
    water_tags, water_mask = pad_tag_batch([f["water_ids"] for f in feature_dicts], pad_id=0)
    humidity_tags, humidity_mask = pad_tag_batch([f["humidity_ids"] for f in feature_dicts], pad_id=0)

    out.update({
        "use_tags": use_tags,
        "use_mask": use_mask,
        "water_tags": water_tags,
        "water_mask": water_mask,
        "humidity_tags": humidity_tags,
        "humidity_mask": humidity_mask,
    })

    # Description embedding (required). Use zero vector if missing.
    semantic_dim = int(os.getenv("SEMANTIC_IN_DIM", "1024"))
    zero_vec = [0.0] * semantic_dim
    desc_embs = [f.get("description_embedding") or zero_vec for f in feature_dicts]
    out["description_embedding"] = torch.tensor(desc_embs, dtype=torch.float32)

    return out


# -----------------------------
# 3b) User preprocessing for training from JSON
# -----------------------------
EXPERIENCE_MAP = {"beginner": 0, "intermediate": 1, "advanced": 2}
HUMIDITY_MAP = {"low": 0, "medium": 1, "high": 2}
SPACE_MAP = {"small": 0, "medium": 1, "large": 2}
COMMIT_MAP = {"low": 0, "medium": 1, "high": 2}
SUN_TIME_BUCKETS = {"low": 0, "medium": 1, "high": 2}
LIGHT_BUCKET_MAP = {"direct_sun": 0, "bright_indirect": 1, "medium": 2, "low": 3}


def user_light_bucket(raw: Any) -> int:
    """Map user light string into one of 4 buckets expected by UserTower."""
    norm = normalize_light(raw)
    return LIGHT_BUCKET_MAP.get(norm, LIGHT_BUCKET_MAP["medium"])


def plant_light_bucket(raw: Any) -> int:
    """Map plant light string into the same buckets as user_light_bucket."""
    norm = normalize_light(raw)
    return LIGHT_BUCKET_MAP.get(norm, LIGHT_BUCKET_MAP["medium"])


def preprocess_user(user_raw: Dict[str, Any], vocabs: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    exp = EXPERIENCE_MAP.get(str(user_raw.get("experience_level", "beginner")).lower(), 0)
    light = user_light_bucket(user_raw.get("light_availability"))
    humid = HUMIDITY_MAP.get(str(user_raw.get("humidity_level", "medium")).lower(), 1)
    space = SPACE_MAP.get(str(user_raw.get("room_size", "medium")).lower(), 1)
    climate = vocab_lookup(vocabs["climate_vocab"], user_raw.get("climate") or "unknown")
    has_pets = 1 if user_raw.get("has_pets") else 0
    commit = COMMIT_MAP.get(str(user_raw.get("time_to_commit", "medium")).lower(), 1)

    # Average room temperature normalized to [0,1] for compatibility with plant temps
    avg_room_temp = normalize_temp_celsius(user_raw.get("average_room_temp"))

    # Sunlight time bucketed
    try:
        sun_hours = float(user_raw.get("average_sunlight_time", 0))
    except Exception:
        sun_hours = 0.0
    if sun_hours < 2:
        sun_bucket = SUN_TIME_BUCKETS["low"]
    elif sun_hours < 6:
        sun_bucket = SUN_TIME_BUCKETS["medium"]
    else:
        sun_bucket = SUN_TIME_BUCKETS["high"]

    # Size preference bucketed similarly to plant size_bucket
    size_pref = user_raw.get("max_plant_size_preference")
    try:
        size_pref_f = float(size_pref) if size_pref is not None else 50.0
    except Exception:
        size_pref_f = 50.0
    if size_pref_f <= 60:
        size_pref_bucket = SPACE_MAP["small"]
    elif size_pref_f <= 150:
        size_pref_bucket = SPACE_MAP["medium"]
    else:
        size_pref_bucket = SPACE_MAP["large"]

    use_list = user_raw.get("use") or []
    use_ids = [vocab_lookup(vocabs["use_vocab"], u) for u in use_list if u]

    # Map watering preferences to the same derived tags as plants for alignment.
    watering_tags = watering_to_tags(user_raw.get("watering_preferences") or user_raw.get("watering"))
    water_ids = [vocab_lookup(vocabs["water_vocab"], t) for t in watering_tags]

    return {
        "experience": exp,
        "light_available": light,
        "humidity": humid,
        "space_size": space,
        "climate": climate,
        "has_pets": has_pets,
        "time_to_commit": commit,
        "sun_time_bucket": sun_bucket,
        "avg_room_temp_n": avg_room_temp,
        "size_pref_bucket": size_pref_bucket,
        "use_ids": use_ids,
        "water_ids": water_ids,
    }


def collate_users(feature_dicts: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "experience": torch.tensor([f["experience"] for f in feature_dicts], dtype=torch.long),
        "light_available": torch.tensor([f["light_available"] for f in feature_dicts], dtype=torch.long),
        "humidity": torch.tensor([f["humidity"] for f in feature_dicts], dtype=torch.long),
        "space_size": torch.tensor([f["space_size"] for f in feature_dicts], dtype=torch.long),
        "climate": torch.tensor([f["climate"] for f in feature_dicts], dtype=torch.long),
        "has_pets": torch.tensor([f["has_pets"] for f in feature_dicts], dtype=torch.long),
        "time_to_commit": torch.tensor([f["time_to_commit"] for f in feature_dicts], dtype=torch.long),
        "sun_time_bucket": torch.tensor([f["sun_time_bucket"] for f in feature_dicts], dtype=torch.long),
        "avg_room_temp_n": torch.tensor([f["avg_room_temp_n"] for f in feature_dicts], dtype=torch.float32),
        "size_pref_bucket": torch.tensor([f["size_pref_bucket"] for f in feature_dicts], dtype=torch.long),
    }

    use_tags, use_mask = pad_tag_batch([f["use_ids"] for f in feature_dicts], pad_id=0)
    water_tags, water_mask = pad_tag_batch([f["water_ids"] for f in feature_dicts], pad_id=0)
    out.update({
        "use": use_tags,
        "use_mask": use_mask,
        "water": water_tags,
        "water_mask": water_mask,
    })
    return out


# -----------------------------
# 4) Model architecture (kept from your working code, but made robust)
# -----------------------------
class MultiLabelTower(nn.Module):
    """Handles list of tags using mean pooling."""
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x: torch.LongTensor, mask: torch.FloatTensor) -> torch.Tensor:
        emb = self.embedding(x)  # [B, T, D]
        emb = emb * mask.unsqueeze(-1)
        summed = torch.sum(emb, dim=1)
        count = torch.sum(mask, dim=1, keepdim=True).clamp(min=1.0)
        return summed / count


class UserTower(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        D = int(config["dim"])

        # Categorical embeddings (kept as in your file)
        self.exp_emb = nn.Embedding(3, D)
        self.light_emb = nn.Embedding(4, D)
        self.humid_emb = nn.Embedding(3, D)
        self.space_emb = nn.Embedding(3, D)
        self.climate_emb = nn.Embedding(int(config["climate_vocab_size"]), D)
        self.pets_emb = nn.Embedding(2, D)
        self.commit_emb = nn.Embedding(3, D)
        self.sun_time_emb = nn.Embedding(3, D)
        self.size_pref_emb = nn.Embedding(3, D)

        # Multi-label (Use cases, watering prefs)
        self.use_tower = MultiLabelTower(int(config["use_vocab_size"]), D)
        self.user_water_tower = MultiLabelTower(int(config["water_vocab_size"]), D)

        # Numeric projections
        self.temp_proj = nn.Linear(1, D)

        # Semantic projection (optional but supported)
        self.use_semantic_user = bool(config.get("use_semantic_user", "semantic_in_dim" in config))
        if self.use_semantic_user:
            self.semantic_proj = nn.Linear(int(config["semantic_in_dim"]), D)

        # MLP input blocks:
        # exp, light, humid, space, climate, pets, commit, sun_time, size_pref, temp_proj, use_tower, water_tower (+ semantic)
        blocks = 12 + (1 if self.use_semantic_user else 0)
        self.mlp = nn.Sequential(
            nn.Linear(D * blocks, D * 2),
            nn.ReLU(),
            nn.Linear(D * 2, int(config["output_dim"])),
        )

    def forward(self, user_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.exp_emb(user_batch["experience"]),
            self.light_emb(user_batch["light_available"]),
            self.humid_emb(user_batch["humidity"]),
            self.space_emb(user_batch["space_size"]),
            self.climate_emb(user_batch["climate"]),
            self.pets_emb(user_batch["has_pets"]),
            self.commit_emb(user_batch["time_to_commit"]),
            self.sun_time_emb(user_batch["sun_time_bucket"]),
            self.size_pref_emb(user_batch["size_pref_bucket"]),
            self.temp_proj(user_batch["avg_room_temp_n"].unsqueeze(1)),
            self.use_tower(user_batch["use"], user_batch["use_mask"]),
            self.user_water_tower(user_batch["water"], user_batch["water_mask"]),
        ]
        if self.use_semantic_user:
            parts.append(self.semantic_proj(user_batch["semantic_embedding"]))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


class PlantTower(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        D = int(config["dim"])

        # Categoricals
        self.light_emb = nn.Embedding(int(config["light_vocab_size"]), D)
        self.climate_emb = nn.Embedding(int(config["climate_vocab_size"]), D)
        self.care_level_emb = nn.Embedding(int(config["care_level_vocab_size"]), D)
        self.category_emb = nn.Embedding(int(config["category_vocab_size"]), D)
        self.family_emb = nn.Embedding(int(config["family_vocab_size"]), D)
        self.origin_emb = nn.Embedding(int(config["origin_vocab_size"]), D)
        self.size_bucket_emb = nn.Embedding(int(config["size_bucket_vocab_size"]), D)
        self.toxic_emb = nn.Embedding(2, D)

        # Numeric projections
        self.temp_proj = nn.Linear(4, D)

        # Multi-label pooled
        self.use_tower = MultiLabelTower(int(config["use_vocab_size"]), D)
        self.water_tower = MultiLabelTower(int(config["water_vocab_size"]), D)
        self.humidity_tower = MultiLabelTower(int(config["humidity_vocab_size"]), D)

        # Description embedding (optional for backward compat with checkpoints trained without it)
        self.use_semantic_plant = bool(config.get("use_semantic_plant", True))
        if self.use_semantic_plant:
            self.desc_proj = nn.Linear(int(config["semantic_in_dim"]), D)

        base_blocks = 9 + 1 + 3 + (1 if self.use_semantic_plant else 0)
        self.mlp = nn.Sequential(
            nn.Linear(D * base_blocks, D * 2),
            nn.ReLU(),
            nn.Linear(D * 2, int(config["output_dim"])),
        )

    def forward(self, plant_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        temp = torch.stack(
            [
                plant_batch["tempmin_n"],
                plant_batch["tempmax_n"],
                plant_batch["temp_mid"],
                plant_batch["temp_range"],
            ],
            dim=1,
        )  # [B, 4]

        blocks = [
            self.light_emb(plant_batch["ideal_light"]),
            self.light_emb(plant_batch["tolerated_light"]),
            self.climate_emb(plant_batch["climate"]),
            self.care_level_emb(plant_batch["care_level"]),
            self.category_emb(plant_batch["category"]),
            self.family_emb(plant_batch["family"]),
            self.origin_emb(plant_batch["origin"]),
            self.size_bucket_emb(plant_batch["size_bucket"]),
            self.toxic_emb(plant_batch["toxic_to_pets"]),
            self.temp_proj(temp),
            self.use_tower(plant_batch["use_tags"], plant_batch["use_mask"]),
            self.water_tower(plant_batch["water_tags"], plant_batch["water_mask"]),
            self.humidity_tower(plant_batch["humidity_tags"], plant_batch["humidity_mask"]),
        ]
        if self.use_semantic_plant:
            blocks.append(self.desc_proj(plant_batch["description_embedding"]))

        x = torch.cat(blocks, dim=-1)
        return self.mlp(x)


class TwoTowerMatchingModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.user_tower = UserTower(config)
        self.plant_tower = PlantTower(config)

    def forward(self, user_data: Dict[str, torch.Tensor], plant_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_vector = self.user_tower(user_data)
        plant_vector = self.plant_tower(plant_data)

        user_vector = F.normalize(user_vector, p=2, dim=1)
        plant_vector = F.normalize(plant_vector, p=2, dim=1)

        score = torch.sum(user_vector * plant_vector, dim=1)
        return score


# -----------------------------
# 5) Mongo helpers + vocab/config builders
# -----------------------------
def connect_to_mongo() -> Tuple[Any, Any]:
    if load_dotenv is None or MongoClient is None:
        raise RuntimeError("Missing dependencies. Install python-dotenv and pymongo to use Mongo functions.")
    load_dotenv()
    mongo_uri = os.getenv("MongoDB_URI") or os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MongoDB_URI missing in environment.")
    client = MongoClient(mongo_uri, server_api=ServerApi("1"))
    db = client[os.getenv("MONGO_DB_NAME", "HowNotToKillYourPlants")]
    return client, db


def load_plants_from_mongo(db: Any) -> List[Dict[str, Any]]:
    """
    Loads all plant docs from Mongo.
    Override collection name via env var MONGO_PLANTS_COLLECTION.
    """
    collection_name = os.getenv("MONGO_PLANTS_COLLECTION", "plants")
    coll = db[collection_name]
    return list(coll.find({}))


def load_users_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_all_vocabs(plants_raw: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    # Categoricals
    light_vocab = build_vocab(
        [normalize_light(p.get("ideallight")) for p in plants_raw]
        + [normalize_light(p.get("toleratedlight")) for p in plants_raw]
    )
    climate_vocab = build_vocab([p.get("climate") for p in plants_raw])
    care_level_vocab = build_vocab([p.get("care_level") for p in plants_raw])
    category_vocab = build_vocab([p.get("category") for p in plants_raw])
    family_vocab = build_vocab([p.get("family") for p in plants_raw])
    origin_vocab = build_vocab([p.get("origin") for p in plants_raw])
    size_bucket_vocab = build_vocab([p.get("size_bucket") for p in plants_raw])

    # Multi-label vocabs (soil removed)
    use_vocab = build_vocab([u for p in plants_raw for u in (p.get("use") or [])])

    # Derived tag vocabs
    water_tag_strings: List[str] = []
    humidity_tag_strings: List[str] = []
    for p in plants_raw:
        water_tag_strings.extend(watering_to_tags(p.get("watering")))
        humidity_tag_strings.extend(humidity_to_tags((p.get("care_guidelines") or {}).get("humidity_detail")))

    water_vocab = build_vocab(water_tag_strings)
    humidity_vocab = build_vocab(humidity_tag_strings)

    return {
        "light_vocab": light_vocab,
        "climate_vocab": climate_vocab,
        "care_level_vocab": care_level_vocab,
        "category_vocab": category_vocab,
        "family_vocab": family_vocab,
        "origin_vocab": origin_vocab,
        "size_bucket_vocab": size_bucket_vocab,
        "use_vocab": use_vocab,
        "water_vocab": water_vocab,
        "humidity_vocab": humidity_vocab,
    }


def build_config(vocabs: Dict[str, Dict[str, int]], *, dim: int = 32, output_dim: int = 64) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "dim": dim,
        "output_dim": output_dim,

        "light_vocab_size": len(vocabs["light_vocab"]),
        "climate_vocab_size": len(vocabs["climate_vocab"]),
        "care_level_vocab_size": len(vocabs["care_level_vocab"]),
        "category_vocab_size": len(vocabs["category_vocab"]),
        "family_vocab_size": len(vocabs["family_vocab"]),
        "origin_vocab_size": len(vocabs["origin_vocab"]),
        "size_bucket_vocab_size": len(vocabs["size_bucket_vocab"]),

        "use_vocab_size": len(vocabs["use_vocab"]),
        "water_vocab_size": len(vocabs["water_vocab"]),
        "humidity_vocab_size": len(vocabs["humidity_vocab"]),
    }

    # If you use Voyage embeddings for user semantic_embedding, set this to your embedding dimension.
    # Leaving it out will disable user semantic features by default unless you explicitly set use_semantic_user=True.
    config.setdefault("semantic_in_dim", int(os.getenv("SEMANTIC_IN_DIM", "1024")))
    config.setdefault("use_semantic_user", True)

    # Description embedding is always a plant feature.
    config.setdefault("use_semantic_plant", True)

    return config


# -----------------------------
# 6) Training utilities (negative sampling inspired by sampleTraining.txt)
# -----------------------------
def plant_id_from_doc(doc: Dict[str, Any], idx: int) -> str:
    """Choose a stable string id for a plant document."""
    return str(doc.get("_id") or doc.get("id") or doc.get("slug") or doc.get("name") or idx)


def user_id_from_doc(doc: Dict[str, Any], idx: int) -> str:
    """Choose a stable string id for a user document."""
    return str(doc.get("username") or doc.get("user_id") or doc.get("id") or idx)


def build_feature_maps(
    plants_raw: List[Dict[str, Any]],
    users_raw: List[Dict[str, Any]],
    vocabs: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Preprocess + index features by id for fast lookup."""
    plant_features: Dict[str, Dict[str, Any]] = {}
    plant_docs_by_id: Dict[str, Dict[str, Any]] = {}
    for i, p in enumerate(plants_raw):
        pid = plant_id_from_doc(p, i)
        plant_features[pid] = preprocess_plant(p, vocabs)
        plant_docs_by_id[pid] = p

    user_features = {
        user_id_from_doc(u, i): preprocess_user(u, vocabs)
        for i, u in enumerate(users_raw)
    }
    return plant_features, user_features, plant_docs_by_id


def build_positive_pairs(
    users_raw: List[Dict[str, Any]],
    plants_raw: List[Dict[str, Any]],
    plant_features: Dict[str, Dict[str, Any]],
    *,
    top_k: int = 3,
) -> List[Tuple[str, str]]:
    """
    Build synthetic (user_id, plant_id) positives using heuristic matching.
    We score plants for each user using related features (light, humidity, climate, size, use overlap)
    and take the top_k as positives per user.
    """

    def plant_score(user: Dict[str, Any], plant_raw: Dict[str, Any], plant_feat: Dict[str, Any]) -> float:
        score = 0.0

        # Light compatibility (bucket distance)
        u_light = user_light_bucket(user.get("light_availability"))
        p_light = plant_light_bucket(plant_raw.get("ideallight"))
        light_dist = abs(u_light - p_light)
        score += 2.0 / (1.0 + light_dist)  # 2 when identical, decays with distance

        # Humidity compatibility
        u_humid = str(user.get("humidity_level", "")).lower()
        humid_tags = humidity_to_tags((plant_raw.get("care_guidelines") or {}).get("humidity_detail"))
        p_humid = None
        for ht in humid_tags:
            if ht.startswith("humidity_level:"):
                p_humid = ht.split(":", 1)[1]
                break
        if p_humid:
            score += 1.0 if p_humid == u_humid else 0.25

        # Climate match (string overlap)
        u_climate = str(user.get("climate", "")).strip().lower()
        p_climate = str(plant_raw.get("climate", "")).strip().lower()
        if u_climate and p_climate:
            score += 1.0 if u_climate == p_climate else 0.2

        # Size bucket vs room size heuristic
        room = str(user.get("room_size", "medium")).lower()
        size_bucket = str(plant_raw.get("size_bucket") or "unknown").lower()
        if room == "small":
            score += 1.0 if size_bucket in {"small", "compact"} else 0.2
        elif room == "medium":
            score += 1.0 if size_bucket in {"medium", "small", "compact"} else 0.2
        else:  # large
            score += 1.0 if size_bucket not in {"small", "compact"} else 0.2

        # Use-case overlap
        u_use = {str(u).strip().lower() for u in (user.get("use") or []) if u}
        p_use = {str(u).strip().lower() for u in (plant_raw.get("use") or []) if u}
        overlap = len(u_use & p_use)
        score += 0.5 * overlap

        # Pets safety
        if user.get("has_pets"):
            score += 1.0 if not plant_raw.get("toxic_to_pets") else -1.0

        # Care level vs time to commit
        care_level = str(plant_raw.get("care_level", "unknown")).lower()
        commit = str(user.get("time_to_commit", "medium")).lower()
        if commit == "low":
            score += 1.0 if care_level in {"easy", "low"} else 0.2
        elif commit == "high":
            score += 1.0
        else:
            score += 0.6

        # Watering preference alignment (simple tag overlap)
        uwater = set(watering_to_tags(user.get("watering_preferences") or user.get("watering")))
        pwater = set(watering_to_tags(plant_raw.get("watering")))
        if uwater and pwater:
            score += 0.5 * len(uwater & pwater)
        return score

        return score

    positives: List[Tuple[str, str]] = []
    for ui, user in enumerate(users_raw):
        uid = user_id_from_doc(user, ui)
        scored: List[Tuple[float, str]] = []
        for pi, plant in enumerate(plants_raw):
            pid = plant_id_from_doc(plant, pi)
            feat = plant_features.get(pid)
            if feat is None:
                continue
            scored.append((plant_score(user, plant, feat), pid))

        if not scored:
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, pid in scored[:top_k]:
            positives.append((uid, pid))

    return positives


def sample_negative_records(
    positives: List[Tuple[str, str]],
    plant_ids: List[str],
    *,
    num_negatives: int = 3,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """For each positive, sample N negatives similar to sampleTraining.txt."""
    rng = np.random.default_rng(seed)
    plant_set = set(plant_ids)
    records: List[Dict[str, Any]] = []

    for user_id, pos_id in positives:
        if pos_id not in plant_set:
            continue
        candidates = [pid for pid in plant_ids if pid != pos_id]
        if not candidates:
            continue

        replace = len(candidates) < num_negatives
        negs = rng.choice(candidates, size=num_negatives, replace=replace).tolist()

        records.append(
            {
                "user_id": user_id,
                "positive": pos_id,
                "negatives": negs,
            }
        )

    return records


class InteractionDataset(Dataset):
    """Dataset of positive + negative plant candidates per user."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        user_features: Dict[str, Dict[str, Any]],
        plant_features: Dict[str, Dict[str, Any]],
        num_negatives: int,
    ) -> None:
        self.records = [
            r
            for r in records
            if r["user_id"] in user_features and r["positive"] in plant_features
        ]
        self.user_features = user_features
        self.plant_features = plant_features
        self.num_negatives = num_negatives
        self.plant_ids = list(plant_features.keys())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        user_id = record["user_id"]
        pos_plant = record["positive"]

        neg_ids = [
            n for n in record.get("negatives", []) if n in self.plant_features and n != pos_plant
        ]
        available = [pid for pid in self.plant_ids if pid != pos_plant]
        if not available:
            raise RuntimeError("No available plants for negative sampling.")
        while len(neg_ids) < self.num_negatives and available:
            neg_ids.append(random.choice(available))
        neg_ids = neg_ids[: self.num_negatives]

        candidates = [self.plant_features[pos_plant]] + [self.plant_features[n] for n in neg_ids]

        return {
            "user": self.user_features[user_id],
            "candidates": candidates,
        }


def collate_interactions(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch positives + negatives for a BCE/CE style ranking loss."""
    users = collate_users([s["user"] for s in samples])

    # Each sample has [pos, neg1, neg2, ...]; zip(*) groups all positives together, etc.
    candidate_lists = list(zip(*[s["candidates"] for s in samples]))
    plant_batches = [collate_plants(list(group)) for group in candidate_lists]

    targets = torch.zeros(len(samples), dtype=torch.long)  # positive is index 0

    return {
        "users": users,
        "plant_candidates": plant_batches,
        "targets": targets,
    }


def train_two_tower(
    model: TwoTowerMatchingModel,
    interactions: List[Dict[str, Any]],
    user_features: Dict[str, Dict[str, Any]],
    plant_features: Dict[str, Dict[str, Any]],
    *,
    num_negatives: int = 3,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = InteractionDataset(interactions, user_features, plant_features, num_negatives)
    if len(dataset) == 0:
        print("No interaction records available for training.")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_interactions,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps = 0

        for batch in loader:
            user_batch = {k: v.to(device) for k, v in batch["users"].items()}
            targets = batch["targets"].to(device)

            user_vec = model.user_tower(user_batch)
            user_vec = F.normalize(user_vec, p=2, dim=1)

            logits_parts: List[torch.Tensor] = []
            for plant_batch in batch["plant_candidates"]:
                plant_batch = {k: v.to(device) for k, v in plant_batch.items()}
                plant_vec = model.plant_tower(plant_batch)
                plant_vec = F.normalize(plant_vec, p=2, dim=1)
                logits_parts.append(torch.sum(user_vec * plant_vec, dim=1))

            logits = torch.stack(logits_parts, dim=1)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        print(f"Epoch {epoch}: loss {avg_loss:.4f}")


def evaluate_confusion(
    model: TwoTowerMatchingModel,
    plant_features: List[Dict[str, Any]],
    user_features: List[Dict[str, Any]],
    *,
    batch_size: int = 64,
) -> None:
    """
    Legacy eval kept for reference; considers paired indices only.
    """
    device = next(model.parameters()).device
    model.eval()

    # Encode plants
    plant_vecs: List[torch.Tensor] = []
    for start in range(0, len(plant_features), batch_size):
        batch = collate_plants(plant_features[start:start + batch_size])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            v = model.plant_tower(batch)
            v = F.normalize(v, p=2, dim=1)
        plant_vecs.append(v.cpu())
    plant_vecs_t = torch.cat(plant_vecs, dim=0)  # [P, D]

    # Encode users
    user_vecs: List[torch.Tensor] = []
    for start in range(0, len(user_features), batch_size):
        batch = collate_users(user_features[start:start + batch_size])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            v = model.user_tower(batch)
            v = F.normalize(v, p=2, dim=1)
        user_vecs.append(v.cpu())
    user_vecs_t = torch.cat(user_vecs, dim=0)  # [U, D]

    num = min(user_vecs_t.size(0), plant_vecs_t.size(0))
    user_vecs_t = user_vecs_t[:num]
    plant_vecs_t = plant_vecs_t[:num]

    sims = user_vecs_t @ plant_vecs_t.T  # [U, P]
    preds = torch.argmax(sims, dim=1)
    true = torch.arange(num)

    acc = (preds == true).float().mean().item()
    print(f"[Legacy] Top-1 accuracy (paired indices): {acc:.4f}")


def evaluate_hits(
    model: TwoTowerMatchingModel,
    interactions: List[Dict[str, Any]],
    user_features: Dict[str, Dict[str, Any]],
    plant_features: Dict[str, Dict[str, Any]],
    *,
    top_k: int = 5,
    batch_size: int = 128,
) -> None:
    """
    Evaluate hit@k using the synthetic positives from interactions.
    """
    device = next(model.parameters()).device
    model.eval()

    # Plant embeddings
    plant_ids = list(plant_features.keys())
    plant_vecs: List[torch.Tensor] = []
    for start in range(0, len(plant_ids), batch_size):
        slice_ids = plant_ids[start:start + batch_size]
        batch_feats = [plant_features[pid] for pid in slice_ids]
        batch = collate_plants(batch_feats)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            v = model.plant_tower(batch)
            v = F.normalize(v, p=2, dim=1)
        plant_vecs.append(v)
    plant_vecs_t = torch.cat(plant_vecs, dim=0)  # [P, D]
    plant_index = {pid: i for i, pid in enumerate(plant_ids)}

    # Gather positives per user
    positives_by_user: Dict[str, List[str]] = {}
    for rec in interactions:
        positives_by_user.setdefault(rec["user_id"], []).append(rec["positive"])

    # User embeddings (only those in interactions)
    users = list(positives_by_user.keys())
    user_vecs: List[torch.Tensor] = []
    user_chunks: List[List[str]] = []
    for start in range(0, len(users), batch_size):
        chunk_ids = users[start:start + batch_size]
        feats = [user_features[u] for u in chunk_ids if u in user_features]
        if not feats:
            continue
        user_chunks.append(chunk_ids)
        batch = collate_users(feats)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            v = model.user_tower(batch)
            v = F.normalize(v, p=2, dim=1)
        user_vecs.append(v)
    if not user_vecs:
        print("No users available for evaluation.")
        return
    user_vecs_t = torch.cat(user_vecs, dim=0)
    flat_users = [u for chunk in user_chunks for u in chunk]

    # Compute hits
    hits_at_1 = 0
    hits_at_k = 0
    total = 0

    sims = user_vecs_t @ plant_vecs_t.T  # [U, P]
    topk_scores, topk_indices = torch.topk(sims, k=min(top_k, plant_vecs_t.size(0)), dim=1)

    for row, user_id in enumerate(flat_users):
        positives = [pid for pid in positives_by_user.get(user_id, []) if pid in plant_index]
        if not positives:
            continue
        total += 1
        pos_indices = {plant_index[p] for p in positives}
        top_indices = set(topk_indices[row].tolist())
        if topk_indices[row, 0].item() in pos_indices:
            hits_at_1 += 1
        if top_indices & pos_indices:
            hits_at_k += 1

    if total == 0:
        print("No positives available for evaluation.")
        return

    print(f"Hit@1: {hits_at_1 / total:.4f} over {total} users")
    print(f"Hit@{top_k}: {hits_at_k / total:.4f} over {total} users")


# -----------------------------
# 8) Embedding export helpers
# -----------------------------
def encode_plant_embeddings(
    model: TwoTowerMatchingModel,
    plant_features: Dict[str, Dict[str, Any]],
    *,
    batch_size: int = 128,
) -> Dict[str, List[float]]:
    """
    Encode plant tower embeddings for all plants.
    Returns dict plant_id -> list[float]
    """
    device = next(model.parameters()).device
    model.eval()
    ids = list(plant_features.keys())
    embeddings: Dict[str, List[float]] = {}
    for start in range(0, len(ids), batch_size):
        slice_ids = ids[start:start + batch_size]
        feats = [plant_features[i] for i in slice_ids]
        batch = collate_plants(feats)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            v = model.plant_tower(batch)
            v = F.normalize(v, p=2, dim=1)
        for pid, vec in zip(slice_ids, v):
            embeddings[pid] = vec.detach().cpu().tolist()
    return embeddings


def upsert_embeddings_into_plants(
    db: Any,
    plant_embeddings: Dict[str, List[float]],
    description_embeddings: Dict[str, Any],
    plant_docs_by_id: Dict[str, Dict[str, Any]],
    *,
    collection_name: str = "plants",
) -> None:
    """
    Upsert embeddings into the plants collection.
    Fields:
      itemTowerEmbeddings: plant tower vector
      descriptionEmbeddings: optional description vector (if provided)
    """
    if db is None:
        raise RuntimeError("Mongo DB handle is None.")
    coll = db[collection_name]
    ops = []

    def make_filter(doc: Dict[str, Any], pid: str) -> Dict[str, Any]:
        if "_id" in doc:
            return {"_id": doc["_id"]}
        if "id" in doc:
            return {"id": doc["id"]}
        if "slug" in doc:
            return {"slug": doc["slug"]}
        if "name" in doc:
            return {"name": doc["name"]}
        return {"plant_id": pid}

    for pid, emb in plant_embeddings.items():
        doc = plant_docs_by_id.get(pid)
        if doc is None:
            continue
        flt = make_filter(doc, pid)
        update_set = {"itemTowerEmbeddings": emb}
        # Always include semantic (description) embedding.
        desc_emb = description_embeddings.get(pid) or doc.get("description_embedding")
        if desc_emb is not None:
            update_set["descriptionEmbeddings"] = desc_emb
        ops.append({"filter": flt, "update": {"$set": update_set}})

    if ops:
        from pymongo import UpdateOne  # type: ignore

        coll.bulk_write([UpdateOne(o["filter"], o["update"], upsert=True) for o in ops])


def save_model_checkpoint(model: TwoTowerMatchingModel, path: str = "two_tower_model.pt") -> None:
    torch.save(model.state_dict(), path)


def save_vocabs_and_config(
    vocabs: Dict[str, Dict[str, int]],
    config: Dict[str, Any],
    dir_path: str,
) -> None:
    """Save vocabs and config for inference. Call after training."""
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "two_tower_vocabs.json"), "w", encoding="utf-8") as f:
        json.dump(vocabs, f, indent=2)
    with open(os.path.join(dir_path, "two_tower_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _convert_vocabs_json_to_full_format(vocabs_arrays: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Convert vocabs.json format (arrays of keys) to full format (PAD/UNK + indices)."""
    result: Dict[str, Dict[str, int]] = {}
    for name, keys in vocabs_arrays.items():
        if not isinstance(keys, list):
            continue
        vocab: Dict[str, int] = {PAD: 0, UNK: 1}
        for k in keys:
            if k and str(k).strip() and k not in vocab:
                vocab[str(k).strip()] = len(vocab)
        result[name] = vocab
    return result


def load_vocabs_and_config(dir_path: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Any]]:
    """Load vocabs and config for inference. Falls back to resources/vocabs.json if two_tower_vocabs.json missing."""
    vocabs_path = os.path.join(dir_path, "two_tower_vocabs.json")
    config_path = os.path.join(dir_path, "two_tower_config.json")

    if os.path.exists(vocabs_path):
        with open(vocabs_path, "r", encoding="utf-8") as f:
            vocabs = json.load(f)
    else:
        # Fallback: try resources/vocabs.json (arrays of keys) and convert
        alt = os.path.normpath(os.path.join(dir_path, "..", "vocabs.json"))
        if os.path.exists(alt):
            with open(alt, "r", encoding="utf-8") as f:
                vocabs_arrays = json.load(f)
            vocabs = _convert_vocabs_json_to_full_format(vocabs_arrays)
        else:
            raise FileNotFoundError(
                f"Neither {vocabs_path} nor {alt} found. Run training first or place vocabs in resources/vocabs.json"
            )

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = build_config(vocabs)
    return vocabs, config


def profile_answers_to_user_raw(profile_answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map profile_answers (from profile_builder schema) to user_raw format expected by preprocess_user.
    """
    raw = dict(profile_answers) if profile_answers else {}
    use_val = raw.get("use")
    if use_val is not None and not isinstance(use_val, list):
        raw["use"] = [use_val] if use_val else []
    raw.setdefault("light_availability", raw.get("average_sunlight_type"))
    raw.setdefault("room_size", raw.get("max_plant_size_preference"))
    raw.setdefault("has_pets", False)
    # Parse average_room_temp if string (e.g. "25", "25°C", "77°F")
    temp_val = raw.get("average_room_temp")
    if isinstance(temp_val, str) and temp_val.strip():
        nums = re.findall(r"-?\d+\.?\d*", temp_val)
        if nums:
            t = float(nums[0])
            if "f" in temp_val.lower() or "°f" in temp_val:
                t = (t - 32) * 5 / 9
            raw["average_room_temp"] = t
    return raw


def load_two_tower_for_inference(
    ckpt_path: str,
    artifacts_dir: Optional[str] = None,
) -> Tuple[TwoTowerMatchingModel, Dict[str, Dict[str, int]], Dict[str, Any]]:
    """Load model, vocabs, config for inference. Returns (model, vocabs, config)."""
    dir_path = artifacts_dir or os.path.dirname(os.path.abspath(ckpt_path)) or "."
    vocabs, config = load_vocabs_and_config(dir_path)
    config["use_semantic_user"] = False

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    # Detect old checkpoints (trained without plant description embedding)
    if "plant_tower.desc_proj.weight" not in state:
        config["use_semantic_plant"] = False

    model = TwoTowerMatchingModel(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, vocabs, config


def encode_user_embedding(
    model: TwoTowerMatchingModel,
    profile_answers: Dict[str, Any],
    vocabs: Dict[str, Dict[str, int]],
) -> List[float]:
    """Encode user profile to embedding using UserTower. Same flow as main_fixed training."""
    user_raw = profile_answers_to_user_raw(profile_answers)
    user_features = preprocess_user(user_raw, vocabs)
    batch = collate_users([user_features])
    with torch.no_grad():
        user_vec = model.user_tower(batch)
        user_vec = F.normalize(user_vec, p=2, dim=1)
    return user_vec[0].tolist()


def extract_description_embeddings(plants_raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collect existing description embeddings on plants if present.
    Expects key 'description_embedding' already on the plant doc.
    """
    descs: Dict[str, Any] = {}
    for i, p in enumerate(plants_raw):
        pid = plant_id_from_doc(p, i)
        if p.get("description_embedding") is not None:
            descs[pid] = p["description_embedding"]
    return descs


def build_description_text(plant_raw: Dict[str, Any]) -> str:
    """
    Build a description string from the description dict.
    Extracts all non-null keys, includes key in output (key: value), concatenates when multiple.
    """
    desc = plant_raw.get("description")
    if not isinstance(desc, dict):
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        summary = plant_raw.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        return ""

    parts = []
    for key, val in desc.items():
        if val is None:
            continue
        s = str(val).strip() if val is not None else ""
        if not s:
            continue
        parts.append(f"{key}: {s}")

    return " ".join(parts) if parts else ""


def encode_description_embeddings_from_text(
    plants_raw: List[Dict[str, Any]],
    *,
    batch_size: int = 32,
    model_name: str = None,
) -> Dict[str, Any]:
    """
    Encode description text via Voyage if available and VOYAGE_API_KEY is set.
    Falls back to existing description_embedding on the plant doc when Voyage is unavailable.
    """
    if voyageai is None:
        # No client available; rely on any existing embeddings.
        return extract_description_embeddings(plants_raw)

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        return extract_description_embeddings(plants_raw)

    client = voyageai.Client(api_key=api_key)
    model = model_name or os.getenv("VOYAGE_MODEL", "voyage-large-2")

    # Collect (pid, text)
    texts: List[Tuple[str, str]] = []
    for i, p in enumerate(plants_raw):
        pid = plant_id_from_doc(p, i)
        desc_text = build_description_text(p)
        if desc_text:
            texts.append((pid, desc_text))

    if not texts:
        return extract_description_embeddings(plants_raw)

    results: Dict[str, Any] = {}
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        ids = [pid for pid, _ in chunk]
        payload = [txt for _, txt in chunk]
        try:
            resp = client.embed(model=model, texts=payload)
            # Newer voyageai client returns an EmbeddingsObject with .data entries each having .embedding
            data = getattr(resp, "data", None)
            for pid, emb_obj in zip(ids, data):
                emb = getattr(emb_obj, "embedding", None)
                if emb is None and isinstance(emb_obj, dict):
                    emb = emb_obj.get("embedding")
                if emb is not None:
                    results[pid] = emb
        except Exception:
            # On failure, skip this batch; rely on existing embeddings later.
            continue

    # Fill any missing with pre-existing description_embedding if present
    fallback = extract_description_embeddings(plants_raw)
    for pid, emb in fallback.items():
        results.setdefault(pid, emb)

    return results


# -----------------------------
# 7) Main entrypoint (training run)
# -----------------------------
def main() -> None:
    client, db = connect_to_mongo()
    try:
        plants_raw = load_plants_from_mongo(db)
        if not plants_raw:
            print("Loaded 0 plants from Mongo. Check your collection name / DB.")
            return

        users_path = os.path.join(os.path.dirname(__file__), "mock_users_130.json")
        if not os.path.exists(users_path):
            print(f"User file not found at {users_path}")
            return

        users_raw = load_users_from_json(users_path)
        if not users_raw:
            print("No users found in mock_users_130.json")
            return

        vocabs = build_all_vocabs(plants_raw)
        config = build_config(vocabs)
        config["use_semantic_user"] = False

        # Encode description embeddings first, inject into plants for build_feature_maps.
        description_embs = encode_description_embeddings_from_text(plants_raw)
        semantic_dim = config.get("semantic_in_dim", 1024)
        zero_vec = [0.0] * int(semantic_dim)
        for i, p in enumerate(plants_raw):
            pid = plant_id_from_doc(p, i)
            p["description_embedding"] = description_embs.get(pid, zero_vec)

        model = TwoTowerMatchingModel(config)

        plant_features, user_features, plant_docs_by_id = build_feature_maps(plants_raw, users_raw, vocabs)

        positives = build_positive_pairs(users_raw, plants_raw, plant_features, top_k=3)
        num_negatives = int(os.getenv("NUM_NEGATIVES", "3"))
        interactions = sample_negative_records(
            positives,
            list(plant_features.keys()),
            num_negatives=num_negatives,
            seed=42,
        )

        print(f"Training with {len(interactions)} interactions (pos + {num_negatives} negs each).")
        train_two_tower(
            model,
            interactions,
            user_features,
            plant_features,
            num_negatives=num_negatives,
            epochs=6,
            batch_size=32,
            lr=1e-3,
        )
        print("Training complete.")
        evaluate_hits(
            model,
            interactions,
            user_features,
            plant_features,
            top_k=5,
            batch_size=64,
        )
        # Legacy paired-index metric (less meaningful with synthetic positives):
        # evaluate_confusion(model, list(plant_features.values()), list(user_features.values()), batch_size=64)

        # Export plant embeddings into PlantsRawCollection and save model checkpoint
        print("Encoding plant embeddings...")
        plant_embs = encode_plant_embeddings(model, plant_features, batch_size=128)
        plants_collection = os.getenv("MONGO_PLANTS_COLLECTION", "plants")
        upsert_embeddings_into_plants(
            db,
            plant_embs,
            description_embs,
            plant_docs_by_id,
            collection_name=plants_collection,
        )
        ckpt_path = os.getenv("MODEL_CKPT_PATH", "two_tower_model.pt")
        save_model_checkpoint(model, ckpt_path)
        artifacts_dir = os.path.dirname(os.path.abspath(ckpt_path)) or "."
        save_vocabs_and_config(vocabs, config, artifacts_dir)
        print(f"Wrote {len(plant_embs)} itemTowerEmbeddings into '{plants_collection}' and saved model checkpoint + vocabs/config.")

    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
