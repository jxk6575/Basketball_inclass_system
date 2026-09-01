"""Per-body-part clothing color descriptors for ReID prior (incl. shoes).

Pipeline
--------
1. Extract (6, 3) mean-HSV for torso / upper_arm / forearm / thigh / shin / shoe
   from pose limb patches (bbox bands as fallback).
2. From the enrollment gallery, estimate **per-part discriminability** =
   mean pairwise distance across students (parts that look the same on everyone
   get low weight; shoes / unique accents get high weight when they separate IDs).
3. Match query↔gallery with those data-driven weights (optional per-student
   uniqueness boost), not fixed torso/shoe hand weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# COCO-17 indices
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# torso / arms / legs / shoes — shoe is last for backward-compat with old (5,3) files
PARTS = ("torso", "upper_arm", "forearm", "thigh", "shin", "shoe")
N_PARTS = len(PARTS)
# Old gallery samples were (5, 3) without shoe
N_PARTS_LEGACY = 5
SHOE_IDX = 5
# Clothing (non-shoe) part indices
CLOTH_IDX = (0, 1, 2, 3, 4)


def _seg_patch(
    frame: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    half_w: float,
) -> np.ndarray | None:
    h, w = frame.shape[:2]
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
    length = float(np.hypot(bx - ax, by - ay))
    if length < 8.0:
        return None
    rw = max(4.0, half_w)
    rh = max(4.0, 0.35 * length)
    x1 = int(max(0, mx - rw))
    x2 = int(min(w, mx + rw))
    y1 = int(max(0, my - rh))
    y2 = int(min(h, my + rh))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return frame[y1:y2, x1:x2]


def _shoe_patch(
    frame: np.ndarray,
    ankle: np.ndarray,
    knee: np.ndarray | None,
    half_w: float,
    bh: float,
) -> np.ndarray | None:
    """
    Compact crop around the ankle (mostly the shoe upper), not deep into the floor.
    """
    h, w = frame.shape[:2]
    ax, ay = float(ankle[0]), float(ankle[1])
    # Prefer a box centered slightly below the ankle — avoid long tip into court
    rw = max(5.0, half_w * 1.05, 0.065 * bh)
    rh_up = max(3.0, 0.025 * bh)
    rh_dn = max(4.0, 0.040 * bh)
    if knee is not None:
        kx, ky = float(knee[0]), float(knee[1])
        vx, vy = ax - kx, ay - ky
        n = float(np.hypot(vx, vy)) + 1e-6
        # Mild lateral bias along shin direction for pointing toes
        ax = ax + 0.08 * (vx / n) * n
        ay = ay + 0.08 * (vy / n) * n
    x1 = int(max(0, ax - rw))
    x2 = int(min(w, ax + rw))
    y1 = int(max(0, ay - rh_up))
    y2 = int(min(h, ay + rh_dn))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return frame[y1:y2, x1:x2]


def _mean_hsv(patch: np.ndarray, *, allow_low_sat: bool = False) -> np.ndarray:
    """
    Mean HSV of a patch.

    Shoes are often black/white/gray (low saturation) — do not require S>15 for
    footwear or dark jersey accents.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    if allow_low_sat:
        # Keep dark / near-gray footwear; drop only crushed blacks & blown highlights
        mask = (v > 12) & (v < 252)
    else:
        mask = (v > 25) & (v < 245) & (s > 15)
    if int(mask.sum()) < 8:
        return np.mean(hsv.reshape(-1, 3), axis=0).astype(np.float32)
    return np.mean(hsv[mask], axis=0).astype(np.float32)


def _mean_hsv_shoe(patch: np.ndarray) -> np.ndarray:
    """
    Shoe mean HSV with court-floor rejection.

    Floor (green/wood) often fills the bottom of the foot crop; keep pixels that
    differ from the bottom-band floor estimate in H/S/V.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, w = hsv.shape[:2]
    if h < 4 or w < 3:
        return _mean_hsv(patch, allow_low_sat=True)
    floor = hsv[max(0, int(h * 0.65)) :, :, :]
    fh = float(np.median(floor[:, :, 0]))
    fs = float(np.median(floor[:, :, 1]))
    fv = float(np.median(floor[:, :, 2]))
    dh = np.minimum(np.abs(hsv[:, :, 0] - fh), 180.0 - np.abs(hsv[:, :, 0] - fh))
    ds = np.abs(hsv[:, :, 1] - fs)
    dv = np.abs(hsv[:, :, 2] - fv)
    # Differ from floor OR clearly darker/brighter footwear
    mask = ((dh > 14.0) | (ds > 28.0) | (dv > 32.0)) & (hsv[:, :, 2] > 12) & (hsv[:, :, 2] < 252)
    # Prefer upper 70% of crop (shoe body) when enough pixels remain
    upper = np.zeros((h, w), dtype=bool)
    upper[: max(1, int(h * 0.70)), :] = True
    mask_u = mask & upper
    use = mask_u if int(mask_u.sum()) >= 8 else mask
    if int(use.sum()) < 8:
        return _mean_hsv(patch, allow_low_sat=True)
    return np.mean(hsv[use], axis=0).astype(np.float32)


def _as_parts(desc: np.ndarray | None) -> np.ndarray | None:
    """Normalize to (N_PARTS, 3); pad legacy (5,3) with zero shoe row."""
    if desc is None:
        return None
    arr = np.asarray(desc, dtype=np.float32).reshape(-1)
    if arr.size == N_PARTS * 3:
        return arr.reshape(N_PARTS, 3)
    if arr.size == N_PARTS_LEGACY * 3:
        out = np.zeros((N_PARTS, 3), dtype=np.float32)
        out[:N_PARTS_LEGACY] = arr.reshape(N_PARTS_LEGACY, 3)
        return out
    return None


def extract_clothing_color(
    frame: np.ndarray,
    bbox: list[float] | tuple[float, ...],
    keypoints: np.ndarray | None = None,
    conf_thr: float = 0.25,
) -> np.ndarray:
    """
    Return a (6, 3) HSV mean descriptor for
    torso / upper_arm / forearm / thigh / shin / shoe.
    Falls back to bbox bands when keypoints are weak.
    """
    desc = np.zeros((N_PARTS, 3), dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if x2 < x1 or y2 < y1:
        x1, y1, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x2, y2 = x1 + bw, y1 + bh
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    k = None
    if keypoints is not None:
        k = np.asarray(keypoints, dtype=np.float64)
        if k.ndim != 2 or k.shape[0] < 17:
            k = None

    half = 0.12 * bw
    used = {p: False for p in PARTS}

    if k is not None:
        conf = k[:, 2] if k.shape[1] >= 3 else np.ones(k.shape[0])

        def ok(i: int) -> bool:
            return float(conf[i]) >= conf_thr

        # torso: shoulder mid → hip mid
        if all(ok(i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
            sm = 0.5 * (k[L_SHOULDER, :2] + k[R_SHOULDER, :2])
            hm = 0.5 * (k[L_HIP, :2] + k[R_HIP, :2])
            patch = _seg_patch(frame, sm, hm, 0.22 * bw)
            if patch is not None:
                desc[0] = _mean_hsv(patch)
                used["torso"] = True

        # upper arms / forearms / thighs / shins (average L/R when both exist)
        pairs = {
            "upper_arm": [(L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW)],
            "forearm": [(L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)],
            "thigh": [(L_HIP, L_KNEE), (R_HIP, R_KNEE)],
            "shin": [(L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE)],
        }
        idx_map = {"upper_arm": 1, "forearm": 2, "thigh": 3, "shin": 4}
        for name, segs in pairs.items():
            cols = []
            for a, b in segs:
                if ok(a) and ok(b):
                    patch = _seg_patch(frame, k[a, :2], k[b, :2], half)
                    if patch is not None:
                        cols.append(_mean_hsv(patch))
            if cols:
                desc[idx_map[name]] = np.mean(cols, axis=0)
                used[name] = True

        # shoes: past each ankle; keep L/R separately then average (both feet)
        shoe_cols = []
        for ankle_i, knee_i in ((L_ANKLE, L_KNEE), (R_ANKLE, R_KNEE)):
            if not ok(ankle_i):
                continue
            knee = k[knee_i, :2] if ok(knee_i) else None
            patch = _shoe_patch(frame, k[ankle_i, :2], knee, half, bh)
            if patch is not None:
                shoe_cols.append(_mean_hsv_shoe(patch))
        if shoe_cols:
            desc[SHOE_IDX] = np.mean(shoe_cols, axis=0)
            used["shoe"] = True

    # Bbox band fallback for missing parts
    bands = {
        "torso": (0.12, 0.48),
        "upper_arm": (0.18, 0.42),
        "forearm": (0.35, 0.55),
        "thigh": (0.48, 0.72),
        "shin": (0.72, 0.88),
        "shoe": (0.90, 0.985),
    }
    h, w = frame.shape[:2]
    for i, name in enumerate(PARTS):
        if used[name]:
            continue
        t0, t1 = bands[name]
        yy1 = int(max(0, y1 + t0 * bh))
        yy2 = int(min(h, y1 + t1 * bh))
        # Shoes: narrower vertical band + slightly wider horizontal crop
        side = 0.08 if name == "shoe" else 0.15
        xx1 = int(max(0, x1 + side * bw))
        xx2 = int(min(w, x2 - side * bw))
        if yy2 - yy1 < 3 or xx2 - xx1 < 3:
            continue
        patch = frame[yy1:yy2, xx1:xx2]
        if patch.size:
            desc[i] = _mean_hsv_shoe(patch) if name == "shoe" else _mean_hsv(patch)

    return desc


def _part_distance(a: np.ndarray, b: np.ndarray, *, shoe: bool = False) -> float:
    """HSV distance in [0,1] (1 = maximally different)."""
    return 1.0 - _part_similarity(a, b, shoe=shoe)


def _part_similarity(a: np.ndarray, b: np.ndarray, *, shoe: bool = False) -> float:
    """Single-part HSV similarity in [0,1]."""
    dh = min(abs(float(a[0]) - float(b[0])), 180.0 - abs(float(a[0]) - float(b[0])))
    ds = abs(float(a[1]) - float(b[1])) / 255.0
    dv = abs(float(a[2]) - float(b[2])) / 255.0
    if shoe:
        # Footwear: black/white/gray differ mainly in V; also chroma (S)
        return float(np.clip(1.0 - (0.20 * (dh / 90.0) + 0.30 * ds + 0.55 * dv), 0.0, 1.0))
    return float(np.clip(1.0 - (0.50 * (dh / 90.0) + 0.25 * ds + 0.25 * dv), 0.0, 1.0))


def student_color_prototype(samples: list[np.ndarray]) -> np.ndarray | None:
    """Mean (N_PARTS, 3) over valid samples; missing parts stay 0."""
    parts = [_as_parts(s) for s in samples]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    stack = np.stack(parts, axis=0)  # S,6,3
    out = np.zeros((N_PARTS, 3), dtype=np.float32)
    for i in range(N_PARTS):
        rows = [stack[s, i] for s in range(len(stack)) if float(np.abs(stack[s, i]).sum()) > 1e-3]
        if rows:
            out[i] = np.mean(rows, axis=0)
    return out


def estimate_part_discriminability(
    prototypes: list[np.ndarray],
    *,
    power: float = 2.4,
    floor: float = 0.03,
) -> np.ndarray:
    """
    Per-part weights that maximize inter-person clothing separation.

    For each body part i, discriminability = mean pairwise HSV distance between
    different students' prototypes. Parts that look the same on everyone
    (e.g. identical jerseys) get near-zero weight; parts that differ (shoes,
    unique pants, …) dominate.

    Returns L1-normalized weights of shape (N_PARTS,).
    """
    protos = [_as_parts(p) for p in prototypes]
    protos = [p for p in protos if p is not None]
    n = len(protos)
    disc = np.zeros(N_PARTS, dtype=np.float32)
    if n < 2:
        disc[:] = 1.0 / N_PARTS
        return disc

    for i in range(N_PARTS):
        dists: list[float] = []
        for a in range(n):
            if float(np.abs(protos[a][i]).sum()) < 1e-3:
                continue
            for b in range(a + 1, n):
                if float(np.abs(protos[b][i]).sum()) < 1e-3:
                    continue
                dists.append(_part_distance(protos[a][i], protos[b][i], shoe=(i == SHOE_IDX)))
        disc[i] = float(np.mean(dists)) if dists else 0.0

    # Emphasize high-separation parts (power > 1); keep small floor so missing
    # disc doesn't zero out a channel entirely when only 2 people enrolled.
    disc = np.power(np.maximum(disc, 0.0), power)
    disc = disc + float(floor) * float(np.max(disc) if float(np.max(disc)) > 1e-8 else 1.0)
    s = float(disc.sum())
    if s < 1e-8:
        disc[:] = 1.0 / N_PARTS
    else:
        disc = disc / s
    return disc.astype(np.float32)


@dataclass
class GalleryColorPrior:
    """
    Session-level clothing prior: part weights that maximize gallery separation,
    plus per-student uniqueness (how distinctive each student's parts are).
    """

    part_weights: np.ndarray  # (6,)
    prototypes: dict[str, np.ndarray]  # sid -> (6,3)
    uniqueness: dict[str, np.ndarray]  # sid -> (6,) relative uniqueness

    def weights_for(self, student_id: str | None = None) -> np.ndarray:
        """
        Effective weights for matching against ``student_id``.

        Combines gallery-wide discriminability with that student's unique parts
        (e.g. only this kid has white shoes → boost shoe when scoring them).
        """
        w = self.part_weights.astype(np.float32).copy()
        blend = float(getattr(self, "_uniqueness_blend", 0.72))
        if student_id and student_id in self.uniqueness:
            u = self.uniqueness[student_id]
            # Blend: keep global structure, amplify this person's distinctive dims
            w = w * ((1.0 - blend) + blend * u)
            s = float(w.sum())
            if s > 1e-8:
                w = w / s
        return w


def relative_color_scores(
    abs_sims: dict[str, float],
    *,
    temperature: float = 0.12,
) -> dict[str, float]:
    """
    Convert absolute clothing similarities into relative [0,1] scores.

    When every gallery entry looks similar (abs sims all ~0.7), absolute
    similarity cannot separate IDs. Scoring by margin vs the runner-up
    restores discriminability: only the best match gets a high score.
    """
    if not abs_sims:
        return {}
    if len(abs_sims) == 1:
        sid, cs = next(iter(abs_sims.items()))
        return {sid: float(np.clip(cs, 0.0, 1.0))}

    items = sorted(abs_sims.items(), key=lambda kv: kv[1], reverse=True)
    best_sid, best_cs = items[0]
    second_cs = float(items[1][1])
    temp = max(1e-3, float(temperature))
    out: dict[str, float] = {}
    for sid, cs in abs_sims.items():
        # Soft margin vs best competitor (for the winner: vs 2nd; else vs best)
        rival = second_cs if sid == best_sid else best_cs
        margin = float(cs) - float(rival)
        # Map margin → (0,1); positive margin → high, negative → low
        rel = 1.0 / (1.0 + np.exp(-margin / temp))
        # Keep a little absolute signal so empty/garbage queries stay low
        out[sid] = float(np.clip(0.70 * rel + 0.30 * float(cs), 0.0, 1.0))
    return out


def build_gallery_color_prior(
    student_colors: dict[str, list[np.ndarray]],
    *,
    power: float | None = None,
    uniqueness_blend: float | None = None,
) -> GalleryColorPrior:
    """Build discriminative part weights from enrolled color samples."""
    from src.config import load_yaml

    idcfg = load_yaml("cameras.yaml").get("identity") or {}
    if power is None:
        power = float(idcfg.get("clothing_color_disc_power", 2.4))
    if uniqueness_blend is None:
        uniqueness_blend = float(idcfg.get("clothing_color_uniqueness_blend", 0.72))
    uniqueness_blend = float(np.clip(uniqueness_blend, 0.0, 0.95))

    prototypes: dict[str, np.ndarray] = {}
    for sid, samples in student_colors.items():
        proto = student_color_prototype(samples)
        if proto is not None:
            prototypes[sid] = proto

    proto_list = list(prototypes.values())
    part_w = estimate_part_discriminability(proto_list, power=float(power))

    uniqueness: dict[str, np.ndarray] = {}
    sids = list(prototypes.keys())
    for sid in sids:
        u = np.zeros(N_PARTS, dtype=np.float32)
        others = [prototypes[o] for o in sids if o != sid]
        if not others:
            u[:] = 1.0 / N_PARTS
        else:
            for i in range(N_PARTS):
                if float(np.abs(prototypes[sid][i]).sum()) < 1e-3:
                    continue
                ds = [
                    _part_distance(prototypes[sid][i], o[i], shoe=(i == SHOE_IDX))
                    for o in others
                    if float(np.abs(o[i]).sum()) > 1e-3
                ]
                u[i] = float(np.mean(ds)) if ds else 0.0
            # Softmax-sharpen uniqueness so only truly distinctive parts dominate
            mx = float(np.max(u)) if float(np.max(u)) > 1e-8 else 1.0
            u = u / mx
            # Square to further separate near-zero vs high uniqueness dims
            u = np.square(u).astype(np.float32)
            umx = float(np.max(u)) if float(np.max(u)) > 1e-8 else 1.0
            u = u / umx
        uniqueness[sid] = u

    prior = GalleryColorPrior(
        part_weights=part_w,
        prototypes=prototypes,
        uniqueness=uniqueness,
    )
    # Stash blend for weights_for (attribute not in dataclass fields → set dynamically)
    prior._uniqueness_blend = uniqueness_blend  # type: ignore[attr-defined]
    return prior


def clothing_color_match_info(
    a: np.ndarray | None,
    b: np.ndarray | None,
    *,
    part_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Weighted clothing match.

    ``part_weights`` should come from ``GalleryColorPrior`` (maximize inter-person
    separation). If omitted, falls back to equal weights over observed parts.
    """
    aa = _as_parts(a)
    bb = _as_parts(b)
    empty = {
        "similarity": 0.0,
        "cloth_sim": 0.0,
        "shoe_sim": None,
        "shoe_decisive": False,
        "shoe_available": False,
        "part_sims": None,
        "part_weights_used": None,
    }
    if aa is None or bb is None:
        return empty

    part_sims = np.full(N_PARTS, np.nan, dtype=np.float32)
    for i in range(N_PARTS):
        if float(np.abs(aa[i]).sum()) < 1e-3 or float(np.abs(bb[i]).sum()) < 1e-3:
            continue
        part_sims[i] = _part_similarity(aa[i], bb[i], shoe=(i == SHOE_IDX))

    if part_weights is None:
        weights = np.ones(N_PARTS, dtype=np.float32)
    else:
        weights = np.asarray(part_weights, dtype=np.float32).reshape(-1)
        if weights.size != N_PARTS:
            weights = np.ones(N_PARTS, dtype=np.float32)

    cloth_vals = [float(part_sims[i]) for i in CLOTH_IDX if not np.isnan(part_sims[i])]
    cloth_ws = [float(weights[i]) for i in CLOTH_IDX if not np.isnan(part_sims[i])]
    cloth_sim = (
        float(np.average(cloth_vals, weights=cloth_ws)) if cloth_vals and sum(cloth_ws) > 1e-8
        else (float(np.mean(cloth_vals)) if cloth_vals else 0.0)
    )
    shoe_available = not np.isnan(part_sims[SHOE_IDX])
    shoe_sim = float(part_sims[SHOE_IDX]) if shoe_available else None

    # shoe_decisive: shoes carry most of the active weight AND disagree vs cloth
    shoe_w_frac = 0.0
    valid_w = 0.0
    for i in range(N_PARTS):
        if np.isnan(part_sims[i]):
            continue
        valid_w += float(weights[i])
        if i == SHOE_IDX:
            shoe_w_frac = float(weights[i])
    if valid_w > 1e-8:
        shoe_w_frac /= valid_w
    shoe_decisive = bool(
        shoe_available
        and shoe_sim is not None
        and shoe_w_frac >= 0.28
        and cloth_sim >= 0.78
        and (cloth_sim - shoe_sim) >= 0.08
    )

    sims = []
    wsum = 0.0
    used_w = np.zeros(N_PARTS, dtype=np.float32)
    for i in range(N_PARTS):
        if np.isnan(part_sims[i]):
            continue
        wi = float(weights[i])
        sims.append(wi * float(part_sims[i]))
        wsum += wi
        used_w[i] = wi
    similarity = float(np.clip(sum(sims) / wsum, 0.0, 1.0)) if sims and wsum > 1e-6 else 0.0

    return {
        "similarity": similarity,
        "cloth_sim": cloth_sim,
        "shoe_sim": shoe_sim,
        "shoe_decisive": shoe_decisive,
        "shoe_available": shoe_available,
        "part_sims": [None if np.isnan(part_sims[i]) else round(float(part_sims[i]), 3) for i in range(N_PARTS)],
        "part_weights_used": [round(float(used_w[i]), 3) for i in range(N_PARTS)],
    }


def clothing_color_similarity(
    a: np.ndarray | None,
    b: np.ndarray | None,
    *,
    part_weights: np.ndarray | None = None,
) -> float:
    """Similarity in [0,1]; optional discriminative ``part_weights``."""
    return float(clothing_color_match_info(a, b, part_weights=part_weights)["similarity"])


def desc_to_list(desc: np.ndarray) -> list[list[float]]:
    return np.asarray(desc, dtype=np.float32).reshape(N_PARTS, 3).tolist()


def desc_from_any(obj: Any) -> np.ndarray | None:
    return _as_parts(obj if not isinstance(obj, list) else np.asarray(obj, dtype=np.float32))
