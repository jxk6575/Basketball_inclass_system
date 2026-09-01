"""Solve per-camera intrinsics (K, distortion) + extrinsics from court landmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.calibration.court_model import landmark_xyz, load_court_model


def default_intrinsics(width: int, height: int, fov_deg: float = 70.0) -> dict[str, Any]:
    """Approximate pinhole K when court-based intrinsic estimation is weak."""
    fx = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return {
        "width": width,
        "height": height,
        "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "source": "approx_fov",
        "fov_deg": fov_deg,
    }


def _K_D(intr: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(intr["camera_matrix"], dtype=np.float64)
    D = np.asarray(intr.get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=np.float64).reshape(-1)
    return K, D


def _pack_obs(
    model: dict[str, Any],
    observations: dict[str, list[float]],
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    ids = [pid for pid, uv in observations.items() if uv is not None and len(uv) >= 2]
    if not ids:
        return None
    obj = np.stack([landmark_xyz(model, pid) for pid in ids], axis=0).astype(np.float64)
    img = np.asarray([observations[pid][:2] for pid in ids], dtype=np.float64)
    return ids, obj, img


def camera_center_world(R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """World-frame camera center. OpenCV: X_cam = R @ X_world + t → C = -Rᵀ t."""
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return (-R.T @ t).reshape(3)


def estimate_intrinsics_planar(
    obj: np.ndarray,
    img: np.ndarray,
    image_size: tuple[int, int],
    *,
    fix_principal: bool = False,
) -> dict[str, Any] | None:
    """
    Estimate K + radial distortion from a single planar view (court ground plane).

    Needs ≥6 well-spread points for a usable estimate. Returns None if unstable.
    """
    n = len(obj)
    if n < 6:
        return None
    w, h = image_size
    obj_cv = obj.reshape(1, -1, 3).astype(np.float32)
    img_cv = img.reshape(1, -1, 2).astype(np.float32)

    # Seed with FOV approx
    seed = default_intrinsics(w, h)
    K0 = np.asarray(seed["camera_matrix"], dtype=np.float64)
    D0 = np.zeros(5, dtype=np.float64)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    if fix_principal:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    try:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_cv, img_cv, (w, h), K0, D0, flags=flags,
        )
    except cv2.error:
        return None

    D = np.asarray(D, dtype=np.float64).reshape(-1)
    # Sanity: focal length and mild distortion (planar 6–7 pts often overfit k1/k2)
    fx = float(K[0, 0])
    if not (0.3 * w < fx < 5.0 * w):
        return None
    if abs(float(D[0])) > 0.5 or abs(float(D[1])) > 0.5:
        return None
    if float(rms) > 5.0:
        return None

    return {
        "width": w,
        "height": h,
        "camera_matrix": K.tolist(),
        "dist_coeffs": [float(x) for x in D.tolist()[:5]],
        "source": "court_planar_calib",
        "rms": float(rms),
        "n_points": n,
    }


def _flip_image_y(img: np.ndarray, height: int) -> np.ndarray:
    """Flip vertical image coordinate: v' = H - 1 - v (plane-chirality fix)."""
    out = np.asarray(img, dtype=np.float64).copy()
    out = out.reshape(-1, 2)
    out[:, 1] = float(height) - 1.0 - out[:, 1]
    return out


def _negate_fy(intr: dict[str, Any]) -> dict[str, Any]:
    """
    Negate fy so a pose solved on v-flipped coords projects correctly on
    original images via standard cv2.projectPoints (keeps C_z > 0).
    """
    out = dict(intr)
    K = np.asarray(out["camera_matrix"], dtype=np.float64).copy()
    K[1, 1] = -abs(float(K[1, 1]))
    out["camera_matrix"] = K.tolist()
    out["fy_sign"] = -1
    out["chirality_fix"] = "flip_v_solve_neg_fy"
    return out


def solve_camera_pnp(
    model: dict[str, Any],
    observations: dict[str, list[float]],
    intrinsics: dict[str, Any],
    min_points: int = 4,
    *,
    image_height: int | None = None,
    enforce_above_ground: bool = True,
) -> dict[str, Any]:
    """
    Solve extrinsics for one camera from {point_id: [u, v]} observations.

    Court landmarks are coplanar (z=0). OpenCV PnP then has a chirality ambiguity:
    the minimum-reprojection pose often places the camera *below* the ground.
    We resolve it by solving on vertically flipped image coordinates and exporting
    K with fy < 0 so standard projectPoints matches the original annotations, with
    camera_center_world.z > 0.
    """
    packed = _pack_obs(model, observations)
    if packed is None or len(packed[0]) < min_points:
        ids = list(observations.keys()) if packed is None else packed[0]
        return {
            "status": "insufficient_points",
            "n_points": len(ids),
            "point_ids": ids,
            "message": f"Need ≥{min_points} landmarks, got {len(ids)}",
        }

    ids, obj, img = packed
    h = int(image_height or intrinsics.get("height") or 1080)
    K_base, D = _K_D(intrinsics)
    D_col = D.reshape(-1, 1)

    # Solve in flipped-v space (selects the above-ground sheet of the plane ambiguity)
    img_solve = _flip_image_y(img, h) if enforce_above_ground else img.copy()
    K_solve = K_base.copy()
    K_solve[1, 1] = abs(float(K_solve[1, 1]))  # always +fy while solving

    def _score_pose(rv, tv):
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K_solve, D_col)
        e = float(np.linalg.norm(proj.reshape(-1, 2) - img_solve, axis=1).mean())
        R_i, _ = cv2.Rodrigues(rv)
        C_i = camera_center_world(R_i, tv)
        pen = 0.0
        # Hard preference: above ground
        if C_i[2] < 0.3:
            pen += 1e3 + abs(float(C_i[2])) * 100.0
        if C_i[2] > 25.0:
            pen += float(C_i[2])
        centroid = obj.mean(axis=0)
        forward_world = R_i.T @ np.array([0.0, 0.0, 1.0])
        to_court = centroid - C_i
        nrm = float(np.linalg.norm(to_court))
        if nrm > 1e-6:
            to_court = to_court / nrm
            align = float(np.dot(forward_world, to_court))
            if align < 0.2:
                pen += 30.0 * (0.2 - align)
        # Points should be in front of camera
        Xc = (R_i @ obj.T + np.asarray(tv, dtype=np.float64).reshape(3, 1)).T
        if float((Xc[:, 2] > 0).mean()) < 0.9:
            pen += 200.0
        return (
            e + pen,
            e,
            R_i,
            C_i,
            np.asarray(rv, dtype=np.float64).reshape(3),
            np.asarray(tv, dtype=np.float64).reshape(3),
        )

    candidates: list[tuple] = []
    for flags in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_SQPNP):
        try:
            ok_g, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                obj.reshape(-1, 1, 3),
                img_solve.reshape(-1, 1, 2),
                K_solve, D_col,
                flags=flags,
            )
            if ok_g and rvecs:
                for rv, tv in zip(rvecs, tvecs):
                    candidates.append(_score_pose(rv, tv))
        except cv2.error:
            continue

    if not candidates:
        return {"status": "pnp_failed", "n_points": len(ids), "point_ids": ids}

    candidates.sort(key=lambda x: x[0])
    _, e_flip, R, center, rvec, tvec = candidates[0]
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    if len(ids) >= 4:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj.reshape(-1, 1, 3), img_solve.reshape(-1, 1, 2),
                K_solve, D_col, rvec, tvec,
            )
            R, _ = cv2.Rodrigues(rvec)
            center = camera_center_world(R, tvec)
        except cv2.error:
            pass

    # If still below ground, force the best above-ground candidate
    if enforce_above_ground and center[2] < 0.3:
        above = [c for c in candidates if c[3][2] >= 0.3]
        if above:
            above.sort(key=lambda x: x[0])
            _, e_flip, R, center, rvec, tvec = above[0]
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    if enforce_above_ground and center[2] < 0.3:
        return {
            "status": "below_ground",
            "n_points": len(ids),
            "point_ids": ids,
            "camera_center_world": center.tolist(),
            "message": "Could not find a pose with Z>0; check landmark left/right labels",
        }

    # Export intrinsics with fy < 0 so projectPoints matches ORIGINAL (u,v)
    if enforce_above_ground:
        intr_out = _negate_fy({**intrinsics, "camera_matrix": K_base.tolist()})
    else:
        intr_out = {**intrinsics, "camera_matrix": K_base.tolist()}

    K_out, D_out = _K_D(intr_out)
    D_out_col = D_out.reshape(-1, 1)
    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K_out, D_out_col)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img, axis=1)

    return {
        "status": "ok",
        "n_points": len(ids),
        "point_ids": ids,
        "rvec": rvec.reshape(-1).tolist(),
        "tvec": tvec.reshape(-1).tolist(),
        "rotation_matrix": R.tolist(),
        "camera_center_world": center.tolist(),
        "reproj_error_px": {
            "mean": float(err.mean()),
            "max": float(err.max()),
            "per_point": {pid: float(e) for pid, e in zip(ids, err)},
        },
        "intrinsics": intr_out,
        "solve_space": "image_v_flipped" if enforce_above_ground else "image_native",
    }


def _share_intrinsics(candidates: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    """Average K/D from successful planar calibrations (same camera model assumption)."""
    usable = [c for c in candidates if c and c.get("source") == "court_planar_calib"]
    if not usable:
        return default_intrinsics(width, height)
    Ks = np.stack([np.asarray(c["camera_matrix"], dtype=np.float64) for c in usable], axis=0)
    Ds = np.stack([
        np.asarray(c.get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=np.float64)[:5]
        for c in usable
    ], axis=0)
    K = Ks.mean(axis=0)
    D = Ds.mean(axis=0)
    return {
        "width": width,
        "height": height,
        "camera_matrix": K.tolist(),
        "dist_coeffs": [float(x) for x in D.tolist()],
        "source": "shared_from_planar_calib",
        "donors": len(usable),
    }


def _softplus(v: float | np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.log1p(np.exp(np.clip(v, -20.0, 20.0)))


def refine_poses_with_priors(
    results: dict[str, Any],
    annotations: dict[str, Any],
    model: dict[str, Any],
    priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Jointly refine extrinsics so that:
      - cam_01 / cam_02 stay outside the sidelines
      - cam_03 XY stays near the measured baseline-3PT offset
      - all cameras share one height Z
    """
    from scipy.optimize import least_squares

    priors = priors if priors is not None else (model.get("camera_pose_priors") or {})
    if not priors or not priors.get("enforce_shared_height", True):
        return results

    cams = [c for c, r in (results.get("cameras") or {}).items() if r.get("status") == "ok"]
    need = [c for c in ("cam_01", "cam_02", "cam_03") if c in cams]
    if len(need) < 2:
        return results

    points = annotations.get("points") or {}
    packs: dict[str, tuple] = {}
    intrs: dict[str, dict] = {}
    r0: dict[str, np.ndarray] = {}
    for c in need:
        packed = _pack_obs(model, points.get(c) or {})
        if packed is None:
            return results
        packs[c] = packed
        intrs[c] = results["cameras"][c]["intrinsics"]
        r0[c] = np.asarray(results["cameras"][c]["rvec"], dtype=np.float64)

    outside = float(priors.get("sideline_outside_m") or 0.05)
    x_left = -7.5 - outside
    x_right = 7.5 + outside
    p03 = priors.get("cam_03") or {}
    xy_t = np.asarray(p03.get("xy_target_m") or [-6.4, -0.15], dtype=np.float64)
    slack = float(p03.get("xy_slack_m") or 0.25)
    z_lo, z_hi = [float(v) for v in (priors.get("height_range_m") or [1.5, 4.0])]

    # Layout: rv01(3), Cy01, out01, rv02(3), Cy02, out02, rv03(3), dxy03(2), Z
    # cam01 Cx = x_left - softplus(out)
    # cam02 Cx = x_right + softplus(out)
    # cam03 Cxy = xy_t + slack * tanh(dxy)
    n = 5 + 5 + 5 + 1  # 16

    def unpack(x: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        Z = float(x[-1])
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if "cam_01" in need:
            out["cam_01"] = (
                x[0:3],
                np.array([x_left - float(_softplus(x[4])), float(x[3]), Z], dtype=np.float64),
            )
        if "cam_02" in need:
            out["cam_02"] = (
                x[5:8],
                np.array([x_right + float(_softplus(x[9])), float(x[8]), Z], dtype=np.float64),
            )
        if "cam_03" in need:
            dxy = x[13:15]
            out["cam_03"] = (
                x[10:13],
                np.array([
                    xy_t[0] + slack * float(np.tanh(dxy[0])),
                    xy_t[1] + slack * float(np.tanh(dxy[1])),
                    Z,
                ], dtype=np.float64),
            )
        return out

    def residuals(x: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for c, (rv, C) in unpack(x).items():
            _ids, obj, img = packs[c]
            R, _ = cv2.Rodrigues(rv.reshape(3, 1))
            t = (-R @ C).reshape(3, 1)
            K, D = _K_D(intrs[c])
            proj, _ = cv2.projectPoints(
                obj.reshape(-1, 1, 3), rv.reshape(3, 1), t, K, D.reshape(-1, 1),
            )
            chunks.append((proj.reshape(-1, 2) - img).ravel())
        return np.concatenate(chunks)

    C0 = {
        c: np.asarray(results["cameras"][c]["camera_center_world"], dtype=np.float64)
        for c in need
    }
    x0 = np.zeros(n, dtype=np.float64)
    x0[0:3] = r0.get("cam_01", np.zeros(3))
    x0[3] = float(np.clip(C0.get("cam_01", [0, 5, 2])[1], 0.5, 12.0))
    x0[4] = 0.4
    x0[5:8] = r0.get("cam_02", np.zeros(3))
    x0[8] = float(np.clip(C0.get("cam_02", [0, 5, 2])[1], 0.5, 12.0))
    x0[9] = 0.4
    x0[10:13] = r0.get("cam_03", np.zeros(3))
    x0[13:15] = 0.0
    z0 = float(np.median([C0[c][2] for c in need]))
    x0[-1] = float(np.clip(max(z0, 2.0), z_lo, z_hi))

    lo = np.array(
        [-np.pi] * 3 + [0.5, 0.0] + [-np.pi] * 3 + [0.5, 0.0] + [-np.pi] * 3 + [-2.0, -2.0] + [z_lo],
        dtype=np.float64,
    )
    hi = np.array(
        [np.pi] * 3 + [12.0, 5.0] + [np.pi] * 3 + [12.0, 5.0] + [np.pi] * 3 + [2.0, 2.0] + [z_hi],
        dtype=np.float64,
    )

    fit = least_squares(
        residuals, x0, method="trf", loss="soft_l1", f_scale=20.0,
        bounds=(lo, hi), max_nfev=500,
    )
    parsed = unpack(fit.x)
    shared_z = float(fit.x[-1])

    for c, (rv, C) in parsed.items():
        _ids, obj, img = packs[c]
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        t = (-R @ C).reshape(3)
        K, D = _K_D(intrs[c])
        proj, _ = cv2.projectPoints(
            obj.reshape(-1, 1, 3), rv.reshape(3, 1), t.reshape(3, 1), K, D.reshape(-1, 1),
        )
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        res = results["cameras"][c]
        res["rvec"] = rv.reshape(-1).tolist()
        res["tvec"] = t.reshape(-1).tolist()
        res["rotation_matrix"] = R.tolist()
        res["camera_center_world"] = C.tolist()
        res["camera_center_world_m"] = {
            "x": float(C[0]), "y": float(C[1]), "z": float(C[2]),
            "description": "先验约束后：场外/cam_03 实测偏移 + 三机同高",
        }
        res["reproj_error_px"] = {
            "mean": float(err.mean()),
            "max": float(err.max()),
            "per_point": {pid: float(e) for pid, e in zip(_ids, err)},
        }
        res["pose_prior"] = {
            "shared_height_m": shared_z,
            "applied": True,
            "optimize_cost": float(fit.cost),
        }

    results["method"] = (results.get("method") or "") + "+pose_priors_sharedZ"
    results.setdefault("summary", {})
    results["summary"]["shared_height_m"] = shared_z
    results["summary"]["camera_centers_world"] = {
        c: results["cameras"][c].get("camera_center_world")
        for c in (results.get("summary") or {}).get("ok_cameras") or need
        if results["cameras"].get(c, {}).get("status") == "ok"
    }
    results["summary"]["pose_priors"] = {
        "cam_03_xy_target_m": xy_t.tolist(),
        "sideline_outside_m": outside,
        "shared_height_m": shared_z,
    }
    return results


def solve_all_cameras(
    annotations: dict[str, Any],
    model: dict[str, Any] | None = None,
    intrinsics_by_cam: dict[str, dict] | None = None,
    *,
    estimate_distortion: bool = True,
    share_intrinsics_when_weak: bool = True,
) -> dict[str, Any]:
    """
    Per camera:
      1) Try planar calibrateCamera → K + radial distortion (needs ≥6 points)
      2) Weak cameras (e.g. cam_03 with 4 pts) reuse shared K/D from stronger cams
      3) PnP/IPPE → R,t and camera_center_world
    """
    model = model or load_court_model()
    results: dict[str, Any] = {
        "standard": model.get("standard"),
        "axes": model.get("axes"),
        "method": "court_landmarks_pnp_flipV_negFy",
        "cameras": {},
    }
    points = annotations.get("points") or {}
    sizes = annotations.get("image_size") or {}

    # Pass 1: estimate intrinsics where possible (on v-flipped coords for consistency)
    estimated: dict[str, dict[str, Any] | None] = {}
    for cam_id, obs in points.items():
        size = sizes.get(cam_id) or [1920, 1080]
        w, h = int(size[0]), int(size[1])
        if intrinsics_by_cam and cam_id in intrinsics_by_cam:
            estimated[cam_id] = intrinsics_by_cam[cam_id]
            continue
        packed = _pack_obs(model, obs)
        if packed is None:
            estimated[cam_id] = default_intrinsics(w, h)
            continue
        ids, obj, img = packed
        if estimate_distortion and len(ids) >= 6:
            estimated[cam_id] = estimate_intrinsics_planar(
                obj, _flip_image_y(img, h), (w, h),
            )
        else:
            estimated[cam_id] = None

    donors = [estimated[c] for c in estimated if estimated[c] and estimated[c].get("source") == "court_planar_calib"]

    # Pass 2: solve extrinsics (enforces camera_center.z > 0)
    for cam_id, obs in points.items():
        size = sizes.get(cam_id) or [1920, 1080]
        w, h = int(size[0]), int(size[1])
        intr = estimated.get(cam_id)
        if intr is None:
            if share_intrinsics_when_weak and donors:
                intr = _share_intrinsics(donors, w, h)
            else:
                intr = default_intrinsics(w, h)

        res = solve_camera_pnp(model, obs, intr, image_height=h, enforce_above_ground=True)
        if res.get("status") == "ok":
            K, D = _K_D(res["intrinsics"])
            K_und = K.copy()
            K_und[1, 1] = abs(float(K_und[1, 1]))  # undistort APIs dislike negative fy
            try:
                newK, roi = cv2.getOptimalNewCameraMatrix(K_und, D, (w, h), alpha=0.0)
            except cv2.error:
                newK, roi = K_und, (0, 0, w, h)
            res["undistort"] = {
                "new_camera_matrix": newK.tolist(),
                "roi": [int(x) for x in roi],
                "note": "fy stored negative for chirality; undistort uses |fy|",
            }
            c = res["camera_center_world"]
            res["camera_center_world_m"] = {
                "x": c[0], "y": c[1], "z": c[2],
                "description": "相机光心在球场坐标系中的位置（米）；Z 必须 > 0",
            }
        results["cameras"][cam_id] = res

    results = refine_poses_with_priors(results, annotations, model)

    ok_cams = [c for c, r in results["cameras"].items() if r.get("status") == "ok"]
    results["summary"] = {
        "ok_cameras": ok_cams,
        "n_ok": len(ok_cams),
        "ready_for_triangulation": len(ok_cams) >= 2,
        "camera_centers_world": {
            c: results["cameras"][c].get("camera_center_world")
            for c in ok_cams
        },
        "intrinsic_sources": {
            c: (results["cameras"][c].get("intrinsics") or {}).get("source")
            for c in ok_cams
        },
        "shared_height_m": (results.get("summary") or {}).get("shared_height_m"),
        "pose_priors": (results.get("summary") or {}).get("pose_priors"),
    }
    return results


def export_calibration(
    solved: dict[str, Any],
    out_dir: Path,
    annotations: dict[str, Any] | None = None,
) -> Path:
    """Write calibration bundle under data/calibration/..."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cameras.json"
    payload = {
        "version": 2,
        "method": solved.get("method") or "court_landmarks_intrinsics_pnp",
        "solved": solved,
        "annotations": annotations,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    centers = {}
    for cam_id, res in (solved.get("cameras") or {}).items():
        if res.get("status") != "ok":
            continue
        centers[cam_id] = res.get("camera_center_world")
        cam_path = out_dir / f"{cam_id}.json"
        cam_path.write_text(json.dumps({
            "camera_id": cam_id,
            "rotation_matrix": res["rotation_matrix"],
            "tvec": res["tvec"],
            "rvec": res["rvec"],
            "camera_center_world": res.get("camera_center_world"),
            "intrinsics": res["intrinsics"],
            "undistort": res.get("undistort"),
            "reproj_error_px": res["reproj_error_px"],
            "point_ids": res["point_ids"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "camera_centers_world.json").write_text(
        json.dumps({
            "unit": "meter",
            "axes": (solved.get("axes") or {}),
            "centers": centers,
            "note": "C = -R^T t ；原点=进攻端底线中点，+x右，+y向中线，+z上",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_dir / "README.txt").write_text(
        "Court-landmark calibration (intrinsics + distortion + extrinsics).\n"
        "cameras.json          = full bundle\n"
        "cam_XX.json           = K, dist, R, t, camera_center_world\n"
        "camera_centers_world.json = 三相机光心世界坐标（米）\n"
        "Undistort: cv2.undistort(img, K, dist, None, newK)\n",
        encoding="utf-8",
    )
    return path
