"""Unit tests for event-based camera sync."""

from __future__ import annotations

from src.cameras.event_sync import (
    SyncEvent,
    apply_offset,
    estimate_offset_from_matches,
    invert_offset,
    match_event_series,
)


def _ev(cam: str, t: float, kind: str = "release") -> SyncEvent:
    return SyncEvent(camera_id=cam, timestamp_ms=t, frame=int(t / 1000 * 30), kind=kind)


def test_match_and_median_offset():
    anchor = [_ev("cam_03", 1000), _ev("cam_03", 5000), _ev("cam_03", 9000)]
    # cam_01 local clock is +350 ms ahead of anchor
    other = [_ev("cam_01", 1350), _ev("cam_01", 5350), _ev("cam_01", 9350)]
    matches = match_event_series(anchor, other, max_match_ms=800)
    assert len(matches) == 3
    off = estimate_offset_from_matches(matches)
    assert off is not None
    assert abs(off - 350.0) < 1e-6
    # common = local - offset
    assert abs(apply_offset(5350, off) - 5000) < 1e-6
    assert abs(invert_offset(5000, off) - 5350) < 1e-6


def test_match_rejects_far_events():
    anchor = [_ev("cam_03", 1000)]
    other = [_ev("cam_01", 5000)]
    matches = match_event_series(anchor, other, max_match_ms=500)
    assert matches == []


def test_one_to_one_matching():
    anchor = [_ev("cam_03", 1000), _ev("cam_03", 1200)]
    other = [_ev("cam_02", 1100)]  # nearer to 1000 and 1200 equally? 100 closer to 1000? |1100-1000|=100, |1100-1200|=100
    matches = match_event_series(anchor, other, max_match_ms=500)
    assert len(matches) == 1  # one-to-one


def test_rim_kind_matching():
    anchor = [_ev("cam_03", 2000, "release"), _ev("cam_03", 6000, "release")]
    rim = [
        SyncEvent(camera_id="cam_04", timestamp_ms=2300, kind="rim_ball"),
        SyncEvent(camera_id="cam_04", timestamp_ms=6400, kind="rim_ball"),
    ]
    matches = match_event_series(anchor, rim, max_match_ms=1000)
    assert len(matches) == 2
    off = estimate_offset_from_matches(matches)
    assert off is not None
    assert abs(off - 350.0) < 1.0  # median of 300 and 400
