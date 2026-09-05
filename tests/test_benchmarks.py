"""The benchmark manifest, and the guard against tests skipping in silence."""

from __future__ import annotations

import json

import pytest

from mikiri.benchmarks import (
    MANIFEST_NAME,
    any_present,
    clip,
    clip_path,
    clips,
    manifest_path,
    missing_clips,
    search_roots,
)


def test_the_manifest_is_present_and_well_formed():
    """It is committed, unlike the footage, so it must be there and be valid."""
    p = manifest_path()
    assert p is not None and p.name == MANIFEST_NAME
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["clips"], "a manifest with no clips describes nothing"
    ids = [c["id"] for c in data["clips"]]
    assert len(ids) == len(set(ids)), f"duplicate clip ids: {ids}"
    for c in data["clips"]:
        assert c["file"] and c["title"] and c["source"], c
        # Either the clip guards something, or it says plainly that it does not.
        assert c.get("guards") or c.get("note"), f"{c['id']} claims neither a guard nor a caveat"


def test_a_clip_is_looked_up_by_id_and_reports_absence_rather_than_guessing():
    assert clip("lindan_rally") is not None
    assert clip("no_such_clip") is None
    assert clip_path("no_such_clip") is None
    assert search_roots(), "there must be somewhere to look"


def test_the_footage_is_all_there_or_all_absent():
    """The failure this folder exists to prevent.

    Every real-footage test skips when its clip is missing, which is correct on
    a machine that does not have the footage and silent when the footage merely
    moved. A partial set means a path is broken: the suite would go green while
    testing nothing. One clip present is therefore taken as "this machine has
    the footage", and then all of them must be.
    """
    if not any_present():
        pytest.skip("no benchmark footage on this machine")
    missing = missing_clips()
    assert not missing, (
        f"benchmark footage is partially present: {missing} cannot be found under "
        f"{[str(r) for r in search_roots()]}. Tests for those clips are skipping silently."
    )


def test_the_ids_the_suite_depends_on_are_all_there():
    """Renaming an id must fail here rather than skip a slow test in silence.

    ``test_real_footage`` reads its trim, its expected event frames and its
    court calibration from the manifest and falls back to the historical values
    when the entry is missing, so a typo would otherwise run nothing and say
    nothing.
    """
    for needed in ("lindan_rally", "jump_smash_behind"):
        assert clip(needed) is not None, f"the test suite references clip id {needed!r}"


def test_the_calibrated_clips_carry_a_usable_calibration():
    """A court in the manifest must be complete, or a test will fail confusingly."""
    for c in clips():
        court = c.get("court")
        if not court:
            continue
        img, world = court["image_points"], court["world_points"]
        assert len(img) == len(world) >= 4, c["id"]
        assert all(len(p) == 2 for p in img + world), c["id"]
        assert court.get("preset"), c["id"]
        assert court.get("note"), f"{c['id']}: say where these points came from"
