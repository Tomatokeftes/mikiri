"""The sessions a player has kept, so a clip can be read against their own earlier ones.

Every comparator the app offers points at the same player over time: the
split-step numbers are a lower bound that only means something against last
month's, and a shot mix is a habit only once it repeats. Until now nothing
kept a session, so the comparison the whole coaching side was built around
could not be made.

A session is kept when the user saves the project. That is the one moment
they have said the clip is worth keeping, and it keeps the store free of the
half-tracked experiments a run-on-every-change rule would collect. The record
is the summaries, not the pose: the tactical summary, the coaching summary the
comparator reads, and enough about the clip to name it. Saving the same
project again replaces its record; deleting one from the store forgets it.

Records are grouped by a free-text player label on the project, empty meaning
the user themselves. The app does not try to recognise who is in the clip:
that is a decision the user makes once, in a text box, and can change.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mikiri.paths import history_file
from mikiri.video import video_id

#: Bumped when a record's shape changes in a way old readers cannot take.
RECORD_VERSION = 1


def player_key(label: str | None) -> str:
    """The label as the store compares it: trimmed and case-folded, empty for the user."""
    return (label or "").strip().casefold()


def clip_key(video_path: str | None, trim_start: int, trim_end: int) -> str | None:
    """What makes two projects the same session: the same file, cut the same way.

    Opening a raw clip again gives it a new project id, and without this a
    second save would count the same match twice. The trim is part of it
    because the same footage cut to one game and to the whole match are two
    different sessions on purpose.
    """
    if not video_path:
        return None
    return f"{video_id(video_path)}:{int(trim_start)}-{int(trim_end)}"


def recorded_at(video_path: str | None, fallback: str) -> str:
    """When the clip was recorded, as far as the file system knows.

    The video file's modification time is the closest thing to a recording
    date a phone clip carries once it has been copied around; a project's
    creation time is the fallback when the file is gone.
    """
    try:
        if video_path:
            stamp = Path(video_path).stat().st_mtime
            return datetime.fromtimestamp(stamp).isoformat(timespec="seconds")
    except OSError:
        pass
    return fallback


def load_history(path: Path | None = None) -> list[dict]:
    """Every kept session, oldest first; an unreadable store reads as empty."""
    p = path or history_file()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    records = data.get("sessions", []) if isinstance(data, dict) else []
    return sorted(
        [r for r in records if isinstance(r, dict) and r.get("project_id")],
        key=lambda r: str(r.get("recorded_at", "")),
    )


def save_history(records: list[dict], path: Path | None = None) -> Path:
    """Write the store; the caller has already decided what belongs in it."""
    p = path or history_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"version": RECORD_VERSION, "sessions": records}, indent=1, default=str),
        encoding="utf-8",
    )
    return p


def _same_session(a: dict, b: dict) -> bool:
    if a.get("project_id") == b.get("project_id"):
        return True
    key = a.get("clip_key")
    return bool(key) and key == b.get("clip_key")


def remember(record: dict, path: Path | None = None) -> list[dict]:
    """Keep ``record``, replacing an earlier record of the same project or the same clip."""
    records = [r for r in load_history(path) if not _same_session(r, record)]
    records.append(record)
    records.sort(key=lambda r: str(r.get("recorded_at", "")))
    save_history(records, path)
    return records


def forget(project_id: str, path: Path | None = None) -> list[dict]:
    """Drop the record of ``project_id``; a missing one is not an error."""
    records = [r for r in load_history(path) if r.get("project_id") != project_id]
    save_history(records, path)
    return records


def sessions_for(records: list[dict], player: str | None) -> list[dict]:
    """The records that belong to ``player``, oldest first."""
    key = player_key(player)
    return [r for r in records if player_key(r.get("player")) == key]


def merge_current(records: list[dict], current: dict | None) -> list[dict]:
    """The kept records with the open session in place of its own kept copy.

    The open project may be newer than what was saved, or never saved at all;
    either way the comparison should include what is on screen, marked as
    such so the reader knows it is not yet kept.
    """
    if current is None:
        return list(records)
    out = [r for r in records if not _same_session(r, current)]
    out.append({**current, "current": True})
    out.sort(key=lambda r: str(r.get("recorded_at", "")))
    return out
