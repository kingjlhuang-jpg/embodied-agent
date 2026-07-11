"""Locate robot scenes from an external unitree_mujoco checkout."""

from __future__ import annotations

import os
from pathlib import Path


SUPPORTED_ROBOTS = {"go2", "g1", "h1", "h1_2", "b2", "b2w", "go2w"}


def _candidate_roots(explicit_root: str | Path | None, anchor: Path) -> list[Path]:
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())

    env_root = os.environ.get("UNITREE_MUJOCO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            anchor / "unitree_mujoco",
            Path.cwd(),
            Path.cwd().parent / "unitree_mujoco",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_unitree_scene(
    robot: str,
    explicit_root: str | Path | None = None,
    *,
    anchor: Path | None = None,
) -> Path:
    """Return the official MuJoCo scene for ``robot`` or raise a useful error."""

    if robot not in SUPPORTED_ROBOTS:
        supported = ", ".join(sorted(SUPPORTED_ROBOTS))
        raise ValueError(f"Unsupported Unitree robot {robot!r}. Choose one of: {supported}")

    if anchor is None:
        anchor = Path(__file__).resolve().parents[2]

    roots = _candidate_roots(explicit_root, anchor)
    for root in roots:
        scene = root / "unitree_robots" / robot / "scene.xml"
        if scene.is_file():
            return scene

    checked = "\n  - ".join(str(root) for root in roots)
    raise FileNotFoundError(
        "Could not find the official unitree_mujoco checkout. "
        "Clone https://github.com/unitreerobotics/unitree_mujoco and pass "
        "--unitree-mujoco-root or set UNITREE_MUJOCO_ROOT.\n"
        f"Checked:\n  - {checked}"
    )
