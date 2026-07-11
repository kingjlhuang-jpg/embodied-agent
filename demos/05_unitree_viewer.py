"""Open any supported official Unitree MuJoCo model in the native viewer."""

from __future__ import annotations

import argparse

from unitree_paths import SUPPORTED_ROBOTS, resolve_unitree_scene


INITIAL_HEIGHTS = {
    "go2": 0.35,
    "g1": 0.75,
    "h1": 0.98,
    "h1_2": 0.98,
    "b2": 0.5,
    "b2w": 0.5,
    "go2w": 0.35,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=sorted(SUPPORTED_ROBOTS), nargs="?", default="g1")
    parser.add_argument(
        "--unitree-mujoco-root",
        help="Path to an official unitree_mujoco checkout (or set UNITREE_MUJOCO_ROOT)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load the model and print its dimensions without opening the viewer",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import mujoco
    import mujoco.viewer

    scene = resolve_unitree_scene(args.robot, args.unitree_mujoco_root)
    model = mujoco.MjModel.from_xml_path(str(scene))
    print(
        f"Unitree {args.robot.upper()}: {scene} "
        f"(nq={model.nq}, nu={model.nu}, njnt={model.njnt})"
    )
    if args.validate_only:
        return

    data = mujoco.MjData(model)
    if model.nq >= 3:
        data.qpos[2] = INITIAL_HEIGHTS.get(args.robot, 0.5)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
