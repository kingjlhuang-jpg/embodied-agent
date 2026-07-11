import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos"))

from unitree_paths import resolve_unitree_scene  # noqa: E402


class UnitreeSceneResolutionTest(unittest.TestCase):
    @staticmethod
    def make_scene(root: Path, robot: str = "g1") -> Path:
        scene = root / "unitree_robots" / robot / "scene.xml"
        scene.parent.mkdir(parents=True)
        scene.write_text("<mujoco/>")
        return scene

    def test_explicit_root_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self.make_scene(root)
            self.assertEqual(resolve_unitree_scene("g1", root, anchor=root / "unused"), expected.resolve())

    def test_environment_root_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self.make_scene(root, "go2")
            with patch.dict(os.environ, {"UNITREE_MUJOCO_ROOT": str(root)}):
                self.assertEqual(resolve_unitree_scene("go2", anchor=root / "unused"), expected.resolve())

    def test_sibling_checkout_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor = Path(tmp)
            expected = self.make_scene(anchor / "unitree_mujoco", "h1")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve_unitree_scene("h1", anchor=anchor), expected.resolve())

    def test_missing_checkout_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("unitree_paths.Path.cwd", return_value=Path(tmp) / "cwd"),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "UNITREE_MUJOCO_ROOT"):
                    resolve_unitree_scene("g1", anchor=Path(tmp))

    def test_unknown_robot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Unitree robot"):
            resolve_unitree_scene("unknown")


if __name__ == "__main__":
    unittest.main()
