import json
import tempfile
import unittest
from pathlib import Path

from verify import check_validation_metadata
from src.datamodules.metadata_paths import (
    dev_set_root_from_metadata_list,
    rebase_dev_set_paths,
)


class MetadataPathTests(unittest.TestCase):
    def test_rebases_nested_released_paths_without_changing_relative_sources(self):
        original = {
            "config": {
                "foreground_dir": "data/dev_set/sound_event/valid",
                "source_file": "Percussion/example.wav",
            },
            "room": [{"sofa_path": "/old/data/dev_set/room_ir/valid/room.sofa"}],
        }
        rebased = rebase_dev_set_paths(original, Path("/mounted/dev_set"))

        self.assertEqual(
            rebased["config"]["foreground_dir"],
            "/mounted/dev_set/sound_event/valid",
        )
        self.assertEqual(rebased["config"]["source_file"], "Percussion/example.wav")
        self.assertEqual(
            rebased["room"][0]["sofa_path"],
            "/mounted/dev_set/room_ir/valid/room.sofa",
        )
        self.assertEqual(original["config"]["foreground_dir"], "data/dev_set/sound_event/valid")

    def test_infers_mount_root_from_validation_index(self):
        self.assertEqual(
            dev_set_root_from_metadata_list("/mounted/dev_set/metadata/valid.json"),
            Path("/mounted/dev_set"),
        )

    def test_verification_checks_paths_inside_validation_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_dir = root / "metadata/valid"
            metadata_dir.mkdir(parents=True)
            (root / "metadata/valid.json").write_text(
                json.dumps([{"metadata_path": "valid/scene.json"}]), encoding="utf-8"
            )
            metadata = {
                "config": {
                    "foreground_dir": "data/dev_set/sound_event/valid",
                    "background_dir": None,
                    "interference_dir": None,
                    "room_config": {
                        "args": {"path": "data/dev_set/room_ir/valid/room.sofa"}
                    },
                },
                "room": {
                    "args": {
                        "metadata": {
                            "sofa_path": "data/dev_set/room_ir/valid/room.sofa"
                        }
                    }
                },
            }
            (metadata_dir / "scene.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            self.assertEqual(
                check_validation_metadata(root),
                [root / "sound_event/valid", root / "room_ir/valid/room.sofa"],
            )
            (root / "sound_event/valid").mkdir(parents=True)
            (root / "room_ir/valid").mkdir(parents=True)
            (root / "room_ir/valid/room.sofa").touch()
            self.assertEqual(check_validation_metadata(root), [])


if __name__ == "__main__":
    unittest.main()
