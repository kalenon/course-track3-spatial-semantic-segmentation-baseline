"""Check metadata caching without requiring full audio/room assets."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.modules.spatial_audio_synthesizer import utils


class FileMetadataCacheTests(unittest.TestCase):
    def setUp(self):
        utils._cached_files_list.cache_clear()
        utils._cached_labels.cache_clear()
        utils.get_audio_duration.cache_clear()

    def test_file_listing_is_cached_but_caller_can_mutate_returned_list(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "event.wav").touch()
            glob_file_list = utils.glob.glob
            with mock.patch.object(utils.glob, "glob", wraps=glob_file_list) as glob_mock:
                first = utils.get_files_list(folder, ".wav")
                first.clear()
                second = utils.get_files_list(folder, ".wav")

            self.assertEqual(second, [str(folder / "event.wav")])
            self.assertEqual(glob_mock.call_count, 1)

    def test_labels_are_cached_and_keep_directory_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "label_a").mkdir()
            (folder / "label_b").mkdir()
            list_directory = utils.os.listdir
            with mock.patch.object(utils.os, "listdir", wraps=list_directory) as list_mock:
                first = utils.get_labels(folder)
                first.clear()
                second = utils.get_labels(folder)

            self.assertEqual(second, ["label_a", "label_b"])
            self.assertEqual(list_mock.call_count, 1)

    def test_duration_reads_header_once_for_reused_source(self):
        duration_mock = mock.Mock(return_value=3.5)
        with mock.patch.object(utils, "librosa",
                               types.SimpleNamespace(get_duration=duration_mock)):
            self.assertEqual(utils.get_audio_duration("/data/source.wav"), 3.5)
            self.assertEqual(utils.get_audio_duration("/data/source.wav"), 3.5)
        duration_mock.assert_called_once_with(path="/data/source.wav")


class SofaMetadataCacheTests(unittest.TestCase):
    def test_room_selection_positions_and_synthesis_reuse_static_metadata(self):
        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        opened = []

        class FakeDatabase:
            def __init__(self):
                self.Data = types.SimpleNamespace(
                    IR=types.SimpleNamespace(
                        dimensions=lambda: ("M", "R", "N"),
                        shape=(2, 4, 3),
                        get_values=lambda **kwargs: np.ones((1, 4, 3)),
                    ),
                    SamplingRate=types.SimpleNamespace(get_values=lambda: [48000]),
                )
                self.Source = types.SimpleNamespace(
                    Position=types.SimpleNamespace(get_values=lambda **kwargs: positions.copy())
                )
                self.closed = False

            def close(self):
                self.closed = True

        def open_database(*args, **kwargs):
            database = FakeDatabase()
            opened.append(database)
            return database

        fake_sofa = types.ModuleType("sofa")
        fake_sofa.Database = types.SimpleNamespace(open=open_database)
        path = Path(__file__).resolve().parents[1] / "src/modules/spatial_audio_synthesizer/room.py"
        spec = importlib.util.spec_from_file_location("track3_room_cache_test", path)
        room_module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"sofa": fake_sofa}):
            spec.loader.exec_module(room_module)

        room = room_module.SofaRoom("/data/room.sofa")
        self.assertEqual(room.room_info["nrir"], 2)
        with mock.patch.object(room_module.random, "randint", return_value=1):
            selected = room.get_position()
        self.assertEqual(selected, [[0.0, 1.0, 0.0]])
        self.assertEqual(np.asarray([selected[0]]).shape, (1, 3))
        # DatasetS3 compares this shape with every room position for duplicates.
        reference = np.atleast_2d([selected[0]])
        all_positions = room.get_all_positions()
        cosine = (all_positions / np.linalg.norm(all_positions, axis=1, keepdims=True)) @ (
            reference / np.linalg.norm(reference, axis=1, keepdims=True)
        ).T
        self.assertEqual(cosine.shape, (2, 1))
        self.assertFalse(room.get_all_positions().flags.writeable)

        second_room = room_module.SofaRoom("/data/room.sofa")
        with mock.patch.object(room_module.random, "randint", return_value=0):
            self.assertEqual(second_room.get_position(mode="point"),
                             [[1.0, 0.0, 0.0]])
        self.assertEqual(len(opened), 2)  # Geometry and positions, once each.

        result = room.synthesize(np.ones(4), 48000, selected, {"dry": False})
        self.assertEqual(result["waveform"].shape, (4, 6))
        self.assertEqual(len(opened), 3)  # Only per-source IR is read each time.
        self.assertTrue(all(database.closed for database in opened))


if __name__ == "__main__":
    unittest.main()
