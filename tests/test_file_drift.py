import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paint_analysis_gui import apply_drift_correction, read_drift_csv


class FileDriftTests(unittest.TestCase):
    def test_applies_nm_drift_to_each_localization_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift.csv"
            path.write_text(
                "Frame,x-drift (nm),y-drift (nm),z-drift (nm)\n"
                "2,260,-130,65\n"
                "0,0,0,0\n"
                "1,130,260,130\n",
                encoding="utf-8",
            )
            locs = pd.DataFrame(
                {
                    "frame": np.asarray([2, 0, 1, 2], dtype=np.uint32),
                    "x": np.asarray([10, 10, 10, 20], dtype=np.float32),
                    "y": np.asarray([5, 5, 5, 10], dtype=np.float32),
                    "z": np.asarray([3, 3, 3, 6], dtype=np.float32),
                }
            )
            info = [{"Frames": 3, "Width": 30, "Height": 30, "Pixelsize": 130.0}]

            corrected, drift, label = apply_drift_correction(
                locs,
                info,
                "file",
                1000,
                20.0,
                60.0,
                drift_file_path=path,
            )

        np.testing.assert_allclose(corrected["x"], [8, 10, 9, 18])
        np.testing.assert_allclose(corrected["y"], [6, 5, 3, 11])
        np.testing.assert_allclose(corrected["z"], [2.5, 3, 2, 5.5])
        np.testing.assert_allclose(drift["x"], [0, 1, 2])
        self.assertEqual(corrected["x"].dtype, np.float32)
        self.assertIn("drift.csv", label)
        np.testing.assert_allclose(locs["x"], [10, 10, 10, 20])

    def test_rejects_missing_drift_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift.csv"
            path.write_text("Frame,x,y\n0,0,0\n2,1,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing 1 frame.*1"):
                read_drift_csv(path, frame_count=3, pixel_size_nm=130.0)

    def test_rejects_duplicate_drift_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift.csv"
            path.write_text("Frame,x,y\n0,0,0\n0,1,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate frame"):
                read_drift_csv(path, frame_count=1, pixel_size_nm=130.0)


if __name__ == "__main__":
    unittest.main()
