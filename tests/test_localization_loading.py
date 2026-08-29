import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

from paint_analysis_gui import histogram_values_for_mode, read_locs, read_locs_csv


class LocalizationLoadingTests(unittest.TestCase):
    def test_reads_nm_csv_export_and_maps_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "localizations.csv"
            pd.DataFrame(
                {
                    "x (nm)": [130.0, 260.0, np.nan],
                    "y (nm)": [390.0, 520.0, 650.0],
                    "sigmaX (nm)": [13.0, 26.0, 39.0],
                    "sigmaY (nm)": [19.5, 32.5, 45.5],
                    "intensity (photons)": [1000.0, 2000.0, 3000.0],
                    "frameIndex": [5.0, 6.0, 7.0],
                    "localization precision (nm)": [2.0, 3.0, 4.0],
                    "channelIndex": [0.0, 0.0, 0.0],
                }
            ).to_csv(path, index=False)

            loaded = read_locs(path)

        self.assertEqual(list(loaded.locs["frame"]), [5, 6])
        np.testing.assert_allclose(loaded.locs["x"], [1.0, 2.0])
        np.testing.assert_allclose(loaded.locs["y"], [3.0, 4.0])
        np.testing.assert_allclose(loaded.locs["sx"], [0.1, 0.2])
        np.testing.assert_allclose(loaded.locs["photons"], [1000.0, 2000.0])
        np.testing.assert_allclose(loaded.locs["precision_nm"], [2.0, 3.0])
        values, label = histogram_values_for_mode(loaded.locs, "precision_nm", 130.0, 100.0, 75.0, 1)
        np.testing.assert_allclose(values, [2.0, 3.0])
        self.assertEqual(label, "Localization precision (nm, QC filtered)")
        self.assertEqual(loaded.info, [{"Frames": 7, "Width": 3, "Height": 5, "Pixelsize": 130.0}])
        self.assertEqual(loaded.metadata["Source format"], "CSV")
        self.assertEqual(loaded.metadata["Localization count"], 2)

    def test_csv_uses_companion_yaml_pixel_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "localizations.csv"
            path.write_text("frameIndex,x (nm),y (nm)\n0,100,200\n", encoding="utf-8")
            path.with_suffix(".yaml").write_text("Pixelsize: 100\n", encoding="utf-8")

            loaded = read_locs_csv(path)

        self.assertAlmostEqual(float(loaded.locs.loc[0, "x"]), 1.0)
        self.assertAlmostEqual(float(loaded.locs.loc[0, "y"]), 2.0)
        self.assertEqual(loaded.info[0]["Pixelsize"], 100.0)

    def test_csv_requires_frame_x_and_y(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("x (nm),y (nm)\n100,200\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required column.*frame"):
                read_locs_csv(path)

    def test_csv_loading_reports_monotonic_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "localizations.csv"
            path.write_text("frame,x,y\n0,1,2\n1,3,4\n", encoding="utf-8")
            updates: list[tuple[float, str]] = []

            read_locs(path, lambda percent, message: updates.append((percent, message)))

        self.assertEqual(updates[0][0], 0.0)
        self.assertEqual(updates[-1][0], 100.0)
        self.assertTrue(all(left[0] <= right[0] for left, right in zip(updates, updates[1:])))
        self.assertTrue(any("Loading CSV" in message for _percent, message in updates))

    def test_hdf5_loading_reports_incremental_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "localizations.hdf5"
            values = np.zeros(3, dtype=[("frame", "u4"), ("x", "f4"), ("y", "f4")])
            values["frame"] = [0, 1, 2]
            values["x"] = [1, 2, 3]
            values["y"] = [4, 5, 6]
            with h5py.File(path, "w") as handle:
                handle.create_dataset("locs", data=values)
            updates: list[tuple[float, str]] = []

            loaded = read_locs(path, lambda percent, message: updates.append((percent, message)))

        self.assertEqual(len(loaded.locs), 3)
        self.assertEqual(updates[0][0], 0.0)
        self.assertEqual(updates[-1][0], 100.0)
        self.assertTrue(any("HDF5" in message for _percent, message in updates))


if __name__ == "__main__":
    unittest.main()
