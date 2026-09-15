import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paint_analysis_gui import export_roi_localizations_csv, read_locs_csv, roi_file_viewport_nm, PaintAnalysisApp
from types import SimpleNamespace


class RoiCsvExportTests(unittest.TestCase):
    def test_raw_preserves_all_fields_and_corrected_uses_corrected_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'locs.csv'
            source.write_text('frameIndex,x (nm),y (nm),z (nm),ID,note\n'
                              '0,100,200,50,0001,"hello, world"\n'
                              '1,nan,200,60,0002,invalid\n'
                              '2,300,200,70,0003,NA\n'
                              '3,500,200,80,0004,last\n')
            source.with_suffix('.yaml').write_text('Pixelsize: 100\n')
            loaded = read_locs_csv(source)
            original = source.read_text()
            raw_path = Path(directory) / 'raw.csv'
            self.assertEqual(export_roi_localizations_csv(
                loaded, raw_path, (300, 100, 250, 150), chunk_size=2), 2)
            raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
            self.assertEqual(raw['ID'].tolist(), ['0001', '0003'])
            self.assertEqual(raw['note'].tolist(), ['hello, world', 'NA'])
            self.assertEqual(raw['x (nm)'].tolist(), ['100', '300'])
            corrected = loaded.locs.reset_index(drop=True).copy()
            corrected['x'] -= 1
            corrected['z'] -= 0.25
            corrected_path = Path(directory) / 'corrected.csv'
            self.assertEqual(export_roi_localizations_csv(
                loaded, corrected_path, (150, 250, 150, 250), corrected, chunk_size=2), 1)
            result = pd.read_csv(corrected_path, dtype=str, keep_default_na=False)
            self.assertEqual(result.columns.tolist(), raw.columns.tolist())
            self.assertEqual(result['ID'].tolist(), ['0003'])
            np.testing.assert_allclose(result['x (nm)'].astype(float), [200])
            np.testing.assert_allclose(result['z (nm)'].astype(float), [45], atol=1e-5)
            reloaded = read_locs_csv(corrected_path)
            self.assertEqual(reloaded.info[0]['Pixelsize'], 100)
            self.assertEqual(roi_file_viewport_nm(reloaded), (150, 250, 150, 250))
            self.assertEqual(PaintAnalysisApp._full_map_viewport_nm(SimpleNamespace(loaded=reloaded)), (150, 250, 150, 250))
            self.assertEqual(source.read_text(), original)

    def test_older_roi_export_fits_occupied_bounds_without_changing_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'example_corrected_roi.csv'
            source.write_text('frame,x,y\n0,400,500\n1,600,700\n')
            source.with_suffix('.yaml').write_text('Pixelsize: 100\nWidth: 1200\nHeight: 1200\n')
            loaded = read_locs_csv(source)
            original = loaded.locs.copy()
            bounds = roi_file_viewport_nm(loaded)
            self.assertEqual(bounds, (39800, 60200, 49800, 70200))
            pd.testing.assert_frame_equal(loaded.locs, original)
            self.assertEqual(loaded.info[0]['Width'], 1200)
            loaded.metadata.pop('ROI bounds (nm)')
            loaded.path = source.with_name('ordinary.csv')
            self.assertIsNone(roi_file_viewport_nm(loaded))

    def test_empty_roi_writes_header_and_source_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'locs.csv'
            source.write_text('frame,x,y,extra\n0,1,2,unchanged\n')
            loaded = read_locs_csv(source)
            target = Path(directory) / 'empty.csv'
            self.assertEqual(export_roi_localizations_csv(loaded, target, (0, 1, 0, 1)), 0)
            self.assertEqual(target.read_text(), 'frame,x,y,extra\n')
            with self.assertRaisesRegex(ValueError, 'source cannot'):
                export_roi_localizations_csv(loaded, source, (0, 1, 0, 1))
            source.write_text('frame,x,y,extra\n0,9,2,changed\n')
            with self.assertRaisesRegex(ValueError, 'changed since loading'):
                export_roi_localizations_csv(loaded, target, (0, 1, 0, 1))
            self.assertEqual(target.read_text(), 'frame,x,y,extra\n')


if __name__ == '__main__':
    unittest.main()
