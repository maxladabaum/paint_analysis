import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paint_analysis_gui import (
    LATEST_DEVELOPMENT_SESSION_FILE,
    PaintAnalysisApp,
    latest_development_session_request,
    set_next_startup_session,
)


class StartupCacheTests(unittest.TestCase):
    def test_cache_requires_approval_and_approval_is_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "localizations.csv"
            source.touch()
            request = (source, {"method": "none"})
            latest = folder / LATEST_DEVELOPMENT_SESSION_FILE.name
            latest.write_text(json.dumps({"source_path": str(source), "settings": request[1]}))

            self.assertEqual(latest_development_session_request(folder), request)
            self.assertIsNone(latest_development_session_request(folder, next_startup=True))
            set_next_startup_session(request, folder)
            self.assertEqual(latest_development_session_request(folder, next_startup=True), request)
            self.assertIsNone(latest_development_session_request(folder, next_startup=True))

            set_next_startup_session(request, folder)
            set_next_startup_session(None, folder)
            self.assertIsNone(latest_development_session_request(folder, next_startup=True))
            self.assertTrue(latest.exists())

    def test_close_saves_yes_or_no_and_closes_window(self):
        request = (Path("localizations.csv"), {})
        for answer in (True, False):
            with self.subTest(answer=answer), patch.dict("os.environ", {}, clear=True), \
                    patch("paint_analysis_gui.latest_development_session_request", return_value=request), \
                    patch("paint_analysis_gui.messagebox.askyesno", return_value=answer) as prompt, \
                    patch("paint_analysis_gui.set_next_startup_session") as save:
                app = SimpleNamespace(destroy=Mock())
                PaintAnalysisApp._on_close(app)
                prompt.assert_called_once()
                self.assertEqual(prompt.call_args.kwargs["default"], "no")
                save.assert_called_once_with(request if answer else None)
                app.destroy.assert_called_once()

    def test_startup_without_approval_does_not_start_worker(self):
        app = SimpleNamespace(loaded=None, session_load_in_progress=False, _run_worker=Mock())
        with patch("paint_analysis_gui.latest_development_session_request", return_value=None) as request:
            PaintAnalysisApp._restore_latest_development_session(app)
        request.assert_called_once_with(next_startup=True)
        app._run_worker.assert_not_called()

    def test_failed_save_keeps_window_open(self):
        app = SimpleNamespace(destroy=Mock())
        with patch("paint_analysis_gui.latest_development_session_request", return_value=None), \
                patch("paint_analysis_gui.set_next_startup_session", side_effect=OSError("disk full")), \
                patch("paint_analysis_gui.messagebox.showerror") as error:
            PaintAnalysisApp._on_close(app)
        error.assert_called_once()
        app.destroy.assert_not_called()
