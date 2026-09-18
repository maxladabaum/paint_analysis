"""Check the Python probes embedded in the Windows batch launcher."""

import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


class WindowsLauncherTests(unittest.TestCase):
    def test_version_probes_compile_and_accept_supported_versions(self):
        launcher = (Path(__file__).resolve().parents[1] / "run_paint.bat").read_text()
        probes = [code for code in re.findall(r'-c "([^"]+)"', launcher)
                  if "sys.version_info" in code]
        self.assertEqual(len(probes), 5)
        for code in probes:
            with self.subTest(code=code):
                # Quoted Python must be valid as written: cmd passes carets
                # inside double quotes through to Python.
                tree = ast.parse(code)
                exit_call = tree.body[-1].exc
                expression = compile(ast.Expression(exit_call.args[0]), "<version probe>", "eval")
                for version in ((2, 7), (3, 9), (3, 10), (3, 11), (3, 12),
                                (3, 13), (3, 14), (3, 15), (4, 0)):
                    expected = version == (3, 12) if "==" in code else (3, 10) <= version < (3, 15)
                    result = eval(expression, {"sys": SimpleNamespace(version_info=version)})
                    self.assertEqual(result, 0 if expected else 1, version)


if __name__ == "__main__":
    unittest.main()
