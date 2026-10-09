from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts" / "smart_rtl.py"


class SmartRTLCLITest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLI), *arguments],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_top_level_help(self) -> None:
        completed = self.run_cli("--help")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("{cvdp,chipverilog}", completed.stdout)

    def test_cvdp_help_is_delegated(self) -> None:
        completed = self.run_cli("cvdp", "--help")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("one Smart-RTL adaptive search on CVDP", completed.stdout)
        self.assertIn("--dataset", completed.stdout)

    def test_chipverilog_help_is_delegated(self) -> None:
        completed = self.run_cli("chipverilog", "--help")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("one Smart-RTL adaptive search on ChipVerilog", completed.stdout)
        self.assertIn("--list-tasks", completed.stdout)


if __name__ == "__main__":
    unittest.main()
