"""Regression checks for the compatibility adapter."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from solver_adapter import simulate


REPO: Path


class SolverAdapterTests(unittest.TestCase):
    def test_documented_baseline(self) -> None:
        result = simulate(REPO)
        self.assertAlmostEqual(result["SD"], 1.64514486781, places=8)
        self.assertAlmostEqual(result["RI"], .392150795003, places=8)
        self.assertAlmostEqual(result["PI"], .52725811502, places=8)

    def test_index_identity(self) -> None:
        result = simulate(REPO)
        self.assertAlmostEqual(result["RI"], 1.0 - 1.0 / result["SD"], places=12)

    def test_radial_narrowing_raises_pi(self) -> None:
        narrow = simulate(REPO, radial_radius=.12)
        baseline = simulate(REPO, radial_radius=.20)
        wide = simulate(REPO, radial_radius=.28)
        self.assertGreater(narrow["PI"], baseline["PI"])
        self.assertGreater(baseline["PI"], wide["PI"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    REPO = args.repo
    unittest.main(argv=[__file__, *remaining])
