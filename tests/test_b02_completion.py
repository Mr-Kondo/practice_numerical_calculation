"""B02 completion reports numerical progress, not physical acceptance."""

import unittest

import numpy as np

from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_FVM_solver import run as run_fvm
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_SPH_solver import (
    _compute_vorticity,
    _project_particles_to_grid,
    _stable_dt,
    run as run_sph,
)


class B02CompletionTest(unittest.TestCase):
    def test_completion_requires_target_time(self) -> None:
        config = {
            "seed": 7,
            "grid_nx": 8,
            "grid_ny": 8,
            "steps": 1,
            "dt": 0.001,
            "max_steps": 1,
            "sph_particles": 16,
            "sph_smoothing_length": 0.1,
            "sph_initial_height": 0.2,
        }
        for runner in (run_fvm, run_sph):
            with self.subTest(method=runner.__module__):
                incomplete = runner({**config, "target_sim_time_s": 0.1}, prefer_gpu=False)
                self.assertEqual(incomplete.metrics["completion_flag"], 0.0)
                self.assertEqual(incomplete.metadata["status"], "incomplete")
                self.assertNotIn("acceptance_pass", incomplete.metrics)
                complete = runner({**config, "target_sim_time_s": 0.0001}, prefer_gpu=True)
                self.assertEqual(complete.metrics["completion_flag"], 1.0)
                self.assertEqual(complete.metadata["status"], "success")
                self.assertEqual(complete.metadata["backend"], "numpy")

    def test_fvm_boundary_mass_budget(self) -> None:
        config = {"grid_nx": 40, "grid_ny": 24, "steps": 1, "dt": 0.001,
                  "target_sim_time_s": 0.4, "max_steps": 1000}
        for boundary in ("reflective", "open"):
            with self.subTest(boundary=boundary):
                result = run_fvm({**config, "right_boundary": boundary}, prefer_gpu=False)
                self.assertLess(result.metrics["mass_error"], 1e-12)
                self.assertAlmostEqual(
                    result.metrics["retained_mass_fraction"] + result.metrics["escaped_mass_fraction"], 1.0
                )
                if boundary == "open":
                    self.assertGreater(result.metrics["escaped_mass_fraction"], 0.0)
                else:
                    self.assertEqual(result.metrics["escaped_mass_fraction"], 0.0)

    def test_sph_boundaries_account_for_active_particles(self) -> None:
        config = {"seed": 7, "grid_nx": 20, "grid_ny": 20, "steps": 1, "dt": 0.001,
                  "target_sim_time_s": 0.4, "max_steps": 1000, "sph_particles": 64,
                  "sph_smoothing_length": 0.06, "sph_initial_height": 0.4,
                  "dam_width_fraction": 0.8, "sph_c0": 10, "right_boundary": "open"}
        for boundary in ("open", "reflective"):
            with self.subTest(boundary=boundary):
                result = run_sph({**config, "right_boundary": boundary}, prefer_gpu=False)
                self.assertLess(result.metrics["mass_error"], 1e-12)
                self.assertAlmostEqual(
                    result.metrics["retained_mass_fraction"] + result.metrics["escaped_mass_fraction"], 1.0
                )
                if boundary == "open":
                    self.assertGreater(result.metrics["escaped_mass_fraction"], 0.0)
                else:
                    self.assertEqual(result.metrics["escaped_mass_fraction"], 0.0)
                for frame in result.metadata["viz_timeseries"]["particle_x_series"]:
                    self.assertTrue(all(0.0 <= x <= 1.0 for x in frame))

    def test_sph_projection_respects_domain_and_uniform_flow_has_no_curl(self) -> None:
        positions = np.array([[0.25, 1.5]])
        velocities = np.array([[1.0, 0.0]])
        ratio, u, v, counts = _project_particles_to_grid(
            positions, velocities, mass=1.0, rho0=1.0, nx=4, ny=4, domain_y=2.0
        )
        self.assertEqual(counts[3, 1], 1.0)
        self.assertEqual(ratio[3, 1], 8.0)
        omega = _compute_vorticity(u, v, counts, dx=0.25, dy=0.5, h=0.1)
        np.testing.assert_allclose(omega, 0.0, atol=1e-12)

    def test_sph_time_step_keeps_acoustic_bound(self) -> None:
        dt = _stable_dt(0.01, np.array([[100.0, 0.0]]), h=0.01, c0=10.0, cfl=0.25)
        self.assertLessEqual(dt, 0.25 * 0.01 / 110.0)


if __name__ == "__main__":
    unittest.main()
