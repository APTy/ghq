import time
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Neural benchmark tests require torch")

from ghqnet.benchmark import _measure, run_benchmark, summarize_latencies
from ghqnet.model import NetworkConfig, PolicyValueNet


class NeuralBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_statistics_are_per_batch_and_throughput_uses_total_boards(self):
        stats = summarize_latencies([1.0, 2.0, 3.0, 4.0], 16)
        self.assertEqual(stats["samples"], 4)
        self.assertEqual(stats["median_ms"], 2.5)
        self.assertAlmostEqual(stats["p95_ms"], 3.85)
        self.assertAlmostEqual(stats["boards_per_second"], 6400)
        self.assertIsNone(summarize_latencies([], 16)["median_ms"])
        with self.assertRaises(ValueError):
            summarize_latencies([float("nan")], 1)

    def test_cpu_tiny_network_reports_inference_and_training_without_mutating_source(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=8)).train()
        original = {name: value.clone() for name, value in model.state_dict().items()}
        report = run_benchmark(model, device="cpu", repeats=2, warmup=1,
                               batch_sizes=(1, 2), train_batch_size=2, train_repeats=1, max_seconds=5)
        self.assertTrue(report["complete"], report)
        self.assertEqual(report["parameters"], sum(p.numel() for p in model.parameters()))
        self.assertGreater(report["legal_actions_per_board"], 1)
        self.assertEqual(report["timing_scope"], "direct_network_device_resident_inputs")
        self.assertEqual(len(report["devices"]), 1)
        device = report["devices"][0]
        self.assertEqual(device["device"], "cpu")
        self.assertEqual([row["batch_size"] for row in device["inference"]], [1, 2])
        for row in [*device["inference"], device["training"]]:
            self.assertGreater(row["boards_per_second"], 0)
            self.assertGreater(row["median_ms"], 0)
            self.assertGreaterEqual(row["p95_ms"], row["median_ms"])
            self.assertTrue(row["complete"])
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_budget_exhaustion_is_reported_without_fabricated_samples(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=8))
        report = run_benchmark(model, device="cpu", repeats=2, warmup=1,
                               batch_sizes=(1,), train_batch_size=1, train_repeats=1, max_seconds=1e-9)
        self.assertFalse(report["complete"])
        for row in [*report["devices"][0]["inference"], report["devices"][0]["training"]]:
            self.assertEqual(row["samples"], 0)
            self.assertIsNone(row["boards_per_second"])
            self.assertFalse(row["complete"])

    def test_cuda_timing_synchronizes_around_work_and_requires_available_gpu(self):
        events = []
        with patch("ghqnet.benchmark.torch.cuda.synchronize", side_effect=lambda _: events.append("sync")):
            result = _measure(lambda: events.append("work"), device=torch.device("cuda"),
                              batch_size=1, warmup=1, repeats=1, deadline=time.perf_counter() + 1)
        self.assertEqual(events, ["sync", "work", "sync", "sync", "work", "sync"])
        self.assertEqual(result["samples"], 1)
        with patch("ghqnet.benchmark.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "cannot see a GPU"):
                run_benchmark(PolicyValueNet(NetworkConfig(blocks=0, channels=8)), device="cuda")


if __name__ == "__main__":
    unittest.main()
