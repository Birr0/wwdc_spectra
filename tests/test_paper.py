"""Protect paper architecture, archived-split requirements and CPU loading."""

import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("DATA_ROOT", "/tmp/unused-wwdc-test-data")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper", ROOT / "paper.py")
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)


class PaperTests(unittest.TestCase):
    def test_recipes_match_archived_architecture_and_condition(self):
        for arm, variable in (("redshift", "Z"), ("norm", "NORM")):
            config = paper.recipe(arm)
            self.assertEqual(list(config["variables"]), [variable])
            self.assertEqual((config["code_dim"], config["hidden_dim"], config["n_layers"]), (10, 1024, 3))
            self.assertEqual(config["batch_size"], 512)
            self.assertEqual(config["lr"], .0001)
        self.assertEqual(paper.recipe("norm")["normalisation_reference"]["retained_rows"], 411742)

    def test_missing_saved_splits_never_falls_back_to_download(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            args = SimpleNamespace(data_root=Path(folder), outdir=Path(folder) / "out", arm="norm")
            with self.assertRaisesRegex(ValueError, "never recreates splits"):
                paper.prepare(args)
            self.assertFalse(args.outdir.exists())

    def test_pretrained_weights_load_on_cpu_and_remain_frozen(self):
        import torch
        from wwdc_spectra.models.load_spender import PretrainedSpender
        module = torch.nn.Linear(3, 2)
        with patch("spender.hub.load", return_value=(None, module)) as load:
            wrapped = PretrainedSpender("sdss_I", 2)
        load.assert_called_once_with("sdss_I", map_location="cpu")
        self.assertFalse(any(p.requires_grad for p in wrapped.parameters()))


if __name__ == "__main__":
    unittest.main()
