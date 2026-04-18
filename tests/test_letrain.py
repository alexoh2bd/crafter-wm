"""
Tests for letrain.py:
  - _load_config    YAML flattening
  - train()         1-epoch smoke test (tiny model, synthetic data)
"""

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import letrain
from letrain import _load_config, train


# ── _load_config ─────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_flattens_nested_sections(self, tmp_path):
        cfg = {"data": {"n_episodes": 99, "save_path": "x.pkl"},
               "training": {"epochs": 5, "batch_size": 4}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        flat = _load_config(str(p))
        assert flat["n_episodes"] == 99
        assert flat["epochs"] == 5

    def test_no_section_key_leaks(self, tmp_path):
        """Section names ('data', 'training') must not appear as keys."""
        cfg = {"data": {"n_episodes": 1}, "training": {"epochs": 2}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        flat = _load_config(str(p))
        assert "data" not in flat
        assert "training" not in flat

    def test_later_sections_override_earlier(self, tmp_path):
        """If two sections share a key, the last wins."""
        cfg = {"section_a": {"lr": 0.1}, "section_b": {"lr": 0.001}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        flat = _load_config(str(p))
        assert flat["lr"] == pytest.approx(0.001)

    def test_teacher_config_keys(self):
        """config_teacher.yaml must contain all keys letrain.py needs."""
        cfg_path = SRC / "config_teacher.yaml"
        if not cfg_path.exists():
            pytest.skip("config_teacher.yaml not found")
        flat = _load_config(str(cfg_path))
        for key in ("context_len", "latent_dim", "epochs", "batch_size", "lr"):
            assert key in flat, f"Missing key: {key}"


# ── train() — 1-epoch smoke test ─────────────────────────────────────────────

class TestTrainSmoke:
    """
    Patches DataLoader to use num_workers=0 so the test doesn't spawn processes.
    Uses the tiny_train_args fixture (batch_size=2, 1 epoch, tiny model).
    """

    @pytest.fixture(autouse=True)
    def patch_dataloader(self, monkeypatch):
        """Force num_workers=0 and pin_memory=False for fast in-process tests."""
        from functools import partial
        from torch.utils.data import DataLoader as _DL

        def safe_loader(*args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs["pin_memory"] = False
            return _DL(*args, **kwargs)

        monkeypatch.setattr(letrain, "DataLoader", safe_loader)

    def test_checkpoint_files_created(self, tiny_train_args):
        train(tiny_train_args)
        logdir = Path(tiny_train_args.logdir)
        assert (logdir / "latest.pt").exists(), "latest.pt not saved"
        assert (logdir / "best.pt").exists(),   "best.pt not saved"

    def test_metrics_jsonl_written(self, tiny_train_args):
        train(tiny_train_args)
        jsonl = Path(tiny_train_args.logdir) / "metrics.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text().strip().splitlines()
        assert len(lines) >= 1   # at least 1 val_loss line per epoch

    def test_checkpoint_has_required_keys(self, tiny_train_args):
        train(tiny_train_args)
        ckpt = torch.load(
            Path(tiny_train_args.logdir) / "best.pt",
            map_location="cpu", weights_only=False,
        )
        for key in ("epoch", "step", "model", "optimizer", "val_loss", "args"):
            assert key in ckpt, f"Missing checkpoint key: {key}"

    def test_checkpoint_args_match_input(self, tiny_train_args):
        train(tiny_train_args)
        ckpt = torch.load(
            Path(tiny_train_args.logdir) / "best.pt",
            map_location="cpu", weights_only=False,
        )
        assert ckpt["args"]["latent_dim"] == tiny_train_args.latent_dim
        assert ckpt["args"]["epochs"] == tiny_train_args.epochs

    def test_val_loss_in_metrics(self, tiny_train_args):
        train(tiny_train_args)
        jsonl = Path(tiny_train_args.logdir) / "metrics.jsonl"
        records = [json.loads(l) for l in jsonl.read_text().strip().splitlines()]
        assert any("val_loss" in r for r in records)

    def test_val_loss_is_finite(self, tiny_train_args):
        train(tiny_train_args)
        jsonl = Path(tiny_train_args.logdir) / "metrics.jsonl"
        for line in jsonl.read_text().strip().splitlines():
            r = json.loads(line)
            if "val_loss" in r:
                import math
                assert math.isfinite(r["val_loss"])

    def test_global_step_advanced_after_train(self, tiny_train_args):
        """Checkpoint must record global_step > 0 confirming optimiser ran."""
        train(tiny_train_args)
        ckpt = torch.load(
            Path(tiny_train_args.logdir) / "best.pt",
            map_location="cpu", weights_only=False,
        )
        assert ckpt["step"] > 0, "global_step is 0 — no gradient updates ran"
