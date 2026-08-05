"""Tests for grian.evaluation.ablations — preconfigured failure experiments."""


from grian.evaluation.ablations import (
    ALL_ABLATIONS,
    correct_baseline,
    future_leakage,
    make_ablation_suite,
    no_embargo,
    no_reconditioning,
    no_transform,
    wrong_loss,
)
from grian.evaluation.trials import DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Individual ablation configs
# ---------------------------------------------------------------------------

class TestCorrectBaseline:
    """Tests for the correct baseline config."""

    def test_has_all_default_keys(self):
        """Correct baseline has every key from DEFAULT_CONFIG."""
        cfg = correct_baseline()
        for key in DEFAULT_CONFIG:
            assert key in cfg

    def test_transform_is_asinh(self):
        """Correct baseline uses asinh transform."""
        cfg = correct_baseline()
        assert cfg["transform"] == "asinh"

    def test_no_leakage(self):
        """Correct baseline does not leak future data."""
        cfg = correct_baseline()
        assert cfg["ablations"]["leak_future"] is False

    def test_embargo_enabled(self):
        """Correct baseline uses embargo."""
        cfg = correct_baseline()
        assert cfg["ablations"]["use_embargo"] is True

    def test_daily_refit(self):
        """Correct baseline refits daily."""
        cfg = correct_baseline()
        assert cfg["refit_days"] == 1


class TestWrongLoss:
    """Tests for the wrong loss ablation."""

    def test_loss_is_mse(self):
        """Wrong loss config uses MSE."""
        cfg = wrong_loss()
        assert cfg["loss"] == "mse"

    def test_transform_still_applied(self):
        """Transform is still asinh — only the loss is wrong."""
        cfg = wrong_loss()
        assert cfg["transform"] == "asinh"


class TestNoTransform:
    """Tests for the no transform ablation."""

    def test_use_transform_false(self):
        """No-transform ablation flag is set."""
        cfg = no_transform()
        assert cfg["ablations"]["use_transform"] is False

    def test_transform_field_is_identity(self):
        """Transform name is identity."""
        cfg = no_transform()
        assert cfg["transform"] == "identity"


class TestNoReconditioning:
    """Tests for the no reconditioning ablation."""

    def test_refit_days_very_large(self):
        """Refit interval is effectively never."""
        cfg = no_reconditioning()
        assert cfg["refit_days"] >= 99999


class TestFutureLeakage:
    """Tests for the future leakage ablation."""

    def test_leak_future_true(self):
        """Future leakage flag is set."""
        cfg = future_leakage()
        assert cfg["ablations"]["leak_future"] is True


class TestNoEmbargo:
    """Tests for the no embargo ablation."""

    def test_use_embargo_false(self):
        """Embargo flag is disabled."""
        cfg = no_embargo()
        assert cfg["ablations"]["use_embargo"] is False

    def test_embargo_value_zero(self):
        """Embargo period is zero."""
        cfg = no_embargo()
        assert cfg["embargo"] == 0


# ---------------------------------------------------------------------------
# Ablation suite
# ---------------------------------------------------------------------------

class TestAblationSuite:
    """Tests for the batch ablation suite generator."""

    def test_suite_length(self):
        """Suite produces one config per ablation."""
        suite = make_ablation_suite()
        assert len(suite) == len(ALL_ABLATIONS)

    def test_unique_trial_names(self):
        """Every ablation in the suite has a unique trial name."""
        suite = make_ablation_suite()
        names = [cfg["trial_name"] for cfg in suite]
        assert len(names) == len(set(names))

    def test_custom_model(self):
        """Suite respects custom model name."""
        suite = make_ablation_suite(model="simple_mlp")
        for cfg in suite:
            assert cfg["model"] == "simple_mlp"

    def test_custom_regions(self):
        """Suite respects custom regions."""
        suite = make_ablation_suite(regions=["SA1", "VIC1"])
        for cfg in suite:
            assert cfg["regions"] == ["SA1", "VIC1"]

    def test_all_configs_complete(self):
        """Every config in the suite has all required keys."""
        suite = make_ablation_suite()
        for cfg in suite:
            for key in DEFAULT_CONFIG:
                assert key in cfg, f"Config {cfg['trial_name']!r} missing key {key!r}"

    def test_exactly_one_correct(self):
        """Exactly one config in the suite has no deliberate mistakes."""
        suite = make_ablation_suite()
        correct_count = sum(
            1 for cfg in suite
            if (cfg["ablations"]["use_transform"]
                and cfg["ablations"]["use_embargo"]
                and not cfg["ablations"]["leak_future"]
                and cfg["refit_days"] == 1
                and cfg["loss"] != "mse")
        )
        assert correct_count == 1
