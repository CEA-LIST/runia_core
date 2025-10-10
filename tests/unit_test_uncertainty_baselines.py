"""
tests/unit_test_uncertainty_baselines.py

Unittest suite for non-covered functions in
`runia.evaluation.baselines`.

Run with:

    python -m unittest -q tests/unit_test_uncertainty_baselines.py

This file focuses on small, fast unit tests that don't require external
resources. Tests are independent and use deterministic seeds where
relevant.
"""

from unittest import TestCase, main
from typing import Dict
import logging

import numpy as np
import torch

from runia.baselines.from_precalculated import (
    ash_s_linear_layer,
    generalized_entropy,
    get_labels_from_logits,
    remove_latent_features,
    mahalanobis_preprocess,
    mahalanobis_postprocess,
    get_baselines_thresholds,
    gmm_fit,
)

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global tolerance for floating point comparisons in tests
TOL = 1e-7


class TestBaselinesUncovered(TestCase):
    """Unittests for small utility functions in baselines.from_precalculated."""

    def setUp(self) -> None:
        # Deterministic seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

    def test_ash_s_linear_layer_percentile_zero(self):
        """When percentile is 0 the function should keep all features and apply
        the sharpening scale exp(1) because scale = s1 / s2 == 1.
        """
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 0.0, 1.0, 3.0, 4.0]], dtype=float)
        out = ash_s_linear_layer(x, percentile=0)

        # Expect equal shapes
        self.assertEqual(out.shape, x.shape)

        # For percentile 0 all features are kept, s2 == s1 and scale == 1 -> exp(1)
        expected = x * np.exp(1.0)
        # compare maximum absolute difference to tolerance
        self.assertAlmostEqual(float(np.max(np.abs(out - expected))), 0.0, delta=TOL)

    def test_generalized_entropy_simple(self):
        """Test the generalized entropy on a small probability matrix against a
        direct implementation.
        """
        probs = np.array([[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]])
        gamma = 1.0
        M = 2

        expected_sorted = np.sort(probs, axis=1)[:, -M:]
        expected_scores = np.sum(expected_sorted**gamma * (1 - expected_sorted) ** gamma, axis=1)
        expected = -expected_scores

        got = generalized_entropy(probs, gamma, M)
        self.assertIsInstance(got, np.ndarray)
        self.assertEqual(got.shape, (2,))
        # compare maximum absolute difference to tolerance
        self.assertAlmostEqual(float(np.max(np.abs(got - expected))), 0.0, delta=TOL)

    def test_get_labels_from_logits_various_branches(self):
        """Cover three branches:
        - numpy logits -> labels are argmax
        - empty lists -> empty label lists
        - unsupported types -> NotImplementedError
        """
        # Case 1: numpy logits
        id_data: Dict[str, np.ndarray] = {
            "train logits": np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]]),
            "valid logits": np.array([[0.4, 0.5, 0.1]]),
        }
        ood_data = {"ood1 logits": np.array([[0.2, 0.7, 0.1]])}
        id_res, ood_res = get_labels_from_logits(id_data.copy(), ood_data.copy(), ["ood1"])

        self.assertIn("train labels", id_res)
        self.assertIn("valid labels", id_res)
        self.assertEqual(id_res["train labels"].shape[0], 2)
        self.assertIn("ood1 labels", ood_res)
        self.assertEqual(ood_res["ood1 labels"].shape[0], 1)

        # Case 2: empty logits lists -> produce empty label lists
        id_data2 = {"train logits": [], "valid logits": []}
        ood_data2 = {"ood1 logits": []}
        id_res2, ood_res2 = get_labels_from_logits(id_data2.copy(), ood_data2.copy(), ["ood1"])  # type: ignore
        self.assertEqual(id_res2["train labels"], [])
        self.assertEqual(id_res2["valid labels"], [])
        self.assertEqual(ood_res2["ood1 labels"], [])

        # Case 3: unsupported types should raise
        bad_id = {"train logits": [1, 2, 3], "valid logits": [4, 5, 6]}
        bad_ood = {"ood1 logits": [1, 2]}
        with self.assertRaises(NotImplementedError):
            get_labels_from_logits(bad_id, bad_ood, ["ood1"])  # type: ignore

    def test_remove_latent_features_removes_keys(self):
        id_data = {"train features": np.ones((2, 3)), "valid features": np.zeros((1, 3))}
        ood_data = {"oodA features": np.full((1, 3), 2.0)}
        id_out, ood_out = remove_latent_features(id_data.copy(), ood_data.copy(), ["oodA"])

        self.assertNotIn("train features", id_out)
        self.assertNotIn("valid features", id_out)
        self.assertNotIn("oodA features", ood_out)

    def test_mahalanobis_preprocess_and_postprocess_basic(self):
        """Create a small synthetic dataset with two classes and run the
        preprocess/postprocess pipeline. Also ensure a warning is emitted when a
        class has no samples.
        """
        # Two classes, feature dim 4
        rng = np.random.RandomState(0)
        train_feats = np.vstack([rng.randn(3, 4) + i for i in range(2)])  # 6 x 4
        train_labels = np.array([0] * 3 + [1] * 3)
        ind_data = {"train features": train_feats, "train labels": train_labels}

        class_mean, precision = mahalanobis_preprocess(ind_data, num_classes=2)
        self.assertEqual(class_mean.shape, (2, 4))
        self.assertEqual(precision.shape[0], precision.shape[1])
        self.assertEqual(precision.shape[0], 4)

        # Create some validation feats
        valid_feats = np.vstack([rng.randn(2, 4) + i for i in range(2)])
        scores = mahalanobis_postprocess(valid_feats, class_mean, precision, num_classes=2)
        self.assertEqual(scores.shape[0], valid_feats.shape[0])
        self.assertTrue(np.isfinite(scores).all())

        # Now test warning when a class has no samples
        ind_data_missing = {"train features": train_feats, "train labels": train_labels}
        with self.assertWarns(UserWarning):
            mahalanobis_preprocess(ind_data_missing, num_classes=3)

    def test_get_baselines_thresholds_basic(self):
        scores = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([10.0, 10.0, 10.0])}
        th = get_baselines_thresholds(["a", "b"], scores, z_score_percentile=1.0)
        self.assertIn("a", th)
        self.assertIn("b", th)
        # For constant array b, std == 0 -> threshold == mean
        self.assertTrue(np.isclose(th["b"], 10.0))

    def test_gmm_fit_returns_distribution_and_jitter(self):
        # Create two classes with 3-dim embeddings
        embeddings = torch.tensor(
            [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1], [5.0, 6.0, 7.0], [5.1, 6.1, 7.1]]
        )
        labels = torch.tensor([0, 0, 1, 1])
        gmm, jitter = gmm_fit(embeddings, labels, num_classes=2)

        # MultivariateNormal loc has shape (n_components, dim)
        self.assertTrue(hasattr(gmm, "loc"))
        self.assertEqual(list(gmm.loc.shape), [2, 3])
        self.assertIsInstance(jitter, (int, float))


if __name__ == "__main__":
    main()
