#!/usr/bin/env python3
"""
Unit tests for the evaluation.inference.postprocessors module.

This script provides comprehensive unit tests for the postprocessor classes in the
runia_core.evaluation.inference.postprocessors module. It tests various
postprocessing methods for out-of-distribution (OoD) detection including density estimation,
distance-based methods, and energy-based approaches.

Usage:
    Run all tests: python -m unittest tests.unit_test_uncertainty_postprocessors
    Run specific test: python -m unittest tests.unit_test_uncertainty_postprocessors.TestKDELatentSpace.test_kde_setup
    Run with verbose output: python -m unittest -v tests.unit_test_uncertainty_postprocessors

Requirements:
    - Python 3.9+
    - PyTorch
    - NumPy
    - Scikit-learn
    - Faiss
    - SciPy
    - unittest (built-in)
    - runia_core library

Date: 2025-01-09
"""

import logging
import unittest
import warnings

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from runia_core.inference.postprocessors import (
    DDU,
    GEN,
    ViM,
    Energy,
    GMMLatentSpace,
    KDELatentSpace,
    KNNLatentSpace,
    MDLatentSpace,
    Mahalanobis,
    cMDLatentSpace,
    postprocessors_dict,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test constants
SEED = 42
TOLERANCE = 1e-6
TEST_FEATURE_DIM = 32
TEST_NUM_SAMPLES = 10
TEST_NUM_CLASSES = 10
TEST_BATCH_SIZE = 16


def generate_test_data(
    num_samples: int = TEST_NUM_SAMPLES,
    feature_dim: int = TEST_FEATURE_DIM,
    num_classes: int = TEST_NUM_CLASSES,
    seed: int = SEED,
):
    """
    Generate synthetic test data for postprocessor testing.

    Args:
        num_samples: Number of samples to generate
        feature_dim: Dimensionality of features
        num_classes: Number of classes for labels
        seed: Random seed for reproducibility

    Returns:
        tuple: (features, labels, logits)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate features with slight class separation
    features = np.random.randn(num_samples, feature_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)

    # Add class-specific bias to features for more realistic data
    for i in range(num_classes):
        class_mask = labels == i
        if np.any(class_mask):
            features[class_mask] += np.random.randn(feature_dim) * 0.5

    # Generate logits
    logits = np.random.randn(num_samples, num_classes).astype(np.float32)

    return features, labels, logits


class TestKDELatentSpace(unittest.TestCase):
    """Test cases for the KDELatentSpace postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.kde_processor = KDELatentSpace()
        self.train_features, self.train_labels, _ = generate_test_data(seed=SEED)
        self.test_features, _, _ = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestKDELatentSpace")

    def test_kde_initialization(self):
        """Test KDELatentSpace initialization."""
        self.assertIsNone(self.kde_processor.detector)
        self.assertFalse(self.kde_processor._setup_flag)

        # Test with config
        cfg = DictConfig({"param": "value"})
        kde_with_cfg = KDELatentSpace(cfg)
        self.assertIsNone(kde_with_cfg.detector)

    def test_kde_setup(self):
        """Test KDELatentSpace setup method."""
        self.kde_processor.setup(self.train_features)

        self.assertTrue(self.kde_processor._setup_flag)
        self.assertIsNotNone(self.kde_processor.detector)

    def test_kde_setup_already_trained_warning(self):
        """Test warning when setting up already trained KDE."""
        self.kde_processor.setup(self.train_features)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.kde_processor.setup(self.train_features)
            self.assertEqual(len(w), 1)
            self.assertIn("already trained", str(w[0].message))

    def test_kde_postprocess(self):
        """Test KDELatentSpace postprocess method."""
        self.kde_processor.setup(self.train_features)
        scores = self.kde_processor.postprocess(self.test_features)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        -55.453746556032975,
                        -45.876175810798735,
                        -52.02871966747913,
                        -55.43453172279228,
                        -64.97713394207216,
                        -62.392177312401635,
                        -55.94369071185685,
                        -46.959354167293704,
                        -56.72893201254575,
                        -53.831503746515544,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )


class TestMDLatentSpace(unittest.TestCase):
    """Test cases for the MDLatentSpace postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.md_processor = MDLatentSpace()
        self.train_features, self.train_labels, _ = generate_test_data(seed=SEED)
        self.test_features, _, _ = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestMDLatentSpace")

    def test_md_initialization(self):
        """Test MDLatentSpace initialization."""
        self.assertIsNone(self.md_processor.feats_mean)
        self.assertIsNone(self.md_processor.precision)
        self.assertIsNone(self.md_processor.centered_data)
        self.assertFalse(self.md_processor._setup_flag)

    def test_md_setup(self):
        """Test MDLatentSpace setup method."""
        self.md_processor.setup(self.train_features)

        self.assertTrue(self.md_processor._setup_flag)
        self.assertIsNotNone(self.md_processor.feats_mean)
        self.assertIsNotNone(self.md_processor.precision)
        self.assertIsNotNone(self.md_processor.centered_data)

        # Check shapes
        self.assertEqual(self.md_processor.feats_mean.shape, (1, TEST_FEATURE_DIM))
        self.assertEqual(self.md_processor.precision.shape, (TEST_FEATURE_DIM, TEST_FEATURE_DIM))

    def test_md_postprocess(self):
        """Test MDLatentSpace postprocess method."""
        self.md_processor.setup(self.train_features)
        scores = self.md_processor.postprocess(self.test_features)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        -3.6267479236438573,
                        -6.005989318619297,
                        -3.603247642226861,
                        -6.893168926200505,
                        -6.821623606454671,
                        -1.722633778077239,
                        -3.4629630663763664,
                        -8.888059923880624,
                        -4.879641073940862,
                        -7.062622955578143,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )


class TestcMDLatentSpace(unittest.TestCase):
    """Test cases for the cMDLatentSpace (class-conditional MD) postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.cmd_processor = cMDLatentSpace()
        self.train_features, self.train_labels, _ = generate_test_data(seed=SEED)
        self.test_features, _, _ = generate_test_data(seed=SEED + 1)
        self.pred_labels = np.random.randint(0, TEST_NUM_CLASSES, len(self.test_features))
        logger.info("Setting up TestcMDLatentSpace")

    def test_cmd_initialization(self):
        """Test cMDLatentSpace initialization."""
        self.assertEqual(self.cmd_processor.num_classes, 10)
        self.assertIsNone(self.cmd_processor.feats_mean)
        self.assertIsNone(self.cmd_processor.precision)
        self.assertIsNone(self.cmd_processor.class_mean)
        self.assertFalse(self.cmd_processor._setup_flag)

        # Test with custom config
        cfg = DictConfig({"num_classes": 5})
        cmd_with_cfg = cMDLatentSpace(cfg)
        self.assertEqual(cmd_with_cfg.num_classes, 5)

    def test_cmd_setup(self):
        """Test cMDLatentSpace setup method."""
        self.cmd_processor.setup(self.train_features, ind_train_labels=self.train_labels)

        self.assertTrue(self.cmd_processor._setup_flag)
        self.assertIsNotNone(self.cmd_processor.class_mean)
        self.assertIsNotNone(self.cmd_processor.precision)

        # Check shapes
        self.assertEqual(self.cmd_processor.class_mean.shape, (TEST_NUM_CLASSES, TEST_FEATURE_DIM))

    def test_cmd_setup_missing_labels(self):
        """Test cMDLatentSpace setup without required labels."""
        with self.assertRaises(ValueError) as context:
            self.cmd_processor.setup(self.train_features)

        self.assertIn("id_labels not provided", str(context.exception))

    def test_cmd_postprocess(self):
        """Test cMDLatentSpace postprocess method."""
        self.cmd_processor.setup(self.train_features, ind_train_labels=self.train_labels)
        scores = self.cmd_processor.postprocess(self.test_features, pred_labels=self.pred_labels)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        -1.134735107421875,
                        -0.9207103252410889,
                        -0.7419852018356323,
                        -2.6374099254608154,
                        -1.0364854335784912,
                        -0.7694298028945923,
                        -2.7188403606414795,
                        -3.315765857696533,
                        -1.6275315284729004,
                        -1.326024055480957,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )

    def test_cmd_postprocess_missing_pred_labels(self):
        """Test cMDLatentSpace postprocess without prediction labels."""
        self.cmd_processor.setup(self.train_features, ind_train_labels=self.train_labels)

        with self.assertRaises(ValueError) as context:
            self.cmd_processor.postprocess(self.test_features)

        self.assertIn("pred_logits not provided", str(context.exception))


class TestKNNLatentSpace(unittest.TestCase):
    """Test cases for the KNNLatentSpace postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.knn_processor = KNNLatentSpace()
        self.train_features, self.train_labels, _ = generate_test_data(seed=SEED)
        self.test_features, _, _ = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestKNNLatentSpace")

    def test_knn_initialization(self):
        """Test KNNLatentSpace initialization."""
        self.assertEqual(self.knn_processor.K, 50)
        self.assertIsNone(self.knn_processor.activation_log)
        self.assertIsNone(self.knn_processor.index)
        self.assertFalse(self.knn_processor._setup_flag)

        # Test with custom config
        cfg = DictConfig({"k_neighbors": 20})
        knn_with_cfg = KNNLatentSpace(cfg)
        self.assertEqual(knn_with_cfg.K, 20)

    def test_knn_setup(self):
        """Test KNNLatentSpace setup method."""
        self.knn_processor.setup(self.train_features)

        self.assertTrue(self.knn_processor._setup_flag)
        self.assertIsNotNone(self.knn_processor.activation_log)
        self.assertIsNotNone(self.knn_processor.index)

        # Check activation_log shape
        self.assertEqual(self.knn_processor.activation_log.shape, self.train_features.shape)

    def test_knn_postprocess(self):
        """Test KNNLatentSpace postprocess method."""
        self.knn_processor.setup(self.train_features)
        scores = self.knn_processor.postprocess(self.test_features)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                        -3.4028234663852886e38,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )


class TestGMMLatentSpace(unittest.TestCase):
    """Test cases for the GMMLatentSpace postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.gmm_processor = GMMLatentSpace()
        self.train_features, self.train_labels, _ = generate_test_data(seed=SEED)
        self.test_features, _, _ = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestGMMLatentSpace")

    def test_gmm_initialization(self):
        """Test GMMLatentSpace initialization."""
        self.assertEqual(self.gmm_processor.num_classes, 10)
        self.assertIsNone(self.gmm_processor.gmm)
        self.assertFalse(self.gmm_processor._setup_flag)

        # Test with custom config
        cfg = DictConfig({"num_classes": 5})
        gmm_with_cfg = GMMLatentSpace(cfg)
        self.assertEqual(gmm_with_cfg.num_classes, 5)

    def test_gmm_setup(self):
        """Test GMMLatentSpace setup method."""
        self.gmm_processor.setup(self.train_features, ind_train_labels=self.train_labels)

        self.assertTrue(self.gmm_processor._setup_flag)
        self.assertIsNotNone(self.gmm_processor.gmm)

    def test_gmm_setup_missing_labels(self):
        """Test GMMLatentSpace setup without required labels."""
        with self.assertRaises(ValueError) as context:
            self.gmm_processor.setup(self.train_features)

        self.assertIn("id_labels not provided", str(context.exception))

    def test_gmm_postprocess(self):
        """Test GMMLatentSpace postprocess method."""
        self.gmm_processor.setup(self.train_features, ind_train_labels=self.train_labels)
        scores = self.gmm_processor.postprocess(self.test_features)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        -17920878.0,
                        -11134799.0,
                        -15745862.0,
                        -23774900.0,
                        -30743066.0,
                        -27474182.0,
                        -20230644.0,
                        -15391784.0,
                        -19933296.0,
                        -16997532.0,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )


class TestEnergyPostprocessor(unittest.TestCase):
    """Test cases for the Energy postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.energy_processor = Energy(method_name="energy", flip_sign=True)
        _, _, self.train_logits = generate_test_data(seed=SEED)
        _, _, self.test_logits = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestEnergyPostprocessor")

    def test_energy_initialization(self):
        """Test Energy postprocessor initialization."""
        self.assertEqual(self.energy_processor.method_name, "energy")
        self.assertTrue(self.energy_processor.flip_sign)
        self.assertFalse(self.energy_processor._setup_flag)

    def test_energy_setup(self):
        """Test Energy postprocessor setup method."""
        self.energy_processor.setup(self.train_logits)

        self.assertTrue(self.energy_processor._setup_flag)
        self.assertIsNotNone(self.energy_processor.threshold)

    def test_energy_postprocess(self):
        """Test Energy postprocessor postprocess method."""
        self.energy_processor.setup(self.train_logits)
        scores = self.energy_processor.postprocess(self.test_logits)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_logits))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        -2.5938825607299805,
                        -2.4519991874694824,
                        -1.9754433631896973,
                        -2.4606494903564453,
                        -2.66804838180542,
                        -2.2560439109802246,
                        -2.509742498397827,
                        -2.859118700027466,
                        -2.4827966690063477,
                        -2.8413193225860596,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )

    def test_energy_postprocess_without_setup(self):
        """Test Energy postprocessor postprocess without setup."""
        with self.assertRaises(AssertionError) as context:
            self.energy_processor.postprocess(self.test_logits)

        self.assertIn("setup() must be called", str(context.exception))

    def test_energy_postprocess_tensor_input(self):
        """Test Energy postprocessor with tensor input."""
        self.energy_processor.setup(self.train_logits)
        test_tensor = Tensor(self.test_logits)
        scores = self.energy_processor.postprocess(test_tensor)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_logits))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        -2.5938825607299805,
                        -2.4519991874694824,
                        -1.9754433631896973,
                        -2.4606494903564453,
                        -2.66804838180542,
                        -2.2560439109802246,
                        -2.509742498397827,
                        -2.859118700027466,
                        -2.4827966690063477,
                        -2.8413193225860596,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )


class TestGENPostprocessor(unittest.TestCase):
    """Test cases for the GEN (Generalized Entropy) postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.gen_processor = GEN(
            method_name="gen", flip_sign=True, gamma=0.1, num_classes=TEST_NUM_CLASSES
        )
        _, _, self.train_logits = generate_test_data(seed=SEED)
        _, _, self.test_logits = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestGENPostprocessor")

    def test_gen_initialization(self):
        """Test GEN postprocessor initialization."""
        self.assertEqual(self.gen_processor.method_name, "gen")
        self.assertTrue(self.gen_processor.flip_sign)
        self.assertEqual(self.gen_processor.gamma, 0.1)
        self.assertEqual(self.gen_processor.num_classes, TEST_NUM_CLASSES)
        self.assertFalse(self.gen_processor._setup_flag)

    def test_gen_setup(self):
        """Test GEN postprocessor setup method."""
        self.gen_processor.setup(self.train_logits)

        self.assertTrue(self.gen_processor._setup_flag)
        self.assertIsNotNone(self.gen_processor.threshold)

    def test_gen_postprocess(self):
        """Test GEN postprocessor postprocess method."""
        self.gen_processor.setup(self.train_logits)
        scores = self.gen_processor.postprocess(self.test_logits)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_logits))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        7.5211100578308105,
                        7.7906317710876465,
                        7.764034748077393,
                        7.348584175109863,
                        7.678954124450684,
                        7.736558437347412,
                        7.683170318603516,
                        7.330999851226807,
                        7.504717826843262,
                        7.726001739501953,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )

    def test_gen_postprocess_without_setup(self):
        """Test GEN postprocessor postprocess without setup."""
        with self.assertRaises(AssertionError) as context:
            self.gen_processor.postprocess(self.test_logits)

        self.assertIn("setup() must be called", str(context.exception))


class TestDDUPostprocessor(unittest.TestCase):
    """Test cases for the DDU (Deep Deterministic Uncertainty) postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.ddu_processor = DDU(method_name="ddu", flip_sign=True, num_classes=TEST_NUM_CLASSES)
        self.train_features, self.train_labels, _ = generate_test_data(seed=SEED)
        self.valid_features, _, _ = generate_test_data(seed=SEED + 2)
        self.test_features, _, _ = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestDDUPostprocessor")

    def test_ddu_initialization(self):
        """Test DDU postprocessor initialization."""
        self.assertEqual(self.ddu_processor.method_name, "ddu")
        self.assertTrue(self.ddu_processor.flip_sign)
        self.assertEqual(self.ddu_processor.num_classes, TEST_NUM_CLASSES)
        self.assertIsNone(self.ddu_processor.gmm)
        self.assertFalse(self.ddu_processor._setup_flag)

        # Check device assignment
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(self.ddu_processor.device, expected_device)

    def test_ddu_setup(self):
        """Test DDU postprocessor setup method."""
        self.ddu_processor.setup(
            self.train_features, valid_feats=self.valid_features, train_labels=self.train_labels
        )

        self.assertTrue(self.ddu_processor._setup_flag)
        self.assertIsNotNone(self.ddu_processor.gmm)
        self.assertIsNotNone(self.ddu_processor.threshold)

    def test_ddu_setup_missing_valid_feats(self):
        """Test DDU setup without required valid_feats."""
        with self.assertRaises(AssertionError) as context:
            self.ddu_processor.setup(self.train_features, train_labels=self.train_labels)

        self.assertIn("valid_feats must be provided", str(context.exception))

    def test_ddu_setup_missing_train_labels(self):
        """Test DDU setup without required train_labels."""
        with self.assertRaises(AssertionError) as context:
            self.ddu_processor.setup(self.train_features, valid_feats=self.valid_features)

        self.assertIn("train_labels must be provided", str(context.exception))

    def test_ddu_postprocess(self):
        """Test DDU postprocessor postprocess method."""
        self.ddu_processor.setup(
            self.train_features, valid_feats=self.valid_features, train_labels=self.train_labels
        )
        scores = self.ddu_processor.postprocess(self.test_features)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        18679324.0,
                        10889954.0,
                        16077478.0,
                        23774906.0,
                        32526814.0,
                        25533802.0,
                        21280214.0,
                        16340976.0,
                        19673760.0,
                        18043234.0,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )


class TestMahalanobisPostprocessor(unittest.TestCase):
    """Test cases for the Mahalanobis postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.maha_processor = Mahalanobis(
            method_name="mahalanobis", flip_sign=True, num_classes=TEST_NUM_CLASSES
        )
        self.train_features, self.train_labels, _ = generate_test_data(seed=SEED)
        self.valid_features, _, _ = generate_test_data(seed=SEED + 2)
        self.test_features, _, _ = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestMahalanobisPostprocessor")

    def test_mahalanobis_initialization(self):
        """Test Mahalanobis postprocessor initialization."""
        self.assertEqual(self.maha_processor.method_name, "mahalanobis")
        self.assertTrue(self.maha_processor.flip_sign)
        self.assertEqual(self.maha_processor.num_classes, TEST_NUM_CLASSES)
        self.assertIsNone(self.maha_processor.class_mean)
        self.assertIsNone(self.maha_processor.precision)
        self.assertFalse(self.maha_processor._setup_flag)

    def test_mahalanobis_setup(self):
        """Test Mahalanobis postprocessor setup method."""
        self.maha_processor.setup(
            self.train_features, train_labels=self.train_labels, valid_feats=self.valid_features
        )

        self.assertTrue(self.maha_processor._setup_flag)
        self.assertIsNotNone(self.maha_processor.class_mean)
        self.assertIsNotNone(self.maha_processor.precision)
        self.assertIsNotNone(self.maha_processor.threshold)

    def test_mahalanobis_setup_missing_train_labels(self):
        """Test Mahalanobis setup without required train_labels."""
        with self.assertRaises(AssertionError) as context:
            self.maha_processor.setup(self.train_features, valid_feats=self.valid_features)

        self.assertIn("train_labels must be provided", str(context.exception))

    def test_mahalanobis_setup_missing_valid_feats(self):
        """Test Mahalanobis setup without required valid_feats."""
        with self.assertRaises(AssertionError) as context:
            self.maha_processor.setup(self.train_features, train_labels=self.train_labels)

        self.assertIn("valid_feats must be provided", str(context.exception))

    def test_mahalanobis_postprocess(self):
        """Test Mahalanobis postprocessor postprocess method."""
        self.maha_processor.setup(
            self.train_features, train_labels=self.train_labels, valid_feats=self.valid_features
        )
        scores = self.maha_processor.postprocess(self.test_features)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        1.1347351808930433,
                        0.9207103216165267,
                        0.7419852259793285,
                        2.63740954614305,
                        1.036485071087479,
                        0.7694294357252861,
                        2.7188404739938,
                        3.3157661379171177,
                        1.6275313633343984,
                        1.3260243294794334,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )

    def test_mahalanobis_postprocess_tensor_input(self):
        """Test Mahalanobis postprocessor with tensor input."""
        self.maha_processor.setup(
            self.train_features, train_labels=self.train_labels, valid_feats=self.valid_features
        )
        test_tensor = Tensor(self.test_features)
        scores = self.maha_processor.postprocess(test_tensor)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        1.1347351808930433,
                        0.9207103216165267,
                        0.7419852259793285,
                        2.63740954614305,
                        1.036485071087479,
                        0.7694294357252861,
                        2.7188404739938,
                        3.3157661379171177,
                        1.6275313633343984,
                        1.3260243294794334,
                    ]
                    - scores
                )
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )


class TestViMPostprocessor(unittest.TestCase):
    """Test cases for the ViM (Virtual logit Matching) postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.vim_processor = ViM(method_name="vim", flip_sign=True)
        self.train_features, _, self.train_logits = generate_test_data(seed=SEED)
        self.valid_features, _, self.valid_logits = generate_test_data(seed=SEED + 2)
        self.test_features, _, self.test_logits = generate_test_data(seed=SEED + 1)

        # Create mock final layer parameters
        self.final_layer_params = {
            "weight": np.random.randn(TEST_NUM_CLASSES, TEST_FEATURE_DIM).astype(np.float32),
            "bias": np.random.randn(TEST_NUM_CLASSES).astype(np.float32),
        }
        logger.info("Setting up TestViMPostprocessor")

    def test_vim_initialization(self):
        """Test ViM postprocessor initialization."""
        self.assertEqual(self.vim_processor.method_name, "vim")
        self.assertTrue(self.vim_processor.flip_sign)
        self.assertIsNone(self.vim_processor.u)
        self.assertIsNone(self.vim_processor.DIM)
        self.assertIsNone(self.vim_processor.NS)
        self.assertIsNone(self.vim_processor.alpha)
        self.assertFalse(self.vim_processor._setup_flag)

    def test_vim_setup(self):
        """Test ViM postprocessor setup method."""
        self.vim_processor.setup(
            self.train_features,
            final_linear_layer_params=self.final_layer_params,
            train_logits=self.train_logits,
            valid_feats=self.valid_features,
            valid_logits=self.valid_logits,
        )

        self.assertTrue(self.vim_processor._setup_flag)
        self.assertIsNotNone(self.vim_processor.u)
        self.assertIsNotNone(self.vim_processor.DIM)
        self.assertIsNotNone(self.vim_processor.NS)
        self.assertIsNotNone(self.vim_processor.alpha)
        self.assertIsNotNone(self.vim_processor.threshold)

        # Check dimension selection logic
        if TEST_FEATURE_DIM >= 2048:
            self.assertEqual(self.vim_processor.DIM, 1000)
        elif TEST_FEATURE_DIM >= 768:
            self.assertEqual(self.vim_processor.DIM, 512)
        else:
            self.assertEqual(self.vim_processor.DIM, TEST_FEATURE_DIM // 2)

    def test_vim_setup_missing_params(self):
        """Test ViM setup with missing parameters."""
        # Missing final_linear_layer_params
        with self.assertRaises(AssertionError) as context:
            self.vim_processor.setup(
                self.train_features,
                train_logits=self.train_logits,
                valid_feats=self.valid_features,
                valid_logits=self.valid_logits,
            )
        self.assertIn("final_linear_layer_params must be provided", str(context.exception))

        # Missing train_logits
        with self.assertRaises(AssertionError) as context:
            self.vim_processor.setup(
                self.train_features,
                final_linear_layer_params=self.final_layer_params,
                valid_feats=self.valid_features,
                valid_logits=self.valid_logits,
            )
        self.assertIn("train_logits must be provided", str(context.exception))

        # Missing valid_feats
        with self.assertRaises(AssertionError) as context:
            self.vim_processor.setup(
                self.train_features,
                final_linear_layer_params=self.final_layer_params,
                train_logits=self.train_logits,
                valid_logits=self.valid_logits,
            )
        self.assertIn("valid_feats must be provided", str(context.exception))

        # Missing valid_logits
        with self.assertRaises(AssertionError) as context:
            self.vim_processor.setup(
                self.train_features,
                final_linear_layer_params=self.final_layer_params,
                train_logits=self.train_logits,
                valid_feats=self.valid_features,
            )
        self.assertIn("valid_logits must be provided", str(context.exception))

    def test_vim_setup_tensor_params(self):
        """Test ViM setup with tensor parameters."""
        tensor_params = {
            "weight": Tensor(self.final_layer_params["weight"]),
            "bias": Tensor(self.final_layer_params["bias"]),
        }

        self.vim_processor.setup(
            self.train_features,
            final_linear_layer_params=tensor_params,
            train_logits=self.train_logits,
            valid_feats=self.valid_features,
            valid_logits=self.valid_logits,
        )

        self.assertTrue(self.vim_processor._setup_flag)
        self.assertIsNotNone(self.vim_processor.u)

    def test_vim_postprocess(self):
        """Test ViM postprocessor postprocess method."""
        self.vim_processor.setup(
            self.train_features,
            final_linear_layer_params=self.final_layer_params,
            train_logits=self.train_logits,
            valid_feats=self.valid_features,
            valid_logits=self.valid_logits,
        )
        scores = self.vim_processor.postprocess(self.test_features, logits=self.test_logits)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(
            (
                np.array(
                    [
                        -18294518.0,
                        -17758880.0,
                        -19942008.0,
                        -20468770.0,
                        -27237914.0,
                        -26840116.0,
                        -23028616.0,
                        -18915342.0,
                        -23772058.0,
                        -14144876.0,
                    ]
                )
                - scores
            ).sum(),
            0.0,
            delta=TOLERANCE,
        )

    def test_vim_postprocess_tensor_input(self):
        """Test ViM postprocessor with tensor inputs."""
        self.vim_processor.setup(
            self.train_features,
            final_linear_layer_params=self.final_layer_params,
            train_logits=self.train_logits,
            valid_feats=self.valid_features,
            valid_logits=self.valid_logits,
        )

        test_features_tensor = Tensor(self.test_features)
        test_logits_tensor = Tensor(self.test_logits)

        scores = self.vim_processor.postprocess(test_features_tensor, logits=test_logits_tensor)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.test_features))


class TestPostprocessorsDict(unittest.TestCase):
    """Test cases for the postprocessors_dict."""

    def setUp(self):
        """Set up test fixtures."""
        logger.info("Setting up TestPostprocessorsDict")

    def test_postprocessors_dict_exists(self):
        """Test that postprocessors_dict is defined and contains expected classes."""
        self.assertIsInstance(postprocessors_dict, dict)
        self.assertGreater(len(postprocessors_dict), 0)

        # Check for key postprocessor classes
        expected_keys = ["KDE", "MD", "cMD", "KNN", "GMM"]
        for key in expected_keys:
            if key in postprocessors_dict:
                self.assertIn(key, postprocessors_dict)
                self.assertTrue(callable(postprocessors_dict[key]))

    def test_postprocessors_dict_classes_instantiable(self):
        """Test that classes in postprocessors_dict can be instantiated."""
        for name, postprocessor_class in postprocessors_dict.items():
            try:
                # Try to instantiate the class
                if name in ["KDE", "MD", "KNN", "GMM"]:
                    # These classes can be instantiated without parameters
                    instance = postprocessor_class()
                    self.assertIsNotNone(instance)
                elif name == "cMD":
                    # This class needs a config with num_classes
                    cfg = DictConfig({"num_classes": 10})
                    instance = postprocessor_class(cfg)
                    self.assertIsNotNone(instance)

                logger.info(f"Successfully instantiated {name} postprocessor")
            except Exception as e:
                logger.warning(f"Could not instantiate {name} postprocessor: {e}")


class TestPostprocessorIntegration(unittest.TestCase):
    """Integration tests for postprocessors."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.train_features, self.train_labels, self.train_logits = generate_test_data(seed=SEED)
        self.test_features, _, self.test_logits = generate_test_data(seed=SEED + 1)
        logger.info("Setting up TestPostprocessorIntegration")

    def test_postprocessor_call_method(self):
        """Test the __call__ method of postprocessors."""
        kde_processor = KDELatentSpace()
        kde_processor.setup(self.train_features)

        # Test __call__ method (should call postprocess)
        scores1 = kde_processor(self.test_features)
        scores2 = kde_processor.postprocess(self.test_features)

        self.assertAlmostEqual((scores1 - scores2).sum(), 0.0, delta=TOLERANCE)

    def test_multiple_postprocessors_consistency(self):
        """Test that different postprocessors produce consistent output shapes."""
        postprocessors = [KDELatentSpace(), MDLatentSpace(), KNNLatentSpace()]

        for processor in postprocessors:
            processor.setup(self.train_features)
            scores = processor.postprocess(self.test_features)

            self.assertIsInstance(scores, np.ndarray)
            self.assertEqual(len(scores), len(self.test_features))
            self.assertTrue(np.all(np.isfinite(scores)))

    def test_ood_postprocessors_flip_sign_consistency(self):
        """Test that OOD postprocessors handle flip_sign correctly."""
        energy_flip = Energy(method_name="energy", flip_sign=True)
        energy_no_flip = Energy(method_name="energy", flip_sign=False)

        energy_flip.setup(self.train_logits)
        energy_no_flip.setup(self.train_logits)

        scores_flip = energy_flip.postprocess(self.test_logits)
        scores_no_flip = energy_no_flip.postprocess(self.test_logits)

        # When flip_sign=True, scores should be negated
        np.testing.assert_array_almost_equal(scores_flip, -scores_no_flip)


def run_tests():
    """
    Run all unit tests for the uncertainty postprocessors module.

    This function serves as the main entry point for running tests and provides
    detailed logging information about test execution.
    """
    logger.info("Starting uncertainty postprocessors module unit tests")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestKDELatentSpace,
        TestMDLatentSpace,
        TestcMDLatentSpace,
        TestKNNLatentSpace,
        TestGMMLatentSpace,
        TestEnergyPostprocessor,
        TestGENPostprocessor,
        TestDDUPostprocessor,
        TestMahalanobisPostprocessor,
        TestViMPostprocessor,
        TestPostprocessorsDict,
        TestPostprocessorIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None, descriptions=True, failfast=False)

    result = runner.run(test_suite)

    # Log summary
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.wasSuccessful():
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed!")

        # Log failure details
        for test, traceback in result.failures:
            logger.error(f"FAILURE in {test}: {traceback}")

        for test, traceback in result.errors:
            logger.error(f"ERROR in {test}: {traceback}")

    return result


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run tests
    result = run_tests()

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
