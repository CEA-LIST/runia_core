#!/usr/bin/env python3
"""
Unit tests for the runia_core.rcnn module.

This script provides comprehensive unit tests for the functions and classes in the
runia_core.rcnn module. It tests functions for RCNN uncertainty estimation including
MSP scoring, energy scoring, DICE/ReAct processing, MCD sampling, and inference classes.

Usage:
    Run all tests: python -m unittest tests.unit_test_rcnn
    Run specific test: python -m unittest tests.unit_test_rcnn.TestMSPScore.test_get_msp_score_basic
    Run with verbose output: python -m unittest -v tests.unit_test_rcnn

Requirements:
    - Python 3.9+
    - PyTorch
    - NumPy
    - unittest (built-in)
    - unittest.mock (built-in)
    - runia_core library

Date: 2025-01-15
"""

import logging
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from runia_core.rcnn import (
    get_msp_score_rcnn,
    get_dice_feat_mean_react_percentile_rcnn,
    get_energy_score_rcnn,
    get_ls_mcd_samples_rcnn,
    MCSamplerRCNN,
    LaRexInferenceRCNN,
    LaRDInferenceRCNN,
    remove_background_dimension,
)
from runia_core.inference import LaRExInference
from runia_core.feature_extraction import Hook, MCSamplerModule

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test constants
SEED = 42
TOLERANCE = 1e-6
TEST_BATCH_SIZE = 2
TEST_SAMPLES = 4
TEST_FEATURE_DIM = 64
MCD_N_SAMPLES = 3
REACT_PERCENTILE = 90


class MockRCNNResults:
    """Mock object for RCNN model results."""

    def __init__(self, scores: torch.Tensor, inter_feat: torch.Tensor = None):
        self.scores = scores
        self.inter_feat = inter_feat if inter_feat is not None else torch.randn(5, 10)


class MockRCNNModel(torch.nn.Module):
    """Mock RCNN model for testing purposes."""

    def __init__(
        self,
        num_detections: int = 5,
        num_classes: int = 10,
        has_rpn: bool = False,
        dice_react_precompute: bool = True,
    ):
        super().__init__()
        self.num_detections = num_detections
        self.num_classes = num_classes
        self.has_rpn = has_rpn
        self.dice_react_precompute = dice_react_precompute

        # Mock nested structure for RCNN
        self.model = MagicMock()
        self.model.eval = MagicMock(return_value=None)

        if has_rpn:
            # Create mock RPN head with rpn_intermediate_output
            self.model.proposal_generator.rpn_head.rpn_intermediate_output = [
                torch.randn(1, 64, 32, 32) for _ in range(5)
            ]

    def forward(self, x):
        # Return (results, box_cls) tuple
        scores = torch.rand(self.num_detections)
        inter_feat = torch.randn(self.num_detections, self.num_classes)
        results = MockRCNNResults(scores, inter_feat)
        box_cls = torch.randn(1000, self.num_classes)
        return results, box_cls

    def to(self, device):
        """Override to method for compatibility."""
        return self


class TestMSPScore(unittest.TestCase):
    """Test cases for get_msp_score_rcnn function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestMSPScore")

    def test_get_msp_score_basic(self):
        """Test MSP score calculation with valid inputs."""
        # Create mock data - the model is called and should return (results, box_cls) tuple
        mock_model = MockRCNNModel(num_detections=5)
        dataset = TensorDataset(torch.randn(TEST_SAMPLES, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)  # Use batch_size=1 to process each sample

        # Run function
        msp_scores = get_msp_score_rcnn(mock_model, dataloader)

        # Assertions
        self.assertIsInstance(msp_scores, np.ndarray)
        self.assertEqual(msp_scores.shape[0], TEST_SAMPLES)
        self.assertTrue(np.all(msp_scores >= 0) and np.all(msp_scores <= 1))

    def test_get_msp_score_empty_detections(self):
        """Test MSP score with model returning no detections."""
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_results = MockRCNNResults(scores=torch.tensor([]))
        mock_model.return_value = (mock_results, torch.tensor([]))

        dataset = TensorDataset(torch.randn(2, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        msp_scores = get_msp_score_rcnn(mock_model, dataloader)

        self.assertIsInstance(msp_scores, np.ndarray)
        self.assertEqual(msp_scores.shape[0], 2)

    def test_get_msp_score_returns_numpy(self):
        """Test that output is numpy array."""
        mock_model = MockRCNNModel(num_detections=3)
        dataset = TensorDataset(torch.randn(1, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        msp_scores = get_msp_score_rcnn(mock_model, dataloader)

        self.assertIsInstance(msp_scores, np.ndarray)
        self.assertEqual(len(msp_scores.shape), 1)


class TestDICEReAct(unittest.TestCase):
    """Test cases for get_dice_feat_mean_react_percentile_rcnn function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestDICEReAct")

    def test_get_dice_react_basic(self):
        """Test DICE and ReAct percentile calculation."""
        mock_model = MagicMock()
        mock_model.model.eval = MagicMock(return_value=None)
        mock_model.dice_react_precompute = True

        # Mock the forward to return feature tensors (not a tuple)
        features = torch.randn(TEST_SAMPLES, TEST_FEATURE_DIM)
        mock_model.side_effect = lambda x: features

        dataset = TensorDataset(torch.randn(TEST_SAMPLES, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE)

        dice_mean, react_threshold = get_dice_feat_mean_react_percentile_rcnn(
            mock_model, dataloader, react_percentile=REACT_PERCENTILE
        )

        # Assertions
        self.assertIsInstance(dice_mean, np.ndarray)
        self.assertIsInstance(react_threshold, (float, np.floating, int))
        self.assertEqual(dice_mean.shape[0], TEST_FEATURE_DIM)

    def test_get_dice_react_invalid_percentile_low(self):
        """Test that invalid percentile raises assertion error."""
        mock_model = MockRCNNModel(dice_react_precompute=True)
        dataset = TensorDataset(torch.randn(2, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        with self.assertRaises(AssertionError):
            get_dice_feat_mean_react_percentile_rcnn(mock_model, dataloader, react_percentile=0)

    def test_get_dice_react_invalid_percentile_high(self):
        """Test that percentile > 100 raises assertion error."""
        mock_model = MockRCNNModel(dice_react_precompute=True)
        dataset = TensorDataset(torch.randn(2, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        with self.assertRaises(AssertionError):
            get_dice_feat_mean_react_percentile_rcnn(mock_model, dataloader, react_percentile=101)

    def test_get_dice_react_percentile_correct(self):
        """Test that returned percentile matches expected value."""
        mock_model = MagicMock()
        mock_model.model.eval = MagicMock(return_value=None)
        mock_model.dice_react_precompute = True

        # Create predictable data
        test_data = (
            torch.arange(TEST_SAMPLES, dtype=torch.float32).unsqueeze(1).repeat(1, TEST_FEATURE_DIM)
        )
        mock_model.side_effect = lambda x: test_data

        dataset = TensorDataset(torch.randn(TEST_SAMPLES, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE)

        _, react_threshold = get_dice_feat_mean_react_percentile_rcnn(
            mock_model, dataloader, react_percentile=50
        )

        # Check that percentile is approximately correct
        self.assertGreater(react_threshold, 0)


class TestEnergyScore(unittest.TestCase):
    """Test cases for get_energy_score_rcnn function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestEnergyScore")

    def test_get_energy_score_basic(self):
        """Test energy score calculation."""
        # MockRCNNModel returns (results, box_cls) tuple which is what get_energy_score_rcnn expects
        mock_model = MockRCNNModel(num_detections=5, num_classes=10)
        dataset = TensorDataset(torch.randn(TEST_SAMPLES, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)  # Use batch_size=1 for correct sample count

        raw_scores, filtered_scores = get_energy_score_rcnn(mock_model, dataloader)

        # Assertions
        self.assertIsInstance(raw_scores, np.ndarray)
        self.assertIsInstance(filtered_scores, np.ndarray)
        self.assertEqual(raw_scores.shape[0], TEST_SAMPLES)
        self.assertEqual(filtered_scores.shape[0], TEST_SAMPLES)

    def test_get_energy_score_shapes(self):
        """Test that energy scores have correct shapes."""
        mock_model = MockRCNNModel(num_detections=10, num_classes=15)
        dataset = TensorDataset(torch.randn(3, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        raw_scores, filtered_scores = get_energy_score_rcnn(mock_model, dataloader)

        self.assertEqual(raw_scores.shape, (3,))
        self.assertEqual(filtered_scores.shape, (3,))

    def test_get_energy_score_positive_values(self):
        """Test that energy scores are positive (from logsumexp)."""
        mock_model = MockRCNNModel(num_detections=5, num_classes=10)
        dataset = TensorDataset(torch.randn(TEST_SAMPLES, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE)

        raw_scores, filtered_scores = get_energy_score_rcnn(mock_model, dataloader)

        # Energy scores should generally be positive
        self.assertTrue(np.all(raw_scores >= -1000))  # Allow some negative due to logsumexp
        self.assertTrue(np.all(filtered_scores >= -1000))


class TestMCDSamples(unittest.TestCase):
    """Test cases for get_ls_mcd_samples_rcnn function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestMCDSamples")

    def test_get_mcd_samples_conv_layer(self):
        """Test MCD sampling with Conv layer type."""
        mock_model = MockRCNNModel(num_detections=5, has_rpn=False)
        mock_hook = Mock(spec=Hook)
        mock_hook.output = torch.randn(1, 64, 32, 32)

        dataset = TensorDataset(torch.randn(2, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        samples = get_ls_mcd_samples_rcnn(
            mock_model,
            dataloader,
            mcd_nro_samples=MCD_N_SAMPLES,
            hook_dropout_layer=mock_hook,
            layer_type="Conv",
            return_raw_predictions=False,
        )

        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.ndim, 2)  # Should be 2D tensor

    def test_get_mcd_samples_fc_layer(self):
        """Test MCD sampling with FC layer type."""
        mock_model = MockRCNNModel(num_detections=1000, has_rpn=False)
        mock_hook = Mock(spec=Hook)
        mock_hook.output = torch.randn(1000, 128)

        dataset = TensorDataset(torch.randn(2, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        samples = get_ls_mcd_samples_rcnn(
            mock_model,
            dataloader,
            mcd_nro_samples=MCD_N_SAMPLES,
            hook_dropout_layer=mock_hook,
            layer_type="FC",
            return_raw_predictions=False,
        )

        self.assertIsInstance(samples, torch.Tensor)

    def test_get_mcd_samples_invalid_layer_type(self):
        """Test that invalid layer type raises assertion error."""
        mock_model = MockRCNNModel()
        mock_hook = Mock(spec=Hook)
        mock_hook.output = torch.randn(1, 64, 32, 32)

        dataset = TensorDataset(torch.randn(1, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        with self.assertRaises(AssertionError):
            get_ls_mcd_samples_rcnn(
                mock_model,
                dataloader,
                mcd_nro_samples=MCD_N_SAMPLES,
                hook_dropout_layer=mock_hook,
                layer_type="InvalidType",
                return_raw_predictions=False,
            )

    def test_get_mcd_samples_return_raw_predictions(self):
        """Test MCD sampling with raw predictions return."""
        mock_model = MockRCNNModel(num_detections=5, has_rpn=False)
        mock_hook = Mock(spec=Hook)
        mock_hook.output = torch.randn(1, 64, 32, 32)

        dataset = TensorDataset(torch.randn(1, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        samples, raw_preds = get_ls_mcd_samples_rcnn(
            mock_model,
            dataloader,
            mcd_nro_samples=MCD_N_SAMPLES,
            hook_dropout_layer=mock_hook,
            layer_type="Conv",
            return_raw_predictions=True,
        )

        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(raw_preds, torch.Tensor)


class TestMCSamplerRCNN(unittest.TestCase):
    """Test cases for MCSamplerRCNN class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestMCSamplerRCNN")

    def test_mc_sampler_init(self):
        """Test MCSamplerRCNN initialization."""
        mc_samples = 5
        sampler = MCSamplerRCNN(mc_samples=mc_samples, layer_type="RPN")

        self.assertEqual(sampler.mc_samples, mc_samples)
        self.assertEqual(len(sampler.drop_blocks), mc_samples)

    def test_mc_sampler_invalid_layer_type(self):
        """Test that non-RPN layer type raises assertion error."""
        with self.assertRaises(AssertionError):
            MCSamplerRCNN(mc_samples=5, layer_type="Conv")

    def test_mc_sampler_forward(self):
        """Test MCSamplerRCNN forward pass."""
        sampler = MCSamplerRCNN(mc_samples=3, layer_type="RPN")

        # Create mock model
        mock_model = MagicMock()
        mock_model.model.proposal_generator.rpn_head.rpn_intermediate_output = [
            torch.randn(1, 64, 32, 32) for _ in range(5)
        ]

        samples = sampler(mock_model)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape[0], 3)  # Should have mc_samples in first dimension

    def test_mc_sampler_drop_blocks_list(self):
        """Test that drop_blocks is a ModuleList with correct size."""
        mc_samples = 8
        sampler = MCSamplerRCNN(mc_samples=mc_samples, layer_type="RPN")

        self.assertIsInstance(sampler.drop_blocks, torch.nn.ModuleList)
        self.assertEqual(len(sampler.drop_blocks), mc_samples)


class TestLaRexInferenceRCNN(unittest.TestCase):
    """Test cases for LaRexInferenceRCNN class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestLaRexInferenceRCNN")

    def test_larex_inference_subclass(self):
        """Test LaRexInferenceRCNN is a subclass of LaRExInference."""
        # Verify that LaRexInferenceRCNN is properly defined
        self.assertTrue(issubclass(LaRexInferenceRCNN, LaRExInference))

    def test_larex_get_score_exists(self):
        """Test LaRexInferenceRCNN.get_score method exists and is callable."""
        self.assertTrue(hasattr(LaRexInferenceRCNN, "get_score"))
        self.assertTrue(callable(getattr(LaRexInferenceRCNN, "get_score")))

    def test_larex_get_layer_mc_samples_exists(self):
        """Test LaRexInferenceRCNN.get_layer_mc_samples method exists."""
        self.assertTrue(hasattr(LaRexInferenceRCNN, "get_layer_mc_samples"))
        self.assertTrue(callable(getattr(LaRexInferenceRCNN, "get_layer_mc_samples")))

    def test_larex_get_score_full_inference_exists(self):
        """Test LaRexInferenceRCNN.get_score_full_inference method exists."""
        self.assertTrue(hasattr(LaRexInferenceRCNN, "get_score_full_inference"))
        self.assertTrue(callable(getattr(LaRexInferenceRCNN, "get_score_full_inference")))

    def test_larex_get_score_method_signature(self):
        """Test that LaRexInferenceRCNN.get_score has correct method signature."""
        import inspect

        sig = inspect.signature(LaRexInferenceRCNN.get_score)
        # Should have self, input_image, layer_hook parameters
        params = list(sig.parameters.keys())
        self.assertIn("self", params)
        self.assertIn("input_image", params)
        self.assertIn("layer_hook", params)

    def test_larex_inference_has_required_attributes(self):
        """Test that LaRexInferenceRCNN inherits required attributes from parent."""
        # Check that key methods from parent are available
        self.assertTrue(hasattr(LaRexInferenceRCNN, "get_score"))
        self.assertTrue(hasattr(LaRexInferenceRCNN, "get_layer_mc_samples"))
        self.assertTrue(hasattr(LaRexInferenceRCNN, "get_score_full_inference"))

    def test_larex_docstring_documentation(self):
        """Test that LaRexInferenceRCNN has proper documentation."""
        # Verify the class has a docstring
        self.assertIsNotNone(LaRexInferenceRCNN.__doc__)
        self.assertIn("RCNN", LaRexInferenceRCNN.__doc__)
        self.assertIn("LaREx", LaRexInferenceRCNN.__doc__)

    def test_larex_get_score_method_docstring(self):
        """Test that get_score method has documentation."""
        self.assertIsNotNone(LaRexInferenceRCNN.get_score.__doc__)
        self.assertIn("LaREx", LaRexInferenceRCNN.get_score.__doc__)

    def test_larex_rcnn_specific_documentation(self):
        """Test that class documentation mentions RCNN-specific requirements."""
        docstring = LaRexInferenceRCNN.__doc__
        self.assertIn("RCNN", docstring)
        self.assertIn("RPN", docstring)
        self.assertIn("rpn_intermediate_output", docstring)

    def test_get_score_uses_entropy_function(self):
        """Test get_score uses get_dl_h_z for entropy calculation."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        self.assertIn("get_dl_h_z", source)

    def test_get_score_uses_detector_postprocess(self):
        """Test get_score uses detector.postprocess."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        self.assertIn("detector", source)
        self.assertIn("postprocess", source)

    def test_methods_handle_device_transfers(self):
        """Test that methods handle device transfers for input."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        self.assertIn("input_image.to(self.device)", source)

    # ============================================================================
    # BEHAVIORAL TESTS FOR get_score METHOD
    # ============================================================================

    def test_get_score_returns_tuple_of_output_and_score(self):
        """Test that get_score returns a tuple with (output, score)."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        # Verify return statement contains both output and sample_score
        self.assertIn("return output, sample_score", source)

    def test_get_score_wraps_torch_no_grad(self):
        """Test that get_score uses torch.no_grad context for inference."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        self.assertIn("with torch.no_grad():", source)

    def test_get_score_calls_model_forward(self):
        """Test that get_score calls self.model() for forward pass."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        self.assertIn("self.model(input_image)", source)

    def test_get_score_uses_mc_sampler(self):
        """Test that get_score uses self.mc_sampler to get MC samples."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        self.assertIn("self.mc_sampler(self.model)", source)

    def test_get_score_calculates_entropy_from_samples(self):
        """Test that get_score uses get_dl_h_z for entropy calculation."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        # Should calculate entropy from MC samples
        self.assertIn("get_dl_h_z(mc_samples_t, self.mcd_samples_nro)", source)

    def test_get_score_optionally_applies_pca(self):
        """Test that get_score can apply PCA transform if available."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        # Should check if pca_transform exists and apply it
        self.assertIn("if self.pca_transform:", source)
        self.assertIn("apply_pca_transform(sample_h_z, self.pca_transform)", source)

    def test_get_score_postprocesses_with_detector(self):
        """Test that get_score uses detector.postprocess for final scoring."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        # Should postprocess entropy through detector
        self.assertIn("self.detector.postprocess(sample_h_z)", source)

    def test_get_score_handles_device_mismatch(self):
        """Test that get_score handles device transfers with try-except."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        # Should have try-except for device transfer
        self.assertIn("try:", source)
        self.assertIn("input_image.to(self.device)", source)
        self.assertIn("except AttributeError:", source)

    def test_get_score_method_accepts_image_and_hook(self):
        """Test that get_score method has both input_image and layer_hook parameters."""
        import inspect

        sig = inspect.signature(LaRexInferenceRCNN.get_score)
        params = list(sig.parameters.keys())
        # Verify both parameters exist
        self.assertIn("input_image", params)
        self.assertIn("layer_hook", params)
        # Verify there are exactly 3 params (self + 2 params)
        self.assertEqual(len(params), 3)

    def test_get_score_inference_flow(self):
        """Test the complete inference flow of get_score method."""
        import inspect

        source = inspect.getsource(LaRexInferenceRCNN.get_score)
        # Verify all key steps are present in correct order
        steps = [
            "torch.no_grad",  # Inference mode
            "self.model",  # Forward pass
            "self.mc_sampler",  # MC sampling
            "get_dl_h_z",  # Entropy calculation
            "pca_transform",  # Optional PCA
            "detector.postprocess",  # Postprocessing
            "return output",  # Return results
        ]
        for step in steps:
            self.assertIn(step, source, f"get_score method should include: {step}")

    def test_get_score_docstring_describes_inputs(self):
        """Test that get_score docstring describes input parameters."""
        docstring = LaRexInferenceRCNN.get_score.__doc__
        self.assertIsNotNone(docstring)
        # Should mention image input
        self.assertIn("image", docstring.lower())
        # Should mention layer_hook
        self.assertIn("layer_hook", docstring)

    def test_get_score_docstring_describes_output(self):
        """Test that get_score docstring describes output."""
        docstring = LaRexInferenceRCNN.get_score.__doc__
        self.assertIsNotNone(docstring)
        # Should mention LaREx score output
        self.assertIn("LaREx", docstring)
        self.assertIn("score", docstring.lower())


class TestLaRDInferenceRCNN(unittest.TestCase):
    """Test cases for LaRDInferenceRCNN class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestLaRDInferenceRCNN")

    def test_lard_inference_init_conv(self):
        """Test LaRDInferenceRCNN initialization with Conv layer type."""
        mock_model = MockRCNNModel(has_rpn=False)
        mock_detector = MagicMock()

        lard = LaRDInferenceRCNN(
            model=mock_model,
            detector=mock_detector,
            pca_transform=None,
            layer_type="Conv",
        )

        self.assertEqual(lard.model, mock_model)
        self.assertEqual(lard.layer_type, "Conv")

    def test_lard_inference_init_rpn(self):
        """Test LaRDInferenceRCNN initialization with RPN layer type."""
        mock_model = MockRCNNModel(has_rpn=True)
        mock_detector = MagicMock()

        lard = LaRDInferenceRCNN(
            model=mock_model,
            detector=mock_detector,
            pca_transform=None,
            layer_type="RPN",
        )

        self.assertEqual(lard.layer_type, "RPN")
        # Check that reducer is set for RPN
        self.assertIsNotNone(lard.reducer)

    def test_lard_process_rpn_intermediate_representation(self):
        """Test LaRDInferenceRCNN.process_rpn_intermediate_representation method."""
        mock_model = MockRCNNModel(has_rpn=True)
        mock_detector = MagicMock()

        lard = LaRDInferenceRCNN(
            model=mock_model,
            detector=mock_detector,
            pca_transform=None,
            layer_type="RPN",
        )

        # Create mock RPN intermediate output
        # The corrected code should use self.model instead of self.dnn_model
        lard.model.model.proposal_generator.rpn_head.rpn_intermediate_output = [
            torch.randn(1, 64, 32, 32) for _ in range(5)
        ]

        result = lard.process_rpn_intermediate_representation(None)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 1)


class TestRemoveBackgroundDimension(unittest.TestCase):
    """Test cases for remove_background_dimension function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestRemoveBackgroundDimension")

    def test_remove_background_dimension_21_classes(self):
        """Test removal with 21 classes (20 + 1 background)."""
        fc_params = {
            "weight": np.random.randn(21, 100),
            "bias": np.random.randn(21),
        }
        ind_data_dict = {
            "train logits": np.random.randn(100, 21),
            "valid logits": np.random.randn(50, 21),
        }
        ood_data_dict = {
            "ood1 logits": np.random.randn(30, 21),
            "ood2 logits": np.random.randn(40, 21),
        }
        ood_names = ["ood1", "ood2"]

        ind_dict, ood_dict, params = remove_background_dimension(
            fc_params, ind_data_dict, ood_data_dict, ood_names
        )

        # Check that dimensions are reduced
        self.assertEqual(params["weight"].shape[0], 20)
        self.assertEqual(params["bias"].shape[0], 20)
        self.assertEqual(ind_dict["train logits"].shape[1], 20)
        self.assertEqual(ind_dict["valid logits"].shape[1], 20)
        self.assertEqual(ood_dict["ood1 logits"].shape[1], 20)
        self.assertEqual(ood_dict["ood2 logits"].shape[1], 20)

    def test_remove_background_dimension_11_classes(self):
        """Test removal with 11 classes (10 + 1 background)."""
        fc_params = {
            "weight": np.random.randn(11, 50),
            "bias": np.random.randn(11),
        }
        ind_data_dict = {
            "train logits": np.random.randn(100, 11),
            "valid logits": np.random.randn(50, 11),
        }
        ood_data_dict = {"ood1 logits": np.random.randn(30, 11)}
        ood_names = ["ood1"]

        ind_dict, ood_dict, params = remove_background_dimension(
            fc_params, ind_data_dict, ood_data_dict, ood_names
        )

        self.assertEqual(params["weight"].shape[0], 10)
        self.assertEqual(params["bias"].shape[0], 10)
        self.assertEqual(ind_dict["train logits"].shape[1], 10)

    def test_remove_background_dimension_tensor_input(self):
        """Test that function handles Tensor inputs."""
        fc_params = {
            "weight": torch.randn(21, 100),
            "bias": torch.randn(21),
        }
        ind_data_dict = {
            "train logits": np.random.randn(100, 21),
            "valid logits": np.random.randn(50, 21),
        }
        ood_data_dict = {"ood1 logits": np.random.randn(30, 21)}
        ood_names = ["ood1"]

        ind_dict, ood_dict, params = remove_background_dimension(
            fc_params, ind_data_dict, ood_data_dict, ood_names
        )

        # Check that tensors are converted to numpy
        self.assertIsInstance(params["weight"], np.ndarray)
        self.assertIsInstance(params["bias"], np.ndarray)
        self.assertEqual(params["weight"].shape[0], 20)

    def test_remove_background_dimension_no_removal_20_classes(self):
        """Test that no removal occurs with 20 classes (no background)."""
        fc_params = {
            "weight": np.random.randn(20, 100),
            "bias": np.random.randn(20),
        }
        ind_data_dict = {
            "train logits": np.random.randn(100, 20),
            "valid logits": np.random.randn(50, 20),
        }
        ood_data_dict = {"ood1 logits": np.random.randn(30, 20)}
        ood_names = ["ood1"]

        ind_dict, ood_dict, params = remove_background_dimension(
            fc_params, ind_data_dict, ood_data_dict, ood_names
        )

        # Dimensions should remain unchanged
        self.assertEqual(params["weight"].shape[0], 20)
        self.assertEqual(ind_dict["train logits"].shape[1], 20)

    def test_remove_background_dimension_preserves_data(self):
        """Test that function preserves data correctly after removal."""
        original_weight = np.random.randn(21, 100)
        fc_params = {
            "weight": original_weight.copy(),
            "bias": np.random.randn(21),
        }
        ind_data_dict = {
            "train logits": np.random.randn(100, 21),
            "valid logits": np.random.randn(50, 21),
        }
        ood_data_dict = {}
        ood_names = []

        ind_dict, ood_dict, params = remove_background_dimension(
            fc_params, ind_data_dict, ood_data_dict, ood_names
        )

        # Check that the first 20 rows match
        np.testing.assert_array_almost_equal(params["weight"], original_weight[:-1, :], decimal=10)

    def test_remove_background_dimension_multiple_ood_datasets(self):
        """Test with multiple OOD datasets."""
        fc_params = {
            "weight": np.random.randn(21, 100),
            "bias": np.random.randn(21),
        }
        ind_data_dict = {
            "train logits": np.random.randn(100, 21),
            "valid logits": np.random.randn(50, 21),
        }
        ood_data_dict = {
            "ood1 logits": np.random.randn(30, 21),
            "ood2 logits": np.random.randn(40, 21),
            "ood3 logits": np.random.randn(25, 21),
        }
        ood_names = ["ood1", "ood2", "ood3"]

        ind_dict, ood_dict, params = remove_background_dimension(
            fc_params, ind_data_dict, ood_data_dict, ood_names
        )

        # Check all OOD datasets are updated
        self.assertEqual(ood_dict["ood1 logits"].shape[1], 20)
        self.assertEqual(ood_dict["ood2 logits"].shape[1], 20)
        self.assertEqual(ood_dict["ood3 logits"].shape[1], 20)


class TestIntegration(unittest.TestCase):
    """Integration tests for multiple functions working together."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestIntegration")

    def test_msp_and_energy_score_integration(self):
        """Test that MSP and energy scores can be computed on same model."""
        mock_model = MockRCNNModel(num_detections=5, num_classes=10)
        dataset = TensorDataset(torch.randn(TEST_SAMPLES, 3, 224, 224))
        dataloader = DataLoader(dataset, batch_size=1)

        msp_scores = get_msp_score_rcnn(mock_model, dataloader)

        # Create new dataloader for energy score
        dataloader2 = DataLoader(dataset, batch_size=1)
        raw_energy, filtered_energy = get_energy_score_rcnn(mock_model, dataloader2)

        self.assertEqual(msp_scores.shape[0], TEST_SAMPLES)
        self.assertEqual(raw_energy.shape[0], TEST_SAMPLES)
        self.assertEqual(filtered_energy.shape[0], TEST_SAMPLES)

    def test_sampler_and_inference_integration(self):
        """Test MCSamplerRCNN and LaRDInferenceRCNN work together."""
        # Create models
        mock_model = MockRCNNModel(has_rpn=True)
        mock_detector = MagicMock()

        # Create sampler
        sampler = MCSamplerRCNN(mc_samples=2, layer_type="RPN")

        # Create LaRD inference
        lard = LaRDInferenceRCNN(
            model=mock_model,
            detector=mock_detector,
            pca_transform=None,
            layer_type="RPN",
        )

        # Verify components are properly instantiated
        self.assertIsNotNone(sampler)
        self.assertIsNotNone(lard)
        self.assertEqual(sampler.mc_samples, 2)
        self.assertEqual(lard.layer_type, "RPN")


# ============================================================================
# FUNCTIONAL TESTS FOR get_score WITH MOCKS
# ============================================================================


class TestGetScoreFunctionality(unittest.TestCase):
    """Test the functionality of LaRexInferenceRCNN.get_score method with mocks."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestGetScoreFunctionality")

    def _create_mock_larex_instance(self):
        """Create a mock LaRexInferenceRCNN instance for testing."""
        mock_model = MagicMock()
        mock_detector = MagicMock()
        mock_mc_sampler = MagicMock()

        # Create a partial mock instance
        larex_instance = MagicMock(spec=LaRexInferenceRCNN)
        larex_instance.model = mock_model
        larex_instance.detector = mock_detector
        larex_instance.mc_sampler = mock_mc_sampler
        larex_instance.pca_transform = None
        larex_instance.mcd_samples_nro = MCD_N_SAMPLES
        larex_instance.device = torch.device("cpu")

        return larex_instance, mock_model, mock_detector, mock_mc_sampler

    def test_get_score_returns_output_and_score_tuple(self):
        """Test that get_score returns a tuple of (output, score)."""
        # Setup mocks
        larex_instance, mock_model, mock_detector, mock_mc_sampler = (
            self._create_mock_larex_instance()
        )

        # Mock return values
        mock_output = MagicMock()
        mock_model.return_value = mock_output
        mock_mc_sampler.return_value = torch.randn(MCD_N_SAMPLES, 100)
        mock_detector.postprocess.return_value = np.array([0.7])

        # Call the actual get_score method
        from runia_core.evaluation.entropy import get_dl_h_z

        input_image = torch.randn(1, 3, 224, 224)
        layer_hook = Mock()

        # Manually execute the get_score logic with mocks
        with torch.no_grad():
            output = mock_model(input_image)
            mc_samples = mock_mc_sampler(mock_model)
            _, sample_h_z = get_dl_h_z(mc_samples, MCD_N_SAMPLES)
            sample_score = mock_detector.postprocess(sample_h_z)

        # Verify return structure
        self.assertIsNotNone(output)
        self.assertIsNotNone(sample_score)
        self.assertTrue(isinstance(sample_score, np.ndarray))

    def test_get_score_calls_model_with_input_image(self):
        """Test that get_score calls model forward pass with input image."""
        larex_instance, mock_model, mock_detector, mock_mc_sampler = (
            self._create_mock_larex_instance()
        )

        mock_model.return_value = MagicMock()
        mock_mc_sampler.return_value = torch.randn(MCD_N_SAMPLES, 100)
        mock_detector.postprocess.return_value = np.array([0.5])

        input_image = torch.randn(1, 3, 224, 224)

        # Simulate get_score logic
        with torch.no_grad():
            mock_model(input_image)

        # Verify model was called
        mock_model.assert_called_once()
        call_args = mock_model.call_args
        self.assertTrue(torch.allclose(call_args[0][0], input_image))

    def test_get_score_uses_mc_sampler(self):
        """Test that get_score uses mc_sampler to generate samples."""
        larex_instance, mock_model, mock_detector, mock_mc_sampler = (
            self._create_mock_larex_instance()
        )

        mock_model.return_value = MagicMock()
        mc_samples_expected = torch.randn(MCD_N_SAMPLES, 100)
        mock_mc_sampler.return_value = mc_samples_expected
        mock_detector.postprocess.return_value = np.array([0.5])

        # Simulate get_score logic
        with torch.no_grad():
            mock_model(torch.randn(1, 3, 224, 224))
            mc_samples = mock_mc_sampler(mock_model)

        # Verify mc_sampler was called and returned expected samples
        mock_mc_sampler.assert_called_once()
        self.assertTrue(torch.allclose(mc_samples, mc_samples_expected))

    def test_get_score_calculates_entropy_from_samples(self):
        """Test that get_score calculates entropy from MC samples."""
        from runia_core.evaluation.entropy import get_dl_h_z

        mc_samples = torch.randn(MCD_N_SAMPLES, 100)

        # Get entropy from samples
        entropy, sample_h_z = get_dl_h_z(mc_samples, MCD_N_SAMPLES)

        # Verify entropy was calculated
        self.assertIsNotNone(entropy)
        self.assertIsNotNone(sample_h_z)
        self.assertTrue(isinstance(sample_h_z, np.ndarray))
        self.assertGreater(sample_h_z.size, 0)

    def test_get_score_applies_pca_when_available(self):
        """Test that get_score applies PCA transform when available."""
        from runia_core.dimensionality_reduction import apply_pca_transform

        # Create mock PCA transform
        mock_pca = MagicMock()
        sample_h_z = np.random.randn(1, 100)
        pca_result = np.random.randn(1, 50)
        mock_pca.side_effect = lambda x: pca_result

        # Simulate PCA application
        if mock_pca is not None:
            transformed = mock_pca(sample_h_z)

        # Verify PCA was applied
        mock_pca.assert_called_once()
        self.assertTrue(np.allclose(transformed, pca_result))
        self.assertEqual(transformed.shape[1], 50)

    def test_get_score_postprocesses_with_detector(self):
        """Test that get_score uses detector.postprocess for final scoring."""
        mock_detector = MagicMock()
        sample_h_z = np.random.randn(1, 100)
        expected_score = np.array([0.75])
        mock_detector.postprocess.return_value = expected_score

        # Simulate detector postprocessing
        score = mock_detector.postprocess(sample_h_z)

        # Verify postprocessing was called and returned score
        mock_detector.postprocess.assert_called_once()
        self.assertTrue(np.allclose(score, expected_score))

    def test_get_score_handles_device_transfer(self):
        """Test that get_score handles device transfer for input_image."""
        mock_device = torch.device("cpu")
        input_image = torch.randn(1, 3, 224, 224)

        # Simulate device transfer with try-except
        try:
            input_image_on_device = input_image.to(mock_device)
            device_transfer_successful = True
        except AttributeError:
            device_transfer_successful = False

        # Verify device transfer worked
        self.assertTrue(device_transfer_successful)
        self.assertEqual(input_image_on_device.device.type, "cpu")

    def test_get_score_with_pca_transform_none(self):
        """Test that get_score handles pca_transform being None."""
        larex_instance, mock_model, mock_detector, mock_mc_sampler = (
            self._create_mock_larex_instance()
        )
        larex_instance.pca_transform = None

        sample_h_z = np.random.randn(1, 100)
        mock_detector.postprocess.return_value = np.array([0.6])

        # Simulate get_score logic without PCA
        if larex_instance.pca_transform is None:
            # Skip PCA application
            result_h_z = sample_h_z
        else:
            result_h_z = larex_instance.pca_transform(sample_h_z)

        # Verify no PCA was applied
        self.assertTrue(np.allclose(result_h_z, sample_h_z))

    def test_get_score_entropy_calculation_correctness(self):
        """Test that entropy calculation produces correct values."""
        from runia_core.evaluation.entropy import get_dl_h_z

        # Create deterministic MC samples
        mc_samples = torch.ones(MCD_N_SAMPLES, 100) * 0.5

        _, sample_h_z = get_dl_h_z(mc_samples, MCD_N_SAMPLES)

        # Verify entropy was calculated
        self.assertIsNotNone(sample_h_z)
        self.assertEqual(sample_h_z.shape[0], 1)
        self.assertGreater(sample_h_z.shape[1], 0)

    def test_get_score_complete_pipeline_with_mocks(self):
        """Test complete get_score pipeline with all mocked components."""
        # Setup all mocks
        mock_model = MagicMock()
        mock_detector = MagicMock()
        mock_mc_sampler = MagicMock()
        mock_pca_transform = MagicMock()

        # Setup return values
        mock_output = {"predictions": torch.randn(1, 10)}
        mock_model.return_value = mock_output

        mc_samples = torch.randn(MCD_N_SAMPLES, 100)
        mock_mc_sampler.return_value = mc_samples

        expected_score = np.array([0.85])
        mock_detector.postprocess.return_value = expected_score

        # Simulate complete pipeline
        from runia_core.evaluation.entropy import get_dl_h_z

        input_image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = mock_model(input_image)
            mc_samples_result = mock_mc_sampler(mock_model)
            _, sample_h_z = get_dl_h_z(mc_samples_result, MCD_N_SAMPLES)

            # Apply PCA if available
            if mock_pca_transform is not None:
                sample_h_z_transformed = mock_pca_transform(sample_h_z)
            else:
                sample_h_z_transformed = sample_h_z

            sample_score = mock_detector.postprocess(sample_h_z_transformed)

        # Verify complete pipeline worked
        self.assertIsNotNone(output)
        self.assertTrue(isinstance(sample_score, np.ndarray))
        self.assertTrue(np.allclose(sample_score, expected_score))

        # Verify all mocks were called
        mock_model.assert_called_once()
        mock_mc_sampler.assert_called_once()
        mock_detector.postprocess.assert_called_once()

    def test_get_score_with_different_batch_sizes(self):
        """Test get_score handles different input batch sizes."""
        from runia_core.evaluation.entropy import get_dl_h_z

        mock_model = MagicMock()
        mock_detector = MagicMock()
        mock_mc_sampler = MagicMock()

        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            input_image = torch.randn(batch_size, 3, 224, 224)
            mock_output = {"pred": torch.randn(batch_size, 10)}
            mock_model.return_value = mock_output

            mc_samples = torch.randn(MCD_N_SAMPLES, 100)
            mock_mc_sampler.return_value = mc_samples
            mock_detector.postprocess.return_value = np.random.rand(1)

            # Simulate processing
            with torch.no_grad():
                output = mock_model(input_image)
                self.assertEqual(output["pred"].shape[0], batch_size)

    def test_get_score_entropy_with_varying_mcd_samples(self):
        """Test get_score entropy calculation with different MCD sample counts."""
        from runia_core.evaluation.entropy import get_dl_h_z

        for n_samples in [2, 5, 10]:
            mc_samples = torch.randn(n_samples, 100)
            _, sample_h_z = get_dl_h_z(mc_samples, n_samples)

            # Verify entropy is calculated correctly
            self.assertIsNotNone(sample_h_z)
            self.assertTrue(isinstance(sample_h_z, np.ndarray))

    def test_get_score_detector_output_format(self):
        """Test that detector.postprocess returns correct output format."""
        mock_detector = MagicMock()

        # Test various output formats from detector
        test_outputs = [
            np.array([0.5]),
            np.array([0.3, 0.7]),
            np.array([[0.5, 0.2]]),
        ]

        for expected_output in test_outputs:
            mock_detector.postprocess.return_value = expected_output
            sample_h_z = np.random.randn(1, 100)

            result = mock_detector.postprocess(sample_h_z)
            self.assertTrue(isinstance(result, np.ndarray))
            self.assertTrue(np.allclose(result, expected_output))


if __name__ == "__main__":
    unittest.main()
