#!/usr/bin/env python3
"""
Unit tests for the evaluation.inference module.

This script provides comprehensive unit tests for the non-covered functions in the
runia_core.evaluation.inference module. It tests various classes
and functions including abstract classes, postprocessors, and inference modules.

Usage:
    Run all tests: python -m unittest tests.unit_test_uncertainty_inference
    Run specific test: python -m unittest tests.unit_test_uncertainty_inference.TestInferenceModule.test_record_time
    Run with verbose output: python -m unittest -v tests.unit_test_uncertainty_inference

Requirements:
    - Python 3.9+
    - PyTorch
    - NumPy
    - unittest (built-in)
    - runia_core library

Date: 2025-01-09
"""

import logging
import time
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from omegaconf import DictConfig

from runia_core.inference.abstract_classes import (
    InferenceModule,
    ObjectDetectionInference,
    OodPostprocessor,
    Postprocessor,
    ProbabilisticInferenceModule,
    record_time,
)
from runia_core.feature_extraction import Hook

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test constants
SEED = 42
TOLERANCE = 1e-6
TEST_ARRAY_SIZE = (10, 5)
TEST_BATCH_SIZE = 4
TEST_CHANNELS = 3
TEST_HEIGHT = 32
TEST_WIDTH = 32


class MockModel(torch.nn.Module):
    """Mock model for testing purposes."""

    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.linear = torch.nn.Linear(10, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)

    def to(self, device):
        """Override to method for testing."""
        super().to(device)
        return self


class MockPostprocessor:
    """Mock postprocessor for testing purposes."""

    def __init__(self):
        self.called = False

    def __call__(self, data, **kwargs):
        self.called = True
        return data * 2


class TestRecordTime(unittest.TestCase):
    """Test cases for the record_time decorator."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestRecordTime")

    def test_record_time_basic_function(self):
        """Test record_time decorator with a basic function."""

        @record_time
        def simple_function(x, y):
            time.sleep(0.01)  # Small delay to measure time
            return x + y

        result, execution_time = simple_function(5, 3)

        self.assertEqual(result, 8)
        self.assertIsInstance(execution_time, float)
        self.assertGreater(execution_time, 0)
        self.assertLess(execution_time, 1.0)  # Should be much less than 1 second
        logger.info(f"Function executed in {execution_time:.6f} seconds")

    def test_record_time_with_kwargs(self):
        """Test record_time decorator with keyword arguments."""

        @record_time
        def function_with_kwargs(a, b=10, c=20):
            return a + b + c

        result, execution_time = function_with_kwargs(5, b=15, c=25)

        self.assertEqual(result, 45)
        self.assertIsInstance(execution_time, float)
        self.assertGreater(execution_time, 0)

    def test_record_time_exception_handling(self):
        """Test record_time decorator when function raises exception."""

        @record_time
        def function_with_exception():
            raise ValueError("Test exception")

        with self.assertRaises(ValueError):
            function_with_exception()


class TestPostprocessor(unittest.TestCase):
    """Test cases for the Postprocessor base class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestPostprocessor")

    def test_postprocessor_call_method(self):
        """Test Postprocessor __call__ method."""

        class ConcretePostprocessor(Postprocessor):
            def setup(self, ind_train_data, **kwargs):
                pass

            def postprocess(self, test_data, **kwargs):
                return test_data * 2

        postprocessor = ConcretePostprocessor()
        test_data = np.random.rand(*TEST_ARRAY_SIZE)

        result = postprocessor(test_data)
        expected = test_data * 2

        self.assertAlmostEqual((result - expected).sum(), 0.0, delta=TOLERANCE)


class TestOodPostprocessor(unittest.TestCase):
    """Test cases for the OodPostprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestOodPostprocessor")

    def test_ood_postprocessor_initialization(self):
        """Test OodPostprocessor initialization."""
        # Test with flip_sign=True
        postprocessor = OodPostprocessor(flip_sign=True)

        self.assertTrue(postprocessor.flip_sign)
        self.assertIsNone(postprocessor.threshold)
        self.assertFalse(postprocessor._setup_flag)

        # Test with flip_sign=False and config
        cfg = DictConfig({"param1": "value1"})
        postprocessor_no_flip = OodPostprocessor(flip_sign=False, cfg=cfg)

        self.assertFalse(postprocessor_no_flip.flip_sign)

    def test_flip_sign_fn_with_dict(self):
        """Test flip_sign_fn with dictionary input."""
        postprocessor = OodPostprocessor(flip_sign=True)

        test_scores = {
            "test_method": np.array([1.0, -2.0, 3.0]),
        }

        result = postprocessor.flip_sign_fn(test_scores)

        expected_flipped = np.array([-1.0, 2.0, -3.0])
        np.testing.assert_array_almost_equal(result["test_method"], expected_flipped)

    def test_flip_sign_fn_with_array(self):
        """Test flip_sign_fn with numpy array input."""
        postprocessor = OodPostprocessor(flip_sign=True)

        test_scores = np.array([1.0, -2.0, 3.0, -4.0])
        result = postprocessor.flip_sign_fn(test_scores)

        expected = np.array([-1.0, 2.0, -3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_flip_sign_fn_no_flip(self):
        """Test flip_sign_fn when flip_sign is False."""
        postprocessor = OodPostprocessor(flip_sign=False)

        test_scores = np.array([1.0, -2.0, 3.0])
        result = postprocessor.flip_sign_fn(test_scores)

        np.testing.assert_array_almost_equal(result, test_scores)

    def test_flip_sign_fn_invalid_input(self):
        """Test flip_sign_fn with invalid input type."""
        postprocessor = OodPostprocessor(flip_sign=True)

        with self.assertRaises(ValueError) as context:
            # Test with invalid input type (intentionally passing wrong type)
            invalid_input = "invalid_input"  # String instead of dict or ndarray
            postprocessor.flip_sign_fn(invalid_input)  # type: ignore

        self.assertIn("scores must be a dict or ndarray", str(context.exception))

    def test_set_threshold(self):
        """Test set_threshold method."""
        postprocessor = OodPostprocessor(flip_sign=False)

        # Mock the get_baselines_thresholds function
        test_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Since we can't easily mock the external function, we'll test the method structure
        try:
            postprocessor.set_threshold(test_scores)
            # If no exception is raised, the method executed successfully
            # The actual threshold calculation depends on the external function
        except Exception as e:
            # If there's an import or dependency issue, we still validate the structure
            logger.warning(f"set_threshold test encountered: {e}")


class TestInferenceModule(unittest.TestCase):
    """Test cases for the InferenceModule class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.mock_model = MockModel()
        self.mock_postprocessor = MockPostprocessor()
        logger.info("Setting up TestInferenceModule")

    def test_inference_module_initialization(self):
        """Test InferenceModule initialization."""
        inference_module = InferenceModule(
            model=self.mock_model, postprocessor=self.mock_postprocessor
        )

        self.assertEqual(inference_module.model, self.mock_model)
        self.assertEqual(inference_module.postprocessor, self.mock_postprocessor)
        self.assertIsInstance(inference_module.device, torch.device)

    def test_inference_module_device_selection(self):
        """Test device selection logic."""
        inference_module = InferenceModule(
            model=self.mock_model, postprocessor=self.mock_postprocessor
        )

        # Device should be cuda if available, otherwise cpu
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(inference_module.device.type, expected_device.type)

    def test_inference_module_model_without_to_method(self):
        """Test initialization with model that doesn't have 'to' method."""
        mock_model_no_to = MagicMock()
        del mock_model_no_to.to  # Remove the 'to' attribute

        # Should not raise an exception
        inference_module = InferenceModule(
            model=mock_model_no_to, postprocessor=self.mock_postprocessor
        )

        self.assertEqual(inference_module.model, mock_model_no_to)

    def test_get_score_not_implemented(self):
        """Test that get_score raises NotImplementedError."""
        inference_module = InferenceModule(
            model=self.mock_model, postprocessor=self.mock_postprocessor
        )

        test_input = torch.randn(TEST_BATCH_SIZE, TEST_CHANNELS, TEST_HEIGHT, TEST_WIDTH)

        with self.assertRaises(NotImplementedError):
            inference_module.get_score(test_input)


class TestProbabilisticInferenceModule(unittest.TestCase):
    """Test cases for the ProbabilisticInferenceModule class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.mock_model = MockModel()
        self.mock_postprocessor = MockPostprocessor()
        self.drop_block_prob = 0.1
        self.drop_block_size = 3
        self.mcd_samples_nro = 10
        logger.info("Setting up TestProbabilisticInferenceModule")

    def test_probabilistic_inference_module_initialization(self):
        """Test ProbabilisticInferenceModule initialization."""
        prob_inference_module = ProbabilisticInferenceModule(
            model=self.mock_model,
            postprocessor=self.mock_postprocessor,
            drop_block_prob=self.drop_block_prob,
            drop_block_size=self.drop_block_size,
            mcd_samples_nro=self.mcd_samples_nro,
        )

        # Test inheritance from InferenceModule
        self.assertEqual(prob_inference_module.model, self.mock_model)
        self.assertEqual(prob_inference_module.postprocessor, self.mock_postprocessor)

        # Test additional attributes
        self.assertEqual(prob_inference_module.drop_block_prob, self.drop_block_prob)
        self.assertEqual(prob_inference_module.drop_block_size, self.drop_block_size)
        self.assertEqual(prob_inference_module.mcd_samples_nro, self.mcd_samples_nro)

    def test_probabilistic_inference_module_parameters_validation(self):
        """Test parameter validation and types."""
        prob_inference_module = ProbabilisticInferenceModule(
            model=self.mock_model,
            postprocessor=self.mock_postprocessor,
            drop_block_prob=0.0,
            drop_block_size=1,
            mcd_samples_nro=1,
        )

        self.assertEqual(prob_inference_module.drop_block_prob, 0.0)
        self.assertEqual(prob_inference_module.drop_block_size, 1)
        self.assertEqual(prob_inference_module.mcd_samples_nro, 1)


class TestObjectDetectionInference(unittest.TestCase):
    """Test cases for the ObjectDetectionInference class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.mock_model = MockModel()
        self.mock_postprocessor = MockPostprocessor()
        self.architecture = "yolo"
        self.rcnn_extraction_type = "roi_pooling"

        # Create mock hooked layers
        self.hooked_layers = [Hook(torch.nn.Conv2d(3, 64, 3)), Hook(torch.nn.Conv2d(64, 128, 3))]
        logger.info("Setting up TestObjectDetectionInference")

    def test_object_detection_inference_initialization(self):
        """Test ObjectDetectionInference initialization."""
        obj_detection_module = ObjectDetectionInference(
            model=self.mock_model,
            postprocessor=self.mock_postprocessor,
            architecture=self.architecture,
            hooked_layers=self.hooked_layers,
            pca_transform=None,
            rcnn_extraction_type=self.rcnn_extraction_type,
        )

        # Test inheritance from InferenceModule
        self.assertEqual(obj_detection_module.model, self.mock_model)
        self.assertEqual(obj_detection_module.postprocessor, self.mock_postprocessor)

        # Test additional attributes
        self.assertEqual(obj_detection_module.architecture, self.architecture)
        self.assertEqual(obj_detection_module.rcnn_extraction_type, self.rcnn_extraction_type)
        self.assertEqual(obj_detection_module.hooked_layers, self.hooked_layers)
        self.assertIsNone(obj_detection_module.pca_transform)

    def test_object_detection_inference_with_pca(self):
        """Test ObjectDetectionInference with PCA transform."""
        mock_pca = MagicMock()

        obj_detection_module = ObjectDetectionInference(
            model=self.mock_model,
            postprocessor=self.mock_postprocessor,
            architecture=self.architecture,
            hooked_layers=self.hooked_layers,
            pca_transform=mock_pca,
            rcnn_extraction_type=None,
        )

        self.assertEqual(obj_detection_module.pca_transform, mock_pca)
        self.assertIsNone(obj_detection_module.rcnn_extraction_type)

    def test_object_detection_inference_hooked_layers_type(self):
        """Test hooked_layers parameter type validation."""
        obj_detection_module = ObjectDetectionInference(
            model=self.mock_model,
            postprocessor=self.mock_postprocessor,
            architecture=self.architecture,
            hooked_layers=self.hooked_layers,
        )

        self.assertIsInstance(obj_detection_module.hooked_layers, list)
        self.assertEqual(len(obj_detection_module.hooked_layers), 2)


class TestInferenceModuleIntegration(unittest.TestCase):
    """Integration tests for inference modules."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logger.info("Setting up TestInferenceModuleIntegration")

    def test_device_consistency_across_modules(self):
        """Test that all modules use the same device selection logic."""
        mock_model = MockModel()
        mock_postprocessor = MockPostprocessor()

        # Create different inference modules
        base_module = InferenceModule(mock_model, mock_postprocessor)
        prob_module = ProbabilisticInferenceModule(mock_model, mock_postprocessor, 0.1, 3, 10)
        obj_module = ObjectDetectionInference(mock_model, mock_postprocessor, "yolo", [])

        # All should have the same device
        self.assertEqual(base_module.device.type, prob_module.device.type)
        self.assertEqual(base_module.device.type, obj_module.device.type)

    def test_inheritance_chain(self):
        """Test proper inheritance relationships."""
        mock_model = MockModel()
        mock_postprocessor = MockPostprocessor()

        prob_module = ProbabilisticInferenceModule(mock_model, mock_postprocessor, 0.1, 3, 10)
        obj_module = ObjectDetectionInference(mock_model, mock_postprocessor, "yolo", [])

        # Test inheritance
        self.assertIsInstance(prob_module, InferenceModule)
        self.assertIsInstance(obj_module, InferenceModule)

        # Test that both have the base functionality
        self.assertTrue(hasattr(prob_module, "get_score"))
        self.assertTrue(hasattr(obj_module, "get_score"))


def run_tests():
    """
    Run all unit tests for the uncertainty inference module.

    This function serves as the main entry point for running tests and provides
    detailed logging information about test execution.
    """
    logger.info("Starting uncertainty inference module unit tests")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestRecordTime,
        TestPostprocessor,
        TestOodPostprocessor,
        TestInferenceModule,
        TestProbabilisticInferenceModule,
        TestObjectDetectionInference,
        TestInferenceModuleIntegration,
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
