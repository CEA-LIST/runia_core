# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Daniel Montoya

from unittest import TestCase, main
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from runia_core.feature_extraction.object_level import (
    BoxFeaturesExtractor,
    BoxFeaturesExtractorAnomalyLoader,
    _reduce_features_to_rois,
    _dropblock_rois_get_entropy,
)
from runia_core.feature_extraction.utils import Hook
from runia_core.feature_extraction.abstract_classes import MCSamplerModule

#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 42
DEVICE = torch.device("cpu")  # Force CPU to avoid CUDA initialization issues
TOL = 1e-6

# Object detection parameters
ROI_OUTPUT_SIZE = (7, 7)
ROI_SAMPLING_RATIO = 2
NUM_HOOKED_REPS = 1
NUM_DETECTED_OBJECTS = 2
NUM_FEATURE_CHANNELS = 64
IMG_SHAPE = (480, 640)  # (height, width)
BATCH_SIZE = 1
MCD_SAMPLES = 3
########################################################################


class TestReduceFeaturesToROIs(TestCase):
    """Test suite for _reduce_features_to_rois function"""

    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = DEVICE

    def test_reduce_features_to_rois_basic(self):
        """Test basic functionality of _reduce_features_to_rois"""
        # Create mock latent features (batch, channels, height, width)
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]

        # Create mock bounding boxes (num_boxes, 4) in xyxy format
        boxes = Tensor(
            [[50, 50, 150, 150], [200, 200, 350, 350]]
        ).to(self.device)

        output_sizes = [ROI_OUTPUT_SIZE]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=NUM_HOOKED_REPS,
            n_detected_objects=NUM_DETECTED_OBJECTS,
            return_stds=False,
        )

        # Check output shapes
        self.assertEqual(len(means), NUM_DETECTED_OBJECTS)
        self.assertEqual(means[0].shape[0], 1)
        self.assertEqual(means[0].shape[1], NUM_FEATURE_CHANNELS)
        # When return_stds=False, stds list should be empty
        self.assertEqual(len(stds), 0)

    def test_reduce_features_to_rois_with_stds(self):
        """Test _reduce_features_to_rois with standard deviations"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        boxes = Tensor(
            [[50, 50, 150, 150], [200, 200, 350, 350]]
        ).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=NUM_HOOKED_REPS,
            n_detected_objects=NUM_DETECTED_OBJECTS,
            return_stds=True,
        )

        # Check that stds are also returned
        self.assertEqual(len(stds), NUM_DETECTED_OBJECTS)
        for std in stds:
            self.assertEqual(std.shape[0], 1)
            self.assertEqual(std.shape[1], NUM_FEATURE_CHANNELS)

    def test_reduce_features_to_rois_multiple_layers(self):
        """Test _reduce_features_to_rois with multiple hooked layers"""
        n_hooked_reps = 3
        latent_sample = [
            torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)
            for _ in range(n_hooked_reps)
        ]
        boxes = Tensor([[50, 50, 150, 150]]).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE] * n_hooked_reps

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=n_hooked_reps,
            n_detected_objects=1,
            return_stds=True,
        )

        # Output should be concatenated features from all layers
        expected_feature_dim = NUM_FEATURE_CHANNELS * n_hooked_reps
        self.assertEqual(means[0].shape[1], expected_feature_dim)
        self.assertEqual(stds[0].shape[1], expected_feature_dim)

    def test_reduce_features_to_rois_single_object(self):
        """Test _reduce_features_to_rois with a single detected object"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        boxes = Tensor([[50, 50, 150, 150]]).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=NUM_HOOKED_REPS,
            n_detected_objects=1,
            return_stds=False,
        )

        self.assertEqual(len(means), 1)
        self.assertEqual(means[0].shape[0], 1)
        self.assertEqual(means[0].shape[1], NUM_FEATURE_CHANNELS)

    def test_reduce_features_to_rois_output_types(self):
        """Test that _reduce_features_to_rois returns tensors"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        boxes = Tensor([[50, 50, 150, 150]]).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=NUM_HOOKED_REPS,
            n_detected_objects=1,
            return_stds=True,
        )

        # Check types
        self.assertIsInstance(means, list)
        self.assertIsInstance(means[0], Tensor)
        self.assertIsInstance(stds, list)
        self.assertIsInstance(stds[0], Tensor)


class TestDropblockRoisGetEntropy(TestCase):
    """Test suite for _dropblock_rois_get_entropy function"""

    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = DEVICE

    def test_dropblock_rois_get_entropy_basic(self):
        """Test basic functionality of _dropblock_rois_get_entropy"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        boxes = Tensor([[50, 50, 150, 150], [200, 200, 350, 350]]).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE]

        # Create a mock MC sampler
        mc_sampler = MCSamplerModule(
            mc_samples=MCD_SAMPLES,
            block_size=7,
            drop_prob=0.5,
            layer_type="Conv",
        ).to(self.device)

        entropies = _dropblock_rois_get_entropy(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=NUM_HOOKED_REPS,
            n_mcd_steps=MCD_SAMPLES,
            mc_sampler=mc_sampler,
        )

        # Check output type and shape
        self.assertIsInstance(entropies, Tensor)
        self.assertEqual(entropies.shape[0], NUM_DETECTED_OBJECTS)

    def test_dropblock_rois_get_entropy_multiple_layers(self):
        """Test _dropblock_rois_get_entropy with multiple hooked layers"""
        n_hooked_reps = 2
        latent_sample = [
            torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)
            for _ in range(n_hooked_reps)
        ]
        boxes = Tensor([[50, 50, 150, 150]]).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE] * n_hooked_reps

        mc_sampler = MCSamplerModule(
            mc_samples=MCD_SAMPLES,
            block_size=7,
            drop_prob=0.5,
            layer_type="Conv",
        ).to(self.device)

        entropies = _dropblock_rois_get_entropy(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=n_hooked_reps,
            n_mcd_steps=MCD_SAMPLES,
            mc_sampler=mc_sampler,
        )

        self.assertIsInstance(entropies, Tensor)
        self.assertEqual(entropies.shape[0], 1)


class TestBoxFeaturesExtractor(TestCase):
    """Test suite for BoxFeaturesExtractor class"""

    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = DEVICE

        # Create a simple mock model
        self.mock_model = Mock(spec=torch.nn.Module)
        self.mock_model.to = Mock(return_value=self.mock_model)
        self.mock_model.eval = Mock(return_value=self.mock_model)

        # Create mock hooked layer
        self.mock_hook = Mock(spec=Hook)

    def test_boxfeaturesextractor_initialization(self):
        """Test BoxFeaturesExtractor initialization"""
        extractor = BoxFeaturesExtractor(
            model=self.mock_model,
            hooked_layers=[self.mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            return_raw_predictions=False,
            return_stds=False,
            mcd_nro_samples=1,
            hook_layer_output=True,
            dropblock_probs=0.0,
            dropblock_sizes=0,
            rcnn_extraction_type=None,
            extract_noise_entropies=False,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        # Check attributes
        self.assertEqual(extractor.roi_sampling_ratio, ROI_SAMPLING_RATIO)
        self.assertIsInstance(extractor.roi_output_sizes, list)
        self.assertFalse(extractor.return_stds)
        self.assertFalse(extractor.extract_noise_entropies)

    def test_boxfeaturesextractor_roi_output_sizes_conversion(self):
        """Test that roi_output_sizes is converted to list"""
        roi_sizes = (7, 7)
        extractor = BoxFeaturesExtractor(
            model=self.mock_model,
            hooked_layers=[self.mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=roi_sizes,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        self.assertIsInstance(extractor.roi_output_sizes, list)
        self.assertEqual(extractor.roi_output_sizes, list(roi_sizes))

    def test_boxfeaturesextractor_rcnn_roi_output_sizes_multiplication(self):
        """Test roi_output_sizes multiplication for RCNN with non-shortcut extraction"""
        roi_sizes = (7, 7)
        extractor = BoxFeaturesExtractor(
            model=self.mock_model,
            hooked_layers=[self.mock_hook],
            device=self.device,
            architecture="rcnn",
            roi_output_sizes=roi_sizes,
            rcnn_extraction_type="backbone",
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        # For RCNN with non-shortcut extraction, roi_output_sizes should be multiplied by 5
        self.assertEqual(len(extractor.roi_output_sizes), 10)

    def test_boxfeaturesextractor_return_stds_parameter(self):
        """Test return_stds parameter in initialization"""
        extractor = BoxFeaturesExtractor(
            model=self.mock_model,
            hooked_layers=[self.mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            return_stds=True,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        self.assertTrue(extractor.return_stds)

    def test_boxfeaturesextractor_mcd_samples_parameter(self):
        """Test mcd_nro_samples parameter in initialization"""
        mcd_samples = 5
        extractor = BoxFeaturesExtractor(
            model=self.mock_model,
            hooked_layers=[self.mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            mcd_nro_samples=mcd_samples,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        self.assertEqual(extractor.mcd_nro_samples, mcd_samples)

    def test_boxfeaturesextractor_extract_noise_entropies_parameter(self):
        """Test extract_noise_entropies parameter in initialization"""
        extractor = BoxFeaturesExtractor(
            model=self.mock_model,
            hooked_layers=[self.mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            extract_noise_entropies=True,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        self.assertTrue(extractor.extract_noise_entropies)


class TestBoxFeaturesExtractorAnomalyLoader(TestCase):
    """Test suite for BoxFeaturesExtractorAnomalyLoader class"""

    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = DEVICE

        # Create a simple mock model
        self.mock_model = Mock(spec=torch.nn.Module)
        self.mock_model.to = Mock(return_value=self.mock_model)
        self.mock_model.eval = Mock(return_value=self.mock_model)

        # Create mock hooked layer
        self.mock_hook = Mock(spec=Hook)

    def test_anomaly_loader_initialization(self):
        """Test BoxFeaturesExtractorAnomalyLoader initialization"""
        extractor = BoxFeaturesExtractorAnomalyLoader(
            model=self.mock_model,
            hooked_layers=[self.mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        # Verify it's an instance of the parent class
        self.assertIsInstance(extractor, BoxFeaturesExtractor)
        self.assertEqual(extractor.roi_sampling_ratio, ROI_SAMPLING_RATIO)

    def test_anomaly_loader_inheritance(self):
        """Test that BoxFeaturesExtractorAnomalyLoader inherits from BoxFeaturesExtractor"""
        self.assertTrue(issubclass(BoxFeaturesExtractorAnomalyLoader, BoxFeaturesExtractor))

    def test_anomaly_loader_with_stds(self):
        """Test BoxFeaturesExtractorAnomalyLoader with return_stds=True"""
        extractor = BoxFeaturesExtractorAnomalyLoader(
            model=self.mock_model,
            hooked_layers=[self.mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            return_stds=True,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        self.assertTrue(extractor.return_stds)

    def test_anomaly_loader_with_different_architecture(self):
        """Test BoxFeaturesExtractorAnomalyLoader with different architectures"""
        for arch in ["rcnn", "detr-backbone", "owlv2"]:
            extractor = BoxFeaturesExtractorAnomalyLoader(
                model=self.mock_model,
                hooked_layers=[self.mock_hook],
                device=self.device,
                architecture=arch,
                roi_output_sizes=ROI_OUTPUT_SIZE,
                roi_sampling_ratio=ROI_SAMPLING_RATIO,
            )
            self.assertEqual(extractor.architecture, arch)


class TestEdgeCasesAndErrors(TestCase):
    """Test suite for edge cases and error handling"""

    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = DEVICE

    def test_reduce_features_empty_boxes(self):
        """Test _reduce_features_to_rois with empty boxes list"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        boxes = Tensor([]).reshape(0, 4).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE]

        # This should handle the case gracefully
        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=NUM_HOOKED_REPS,
            n_detected_objects=0,
            return_stds=False,
        )

        self.assertEqual(len(means), 0)
        self.assertEqual(len(stds), 0)

    def test_reduce_features_large_number_of_objects(self):
        """Test _reduce_features_to_rois with large number of detected objects"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 120, 160, device=self.device)]
        n_objects = 10
        boxes = Tensor(
            [[i * 10, i * 10, i * 10 + 50, i * 10 + 50] for i in range(n_objects)]
        ).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=(IMG_SHAPE[0] * 2, IMG_SHAPE[1] * 2),
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=NUM_HOOKED_REPS,
            n_detected_objects=n_objects,
            return_stds=False,
        )

        self.assertEqual(len(means), n_objects)
        for mean in means:
            self.assertEqual(mean.shape[0], 1)
            self.assertEqual(mean.shape[1], NUM_FEATURE_CHANNELS)

    def test_reduce_features_different_output_sizes(self):
        """Test _reduce_features_to_rois with different output sizes for each layer"""
        n_hooked_reps = 4
        latent_sample = [
            torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)
            for _ in range(n_hooked_reps)
        ]
        boxes = Tensor([[50, 50, 150, 150]]).to(self.device)
        output_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=n_hooked_reps,
            n_detected_objects=1,
            return_stds=True,
        )

        self.assertEqual(len(means), 1)
        # Feature dimension should be sum of all channels from all layers
        expected_dim = NUM_FEATURE_CHANNELS * n_hooked_reps
        self.assertEqual(means[0].shape[1], expected_dim)
        self.assertEqual(stds[0].shape[1], expected_dim)

    def test_boxfeaturesextractor_with_dropblock(self):
        """Test BoxFeaturesExtractor with dropblock configuration"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        mock_hook = Mock(spec=Hook)

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            dropblock_probs=0.5,
            dropblock_sizes=7,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        self.assertEqual(extractor.dropblock_probs, 0.5)
        self.assertEqual(extractor.dropblock_sizes, 7)

    def test_boxfeaturesextractor_with_multiple_hooked_layers(self):
        """Test BoxFeaturesExtractor with multiple hooked layers"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        hooks = [Mock(spec=Hook) for _ in range(3)]

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=hooks,
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=(7, 7),
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        self.assertEqual(len(extractor.hooked_layers), 3)

    def test_boxfeaturesextractor_yolov8_architecture(self):
        """Test BoxFeaturesExtractor specifically with YOLO architecture"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        mock_hook = Mock(spec=Hook)

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        self.assertEqual(extractor.architecture, "yolov8")

    def test_boxfeaturesextractor_rcnn_architecture_shortcut(self):
        """Test BoxFeaturesExtractor with RCNN architecture using shortcut extraction"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        mock_hook = Mock(spec=Hook)

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="rcnn",
            rcnn_extraction_type="shortcut",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        # With shortcut extraction, roi_output_sizes should NOT be multiplied
        self.assertEqual(len(extractor.roi_output_sizes), 2)

    def test_reduce_features_to_rois_high_spatial_scale(self):
        """Test _reduce_features_to_rois with high spatial scales"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 120, 160, device=self.device)]
        boxes = Tensor([[50, 50, 150, 150]]).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=(480, 640),
            sampling_ratio=1,  # High sampling ratio
            n_hooked_reps=NUM_HOOKED_REPS,
            n_detected_objects=1,
            return_stds=True,
        )

        self.assertEqual(len(means), 1)
        self.assertEqual(means[0].shape[1], NUM_FEATURE_CHANNELS)

    def test_reduce_features_to_rois_low_spatial_scale(self):
        """Test _reduce_features_to_rois with low spatial scales"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 30, 40, device=self.device)]
        boxes = Tensor([[10, 10, 30, 30]]).to(self.device)
        output_sizes = [(3, 3)]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=(60, 80),
            sampling_ratio=-1,  # Negative sampling ratio (adaptive)
            n_hooked_reps=1,
            n_detected_objects=1,
            return_stds=False,
        )

        self.assertEqual(len(means), 1)
        self.assertEqual(means[0].shape[0], 1)


class TestBoxFeaturesExtractorGetLSSamples(TestCase):
    """Test suite for BoxFeaturesExtractor.get_ls_samples method"""

    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = DEVICE

    def _create_mock_extractor(self):
        """Helper to create a mock extractor"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        mock_hook = Mock(spec=Hook)

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
            return_stds=False,
        )

        # Mock necessary methods
        extractor.check_dataloader = Mock()
        extractor.unpack_dataloader = Mock(return_value=("path", torch.randn(1, 3, 416, 416), 0))
        extractor._get_samples_one_image = Mock(
            return_value=(
                {
                    "latent_space_means": torch.randn(1, NUM_FEATURE_CHANNELS),
                    "features": torch.randn(1, 256),
                    "logits": torch.randn(1, 80),
                    "boxes": torch.randn(1, 4),
                },
                True,
            )
        )

        return extractor

    def test_get_ls_samples_result_structure(self):
        """Test that get_ls_samples returns correct structure"""
        extractor = self._create_mock_extractor()

        # Create a mock dataloader
        mock_loader = Mock(spec=DataLoader)
        mock_loader.__len__ = Mock(return_value=2)
        mock_loader.__iter__ = Mock(return_value=iter([None, None]))

        results = extractor.get_ls_samples(mock_loader, predict_conf=0.25)

        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn("no_obj", results)

    def test_get_ls_samples_calls_check_dataloader(self):
        """Test that get_ls_samples calls check_dataloader"""
        extractor = self._create_mock_extractor()

        mock_loader = Mock(spec=DataLoader)
        mock_loader.__len__ = Mock(return_value=0)
        mock_loader.__iter__ = Mock(return_value=iter([]))

        extractor.get_ls_samples(mock_loader)

        extractor.check_dataloader.assert_called_once()

    def test_get_ls_samples_processes_multiple_images(self):
        """Test get_ls_samples with multiple images"""
        extractor = self._create_mock_extractor()

        mock_loader = Mock(spec=DataLoader)
        mock_loader.__len__ = Mock(return_value=3)
        mock_loader.__iter__ = Mock(return_value=iter([None, None, None]))

        results = extractor.get_ls_samples(mock_loader, predict_conf=0.25)

        # Verify _get_samples_one_image was called 3 times
        self.assertEqual(extractor._get_samples_one_image.call_count, 3)


class TestGetSamplesOneImage(TestCase):
    """Test suite for BoxFeaturesExtractor._get_samples_one_image method"""

    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = DEVICE

    def _create_mock_extractor_with_inference(self):
        """Helper to create a mock extractor with inference"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        mock_hook = Mock(spec=Hook)

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        return extractor

    def test_get_samples_one_image_with_detected_objects(self):
        """Test _get_samples_one_image with detected objects"""
        extractor = self._create_mock_extractor_with_inference()

        # Mock the model_dependent_inference method
        detected_boxes = Tensor([[50, 50, 150, 150], [200, 200, 350, 350]])
        extractor.model_dependent_inference = Mock(
            return_value=(
                {
                    "features": torch.randn(2, 256),
                    "logits": torch.randn(2, 80),
                },
                detected_boxes,
                None,
                IMG_SHAPE,
            )
        )

        # Mock model_dependent_feature_extraction
        extractor.model_dependent_feature_extraction = Mock(
            return_value=[torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        )

        image = torch.randn(1, 3, 416, 416)
        results, found_objs_flag = extractor._get_samples_one_image(image, predict_conf=0.25)

        # Check results
        self.assertTrue(found_objs_flag)
        self.assertIn("boxes", results)
        self.assertIn("latent_space_means", results)

    def test_get_samples_one_image_no_detected_objects(self):
        """Test _get_samples_one_image with no detected objects"""
        extractor = self._create_mock_extractor_with_inference()

        # Mock the model_dependent_inference method to return no boxes
        extractor.model_dependent_inference = Mock(
            return_value=(
                {"features": torch.randn(0, 256), "logits": torch.randn(0, 80)},
                Tensor([]),
                None,
                IMG_SHAPE,
            )
        )

        # Mock model_dependent_feature_extraction
        extractor.model_dependent_feature_extraction = Mock(
            return_value=[torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        )

        image = torch.randn(1, 3, 416, 416)
        results, found_objs_flag = extractor._get_samples_one_image(image, predict_conf=0.25)

        # When no objects are detected, the whole image is used as a single object
        self.assertFalse(found_objs_flag)
        self.assertEqual(results["boxes"].shape[0], 1)

    def test_get_samples_one_image_return_raw_predictions(self):
        """Test _get_samples_one_image with return_raw_predictions enabled"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        mock_hook = Mock(spec=Hook)

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            return_raw_predictions=True,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        detected_boxes = Tensor([[50, 50, 150, 150]])
        raw_preds = {"predictions": torch.randn(1, 85)}

        extractor.model_dependent_inference = Mock(
            return_value=(
                {},
                detected_boxes,
                raw_preds,
                IMG_SHAPE,
            )
        )

        extractor.model_dependent_feature_extraction = Mock(
            return_value=[torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        )

        image = torch.randn(1, 3, 416, 416)
        results, _ = extractor._get_samples_one_image(image, predict_conf=0.25)

        # Check that raw predictions are included
        self.assertIn("raw_preds", results)
        self.assertEqual(results["raw_preds"], raw_preds)

    def test_get_samples_one_image_return_stds(self):
        """Test _get_samples_one_image with return_stds enabled"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)

        mock_hook = Mock(spec=Hook)

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            return_stds=True,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        detected_boxes = Tensor([[50, 50, 150, 150]])

        extractor.model_dependent_inference = Mock(
            return_value=(
                {},
                detected_boxes,
                None,
                IMG_SHAPE,
            )
        )

        extractor.model_dependent_feature_extraction = Mock(
            return_value=[torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        )

        image = torch.randn(1, 3, 416, 416)
        results, _ = extractor._get_samples_one_image(image, predict_conf=0.25)

        # Check that stds are included
        self.assertIn("stds", results)


class TestIntegrationScenarios(TestCase):
    """Test suite for integration scenarios and realistic workflows"""

    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = DEVICE

    def test_workflow_boxfeaturesextractor_with_multiple_configurations(self):
        """Test a realistic workflow with multiple extractor configurations"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        mock_hook = Mock(spec=Hook)

        # Test with various realistic configurations
        configs = [
            {
                "architecture": "yolov8",
                "return_stds": True,
                "extract_noise_entropies": False,
                "mcd_nro_samples": 1,
            },
            {
                "architecture": "rcnn",
                "return_stds": False,
                "extract_noise_entropies": True,
                "mcd_nro_samples": 5,
                "rcnn_extraction_type": "backbone",
            },
            {
                "architecture": "detr-backbone",
                "return_stds": True,
                "extract_noise_entropies": False,
                "mcd_nro_samples": 3,
            },
        ]

        for config in configs:
            arch = config.pop("architecture")
            extractor = BoxFeaturesExtractor(
                model=mock_model,
                hooked_layers=[mock_hook],
                device=self.device,
                architecture=arch,
                roi_output_sizes=ROI_OUTPUT_SIZE,
                roi_sampling_ratio=ROI_SAMPLING_RATIO,
                **config,
            )
            self.assertEqual(extractor.architecture, arch)
            self.assertEqual(extractor.return_stds, config["return_stds"])
            self.assertEqual(extractor.extract_noise_entropies, config["extract_noise_entropies"])
            self.assertEqual(extractor.mcd_nro_samples, config["mcd_nro_samples"])

    def test_feature_extraction_workflow_with_roi_align(self):
        """Test realistic feature extraction workflow with ROI alignment"""
        # Simulate a realistic scenario with multiple layers and objects
        latent_samples = [
            torch.rand(1, 256, 60, 80, device=self.device),
            torch.rand(1, 512, 30, 40, device=self.device),
        ]

        boxes = Tensor(
            [
                [100, 100, 200, 200],
                [250, 250, 350, 350],
                [50, 50, 150, 150],
            ]
        ).to(self.device)

        output_sizes = [(14, 14), (7, 7)]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_samples,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=2,
            n_detected_objects=3,
            return_stds=True,
        )

        # Verify output
        self.assertEqual(len(means), 3)
        self.assertEqual(len(stds), 3)
        # Total feature dimension should be sum of channels from both layers
        expected_dim = 256 + 512
        for mean, std in zip(means, stds):
            self.assertEqual(mean.shape, (1, expected_dim))
            self.assertEqual(std.shape, (1, expected_dim))

    def test_monte_carlo_sampling_integration(self):
        """Test integration with Monte Carlo sampling"""
        latent_sample = [torch.rand(1, NUM_FEATURE_CHANNELS, 60, 80, device=self.device)]
        boxes = Tensor([[50, 50, 150, 150]]).to(self.device)
        output_sizes = [ROI_OUTPUT_SIZE]

        mc_sampler = MCSamplerModule(
            mc_samples=MCD_SAMPLES,
            block_size=7,
            drop_prob=0.5,
            layer_type="Conv",
        ).to(self.device)

        entropies = _dropblock_rois_get_entropy(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=ROI_SAMPLING_RATIO,
            n_hooked_reps=1,
            n_mcd_steps=MCD_SAMPLES,
            mc_sampler=mc_sampler,
        )

        # Verify entropy output
        self.assertIsInstance(entropies, Tensor)
        self.assertEqual(entropies.shape[0], 1)

    def test_boxfeaturesextractor_tensor_device_consistency(self):
        """Test that BoxFeaturesExtractor maintains device consistency"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        mock_hook = Mock(spec=Hook)

        extractor = BoxFeaturesExtractor(
            model=mock_model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="yolov8",
            roi_output_sizes=ROI_OUTPUT_SIZE,
            roi_sampling_ratio=ROI_SAMPLING_RATIO,
        )

        # Verify device is set correctly
        self.assertEqual(extractor.device, self.device)

    def test_reduce_features_statistical_properties(self):
        """Test statistical properties of reduced features"""
        # Create latent features with known properties
        latent_sample = [torch.ones(1, NUM_FEATURE_CHANNELS, 30, 30, device=self.device)]
        boxes = Tensor([[10, 10, 20, 20]]).to(self.device)
        output_sizes = [(5, 5)]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_sample,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=(60, 60),
            sampling_ratio=0,
            n_hooked_reps=1,
            n_detected_objects=1,
            return_stds=True,
        )

        # Since input is all ones, mean should be close to 1 and std should be close to 0
        self.assertTrue(torch.allclose(means[0], torch.ones_like(means[0]), atol=0.1))
        self.assertTrue(torch.allclose(stds[0], torch.zeros_like(stds[0]), atol=0.1))

    def test_anomaly_loader_multiple_configurations(self):
        """Test BoxFeaturesExtractorAnomalyLoader with various configurations"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        mock_hook = Mock(spec=Hook)

        configurations = [
            {"return_raw_predictions": True, "return_stds": True},
            {"return_raw_predictions": False, "return_stds": False},
            {"return_raw_predictions": True, "return_stds": False},
        ]

        for config in configurations:
            extractor = BoxFeaturesExtractorAnomalyLoader(
                model=mock_model,
                hooked_layers=[mock_hook],
                device=self.device,
                architecture="yolov8",
                roi_output_sizes=ROI_OUTPUT_SIZE,
                roi_sampling_ratio=ROI_SAMPLING_RATIO,
                **config,
            )

            self.assertEqual(extractor.return_raw_predictions, config["return_raw_predictions"])
            self.assertEqual(extractor.return_stds, config["return_stds"])

    def test_mixed_object_detection_scenario(self):
        """Test realistic mixed scenario with detected and undetected objects"""
        latent_samples = [torch.rand(1, 128, 40, 56, device=self.device)]

        # Multiple boxes of different scales
        boxes = Tensor(
            [
                [10, 10, 100, 100],  # Large box
                [200, 200, 250, 250],  # Small box
                [400, 400, 600, 600],  # Large box
                [150, 100, 180, 140],  # Medium box
            ]
        ).to(self.device)

        output_sizes = [(14, 14)]

        means, stds = _reduce_features_to_rois(
            latent_mcd_sample=latent_samples,
            output_sizes=output_sizes,
            boxes=boxes,
            img_shape=IMG_SHAPE,
            sampling_ratio=1,
            n_hooked_reps=1,
            n_detected_objects=4,
            return_stds=True,
        )

        # Verify all boxes produce valid features
        self.assertEqual(len(means), 4)
        self.assertEqual(len(stds), 4)
        for mean, std in zip(means, stds):
            self.assertEqual(mean.shape[0], 1)
            self.assertEqual(std.shape[0], 1)
            self.assertEqual(mean.shape[1], 128)
            self.assertEqual(std.shape[1], 128)



if __name__ == "__main__":
    main()
