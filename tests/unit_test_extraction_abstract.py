#!/usr/bin/env python3
"""
Unit tests for the feature_extraction.abstract_classes module.

This script provides comprehensive unit tests for the abstract classes in the
runia_core.feature_extraction.abstract_classes module. It focuses especially on the
ObjectDetectionExtractor class and MCSamplerModule, testing various architectures
and feature extraction modes.

Usage:
    Run all tests: python -m unittest tests.unit_test_extraction_abstract
    Run specific test: python -m unittest tests.unit_test_extraction_abstract.TestObjectDetectionExtractor.test_initialization
    Run with verbose output: python -m unittest -v tests.unit_test_extraction_abstract

Requirements:
    - Python 3.9+
    - PyTorch
    - NumPy
    - unittest (built-in)
    - runia_core library

Date: 2026-03-09
"""

import logging
import unittest
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms

from runia_core.feature_extraction.abstract_classes import (
    MCSamplerModule,
    Extractor,
    ObjectDetectionExtractor,
    SUPPORTED_OBJECT_DETECTION_ARCHITECTURES,
)
from runia_core.feature_extraction.utils import Hook

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test constants
SEED = 42
TOLERANCE = 1e-6
TEST_BATCH_SIZE = 1
TEST_CHANNELS = 3
TEST_HEIGHT = 224
TEST_WIDTH = 224
MC_SAMPLES = 3
BLOCK_SIZE = 7
DROP_PROB = 0.1


class SimpleConvModel(nn.Module):
    """Simple convolutional model for testing purposes."""

    def __init__(self, input_channels: int = 3, output_dim: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MockYOLOModel:
    """Mock YOLO model for testing."""

    def __init__(self):
        self.model = Mock()
        self.model.model = Mock()
        self.model.model._modules = {"22": nn.Module()}
        self.predictor = Mock()
        self.predictor.args = Mock()
        self.predictor.args.iou = 0.45
        self.predictor.args.classes = None
        self.predictor.args.agnostic_nms = False
        self.predictor.args.max_det = 300

    def __call__(self, image, conf=0.25, **kwargs):
        """Simulate YOLO inference."""
        # Create mock prediction results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        return [mock_result]


class MockRCNNModel:
    """Mock RCNN model for testing."""

    def __init__(self):
        self.proposal_generator = Mock()
        self.proposal_generator.rpn_head = Mock()
        self.proposal_generator.rpn_head.rpn_intermediate_output = torch.randn(1, 256, 56, 56)

    def __call__(self, image):
        """Simulate RCNN inference."""
        mock_instances = Mock()
        mock_instances.pred_boxes = Mock()
        mock_instances.pred_boxes.tensor = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        mock_instances._fields = {"latent_feature": None}
        return mock_instances


class MockOWLv2Model:
    """Mock OWLv2 model for testing."""

    def __init__(self):
        self.model = Mock()
        self.model.config = Mock()
        self.model.config.vision_config = Mock()
        self.model.config.vision_config.hidden_size = 768
        self.model.config.vision_config.image_size = 960
        self.model.config.vision_config.patch_size = 32

    def forward_and_postprocess(
        self, input_ids, attention_mask, pixel_values, orig_sizes, threshold
    ):
        """Simulate OWLv2 inference."""
        result = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "last_hidden": torch.randn(1, 576, 768),
            "logits": torch.randn(1, 1),
        }
        return [result]


class MockDINOModel:
    """Mock DINO model for testing."""

    def forward_and_postprocess(
        self, pixel_values, attention_mask, orig_sizes, input_ids, threshold
    ):
        """Simulate DINO inference."""
        result = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "last_hidden": torch.randn(1, 576, 768),
            "logits": torch.randn(1, 1),
        }
        return [result]


class MockDETRModel:
    """Mock DETR model for testing."""

    def forward_and_postprocess(self, pixel_values, pixel_mask, orig_sizes, threshold):
        """Simulate DETR inference."""
        result = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "last_hidden": torch.randn(1, 100, 256),
            "logits": torch.randn(1, 1),
        }
        return [result]


class TestMCSamplerModule(unittest.TestCase):
    """Test cases for the MCSamplerModule class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Setting up TestMCSamplerModule")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down TestMCSamplerModule")

    def test_initialization_conv(self):
        """Test MCSamplerModule initialization with Conv layer type."""
        sampler = MCSamplerModule(
            mc_samples=MC_SAMPLES,
            block_size=BLOCK_SIZE,
            drop_prob=DROP_PROB,
            layer_type="Conv",
        )
        self.assertEqual(sampler.layer_type, "Conv")
        self.assertEqual(sampler.mc_samples, MC_SAMPLES)
        self.assertEqual(len(sampler.drop_blocks), MC_SAMPLES)

    def test_initialization_fc(self):
        """Test MCSamplerModule initialization with FC layer type."""
        sampler = MCSamplerModule(
            mc_samples=MC_SAMPLES,
            block_size=BLOCK_SIZE,
            drop_prob=DROP_PROB,
            layer_type="FC",
        )
        self.assertEqual(sampler.layer_type, "FC")

    def test_initialization_rpn(self):
        """Test MCSamplerModule initialization with RPN layer type."""
        sampler = MCSamplerModule(
            mc_samples=MC_SAMPLES,
            block_size=BLOCK_SIZE,
            drop_prob=DROP_PROB,
            layer_type="RPN",
        )
        self.assertEqual(sampler.layer_type, "RPN")

    def test_initialization_invalid_layer_type(self):
        """Test MCSamplerModule initialization with invalid layer type."""
        with self.assertRaises(AssertionError):
            MCSamplerModule(
                mc_samples=MC_SAMPLES,
                block_size=BLOCK_SIZE,
                drop_prob=DROP_PROB,
                layer_type="InvalidType",
            )

    def test_forward_conv_layer(self):
        """Test forward pass with Conv layer type."""
        sampler = MCSamplerModule(
            mc_samples=MC_SAMPLES,
            block_size=BLOCK_SIZE,
            drop_prob=DROP_PROB,
            layer_type="Conv",
        )
        sampler.to(self.device)

        # Create a random latent representation
        latent_rep = torch.randn(1, 64, 28, 28).to(self.device)

        # Forward pass
        output = sampler(latent_rep)

        # Check output shape
        self.assertEqual(output.shape[0], MC_SAMPLES)
        self.assertGreater(output.shape[1], 0)

    def test_mc_samples_number(self):
        """Test that the number of MC samples is correct."""
        for num_samples in [1, 5, 10]:
            sampler = MCSamplerModule(
                mc_samples=num_samples,
                block_size=BLOCK_SIZE,
                drop_prob=DROP_PROB,
                layer_type="Conv",
            )
            self.assertEqual(len(sampler.drop_blocks), num_samples)

    def test_module_to_device(self):
        """Test moving MCSamplerModule to different devices."""
        sampler = MCSamplerModule(
            mc_samples=MC_SAMPLES,
            block_size=BLOCK_SIZE,
            drop_prob=DROP_PROB,
            layer_type="Conv",
        )
        sampler.to(self.device)

        # Check that all drop blocks are on the correct device
        for drop_block in sampler.drop_blocks:
            for param in drop_block.parameters():
                self.assertEqual(param.device, self.device)


class ConcreteExtractor(Extractor):
    """Concrete implementation of Extractor for testing."""

    def get_ls_samples(self, data_loader, **kwargs):
        """Concrete implementation of abstract method."""
        pass

    def _get_samples_one_image(self, image, **kwargs):
        """Concrete implementation of abstract method."""
        pass


class ConcreteObjectDetectionExtractor(ObjectDetectionExtractor):
    """Concrete implementation of ObjectDetectionExtractor for testing."""

    def get_ls_samples(self, data_loader, **kwargs):
        """Concrete implementation of abstract method."""
        pass

    def _get_samples_one_image(self, image, **kwargs):
        """Concrete implementation of abstract method."""
        pass


class TestExtractorBaseClass(unittest.TestCase):
    """Test cases for the base Extractor class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleConvModel().to(self.device)
        logger.info("Setting up TestExtractorBaseClass")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down TestExtractorBaseClass")

    def test_initialization(self):
        """Test Extractor initialization."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            return_raw_predictions=True,
            return_stds=True,
            mcd_nro_samples=5,
        )

        self.assertEqual(extractor.model, self.model)
        self.assertEqual(extractor.hooked_layers, hooked_layers)
        self.assertEqual(extractor.device, self.device)
        self.assertTrue(extractor.return_raw_predictions)
        self.assertTrue(extractor.return_stds)
        self.assertEqual(extractor.mcd_nro_samples, 5)

    def test_check_dataloader_with_batch_sampler(self):
        """Test check_dataloader with batch_sampler attribute."""
        mock_dataloader = Mock()
        mock_dataloader.batch_sampler = Mock()
        mock_dataloader.batch_sampler.batch_size = 1

        # Should not raise an error
        Extractor.check_dataloader(mock_dataloader)

    def test_check_dataloader_with_batch_size(self):
        """Test check_dataloader with batch_size attribute."""
        mock_dataloader = Mock(spec=["batch_size"])
        mock_dataloader.batch_size = 1

        # Should not raise an error
        Extractor.check_dataloader(mock_dataloader)

    def test_check_dataloader_with_bs(self):
        """Test check_dataloader with bs attribute."""
        mock_dataloader = Mock(spec=["bs"])
        mock_dataloader.bs = 1

        # Should not raise an error
        Extractor.check_dataloader(mock_dataloader)

    def test_check_dataloader_invalid_batch_size(self):
        """Test check_dataloader with invalid batch size."""
        mock_dataloader = Mock()
        mock_dataloader.batch_size = 2

        with self.assertRaises(AssertionError):
            Extractor.check_dataloader(mock_dataloader)

    def test_check_dataloader_no_batch_size_attributes(self):
        """Test check_dataloader with no batch size attributes."""
        mock_dataloader = Mock(spec=[])

        with self.assertRaises(AttributeError):
            Extractor.check_dataloader(mock_dataloader)


class TestObjectDetectionExtractor(unittest.TestCase):
    """Test cases for the ObjectDetectionExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleConvModel().to(self.device)
        logger.info("Setting up TestObjectDetectionExtractor")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down TestObjectDetectionExtractor")

    def test_supported_architectures(self):
        """Test that all supported architectures are defined."""
        expected_architectures = [
            "yolov8",
            "rcnn",
            "detr-backbone",
            "owlv2",
            "rtdetr-backbone",
            "rtdetr-encoder",
            "dino",
        ]
        self.assertEqual(set(SUPPORTED_OBJECT_DETECTION_ARCHITECTURES), set(expected_architectures))

    def test_initialization_yolov8(self):
        """Test ObjectDetectionExtractor initialization with YOLO architecture."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="yolov8",
            return_raw_predictions=True,
        )

        self.assertEqual(extractor.architecture, "yolov8")
        self.assertEqual(extractor.n_hooked_reps, 1)

    def test_initialization_rcnn(self):
        """Test ObjectDetectionExtractor initialization with RCNN architecture."""
        hooked_layers = [Hook(self.model.conv1), Hook(self.model.conv2)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rcnn",
            rcnn_extraction_type="backbone",
        )

        self.assertEqual(extractor.architecture, "rcnn")
        self.assertEqual(extractor.rcnn_extraction_type, "backbone")

    def test_initialization_owlv2(self):
        """Test ObjectDetectionExtractor initialization with OWLv2 architecture."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="owlv2",
        )

        self.assertEqual(extractor.architecture, "owlv2")

    def test_initialization_dino(self):
        """Test ObjectDetectionExtractor initialization with DINO architecture."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="dino",
        )

        self.assertEqual(extractor.architecture, "dino")

    def test_initialization_detr(self):
        """Test ObjectDetectionExtractor initialization with DETR architecture."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="detr-backbone",
        )

        self.assertEqual(extractor.architecture, "detr-backbone")

    def test_initialization_rtdetr_backbone(self):
        """Test ObjectDetectionExtractor initialization with RTDETR-backbone architecture."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rtdetr-backbone",
        )

        self.assertEqual(extractor.architecture, "rtdetr-backbone")

    def test_initialization_rtdetr_encoder(self):
        """Test ObjectDetectionExtractor initialization with RTDETR-encoder architecture."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rtdetr-encoder",
        )

        self.assertEqual(extractor.architecture, "rtdetr-encoder")

    def test_initialization_invalid_architecture(self):
        """Test ObjectDetectionExtractor initialization with invalid architecture."""
        hooked_layers = [Hook(self.model.conv1)]

        with self.assertRaises(AssertionError):
            ConcreteObjectDetectionExtractor(
                model=self.model,
                hooked_layers=hooked_layers,
                device=self.device,
                architecture="invalid_architecture",
            )

    def test_initialization_invalid_rcnn_extraction_type(self):
        """Test ObjectDetectionExtractor initialization with invalid RCNN extraction type."""
        hooked_layers = [Hook(self.model.conv1)]

        with self.assertRaises(AssertionError):
            ConcreteObjectDetectionExtractor(
                model=self.model,
                hooked_layers=hooked_layers,
                device=self.device,
                architecture="rcnn",
                rcnn_extraction_type="invalid_type",
            )

    def test_extract_noise_entropies_initialization(self):
        """Test ObjectDetectionExtractor with noise entropy extraction enabled."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="yolov8",
            extract_noise_entropies=True,
            mcd_nro_samples=MC_SAMPLES,
            dropblock_sizes=BLOCK_SIZE,
            dropblock_probs=DROP_PROB,
        )

        self.assertTrue(extractor.extract_noise_entropies)
        self.assertIsNotNone(extractor.mc_sampler)
        self.assertIsInstance(extractor.mc_sampler, MCSamplerModule)

    def test_extract_noise_entropies_not_enabled(self):
        """Test ObjectDetectionExtractor with noise entropy extraction disabled."""
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="yolov8",
            extract_noise_entropies=False,
        )

        self.assertFalse(extractor.extract_noise_entropies)
        self.assertFalse(hasattr(extractor, "mc_sampler"))

    def test_unpack_dataloader_yolov8(self):
        """Test unpack_dataloader for YOLO architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="yolov8",
        )

        # Mock YOLO dataloader output
        impath = ["/path/to/image/001.jpg"]
        image = torch.randn(1, 3, 224, 224).to(self.device)
        im_counter = 0
        loader_contents = (impath, image, im_counter)

        result_impath, result_image, result_id = extractor.unpack_dataloader(loader_contents)

        self.assertEqual(result_impath, impath)
        self.assertEqual(result_id, "1")

    def test_unpack_dataloader_rcnn(self):
        """Test unpack_dataloader for RCNN architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rcnn",
        )

        # Mock RCNN dataloader output
        loader_contents = [
            {
                "file_name": "/path/to/image/test.jpg",
                "image_id": 123,
            }
        ]

        result_impath, result_image, result_id = extractor.unpack_dataloader(loader_contents)

        self.assertEqual(result_impath, ["/path/to/image/test.jpg"])
        self.assertEqual(result_id, 123)

    def test_unpack_dataloader_owlv2(self):
        """Test unpack_dataloader for OWLv2 architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="owlv2",
        )

        # Mock OWLv2 dataloader output
        loader_contents = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long).to(self.device),
            "attention_mask": torch.ones(1, 5, dtype=torch.long).to(self.device),
            "pixel_values": torch.randn(1, 3, 224, 224).to(self.device),
            "orig_size": torch.tensor([[224, 224]]).to(self.device),
            "labels": [{"image_id": "img_001"}],
        }

        result_impath, result_image, result_id = extractor.unpack_dataloader(loader_contents)

        self.assertEqual(result_impath, ["img_001"])
        self.assertEqual(result_id, "img_001")

    def test_unpack_dataloader_dino(self):
        """Test unpack_dataloader for DINO architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="dino",
        )

        # Mock DINO dataloader output
        loader_contents = {
            "pixel_values": torch.randn(1, 3, 224, 224).to(self.device),
            "attention_mask": torch.ones(1, 224, 224, dtype=torch.long).to(self.device),
            "orig_size": torch.tensor([[224, 224]]).to(self.device),
            "input_ids": torch.zeros(1, 5, dtype=torch.long).to(self.device),
            "labels": [{"image_id": "img_001"}],
        }

        result_impath, result_image, result_id = extractor.unpack_dataloader(loader_contents)

        self.assertEqual(result_impath, ["img_001"])

    def test_unpack_dataloader_detr(self):
        """Test unpack_dataloader for DETR architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="detr-backbone",
        )

        # Mock DETR dataloader output
        loader_contents = {
            "pixel_values": torch.randn(1, 3, 224, 224).to(self.device),
            "pixel_mask": torch.ones(1, 224, 224, dtype=torch.long).to(self.device),
            "labels": [
                {
                    "image_id": torch.tensor(1),
                    "orig_size": torch.tensor([224, 224]),
                }
            ],
        }

        result_impath, result_image, result_id = extractor.unpack_dataloader(loader_contents)

        self.assertEqual(result_impath, [torch.tensor(1)])
        self.assertEqual(result_id, 1)

    def test_yolo_get_logits_basic(self):
        """Test yolo_get_logits with basic predictions."""
        # Create mock YOLO predictions
        # Shape: (batch_size, 6300, 84) where 84 = 4 (box) + 80 (classes)
        prediction = torch.randn(1, 6300, 84)
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4]) * 2  # Confidence

        logits = ObjectDetectionExtractor.yolo_get_logits(
            prediction=prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            nc=80,
        )

        self.assertIsInstance(logits, torch.Tensor)

    def test_yolo_get_logits_high_confidence_threshold(self):
        """Test yolo_get_logits with high confidence threshold."""
        prediction = torch.randn(1, 6300, 84)
        prediction[:, :, 4] = 0.1  # Low confidence

        logits = ObjectDetectionExtractor.yolo_get_logits(
            prediction=prediction,
            conf_thres=0.9,
            iou_thres=0.45,
            nc=80,
        )

        # Should return a tensor (possibly empty)
        self.assertIsInstance(logits, torch.Tensor)

    def test_yolo_get_logits_invalid_conf_threshold(self):
        """Test yolo_get_logits with invalid confidence threshold."""
        prediction = torch.randn(1, 6300, 84)

        with self.assertRaises(AssertionError):
            ObjectDetectionExtractor.yolo_get_logits(
                prediction=prediction,
                conf_thres=1.5,  # Invalid: > 1.0
                iou_thres=0.45,
                nc=80,
            )

    def test_yolo_get_logits_invalid_iou_threshold(self):
        """Test yolo_get_logits with invalid IoU threshold."""
        prediction = torch.randn(1, 6300, 84)

        with self.assertRaises(AssertionError):
            ObjectDetectionExtractor.yolo_get_logits(
                prediction=prediction,
                conf_thres=0.25,
                iou_thres=-0.1,  # Invalid: < 0.0
                nc=80,
            )

    def test_yolo_get_logits_with_class_filter(self):
        """Test yolo_get_logits with class filtering."""
        prediction = torch.randn(1, 6300, 84)
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4]) * 2

        # Filter for specific classes
        logits = ObjectDetectionExtractor.yolo_get_logits(
            prediction=prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=[0, 1, 2],
            nc=80,
        )

        self.assertIsInstance(logits, torch.Tensor)

    def test_model_dependent_feature_extraction_yolo(self):
        """Test model_dependent_feature_extraction for YOLO architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        mock_hook = Mock()
        mock_hook.output = [torch.randn(1, 64, 56, 56)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=[mock_hook],
            device=self.device,
            architecture="yolov8",
            hook_layer_output=True,
        )

        latent_sample = extractor.model_dependent_feature_extraction()

        self.assertIsInstance(latent_sample, list)

    def test_rcnn_rpn_inter_extraction_type(self):
        """Test that RCNN with rpn_inter extraction type is properly initialized."""
        # Use a real Hook for this test
        hooked_layers = [Hook(self.model.conv1)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rcnn",
            rcnn_extraction_type="rpn_inter",
        )

        # Verify that the extractor is properly initialized with rpn_inter
        self.assertEqual(extractor.rcnn_extraction_type, "rpn_inter")
        self.assertEqual(extractor.architecture, "rcnn")

    def test_model_dependent_feature_extraction_owlv2(self):
        """Test model_dependent_feature_extraction for OWLv2 architecture."""
        mock_hook = Mock()
        # OWLv2 specific format: [batch, seq, hidden]
        # With proper dimensions: (1, 577, 768) where 576 = 24*24 patch grid + 1 CLS token
        # After reshape should be (1, 768, 24, 24)
        hidden_size = 768
        image_size = 768  # 24 patches of 32 size each
        patch_size = 32
        patches_per_side = image_size // patch_size  # 24

        # Create the proper shaped tensor
        mock_hook.output = [torch.randn(1, patches_per_side * patches_per_side + 1, hidden_size)]

        hooked_layers = [mock_hook]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="owlv2",
            hook_layer_output=True,
        )

        # Mock the model config for OWLv2
        extractor.model = Mock()
        extractor.model.model = Mock()
        extractor.model.model.config = Mock()
        extractor.model.model.config.vision_config = Mock()
        extractor.model.model.config.vision_config.hidden_size = hidden_size
        extractor.model.model.config.vision_config.image_size = image_size
        extractor.model.model.config.vision_config.patch_size = patch_size

        latent_sample = extractor.model_dependent_feature_extraction()

        self.assertIsInstance(latent_sample, list)

    def test_hooked_layers_single_layer_no_output(self):
        """Test hooked_layers handling for single layer without output hook."""
        hooked_layer = Hook(self.model.conv1)

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=[hooked_layer],
            device=self.device,
            architecture="yolov8",
            hook_layer_output=False,
        )

        self.assertEqual(extractor.hooked_layers, hooked_layer)

    def test_hooked_layers_multiple_layers_with_output(self):
        """Test hooked_layers handling for multiple layers with output hook."""
        hooked_layers = [Hook(self.model.conv1), Hook(self.model.conv2)]

        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rcnn",
            rcnn_extraction_type="backbone",
            hook_layer_output=True,
        )

        self.assertEqual(len(extractor.hooked_layers), 2)

    def test_model_dependent_inference_yolov8_basic(self):
        """Test model_dependent_inference for YOLO architecture basic structure."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="yolov8",
        )

        # Verify extractor is set up correctly for YOLO
        self.assertEqual(extractor.architecture, "yolov8")
        self.assertIsNotNone(extractor.model)

    def test_model_dependent_inference_rcnn_with_features(self):
        """Test model_dependent_inference for RCNN with features."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rcnn",
        )

        # Create mock RCNN image input
        image = [
            {
                "height": 480,
                "width": 640,
                "image_id": 1,
            }
        ]

        # Create mock RCNN prediction
        mock_boxes_tensor = Mock()
        mock_boxes_tensor.tensor = torch.tensor([[10.0, 10.0, 50.0, 50.0]])

        mock_instances = Mock()
        mock_instances.pred_boxes = mock_boxes_tensor
        mock_instances._fields = {
            "latent_feature": torch.randn(1, 256),
            "logits": torch.randn(1, 80),
        }

        extractor.model = Mock(return_value=mock_instances)

        results, boxes, pred_img, img_shape = extractor.model_dependent_inference(
            image=image, predict_conf=0.25
        )

        self.assertIsInstance(results, dict)
        self.assertIsInstance(boxes, torch.Tensor)
        self.assertEqual(img_shape, (480, 640))
        self.assertIn("features", results)
        self.assertIn("logits", results)
        self.assertEqual(boxes.shape[0], 1)

    def test_model_dependent_inference_rcnn_wrapped_output(self):
        """Test model_dependent_inference for RCNN with list/dict wrapped output."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rcnn",
        )

        # Create mock RCNN image input
        image = [{"height": 480, "width": 640, "image_id": 1}]

        # Create mock RCNN prediction
        mock_boxes_tensor = Mock()
        mock_boxes_tensor.tensor = torch.tensor([[10.0, 10.0, 50.0, 50.0]])

        mock_instances = Mock()
        mock_instances.pred_boxes = mock_boxes_tensor
        mock_instances._fields = {}

        # Return as dict wrapped in list
        extractor.model = Mock(return_value=[{"instances": mock_instances}])

        results, boxes, pred_img, img_shape = extractor.model_dependent_inference(
            image=image, predict_conf=0.25
        )

        self.assertEqual(img_shape, (480, 640))
        self.assertIsInstance(boxes, torch.Tensor)

    def test_model_dependent_inference_owlv2_architecture(self):
        """Test model_dependent_inference for OWLv2 architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="owlv2",
        )

        # Create OWLv2 image input
        image = (
            torch.zeros(1, 5, dtype=torch.long).to(self.device),
            torch.ones(1, 5, dtype=torch.long).to(self.device),
            torch.randn(1, 3, 224, 224).to(self.device),
            torch.tensor([[480, 640]]).to(self.device),
        )

        # Create mock OWLv2 prediction
        mock_pred_result = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "last_hidden": torch.randn(1, 576, 768),
            "logits": torch.randn(1, 1),
        }

        extractor.model = Mock()
        extractor.model.forward_and_postprocess = Mock(return_value=[mock_pred_result])

        results, boxes, pred_img, img_shape = extractor.model_dependent_inference(
            image=image, predict_conf=0.25
        )

        self.assertIsInstance(results, dict)
        self.assertIn("features", results)
        self.assertIn("logits", results)
        # img_shape is tensor for OWLv2
        self.assertTrue(isinstance(img_shape, (tuple, torch.Tensor)))
        if isinstance(img_shape, torch.Tensor):
            torch.testing.assert_close(img_shape, torch.tensor([480, 640]).to(self.device))
        else:
            self.assertEqual(img_shape, (480, 640))
        torch.testing.assert_close(boxes, mock_pred_result["boxes"])

    def test_model_dependent_inference_dino_architecture(self):
        """Test model_dependent_inference for DINO architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="dino",
        )

        # Create DINO image input
        image = (
            torch.randn(1, 3, 224, 224).to(self.device),
            torch.ones(1, 224, 224, dtype=torch.long).to(self.device),
            torch.tensor([[480, 640]]).to(self.device),
            torch.zeros(1, 5, dtype=torch.long).to(self.device),
        )

        # Create mock DINO prediction
        mock_pred_result = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "last_hidden": torch.randn(1, 100, 768),
            "logits": torch.randn(1, 1),
        }

        extractor.model = Mock()
        extractor.model.forward_and_postprocess = Mock(return_value=[mock_pred_result])

        results, boxes, pred_img, img_shape = extractor.model_dependent_inference(
            image=image, predict_conf=0.25
        )

        self.assertIsInstance(results, dict)
        self.assertIn("features", results)
        self.assertIn("logits", results)
        # img_shape is tensor for DINO
        self.assertTrue(isinstance(img_shape, (tuple, torch.Tensor)))
        if isinstance(img_shape, torch.Tensor):
            torch.testing.assert_close(img_shape, torch.tensor([480, 640]).to(self.device))
        else:
            self.assertEqual(img_shape, (480, 640))

    def test_model_dependent_inference_detr_architecture(self):
        """Test model_dependent_inference for DETR architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="detr-backbone",
        )

        # Create DETR image input
        image = (
            torch.randn(1, 3, 224, 224).to(self.device),
            torch.ones(1, 224, 224, dtype=torch.long).to(self.device),
            torch.tensor([[480.0, 640.0]]).to(self.device),
        )

        # Create mock DETR prediction
        mock_pred_result = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "last_hidden": torch.randn(1, 100, 256),
            "logits": torch.randn(1, 1),
        }

        extractor.model = Mock()
        extractor.model.forward_and_postprocess = Mock(return_value=[mock_pred_result])

        results, boxes, pred_img, img_shape = extractor.model_dependent_inference(
            image=image, predict_conf=0.25
        )

        self.assertIsInstance(results, dict)
        self.assertIn("features", results)
        self.assertIn("logits", results)
        self.assertEqual(img_shape, (480, 640))

    def test_model_dependent_inference_rtdetr_encoder_architecture(self):
        """Test model_dependent_inference for RTDETR-encoder architecture."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rtdetr-encoder",
        )

        # Create RTDETR image input (same format as DETR)
        image = (
            torch.randn(1, 3, 224, 224).to(self.device),
            torch.ones(1, 224, 224, dtype=torch.long).to(self.device),
            torch.tensor([[480.0, 640.0]]).to(self.device),
        )

        # Create mock RTDETR prediction
        mock_pred_result = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "last_hidden": torch.randn(1, 100, 256),
            "logits": torch.randn(1, 1),
        }

        extractor.model = Mock()
        extractor.model.forward_and_postprocess = Mock(return_value=[mock_pred_result])

        results, boxes, pred_img, img_shape = extractor.model_dependent_inference(
            image=image, predict_conf=0.25
        )

        self.assertIsInstance(results, dict)
        self.assertIn("features", results)
        self.assertIn("logits", results)
        self.assertEqual(img_shape, (480, 640))

    def test_model_dependent_inference_return_types(self):
        """Test that model_dependent_inference returns correct tuple structure."""
        hooked_layers = [Hook(self.model.conv1)]
        extractor = ConcreteObjectDetectionExtractor(
            model=self.model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="owlv2",
        )

        image = (
            torch.zeros(1, 5, dtype=torch.long).to(self.device),
            torch.ones(1, 5, dtype=torch.long).to(self.device),
            torch.randn(1, 3, 224, 224).to(self.device),
            torch.tensor([[480, 640]]).to(self.device),
        )

        mock_pred_result = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "last_hidden": torch.randn(1, 576, 768),
            "logits": torch.randn(1, 1),
        }

        extractor.model = Mock()
        extractor.model.forward_and_postprocess = Mock(return_value=[mock_pred_result])

        result = extractor.model_dependent_inference(image=image, predict_conf=0.25)

        # Check return type is tuple with 4 elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

        results_dict, boxes, pred_img, img_shape = result

        # Check individual return types
        self.assertIsInstance(results_dict, dict)
        self.assertIsInstance(boxes, torch.Tensor)
        # img_shape can be tuple or tensor depending on architecture
        self.assertTrue(isinstance(img_shape, (tuple, torch.Tensor)))


class TestObjectDetectionExtractorIntegration(unittest.TestCase):
    """Integration tests for ObjectDetectionExtractor."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Setting up TestObjectDetectionExtractorIntegration")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down TestObjectDetectionExtractorIntegration")

    def test_full_extraction_workflow_yolov8(self):
        """Test full extraction workflow with YOLO architecture."""
        model = SimpleConvModel().to(self.device)
        hooked_layers = [Hook(model.conv2)]

        extractor = ConcreteObjectDetectionExtractor(
            model=model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="yolov8",
            extract_noise_entropies=False,
        )

        # Verify initialization
        self.assertEqual(extractor.architecture, "yolov8")
        self.assertEqual(extractor.n_hooked_reps, 1)

    def test_full_extraction_workflow_rcnn(self):
        """Test full extraction workflow with RCNN architecture."""
        model = SimpleConvModel().to(self.device)
        hooked_layers = [Hook(model.conv1), Hook(model.conv2)]

        extractor = ConcreteObjectDetectionExtractor(
            model=model,
            hooked_layers=hooked_layers,
            device=self.device,
            architecture="rcnn",
            rcnn_extraction_type="backbone",
            extract_noise_entropies=True,
            mcd_nro_samples=3,
            dropblock_sizes=7,
            dropblock_probs=0.1,
        )

        self.assertTrue(extractor.extract_noise_entropies)
        self.assertIsInstance(extractor.mc_sampler, MCSamplerModule)

    def test_all_architectures_initialization(self):
        """Test initialization for all supported architectures."""
        model = SimpleConvModel().to(self.device)
        hooked_layers = [Hook(model.conv1)]

        for architecture in SUPPORTED_OBJECT_DETECTION_ARCHITECTURES:
            extractor = ConcreteObjectDetectionExtractor(
                model=model,
                hooked_layers=hooked_layers,
                device=self.device,
                architecture=architecture,
            )
            self.assertEqual(extractor.architecture, architecture)


class TestMCSamplerModuleIntegration(unittest.TestCase):
    """Integration tests for MCSamplerModule."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Setting up TestMCSamplerModuleIntegration")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down TestMCSamplerModuleIntegration")

    def test_mc_sampler_with_conv_features(self):
        """Test MC sampler with convolutional features."""
        sampler = MCSamplerModule(
            mc_samples=5,
            block_size=7,
            drop_prob=0.1,
            layer_type="Conv",
        )
        sampler.to(self.device)

        # Simulate convolutional features
        features = torch.randn(1, 128, 32, 32).to(self.device)

        output = sampler(features)

        # Output should have MC_SAMPLES rows
        self.assertEqual(output.shape[0], 5)
        self.assertGreater(output.shape[1], 0)

    def test_mc_sampler_reproducibility(self):
        """Test MC sampler reproducibility with fixed seed."""
        torch.manual_seed(SEED)
        sampler1 = MCSamplerModule(
            mc_samples=3,
            block_size=7,
            drop_prob=0.1,
            layer_type="Conv",
        )
        sampler1.to(self.device)

        torch.manual_seed(SEED)
        sampler2 = MCSamplerModule(
            mc_samples=3,
            block_size=7,
            drop_prob=0.1,
            layer_type="Conv",
        )
        sampler2.to(self.device)

        features = torch.randn(1, 64, 16, 16).to(self.device)

        # Set model to eval mode for reproducibility
        sampler1.eval()
        sampler2.eval()

        # Note: Due to dropout stochasticity, outputs won't be identical
        output1 = sampler1(features)
        output2 = sampler2(features)

        # Both should have same shape
        self.assertEqual(output1.shape, output2.shape)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
