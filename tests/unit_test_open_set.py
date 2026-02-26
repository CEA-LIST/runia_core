import json
import os
import tempfile
from unittest import TestCase, main

import numpy as np
import pandas as pd
import torch

from runia_core.evaluation.open_set import (
    COCOParser,
    OpenSetEvaluator,
    convert_osod_results_to_pandas_df,
    convert_osod_results_to_hierarchical_pandas_df,
    convert_osod_results_for_mlflow_logging,
    plot_two_osod_datasets_metrics,
    plot_two_osod_datasets_per_metric,
    voc_eval,
    _compute_overlaps,
    voc_ap,
    get_boxes_from_precalculated,
    get_labels_and_scores_from_logits,
    convert_xywh_to_xyxy,
    get_n_unk_ood_dataset,
    evaluate_open_set_detection_one_method,
    get_overall_open_set_results,
)

# Test parameters
SEED = 42
NUM_CLASSES = 3
TEST_IMAGE_SIZE = 100


class TestCOCOParser(TestCase):
    """Test suite for COCOParser class."""

    def setUp(self):
        """Create a temporary COCO format JSON file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.coco_path = os.path.join(self.temp_dir.name, "annotations.json")

        # Create sample COCO data
        self.coco_data = {
            "info": {"description": "Test dataset"},
            "licenses": [{"id": 1, "name": "Test License"}],
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640, "license": 1},
                {"id": 2, "file_name": "img2.jpg", "height": 480, "width": 640, "license": 1},
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [100, 100, 60, 60],
                    "area": 3600,
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [50, 50, 40, 40],
                    "area": 1600,
                    "iscrowd": 0,
                },
            ],
        }

        with open(self.coco_path, "w") as f:
            json.dump(self.coco_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_coco_parser_initialization(self):
        """Test COCOParser initialization."""
        parser = COCOParser(self.coco_path)
        self.assertIsNotNone(parser)
        self.assertEqual(len(parser.cat_dict), 2)
        self.assertEqual(len(parser.im_dict), 2)

    def test_get_imgIds(self):
        """Test retrieving image IDs."""
        parser = COCOParser(self.coco_path)
        img_ids = parser.get_imgIds()
        self.assertEqual(len(img_ids), 2)
        self.assertIn(1, img_ids)
        self.assertIn(2, img_ids)

    def test_get_annIds(self):
        """Test retrieving annotation IDs for given image IDs."""
        parser = COCOParser(self.coco_path)
        ann_ids = parser.get_annIds(1)
        self.assertEqual(len(ann_ids), 2)
        self.assertIn(1, ann_ids)
        self.assertIn(2, ann_ids)

    def test_get_annIds_multiple_images(self):
        """Test retrieving annotation IDs for multiple image IDs."""
        parser = COCOParser(self.coco_path)
        ann_ids = parser.get_annIds([1, 2])
        self.assertEqual(len(ann_ids), 3)

    def test_load_anns(self):
        """Test loading annotations by ID."""
        parser = COCOParser(self.coco_path)
        anns = parser.load_anns([1, 2])
        self.assertEqual(len(anns), 2)
        self.assertEqual(anns[0]["id"], 1)
        self.assertEqual(anns[1]["id"], 2)

    def test_load_cats(self):
        """Test loading categories by ID."""
        parser = COCOParser(self.coco_path)
        cats = parser.load_cats([1, 2])
        self.assertEqual(len(cats), 2)
        self.assertEqual(cats[0]["name"], "cat")
        self.assertEqual(cats[1]["name"], "dog")

    def test_get_img_info(self):
        """Test retrieving image information."""
        parser = COCOParser(self.coco_path)
        img_info = parser.get_img_info(1)
        self.assertEqual(len(img_info), 1)
        self.assertEqual(img_info[0]["id"], 1)

    def test_get_img_ids_per_cat_name(self):
        """Test retrieving image IDs for a specific category."""
        parser = COCOParser(self.coco_path)
        img_ids = parser.get_img_ids_per_cat_name("cat")
        self.assertEqual(len(img_ids), 2)

    def test_coco_parser_with_subset(self):
        """Test COCOParser with a subset of images."""
        parser = COCOParser(self.coco_path, using_subset=[1])
        self.assertEqual(len(parser.im_dict), 1)
        self.assertEqual(len(parser.annIm_dict), 1)

    def test_cat_dict_count(self):
        """Test that category counts are computed correctly."""
        parser = COCOParser(self.coco_path)
        # Cat category appears in 2 annotations
        self.assertEqual(parser.cat_dict[1]["count"], 2)
        # Dog category appears in 1 annotation
        self.assertEqual(parser.cat_dict[2]["count"], 1)


class TestOpenSetEvaluator(TestCase):
    """Test suite for OpenSetEvaluator class."""

    def setUp(self):
        """Create a temporary COCO format JSON file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.coco_path = os.path.join(self.temp_dir.name, "annotations.json")

        self.coco_data = {
            "info": {"description": "Test dataset"},
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640},
                {"id": 2, "file_name": "img2.jpg", "height": 480, "width": 640},
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 2,
                    "bbox": [100, 100, 60, 60],
                    "area": 3600,
                    "iscrowd": 0,
                },
            ],
        }

        with open(self.coco_path, "w") as f:
            json.dump(self.coco_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_evaluator_initialization(self):
        """Test OpenSetEvaluator initialization."""
        evaluator = OpenSetEvaluator(
            id_dataset_name="test_dataset",
            ground_truth_annotations_path=self.coco_path,
            metric_2007=True,
        )
        self.assertEqual(evaluator._dataset_name, "test_dataset")
        self.assertEqual(evaluator.total_num_class, 3)  # 2 known + 1 unknown
        self.assertEqual(evaluator.num_known_classes, 2)
        self.assertEqual(evaluator.unknown_class_index, 2)
        self.assertIn("unknown", evaluator._class_names)

    def test_evaluator_reset(self):
        """Test resetting the evaluator."""
        evaluator = OpenSetEvaluator(
            id_dataset_name="test_dataset",
            ground_truth_annotations_path=self.coco_path,
            metric_2007=True,
        )
        evaluator._predictions[0].append("test")
        evaluator.reset()
        self.assertEqual(len(evaluator._predictions), 0)

    def test_evaluator_process(self):
        """Test processing predictions."""
        evaluator = OpenSetEvaluator(
            id_dataset_name="test_dataset",
            ground_truth_annotations_path=self.coco_path,
            metric_2007=True,
        )
        boxes = np.array([[10, 10, 50, 50], [100, 100, 160, 160]])
        scores = np.array([0.95, 0.85])
        classes = np.array([0, 1])

        evaluator.process(image_id=1, boxes=boxes, scores=scores, classes=classes)
        self.assertEqual(len(evaluator._predictions[0]), 1)
        self.assertEqual(len(evaluator._predictions[1]), 1)

    def test_evaluator_known_classes(self):
        """Test that known classes are correctly set."""
        evaluator = OpenSetEvaluator(
            id_dataset_name="test_dataset",
            ground_truth_annotations_path=self.coco_path,
            metric_2007=True,
        )
        self.assertEqual(len(evaluator.known_classes), 2)
        self.assertIn("cat", evaluator.known_classes)
        self.assertIn("dog", evaluator.known_classes)

    def test_evaluator_evaluate(self):
        """Test evaluator.evaluate method."""
        evaluator = OpenSetEvaluator(
            id_dataset_name="test_dataset",
            ground_truth_annotations_path=self.coco_path,
            metric_2007=True,
        )
        # Add some predictions
        boxes = np.array([[10, 10, 50, 50], [100, 100, 160, 160]])
        scores = np.array([0.95, 0.85])
        classes = np.array([0, 1])
        evaluator.process(image_id=1, boxes=boxes, scores=scores, classes=classes)

        # Evaluate on the same annotations
        results = evaluator.evaluate(
            test_annotations_path=self.coco_path,
            is_ood=False,
            get_known_classes_metrics=True,
            using_subset=False,
        )

        # Check that results is a dictionary with expected metrics
        self.assertIsInstance(results, dict)
        self.assertIn("AP_U", results)
        self.assertIn("P_U", results)
        self.assertIn("R_U", results)


class TestUtilityFunctions(TestCase):
    """Test suite for utility functions."""

    def test_convert_xywh_to_xyxy(self):
        """Test converting bounding box format from xywh to xyxy."""
        bbox = [10, 20, 30, 40]  # x, y, width, height
        result = convert_xywh_to_xyxy(bbox)
        expected = [10, 20, 40, 60]  # x_min, y_min, x_max, y_max
        self.assertEqual(result, expected)

    def test_get_boxes_from_precalculated_numpy(self):
        """Test getting boxes from numpy array."""
        boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        result = get_boxes_from_precalculated(boxes)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 4))

    def test_get_boxes_from_precalculated_torch(self):
        """Test getting boxes from torch tensor."""
        boxes = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float)
        result = get_boxes_from_precalculated(boxes)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 4))

    def test_get_boxes_from_precalculated_list(self):
        """Test getting boxes from list."""
        boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]
        result = get_boxes_from_precalculated(boxes)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 4))

    def test_get_boxes_from_precalculated_invalid(self):
        """Test error handling for invalid input."""
        with self.assertRaises(ValueError):
            get_boxes_from_precalculated("invalid")

    def test_get_labels_and_scores_from_logits_numpy(self):
        """Test extracting labels and scores from numpy logits."""
        np.random.seed(SEED)
        logits = np.random.randn(5, 3)
        labels, scores = get_labels_and_scores_from_logits(logits)
        self.assertEqual(len(labels), 5)
        self.assertEqual(len(scores), 5)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_get_labels_and_scores_from_logits_torch(self):
        """Test extracting labels and scores from torch logits."""
        torch.manual_seed(SEED)
        logits = torch.randn(5, 3)
        labels, scores = get_labels_and_scores_from_logits(logits)
        self.assertEqual(len(labels), 5)
        self.assertEqual(len(scores), 5)

    def test_get_labels_and_scores_from_logits_list(self):
        """Test extracting labels and scores from list logits."""
        logits = [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]
        labels, scores = get_labels_and_scores_from_logits(logits)
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(scores), 2)

    def test_get_labels_and_scores_from_logits_invalid(self):
        """Test error handling for invalid logits input."""
        with self.assertRaises(ValueError):
            get_labels_and_scores_from_logits("invalid")

    def test_compute_overlaps(self):
        """Test computing IoU between bounding boxes."""
        # Ground truth box: [0, 0, 10, 10]
        BBGT = np.array([[0, 0, 10, 10]])
        # Detected box: [5, 5, 15, 15]
        bb = np.array([5, 5, 15, 15])
        overlaps = _compute_overlaps(BBGT, bb)
        # Expected IoU with +1 for box area calculation
        # Intersection: (10-5+1)*(10-5+1) = 36, Union: 11*11 + 11*11 - 36 = 206
        # IoU = 36/206 ≈ 0.1748
        self.assertAlmostEqual(overlaps[0], 36 / 206, places=4)

    def test_compute_overlaps_no_overlap(self):
        """Test IoU with no overlap."""
        BBGT = np.array([[0, 0, 10, 10]])
        bb = np.array([20, 20, 30, 30])
        overlaps = _compute_overlaps(BBGT, bb)
        self.assertEqual(overlaps[0], 0.0)

    def test_compute_overlaps_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        BBGT = np.array([[0, 0, 10, 10]])
        bb = np.array([0, 0, 10, 10])
        overlaps = _compute_overlaps(BBGT, bb)
        self.assertEqual(overlaps[0], 1.0)

    def test_voc_ap_07_metric(self):
        """Test VOC AP calculation with 2007 metric."""
        rec = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        prec = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        ap = voc_ap(rec, prec, use_07_metric=True)
        self.assertGreater(ap, 0)
        self.assertLessEqual(ap, 1)

    def test_voc_ap_all_points_metric(self):
        """Test VOC AP calculation with all points metric."""
        rec = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        prec = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        ap = voc_ap(rec, prec, use_07_metric=False)
        self.assertGreater(ap, 0)
        self.assertLessEqual(ap, 1)


class TestDataFrameConversion(TestCase):
    """Test suite for DataFrame conversion functions."""

    def test_convert_osod_results_to_pandas_df(self):
        """Test converting OSOD results to pandas DataFrame."""
        results = {
            "method1": {"mAP": 0.75, "WI": 0.85, "AOSE": 10},
            "method2": {"mAP": 0.80, "WI": 0.90, "AOSE": 5},
        }
        df = convert_osod_results_to_pandas_df(
            results, methods_names=["method1", "method2"], save_method_as_data=False
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

    def test_convert_osod_results_to_pandas_df_with_method_column(self):
        """Test converting OSOD results to DataFrame with method column."""
        results = {
            "method1": {"mAP": 0.75, "WI": 0.85},
            "method2": {"mAP": 0.80, "WI": 0.90},
        }
        df = convert_osod_results_to_pandas_df(
            results, methods_names=["method1", "method2"], save_method_as_data=True
        )
        self.assertIn("Method", df.columns)
        self.assertEqual(df.loc["method1", "Method"], "method1")

    def test_convert_osod_results_to_hierarchical_pandas_df(self):
        """Test converting OSOD results to hierarchical DataFrame."""
        results_a = {
            "method1": {"mAP": 0.75, "WI": 0.85},
            "method2": {"mAP": 0.80, "WI": 0.90},
        }
        results_b = {
            "method1": {"mAP": 0.70, "WI": 0.80},
            "method2": {"mAP": 0.75, "WI": 0.85},
        }
        df = convert_osod_results_to_hierarchical_pandas_df(
            results_a,
            results_b,
            methods_names=["method1", "method2"],
            save_method_as_data=False,
            datasets_names=["dataset_a", "dataset_b"],
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("dataset_a", df.columns.get_level_values(0))
        self.assertIn("dataset_b", df.columns.get_level_values(0))

    def test_convert_osod_results_for_mlflow_logging(self):
        """Test converting OSOD results for MLflow logging."""
        open_set_results = {
            "ood_dataset1": {
                "method1": {"mAP": 0.75, "WI": 0.85},
                "method2": {"mAP": 0.80, "WI": 0.90},
            },
            "ood_dataset2": {
                "method1": {"mAP": 0.70, "WI": 0.80},
                "method2": {"mAP": 0.75, "WI": 0.85},
            },
        }
        mlflow_dict = convert_osod_results_for_mlflow_logging(
            open_set_results,
            ood_datasets_names=["ood_dataset1", "ood_dataset2"],
            methods_names=["method1", "method2"],
        )
        self.assertIsInstance(mlflow_dict, dict)
        self.assertIn("ood_dataset1 method1 mAP", mlflow_dict)
        self.assertAlmostEqual(mlflow_dict["ood_dataset1 method1 mAP"], 0.75)


class TestPlottingFunctions(TestCase):
    """Test suite for plotting functions."""

    def test_plot_two_osod_datasets_metrics(self):
        """Test plotting OSOD metrics for two datasets."""
        results_a = pd.DataFrame(
            {
                "mAP": [0.75, 0.80],
                "WI": [0.85, 0.90],
            },
            index=["method1", "method2"],
        )
        results_b = pd.DataFrame(
            {
                "mAP": [0.70, 0.75],
                "WI": [0.80, 0.85],
            },
            index=["method1", "method2"],
        )
        fig = plot_two_osod_datasets_metrics(
            results_a,
            results_b,
            methods_names=["method1", "method2"],
            datasets_names=["dataset_a", "dataset_b"],
            metrics_to_plot=["mAP", "WI"],
            show_plot=False,
        )
        self.assertIsNotNone(fig)

    def test_plot_two_osod_datasets_per_metric(self):
        """Test plotting a specific OSOD metric for two datasets."""
        results_a = pd.DataFrame(
            {"mAP": [0.75, 0.80]},
            index=["method1", "method2"],
        )
        results_b = pd.DataFrame(
            {"mAP": [0.70, 0.75]},
            index=["method1", "method2"],
        )
        fig = plot_two_osod_datasets_per_metric(
            results_a,
            results_b,
            methods_names=["method1", "method2"],
            datasets_names=["dataset_a", "dataset_b"],
            metric_to_plot="mAP",
            show_plot=False,
        )
        self.assertIsNotNone(fig)


class TestVocEval(TestCase):
    """Test suite for VOC evaluation function."""

    def setUp(self):
        """Create temporary COCO format JSON files for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.coco_path = os.path.join(self.temp_dir.name, "annotations.json")

        self.coco_data = {
            "info": {"description": "Test dataset"},
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640},
                {"id": 2, "file_name": "img2.jpg", "height": 480, "width": 640},
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
                {"id": 3, "name": "unknown", "supercategory": "unknown"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [100, 100, 60, 60],
                    "area": 3600,
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [50, 50, 40, 40],
                    "area": 1600,
                    "iscrowd": 0,
                },
            ],
        }

        with open(self.coco_path, "w") as f:
            json.dump(self.coco_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_voc_eval_known_class(self):
        """Test VOC evaluation for known class."""
        annotations = COCOParser(self.coco_path)
        predictions = ["1 0.95 11 11 61 61", "1 0.90 101 101 161 161"]

        rec, prec, ap, unk_sum, n_unk, tp_fp_cs, fp_os = voc_eval(
            predictions, annotations, "cat", ovthresh=0.5, use_07_metric=True, is_ood=False
        )
        self.assertIsInstance(rec, np.ndarray)
        self.assertIsInstance(prec, np.ndarray)
        self.assertGreaterEqual(ap, 0)
        self.assertLessEqual(ap, 1)

    def test_voc_eval_empty_predictions(self):
        """Test VOC evaluation with empty predictions."""
        annotations = COCOParser(self.coco_path)
        predictions = [""]

        rec, prec, ap, unk_sum, n_unk, tp_fp_cs, fp_os = voc_eval(
            predictions, annotations, "cat", ovthresh=0.5, use_07_metric=True, is_ood=False
        )
        self.assertEqual(len(rec), 0)
        self.assertEqual(len(prec), 0)


class TestGetNUnkOodDataset(TestCase):
    """Test suite for get_n_unk_ood_dataset function."""

    def setUp(self):
        """Create temporary COCO format JSON file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.coco_path = os.path.join(self.temp_dir.name, "annotations.json")

        self.coco_data = {
            "info": {"description": "Test dataset"},
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640},
                {"id": 2, "file_name": "img2.jpg", "height": 480, "width": 640},
            ],
            "categories": [{"id": 1, "name": "unknown", "supercategory": "unknown"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [100, 100, 60, 60],
                    "area": 3600,
                    "iscrowd": 0,
                },
            ],
        }

        with open(self.coco_path, "w") as f:
            json.dump(self.coco_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_get_n_unk_ood_dataset(self):
        """Test counting unknown annotations in OOD dataset."""
        n_unk = get_n_unk_ood_dataset(self.coco_path)
        self.assertEqual(n_unk, 2)


class TestEvaluateOpenSetDetectionOneMethod(TestCase):
    """Test suite for evaluate_open_set_detection_one_method function."""

    def setUp(self):
        """Create temporary COCO format JSON files for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.id_coco_path = os.path.join(self.temp_dir.name, "id_annotations.json")
        self.test_coco_path = os.path.join(self.temp_dir.name, "test_annotations.json")

        # ID dataset with known classes
        self.id_coco_data = {
            "info": {"description": "ID dataset"},
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640},
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
            ],
        }

        # Test dataset (can be ID or OOD)
        self.test_coco_data = {
            "info": {"description": "Test dataset"},
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640},
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
            ],
        }

        with open(self.id_coco_path, "w") as f:
            json.dump(self.id_coco_data, f)
        with open(self.test_coco_path, "w") as f:
            json.dump(self.test_coco_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_evaluate_open_set_detection_one_method_basic(self):
        """Test basic evaluation of open set detection for one method."""
        # Create predictions dictionary
        predictions_dict = {
            1: {
                "boxes": np.array([[10, 10, 50, 50]]),
                "logits": np.array([[1.0, 0.5]]),
                "uncertainty_score": np.array([0.9]),
            }
        }

        results = evaluate_open_set_detection_one_method(
            id_dataset_name="test_dataset",
            id_gt_annotations_path=self.id_coco_path,
            predictions_dict=predictions_dict,
            method_name="uncertainty_score",
            threshold=0.5,
            test_gt_annotations_path=self.test_coco_path,
            metric_2007=True,
            evaluating_ood=False,
            get_known_classes_metrics=True,
            is_open_set_model=False,
            unk_class_number=None,
            using_subset=False,
        )

        # Check that results is a dictionary with expected metrics
        self.assertIsInstance(results, dict)
        self.assertIn("AP_U", results)
        self.assertIn("P_U", results)
        self.assertIn("R_U", results)

    def test_evaluate_open_set_detection_one_method_with_subset(self):
        """Test evaluation with a subset of image IDs."""
        predictions_dict = {
            1: {
                "boxes": np.array([[10, 10, 50, 50]]),
                "logits": np.array([[1.0, 0.5]]),
                "uncertainty_score": np.array([0.9]),
            }
        }

        results = evaluate_open_set_detection_one_method(
            id_dataset_name="test_dataset",
            id_gt_annotations_path=self.id_coco_path,
            predictions_dict=predictions_dict,
            method_name="uncertainty_score",
            threshold=0.5,
            test_gt_annotations_path=self.test_coco_path,
            metric_2007=True,
            evaluating_ood=False,
            get_known_classes_metrics=True,
            is_open_set_model=False,
            unk_class_number=None,
            using_subset=[1],
        )

        # Check that results is a dictionary
        self.assertIsInstance(results, dict)

    def test_evaluate_open_set_detection_one_method_ood(self):
        """Test evaluation for OOD dataset."""
        predictions_dict = {
            1: {
                "boxes": np.array([[10, 10, 50, 50]]),
                "logits": np.array([[1.0, 0.5]]),
                "uncertainty_score": np.array([0.9]),
            }
        }

        results = evaluate_open_set_detection_one_method(
            id_dataset_name="test_dataset",
            id_gt_annotations_path=self.id_coco_path,
            predictions_dict=predictions_dict,
            method_name="uncertainty_score",
            threshold=0.5,
            test_gt_annotations_path=self.test_coco_path,
            metric_2007=True,
            evaluating_ood=True,
            get_known_classes_metrics=False,
            is_open_set_model=False,
            unk_class_number=None,
            using_subset=False,
        )

        self.assertIsInstance(results, dict)

    def test_evaluate_open_set_detection_one_method_empty_predictions(self):
        """Test evaluation with empty predictions."""
        predictions_dict = {}

        results = evaluate_open_set_detection_one_method(
            id_dataset_name="test_dataset",
            id_gt_annotations_path=self.id_coco_path,
            predictions_dict=predictions_dict,
            method_name="uncertainty_score",
            threshold=0.5,
            test_gt_annotations_path=self.test_coco_path,
            metric_2007=True,
            evaluating_ood=False,
            get_known_classes_metrics=True,
            is_open_set_model=False,
            unk_class_number=None,
            using_subset=False,
        )

        self.assertIsInstance(results, dict)

    def test_evaluate_open_set_detection_open_set_model(self):
        """Test evaluation with is_open_set_model=True."""
        predictions_dict = {
            1: {
                "boxes": np.array([[10, 10, 50, 50]]),
                "logits": np.array([[1.0, 0.5]]),
                "uncertainty_score": np.array([0.9]),
            }
        }

        results = evaluate_open_set_detection_one_method(
            id_dataset_name="test_dataset",
            id_gt_annotations_path=self.id_coco_path,
            predictions_dict=predictions_dict,
            method_name="uncertainty_score",
            threshold=0.5,
            test_gt_annotations_path=self.test_coco_path,
            metric_2007=True,
            evaluating_ood=False,
            get_known_classes_metrics=True,
            is_open_set_model=True,
            unk_class_number=2,
            using_subset=False,
        )

        self.assertIsInstance(results, dict)


class TestGetOverallOpenSetResults(TestCase):
    """Test suite for get_overall_open_set_results function."""

    def setUp(self):
        """Create temporary COCO format JSON files for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.id_coco_path = os.path.join(self.temp_dir.name, "id_annotations.json")
        self.ood1_coco_path = os.path.join(self.temp_dir.name, "ood1_annotations.json")
        self.ood2_coco_path = os.path.join(self.temp_dir.name, "ood2_annotations.json")

        # ID dataset
        self.id_coco_data = {
            "info": {"description": "ID dataset"},
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640},
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
            ],
        }

        # OOD datasets
        self.ood_coco_data = {
            "info": {"description": "OOD dataset"},
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640},
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
            ],
        }

        with open(self.id_coco_path, "w") as f:
            json.dump(self.id_coco_data, f)
        with open(self.ood1_coco_path, "w") as f:
            json.dump(self.ood_coco_data, f)
        with open(self.ood2_coco_path, "w") as f:
            json.dump(self.ood_coco_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_get_overall_open_set_results_basic(self):
        """Test get_overall_open_set_results with basic parameters."""
        predictions = {
            1: {
                "boxes": np.array([[10, 10, 50, 50]]),
                "logits": np.array([[1.0, 0.5]]),
                "uncertainty_score": np.array([0.9]),
            }
        }

        ind_data_dict = {"valid": predictions}
        ood_data_dict = {
            "ood_dataset1": predictions,
            "ood_dataset2": predictions,
        }

        results = get_overall_open_set_results(
            ind_dataset_name="id_dataset",
            ind_gt_annotations_path=self.id_coco_path,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_datasets_names=["ood_dataset1", "ood_dataset2"],
            ood_annotations_paths={
                "ood_dataset1": self.ood1_coco_path,
                "ood_dataset2": self.ood2_coco_path,
            },
            methods_names=["uncertainty_score"],
            methods_thresholds={"uncertainty_score": 0.5},
            metric_2007=True,
            evaluate_on_ind=True,
            get_known_classes_metrics=True,
            is_open_set_model=False,
            unk_class_number=None,
            using_id_val_subset=False,
        )

        # Check structure
        self.assertIsInstance(results, dict)
        self.assertIn("id_dataset", results)
        self.assertIn("ood_dataset1", results)
        self.assertIn("ood_dataset2", results)

        # Check inner dictionaries
        self.assertIn("uncertainty_score", results["id_dataset"])
        self.assertIn("uncertainty_score", results["ood_dataset1"])
        self.assertIn("uncertainty_score", results["ood_dataset2"])

    def test_get_overall_open_set_results_without_ind_eval(self):
        """Test get_overall_open_set_results without evaluating on ID data."""
        predictions = {
            1: {
                "boxes": np.array([[10, 10, 50, 50]]),
                "logits": np.array([[1.0, 0.5]]),
                "uncertainty_score": np.array([0.9]),
            }
        }

        ind_data_dict = {"valid": predictions}
        ood_data_dict = {
            "ood_dataset1": predictions,
        }

        results = get_overall_open_set_results(
            ind_dataset_name="id_dataset",
            ind_gt_annotations_path=self.id_coco_path,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_datasets_names=["ood_dataset1"],
            ood_annotations_paths={
                "ood_dataset1": self.ood1_coco_path,
            },
            methods_names=["uncertainty_score"],
            methods_thresholds={"uncertainty_score": 0.5},
            metric_2007=True,
            evaluate_on_ind=False,
            get_known_classes_metrics=True,
            is_open_set_model=False,
            unk_class_number=None,
            using_id_val_subset=False,
        )

        # Check that ID dataset is not in results when evaluate_on_ind=False
        self.assertNotIn("id_dataset", results)
        self.assertIn("ood_dataset1", results)

    def test_get_overall_open_set_results_multiple_methods(self):
        """Test get_overall_open_set_results with multiple methods."""
        predictions = {
            1: {
                "boxes": np.array([[10, 10, 50, 50]]),
                "logits": np.array([[1.0, 0.5]]),
                "uncertainty_score": np.array([0.9]),
                "confidence_score": np.array([0.8]),
            }
        }

        ind_data_dict = {"valid": predictions}
        ood_data_dict = {
            "ood_dataset1": predictions,
        }

        results = get_overall_open_set_results(
            ind_dataset_name="id_dataset",
            ind_gt_annotations_path=self.id_coco_path,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_datasets_names=["ood_dataset1"],
            ood_annotations_paths={
                "ood_dataset1": self.ood1_coco_path,
            },
            methods_names=["uncertainty_score", "confidence_score"],
            methods_thresholds={
                "uncertainty_score": 0.5,
                "confidence_score": 0.6,
            },
            metric_2007=True,
            evaluate_on_ind=True,
            get_known_classes_metrics=True,
            is_open_set_model=False,
            unk_class_number=None,
            using_id_val_subset=False,
        )

        # Check that both methods are present in results
        self.assertIn("uncertainty_score", results["ood_dataset1"])
        self.assertIn("confidence_score", results["ood_dataset1"])

    def test_get_overall_open_set_results_with_subset(self):
        """Test get_overall_open_set_results with ID validation subset."""
        predictions = {
            1: {
                "boxes": np.array([[10, 10, 50, 50]]),
                "logits": np.array([[1.0, 0.5]]),
                "uncertainty_score": np.array([0.9]),
            }
        }

        ind_data_dict = {"valid": predictions}
        ood_data_dict = {
            "ood_dataset1": predictions,
        }

        results = get_overall_open_set_results(
            ind_dataset_name="id_dataset",
            ind_gt_annotations_path=self.id_coco_path,
            ind_data_dict=ind_data_dict,
            ood_data_dict=ood_data_dict,
            ood_datasets_names=["ood_dataset1"],
            ood_annotations_paths={
                "ood_dataset1": self.ood1_coco_path,
            },
            methods_names=["uncertainty_score"],
            methods_thresholds={"uncertainty_score": 0.5},
            metric_2007=True,
            evaluate_on_ind=True,
            get_known_classes_metrics=True,
            is_open_set_model=False,
            unk_class_number=None,
            using_id_val_subset=[1],
        )

        self.assertIsInstance(results, dict)
        self.assertIn("id_dataset", results)


if __name__ == "__main__":
    main()
