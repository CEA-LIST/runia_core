from unittest import TestCase, main
import numpy as np
from runia_core.evaluation.metrics import (
    get_auroc_results,
    log_evaluate_postprocessors,
    subset_boxes,
)

#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 1
TEST_SET_SIZE = 1000
LATENT_SPACE_DIM = 20
N_CATEGORIES = 5
TOL = 1e-7
########################################################################


class Test(TestCase):
    def test_hz_detector_results(self):
        np.random.seed(SEED)
        test_ind = 0.5 + np.random.randn(TEST_SET_SIZE)
        test_ood = -0.5 + np.random.randn(TEST_SET_SIZE)
        test_name = "test"
        results = get_auroc_results(test_name, test_ind, test_ood, False)
        self.assertAlmostEqual(0.7329999804496765, results["fpr@95"].values[0], delta=TOL)
        self.assertAlmostEqual(0.7484172582626343, results["aupr"].values[0], delta=TOL)
        self.assertAlmostEqual(0.7622030377388, results["auroc"].values[0], delta=TOL)

    def test_evaluate_postprocessors(self):
        np.random.seed(SEED)
        valid_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        train_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        valid_labels = np.random.randint(N_CATEGORIES, size=TEST_SET_SIZE)
        train_labels = np.random.randint(N_CATEGORIES, size=TEST_SET_SIZE)
        ind_dict = {
            "train latent_space_means": train_ind,
            "valid latent_space_means": valid_ind,
            "train labels": train_labels,
            "valid labels": valid_labels,
        }
        ood_ds_name = "test"
        ood_labels = np.random.randint(N_CATEGORIES, size=TEST_SET_SIZE)
        ood_dict = {
            f"{ood_ds_name} latent_space_means": -0.5
            + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM),
            f"{ood_ds_name} labels": ood_labels,
        }
        results = log_evaluate_postprocessors(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            postprocessors=["KDE", "MD"],
            ood_datasets_names=[ood_ds_name],
        )
        self.assertAlmostEqual(
            0.9449479579925537,
            results["results_df"].loc[f"{ood_ds_name} KDE"]["auroc"],
            delta=TOL,
        )
        self.assertAlmostEqual(
            0.9474190473556519, results["results_df"].loc[f"{ood_ds_name} KDE"]["aupr"], delta=TOL
        )
        self.assertAlmostEqual(
            0.2770000100135803,
            results["results_df"].loc[f"{ood_ds_name} KDE"]["fpr@95"],
            delta=TOL,
        )
        self.assertAlmostEqual(
            0.9514310359954834,
            results["results_df"].loc[f"{ood_ds_name} MD"]["auroc"],
            delta=TOL,
        )
        self.assertAlmostEqual(
            0.9535703659057617, results["results_df"].loc[f"{ood_ds_name} MD"]["aupr"], delta=TOL
        )
        self.assertAlmostEqual(
            0.2540000081062317,
            results["results_df"].loc[f"{ood_ds_name} MD"]["fpr@95"],
            delta=TOL,
        )


class TestSubsetBoxes(TestCase):
    """Test cases for the subset_boxes function"""

    def setUp(self):
        """Set up common test data"""
        np.random.seed(SEED)
        self.latent_dim = 10
        self.ood_names = ["ood_dataset_1", "ood_dataset_2"]

    def test_subset_boxes_no_subsetting_needed(self):
        """Test when data size is below limits - no subsetting should occur"""
        # Create small data below limits
        ind_dict = {
            "train latent_space_means": np.random.randn(50, self.latent_dim),
            "valid latent_space_means": np.random.randn(30, self.latent_dim),
            "train labels": np.random.randint(5, size=50),
            "valid labels": np.random.randint(5, size=30),
        }
        ood_dict = {
            f"{ood} latent_space_means": np.random.randn(40, self.latent_dim)
            for ood in self.ood_names
        }

        ind_dict_orig_size = {k: v.shape[0] for k, v in ind_dict.items() if "means" in k}
        ood_dict_orig_size = {k: v.shape[0] for k, v in ood_dict.items() if "means" in k}

        # Call with high limits
        result_ind, result_ood = subset_boxes(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            ind_train_limit=100,
            ood_limit=100,
            random_seed=SEED,
            ood_names=self.ood_names,
        )

        # Verify sizes remain unchanged
        self.assertEqual(
            result_ind["train latent_space_means"].shape[0],
            ind_dict_orig_size["train latent_space_means"],
        )
        self.assertEqual(
            result_ind["valid latent_space_means"].shape[0],
            ind_dict_orig_size["valid latent_space_means"],
        )
        for ood_name in self.ood_names:
            self.assertEqual(
                result_ood[f"{ood_name} latent_space_means"].shape[0],
                ood_dict_orig_size[f"{ood_name} latent_space_means"],
            )

    def test_subset_boxes_train_subsetting(self):
        """Test subsetting of training data"""
        train_size = 500
        valid_size = 100
        ind_train_limit = 200

        ind_dict = {
            "train latent_space_means": np.random.randn(train_size, self.latent_dim),
            "valid latent_space_means": np.random.randn(valid_size, self.latent_dim),
            "train labels": np.random.randint(5, size=train_size),
            "valid labels": np.random.randint(5, size=valid_size),
        }
        ood_dict = {
            f"{ood} latent_space_means": np.random.randn(150, self.latent_dim)
            for ood in self.ood_names
        }

        result_ind, result_ood = subset_boxes(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            ind_train_limit=ind_train_limit,
            ood_limit=500,
            random_seed=SEED,
            ood_names=self.ood_names,
        )

        # Verify train set was subset correctly
        self.assertEqual(result_ind["train latent_space_means"].shape[0], ind_train_limit)
        # Valid set should remain unchanged
        self.assertEqual(result_ind["valid latent_space_means"].shape[0], valid_size)

    def test_subset_boxes_ood_subsetting(self):
        """Test subsetting of OoD data"""
        ood_size = 400
        ood_limit = 150

        ind_dict = {
            "train latent_space_means": np.random.randn(100, self.latent_dim),
            "valid latent_space_means": np.random.randn(100, self.latent_dim),
            "train labels": np.random.randint(5, size=100),
            "valid labels": np.random.randint(5, size=100),
        }
        ood_dict = {
            f"{ood} latent_space_means": np.random.randn(ood_size, self.latent_dim)
            for ood in self.ood_names
        }

        result_ind, result_ood = subset_boxes(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            ind_train_limit=500,
            ood_limit=ood_limit,
            random_seed=SEED,
            ood_names=self.ood_names,
        )

        # Verify OoD sets were subset correctly
        for ood_name in self.ood_names:
            self.assertEqual(result_ood[f"{ood_name} latent_space_means"].shape[0], ood_limit)

    def test_subset_boxes_with_logits_and_features(self):
        """Test subsetting with additional data (logits and features)"""
        train_size = 300
        valid_size = 100
        ood_size = 200
        ind_train_limit = 150
        ood_limit = 100
        n_classes = 5

        ind_dict = {
            "train latent_space_means": np.random.randn(train_size, self.latent_dim),
            "valid latent_space_means": np.random.randn(valid_size, self.latent_dim),
            "train logits": np.random.randn(train_size, n_classes),
            "valid logits": np.random.randn(valid_size, n_classes),
            "train features": np.random.randn(train_size, 256),
            "valid features": np.random.randn(valid_size, 256),
            "train labels": np.random.randint(5, size=train_size),
            "valid labels": np.random.randint(5, size=valid_size),
        }
        ood_dict = {
            f"{ood} latent_space_means": np.random.randn(ood_size, self.latent_dim)
            for ood in self.ood_names
        }
        # Add logits and features to OoD dict
        for ood_name in self.ood_names:
            ood_dict[f"{ood_name} logits"] = np.random.randn(ood_size, n_classes)
            ood_dict[f"{ood_name} features"] = np.random.randn(ood_size, 256)

        result_ind, result_ood = subset_boxes(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            ind_train_limit=ind_train_limit,
            ood_limit=ood_limit,
            random_seed=SEED,
            ood_names=self.ood_names,
        )

        # Verify train logits and features were subset
        self.assertEqual(result_ind["train logits"].shape[0], ind_train_limit)
        self.assertEqual(result_ind["train features"].shape[0], ind_train_limit)

        # Verify OoD logits and features were subset
        for ood_name in self.ood_names:
            self.assertEqual(result_ood[f"{ood_name} logits"].shape[0], ood_limit)
            self.assertEqual(result_ood[f"{ood_name} features"].shape[0], ood_limit)

    def test_subset_boxes_with_prediction_tracking(self):
        """Test subsetting with non-empty predictions tracking"""
        train_size = 200
        valid_size = 150
        ood_size = 100

        ind_dict = {
            "train latent_space_means": np.random.randn(train_size, self.latent_dim),
            "valid latent_space_means": np.random.randn(valid_size, self.latent_dim),
            "train labels": np.random.randint(5, size=train_size),
            "valid labels": np.random.randint(5, size=valid_size),
        }
        ood_dict = {
            f"{ood} latent_space_means": np.random.randn(ood_size, self.latent_dim)
            for ood in self.ood_names
        }

        # Create prediction tracking dicts
        non_empty_predictions_id = {"valid": [f"img_{i%20}" for i in range(valid_size)]}
        non_empty_predictions_ood = {
            ood: [f"ood_img_{i%10}" for i in range(ood_size)] for ood in self.ood_names
        }

        result_ind, result_ood, result_pred_id, result_pred_ood = subset_boxes(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            ind_train_limit=500,
            ood_limit=50,
            random_seed=SEED,
            ood_names=self.ood_names,
            non_empty_predictions_id=non_empty_predictions_id,
            non_empty_predictions_ood=non_empty_predictions_ood,
        )

        # Verify predictions were subset along with data
        self.assertEqual(
            result_pred_id["valid"].__len__(),
            result_ind["valid latent_space_means"].shape[0],
        )
        for ood_name in self.ood_names:
            self.assertEqual(
                result_pred_ood[ood_name].__len__(),
                result_ood[f"{ood_name} latent_space_means"].shape[0],
            )

    def test_subset_boxes_reproducibility(self):
        """Test that same seed produces consistent results"""
        ind_dict_1 = {
            "train latent_space_means": np.random.randn(300, self.latent_dim),
            "valid latent_space_means": np.random.randn(100, self.latent_dim),
            "train labels": np.random.randint(5, size=300),
            "valid labels": np.random.randint(5, size=100),
        }
        ood_dict_1 = {
            f"{ood} latent_space_means": np.random.randn(200, self.latent_dim)
            for ood in self.ood_names
        }

        # Create copies for second run
        ind_dict_2 = {k: v.copy() for k, v in ind_dict_1.items()}
        ood_dict_2 = {k: v.copy() for k, v in ood_dict_1.items()}

        result_ind_1, result_ood_1 = subset_boxes(
            ind_dict=ind_dict_1,
            ood_dict=ood_dict_1,
            ind_train_limit=150,
            ood_limit=100,
            random_seed=42,
            ood_names=self.ood_names,
        )

        result_ind_2, result_ood_2 = subset_boxes(
            ind_dict=ind_dict_2,
            ood_dict=ood_dict_2,
            ind_train_limit=150,
            ood_limit=100,
            random_seed=42,
            ood_names=self.ood_names,
        )

        # Verify reproducibility with same seed
        np.testing.assert_array_equal(
            result_ind_1["train latent_space_means"],
            result_ind_2["train latent_space_means"],
        )
        for ood_name in self.ood_names:
            np.testing.assert_array_equal(
                result_ood_1[f"{ood_name} latent_space_means"],
                result_ood_2[f"{ood_name} latent_space_means"],
            )

    def test_subset_boxes_different_seeds(self):
        """Test that different seeds produce different results"""
        ind_dict_1 = {
            "train latent_space_means": np.random.randn(300, self.latent_dim),
            "valid latent_space_means": np.random.randn(100, self.latent_dim),
            "train labels": np.random.randint(5, size=300),
            "valid labels": np.random.randint(5, size=100),
        }
        ood_dict_1 = {
            f"{ood} latent_space_means": np.random.randn(200, self.latent_dim)
            for ood in self.ood_names
        }

        # Create copies for second run
        ind_dict_2 = {k: v.copy() for k, v in ind_dict_1.items()}
        ood_dict_2 = {k: v.copy() for k, v in ood_dict_1.items()}

        result_ind_1, result_ood_1 = subset_boxes(
            ind_dict=ind_dict_1,
            ood_dict=ood_dict_1,
            ind_train_limit=150,
            ood_limit=100,
            random_seed=42,
            ood_names=self.ood_names,
        )

        result_ind_2, result_ood_2 = subset_boxes(
            ind_dict=ind_dict_2,
            ood_dict=ood_dict_2,
            ind_train_limit=150,
            ood_limit=100,
            random_seed=99,
            ood_names=self.ood_names,
        )

        # Verify different seeds produce different subsets (with high probability)
        self.assertFalse(
            np.array_equal(
                result_ind_1["train latent_space_means"],
                result_ind_2["train latent_space_means"],
            )
        )

    def test_subset_boxes_returns_correct_types(self):
        """Test that function returns correct types"""
        ind_dict = {
            "train latent_space_means": np.random.randn(100, self.latent_dim),
            "valid latent_space_means": np.random.randn(100, self.latent_dim),
            "train labels": np.random.randint(5, size=100),
            "valid labels": np.random.randint(5, size=100),
        }
        ood_dict = {
            f"{ood} latent_space_means": np.random.randn(100, self.latent_dim)
            for ood in self.ood_names
        }

        result = subset_boxes(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            ind_train_limit=500,
            ood_limit=500,
            random_seed=SEED,
            ood_names=self.ood_names,
        )

        # Should return tuple of 2 dicts when prediction dicts are None
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], dict)

    def test_subset_boxes_returns_four_items_with_predictions(self):
        """Test that function returns 4 items when prediction dicts are provided"""
        ind_dict = {
            "train latent_space_means": np.random.randn(100, self.latent_dim),
            "valid latent_space_means": np.random.randn(100, self.latent_dim),
            "train labels": np.random.randint(5, size=100),
            "valid labels": np.random.randint(5, size=100),
        }
        ood_dict = {
            f"{ood} latent_space_means": np.random.randn(100, self.latent_dim)
            for ood in self.ood_names
        }
        non_empty_predictions_id = {"valid": [f"img_{i}" for i in range(100)]}
        non_empty_predictions_ood = {
            ood: [f"ood_img_{i}" for i in range(100)] for ood in self.ood_names
        }

        result = subset_boxes(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            ind_train_limit=500,
            ood_limit=500,
            random_seed=SEED,
            ood_names=self.ood_names,
            non_empty_predictions_id=non_empty_predictions_id,
            non_empty_predictions_ood=non_empty_predictions_ood,
        )

        # Should return tuple of 4 items when prediction dicts are provided
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[2], dict)
        self.assertIsInstance(result[3], dict)

    def test_subset_boxes_preserves_data_dimension(self):
        """Test that subsetting preserves data dimensions"""
        ind_dict = {
            "train latent_space_means": np.random.randn(300, self.latent_dim),
            "valid latent_space_means": np.random.randn(100, self.latent_dim),
            "train labels": np.random.randint(5, size=300),
            "valid labels": np.random.randint(5, size=100),
        }
        ood_dict = {
            f"{ood} latent_space_means": np.random.randn(200, self.latent_dim)
            for ood in self.ood_names
        }

        result_ind, result_ood = subset_boxes(
            ind_dict=ind_dict,
            ood_dict=ood_dict,
            ind_train_limit=150,
            ood_limit=100,
            random_seed=SEED,
            ood_names=self.ood_names,
        )

        # Verify dimensions are preserved
        self.assertEqual(result_ind["train latent_space_means"].shape[1], self.latent_dim)
        self.assertEqual(result_ind["valid latent_space_means"].shape[1], self.latent_dim)
        for ood_name in self.ood_names:
            self.assertEqual(result_ood[f"{ood_name} latent_space_means"].shape[1], self.latent_dim)


if __name__ == "__main__":
    main()
