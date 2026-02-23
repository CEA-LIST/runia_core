from unittest import TestCase, main
import numpy as np
from runia_core.evaluation.metrics import get_auroc_results, log_evaluate_postprocessors

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

    def test_evaluate_lared_larem(self):
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


if __name__ == "__main__":
    main()
