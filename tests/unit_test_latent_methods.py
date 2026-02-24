from unittest import TestCase, main
import numpy as np
import torch
from omegaconf import OmegaConf

from runia_core.evaluation import (
    calculate_all_baselines,
    remove_latent_features,
    log_evaluate_larex,
)
from runia_core.inference import get_baselines_thresholds

#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 1
TEST_SET_PROPORTION = 0.02
TEST_SAMPLES = 200
MCD_N_SAMPLES = 3
LATENT_SPACE_DIM = 20
TOL = 1e-6
LAYER_TYPE = "Conv"
REDUCTION_METHOD = "fullmean"
BASELINES_NAMES = ["msp"]
LS_POSTPROCESSORS = ["KNN", "MD", "GMM"]
VISUALIZE_LS_POSTPROCESSOR = "MD"
########################################################################


class TestLatentMethods(TestCase):

    def setUp(self) -> None:
        # Deterministic seeds for reproducibility
        np.random.seed(seed=SEED)
        torch.manual_seed(seed=SEED)

    def test_latent_methods_eval(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        cfg = OmegaConf.create(
            {
                "ood_datasets": ["test_ood"],
                "n_pca_components": [1, 2, 4],
                "log_dir": "logs",
                "k_neighbors": 10,
                "ind_dataset": "test_id",
            }
        )
        fc_params = {
            "weight": np.random.rand(LATENT_SPACE_DIM, LATENT_SPACE_DIM).astype(np.float32),
            "bias": np.random.rand(LATENT_SPACE_DIM).astype(np.float32),
        }

        # Here we start from a supposed already calculated logits and features
        # ID
        train_ind_feats = np.float32(0.5 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        train_ind_logits = np.float32(0.5 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        train_ind_latent_feats = np.float32(0.4 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        val_ind_feats = np.float32(0.5 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        val_ind_logits = np.float32(0.5 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        val_ind_latent_feats = np.float32(0.4 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        id_data = {
            "train features": train_ind_feats,
            "train logits": train_ind_logits,
            "valid features": val_ind_feats,
            "valid logits": val_ind_logits,
            "train latent_space_means": train_ind_latent_feats,
            "valid latent_space_means": val_ind_latent_feats,
        }
        # OOD
        test_ood_feats = np.float32(-0.5 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        test_ood_logits = np.float32(-0.5 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        test_ood_latent_feats = np.float32(-0.4 + np.random.randn(TEST_SAMPLES, LATENT_SPACE_DIM))
        ood_ds_name = "test_ood"
        ood_data = {
            f"{ood_ds_name} features": test_ood_feats,
            f"{ood_ds_name} logits": test_ood_logits,
            f"{ood_ds_name} latent_space_means": test_ood_latent_feats,
        }
        id_data, ood_data, ood_baselines_scores = calculate_all_baselines(
            baselines_names=BASELINES_NAMES,
            ind_data_dict=id_data,
            ood_data_dict=ood_data,
            fc_params=fc_params,
            cfg=cfg,
            num_classes=LATENT_SPACE_DIM,
        )

        id_data, ood_data = remove_latent_features(
            id_data=id_data, ood_data=ood_data, ood_names=cfg.ood_datasets
        )

        df, best_postprocessors_dict, postprocessor_thresholds, ood_data = log_evaluate_larex(
            cfg=cfg,
            baselines_names=BASELINES_NAMES,
            ind_data_dict=id_data,
            ood_data_dict=ood_data,
            ood_baselines_scores=ood_baselines_scores,
            mlflow_run_name="my_run",
            mlflow_logging=False,
            visualize_score=VISUALIZE_LS_POSTPROCESSOR,
            postprocessors=LS_POSTPROCESSORS,
        )
        self.assertIn("KNN", best_postprocessors_dict)
        self.assertAlmostEqual(
            best_postprocessors_dict["KNN"]["auroc"], 0.9881750345230103, delta=TOL
        )
        self.assertIn("MD", best_postprocessors_dict)
        self.assertAlmostEqual(
            best_postprocessors_dict["MD"]["auroc"], 0.837399959564209, delta=TOL
        )
        self.assertIn("GMM", best_postprocessors_dict)
        self.assertAlmostEqual(
            best_postprocessors_dict["GMM"]["auroc"], 0.801800012588501, delta=TOL
        )


if __name__ == "__main__":
    main()
