"""
tests/unit_test_baselines.py

Run with:

    python -m unittest -q tests/unit_test_baselines.py

This file focuses on small, fast unit tests that don't require external
resources. Tests are independent and use deterministic seeds where
relevant.
"""

from unittest import TestCase, main
from typing import Dict
from omegaconf import OmegaConf
import logging

import numpy as np
import torch
import torchvision

from runia_core.inference import (
    get_mcd_pred_uncertainty_score,
    get_predictive_uncertainty_score,
    ash_s_linear_layer,
    generalized_entropy,
    mahalanobis_preprocess,
    mahalanobis_postprocess,
    gmm_fit,
    MDLatentSpace,
    KDELatentSpace,
)
from runia_core.evaluation.baselines import (
    get_labels_from_logits,
    remove_latent_features,
    calculate_all_baselines,
)
from runia_core.inference.abstract_classes import get_baselines_thresholds
from tests_architecture import Net


# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global tolerance for floating point comparisons in tests
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
BASELINES_NAMES = [
    "vim",
    "mdist",
    "msp",
    "knn",
    "energy",
    "ash",
    "dice",
    "react",
    "gen",
    "dice_react",
    "ddu",
    "raw",
]
########################################################################


class TestBaselinesFromPrecalculated(TestCase):
    """Unittests for small utility functions in baselines.from_precalculated."""

    def setUp(self) -> None:
        # Deterministic seeds for reproducibility
        np.random.seed(seed=SEED)
        torch.manual_seed(seed=SEED)

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
        self.assertEqual(len(id_res2["train labels"]), 0)
        self.assertEqual(len(id_res2["valid labels"]), 0)
        self.assertEqual(len(ood_res2["ood1 labels"]), 0)
        self.assertEqual(len(id_res2["valid labels"]), 0)
        self.assertEqual(len(ood_res2["ood1 labels"]), 0)

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

    def test_all_baselines_postp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        cfg = OmegaConf.create(
            {
                "ood_datasets": ["test_ood"],
                "ash_percentile": 90,
                "react_percentile": 90,
                "dice_percentile": 90,
                "gen_gamma": 0.1,
                "k_neighbors": 10,
            }
        )
        fc_params = {
            "weight": np.random.rand(LATENT_SPACE_DIM, LATENT_SPACE_DIM).astype(np.float32),
            "bias": np.random.rand(LATENT_SPACE_DIM).astype(np.float32),
        }

        # Here we start from a supposed already calculated logits and features
        # ID
        train_ind_feats = np.float32(np.random.random((TEST_SAMPLES, LATENT_SPACE_DIM)))
        train_ind_logits = np.float32(np.random.random((TEST_SAMPLES, LATENT_SPACE_DIM)))
        val_ind_feats = np.float32(np.random.random((TEST_SAMPLES, LATENT_SPACE_DIM)))
        val_ind_logits = np.float32(np.random.random((TEST_SAMPLES, LATENT_SPACE_DIM)))
        id_data = {
            "train features": train_ind_feats,
            "train logits": train_ind_logits,
            "valid features": val_ind_feats,
            "valid logits": val_ind_logits,
        }
        # OOD
        test_ood_feats = np.float32(np.random.random((TEST_SAMPLES, LATENT_SPACE_DIM)))
        test_ood_logits = np.float32(np.random.random((TEST_SAMPLES, LATENT_SPACE_DIM)))
        ood_ds_name = "test_ood"
        ood_data = {
            f"{ood_ds_name} features": test_ood_feats,
            f"{ood_ds_name} logits": test_ood_logits,
        }
        id_data, ood_data, ood_baselines_scores = calculate_all_baselines(
            baselines_names=BASELINES_NAMES,
            ind_data_dict=id_data,
            ood_data_dict=ood_data,
            fc_params=fc_params,
            cfg=cfg,
            num_classes=LATENT_SPACE_DIM,
        )
        self.assertAlmostEqual(ood_baselines_scores["test_ood msp"].mean(), 0.07561022, delta=TOL)
        self.assertAlmostEqual(ood_baselines_scores["test_ood knn"].mean(), -0.28827268, delta=TOL)
        self.assertAlmostEqual(ood_baselines_scores["test_ood energy"].mean(), 3.5367718, delta=TOL)
        self.assertAlmostEqual(ood_baselines_scores["test_ood ash"].mean(), 437.55548, delta=TOL)
        self.assertAlmostEqual(ood_baselines_scores["test_ood gen"].mean(), -14.69404, delta=TOL)
        self.assertAlmostEqual(ood_baselines_scores["test_ood react"].mean(), 8.930155, delta=TOL)
        self.assertAlmostEqual(ood_baselines_scores["test_ood dice"].mean(), 4.779826, delta=TOL)
        self.assertAlmostEqual(
            ood_baselines_scores["test_ood dice_react"].mean(), 4.7608514, delta=TOL
        )
        self.assertAlmostEqual(
            ood_baselines_scores["test_ood mdist"].mean(), -20.75197064883483, delta=TOL
        )
        self.assertAlmostEqual(ood_baselines_scores["test_ood ddu"].mean(), -863839.4375, delta=TOL)


class TestBaselinesFromModel(TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define dataset for testing
        self.mnist_data = torchvision.datasets.MNIST(
            "./tests/mnist-data/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        self.subset_ds_len = int(len(self.mnist_data) * TEST_SET_PROPORTION)
        test_subset = torch.utils.data.random_split(
            self.mnist_data,
            [self.subset_ds_len, len(self.mnist_data) - self.subset_ds_len],
            torch.Generator().manual_seed(SEED),
        )[0]
        # DataLoader
        self.test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True)
        # Define toy model for testing
        self.tests_model = Net(latent_space_dimension=LATENT_SPACE_DIM)
        self.tests_model.to(device)
        self.tests_model.eval()

    def test_mcd_pred_uncertainty_score(self):
        torch.manual_seed(SEED)
        samples, pred_h, mi = get_mcd_pred_uncertainty_score(
            dnn_model=self.tests_model,
            input_dataloader=self.test_loader,
            mcd_nro_samples=MCD_N_SAMPLES,
        )
        self.assertEqual(
            samples.shape,
            torch.Size([self.subset_ds_len, MCD_N_SAMPLES, len(self.mnist_data.classes)]),
        )
        self.assertTrue(
            (
                samples[0, 0].cpu()
                - torch.Tensor(
                    [
                        0.1086513326,
                        0.1488786936,
                        0.0802124515,
                        0.0924480408,
                        0.1231049076,
                        0.0829655901,
                        0.0918756202,
                        0.0897051468,
                        0.0857857615,
                        0.0963724107,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(pred_h.shape, torch.Size([self.subset_ds_len]))
        self.assertTrue(
            (
                pred_h[:LATENT_SPACE_DIM].cpu().numpy()
                - np.array(
                    [
                        2.2834153,
                        2.294122,
                        2.2849278,
                        2.290954,
                        2.2931917,
                        2.285399,
                        2.2858777,
                        2.2884946,
                        2.2849176,
                        2.2858899,
                        2.2869554,
                        2.2908378,
                        2.289379,
                        2.2784553,
                        2.287322,
                        2.283883,
                        2.285529,
                        2.2853007,
                        2.2854085,
                        2.2908926,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(mi.shape, torch.Size([self.subset_ds_len]))
        self.assertTrue(
            (
                mi[:LATENT_SPACE_DIM].cpu().numpy()
                - np.array(
                    [
                        0.0000000e00,
                        -2.3841858e-07,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        2.3841858e-07,
                        -2.3841858e-07,
                        0.0000000e00,
                        0.0000000e00,
                        -2.3841858e-07,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        2.3841858e-07,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_get_pred_uncertainty_score(self):
        torch.manual_seed(SEED)
        input_samples = torch.rand(self.subset_ds_len * MCD_N_SAMPLES, len(self.mnist_data.classes))
        pred_h, mi = get_predictive_uncertainty_score(input_samples, MCD_N_SAMPLES)
        self.assertEqual(pred_h.shape, torch.Size([self.subset_ds_len]))
        self.assertTrue(
            (
                pred_h[:LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        2.2954239845,
                        2.2936100960,
                        2.2913274765,
                        2.2875323296,
                        2.2929368019,
                        2.2918679714,
                        2.2960093021,
                        2.2968566418,
                        2.2959966660,
                        2.2814788818,
                        2.2842481136,
                        2.2999031544,
                        2.2889721394,
                        2.2876074314,
                        2.2898373604,
                        2.2722413540,
                        2.2952365875,
                        2.2890031338,
                        2.2956945896,
                        2.2963838577,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(mi.shape, torch.Size([self.subset_ds_len]))
        self.assertTrue(
            (
                mi[:LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        0.0174179077,
                        0.0362265110,
                        0.0232129097,
                        0.0191953182,
                        0.0291962624,
                        0.0250487328,
                        0.0286319256,
                        0.0462374687,
                        0.0337688923,
                        0.0085363388,
                        0.0118212700,
                        0.0289292336,
                        0.0174930096,
                        0.0185499191,
                        0.0275359154,
                        0.0115809441,
                        0.0252931118,
                        0.0210585594,
                        0.0264883041,
                        0.0213701725,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_larem_postprocessor(self):
        np.random.seed(SEED)
        test_features = np.random.rand(self.subset_ds_len, LATENT_SPACE_DIM)
        larem_processor = MDLatentSpace()
        larem_processor.setup(test_features)
        self.assertEqual(larem_processor.precision.shape, (LATENT_SPACE_DIM, LATENT_SPACE_DIM))
        self.assertTrue(
            np.allclose(
                larem_processor.precision[0],
                np.array(
                    [
                        12.44132297,
                        -1.83394915,
                        -0.79040659,
                        -1.8327401,
                        -1.02820474,
                        -0.8484738,
                        0.68634041,
                        -0.95924792,
                        -1.37500991,
                        1.07289637,
                        1.44918455,
                        0.26444471,
                        0.12702008,
                        -1.58528515,
                        1.01321551,
                        -0.73969692,
                        0.7707137,
                        0.89374538,
                        -0.18806438,
                        0.44323749,
                    ]
                ),
                atol=TOL,
            )
        )
        postprocessed = larem_processor.postprocess(test_features)
        self.assertEqual(postprocessed.shape, (self.subset_ds_len,))
        self.assertTrue(
            np.allclose(
                postprocessed[:LATENT_SPACE_DIM],
                np.array(
                    [
                        -18.91857476,
                        -24.91446197,
                        -18.25391816,
                        -23.0257814,
                        -21.92124241,
                        -26.56679895,
                        -21.8830617,
                        -28.06454144,
                        -18.53972256,
                        -27.31282806,
                        -22.5506664,
                        -18.53799936,
                        -21.37307662,
                        -12.66427075,
                        -16.15089713,
                        -16.85529228,
                        -16.74306546,
                        -19.8072294,
                        -25.18734225,
                        -20.55324483,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_lared_postprocessor(self):
        np.random.seed(SEED)
        test_features = np.random.rand(self.subset_ds_len, LATENT_SPACE_DIM)
        lared_processor = KDELatentSpace()
        lared_processor.setup(test_features)
        postprocessed = lared_processor.postprocess(test_features)
        self.assertEqual(postprocessed.shape, (self.subset_ds_len,))
        self.assertTrue(
            np.allclose(
                postprocessed[:LATENT_SPACE_DIM],
                np.array(
                    [
                        -19.91348719,
                        -20.19627582,
                        -19.92684543,
                        -19.94015669,
                        -19.90663336,
                        -20.24527737,
                        -20.11600327,
                        -20.19620542,
                        -19.82373531,
                        -20.15918133,
                        -20.12760057,
                        -19.85977362,
                        -19.98236152,
                        -19.65662305,
                        -19.85863884,
                        -19.84679279,
                        -19.76649333,
                        -19.95602029,
                        -20.07514339,
                        -19.79483468,
                    ]
                ),
                atol=TOL,
            )
        )


if __name__ == "__main__":
    main()
