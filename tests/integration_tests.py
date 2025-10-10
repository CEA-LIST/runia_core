from unittest import TestCase, main
import torch
import torchvision
import numpy as np
import pandas as pd
import urllib
import tarfile
import os
from runia.evaluation.entropy import get_dl_h_z
from runia.evaluation.metrics import (
    select_and_log_best_larex,
    save_roc_ood_detector,
    get_pred_scores_plots,
)
from runia.feature_extraction import (
    get_latent_representation_mcd_samples,
    MCSamplerModule,
)
from runia.inference import MDLatentSpace, LaRExInference
from runia.evaluation.latent_space import log_evaluate_postprocessors
from runia.dimensionality_reduction import apply_pca_ds_split, apply_pca_transform
from runia.feature_extraction import apply_dropout, Hook
from tests_architecture import Net

#########################################################################
# PARAMETERS
# If you change any of the following parameters, the tests will not pass!!
SEED = 1
TEST_SET_PROPORTION = 0.1
MCD_N_SAMPLES = 3
LATENT_SPACE_DIM = 20
TOL = 1e-7
N_PCA_COMPONENTS = 4
########################################################################

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define datasets for testing
transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
)
mnist_data = torchvision.datasets.MNIST(
    "./mnist-data/",
    train=False,
    download=True,
    transform=transforms,
)
if not os.path.exists("./emnist_data_source/EMNIST/raw"):
    print("getting emnist raw data from confianceai repository ..")
    if not os.path.exists("./emnist_data_source/EMNIST"):
        os.makedirs("./emnist_data_source/EMNIST")

    urllib.request.urlretrieve(
        "https://minio-storage.apps.confianceai-public.irtsysx.fr/ml-models/emnist.tar.gz",
        "./emnist_data_source/EMNIST/emnist.tar.gz",
    )

    file = tarfile.open("./emnist_data_source/EMNIST/emnist.tar.gz")
    file.extractall("./emnist_data_source/EMNIST")
    file.close()
    print("emnist raw data have been downloaded")

emnist_data = torchvision.datasets.EMNIST(
    "./emnist_data_source", split="letters", train=False, download=False, transform=transforms
)
# Subset InD dataset
ind_subset_ds_len = int(len(mnist_data) * TEST_SET_PROPORTION)
ind_test_subset = torch.utils.data.random_split(
    mnist_data,
    [ind_subset_ds_len, len(mnist_data) - ind_subset_ds_len],
    torch.Generator().manual_seed(SEED),
)[0]
# Subset OoD dataset
ood_subset_ds_len = int(len(emnist_data) * TEST_SET_PROPORTION * 0.5)
ood_test_subset = torch.utils.data.random_split(
    emnist_data,
    [ood_subset_ds_len, len(emnist_data) - ood_subset_ds_len],
    torch.Generator().manual_seed(SEED),
)[0]

# DataLoaders
ind_test_loader = torch.utils.data.DataLoader(ind_test_subset, batch_size=1, shuffle=True)
ood_test_loader = torch.utils.data.DataLoader(ood_test_subset, batch_size=1, shuffle=True)
# Define toy model for testing
tests_model = Net(latent_space_dimension=LATENT_SPACE_DIM)
tests_model.to(device)
tests_model.eval()


class Test(TestCase):
    def test_select_best_larex(self):
        np.random.seed(SEED)
        # Here we start from a supposed already calculated entropy
        test_ind = np.float32(0.5 + np.random.randn(ind_subset_ds_len, LATENT_SPACE_DIM))
        train_ind = np.float32(0.5 + np.random.randn(ind_subset_ds_len, LATENT_SPACE_DIM))
        train_labels = train_ind.argmax(axis=1)
        test_labels = test_ind.argmax(axis=1)
        ind_pca_dict = {"train labels": train_labels, "valid labels": test_labels}
        ood_ds_name = "test_ood"
        ood_dict = {
            f"{ood_ds_name} latent_space_means": np.float32(
                -0.5 + np.random.randn(ind_subset_ds_len, LATENT_SPACE_DIM)
            )
        }
        ood_dict[f"{ood_ds_name} labels"] = ood_dict[f"{ood_ds_name} latent_space_means"].argmax(
            axis=1
        )
        pca_components = (2, 6, 10)
        overall_metrics_df = pd.DataFrame(
            columns=[
                "auroc",
                "fpr@95",
                "aupr",
                "fpr",
                "tpr",
            ]
        )
        for n_components in pca_components:
            # Perform PCA dimension reduction
            pca_h_z_ind_train, pca_transformation = apply_pca_ds_split(
                samples=train_ind, nro_components=n_components
            )
            pca_h_z_ind_test = apply_pca_transform(test_ind, pca_transformation)
            ind_pca_dict["train latent_space_means"] = pca_h_z_ind_train
            ind_pca_dict["valid latent_space_means"] = pca_h_z_ind_test
            ood_pca_dict = {}
            ood_pca_dict[f"{ood_ds_name} latent_space_means"] = apply_pca_transform(
                ood_dict[f"{ood_ds_name} latent_space_means"], pca_transformation
            )
            ood_pca_dict[f"{ood_ds_name} labels"] = ood_dict[f"{ood_ds_name} labels"]

            results_eval = log_evaluate_postprocessors(
                ind_dict=ind_pca_dict,
                ood_dict=ood_pca_dict,
                ood_datasets_names=[ood_ds_name],
                experiment_name_extension=f" PCA {n_components}",
                return_density_scores="MD",
                log_step=n_components,
                mlflow_logging=False,
                postprocessors=["MD", "KDE"],
            )
            # Add results to df
            for results in results_eval["results_df"].index.values:
                overall_metrics_df.loc[results] = results_eval["results_df"].loc[results]

        # Check LaRED results
        auroc_lared, aupr_lared, fpr_lared, best_n_comps_lared = select_and_log_best_larex(
            overall_metrics_df,
            pca_components,
            postprocessor_name="KDE",
            log_mlflow=False,
            multiple_ood_datasets_flag=False,
        )
        self.assertAlmostEqual(0.8123340606689453, auroc_lared, delta=TOL)
        self.assertAlmostEqual(0.7958822250366211, aupr_lared, delta=TOL)
        self.assertAlmostEqual(0.5989999771118164, fpr_lared, delta=TOL)

        # Check LaREM results
        auroc_larem, aupr_larem, fpr_larem, best_n_comps_larem = select_and_log_best_larex(
            overall_metrics_df,
            pca_components,
            postprocessor_name="MD",
            log_mlflow=False,
            multiple_ood_datasets_flag=False,
        )
        self.assertAlmostEqual(0.8106600642204285, auroc_larem, delta=TOL)
        self.assertAlmostEqual(0.7947195768356323, aupr_larem, delta=TOL)
        self.assertAlmostEqual(0.6159999966621399, fpr_larem, delta=TOL)

        roc_curve_test = save_roc_ood_detector(overall_metrics_df, "Test title")
        self.assertEqual(1.0, roc_curve_test.axes[0].dataLim.max[0])
        self.assertEqual(1.0, roc_curve_test.axes[0].dataLim.max[1])
        self.assertEqual(0.0, roc_curve_test.axes[0].dataLim.min[0])
        self.assertEqual(0.0, roc_curve_test.axes[0].dataLim.min[1])
        self.assertAlmostEqual(
            0.0010000000474974513, roc_curve_test.axes[0].dataLim.minposx, delta=TOL
        )
        self.assertAlmostEqual(
            0.0010000000474974513, roc_curve_test.axes[0].dataLim.minposy, delta=TOL
        )

        experiment_dict = {
            "InD": results_eval["InD"],
            "x_axis": "MD score",
            "plot_name": "MD test plot",
            "test_ood": results_eval["OoD"][ood_ds_name],
        }
        pred_scores_plot_test = get_pred_scores_plots(
            experiment=experiment_dict,
            ood_datasets_list=[ood_ds_name],
            title="Test title",
            ind_dataset_name="Test InD",
        )
        self.assertAlmostEqual(594.5, pred_scores_plot_test.ax.bbox.max[0], delta=TOL)
        self.assertAlmostEqual(462.66666666666663, pred_scores_plot_test.ax.bbox.max[1], delta=TOL)
        self.assertAlmostEqual(70.65277777777779, pred_scores_plot_test.ax.bbox.min[0], delta=TOL)
        self.assertAlmostEqual(58.277777777777764, pred_scores_plot_test.ax.bbox.min[1], delta=TOL)

    def test_extract_entropy_larex(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model.apply(apply_dropout)  # enable dropout
        ind_mcd_latent_samples = get_latent_representation_mcd_samples(
            tests_model, ind_test_loader, MCD_N_SAMPLES, hooked_layer, "Conv"
        )
        ood_mcd_latent_samples = get_latent_representation_mcd_samples(
            tests_model, ood_test_loader, MCD_N_SAMPLES, hooked_layer, "Conv"
        )
        ind_test_entropy = get_dl_h_z(ind_mcd_latent_samples, MCD_N_SAMPLES, parallel_run=True)[1]
        ood_test_entropy = get_dl_h_z(ood_mcd_latent_samples, MCD_N_SAMPLES, parallel_run=True)[1]
        self.assertEqual((ind_subset_ds_len, LATENT_SPACE_DIM), ind_test_entropy.shape)
        self.assertEqual((ood_subset_ds_len, LATENT_SPACE_DIM), ood_test_entropy.shape)
        self.assertAlmostEqual(
            0.0,
            (
                ind_test_entropy[0]
                - np.array(
                    [
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                    ]
                )
            ).sum(),
            delta=TOL,
        )
        self.assertAlmostEqual(
            0.0,
            (
                ood_test_entropy[0]
                - np.array(
                    [
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                    ]
                )
            ).sum(),
            delta=TOL,
        )

    def test_larex_inference(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        ind_test_features = np.random.rand(ind_subset_ds_len, LATENT_SPACE_DIM)
        pca_ind_train, pca_transformation = apply_pca_ds_split(
            samples=ind_test_features, nro_components=N_PCA_COMPONENTS
        )
        larem_processor = MDLatentSpace()
        larem_processor.setup(pca_ind_train)
        larem_inference = LaRExInference(
            model=tests_model,
            postprocessor=larem_processor,
            mcd_sampler=MCSamplerModule,
            pca_transform=pca_transformation,
            mcd_samples_nro=MCD_N_SAMPLES,
            drop_block_prob=0.5,
            drop_block_size=8,
            layer_type="Conv",
        )
        ood_iterator = iter(ood_test_loader)
        ood_test_image = next(ood_iterator)[0]
        ood_prediction, ood_img_score = larem_inference.get_score(
            ood_test_image, layer_hook=hooked_layer
        )
        self.assertAlmostEqual(-6103.11052918, ood_img_score, delta=TOL)
        self.assertAlmostEqual(
            0.0,
            (
                ood_prediction[0].cpu().numpy()
                - np.array(
                    [
                        -2.275621,
                        -2.007265,
                        -2.4919932,
                        -2.2528067,
                        -2.1876812,
                        -2.345544,
                        -2.314673,
                        -2.4217446,
                        -2.4728994,
                        -2.3519247,
                    ]
                )
            ).sum(),
            delta=1e-6,
        )


if __name__ == "__main__":
    main()
