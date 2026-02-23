from unittest import TestCase, main
import numpy as np

from runia_core.evaluation import single_image_entropy_calculation, get_dl_h_z
from runia_core.feature_extraction import (
    get_latent_representation_mcd_samples,
    MCDSamplesExtractor,
    apply_dropout,
    Hook,
)
from .tests_architecture import Net
import torch
import torchvision

#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 1
TEST_SET_PROPORTION = 0.02
MCD_N_SAMPLES = 3
LATENT_SPACE_DIM = 20
TOL = 1e-6
LAYER_TYPE = "Conv"
REDUCTION_METHOD = "fullmean"
########################################################################


class Test(TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.tests_model.to(self.device)
        self.tests_model.eval()

    ############################################
    # Uncertainty estimation Tests
    ############################################
    def test_hook_module(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(self.tests_model.conv2_drop)
        self.tests_model(next(iter(self.test_loader))[0].to(self.device))
        self.assertEqual(hooked_layer.output.shape, torch.Size([1, LATENT_SPACE_DIM, 8, 8]))
        self.assertTrue(
            (
                hooked_layer.output[0, 0, 0].cpu()
                - torch.Tensor(
                    [
                        0.0800926983,
                        0.2299261838,
                        0.1115054339,
                        0.4455563724,
                        0.1744501591,
                        0.3627213538,
                        0.4317324460,
                        0.0737401918,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_latent_representation_mcd_samples(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(self.tests_model.conv2_drop)
        self.tests_model.apply(apply_dropout)  # enable dropout
        mcd_samples = get_latent_representation_mcd_samples(
            self.tests_model, self.test_loader, MCD_N_SAMPLES, hooked_layer, "Conv"
        )
        self.assertEqual(
            mcd_samples.shape, torch.Size([MCD_N_SAMPLES * self.subset_ds_len, LATENT_SPACE_DIM])
        )
        self.assertTrue(
            (
                mcd_samples[0].cpu().numpy()
                - np.array(
                    [
                        0.02988447,
                        0.09705469,
                        0.0450424,
                        0.12505123,
                        -0.01840811,
                        0.0881443,
                        -0.06746943,
                        0.0074361,
                        -0.43022513,
                        0.10256019,
                        -0.5561371,
                        -0.09670735,
                        -0.5716977,
                        0.0211003,
                        -0.30657297,
                        -0.06945786,
                        -0.4759769,
                        0.70155424,
                        -0.05476333,
                        0.2565453,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_mcd_samples_extractor(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(self.tests_model.conv2_drop)
        self.tests_model.apply(apply_dropout)  # enable dropout
        samples_extractor = MCDSamplesExtractor(
            model=self.tests_model,
            mcd_nro_samples=MCD_N_SAMPLES,
            hooked_layers=[hooked_layer],
            layer_type=LAYER_TYPE,
            device=self.device,
            reduction_method=REDUCTION_METHOD,
            return_raw_predictions=False,
        )
        mcd_samples = samples_extractor.get_ls_samples(data_loader=self.test_loader)
        self.assertEqual(
            mcd_samples.shape, torch.Size([MCD_N_SAMPLES * self.subset_ds_len, LATENT_SPACE_DIM])
        )
        self.assertTrue(
            (
                mcd_samples[0].cpu().numpy()
                - np.array(
                    [
                        0.02988447,
                        0.09705469,
                        0.0450424,
                        0.12505123,
                        -0.01840811,
                        0.0881443,
                        -0.06746943,
                        0.0074361,
                        -0.43022513,
                        0.10256019,
                        -0.5561371,
                        -0.09670735,
                        -0.5716977,
                        0.0211003,
                        -0.30657297,
                        -0.06945786,
                        -0.4759769,
                        0.70155424,
                        -0.05476333,
                        0.2565453,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_single_image_entropy_calculation(self):
        np.random.seed(SEED)
        sample_random_image = np.random.rand(MCD_N_SAMPLES, LATENT_SPACE_DIM)
        single_image_entropy = single_image_entropy_calculation(
            sample_random_image, MCD_N_SAMPLES - 1
        )
        self.assertEqual(single_image_entropy.shape, (LATENT_SPACE_DIM,))
        self.assertTrue(
            np.allclose(
                single_image_entropy,
                np.array(
                    [
                        0.50127026,
                        -0.24113694,
                        -0.00449107,
                        0.3995355,
                        0.91656596,
                        0.77765932,
                        0.9553056,
                        -0.05127198,
                        -0.50808706,
                        0.70149445,
                        0.20306963,
                        -0.14639144,
                        0.90684774,
                        0.5116368,
                        0.66483277,
                        0.52607873,
                        -0.29929704,
                        0.64812784,
                        0.55261872,
                        0.56711444,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_get_dl_h_z(self):
        torch.manual_seed(SEED)
        test_latent_rep = torch.rand(MCD_N_SAMPLES * self.subset_ds_len, LATENT_SPACE_DIM)
        test_entropy = get_dl_h_z(test_latent_rep, MCD_N_SAMPLES, parallel_run=True)[1]
        self.assertEqual(test_entropy.shape, (self.subset_ds_len, LATENT_SPACE_DIM))
        self.assertTrue(
            np.allclose(
                test_entropy[0],
                np.array(
                    [
                        0.66213845,
                        -1.24553263,
                        0.43274342,
                        0.16964551,
                        -0.11109334,
                        -0.15076459,
                        0.50606536,
                        -0.938645,
                        0.4492352,
                        0.1962833,
                        0.7151791,
                        0.70086446,
                        -0.89067232,
                        0.01402643,
                        0.96637271,
                        0.6283722,
                        0.34451879,
                        -0.00393629,
                        -0.48039907,
                        0.2459123,
                    ]
                ),
                atol=TOL,
            )
        )


if __name__ == "__main__":
    main()
