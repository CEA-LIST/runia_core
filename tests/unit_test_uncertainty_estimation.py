from unittest import TestCase, main
import numpy as np

from runia.uncertainty_estimation import single_image_entropy_calculation, get_dl_h_z
from runia.uncertainty_estimation.baselines import (
    get_mcd_pred_uncertainty_score,
    get_predictive_uncertainty_score,
    get_msp_score,
    get_energy_score,
    MDSPostprocessorFromModelInference,
    KNNPostprocessorFromModelInference,
)
from runia.uncertainty_estimation.feature_extraction import (
    get_latent_representation_mcd_samples,
    MCDSamplesExtractor,
)
from runia.uncertainty_estimation.feature_extraction.utils import apply_dropout, Hook
from runia.uncertainty_estimation.inference import MDLatentSpace, KDELatentSpace
from tests_architecture import Net
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

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset for testing
mnist_data = torchvision.datasets.MNIST(
    "./mnist-data/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
subset_ds_len = int(len(mnist_data) * TEST_SET_PROPORTION)
test_subset = torch.utils.data.random_split(
    mnist_data,
    [subset_ds_len, len(mnist_data) - subset_ds_len],
    torch.Generator().manual_seed(SEED),
)[0]
# DataLoader
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True)
# Define toy model for testing
tests_model = Net(latent_space_dimension=LATENT_SPACE_DIM)
tests_model.to(device)
tests_model.eval()


class Test(TestCase):
    ############################################
    # Uncertainty estimation Tests
    ############################################
    def test_hook_module(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model(next(iter(test_loader))[0].to(device))
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
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model.apply(apply_dropout)  # enable dropout
        mcd_samples = get_latent_representation_mcd_samples(
            tests_model, test_loader, MCD_N_SAMPLES, hooked_layer, "Conv"
        )
        self.assertEqual(
            mcd_samples.shape, torch.Size([MCD_N_SAMPLES * subset_ds_len, LATENT_SPACE_DIM])
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
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model.apply(apply_dropout)  # enable dropout
        samples_extractor = MCDSamplesExtractor(
            model=tests_model,
            mcd_nro_samples=MCD_N_SAMPLES,
            hooked_layers=[hooked_layer],
            layer_type=LAYER_TYPE,
            device=device,
            reduction_method=REDUCTION_METHOD,
            return_raw_predictions=False,
        )
        mcd_samples = samples_extractor.get_ls_samples(data_loader=test_loader)
        self.assertEqual(
            mcd_samples.shape, torch.Size([MCD_N_SAMPLES * subset_ds_len, LATENT_SPACE_DIM])
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
        test_latent_rep = torch.rand(MCD_N_SAMPLES * subset_ds_len, LATENT_SPACE_DIM)
        test_entropy = get_dl_h_z(test_latent_rep, MCD_N_SAMPLES, parallel_run=True)[1]
        self.assertEqual(test_entropy.shape, (subset_ds_len, LATENT_SPACE_DIM))
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

    def test_mcd_pred_uncertainty_score(self):
        torch.manual_seed(SEED)
        samples, pred_h, mi = get_mcd_pred_uncertainty_score(
            dnn_model=tests_model, input_dataloader=test_loader, mcd_nro_samples=MCD_N_SAMPLES
        )
        self.assertEqual(
            samples.shape, torch.Size([subset_ds_len, MCD_N_SAMPLES, len(mnist_data.classes)])
        )
        self.assertTrue(
            (
                samples[0, 0].cpu()
                - torch.Tensor(
                    [
                        0.1046028882,
                        0.1448072642,
                        0.0831882283,
                        0.0958161354,
                        0.1245612875,
                        0.0838165581,
                        0.0954214483,
                        0.0930665582,
                        0.0805836692,
                        0.0941359326,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(pred_h.shape, torch.Size([subset_ds_len]))
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
        self.assertEqual(mi.shape, torch.Size([subset_ds_len]))
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
        input_samples = torch.rand(subset_ds_len * MCD_N_SAMPLES, len(mnist_data.classes))
        pred_h, mi = get_predictive_uncertainty_score(input_samples, MCD_N_SAMPLES)
        self.assertEqual(pred_h.shape, torch.Size([subset_ds_len]))
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
        self.assertEqual(mi.shape, torch.Size([subset_ds_len]))
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

    def test_get_msp_score(self):
        torch.manual_seed(SEED)
        msp_score = get_msp_score(tests_model, test_loader)
        self.assertEqual(msp_score.shape, (subset_ds_len,))
        self.assertTrue(
            np.allclose(
                msp_score[:LATENT_SPACE_DIM],
                np.array(
                    [
                        0.1488787,
                        0.12976062,
                        0.14545625,
                        0.1325143,
                        0.13224202,
                        0.14340535,
                        0.14497909,
                        0.14206077,
                        0.1483772,
                        0.14066638,
                        0.13316885,
                        0.13690054,
                        0.13229437,
                        0.15583877,
                        0.14118572,
                        0.15055893,
                        0.13714625,
                        0.13952997,
                        0.13808227,
                        0.13906068,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_get_energy_score(self):
        torch.manual_seed(SEED)
        energy_score = get_energy_score(tests_model, test_loader)
        self.assertEqual(energy_score.shape, (subset_ds_len,))
        self.assertTrue(
            np.allclose(
                energy_score[:LATENT_SPACE_DIM],
                np.array(
                    [
                        -1.1920929e-07,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        2.3841858e-07,
                        0.0000000e00,
                        0.0000000e00,
                        1.1920929e-07,
                        0.0000000e00,
                        -1.1920929e-07,
                        1.1920929e-07,
                        1.1920929e-07,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_MDSPostprocessor(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(tests_model.fc1)
        mahalanobis_distance_estimator = MDSPostprocessorFromModelInference(
            num_classes=len(mnist_data.classes)
        )
        mahalanobis_distance_estimator.setup(tests_model, test_loader, hooked_layer)
        self.assertEqual(mahalanobis_distance_estimator.precision.shape, torch.Size([50, 50]))
        self.assertTrue(
            (
                mahalanobis_distance_estimator.precision[0, :LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        165.2596282959,
                        -12.5079212189,
                        8.9375352859,
                        -3.4928135872,
                        6.0702023506,
                        -17.6381549835,
                        -1.1519905329,
                        14.5131769180,
                        -36.5683860779,
                        30.9377803802,
                        -5.0159239769,
                        30.1570549011,
                        -9.7787103653,
                        11.7463140488,
                        -5.8539495468,
                        -2.8339817524,
                        -1.1148993969,
                        -21.3802356720,
                        -43.9337997437,
                        -15.5770769119,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(mahalanobis_distance_estimator.class_mean.shape, torch.Size([10, 50]))
        self.assertTrue(
            (
                mahalanobis_distance_estimator.class_mean[0, :LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        0.5837824345,
                        0.0068388470,
                        0.2311379015,
                        0.3187698424,
                        0.2801628113,
                        0.0079951715,
                        0.0159174353,
                        -0.5741344094,
                        0.1619823128,
                        -0.6143783927,
                        0.2275377214,
                        -0.2329114527,
                        -0.0491828360,
                        0.1668033451,
                        0.0840948597,
                        0.0450854897,
                        0.0546373986,
                        0.0204664711,
                        -0.3509032130,
                        0.0088814823,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_kNNPostprocessor(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        hooked_layer = Hook(tests_model.fc1)
        knn_processor = KNNPostprocessorFromModelInference(k=50)
        knn_processor.setup(tests_model, test_loader, hooked_layer)
        self.assertEqual(knn_processor.activation_log.shape, (subset_ds_len, 50))
        self.assertTrue(
            (
                knn_processor.activation_log[0, :LATENT_SPACE_DIM]
                - np.array(
                    [
                        0.38499776,
                        0.09457733,
                        -0.03053672,
                        0.04870516,
                        0.16147998,
                        0.11905161,
                        -0.03847426,
                        -0.25957152,
                        0.06517255,
                        -0.16683035,
                        0.06689644,
                        -0.09510326,
                        0.13610193,
                        0.03214959,
                        0.08813315,
                        0.00172174,
                        0.23592833,
                        0.1691616,
                        -0.05791634,
                        0.06105684,
                    ]
                )
            ).sum()
            < TOL
        )
        post_processed = knn_processor.postprocess(tests_model, test_loader, hooked_layer)[1]
        self.assertEqual(post_processed.shape, (subset_ds_len,))
        self.assertTrue(
            (
                post_processed[:LATENT_SPACE_DIM]
                - np.array(
                    [
                        -0.38611755,
                        -0.319071,
                        -0.30711156,
                        -0.30398387,
                        -0.35546955,
                        -0.36445114,
                        -0.25926197,
                        -0.42174166,
                        -0.28614116,
                        -0.40499124,
                        -0.34399787,
                        -0.29860055,
                        -0.27776933,
                        -0.31202078,
                        -0.4048146,
                        -0.29887676,
                        -0.262555,
                        -0.25697282,
                        -0.28162727,
                        -0.30161116,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_larem_postprocessor(self):
        np.random.seed(SEED)
        test_features = np.random.rand(subset_ds_len, LATENT_SPACE_DIM)
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
        self.assertEqual(postprocessed.shape, (subset_ds_len,))
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
        test_features = np.random.rand(subset_ds_len, LATENT_SPACE_DIM)
        lared_processor = KDELatentSpace()
        lared_processor.setup(test_features)
        postprocessed = lared_processor.postprocess(test_features)
        self.assertEqual(postprocessed.shape, (subset_ds_len,))
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
