# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Based on https://github.com/fregu856/deeplabv3
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya
import torch
import faiss
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from typing import Tuple, Union
from torch import Tensor
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from runia.feature_extraction.utils import Hook

__all__ = [
    "get_mcd_pred_uncertainty_score",
    "get_predictive_uncertainty_score",
    "get_msp_score",
    "get_energy_score",
    "MDSPostprocessorFromModelInference",
    "KNNPostprocessorFromModelInference",
    "normalizer",
    "get_dice_feat_mean_react_percentile",
    "RouteDICE",
]


def get_mcd_pred_uncertainty_score(
    dnn_model: torch.nn.Module, input_dataloader: DataLoader, mcd_nro_samples: int = 2
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    This function performs inference and calculates the predictive uncertainty, the mutual
    information, and returns the predictions, given a model, a dataloader and a number of MCD steps.

    Args:
        dnn_model: Trained model
        input_dataloader: Data Loader
        mcd_nro_samples: Number of samples for MCD dropout

    Returns:
        MCD samples, predictive entropy and mutual information scores
    """
    softmax_fn = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gtsrb_model.to(device)
    with torch.no_grad():
        img_pred_mcd_samples = []

        for image, _ in tqdm(input_dataloader):
            image = image.to(device)

            for sample in range(mcd_nro_samples):
                pred_img = dnn_model(image)

                img_pred_mcd_samples.append(pred_img)

        img_pred_mcd_samples_t = torch.cat(img_pred_mcd_samples, dim=0)

        # compute softmax output - normalized output:
        img_pred_softmax_mcd_samples_t = softmax_fn(img_pred_mcd_samples_t)

        dl_pred_mcd_samples = torch.split(img_pred_softmax_mcd_samples_t, mcd_nro_samples)
        # Get dataloader mcd predictions:
        dl_pred_mcd_samples_t = torch.stack(dl_pred_mcd_samples)

        # get predictive entropy:
        expect_preds = torch.mean(dl_pred_mcd_samples_t, dim=1)
        pred_h_t = -torch.sum((expect_preds * torch.log(expect_preds)), dim=1)
        # get expected entropy:
        preds_h = -torch.sum(dl_pred_mcd_samples_t * torch.log(dl_pred_mcd_samples_t), dim=-1)
        expected_h_preds_t = torch.mean(preds_h, dim=1)
        # get mutual information:
        mi_t = pred_h_t - expected_h_preds_t

    return dl_pred_mcd_samples_t, pred_h_t, mi_t


def get_predictive_uncertainty_score(
    input_samples: Tensor, mcd_nro_samples: int
) -> Tuple[Tensor, Tensor]:
    """
    This function calculates the predictive uncertainty and the mutual information given some
    already calculated activations for a number of MCD steps.

    Args:
        input_samples: Already calculated outputs from a model with the given MCD steps
        mcd_nro_samples: Number of samples for MCD dropout

    Returns:
        Predictive uncertainty, mutual information
    """
    # Check correct dimensions
    assert input_samples.shape[0] % mcd_nro_samples == 0, (
        "Input tensor first dimension must be " "divisible by the mcd_nro_samples"
    )
    softmax_fn = torch.nn.Softmax(dim=1)
    # compute softmax output - normalized output:
    img_pred_softmax_mcd_samples_t = softmax_fn(input_samples)

    dl_pred_mcd_samples = torch.split(img_pred_softmax_mcd_samples_t, mcd_nro_samples)
    # Get dataloader mcd predictions:
    dl_pred_mcd_samples_t = torch.stack(dl_pred_mcd_samples)

    # get predictive entropy:
    expect_preds = torch.mean(dl_pred_mcd_samples_t, dim=1)
    pred_h_t = -torch.sum((expect_preds * torch.log(expect_preds)), dim=1)
    # get expected entropy:
    preds_h = -torch.sum(dl_pred_mcd_samples_t * torch.log(dl_pred_mcd_samples_t), dim=-1)
    expected_h_preds_t = torch.mean(preds_h, dim=1)
    # get mutual information:
    mi_t = pred_h_t - expected_h_preds_t

    return pred_h_t, mi_t


def get_msp_score(dnn_model: torch.nn.Module, input_dataloader: DataLoader) -> np.ndarray:
    """
    Calculates the Maximum softmax probability score given a model and a dataloader

    Args:
        dnn_model: Trained torch or lightning model
        input_dataloader: Dataloader

    Returns:
        MSP scores
    """
    softmax_fn = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_model.to(device)
    dl_preds_msp_scores = []

    with torch.no_grad():
        for image, _ in tqdm(input_dataloader, desc="Getting MSP score"):
            image = image.to(device)
            pred_logits = dnn_model(image)

            pred_score = torch.max(softmax_fn(pred_logits), dim=1)
            # get the max values:
            dl_preds_msp_scores.append(pred_score[0])

        dl_preds_msp_scores_t = torch.cat(dl_preds_msp_scores, dim=0)
        # pred = np.max(softmax_fn(pred_logits).detach().cpu().numpy(), axis=1)
        dl_preds_msp_scores = dl_preds_msp_scores_t.detach().cpu().numpy()

    return dl_preds_msp_scores


def get_energy_score(dnn_model: torch.nn.Module, input_dataloader: DataLoader) -> np.ndarray:
    """
    Calculates the energy uncertainty score

    Args:
        dnn_model: Trained torch or lightning model
        input_dataloader: Dataloader

    Returns:
        Energy scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_preds_energy_scores = []

    with torch.no_grad():
        for image, _ in tqdm(input_dataloader, desc="Getting energy score"):
            image = image.to(device)
            pred_logits = dnn_model(image)

            pred_energy_score = torch.logsumexp(pred_logits, dim=1)

            dl_preds_energy_scores.append(pred_energy_score)

        dl_preds_energy_scores_t = torch.cat(dl_preds_energy_scores, dim=0)

        dl_preds_energy_scores = dl_preds_energy_scores_t.detach().cpu().numpy()

    return dl_preds_energy_scores


class MDSPostprocessorFromModelInference:
    """
    Mahalanobis Distance Score uncertainty estimator class

    Args:
        num_classes: Number of In-distribution samples
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, num_classes: int = 43, setup_flag: bool = False):
        """
        Mahalanobis Distance Score uncertainty estimator class

        Args:
            num_classes: Number of In-distribution samples
            setup_flag: Whether the postprocessor is already trained
        """
        self.num_classes = num_classes
        self.setup_flag = setup_flag
        self.precision = None
        self.class_mean = None

    def setup(
        self, dnn_model: torch.nn.Module, ind_dataloader: DataLoader, layer_hook: Hook
    ) -> None:
        """
        Estimate the parameters of a multivariate normal distribution from a set of data

        Args:
            dnn_model: Trained torch or lightning model
            ind_dataloader: Dataloader
            layer_hook: Hook to the layer to take samples from

        """
        if not self.setup_flag:
            # estimate mean and variance from training set
            print("\n Estimating mean and variance from training set...")
            all_feats = []
            all_labels = []
            # get features/representations:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dnn_model.to(device)
            # get features:
            with torch.no_grad():
                for image, label in tqdm(ind_dataloader, desc="Setting MDist"):
                    image = image.to(device)
                    _ = dnn_model(image)
                    latent_rep = torch.flatten(layer_hook.output, 1)  # latent representation sample
                    all_feats.append(latent_rep.cpu())
                    all_labels.append(deepcopy(label))

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            # compute class-conditional statistics:
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples - self.class_mean[c].view(1, -1))

            self.class_mean = torch.stack(self.class_mean)  # shape [#classes, feature dim]

            group_lasso = EmpiricalCovariance(assume_centered=False)

            group_lasso.fit(torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(
        self, dnn_model: torch.nn.Module, dataloader: DataLoader, layer_hook: Hook
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform inference with the set-up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            dnn_model: Trained torch or lightning model
            dataloader: Dataloader
            layer_hook: Hook to the layer to take samples from

        Returns:
            (tuple): Model predictions and confidence scores
        """
        all_preds = []
        all_conf_score = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dnn_model.to(device)

        for image, _ in tqdm(dataloader, desc="Calculating MDist"):
            image = image.to(device)
            pred_logits = dnn_model(image)
            latent_rep = torch.flatten(layer_hook.output, 1)
            pred = pred_logits.argmax(1)

            all_preds.append(pred)

            class_scores = torch.zeros((pred_logits.shape[0], self.num_classes))
            for c in range(self.num_classes):
                tensor = latent_rep.cpu() - self.class_mean[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, self.precision), tensor.t()
                ).diag()

            conf = torch.max(class_scores, dim=1)[0]

            all_conf_score.append(conf)

        all_preds_t = torch.cat(all_preds)
        all_conf_score_t = torch.cat(all_conf_score)

        return all_preds_t, all_conf_score_t


def normalizer(x):
    """
    Auxiliary function that normalizes the input data.

    Args:
        x: Input data

    Returns:
        Normalized data
    """
    return x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)


class KNNPostprocessorFromModelInference:
    """
    kNN Distance Score uncertainty estimator class

    Args:
        k: Number of neighbors for calculations
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, k: int = 50, setup_flag: bool = False):
        """
        kNN Distance Score uncertainty estimator class

        Args:
            k: Number of neighbors for calculations
            setup_flag: Whether the postprocessor is already trained
        """
        self.K = k
        self.activation_log = None
        self.setup_flag = setup_flag
        self.index = None

    def setup(
        self, dnn_model: torch.nn.Module, ind_dataloader: DataLoader, layer_hook: Hook
    ) -> None:
        """
        Estimate the parameters of a kNN estimator

        Args:
            dnn_model: Trained torch or lightning model
            ind_dataloader: Dataloader
            layer_hook: Hook to the layer to take samples from

        """
        if not self.setup_flag:
            print("\n Get latent embeddings z from training set...")
            activation_log = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dnn_model.to(device)

            with torch.no_grad():
                for image, _ in tqdm(ind_dataloader, desc="Setting kNN"):
                    image = image.to(device)
                    _ = dnn_model(image)

                    latent_rep = torch.flatten(layer_hook.output, 1)  # latent representation sample
                    # ic(layer_hook.output)
                    activation_log.append(normalizer(latent_rep.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.index = faiss.IndexFlatL2(latent_rep.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(
        self, dnn_model: torch.nn.Module, dataloader: DataLoader, layer_hook: Hook
    ) -> Tuple[Tensor, np.ndarray]:
        """
        Perform inference with the set-up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            dnn_model: Trained torch or lightning model
            dataloader: Dataloader
            layer_hook: Hook to the layer to take samples from

        Returns:
            (tuple): Model predictions and confidence scores
        """
        all_preds = []
        all_kth_dist_score = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dnn_model.to(device)

        for image, _ in tqdm(dataloader, desc="Calculating kNN scores"):
            image = image.to(device)
            pred_logits = dnn_model(image)
            # ic(layer_hook.output)
            latent_rep = torch.flatten(layer_hook.output, 1)  # latent representation sample

            pred = torch.max(torch.softmax(pred_logits, dim=1), dim=1)
            latent_rep_normed = normalizer(latent_rep.data.cpu().numpy())

            D, _ = self.index.search(latent_rep_normed, self.K)
            kth_dist = -D[:, -1]

            all_preds.append(pred[0])
            all_kth_dist_score.append(kth_dist)

        all_preds_t = torch.cat(all_preds)
        # all_kth_dist_score_t = torch.cat(all_kth_dist_score)
        all_kth_dist_score_np = np.concatenate(all_kth_dist_score, axis=0)

        return all_preds_t, all_kth_dist_score_np

    def set_K_hyperparam(self, hyperparam: int = 50):
        self.K = hyperparam

    def get_K_hyperparam(self):
        return self.K


def get_dice_feat_mean_react_percentile(
    dnn_model: torch.nn.Module, ind_dataloader: DataLoader, react_percentile: int = 90
) -> Tuple[np.ndarray, float]:
    """
    Get the DICE and ReAct thresholds for sparsifying and clipping from a given model.

    Args:
        dnn_model: The RCNN model
        ind_dataloader: The Data loader
        react_percentile: Desired percentile for ReAct

    Returns:
        Tuple[np.ndarray, float]: The DICE expected values, and the ReAct threshold
    """
    assert 0 < react_percentile < 100, "react_percentile must be greater than 0 and less than 100"
    feat_log = []
    dnn_model.eval()
    assert dnn_model.dice_precompute
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs, targets in tqdm(ind_dataloader, desc="Setting up DICE/ReAct"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = dnn_model(inputs)
        out = adaptive_avg_pool2d(outputs, 1)
        out = out.view(out.size(0), -1)
        # score = dnn_model.fc(out)
        feat_log.append(out.data.cpu().numpy())
    feat_log_array = np.array(feat_log).squeeze()
    return feat_log_array.mean(0), np.percentile(feat_log_array, react_percentile)


"""
DICE Code taken from https://github.com/deeplearning-wisc/dice/blob/master/models/route.py
All credits to authors
"""


class RouteDICE(torch.nn.Linear):
    """
    Class to replace the penultimate fully connected layer of a network in order to use the
    DICE method

    Args:
        in_features: Dimension of the input vector
        out_features: Dimension of the output vector
        bias: Bias for the Linear layer
        p: Percentile for sparsifying
        conv1x1: Whether using a 1x1 conv layer
        info: The previously calculated expected values
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        p: int = 90,
        conv1x1: bool = False,
        info: Union[None, np.ndarray] = None,
    ):
        """
        Class to replace the penultimate fully connected layer of a network in order to use the
        DICE method

        Args:
            in_features: Dimension of the input vector
            out_features: Dimension of the output vector
            bias: Bias for the Linear layer
            p: Percentile for sparsifying
            conv1x1: Whether using a 1x1 conv layer
            info: The previously calculated expected values
        """
        assert 0 < p < 100, "p must be greater than 0 and less than 100"
        if info is not None:
            assert isinstance(info, np.ndarray), "info must be a numpy array or None"

        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p
        self.info = info
        self.masked_w = None
        self.contrib = None
        self.thresh = None

    def calculate_mask_weight(self):
        self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()
        # self.contrib = np.abs(self.contrib)
        # self.contrib = np.random.rand(*self.contrib.shape)
        # self.contrib = self.info[None, :]
        # self.contrib = np.random.rand(*self.info[None, :].shape)
        self.thresh = np.percentile(self.contrib, self.p)
        mask = torch.Tensor((self.contrib > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, x):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = x[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out
