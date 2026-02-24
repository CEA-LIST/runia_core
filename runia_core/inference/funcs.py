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
import numpy as np
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from sklearn.covariance import EmpiricalCovariance
from typing import Union, Dict, Tuple

__all__ = [
    "RouteDICE",
    "ash_s_conv_layer",
    "ash_s_linear_layer",
    "gmm_fit",
    "generalized_entropy",
    "get_mcd_pred_uncertainty_score",
    "get_predictive_uncertainty_score",
    "get_dice_feat_mean_react_percentile",
    "mahalanobis_preprocess",
    "mahalanobis_postprocess",
]


def mahalanobis_preprocess(
    ind_data: Dict[str, np.ndarray], num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate class means and precision (inverse covariance) for Mahalanobis.

    The function computes per-class means from the InD training features and
    fits an EmpiricalCovariance on the centered data to obtain a precision matrix
    for subsequent Mahalanobis distance computations.

    Args:
        ind_data: Dictionary with InD data arrays. Expects keys "train features"
            and "train labels".
        num_classes: Number of classes to compute statistics for.

    Returns:
        A tuple (class_mean, precision) where `class_mean` has shape
        [num_classes, feature_dim] and `precision` is the precision matrix
        returned by the fitted EmpiricalCovariance.
    """
    class_mean = []
    centered_data = []
    for c in range(num_classes):
        class_samples = ind_data["train features"][ind_data["train labels"] == c]
        if len(class_samples) == 0:
            warnings.warn(f"No train examples for class {c}")
        class_mean.append(class_samples.mean(0))
        centered_data.append(class_samples - class_mean[c].reshape(1, -1))
    class_mean = np.stack(class_mean)  # shape [#classes, feature dim]

    group_lasso = EmpiricalCovariance(assume_centered=False)

    group_lasso.fit(np.concatenate(centered_data).astype(np.float32))
    # inverse of covariance
    return class_mean, group_lasso.precision_


def mahalanobis_postprocess(
    feats: np.ndarray, class_mean: np.ndarray, precision: np.ndarray, num_classes: int
) -> np.ndarray:
    """Postprocess features into Mahalanobis-based confidence scores.

    For each sample, the function computes the Mahalanobis quadratic form with
    respect to each class mean and returns the maximum (best) confidence per
    sample. Classes without training examples are excluded by replacing NaNs
    with -inf.

    Args:
        feats: Array of shape (N, feature_dim) with features to score.
        class_mean: Array of per-class means with shape (num_classes, feature_dim).
        precision: Precision (inverse covariance) matrix used for the quadratic form.
        num_classes: Number of classes considered.

    Returns:
        1D numpy array of length N with the per-sample Mahalanobis confidence scores.
    """
    all_conf_score = []
    for feats in feats:
        class_scores = np.zeros((1, num_classes))
        for c in range(num_classes):
            tensor = feats - class_mean[c].reshape(1, -1)
            class_scores[:, c] = np.diag(-np.matmul(np.matmul(tensor, precision), tensor.T))
        # Exclude the score for classes with no examples in the training data!
        class_scores[np.isnan(class_scores)] = -np.inf
        conf = np.max(class_scores, axis=1)

        all_conf_score.append(conf)

    all_conf_score_t = np.concatenate(all_conf_score)

    return all_conf_score_t


def normalizer(x):
    """
    Auxiliary function that normalizes the input data.

    Args:
        x: Input data

    Returns:
        Normalized data
    """
    return x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)


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


# ASH with scaling for convolutional layers
def ash_s_conv_layer(x: torch.Tensor, percentile: int = 65):
    """Apply ASH-S scaling for convolutional feature maps.

    This function implements the ASH-S pruning and scaling operation for 4D
    convolutional tensors (B, C, H, W). It zeros out lower-importance elements
    per sample and rescales the remaining values to preserve global energy.

    Args:
        x: A 4D torch Tensor with shape (batch, channels, height, width).
        percentile: Percentile used to determine how many elements to keep.

    Returns:
        The sharpened tensor with the same shape as the input.
    """
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


def ash_s_linear_layer(x: np.ndarray, percentile: int = 85):
    """Apply ASH-S scaling for linear (2D) activations.

    This function keeps the top-k elements per row of a 2D numpy array and
    applies a scaling similar to the convolutional version.

    Args:
        x: 2D numpy array with shape (batch, features).
        percentile: Percentile used to determine how many features to keep.

    Returns:
        A 2D numpy array with the pruned-and-scaled activations.
    """
    assert x.ndim == 2
    assert 0 <= percentile <= 100
    # calculate the sum of the input per sample
    s1 = x.sum(axis=1)
    n = x.shape[1]
    k = n - int(np.round(n * percentile / 100.0))
    idx = np.argpartition(x, -k)[:, -k:]
    top_k = np.partition(x, -k)[:, -k:]
    scattered = np.zeros_like(x)
    np.put_along_axis(scattered, indices=idx, values=top_k, axis=1)

    # calculate new sum of the input per sample after pruning
    s2 = scattered.sum(axis=1)

    # apply sharpening
    scale = s1 / s2
    scattered = scattered * np.exp(scale[:, None])

    return scattered


# DDU calculations
def gmm_fit(
    embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> Tuple[torch.distributions.MultivariateNormal, float]:
    """Fit a class-conditional Gaussian Mixture approximation (per-class MVN).

    The function computes per-class empirical means and covariance matrices
    (with handling for classes that have no examples). It then attempts to build
    a torch.distributions.MultivariateNormal object for the collection of
    class means and covariances, adding a small jitter if necessary to ensure
    positive-definiteness.

    Args:
        embeddings: 2D torch Tensor of shape (N, D) containing feature embeddings.
        labels: 1D torch Tensor with integer class labels for the embeddings.
        num_classes: Number of classes expected.

    Returns:
        A tuple (gmm, jitter_eps) where `gmm` is a torch.distributions.MultivariateNormal
        instance parameterized by stacked class means and covariance matrices,
        and `jitter_eps` is the last jitter scalar tried (0 or a small power of 10)
        that permitted successful construction.
    """
    jitters = [0] + [10**exp for exp in range(-20, 0, 1)]

    def centered_cov_torch(x):
        n = x.shape[0]
        if n == 1:
            n += 1
        res = 1 / (n - 1) * x.t().mm(x)
        return res

    with torch.no_grad():
        classwise_mean_features = torch.stack(
            [torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]
        )
        classwise_cov_features = torch.stack(
            [
                centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c])
                for c in range(num_classes)
            ]
        )
    # Control for classes with no examples
    n_cols_to_remove = torch.any(classwise_mean_features.isnan(), dim=1, keepdim=True).sum().item()
    if n_cols_to_remove > 0:
        cols_bool = ~torch.any(classwise_mean_features.isnan(), dim=1, keepdim=True)
        remaining_cols = classwise_mean_features.shape[0] - n_cols_to_remove
        hidden_dim = classwise_mean_features.shape[1]
        # Subset means
        mean_mask = cols_bool.repeat(1, hidden_dim)
        classwise_mean_features = classwise_mean_features[mean_mask].reshape(remaining_cols, -1)
        # Subset covariances
        cov_mask = (
            cols_bool.repeat(1, hidden_dim)
            .repeat(1, hidden_dim)
            .reshape(num_classes, hidden_dim, hidden_dim)
        )
        classwise_cov_features = classwise_cov_features[cov_mask].reshape(
            remaining_cols, hidden_dim, hidden_dim
        )

    with torch.no_grad():
        for jitter_eps in jitters:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1],
                    device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features,
                    covariance_matrix=(classwise_cov_features + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "found invalid values" in str(e):
                    continue
            break

    return gmm, jitter_eps


def generalized_entropy(probs, gamma, M):
    """
    Calculates a generalized entropy score based on the top M probabilities.

    This function determines the generalized entropy for a set of probability
    distributions. It focuses on the top M probabilities (sorted in descending
    order) and applies a generalized entropy formula with a specific gamma value
    to emphasize concentration or dispersion.

    Args:
        probs: ndarray
            A 2D array where each row represents a probability distribution.
        gamma: float
            A parameter used to adjust the sensitivity to concentration within
            the probability distribution. Higher values of gamma result in
            stronger emphasis on higher probabilities.
        M: int
            The number of top probabilities from each distribution to consider in
            the computation.

    Returns:
        ndarray:
            An array containing the negative generalized entropy scores for each
            distribution row in the input.
    """
    probs_sorted = np.sort(probs, axis=1)[:, -M:]
    scores = np.sum(probs_sorted**gamma * (1 - probs_sorted) ** (gamma), axis=1)

    return -scores


def get_mcd_pred_uncertainty_score(
    dnn_model: torch.nn.Module, input_dataloader: DataLoader, mcd_nro_samples: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    input_samples: torch.Tensor, mcd_nro_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
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
