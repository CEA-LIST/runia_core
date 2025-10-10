# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Based on https://github.com/fregu856/deeplabv3
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya
import warnings
from typing import Union

import faiss
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import KernelDensity
from torch import Tensor

from runia.baselines.from_model_inference import (
    normalizer,
)
from runia.baselines.from_precalculated import (
    mahalanobis_preprocess,
    mahalanobis_postprocess,
    gmm_fit,
    generalized_entropy,
)
from runia.inference.abstract_classes import (
    Postprocessor,
    OodPostprocessor,
)

__all__ = [
    "KDELatentSpace",
    "MDLatentSpace",
    "cMDLatentSpace",
    "KNNLatentSpace",
    "GMMLatentSpace",
    "Energy",
    "GEN",
    "DDU",
    "Mahalanobis",
    "ViM",
    "postprocessors_dict",
    "postprocessor_input_dict",
]


class DetectorKDE:
    """
    Instantiates a Kernel Density Estimation Estimator. See
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html for
    more details

    Args:
        train_embeddings: Samples to train the estimator
        save_path: Optional path to save the estimator
        kernel: Kernel. Default='gaussian'
        bandwidth: Bandwidth of the estimator.
    """

    def __init__(self, train_embeddings, save_path=None, kernel="gaussian", bandwidth=1.0) -> None:
        """
        Instantiates a Kernel Density Estimation Estimator. See
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html for
        more details

        Args:
            train_embeddings: Samples to train the estimator
            save_path: Optional path to save the estimator
            kernel: Kernel. Default='gaussian'
            bandwidth: Bandwidth of the estimator.
        """
        assert isinstance(train_embeddings, np.ndarray), "train_embeddings must be a numpy array"
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.train_embeddings = train_embeddings
        self.save_path = save_path
        self.density = self.density_fit()

    def density_fit(self):
        """
        Fit the KDE Estimator
        """
        density = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(
            self.train_embeddings
        )
        return density

    def get_density_scores(self, test_embeddings):
        """
        Transforms the scores from a second distribution while normalizing the scores

        Args:
            test_embeddings: The new samples to get the density scores

        Returns:
            Density scores
        """
        return self.density.score_samples(test_embeddings)


class KDELatentSpace(Postprocessor):
    """
    Kernel Density Estimator Distance Score uncertainty estimator class for already calculated representations.

    Args:
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, cfg: DictConfig = None):
        """
        Kernel Density Distance Score uncertainty estimator class for already calculated representations.

        Args:
            setup_flag: Whether the postprocessor is already trained
        """
        super().__init__(cfg)
        self.detector = None

    def setup(self, ind_train_feats: np.ndarray, **kwargs) -> None:
        """
        Estimate the Kernel density estimator distribution from a set of data

        Args:
            ind_train_feats: InD features to estimate the distribution

        """
        assert isinstance(ind_train_feats, np.ndarray), "ind_feats must be a numpy array"
        assert ind_train_feats.ndim == 2, "ind_feats must be 2 dimensional"
        if not self._setup_flag:
            self.detector = DetectorKDE(train_embeddings=ind_train_feats)
            self._setup_flag = True
        else:
            warnings.warn("KDEPostprocessor already trained")

    def postprocess(self, test_feats: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform inference with the set-up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            test_feats: Features (either InD or OoD) to estimate if they belong to the InD
                distribution

        Returns:
            (tuple): Confidence scores
        """
        assert isinstance(test_feats, np.ndarray), "test_feats must be a numpy array"
        assert test_feats.ndim == 2, "ood_feats must be 2 dimensional"
        return self.detector.get_density_scores(test_feats)


class MDLatentSpace(Postprocessor):
    """
    Mahalanobis distance Score uncertainty estimator class for already calculated representations.

    Args:
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, cfg: DictConfig = None):
        """
        Mahalanobis Distance Score uncertainty estimator class for already calculated representations.

        Args:
            setup_flag: Whether the postprocessor is already trained
        """
        super().__init__(cfg)
        self.feats_mean = None
        self.precision = None
        self.centered_data = None

    def setup(self, ind_train_feats: np.ndarray, **kwargs) -> None:
        """
        Estimate the parameters of a multivariate normal distribution from a set of data

        Args:
            ind_train_feats: InD features to estimate the distribution

        """
        assert isinstance(ind_train_feats, np.ndarray), "ind_feats must be a numpy array"
        assert ind_train_feats.ndim == 2, "ind_feats must be 2 dimensional"
        if not self._setup_flag:
            # estimate mean and variance from training set
            self.feats_mean = np.mean(ind_train_feats, 0, keepdims=True)

            self.centered_data = ind_train_feats - self.feats_mean

            group_lasso = EmpiricalCovariance(assume_centered=False)
            group_lasso.fit(self.centered_data)

            self.precision = group_lasso.precision_

            self._setup_flag = True
            # we need to use:
            # self.feats_mean & self.precision
        else:
            warnings.warn("MDPostprocessor already trained")

    def postprocess(self, test_feats: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform inference with the set-up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            test_feats: Features (either InD or OoD) to estimate if they belong to the InD
                distribution

        Returns:
            (np.ndarray): Confidence scores
        """
        assert isinstance(test_feats, np.ndarray), "test_feats must be a numpy array"
        assert test_feats.ndim == 2, "test_feats must be 2 dimensional"
        diff = test_feats - self.feats_mean
        conf_score = -np.diag(np.matmul(np.matmul(diff, self.precision), np.transpose(diff)))

        return conf_score


class cMDLatentSpace(Postprocessor):
    """
    LaREM with category specific information Distance Score uncertainty estimator class for
    already calculated representations.

    Args:
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, cfg: DictConfig = None):
        """
        LaREM Distance Score uncertainty estimator class for already calculated representations.

        Args:
            setup_flag: Whether the postprocessor is already trained
        """
        super().__init__(cfg)
        try:
            self.num_classes = cfg.num_classes
        except AttributeError:
            self.num_classes = 10

        self.feats_mean = None
        self.precision = None
        self.class_mean = None

    def setup(self, ind_train_feats: np.ndarray, **kwargs) -> None:
        """
        Estimate the parameters of a multivariate normal distribution from a set of data

        Args:
            ind_train_feats: InD features to estimate the distribution

        """
        # Get ground truth InD labels
        try:
            ind_train_labels = kwargs["ind_train_labels"]
            if isinstance(ind_train_labels, np.ndarray):
                ind_train_labels = Tensor(ind_train_labels)
        except KeyError:
            raise ValueError(
                "id_labels not provided. Pass ID train labels as 'ind_train_labels' argument."
            )

        if isinstance(ind_train_feats, np.ndarray):
            ind_train_feats = Tensor(ind_train_feats)
        assert ind_train_feats.ndim == 2, "ind_feats must be 2 dimensional"
        if not self._setup_flag:
            # compute class-conditional statistics:
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = ind_train_feats[ind_train_labels.eq(c)].data
                if len(class_samples) == 0:
                    warnings.warn(
                        f"No examples for class {c} to build class-wise Mahalanobis Distance score"
                    )
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples - self.class_mean[c].view(1, -1))

            self.class_mean = torch.stack(self.class_mean)  # shape [#classes, feature dim]

            group_lasso = EmpiricalCovariance(assume_centered=False)

            group_lasso.fit(torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(group_lasso.precision_).float()
            self._setup_flag = True

        else:
            warnings.warn("cMDPostprocessor already trained")

    def postprocess(self, test_feats: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform inference with the set-up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            test_feats: Features (either InD or OoD) to estimate if they belong to the InD
                distribution

        Returns:
            (tuple): Confidence scores
        """
        try:
            pred_labels = kwargs["pred_labels"]
            if isinstance(pred_labels, np.ndarray):
                pred_labels = Tensor(pred_labels)
        except KeyError:
            raise ValueError("pred_logits not provided")
        # assert isinstance(ood_feats, np.ndarray), "ood_feats must be a numpy array"
        if isinstance(test_feats, np.ndarray):
            test_feats = Tensor(test_feats)
        assert test_feats.ndim == 2, "test_feats must be 2 dimensional"
        all_conf_score = []
        for feat in test_feats:
            class_scores = torch.zeros((1, self.num_classes))
            for c in range(self.num_classes):
                tensor = feat - self.class_mean[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, self.precision), tensor.t()
                ).diag()
            # Exclude the score for classes with no examples in the training data!
            class_scores[torch.isnan(class_scores)] = torch.tensor(-np.inf)
            conf = torch.max(class_scores, dim=1)[0]

            all_conf_score.append(conf)

        all_conf_score_t = torch.cat(all_conf_score)

        return all_conf_score_t.numpy()


class KNNLatentSpace(Postprocessor):
    """
    KNN Distance Score uncertainty estimator class for already calculated representations.

    Args:
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, cfg: DictConfig = None):
        """
        KNN Distance Score uncertainty estimator class for already calculated representations.

        Args:
            setup_flag: Whether the postprocessor is already trained
            cfg: Config class that may contain the k_neighbors key to control k
        """
        super().__init__(cfg)
        try:
            self.K = cfg.k_neighbors
        except AttributeError:
            self.K = 50
        self.activation_log = None
        self.index = None

    def setup(self, ind_train_feats: np.ndarray, **kwargs) -> None:
        """
        Add the precalculated points to a point cloud to build the reference ID distribution.

        Args:
            ind_train_feats: InD features to estimate the distribution

        """
        assert isinstance(ind_train_feats, np.ndarray), "ind_train_feats must be a numpy array"
        assert ind_train_feats.ndim == 2, "ind_train_feats must be 2 dimensional"
        if not self._setup_flag:
            self.activation_log = np.array([normalizer(feat) for feat in ind_train_feats])
            self.index = faiss.IndexFlatL2(ind_train_feats.shape[1])
            self.index.add(self.activation_log)
            self._setup_flag = True

        else:
            warnings.warn("KNNPostprocessor already trained")

    def postprocess(self, test_feats: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform inference with the set-up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            test_feats: Features (either InD or OoD) to estimate if they belong to the InD
                distribution

        Returns:
            (tuple): Confidence scores
        """
        assert isinstance(test_feats, np.ndarray), "test_feats must be a numpy array"
        assert test_feats.ndim == 2, "test_feats must be 2 dimensional"
        all_kth_dist_score = []
        for feat in test_feats:
            latent_rep_normed = normalizer(feat.reshape(1, -1))
            D, _ = self.index.search(latent_rep_normed, self.K)
            kth_dist = -D[:, -1]
            all_kth_dist_score.append(kth_dist)
        all_kth_dist_score_np = np.concatenate(all_kth_dist_score, axis=0)
        return all_kth_dist_score_np


class GMMLatentSpace(Postprocessor):
    """
    LaREG Gaussian Mixture model Score estimator class for already calculated representations.

    Args:
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, cfg: DictConfig = None):
        """
        LaREG Gaussian Mixture model Score estimator class for already calculated representations.

        Args:
            setup_flag: Whether the postprocessor is already trained
        """
        super().__init__(cfg)
        try:
            self.num_classes = cfg.num_classes
        except AttributeError:
            self.num_classes = 10

        self.gmm = None

    def setup(self, ind_train_feats: np.ndarray, **kwargs) -> None:
        """
        Estimate the parameters of a multivariate normal distribution from a set of data

        Args:
            ind_train_feats: InD features to estimate the distribution

        """
        assert isinstance(ind_train_feats, np.ndarray), "ind_train_feats must be a numpy array"
        assert ind_train_feats.ndim == 2, "ind_train_feats must be 2 dimensional"
        if not self._setup_flag:
            # Get ground truth InD labels
            try:
                ind_predicted_labels = kwargs["ind_train_labels"]
                if isinstance(ind_predicted_labels, np.ndarray):
                    ind_predicted_labels = Tensor(ind_predicted_labels)
            except KeyError:
                raise ValueError("id_labels not provided")
            self.gmm, _ = gmm_fit(
                embeddings=Tensor(ind_train_feats),
                labels=ind_predicted_labels,
                num_classes=self.num_classes,
            )
            self._setup_flag = True

        else:
            warnings.warn("GMMPostprocessor already trained")

    def postprocess(self, test_feats: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform inference with the set-up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            test_feats: Features (either InD or OoD) to estimate if they belong to the InD
                distribution

        Returns:
            (tuple): Confidence scores
        """
        assert isinstance(test_feats, np.ndarray), "test_feats must be a numpy array"
        assert test_feats.ndim == 2, "test_feats must be 2 dimensional"
        log_probs = self.gmm.log_prob(Tensor(test_feats[:, None, :]))
        energy: np.ndarray = logsumexp(log_probs.numpy(), axis=1)
        return energy


class Energy(OodPostprocessor):
    """
    Performs energy-based postprocessing for out-of-distribution (OOD) detection.

    This class calculates energy-based scores for inputs and applies a threshold
    for determining OOD samples. It allows customization of the threshold setup
    and postprocessing behavior through its methods.

    Attributes:
        method_name (str): Name of the method used for scoring, which is a
            key in the output score dictionary.
        flip_sign (bool): Indicates whether the calculated scores should
            be inverted during postprocessing.
    """

    def setup(self, ind_train_data: np.ndarray, **kwargs):
        """
        Sets up the threshold for the post-hoc OOD detection method.
        This includes computing scores based on the method for the InD train logits, transforming
        them, and setting a threshold for further processing.

        Args:
            ind_train_data (np.ndarray): A numpy array containing the features of
                in-distribution training data.
            **kwargs: Additional keyword arguments that may be required by the
                method or processing logic.

        """
        # Need ID scores in Dict format for threshold setup
        ind_scores = {self.method_name: logsumexp(ind_train_data, axis=1)}
        ind_scores = self.flip_sign_fn(ind_scores)
        self.set_threshold(ind_scores)

    def postprocess(self, test_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Post-processes the given test data array and computes confidence scores.

        This method performs post-processing operations on the provided test data,
        including conversion from PyTorch Tensor to NumPy array (if applicable) and
        applying a scoring function to modify the data.

        Args:
            test_data (np.ndarray): Input array containing test data. If the input
                is of type `Tensor`, it will be converted to a NumPy array.
            **kwargs: Additional parameters that may be passed but are not utilized
                explicitly in this method.

        Returns:
            np.ndarray: The computed scores after applying the scoring function.
        """
        assert self._setup_flag, "setup() must be called before postprocess()"
        if isinstance(test_data, Tensor):
            test_data = test_data.cpu().numpy()
        scores = logsumexp(test_data, axis=1)
        scores = self.flip_sign_fn(scores)
        return scores


class GEN(OodPostprocessor):
    """
    Applies a generalized entropy-based method for postprocessing out-of-distribution (OOD) detection scores.

    This class extends the OodPostprocessor class to implement a postprocessing method
    based on generalized entropy for handling OOD detection tasks. It provides
    functionality to configure the method using training data, set thresholds,
    and computes entropy-based OOD detection scores for test data.

    Attributes:
        method_name (str): The name of the postprocessing method.
        gamma (float): The entropy regularization parameter that controls the level of
            entropy applied to scores.
        num_classes (int): The number of classes in the classification task.
    """

    def __init__(
        self,
        method_name: str,
        flip_sign: bool,
        gamma: float,
        num_classes: int,
        cfg: DictConfig = None,
    ):
        """
        Initializes the class with the provided parameters necessary for its functionality. Sets up
        the method with its configuration including gamma and class numbers, and specifies whether
        to invert the score or use a setup flag.

        Args:
            method_name (str): Name of the method to initialize.
            flip_sign (bool): Whether the score inversion is applied.
            gamma (float): The gamma parameter for the method.
            num_classes (int): Number of classes being used.
            setup_flag (bool): A flag to determine if additional setup is required.
            cfg (DictConfig): Configuration dictionary for additional settings.
        """
        super().__init__(method_name, flip_sign, cfg)
        self.method_name = method_name
        self.gamma = gamma
        self.num_classes = num_classes

    def setup(self, ind_train_data: np.ndarray, **kwargs):
        """
        Sets up the necessary components to compute thresholds for confidence scores.

        The function processes the provided independent training data by applying the softmax
        operation along a specified axis. The resultant softmax values are then used to calculate
        generalized entropy scores, which are associated to the current method name. These scores
        are further transformed using the inversion function and finally used to set the
        threshold for the method.

        Args:
            ind_train_data (np.ndarray): Independent training data to be used for calculating
                identification scores and thresholds.
            **kwargs: Additional keyword arguments for configuration or optional parameters.
        """
        softmax_ind_train = softmax(ind_train_data, axis=1)
        # Need ID scores in Dict format for threshold setup
        ind_scores = {
            self.method_name: generalized_entropy(softmax_ind_train, self.gamma, self.num_classes)
        }
        ind_scores = self.flip_sign_fn(ind_scores)
        self.set_threshold(ind_scores)

    def postprocess(self, test_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Postprocesses the given test data to compute scores using the generalized entropy score.
        This method assumes that the setup() method has been called and the internal state is
        properly initialized before execution.

        Args:
            test_data (np.ndarray): Input test data that needs to be postprocessed. If the input is a
                Tensor, it will automatically be converted to a NumPy array.
            **kwargs: Arbitrary keyword arguments for additional configurations or options.

        Returns:
            np.ndarray: The computed scores after postprocessing the input data.
        """
        assert self._setup_flag, "setup() must be called before postprocess()"
        if isinstance(test_data, Tensor):
            test_data = test_data.cpu().numpy()
        softmax_test = softmax(test_data, axis=1)
        scores = generalized_entropy(softmax_test, self.gamma, self.num_classes)
        scores = self.flip_sign_fn(scores)
        return scores


class DDU(OodPostprocessor):
    """
    DDU (Deep Deterministic Uncertainty) postprocessor class.

    This class implements a postprocessor for managing uncertainty utilizing Gaussian Mixture
    Models (GMM). The purpose is to compute scores for given test data, indicating uncertainty
    or out-of-distribution likelihood. It requires setup with appropriate training data and
    parameters before postprocessing can occur.

    Attributes:
        num_classes (int): The number of classes in the classification problem.
        gmm (GMM or None): Instance of a Gaussian Mixture Model for embeddings. Default is None
            and assigned during the `setup` phase.
        device (str): Device to be used for computation, either "cuda" (if available) or "cpu".
    """

    def __init__(
        self,
        method_name: str,
        flip_sign: bool,
        num_classes: int,
        cfg: DictConfig = None,
    ):
        """
        Initializes the instance of the class with the given parameters.

        Args:
            method_name (str): The name of the method.
            flip_sign (bool): Flag indicating whether the score should be multiplied by -1 to ensure ID scores are higher than OOD scores.
            num_classes (int): The number of classes for the classification task.
            cfg (DictConfig): Optional configuration dictionary.

        Attributes:
            num_classes (int): The number of classes for the classification task.
            gmm: A placeholder for the Gaussian Mixture Model, initially set to None.
            device (str): The device type used for computations, either "cuda" if a
                GPU is available, or "cpu" otherwise.
        """
        super().__init__(method_name, flip_sign, cfg)
        self.num_classes = num_classes
        self.gmm = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self, ind_train_data: np.ndarray, **kwargs):
        """
        Sets up the configuration and initializes the necessary components for the DDU
        method. This function fits a Gaussian Mixture Model (GMM) using the supplied
        training embeddings and labels, then computes log probabilities and energy
        scores required for threshold setup.

        Args:
            ind_train_data (np.ndarray): In-distribution training data features.
            **kwargs: Additional keyword arguments:
                - valid_feats (np.ndarray): Validation set features for GMM score
                  evaluation.
                - train_labels (np.ndarray): Training set labels corresponding to the
                  `ind_train_data`.

        Raises:
            AssertionError: If "valid_feats" is not provided in `kwargs`.
            AssertionError: If "train_labels" is not provided in `kwargs`.
        """
        assert "valid_feats" in kwargs, "valid_feats must be provided for DDU"
        assert "train_labels" in kwargs, "train_labels must be provided for DDU"
        self.gmm, _ = gmm_fit(
            embeddings=Tensor(ind_train_data).to(self.device),
            labels=Tensor(kwargs["train_labels"]).to(self.device),
            num_classes=self.num_classes,
        )
        ind_test_log_probs = self.gmm.log_prob(
            Tensor(kwargs["valid_feats"][:, None, :]).to(self.device)
        )
        ind_energy = logsumexp(ind_test_log_probs.cpu().numpy(), axis=1)
        # Need ID scores in Dict format for threshold setup
        ind_scores = {self.method_name: ind_energy}
        ind_scores = self.flip_sign_fn(ind_scores)
        self.set_threshold(ind_scores)

    def postprocess(self, test_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies post-processing steps to the input test data.

        This method processes the given test data using a Gaussian Mixture Model
        (GMM). It calculates the log probabilities of the test samples against the
        GMM, derives scores using a logarithmic sum exponential approach, and applies
        an inversion function on the scores for final output.

        Args:
            test_data: Input data to process as a NumPy array.
            **kwargs: Additional keyword arguments for the method.

        Returns:
            np.ndarray: The processed scores as a NumPy array.
        """
        assert self._setup_flag, "setup() must be called before postprocess()"
        if isinstance(test_data, np.ndarray):
            test_data = Tensor(test_data)
        test_log_probs = self.gmm.log_prob(test_data[:, None, :].to(self.device))
        scores = logsumexp(test_log_probs.cpu().numpy(), axis=1)
        scores = self.flip_sign_fn(scores)
        return scores


class Mahalanobis(OodPostprocessor):
    """Handles Mahalanobis distance calculation for out-of-distribution detection.

    The Mahalanobis class provides methods for calculating the Mahalanobis distance
    based on in-distribution data and utilizes it for postprocessing test data to
    perform out-of-distribution (OOD) detection. It involves setting up the Mahalanobis
    preprocessing pipeline using training data and then applying it to test data
    to compute scores.

    Attributes:
        num_classes (int): Number of classes in the in-distribution data.
        class_mean (np.ndarray, optional): Class mean vectors calculated during
            the setup phase, used for Mahalanobis distance computation.
        precision (np.ndarray, optional): Precision matrix calculated during the
            setup phase, used for Mahalanobis distance computation.
    """

    def __init__(
        self,
        method_name: str,
        flip_sign: bool,
        num_classes: int,
        cfg: DictConfig = None,
    ):
        """
        Initializes an instance with the specified method parameters, including the
        number of classes and configuration. Additionally, initializes placeholders
        for attributes like `class_mean` and `precision`.

        Args:
            method_name (str): Name of the specified method to be used.
            flip_sign (bool): Indicates whether to multiply the scores by -1 to ensure ID scores are higher than OOD scores.
            num_classes (int): Number of classes to be utilized in the method.
            cfg (DictConfig or None): Optional configuration settings for the method.
        """
        super().__init__(method_name, flip_sign, cfg)
        self.num_classes = num_classes
        self.class_mean = None
        self.precision = None

    def setup(self, ind_train_data: np.ndarray, **kwargs):
        """
        Preprocesses training data and calculates class means and precision matrix using
        Mahalanobis distance. Validates the features and stores the calculated threshold
        based on validation set scores.

        Args:
            ind_train_data (np.ndarray): The training data features for in-distribution data.
            **kwargs: Additional parameters required for processing.
                train_labels: Labels corresponding to training data.
                valid_feats: Features for validation.

        """
        assert "train_labels" in kwargs, "train_labels must be provided for Mahalanobis"
        assert "valid_feats" in kwargs, "valid_feats must be provided for Mahalanobis"
        ind_data_dict = {"train features": ind_train_data, "train labels": kwargs["train_labels"]}
        self.class_mean, self.precision = mahalanobis_preprocess(
            ind_data=ind_data_dict, num_classes=self.num_classes
        )
        ind_valid_score = mahalanobis_postprocess(
            feats=kwargs["valid_feats"],
            class_mean=self.class_mean,
            precision=self.precision,
            num_classes=self.num_classes,
        )
        ind_scores = {self.method_name: ind_valid_score}
        ind_scores = self.flip_sign_fn(ind_scores)
        self.set_threshold(ind_scores)

    def postprocess(self, test_data: Union[np.ndarray, Tensor], **kwargs) -> np.ndarray:
        """
        Postprocesses the test data to compute scores using Mahalanobis distance.

        This method takes the test data, which can be either a Numpy array or a Tensor,
        and computes the Mahalanobis distance scores based on class means and precision
        matrix. If the input is a Tensor, it is first converted to a Numpy array. The
        computed scores are then inverted if needed using the invert_score_fn function.

        Args:
            test_data (Union[np.ndarray, Tensor]): Test data that needs to be processed.
                It can either be a Numpy array or a PyTorch Tensor.
            **kwargs: Additional arguments that may be required for postprocessing.

        Returns:
            np.ndarray: The processed scores computed from the input test data.
        """
        assert self._setup_flag, "setup() must be called before postprocess()"
        if isinstance(test_data, Tensor):
            test_data = test_data.cpu().numpy()
        test_scores = mahalanobis_postprocess(
            feats=test_data,
            class_mean=self.class_mean,
            precision=self.precision,
            num_classes=self.num_classes,
        )
        test_scores = self.flip_sign_fn(test_scores)
        return test_scores


class ViM(OodPostprocessor):
    """
    Virtual logit Matching (ViM) class.

    The ViM class is designed for post-processing in out-of-distribution detection tasks.
    It focuses on transforming input data using specific mathematical procedures to calculate
    scores for identifying in-distribution and out-of-distribution data. This class applies
    dimension reduction, covariance matrix analysis, and score adjustments based on empirical
    data. The class relies on proper configuration and expects relevant parameters during
    setup for effective functionality.

    Attributes:
        u (np.ndarray): A vector computed based on the pseudo-inverse of the weight matrix
            and the bias vector from the final linear layer.
        DIM (int): Dimension of the subspace used for the computation of null space.
        NS (np.ndarray): Null space vector representation calculated during the setup process.
        alpha (float): Scaling factor calculated from training logits and null space projections.
    """

    def __init__(
        self,
        method_name: str,
        flip_sign: bool,
        cfg: DictConfig = None,
    ):
        """
        Initializes the class with specified method configuration and parameters.

        Args:
            method_name (str): Name of the method or algorithm to initialize.
            flip_sign (bool): Indicates whether to multiply the scores by -1 to ensure ID scores are higher than OOD scores.
            cfg (DictConfig, optional): Configuration object containing additional settings.
        """
        super().__init__(method_name, flip_sign, cfg)
        self.u = None
        self.DIM = None
        self.NS = None
        self.alpha = None

    def setup(self, ind_train_data: np.ndarray, **kwargs):
        """
        Sets up the ViM model for specific data inputs and configurations. This
        method configures the model based on training and validation data,
        calculates necessary transformations, and determines thresholds for
        use with in-distribution and out-of-distribution detection.

        Args:
            ind_train_data (np.ndarray): In-distribution training data used to
                compute the empirical covariance and null space transformations.
            **kwargs: Additional keyword arguments required for setup. Must
                include:
                - "final_linear_layer_params": Dictionary containing the
                  "weight" and "bias" of the final linear layer in the model.
                - "train_logits": ndarray of logits corresponding to the
                  training data.
                - "valid_feats": Validation dataset features for null space
                  computations.
                - "valid_logits": ndarray of logits corresponding to the
                  validation data.
        """
        assert (
            "final_linear_layer_params" in kwargs
        ), "final_linear_layer_params must be provided for ViM"
        assert "train_logits" in kwargs, "train_logits must be provided for ViM"
        assert "valid_feats" in kwargs, "valid_feats must be provided for ViM"
        assert "valid_logits" in kwargs, "valid_logits must be provided for ViM"
        w, b = (
            kwargs["final_linear_layer_params"]["weight"],
            kwargs["final_linear_layer_params"]["bias"],
        )
        if isinstance(w, Tensor):
            w = w.numpy()
        if isinstance(b, Tensor):
            b = b.numpy()
        self.u = -np.matmul(np.linalg.pinv(w), b)
        # Dimension of the null space
        if ind_train_data.shape[-1] >= 2048:
            self.DIM = 1000
        elif ind_train_data.shape[-1] >= 768:
            self.DIM = 512
        else:
            self.DIM = ind_train_data.shape[-1] // 2

        # Setup the model
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(ind_train_data - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        self.NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[self.DIM :]]).T)
        vlogit_id_train = np.linalg.norm(np.matmul(ind_train_data - self.u, self.NS), axis=-1)
        self.alpha = kwargs["train_logits"].max(axis=-1).mean() / vlogit_id_train.mean()

        # Apply on valid set
        vlogit_id_val = (
            np.linalg.norm(np.matmul(kwargs["valid_feats"] - self.u, self.NS), axis=-1) * self.alpha
        )
        energy_id_val = logsumexp(kwargs["valid_logits"], axis=-1)
        score_id = -vlogit_id_val + energy_id_val
        ind_scores = {self.method_name: score_id}
        ind_scores = self.flip_sign_fn(ind_scores)
        self.set_threshold(ind_scores)

    def postprocess(self, test_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Postprocesses the input test data to compute a confidence score.

        The method calculates a scoring metric based on the input test data and additional
        logits provided. It converts PyTorch Tensors to NumPy arrays (if necessary), computes
        norms and energy statistics, and produces a score combining both components.

        Args:
            test_data (np.ndarray): The input test data, either in the form of a NumPy
                array or a PyTorch tensor that will be converted to NumPy.
            **kwargs: Additional keyword arguments, including "logits" which should
                provide the logits either as a PyTorch tensor or a NumPy array.

        Returns:
            np.ndarray: An array of scores computed based on the inputs and the internal
            configuration of the object.

        Raises:
            AssertionError: If the method is called before setup() has been successfully
                executed to configure the necessary internal state.
        """
        assert self._setup_flag, "setup() must be called before postprocess()"
        if isinstance(test_data, Tensor):
            test_data = test_data.cpu().numpy()
        if isinstance(kwargs["logits"], Tensor):
            kwargs["logits"] = kwargs["logits"].cpu().numpy()
        vlogit_test = np.linalg.norm(np.matmul(test_data - self.u, self.NS), axis=-1) * self.alpha
        energy_test = logsumexp(kwargs["logits"], axis=-1)
        score = -vlogit_test + energy_test
        return score


postprocessors_dict = {
    "KDE": KDELatentSpace,
    "MD": MDLatentSpace,
    "KNN": KNNLatentSpace,
    "cMD": cMDLatentSpace,
    "GMM": GMMLatentSpace,
    "energy": Energy,
    "gen": GEN,
    "ddu": DDU,
    "mahalanobis": Mahalanobis,
    "vim": ViM,
}

postprocessor_input_dict = {
    "KDE": ["latent_space_means"],
    "KNN": ["latent_space_means"],
    "GMM": ["latent_space_means"],
    "MD": ["latent_space_means"],
    "energy": ["logits"],
    "gen": ["logits"],
    "ddu": ["features"],
    "mahalanobis": ["features"],
    "vim": ["features", "logits"],
}
