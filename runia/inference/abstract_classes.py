# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya
from time import monotonic
from typing import List, Union, Dict

import torch
from numpy import ndarray
from omegaconf import DictConfig

from runia.baselines.from_precalculated import get_baselines_thresholds
from runia.feature_extraction.utils import Hook

__all__ = [
    "record_time",
    "Postprocessor",
    "OodPostprocessor",
    "InferenceModule",
    "ProbabilisticInferenceModule",
    "ObjectDetectionInference",
]


###########################################################################
# Time-measuring Inference module
###########################################################################
# Record time wrapper
def record_time(function):
    """
    Decorator to record the execution time

    Args:
        function: Function to decorate

    Returns:
        Function return and execution time
    """

    def wrap(*args, **kwargs):
        start_time = monotonic()
        function_return = function(*args, **kwargs)
        delta_t = monotonic() - start_time
        return function_return, delta_t

    return wrap


###########################################################################
# Post-processor base classes
###########################################################################
class Postprocessor:
    """
    A base class for implementing post-hoc OOD detection workflows.

    This class provides a structured way to set up postprocessing configurations and
    execute postprocessing functions on test data. It serves as a template for
    building specific postprocessing pipelines by subclassing and implementing the
    required methods. The setup method is intended for initializing configurations
    and settings, while the postprocess method handles the transformation or analysis
    of the input data.

    Attributes:
        cfg (DictConfig): An optional configuration object passed during initialization.
    """

    def __init__(self, cfg: DictConfig = None):
        """
        Initializes the class with the provided configuration and sets up the initial state.

        Args:
            cfg (DictConfig, optional): Configuration dictionary used for initializing the
                attributes of the class. Defaults to None.
        """
        self._setup_flag = False

    def setup(self, ind_train_data: ndarray, **kwargs) -> None:
        """
        Sets up the necessary configurations or operations for the current instance
        using the provided independent training data. This method is expected to be
        implemented by subclasses to define specific setup behavior.

        Args:
            ind_train_data: Independent training data used for setup operations.
            **kwargs: Additional keyword arguments that subclasses can utilize for
                extended setup configurations.
        """
        raise NotImplementedError

    def postprocess(self, test_data: ndarray, **kwargs) -> ndarray:
        """
        Post-processes the test data based on specific implementation. This method is intended
        to be overridden in subclasses to apply further manipulations, filtering, or adjustments
        to the raw test data input. Results are expected to be returned in a structured format.

        Args:
            test_data: Input test data as a multidimensional array to be processed.
            **kwargs: Additional keyword arguments that can optionally provide context or
                settings for the post-processing step.

        Returns:
            ndarray: Modified or processed test data in the same or different structured format.
        """
        raise NotImplementedError

    def __call__(self, test_data: ndarray, **kwargs) -> ndarray:
        """
        Processes the given data by applying post-processing methods.

        This method is designed to handle input test data, process it internally,
        and return the processed output. The specific operation of post-processing
        is defined within the `postprocess` method.

        Args:
            test_data (ndarray): The input test data that requires post-processing.
            **kwargs: Additional keyword arguments that can affect the behavior of
                the postprocessing method.

        Returns:
            ndarray: The processed data after applying the postprocessing method.
        """
        return self.postprocess(test_data, **kwargs)


class OodPostprocessor(Postprocessor):
    """Handles postprocessing for out-of-distribution (OOD) detection methods.

    This class is designed to allow the customization of OOD postprocessing
    behaviors such as score inversion and threshold configuration for specific
    methods. It builds upon the standard functionality of a base Postprocessor
    class and extends it with specific behaviors for OOD processing.

    Attributes:
        flip_sign (bool): Indicates whether the scores should be inverted
            (multiplied by -1).
        method_name (str): The name of the processing method to be used for
            handling OOD scores.
        threshold (Union[float, None]): The threshold value for the method.
            Typically used for OOD decision-making.
    """

    def __init__(self, method_name: str, flip_sign: bool, cfg: DictConfig = None):
        """
        Initializes the class with the method name, score inversion flag, and optional configuration.

        Args:
            method_name (str): The name of the method to be initialized.
            flip_sign (bool): A flag indicating whether the score should be multiplied by -1.
            cfg (DictConfig, optional): An optional configuration object. Defaults to None.
        """
        super().__init__(cfg)
        self.flip_sign = flip_sign
        self.method_name = method_name
        self.threshold: Union[float, None] = None

    def flip_sign_fn(
        self, scores: Union[Dict[str, ndarray], ndarray]
    ) -> Union[Dict[str, ndarray], ndarray]:
        """
        Flip the sign of the given scores based on the `flip_sign` class property. This method modifies
        the scores to be their negative values if the `flip_sign` flag is set to True. The input can be
        either a dictionary of numpy arrays or a single numpy array.

        Raises:
            ValueError: If the provided `scores` is neither a dictionary nor a numpy array.

        Args:
            scores: The input scores to potentially flip the sign of. It can be either a dictionary with
                string keys and numpy array values or a single numpy array.

        Returns:
            Union[Dict[str, ndarray], ndarray]: The scores after flipping their signs (if applicable).
            The returned structure matches the input structure.
        """
        if self.flip_sign:
            if isinstance(scores, dict):
                scores[self.method_name] = scores[self.method_name] * -1
            elif isinstance(scores, ndarray):
                scores = scores * -1
            else:
                raise ValueError("scores must be a dict or ndarray")
        return scores

    def set_threshold(self, ind_test_scores: Dict[str, ndarray]) -> None:
        """
        Updates the threshold value by calculating baseline thresholds using the
        provided individual test scores. Sets the setup flag to True once the
        threshold is updated.

        Args:
            ind_test_scores (Dict[str, ndarray]): A dictionary where keys are the
                names of individual test metrics and values are NumPy arrays
                containing the scores for the respective metrics.
        """
        self.threshold = get_baselines_thresholds(
            baselines_names=[self.method_name],
            baselines_scores_dict=ind_test_scores,
        )[self.method_name]
        self._setup_flag = True

    def setup(self, ind_train_data: ndarray, **kwargs) -> None:
        raise NotImplementedError

    def postprocess(self, test_data: ndarray, **kwargs) -> ndarray:
        raise NotImplementedError


###########################################################################
# Inference base class
###########################################################################
class InferenceModule:
    """
    Handles the process of model inference with uncertainty estimation

    Allows for abstract model interaction and result postprocessing. This
    class is designed to serve as a base class for specific implementations,
    requiring the subclass to implement key functionality such as the
    `get_score` method. It also manages device allocation for the model.

    Attributes:
        model: The machine learning model that is used for inference.
        postprocessor: A callable or processing class used to postprocess
            the results of the inference.
        device: The device (CPU or CUDA) where the model is allocated for
            inference.
    """

    def __init__(
        self,
        model,
        postprocessor,
    ):
        """
        Initializes the class with a model and postprocessor, and sets the device for the
        model to be used on either CUDA or CPU.

        Args:
            model: The model to be initialized. It should support being moved to a
                specific device (e.g., CUDA or CPU).
            postprocessor: The postprocessor to be used alongside the model.
        """
        self.model = model
        self.postprocessor = postprocessor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model.to(self.device)
        except AttributeError:
            pass

    def get_score(self, input_image, *args, **kwargs):
        """
        Computes a score for the given input image based on the implemented logic
        in a subclass.

        This method is intended to be overridden in subclasses to provide the
        specific scoring algorithm for the input image. If not implemented, it
        raises a NotImplementedError.

        Args:
            input_image: Represents the input image data for which the score needs
                to be computed. The specific type and structure of this parameter
                depend on the scoring algorithm implemented in the subclass.
            *args: Additional positional arguments that may be used in the scoring
                algorithm.
            **kwargs: Additional keyword arguments that may be used in the scoring
                algorithm.

        Raises:
            NotImplementedError: If the subclass does not implement this method,
                indicating scoring functionality is missing.
        """
        raise NotImplementedError


class ProbabilisticInferenceModule(InferenceModule):
    """
    Performs probabilistic inference utilizing a specified model and postprocessor.

    This class extends the functionality of the `InferenceModule` to enable
    probabilistic inference by incorporating parameters for dropblock probability,
    dropblock block size, and Monte Carlo (MC) sampling. It is designed to allow
    flexibility in model inference by enabling techniques such as MC Dropout.

    Attributes:
        drop_block_prob (float): The dropblock probability for MC dropout. To be used in feature maps.
        drop_block_size (int): The dropblock size for MC dropout. To be used in feature maps.
        mcd_samples_nro (int): The number of Monte Carlo samples to use for
            probabilistic inference.
    """

    def __init__(
        self,
        model,
        postprocessor,
        drop_block_prob: float,
        drop_block_size: int,
        mcd_samples_nro: int,
    ):
        """
        Initializes an instance of the class with the specified attributes. This constructor
        inherits from the parent class and initializes parameters related to drop block
        probability, size, and Monte Carlo dropout samples.

        Args:
            model: Model instance to be passed to the parent class initializer.
            postprocessor: Postprocessor object passed to the parent class.
            drop_block_prob: Probability value for Drop Block regularization.
            drop_block_size: Integer size of the Drop Block.
            mcd_samples_nro: Number of Monte Carlo Dropout samples to be used.
        """
        super().__init__(model, postprocessor)
        self.drop_block_prob = drop_block_prob
        self.drop_block_size = drop_block_size
        self.mcd_samples_nro = mcd_samples_nro


class ObjectDetectionInference(InferenceModule):
    """Module for object detection inference.

    This class is designed for performing object detection inference using a specified
    model, postprocessing techniques, and other configurations. It allows easy handling of
    different architectures and extraction methods, as well as supporting optional PCA transformation
    for dimensionality reduction.

    Attributes:
        architecture (str): Architecture type of the object detection model.
        rcnn_extraction_type (str): Extraction type for region-based convolutional neural network
            features, if applicable.
        hooked_layers (List[Hook]): List of layers hooked for model introspection during inference.
        pca_transform: Optional PCA transformation used for dimensionality reduction. The type is
            dependent on the PCA implementation/library used.
    """

    def __init__(
        self,
        model,
        postprocessor,
        architecture: str,
        hooked_layers: List[Hook],
        pca_transform=None,
        rcnn_extraction_type: str = None,
    ):
        """
        Initializes an instance of the class.

        This constructor sets up the class attributes and ensures proper initialization
        for functionality involving architecture definition, hooked layer management,
        PCA transformation, and RCNN extraction.

        Args:
            model: The model to be used with the instance.
            postprocessor: The postprocessor function or object to process outputs.
            architecture (str): The name or type of architecture being utilized.
            hooked_layers (List[Hook]): The list of hooked layers for feature extraction for latent space methods
            pca_transform: Optional PCA transformation to apply, if available.
            rcnn_extraction_type (str, optional): The type of RCNN extraction to use,
                if applicable.
        """
        super().__init__(model=model, postprocessor=postprocessor)
        self.architecture = architecture
        self.rcnn_extraction_type = rcnn_extraction_type
        self.hooked_layers = hooked_layers
        self.pca_transform = pca_transform
