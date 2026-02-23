# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya
from typing import Tuple, Union, List, Any

import numpy as np
import torch
from torch import Tensor

from runia_core.inference.abstract_classes import (
    InferenceModule,
    record_time,
    ObjectDetectionInference,
)
import runia_core.feature_extraction.object_level
from runia_core.feature_extraction.utils import Hook
from runia_core.dimensionality_reduction import apply_pca_ds_split, apply_pca_transform
from runia_core.import_helper_functions import module_exists
from runia_core.inference.postprocessors import postprocessors_dict

if module_exists("ultralytics"):
    from ultralytics.engine.results import Boxes

__all__ = ["BoxInferenceYolo", "ObjectLevelInference"]


class BoxInferenceYolo(InferenceModule):
    """
    Class intended to perform inference on new data using the LaREx methods (either LaRED or LaREM).
    LaRD performs representation reduction and density calculations to return a confidence score.
    This method does not perform zMCD sampling nor entropy calculations.
    It assumes that an optional PCA reducer plus LaRED or LaREM postprocessors are already trained.
    It can also perform testing of inference time.

    Args:
            model: Trained model
            postprocessor_type: String stating either 'LaRED', 'LaREM' or 'LaREK'
            ind_samples: Numpy ndarray with the training InD latent space samples
            n_pca_components: Optionally the number of PCA components to use for dimension reduction.
                Default: None if not using PCA reduction
    """

    def __init__(
        self,
        model,
        postprocessor,
        postprocessor_type: str,
        ind_samples: np.ndarray,
        roi_output_sizes: Tuple[int],
        roi_sampling_ratio: int = -1,
        n_pca_components=None,
    ):
        """
        Class intended to perform inference on new data using the LaREx methods (either LaRED or
        LaREM). LaRD performs representation reduction and density calculations to return a
        confidence score. This method does not perform zMCD sampling nor entropy calculations.
        It assumes that an optional PCA reducer plus LaRED or LaREM postprocessors are already
        trained. It can also perform testing of inference time.

        Args:
            model: Trained model
            postprocessor_type: String stating either 'LaRED', 'LaREM' or 'LaREK'
            ind_samples: Numpy ndarray with the training InD latent space samples
            n_pca_components: Optionally the number of PCA components to use for dimension reduction.
                Default: None if not using PCA reduction
        """
        super().__init__(model, postprocessor)
        assert (
            postprocessor_type in postprocessors_dict.keys()
        ), f"postprocessor_type must be one of {postprocessors_dict.keys()}"
        # Optionally do PCA reduction
        self.pca_transformation = None
        if n_pca_components:
            self.pca_components = n_pca_components
            # Train PCA module and transform InD train samples
            ind_samples, self.pca_transformation = apply_pca_ds_split(
                samples=ind_samples, nro_components=n_pca_components
            )

        # Setup postprocessor with the features
        self.postprocessor = postprocessors_dict[postprocessor_type]
        self.postprocessor.setup(ind_samples)
        # Roi parameters
        self.roi_output_sizes = roi_output_sizes
        self.roi_sampling_ratio = roi_sampling_ratio

    def get_score(
        self,
        input_image: Union[List[Tensor], List[np.ndarray], List[str]],
        confidence_score: float,
        layer_hook: List[Hook],
        threshold: float,
        use_stds: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Compute LaRx score for a single image. Batch size must be 1.

        Args:
            input_image: New image, a list with only one image in tensor, numpy array, os string with path format
            confidence_score: Confidence score for object detection inference.
            layer_hook: List with the hooked layers.
            threshold: Binary classification threshold previously calculated to classify InD or OoD
            use_stds: Whether to use latent space standard deviations. Default: False

        Returns:
            Network prediction with updated classes in case of detection of OoD objects.
        """
        assert len(input_image) == 1, "Only batch 1 is supported"

        detected_objects_flag = True
        with torch.no_grad():
            try:
                input_image = input_image.to(self.device)
            except AttributeError:
                pass
            output = self.model(input_image, conf=confidence_score, **kwargs)
            img_shape = output[0].orig_shape  # Height, width
            # img_shape = input_image[0].shape[:2]  # Height, width
            # Get detected boxes
            boxes = output[0].boxes.xyxy
            n_detected_objects = boxes.shape[0]
            # If not detected object according to threshold, take the whole image as region of interest
            if n_detected_objects == 0:
                boxes = (
                    Tensor([0.0, 0.0, img_shape[1], img_shape[0]]).reshape(1, -1).to(self.device)
                )
                n_detected_objects = 1
                detected_objects_flag = False

            latent_rep = [layer.output for layer in layer_hook]  # latent representation sample
            # Get ROIs
            latent_rep_means, latent_rep_stds = (
                runia_core.feature_extraction.object_level._reduce_features_to_rois(
                    latent_mcd_sample=latent_rep,
                    output_sizes=self.roi_output_sizes,
                    boxes=boxes,
                    img_shape=img_shape,
                    sampling_ratio=self.roi_sampling_ratio,
                    n_hooked_reps=len(self.roi_output_sizes),
                    n_detected_objects=n_detected_objects,
                )
            )
            latent_rep = torch.cat(latent_rep_means, dim=0)
            if use_stds:
                latent_rep_stds = torch.cat(latent_rep_stds, dim=1)
                latent_rep = torch.cat((latent_rep, latent_rep_stds), dim=1)
            # Convert to numpy array
            latent_rep = latent_rep.cpu().numpy()
            if self.pca_transformation:
                latent_rep = apply_pca_transform(latent_rep, self.pca_transformation)
            # Add OoD class to results
            if "OoD" not in output[0].names.values():
                output[0].names[len(output[0].names)] = "OOD"
            # Evaluate each detected object
            objects_to_update, objects_ood_scores = self.postprocess_detected_objects(
                latent_rep=latent_rep,
                threshold=threshold,
                detected_obj_flag=detected_objects_flag,
                boxes=boxes,
                output=output,
                img_shape=img_shape,
                conf_score=confidence_score,
            )
            # Update boxes if necessary
            if len(objects_to_update) > 0:
                output[0].boxes = Boxes(torch.cat(objects_to_update, dim=0), orig_shape=img_shape)
            # Add OoD scores to Results boxes
            output[0].boxes.ood_scores = objects_ood_scores
        return output

    def postprocess_detected_objects(
        self,
        latent_rep: np.ndarray,
        threshold: float,
        detected_obj_flag: bool,
        boxes: Tensor,
        output: Tensor,
        img_shape: Tuple[int, int],
        conf_score: float,
    ):
        """
        Processes detected objects against a threshold for classification as in-distribution
        (InD) or out-of-distribution (OoD). This method performs postprocessing on latent
        representations and updates the detected object information based on the classification.
        The method returns the updated objects and their corresponding OoD scores.

        Args:
            latent_rep (np.ndarray): Latent representation of detected objects.
            threshold (float): Threshold value for classifying objects as InD or OoD.
            detected_obj_flag (bool): Flag indicating whether objects are already detected.
            boxes (Tensor): Bounding boxes for detected objects.
            output (Tensor): Output from the detection model containing confidence scores
                and object class information.
            img_shape (Tuple[int, int]): Shape of the input image (height, width).
            conf_score (float): Confidence score to assign for objects classified as OoD.

        Returns:
            Tuple[List[Tensor], List[float]]: A tuple containing a list of tensors representing
            updated objects and a list of OoD scores for the objects.
        """
        objects_to_update = []
        objects_ood_scores = []
        for i, found_object_latent_rep in enumerate(latent_rep):
            sample_score = self.postprocessor.postprocess(found_object_latent_rep.reshape(1, -1))
            objects_ood_scores.append(sample_score)
            # Here is where classification in InD or OoD takes place
            if sample_score < threshold:
                if detected_obj_flag:
                    objects_to_update.append(
                        Tensor(
                            [
                                boxes[i][0].item(),
                                boxes[i][1].item(),
                                boxes[i][2].item(),
                                boxes[i][3].item(),
                                output[0].boxes.conf[i],
                                len(output[0].names) - 1,
                            ]
                        )
                        .reshape(1, -1)
                        .to(self.device),
                    )
                else:
                    objects_to_update.append(
                        Tensor(
                            [
                                0.0,
                                0.0,
                                img_shape[1],
                                img_shape[0],
                                conf_score,
                                len(output[0].names) - 1,
                            ]
                        )
                        .reshape(1, -1)
                        .to(self.device),
                    )
            # InD object detected
            else:
                if detected_obj_flag:
                    objects_to_update.append(
                        Tensor(
                            [
                                boxes[i][0].item(),
                                boxes[i][1].item(),
                                boxes[i][2].item(),
                                boxes[i][3].item(),
                                output[0].boxes.conf[i],
                                output[0].boxes.cls[i],
                            ]
                        )
                        .reshape(1, -1)
                        .to(self.device),
                    )
        return objects_to_update, objects_ood_scores

    @record_time
    def test_time_inference(self, **kwargs):
        """
        Call the inference function and get the execution time

        Args:
            input_image: Input image
            layer_hook: Hooked layer

        Returns:
            LaREx inference results plus execution time
        """
        return self.get_score(**kwargs)


class ObjectLevelInference(ObjectDetectionInference):
    """Class for object-level inference for object detection models.

    This class extends the functionality of object detection inference by providing
    methods to compute confidence scores and adjust predictions based on various
    parameters. It also utilizes feature extraction through the integration
    of BoxFeaturesExtractor.

    Attributes:
        latent_space_method (bool): Indicates whether a latent space method is used
            for extracting feature representations.
        postprocessor_input (List[str]): List of keys to access specific data in
            inference results for postprocessing.
        features_extractor (BoxFeaturesExtractor): Instance of BoxFeaturesExtractor
            used to handle feature extraction and predictions.
    """

    def __init__(
        self,
        model,
        postprocessor,
        architecture: str,
        latent_space_method: bool,
        hooked_layers: List[Hook],
        postprocessor_input: List[str],
        roi_output_sizes: Tuple[int],
        roi_sampling_ratio: int = -1,
        pca_transform=None,
        rcnn_extraction_type: str = None,
    ):
        """
        Initializes the class with the given model, postprocessor, architecture, latent space
        method, hooked layers, postprocessor input, and other optional parameters.

        The initializer sets up the feature extractor to enable inference and extraction
        functionality based on the given model and specifications.

        Args:
            model: The PyTorch model used for inference and feature extraction.
            postprocessor: A postprocessor used for transforming model outputs into
                interpretable results.
            architecture: The architecture of the model, usually represented as a string.
            latent_space_method: A boolean indicating if a latent space method is used.
            hooked_layers: A list of Hook objects representing the layers to hook for
                feature extraction.
            postprocessor_input: A list of strings specifying the input names required by
                the postprocessor.
            roi_output_sizes: A tuple of integers indicating the Region of Interest (ROI)
                output sizes.
            roi_sampling_ratio: An integer specifying the sampling ratio for ROI (default
                is -1 if not provided).
            pca_transform: Optional; a PCA transform to reduce dimensionality of the
                extracted features.
            rcnn_extraction_type: Optional; a string representing the type of RCNN
                extraction being used, if applicable.
        """
        super().__init__(
            model=model,
            postprocessor=postprocessor,
            architecture=architecture,
            hooked_layers=hooked_layers,
            rcnn_extraction_type=rcnn_extraction_type,
            pca_transform=pca_transform,
        )
        self.latent_space_method = latent_space_method
        self.postprocessor_input = postprocessor_input

        # Instantiate feature extractor, which has already implemented inference + extraction
        self.features_extractor = runia_core.feature_extraction.object_level.BoxFeaturesExtractor(
            model=self.model,
            device=self.device,
            hooked_layers=self.hooked_layers,
            architecture=self.architecture,
            rcnn_extraction_type=self.rcnn_extraction_type,
            roi_output_sizes=roi_output_sizes,
            roi_sampling_ratio=roi_sampling_ratio,
            return_raw_predictions=True,
        )

    def get_score(self, input_image, predict_conf, **kwargs):
        """
        Generates scores for a given input image by extracting features,
        transforming latent space if applicable, and applying post-processing
        to compute confidence scores.

        Args:
            input_image: Input image to be processed.
            predict_conf: Configuration settings used during the prediction phase.
            **kwargs: Additional arguments that may be passed to the underlying
                feature extraction process.

        Returns:
            Tuple containing raw predictions from inference results and confidence
            scores after post-processing.

        """
        with torch.no_grad():
            inference_results, found_objects_flag = self.features_extractor._get_samples_one_image(
                input_image, predict_conf, **kwargs
            )
            if self.latent_space_method:
                inference_results["latent_space_means"] = (
                    inference_results["latent_space_means"].cpu().numpy()
                )
        if self.pca_transform:
            inference_results["latent_space_means"] = apply_pca_transform(
                inference_results["latent_space_means"], self.pca_transform
            )
        if found_objects_flag:
            if len(self.postprocessor_input) == 1:
                confidence_scores = self.postprocessor.postprocess(
                    inference_results[self.postprocessor_input[0]]
                )
            else:
                confidence_scores = self.postprocessor.postprocess(
                    test_data=inference_results[self.postprocessor_input[0]],
                    logits=inference_results[self.postprocessor_input[1]],
                )
        else:
            confidence_scores = []

        return inference_results["raw_preds"], confidence_scores

    def adjust_predictions_faster_rcnn(
        self, predictions: Any, scores: np.ndarray, ood_class_number: int, **kwargs
    ) -> Any:
        """
        Adjusts the prediction labels for a Faster R-CNN model based on a score threshold.

        This method modifies the predictions by setting the label of predictions below
        a specified score threshold to the out-of-distribution (OOD) class number. This
        is useful for filtering predictions that are considered uncertain or out-of-distribution
        according to the specified threshold.

        Args:
            predictions: Predictions object to be modified. The exact type and structure
                of this object depend on the specific implementation of the Faster R-CNN
                model.
            scores: Array of confidence scores corresponding to the predictions. These
                scores determine if a prediction should be marked as OOD based on the
                threshold.
            ood_class_number: Integer representing the label or class number used to
                identify out-of-distribution predictions.
            **kwargs: Additional keyword arguments that may be required for specific
                configurations or extensions of the processing logic.

        Returns:
            Predictions: The modified predictions object with adjusted labels for
            out-of-distribution predictions.
        """
        for i, score in enumerate(scores):
            if score < self.postprocessor.threshold:
                predictions.det_labels[i] = ood_class_number
        return predictions
