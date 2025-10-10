# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez
#    Daniel Montoya
from typing import List, Tuple, Union, Any, Dict

import torch
from numpy import ndarray, ascontiguousarray
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

from runia.feature_extraction.utils import Hook
from runia.feature_extraction.abstract_classes import (
    ObjectDetectionExtractor,
    MCSamplerModule,
)
from runia.evaluation.entropy import get_dl_h_z

__all__ = [
    "BoxFeaturesExtractor",
    "BoxFeaturesExtractorAnomalyLoader",
]


class BoxFeaturesExtractor(ObjectDetectionExtractor):
    """
    Handles the extraction of features specific to object detection at the object level
    with support for various configurations and architectures.

    This class facilitates the process of extracting features from object detection models,
    performing sampling, handling intermediate layer outputs, and calculating noise entropies
    as needed. It ensures compatibility with supported architectures and provides flexibility
    for model-dependent inference tasks.

    Attributes:
        roi_output_sizes (Tuple[int]): ROIAlign parameter to fix the sizes of the output feature maps. Must
            match the number of hooked layers in certain configurations.
        roi_sampling_ratio (int): ROIAlign parameter used for sampling operations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        hooked_layers: List[Hook],
        device: torch.device,
        architecture: str,
        roi_output_sizes: Tuple[int],
        return_raw_predictions: bool = False,
        return_stds: bool = False,
        mcd_nro_samples: int = 1,
        hook_layer_output: bool = True,
        dropblock_probs: Union[float, List] = 0.0,
        dropblock_sizes: Union[int, List] = 0,
        rcnn_extraction_type: str = None,
        extract_noise_entropies: bool = False,
        roi_sampling_ratio: int = -1,
    ):
        """
        Initializes a latent features extractor for object-level Out-of-Distribution Object Detection
        with configurable extraction and sampling.

        This class supports multiple configurations for hooking intermediate layers,
        extracting features, and noise entropy calculations. It also performs assertions
        to ensure the compatibility of provided arguments with supported architectures and
        extraction types.

        Args:
            roi_output_sizes (Tuple[int]): ROIAlign parameter to fix the sizes of the output feature maps. This must
                match the number of hooked layers when hooking the output layers.
            roi_sampling_ratio (int, optional): ROIAlign parameter used for sampling operations. Defaults
                to -1.
        """
        super().__init__(
            model=model,
            hooked_layers=hooked_layers,
            device=device,
            return_raw_predictions=return_raw_predictions,
            return_stds=return_stds,
            mcd_nro_samples=mcd_nro_samples,
            hook_layer_output=hook_layer_output,
            dropblock_probs=dropblock_probs,
            dropblock_sizes=dropblock_sizes,
            architecture=architecture,
            rcnn_extraction_type=rcnn_extraction_type,
            extract_noise_entropies=extract_noise_entropies,
        )
        if not isinstance(roi_output_sizes, list):
            roi_output_sizes = list(roi_output_sizes)
        self.roi_output_sizes = roi_output_sizes
        self.roi_sampling_ratio = roi_sampling_ratio

        # RPN intermediate extraction is tricky, the architecture was modified to catch intermediate
        # representations by using a list stored created during the forward
        # Also the backbone and the RPN final, output a dictionary of five tensors
        if self.architecture == "rcnn" and self.rcnn_extraction_type != "shortcut":
            self.roi_output_sizes = self.roi_output_sizes * 5
            self.n_hooked_reps = 5

    def get_ls_samples(
        self, data_loader: Union[DataLoader, Any], predict_conf: float = 0.25, **kwargs
    ) -> Dict:
        """
        Extracts latent space activations for detected objects from the images in the given data loader.
        Uses the ROI Align algorithm to achieve this.

        This method iterates over the items in a provided data loader, processes images
        to extract latent space representations, and organizes the results into a
        dictionary. It includes various features such as tracking progress via a progress
        bar, handling batches without detected objects, and optionally including standard
        deviations in the results.

        Args:
            data_loader (Union[DataLoader, Any]): The data loader providing images and associated data.
            predict_conf (float): Confidence threshold for predictions. Defaults to 0.25.
            **kwargs: Additional arguments passed to the image processing method.

        Returns:
            Dict: A dictionary containing latent space results per image. The results include:
                - 'means', 'features', 'logits', 'boxes' for detected objects.
                - 'no_obj' listing images with no detected objects.

        Raises:
            ValueError: If the data loader batch size is not 1, as the method assumes
                individual image processing.
        """
        # Check data loader batch size is 1
        self.check_dataloader(data_loader)
        results = {}
        no_obj_imgs = []
        if self.return_stds:
            results["stds"] = []
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting latent space box samples") as pbar:
                # for impath, image, im_counter in data_loader:
                for loader_contents in data_loader:
                    impath, image, im_id = self.unpack_dataloader(loader_contents)
                    result_img, found_obj_flag = self._get_samples_one_image(
                        image=image, predict_conf=predict_conf, **kwargs
                    )
                    results[im_id] = {
                        "latent_space_means": [],
                        "features": [],
                        "logits": [],
                        "boxes": [],
                    }
                    if found_obj_flag:
                        for result_type, result_value in result_img.items():
                            results[im_id][result_type].append(result_value)
                    else:
                        # impath is a list, with batch size 1 we only need the first element (the string)
                        no_obj_imgs.append(impath[0])
                    # Update progress bar
                    pbar.update(1)
                for im_id in results.keys():
                    for result_type, result_value in results[im_id].items():
                        results[im_id][result_type] = (
                            torch.cat(result_value, dim=0)
                            if len(result_value) > 0
                            else result_value
                        )
        results["no_obj"] = no_obj_imgs
        # print("Latent representation vector size: ", results["latent_space_means"].shape[1])
        print(f"No objects in {len(no_obj_imgs)} images")
        return results

    def _get_samples_one_image(
        self, image: Union[Tensor, ndarray], predict_conf: float, **kwargs
    ) -> Tuple[Dict[str, Tensor], bool]:
        """
        Processes an image using different object detection architectures and extracts
        features, logits, and bounding boxes for detected objects. Supports YOLO, RCNN,
        OWL-V2, DETR, RT-DETR architectures, and allows handling outputs and latent
        features specific to each.

        Args:
            image: Input image in the form of a tensor or numpy array whose format
                depends on the detection architecture being used.
            predict_conf: Confidence threshold to filter predictions. Controls which
                predictions are retained by applying a confidence score filter.
            **kwargs: Additional keyword arguments passed to the specific detection
                model's forward or inference method.

        Returns:
            tuple: A tuple containing a dictionary (`results`) with extracted
            information including features, logits, and bounding boxes, and a
            boolean flag indicating whether objects were detected (`found_objs_flag`).
            The `results` dictionary can include:
                - "features": Extracted latent features, if available.
                - "logits": Logits from the detection model's output or intermediate
                  layers.
                - "boxes": Bounding boxes for the detected objects in (x1, y1, x2, y2)
                  format.
                - "latent_space_means": ROIs-related feature means.
                - "stds": ROIs-related feature standard deviations (if enabled).
                - "raw_preds": Raw predictions from the detection model (if enabled).
        """
        # Found objects flag
        found_objs_flag = True
        # Perform inference
        results, boxes, pred_img, img_shape = self.model_dependent_inference(
            image, predict_conf, **kwargs
        )
        # Check if objects were detected
        n_detected_objects = boxes.shape[0]
        if n_detected_objects == 0:
            # Get whole image as single object if no objects are detected
            boxes = Tensor([0.0, 0.0, img_shape[1], img_shape[0]]).reshape(1, -1).to(self.device)
            n_detected_objects = 1
            found_objs_flag = False
        # Catch the latent activations
        latent_sample = self.model_dependent_feature_extraction()
        if len(latent_sample) > 0:
            # Deterministic algorithm
            if not self.extract_noise_entropies:
                n_objects_means, n_objects_stds = _reduce_features_to_rois(
                    latent_mcd_sample=latent_sample,
                    output_sizes=self.roi_output_sizes,
                    boxes=boxes,
                    img_shape=img_shape,
                    sampling_ratio=self.roi_sampling_ratio,
                    n_hooked_reps=self.n_hooked_reps,
                    n_detected_objects=n_detected_objects,
                    return_stds=self.return_stds,
                )
                results["latent_space_means"] = torch.cat(n_objects_means, dim=0)
            else:
                # MCD sampling-based algorithm
                n_objects_means = _dropblock_rois_get_entropy(
                    latent_mcd_sample=latent_sample,
                    output_sizes=self.roi_output_sizes,
                    boxes=boxes,
                    img_shape=img_shape,
                    sampling_ratio=self.roi_sampling_ratio,
                    n_hooked_reps=self.n_hooked_reps,
                    n_mcd_steps=self.mcd_nro_samples,
                    mc_sampler=self.mc_sampler,
                )
                results["latent_space_means"] = n_objects_means
        else:
            results["latent_space_means"] = []
        results["boxes"] = boxes
        if self.return_stds:
            results["stds"] = torch.cat(n_objects_stds, dim=0)
        if self.return_raw_predictions:
            results["raw_preds"] = pred_img
        return results, found_objs_flag


class BoxFeaturesExtractorAnomalyLoader(BoxFeaturesExtractor):
    """
    Used to extract latent features of a yolo model with a dataloader that generates anomalies on the fly
    like blur, brightness, and fog.
    """

    def get_ls_samples(
        self, data_loader: Union[DataLoader, Any], predict_conf=0.25, **kwargs
    ) -> Dict:
        results = {"latent_space_means": []}
        no_obj_imgs = 0
        if self.return_stds:
            results["stds"] = []
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting latent space box samples") as pbar:
                for image, label in data_loader:
                    # Here, a BGR 2 RGB inversion is performed, since the torch Dataloader seems to feed yolo
                    # Images in the wrong ordering
                    image = [ascontiguousarray(image[0].numpy().transpose(1, 2, 0)[..., ::-1])]
                    result_img, found_obj_flag = self._get_samples_one_image(
                        image=image, predict_conf=predict_conf
                    )
                    for result_type, result_value in result_img.items():
                        results[result_type].append(result_value)
                    if not found_obj_flag:
                        # impath is a list, with batch size 1 we only need the first element (the string)
                        no_obj_imgs += 1
                    # Update progress bar
                    pbar.update(1)
                for result_type, result_value in results.items():
                    results[result_type] = torch.cat(result_value, dim=0)
        results["no_obj"] = no_obj_imgs
        print("Latent representation vector size: ", results["latent_space_means"].shape[1])
        print(f"No objects in {no_obj_imgs} images")
        return results


def _reduce_features_to_rois(
    latent_mcd_sample: List[Tensor],
    output_sizes: Tuple[int],
    boxes: Tensor,
    img_shape: Tuple[int, ...],
    sampling_ratio: int,
    n_hooked_reps: int,
    n_detected_objects: int,
    return_stds: bool = False,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    This function takes as input the bounding boxes predictions, the latent representations, and the output sizes
    to obtain the means and optionally the standard deviations of these representations

    Args:
        latent_mcd_sample: The latent representation samples
        output_sizes: Tuple of output sizes as integers
        boxes: Box predictions in the format xyxy
        img_shape: Tuple with the image shape
        sampling_ratio: Roi Align sampling ratio
        n_hooked_reps: Number of hooked latent representations
        n_detected_objects: Number of detected objects in the image
        return_stds: Whether to return standard deviations

    Returns:
        A tuple with the means and standard deviations of rois
    """
    # Extract latent space ROIs
    # Several representations per layer
    rois = [
        roi_align(
            latent_mcd_sample[i],
            [boxes],
            output_size=output_sizes[i],
            spatial_scale=latent_mcd_sample[i].shape[3] / img_shape[1],
            sampling_ratio=sampling_ratio,
            aligned=True,
        )
        for i in range(n_hooked_reps)
    ]
    # Get means and optionally stds from rois
    n_objects_means = []
    n_objects_stds = []

    for i in range(n_detected_objects):
        object_i_latent_means = []
        object_i_latent_stds = []
        for j in range(n_hooked_reps):
            object_i_latent_means.append(torch.mean(rois[j][i], dim=(1, 2)).reshape(-1))
            if return_stds:
                object_i_latent_stds.append(torch.std(rois[j][i], dim=(1, 2)).reshape(-1))
        n_objects_means.append(torch.cat(object_i_latent_means, dim=0).reshape(1, -1))
        if return_stds:
            n_objects_stds.append(torch.cat(object_i_latent_stds, dim=0).reshape(1, -1))

    return n_objects_means, n_objects_stds


def _dropblock_rois_get_entropy(
    latent_mcd_sample: List[Tensor],
    output_sizes: Tuple[int],
    boxes: Tensor,
    img_shape: Tuple[int, ...],
    sampling_ratio: int,
    n_hooked_reps: int,
    n_mcd_steps: int,
    mc_sampler: MCSamplerModule,
) -> Tensor:
    """
    This function takes as input the bounding boxes predictions, the latent representations, the output sizes,
    the number of Monte Carlo Dropblock repetitions, the dropblock size, and dropblock probability,
    to calculate the entropy of the means of the activations

    Args:
        latent_mcd_sample: The latent representation samples
        output_sizes: Tuple of output sizes as integers
        boxes: Box predictions in the format xyxy
        img_shape: Tuple with the image shape
        sampling_ratio: Roi Align sampling ratio
        n_hooked_reps: Number of hooked latent representations
        n_mcd_steps: Number of dropblock noising steps
        mc_sampler: Monte Carlo dropblock sampler class

    Returns:
        The entropy of the means of the noised activations
    """
    # Extract latent space ROIs
    # Several representations per layer
    rois = [
        roi_align(
            latent_mcd_sample[i],
            [boxes],
            output_size=output_sizes[i],
            spatial_scale=latent_mcd_sample[i].shape[3] / img_shape[1],
            sampling_ratio=sampling_ratio,
            aligned=True,
        )
        for i in range(n_hooked_reps)
    ]
    if len(rois) > 1:
        rois = torch.cat(rois, dim=1)
    else:
        rois = rois[0]
    # Obtain noised representations
    all_noised_objects = []
    for detection in rois:
        all_noised_objects.append(mc_sampler(detection.unsqueeze(0)))
    # mc_samples = mc_sampler(rois)
    all_noised_objects = torch.cat(all_noised_objects, dim=0)
    # Get entropies
    _, entropies = get_dl_h_z(all_noised_objects, mcd_samples_nro=n_mcd_steps, parallel_run=True)
    entropies = Tensor(entropies)

    return entropies
