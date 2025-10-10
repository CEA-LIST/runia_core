from typing import List, Union, Any, Tuple, Dict

import torch
from dropblock import DropBlock2D
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import nms

from runia.feature_extraction.utils import (
    Hook,
    get_mean_or_fullmean_ls_sample,
)

SUPPORTED_OBJECT_DETECTION_ARCHITECTURES = [
    "yolov8",
    "rcnn",
    "detr-backbone",
    "owlv2",
    "rtdetr-backbone",
    "rtdetr-encoder",
]

__all__ = [
    "Extractor",
    "ObjectDetectionExtractor",
    "MCSamplerModule",
    "SUPPORTED_OBJECT_DETECTION_ARCHITECTURES",
]


class MCSamplerModule(torch.nn.Module):
    """
    Module for Monte Carlo (MC) sampling from a trained model.

    This class applies Monte Carlo Dropout (MCD) sampling to a given latent
    representation using DropBlock2D layers. It is designed to sample
    noised and reduced activations from the input, which can aid in
    quantifying uncertainty in predictions. Depending on the specified
    layer type, it processes the activations accordingly for convolutional
    or fully connected layers.

    Attributes:
        layer_type (str): Specifies the type of layer, either "Conv" or "FC".
        mc_samples (int): Number of Monte Carlo samples to be taken.
        drop_blocks (torch.nn.ModuleList): A list of DropBlock2D layers used for
            adding stochastic noise to the input.
    """

    def __init__(
        self,
        mc_samples: int,
        block_size: int,
        drop_prob: float,
        layer_type: str = "Conv",
    ):
        """
        Initializes MCSamplerModule.

        This class leverages the ability to integrate multiple `DropBlock2D` modules
        while maintaining the consistency specified by the chosen parameters.

        Args:
            mc_samples: Number of Monte Carlo Dropout (MCD) samples to take
            block_size: Size of Dropblock
            drop_prob: Dropblock probability
            layer_type: Either 'Conv' or 'FC'
        """
        super(MCSamplerModule, self).__init__()
        assert layer_type in ("Conv", "FC", "RPN")
        self.layer_type = layer_type
        self.mc_samples = mc_samples
        self.drop_blocks = torch.nn.ModuleList(
            [
                DropBlock2D(block_size=block_size, drop_prob=drop_prob)
                for _ in range(self.mc_samples)
            ]
        )

    def forward(self, latent_rep):
        """
        Apply the MCD sampling module to a latent representation.

        Args:
            latent_rep: Latent representation

        Returns:
            Noised and reduced activations
        """
        samples = []
        for drop_layer in self.drop_blocks:
            mc_sample = drop_layer(latent_rep)

            if self.layer_type == "Conv":
                # Get image HxW mean:
                mc_sample = get_mean_or_fullmean_ls_sample(mc_sample, method="fullmean")

            samples.append(mc_sample.reshape(1, -1))
        samples_t = torch.cat(samples)
        return samples_t


class Extractor:
    """
    Handles the extraction of latent space activations and features from a given model for
    various scenarios.

    This class is designed to facilitate data extraction from machine learning
    models by applying hooks to specified layers. It allows for customization of
    predictions, sampling, and additional processing, such as DropBlock Monte Carlo sampling.

    Attributes:
        model (torch.nn.Module): The model from which features and predictions
            will be extracted.
        hooked_layers (List[Hook]): List of layers where hooks are applied to
            capture intermediate outputs.
        device (torch.device): The device used to perform computations (e.g.,
            CPU or GPU).
        return_raw_predictions (bool): Flag to indicate if raw predictions from
            the model should be returned.
        hook_layer_output (bool): Indicates whether the output of the hooked
            layers should be recorded.
        return_stds (bool): Specifies if standard deviations of the latent
            representations should be computed and returned.
        mcd_nro_samples (int): Number of Monte Carlo Dropout samples to be used
            for uncertainty estimation.
        dropblock_probs (Union[float, List]): Probability/probabilities for the
            DropBlock regularization when applied to the model.
        dropblock_sizes (Union[int, List]): Block size(s) for the DropBlock
            sampling when applied.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        hooked_layers: List[Hook],
        device: torch.device,
        return_raw_predictions: bool = False,
        return_stds: bool = False,
        mcd_nro_samples: int = 1,
        hook_layer_output: bool = True,
        dropblock_probs: Union[float, List] = 0.0,
        dropblock_sizes: Union[int, List] = 0,
    ):
        """
        Initializes the class with the given parameters and configurations. This constructor sets up the
        model, hooked layers, device, and additional Monte Carlo Dropout (MCD) settings, enabling the user
        to specify the behavior and functionality of the model's forward passes.

        Args:
            model (torch.nn.Module): The model from which features and predictions
                will be extracted.
            hooked_layers (List[Hook]): List of layers where hooks are applied to
                capture intermediate outputs.
            device (torch.device): The device used to perform computations (e.g.,
                CPU or GPU).
            return_raw_predictions (bool): Flag to indicate if raw predictions from
                the model should be returned.
            hook_layer_output (bool): Indicates whether the output of the hooked
                layers should be recorded.
            return_stds (bool): Specifies if standard deviations of the latent
                representations should be computed and returned.
            mcd_nro_samples (int): Number of Monte Carlo Dropout samples to be used
                for uncertainty estimation.
            dropblock_probs (Union[float, List]): Probability/probabilities for the
                DropBlock regularization when applied to the model.
            dropblock_sizes (Union[int, List]): Block size(s) for the DropBlock
                sampling when applied.
        """
        self.model = model
        self.mcd_nro_samples = mcd_nro_samples
        self.hooked_layers = hooked_layers
        self.device = device
        self.return_raw_predictions = return_raw_predictions
        self.hook_layer_output = hook_layer_output
        self.return_stds = return_stds
        self.dropblock_sizes = dropblock_sizes
        self.dropblock_probs = dropblock_probs

    def get_ls_samples(self, data_loader, **kwargs):
        """
        Fetches labeled samples from the provided data loader using additional parameters.

        This method is designed to fetch labeled samples based on the given data loader
        and any additional arguments. The specific implementation of how the labeled
        samples are retrieved is not provided, as the method is intentionally abstract
        and should be overridden in a subclass.

        Args:
            data_loader: A data loader object from which labeled samples are to be
                fetched. The exact nature of the data loader is determined by the
                implementation.
            **kwargs: Additional keyword arguments necessary for the retrieval of
                labeled samples. The specific arguments and their usage are dependent
                on the subclass implementation.

        Raises:
            NotImplementedError: This method is abstract and must be implemented by
                subclasses. Attempting to invoke it directly without overriding will
                result in this error.
        """
        raise NotImplementedError

    def _get_samples_one_image(self, image, **kwargs):
        """
        Processes a single image to extract samples for further analysis or
        computation. This method is abstract and must be implemented by subclasses
        to define specific behavior.

        Args:
            image: The input image to process.
            **kwargs: Arbitrary keyword arguments to customize the behavior of
                the implementation.
        """
        raise NotImplementedError

    @staticmethod
    def check_dataloader(data_loader: Union[DataLoader, Any]) -> None:
        """
        Ensures that the provided data loader operates with a batch size of 1. This function
        validates the batch size based on specific attributes (`batch_sampler`, `batch_size`,
        or `bs`) expected to be present in the `data_loader` object. If none of these
        attributes exist or the batch size does not meet the required condition, an
        appropriate exception is raised.

        Args:
            data_loader: The data loader object to validate. It should either be an instance
                of `DataLoader` or any class that has one of the specified attributes.

        Raises:
            AssertionError: If the batch size is not 1.
            AttributeError: If the `data_loader` lacks any recognized batch size attributes.
        """
        if hasattr(data_loader, "batch_sampler"):
            assert data_loader.batch_sampler.batch_size == 1, "Only batch size 1 is supported"
        elif hasattr(data_loader, "batch_size"):
            assert data_loader.batch_size == 1, "Only batch size 1 is supported"
        elif hasattr(data_loader, "bs"):
            assert data_loader.bs == 1, "Only batch size 1 is supported"
        else:
            raise AttributeError(
                "Data loader must have attribute batch size and should be equal to 1"
            )


class ObjectDetectionExtractor(Extractor):
    """
    Handles object detection feature extraction for various model architectures. This class
    specializes in setting up the required extraction workflow including hooking relevant
    representations within the model, configuring backbones, and managing architecture-specific
    processes for obtaining predictions and intermediate features.

    The class supports architectures like YOLO, RCNN, OWLv2, and others, offering the ability
    to handle object detection pipelines and manage nuanced behaviors like Region Proposal
    Networks (RPNs) and intermediate representation extraction.

    Attributes:
        architecture (str): The specific object detection model architecture being used.
            Supported options are defined in `SUPPORTED_OBJECT_DETECTION_ARCHITECTURES`.
        rcnn_extraction_type (str): Defines the mode of extracting intermediate data from RCNN
            models. Options include "rpn_inter", "rpn_head", "shortcut", "backbone", or None.
        n_hooked_reps (int): Number of hooked representations or layers being observed during
            the workflow.
        extract_noise_entropies (bool): Determines whether Monte Carlo (MC) sampling is applied
            to extract noise entropies. If True, an MC sampler is initialized.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        hooked_layers: List[Hook],
        device: torch.device,
        architecture: str,
        return_raw_predictions: bool = False,
        return_stds: bool = False,
        mcd_nro_samples: int = 1,
        hook_layer_output: bool = True,
        dropblock_probs: Union[float, List] = 0.0,
        dropblock_sizes: Union[int, List] = 0,
        rcnn_extraction_type: str = None,
        extract_noise_entropies: bool = False,
    ):
        """
        Initializes an object detection module with specific configurations and settings based on
        the provided architecture, extraction types, and additional parameters.

        Args:
            architecture (str): Indicates the architecture type being used.
                Only values specified in the `SUPPORTED_OBJECT_DETECTION_ARCHITECTURES`
                are allowed.
            rcnn_extraction_type (str, optional): Specifies the type of RCNN extraction.
                Valid values include "rpn_inter", "rpn_head", "shortcut", "backbone", or None.
            extract_noise_entropies (bool, optional): Determines whether Monte Carlo sampling
                and noise entropy extraction are performed.
            **kwargs: Additional keyword arguments passed to the base class initialization.

        Raises:
            AssertionError: If the architecture provided is not in `SUPPORTED_OBJECT_DETECTION_ARCHITECTURES`.
            AssertionError: If `rcnn_extraction_type` is not one of the valid values.
            AssertionError: If the number of hooked layers does not match the number of hooked representations
                when `hook_layer_output` is True and the RCNN extraction type is not "rpn_inter".

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
        )
        assert (
            architecture in SUPPORTED_OBJECT_DETECTION_ARCHITECTURES
        ), f"Only {SUPPORTED_OBJECT_DETECTION_ARCHITECTURES} are supported"
        assert rcnn_extraction_type in ("rpn_inter", "rpn_head", "shortcut", "backbone", None)
        self.architecture = architecture
        self.rcnn_extraction_type = rcnn_extraction_type
        self.n_hooked_reps = len(self.hooked_layers)
        # When hooking the input, a direct layer Hook is expected
        if len(self.hooked_layers) == 1 and not self.hook_layer_output:
            self.hooked_layers = self.hooked_layers[0]
        # When hooking output, a list of Hooked layers is expected
        if self.hook_layer_output and self.rcnn_extraction_type != "rpn_inter":
            assert (
                len(self.hooked_layers) == self.n_hooked_reps
            ), "Specify an equal number of hooked layers and output sizes"

        # In case of performing MCD sampling
        self.extract_noise_entropies = extract_noise_entropies
        if self.extract_noise_entropies:
            self.mc_sampler = MCSamplerModule(
                mc_samples=self.mcd_nro_samples,
                block_size=self.dropblock_sizes,
                drop_prob=self.dropblock_probs,
                layer_type="Conv",
            )
            self.mc_sampler.to(self.device)

    def unpack_dataloader(
        self, loader_contents: Union[Dict, List, Tuple]
    ) -> Tuple[List[str], Any, str]:
        """
        Unpacks and processes the contents of a dataloader based on the model architecture.
        This method extracts image paths (or identifiers), raw images, and image IDs
        from the provided dataloader inputs.

        Args:
            loader_contents: Data retrieved from the dataloader. Contents vary depending on
                the model architecture and may include image paths, tensors, or metadata.

        Returns:
            Tuple[List[str], Any, str]: A tuple containing the processed image paths as a list
                of strings, the formatted image data (as tensors or appropriate format),
                and the corresponding image identifier.

        Raises:
            ValueError: If unable to process or format the image path correctly
                for certain architectures due to unexpected data.
        """
        if self.architecture == "yolov8":
            (impath, image, im_counter) = loader_contents
            try:
                int(impath[0].split("/")[-1].split(".")[0])
                im_id = impath[0].split("/")[-1].split(".")[0].lstrip("0")
            except ValueError:
                im_id = impath[0].split("/")[-1].split(".")[0]

        elif self.architecture == "rcnn":
            image = loader_contents
            impath = [image[0]["file_name"]]
            im_id = image[0]["image_id"]
        elif self.architecture == "owlv2":
            image = (
                loader_contents["input_ids"].to(self.device),
                loader_contents["attention_mask"].to(self.device),
                loader_contents["pixel_values"].to(self.device),
                loader_contents["orig_size"],
            )
            impath = [loader_contents["labels"][0]["image_id"]]
            im_id = impath[0]
        # DETR or RTDETR
        else:
            image = (
                loader_contents["pixel_values"].to(self.device),
                loader_contents["pixel_mask"].to(self.device),
                torch.stack(
                    [target["orig_size"] for target in loader_contents["labels"]], dim=0
                ).to(self.device),
            )
            impath = [loader_contents["labels"][0]["image_id"]]
            im_id = loader_contents["labels"][0]["image_id"].item()

        return impath, image, im_id

    def model_dependent_inference(
        self, image, predict_conf: float, **kwargs: Any
    ) -> Tuple[Dict, Tensor, Any, Tuple[int, int]]:
        """
        Performs model-dependent inference by applying the appropriate method based on the
        architecture type. It processes the input image and extracts predictions, bounding
        boxes, and relevant intermediate outputs such as logits or features, depending on
        the architecture.

        Args:
            image (list or tuple or Tensor): Input image(s) or related input data required
                for inference, varying by the selected model architecture.
            predict_conf (float): Confidence threshold used during prediction to filter
                outputs.
            **kwargs (Any): Additional keyword arguments for architecture-specific behavior
                or configuration during inference.

        Returns:
            Tuple[Tensor, Any, Tuple[int, int]]: A tuple containing:
                - `boxes` (Tensor): Predicted bounding boxes from the inference step.
                - `pred_img` (Any): Inference output specific to the model architecture.
                - `img_shape` (Tuple[int, int]): Dimensions of the input image (height, width).

        Raises:
            AssertionError: If the size of the logits in the results dictionary does not
                match the number of predicted boxes for "yolov8" architecture.
        """
        # Final results dictionary
        results = {}
        if self.architecture == "yolov8":
            img_shape = image[0].shape[:2]  # Height, width
            # Hook the Predict module
            hook_detect = Hook(self.model.model.model._modules["22"])
            # Perform inference just once per image
            pred_img = self.model(image, conf=predict_conf, **kwargs)
            if len(pred_img[0]) > 0:
                activation_detect = hook_detect.output[0]
                results["logits"] = self.yolo_get_logits(
                    prediction=activation_detect,
                    conf_thres=predict_conf,
                    iou_thres=self.model.predictor.args.iou,
                    classes=self.model.predictor.args.classes,
                    agnostic=self.model.predictor.args.agnostic_nms,
                    max_det=self.model.predictor.args.max_det,
                )
                assert len(results["logits"]) == len(pred_img[0])
            boxes = pred_img[0].boxes.xyxy

        elif self.architecture == "rcnn":
            img_shape = image[0]["height"], image[0]["width"]
            pred_img = self.model(image)
            if isinstance(pred_img, list):
                pred_img = pred_img[0]
            if isinstance(pred_img, dict):
                pred_img = pred_img["instances"]
            # The output of the rcnn seems to be already in the format xyxy
            boxes = pred_img.pred_boxes.tensor
            if "latent_feature" in pred_img._fields.keys():
                # Store previous-to-last features
                results["features"] = pred_img.latent_feature
            if "inter_feat" in pred_img._fields.keys():
                results["logits"] = pred_img.inter_feat
            elif "logits" in pred_img._fields.keys():
                results["logits"] = pred_img.logits

        elif self.architecture == "owlv2":
            img_shape = image[3][0]
            pred_img = self.model.forward_and_postprocess(
                input_ids=image[0],
                attention_mask=image[1],
                pixel_values=image[2],
                orig_sizes=image[3],
                threshold=predict_conf,
            )[
                0
            ]  # Batch size 1, therefore just one image
            boxes = pred_img["boxes"]
            results["features"] = pred_img["last_hidden"]
            results["logits"] = pred_img["logits"]
        # DETR or RTDETR
        else:
            img_shape = (image[2][0][0].item(), image[2][0][1].item())
            # Made a custom function on the Detr class to handle this input
            pred_img = self.model.forward_and_postprocess(
                pixel_values=image[0],
                pixel_mask=image[1],
                orig_sizes=image[2],
                threshold=predict_conf,
            )[
                0
            ]  # Batch size 1, therefore just one image
            boxes = pred_img["boxes"]
            results["features"] = pred_img["last_hidden"]
            results["logits"] = pred_img["logits"]
        return results, boxes, pred_img, img_shape

    def model_dependent_feature_extraction(
        self,
    ) -> Any:
        """
        Extracts and processes latent feature representations from models depending on
        the specified architecture and feature extraction type.

        This function provides a mechanism to retrieve intermediate or latent representations
        from multiple types of neural network architectures like RCNN, YOLO, DETR, OWLv2,
        and RTDETR-encoder. It handles the complexities of accessing the appropriate
        layers or tensors depending on the architecture type and specific configuration.

        Args:
            None

        Returns:
            Any: Processed latent feature representations extracted from the model.

        Raises:
            AssertionError: If the latent sample's length or structure does not match the
                expected number of hooked representations.
        """
        if self.architecture == "rcnn" and self.rcnn_extraction_type == "rpn_inter":
            # Ugly but still a simple solution for a complex architecture
            if hasattr(self.model, "model"):
                latent_sample = self.model.model.proposal_generator.rpn_head.rpn_intermediate_output
            else:
                latent_sample = self.model.proposal_generator.rpn_head.rpn_intermediate_output

        # Yolo, DETR, or other rcnn locations
        else:
            if self.hook_layer_output:
                latent_sample = [layer.output for layer in self.hooked_layers]
            else:
                latent_sample = self.hooked_layers.input
                # Input might be a one-element tuple, containing the desired list
                if len(latent_sample) == 1 and self.n_hooked_reps != 1:
                    try:
                        assert len(latent_sample[0]) == self.n_hooked_reps
                        latent_sample = latent_sample[0]
                    except AssertionError:
                        print("Cannot find a suitable latent space sample")
        # Check if rcnn backbone output
        if (
            self.architecture == "rcnn"
            and len(latent_sample) == 1
            and isinstance(latent_sample[0], dict)
            and self.rcnn_extraction_type == "backbone"
        ):
            latent_sample = [v for k, v in latent_sample[0].items()]
        # Check for rpn_head output extraction
        if (
            self.architecture == "rcnn"
            and len(latent_sample) == 1
            and isinstance(latent_sample[0], tuple)
            and len(latent_sample[0]) == 2
            and self.rcnn_extraction_type == "rpn_head"
        ):
            latent_sample = [
                torch.cat([obj_logit, anch_delta], dim=1)
                for obj_logit, anch_delta in zip(latent_sample[0][0], latent_sample[0][1])
            ]
        if self.architecture == "owlv2":
            latent_sample = [
                latent_sample[0][0][:, 1:, :].reshape(
                    1,
                    self.model.model.config.vision_config.hidden_size,
                    int(
                        self.model.model.config.vision_config.image_size
                        / self.model.model.config.vision_config.patch_size
                    ),
                    int(
                        self.model.model.config.vision_config.image_size
                        / self.model.model.config.vision_config.patch_size
                    ),
                )
            ]
        if self.architecture == "rtdetr-encoder":
            latent_sample = [
                latent_sample[0][0].permute(0, 2, 1).reshape(-1, 256, 20, 20).contiguous()
            ]
        return latent_sample

    @staticmethod
    def yolo_get_logits(
        prediction: Tensor,
        conf_thres: float,
        iou_thres: float,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det: int = 300,
        nc: int = 0,  # number of classes (optional)
        max_nms: int = 30000,
        max_wh: int = 7680,
    ):
        """
        Processes YOLO predictions and applies non-maximum suppression (NMS) on detected objects.

        This method takes raw predictions from a YOLO model and filters them based on confidence
        and IoU thresholds. It supports multi-class detections, class-agnostic filtering, and
        applies NMS to reduce duplicate detections. Additionally, the output logits can be
        calculated for the remaining bounding boxes.

        Args:
            prediction: Tensor containing raw YOLO model predictions with dimensions
                corresponding to bounding box coordinates, confidence scores, and class scores.
            conf_thres: Floating-point confidence threshold for filtering predictions, must be
                between 0.0 and 1.0.
            iou_thres: Floating-point IoU threshold used during non-maximum suppression, must
                be between 0.0 and 1.0.
            classes: Optional list of class IDs to filter predictions. If specified, only
                detections of these classes are retained. Defaults to None.
            agnostic: Boolean indicating whether NMS should ignore classes (class-agnostic) or
                not. Defaults to False.
            multi_label: Boolean specifying whether detections may contain multiple labels per
                bounding box. Defaults to False.
            max_det: Integer specifying the maximum number of retained detections per image.
                Defaults to 300.
            nc: Integer number of classes, optional. If not provided, it is derived from the
                prediction tensor's dimensions.
            max_nms: Integer specifying the maximum number of boxes allowed before applying NMS.
                Defaults to 30000.
            max_wh: Integer value used to scale class identifiers for offsetting class
                information in NMS. Defaults to 7680.

        Returns:
            Tensor: A concatenated tensor of object class logits with shape (n, nc) where 'n'
                represents the total number of valid detections after applying thresholds and
                filtering.

        Raises:
            AssertionError: If the provided values for `conf_thres` or `iou_thres` are not
                within the expected range [0.0, 1.0].
        """
        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[
                    x[:, 4].argsort(descending=True)[:max_nms]
                ]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores

            boxes = x[:, :4] + c  # boxes (offset by class)
            i = nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = torch.log(cls[i])

        return torch.cat(output, dim=0)
