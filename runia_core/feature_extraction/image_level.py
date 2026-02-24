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
import pytorch_lightning as pl
from warnings import warn
from typing import Union, List, Tuple, Dict, Any

from numpy import ndarray
from torch.nn.functional import avg_pool2d
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor
from dropblock import DropBlock2D

from runia_core.feature_extraction.abstract_classes import (
    Extractor,
    ObjectDetectionExtractor,
)
from runia_core.feature_extraction.utils import (
    Hook,
    get_mean_or_fullmean_ls_sample,
    get_std_ls_sample,
)

__all__ = [
    "FastMCDSamplesExtractor",
    "MCDSamplesExtractor",
    "ImageLvlFeatureExtractor",
    "deeplabv3p_get_ls_mcd_samples",
    "get_latent_representation_mcd_samples",
]


class FastMCDSamplesExtractor(Extractor):
    """
    Class that extracts MCD samples in a non-invasive way, that is, it catches the
    intermediate representation that we want to perturb by doing one single forward pass
    through the network, then it uses a local Dropout module to repeat the MC dropout inferences. It supports
    both fully connected (FC) and convolutional (Conv) layer types and provides options for dimensionality
    reduction and ground truth label return.

    Attributes:
        layer_type (str): Specifies the type of layer (either 'FC' or 'Conv') from which
            Monte Carlo samples are extracted.
        reduction_method (str): Method for dimensionality reduction of the hooked
            representation ('mean' or 'fullmean').
        return_gt_labels (bool): If True, includes the ground truth labels in the results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        hooked_layers: List[Hook],
        device: torch.device,
        layer_type: str,
        reduction_method: str,
        return_raw_predictions: bool = False,
        return_stds: bool = False,
        mcd_nro_samples: int = 1,
        hook_layer_output: bool = True,
        dropblock_probs: Union[float, List] = 0.0,
        dropblock_sizes: Union[int, List] = 0,
        return_gt_labels: bool = False,
    ):
        """
        Initializes the class with specified configurations for layer type, reduction
        method, and ground truth label return. Validates the provided configurations,
        ensuring they match the expected formats and values. Handles initialization
        of dropout layers based on given parameters, adapting them for either
        fully-connected (FC) or convolutional (Conv) layers.

        Args:
            layer_type (str): Specifies the type of layer. Must be either "FC"
                for fully-connected layers or "Conv" for convolutional layers.
            reduction_method (str): Specifies the method of reduction. Supported
                options include "mean" and "fullmean".
            return_gt_labels (bool, optional): Indicates whether ground truth
                labels should be returned. Defaults to False.
            **kwargs: Additional keyword arguments to customize functionality.
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
        self.hooked_layer = self.hooked_layers[0]
        assert layer_type in ("FC", "Conv"), "Layer type must be either 'FC' or 'Conv'"
        assert reduction_method in (
            "mean",
            "fullmean",
        ), "Only mean and fullmean reduction methods supported"

        self.layer_type = layer_type
        self.reduction_method = reduction_method
        self.return_gt_labels = return_gt_labels
        try:
            self.dropout_n_layers = len(self.dropblock_probs)
        except TypeError:
            self.dropout_n_layers = 1
            self.dropblock_probs = [self.dropblock_probs]
            self.dropblock_sizes = [self.dropblock_sizes]

        if self.layer_type == "Conv":
            self.dropout_layers = torch.nn.ModuleList(
                DropBlock2D(drop_prob=self.dropblock_probs[i], block_size=self.dropblock_sizes[i])
                for i in range(self.dropout_n_layers)
            )
        # FC
        else:
            self.dropout_layers = torch.nn.ModuleList(
                torch.nn.Dropout(self.dropblock_probs[i]) for i in range(self.dropout_n_layers)
            )

    def get_ls_samples(self, data_loader: DataLoader, **kwargs) -> Dict[str, Tensor]:
        """
        Perform the fast Monte Carlo Dropout inference given a dataloader. This class does not perform
        full model inference at each time, instead it uses the intermediate representation several times
        in Monte Carlo Dropblock.

        Args:
            data_loader: DataLoader

        Returns:
            Latent MCD samples and optionally the raw inference results
        """
        results = {"latent_space_means": []}
        if self.return_raw_predictions:
            results["raw_preds"] = []
        if self.return_stds:
            results["stds"] = []
        if self.return_gt_labels:
            results["gt_labels"] = []
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
                for image, gt_labels in data_loader:
                    image = image.to(self.device)
                    # Extraction per image
                    result_img = self._get_samples_one_image(image=image, **kwargs)
                    for result_type, result_value in result_img.items():
                        results[result_type].append(result_value)
                    if self.return_gt_labels:
                        results["gt_labels"].append(gt_labels.reshape(1, -1))
                    # Update progress bar
                    pbar.update(1)

                for result_type, result_value in results.items():
                    results[result_type] = torch.cat(result_value, dim=0)
        print("Latent representation vector size: ", results["latent_space_means"].shape[1])
        return results

    def _get_samples_one_image(self, image, **kwargs) -> Dict:
        """
        Processes a single image using Monte Carlo Dropout (MCD) sampling, returning derived
        statistics such as means, standard deviations, and optional raw predictions.

        Args:
            image: Image input to be processed by the model.
            **kwargs: Additional keyword arguments passed to the model during inference.

        Returns:
            Dict: A dictionary containing the following keys:
                - 'means': Tensor of mean values computed from MCD samples.
                - 'stds' (optional): Tensor of standard deviations computed from MCD samples,
                  only included if `self.return_stds` is True.
                - 'raw_preds' (optional): Raw model predictions, included if
                  `self.return_raw_predictions` is True.
        """
        # Final results dictionary
        results = {}
        img_mcd_samples_means = []
        img_mcd_samples_stds = []
        # Perform inference just once per image
        pred_img = self.model(image, **kwargs)
        # pred_img = getattr(self.model, "predict")(source=image, verbose=False)
        if self.hook_layer_output:
            latent_mcd_sample = self.hooked_layer.output
        else:
            latent_mcd_sample = self.hooked_layer.input
            # Input might be a one-element tuple, containing the desired list
            if len(latent_mcd_sample) == 1 and self.dropout_n_layers != 1:
                try:
                    assert len(latent_mcd_sample[0]) == self.dropout_n_layers
                    latent_mcd_sample = latent_mcd_sample[0]
                except AssertionError:
                    print("Cannot find a suitable latent space sample")
        for _ in range(self.mcd_nro_samples):
            # Apply dropout/dropblock first
            if self.dropout_n_layers == 1:
                latent_mcd_sample_noised = self.dropout_layers[0](latent_mcd_sample)
            else:
                latent_mcd_sample_noised = [
                    self.dropout_layers[i](latent_mcd_sample[i])
                    for i in range(self.dropout_n_layers)
                ]
            if self.dropout_n_layers == 1:
                if self.layer_type == "Conv":
                    latent_mcd_sample_means = get_mean_or_fullmean_ls_sample(
                        latent_mcd_sample_noised, self.reduction_method
                    )
                    if self.return_stds:
                        latent_mcd_sample_stds = get_std_ls_sample(latent_mcd_sample_noised)
                # FC
                else:
                    # It is already a 1d tensor
                    latent_mcd_sample_means = torch.squeeze(latent_mcd_sample_noised)
            # Several representations per layer
            else:
                if self.layer_type == "Conv":
                    n_latent_layers_means = []
                    n_latent_layers_stds = []
                    for i in range(self.dropout_n_layers):
                        n_latent_layers_means.append(
                            get_mean_or_fullmean_ls_sample(
                                latent_mcd_sample_noised[i], self.reduction_method
                            ).reshape(-1)
                        )
                        if self.return_stds:
                            n_latent_layers_stds.append(
                                get_std_ls_sample(latent_mcd_sample_noised[i]).reshape(-1)
                            )
                    latent_mcd_sample_means = torch.cat(n_latent_layers_means, dim=0)
                    if self.return_stds:
                        latent_mcd_sample_stds = torch.cat(n_latent_layers_stds, dim=0)
                # FC
                else:
                    raise NotImplementedError
            img_mcd_samples_means.append(latent_mcd_sample_means.reshape(1, -1))
            if self.return_stds:
                img_mcd_samples_stds.append(latent_mcd_sample_stds.reshape(1, -1))

        results["latent_space_means"] = torch.cat(img_mcd_samples_means, dim=0)
        if self.return_stds:
            results["stds"] = torch.cat(img_mcd_samples_stds, dim=0)
        if self.return_raw_predictions:
            results["raw_preds"] = pred_img
        return results


class MCDSamplesExtractor(Extractor):
    """
    Class to get Monte-Carlo samples from any torch model Dropout or Dropblock Layer. This class implements
    the classic MCD algorithm performing multiple inferences on a model.

    This class is designed to provide Monte-Carlo (MC) Dropout samples for a given PyTorch model,
    specifically for layers implementing dropout or dropblock functionality. The purpose is
    to perform uncertainty quantification or latent space sampling. It supports both fully
    connected (FC) and convolutional (Conv) layers with optional dimensionality reduction methods.

    Attributes:
        layer_type (str): Type of the target layer, either 'FC' for Fully Connected or
            'Conv' for Convolutional layers.
        reduction_method (str): Method of dimensionality reduction for hooked representations.
            Acceptable values are 'mean', 'fullmean', or 'avgpool'.
        avg_pooling_parameters (Union[Tuple, List, None]): Parameters for average pooling
            if the reduction method is set to 'avgpool'. Should consist of a tuple (kernel_size,
            stride, padding).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        hooked_layers: List[Hook],
        device: torch.device,
        layer_type: str,
        reduction_method: str,
        return_raw_predictions: bool = False,
        return_stds: bool = False,
        mcd_nro_samples: int = 1,
        hook_layer_output: bool = True,
        dropblock_probs: Union[float, List] = 0.0,
        dropblock_sizes: Union[int, List] = 0,
        avg_pooling_parameters: Union[Tuple, List, None] = None,
    ):
        """
        Class to get Monte-Carlo samples from any torch model Dropout or Dropblock Layer

        Args:
            layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or
                Conv (Convolutional)
            reduction_method: Whether to use fullmean, mean, or avgpool to reduce dimensionality
                of hooked representation
            avg_pooling_parameters: Optionally pass parameters for average pooling reduction, as a
                tuple of integers as: kernel size, stride, padding.

        Returns:
            Monte-Carlo Dropout samples for the input dataloader
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
        self.hooked_layer = self.hooked_layers[0]
        assert layer_type in ("FC", "Conv"), "Layer type must be either 'FC' or 'Conv'"
        assert reduction_method in (
            "mean",
            "fullmean",
            "avgpool",
        ), "Only mean, fullmean and avg pool reduction methods supported"

        if avg_pooling_parameters is not None:
            assert (
                len(avg_pooling_parameters) == 3
            ), "Three parameters are needed for average pooling"
        self.layer_type = layer_type
        self.reduction_method = reduction_method
        self.avg_pooling_parameters = avg_pooling_parameters

    def get_ls_samples(
        self, data_loader: DataLoader, **kwargs
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        Perform the Monte Carlo Dropout inference given a dataloader

        Args:
            data_loader: DataLoader

        Returns:
            Latent MCD samples and optionally the raw inference results
        """
        assert isinstance(data_loader, DataLoader)
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
                dl_imgs_latent_mcd_samples = []
                if self.return_raw_predictions:
                    raw_predictions = []
                for image, _ in data_loader:
                    # image = image.view(1, 1, 28, 28).to(device)
                    image = image.to(self.device)
                    if self.return_raw_predictions:
                        latent_samples, raw_preds = self._get_samples_one_image(
                            image=image, **kwargs
                        )
                        dl_imgs_latent_mcd_samples.append(latent_samples)
                        raw_predictions.extend(raw_preds)
                    else:
                        dl_imgs_latent_mcd_samples.append(self._get_samples_one_image(image=image))
                    # Update progress bar
                    pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)
        print("MCD N_samples: ", dl_imgs_latent_mcd_samples_t.shape[1])
        if self.return_raw_predictions:
            return dl_imgs_latent_mcd_samples_t, torch.cat(raw_predictions, dim=0)
        else:
            return dl_imgs_latent_mcd_samples_t

    def _get_samples_one_image(self, image, **kwargs) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        Private method that extracts latent samples from one image, and optionally returns
        raw predictions

        Args:
            image: Image

        Returns:
            Latent activation map reduced, and optionally raw predictions.
        """
        img_mcd_samples = []
        if self.return_raw_predictions:
            raw_predictions = []
        for _ in range(self.mcd_nro_samples):
            pred_img = self.model(image, **kwargs)
            if self.return_raw_predictions:
                raw_predictions.append(pred_img)
            latent_mcd_sample = self.hooked_layer.output
            if self.layer_type == "Conv":
                if self.reduction_method == "mean" or self.reduction_method == "fullmean":
                    latent_mcd_sample = get_mean_or_fullmean_ls_sample(
                        latent_mcd_sample, method=self.reduction_method
                    )
                # Avg pool
                else:
                    # Perform average pooling over latent representations
                    latent_mcd_sample = avg_pool2d(
                        latent_mcd_sample,
                        kernel_size=self.avg_pooling_parameters[0],
                        stride=self.avg_pooling_parameters[1],
                        padding=self.avg_pooling_parameters[2],
                    )
            # FC
            else:
                # It is already a 1d tensor
                latent_mcd_sample = torch.squeeze(latent_mcd_sample)
            img_mcd_samples.append(latent_mcd_sample.reshape(1, -1))

        img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
        if self.return_raw_predictions:
            raw_predictions = torch.cat(raw_predictions, dim=0)
            return img_mcd_samples_t, raw_predictions
        else:
            return img_mcd_samples_t


class ImageLvlFeatureExtractor(ObjectDetectionExtractor):
    """
    Extracts image-level features using various object detection architectures.

    This class supports different architectures such as YOLOv8, RCNN, DETR, and
    OWL-V2. It provides methods for feature extraction and latent space sampling
    and accommodates specific configurations for RCNN extraction types.
    Additionally, it includes functionality for noise entropy estimation and
    Monte Carlo dropout sampling.
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
        Initializes the extractor with the specified architecture, RCNN extraction type, and optional noise
        entropy extraction configurations. This constructor also ensures the validity of specified
        parameters, configures the hooking layers based on the chosen architecture, and optionally
        prepares for Monte Carlo Dropout (MCD) sampling.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class initializer.

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

        # When hooking the input, a direct layer Hook is expected
        if len(self.hooked_layers) == 1 and not self.hook_layer_output:
            # Ugly way to ensure that when the hooked layer is the last detect of yolo, its input comes from 3 previous layers
            if self.architecture == "yolov8":
                self.n_hooked_reps = 3

    def get_ls_samples(
        self, data_loader: Union[DataLoader, Any], predict_conf=0.25, **kwargs
    ) -> Dict:
        """
        Extracts latent space samples from images provided by the data loader using a specific model architecture.

        This method processes images in a data loader and extracts latent representations, features, and logits of
        the images based on the specified model architecture. It supports different model architectures such as
        "yolov8", "rcnn", and others (e.g., DETR). The method processes images individually (only batch size
        of 1 is supported) and ensures that latent space information is collected. If no objects are detected
        in an image, it tracks such images.

        Args:
            data_loader (Union[DataLoader, Any]): Data loader providing images and metadata for processing.
                The data loader must have a batch size and it should be equal to 1.
            predict_conf (float, optional): Prediction confidence threshold for object detection. Defaults to 0.25.
            **kwargs: Additional keyword arguments to customize behavior during latent sample extraction.

        Returns:
            Dict: A dictionary containing latent representations (means), additional features, logits,
            and a list of image paths (`no_obj`) where no objects were detected. If enabled during initialization,
            standard deviations (`stds`) are also included.
        """
        self.check_dataloader(data_loader)
        results = {"latent_space_means": [], "features": [], "logits": []}
        no_obj_imgs = []
        if self.return_stds:
            results["stds"] = []
        with torch.no_grad():
            with tqdm(
                total=len(data_loader), desc="Extracting latent space image level samples"
            ) as pbar:
                # for impath, image, im_counter in data_loader:
                for loader_contents in data_loader:
                    impath, image, im_id = self.unpack_dataloader(loader_contents)
                    result_img, found_obj_flag = self._get_samples_one_image(
                        image=image, predict_conf=predict_conf, **kwargs
                    )
                    for result_type, result_value in result_img.items():
                        results[result_type].append(result_value)
                    if not found_obj_flag:
                        # impath is a list, with batch size 1 we only need the first element (the string)
                        no_obj_imgs.append(impath[0])
                    # Update progress bar
                    pbar.update(1)
                for result_type, result_value in results.items():
                    results[result_type] = (
                        torch.cat(result_value, dim=0) if len(result_value) > 0 else result_value
                    )
        results["no_obj"] = no_obj_imgs
        print("Latent representation vector size: ", results["latent_space_means"].shape[1])
        print(f"No objects in {len(no_obj_imgs)} images")
        return results

    def _get_samples_one_image(
        self, image: Union[Tensor, ndarray], predict_conf: float, **kwargs
    ) -> Tuple[Dict[str, Tensor], bool]:
        """
        Processes a single image to extract image-level latent samples, along with predictions.
        This method utilizes a model's inference capability and feature extraction to generate
        results for downstream tasks.

        Args:
            image: Input image, can either be a PyTorch Tensor or a NumPy ndarray. Represents
                the image on which inference and feature extraction are performed.
            predict_conf: Float value representing the confidence threshold for predictions. Used
                to determine objects detected during model inference.
            **kwargs: Additional optional keyword arguments supporting customized inference
                processes or model-specific configurations.

        Returns:
            Tuple[Dict[str, Tensor], bool]: A tuple containing a results dictionary and a boolean
            flag. The results dictionary includes latent vector means, raw predictions if
            applicable, and other extracted features. The boolean flag indicates whether objects
            were found in the image (True) or if the entire image is treated as one object (False).
        """
        # Found objects flag
        found_objs_flag = True
        results, boxes, pred_img, img_shape = self.model_dependent_inference(
            image, predict_conf, **kwargs
        )
        n_detected_objects = boxes.shape[0]
        if n_detected_objects == 0:
            # Get whole image as single object if no objects are detected
            boxes = Tensor([0.0, 0.0, img_shape[1], img_shape[0]]).reshape(1, -1).to(self.device)
            n_detected_objects = 1
            found_objs_flag = False
        # Catch the latent activations
        latent_sample = self.model_dependent_feature_extraction()
        # Deterministic algorithm
        if not self.extract_noise_entropies:
            # latent samples is a list of tensors
            for i in range(len(latent_sample)):
                latent_sample[i] = get_mean_or_fullmean_ls_sample(
                    latent_sample[i], "fullmean"
                ).reshape(1, -1)
            results["latent_space_means"] = torch.cat(latent_sample, dim=1)
            if self.return_stds:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.return_stds:
            # results["stds"] = torch.cat(latent_mcd_sample_stds, dim=0)
            raise NotImplementedError
        if self.return_raw_predictions:
            results["raw_preds"] = pred_img
        return results, found_objs_flag


def deeplabv3p_get_ls_mcd_samples(
    model_module: pl.LightningModule,
    dataloader: DataLoader,
    mcd_nro_samples: int,
    hook_dropout_layer: Hook,
) -> Tensor:
    """
     Get Monte-Carlo samples form Deeplabv3+ DNN Dropout Layer

    Args:
        model_module (pl.LightningModule): Deeplabv3+ Neural Network Lightning Module
        dataloader (DataLoader): Input samples (torch) Dataloader
        mcd_nro_samples (int): Number of Monte-Carlo Samples
        hook_dropout_layer (Hook): Hook at the Dropout Layer from the Neural Network Module

    Returns:
        (Tensor): Monte-Carlo Dropout samples for the input dataloader
    """
    warn(
        "This method is deprecated. Use one of the Extractor classes instead",
        DeprecationWarning,
        stacklevel=2,
    )
    assert isinstance(model_module, torch.nn.Module), "model_module must be a pytorch model"
    assert isinstance(dataloader, DataLoader), "dataloader must be a DataLoader"
    assert isinstance(mcd_nro_samples, int), "mcd_nro_samples must be an integer"
    assert isinstance(hook_dropout_layer, Hook), "hook_dropout_layer must be an Hook"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        dl_imgs_latent_mcd_samples = []
        with tqdm(total=len(dataloader), desc="Extracting MCD samples") as pbar:
            for image, _ in dataloader:
                image = image.to(device)
                img_mcd_samples = []
                for _ in range(mcd_nro_samples):
                    _ = model_module.deeplab_v3plus_model(image)
                    # pred = torch.argmax(pred_img, dim=1)
                    latent_mcd_sample = hook_dropout_layer.output
                    # Get image HxW mean:
                    latent_mcd_sample = get_mean_or_fullmean_ls_sample(
                        latent_mcd_sample, method="fullmean"
                    )
                    img_mcd_samples.append(latent_mcd_sample.reshape(1, -1))

                img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
                dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)

            pbar.update(1)

        dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    return dl_imgs_latent_mcd_samples_t


def get_latent_representation_mcd_samples(
    dnn_model: torch.nn.Module,
    dataloader: DataLoader,
    mcd_nro_samples: int,
    layer_hook: Hook,
    layer_type: str,
) -> Tensor:
    """
    Get latent representations Monte-Carlo samples from DNN using a layer hook

    Args:
        dnn_model: Neural Network Torch or Lightning Module
        dataloader: Input samples (torch) Data loader
        mcd_nro_samples: Number of Monte-Carlo Samples
        layer_hook: DNN layer hook
        layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected)
        or Conv (Convolutional)

    Returns:
        Input dataloader latent representations MC samples tensor
    """
    warn(
        "This method is deprecated. Use one of the Extractor classes instead",
        DeprecationWarning,
        stacklevel=2,
    )
    assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
    assert isinstance(dataloader, DataLoader), "dataloader must be a DataLoader"
    assert isinstance(mcd_nro_samples, int), "mcd_nro_samples must be an integer"
    assert isinstance(layer_hook, Hook), "layer_hook must be an Hook"
    assert layer_type in ("FC", "Conv"), "Layer type must be either 'FC' or 'Conv'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as pbar:
            dl_imgs_latent_mcd_samples = []
            for image, _ in dataloader:
                image = image.to(device)
                img_mcd_samples = []
                for _ in range(mcd_nro_samples):
                    _ = dnn_model(image)
                    latent_mcd_sample = layer_hook.output

                    if layer_type == "Conv":
                        # Get image HxW mean:
                        latent_mcd_sample = get_mean_or_fullmean_ls_sample(
                            latent_mcd_sample, method="fullmean"
                        )
                    else:
                        # Aggregate the second dimension (dim 1) to keep the proposed boxes
                        # dimension
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=1)

                    img_mcd_samples.append(latent_mcd_sample.reshape(1, -1))

                img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
                dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                # Update progress bar
                pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    return dl_imgs_latent_mcd_samples_t
