# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez
#    Daniel Montoya

from typing import Tuple, Dict, List
import torch
import numpy as np
from dropblock import DropBlock2D
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

from runia.dimensionality_reduction import apply_pca_transform
from runia.inference import (
    LaRExInference,
    record_time,
    LaRDInference,
)
from runia.feature_extraction import Hook
from runia.evaluation.entropy import get_dl_h_z

# Very bad practice
dropblock_ext = DropBlock2D(drop_prob=0.4, block_size=1)

__all__ = [
    "get_msp_score_rcnn",
    "get_dice_feat_mean_react_percentile_rcnn",
    "get_energy_score_rcnn",
    "get_ls_mcd_samples_rcnn",
    "MCSamplerRCNN",
    "LaRexInferenceRCNN",
    "LaRDInferenceRCNN",
    "remove_background_dimension",
]


def get_msp_score_rcnn(dnn_model: torch.nn.Module, input_dataloader: DataLoader) -> np.ndarray:
    """
    Calculates the Maximum softmax probability score from an RCNN architecture coded with the
    Detectron 2 library, where the results are the first element of the output of the network,
    and the softmax are already calculated within the scores attribute of the results.

    Args:
        dnn_model (torch.nn.Module): The RCNN model
        input_dataloader (DataLoader): The Dataloader

    Returns:
        np.ndarray: The MSP scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_model.to(device)
    dl_preds_msp_scores = []

    with torch.no_grad():
        for image in tqdm(input_dataloader, desc="Getting MSP score"):
            # image = image.to(device)
            results, _ = dnn_model(image)

            pred_scores = results.scores
            # ic(pred_score.shape)
            # get the max values:
            if len(pred_scores) > 0:
                dl_preds_msp_scores.append(pred_scores.max().reshape(1))
            else:
                dl_preds_msp_scores.append((torch.Tensor([0.0])).to(device))

        dl_preds_msp_scores_t = torch.cat(dl_preds_msp_scores, dim=0)
        # ic(dl_preds_msp_scores_t.shape)
        dl_preds_msp_scores = dl_preds_msp_scores_t.detach().cpu().numpy()

    return dl_preds_msp_scores


def get_dice_feat_mean_react_percentile_rcnn(
    dnn_model: torch.nn.Module, ind_dataloader: DataLoader, react_percentile: int = 90
) -> Tuple[np.ndarray, float]:
    """
    Get the DICE and ReAct thresholds for sparsifying and clipping from an RCNN architecture, where
    the output has been modified to return the previous-to-last layer activations.

    Args:
        dnn_model: The RCNN model
        ind_dataloader: The Data loader
        react_percentile: Desired percentile for ReAct

    Returns:
        Tuple[np.ndarray, float]: The DICE expected values, and the ReAct threshold
    """
    assert 0 < react_percentile < 100, "react_percentile must be greater than 0 and less than 100"
    feat_log = []
    dnn_model.model.eval()
    assert dnn_model.dice_react_precompute
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs in tqdm(ind_dataloader, desc="Setting up DICE/ReAct"):
        outputs = dnn_model(inputs)
        out = outputs.mean(0)
        out = out.view(1, -1)
        # score = dnn_model.fc(out)
        feat_log.append(out.data.cpu().numpy())
    feat_log_array = np.array(feat_log).squeeze()
    return feat_log_array.mean(0), np.percentile(feat_log_array, react_percentile)


def get_energy_score_rcnn(dnn_model: torch.nn.Module, input_dataloader: DataLoader):
    """
    Calculates the energy uncertainty score from an RCNN architecture where the output has been
    modified to return the raw activations before NMS alongside the normal (NMS filtered) ones.

    Args:
        dnn_model: The RCNN model
        input_dataloader: The Data loader

    Returns:
        Tuple[np.ndarray, np.ndarray]: Energy scores from the Raw and the filtered outputs
    """
    # Here we take the enrgy as a mean of the whole 1000 proposals
    raw_preds_energy_scores = []
    # Here we take the enrgy as a mean of the filtered detections after NMS
    filtered_preds_energy_scores = []

    with torch.no_grad():
        for image in tqdm(input_dataloader, desc="Getting energy score"):
            results, box_cls = dnn_model(image)
            # Raw energy
            raw_energy_score = torch.logsumexp(box_cls[:, :-1], dim=1)
            raw_preds_energy_scores.append(raw_energy_score.mean().reshape(1))
            # Filtered energy
            filtered_energy_score = torch.logsumexp(results.inter_feat[:, :-1], dim=1)
            filtered_preds_energy_scores.append(filtered_energy_score.mean().reshape(1))

        raw_preds_energy_scores_t = torch.cat(raw_preds_energy_scores, dim=0)
        raw_preds_energy_scores = raw_preds_energy_scores_t.detach().cpu().numpy()
        filtered_preds_energy_scores_t = torch.cat(filtered_preds_energy_scores, dim=0)
        filtered_preds_energy_scores = filtered_preds_energy_scores_t.detach().cpu().numpy()

    return raw_preds_energy_scores, filtered_preds_energy_scores


# Get latent space Monte Carlo Dropout samples
def get_ls_mcd_samples_rcnn(
    model: torch.nn.Module,
    data_loader: torch.utils.data.dataloader.DataLoader,
    mcd_nro_samples: int,
    hook_dropout_layer: Hook,
    layer_type: str,
    return_raw_predictions: bool,
) -> torch.tensor:
    """
    Get Monte-Carlo Dropout samples from RCNN's Dropout or Dropblock Layer.
    For this function to work on the RPN, it must have defined a list called
    'rpn_intermediate_output' that collects the intermediate representations at inference time
    i.e. the hook is not enough since it only captures one output per layer and this layer
    outputs five elements. A modification in the RPN is therefore needed.
    An example below::

        def forward(self, features: List[torch.Tensor]):
            self.rpn_intermediate_output.clear()
            pred_objectness_logits = []
            pred_anchor_deltas = []
            for x in features:
                t = self.dropblock(self.conv(x))
                # Normal NN operations
                pred_objectness_logits.append(self.objectness_logits(t))
                pred_anchor_deltas.append(self.anchor_deltas(t))
                # Hook operations to catch RPN intermediate output
                self.rpn_intermediate_output.append(t)
            return pred_objectness_logits, pred_anchor_deltas

    Args:
        model: Torch model
        data_loader: Input samples (torch) DataLoader
        mcd_nro_samples: Number of Monte-Carlo Samples
        hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
        layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or
            Conv (Convolutional)
        return_raw_predictions: Returns the raw logits output

    Returns:
        Monte-Carlo Dropout samples for the input dataloader
    """
    assert layer_type in (
        "FC",
        "Conv",
        "RPN",
        "backbone",
    ), "Layer type must be either 'FC','backbone', 'RPN' or 'Conv'"
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
            dl_imgs_latent_mcd_samples = []
            if return_raw_predictions:
                raw_predictions = []
            for i, image in enumerate(data_loader):
                img_mcd_samples = []
                for _ in range(mcd_nro_samples):
                    instances, _ = model(image)
                    if return_raw_predictions:
                        raw_predictions.append(instances.inter_feat[:, :-1].mean(0))
                    # pred = torch.argmax(pred_img, dim=1)
                    latent_mcd_sample = hook_dropout_layer.output
                    if layer_type == "Conv":
                        # Get image HxW mean:
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                        # Remove useless dimensions:
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=3)
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                    elif layer_type == "RPN":
                        latent_mcd_sample = (
                            model.model.proposal_generator.rpn_head.rpn_intermediate_output
                        )
                        for i in range(len(latent_mcd_sample)):
                            latent_mcd_sample[i] = torch.mean(
                                latent_mcd_sample[i], dim=2, keepdim=True
                            )
                            latent_mcd_sample[i] = torch.mean(
                                latent_mcd_sample[i], dim=3, keepdim=True
                            )
                            # Remove useless dimensions:
                            latent_mcd_sample[i] = torch.squeeze(latent_mcd_sample[i])
                        latent_mcd_sample = torch.cat(latent_mcd_sample, dim=0)
                    elif layer_type == "backbone":
                        # Apply dropblock
                        for k, v in latent_mcd_sample.items():
                            latent_mcd_sample[k] = dropblock_ext(v)
                            # Get image HxW mean:
                            latent_mcd_sample[k] = torch.mean(
                                latent_mcd_sample[k], dim=2, keepdim=True
                            )
                            latent_mcd_sample[k] = torch.mean(
                                latent_mcd_sample[k], dim=3, keepdim=True
                            )
                            # Remove useless dimensions:
                            latent_mcd_sample[k] = torch.squeeze(latent_mcd_sample[k])
                        latent_mcd_sample = torch.cat(list(latent_mcd_sample.values()), dim=0)
                    # FC
                    else:
                        # Aggregate the second dimension (dim 1) to keep the proposed boxes dim
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=1)
                    if (
                        (layer_type == "FC" and latent_mcd_sample.shape[0] == 1000)
                        or layer_type == "RPN"
                        or layer_type == "Conv"
                    ):
                        img_mcd_samples.append(latent_mcd_sample)
                    elif layer_type == "FC" and latent_mcd_sample.shape[0] != 1000:
                        pass
                    else:
                        raise NotImplementedError
                if layer_type == "Conv":
                    img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
                    dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                else:
                    if (
                        (layer_type == "FC" and latent_mcd_sample.shape[0] == 1000)
                        or layer_type == "RPN"
                        or layer_type == "Conv"
                    ):
                        img_mcd_samples_t = torch.stack(img_mcd_samples, dim=0)
                        dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                    elif layer_type == "FC" and latent_mcd_sample.shape[0] != 1000:
                        print(f"Omitted image: {image[0]['image_id']}")
                    else:
                        raise NotImplementedError

                # Update progress bar
                pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    if return_raw_predictions:
        return dl_imgs_latent_mcd_samples_t, torch.stack(raw_predictions, dim=0)
    else:
        return dl_imgs_latent_mcd_samples_t


class MCSamplerRCNN(torch.nn.Module):
    """
    Class that takes MCD samples from an already trained model. To take samples from the RPN,
    a modification is needed in its forward function.
    An example below::

        def forward(self, features: List[torch.Tensor]):
            self.rpn_intermediate_output.clear()
            pred_objectness_logits = []
            pred_anchor_deltas = []
            for x in features:
                t = self.dropblock(self.conv(x))
                # Normal NN operations
                pred_objectness_logits.append(self.objectness_logits(t))
                pred_anchor_deltas.append(self.anchor_deltas(t))
                # Hook operations to catch RPN intermediate output
                self.rpn_intermediate_output.append(t)
            return pred_objectness_logits, pred_anchor_deltas

    Args:
        mc_samples: Number of MCD samples to take
        layer_type: Either 'Conv' or 'FC'
    """

    def __init__(self, mc_samples=8, layer_type: str = "RPN"):
        """
        Class that takes MCD samples from an already trained model.
        Args:
            mc_samples: Number of MCD samples to take
            layer_type: Either 'Conv' or 'FC'
        """
        super(MCSamplerRCNN, self).__init__()
        assert layer_type == "RPN", "layer_type must be 'RPN'"
        self.mc_samples = mc_samples
        self.drop_blocks = torch.nn.ModuleList(
            [DropBlock2D(block_size=8, drop_prob=0.5) for i in range(self.mc_samples)]
        )

    def forward(self, model):
        latent_mcd_sample_init = model.model.proposal_generator.rpn_head.rpn_intermediate_output
        samples = []
        for drop_layer in self.drop_blocks:
            latent_mcd_sample = latent_mcd_sample_init.copy()
            for i in range(len(latent_mcd_sample)):
                latent_mcd_sample[i] = drop_layer(latent_mcd_sample[i])
                latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=2, keepdim=True)
                latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=3, keepdim=True)
                # Remove useless dimensions:
                latent_mcd_sample[i] = torch.squeeze(latent_mcd_sample[i])
            latent_mcd_sample = torch.cat(latent_mcd_sample, dim=0)
            samples.append(latent_mcd_sample)
        samples_t = torch.stack(samples)
        return samples_t


class LaRexInferenceRCNN(LaRExInference):
    """
    Module to perform inference on a trained RCNN model with LaREx. The RCNN must have the
    compatible architectural modifications; namely: need to catch the intermediate representations
    from the convolutional layer at the RPN with a list.
    An example below::

        def forward(self, features: List[torch.Tensor]):
            self.rpn_intermediate_output.clear()
            pred_objectness_logits = []
            pred_anchor_deltas = []
            for x in features:
                t = self.dropblock(self.conv(x))
                # Normal NN operations
                pred_objectness_logits.append(self.objectness_logits(t))
                pred_anchor_deltas.append(self.anchor_deltas(t))
                # Hook operations to catch RPN intermediate output
                self.rpn_intermediate_output.append(t)
            return pred_objectness_logits, pred_anchor_deltas
    """

    def get_score(self, input_image, layer_hook):
        """
        Compute LaREx score for a single image for the RCNN architecture

        Args:
            input_image: New image, in tensor format
            layer_hook: Hooked layer

        Returns:
            LaREx score
        """
        with torch.no_grad():
            try:
                input_image = input_image.to(self.device)
            except AttributeError:
                pass
            output = self.dnn_model(input_image)

        mc_samples_t = self.mc_sampler(self.dnn_model)
        _, sample_h_z = get_dl_h_z(mc_samples_t, self.mcd_samples_nro)
        if self.pca_transform:
            sample_h_z = apply_pca_transform(sample_h_z, self.pca_transform)
        sample_score = self.detector.postprocess(sample_h_z)
        return output, sample_score

    @record_time
    def get_layer_mc_samples(self, input_image, layer_hook):
        """
        Function that benchmarks how much time it takes to perform one inference pass through
        the network plus the monte carlo sampling.

        Args:
            input_image: Input image
            layer_hook: Hooked layer

        Returns:
            MC samples
        """
        with torch.no_grad():
            try:
                input_image = input_image.to(self.device)
            except AttributeError:
                pass
            _ = self.dnn_model(input_image)

        mc_samples_t = self.mc_sampler(self.dnn_model)
        return mc_samples_t

    @record_time
    def get_score_full_inference(self, input_image, layer_hook):
        """
        Function that allows to perform on inference pass using the slow version of the method,
        where N inference passes are performed through the network, instead of monte carlo
        sampling only the intermediate representation.

        Args:
            input_image: Input image
            layer_hook: Hooked layer

        Returns:
            Network prediction and LaREX score
        """
        with torch.no_grad():
            mc_samples_t = []
            for i in range(self.mcd_samples_nro):
                output = self.dnn_model(input_image)
                if self.layer_type == "RPN":
                    # Take latent sample
                    latent_mcd_sample = (
                        self.dnn_model.model.proposal_generator.rpn_head.rpn_intermediate_output
                    )
                    for i in range(len(latent_mcd_sample)):
                        latent_mcd_sample[i] = self.mc_sampler.drop_blocks[0](latent_mcd_sample[i])
                        latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=2, keepdim=True)
                        latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=3, keepdim=True)
                        # Remove useless dimensions:
                        latent_mcd_sample[i] = torch.squeeze(latent_mcd_sample[i])
                    latent_mcd_sample = torch.cat(latent_mcd_sample, dim=0)
                    mc_samples_t.append(latent_mcd_sample)
            mc_samples_t = torch.stack(mc_samples_t, dim=0)

            _, sample_h_z = get_dl_h_z(mc_samples_t, self.mcd_samples_nro)
            if self.pca_transform:
                sample_h_z = apply_pca_transform(sample_h_z, self.pca_transform)
            sample_larex_score = self.detector.postprocess(sample_h_z)
            return output, sample_larex_score


class LaRDInferenceRCNN(LaRDInference):
    """
    LaRDInference class that allows inference using the LaRD model, which leverages a detector
    of latent feature statistics for distribution shift detection

    Args:
            dnn_model: RCNN trained model
            detector: KDE or Mahalanobis module
            pca_transform: PCA trained tranformation function
            layer_type: Either "Conv", "RPN" or "FC"
    """

    def __init__(
        self,
        dnn_model: torch.nn.Module,
        detector,
        pca_transform=None,
        layer_type="Conv",
    ):
        """
        LaRDInference class that allows inference using the LaRD model, which leverages a detector
        of latent feature statistics for distribution shift detection

        Args:
            dnn_model: RCNN trained model
            detector: KDE or Mahalanobis module
            pca_transform: PCA trained tranformation function
            layer_type: Either "Conv", "RPN" or "FC"
        """
        super().__init__(dnn_model, detector, pca_transform, layer_type)
        if layer_type == "RPN":
            self.reducer = self.process_rpn_intermediate_representation

    def process_rpn_intermediate_representation(self, intermediate_representation):
        """
        This reduction function assumes that the RCNN has the compatible architectural
        modifications; namely: need to catch the intermediate representations from the
        convolutional layer at the RPN with a list. An example below::

            def forward(self, features: List[torch.Tensor]):
                self.rpn_intermediate_output.clear()
                pred_objectness_logits = []
                pred_anchor_deltas = []
                for x in features:
                    t = self.dropblock(self.conv(x))
                    # Normal NN operations
                    pred_objectness_logits.append(self.objectness_logits(t))
                    pred_anchor_deltas.append(self.anchor_deltas(t))
                    # Hook operations to catch RPN intermediate output
                    self.rpn_intermediate_output.append(t)
                return pred_objectness_logits, pred_anchor_deltas

        Args:
            intermediate_representation: Dummy argument for compatibility reasons.

        Returns:
            Reduced version of the latent representation in the RPN
        """
        latent_mcd_sample = self.dnn_model.model.proposal_generator.rpn_head.rpn_intermediate_output
        for i in range(len(latent_mcd_sample)):
            latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=2, keepdim=True)
            latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=3, keepdim=True)
            # Remove useless dimensions:
            latent_mcd_sample[i] = torch.squeeze(latent_mcd_sample[i])
        return torch.cat(latent_mcd_sample, dim=0).cpu().numpy().reshape(1, -1)


def remove_background_dimension(
    fc_params: Dict,
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    ood_names: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Remove background dimension for logits in train, valid and ood datasets dictionaries, and in weights parameters.
    Args:
        fc_params: Dictionary of weight and bias parameters for the last fully connected layer
        ind_data_dict: Dictionary with the keys 'train logits', 'valid logits'
        ood_data_dict: Dictionary with the keys 'ood_dataset logits' for every ood dataset
        ood_names: Names of the ood_datsets in the ood data dict

    Returns:
        Tuple of ind data, ood data and fc parameters with the last dimension removed
    """
    w, b = fc_params["weight"], fc_params["bias"]
    if isinstance(w, Tensor):
        w = w.numpy()
    if isinstance(b, Tensor):
        b = b.numpy()
    # Control for background category in rcnn
    if w.shape[0] == 21 or w.shape[0] == 11:
        w = w[:-1, :]
    if b.shape[0] == 21 or b.shape[0] == 11:
        b = b[:-1]
    fc_params["weight"] = w
    fc_params["bias"] = b
    if (
        ind_data_dict["train logits"].shape[-1] == 21
        or ind_data_dict["train logits"].shape[-1] == 11
    ):
        ind_data_dict["train logits"] = ind_data_dict["train logits"][:, :-1]
    if (
        ind_data_dict["valid logits"].shape[-1] == 21
        or ind_data_dict["valid logits"].shape[-1] == 11
    ):
        ind_data_dict["valid logits"] = ind_data_dict["valid logits"][:, :-1]

    for ood_dataset in ood_names:
        if (
            ood_data_dict[f"{ood_dataset} logits"].shape[-1] == 21
            or ood_data_dict[f"{ood_dataset} logits"].shape[-1] == 11
        ):
            ood_data_dict[f"{ood_dataset} logits"] = ood_data_dict[f"{ood_dataset} logits"][:, :-1]

    return ind_data_dict, ood_data_dict, fc_params
