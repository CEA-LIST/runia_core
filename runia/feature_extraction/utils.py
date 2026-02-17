# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez
#    Daniel Montoya
from typing import Dict, Tuple, List

import torch
from numpy import ndarray
from torch import Tensor
from dropblock import DropBlock2D

__all__ = [
    "Hook",
    "apply_dropout",
    "get_mean_or_fullmean_ls_sample",
    "get_variance_ls_sample",
    "get_std_ls_sample",
    "get_aggregated_data_dict",
    "associate_precalculated_baselines_with_raw_predictions",
]


class Hook:
    """
    This class will catch the input and output of any torch layer during forward/backward pass.

    Args:
        module (torch.nn.Module): Layer block from Neural Network Module
        backward (bool): backward-poss hook
    """

    def __init__(self, module: torch.nn.Module, backward: bool = False):
        """
        This class will catch the input and output of any torch layer during forward/backward pass.

        Args:
            module (torch.nn.Module): Layer block from Neural Network Module
            backward (bool): backward-poss hook
        """
        self.input = None
        self.output = None
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, outputs):
        self.input = inputs
        self.output = outputs

    def close(self):
        self.hook.remove()


def apply_dropout(m):
    """
    Activate Dropout or Dropblock layers.

    Args:
        m: Pytorch module
    """
    if isinstance(m, torch.nn.Dropout) or isinstance(m, DropBlock2D):
        m.train()


def get_mean_or_fullmean_ls_sample(latent_sample: Tensor, method: str = "fullmean") -> Tensor:
    """
    Get either the mean (get a $W \times C$-sized vector) or fullmean (get a $C$-sized vector)
    from the convolutional activation map. (From a $C \times H \times W_sized convolutional
    activation map)

    Args:
        latent_sample: Convolutional activation map
        method: Either 'mean' or 'fullmean'

    Returns:
        The reduced activation map
    """
    assert method in ("mean", "fullmean")
    if method == "mean":
        latent_sample = torch.mean(latent_sample, dim=3, keepdim=True)
        latent_sample = torch.squeeze(latent_sample)
    # fullmean
    else:
        latent_sample = torch.mean(latent_sample, dim=3, keepdim=True)
        latent_sample = torch.mean(latent_sample, dim=2, keepdim=True)
        latent_sample = torch.squeeze(latent_sample)
    return latent_sample


def get_variance_ls_sample(latent_sample: Tensor) -> Tensor:
    """
    Get the variance for each channel from the convolutional activation map.

    Args:
        latent_sample: Convolutional activation map

    Returns:
        The reduced activation map
    """
    latent_sample = torch.var(latent_sample, dim=3, keepdim=True)
    latent_sample = torch.var(latent_sample, dim=2, keepdim=True)
    latent_sample = torch.squeeze(latent_sample)
    return latent_sample


def get_std_ls_sample(latent_sample: Tensor) -> Tensor:
    """
    Get the standard deviation for each channel from the convolutional activation map.

    Args:
        latent_sample: Convolutional activation map

    Returns:
        The reduced activation map
    """
    latent_sample = torch.std(latent_sample, dim=3, keepdim=True)
    latent_sample = torch.std(latent_sample, dim=2, keepdim=True)
    latent_sample = torch.squeeze(latent_sample)
    return latent_sample


def get_aggregated_data_dict(
    data_dict: Dict,
    dataset_name: str,
    aggregated_data_dict: Dict[str, Tensor],
    no_obj_dict: Dict[str, List],
    non_empty_predictions_ids: Dict[str, List],
    probs_as_logits: bool,
) -> Tuple[Dict, Dict, Dict]:
    """
    Extracts and aggregates data from a given dataset dictionary for a specific dataset. Function
    most useful when dealing with object-level OOD detection and OSOD benchmark.

    Iterates through features, logits, and means of the dataset and processes them into
    aggregated dictionaries. Handles empty or non-existent keys and manages the conversion
    of logits to probabilities if specified.

    Args:
        data_dict (Dict): Input dictionary containing dataset-specific feature data,
            logits, and means information.
        dataset_name (str): Name of the dataset being processed. Must be in dara_dict.
        aggregated_data_dict (Dict): Dictionary to store aggregated features, logits,
            and means for the dataset.
        no_obj_dict (Dict): Dictionary to store "no_obj" data extracted from the
            dataset if available.
        non_empty_predictions_ids (Dict): Dictionary to store IDs of non-empty
            predictions found in the dataset.
        probs_as_logits (bool): Flag to determine whether the logits should be
            converted to probabilities using log transformation.

    Returns:
        Tuple[Dict, Dict, Dict]: A tuple containing the updated `aggregated_data_dict`,
            `no_obj_dict`, and `non_empty_predictions_ids`.
    """
    if "no_obj" in data_dict[dataset_name].keys():
        no_obj_dict[dataset_name] = data_dict[dataset_name].pop("no_obj")
    all_features = []
    for im_results in data_dict[f"{dataset_name}"].values():
        if len(im_results["features"]) > 0:
            all_features.append(im_results["features"])
    if len(all_features) > 0:
        aggregated_data_dict[f"{dataset_name} features"] = (
            torch.cat(all_features, dim=0).cpu().numpy()
        )

    all_logits = []
    for im_results in data_dict[f"{dataset_name}"].values():
        if len(im_results["logits"]) > 0:
            if probs_as_logits:
                all_logits.append(torch.log(im_results["logits"] + 1e-10))
            else:
                all_logits.append(im_results["logits"])
    if len(all_logits) > 0:
        aggregated_data_dict[f"{dataset_name} logits"] = torch.cat(all_logits, dim=0).cpu().numpy()

    all_latent_space_means = []
    non_empty_predictions_ids[dataset_name] = []
    for im_id, im_results in data_dict[f"{dataset_name}"].items():
        if len(im_results["latent_space_means"]) > 0:
            all_latent_space_means.append(im_results["latent_space_means"])
            non_empty_predictions_ids[dataset_name].extend(
                [im_id] * len(im_results["latent_space_means"])
            )
    aggregated_data_dict[f"{dataset_name} latent_space_means"] = (
        torch.cat(all_latent_space_means, dim=0).cpu().numpy()
    )
    return aggregated_data_dict, no_obj_dict, non_empty_predictions_ids


def associate_precalculated_baselines_with_raw_predictions(
    data_dict: Dict[str, Dict[str, torch.Tensor]],
    dataset_name: str,
    ood_baselines_dict: Dict[str, ndarray],
    baselines_names: List[str],
    non_empty_ids: List[str],
    is_ood: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Associates precomputed baseline results with raw predictions for a given dataset. Function
    most useful when dealing with object-level OOD detection and OSOD benchmark.

    This function updates a nested dictionary (`data_dict`) with precalculated
    baseline information from a separate dictionary (`ood_baselines_dict`), matching it
    to raw predictions using a list of image IDs (`non_empty_ids`). The baselines'
    association can be toggled based on whether the dataset is Out-Of-Distribution (OOD) or
    not using a boolean flag (`is_ood`). If a baseline name does not exist in the inner
    dictionary of `data_dict`, an empty list is created for initialization.

    Args:
        data_dict (Dict[str, Dict[str, torch.Tensor]]): A nested dictionary where predictions
            and baselines are stored for each image ID.
        dataset_name (str): The name of the dataset being processed, used to build the keys
            for OOD baseline values in `ood_baselines_dict`.
        ood_baselines_dict (Dict[str, ndarray]): Dictionary containing precomputed baselines
            for either OOD or non-OOD datasets. Keys are constructed using baseline names
            and dataset names.
        baselines_names (List[str]): A list of baseline model names to associate with each
            image's data in `data_dict`.
        non_empty_ids (List[str]): A list of image IDs for which baselines are to be
            associated. These image IDs correspond to datasets with existing predictions.
        is_ood (bool): A flag indicating whether the dataset is Out-Of-Distribution (OOD).
            When True, keys in `ood_baselines_dict` are constructed using the dataset name
            and baseline names.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: The updated `data_dict`, now containing
        additional information about the associated baseline values.
    """
    for idx, im_id in enumerate(non_empty_ids):
        for baseline_name in baselines_names:
            if not baseline_name in data_dict[im_id].keys():
                data_dict[im_id][baseline_name] = []
            if is_ood:
                data_dict[im_id][baseline_name].append(
                    ood_baselines_dict[f"{dataset_name} {baseline_name}"][idx]
                )
            else:
                data_dict[im_id][baseline_name].append(ood_baselines_dict[f"{baseline_name}"][idx])
    return data_dict
