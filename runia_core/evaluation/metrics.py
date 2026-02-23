# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez
#    Daniel Montoya

from typing import Union, Tuple, Dict, List, Optional
import numpy as np
import torch
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import auc
import torchmetrics.functional as tmf
import seaborn as sns
from collections import defaultdict

from runia_core.inference.postprocessors import postprocessors_dict

__all__ = [
    "get_auroc_results",
    "plot_roc_ood_detector",
    "save_roc_ood_detector",
    "save_scores_plots",
    "get_pred_scores_plots",
    "log_evaluate_postprocessors",
    "select_and_log_best_larex",
    "subset_boxes",
]


def get_auroc_results(
    detect_exp_name: str,
    ind_samples_scores: np.ndarray,
    ood_samples_scores: np.ndarray,
    return_results_for_mlflow: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Calculates the metrics relevant for OoD detection: AUROC, FPR, AUPR, TPR, precision, recall,
    and classification thresholds. Can optionally format results for mlflow logging (no @ allowed).
    Automatically inverts labels if AUROC < 0.5.

    Args:
        detect_exp_name: Name of the current experiment. This will be of the name of the row
         of the returned pandas df
        ind_samples_scores: Array of InD scores
        ood_samples_scores: Array of OoD scores
        return_results_for_mlflow: Optionally return AUROC, FPR and AUPR formatted for mlflow
         logging

    Returns:
        (pd.Dataframe): Results in a pandas dataframe format and optionally a dictionary with
            results for mlflow
    """
    labels_ind_test = np.ones((ind_samples_scores.shape[0], 1))  # positive class
    labels_ood_test = np.zeros((ood_samples_scores.shape[0], 1))  # negative class

    ind_samples_scores = np.expand_dims(ind_samples_scores, 1)
    ood_samples_scores = np.expand_dims(ood_samples_scores, 1)

    scores = np.vstack((ind_samples_scores, ood_samples_scores))
    labels = np.vstack((labels_ind_test, labels_ood_test))
    labels = labels.astype("int32")

    roc_auc = tmf.auroc(torch.from_numpy(scores), torch.from_numpy(labels), task="binary")

    fpr, tpr, roc_thresholds = tmf.roc(
        torch.from_numpy(scores), torch.from_numpy(labels), task="binary"
    )

    fpr_95 = fpr[torch.where(tpr >= 0.95)[0][0]]

    precision, recall, pr_thresholds = tmf.precision_recall_curve(
        torch.from_numpy(scores), torch.from_numpy(labels), task="binary"
    )
    aupr = auc(recall.numpy(), precision.numpy())

    results_table = pd.DataFrame.from_dict(
        {detect_exp_name: [roc_auc.item(), fpr_95.item(), aupr, fpr.tolist(), tpr.tolist()]},
        orient="index",
        columns=[
            "auroc",
            "fpr@95",
            "aupr",
            "fpr",
            "tpr",
        ],
    )

    if not return_results_for_mlflow:
        return results_table
    results_for_mlflow = results_table.loc[detect_exp_name, ["auroc", "fpr@95", "aupr"]].to_dict()
    # MLFlow doesn't accept the character '@'
    results_for_mlflow["fpr_95"] = results_for_mlflow.pop("fpr@95")
    return results_table, results_for_mlflow


def plot_roc_ood_detector(results_table, plot_title: str = "Plot Title"):
    """
    Plot ROC curve from the results table from the function get_hz_detector_results.

    Args:
        results_table: Pandas table obtained with the get_hz_detector_results function
        plot_title: Title of the plot

    """
    plt.figure(figsize=(8, 6))
    for i in results_table.index:
        # print(i)
        plt.plot(
            results_table.loc[i]["fpr"],
            results_table.loc[i]["tpr"],
            label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
        )

    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(plot_title, fontweight="bold", fontsize=15)
    plt.legend(prop={"size": 12}, loc="lower right")
    plt.show()


def save_roc_ood_detector(
    results_table: pd.DataFrame, postprocessors: List[str], plot_title: str = "Plot Title"
) -> plt.Figure:
    """
    Returns a ROC plot figure that can be saved or logged with mlflow. Does not display any
    figure to screen

    Args:
        results_table (pd.Dataframe): Dataframe with results as rows and experiment names as
            indexes
        postprocessors: List of strings of postprocessors names
        plot_title (str): Title of the plot

    Returns:
        (plt.Figure): A figure to be saved or logged with mlflow
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in results_table.index:
        if any([postp in i for postp in postprocessors]):
            ax.plot(
                results_table.loc[i]["fpr"],
                results_table.loc[i]["tpr"],
                label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
                linestyle="solid",
                linewidth=3.0,
            )
        else:
            ax.plot(
                results_table.loc[i]["fpr"],
                results_table.loc[i]["tpr"],
                label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
                linestyle="dashed",
                linewidth=1.7,
            )

    ax.plot([0, 1], [0, 1], color="orange", linestyle="--")
    ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_title(plot_title, fontweight="bold", fontsize=15)
    ax.legend(prop={"size": 12}, loc="lower right")
    return fig


def save_scores_plots(
    scores_ind: np.ndarray,
    ood_scores_dict: Dict,
    ood_datasets_list: List[str],
    ind_dataset_name: str,
    post_processor_name: str = "LaREM",
) -> Dict:
    """
    InD and OoD agnostic function that takes as input the InD numpy ndarray with the LaRED scores,
    a dictionary of OoD LaRED scores, a list of the names of the OoD dataset, and the name of the
    InD dataset, and returns a histogram of pairwise comparisons, that can be saved to a
    file, logged with mlflow, or shown in screen

    Args:
        scores_ind: InD LaRED scores as numpy ndarray
        ood_scores_dict: Dictionary keys as ood datasets names and values as ndarrays of
            LaRED scores per each
        ood_datasets_list: List of OoD datasets names
        ind_dataset_name: String with the name of the InD dataset
        post_processor_name: String with the name of the post-processing function. One of "LaRED", "LaREM", or "LaREK"

    Returns:
        Dictionary of plots where the keys are the plot names and the values are the figures
    """
    assert post_processor_name in postprocessors_dict.keys()
    df_scores_ind = pd.DataFrame(scores_ind, columns=[f"{post_processor_name} score"])
    df_scores_ind.insert(0, "Dataset", "")
    df_scores_ind.loc[:, "Dataset"] = ind_dataset_name
    ood_df_dict = {}
    for ood_dataset_name in ood_datasets_list:
        ood_df_dict[ood_dataset_name] = pd.DataFrame(
            ood_scores_dict[ood_dataset_name], columns=[f"{post_processor_name} score"]
        )
        ood_df_dict[ood_dataset_name].insert(0, "Dataset", "")
        ood_df_dict[ood_dataset_name].loc[:, "Dataset"] = ood_dataset_name

    plots_dict = {}
    for ood_dataset_name in ood_datasets_list:
        df_h_z_scores = pd.concat([df_scores_ind, ood_df_dict[ood_dataset_name]]).reset_index(
            drop=True
        )
        plots_dict[f"{ood_dataset_name}_{post_processor_name}_scores"] = sns.displot(
            df_h_z_scores, x=f"{post_processor_name} score", hue="Dataset", kind="hist", fill=True
        )

    return plots_dict


def get_pred_scores_plots(
    experiment: Dict, ood_datasets_list: list, title: str, ind_dataset_name: str
):
    """
    Function that takes as input an experiment dictionary (one classification technique), a list
    of ood datasets, a plot title, and the InD dataset name and returns a plot of the predictive
    score density

    Args:
        experiment: Dictionary with keys 'InD':ndarray, 'x_axis':str, and 'plot_name':str and other
            keys as ood dataset names with values as ndarray
        ood_datasets_list: List with OoD datasets names
        title: Title of the plot
        ind_dataset_name: String with the name of the InD dataset

    Returns:
        Figure with the density scores of the InD and the OoD datasets
    """
    df_pred_h_scores_ind = pd.DataFrame(experiment["InD"], columns=[experiment["x_axis"]])
    df_pred_h_scores_ind.insert(0, "Dataset", "")
    df_pred_h_scores_ind.loc[:, "Dataset"] = ind_dataset_name
    ood_df_dict = {}
    for ood_dataset_name in ood_datasets_list:
        ood_df_dict[ood_dataset_name] = pd.DataFrame(
            experiment[ood_dataset_name], columns=[experiment["x_axis"]]
        )
        ood_df_dict[ood_dataset_name].insert(0, "Dataset", "")
        ood_df_dict[ood_dataset_name].loc[:, "Dataset"] = ood_dataset_name

    all_dfs = [df_pred_h_scores_ind]
    all_dfs.extend(list(ood_df_dict.values()))
    df_pred_h_scores = pd.concat(all_dfs).reset_index(drop=True)

    ax = sns.displot(
        df_pred_h_scores, x=experiment["x_axis"], hue="Dataset", kind="hist", fill=True
    ).set(title=title)
    plt.tight_layout()
    plt.legend(loc="best")
    return ax


def log_evaluate_postprocessors(
    ind_dict: Dict[str, np.ndarray],
    ood_dict: Dict[str, np.ndarray],
    ood_datasets_names: List[str],
    experiment_name_extension: str = "",
    return_density_scores: Optional[str] = None,
    log_step: Optional[int] = None,
    mlflow_logging: bool = False,
    postprocessors=None,
    cfg: DictConfig = None,
) -> Dict[str, Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]]:
    """
    Function that takes as input InD numpy arrays of entropies and one dictionary for all OoD
    datasets and returns LaRED and LaREM results in the form of a pandas dataframe.
    Optionally logs to a running mlflow experiment.

    Args:
        ind_dict: InD data in the format {"train latent_space_means": np.ndarray, "valid latent_space_means": np.ndarray, "train labels": np.ndarray,
         "valid labels": np.ndarray}
        ood_dict: OoD dictionary where keys are the OoD datasets and the values are the
            numpy arrays of latent representations
        ood_datasets_names: List of strings with the names of the OOD datasets
        experiment_name_extension: Extra string to add to the default experiment name, useful for
            PCA experiments
        return_density_scores: return one of the postprocessor density scores for further analysis. Either 'LaRED',
            'LaREM' or 'LaREK'
        log_step: optional step useful for PCA experiments. None if not performing PCA with
            several components
        mlflow_logging: Optionally log to an existing mlflow run
        postprocessors: List of postprocessors to apply to precalculated ls samples.
            Default: ["LaRED", "LaREM", "LaREK"]
        cfg: Configuration class, useful for postprocessor parameters

    Returns:
        Pandas dataframe with results, optionally LaRED density score
    """
    if return_density_scores is not None:
        assert return_density_scores in postprocessors_dict.keys()
    if postprocessors is None:
        postprocessors = postprocessors_dict.keys()

    # Initialize df to store all the results
    overall_metrics_df = pd.DataFrame(
        columns=[
            "auroc",
            "fpr@95",
            "aupr",
            "fpr",
            "tpr",
        ]
    )
    ##############################
    # Calculate scores
    ##############################
    # Initialize dictionaries
    ind_scores_dict = {}
    ood_scores_dict = {}
    for postprocessor in postprocessors:
        # Instantiate postprocessor
        postp_instance = postprocessors_dict[postprocessor](cfg=cfg)
        postp_instance._setup_flag = False
        # Train postprocessor
        postp_instance.setup(
            ind_dict["train latent_space_means"], ind_train_labels=ind_dict["train labels"]
        )
        # InD Inference
        ind_scores_dict[postprocessor] = postp_instance.postprocess(
            ind_dict["valid latent_space_means"], pred_labels=ind_dict["valid labels"]
        )
        # OoD Inference
        ood_scores_dict[postprocessor] = {}
        for ood_dataset_name in ood_datasets_names:
            ood_scores_dict[postprocessor][ood_dataset_name] = postp_instance.postprocess(
                ood_dict[f"{ood_dataset_name} latent_space_means"],
                pred_labels=ood_dict[f"{ood_dataset_name} labels"],
            )

    #########################
    # Prepare logging of results
    postprocessors_experiments = {}
    for ood_dataset_name in ood_datasets_names:
        for postprocessor in postprocessors:
            postprocessors_experiments[f"{ood_dataset_name} {postprocessor}"] = {
                "InD": ind_scores_dict[postprocessor],
                "OoD": ood_scores_dict[postprocessor][ood_dataset_name],
            }

    # Log Results
    for experiment_name, experiment in postprocessors_experiments.items():
        experiment_name = experiment_name + experiment_name_extension
        results_df, results_mlflow = get_auroc_results(
            detect_exp_name=experiment_name,
            ind_samples_scores=experiment["InD"],
            ood_samples_scores=experiment["OoD"],
            return_results_for_mlflow=True,
        )
        # Add OoD dataset to metrics name
        if "PCA" in experiment_name:
            results_mlflow = {
                f"{' '.join(experiment_name.split()[:-1])}_{k}": v
                for k, v in results_mlflow.items()
            }

        else:
            results_mlflow = {f"{experiment_name}_{k}": v for k, v in results_mlflow.items()}
        if mlflow_logging:
            mlflow.log_metrics(results_mlflow, step=log_step)
        # Update overall metrics dataframe
        for result in results_df.index.values:
            overall_metrics_df.loc[result] = results_df.loc[result]

    results = {"results_df": overall_metrics_df}
    if return_density_scores is not None:
        results["InD"] = ind_scores_dict[return_density_scores]
        results["OoD"] = ood_scores_dict[return_density_scores]
    return results


def select_and_log_best_larex(
    overall_metrics_df: pd.DataFrame,
    n_pca_components_list: Union[list, Tuple],
    postprocessor_name: str,
    multiple_ood_datasets_flag: bool,
    log_mlflow: bool = False,
) -> Tuple[float, float, float, int]:
    """
    Takes as input a Dataframe with the columns 'auroc', 'aupr' and 'fpr@95', a list of PCA number
    of components and the name of the technique: either 'LaRED' or 'LaREM', and logs to and
    existing mlflow run the best metrics

    Args:
        overall_metrics_df: Pandas DataFrame with the LaRED or LaREM experiments results
        n_pca_components_list: List with the numbers of PCA components
        postprocessor_name: One of the postprocessor modules in postprocessors_dict.keys()
        multiple_ood_datasets_flag: Flag that indicates whether there are multiple ood datasets or not
        log_mlflow: Log to mlflow boolean flag

    Returns:
        Tuple with the best auroc, aupr, fpr and the N components.
    """
    assert postprocessor_name in postprocessors_dict.keys(), f"Got {postprocessor_name}"
    means_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
    temp_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
    # Calculate mean of no PCA run
    for row_name in overall_metrics_df.index:
        if postprocessor_name in row_name and "anomalies" not in row_name and "PCA" not in row_name:
            temp_df.loc[row_name] = overall_metrics_df.loc[row_name, ["auroc", "fpr@95", "aupr"]]

    means_temp_df = temp_df.mean()
    means_df.loc[postprocessor_name] = means_temp_df

    if multiple_ood_datasets_flag:
        stds_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
        stds_temp_df = temp_df.std()
        stds_df.loc[postprocessor_name] = stds_temp_df

    # Calculate means of PCA runs
    for n_components in n_pca_components_list:
        temp_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
        for row_name in overall_metrics_df.index:
            if (
                postprocessor_name in row_name
                and f"PCA {n_components}" in row_name
                and row_name.split(f"PCA {n_components}")[-1] == ""
            ):
                temp_df.loc[row_name] = overall_metrics_df.loc[
                    row_name, ["auroc", "fpr@95", "aupr"]
                ]

        means_temp_df = temp_df.mean()
        means_df.loc[f"{postprocessor_name} PCA {n_components}"] = means_temp_df

        if multiple_ood_datasets_flag:
            stds_temp_df = temp_df.std()
            stds_df.loc[f"{postprocessor_name} PCA {n_components}"] = stds_temp_df

    best_index = means_df[means_df.auroc == means_df.auroc.max()].index[0]
    # Here we assume the convention that 0 PCA components would mean the no PCA case
    if "PCA" in best_index:
        best_n_comps = int(best_index.split()[-1])
    else:
        best_n_comps = 0

    if log_mlflow:
        mlflow.log_metric(f"{postprocessor_name}_auroc_mean", means_df.loc[best_index, "auroc"])
        mlflow.log_metric(f"{postprocessor_name}_aupr_mean", means_df.loc[best_index, "aupr"])
        mlflow.log_metric(f"{postprocessor_name}_fpr95_mean", means_df.loc[best_index, "fpr@95"])
        mlflow.log_metric(f"Best {postprocessor_name}", best_n_comps)
        if multiple_ood_datasets_flag:
            mlflow.log_metric(f"{postprocessor_name}_auroc_std", stds_df.loc[best_index, "auroc"])
            mlflow.log_metric(f"{postprocessor_name}_aupr_std", stds_df.loc[best_index, "aupr"])
            mlflow.log_metric(f"{postprocessor_name}_fpr95_std", stds_df.loc[best_index, "fpr@95"])
    return (
        means_df.loc[best_index, "auroc"],
        means_df.loc[best_index, "aupr"],
        means_df.loc[best_index, "fpr@95"],
        best_n_comps,
    )


def subset_boxes(
    ind_dict: Dict[str, np.ndarray],
    ood_dict: Dict[str, np.ndarray],
    ind_train_limit: int,
    ood_limit: int,
    random_seed: int,
    ood_names: List[str],
    non_empty_predictions_id: Optional[Dict[str, List]] = None,
    non_empty_predictions_ood: Optional[Dict[str, List]] = None,
) -> Union[
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List], Dict[str, List]],
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
]:
    """
    Function that subsets a given number of box predictions into a smaller number of them, to speed up caluclations
    during evaluation.

    Args:
        ind_dict: InD data dictionary, with the entries 'train latent_space_means' and 'valid latent_space_means'
        ood_dict: OoD data dictionary where each ood dataset is its own key-value pair
        ind_train_limit: Max number of allowed InD train boxes
        ood_limit: Max number of allowed OoD boxes
        random_seed: Random generator seed
        ood_names: List with the names of the OOD datasets
        non_empty_predictions_id: List with the ids of the images that have non-empty predictions in the InD valid dataset
        non_empty_predictions_ood: List with the ids of the images that have non-empty predictions in the OoD datasets

    Returns:
        Tuple of InD and OoD subset dictionaries
    """
    np.random.seed(random_seed)
    # Subset train
    if (
        "train latent_space_means" in ind_dict.keys()
        and ind_dict["train latent_space_means"].shape[0] > ind_train_limit
    ):
        print(
            f"Subsetting train set to {ind_train_limit} from {ind_dict['train latent_space_means'].shape[0]} extracted boxes"
        )
        chosen_idx_train = np.random.choice(
            ind_dict["train latent_space_means"].shape[0], size=ind_train_limit, replace=False
        )
        ind_dict["train latent_space_means"] = ind_dict["train latent_space_means"][
            chosen_idx_train
        ]
        if "train logits" in ind_dict.keys():
            ind_dict["train logits"] = ind_dict["train logits"][chosen_idx_train, :]
        if "train features" in ind_dict.keys():
            ind_dict["train features"] = ind_dict["train features"][chosen_idx_train, :]

    # Subset InD valid to be the same size as the ood length
    if (
        "valid latent_space_means" in ind_dict.keys()
        and ind_dict["valid latent_space_means"].shape[0] > ood_limit
    ):
        non_emp_test = defaultdict(int)
        for im_id in non_empty_predictions_id["valid"]:
            non_emp_test[im_id] += 1
        avg_obj_per_id_img = int(ind_dict["valid latent_space_means"].shape[0] / len(non_emp_test))
        choice_test = np.random.choice(list(non_emp_test.keys()), size=int(ood_limit/avg_obj_per_id_img), replace=False)
        chosen_idx_valid = []
        choice_test = np.delete(choice_test, np.where(choice_test == "default_factory"))
        for i, idx in enumerate(non_empty_predictions_id["valid"]):
            # chosen_idx_valid.extend([idx] * non_emp_test[int(idx)])
            if idx in choice_test:
                chosen_idx_valid.append(i)
        print(
            f"Subsetting valid set to {len(chosen_idx_valid)} from {ind_dict['valid latent_space_means'].shape[0]} extracted boxes"
        )
        # chosen_idx_valid = np.random.choice(
        #     ind_dict["valid latent_space_means"].shape[0], size=ood_limit, replace=False
        # )
        ind_dict["valid latent_space_means"] = ind_dict["valid latent_space_means"][
            chosen_idx_valid
        ]
        if "valid logits" in ind_dict.keys():
            ind_dict["valid logits"] = ind_dict["valid logits"][chosen_idx_valid, :]
        if "valid features" in ind_dict.keys():
            ind_dict["valid features"] = ind_dict["valid features"][chosen_idx_valid, :]
        if non_empty_predictions_id is not None:
            non_empty_predictions_id["valid"] = [
                non_empty_predictions_id["valid"][i] for i in chosen_idx_valid
            ]

    # Subset OoD
    for ood_dataset_name in ood_names:
        data = ood_dict[f"{ood_dataset_name} latent_space_means"]
        if data.shape[0] > ood_limit:
            print(
                f"Subsetting {ood_dataset_name} to {ood_limit} from {data.shape[0]} extracted boxes"
            )
            chosen_idx_ood = np.random.choice(data.shape[0], size=ood_limit, replace=False)
            ood_dict[f"{ood_dataset_name} latent_space_means"] = data[chosen_idx_ood]
            if f"{ood_dataset_name} logits" in ood_dict.keys():
                ood_dict[f"{ood_dataset_name} logits"] = ood_dict[f"{ood_dataset_name} logits"][
                    chosen_idx_ood, :
                ]
            if f"{ood_dataset_name} features" in ood_dict.keys():
                ood_dict[f"{ood_dataset_name} features"] = ood_dict[f"{ood_dataset_name} features"][
                    chosen_idx_ood, :
                ]
            if non_empty_predictions_ood is not None:
                non_empty_predictions_ood[ood_dataset_name] = [
                    non_empty_predictions_ood[ood_dataset_name][i] for i in chosen_idx_ood
                ]

    if non_empty_predictions_id is not None and non_empty_predictions_ood is not None:
        return ind_dict, ood_dict, non_empty_predictions_id, non_empty_predictions_ood
    return ind_dict, ood_dict
