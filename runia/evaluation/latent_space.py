import os
from typing import List, Union, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt

from runia.evaluation.baselines import baseline_name_dict
from runia.evaluation.metrics import (
    get_pred_scores_plots,
    get_auroc_results,
    log_evaluate_postprocessors,
    save_scores_plots,
    save_roc_ood_detector,
    select_and_log_best_larex,
)
from runia.inference.postprocessors import postprocessors_dict
from runia.dimensionality_reduction import apply_pca_ds_split, apply_pca_transform

__all__ = [
    "log_evaluate_larex",
    "plot_roc_curves",
]


def log_evaluate_larex(
    cfg: DictConfig,
    baselines_names: List[str],
    ood_baselines_scores: Dict[str, np.ndarray],
    ind_data_dict: Dict[str, np.ndarray],
    ood_data_dict: Dict[str, np.ndarray],
    mlflow_run_name: str,
    mlflow_logging: bool,
    visualize_score: Union[None, str] = None,
    postprocessors: Union[None, List[str]] = None,
    save_csv: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, float], Dict[str, np.ndarray]]:
    """
    This function performs the evaluation of InD vs OoD for One InD dataset and several OoD datasets, testing the
    LaREx methods and optionally some baseline methods. It can optionally log all results to an MlFlow server.
    This is the recommended setting.

    Args:
        cfg: Configuration object.
        baselines_names: List of baselines names.
        ood_baselines_scores: Dictionary with the OoD baselines in the format {f"{ood_dataset} {baseline}": np.ndarray}
        ind_data_dict: Dictionary with the InD data in the format {"train latent_space_means": np.ndarray, "valid latent_space_means": np.ndarray} for the
         InD samples, and {baseline: np.ndarray} for each baseline.
        ood_data_dict: Dictionary with the OOD data in the format {ood_dataset: np.ndarray}
        mlflow_run_name: String with the name of the MlFlow run.
        mlflow_logging: Boolean indicating whether to log to mlflow
        visualize_score: The name of one postprocessor, to visualize its score in the logs.
            One of ["LaRED", "LaREM", "LaREK"]
        postprocessors: List of postprocessors to apply to precalculated ls samples.
            Default: ["LaRED", "LaREM", "LaREK"]
        save_csv: Boolean indicating whether to save the evaluation results in a csv file.

    Returns:
        Pandas Dataframe with the evaluation metrics
    """
    if postprocessors is None:
        postprocessors = postprocessors_dict.keys()
    if visualize_score is not None:
        assert visualize_score in postprocessors_dict.keys()
    current_date = cfg.log_dir.split("/")[-1]
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
    # Save to a local folder instead
    logs_folder = f"./results_logs/ind_{cfg.ind_dataset}/{mlflow_run_name}"
    if not mlflow_logging:
        os.makedirs(logs_folder, exist_ok=False)
    #################################################################
    # Baselines analysis
    #################################################################
    if len(baselines_names) > 0:
        overall_metrics_df = log_baselines(
            baselines_names=baselines_names,
            ind_dataset=cfg.ind_dataset,
            ind_data_dict=ind_data_dict,
            ood_baselines_scores=ood_baselines_scores,
            ood_datasets=cfg.ood_datasets,
            overall_metrics_df=overall_metrics_df,
            mlflow_logging=mlflow_logging,
            logs_folder=logs_folder,
        )
    ######################################################
    # Evaluate OoD detection methods LaRx
    ######################################################
    print(f"{postprocessors} running...")
    # ################# Perform evaluation with the complete vector of latent representations ############
    results_eval = log_evaluate_postprocessors(
        ind_dict=ind_data_dict,
        ood_dict=ood_data_dict,
        ood_datasets_names=cfg.ood_datasets,
        experiment_name_extension="",
        return_density_scores=visualize_score,
        mlflow_logging=mlflow_logging,
        postprocessors=postprocessors,
        cfg=cfg,
    )
    # Add results to df
    for result in results_eval["results_df"].index.values:
        overall_metrics_df.loc[result] = results_eval["results_df"].loc[result]
    if visualize_score is not None:
        # Plots comparison of densities
        postp_scores_plots_dict = save_scores_plots(
            scores_ind=results_eval["InD"],
            ood_scores_dict=results_eval["OoD"],
            ood_datasets_list=cfg.ood_datasets,
            ind_dataset_name=cfg.ind_dataset,
            post_processor_name=visualize_score,
        )

        for plot_name, plot in postp_scores_plots_dict.items():
            if mlflow_logging:
                mlflow.log_figure(figure=plot.figure, artifact_file=f"figs/{plot_name}.png")
            else:
                plot.figure.savefig(logs_folder + f"/{plot_name}.png")

    # #################### Perform evaluation with PCA reduced vectors #####################
    for n_components in tqdm(cfg.n_pca_components, desc="Evaluating PCA"):
        # Perform PCA dimension reduction
        ind_dict_pca = {}
        pca_ind_train, pca_transformation = apply_pca_ds_split(
            samples=ind_data_dict["train latent_space_means"], nro_components=n_components
        )
        pca_ind_valid = apply_pca_transform(
            ind_data_dict["valid latent_space_means"], pca_transformation
        )
        ind_dict_pca["train latent_space_means"] = pca_ind_train
        ind_dict_pca["valid latent_space_means"] = pca_ind_valid
        if "train labels" in ind_data_dict:
            ind_dict_pca["train labels"] = ind_data_dict["train labels"]
        if "valid labels" in ind_data_dict:
            ind_dict_pca["valid labels"] = ind_data_dict["valid labels"]
        ood_dict_pca = {}
        for ood_dataset_name in cfg.ood_datasets:
            ood_dict_pca[f"{ood_dataset_name} latent_space_means"] = apply_pca_transform(
                ood_data_dict[f"{ood_dataset_name} latent_space_means"], pca_transformation
            )
            if f"{ood_dataset_name} labels" in ood_data_dict:
                ood_dict_pca[f"{ood_dataset_name} labels"] = ood_data_dict[
                    f"{ood_dataset_name} labels"
                ]

        results_eval = log_evaluate_postprocessors(
            ind_dict=ind_dict_pca,
            ood_dict=ood_dict_pca,
            ood_datasets_names=cfg.ood_datasets,
            experiment_name_extension=f" PCA {n_components}",
            return_density_scores=None,
            log_step=n_components,
            mlflow_logging=mlflow_logging,
            postprocessors=postprocessors,
            cfg=cfg,
        )
        # Add results to df
        for result in results_eval["results_df"].index.values:
            overall_metrics_df.loc[result] = results_eval["results_df"].loc[result]

    # Optionally save csv in file
    if save_csv:
        os.makedirs(f"./results_csvs/{cfg.mlflow_experiment_name}", exist_ok=True)
        overall_metrics_df_name = (
            f"./results_csvs/{cfg.mlflow_experiment_name}/{mlflow_run_name}_{current_date}.csv.gz"
        )
        print(f"Saving csv to {overall_metrics_df_name}")
        overall_metrics_df.to_csv(path_or_buf=overall_metrics_df_name, compression="gzip")
        if mlflow_logging:
            mlflow.log_artifact(overall_metrics_df_name)

    # Get best postprocessors:
    best_postprocessors_dict = _get_best_postprocessors_metrics(
        baselines_names=baselines_names,
        overall_metrics_df=overall_metrics_df,
        mlflow_logging=mlflow_logging,
        postprocessors=postprocessors,
        n_pca_components=cfg.n_pca_components,
        ood_datasets_names=cfg.ood_datasets,
    )
    print(
        f"Best postprocessors metrics: { {k: v for k, v in best_postprocessors_dict.items() if k != 'best'} }"
    )
    # Get thresholds for binary classification
    postprocessor_thresholds, ood_data_dict = _get_best_post_processor_thresholds(
        postprocessors_names=postprocessors,
        best_postprocessors_dict=best_postprocessors_dict,
        cfg=cfg,
        ind_data=ind_data_dict,
        ood_data=ood_data_dict,
        logs_folder=logs_folder,
        log_mlflow=mlflow_logging,
    )
    print(f"Best postprocessor thresholds: {postprocessor_thresholds}")

    # Plot the postprocessors ROC curves
    plot_roc_curves(
        ood_datasets=cfg.ood_datasets,
        postprocessors=postprocessors,
        overall_metrics_df=overall_metrics_df,
        best_postprocessors_dict=best_postprocessors_dict,
        mlflow_logging=mlflow_logging,
        ind_dataset=cfg.ind_dataset,
        logs_folder=logs_folder,
        baselines_names=baselines_names,
    )
    return overall_metrics_df, best_postprocessors_dict, postprocessor_thresholds, ood_data_dict


def log_baselines(
    baselines_names: List[str],
    ind_dataset: str,
    ind_data_dict: Dict[str, np.ndarray],
    ood_baselines_scores: Dict[str, np.ndarray],
    ood_datasets: List[str],
    overall_metrics_df: pd.DataFrame,
    mlflow_logging: bool,
    logs_folder: str,
) -> pd.DataFrame:
    """
    Log baselines if previously calculated.

    Args:
        baselines_names: List of strings with the names of the baselines
        ind_dataset: String with the name of the InD dataset
        ind_data_dict: Dictionary with the train and valid InD samples
        ood_baselines_scores: Dictionary with the name of each baseline and the ood_dataset and their scores
        ood_datasets: List of strings with the names of the OOD datasets
        overall_metrics_df: Pandas Dataframe with the compatible formatting
        mlflow_logging: Whether to log to mlflow
        logs_folder: If not logging to mlflow, string with the folder to store plots and metrics

    Returns:
        Pandas DataFrame with the updated fields for the baselines
    """
    print("Logging baselines")
    # Dictionary that defines experiments names, InD and OoD datasets
    # We use some negative uncertainty scores to align with the convention that positive
    # (in-distribution) samples have higher scores (see plots)
    baselines_experiments = {}
    for baseline in baselines_names:
        for ood_dataset in ood_datasets:
            if baseline == "pred_h" or baseline == "mi":
                baselines_experiments[f"{ood_dataset} {baseline}"] = {
                    "InD": -ind_data_dict[baseline],
                    "OoD": -ood_baselines_scores[f"{ood_dataset} {baseline}"],
                }
            else:
                baselines_experiments[f"{ood_dataset} {baseline}"] = {
                    "InD": ind_data_dict[baseline],
                    "OoD": ood_baselines_scores[f"{ood_dataset} {baseline}"],
                }

    baselines_plots = {}
    for baseline in baselines_names:
        baselines_plots[baseline_name_dict[baseline]["plot_title"]] = {
            "InD": ind_data_dict[baseline]
        }
        baselines_plots[baseline_name_dict[baseline]["plot_title"]]["x_axis"] = baseline_name_dict[
            baseline
        ]["x_axis"]
        baselines_plots[baseline_name_dict[baseline]["plot_title"]]["plot_name"] = (
            baseline_name_dict[baseline]["plot_name"]
        )
        for ood_dataset in ood_datasets:
            baselines_plots[baseline_name_dict[baseline]["plot_title"]][ood_dataset] = (
                ood_baselines_scores[f"{ood_dataset} {baseline}"]
            )

    # Make all baselines plots
    for plot_title, experiment in tqdm(baselines_plots.items(), desc="Plotting baselines"):
        # Plot score values predictive entropy
        pred_score_plot = get_pred_scores_plots(
            experiment, ood_datasets, title=plot_title, ind_dataset_name=ind_dataset
        )
        if mlflow_logging:
            mlflow.log_figure(
                figure=pred_score_plot.figure,
                artifact_file=f"figs/{experiment['plot_name']}.png",
            )
        else:
            pred_score_plot.figure.savefig(logs_folder + f"/{experiment['plot_name']}.png")

    # Log all baselines experiments
    for experiment_name, experiment in tqdm(
        baselines_experiments.items(), desc="Logging baselines"
    ):
        results_df, results_mlflow = get_auroc_results(
            detect_exp_name=experiment_name,
            ind_samples_scores=experiment["InD"],
            ood_samples_scores=experiment["OoD"],
            return_results_for_mlflow=True,
        )
        results_mlflow = dict([(f"{experiment_name}_{k}", v) for k, v in results_mlflow.items()])
        if mlflow_logging:
            mlflow.log_metrics(results_mlflow)
        # Plot each ROC curve individually LEAVE COMMENTED
        # roc_curve = save_roc_ood_detector(
        #     results_table=r_df,
        #     plot_title=f"ROC {cfg.ind_dataset} vs {experiment_name} {cfg.layer_type} layer"
        # )
        # mlflow.log_figure(figure=roc_curve,
        #                   artifact_file=f"figs/roc_{experiment_name}.png")
        # END COMMENTED SECTION
        for results in results_df.index.values:
            overall_metrics_df.loc[results] = results_df.loc[results]

    return overall_metrics_df


def plot_roc_curves(
    ood_datasets: List[str],
    postprocessors: List[str],
    overall_metrics_df: pd.DataFrame,
    best_postprocessors_dict: Dict,
    mlflow_logging: bool,
    ind_dataset: str,
    logs_folder: str,
    baselines_names: List[str],
) -> None:
    """
    This function takes as input the overall datasets dataframe and plots ROC curves for the best postprocessors and
    for the PCA evaluation of postprocessors

    Args:
        ood_datasets: List of strings with the names of the OoD datasets
        postprocessors: List of strings with the names of the postprocessors
        overall_metrics_df: Pandas dataframe with the results of previous evaluation of postprocessors
        best_postprocessors_dict: Dictionary that comes from the `get_best_postprocessors_metrics` function
        mlflow_logging: Whether to log to mlflow
        ind_dataset: String with the name of the InD dataset
        logs_folder: If not logging to mlflow, the plots will be saved here
        baselines_names: List of strings with the names of the baselines

    """
    # Plot Roc curves together, by OoD dataset
    dfs_dict = {}
    for ood_dataset in ood_datasets:
        dfs_dict["base"] = pd.DataFrame(
            columns=[
                "auroc",
                "fpr@95",
                "aupr",
                "fpr",
                "tpr",
            ]
        )
        for postprocessor in postprocessors:
            dfs_dict[postprocessor] = pd.DataFrame(
                columns=[
                    "auroc",
                    "fpr@95",
                    "aupr",
                    "fpr",
                    "tpr",
                ]
            )
        for row_name in overall_metrics_df.index:
            # Log baselines and methods other than LaRED or LaREM that use the whole latent space without PCA
            if ood_dataset in row_name and (
                row_name in best_postprocessors_dict["best"]
                or row_name.split(f"{ood_dataset} ")[-1] in baselines_names
            ):
                dfs_dict["base"].loc[row_name] = overall_metrics_df.loc[row_name]
                dfs_dict["base"].rename(
                    index={row_name: row_name.split(ood_dataset)[1]}, inplace=True
                )
            for postprocessor in postprocessors:
                # Log postprocessor with PCA
                if ood_dataset in row_name and "PCA" in row_name and postprocessor in row_name:
                    dfs_dict[postprocessor].loc[row_name] = overall_metrics_df.loc[row_name]
                    dfs_dict[postprocessor].rename(
                        index={row_name: row_name.split(ood_dataset)[1]}, inplace=True
                    )
        # Plot ROC curve baselines and methods other than LaRED or LaREM that use the whole latent space without PCA
        roc_curve = save_roc_ood_detector(
            results_table=dfs_dict["base"],
            plot_title=f"ROC {ind_dataset} vs {ood_dataset}",
            postprocessors=postprocessors,
        )
        if mlflow_logging:
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve, artifact_file=f"figs/roc_{ood_dataset}.png")
        else:
            roc_curve.savefig(logs_folder + f"/roc_{ood_dataset}.png")

        for postprocessor in postprocessors:
            roc_curve_pca_postp = save_roc_ood_detector(
                results_table=dfs_dict[postprocessor],
                plot_title=f"ROC {ind_dataset} vs {ood_dataset} {postprocessor} PCA",
                postprocessors=postprocessors,
            )
            if mlflow_logging:
                # Log the plot with mlflow
                mlflow.log_figure(
                    figure=roc_curve_pca_postp,
                    artifact_file=f"figs/roc_{ood_dataset}_pca_{postprocessor}.png",
                )
            else:
                roc_curve_pca_postp.savefig(
                    logs_folder + f"/roc_{ood_dataset}_pca_{postprocessor}.png"
                )


def _get_best_postprocessors_metrics(
    baselines_names: List[str],
    overall_metrics_df: pd.DataFrame,
    mlflow_logging: bool,
    postprocessors: List[str],
    n_pca_components: List[int],
    ood_datasets_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Gets the best postprocessors metrics across multiple baselines and out-of-distribution (OoD) datasets.
    The function computes various performance metrics such as AUROC, AUPR, and FPR@95 for the baselines
    and postprocessors provided. It also optionally logs these metrics using MLflow for tracking.
    This helps in evaluating the global performance of baselines and postprocessors across OoD datasets.

    Args:
        baselines_names: List of baseline names to evaluate.
        overall_metrics_df: Dataframe containing overall performance metrics.
        mlflow_logging: Boolean flag to indicate whether metrics should be logged with MLflow or not.
        postprocessors: List of postprocessing techniques to evaluate.
        n_pca_components: List of PCA components to be considered for dimensionality reduction.
        ood_datasets_names: List of out-of-distribution datasets used in evaluation.

    Returns:
        A dictionary where each key represents a postprocessor or aggregated results, mapping to another
        dictionary containing the best configurations and associated performance metrics.
    """
    # Check if there are multiple ood datasets
    mutliple_ood_datasets_flag = len(ood_datasets_names) > 1
    # We collect all metrics to estimate global performance of all metrics
    all_aurocs = []
    all_auprs = []
    all_fprs = []
    if len(baselines_names) > 0:
        # Extract mean for each baseline across datasets
        for baseline in baselines_names:
            temp_df = pd.DataFrame(
                columns=[
                    "auroc",
                    "fpr@95",
                    "aupr",
                    "fpr",
                    "tpr",
                ]
            )
            for row_name in overall_metrics_df.index:
                if baseline in row_name:
                    temp_df.loc[row_name] = overall_metrics_df.loc[row_name]
                    temp_df.rename(index={row_name: row_name.split(baseline)[0]}, inplace=True)

            all_aurocs.append(temp_df["auroc"].mean())
            all_auprs.append(temp_df["aupr"].mean())
            all_fprs.append(temp_df["fpr@95"].mean())
            if mlflow_logging:
                mlflow.log_metric(f"{baseline}_auroc_mean", temp_df["auroc"].mean())
                mlflow.log_metric(f"{baseline}_auroc_std", temp_df["auroc"].std())
                mlflow.log_metric(f"{baseline}_aupr_mean", temp_df["aupr"].mean())
                mlflow.log_metric(f"{baseline}_aupr_std", temp_df["aupr"].std())
                mlflow.log_metric(f"{baseline}_fpr95_mean", temp_df["fpr@95"].mean())
                mlflow.log_metric(f"{baseline}_fpr95_std", temp_df["fpr@95"].std())

    # Extract mean for best postprocessors across OoD datasets
    best_postprocessors_dict = {"best": []}
    for postprocessor in postprocessors:
        best_postprocessors_dict[postprocessor] = {}
        auroc, aupr, fpr, best_comp = select_and_log_best_larex(
            overall_metrics_df,
            n_pca_components,
            postprocessor_name=postprocessor,
            log_mlflow=mlflow_logging,
            multiple_ood_datasets_flag=mutliple_ood_datasets_flag,
        )
        if best_comp == 0:
            best_postprocessors_dict[postprocessor]["best_comp"] = f"{postprocessor}"
        else:
            best_postprocessors_dict[postprocessor][
                "best_comp"
            ] = f"{postprocessor} PCA {best_comp}"
        best_postprocessors_dict[postprocessor]["auroc"] = auroc
        best_postprocessors_dict[postprocessor]["aupr"] = aupr
        best_postprocessors_dict[postprocessor]["fpr"] = fpr
        all_aurocs.append(auroc)
        all_auprs.append(aupr)
        all_fprs.append(fpr)
        for ood_dataset in ood_datasets_names:
            best_postprocessors_dict["best"].append(
                f"{ood_dataset} {best_postprocessors_dict[postprocessor]['best_comp']}"
            )

    # Log average performances across OoD datasets
    if mlflow_logging and len(ood_datasets_names) > 1:
        mlflow.log_metric(f"global_auroc_mean", np.mean(all_aurocs))
        mlflow.log_metric(f"global_auroc_std", np.std(all_aurocs))
        mlflow.log_metric(f"global_aupr_mean", np.mean(all_auprs))
        mlflow.log_metric(f"global_aupr_std", np.std(all_auprs))
        mlflow.log_metric(f"global_fpr_mean", np.mean(all_fprs))
        mlflow.log_metric(f"global_fpr_std", np.std(all_fprs))

    return best_postprocessors_dict


def _get_best_post_processor_thresholds(
    postprocessors_names: List,
    best_postprocessors_dict: Dict,
    cfg: DictConfig,
    ind_data: Dict[str, np.ndarray],
    ood_data: Dict[str, np.ndarray],
    logs_folder: str,
    log_mlflow: bool = False,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    This function takes as input the best post processors dictionary obtained with the `get_best_postprocessors_metrics`
    function and returns a dictionary with the best post processor thresholds. A forward pass through the
    train and test InD samples is necessary.

    Args:
        postprocessors_names: List with the post processors names.
        best_postprocessors_dict: Dictionary obtained form the `get_best_postprocessors_metrics` function.
        cfg: Config class from config file
        ind_data: In-Distribution data dictionary in the format {'train': np.ndarray, 'valid': np.ndarray}
        ood_data: Out-of-Distribution data dictionary in the format {'ood1': np.ndarray,..., 'oodn': np.ndarray}
        logs_folder: Folder where the logs are stored if not logging to mlflow
        log_mlflow: Boolean that indicates whether logging to mlflow.

    Returns:
        Dictionary with the best postprocessors thresholds,
        and a dictionary with the postprocessors scores attached to the ood datasets
    """
    postprocessor_thresholds = {}
    for postprocessor_name in postprocessors_names:
        train_data = ind_data["train latent_space_means"].copy()
        valid_data = ind_data["valid latent_space_means"].copy()
        pca_transformation = None
        postp_instance = postprocessors_dict[postprocessor_name](cfg=cfg)
        postp_instance._setup_flag = False
        best_postp = best_postprocessors_dict[postprocessor_name]["best_comp"]
        if "PCA" in best_postp:
            n_pca_comps = int(best_postp.split("PCA")[1])
            train_data, pca_transformation = apply_pca_ds_split(
                samples=train_data, nro_components=n_pca_comps
            )
        # Train postprocessor
        postp_instance.setup(train_data, ind_train_labels=ind_data["train labels"])
        # InD Inference
        if "PCA" in best_postp:
            valid_data = apply_pca_transform(valid_data, pca_transformation)
        ind_valid_postp = postp_instance.postprocess(
            valid_data, pred_labels=ind_data["valid labels"]
        )
        mean_ind_valid, std_ind_valid = np.mean(ind_valid_postp), np.std(ind_valid_postp)
        # Here we use the 95% confidence z score
        threshold_postp = mean_ind_valid - (1.645 * std_ind_valid)
        postprocessor_thresholds[best_postp] = threshold_postp
        ############################
        # Plot InD and OoD scores distribution along with 95% InD samples threshold
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(ind_valid_postp, bins=100, label="InD valid set", alpha=0.4)
        # Visualize OoD samples at the same time
        for ood_dataset_name in cfg.ood_datasets:
            ood_dataset = ood_data[f"{ood_dataset_name} latent_space_means"].copy()
            if "PCA" in best_postp:
                ood_dataset = apply_pca_transform(ood_dataset, pca_transformation)
            ood_postp = postp_instance.postprocess(
                ood_dataset, pred_labels=ood_data[f"{ood_dataset_name} labels"]
            )
            ood_data[f"{ood_dataset_name} {best_postp}"] = ood_postp
            ax.hist(ood_postp, bins=100, label=f"OoD {ood_dataset_name} ", alpha=0.4)
        ax.vlines(
            x=threshold_postp,
            ymin=0,
            ymax=ax.dataLim.bounds[3],
            colors="r",
            label=f"95% threshold={round(threshold_postp, 2)}",
        )
        ax.legend()
        ax.set_xlabel("Score")
        ax.set_ylabel("Frquency")
        ax.set_title(f"Empirical {best_postp} score distribution")
        if log_mlflow:
            mlflow.log_metric(f"Threshold_{best_postp}", threshold_postp)
            # Plot empirical score distribution and threshold
            mlflow.log_figure(figure=fig, artifact_file=f"figs/{best_postp}_score_threshold.png")
        else:
            fig.savefig(f"{logs_folder}/{best_postp}_score_threshold.png")

    return postprocessor_thresholds, ood_data
