"""Module containing baselines scripts for OoD detection"""

from . import from_model_inference
from . import from_precalculated

from .from_model_inference import *
from .from_precalculated import *


baseline_name_dict = {
    "pred_h": {
        "plot_title": "Predictive H distribution",
        "x_axis": "Predictive H score",
        "plot_name": "pred_h",
    },
    "mi": {
        "plot_title": "Predictive MI distribution",
        "x_axis": "Predictive MI score",
        "plot_name": "pred_mi",
    },
    "msp": {
        "plot_title": "Predictive MSP distribution",
        "x_axis": "Predictive MSP score",
        "plot_name": "pred_msp",
    },
    "energy": {
        "plot_title": "Predictive energy score distribution",
        "x_axis": "Predictive energy score",
        "plot_name": "pred_energy",
    },
    "mdist": {
        "plot_title": "Mahalanobis Distance distribution",
        "x_axis": "Mahalanobis Distance score",
        "plot_name": "pred_mdist",
    },
    "knn": {
        "plot_title": "kNN distance distribution",
        "x_axis": "kNN Distance score",
        "plot_name": "pred_knn",
    },
    "ash": {
        "plot_title": "ASH score distribution",
        "x_axis": "ASH score",
        "plot_name": "ash_score",
    },
    "dice": {
        "plot_title": "DICE score distribution",
        "x_axis": "DICE score",
        "plot_name": "dice_score",
    },
    "react": {
        "plot_title": "ReAct score distribution",
        "x_axis": "ReAct score",
        "plot_name": "react_score",
    },
    "dice_react": {
        "plot_title": "DICE + ReAct score distribution",
        "x_axis": "DICE + ReAct score",
        "plot_name": "dice_react_score",
    },
    "vim": {
        "plot_title": "ViM score distribution",
        "x_axis": "ViM score",
        "plot_name": "vim_score",
    },
    "gen": {
        "plot_title": "GEN score distribution",
        "x_axis": "GEN score",
        "plot_name": "gen_score",
    },
    "ddu": {
        "plot_title": "DDU score distribution",
        "x_axis": "DDU score",
        "plot_name": "ddu_score",
    },
    "raw": {
        "plot_title": "Raw predictions",
        "x_axis": "Raw predictions",
        "plot_name": "raw_predictions",
    }
}


__all__ = ["baseline_name_dict"]
__all__ += from_model_inference.__all__
__all__ += from_precalculated.__all__
