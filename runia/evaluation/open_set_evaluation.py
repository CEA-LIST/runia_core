import json
from collections import defaultdict
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import softmax
import numpy as np
import torch
from tqdm import tqdm

__all__ = [
    "COCOParser",
    "OpenSetEvaluator",
    "evaluate_open_set_detection_one_method",
    "get_overall_open_set_results",
    "convert_osod_results_to_pandas_df",
    "convert_osod_results_to_hierarchical_pandas_df",
    "convert_osod_results_for_mlflow_logging",
    "plot_two_osod_datasets_metrics",
    "plot_two_osod_datasets_per_metric",
]


class COCOParser:
    """
    Parses a COCO (Common Objects in Context) dataset JSON file and provides
    methods for querying annotations, images, categories, licenses, and other
    structured information. Designed to handle datasets adhering to the COCO
    format.

    Attributes:
        annIm_dict (defaultdict): A dictionary mapping image IDs to their
            corresponding annotations.
        cat_dict (dict): A dictionary mapping category IDs to category
            metadata, including count of annotations for that category.
        categories_original (dict): A copy of the original categories section
            from the COCO file.
        annId_dict (dict): A dictionary mapping annotation IDs to annotation
            data.
        im_dict (dict): A dictionary mapping image IDs to image metadata.
        licenses_dict (dict): A dictionary holding license information if
            available in the dataset file.
        info_dict (dict): A dictionary holding detailed dataset information if
            available in the COCO file.
    """

    def __init__(self, anns_file: str, using_subset: Optional[List[Union[str, int]]] = False):
        """
        Initializes and processes the COCO (Common Objects in Context) data structure by parsing
        the input JSON file, creating dictionaries for images, annotations, and categories.

        The constructor reads and processes the COCO JSON file, storing necessary mappings for
        categories, annotations, and images either for a subset of the dataset (if provided)
        or for the entire dataset.

        Args:
            anns_file (str): Path to the JSON file containing COCO dataset annotations.
            using_subset (Optional[List[Union[str, int]]], optional): List of image IDs to restrict
                processing to a subset of the dataset. If False or not provided, the entire dataset
                is processed.
        """
        with open(anns_file, "r") as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        # Dict of id: category pairs
        self.cat_dict = {}
        # Dict of original categories (copy of original entry)
        self.categories_original = {"categories": coco["categories"]}
        self.annId_dict = {}
        self.im_dict = {}
        if "licenses" in coco:
            self.licenses_dict = {"licenses": coco["licenses"]}
        else:
            self.licenses_dict = {}
        if "info" in coco:
            self.info_dict = {"info": coco["info"]}
        else:
            self.info_dict = {}
        for cat in coco["categories"]:
            self.cat_dict[cat["id"]] = cat
            self.cat_dict[cat["id"]]["count"] = 0
        for ann in coco["annotations"]:
            if using_subset and ann["image_id"] in using_subset:
                self.annIm_dict[ann["image_id"]].append(ann)
                self.annId_dict[ann["id"]] = ann
                self.cat_dict[ann["category_id"]]["count"] += 1
            elif not using_subset:
                self.annIm_dict[ann["image_id"]].append(ann)
                self.annId_dict[ann["id"]] = ann
                self.cat_dict[ann["category_id"]]["count"] += 1
        for img in coco["images"]:
            if using_subset and img["id"] in using_subset:
                self.im_dict[img["id"]] = img
            elif not using_subset:
                self.im_dict[img["id"]] = img

        # Licenses not actually needed per image
        # for license in coco['licenses']:
        #     self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        """
        Retrieves a list of all image IDs available in the current image dictionary.

        Returns:
            list: A list containing the image IDs as keys from the image dictionary.
        """
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids: Union[List, int, str]) -> List[int]:
        """
        Fetches the annotation IDs for a given list of image IDs, a single image ID, or a string representation
        of an image ID. The function ensures compatibility for various input types and returns a list of
        associated annotation IDs, maintaining consistency across multiple formats of input.

        Args:
            im_ids (Union[List, int, str]): A single image ID, string representation of an image ID,
                or a list of image IDs, for which annotation IDs are to be fetched.

        Returns:
            List[int]: A list of annotation IDs associated with the given image ID(s).
        """
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann["id"] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids: Union[List[int], int]) -> List[Dict]:
        """
        Loads annotations based on the provided annotation IDs.

        This method retrieves annotations corresponding to the given annotation IDs,
        returning the annotations as a list of dictionaries. If a single annotation ID
        is provided, it is internally handled as a list with one element.

        Args:
            ann_ids (Union[List[int], int]): List of annotation IDs or a single
                annotation ID.

        Returns:
            List[Dict]: A list of annotations corresponding to the provided IDs.
        """
        im_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def load_cats(self, class_ids: Union[List[int], int]) -> List[Dict]:
        """
        Loads category information for the specified class IDs.

        Args:
            class_ids (Union[List[int], int]): A single class ID or a list of class IDs for which
                category information needs to be loaded.

        Returns:
            List[Dict]: A list of dictionaries where each dictionary contains the category
                information for the provided class IDs.
        """
        class_ids = class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self, im_ids: Union[List, int, str]) -> List[Dict]:
        """
        Retrieves the license details for the given images.

        This method fetches the license information for the provided image ID(s)
        using the internal dictionaries that map image IDs to license IDs, and
        license IDs to license information.

        Args:
            im_ids: A single image ID or a list of image IDs for which license
                details are to be retrieved.

        Returns:
            A list of dictionaries, where each dictionary contains the license
            details corresponding to the provided image IDs.
        """
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]

    def get_img_info(self, im_ids: Union[List, int, str]) -> List[Dict]:
        """
        Retrieves information for the specified image ID(s) from the internal image data.

        This function takes a single image ID, a list of image IDs, or a string representing
        an image ID, and returns a list of dictionaries containing information about the
        specified images. The information is fetched from an existing internal mapping of
        image IDs to their corresponding data.

        Args:
            im_ids (Union[List, int, str]): The ID or a collection of IDs for which image
                information is to be retrieved. This can be provided as a single integer,
                string, or a list of either.

        Returns:
            List[Dict]: A list of dictionaries containing information corresponding to each
                requested image ID.
        """
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [self.im_dict[im_id] for im_id in im_ids]

    def get_img_ids_per_cat_name(self, cat_name: str) -> List:
        """
        Retrieve a list of unique image IDs corresponding to a given category name.

        Args:
            cat_name (str): The name of the category to fetch the respective image IDs.

        Returns:
            List: A list of unique image IDs associated with the specified category name.
        """
        cat_id = [cat["id"] for cat in self.cat_dict.values() if cat["name"] == cat_name][0]
        return list(
            set(
                [
                    ann["image_id"]
                    for ann in self.annId_dict.values()
                    if ann["category_id"] == cat_id
                ]
            )
        )


class OpenSetEvaluator:
    """Handles open-set evaluation of object detection models.

    This class is designed to provide functionality for evaluating object detection
    models with open-set recognition capabilities. The evaluation is performed
    using detection metrics and considers both known and unknown classes.

    Attributes:
        _dataset_name (str): Name of the in-domain dataset.
        _class_names (List[str]): List of class names excluding "unknown".
        total_num_class (int): Total number of classes including the "unknown" class.
        unknown_class_index (int): Index of the "unknown" class.
        num_known_classes (int): Number of known classes in the dataset.
        known_classes (List[str]): List of known class names.
        _is_2007 (bool): Whether to use the 2007 metric for evaluation.
    """

    def __init__(self, id_dataset_name: str, ground_truth_annotations_path: str, metric_2007: bool):
        """
        Initializes the class with dataset metadata, ground truth annotations, and a metric option
        indicating whether to use the 2007 evaluation metric. Processes the provided annotations
        to compute class names, define the number of known and unknown classes, and sets up class
        attributes for later use.

        Args:
            id_dataset_name (str): Unique name of the dataset.
            ground_truth_annotations_path (str): File path to the ground truth annotations.
            metric_2007 (bool): Indicates whether to use the 2007 metric for evaluation.
        """
        ground_truth_annotations = COCOParser(ground_truth_annotations_path)
        self._dataset_name = id_dataset_name
        self._class_names = [cat["name"] for cat in ground_truth_annotations.cat_dict.values()] + [
            "unknown"
        ]
        self.total_num_class = len(ground_truth_annotations.cat_dict) + 1
        self.unknown_class_index = self.total_num_class - 1
        self.num_known_classes = len(ground_truth_annotations.cat_dict)
        self.known_classes = self._class_names[: self.num_known_classes]
        self._is_2007 = metric_2007
        self._predictions = defaultdict(list)

    def reset(self):
        """
        Resets the internal state of the predictions object.

        Clears all stored predictions by resetting the internal data structure
        used to store them. This method is useful for reinitializing the state
        of the object if subsequent predictions need to be stored from scratch.

        """
        # class name -> list of prediction strings
        self._predictions = defaultdict(list)

    def process(
        self, image_id: Union[str, int], boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray
    ) -> None:
        """
        Processes the detection outputs by converting bounding box values, appending formatted
        detections to the internal predictions for each detected object class.

        Args:
            image_id (Union[str, int]): The unique identifier for the image.
            boxes (np.ndarray): Array of bounding boxes for detected objects. Each box is defined by
                [xmin, ymin, xmax, ymax].
            scores (np.ndarray): Array of confidence scores for the detected objects.
            classes (np.ndarray): Array of class labels for the detected objects.

        Returns:
            None
        """
        for box, score, cls in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            # The inverse of data loading logic in `datasets/pascal_voc.py`
            xmin += 1
            ymin += 1
            self._predictions[cls].append(
                f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            )

    def evaluate(
        self,
        test_annotations_path: str,
        is_ood: bool,
        get_known_classes_metrics: bool,
        using_subset: Optional[List[Union[str, int]]] = False,
    ) -> Dict[str, float]:
        """
        Evaluates predictions against ground truth annotations and computes various metrics.

        This function reads annotations, evaluates predictions per class using a VOC evaluation
        method, and calculates metrics such as mean Average Precision (mAP), Open Set Error (OSE),
        and related metrics for known and unknown classes.

        Args:
            test_annotations_path (str): Path to the file containing test annotations
                in COCO format.
            is_ood (bool): Indicates whether all objects in the evaluated dataset are out-of-distribution (OOD).
            get_known_classes_metrics (bool): Specifies if metrics for known classes should
                be computed and included in the results.
            using_subset (Optional[List[Union[str, int]]], optional): A subset of image ids
                from the annotations to use during evaluation. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary containing the computed evaluation metrics. Each key
            corresponds to a specific metric, and the values are rounded to 3 decimal places.
        """
        # Read annotations file
        test_annotations = COCOParser(test_annotations_path, using_subset)
        # Get the predictions per class
        predictions = defaultdict(list)
        for clsid, lines in self._predictions.items():
            predictions[clsid].extend(lines)

        aps = defaultdict(list)  # iou -> ap per class
        recs = defaultdict(list)
        precs = defaultdict(list)
        all_recs = defaultdict(list)
        all_precs = defaultdict(list)
        unk_det_as_knowns = defaultdict(list)
        num_unks = defaultdict(list)
        tp_plus_fp_cs = defaultdict(list)
        fp_os = defaultdict(list)

        for cls_id, cls_name in enumerate(self._class_names):
            lines = predictions.get(cls_id, [""])

            for thresh in [
                50,
            ]:
                # for thresh in range(50, 100, 5):
                (
                    rec,
                    prec,
                    ap,
                    unk_det_as_known,
                    num_unk,
                    tp_plus_fp_closed_set,
                    fp_open_set,
                ) = voc_eval(
                    lines,
                    test_annotations,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                    is_ood=is_ood,
                )
                aps[thresh].append(ap * 100)
                unk_det_as_knowns[thresh].append(unk_det_as_known)
                num_unks[thresh].append(num_unk)
                all_precs[thresh].append(prec)
                all_recs[thresh].append(rec)
                tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
                fp_os[thresh].append(fp_open_set)
                try:
                    recs[thresh].append(rec[-1] * 100)
                    precs[thresh].append(prec[-1] * 100)
                except:
                    recs[thresh].append(0)
                    precs[thresh].append(0)

        results_2d = {}
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        if get_known_classes_metrics:
            results_2d["mAP"] = mAP[50]

        wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os)
        results_2d["WI"] = wi[0.8][50] * 100

        total_num_unk_det_as_known = {iou: np.sum(x) for iou, x in unk_det_as_knowns.items()}
        # total_num_unk = num_unks[50][0]
        # self.logger.info('num_unk ' + str(total_num_unk))
        results_2d["AOSE"] = total_num_unk_det_as_known[50]
        if num_unk > 0:
            results_2d["nOSE"] = round(total_num_unk_det_as_known[50] * 100 / num_unk, 3)
            # Get error counts
            if is_ood:
                total_fp_ood = 0
                for cls in tp_plus_fp_cs[50]:
                    if cls is not None and len(cls) > 0:
                        total_fp_ood += cls.max()
                # Error of detecting an unlabeled section as a known class
                results_2d["E_BK"] = total_fp_ood - total_num_unk_det_as_known[50]

        else:
            results_2d["nOSE"] = 0.0

        # Known
        if get_known_classes_metrics:
            results_2d.update(
                {
                    "AP_K": np.mean(aps[50][: self.num_known_classes]),
                    "P_K": np.mean(precs[50][: self.num_known_classes]),
                    "R_K": np.mean(recs[50][: self.num_known_classes]),
                }
            )

        # Unknown
        results_2d.update(
            {
                "AP_U": np.mean(aps[50][-1]),
                "P_U": np.mean(precs[50][-1]),
                "R_U": np.mean(recs[50][-1]),
            }
        )
        results_head = list(results_2d.keys())
        results_data = [[float(results_2d[k]) for k in results_2d]]

        return {metric: round(x, 3) for metric, x in zip(results_head, results_data[0])}

    def compute_WI_at_many_recall_level(
        self, recalls: Dict[int, List], tp_plus_fp_cs: Dict[int, List], fp_os: Dict[int, List]
    ) -> Dict[float, Dict[int, float]]:
        """
        Computes WI (Wilderness Impact) at multiple recall levels.

        This method calculates the wilderness impact (WI) at multiple predefined recall
        levels by invoking the `compute_WI_at_a_recall_level` function for each recall level.
        The calculated WI values are stored in a dictionary, with recall levels as keys and
        a nested dictionary containing WI values for different classes as values.

        Args:
            recalls (Dict[int, List]): A dictionary where the key is the class ID and the
                value is a list representing recall values for the corresponding class.
            tp_plus_fp_cs (Dict[int, List]): A dictionary where the key is the class ID and
                the value is a list representing the true positive counts and false positive
                counts for the corresponding class.
            fp_os (Dict[int, List]): A dictionary where the key is the class ID and the
                value is a list representing the false positives for the corresponding class.

        Returns:
            Dict[float, Dict[int, float]]: A dictionary where recall levels are the keys, and
            values are nested dictionaries. Each nested dictionary has class IDs as keys
            and their corresponding wilderness impact (WI) values as values.

        """
        wi_at_recall = {}
        # for r in range(1, 10):
        for r in [8]:
            r = r / 10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(
        self,
        recalls: Dict[int, List],
        tp_plus_fp_cs: Dict[int, List],
        fp_os: Dict[int, List],
        recall_level: float = 0.5,
    ) -> Dict[int, float]:
        """
        Computes the WI (Wilderness Impact) at a specified recall level for a set of IoUs.

        This function calculates the wilderness impact for specific recall levels across multiple
        Intersection over Union (IoU) thresholds. The WI metric is determined by comparing false positives
        to the sum of true positives and false positives for known classes only. If no data is available
        for a specific IoU, the result will default to 0 for that IoU.

        Args:
            recalls (Dict[int, List]): A dictionary where keys are IoU levels, and values are lists containing
                recall values for each class.
            tp_plus_fp_cs (Dict[int, List]): A dictionary where keys are IoU levels, and values are lists of
                the counts of true positives plus false positives for each class.
            fp_os (Dict[int, List]): A dictionary where keys are IoU levels, and values are lists of false
                positives across different classes.
            recall_level (float): The recall level at which the WI needs to be computed. Default is 0.5.

        Returns:
            Dict[int, float]: A dictionary where keys are IoU thresholds, and values are the corresponding
                WI values at the specified recall level.
        """
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_known_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou


def voc_eval(
    predictions_per_class: List[str],
    test_annotations: COCOParser,
    classname: str,
    ovthresh: float = 0.5,
    use_07_metric: bool = True,
    is_ood: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, float, float, int, Union[np.ndarray, None], Union[np.ndarray, None]
]:
    """
    Evaluates the performance of the object detection model using VOC-style metrics.

    This function calculates precision, recall, and average precision (AP) for a given class
    based on model predictions and ground-truth annotations. It also supports evaluation
    for Open Set Object Detection (OSOD) and includes functionality to consider unknown classes.
    If class predictions overlap with unknown class ground truths above a specified threshold,
    the function flags them as false positives in an open-set scenario.

    Args:
        predictions_per_class: List of predictions for a specific class. Each element is expected
            to contain image ID, confidence score, and bounding box information.
        test_annotations: Dictionary of ground-truth annotations, containing image-wise object
            annotations and category details.
        classname: Name of the class being evaluated. It can also be "unknown" for OSOD evaluation.
        ovthresh: Minimum overlap threshold (IoU) for a correct detection. Defaults to 0.5.
        use_07_metric: Boolean indicating whether to use PASCAL VOC 2007's 11-point AP evaluation
            metric (if True) or comprehensive precision-recall curve integration (if False).
            Defaults to True.
        is_ood: Boolean determining whether the dataset only includes out-of-distribution (OOD) objects.
            If True, all objects are considered OOD. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - rec (np.ndarray): Array of recall values at different prediction thresholds.
            - prec (np.ndarray): Array of precision values at different prediction thresholds.
            - ap (float): Average precision (AP) for the evaluated class.
            - is_unk_sum (float): Sum of OOD detections classified as known.
            - n_unk (int): Total number of unknown objects in the ground-truth data.
            - tp_plus_fp_closed_set (np.ndarray or None): Cumulative true positives plus false positives
              for closed-set evaluation.
            - fp_open_set (np.ndarray or None): Cumulative false positives due to open-set misclassifications.
    """
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    # if not is_ood:
    for imagename in test_annotations.annIm_dict.keys():
        # If is_ood, all objects in dataset are ood
        if is_ood:
            if classname == "unknown":
                R = [obj for obj in test_annotations.annIm_dict[imagename]]
            else:
                R = []
        else:
            R = [
                obj
                for obj in test_annotations.annIm_dict[imagename]
                if test_annotations.cat_dict[obj["category_id"]]["name"] == classname
            ]
        bbox = np.array([convert_xywh_to_xyxy(x["bbox"]) for x in R])
        difficult = np.array([False for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        if isinstance(imagename, int):
            imagename = str(imagename)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    splitlines = [x.strip().split(" ") for x in predictions_per_class]
    image_ids = [x[0] for x in splitlines]
    # If there exists detections for this class
    if len(image_ids[0]) > 0:
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
    else:
        image_ids = []

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] in class_recs.keys():
            R = class_recs[image_ids[d]]
        else:
            continue
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            overlaps = _compute_overlaps(BBGT, bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    if npos > 0:
        rec = tp / float(npos)
    elif npos == 0:
        rec = tp
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # compute unknown det as known
    unknown_class_recs = {}
    n_unk = 0
    for imagename in test_annotations.annIm_dict.keys():
        # If is_ood, all objects in dataset are ood
        if is_ood:
            R = [obj for obj in test_annotations.annIm_dict[imagename]]
        else:
            R = [
                obj
                for obj in test_annotations.annIm_dict[imagename]
                if test_annotations.cat_dict[obj["category_id"]]["name"] == "unknown"
            ]
        bbox = np.array([convert_xywh_to_xyxy(x["bbox"]) for x in R])
        difficult = np.array([False for x in R]).astype(bool)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        if isinstance(imagename, int):
            imagename = str(imagename)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == "unknown":
        return rec, prec, ap, 0, n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] in unknown_class_recs.keys():
            R = unknown_class_recs[image_ids[d]]
        else:
            continue
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            overlaps = _compute_overlaps(BBGT, bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    tp_plus_fp_closed_set = tp + fp
    fp_open_set = np.cumsum(is_unk)

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set


def _compute_overlaps(BBGT: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """
    Computes the overlap ratios (Intersection over Union - IoU) between a given bounding box
    and an array of ground truth bounding boxes.

    The IoU quantifies the overlap between bounding boxes by dividing the area of intersection
    by the area of the union of the two bounding boxes. A higher IoU indicates greater overlap
    between the bounding boxes.

    Args:
        BBGT (np.ndarray): A 2D numpy array of ground truth bounding boxes in the format
            [x_min, y_min, x_max, y_max].
        bb (np.ndarray): A 1D numpy array representing a single bounding box in the format
            [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: A 1D numpy array containing the IoU values for the given bounding box (bb)
            with each ground truth bounding box in BBGT.
    """
    # compute overlaps
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
        + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
        - inters
    )

    return inters / uni


def voc_ap(rec: np.ndarray, prec: np.ndarray, use_07_metric: bool = False) -> float:
    """
    Computes the Average Precision (AP) for the given recall and precision arrays.
    The method supports both the 11-point metric and the more precise method using
    the integration of the precision-recall curve.

    This function computes AP either with the 11-point interpolation method (if
    `use_07_metric` is set to True) or through a corrected method, which integrates
    the area under the precision-recall curve.

    Args:
        rec (np.ndarray): Recall values ranging from 0 to 1.
        prec (np.ndarray): Precision values corresponding to the recall values.
        use_07_metric (bool): Whether to use VOC 2007 11-point metric for AP
            computation. If False, the integration-based method is used.

    Returns:
        float: The computed average precision (AP) for the given recall and
            precision arrays.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_open_set_detection_one_method(
    id_dataset_name: str,
    id_gt_annotations_path: str,
    predictions_dict: Dict,
    method_name: str,
    threshold: float,
    test_gt_annotations_path: str,
    metric_2007: bool,
    evaluating_ood: bool,
    get_known_classes_metrics: bool,
    is_open_set_model: bool,
    unk_class_number: Union[int, None] = None,
    using_subset: Optional[List[Union[str, int]]] = False,
) -> Dict[str, float]:
    """
    Evaluates open-set detection for a given method using specified parameters and
    dataset annotations. Processes predictions, applies post-processing or open-set
    specific handling, and evaluates detection results against ground-truth data.

    Args:
        id_dataset_name (str): The name of the in-distribution dataset being
            evaluated.
        id_gt_annotations_path (str): Path to the ground-truth annotations file
            of the in-distribution dataset.
        predictions_dict (Dict): Dictionary containing prediction results for
            each image, including bounding boxes, logits, and other relevant
            information.
        method_name (str): The key name in predictions_dict representing the
            method-specific prediction scores or metrics.
        threshold (float): The threshold value to use for post-processing
            predictions.
        test_gt_annotations_path (str): Path to the ground-truth annotations file
            used for evaluation.
        metric_2007 (bool): Flag indicating whether to use the evaluation metric
            from 2007 or the newer evaluation setup.
        evaluating_ood (bool): Indicates whether the evaluation is for
            a fully out-of-distribution dataset (No ID samples).
        get_known_classes_metrics (bool): Whether to compute metrics
            for known in-distribution classes.
        is_open_set_model (bool): Indicates whether the evaluated model is
            specifically an open-set model. In this case postprocessing is not done.
        unk_class_number (Union[int, None], optional): The label index for the
            unknown class, if applicable. Default is None.
        using_subset (Optional[List[Union[str, int]]], optional): Subset of
            image IDs to evaluate on, or False if not applicable.

    Returns:
        Dict[str, float]: A dictionary of evaluation results containing metric
        names as keys and their respective scores as values.
    """
    evaluator = OpenSetEvaluator(id_dataset_name, id_gt_annotations_path, metric_2007=metric_2007)
    evaluator.reset()
    for im_id, im_pred in predictions_dict.items():
        # If using ID val subset, process only if in the used subset
        if using_subset and im_id in using_subset:
            if len(im_pred["boxes"]) > 0:
                labels, scores = get_labels_and_scores_from_logits(im_pred["logits"])
                boxes = get_boxes_from_precalculated(im_pred["boxes"])
                if not is_open_set_model:
                    # Postprocess according to score and threshold
                    unk_boxes = np.where(np.array(predictions_dict[im_id][method_name]) < threshold)
                else:
                    unk_boxes = np.where(labels == unk_class_number)
                labels[unk_boxes] = evaluator.unknown_class_index
                # Add results to evaluator
                evaluator.process(im_id, boxes, scores, labels)
        # Otherwise process all
        elif not using_subset:
            if len(im_pred["boxes"]) > 0:
                labels, scores = get_labels_and_scores_from_logits(im_pred["logits"])
                boxes = get_boxes_from_precalculated(im_pred["boxes"])
                if not is_open_set_model:
                    # Postprocess according to score and threshold
                    unk_boxes = np.where(np.array(predictions_dict[im_id][method_name]) < threshold)
                else:
                    unk_boxes = np.where(labels == unk_class_number)
                labels[unk_boxes] = evaluator.unknown_class_index
                # Add results to evaluator
                evaluator.process(im_id, boxes, scores, labels)
    evaluation_results = evaluator.evaluate(
        test_gt_annotations_path,
        is_ood=evaluating_ood,
        get_known_classes_metrics=get_known_classes_metrics,
        using_subset=using_subset,
    )
    return evaluation_results


def get_boxes_from_precalculated(boxes: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    """
    Converts a list of bounding boxes from various input formats into a numpy array.

    This function accepts bounding boxes as PyTorch tensors, numpy arrays, or lists,
    and ensures consistent conversion into a numpy array for further processing.
    An exception is raised if the input type is invalid.

    Args:
        boxes: Bounding boxes provided as a torch.Tensor, np.ndarray, or list.

    Returns:
        np.ndarray: A numpy array representation of the input bounding boxes.

    Raises:
        ValueError: If the input parameter `boxes` is not a torch.Tensor, np.ndarray, or list.
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    elif isinstance(boxes, np.ndarray):
        boxes = boxes
    elif isinstance(boxes, list):
        boxes = np.array(boxes)
    else:
        raise ValueError("boxes must be a torch.Tensor, np.ndarray or list")
    return boxes


def get_labels_and_scores_from_logits(
    logits: Union[torch.Tensor, np.ndarray, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts predicted labels and their corresponding scores from the logits using the softmax function.

    The function takes a tensor, numpy array, or list representing the logits, applies the softmax
    function, and computes the maximum predicted class and its corresponding score. If the logits have
    a specific shape (21 or 11 classes), the function excludes the last column before determining the
    predicted class and scores.

    Args:
        logits: The input logits, which can be a PyTorch tensor, a NumPy array, or a Python list.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - The first array consists of the predicted class labels.
            - The second array contains the maximum scores for each prediction.

    Raises:
        ValueError: If the input logits are not a torch.Tensor, np.ndarray, or list.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    elif isinstance(logits, np.ndarray):
        logits = logits
    elif isinstance(logits, list):
        logits = np.array(logits)
    else:
        raise ValueError("logits must be a torch.Tensor, np.ndarray or list")

    # scores = softmax(logits, axis=-1).max(axis=-1)
    scores = softmax(logits, axis=-1)
    if logits.shape[1] == 21 or logits.shape[1] == 11:
        scores = scores[:, :-1]
    pred_classes = np.argmax(scores, axis=-1)
    return pred_classes, scores.max(axis=-1)


def convert_xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """
    Converts a bounding box represented in (x, y, width, height) format to
    (x_min, y_min, x_max, y_max) format.

    This conversion is often used to switch from relative coordinate formats
    to absolute coordinate formats required for certain computer vision tasks.

    Args:
        bbox (List[float]): A list containing four float values in the format
            [x, y, width, height], where:
            x: The x-coordinate of the top-left corner.
            y: The y-coordinate of the top-left corner.
            width: The width of the bounding box.
            height: The height of the bounding box.

    Returns:
        List[float]: A list containing four float values representing the bounding box
        in the format [x_min, y_min, x_max, y_max], where:
            x_min: The x-coordinate of the top-left corner.
            y_min: The y-coordinate of the top-left corner.
            x_max: The x-coordinate of the bottom-right corner.
            y_max: The y-coordinate of the bottom-right corner.
    """
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def get_overall_open_set_results(
    ind_dataset_name: str,
    ind_gt_annotations_path: str,
    ind_data_dict: Dict,
    ood_data_dict: Dict,
    ood_datasets_names: List[str],
    ood_annotations_paths: Dict[str, str],
    methods_names: List[str],
    methods_thresholds: Dict[str, float],
    metric_2007: bool,
    evaluate_on_ind: bool,
    get_known_classes_metrics: bool,
    is_open_set_model: bool,
    unk_class_number: Union[int, None] = None,
    using_id_val_subset: Optional[List[Union[str, int]]] = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate overall open set detection results on both in-distribution (ID) and out-of-distribution (OOD) datasets.

    This function computes open set detection metrics for a given set of methods and thresholds, across multiple
    datasets (ID and OOD). It returns a dictionary containing the results for each dataset and method.

    Args:
        ind_dataset_name (str): Name of the in-distribution dataset.
        ind_gt_annotations_path (str): Path to the ground truth annotations of the in-distribution dataset.
        ind_data_dict (Dict): Dictionary containing predictions for the in-distribution dataset.
        ood_data_dict (Dict): Dictionary containing predictions for the OOD datasets.
        ood_datasets_names (List[str]): List of names of the OOD datasets.
        ood_annotations_paths (Dict[str, str]): Dictionary mapping OOD dataset names to their ground truth
            annotations paths.
        methods_names (List[str]): List of method names used for open set or OOD detection.
        methods_thresholds (Dict[str, float]): Dictionary mapping OOD method names to their respective detection
            thresholds.
        metric_2007 (bool): Whether to use the Pascal VOC 2007 evaluation metric.
        evaluate_on_ind (bool): Indicates whether to evaluate open set detection on the in-distribution data.
        get_known_classes_metrics (bool): Whether to compute metrics for known classes in the datasets.
        is_open_set_model (bool): Flag indicating if the model being evaluated is an open set detection model. In this
            case postprocessing is not done.
        unk_class_number (Union[int, None], optional): Class number assigned to unknown samples. Defaults to None.
        using_id_val_subset (Optional[List[Union[str, int]]], optional): Subset of in-distribution validation data to use.
            Defaults to False.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: A nested dictionary where each key is a dataset name (in-distribution
            or OOD), and its value is another dictionary containing method names as keys and their corresponding
            metrics as values.
    """
    open_set_results = {}
    if evaluate_on_ind:
        open_set_results[ind_dataset_name] = {}
        for baseline_name in methods_names:
            open_set_results[ind_dataset_name][baseline_name] = (
                evaluate_open_set_detection_one_method(
                    id_dataset_name=ind_dataset_name,
                    id_gt_annotations_path=ind_gt_annotations_path,
                    predictions_dict=ind_data_dict["valid"],
                    method_name=baseline_name,
                    threshold=methods_thresholds[baseline_name],
                    test_gt_annotations_path=ind_gt_annotations_path,
                    metric_2007=metric_2007,
                    evaluating_ood=False,
                    get_known_classes_metrics=True,
                    using_subset=using_id_val_subset,
                    is_open_set_model=is_open_set_model,
                    unk_class_number=unk_class_number,
                )
            )
    for ood_dataset_name in tqdm(
        ood_datasets_names, desc=f"Evaluating OSOD on OOD datasets {ood_datasets_names}"
    ):
        open_set_results[ood_dataset_name] = {}
        for baseline_name in methods_names:
            open_set_results[ood_dataset_name][baseline_name] = (
                evaluate_open_set_detection_one_method(
                    id_dataset_name=ind_dataset_name,
                    id_gt_annotations_path=ind_gt_annotations_path,
                    predictions_dict=ood_data_dict[ood_dataset_name],
                    method_name=baseline_name,
                    threshold=methods_thresholds[baseline_name],
                    test_gt_annotations_path=ood_annotations_paths[ood_dataset_name],
                    metric_2007=metric_2007,
                    evaluating_ood=True,
                    get_known_classes_metrics=get_known_classes_metrics,
                    is_open_set_model=is_open_set_model,
                    unk_class_number=unk_class_number,
                )
            )
    return open_set_results


def convert_osod_results_to_pandas_df(
    open_set_results: Dict[str, Dict[str, float]],
    methods_names: List[str],
    save_method_as_data: bool,
):
    """
    Converts a dictionary of open set results into a pandas DataFrame for easier data
    manipulation and analysis. This function processes the results of open set methods
    and structures them into a tabular format.

    Args:
        open_set_results (Dict[str, Dict[str, float]]): A dictionary containing the
            results of open set methods, where keys are method names and values are
            dictionaries of metrics and their corresponding scores.
        methods_names (List[str]): A list of method names to include in the DataFrame.
        save_method_as_data (bool): A flag indicating whether to include the method name
            as a separate column in the resulting DataFrame.

    Returns:
        pd.DataFrame: A pandas DataFrame with the specified format, containing the open
        set results. If `save_method_as_data` is `True`, an additional column with
        the method names is included.
    """
    if save_method_as_data:
        col_names = ["Method"] + list(open_set_results[list(open_set_results.keys())[0]].keys())
    else:
        col_names = list(open_set_results[list(open_set_results.keys())[0]].keys())
    new_dict = {}
    for method_name in methods_names:
        if save_method_as_data:
            new_dict[method_name] = [method_name] + list(open_set_results[method_name].values())
        else:
            new_dict[method_name] = list(open_set_results[method_name].values())
    df = pd.DataFrame.from_dict(new_dict, orient="index", columns=col_names)
    return df


def convert_osod_results_to_hierarchical_pandas_df(
    osod_results_a: Dict[str, Dict[str, float]],
    osod_results_b: Dict[str, Dict[str, float]],
    methods_names: List[str],
    save_method_as_data: bool,
    datasets_names: List[str],
):
    """
    Converts OSOD (Open Set Object Detection) results into a hierarchical
    pandas DataFrame for improved organization and readability. The resulting DataFrame
    is structured with datasets and metrics forming a multi-level column index.

    Args:
        osod_results_a: A dictionary mapping method names to their metric values
            for the first set of results. The inner dictionary maps metric names
            to their respective float values.
        osod_results_b: A dictionary similar to osod_results_a but containing
            metric values for the second set of results.
        methods_names: A list of method names that specify the order of processing
            for the OSOD results.
        save_method_as_data: A boolean indicating whether to include method names as
            a separate column in the resulting DataFrame.
        datasets_names: A list of dataset names to use for labeling the hierarchical
            columns, providing separation of metrics across datasets.

    Returns:
        pd.DataFrame: A hierarchical pandas DataFrame representing the OSOD
        results, where columns are organized by dataset names and their associated
        metrics, while rows represent different methods.
    """
    if save_method_as_data:
        col_names = ["Method"] + list(osod_results_a[list(osod_results_a.keys())[0]].keys())
    else:
        col_names = list(osod_results_a[list(osod_results_a.keys())[0]].keys())
    columns = pd.MultiIndex.from_product([datasets_names, col_names], names=["Dataset", "Metric"])
    new_dict = {}
    for method_name in methods_names:
        if save_method_as_data:
            new_dict[method_name] = (
                [method_name]
                + list(osod_results_a[method_name].values())
                + list(osod_results_b[method_name].values())
            )
        else:
            new_dict[method_name] = list(osod_results_a[method_name].values()) + list(
                osod_results_b[method_name].values()
            )
    df = pd.DataFrame.from_dict(new_dict, orient="index", columns=columns)
    return df


def plot_two_osod_datasets_metrics(
    osod_results_a: Dict[str, Dict[str, float]],
    osod_results_b: Dict[str, Dict[str, float]],
    methods_names: List[str],
    datasets_names: List[str],
    metrics_to_plot: List[str],
    show_plot: bool,
):
    """
    Plots OSOD (Open Set Object Detection) metrics for two datasets, comparing performance
    across different methods and metrics.

    This function creates a grouped bar chart for visualizing the performance of various
    methods on specified metrics for two different datasets. The bars for each dataset
    are grouped together with corresponding labels for better comparison. The user can
    display the plot directly or use the returned figure object for further customization.

    Args:
        osod_results_a (Dict[str, Dict[str, float]]): Metrics results for the first dataset,
            indexed by method names and containing values for specified metrics.
        osod_results_b (Dict[str, Dict[str, float]]): Metrics results for the second dataset,
            indexed by method names and containing values for specified metrics.
        methods_names (List[str]): List of method names to include in the plot. Each method's
            metrics will be plotted as bars.
        datasets_names (List[str]): List containing names of the two datasets to be labeled
            on the plot.
        metrics_to_plot (List[str]): List of metric names to include as x-axis labels. These
            metrics will be grouped for comparison.
        show_plot (bool): If set to True, the generated plot will be displayed immediately.

    Returns:
        matplotlib.figure.Figure: The figure object containing the generated bar chart,
        which can be further manipulated or saved.

    """
    x = np.arange(len(metrics_to_plot))  # the label locations
    width = 1 / (len(methods_names) * 2 + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained", figsize=(4 * len(methods_names), 6))

    for method in methods_names:
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            osod_results_a.loc[method][metrics_to_plot],
            width,
            label=f"{method} {datasets_names[0]}",
        )
        ax.bar_label(rects, padding=3, fontsize=8, fmt="%.2f")
        multiplier += 1
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            osod_results_b.loc[method][metrics_to_plot],
            width,
            label=f"{method} {datasets_names[1]}",
        )
        ax.bar_label(rects, padding=3, fontsize=8, fmt="%.2f")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage")
    ax.set_title(f"OSOD metrics for {datasets_names[0]} and {datasets_names[1]}")
    ax.set_xticks(x + 0.5 - 0.5 * width, metrics_to_plot)
    # ax.legend(loc='upper left', ncols=3)
    ax.legend(ncols=max(1, int(len(methods_names) / 3)))
    ax.set_ylim(0, 100)
    if show_plot:
        plt.show()
    return fig


def plot_two_osod_datasets_per_metric(
    osod_results_a: Dict[str, Dict[str, float]],
    osod_results_b: Dict[str, Dict[str, float]],
    methods_names: List[str],
    datasets_names: List[str],
    metric_to_plot: str,
    show_plot: bool,
):
    """
    Plots and compares a specified metric for two datasets in Open Set Object Detection (OSOD)
    experiments across given methods.

    This function generates a bar chart that displays performance metrics for
    two datasets side-by-side for each method provided. It allows for visual
    comparison of the datasets and enables clearer observation of metric-level
    differences. The chart title and labels are dynamically derived based on
    the datasets and the specified metric.

    Args:
        osod_results_a: Metric data for the first dataset. The data should be
            structured as a dictionary where keys are metric names and values
            are dictionaries with method names as keys and their corresponding
            metric values as floats.
        osod_results_b: Metric data for the second dataset. The schema should
            match that of `osod_results_a`.
        methods_names: List of method names to be used as x-axis labels and as
            keys for accessing metric data in the datasets.
        datasets_names: List containing the names of the two datasets being
            compared. The first name corresponds to `osod_results_a` and the
            second to `osod_results_b`.
        metric_to_plot: String representing the metric key to extract and
            visualize from the provided datasets.
        show_plot: Boolean indicating whether to display the plot immediately.
            If `False`, the plot is simply returned without being displayed.

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object representing the
        generated bar chart. This can be further customized or saved to disk
        as required.
    """
    # colors = ['tab:blue', 'tab:orange']
    x = np.arange(len(methods_names))  # the label locations
    width = 1 / (len(datasets_names) + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained", figsize=(1.5 * len(methods_names), 5))
    ax.grid(axis="y", linestyle="--")
    for dataset, dataset_name in zip([osod_results_a, osod_results_b], datasets_names):
        offset = width * multiplier
        rects = ax.bar(x + offset, dataset[metric_to_plot], width, label=f"{dataset_name}")
        ax.bar_label(rects, padding=3, fontsize=8, fmt="%.2f")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage")
    ax.set_title(f"OSOD {metric_to_plot} for {datasets_names[0]} and {datasets_names[1]}")
    ax.set_xticks(x + 0.5 - width, methods_names)
    # ax.legend(loc='upper left', ncols=3)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[-2:], labels[-2:], frameon=True)

    ax.legend()
    ax.set_ylim(0, 100)
    if show_plot:
        plt.show()
    return fig


def convert_osod_results_for_mlflow_logging(
    open_set_results: Dict[str, Dict[str, Dict[str, float]]],
    ood_datasets_names: List[str],
    methods_names: List[str],
) -> Dict[str, float]:
    """
    Converts open set object detection (OSOD) results into a format suitable
    for logging in MLflow by restructuring and flattening nested dictionaries.

    Args:
        open_set_results (Dict[str, Dict[str, Dict[str, float]]]): The input OSOD results
            structured as a dictionary where keys represent out-of-distribution dataset names,
            values are dictionaries of baseline methods, and each method contains
            metrics and their values.
        ood_datasets_names (List[str]): A list of out-of-distribution dataset names
            for which the results are structured.
        methods_names (List[str]): A list of method names to retrieve
            metrics for from the OSOD results.

    Returns:
        Dict[str, float]: A flattened dictionary suitable for MLflow where keys represent
            concatenations of out-of-distribution dataset names, method names, and metric names,
            and values represent the corresponding metric values.
    """
    results_for_mlflow = {}
    for ood_dataset_name in ood_datasets_names:
        for baseline_name in methods_names:
            for metric_name, value in open_set_results[ood_dataset_name][baseline_name].items():
                results_for_mlflow[f"{ood_dataset_name} {baseline_name} {metric_name}"] = value
    return results_for_mlflow


def get_n_unk_ood_dataset(annotations_path: str):
    """
    Calculates and returns the number of unknown instances for an OOD dataset based on the provided annotations path.
    Since it expects an OOD dataset annotations path, the function assumes that all instances are unknown.

    This function utilizes the COCOParser class to parse the given annotations file,
    extract image IDs, and subsequently retrieve annotation IDs associated with those images.
    The total count of these annotation IDs is then computed and returned.

    Args:
        annotations_path (COCOParser): A COCOParser instance or an object compatible with it
            containing the path to the dataset annotations.

    Returns:
        int: The total number of annotations in the dataset.
    """
    annotations = COCOParser(annotations_path)
    im_ids = annotations.get_imgIds()
    ann_ids = annotations.get_annIds(im_ids)
    return len(ann_ids)
