from typing import Union, Any
import torch
from numpy import ascontiguousarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from runia.feature_extraction import FastMCDSamplesExtractor

__all__ = ["FastMCDSamplesExtractorYolov8"]


class FastMCDSamplesExtractorYolov8(FastMCDSamplesExtractor):
    """
    Class to get either zMCD samples or raw latent samples from any Yolo v8 model layer different from a Dropout or
    Dropblock, so absolutely no architecture modification is needed. Inherits from the base
    class FastMCDSamplesExtractor.
    """

    def get_ls_samples(self, data_loader: Union[DataLoader, Any], **kwargs) -> dict:
        """
        Extract latent space samples given a dataloader.

        Args:
            data_loader: DataLoader or Dataloader like. Already tested with LoadImages class
                from ultralytics

        Returns:
            Latent MCD samples and optionally the raw inference results
        """
        results = {"latent_space_means": []}
        if self.return_raw_predictions:
            results["raw_preds"] = []
        if self.return_variances:
            results["variances"] = []
        if self.return_gt_labels:
            results["gt_labels"] = []

        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting latent space samples") as pbar:
                for image, label in data_loader:
                    # Here, a BGR 2 RGB inversion is performed, since the torch Dataloader seems to feed yolo
                    # Images in the wrong ordering
                    image = [ascontiguousarray(image[0].numpy().transpose(1, 2, 0)[..., ::-1])]
                    result_img = self._get_samples_one_image(image=image)
                    for result_type, result_value in result_img.items():
                        results[result_type].append(result_value)
                    # Update progress bar
                    pbar.update(1)
                for result_type, result_value in results.items():
                    results[result_type] = torch.cat(result_value, dim=0)

        print("Latent representation vector size: ", results["latent_space_means"].shape[1])
        return results
