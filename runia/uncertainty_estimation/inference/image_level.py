# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez, probabilistic adaptation
#    Daniel Montoya
from typing import Tuple

import numpy as np
import torch

from runia.dimensionality_reduction import apply_pca_transform
from runia.uncertainty_estimation.feature_extraction import (
    MCSamplerModule,
    get_mean_or_fullmean_ls_sample,
    Hook,
)
from runia.uncertainty_estimation.entropy import get_dl_h_z
from runia.uncertainty_estimation.inference.abstract_classes import (
    ProbabilisticInferenceModule,
    record_time,
    InferenceModule,
    Postprocessor,
)

__all__ = ["LaRExInference", "LaRDInference"]


class LaRExInference(ProbabilisticInferenceModule):
    """
    Class intended to perform inference on new data using the LaREx methods (either LaRED or LaREM).
    LaREx performs zMCD sampling, entropy and density calculations to return a confidence score.
    It assumes that an optional PCA reducer plus LaRED or LaREM postprocessors are already trained.
    It can also perform testing of inference time.

    Args:
        model: Trained model
        postprocessor: LaRED or laREM trained postprocessor.
        mcd_sampler: Monte Carlo Dropout sampler module
        pca_transform: Optionally PCA already trained for dimension reduction. Default: None
        mcd_samples_nro: Number of MCD samples.
        layer_type: Either 'Conv' or 'FC'
    """

    def __init__(
        self,
        model,
        postprocessor,
        drop_block_prob: float,
        drop_block_size: int,
        mcd_samples_nro: int,
        mcd_sampler: MCSamplerModule,
        pca_transform=None,
        layer_type="Conv",
    ):
        """
        Class intended to perform inference on new data using the LaREx methods (either LaRED or
        LaREM). LaREx performs zMCD sampling, entropy and density calculations to return a
        confidence score. It assumes that an optional PCA reducer plus LaRED or LaREM postprocessors
        are already trained. It can also perform testing of inference time.

        Args:
            model: Trained model
            postprocessor: LaRED or laREM trained postprocessor.
            mcd_sampler: Monte Carlo Dropout sampler module
            drop_block_prob: Dropblock probability
            drop_block_size: Size of Dropblock
            pca_transform: Optionally PCA trained for dimension reduction. Default: None
            mcd_samples_nro: Number of MCD samples.
            layer_type: Either 'Conv' or 'FC'
        """
        super().__init__(
            model=model,
            postprocessor=postprocessor,
            drop_block_prob=drop_block_prob,
            drop_block_size=drop_block_size,
            mcd_samples_nro=mcd_samples_nro,
        )

        self.layer_type = layer_type
        self.pca_transform = pca_transform

        self.mc_sampler = mcd_sampler(
            mc_samples=self.mcd_samples_nro,
            layer_type=layer_type,
            drop_prob=self.drop_block_prob,
            block_size=self.drop_block_size,
        )
        self.mc_sampler.to(self.device)
        self.mc_sampler.train()

        # self.sample_larex_score = None

    def get_score(self, input_image, layer_hook):
        """
        Compute LaREx score for a single image.

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
            output = self.model(input_image)
            latent_rep = layer_hook.output  # latent representation sample

        mc_samples_t = self.mc_sampler(latent_rep)
        _, sample_h_z = get_dl_h_z(mc_samples_t, self.mcd_samples_nro)
        if self.pca_transform:
            sample_h_z = apply_pca_transform(sample_h_z, self.pca_transform)
        sample_larex_score = self.postprocessor.postprocess(sample_h_z)
        return output, sample_larex_score

    @record_time
    def test_time_inference(self, input_image, layer_hook):
        """
        Call the inference function and get the execution time

        Args:
            input_image: Input image
            layer_hook: Hooked layer

        Returns:
            LaREx inference results plus execution time
        """
        return self.get_score(input_image, layer_hook)

    @record_time
    def get_layer_mc_samples(self, input_image, layer_hook):
        """
        Function used to benchmark execution time of model inference and MC sampling

        Args:
            input_image: Input image
            layer_hook: Hooked layer

        Returns:
            Mc samples plus execution time
        """
        with torch.no_grad():
            input_image = input_image.to(self.device)
            _ = self.model(input_image)
            latent_rep = layer_hook.output  # latent representation sample

        mc_samples_t = self.mc_sampler(latent_rep)
        return mc_samples_t

    @record_time
    def get_mc_samples_full_inference(self, input_image, layer_hook):
        """
        Function used to benchmark execution time of full model inferences, instead of the fast
        method with only one inference.

        Args:
            input_image: Input image
            layer_hook: Hooked layer

        Returns:
            MC samples plus execution time
        """
        mc_samples = []
        with torch.no_grad():
            for i in range(self.mcd_samples_nro):
                try:
                    input_image = input_image.to(self.device)
                except AttributeError:
                    pass
                _ = self.model(input_image)
                latent_rep = layer_hook.output
                mc_samples.append(latent_rep)

            mc_samples_t = torch.cat(mc_samples)
            mc_samples_np = mc_samples_t.cpu().numpy()
        return mc_samples_np

    @record_time
    def get_score_full_inference(self, input_image, layer_hook):
        """
        Abstract function that should be implemented in child class to benchmark execution time
        of the full method using complete inferences instead of the fast method. Do not use this
        function for normal inference.

        Args:
            input_image: Input image
            layer_hook: Hooked layer

        Returns:
            LaREx inference results and execution time
        """
        raise NotImplementedError


class LaRDInference(InferenceModule):
    """
    Class intended to perform uncertainty estimation inference on new data.
    LaRD performs representation reduction and density calculations to return a confidence score.
    This method does not perform zMCD sampling nor entropy calculations.
    It assumes that an optional PCA reducer plus KDE or MD postprocessors are already trained.
    It can also perform testing of inference time.

    Args:
            model: Trained model
            postprocessor: KDE or MD trained postprocessor.
            pca_transform: Optionally PCA already trained for dimension reduction. Default: None
            layer_type: Either 'Conv' or 'FC'
    """

    def __init__(
        self, model, postprocessor: Postprocessor, pca_transform=None, layer_type="Conv"
    ) -> None:
        """
        Class intended to perform inference on new data using postprocessor methods (either KDE or
        MD). LaRD performs representation reduction and density calculations to return a
        confidence score. This method does not perform zMCD sampling nor entropy calculations.
        It assumes that an optional PCA reducer plus KDE or MD postprocessors are already
        trained. It can also perform testing of inference time.

        Args:
            model: Trained model
            postprocessor: KDE or MD trained postprocessor.
            pca_transform: Optionally PCA trained for dimension reduction. Default: None
            layer_type: Either 'Conv' or 'FC'
        """
        super().__init__(model, postprocessor)
        self.layer_type = layer_type
        if self.layer_type == "Conv":
            self._reducer = self._reduce_conv_representation
        elif self.layer_type == "FC":
            self._reducer = self._reduce_fc_representation
        else:
            pass  # The only other possibility so far is RPN, implemented in their own subclass
        self.pca_transform = pca_transform
        # self.sample_larex_score = None

    def get_score(self, input_image: torch.Tensor, layer_hook: Hook) -> Tuple[torch.Tensor, float]:
        """
        Compute LaRx score for a single image

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
            output = self.model(input_image)
            latent_rep = layer_hook.output  # latent representation sample
        latent_rep = self._reducer(latent_rep)
        if self.pca_transform:
            latent_rep = apply_pca_transform(latent_rep, self.pca_transform)
        sample_score = self.postprocessor.postprocess(latent_rep)
        return output, sample_score

    @record_time
    def test_time_inference(
        self, input_image: torch.Tensor, layer_hook: Hook
    ) -> Tuple[torch.Tensor, float]:
        """
        Call the inference function and get the execution time

        Args:
            input_image: Input image
            layer_hook: Hooked layer

        Returns:
            LaREx inference results plus execution time
        """
        return self.get_score(input_image, layer_hook)

    @staticmethod
    def _reduce_conv_representation(representation: torch.Tensor) -> np.ndarray:
        """
        Private method that performs reduction of latent representations from convolutional layers.

        Args:
            representation: Latent representation

        Returns:
            Average per channel from convolutional activation maps.
        """
        return (
            get_mean_or_fullmean_ls_sample(representation, "fullmean").cpu().numpy().reshape(1, -1)
        )

    @staticmethod
    def _reduce_fc_representation(representation: torch.Tensor) -> np.ndarray:
        """
        Private method that performs reduction of latent representations from Fully connected
        layers.

        Args:
            representation: Latent representation

        Returns:
            If dim(representation) > 1, average per column, reshaped representation otherwise.
        """
        if representation.ndim > 1:
            return torch.mean(representation, dim=1).cpu().numpy().reshape(1, -1)
        else:
            return representation.reshape(1, -1).cpu().numpy()
