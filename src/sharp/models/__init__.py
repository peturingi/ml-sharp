"""Contains different Gaussian predictors.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Final

from sharp.models.monodepth import (
    create_monodepth_adaptor,
    create_monodepth_dpt, MonodepthDensePredictionTransformer, MonodepthWithEncodingAdaptor,
)

from .alignment import create_alignment
from .composer import GaussianComposer
from .gaussian_decoder import create_gaussian_decoder, GaussianDensePredictionTransformer
from .heads import DirectPredictionHead
from .initializer import create_initializer, MultiLayerInitializer
from .params import PredictorParams
from .predictor import RGBGaussianPredictor


def create_predictor(predictor_params: PredictorParams) -> RGBGaussianPredictor:
    """Create gaussian predictor model specified by name."""
    if predictor_params.gaussian_decoder.stride < predictor_params.initializer.stride:
        raise ValueError(
            "We do not expected gaussian_decoder has higher resolution than initializer."
        )

    scale_factor: Final[int] = predictor_params.gaussian_decoder.stride // predictor_params.initializer.stride
    gaussian_composer: Final[GaussianComposer] = GaussianComposer(
        delta_factor=predictor_params.delta_factor,
        min_scale=predictor_params.min_scale,
        max_scale=predictor_params.max_scale,
        color_activation_type=predictor_params.color_activation_type,
        opacity_activation_type=predictor_params.opacity_activation_type,
        color_space=predictor_params.color_space,
        scale_factor=scale_factor,
        base_scale_on_predicted_mean=predictor_params.base_scale_on_predicted_mean,
    )
    if predictor_params.num_monodepth_layers > 1 and predictor_params.initializer.num_layers != 2:
        raise KeyError("We only support num_layers = 2 when num_monodepth_layers > 1.")

    monodepth_model: Final[MonodepthDensePredictionTransformer] = create_monodepth_dpt(predictor_params.monodepth)
    monodepth_adaptor: Final[MonodepthWithEncodingAdaptor] = create_monodepth_adaptor(
        monodepth_model,
        predictor_params.monodepth_adaptor,
        predictor_params.num_monodepth_layers,
        predictor_params.sorting_monodepth,
    )

    if predictor_params.num_monodepth_layers == 2:
        monodepth_adaptor.replicate_head(predictor_params.num_monodepth_layers)

    gaussian_decoder: Final[GaussianDensePredictionTransformer] = create_gaussian_decoder(
        predictor_params.gaussian_decoder,
        dims_depth_features=monodepth_adaptor.get_feature_dims(),
    )
    initializer: Final[MultiLayerInitializer] = create_initializer(
        predictor_params.initializer,
    )
    prediction_head: Final[DirectPredictionHead] = DirectPredictionHead(
        feature_dim=gaussian_decoder.dim_out, num_layers=initializer.num_layers
    )
    decoder_dim = monodepth_model.decoder.dims_decoder[-1]
    return RGBGaussianPredictor(
        init_model=initializer,
        feature_model=gaussian_decoder,
        prediction_head=prediction_head,
        monodepth_model=monodepth_adaptor,
        gaussian_composer=gaussian_composer,
        scale_map_estimator=create_alignment(predictor_params.depth_alignment, depth_decoder_dim=decoder_dim),
    )


__all__ = [
    "PredictorParams",
    "create_predictor",
    "RGBGaussianPredictor",
]
