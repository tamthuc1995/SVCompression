# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import numpy as np
from tqdm import tqdm
from os import makedirs
import shutil
import imageio

import torch

from src.config import cfg, update_argparser, update_config

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.directional_color_utils import get_directional_color_transform


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster rendering.")
    parser.add_argument('model_path')
    parser.add_argument('--model_bitstreams_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--geo_params_stepsize", default=1.0, type=float)
    parser.add_argument("--color_coefs_stepsize", default=0.5, type=float)
    parser.add_argument("--dont_do_shs_transform", action="store_true")
    parser.add_argument("--color_channel_independent", action="store_true")
    parser.add_argument("--use_directional_color_transform", action="store_true")
    parser.add_argument("--suffix", default="config00", type=str)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    print("Encode to bitstreams " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))

    # Load data
    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
        camera_params_only=False,
    )

    # Load model
    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    )
    loaded_iter = voxel_model.load_iteration(args.model_path, args.iteration)
    voxel_model.freeze_vox_geo()

    compression_config = {
        "geo_params_stepsize": args.geo_params_stepsize,
        "color_coefs_stepsize": args.color_coefs_stepsize,
        "color_channel_independent": args.color_channel_independent,
        "do_shs_transform": not args.dont_do_shs_transform,
        "debug": args.debug
    }
    
    # Output path suffix
    suffix = args.suffix
    encode_bitstream_path = os.path.join(
        args.model_bitstreams_path, suffix, "compressed_bitstream")
    makedirs(encode_bitstream_path, exist_ok=True)
    shutil.copy(
        os.path.join(args.model_path, 'config.yaml'),
        os.path.join(args.model_bitstreams_path, 'config.yaml')
    )

    if args.use_directional_color_transform:
        # check pre-computed color transform 
        path_color_transform = os.path.join(args.model_bitstreams_path, 'directional_color_transform_precomputed.npz')
        if os.path.exists(path_color_transform):
            color_transforms = np.load(path_color_transform)
        else:
            list_decoding_views = data_pack.get_test_cameras()
            _, transform, inverse_transform = get_directional_color_transform(list_decoding_views, voxel_model)
            color_transforms = {
                "directional_color_transform": transform, 
                "directional_color_transform_inverse": inverse_transform
            }
            np.savez_compressed(path_color_transform, **color_transforms)

        print("Use directional color transform")
        compression_config["use_directional_color_transform"] = True
        compression_config["directional_color_transform"] = color_transforms["directional_color_transform"]
        compression_config["directional_color_transform_inverse"] = color_transforms["directional_color_transform_inverse"]


    s = time.time()
    total_bytes_encoded = voxel_model.encode(encode_bitstream_path, compression_config)
    total_Mb_encoded    = total_bytes_encoded / 1024 / 1024
    total_mins_encoded  = (time.time() - s)/60.0
    print(f'total_bytes_encoded: {total_Mb_encoded:.3f} Mb')
    print(f'total_times_encoded: {total_mins_encoded:.3f} mins')


