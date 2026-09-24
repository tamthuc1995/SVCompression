import os
import re
import torch

#
from src.utils import octree_utils

# 
import numpy as np
from collections import deque
import constriction
from src.utils import raht_utils
from src.utils import comression_utils

import svraster_cuda
from svraster_cuda.meta import MAX_NUM_LEVELS

class SVCompression:

    def encode(self, path, compression_config={}):

        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'active_sh_degree': self.active_sh_degree,
            'scene_center': self.scene_center.data.contiguous().detach().cpu().numpy(),
            'inside_extent': self.inside_extent.data.contiguous().detach().cpu().numpy(),
            'scene_extent': self.scene_extent.data.contiguous().detach().cpu().numpy(),
            'octpath': self.octpath.data.contiguous().detach().cpu().numpy(),
            'octlevel': self.octlevel.data.contiguous().detach().cpu().numpy(),
            '_geo_grid_pts': self._geo_grid_pts.data.contiguous().detach().cpu().numpy(),
            '_sh0': self._sh0.data.contiguous().detach().cpu().numpy(),
            '_shs': self._shs.data.contiguous().detach().cpu().numpy(),
        }
        # REORDER EVERYTHING BY OCTPATH
        order = np.argsort(state_dict["octpath"][:, 0])
        state_dict["octpath"]  = state_dict["octpath"][order]
        state_dict["octlevel"] = state_dict["octlevel"][order]
        state_dict["_sh0"]     = state_dict["_sh0"][order]
        state_dict["_shs"]     = state_dict["_shs"][order]
        del order
        

        #### HEAD BITSTREAM
        META_INFO = {
            "num_voxels": state_dict["octpath"].shape[0],
            "active_sh_degree": state_dict["active_sh_degree"],
            "scene_center": state_dict["scene_center"],
            "inside_extent": state_dict["inside_extent"],
            "scene_extent": state_dict["scene_extent"],
            #######
            "geo_params_stepsize": np.float16(compression_config.get("geo_params_stepsize", 1.0)),
            "geo_params_entropy_min": np.int32(-compression_config.get("geo_params_half_num_bins", 10000)),
            "geo_params_entropy_max": np.int32( compression_config.get("geo_params_half_num_bins", 10000)),
            #######
            "color_coefs_stepsize": np.float16(compression_config.get("color_coefs_stepsize", 0.5)),
            "color_coefs_entropy_min": np.int32(-compression_config.get("color_coefs_half_num_bins", 10000)),
            "color_coefs_entropy_max": np.int32( compression_config.get("color_coefs_half_num_bins", 10000)),
            #######
            "use_directional_color_transform": compression_config.get("use_directional_color_transform", False)
        }

        ####################################################################################
        ####################################################################################
        #####                    COMPRESSION OF OCTTREE OCCUPACY                       #####
        ####################################################################################
        ####################################################################################

        occupacy_bitstream = comression_utils.encode_morton_to_bitstream(
            state_dict["octpath"][:, 0], 
            state_dict["octlevel"][:, 0],
            max_level=MAX_NUM_LEVELS
        )

        decoded_octpath, decoded_octlevel = comression_utils.decode_morton_from_bitstream(
            occupacy_bitstream, 
            max_level=MAX_NUM_LEVELS
        )
        if compression_config["debug"]:
            assert np.all(decoded_octpath == state_dict["octpath"][:, 0]), "decoded_octpath is mismatch"
            assert np.all(decoded_octlevel == state_dict["octlevel"][:, 0]), "decoded_octlevel is mismatch"

        state_dict["octpath"] = None
        state_dict["octlevel"] = None
        

        np.savez_compressed(f"{path}/occupacy_bitstreams.npz", bitstreams=occupacy_bitstream)

        ####################################################################################
        ####################################################################################
        #####                  COMPRESSION OF GEOMETRY PARAMETERs                      #####
        ####################################################################################
        ####################################################################################
        decoded_octpath=torch.tensor(decoded_octpath[:, np.newaxis], dtype=torch.int64).cuda()
        decoded_octlevel=torch.tensor(decoded_octlevel[:, np.newaxis], dtype=torch.int8).cuda()

        # Available at decoding
        vox_center, _ = octree_utils.octpath_decoding(
            decoded_octpath,
            decoded_octlevel,
            torch.tensor(META_INFO["scene_center"]).cuda(),
            torch.tensor(META_INFO["scene_extent"]).cuda()
        )
        grid_pts_key, _ = octree_utils.build_grid_pts_link(
            decoded_octpath, 
            decoded_octlevel
        )
        max_levels         = torch.ones((grid_pts_key.shape[0], 1), dtype=torch.int8).cuda() * MAX_NUM_LEVELS
        corners_mortoncode = svraster_cuda.utils.ijk_2_octpath(grid_pts_key, max_levels)
        corners_mortoncode = corners_mortoncode[:, 0].cpu().numpy()
        num_corners        = corners_mortoncode.shape[0]

        # 
        index2corners = np.arange(corners_mortoncode.shape[0]).astype(np.int32)
        cornersorder  = np.argsort(corners_mortoncode)
        index2corners      = index2corners[cornersorder]
        corners_mortoncode = corners_mortoncode[cornersorder]

        # Construct raht tree for corners
        corners_binlevel, corners_coeff_binlevel = raht_utils.RahtPrologue(corners_mortoncode)
        order_coefs = np.argsort(corners_coeff_binlevel)
        index_reverse = np.arange(num_corners).astype(np.int32)
        index_reverse = index_reverse[order_coefs]


        # NOT Available at decoding
        # encode geo params at corners
        # Transform
        transformed_attributes = raht_utils.Raht(
            corners_binlevel,
            state_dict["_geo_grid_pts"][cornersorder]
        )[order_coefs]
        
        geo_params_stepsize = META_INFO["geo_params_stepsize"]
        META_INFO["geo_params_entropy_mean"] = np.mean(transformed_attributes[8:]).astype(np.float32)
        META_INFO["geo_params_entropy_std"] = np.std(transformed_attributes[8:]).astype(np.float32)


        DC_coef = transformed_attributes[:8]
        AC_coefs = np.round(
            transformed_attributes[8:, 0] / geo_params_stepsize
        ).astype(np.int32)

        #
        geo_params_entropy_model = constriction.stream.model.QuantizedGaussian(
            META_INFO["geo_params_entropy_min"],
            META_INFO["geo_params_entropy_max"],
            META_INFO["geo_params_entropy_mean"],
            META_INFO["geo_params_entropy_std"]
        )

        # ---------- Encoding ----------
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(AC_coefs, geo_params_entropy_model)
        geo_params_bitstream = encoder.get_compressed()
        print(
            f"Compressed AC geo params size: "
            f"({len(geo_params_bitstream) * 32/1024/1024/8} Mb including padding)"
        )
        
        if compression_config["debug"]:
            decoder = constriction.stream.stack.AnsCoder(geo_params_bitstream)
            decoded_AC_coefs = decoder.decode(geo_params_entropy_model, len(AC_coefs))  # decode exactly N symbols
            assert np.all(decoded_AC_coefs == AC_coefs), "decoded_AC_coefs is not same as quantized AC_coefs"


        ### WRITE TO DISC
        np.savez_compressed(f"{path}/geoparams_bitstreams.npz", DC_coef=DC_coef, geo_params_bitstream=geo_params_bitstream)

        # cleaning
        del corners_mortoncode
        del index2corners, cornersorder
        del corners_binlevel, corners_coeff_binlevel, order_coefs, index_reverse
        del transformed_attributes
        del AC_coefs
        del DC_coef


        ####################################################################################
        ####################################################################################
        #####                  COMPRESSION OF COLOR PARAMETERs                         #####
        ####################################################################################
        ####################################################################################
        

        ### WRITE TO DISC
        if META_INFO["use_directional_color_transform"]:

            print("Use directional color transform")
            shs_coefficients = np.concatenate([
                state_dict["_sh0"][:, np.newaxis, :], 
                state_dict["_shs"]
            ], axis=1)

            shs_coefficients = (
                shs_coefficients.reshape(-1, 16 * 3) @ compression_config["directional_color_transform"]
            ).reshape(-1, 16, 3)

        else:

            shs_coefficients = np.concatenate([
                state_dict["_sh0"][:, np.newaxis, :], 
                state_dict["_shs"]
            ], axis=1).reshape(-1, 16, 3)


        # Build Color Transform
        if compression_config["do_shs_transform"]:
            if compression_config["color_channel_independent"]:
                kernels_klt_transform = np.eye(48)
                for d in range(3):
                    XtX = shs_coefficients[:, :, d].T @ shs_coefficients[:, :, d]
                    U, S, V = np.linalg.svd(XtX)
                    diag_begin, diag_end = d*16, (d+1)*16
                    kernels_klt_transform[diag_begin:diag_end, diag_begin:diag_end] = U
            else:
                XtX = shs_coefficients.reshape(-1, 48).T @ shs_coefficients.reshape(-1, 48)
                U, S, V = np.linalg.svd(XtX)
                kernels_klt_transform = U
        else:
            kernels_klt_transform = np.eye(48)

        kernels_klt_transform = kernels_klt_transform.astype(np.float32)
        # Prepare for RAHT transform
        decoded_octpath_numpy = decoded_octpath.cpu().numpy()
        shs_binlevel, shs_coeff_binlevel = raht_utils.RahtPrologue(decoded_octpath_numpy[:, 0])
        order_shs_coefs = np.argsort(shs_coeff_binlevel)
        shs_index_reverse = np.arange(shs_coeff_binlevel.shape[0]).astype(np.int32)
        shs_index_reverse = shs_index_reverse[order_shs_coefs]
        shs_coefficients = (
            shs_coefficients.reshape(-1, 48) @ kernels_klt_transform
        ).reshape(-1, 16, 3)


        # Apply RAHT transform
        num_voxels       = decoded_octpath_numpy.shape[0]
        DC_color_coeffs  = []
        AC_bitstreams    = []
        META_INFO["color_coefs_entropy_mean"] = []
        META_INFO["color_coefs_entropy_std"]  = []
        color_coefs_stepsize = META_INFO["color_coefs_stepsize"]
        size = 0.0
        for k in range(16):
            coefs = raht_utils.Raht(shs_binlevel, shs_coefficients[:, k, :])[order_shs_coefs]
            DC_color_coeffs.append(coefs[:8, :].flatten().astype(np.float32))

            META_INFO["color_coefs_entropy_mean"].append(
                np.mean(coefs[8:, :].transpose().flatten(), axis=0).astype(np.float32)
            )
            META_INFO["color_coefs_entropy_std"].append(
                np.std(coefs[8:, :].transpose().flatten(), axis=0).astype(np.float32)
            )

            AC_coefs_quantized = np.round(
                coefs[8:, :].transpose().flatten() / color_coefs_stepsize
            ).astype(np.int32)

            if compression_config["debug"]:
                print(f"min_bin={np.min(AC_coefs_quantized)} max_bin={np.max(AC_coefs_quantized)}")
                assert np.min(AC_coefs_quantized) > META_INFO["color_coefs_entropy_min"]
                assert np.max(AC_coefs_quantized) < META_INFO["color_coefs_entropy_max"]
                                                
            color_entropy_model = constriction.stream.model.QuantizedGaussian(
                META_INFO["color_coefs_entropy_min"],
                META_INFO["color_coefs_entropy_max"],
                META_INFO["color_coefs_entropy_mean"][k],
                META_INFO["color_coefs_entropy_std"][k]
            )

            encoder = constriction.stream.stack.AnsCoder()
            encoder.encode_reverse(AC_coefs_quantized, color_entropy_model)
            AC_bitstream_k = encoder.get_compressed()

            print(
                f"Compressed AC color k={k} coefficients size: "
                f"({len(AC_bitstream_k) * 32/1024/1024/8} Mb including padding)"
            )
            size += len(AC_bitstream_k) * 32/1024/1024/8

            if compression_config["debug"]:
                decoder = constriction.stream.stack.AnsCoder(AC_bitstream_k)
                decoded_AC_coefs_quantized = decoder.decode(color_entropy_model, (num_voxels-8) * 3)  # decode exactly N symbols
                assert np.all(decoded_AC_coefs_quantized == AC_coefs_quantized)

            AC_bitstreams.append(AC_bitstream_k)

        print(
            f"Compressed AC color totally coefficients size: "
            f"({size} Mb including padding)"
        )

        ## WRITE COLOR BITSTREAMS
        META_INFO["color_coefs_entropy_mean"] = np.array(
            META_INFO["color_coefs_entropy_mean"]
        ).astype(np.float32)

        META_INFO["color_coefs_entropy_std"] = np.array(
            META_INFO["color_coefs_entropy_std"]
        ).astype(np.float32)

        np.savez_compressed(f"{path}/meta_info.npz", **META_INFO)
        np.savez_compressed(f"{path}/color_coefs_bitsteams.npz", **{
            "DC_color_coeffs": np.stack(DC_color_coeffs, axis=0),
            "AC_bitstream_0": AC_bitstreams[0],
            "AC_bitstream_1": AC_bitstreams[1],
            "AC_bitstream_2": AC_bitstreams[2],
            "AC_bitstream_3": AC_bitstreams[3],
            "AC_bitstream_4": AC_bitstreams[4],
            "AC_bitstream_5": AC_bitstreams[5],
            "AC_bitstream_6": AC_bitstreams[6],
            "AC_bitstream_7": AC_bitstreams[7],
            "AC_bitstream_8": AC_bitstreams[8],
            "AC_bitstream_9": AC_bitstreams[9],
            "AC_bitstream_10": AC_bitstreams[10],
            "AC_bitstream_11": AC_bitstreams[11],
            "AC_bitstream_12": AC_bitstreams[12],
            "AC_bitstream_13": AC_bitstreams[13],
            "AC_bitstream_14": AC_bitstreams[14],
            "AC_bitstream_15": AC_bitstreams[15],
        })

        if META_INFO["use_directional_color_transform"]:
            print("Saving with directional color transform inverse")
            color_transform_inverse = kernels_klt_transform.T @ compression_config["directional_color_transform_inverse"]
        else:
            print("Saving with KLT color transform only")
            color_transform_inverse = kernels_klt_transform.T


        np.savez_compressed(f"{path}/color_inverse_transform.npz", **{
            "color_transform_inverse": color_transform_inverse,
        })
        ## GET TOTAL BYTES SIZE ENCODED
        TOTAL_SIZES = os.path.getsize(f"{path}/meta_info.npz")
        TOTAL_SIZES += os.path.getsize(f"{path}/occupacy_bitstreams.npz")
        TOTAL_SIZES += os.path.getsize(f"{path}/geoparams_bitstreams.npz")
        TOTAL_SIZES += os.path.getsize(f"{path}/color_inverse_transform.npz")
        TOTAL_SIZES += os.path.getsize(f"{path}/color_coefs_bitsteams.npz")


        # cleaning
        del decoded_octpath_numpy
        del shs_binlevel
        del shs_coeff_binlevel
        del order_shs_coefs
        del shs_index_reverse
        del shs_coefficients
        del DC_color_coeffs
        del AC_bitstreams


        return TOTAL_SIZES


    def decode(self, path):

        META_INFO = np.load(f"{path}/meta_info.npz")
        active_sh_degree = META_INFO["active_sh_degree"]
        scene_center = torch.tensor(META_INFO["scene_center"]).cuda()
        inside_extent = torch.tensor(META_INFO["inside_extent"]).cuda()
        scene_extent = torch.tensor(META_INFO["scene_extent"]).cuda()


        ####################################################################################
        ####################################################################################
        #####                    DE-COMPRESS OCTTREE OCCUPACY                          #####
        ####################################################################################
        ####################################################################################
        print("DE-COMPRESS OCTTREE OCCUPACY")
        
        bitstreams = np.load(f"{path}/occupacy_bitstreams.npz")["bitstreams"]
        decoded_octpath, decoded_octlevel = comression_utils.decode_morton_from_bitstream(
            bitstreams,
            max_level=16
        )
        decoded_octpath = torch.tensor(decoded_octpath[:, np.newaxis], dtype=torch.int64).cuda()
        decoded_octlevel = torch.tensor(decoded_octlevel[:, np.newaxis], dtype=torch.int8).cuda()
        num_voxels = decoded_octpath.shape[0]


        ####################################################################################
        ####################################################################################
        #####                  DE-COMPRESS  GEOMETRY PARAMETERs                        #####
        ####################################################################################
        ####################################################################################
        print("DE-COMPRESS  GEOMETRY PARAMETERs ")
        # vox_center, _ = octree_utils.octpath_decoding(
        #     decoded_octpath,
        #     decoded_octlevel,
        #     scene_center,
        #     scene_extent
        # )
        grid_pts_key, _ = octree_utils.build_grid_pts_link(
            decoded_octpath, 
            decoded_octlevel
        )

        max_levels         = torch.ones((grid_pts_key.shape[0], 1), dtype=torch.int8).cuda() *MAX_NUM_LEVELS
        corners_mortoncode = svraster_cuda.utils.ijk_2_octpath(grid_pts_key, max_levels)
        corners_mortoncode = corners_mortoncode[:, 0].cpu().numpy()
        num_corners        = corners_mortoncode.shape[0]

        # 
        index2corners = np.arange(corners_mortoncode.shape[0]).astype(np.int32)
        cornersorder  = np.argsort(corners_mortoncode)
        index2corners      = index2corners[cornersorder]
        corners_mortoncode = corners_mortoncode[cornersorder]

        # Construct raht tree for corners
        corners_binlevel, corners_coeff_binlevel = raht_utils.RahtPrologue(corners_mortoncode)
        order_coefs = np.argsort(corners_coeff_binlevel)
        index_reverse = np.arange(num_corners).astype(np.int32)
        index_reverse = index_reverse[order_coefs]

        temp = np.load(f"{path}/geoparams_bitstreams.npz")
        DC_coef              = temp["DC_coef"]
        geo_params_bitstream = temp["geo_params_bitstream"]
        geo_params_stepsize  = META_INFO["geo_params_stepsize"]
        del temp

        geo_params_entropy_model = constriction.stream.model.QuantizedGaussian(
            META_INFO["geo_params_entropy_min"],
            META_INFO["geo_params_entropy_max"],
            META_INFO["geo_params_entropy_mean"],
            META_INFO["geo_params_entropy_std"]
        )

        decoder = constriction.stream.stack.AnsCoder(geo_params_bitstream)
        decoded_AC_coefs = decoder.decode(geo_params_entropy_model, num_corners-8)  # decode exactly N symbols
        decoded_AC_coefs = decoded_AC_coefs * geo_params_stepsize

        decoded_transformed_attributes = np.concatenate([DC_coef, decoded_AC_coefs[:, np.newaxis]])

        decoded_temp = np.zeros_like(decoded_transformed_attributes)
        decoded_temp[index_reverse] = decoded_transformed_attributes
        decoded_attributes_ordered = raht_utils.InvRaht(corners_binlevel, decoded_temp)

        decoded_attributes = np.zeros_like(decoded_attributes_ordered)
        decoded_attributes[index2corners] = decoded_attributes_ordered
        decoded_geo_grid_params = decoded_attributes

        # Cleaning 
        del corners_mortoncode
        del decoded_transformed_attributes
        del decoded_temp
        del decoded_attributes_ordered

        ####################################################################################
        ####################################################################################
        #####                  DE-COMPRESS COLOR PARAMETERs                            #####
        ####################################################################################
        ####################################################################################   
        print("DE-COMPRESS COLOR PARAMETERs")

        # Prepare inverse transform
        color_transform_inverse = np.load(f"{path}/color_inverse_transform.npz")["color_transform_inverse"]
        color_decoded = np.load(f"{path}/color_coefs_bitsteams.npz")
        color_coefs_stepsize = META_INFO["color_coefs_stepsize"]
        shs_binlevel, shs_coeff_binlevel = raht_utils.RahtPrologue(decoded_octpath.cpu().numpy()[:, 0])
        order_shs_coefs = np.argsort(shs_coeff_binlevel)
        shs_index_reverse = np.arange(shs_coeff_binlevel.shape[0]).astype(np.int32)
        shs_index_reverse = shs_index_reverse[order_shs_coefs]


        decoded_shs_coefficients_transformed = []
        for k in range(16):
            color_entropy_model = constriction.stream.model.QuantizedGaussian(
                META_INFO["color_coefs_entropy_min"],
                META_INFO["color_coefs_entropy_max"],
                META_INFO["color_coefs_entropy_mean"][k],
                META_INFO["color_coefs_entropy_std"][k]
            )

            AC_bitstream_k = color_decoded[f"AC_bitstream_{k}"]
            decoder = constriction.stream.stack.AnsCoder(AC_bitstream_k)
            decoded_AC_coefs_quantized = decoder.decode(color_entropy_model, (num_voxels-8) * 3)  # decode exactly N symbols
            decoded_AC_coefs_quantized = decoded_AC_coefs_quantized * color_coefs_stepsize

            decoded_coefs_ordered = np.concatenate([
                color_decoded["DC_color_coeffs"][k].reshape(8, 3),
                decoded_AC_coefs_quantized.reshape(3, (num_voxels-8)).transpose(),
            ], axis=0)

            decoded_coefs = np.zeros_like(decoded_coefs_ordered)
            decoded_coefs[shs_index_reverse] = decoded_coefs_ordered
            decoded_shs_coefficients_k = raht_utils.InvRaht(shs_binlevel, decoded_coefs)
            decoded_shs_coefficients_transformed.append(decoded_shs_coefficients_k)

        # Combined
        decoded_shs_coefficients_transformed = np.stack(
            decoded_shs_coefficients_transformed,
            axis=1
        )
        decoded_shs_coefficients = (
            decoded_shs_coefficients_transformed.reshape(-1, 48) @ color_transform_inverse
        ).reshape(-1, 16, 3)

        # Cleaning 
        del decoded_shs_coefficients_transformed
        del decoded_coefs
        del decoded_coefs_ordered
        del decoded_AC_coefs_quantized

        ####################################################################################
        ####################################################################################
        #####                  Assign Voxel Splat model                                #####
        ####################################################################################
        ####################################################################################   

        self.active_sh_degree = active_sh_degree
        self.scene_center     = scene_center
        self.inside_extent    = inside_extent
        self.scene_extent     = scene_extent

        self.octpath  = decoded_octpath
        self.octlevel = decoded_octlevel

        self._geo_grid_pts = torch.tensor(decoded_geo_grid_params).cuda().to(torch.float32)
        self._sh0          = torch.tensor(decoded_shs_coefficients[:, 0, :]).cuda().to(torch.float32)
        self._shs          = torch.tensor(decoded_shs_coefficients[:, 1:]).cuda().to(torch.float32)

