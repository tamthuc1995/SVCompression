
import numpy as np
from sklearn.cluster import KMeans


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

def computeSHBasisInstance(directions):

    directions_norm = np.sqrt((directions**2).sum(axis=1, keepdims=True))
    directions_norm = directions / directions_norm
    size = directions_norm.shape[0]
    print(f"Dir size={size}")

    # DEG 0
    # result = SH_C0 * sh0
    result = [np.ones(size) * SH_C0]

    #DEG 1
    x, y, z = directions_norm[:, 0], directions_norm[:, 1], directions_norm[:, 2]
    result.append( - SH_C1 * y )
    result.append( + SH_C1 * z )
    result.append( - SH_C1 * x )

    #DEG 2
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z
    result.append( SH_C2[0] * xy )
    result.append( SH_C2[1] * yz )
    result.append( SH_C2[2] * (2.0 * zz - xx - yy) )
    result.append( SH_C2[3] * xz )
    result.append( SH_C2[4] * (xx - yy) )

    
    # DEG 3
    result.append( SH_C3[0] * y * (3.0 * xx - yy) )
    result.append( SH_C3[1] * xy * z)
    result.append( SH_C3[2] * y * (4.0 * zz - xx - yy) )
    result.append( SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) )
    result.append( SH_C3[4] * x * (4.0 * zz - xx - yy) )
    result.append( SH_C3[5] * z * (xx - yy) )
    result.append( SH_C3[6] * x * (xx - 3.0 * yy) )

    result = np.stack(result, axis=-1)
    return result


def get_directional_color_transform(list_view, voxel_model):
    tr_render_opt = {
        'track_max_w': False,
        'output_depth': False,
        'output_normal': False,
        'output_T': False,
    }

    final_multi_views_gram_matrix = np.zeros((16, 16))
    for idx, view in enumerate(list_view):
        print(f"View {idx}")
        ndup_per_voxels, _ = voxel_model.collect_render_infos(view, **tr_render_opt)
        view_gram_matrix = np.zeros((16, 16))
        view_dir_size = 0.0
        for ndup in [0, 1, 2, 3, 4]: 
            all_directions_view_i = (voxel_model.vox_center[ndup_per_voxels>ndup] - view.position[None, :]).cpu().numpy()
            shs_basis = computeSHBasisInstance(all_directions_view_i)
            print(f"+ collect shs_basis more than {ndup} duplicates: {shs_basis.shape}")
            view_gram_matrix += (shs_basis.T @ shs_basis)
            view_dir_size += shs_basis.shape[0]
        view_gram_matrix = view_gram_matrix / view_dir_size
        final_multi_views_gram_matrix += view_gram_matrix

    D_sqrt_inv = np.diag(1/np.sqrt(np.diag(final_multi_views_gram_matrix)))
    gram_matrix_normed = D_sqrt_inv @ final_multi_views_gram_matrix @ D_sqrt_inv

    U, S, Vt = np.linalg.svd(gram_matrix_normed)
    sqrt_inv = U @ np.diag(1/np.sqrt(S)) @ Vt

    transform = gram_matrix_normed @ sqrt_inv
    inv_transform = sqrt_inv

    directional_color_transform = np.eye(48).reshape(16, 3, 16, 3)
    directional_color_inverse_transform = np.eye(48).reshape(16, 3, 16, 3)
    for d in range(3):
        directional_color_transform[:, d, :, d] = directional_color_transform[:, d, :, d] @ transform
        directional_color_inverse_transform[:, d, :, d] = directional_color_inverse_transform[:, d, :, d] @ inv_transform

    directional_color_transform = directional_color_transform.reshape(48, 48)
    directional_color_inverse_transform = directional_color_inverse_transform.reshape(48, 48)

    return gram_matrix_normed, directional_color_transform, directional_color_inverse_transform
