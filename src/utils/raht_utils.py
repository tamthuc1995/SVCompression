"""This module provides functions for the Region-Adaptive Hierarchical Transform (RAHT)

Functions:
    RahtPrologue()
    Raht()
    InvRaht()

Example usage:

infilename = '/home/pac/data/mpeg/PCC/CfP/Static_Objects_and_Scenes/Cat1_quantized_v2/longdress_vox10_1300.ply'
pc = PointCloud3d(filename=infilename)
pc.mortonize_and_sort()
attributes = np.asarray(pc.color, dtype=np.float32)
binlevel, coeff_binlevel = RahtPrologue(pc.morton_code)
transformed_attributes = Raht(binlevel, attributes)
recon_attributes = InvRaht(binlevel, transformed_attributes)
assert np.all(attributes.astype(np.uint8) == np.rint(recon_attributes).astype(np.uint8))
"""
import numpy as np


class Object(object):
    def __init__(self):
        return
    

def RahtPrologue(morton_code, mass=None):
    """Prologue to run Raht() and InvRaht()
    
    Args:
        morton_code: length-N int64 array of Morton codes, sorted
        mass: length-N float32 array of point masses (optional)
    
    Returns:
        binlevel: length-(bindepth+1) array of objects, where
            for binary level b=0 (root) to b=bindepth (leaves), and Nb nodes at level b,
            binlevel[b].first_descendant: length-Nb int32 array of first descendants indices
            binlevel[b].path_prefix: length-Nb int64 array of path prefix to nodes
            binlevel[b].is_left_sibling: length-Nb bool array of left sibling indicators
            binlevel[b].is_first_sibling: length-Nb bool array of first sibling indicators
            binlevel[b].mass: length-Nb float32 array of node masses
            binlevel[b].descendant_count: length-Nb int32 array of descendant counts

        coeff_binlevel: length-N int32 array of levels at which AC coefficients were created
    """

    # Validate input arrays.
    assert morton_code.dtype == np.int64
    assert np.all(morton_code[:-1] < morton_code[1:])  # is sorted and unique
    N = len(morton_code)  # number of transform coefficients
    if mass is None:
        mass = np.ones(N, dtype=np.float64)
    else:
        assert len(mass) == N
        assert mass.dtype == np.float64
    cum_mass = np.r_[np.array([0], dtype=np.float64), np.cumsum(mass)]  # exclusive cumsum
    cum_unit = np.arange(N+1, dtype=np.int64)  # exclusive cumsum of unit masses

    # Allocate output arrays.
    m = morton_code[-1]  # maximum morton code, since list is sorted
    bindepth = int(m).bit_length()  # convert to int to access convenient bit_length()
    binlevel = [Object() for b in range(bindepth+1)]
    coeff_binlevel = np.zeros(N, dtype=np.int64)

    # Process one level at a time, from leaves at b=bindepth to root at b=0.
    for b in range(bindepth, -1, -1):
        if b == bindepth:  # initialize
            first_descendant = np.arange(N, dtype=np.int64)
        else:
            first_descendant = binlevel[b+1].first_descendant[binlevel[b+1].is_first_sibling]
        path_prefix = morton_code[first_descendant] >> (bindepth - b)
        path_diff = path_prefix[:-1] ^ path_prefix[1:]
        is_left_sibling = np.r_[(path_diff >> 1) == 0, False]
        is_right_sibling = np.r_[False, is_left_sibling[:-1]]
        is_first_sibling = np.r_[True, ~is_left_sibling[:-1]]
        next_first_descendant = np.r_[first_descendant[1:], N]
        mass = cum_mass[next_first_descendant] - cum_mass[first_descendant]
        descendant_count = cum_unit[next_first_descendant] - cum_unit[first_descendant]
        right_sibling_index = first_descendant[is_right_sibling]
        coeff_binlevel[right_sibling_index] = b

        binlevel[b].first_descendant = first_descendant
        binlevel[b].path_prefix = path_prefix
        binlevel[b].is_left_sibling = is_left_sibling
        binlevel[b].is_right_sibling = is_right_sibling
        binlevel[b].is_first_sibling = is_first_sibling
        binlevel[b].mass = mass
        binlevel[b].descendant_count = descendant_count
        Nb = len(binlevel[b].first_descendant)
        assert len(binlevel[b].path_prefix) == Nb
        assert len(binlevel[b].is_left_sibling) == Nb
        assert len(binlevel[b].is_right_sibling) == Nb
        assert len(binlevel[b].is_first_sibling) == Nb
        assert len(binlevel[b].mass) == Nb
        assert len(binlevel[b].descendant_count) == Nb
        binlevel[b].count = Nb

    return binlevel, coeff_binlevel


def Raht(binlevel, attributes):
    """Transform using Region Adaptive Haar Transform (RAHT)
    
    Args:
        binlevel: length-(depth+1) array of objects, from RahtPrologue()
        attributes: NxD float32 array of attributes

    Returns:
        transformed_attributes: NxD float32 array of transformed attributes
    """

    # Allocate output array.
    transformed_attributes = np.array(attributes, dtype=np.float64)

    # Process one level at a time, from leaves at b=bindepth to root at b=0.
    bindepth = len(binlevel) - 1
    for b in range(bindepth, -1, -1):
        if not np.any(binlevel[b].is_left_sibling):
            continue
        f0 = binlevel[b].is_left_sibling
        f1 = binlevel[b].is_right_sibling
        i0 = binlevel[b].first_descendant[f0]
        i1 = binlevel[b].first_descendant[f1]
        m0 = binlevel[b].mass[f0]
        m1 = binlevel[b].mass[f1]
        x0 = transformed_attributes[i0, :]  # left sibling coefficients
        x1 = transformed_attributes[i1, :]  # right sibling coefficients
        frac = m0 / (m0 + m1)
        alpha = np.sqrt(frac)[:, np.newaxis]
        beta = np.sqrt(1 - frac)[:, np.newaxis]
        assert beta.dtype == np.float64
        assert alpha.dtype == np.float64
        transformed_attributes[i0] = alpha * x0 + beta * x1
        transformed_attributes[i1] = -beta * x0 + alpha * x1

    return transformed_attributes


def InvRaht(binlevel, transformed_attributes):
    """Inverse transform using Inverse Region Adaptive Haar Transform (IRAHT)
    
    Args:
        binlevel: length-(depth+1) array of objects, from RahtPrologue()
        transformed_attributes: NxD float32 array of transformed attributes

    Returns:
        attributes: NxD float32 array of attributes
    """

    # Allocate output array.
    attributes = np.array(transformed_attributes, dtype=np.float64)

    # Process one level at a time, from root at b=0 to leaves at b=bindepth.
    bindepth = len(binlevel) - 1
    for b in range(bindepth + 1):
        if not np.any(binlevel[b].is_left_sibling):
            continue
        f0 = binlevel[b].is_left_sibling
        f1 = binlevel[b].is_right_sibling
        i0 = binlevel[b].first_descendant[f0]
        i1 = binlevel[b].first_descendant[f1]
        m0 = binlevel[b].mass[f0]
        m1 = binlevel[b].mass[f1]
        x0 = attributes[i0, :]  # left sibling coefficients
        x1 = attributes[i1, :]  # right sibling coefficients
        frac = m0 / (m0 + m1)
        alpha = np.sqrt(frac)[:, np.newaxis]
        beta = np.sqrt(1 - frac)[:, np.newaxis]
        assert beta.dtype == np.float64
        assert alpha.dtype == np.float64
        attributes[i0] = alpha * x0 - beta * x1
        attributes[i1] = beta * x0 + alpha * x1

    return attributes
