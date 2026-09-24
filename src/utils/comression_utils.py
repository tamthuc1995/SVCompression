import numpy as np
from collections import deque

# ------------------------------------------------------------------
# Path table (unchanged)
# ------------------------------------------------------------------
def morton_to_path(keys, max_level):
    """keys : (L,) uint64  →  path : (L, max_level) uint8"""
    keys = np.asarray(keys, dtype=np.uint64)
    L = len(keys)
    path = np.empty((L, max_level), dtype=np.uint8)
    for d in range(max_level):
        shift = 3 * (max_level - 1 - d)
        path[:, d] = (keys >> shift) & 7
    return path

# ------------------------------------------------------------------
# Encode – sequential BFS + np.diff for run boundaries
# ------------------------------------------------------------------
def encode_from_path(path, depths, max_level):
    """
    path   : (L, max_level) uint8
    depths : (L,) int          terminal depth of each point
    Returns a compact stream of uint8:
        0          → leaf (terminal flag)
        1..255     → occupancy mask of an internal node
    """
    path   = np.asarray(path,   dtype=np.uint8)
    depths = np.asarray(depths, dtype=np.int32)
    L = path.shape[0]

    stream = []
    # queue stores (start, end, depth) – integer ranges into the sorted array
    q = deque([(0, L, 0)])

    while q:
        start, end, depth = q.popleft()
        if start >= end:
            continue

        # ---------- terminal flag ----------
        if np.all(depths[start:end] == depth):
            stream.append(0)
            continue

        # ---------- internal node ----------
        children = path[start:end, depth]          # view of child indices

        # locate every place where the child index changes
        # np.diff == 0 → same child, != 0 → new run starts
        changes = np.flatnonzero(np.diff(children)) + 1
        # boundaries of the runs: [0, change1, change2, ..., length]
        bounds  = np.concatenate(([0], changes, [end - start]))

        mask = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            c = int(children[a])                   # child index of this run
            mask |= (1 << c)
            # push the corresponding sub-range
            q.append((start + a, start + b, depth + 1))

        stream.append(mask)

    return np.asarray(stream, dtype=np.uint8)

# ------------------------------------------------------------------
# Wrappers
# ------------------------------------------------------------------
def encode_morton_to_bitstream(keys, depths, max_level, sorted=True):
    if not sorted:
        # make sure sorted
        order  = np.argsort(keys)
        keys   = keys[order]
        depths = depths[order]
        
    path = morton_to_path(keys, max_level)
    return encode_from_path(path, depths, max_level)


# ------------------------------------------------------------------
# Decode (unchanged – reconstructs keys purely from the stream)
# ------------------------------------------------------------------
def decode_morton_from_bitstream(stream, max_level):
    stream = np.asarray(stream, dtype=np.uint8)
    leaf_keys, leaf_depths = [], []
    q = deque([(np.int64(0), 0)])
    idx = 0

    while q and idx < len(stream):
        prefix, depth = q.popleft()
        mask = int(stream[idx])
        idx += 1

        if mask == 0:                               # terminal flag
            key = prefix << (3 * (max_level - depth))
            leaf_keys.append(key)
            leaf_depths.append(depth)
            continue

        for c in range(8):
            if mask & (1 << c):
                q.append(((prefix << 3) | np.int64(c), depth + 1))

    codes = np.asarray(leaf_keys,   dtype=np.int64)
    depths = np.asarray(leaf_depths, dtype=np.int32)

    order = np.argsort(codes)
    codes = codes[order]
    depths = depths[order]

    return (codes, depths)

