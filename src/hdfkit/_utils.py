import numpy as np
from numbers import Number

def int2binarystring(num: int, length: int = 0) -> str:
    return bin(num)[2:].zfill(length)

def bitoffset(data:np.ndarray | Number, bit_start_pos:int, bit_end_pos:int) -> np.ndarray | Number:
    # 左开右闭区间，即从bit_start_pos开始，到bit_end_pos结束，不包含bit_end_pos
    return (data >> bit_start_pos) % (2 ** (bit_end_pos - bit_start_pos))

def scale(data:np.ndarray | Number, scale_factor:Number=1, add_offset:Number=0) -> np.ndarray | Number:
    return data * scale_factor + add_offset

def mask(data:np.ndarray | Number, fill_value:Number) -> np.ma.MaskedArray:
    return np.ma.masked_values(data, fill_value, rtol=1e-8, atol=1e-9)

def split_slice_1d(sl: slice, grid_size: int) -> dict[str, slice]:
    """
    将 1D slice 按照 [0, grid_size) 边界切分为 before, middle, after 三个部分。
    """
    start = sl.start if sl.start is not None else 0
    stop = sl.stop if sl.stop is not None else grid_size
    step = sl.step if sl.step is not None else 1

    if step != 1:
        raise NotImplementedError("Only step=1 is supported for slice splitting")

    res = {}
    # Before: [-grid_size, 0) -> [0, grid_size)
    if start < 0:
        b_start = start
        b_stop = min(stop, 0)
        res["before"] = slice(b_start + grid_size, b_stop + grid_size)

    # Middle: [0, grid_size) -> [0, grid_size)
    if start < grid_size and stop > 0:
        m_start = max(start, 0)
        m_stop = min(stop, grid_size)
        res["middle"] = slice(m_start, m_stop)

    # After: [grid_size, 2*grid_size) -> [0, grid_size)
    if stop > grid_size:
        a_start = max(start, grid_size)
        a_stop = stop
        res["after"] = slice(a_start - grid_size, a_stop - grid_size)

    return res

def split_slice_2d(item: tuple[slice, slice], grid_size: int) -> tuple[dict[str, dict], tuple[int, int]]:
    """
    将 2D slice (row_slice, col_slice) 切分为 9 个区域。
    Returns:
        (slices_map, target_shape)
        slices_map: {direction: {'src': (r_slice, c_slice), 'dst': (r_slice, c_slice)}}
        target_shape: (rows, cols)
    """
    row_sl, col_sl = item
    row_parts = split_slice_1d(row_sl, grid_size)
    col_parts = split_slice_1d(col_sl, grid_size)

    # 计算目标切片 (dst)
    def calc_dst(parts):
        dsts = {}
        curr = 0
        for k, sl in parts.items():
            length = sl.stop - sl.start
            dsts[k] = slice(curr, curr + length)
            curr += length
        return dsts, curr

    row_dsts, r_len = calc_dst(row_parts)
    col_dsts, c_len = calc_dst(col_parts)

    # 映射表
    row_map = {"before": "top", "middle": "", "after": "bottom"}
    col_map = {"before": "left", "middle": "", "after": "right"}

    res = {}
    for r_key, r_sl in row_parts.items():
        for c_key, c_sl in col_parts.items():
            # 组合方向名称
            d_row = row_map[r_key]
            d_col = col_map[c_key]
            
            if not d_row and not d_col:
                direction = "center"
            else:
                direction = d_row + d_col
            
            res[direction] = {
                "src": (r_sl, c_sl),
                "dst": (row_dsts[r_key], col_dsts[c_key])
            }
    
    return res, (r_len, c_len)
