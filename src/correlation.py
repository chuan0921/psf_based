import numpy as np


def gaussian_subpixel_refinement(ncc_values, peak_idx):
    """
    Three-point Gaussian peak fitting for sub-pixel accuracy.

    NCC 曲線在峰值附近近似高斯分布，用三個點擬合可獲得 sub-pixel 精度。

    公式：δ = 0.5 × (NCC[i-1] - NCC[i+1]) / (NCC[i-1] - 2×NCC[i] + NCC[i+1])

    Args:
        ncc_values: NCC values array (1D)
        peak_idx: Integer index of the peak

    Returns:
        sub_pixel_position: Refined position (float)
    """
    if peak_idx <= 0 or peak_idx >= len(ncc_values) - 1:
        return float(peak_idx)  # 邊界點無法做 sub-pixel

    y_minus = ncc_values[peak_idx - 1]
    y_center = ncc_values[peak_idx]
    y_plus = ncc_values[peak_idx + 1]

    # 確保峰值有效（中心點應該是最大的）
    if y_center <= y_minus or y_center <= y_plus:
        return float(peak_idx)

    # 拋物線擬合（等價於高斯對數擬合）
    denominator = 2 * (y_minus - 2 * y_center + y_plus)
    if abs(denominator) < 1e-10:
        return float(peak_idx)

    delta = (y_minus - y_plus) / denominator

    # 限制 delta 在 [-0.5, 0.5] 範圍內（sub-pixel 偏移不應超過半個像素）
    delta = np.clip(delta, -0.5, 0.5)

    return peak_idx + delta


def compute_multi_roi_ncc(ref_image, moving_image, roi_masks):
    num_roi = len(roi_masks)
    roi_cc = np.zeros(num_roi)

    for i, mask in enumerate(roi_masks):
        ref_pixels = ref_image[mask]
        mov_pixels = moving_image[mask]

        if len(ref_pixels) > 0 and len(mov_pixels) > 0:
            roi_cc[i] = _compute_ncc(ref_pixels, mov_pixels)
        else:
            roi_cc[i] = np.nan

    return roi_cc

def _compute_ncc(ref_pixels, mov_pixels):
    ref_mean = ref_pixels.mean()
    mov_mean = mov_pixels.mean()
    ref_std = ref_pixels.std()
    mov_std = mov_pixels.std()

    if ref_std > 0 and mov_std > 0:
        return np.mean((ref_pixels - ref_mean) * (mov_pixels - mov_mean)) / (ref_std * mov_std)
    else:
        return np.nan

def _shift_mask_x(mask, dx_pixels):
    """沿 X 方向平移 mask"""
    if dx_pixels == 0:
        return mask
    shifted = np.zeros_like(mask, dtype=bool)
    if dx_pixels > 0:
        shifted[:, dx_pixels:] = mask[:, :-dx_pixels]
    else:
        shifted[:, :dx_pixels] = mask[:, -dx_pixels:]
    return shifted

def _shift_mask_z(mask, dz_pixels):
    """沿 Z 方向平移 mask"""
    if dz_pixels == 0:
        return mask
    shifted = np.zeros_like(mask, dtype=bool)
    if dz_pixels > 0:
        shifted[dz_pixels:, :] = mask[:-dz_pixels, :]
    else:
        shifted[:dz_pixels, :] = mask[-dz_pixels:, :]
    return shifted

def _is_valid_mask(mask, image_shape):
    """檢查 mask 是否有效（有足夠的像素）"""
    return np.sum(mask) > 10  # 至少需要 10 個像素

def compute_lateral_displacement_1d(ref_image, mov_image, roi_mask,
                                   search_range_x, search_step_x, x_grid):
    """
    在 X 方向（橫向）搜尋最佳位移

    Parameters:
    -----------
    ref_image, mov_image : ndarray (Nz, Nx)
        參考和移動影像
    roi_mask : ndarray (Nz, Nx), bool
        ROI mask
    search_range_x : float
        搜尋範圍 (mm)
    search_step_x : float
        搜尋步長 (mm)
    x_grid : ndarray
        X 座標軸 (mm)

    Returns:
    --------
    best_dx : float
        最佳位移 (mm)
    max_ncc : float
        最大 NCC 值
    """
    # 提取參考 ROI
    ref_roi = ref_image[roi_mask]

    # 轉換到像素
    dx_mm_per_pixel = np.abs(x_grid[1] - x_grid[0])
    x_offsets_mm = np.arange(-search_range_x, search_range_x + search_step_x, search_step_x)
    x_offsets_pixels = np.round(x_offsets_mm / dx_mm_per_pixel).astype(int)

    max_ncc = -1
    best_dx = 0

    # 搜尋 X 方向位移
    for i, dx_pix in enumerate(x_offsets_pixels):
        # 平移 ROI mask
        shifted_mask = _shift_mask_x(roi_mask, dx_pix)

        # 邊界檢查
        if not _is_valid_mask(shifted_mask, mov_image.shape):
            continue

        # 提取移動後的 ROI
        mov_roi = mov_image[shifted_mask]

        # 確保兩個 ROI 大小一樣
        if len(mov_roi) != len(ref_roi):
            continue

        # 計算 NCC
        ncc = _compute_ncc(ref_roi, mov_roi)

        if ncc > max_ncc:
            max_ncc = ncc
            best_dx = x_offsets_mm[i]

    return best_dx, max_ncc

def compute_axial_displacement_1d(ref_image, mov_image, roi_mask,
                                 search_range_z, search_step_z, z_grid):
    """
    在 Z 方向（軸向）搜尋最佳位移

    Parameters:
    -----------
    ref_image, mov_image : ndarray (Nz, Nx)
        參考和移動影像
    roi_mask : ndarray (Nz, Nx), bool
        ROI mask
    search_range_z : float
        搜尋範圍 (mm)
    search_step_z : float
        搜尋步長 (mm)
    z_grid : ndarray
        Z 座標軸 (mm)

    Returns:
    --------
    best_dz : float
        最佳位移 (mm)
    max_ncc : float
        最大 NCC 值
    """
    # 提取參考 ROI
    ref_roi = ref_image[roi_mask]

    # 轉換到像素
    dz_mm_per_pixel = np.abs(z_grid[1] - z_grid[0])
    z_offsets_mm = np.arange(-search_range_z, search_range_z + search_step_z, search_step_z)
    z_offsets_pixels = np.round(z_offsets_mm / dz_mm_per_pixel).astype(int)

    max_ncc = -1
    best_dz = 0

    # 搜尋 Z 方向位移
    for i, dz_pix in enumerate(z_offsets_pixels):
        # 平移 ROI mask
        shifted_mask = _shift_mask_z(roi_mask, dz_pix)

        # 邊界檢查
        if not _is_valid_mask(shifted_mask, mov_image.shape):
            continue

        # 提取移動後的 ROI
        mov_roi = mov_image[shifted_mask]

        # 確保兩個 ROI 大小一樣
        if len(mov_roi) != len(ref_roi):
            continue

        # 計算 NCC
        ncc = _compute_ncc(ref_roi, mov_roi)

        if ncc > max_ncc:
            max_ncc = ncc
            best_dz = z_offsets_mm[i]

    return best_dz, max_ncc

def estimate_displacements_batch_xz(ref_image, mov_image, roi_masks,
                                   search_range_x, search_range_z,
                                   search_step_x, search_step_z,
                                   x_grid, z_grid, ncc_threshold=0.6):
    """
    批次估算多個 ROI 的 X-Z 位移

    Parameters:
    -----------
    ref_image, mov_image : ndarray (Nz, Nx)
        參考和移動影像
    roi_masks : list of ndarray
        ROI masks
    search_range_x, search_range_z : float
        搜尋範圍 (mm)
    search_step_x, search_step_z : float
        搜尋步長 (mm)
    x_grid, z_grid : ndarray
        座標軸 (mm)
    ncc_threshold : float
        NCC 閾值

    Returns:
    --------
    displacements : ndarray (N, 2)
        [dx, dz] in mm
    ncc_values : ndarray (N, 2)
        [ncc_x, ncc_z]
    """
    num_roi = len(roi_masks)
    displacements = np.zeros((num_roi, 2))
    ncc_values = np.zeros((num_roi, 2))

    for i in range(num_roi):
        # X 方向
        dx, ncc_x = compute_lateral_displacement_1d(
            ref_image, mov_image, roi_masks[i],
            search_range_x, search_step_x, x_grid
        )

        # Z 方向
        dz, ncc_z = compute_axial_displacement_1d(
            ref_image, mov_image, roi_masks[i],
            search_range_z, search_step_z, z_grid
        )

        # 檢查閾值
        if ncc_x < ncc_threshold:
            dx = np.nan
        if ncc_z < ncc_threshold:
            dz = np.nan

        displacements[i] = [dx, dz]
        ncc_values[i] = [ncc_x, ncc_z]

    return displacements, ncc_values

def estimate_x_displacement_batch(ref_image, mov_image, roi_masks,
                                   search_range_x, search_step_x, x_grid):
    """
    批次估算多個 IW 的 X 方向位移（只追蹤 X，不管 Z）

    做法：Template Matching + Sub-pixel Refinement
    - ref_patch = 從 ref_image 提取的 IW patch (template)
    - 在 mov_image 上全域搜索（X 方向），找到 NCC 最高的位置
    - 使用 Three-Point Gaussian Peak Fitting 獲得 sub-pixel 精度
    - 位移 = 找到的位置 - 原始 IW 位置

    Parameters:
    -----------
    ref_image, mov_image : ndarray (Nz, Nx)
        參考和移動影像
    roi_masks : list of ndarray
        IW masks
    search_range_x : float
        X 方向搜尋範圍 (mm)，從原始位置向左右各搜索這麼多
    search_step_x : float
        X 方向搜尋步長 (mm)（目前未使用，步長固定為 1 pixel）
    x_grid : ndarray
        X 座標軸 (mm)

    Returns:
    --------
    dx_array : ndarray (N,)
        每個 IW 的 X 位移 (mm)，正值 = 向右移動（含 sub-pixel 精度）
    ncc_array : ndarray (N,)
        每個 IW 的最大 NCC 值
    """
    num_iw = len(roi_masks)
    dx_array = np.zeros(num_iw)
    ncc_array = np.zeros(num_iw)

    dx_mm_per_pixel = np.abs(x_grid[1] - x_grid[0])
    search_range_pixels = int(search_range_x / dx_mm_per_pixel)

    image_width = mov_image.shape[1]

    for i in range(num_iw):
        mask = roi_masks[i]

        # 找到 mask 的 bounding box（原始 IW 位置）
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            dx_array[i] = 0
            ncc_array[i] = 0
            continue

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Patch 尺寸
        patch_w = cmax - cmin + 1

        # 提取參考 patch (template)
        ref_patch = ref_image[rmin:rmax+1, cmin:cmax+1].astype(np.float64)

        # 參考 patch 正規化
        ref_mean = ref_patch.mean()
        ref_std = ref_patch.std()
        if ref_std < 1e-10:
            dx_array[i] = 0
            ncc_array[i] = 0
            continue
        ref_norm = (ref_patch - ref_mean) / ref_std

        # 全域搜索範圍（在 mov_image 上）
        # search_x 是 mov_patch 的左邊界（column index）
        search_start = max(0, cmin - search_range_pixels)
        search_end = min(image_width - patch_w, cmin + search_range_pixels)
        num_search_positions = search_end - search_start + 1

        # 儲存所有搜索位置的 NCC 值（用於 sub-pixel refinement）
        ncc_curve = np.full(num_search_positions, -1.0)

        # 在 mov_image 上滑動搜索
        for idx, search_x in enumerate(range(search_start, search_end + 1)):
            # 提取 mov_image 在 search_x 位置的 patch
            mov_patch = mov_image[rmin:rmax+1, search_x:search_x+patch_w].astype(np.float64)

            if mov_patch.shape != ref_patch.shape:
                continue

            # 計算 NCC
            mov_mean = mov_patch.mean()
            mov_std = mov_patch.std()
            if mov_std < 1e-10:
                continue

            mov_norm = (mov_patch - mov_mean) / mov_std
            ncc = np.mean(ref_norm * mov_norm)
            ncc_curve[idx] = ncc

        # 找到整數峰值位置
        best_idx = np.argmax(ncc_curve)
        best_ncc = ncc_curve[best_idx]

        # Sub-pixel refinement: 使用 Three-Point Gaussian Peak Fitting
        refined_idx = gaussian_subpixel_refinement(ncc_curve, best_idx)

        # 位移 = (搜索起點 + 精確位置) - 原始位置
        # search_start + refined_idx = 在 mov_image 中找到的精確 column 位置
        # cmin = 原始 IW 的 column 位置
        displacement_pixels = (search_start + refined_idx) - cmin
        dx_array[i] = displacement_pixels * dx_mm_per_pixel
        ncc_array[i] = best_ncc

    return dx_array, ncc_array


def create_iw_results(iw_info, displacements, ncc_values, velocities, dt):
    """
    Package IW grid results with metadata.

    Parameters:
    -----------
    iw_info : list of dict
        IW grid metadata from create_iw_grid()
    displacements : ndarray (N, 2)
        [dx, dz] in mm
    ncc_values : ndarray (N, 2)
        [ncc_x, ncc_z]
    velocities : ndarray (N, 2)
        [vx, vz] in mm/s
    dt : float
        Time interval (s)

    Returns:
    --------
    results : list of dict
        Per-IW results with metadata
    """
    results = []

    for i, iw in enumerate(iw_info):
        dx, dz = displacements[i]
        ncc_x, ncc_z = ncc_values[i]
        vx, vz = velocities[i]

        # Validity flags (NCC >= 0.6)
        valid_x = not np.isnan(dx) and ncc_x >= 0.6
        valid_z = not np.isnan(dz) and ncc_z >= 0.6

        results.append({
            # Position
            'cx': iw['cx'],
            'cz': iw['cz'],
            'index': iw['index'],

            # Geometry
            'inside_vessel': iw['inside_vessel'],
            'vessel_overlap': iw['vessel_overlap'],

            # Displacement
            'dx': dx,
            'dz': dz,
            'displacement_mag': np.sqrt(dx**2 + dz**2) if valid_x and valid_z else np.nan,

            # Velocity
            'vx': vx,
            'vz': vz,
            'velocity_mag': np.sqrt(vx**2 + vz**2) if valid_x and valid_z else np.nan,

            # Quality
            'ncc_x': ncc_x,
            'ncc_z': ncc_z,
            'ncc_mean': (ncc_x + ncc_z) / 2,
            'valid_x': valid_x,
            'valid_z': valid_z,
            'valid': valid_x and valid_z
        })

    return results
