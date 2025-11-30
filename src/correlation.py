import numpy as np

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
