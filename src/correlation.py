import numpy as np
import cv2


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


# =============================================================================
# 2D Block Matching with cv2.matchTemplate (True 2D Search)
# =============================================================================

def gaussian_subpixel_refinement_2d(ncc_map, peak_row, peak_col):
    """
    2D sub-pixel refinement using parabolic fitting in both directions.

    Parameters:
    -----------
    ncc_map : ndarray (H, W)
        2D NCC correlation map
    peak_row, peak_col : int
        Integer peak position

    Returns:
    --------
    refined_row, refined_col : float
        Sub-pixel refined position
    """
    h, w = ncc_map.shape

    # Default to integer position
    refined_row = float(peak_row)
    refined_col = float(peak_col)

    # X (column) direction refinement
    if 0 < peak_col < w - 1:
        y_minus = ncc_map[peak_row, peak_col - 1]
        y_center = ncc_map[peak_row, peak_col]
        y_plus = ncc_map[peak_row, peak_col + 1]

        if y_center > y_minus and y_center > y_plus:
            denom = 2 * (y_minus - 2 * y_center + y_plus)
            if abs(denom) > 1e-10:
                delta_x = (y_minus - y_plus) / denom
                refined_col = peak_col + np.clip(delta_x, -0.5, 0.5)

    # Z (row) direction refinement
    if 0 < peak_row < h - 1:
        y_minus = ncc_map[peak_row - 1, peak_col]
        y_center = ncc_map[peak_row, peak_col]
        y_plus = ncc_map[peak_row + 1, peak_col]

        if y_center > y_minus and y_center > y_plus:
            denom = 2 * (y_minus - 2 * y_center + y_plus)
            if abs(denom) > 1e-10:
                delta_z = (y_minus - y_plus) / denom
                refined_row = peak_row + np.clip(delta_z, -0.5, 0.5)

    return refined_row, refined_col


def compute_2d_displacement_ncc(ref_image, mov_image, template_mask,
                                 search_range_x, search_range_z,
                                 x_grid, z_grid,
                                 vessel_params=None):
    """
    真正的 2D block matching with NCC (使用 cv2.matchTemplate)

    搜尋視窗：
    - X 方向: [0, +search_range_x] mm (正向流)
    - Z 方向: [-search_range_z, +search_range_z] mm (以 IW 中心為基準)
    - 若提供 vessel_params，會裁切到血管邊界內

    Parameters:
    -----------
    ref_image : ndarray (Nz, Nx)
        參考影像
    mov_image : ndarray (Nz, Nx)
        移動影像
    template_mask : ndarray (Nz, Nx), bool
        IW 區域 mask
    search_range_x : float
        X 方向搜尋範圍 (mm)，只搜正向 [0, +search_range_x]
    search_range_z : float
        Z 方向搜尋範圍 (mm)，搜 [-search_range_z, +search_range_z]
    x_grid, z_grid : ndarray
        座標軸 (mm)
    vessel_params : dict, optional
        血管參數 {'cx': float, 'cz': float, 'R': float}
        若提供，會將 search window 裁切到血管內

    Returns:
    --------
    best_dx : float
        X 位移 (mm)，正值表示正向流
    best_dz : float
        Z 位移 (mm)，正值表示向下
    max_ncc : float
        最大 NCC 值
    """
    # 1. 從 ref_image 提取 template (IW 區域)
    rows, cols = np.where(template_mask)
    if len(rows) == 0 or len(cols) == 0:
        return 0.0, 0.0, 0.0

    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    template = ref_image[r_min:r_max+1, c_min:c_max+1]

    template_h, template_w = template.shape

    # 2. 計算像素尺寸
    dx_per_pix = abs(x_grid[1] - x_grid[0])
    dz_per_pix = abs(z_grid[1] - z_grid[0])

    # 3. 定義 search region 邊界（在 mov_image 中）
    # X: 從 template 原始位置開始，向正向搜索 search_range_x
    # Z: 從 template 原始位置上下各搜索 search_range_z
    search_range_x_pix = int(search_range_x / dx_per_pix)
    search_range_z_pix = int(search_range_z / dz_per_pix)

    # Search region 邊界（基礎）
    # X: [c_min, c_min + search_range_x_pix + template_w]
    # Z: [r_min - search_range_z_pix, r_min + search_range_z_pix + template_h]
    sr_c_start = c_min
    sr_c_end = min(c_min + search_range_x_pix + template_w, mov_image.shape[1])
    sr_r_start = max(r_min - search_range_z_pix, 0)
    sr_r_end = min(r_min + search_range_z_pix + template_h, mov_image.shape[0])

    # 4. 血管邊界裁切（若提供 vessel_params）
    if vessel_params is not None:
        vessel_cx = vessel_params['cx']
        vessel_cz = vessel_params['cz']
        vessel_R = vessel_params['R']

        # IW 中心在 mm 座標
        iw_cx = x_grid[c_min + template_w // 2]
        iw_cz = z_grid[r_min + template_h // 2]

        # 計算該 X 位置的血管 Z 邊界
        # 圓方程: (x - cx)^2 + (z - cz)^2 = R^2
        # 解 z: z = cz ± sqrt(R^2 - (x - cx)^2)
        dx_from_center = iw_cx - vessel_cx
        if abs(dx_from_center) < vessel_R:
            z_extent = np.sqrt(vessel_R**2 - dx_from_center**2)
            z_top_mm = vessel_cz - z_extent  # 血管上邊界 (mm)
            z_bottom_mm = vessel_cz + z_extent  # 血管下邊界 (mm)

            # 轉換為像素座標
            # z_grid[0] 是影像頂部，值較小
            z_top_pix = int((z_top_mm - z_grid[0]) / dz_per_pix)
            z_bottom_pix = int((z_bottom_mm - z_grid[0]) / dz_per_pix)

            # 裁切 search region 到血管內
            sr_r_start = max(sr_r_start, z_top_pix)
            sr_r_end = min(sr_r_end, z_bottom_pix)

    # 確保 search region 足夠大
    if sr_c_end - sr_c_start < template_w or sr_r_end - sr_r_start < template_h:
        return 0.0, 0.0, 0.0

    search_region = mov_image[sr_r_start:sr_r_end, sr_c_start:sr_c_end]

    # 4. 使用 cv2.matchTemplate 計算 NCC map（一次計算所有位置！）
    ncc_map = cv2.matchTemplate(
        search_region.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCOEFF_NORMED
    )
    # ncc_map shape: (sr_h - template_h + 1, sr_w - template_w + 1)

    if ncc_map.size == 0:
        return 0.0, 0.0, 0.0

    # 5. 找到最大 NCC 位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc_map)
    best_col, best_row = max_loc  # (x, y) in ncc_map coordinates

    # 6. Sub-pixel refinement
    refined_row, refined_col = gaussian_subpixel_refinement_2d(
        ncc_map, best_row, best_col
    )

    # 7. 轉換回位移 (mm)
    # ncc_map 的 (0, 0) 對應 search_region 的 (0, 0)
    # search_region 的 (0, 0) 對應 mov_image 的 (sr_r_start, sr_c_start)
    # template 原本在 mov_image 的 (r_min, c_min)
    #
    # 在 ncc_map 座標系中：
    # - (0, 0) 表示 template 放在 search_region 的左上角
    # - template 原始位置對應 ncc_map 的 (r_min - sr_r_start, c_min - sr_c_start) = (r_min - sr_r_start, 0)
    #
    # 位移計算：
    # dx_pix = refined_col - (c_min - sr_c_start) = refined_col - 0 = refined_col
    # dz_pix = refined_row - (r_min - sr_r_start)

    origin_row_in_ncc = r_min - sr_r_start
    origin_col_in_ncc = 0  # 因為 sr_c_start = c_min

    dx_pix = refined_col - origin_col_in_ncc
    dz_pix = refined_row - origin_row_in_ncc

    best_dx = dx_pix * dx_per_pix
    best_dz = dz_pix * dz_per_pix

    return best_dx, best_dz, max_val


def estimate_2d_displacement_batch(ref_image, mov_image, iw_masks,
                                    search_range_x, search_range_z,
                                    x_grid, z_grid,
                                    vessel_params=None):
    """
    批次處理所有 IW 的 2D 位移估計

    使用 cv2.matchTemplate 進行高效 2D block matching。

    Parameters:
    -----------
    ref_image, mov_image : ndarray (Nz, Nx)
        參考和移動影像
    iw_masks : list of ndarray
        IW masks
    search_range_x : float
        X 方向搜尋範圍 (mm)，只搜正向 [0, +search_range_x]
    search_range_z : float
        Z 方向搜尋範圍 (mm)，搜 [-search_range_z, +search_range_z]
    x_grid, z_grid : ndarray
        座標軸 (mm)
    vessel_params : dict, optional
        血管參數 {'cx': float, 'cz': float, 'R': float}
        若提供，會將 search window 裁切到血管內

    Returns:
    --------
    displacements : ndarray (num_iw, 2)
        [dx, dz] in mm
    ncc_values : ndarray (num_iw,)
        max NCC per IW
    """
    num_iw = len(iw_masks)
    displacements = np.zeros((num_iw, 2))
    ncc_values = np.zeros(num_iw)

    for i, mask in enumerate(iw_masks):
        dx, dz, ncc = compute_2d_displacement_ncc(
            ref_image, mov_image, mask,
            search_range_x, search_range_z,
            x_grid, z_grid,
            vessel_params=vessel_params
        )
        displacements[i] = [dx, dz]
        ncc_values[i] = ncc

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
