import numpy as np

def estimate_velocity(cc_curve, dt, y_distance):
    peak_frame = _detect_peak_parabolic(cc_curve)
    time_to_peak = peak_frame * dt
    
    # 避免除以零：如果 time_to_peak 為 0 或太小，返回 nan
    if time_to_peak <= 1e-9:
        return np.nan
    
    return y_distance / time_to_peak

def _detect_peak_parabolic(cc_curve):
    if len(cc_curve) < 2:
        return 0.0
    
    # 跳過 frame 0，從 frame 1 開始找 peak
    if len(cc_curve) > 1:
        peak_idx = np.argmax(cc_curve[1:]) + 1
    else:
        peak_idx = np.argmax(cc_curve)

    if 1 <= peak_idx < len(cc_curve) - 1:
        y1 = cc_curve[peak_idx - 1]
        y2 = cc_curve[peak_idx]
        y3 = cc_curve[peak_idx + 1]

        denom = y1 - 2*y2 + y3
        if abs(denom) > 1e-9:
            delta = 0.5 * (y1 - y3) / denom
            return peak_idx + delta

    return float(peak_idx)

def estimate_velocities_batch(all_roi_cc, dt, y_distance):
    num_roi, num_iter, _ = all_roi_cc.shape
    velocities = np.zeros((num_roi, num_iter))

    for i in range(num_roi):
        for j in range(num_iter):
            velocities[i, j] = estimate_velocity(all_roi_cc[i, j, :], dt, y_distance)

    # 統計有效速度數量
    valid_count = np.sum(~np.isnan(velocities))
    total_count = velocities.size
    if valid_count < total_count:
        print(f"  警告: {total_count - valid_count}/{total_count} 個速度估算無效 (peak at frame 0)")

    return velocities

def estimate_velocity_from_displacement_xz(displacement_mm, dt):
    """
    從 X-Z 位移計算速度

    Parameters:
    -----------
    displacement_mm : tuple or array (dx, dz)
        位移 (mm)
    dt : float
        時間間隔 (s)

    Returns:
    --------
    velocity : tuple (vx, vz)
        速度 (mm/s)
    """
    dx, dz = displacement_mm

    if np.isnan(dx) or np.isnan(dz) or dt <= 1e-9:
        return (np.nan, np.nan)

    vx = dx / dt
    vz = dz / dt

    return vx, vz

def estimate_velocities_batch_xz(displacements, dt):
    """
    批次計算 X-Z 速度

    Parameters:
    -----------
    displacements : ndarray (N, 2)
        位移 [dx, dz] (mm)
    dt : float
        時間間隔 (s)

    Returns:
    --------
    velocities : ndarray (N, 2)
        速度 [vx, vz] (mm/s)
    """
    num_roi = displacements.shape[0]
    velocities = np.zeros((num_roi, 2))

    for i in range(num_roi):
        velocities[i] = estimate_velocity_from_displacement_xz(
            displacements[i], dt
        )

    return velocities

def estimate_velocity_unified(ref_data, mov_data, dt, mode='xz', **kwargs):
    """
    統一的速度估算接口

    Parameters:
    -----------
    ref_data, mov_data : ndarray
        參考和移動數據（影像或體積）
    dt : float
        時間間隔 (s)
    mode : str
        'xz': X-Z 平面 speckle tracking
        'y_only': Y 方向多陣元方法（現有）
        '3d': 結合 X-Z 和 Y（未來）

    For 'xz' mode:
        roi_masks : list of masks
        search_range_x, search_range_z : float
        search_step_x, search_step_z : float
        x_grid, z_grid : ndarray
        ncc_threshold : float

    For 'y_only' mode:
        roi_masks : list of masks
        y_distance : float

    Returns:
    --------
    result : dict
        'xz': {'velocities': (N,2), 'displacements': (N,2), 'ncc': (N,2)}
        'y_only': {'velocities': (N,), ...}
    """
    if mode == 'xz':
        from src.correlation import estimate_displacements_batch_xz

        roi_masks = kwargs['roi_masks']
        search_range_x = kwargs['search_range_x']
        search_range_z = kwargs['search_range_z']
        search_step_x = kwargs['search_step_x']
        search_step_z = kwargs['search_step_z']
        x_grid = kwargs['x_grid']
        z_grid = kwargs['z_grid']
        ncc_threshold = kwargs.get('ncc_threshold', 0.3)

        # 1. 估算位移
        displacements, ncc_values = estimate_displacements_batch_xz(
            ref_data, mov_data, roi_masks,
            search_range_x, search_range_z,
            search_step_x, search_step_z,
            x_grid, z_grid, ncc_threshold
        )

        # 2. 計算速度
        velocities = estimate_velocities_batch_xz(displacements, dt)

        return {
            'velocities': velocities,  # (N, 2): [vx, vz]
            'displacements': displacements,  # (N, 2): [dx, dz]
            'ncc': ncc_values  # (N, 2): [ncc_x, ncc_z]
        }

    elif mode == 'y_only':
        # 使用現有方法
        from src.correlation import compute_multi_roi_ncc

        roi_masks = kwargs['roi_masks']
        y_distance = kwargs['y_distance']

        roi_cc = compute_multi_roi_ncc(ref_data, mov_data, roi_masks)

        # 返回與現有格式兼容的結果
        return {
            'velocities': roi_cc,  # 暫時返回相關係數
            'roi_cc': roi_cc
        }

    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: 'xz', 'y_only'")
