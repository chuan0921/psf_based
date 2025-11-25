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
