import numpy as np

def compute_error_metrics(measured, theoretical):
    abs_error = np.abs(measured - theoretical)
    return {
        'rmse': np.sqrt(np.mean((measured - theoretical)**2)),
        'mae': np.mean(abs_error),
        'max_error': abs_error.max()
    }

def compute_stability_metrics(velocities_multi_iter):
    return {
        'mean': np.mean(velocities_multi_iter, axis=1),
        'std': np.std(velocities_multi_iter, axis=1),
        'cv': np.std(velocities_multi_iter, axis=1) / np.abs(np.mean(velocities_multi_iter, axis=1))
    }

def generate_statistics_report(roi_x_positions, theo_vel, meas_vel, std_vel, R, Vmax, num_iter):
    report = []
    report.append('=' * 80)
    report.append('VELOCITY PROFILE ANALYSIS')
    report.append('=' * 80)
    report.append(f'Vessel Radius (R): {R} mm')
    report.append(f'Maximum Velocity (Vmax): {Vmax} mm/s')
    report.append(f'Number of ROI positions: {len(roi_x_positions)}')
    report.append(f'Number of iterations: {num_iter}')
    report.append('=' * 80)
    report.append('')
    report.append('DETAILED VELOCITY COMPARISON:')
    report.append('-' * 80)
    report.append(f'{"Position":>8} | {"Theoretical":>11} | {"Measured":>9} | {"Std Dev":>8} | {"Abs Error":>9}')
    report.append(f'{"(mm)":>8} | {"(mm/s)":>11} | {"(mm/s)":>9} | {"(mm/s)":>8} | {"(mm/s)":>9}')
    report.append('-' * 80)

    for i, pos in enumerate(roi_x_positions):
        abs_err = abs(meas_vel[i] - theo_vel[i])
        report.append(f'{pos:8.1f} | {theo_vel[i]:11.2f} | {meas_vel[i]:9.2f} | {std_vel[i]:8.3f} | {abs_err:9.3f}')

    report.append('-' * 80)

    metrics = compute_error_metrics(meas_vel, theo_vel)
    report.append('')
    report.append('OVERALL STATISTICS:')
    report.append(f'  RMSE: {metrics["rmse"]:.4f} mm/s')
    report.append(f'  MAE:  {metrics["mae"]:.4f} mm/s')
    report.append(f'  Max Error: {metrics["max_error"]:.4f} mm/s')
    report.append('=' * 80)

    return '\n'.join(report)

def save_results(filename, **data):
    np.savez_compressed(filename, **data)


def compute_iw_error_metrics(iw_positions, iw_velocities, iw_valid,
                              vessel_cx, vessel_cz, R, Vmax, flow_direction):
    """
    計算 IW 速度估算的誤差指標（分別計算 vx, vz, |v|）

    Parameters:
    -----------
    iw_positions : ndarray (N, 2)
        IW 中心位置 [cx, cz] (mm)
    iw_velocities : ndarray (N, 2)
        估算速度向量 [vx, vz] (mm/s)
    iw_valid : ndarray (N,)
        有效性旗標
    vessel_cx, vessel_cz : float
        血管中心座標 (mm)
    R : float
        血管半徑 (mm)
    Vmax : float
        中心最大速度 (mm/s)
    flow_direction : ndarray (3,)
        流動方向向量

    Returns:
    --------
    metrics : dict
        包含各項誤差指標
    """
    # 1. 計算徑向距離
    cx = iw_positions[:, 0]
    cz = iw_positions[:, 1]
    r = np.sqrt((cx - vessel_cx)**2 + (cz - vessel_cz)**2)

    # 2. 正規化流動方向
    flow_dir_norm = flow_direction / np.linalg.norm(flow_direction)

    # 3. 計算理論速度
    v_theo_mag = np.where(r <= R, Vmax * (1 - (r/R)**2), 0)
    vx_theo = v_theo_mag * flow_dir_norm[0]
    vz_theo = v_theo_mag * flow_dir_norm[2]

    # 4. 估算速度
    vx_est = iw_velocities[:, 0]
    vz_est = iw_velocities[:, 1]
    v_est_mag = np.sqrt(vx_est**2 + vz_est**2)

    # 5. 過濾有效 IW（在血管內）
    inside_vessel = r <= R
    valid_inside = iw_valid & inside_vessel & ~np.isnan(v_est_mag)

    if np.sum(valid_inside) == 0:
        return {
            'magnitude': {'rmse': np.nan, 'nrmse': np.nan, 'mae': np.nan, 'r_squared': np.nan},
            'vx': {'rmse': np.nan, 'nrmse': np.nan, 'mae': np.nan},
            'vz': {'rmse': np.nan, 'nrmse': np.nan, 'mae': np.nan},
            'num_valid': 0,
            'num_inside_vessel': int(np.sum(inside_vessel)),
            'relative_error_mean': np.nan,
            'relative_error_std': np.nan
        }

    # === 速度大小誤差 ===
    v_est_valid = v_est_mag[valid_inside]
    v_theo_valid = v_theo_mag[valid_inside]

    errors_mag = v_est_valid - v_theo_valid
    rmse_mag = np.sqrt(np.mean(errors_mag**2))
    nrmse_mag = (rmse_mag / Vmax) * 100
    mae_mag = np.mean(np.abs(errors_mag))

    # R-squared
    ss_res = np.sum(errors_mag**2)
    ss_tot = np.sum((v_theo_valid - np.mean(v_theo_valid))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # 相對誤差
    nonzero_mask = v_theo_valid > 1e-6
    if np.any(nonzero_mask):
        rel_errors = np.abs(errors_mag[nonzero_mask]) / v_theo_valid[nonzero_mask] * 100
        rel_error_mean = np.mean(rel_errors)
        rel_error_std = np.std(rel_errors)
    else:
        rel_error_mean = np.nan
        rel_error_std = np.nan

    # === vx 分量誤差 ===
    vx_est_valid = vx_est[valid_inside]
    vx_theo_valid = vx_theo[valid_inside]
    errors_vx = vx_est_valid - vx_theo_valid
    rmse_vx = np.sqrt(np.mean(errors_vx**2))
    nrmse_vx = (rmse_vx / (Vmax * abs(flow_dir_norm[0]))) * 100 if abs(flow_dir_norm[0]) > 1e-6 else np.nan
    mae_vx = np.mean(np.abs(errors_vx))

    # === vz 分量誤差 ===
    vz_est_valid = vz_est[valid_inside]
    vz_theo_valid = vz_theo[valid_inside]
    errors_vz = vz_est_valid - vz_theo_valid
    rmse_vz = np.sqrt(np.mean(errors_vz**2))
    nrmse_vz = (rmse_vz / (Vmax * abs(flow_dir_norm[2]))) * 100 if abs(flow_dir_norm[2]) > 1e-6 else np.nan
    mae_vz = np.mean(np.abs(errors_vz))

    return {
        'magnitude': {
            'rmse': rmse_mag,
            'nrmse': nrmse_mag,
            'mae': mae_mag,
            'r_squared': r_squared,
            'max_error': np.max(np.abs(errors_mag))
        },
        'vx': {
            'rmse': rmse_vx,
            'nrmse': nrmse_vx,
            'mae': mae_vx,
            'max_error': np.max(np.abs(errors_vx))
        },
        'vz': {
            'rmse': rmse_vz,
            'nrmse': nrmse_vz,
            'mae': mae_vz,
            'max_error': np.max(np.abs(errors_vz))
        },
        'num_valid': int(np.sum(valid_inside)),
        'num_inside_vessel': int(np.sum(inside_vessel)),
        'relative_error_mean': rel_error_mean,
        'relative_error_std': rel_error_std,
        'radial_distances': r[valid_inside],
        'v_estimated': v_est_valid,
        'v_theoretical': v_theo_valid
    }


def generate_iw_error_report(metrics, vessel_params, flow_direction):
    """
    生成格式化的誤差分析報告

    Parameters:
    -----------
    metrics : dict
        compute_iw_error_metrics() 的輸出
    vessel_params : dict
        {'R': float, 'Vmax': float}
    flow_direction : ndarray (3,)
        流動方向向量

    Returns:
    --------
    report : str
        格式化的報告字串
    """
    flow_dir_norm = flow_direction / np.linalg.norm(flow_direction)

    report = []
    report.append("=" * 80)
    report.append("VELOCITY ESTIMATION ERROR ANALYSIS")
    report.append("=" * 80)
    report.append("")
    report.append("CONFIGURATION:")
    report.append(f"  Vessel Radius (R):     {vessel_params['R']:.2f} mm")
    report.append(f"  Maximum Velocity:      {vessel_params['Vmax']:.1f} mm/s")
    report.append(f"  Flow Direction:        [{flow_dir_norm[0]:.3f}, {flow_dir_norm[1]:.3f}, {flow_dir_norm[2]:.3f}]")
    report.append("")
    report.append("SAMPLE STATISTICS:")
    report.append(f"  Valid IWs (inside vessel): {metrics['num_valid']}")
    report.append(f"  Total IWs inside vessel:   {metrics['num_inside_vessel']}")
    report.append("")
    report.append("=" * 80)
    report.append("ERROR METRICS")
    report.append("=" * 80)

    # 速度大小誤差
    mag = metrics['magnitude']
    report.append("")
    report.append("VELOCITY MAGNITUDE |v|:")
    report.append("-" * 40)
    if not np.isnan(mag['rmse']):
        report.append(f"  RMSE:           {mag['rmse']:.2f} mm/s")
        report.append(f"  NRMSE:          {mag['nrmse']:.2f}% (of Vmax)")
        report.append(f"  MAE:            {mag['mae']:.2f} mm/s")
        report.append(f"  Max Error:      {mag['max_error']:.2f} mm/s")
        report.append(f"  R-squared:      {mag['r_squared']:.4f}")
    else:
        report.append("  ERROR: No valid data")

    # vx 分量誤差
    vx = metrics['vx']
    report.append("")
    report.append(f"X-COMPONENT vx (flow_dir_x = {flow_dir_norm[0]:.3f}):")
    report.append("-" * 40)
    if not np.isnan(vx['rmse']):
        report.append(f"  RMSE:           {vx['rmse']:.2f} mm/s")
        report.append(f"  NRMSE:          {vx['nrmse']:.2f}%")
        report.append(f"  MAE:            {vx['mae']:.2f} mm/s")
        report.append(f"  Max Error:      {vx['max_error']:.2f} mm/s")
    else:
        report.append("  ERROR: No valid data")

    # vz 分量誤差
    vz = metrics['vz']
    report.append("")
    report.append(f"Z-COMPONENT vz (flow_dir_z = {flow_dir_norm[2]:.3f}):")
    report.append("-" * 40)
    if not np.isnan(vz['rmse']):
        report.append(f"  RMSE:           {vz['rmse']:.2f} mm/s")
        report.append(f"  NRMSE:          {vz['nrmse']:.2f}%")
        report.append(f"  MAE:            {vz['mae']:.2f} mm/s")
        report.append(f"  Max Error:      {vz['max_error']:.2f} mm/s")
    else:
        report.append("  ERROR: No valid data")

    # 相對誤差統計
    report.append("")
    report.append("RELATIVE ERROR STATISTICS:")
    report.append("-" * 40)
    if not np.isnan(metrics['relative_error_mean']):
        report.append(f"  Mean Relative Error:   {metrics['relative_error_mean']:.1f}%")
        report.append(f"  Std of Relative Error: {metrics['relative_error_std']:.1f}%")
    else:
        report.append("  ERROR: Cannot compute relative error")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)
