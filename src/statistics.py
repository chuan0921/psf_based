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
