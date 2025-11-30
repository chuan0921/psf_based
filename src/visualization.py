import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def save_bmode_image(image, x, z, filename, title='B-mode Image'):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, extent=[x.min(), x.max(), z.max(), z.min()],
               cmap='gray', aspect='auto')
    plt.colorbar(label='dB')
    plt.xlabel('Lateral (mm)')
    plt.ylabel('Axial (mm)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_bmode_image(image, x, z, title='B-mode Image'):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, extent=[x.min(), x.max(), z.max(), z.min()],
               cmap='gray', aspect='auto')
    plt.colorbar(label='dB')
    plt.xlabel('Lateral (mm)')
    plt.ylabel('Axial (mm)')
    plt.title(title)
    plt.tight_layout()

def plot_rayleigh_distribution(envelope_data):
    env_vals = envelope_data.flatten()
    sigma_env = env_vals.mean() / np.sqrt(np.pi/2)

    plt.figure(figsize=(6, 4))
    plt.hist(env_vals, bins=50, density=True, color='gray', alpha=0.7, label='Empirical')

    x_vals = np.linspace(0, env_vals.max(), 100)
    y_theo = (x_vals / sigma_env**2) * np.exp(-x_vals**2 / (2*sigma_env**2))
    plt.plot(x_vals, y_theo, 'r-', linewidth=2, label='Rayleigh Fit')

    plt.xlabel('Normalized Envelope')
    plt.ylabel('PDF')
    plt.title('Speckle Envelope Distribution')
    plt.legend()
    plt.tight_layout()

def plot_velocity_profile(roi_x_positions, theoretical_vel, measured_vel, std_vel, R, Vmax):
    x_smooth = np.linspace(-5, 5, 100)
    r_smooth = np.abs(x_smooth)
    theo_smooth = np.where(r_smooth <= R, Vmax * (1 - (r_smooth/R)**2), 0)

    plt.figure(figsize=(12, 8))
    plt.plot(x_smooth, theo_smooth, 'k-', linewidth=3, label='Theoretical')
    plt.errorbar(roi_x_positions, measured_vel, yerr=std_vel,
                 fmt='ro-', linewidth=2.5, markersize=6, capsize=4,
                 label='Measured ± 1σ')

    plt.axvline(-R, color='b', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.axvline(R, color='b', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(-R-0.3, Vmax*0.9, '-R', fontsize=12, color='blue', fontweight='bold')
    plt.text(R+0.1, Vmax*0.9, '+R', fontsize=12, color='blue', fontweight='bold')

    plt.grid(True)
    plt.xlabel('Radial Position (mm)', fontsize=14)
    plt.ylabel('Velocity (mm/s)', fontsize=14)
    plt.title(f'Blood Flow Velocity Profile (Vmax={Vmax} mm/s, R={R} mm)', fontsize=16)
    plt.legend(fontsize=12)
    plt.xlim([-6, 6])
    plt.ylim([0, Vmax*1.1])
    plt.tight_layout()

def create_animation(frames, x, z, vessel_params, roi_params, filename='animation.gif'):
    import os
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame_idx):
        ax.clear()
        ax.imshow(frames[frame_idx], extent=[x.min(), x.max(), z.max(), z.min()],
                  cmap='gray', aspect='auto')
        ax.set_xlabel('Lateral (mm)')
        ax.set_ylabel('Axial (mm)')
        ax.set_title(f'Frame {frame_idx+1}')

        vessel_circle = plt.Circle((vessel_params['cx'], vessel_params['cz']),
                                   vessel_params['R'], color='r', fill=False, linewidth=2)
        ax.add_patch(vessel_circle)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100)

    try:
        anim.save(filename, writer='pillow', fps=10)
        print(f"Animation saved as: {filename}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Saving individual frames instead...")
        frame_dir = os.path.dirname(filename) if os.path.dirname(filename) else '.'
        for i, frame in enumerate(frames):
            frame_filename = os.path.join(frame_dir, f"frame_{i:03d}.png")
            save_bmode_image(frame, x, z, frame_filename, f'Frame {i}')

    plt.close()

def plot_velocity_vector_field(bmode_image, x, z, iw_positions, iw_velocities,
                                iw_valid, vessel_params, filename=None,
                                scale_factor=0.03, arrow_width=0.008):
    """
    在 B-mode 影像上繪製速度向量場

    Parameters:
    -----------
    bmode_image : ndarray (Nz, Nx)
        B-mode 超音波影像
    x, z : ndarray
        空間座標軸 (mm)
    iw_positions : ndarray (N, 2)
        IW 中心位置 [cx, cz] (mm)
    iw_velocities : ndarray (N, 2)
        速度向量 [vx, vz] (mm/s)
    iw_valid : ndarray (N,)
        有效性旗標
    vessel_params : dict
        {'cx': float, 'cz': float, 'R': float}
    filename : str, optional
        輸出檔名
    scale_factor : float
        箭頭長度縮放因子
    arrow_width : float
        箭頭寬度
    """
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. 繪製 B-mode 背景
    ax.imshow(bmode_image, extent=[x.min(), x.max(), z.max(), z.min()],
              cmap='gray', aspect='auto', alpha=0.8)

    # 2. 準備速度資料
    cx = iw_positions[:, 0]
    cz = iw_positions[:, 1]
    vx = iw_velocities[:, 0]
    vz = iw_velocities[:, 1]
    v_mag = np.sqrt(vx**2 + vz**2)

    # 3. 過濾有效向量
    valid_mask = iw_valid & ~np.isnan(v_mag)

    if np.sum(valid_mask) == 0:
        print("警告: 沒有有效的速度向量可顯示")
        plt.close()
        return None, None

    # 4. 顏色正規化
    vmax = np.nanmax(v_mag[valid_mask])
    norm = Normalize(vmin=0, vmax=vmax)
    cmap_obj = cm.get_cmap('coolwarm')

    # 5. 繪製速度箭頭
    Q = ax.quiver(cx[valid_mask], cz[valid_mask],
                  vx[valid_mask], vz[valid_mask],
                  v_mag[valid_mask],
                  cmap=cmap_obj, norm=norm,
                  scale=1/scale_factor, scale_units='xy',
                  width=arrow_width, headwidth=3, headlength=4,
                  alpha=0.9)

    # 6. 加入 colorbar
    cbar = plt.colorbar(Q, ax=ax, label='Velocity (mm/s)')

    # 7. 加入血管邊界圓圈
    vessel_circle = plt.Circle(
        (vessel_params['cx'], vessel_params['cz']),
        vessel_params['R'],
        color='lime', fill=False, linewidth=2, linestyle='--',
        label='Vessel boundary'
    )
    ax.add_patch(vessel_circle)

    # 8. 標籤和標題
    ax.set_xlabel('Lateral Position (mm)', fontsize=12)
    ax.set_ylabel('Axial Position (mm)', fontsize=12)
    ax.set_title(f'Velocity Vector Field (N={np.sum(valid_mask)} valid IWs)', fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()

    # 9. 儲存或顯示
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.close()
    return fig, ax


def plot_velocity_profile_comparison(iw_positions, iw_velocities, iw_valid,
                                      vessel_cx, vessel_cz, R, Vmax,
                                      flow_direction, filename=None):
    """
    繪製速度剖面比較圖（估算 vs 理論）

    Parameters:
    -----------
    iw_positions : ndarray (N, 2)
        IW 中心位置 [cx, cz] (mm)
    iw_velocities : ndarray (N, 2)
        速度向量 [vx, vz] (mm/s)
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
    filename : str, optional
        輸出檔名
    """
    # 1. 計算徑向距離
    cx = iw_positions[:, 0]
    cz = iw_positions[:, 1]
    r = np.sqrt((cx - vessel_cx)**2 + (cz - vessel_cz)**2)

    # 2. 計算估算速度大小
    vx_est = iw_velocities[:, 0]
    vz_est = iw_velocities[:, 1]
    v_mag_est = np.sqrt(vx_est**2 + vz_est**2)

    # 3. 正規化流動方向
    flow_dir_norm = flow_direction / np.linalg.norm(flow_direction)

    # 4. 理論 Poiseuille 速度曲線
    r_smooth = np.linspace(0, R * 1.2, 200)
    v_theo_smooth = np.where(r_smooth <= R,
                              Vmax * (1 - (r_smooth/R)**2),
                              0)

    # 5. 每個 IW 的理論速度
    v_theo_per_iw = np.where(r <= R, Vmax * (1 - (r/R)**2), 0)
    vx_theo = v_theo_per_iw * flow_dir_norm[0]
    vz_theo = v_theo_per_iw * flow_dir_norm[2]

    # 6. 過濾有效 IW（在血管內）
    valid_inside = iw_valid & (r <= R) & ~np.isnan(v_mag_est)

    # 建立圖表（3 個子圖）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # === 子圖 1: 速度大小 ===
    ax1 = axes[0]
    ax1.plot(r_smooth, v_theo_smooth, 'b-', linewidth=2.5, label='Theoretical')
    ax1.fill_between(r_smooth, v_theo_smooth, alpha=0.2, color='blue')
    ax1.scatter(r[valid_inside], v_mag_est[valid_inside],
                c='red', s=50, alpha=0.7, edgecolors='darkred',
                label='Estimated')
    ax1.axvline(x=R, color='green', linestyle='--', linewidth=2,
                label=f'R={R} mm')
    ax1.set_xlabel('Radial Distance (mm)', fontsize=11)
    ax1.set_ylabel('Velocity Magnitude (mm/s)', fontsize=11)
    ax1.set_title('|v| vs Radial Distance', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, R * 1.3])
    ax1.set_ylim([0, Vmax * 1.1])

    # === 子圖 2: vx 分量 ===
    ax2 = axes[1]
    vx_theo_smooth = v_theo_smooth * flow_dir_norm[0]
    ax2.plot(r_smooth, vx_theo_smooth, 'b-', linewidth=2.5, label='Theoretical vx')
    ax2.scatter(r[valid_inside], vx_est[valid_inside],
                c='red', s=50, alpha=0.7, edgecolors='darkred',
                label='Estimated vx')
    ax2.axvline(x=R, color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Radial Distance (mm)', fontsize=11)
    ax2.set_ylabel('vx (mm/s)', fontsize=11)
    ax2.set_title(f'vx Component (flow_dir_x={flow_dir_norm[0]:.3f})', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, R * 1.3])

    # === 子圖 3: vz 分量 ===
    ax3 = axes[2]
    vz_theo_smooth = v_theo_smooth * flow_dir_norm[2]
    ax3.plot(r_smooth, vz_theo_smooth, 'b-', linewidth=2.5, label='Theoretical vz')
    ax3.scatter(r[valid_inside], vz_est[valid_inside],
                c='red', s=50, alpha=0.7, edgecolors='darkred',
                label='Estimated vz')
    ax3.axvline(x=R, color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('Radial Distance (mm)', fontsize=11)
    ax3.set_ylabel('vz (mm/s)', fontsize=11)
    ax3.set_title(f'vz Component (flow_dir_z={flow_dir_norm[2]:.3f})', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, R * 1.3])

    plt.suptitle(f'Velocity Profile Comparison (Vmax={Vmax} mm/s, R={R} mm)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 儲存或顯示
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.close()
    return fig, axes


def plot_iw_boundaries(bmode_image, x, z, iw_info, iw_config,
                       vessel_params, speckle_info=None,
                       filename=None, max_iw_to_show=None):
    """
    在 B-mode 影像上繪製 IW 邊界框

    Parameters:
    -----------
    bmode_image : ndarray (Nz, Nx)
        B-mode 影像
    x, z : ndarray
        座標軸 (mm)
    iw_info : list of dict
        來自 create_iw_grid() 的 IW 資訊
    iw_config : InterrogationWindowConfig
        IW 配置物件
    vessel_params : dict
        {'cx': float, 'cz': float, 'R': float}
    speckle_info : dict, optional
        {'lateral': float, 'axial': float} speckle FWHM (mm)
    filename : str, optional
        輸出檔名
    max_iw_to_show : int, optional
        最多顯示幾個 IW (避免太擁擠)
    """
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. 繪製 B-mode 背景
    ax.imshow(bmode_image, extent=[x.min(), x.max(), z.max(), z.min()],
              cmap='gray', aspect='auto')

    # 2. 繪製血管邊界
    vessel_circle = plt.Circle(
        (vessel_params['cx'], vessel_params['cz']),
        vessel_params['R'],
        color='red', fill=False, linewidth=2.5, linestyle='-',
        label='Vessel boundary'
    )
    ax.add_patch(vessel_circle)

    # 3. 選擇要顯示的 IW
    iws_to_show = iw_info
    if max_iw_to_show is not None and len(iw_info) > max_iw_to_show:
        # 優先選擇在血管內的 IW
        inside_iws = [iw for iw in iw_info if iw['inside_vessel']]
        if len(inside_iws) >= max_iw_to_show:
            iws_to_show = inside_iws[:max_iw_to_show]
        else:
            iws_to_show = iw_info[:max_iw_to_show]

    # 4. 繪製 IW 邊界框
    half_x = iw_config.iw_size_x / 2
    half_z = iw_config.iw_size_z / 2

    for i, iw in enumerate(iws_to_show):
        cx, cz = iw['cx'], iw['cz']

        # 矩形左下角座標 (注意 imshow 的 y 軸反轉)
        rect = Rectangle(
            (cx - half_x, cz - half_z),
            iw_config.iw_size_x, iw_config.iw_size_z,
            linewidth=1.5, edgecolor='lime', facecolor='none',
            linestyle='--', alpha=0.8
        )
        ax.add_patch(rect)

        # 在中心標記點
        ax.plot(cx, cz, 'g.', markersize=3)

    # 5. 建立標題資訊
    title_lines = [f'IW Boundaries on B-mode Image']
    title_lines.append(f'IW size: {iw_config.iw_size_x:.2f} × {iw_config.iw_size_z:.2f} mm')

    if speckle_info:
        speckles_x = iw_config.iw_size_x / speckle_info['lateral']
        speckles_z = iw_config.iw_size_z / speckle_info['axial']
        total_speckles = speckles_x * speckles_z
        title_lines.append(f'Expected speckles per IW: ~{total_speckles:.0f} ({speckles_x:.1f} × {speckles_z:.1f})')

    title_lines.append(f'Showing {len(iws_to_show)}/{len(iw_info)} IWs')

    ax.set_title('\n'.join(title_lines), fontsize=12)
    ax.set_xlabel('Lateral Position (mm)', fontsize=11)
    ax.set_ylabel('Axial Position (mm)', fontsize=11)
    ax.legend(loc='upper right')

    plt.tight_layout()

    # 6. 儲存
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.close()
    return fig, ax


def plot_ncc_curves(all_ncc_mean, peak_frames, iw_positions, iw_valid,
                    vessel_cx, vessel_cz, R, dt, filename=None,
                    num_curves=10):
    """
    繪製 NCC 隨幀數變化的曲線（選擇性顯示部分 IW）

    Parameters:
    -----------
    all_ncc_mean : ndarray (num_iw, num_frames)
        每個 IW 在每幀的平均 NCC 值
    peak_frames : ndarray (num_iw,)
        每個 IW 的峰值幀索引
    iw_positions : ndarray (num_iw, 2)
        IW 位置 [cx, cz]
    iw_valid : ndarray (num_iw,)
        有效性旗標
    vessel_cx, vessel_cz : float
        血管中心
    R : float
        血管半徑
    dt : float
        幀間隔 (s)
    filename : str, optional
        輸出檔名
    num_curves : int
        顯示幾條曲線
    """
    num_iw, num_frames = all_ncc_mean.shape
    frames = np.arange(num_frames)
    time_axis = (frames + 1) * dt * 1000  # 轉換為 ms

    # 計算徑向距離
    cx = iw_positions[:, 0]
    cz = iw_positions[:, 1]
    r = np.sqrt((cx - vessel_cx)**2 + (cz - vessel_cz)**2)

    # 選擇在血管內且有效的 IW
    inside_mask = (r <= R) & iw_valid
    inside_indices = np.where(inside_mask)[0]

    if len(inside_indices) == 0:
        print("警告: 沒有有效的血管內 IW")
        return None, None

    # 依據徑向距離排序，選擇不同位置的 IW
    sorted_indices = inside_indices[np.argsort(r[inside_indices])]
    step = max(1, len(sorted_indices) // num_curves)
    selected_indices = sorted_indices[::step][:num_curves]

    # 建立圖表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # === 子圖 1: 個別 IW 的 NCC 曲線 ===
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

    for i, idx in enumerate(selected_indices):
        ncc_curve = all_ncc_mean[idx, :]
        peak_frame = peak_frames[idx]
        r_val = r[idx]

        ax1.plot(time_axis, ncc_curve, '-', color=colors[i], linewidth=1.5,
                 label=f'IW {idx} (r={r_val:.2f}mm)')

        # 標記峰值點
        peak_time = (peak_frame + 1) * dt * 1000
        peak_ncc = ncc_curve[peak_frame]
        ax1.plot(peak_time, peak_ncc, 'o', color=colors[i], markersize=8)

    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('NCC (mean of X and Z)', fontsize=11)
    ax1.set_title('NCC Curves for Selected IWs (inside vessel)', fontsize=12)
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time_axis[-1]])

    # === 子圖 2: 全域平均 NCC 曲線 ===
    ax2 = axes[1]
    global_mean_ncc = np.mean(all_ncc_mean[inside_mask], axis=0)
    global_std_ncc = np.std(all_ncc_mean[inside_mask], axis=0)

    ax2.fill_between(time_axis, global_mean_ncc - global_std_ncc,
                     global_mean_ncc + global_std_ncc, alpha=0.3, color='blue')
    ax2.plot(time_axis, global_mean_ncc, 'b-', linewidth=2, label='Mean ± Std')

    # 標記全域峰值
    global_peak_frame = np.argmax(global_mean_ncc)
    global_peak_time = (global_peak_frame + 1) * dt * 1000
    global_peak_ncc = global_mean_ncc[global_peak_frame]
    ax2.plot(global_peak_time, global_peak_ncc, 'ro', markersize=12,
             label=f'Peak at {global_peak_time:.1f}ms (NCC={global_peak_ncc:.3f})')
    ax2.axvline(global_peak_time, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('NCC (mean of X and Z)', fontsize=11)
    ax2.set_title(f'Global Average NCC (N={np.sum(inside_mask)} IWs inside vessel)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, time_axis[-1]])

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.close()
    return fig, axes


def plot_peak_frame_distribution(peak_frames, iw_positions, iw_valid,
                                  vessel_cx, vessel_cz, R, dt,
                                  bmode_image=None, x=None, z=None,
                                  filename=None):
    """
    繪製峰值幀分佈圖

    Parameters:
    -----------
    peak_frames : ndarray (num_iw,)
        每個 IW 的峰值幀索引
    iw_positions : ndarray (num_iw, 2)
        IW 位置 [cx, cz]
    iw_valid : ndarray (num_iw,)
        有效性旗標
    vessel_cx, vessel_cz : float
        血管中心
    R : float
        血管半徑
    dt : float
        幀間隔 (s)
    bmode_image : ndarray, optional
        B-mode 影像背景
    x, z : ndarray, optional
        座標軸
    filename : str, optional
        輸出檔名
    """
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # 計算徑向距離
    cx_arr = iw_positions[:, 0]
    cz_arr = iw_positions[:, 1]
    r = np.sqrt((cx_arr - vessel_cx)**2 + (cz_arr - vessel_cz)**2)

    # 選擇血管內且有效的 IW
    inside_mask = (r <= R) & iw_valid

    # 建立圖表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === 子圖 1: 峰值幀直方圖 ===
    ax1 = axes[0]
    valid_peaks = peak_frames[inside_mask]
    time_to_peak_ms = (valid_peaks + 1) * dt * 1000

    ax1.hist(time_to_peak_ms, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(time_to_peak_ms), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(time_to_peak_ms):.1f} ms')
    ax1.axvline(np.median(time_to_peak_ms), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(time_to_peak_ms):.1f} ms')

    ax1.set_xlabel('Time to Peak (ms)', fontsize=11)
    ax1.set_ylabel('Number of IWs', fontsize=11)
    ax1.set_title(f'Peak Time Distribution (N={len(valid_peaks)} IWs inside vessel)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === 子圖 2: 空間分佈圖 ===
    ax2 = axes[1]

    if bmode_image is not None and x is not None and z is not None:
        ax2.imshow(bmode_image, extent=[x.min(), x.max(), z.max(), z.min()],
                   cmap='gray', aspect='auto', alpha=0.5)

    # 繪製峰值幀顏色編碼的散點
    vmin, vmax = np.min(valid_peaks), np.max(valid_peaks)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap('coolwarm')

    scatter = ax2.scatter(cx_arr[inside_mask], cz_arr[inside_mask],
                          c=peak_frames[inside_mask], cmap=cmap_obj, norm=norm,
                          s=80, edgecolors='black', linewidths=0.5, alpha=0.8)

    # 血管邊界
    vessel_circle = plt.Circle(
        (vessel_cx, vessel_cz), R,
        color='lime', fill=False, linewidth=2, linestyle='--',
        label='Vessel boundary'
    )
    ax2.add_patch(vessel_circle)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2, label='Peak Frame Index')

    ax2.set_xlabel('Lateral Position (mm)', fontsize=11)
    ax2.set_ylabel('Axial Position (mm)', fontsize=11)
    ax2.set_title('Spatial Distribution of Peak Frames', fontsize=12)
    ax2.legend(loc='upper right')

    if x is not None:
        ax2.set_xlim([x.min(), x.max()])
    if z is not None:
        ax2.set_ylim([z.max(), z.min()])  # 反轉 y 軸

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.close()
    return fig, axes
