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


def save_combined_velocity_profiles(iw_positions, iw_velocities, iw_valid,
                                     vessel_cx, vessel_cz, R, Vmax,
                                     flow_direction, filename=None):
    """
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
    from matplotlib.gridspec import GridSpec

    # 計算基本數據
    cx = iw_positions[:, 0]
    cz = iw_positions[:, 1]
    r = np.sqrt((cx - vessel_cx)**2 + (cz - vessel_cz)**2)

    vx_est = iw_velocities[:, 0]
    vz_est = iw_velocities[:, 1]
    v_mag_est = np.sqrt(vx_est**2 + vz_est**2)

    flow_dir_norm = flow_direction / np.linalg.norm(flow_direction)

    # 理論曲線
    r_smooth = np.linspace(0, R * 1.2, 200)
    v_theo_smooth = np.where(r_smooth <= R, Vmax * (1 - (r_smooth/R)**2), 0)

    # 過濾有效 IW
    valid_inside = iw_valid & (r <= R) & ~np.isnan(v_mag_est)

    # === 建立圖表 ===
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # 排序有效數據（依 r 排序，用於連線）
    sort_idx = np.argsort(r[valid_inside])
    r_sorted = r[valid_inside][sort_idx]
    v_mag_sorted = v_mag_est[valid_inside][sort_idx]
    vx_sorted = vx_est[valid_inside][sort_idx]
    vz_sorted = vz_est[valid_inside][sort_idx]

    # === 上排：折線圖 ===
    # 子圖 1: |v| vs r
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(r_smooth, v_theo_smooth, 'b-', linewidth=2.5, label='Theoretical')
    ax1.fill_between(r_smooth, v_theo_smooth, alpha=0.2, color='blue')
    ax1.plot(r_sorted, v_mag_sorted, 'ro-', linewidth=1.5, markersize=6,
             markeredgecolor='darkred', label='Estimated')
    ax1.axvline(x=R, color='green', linestyle='--', linewidth=2, label=f'R={R} mm')
    ax1.set_xlabel('Radial Distance (mm)', fontsize=10)
    ax1.set_ylabel('Velocity Magnitude (mm/s)', fontsize=10)
    ax1.set_title('|v| vs Radial Distance', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, R * 1.3])
    ax1.set_ylim([0, Vmax * 1.1])

    # 子圖 2: vx vs r
    ax2 = fig.add_subplot(gs[0, 1])
    vx_theo_smooth = v_theo_smooth * flow_dir_norm[0]
    ax2.plot(r_smooth, vx_theo_smooth, 'b-', linewidth=2.5, label='Theoretical vx')
    ax2.plot(r_sorted, vx_sorted, 'ro-', linewidth=1.5, markersize=6,
             markeredgecolor='darkred', label='Estimated vx')
    ax2.axvline(x=R, color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Radial Distance (mm)', fontsize=10)
    ax2.set_ylabel('vx (mm/s)', fontsize=10)
    ax2.set_title(f'vx Component (flow_dir_x={flow_dir_norm[0]:.3f})', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, R * 1.3])

    # 子圖 3: vz vs r
    ax3 = fig.add_subplot(gs[0, 2])
    vz_theo_smooth = v_theo_smooth * flow_dir_norm[2]
    ax3.plot(r_smooth, vz_theo_smooth, 'b-', linewidth=2.5, label='Theoretical vz')
    ax3.plot(r_sorted, vz_sorted, 'ro-', linewidth=1.5, markersize=6,
             markeredgecolor='darkred', label='Estimated vz')
    ax3.axvline(x=R, color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('Radial Distance (mm)', fontsize=10)
    ax3.set_ylabel('vz (mm/s)', fontsize=10)
    ax3.set_title(f'vz Component (flow_dir_z={flow_dir_norm[2]:.3f})', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, R * 1.3])

    # === 下排：Centerline 曲線圖 (span 整排) ===
    ax4 = fig.add_subplot(gs[1, :])

    # 計算中心線數據
    x_rel = iw_positions[:, 0] - vessel_cx
    z_rel = iw_positions[:, 1] - vessel_cz

    # 找中心線 IW
    unique_z = np.unique(np.round(z_rel, 2))
    center_z = unique_z[np.argmin(np.abs(unique_z))]
    z_tolerance = 0.1
    on_centerline = np.abs(z_rel - center_z) < z_tolerance

    # 過濾
    r_abs = np.abs(x_rel)
    valid_curve = iw_valid & (r_abs <= R) & ~np.isnan(vx_est) & on_centerline

    # 排序
    sort_idx = np.argsort(x_rel[valid_curve])
    x_sorted = x_rel[valid_curve][sort_idx]
    vx_sorted = vx_est[valid_curve][sort_idx]

    # 理論曲線
    x_theo = np.linspace(-R, R, 200)
    vx_theo = Vmax * (1 - (x_theo/R)**2)

    ax4.plot(x_theo, vx_theo, 'b-', linewidth=2.5, label='Theoretical (Poiseuille)')
    ax4.plot(x_sorted, vx_sorted, 'ro-', linewidth=1.5, markersize=6, label='Estimated')
    ax4.axvline(-R, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Vessel wall (±{R}mm)')
    ax4.axvline(R, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Position relative to vessel center (mm)', fontsize=11)
    ax4.set_ylabel('vx (mm/s)', fontsize=11)
    ax4.set_title('Centerline Velocity Profile', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-R*1.1, R*1.1])
    ax4.set_ylim([min(0, vx_sorted.min()*1.1) if len(vx_sorted) > 0 else 0, Vmax*1.1])

    plt.suptitle(f'Velocity Profile Analysis (Vmax={Vmax} mm/s, R={R} mm)',
                 fontsize=14, fontweight='bold')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.close()
    return fig


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


def save_combined_bmode_images(ref_image, matched_image, x, z,
                                iw_info, iw_config, vessel_params,
                                matched_frame_idx, filename=None):
    """
    Parameters:
    -----------
    ref_image : ndarray (Nz, Nx)
        參考影像 (Frame 0)
    matched_image : ndarray (Nz, Nx)
        匹配影像 (Frame N)
    x, z : ndarray
        座標軸 (mm)
    iw_info : list of dict
        IW 資訊 (來自 create_iw_grid)
    iw_config : InterrogationWindowConfig
        IW 配置
    vessel_params : dict
        {'cx': float, 'cz': float, 'R': float}
    matched_frame_idx : int
        匹配影像的幀索引
    filename : str, optional
        輸出檔名
    """
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    extent = [x.min(), x.max(), z.max(), z.min()]

    # === 左側：Reference Image + IW 網格 ===
    ax1 = axes[0]
    ax1.imshow(ref_image, extent=extent, cmap='gray', aspect='auto')
    ax1.set_title('Reference Image (Frame 0)', fontsize=12)
    ax1.set_xlabel('Lateral (mm)')
    ax1.set_ylabel('Axial (mm)')

    # 繪製半透明 IW 網格
    half_x = iw_config.iw_size_x / 2
    half_z = iw_config.iw_size_z / 2

    for iw in iw_info:
        cx, cz = iw['cx'], iw['cz']
        rect = Rectangle(
            (cx - half_x, cz - half_z),
            iw_config.iw_size_x, iw_config.iw_size_z,
            linewidth=1, edgecolor='cyan', facecolor='none',
            linestyle='-', alpha=0.4
        )
        ax1.add_patch(rect)

    # 血管邊界
    vessel_circle = plt.Circle(
        (vessel_params['cx'], vessel_params['cz']),
        vessel_params['R'],
        color='red', fill=False, linewidth=2, linestyle='--'
    )
    ax1.add_patch(vessel_circle)

    # === 右側：Matched Image ===
    ax2 = axes[1]
    ax2.imshow(matched_image, extent=extent, cmap='gray', aspect='auto')
    ax2.set_title(f'Matched Image (Frame {matched_frame_idx})', fontsize=12)
    ax2.set_xlabel('Lateral (mm)')
    ax2.set_ylabel('Axial (mm)')

    # 血管邊界
    vessel_circle2 = plt.Circle(
        (vessel_params['cx'], vessel_params['cz']),
        vessel_params['R'],
        color='red', fill=False, linewidth=2, linestyle='--'
    )
    ax2.add_patch(vessel_circle2)

    plt.suptitle('B-mode Images: Reference vs Matched', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.close()
    return fig, axes


def plot_ncc_curves(all_ncc_iterations, peak_frames, iw_positions, iw_valid,
                    vessel_cx, vessel_cz, R, filename=None,
                    num_curves=10):
    """
    繪製 NCC 隨幀數變化的曲線（支援多次迭代數據）

    Parameters:
    -----------
    all_ncc_iterations : ndarray
        Shape (num_iterations, num_iw, num_frames) 或 (num_iw, num_frames)
        若為 2D 則自動擴展為 3D
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
    filename : str, optional
        輸出檔名
    num_curves : int
        顯示幾條曲線
    """
    # 處理 2D/3D 輸入
    if all_ncc_iterations.ndim == 2:
        # 若為 2D，擴展為 3D (1 iteration)
        all_ncc_iterations = all_ncc_iterations[np.newaxis, :, :]

    num_iterations, num_iw, num_frames = all_ncc_iterations.shape
    frames = np.arange(1, num_frames + 1)  # 1-indexed (Frame 1 = first comparison after 1*dt)

    # 計算徑向距離
    cx = iw_positions[:, 0]
    cz = iw_positions[:, 1]
    r = np.sqrt((cx - vessel_cx)**2 + (cz - vessel_cz)**2)

    # 選擇在血管內且有效的 IW
    inside_mask = (r <= R) & iw_valid
    inside_indices = np.where(inside_mask)[0]

    if len(inside_indices) == 0:
        print("Warning: No valid IWs inside vessel")
        return None, None

    # 依據徑向距離排序，選擇不同位置的 IW
    sorted_indices = inside_indices[np.argsort(r[inside_indices])]
    step = max(1, len(sorted_indices) // num_curves)
    selected_indices = sorted_indices[::step][:num_curves]

    # 建立圖表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # === 子圖 1: 個別 IW 的 NCC 曲線（平均值）===
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

    # 計算平均 NCC（跨迭代）
    mean_ncc = np.mean(all_ncc_iterations, axis=0)  # (num_iw, num_frames)

    for i, idx in enumerate(selected_indices):
        ncc_curve = mean_ncc[idx, :]
        peak_frame_idx = peak_frames[idx]  # 0-indexed array index
        r_val = r[idx]

        ax1.plot(frames, ncc_curve, '-', color=colors[i], linewidth=1.5,
                 label=f'IW {idx} (r={r_val:.2f}mm)')

        # 標記峰值點 (轉換為 1-indexed)
        peak_ncc = ncc_curve[peak_frame_idx]
        ax1.plot(peak_frame_idx + 1, peak_ncc, 'o', color=colors[i], markersize=8)

    ax1.set_xlabel('Frame', fontsize=11)
    ax1.set_ylabel('NCC', fontsize=11)
    ax1.set_title('NCC Curves for Selected IWs (dots = last valid frame)', fontsize=12)
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, num_frames])

    # === 子圖 2: 全域 NCC 曲線（min-max 範圍 + 平均值）===
    ax2 = axes[1]

    # 計算每個迭代的全域平均（跨 IW）
    # Shape: (num_iterations, num_frames)
    global_per_iter = np.mean(all_ncc_iterations[:, inside_mask, :], axis=1)

    # 計算 min-max 範圍和平均值（跨迭代）
    global_mean = np.mean(global_per_iter, axis=0)  # (num_frames,)
    global_min = np.min(global_per_iter, axis=0)    # (num_frames,)
    global_max = np.max(global_per_iter, axis=0)    # (num_frames,)

    # 繪製 min-max 範圍（淺紫色）
    ax2.fill_between(frames, global_min, global_max, alpha=0.3, color='purple',
                     label=f'Min-Max range ({num_iterations} iterations)')

    # 繪製平均值（深藍色）
    ax2.plot(frames, global_mean, 'darkblue', linewidth=2, label='Mean')

    # 標記全域峰值 (轉換為 1-indexed)
    global_peak_idx = np.argmax(global_mean)  # 0-indexed array index
    global_peak_frame = global_peak_idx + 1   # 1-indexed for display
    global_peak_ncc = global_mean[global_peak_idx]
    ax2.plot(global_peak_frame, global_peak_ncc, 'ro', markersize=12,
             label=f'Peak at Frame {global_peak_frame} (NCC={global_peak_ncc:.3f})')
    ax2.axvline(global_peak_frame, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Frame', fontsize=11)
    ax2.set_ylabel('NCC', fontsize=11)
    ax2.set_title(f'Global Average NCC (N={np.sum(inside_mask)} IWs, {num_iterations} iterations)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, num_frames])

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


def compute_ncc_curve(ref_patch, mov_image, rmin, rmax, cmin, search_range_pixels):
    """
    計算 NCC 曲線：在 mov_image 上滑動搜索

    Parameters:
    -----------
    ref_patch : ndarray
        參考 patch
    mov_image : ndarray
        移動影像
    rmin, rmax : int
        行範圍
    cmin : int
        參考 patch 的起始列
    search_range_pixels : int
        搜索範圍（像素）

    Returns:
    --------
    search_positions : ndarray
        搜索位置偏移量
    ncc_values : ndarray
        NCC 值
    """
    patch_w = ref_patch.shape[1]
    image_width = mov_image.shape[1]

    ref_mean = ref_patch.mean()
    ref_std = ref_patch.std()
    if ref_std < 1e-10:
        return np.array([0]), np.array([0])
    ref_norm = (ref_patch - ref_mean) / ref_std

    search_start = max(0, cmin - search_range_pixels)
    search_end = min(image_width - patch_w, cmin + search_range_pixels)

    search_positions = []
    ncc_values = []

    for search_x in range(search_start, search_end + 1):
        mov_patch = mov_image[rmin:rmax+1, search_x:search_x+patch_w].astype(np.float64)

        if mov_patch.shape != ref_patch.shape:
            continue

        mov_mean = mov_patch.mean()
        mov_std = mov_patch.std()
        if mov_std < 1e-10:
            ncc = 0
        else:
            mov_norm = (mov_patch - mov_mean) / mov_std
            ncc = np.mean(ref_norm * mov_norm)

        offset = search_x - cmin
        search_positions.append(offset)
        ncc_values.append(ncc)

    return np.array(search_positions), np.array(ncc_values)


def create_ncc_tracking_animation(all_frames, ref_image, iw_mask, iw_index,
                                   x, z, search_range_pixels,
                                   output_path, fps=5, max_frames=50):
    """
    建立 NCC tracking 視覺化動畫

    Parameters:
    -----------
    all_frames : list of ndarray
        所有影像幀
    ref_image : ndarray
        參考影像
    iw_mask : ndarray
        IW mask
    iw_index : int
        IW 索引
    x, z : ndarray
        座標軸 (mm)
    search_range_pixels : int
        搜索範圍（像素）
    output_path : str
        輸出路徑
    fps : int
        動畫幀率
    max_frames : int
        最大幀數
    """
    from matplotlib.patches import Rectangle
    from tqdm import tqdm

    # 找到 IW 的 bounding box
    rows = np.any(iw_mask, axis=1)
    cols = np.any(iw_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 提取參考 patch
    ref_patch = ref_image[rmin:rmax+1, cmin:cmax+1].astype(np.float64)

    dx = np.abs(x[1] - x[0])

    # IW 在 mm 座標中的位置
    x_min_mm = x[cmin]
    x_max_mm = x[cmax]
    z_min_mm = z[rmin]
    z_max_mm = z[rmax]

    # 預計算所有幀的 NCC 曲線
    num_frames = min(len(all_frames), max_frames)
    all_ncc_data = []

    print(f"預計算 {num_frames} 幀的 NCC 曲線...")
    for frame_idx in tqdm(range(num_frames)):
        mov_image = all_frames[frame_idx]
        positions, ncc_curve = compute_ncc_curve(
            ref_patch, mov_image, rmin, rmax, cmin, search_range_pixels
        )

        if len(ncc_curve) > 0:
            best_idx = np.argmax(ncc_curve)
            best_ncc = ncc_curve[best_idx]
            best_offset = positions[best_idx]
        else:
            best_idx, best_ncc, best_offset = 0, 0, 0

        all_ncc_data.append({
            'positions': positions,
            'ncc_curve': ncc_curve,
            'best_idx': best_idx,
            'best_ncc': best_ncc,
            'best_offset': best_offset
        })

    # 建立圖表
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.5], hspace=0.3, wspace=0.2)

    ax_ref = fig.add_subplot(gs[0, 0])
    ax_mov = fig.add_subplot(gs[0, 1])
    ax_ref_zoom = fig.add_subplot(gs[1, 0])
    ax_mov_zoom = fig.add_subplot(gs[1, 1])
    ax_ncc = fig.add_subplot(gs[2, :])

    extent = [x.min(), x.max(), z.max(), z.min()]

    # Reference Image (固定)
    ax_ref.imshow(ref_image, extent=extent, cmap='gray', aspect='auto')
    ax_ref.set_title('Reference Image (Frame 0)', fontsize=12)
    ax_ref.set_xlabel('Lateral (mm)')
    ax_ref.set_ylabel('Axial (mm)')
    rect_ref = Rectangle((x_min_mm, z_min_mm), x_max_mm - x_min_mm, z_max_mm - z_min_mm,
                          linewidth=2, edgecolor='yellow', facecolor='none')
    ax_ref.add_patch(rect_ref)

    # Reference zoom (固定)
    ax_ref_zoom.imshow(ref_patch, cmap='gray', aspect='auto')
    ax_ref_zoom.set_title('Reference IW Patch (Template)', fontsize=11)
    ax_ref_zoom.axis('off')

    # Moving Image
    im_mov = ax_mov.imshow(all_frames[0], extent=extent, cmap='gray', aspect='auto')
    ax_mov.set_title('Moving Image (Frame 0)', fontsize=12)
    ax_mov.set_xlabel('Lateral (mm)')
    ax_mov.set_ylabel('Axial (mm)')

    # Search Window (紅色虛線框) - 固定顯示搜索範圍
    search_range_mm = search_range_pixels * dx
    search_window_x_min = x_min_mm - search_range_mm
    search_window_x_max = x_max_mm + search_range_mm
    search_window_width = search_window_x_max - search_window_x_min
    rect_search = Rectangle((search_window_x_min, z_min_mm), search_window_width, z_max_mm - z_min_mm,
                              linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
                              label='Search Window')
    ax_mov.add_patch(rect_search)

    # Best Match (綠色框) - 會隨 NCC peak 移動
    rect_mov = Rectangle((x_min_mm, z_min_mm), x_max_mm - x_min_mm, z_max_mm - z_min_mm,
                          linewidth=2, edgecolor='lime', facecolor='none', label='Best Match')
    ax_mov.add_patch(rect_mov)

    # 圖例
    ax_mov.legend(loc='upper right', fontsize=8)

    # Moving zoom
    im_mov_zoom = ax_mov_zoom.imshow(ref_patch, cmap='gray', aspect='auto')
    ax_mov_zoom.set_title('Best Match Patch', fontsize=11)
    ax_mov_zoom.axis('off')

    # NCC curve
    line_ncc, = ax_ncc.plot([], [], 'b-', linewidth=2, label='NCC')
    peak_marker, = ax_ncc.plot([], [], 'ro', markersize=10, label='Peak')
    ax_ncc.set_xlim(-search_range_pixels - 5, search_range_pixels + 5)
    ax_ncc.set_ylim(-0.2, 1.1)
    ax_ncc.set_xlabel('Search Offset (pixels)', fontsize=11)
    ax_ncc.set_ylabel('NCC', fontsize=11)
    ax_ncc.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax_ncc.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax_ncc.legend(loc='upper right')
    ax_ncc.grid(True, alpha=0.3)

    fig.suptitle(f'NCC Template Matching - IW {iw_index}', fontsize=14, fontweight='bold')

    info_text = ax_ncc.text(0.02, 0.95, '', transform=ax_ncc.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def update(frame_idx):
        data = all_ncc_data[frame_idx]
        mov_image = all_frames[frame_idx]

        im_mov.set_array(mov_image)
        ax_mov.set_title(f'Moving Image (Frame {frame_idx})', fontsize=12)

        offset_mm = data['best_offset'] * dx
        rect_mov.set_xy((x_min_mm + offset_mm, z_min_mm))

        best_cmin = cmin + data['best_offset']
        if 0 <= best_cmin < mov_image.shape[1] - (cmax - cmin):
            mov_patch = mov_image[rmin:rmax+1, best_cmin:best_cmin+(cmax-cmin+1)]
            im_mov_zoom.set_array(mov_patch)

        line_ncc.set_data(data['positions'], data['ncc_curve'])
        peak_marker.set_data([data['best_offset']], [data['best_ncc']])

        displacement_mm = data['best_offset'] * dx
        info_str = (f"Frame: {frame_idx}/{num_frames-1}  |  "
                    f"Best NCC: {data['best_ncc']:.3f}  |  "
                    f"Offset: {data['best_offset']} px ({displacement_mm:.3f} mm)")
        info_text.set_text(info_str)

        return [im_mov, rect_mov, im_mov_zoom, line_ncc, peak_marker, info_text]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=False)

    print(f"儲存動畫到 {output_path}...")
    anim.save(output_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    print(f"完成！")
