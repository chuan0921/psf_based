"""
NCC Template Matching 視覺化動畫

展示 Reference Image 與 Moving Image 之間的 NCC 計算過程
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import os

# 設定使用 GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
GPU_ID = 0

import cupy as cp
cp.cuda.Device(GPU_ID).use()

from tqdm import tqdm
from src.setting import *
from src.setting import calculate_element_centers, calculate_elevational_pitch
from src.phantom import generate_vessel_phantom, update_scatterer_positions, check_and_regenerate_scatterers
from src.speckle_image import calculate_psf_parameters, generate_ultrasound_image, apply_envelope_detection
from src.grid import create_spatial_grid, create_iw_grid, extract_iw_masks


def compute_ncc_curve(ref_patch, mov_image, rmin, rmax, cmin, search_range_pixels):
    """
    計算 NCC 曲線：在 mov_image 上滑動搜索
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
    """建立 NCC tracking 視覺化動畫"""

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
    rect_mov = Rectangle((x_min_mm, z_min_mm), x_max_mm - x_min_mm, z_max_mm - z_min_mm,
                          linewidth=2, edgecolor='lime', facecolor='none')
    ax_mov.add_patch(rect_mov)

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


def main():
    """主程式"""

    output_dir = 'results/ncc_visualization'
    os.makedirs(output_dir, exist_ok=True)

    print("初始化模擬環境...")
    sim_config = init_simulation()

    trans_cfg = TransducerConfig()
    psf_cfg = PSFConfig()
    grid_cfg = GridConfig()
    roi_cfg = ROIConfig()
    sim_cfg = SimulationConfig()
    flow_cfg = FlowConfig()
    iw_cfg = InterrogationWindowConfig()
    track_cfg = SpeckleTrackingConfig()

    # 建立網格
    X, Z, x, z = create_spatial_grid(grid_cfg.x_size, grid_cfg.z_size, grid_cfg.dx)
    dx = np.abs(x[1] - x[0])

    # IW 網格
    iw_info, grid_shape = create_iw_grid(X, Z, iw_cfg)
    iw_masks = extract_iw_masks(iw_info)
    num_iw = len(iw_masks)
    print(f"IW 數量: {num_iw}")

    # 選擇中心和邊緣 IW
    vessel_cx = roi_cfg.vessel_cx
    vessel_cz = roi_cfg.vessel_cz
    vessel_R = flow_cfg.vessel_radius

    center_iw_idx = None
    edge_iw_idx = None
    min_dist_center = float('inf')
    best_edge_dist = float('inf')

    for i, iw in enumerate(iw_info):
        cx_iw, cz_iw = iw['cx'], iw['cz']
        r = np.sqrt((cx_iw - vessel_cx)**2 + (cz_iw - vessel_cz)**2)

        if r < min_dist_center and iw['inside_vessel']:
            min_dist_center = r
            center_iw_idx = i

        dist_to_target = abs(r - vessel_R * 0.85)
        if dist_to_target < best_edge_dist and iw['inside_vessel']:
            best_edge_dist = dist_to_target
            edge_iw_idx = i

    print(f"中心 IW: {center_iw_idx} (r={min_dist_center:.2f} mm)")
    print(f"邊緣 IW: {edge_iw_idx}")

    # PSF 參數
    sigma_x, sigma_z = calculate_psf_parameters(
        trans_cfg.fx, trans_cfg.c, psf_cfg.Fnum, psf_cfg.ncycles, psf_cfg.sigma_scale
    )
    sigma_y = 0.3

    element_centers = calculate_element_centers(trans_cfg.element_data, trans_cfg.N_elements)

    # 產生 phantom
    print("\n產生影像序列...")
    x_sc, y_sc, z_sc, amps, v_profile, flow_dir = generate_vessel_phantom(
        sim_cfg.num_scatterers,
        grid_cfg.x_size, flow_cfg.vessel_length_y, grid_cfg.z_size,
        roi_cfg.vessel_cx, roi_cfg.vessel_cz,
        flow_cfg.vessel_radius, flow_cfg.max_velocity,
        flow_direction=flow_cfg.flow_direction,
        vessel_axis=flow_cfg.vessel_axis,
        vessel_cy=0.0
    )

    # 產生參考影像
    ref_rf = generate_ultrasound_image(
        X, Z, x_sc, y_sc, z_sc, amps,
        element_centers, [0], [0],
        trans_cfg.fx, trans_cfg.c, sigma_x, sigma_z, sigma_y, psf_cfg.p_comp
    )
    ref_image = apply_envelope_detection(ref_rf)

    # 產生所有幀
    num_frames = 50
    all_frames = [ref_image.copy()]

    print(f"產生 {num_frames} 幀...")
    for frame_idx in tqdm(range(1, num_frames)):
        x_sc, y_sc, z_sc = update_scatterer_positions(
            x_sc, y_sc, z_sc, v_profile, flow_dir, sim_cfg.dt
        )

        if flow_cfg.enable_scatterer_regeneration:
            x_sc, y_sc, z_sc, amps, _ = check_and_regenerate_scatterers(
                x_sc, y_sc, z_sc, amps,
                grid_cfg.x_size, flow_cfg.vessel_length_y, grid_cfg.z_size,
                roi_cfg.vessel_cx, 0.0, roi_cfg.vessel_cz,
                flow_cfg.vessel_radius, flow_dir,
                vessel_axis=flow_cfg.vessel_axis,
                boundary_margin=flow_cfg.boundary_margin
            )

        mov_rf = generate_ultrasound_image(
            X, Z, x_sc, y_sc, z_sc, amps,
            element_centers, [0], [0],
            trans_cfg.fx, trans_cfg.c, sigma_x, sigma_z, sigma_y, psf_cfg.p_comp
        )
        mov_image = apply_envelope_detection(mov_rf)
        all_frames.append(mov_image.copy())

    search_range_pixels = int(track_cfg.search_range_x / dx)

    # 產生動畫
    print("\n=== 產生中心 IW 動畫 ===")
    create_ncc_tracking_animation(
        all_frames, ref_image, iw_masks[center_iw_idx], center_iw_idx,
        x, z, search_range_pixels,
        os.path.join(output_dir, 'ncc_tracking_center_iw.gif'),
        fps=5, max_frames=num_frames
    )

    print("\n=== 產生邊緣 IW 動畫 ===")
    create_ncc_tracking_animation(
        all_frames, ref_image, iw_masks[edge_iw_idx], edge_iw_idx,
        x, z, search_range_pixels,
        os.path.join(output_dir, 'ncc_tracking_edge_iw.gif'),
        fps=5, max_frames=num_frames
    )

    print(f"\n所有動畫已儲存到 {output_dir}/")


if __name__ == '__main__':
    main()
