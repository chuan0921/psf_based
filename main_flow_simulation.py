import numpy as np
import os
import warnings

# 隱藏 matplotlib 的 masked element 警告
warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan')

# 設定使用 GPU 1 (必須在 import cupy 之前設定)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
GPU_ID = 0  # 因為只有 GPU 1 可見，所以對 CuPy 來說是 device 0

import cupy as cp
from tqdm import tqdm
from src.setting import *

cp.cuda.Device(GPU_ID).use()
from src.phantom import generate_vessel_phantom, update_scatterer_positions, check_and_regenerate_scatterers
from src.speckle_image import calculate_psf_parameters, generate_ultrasound_image, apply_envelope_detection
from src.grid import create_spatial_grid, create_roi_masks
from src.correlation import compute_multi_roi_ncc
from src.velocity_estimation import estimate_velocities_batch, estimate_velocity_unified
from src.visualization import plot_velocity_profile, plot_rayleigh_distribution, create_animation, save_bmode_image
from src.statistics import generate_statistics_report, save_results

def main():
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    print("初始化模擬環境...")
    print(f"使用 GPU 1 (NVIDIA H100)")
    free_mem, total_mem = cp.cuda.Device(GPU_ID).mem_info
    print(f"GPU 記憶體: {free_mem/1024**3:.1f} GB free / {total_mem/1024**3:.1f} GB total")
    sim_config = init_simulation()

    trans_cfg = TransducerConfig()
    psf_cfg = PSFConfig()
    grid_cfg = GridConfig()
    roi_cfg = ROIConfig()
    sim_cfg = SimulationConfig()
    flow_cfg = FlowConfig()  # 新增：3D 流動配置
    track_cfg = SpeckleTrackingConfig()  # 新增：Speckle tracking 配置

    print(f"\n流動配置:")
    print(f"  模式: {flow_cfg.velocity_estimation_mode}")
    print(f"  流動方向: {flow_cfg.flow_direction}")
    print(f"  最大速度: {flow_cfg.max_velocity} mm/s")
    print(f"  血管半徑: {flow_cfg.vessel_radius} mm")

    element_centers = calculate_element_centers(trans_cfg.element_data, trans_cfg.N_elements)
    y_distance = calculate_elevational_pitch(element_centers)
    print(f"Elevational pitch: {y_distance:.4f} mm")

    X, Z, x, z = create_spatial_grid(grid_cfg.x_size, grid_cfg.z_size, grid_cfg.dx)
    roi_masks = create_roi_masks(X, Z, roi_cfg.roi_x_positions, roi_cfg.vessel_cz, roi_cfg.roi_size)
    print(f"Grid shape: {X.shape}, ROI count: {len(roi_masks)}")

    sigma_x, sigma_z = calculate_psf_parameters(trans_cfg.fx, trans_cfg.c,
                                                psf_cfg.Fnum, psf_cfg.ncycles, psf_cfg.sigma_scale)
    sigma_y = 0.3
    print(f"PSF parameters: sigma_x={sigma_x:.3f} mm, sigma_z={sigma_z:.3f} mm")

    num_roi = len(roi_cfg.roi_x_positions)
    all_roi_cc = np.zeros((num_roi, sim_cfg.num_iterations, sim_cfg.num_frames))
    animation_frames = []
    saved_ref_image = None
    saved_matched_image = None
    max_cc_frame_idx = None

    for iter_idx in tqdm(range(sim_cfg.num_iterations), desc="Iterations", unit="iter"):
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"Iteration {iter_idx+1}/{sim_cfg.num_iterations}")
        tqdm.write(f"{'='*50}")

        # 使用新的 3D phantom 生成
        x_sc, y_sc, z_sc, amps, v_profile, flow_dir = generate_vessel_phantom(
            sim_cfg.num_scatterers,
            grid_cfg.x_size, flow_cfg.vessel_length_y, grid_cfg.z_size,
            roi_cfg.vessel_cx, roi_cfg.vessel_cz,
            flow_cfg.vessel_radius, flow_cfg.max_velocity,
            flow_direction=flow_cfg.flow_direction,
            vessel_cy=0.0
        )

        print("生成參考影像...")
        ref_rf = generate_ultrasound_image(
            X, Z, x_sc, y_sc, z_sc, amps,
            element_centers, [0], [0],
            trans_cfg.fx, trans_cfg.c, sigma_x, sigma_z, sigma_y, psf_cfg.p_comp
        )
        ref_image = apply_envelope_detection(ref_rf)

        if iter_idx == 0:
            env_ref = np.abs(np.fft.ifft(ref_rf, axis=0))
            plot_rayleigh_distribution(env_ref / env_ref.max())
            saved_ref_image = ref_image.copy()

        frame_cc_values = []

        for frame_idx in tqdm(range(sim_cfg.num_frames), desc="  Frames", unit="frame", leave=False):
            # 使用新的 3D 位置更新
            x_sc, y_sc, z_sc = update_scatterer_positions(
                x_sc, y_sc, z_sc, v_profile, flow_dir, sim_cfg.dt
            )

            # 散射體重生（如果啟用）
            if flow_cfg.enable_scatterer_regeneration:
                x_sc, y_sc, z_sc, amps, num_regen = check_and_regenerate_scatterers(
                    x_sc, y_sc, z_sc, amps,
                    grid_cfg.x_size, flow_cfg.vessel_length_y, grid_cfg.z_size,
                    roi_cfg.vessel_cx, 0.0, roi_cfg.vessel_cz,
                    flow_cfg.vessel_radius, flow_dir,
                    boundary_margin=flow_cfg.boundary_margin
                )
                if num_regen > 0 and frame_idx % 10 == 0:
                    tqdm.write(f"    Frame {frame_idx}: 重生 {num_regen} 個散射體")

            moving_rf = generate_ultrasound_image(
                X, Z, x_sc, y_sc, z_sc, amps,
                element_centers, [1], [1],
                trans_cfg.fx, trans_cfg.c, sigma_x, sigma_z, sigma_y, psf_cfg.p_comp
            )
            moving_image = apply_envelope_detection(moving_rf)

            roi_cc = compute_multi_roi_ncc(ref_image, moving_image, roi_masks)
            all_roi_cc[:, iter_idx, frame_idx] = roi_cc

            if iter_idx == 0:
                animation_frames.append(moving_image.copy())
                avg_cc = np.nanmean(roi_cc)
                frame_cc_values.append(avg_cc)

        if iter_idx == 0 and len(frame_cc_values) > 0:
            max_cc_frame_idx = np.argmax(frame_cc_values)
            saved_matched_image = animation_frames[max_cc_frame_idx].copy()
            tqdm.write(f"\nMax CC at frame {max_cc_frame_idx}: {frame_cc_values[max_cc_frame_idx]:.3f}")

    print("\n速度估算...")

    # 計算理論速度（基於 3D 流動）
    # 計算每個 ROI 位置到血管軸線的徑向距離
    theoretical_vel = []
    for x_pos in roi_cfg.roi_x_positions:
        # ROI 位置（假設在血管中心 Z 位置）
        dx = x_pos - roi_cfg.vessel_cx
        dy = 0 - 0  # vessel_cy = 0
        dz = roi_cfg.vessel_cz - roi_cfg.vessel_cz  # 0
        pos_vec = np.array([dx, dy, dz])

        # 投影到流動方向
        proj = np.dot(pos_vec, flow_dir) * flow_dir
        perp = pos_vec - proj
        r = np.linalg.norm(perp)

        # Poiseuille 速度剖面
        theo_speed = flow_cfg.max_velocity * (1 - (r / flow_cfg.vessel_radius)**2) \
                     if r <= flow_cfg.vessel_radius else 0
        theoretical_vel.append(theo_speed)

    theoretical_vel = np.array(theoretical_vel)

    # 速度估算（保持現有的 Y 方向方法）
    measured_velocities = estimate_velocities_batch(all_roi_cc, sim_cfg.dt, y_distance)

    # 使用 nanmean/nanstd 忽略無效的速度估算
    avg_vel = np.nanmean(measured_velocities, axis=1)
    std_vel = np.nanstd(measured_velocities, axis=1)

    report = generate_statistics_report(roi_cfg.roi_x_positions, theoretical_vel,
                                       avg_vel, std_vel,
                                       flow_cfg.vessel_radius, flow_cfg.max_velocity,
                                       sim_cfg.num_iterations)
    print("\n" + report)

    plot_velocity_profile(roi_cfg.roi_x_positions, theoretical_vel, avg_vel, std_vel,
                         flow_cfg.vessel_radius, flow_cfg.max_velocity)

    if saved_ref_image is not None:
        print("\n儲存影像...")
        save_bmode_image(saved_ref_image, x, z,
                        os.path.join(output_dir, 'reference_image.png'),
                        'Reference Image (Frame 0)')

    if saved_matched_image is not None:
        save_bmode_image(saved_matched_image, x, z,
                        os.path.join(output_dir, f'matched_image_frame{max_cc_frame_idx}.png'),
                        f'Best Matched Image (Frame {max_cc_frame_idx}, Max CC)')

    if animation_frames:
        print("\n建立動畫...")
        create_animation(animation_frames, x, z,
                        {'cx': roi_cfg.vessel_cx, 'cz': roi_cfg.vessel_cz,
                         'R': flow_cfg.vessel_radius},
                        {}, os.path.join(output_dir, 'flow_simulation.gif'))

    save_results(os.path.join(output_dir, 'results.npz'),
                 roi_x_positions=roi_cfg.roi_x_positions,
                 theoretical_velocities=theoretical_vel,
                 measured_velocities=measured_velocities,
                 avg_velocities=avg_vel,
                 std_velocities=std_vel,
                 all_roi_cc=all_roi_cc)

    cleanup_simulation()
    print("\n模擬完成!")

if __name__ == '__main__':
    main()
