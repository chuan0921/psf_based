import numpy as np
import os
import warnings
from datetime import datetime

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
from src.grid import create_spatial_grid, create_roi_masks, create_iw_grid, extract_iw_masks, validate_iw_grid, print_iw_validation_report
from src.correlation import compute_multi_roi_ncc, create_iw_results
from src.velocity_estimation import estimate_velocities_batch, estimate_velocity_unified
from src.visualization import (plot_velocity_profile, plot_rayleigh_distribution,
                               create_animation, save_bmode_image,
                               plot_velocity_vector_field, plot_velocity_profile_comparison,
                               plot_iw_boundaries, plot_ncc_curves, plot_peak_frame_distribution)
from src.statistics import (generate_statistics_report, save_results,
                            compute_iw_error_metrics, generate_iw_error_report)

def main():
    # 使用時間戳建立獨立的輸出資料夾，避免覆蓋先前結果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"輸出資料夾: {output_dir}")

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
    iw_cfg = InterrogationWindowConfig()  # 新增：IW 網格配置

    print(f"\n流動配置:")
    print(f"  模式: {flow_cfg.velocity_estimation_mode}")
    print(f"  血管軸線: {flow_cfg.vessel_axis}")
    print(f"  流動方向: {flow_cfg.flow_direction}")
    print(f"  最大速度: {flow_cfg.max_velocity} mm/s")
    print(f"  血管半徑: {flow_cfg.vessel_radius} mm")

    element_centers = calculate_element_centers(trans_cfg.element_data, trans_cfg.N_elements)
    y_distance = calculate_elevational_pitch(element_centers)
    print(f"Elevational pitch: {y_distance:.4f} mm")

    X, Z, x, z = create_spatial_grid(grid_cfg.x_size, grid_cfg.z_size, grid_cfg.dx)

    # 使用 IW 網格系統
    iw_info, grid_shape = create_iw_grid(X, Z, iw_cfg)
    print(f"Grid shape: {X.shape}, IW grid: {grid_shape}, Total IWs: {len(iw_info)}")

    # 驗證 IW 網格
    validation = validate_iw_grid(iw_info, iw_cfg)
    print_iw_validation_report(validation)

    # 提取 masks 供相關性計算使用
    iw_masks = extract_iw_masks(iw_info)

    sigma_x, sigma_z = calculate_psf_parameters(trans_cfg.fx, trans_cfg.c,
                                                psf_cfg.Fnum, psf_cfg.ncycles, psf_cfg.sigma_scale)
    sigma_y = 0.3
    print(f"PSF parameters: sigma_x={sigma_x:.3f} mm, sigma_z={sigma_z:.3f} mm")

    # === Speckle / IW 診斷資訊 ===
    wavelength_mm = (trans_cfg.c / trans_cfg.fx) * 1e3
    speckle_lateral = psf_cfg.Fnum * wavelength_mm  # FWHM
    speckle_axial = (psf_cfg.ncycles * wavelength_mm) / 2

    speckles_x = iw_cfg.iw_size_x / speckle_lateral
    speckles_z = iw_cfg.iw_size_z / speckle_axial
    total_speckles = speckles_x * speckles_z

    print(f"\n{'='*60}")
    print("SPECKLE / IW DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"Speckle size (FWHM):")
    print(f"  Lateral:  {speckle_lateral:.3f} mm")
    print(f"  Axial:    {speckle_axial:.3f} mm")
    print(f"\nIW size: {iw_cfg.iw_size_x:.2f} × {iw_cfg.iw_size_z:.2f} mm")
    print(f"Expected speckles per IW:")
    print(f"  Lateral:  {speckles_x:.1f}")
    print(f"  Axial:    {speckles_z:.1f}")
    print(f"  Total:    {total_speckles:.1f}")
    print(f"{'='*60}")

    num_iw = len(iw_info)
    num_frames = sim_cfg.num_frames

    # === 新增：儲存逐幀 NCC 和位移的陣列 ===
    # Shape: (num_iw, num_frames)
    all_ncc_x = np.zeros((num_iw, num_frames))      # X 方向 NCC
    all_ncc_z = np.zeros((num_iw, num_frames))      # Z 方向 NCC
    all_ncc_mean = np.zeros((num_iw, num_frames))   # 平均 NCC (用於找峰值)
    all_dx = np.zeros((num_iw, num_frames))         # X 方向位移 (mm)
    all_dz = np.zeros((num_iw, num_frames))         # Z 方向位移 (mm)

    animation_frames = []
    saved_ref_image = None
    saved_matched_image = None
    max_cc_frame_idx = None

    for iter_idx in tqdm(range(sim_cfg.num_iterations), desc="Iterations", unit="iter"):
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"Iteration {iter_idx+1}/{sim_cfg.num_iterations}")
        tqdm.write(f"{'='*50}")

        # 使用新的 3D phantom 生成（分離血管幾何與流動方向）
        x_sc, y_sc, z_sc, amps, v_profile, flow_dir = generate_vessel_phantom(
            sim_cfg.num_scatterers,
            grid_cfg.x_size, flow_cfg.vessel_length_y, grid_cfg.z_size,
            roi_cfg.vessel_cx, roi_cfg.vessel_cz,
            flow_cfg.vessel_radius, flow_cfg.max_velocity,
            flow_direction=flow_cfg.flow_direction,
            vessel_axis=flow_cfg.vessel_axis,
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

        for frame_idx in tqdm(range(num_frames), desc="  Frames", unit="frame", leave=False):
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
                    vessel_axis=flow_cfg.vessel_axis,
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

            # 使用 IW 網格進行 X-Z 位移估算（每幀都跟參考幀比較）
            result = estimate_velocity_unified(
                ref_image, moving_image, sim_cfg.dt,
                mode='xz',
                roi_masks=iw_masks,
                search_range_x=iw_cfg.search_range_x,
                search_range_z=iw_cfg.search_range_z,
                search_step_x=iw_cfg.search_step_x,
                search_step_z=iw_cfg.search_step_z,
                x_grid=x, z_grid=z,
                ncc_threshold=0.0  # 暫時不過濾，儲存所有 NCC 值
            )

            # === 儲存逐幀結果 ===
            if iter_idx == 0:
                # 儲存 NCC 值
                all_ncc_x[:, frame_idx] = result['ncc'][:, 0]
                all_ncc_z[:, frame_idx] = result['ncc'][:, 1]
                all_ncc_mean[:, frame_idx] = (result['ncc'][:, 0] + result['ncc'][:, 1]) / 2

                # 儲存位移值
                all_dx[:, frame_idx] = result['displacements'][:, 0]
                all_dz[:, frame_idx] = result['displacements'][:, 1]

                animation_frames.append(moving_image.copy())

        # === 找峰值幀並計算速度 ===
        if iter_idx == 0:
            tqdm.write(f"\n找尋每個 IW 的 NCC 峰值幀...")

            # 對每個 IW 找到 NCC 最大的幀
            peak_frames = np.argmax(all_ncc_mean, axis=1)  # Shape: (num_iw,)

            # 提取峰值幀的數據
            final_dx = np.array([all_dx[i, peak_frames[i]] for i in range(num_iw)])
            final_dz = np.array([all_dz[i, peak_frames[i]] for i in range(num_iw)])
            final_ncc_x = np.array([all_ncc_x[i, peak_frames[i]] for i in range(num_iw)])
            final_ncc_z = np.array([all_ncc_z[i, peak_frames[i]] for i in range(num_iw)])
            final_ncc_mean = np.array([all_ncc_mean[i, peak_frames[i]] for i in range(num_iw)])

            # 計算到達峰值的時間 (幀數從1開始計算，因為 frame 0 表示 dt 已經過了)
            time_to_peak = (peak_frames + 1) * sim_cfg.dt  # +1 因為 frame 0 已經經過 dt 時間

            # 計算速度 = 位移 / 時間
            final_vx = final_dx / time_to_peak
            final_vz = final_dz / time_to_peak
            final_v_mag = np.sqrt(final_vx**2 + final_vz**2)

            # 有效性判斷
            ncc_threshold = iw_cfg.ncc_threshold
            valid_mask = (final_ncc_mean >= ncc_threshold) & (peak_frames > 0)

            # 找全域最大 NCC 幀
            global_avg_ncc = np.mean(all_ncc_mean, axis=0)
            max_cc_frame_idx = np.argmax(global_avg_ncc)
            saved_matched_image = animation_frames[max_cc_frame_idx].copy()

            tqdm.write(f"峰值幀統計:")
            tqdm.write(f"  全域最大 NCC 幀: {max_cc_frame_idx} (NCC={global_avg_ncc[max_cc_frame_idx]:.3f})")
            tqdm.write(f"  峰值幀範圍: [{np.min(peak_frames)}, {np.max(peak_frames)}]")
            tqdm.write(f"  峰值幀平均: {np.mean(peak_frames):.1f}")

    print("\n速度估算與統計...")

    # 分析 IW 結果（使用峰值幀法計算的速度）
    if 'final_vx' in dir():
        # 計算統計量
        valid_count = np.sum(valid_mask)
        total_count = num_iw
        valid_ratio = valid_count / total_count if total_count > 0 else 0

        print(f"\nIW 品質統計 (峰值幀法):")
        print(f"  總 IW 數: {total_count}")
        print(f"  有效 IW 數: {valid_count}")
        print(f"  有效率: {valid_ratio:.1%}")

        # 取得所有有效 IW 的速度大小
        valid_velocities = final_v_mag[valid_mask & ~np.isnan(final_v_mag)]
        if len(valid_velocities) > 0:
            print(f"\n速度統計 (有效 IWs):")
            print(f"  平均速度: {np.mean(valid_velocities):.2f} mm/s")
            print(f"  速度範圍: [{np.min(valid_velocities):.2f}, {np.max(valid_velocities):.2f}] mm/s")
            print(f"  標準差: {np.std(valid_velocities):.2f} mm/s")

        # 峰值幀統計
        valid_peak_frames = peak_frames[valid_mask]
        if len(valid_peak_frames) > 0:
            print(f"\n峰值幀統計 (有效 IWs):")
            print(f"  峰值幀平均: {np.mean(valid_peak_frames):.1f}")
            print(f"  峰值幀範圍: [{np.min(valid_peak_frames)}, {np.max(valid_peak_frames)}]")

        # 儲存詳細 IW 結果
        print(f"\n儲存 IW 結果到 {output_dir}/iw_results.npz...")
        iw_positions = np.array([[iw['cx'], iw['cz']] for iw in iw_info])
        iw_overlaps = np.array([iw['vessel_overlap'] for iw in iw_info])

        np.savez(
            os.path.join(output_dir, 'iw_results.npz'),
            # 位置與幾何
            iw_positions=iw_positions,
            iw_overlap=iw_overlaps,
            # 峰值幀數據
            peak_frames=peak_frames,
            time_to_peak=time_to_peak,
            # 最終速度
            iw_velocities=np.stack([final_vx, final_vz], axis=1),
            iw_velocity_mag=final_v_mag,
            # 最終 NCC
            iw_ncc=np.stack([final_ncc_x, final_ncc_z], axis=1),
            iw_ncc_mean=final_ncc_mean,
            # 有效性
            iw_valid=valid_mask,
            # 完整逐幀數據
            all_ncc_mean=all_ncc_mean,
            all_dx=all_dx,
            all_dz=all_dz
        )

        # =========================================================
        # 速度場視覺化與誤差分析
        # =========================================================
        print("\n" + "="*60)
        print("VELOCITY FIELD VISUALIZATION AND ERROR ANALYSIS")
        print("="*60)

        # 準備 IW 資料
        iw_pos = iw_positions
        iw_vel = np.stack([final_vx, final_vz], axis=1)
        iw_valid_arr = valid_mask

        vessel_params = {
            'cx': roi_cfg.vessel_cx,
            'cz': roi_cfg.vessel_cz,
            'R': flow_cfg.vessel_radius,
            'Vmax': flow_cfg.max_velocity
        }

        # 1. 向量場視覺化
        if saved_ref_image is not None:
            print("\n1. 生成速度向量場圖...")
            plot_velocity_vector_field(
                saved_ref_image, x, z,
                iw_pos, iw_vel, iw_valid_arr,
                vessel_params,
                filename=os.path.join(output_dir, 'velocity_vector_field.png'),
                scale_factor=0.02
            )

            # 1.5 IW 邊界視覺化
            print("1.5 生成 IW 邊界圖...")
            speckle_info = {
                'lateral': speckle_lateral,
                'axial': speckle_axial
            }
            plot_iw_boundaries(
                saved_ref_image, x, z, iw_info, iw_cfg,
                vessel_params, speckle_info,
                filename=os.path.join(output_dir, 'iw_boundaries.png'),
                max_iw_to_show=None  # 顯示所有 IW
            )

            # 1.6 NCC 曲線圖
            print("1.6 生成 NCC 曲線圖...")
            plot_ncc_curves(
                all_ncc_mean, peak_frames, iw_positions, valid_mask,
                vessel_params['cx'], vessel_params['cz'],
                vessel_params['R'], sim_cfg.dt,
                filename=os.path.join(output_dir, 'ncc_curves.png'),
                num_curves=10
            )

            # 1.7 峰值幀分佈圖
            print("1.7 生成峰值幀分佈圖...")
            plot_peak_frame_distribution(
                peak_frames, iw_positions, valid_mask,
                vessel_params['cx'], vessel_params['cz'],
                vessel_params['R'], sim_cfg.dt,
                bmode_image=saved_ref_image, x=x, z=z,
                filename=os.path.join(output_dir, 'peak_frame_distribution.png')
            )

        # 2. 速度剖面比較圖
        print("2. 生成速度剖面比較圖...")
        plot_velocity_profile_comparison(
            iw_pos, iw_vel, iw_valid_arr,
            vessel_params['cx'], vessel_params['cz'],
            vessel_params['R'], vessel_params['Vmax'],
            flow_cfg.flow_direction,
            filename=os.path.join(output_dir, 'velocity_profile_comparison.png')
        )

        # 3. 誤差分析
        print("3. 計算誤差指標...")
        error_metrics = compute_iw_error_metrics(
            iw_pos, iw_vel, iw_valid_arr,
            vessel_params['cx'], vessel_params['cz'],
            vessel_params['R'], vessel_params['Vmax'],
            flow_cfg.flow_direction
        )

        # 4. 生成並印出誤差報告
        error_report = generate_iw_error_report(
            error_metrics, vessel_params, flow_cfg.flow_direction
        )
        print(error_report)

        # 5. 儲存誤差報告
        report_path = os.path.join(output_dir, 'error_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(error_report)
        print(f"\n誤差報告已儲存至: {report_path}")

        # 6. 儲存誤差指標到 npz
        np.savez(
            os.path.join(output_dir, 'error_metrics.npz'),
            rmse_mag=error_metrics['magnitude']['rmse'],
            nrmse_mag=error_metrics['magnitude']['nrmse'],
            mae_mag=error_metrics['magnitude']['mae'],
            r_squared=error_metrics['magnitude']['r_squared'],
            rmse_vx=error_metrics['vx']['rmse'],
            rmse_vz=error_metrics['vz']['rmse'],
            num_valid=error_metrics['num_valid'],
            relative_error_mean=error_metrics['relative_error_mean'],
            relative_error_std=error_metrics['relative_error_std']
        )
        print(f"誤差指標已儲存至: {output_dir}/error_metrics.npz")

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

    cleanup_simulation()
    print("\n模擬完成!")

if __name__ == '__main__':
    main()
