import numpy as np
import matlab.engine
import os
from src.setting import *
from src.phantom import generate_vessel_phantom, update_scatterer_positions
from src.speckle_image import calculate_psf_parameters, generate_ultrasound_image, apply_envelope_detection
from src.grid import create_spatial_grid, create_roi_masks
from src.correlation import compute_multi_roi_ncc
from src.velocity_estimation import estimate_velocities_batch
from src.visualization import plot_velocity_profile, plot_rayleigh_distribution, create_animation, save_bmode_image
from src.statistics import generate_statistics_report, save_results

def main():
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    print("初始化 MATLAB Engine 與 Field II...")
    eng = matlab.engine.start_matlab()
    init_field_ii(eng)

    trans_cfg = TransducerConfig()
    psf_cfg = PSFConfig()
    grid_cfg = GridConfig()
    roi_cfg = ROIConfig()
    sim_cfg = SimulationConfig()

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

    for iter_idx in range(sim_cfg.num_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iter_idx+1}/{sim_cfg.num_iterations}")
        print(f"{'='*50}")

        x_sc, y_sc, z_sc, amps, v_profile = generate_vessel_phantom(
            sim_cfg.num_scatterers, grid_cfg.x_size, 5, grid_cfg.z_size,
            roi_cfg.vessel_cx, roi_cfg.vessel_cz, 5, 10
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

        for frame_idx in range(sim_cfg.num_frames):
            y_sc = update_scatterer_positions(y_sc, v_profile, sim_cfg.dt)

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

            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{sim_cfg.num_frames}: avg CC = {np.nanmean(roi_cc):.3f}")

        if iter_idx == 0 and len(frame_cc_values) > 0:
            max_cc_frame_idx = np.argmax(frame_cc_values)
            saved_matched_image = animation_frames[max_cc_frame_idx].copy()
            print(f"\nMax CC at frame {max_cc_frame_idx}: {frame_cc_values[max_cc_frame_idx]:.3f}")

    print("\n速度估算...")
    measured_velocities = estimate_velocities_batch(all_roi_cc, sim_cfg.dt, y_distance)

    avg_vel = np.mean(measured_velocities, axis=1)
    std_vel = np.std(measured_velocities, axis=1)

    r_pos = np.abs(roi_cfg.roi_x_positions)
    theoretical_vel = np.where(r_pos <= 5, 10 * (1 - (r_pos/5)**2), 0)

    report = generate_statistics_report(roi_cfg.roi_x_positions, theoretical_vel,
                                       avg_vel, std_vel, 5, 10, sim_cfg.num_iterations)
    print("\n" + report)

    plot_velocity_profile(roi_cfg.roi_x_positions, theoretical_vel, avg_vel, std_vel, 5, 10)

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
                        {'cx': roi_cfg.vessel_cx, 'cz': roi_cfg.vessel_cz, 'R': 5},
                        {}, os.path.join(output_dir, 'flow_simulation.gif'))

    save_results(os.path.join(output_dir, 'results.npz'),
                 roi_x_positions=roi_cfg.roi_x_positions,
                 theoretical_velocities=theoretical_vel,
                 measured_velocities=measured_velocities,
                 avg_velocities=avg_vel,
                 std_velocities=std_vel,
                 all_roi_cc=all_roi_cc)

    cleanup_field_ii(eng)
    eng.quit()
    print("\n模擬完成!")

if __name__ == '__main__':
    main()
