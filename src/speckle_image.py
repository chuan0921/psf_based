import numpy as np
import os

# 確保使用 GPU 1 (如果環境變數還沒設定)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cupy as cp
from scipy.signal import hilbert

# 使用可見的第一個 GPU (即實際的 GPU 1)
GPU_ID = 0
cp.cuda.Device(GPU_ID).use()

def calculate_psf_parameters(fx, c, Fnum, ncycles, sigma_scale):
    wavelength = c / fx
    lambda_mm = wavelength * 1e3

    FWHM_lat = Fnum * lambda_mm
    sigma_x = FWHM_lat / (2 * np.sqrt(2 * np.log(2)))

    pulse_len = ncycles * lambda_mm
    FWHM_ax = pulse_len / 2
    sigma_z = FWHM_ax / (2 * np.sqrt(2 * np.log(2)))

    return sigma_x * sigma_scale, sigma_z * sigma_scale

def generate_ultrasound_image(X, Z, x_scatter, y_scatter, z_scatter, amplitudes,
                               element_centers, tx_idx, rx_idx,
                               fx, c, sigma_x, sigma_z, sigma_y, p_comp):
    # 將資料轉移到 GPU
    X_gpu = cp.asarray(X)
    Z_gpu = cp.asarray(Z)
    x_sc_gpu = cp.asarray(x_scatter)
    y_sc_gpu = cp.asarray(y_scatter)
    z_sc_gpu = cp.asarray(z_scatter)
    amps_gpu = cp.asarray(amplitudes)
    
    composite_image = cp.zeros_like(X_gpu)

    for tx in tx_idx:
        for rx in rx_idx:
            tx_center = cp.asarray(element_centers[tx])
            rx_center = cp.asarray(element_centers[rx])
            sub_image = _compute_psf_contribution_gpu(
                X_gpu, Z_gpu, x_sc_gpu, y_sc_gpu, z_sc_gpu, amps_gpu,
                tx_center, rx_center,
                fx, c, sigma_x, sigma_z, sigma_y, p_comp
            )
            composite_image += sub_image

    # 將結果轉回 CPU
    return cp.asnumpy(composite_image)

def _compute_psf_contribution_gpu(X, Z, x_sc, y_sc, z_sc, amps,
                                   tx_center, rx_center,
                                   fx, c, sigma_x, sigma_z, sigma_y, p_comp):
    """GPU 加速版本的 PSF 計算"""
    num_scatterers = len(x_sc)
    batch_size = 2000  # GPU 可以處理更大的 batch
    result = cp.zeros_like(X)

    for i in range(0, num_scatterers, batch_size):
        end_idx = min(i + batch_size, num_scatterers)

        x_batch = x_sc[i:end_idx]
        y_batch = y_sc[i:end_idx]
        z_batch = z_sc[i:end_idx]
        amps_batch = amps[i:end_idx]

        tx_dist = cp.sqrt((x_batch - tx_center[0])**2 +
                          (y_batch - tx_center[1])**2 +
                          (z_batch - tx_center[2])**2)
        rx_dist = cp.sqrt((x_batch - rx_center[0])**2 +
                          (y_batch - rx_center[1])**2 +
                          (z_batch - rx_center[2])**2)
        total_delay = (tx_dist + rx_dist) / c

        r_tx_z = z_batch - tx_center[2]
        r_rx_z = z_batch - rx_center[2]
        directivity_tx = r_tx_z / tx_dist
        directivity_rx = r_rx_z / rx_dist
        directivity = (directivity_tx * directivity_rx) ** (1 - p_comp)

        elev_weight = cp.exp(- y_batch**2 / (2 * sigma_y**2))

        phase = 2 * cp.pi * fx * total_delay

        dx = X[cp.newaxis, :, :] - x_batch[:, cp.newaxis, cp.newaxis]
        dz = Z[cp.newaxis, :, :] - z_batch[:, cp.newaxis, cp.newaxis]

        psf = (amps_batch[:, cp.newaxis, cp.newaxis] *
               directivity[:, cp.newaxis, cp.newaxis] *
               elev_weight[:, cp.newaxis, cp.newaxis] *
               cp.exp(-(dx**2 / (2*sigma_x**2) + dz**2 / (2*sigma_z**2))) *
               cp.cos(2*cp.pi*fx/c * dz + phase[:, cp.newaxis, cp.newaxis]))

        result += cp.sum(psf, axis=0)

    return result

def apply_envelope_detection(rf_image, dynamic_range=30):
    env = np.abs(hilbert(rf_image, axis=0))
    env_norm = env / env.max()
    return np.maximum(20 * np.log10(env_norm), -dynamic_range)
