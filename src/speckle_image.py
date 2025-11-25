import numpy as np
from scipy.signal import hilbert

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
    composite_image = np.zeros_like(X)

    for tx in tx_idx:
        for rx in rx_idx:
            sub_image = _compute_psf_contribution(
                X, Z, x_scatter, y_scatter, z_scatter, amplitudes,
                element_centers[tx], element_centers[rx],
                fx, c, sigma_x, sigma_z, sigma_y, p_comp
            )
            composite_image += sub_image

    return composite_image

def _compute_psf_contribution(X, Z, x_sc, y_sc, z_sc, amps,
                              tx_center, rx_center,
                              fx, c, sigma_x, sigma_z, sigma_y, p_comp):
    num_scatterers = len(x_sc)
    batch_size = 500
    result = np.zeros_like(X)

    for i in range(0, num_scatterers, batch_size):
        end_idx = min(i + batch_size, num_scatterers)

        x_batch = x_sc[i:end_idx]
        y_batch = y_sc[i:end_idx]
        z_batch = z_sc[i:end_idx]
        amps_batch = amps[i:end_idx]

        tx_dist = np.sqrt((x_batch - tx_center[0])**2 +
                          (y_batch - tx_center[1])**2 +
                          (z_batch - tx_center[2])**2)
        rx_dist = np.sqrt((x_batch - rx_center[0])**2 +
                          (y_batch - rx_center[1])**2 +
                          (z_batch - rx_center[2])**2)
        total_delay = (tx_dist + rx_dist) / c

        r_tx_z = z_batch - tx_center[2]
        r_rx_z = z_batch - rx_center[2]
        directivity_tx = r_tx_z / tx_dist
        directivity_rx = r_rx_z / rx_dist
        directivity = (directivity_tx * directivity_rx) ** (1 - p_comp)

        elev_weight = np.exp(- y_batch**2 / (2 * sigma_y**2))

        phase = 2 * np.pi * fx * total_delay

        dx = X[np.newaxis, :, :] - x_batch[:, np.newaxis, np.newaxis]
        dz = Z[np.newaxis, :, :] - z_batch[:, np.newaxis, np.newaxis]

        psf = (amps_batch[:, np.newaxis, np.newaxis] *
               directivity[:, np.newaxis, np.newaxis] *
               elev_weight[:, np.newaxis, np.newaxis] *
               np.exp(-(dx**2 / (2*sigma_x**2) + dz**2 / (2*sigma_z**2))) *
               np.cos(2*np.pi*fx/c * dz + phase[:, np.newaxis, np.newaxis]))

        result += np.sum(psf, axis=0)

    return result

def apply_envelope_detection(rf_image, dynamic_range=30):
    env = np.abs(hilbert(rf_image, axis=0))
    env_norm = env / env.max()
    return np.maximum(20 * np.log10(env_norm), -dynamic_range)
