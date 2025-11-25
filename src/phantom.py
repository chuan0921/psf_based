import numpy as np

def generate_point_targets(positions, amplitudes=None):
    if amplitudes is None:
        amplitudes = np.ones(len(positions))

    return (positions[:, 0], positions[:, 1], positions[:, 2], amplitudes)

def generate_vessel_phantom(num_scatterers, x_size, y_size, z_size,
                           vessel_cx, vessel_cz, R, Vmax):
    x_scatter = x_size * np.random.rand(num_scatterers) - x_size/2
    z_scatter = z_size * np.random.rand(num_scatterers)
    y_scatter = y_size * np.random.rand(num_scatterers) - y_size/2

    amplitudes = np.random.randn(num_scatterers) * (1 + 0.5*np.random.rand(num_scatterers))

    r = np.sqrt((x_scatter - vessel_cx)**2 + (z_scatter - vessel_cz)**2)
    v_profile = Vmax * (1 - (r / R)**2)
    v_profile[r > R] = 0

    return x_scatter, y_scatter, z_scatter, amplitudes, v_profile

def update_scatterer_positions(y_scatter, v_profile, dt):
    y_move = v_profile * dt * 1e-3
    return y_scatter + y_move
