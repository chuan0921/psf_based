import numpy as np

class TransducerConfig:
    fx = 5e6
    c = 1540
    N_elements = 2
    gap = 0.01e-3

    element_data = np.array([
        [1, -0.3e-3 - gap/2, (6e-3+gap)/2 - (2e-3) - gap, 0,
         -0.3e-3 - gap/2, (-6e-3-gap)/2, 0,
         -gap/2, (-6e-3-gap)/2, 0,
         -gap/2, (-6e-3-gap)/2 + (2e-3), 0,
         1, 0.3e-3, 6e-3+gap, 0, 0, 0],
        [2, -0.3e-3 - gap/2, (6e-3+gap)/2, 0,
         -gap/2, (6e-3+gap)/2, 0,
         -gap/2, (-6e-3-gap)/2 + 2e-3 + gap, 0,
         -0.3e-3 - gap/2, (6e-3+gap)/2 - (2e-3), 0,
         1, 0.3e-3, 6e-3+gap, 0, 0, 0]
    ])

class PSFConfig:
    Fnum = 2
    ncycles = 2
    sigma_scale = 0.9
    p_comp = 1

class GridConfig:
    x_size = 10
    z_size = 13
    dx = 0.02

class ROIConfig:
    vessel_cx = 0
    vessel_cz = 6.5
    roi_size = 2.5
    roi_x_positions = np.arange(-5, 5.25, 0.25)

class SimulationConfig:
    num_scatterers = 8000
    num_frames = 1
    num_iterations = 1
    dt = 1/100

def init_simulation():
    """
    初始化超音波模擬環境（純 Python 實現）
    不再需要 MATLAB 或 Field II
    """
    print("Initializing Python-based ultrasound simulation...")
    # 設定基本參數
    config = {
        'fs': 100e6,  # 採樣頻率 100 MHz
        'initialized': True
    }
    return config

def cleanup_simulation():
    """
    清理模擬環境（純 Python 實現）
    """
    print("Cleaning up simulation...")
    pass

def calculate_element_centers(element_data, N_elements):
    element_centers = np.zeros((N_elements, 3))
    for i in range(N_elements):
        x_coords = element_data[i, [1, 4, 7, 10]]
        y_coords = element_data[i, [2, 5, 8, 11]]
        z_coords = element_data[i, [3, 6, 9, 12]]
        element_centers[i] = [x_coords.mean(), y_coords.mean(), z_coords.mean()]
    return element_centers

def calculate_elevational_pitch(element_centers):
    odd_y = element_centers[0::2, 1].mean()
    even_y = element_centers[1::2, 1].mean()
    return abs(odd_y - even_y) * 1e3
