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
    num_frames = 10
    num_iterations = 1
    dt = 1/100

class FlowConfig:
    """3D 流動配置"""
    # 流動方向向量 (將自動正規化) [vx, vy, vz]
    flow_direction = np.array([0.0, 1.0, 0.0])  # 預設 Y 方向

    # 速度估算模式
    velocity_estimation_mode = 'xz'  # 'xz', 'y_only', '3d'

    # 血管幾何參數
    vessel_radius = 5.0  # mm
    vessel_length_y = 20.0  # mm, Y 方向血管長度
    max_velocity = 10.0  # mm/s, 中心最大速度

    # 散射體重生參數
    enable_scatterer_regeneration = True
    boundary_margin = 0.5  # mm

class SpeckleTrackingConfig:
    """Speckle tracking 搜尋參數"""
    # 搜尋範圍 (mm)
    search_range_x = 2.0  # 橫向
    search_range_z = 4.0  # 軸向
    search_range_y = 1.5  # 側向

    # 搜尋步長 (mm)
    search_step_x = 0.1
    search_step_z = 0.1

    # ROI 大小 (mm) - 用於 speckle tracking
    roi_size_x = 2.0
    roi_size_z = 2.0

    # NCC 閾值
    ncc_threshold = 0.3

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
