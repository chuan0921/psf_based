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
    num_frames = 100
    num_iterations = 1
    dt = 0.003  # 3 ms (333 Hz frame rate) - for max displacement constraint

class FlowConfig:
    """3D 流動配置"""
    # 血管軸線方向 (控制血管幾何形狀，將自動正規化)
    vessel_axis = np.array([0.0, 1.0, 0.0])  # Y 方向 → XZ 影像中顯示圓形切面

    # 流動方向向量 (控制散射體移動方向，將自動正規化)
    flow_direction = np.array([1.0, 0.0, 1.0])  # XZ 方向 (45度) → 測試 XZ speckle tracking

    # 速度估算模式
    velocity_estimation_mode = 'xz'  # 'xz', 'y_only', '3d'

    # 血管幾何參數
    vessel_radius = 5.0  # mm
    vessel_length_y = 20.0  # mm, Y 方向血管長度
    max_velocity = 120.0  # mm/s, 中心最大速度 (increased for realistic flow)

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

class InterrogationWindowConfig:
    """Grid-based Interrogation Window configuration for speckle tracking"""

    # IW dimensions (mm) - designed to contain ~10 speckles
    iw_size_x = 1.5  # Lateral (mm)
    iw_size_z = 1.5  # Axial (mm)

    # Overlap and step size
    overlap_ratio = 0.5  # 50% overlap
    step_x = iw_size_x * (1 - overlap_ratio)  # 0.75 mm
    step_z = iw_size_z * (1 - overlap_ratio)  # 0.75 mm

    # Coverage area (vessel-based)
    vessel_cx = 0.0   # mm (from ROIConfig)
    vessel_cz = 6.5   # mm
    vessel_radius = 5.0  # mm
    coverage_margin = 0.5  # mm, extra margin beyond vessel radius

    # Quality control
    ncc_threshold = 0.6  # Below this = invalid vector
    min_pixels_per_iw = 50  # Minimum pixels for valid IW

    # Search parameters for correlation
    # Max displacement = max_velocity * dt = 120 mm/s * 0.003 s = 0.36 mm
    # Use 2x safety margin
    search_range_x = 0.8  # mm (>2x max displacement)
    search_range_z = 0.8  # mm
    search_step_x = 0.05  # mm (subpixel with dx=0.02 mm)
    search_step_z = 0.05  # mm

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
