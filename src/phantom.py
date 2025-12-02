import numpy as np

def generate_point_targets(positions, amplitudes=None):
    if amplitudes is None:
        amplitudes = np.ones(len(positions))

    return (positions[:, 0], positions[:, 1], positions[:, 2], amplitudes)

def generate_vessel_phantom(num_scatterers, x_size, y_size, z_size,
                           vessel_cx, vessel_cz, R, Vmax,
                           flow_direction=None, vessel_axis=None, vessel_cy=0.0):
    """
    生成 3D 血管 phantom，支援分離的血管幾何與流動方向

    Parameters:
    -----------
    num_scatterers : int
        散射體數量
    x_size, y_size, z_size : float
        空間尺寸 (mm)
    vessel_cx, vessel_cz : float
        血管中心 X 和 Z 座標 (mm)
    R : float
        血管半徑 (mm)
    Vmax : float
        中心最大速度 (mm/s)
    flow_direction : ndarray (3,), optional
        流動方向單位向量 [vx, vy, vz]，控制散射體移動方向
        如果為 None，預設為 [0, 1, 0] (Y 方向)
    vessel_axis : ndarray (3,), optional
        血管軸線方向 [ax, ay, az]，控制血管幾何形狀
        如果為 None，與 flow_direction 相同（向後兼容）
    vessel_cy : float, optional
        血管中心 Y 座標 (mm)，預設為 0.0

    Returns:
    --------
    x_scatter, y_scatter, z_scatter : ndarray
        散射體 3D 座標
    amplitudes : ndarray
        散射體振幅
    v_profile : ndarray
        每個散射體的速度大小 (Poiseuille profile)
    flow_direction_normalized : ndarray (3,)
        正規化後的流動方向向量
    """
    # 0. 處理預設參數
    if flow_direction is None:
        flow_direction = np.array([0.0, 1.0, 0.0])  # 預設 Y 方向
    flow_direction_normalized = flow_direction / np.linalg.norm(flow_direction)

    # 如果 vessel_axis 為 None，使用 flow_direction（向後兼容）
    if vessel_axis is None:
        vessel_axis = flow_direction_normalized.copy()
    else:
        vessel_axis = vessel_axis / np.linalg.norm(vessel_axis)

    # 1. 生成散射體位置（只在血管內）
    # 使用拒絕採樣法：生成隨機點，只保留在血管內的
    x_scatter = []
    y_scatter = []
    z_scatter = []
    amplitudes = []

    generated = 0
    max_attempts = num_scatterers * 10  # 最多嘗試次數
    attempts = 0

    while generated < num_scatterers and attempts < max_attempts:
        # 生成一批候選點
        batch_size = min(num_scatterers - generated, 1000)
        x_candidates = x_size * np.random.rand(batch_size) - x_size/2
        y_candidates = y_size * np.random.rand(batch_size) - y_size/2
        z_candidates = z_size * np.random.rand(batch_size)

        # 檢查哪些點在血管內（使用 vessel_axis 決定血管形狀）
        dx = x_candidates - vessel_cx
        dy = y_candidates - vessel_cy
        dz = z_candidates - vessel_cz
        pos_vecs = np.column_stack([dx, dy, dz])

        # 計算到血管軸線的徑向距離（使用 vessel_axis）
        proj_length = np.dot(pos_vecs, vessel_axis)
        proj_vecs = proj_length[:, np.newaxis] * vessel_axis
        perp_vecs = pos_vecs - proj_vecs
        r = np.linalg.norm(perp_vecs, axis=1)

        # 只保留在血管內的點
        inside = r <= R

        x_scatter.extend(x_candidates[inside])
        y_scatter.extend(y_candidates[inside])
        z_scatter.extend(z_candidates[inside])
        amplitudes.extend(np.random.randn(np.sum(inside)) * (1 + 0.5*np.random.rand(np.sum(inside))))

        generated += np.sum(inside)
        attempts += batch_size

    # 轉換為 numpy arrays
    x_scatter = np.array(x_scatter[:num_scatterers])
    y_scatter = np.array(y_scatter[:num_scatterers])
    z_scatter = np.array(z_scatter[:num_scatterers])
    amplitudes = np.array(amplitudes[:num_scatterers])

    # 如果沒有生成足夠的散射體，警告
    if len(x_scatter) < num_scatterers:
        print(f"警告: 只生成了 {len(x_scatter)}/{num_scatterers} 個散射體在血管內")

    # 2. 計算到血管軸線的徑向距離（用於 Poiseuille profile）
    # 血管軸線: 通過 (vessel_cx, vessel_cy, vessel_cz) 沿著 vessel_axis 的直線
    dx = x_scatter - vessel_cx
    dy = y_scatter - vessel_cy
    dz = z_scatter - vessel_cz
    pos_vectors = np.column_stack([dx, dy, dz])  # (N, 3)

    # 投影到血管軸線方向的長度
    proj_length = np.dot(pos_vectors, vessel_axis)  # (N,)

    # 投影向量
    proj_vectors = proj_length[:, np.newaxis] * vessel_axis  # (N, 3)

    # 垂直距離向量 = 總向量 - 投影向量
    perp_vectors = pos_vectors - proj_vectors

    # 到軸線的徑向距離
    r = np.linalg.norm(perp_vectors, axis=1)  # (N,)

    # 3. 計算 Poiseuille 速度剖面
    # 速度大小基於到血管軸線的距離，但移動方向由 flow_direction 決定
    v_profile = Vmax * (1 - (r / R)**2)

    return x_scatter, y_scatter, z_scatter, amplitudes, v_profile, flow_direction_normalized

def update_scatterer_positions(x_scatter, y_scatter, z_scatter,
                               v_profile, flow_direction, dt):
    """
    更新散射體 3D 位置

    Parameters:
    -----------
    x_scatter, y_scatter, z_scatter : ndarray
        當前散射體座標 (mm)
    v_profile : ndarray
        每個散射體的速度大小 (mm/s)
    flow_direction : ndarray (3,)
        流動方向單位向量 [vx, vy, vz]
    dt : float
        時間步長 (s)

    Returns:
    --------
    x_new, y_new, z_new : ndarray
        更新後的座標 (mm)
    """
    # 計算位移量 (mm)
    # v_profile: mm/s, dt: s → v_profile * dt = mm
    displacement = v_profile * dt

    # 3D 位移向量
    dx = displacement * flow_direction[0]
    dy = displacement * flow_direction[1]
    dz = displacement * flow_direction[2]

    # 更新位置
    x_new = x_scatter + dx
    y_new = y_scatter + dy
    z_new = z_scatter + dz

    return x_new, y_new, z_new

def check_and_regenerate_scatterers(x_scatter, y_scatter, z_scatter, amplitudes,
                                   x_size, y_size, z_size,
                                   vessel_cx, vessel_cy, vessel_cz, R,
                                   flow_direction, vessel_axis=None, boundary_margin=0.5):
    """
    檢測離開邊界的散射體並在血管內重新生成

    Parameters:
    -----------
    x_scatter, y_scatter, z_scatter : ndarray
        當前散射體座標 (mm)
    amplitudes : ndarray
        散射體振幅
    x_size, y_size, z_size : float
        空間尺寸 (mm)
    vessel_cx, vessel_cy, vessel_cz : float
        血管中心座標 (mm)
    R : float
        血管半徑 (mm)
    flow_direction : ndarray (3,)
        流動方向單位向量
    vessel_axis : ndarray (3,), optional
        血管軸線方向，控制血管幾何形狀
        如果為 None，使用 flow_direction（向後兼容）
    boundary_margin : float
        邊界容差 (mm)

    Returns:
    --------
    x_scatter, y_scatter, z_scatter : ndarray
        更新後的座標
    amplitudes : ndarray
        更新後的振幅
    num_regenerated : int
        重生的散射體數量
    """
    # 如果 vessel_axis 為 None，使用 flow_direction（向後兼容）
    if vessel_axis is None:
        vessel_axis = flow_direction / np.linalg.norm(flow_direction)
    else:
        vessel_axis = vessel_axis / np.linalg.norm(vessel_axis)

    # 1. 檢測邊界
    x_min, x_max = -x_size/2 - boundary_margin, x_size/2 + boundary_margin
    y_min, y_max = -y_size/2 - boundary_margin, y_size/2 + boundary_margin
    z_min, z_max = -boundary_margin, z_size + boundary_margin

    outside = ((x_scatter < x_min) | (x_scatter > x_max) |
               (y_scatter < y_min) | (y_scatter > y_max) |
               (z_scatter < z_min) | (z_scatter > z_max))

    num_regenerated = np.sum(outside)

    if num_regenerated == 0:
        return x_scatter, y_scatter, z_scatter, amplitudes, 0

    # 2. 對離開的散射體重新生成（在血管內）
    indices = np.where(outside)[0]

    for idx in indices:
        # 重試直到生成在血管內的點
        max_attempts = 100
        for attempt in range(max_attempts):
            x_new = x_size * np.random.rand() - x_size/2
            y_new = y_size * np.random.rand() - y_size/2
            z_new = z_size * np.random.rand()

            # 檢查是否在血管內（使用 vessel_axis）
            dx, dy, dz = x_new - vessel_cx, y_new - vessel_cy, z_new - vessel_cz
            pos_vec = np.array([dx, dy, dz])
            proj_length = np.dot(pos_vec, vessel_axis)
            proj_vec = proj_length * vessel_axis
            perp_vec = pos_vec - proj_vec
            r = np.linalg.norm(perp_vec)

            if r <= R:
                x_scatter[idx] = x_new
                y_scatter[idx] = y_new
                z_scatter[idx] = z_new
                amplitudes[idx] = np.random.randn() * (1 + 0.5*np.random.rand())
                break

    return x_scatter, y_scatter, z_scatter, amplitudes, num_regenerated
