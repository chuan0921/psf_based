import numpy as np
from scipy.signal import savgol_filter
from scipy.special import jv  # Bessel functions


def compute_wall_shear_stress(velocity_field, wall_coords, mu, sg_order=3, sg_window_ratio=0.3):
    """計算血管壁剪切應力 (Wall Shear Stress, WSS)。
    
    基於 Leow & Tang (2018) 提出的方法，使用 Savitzky-Golay 濾波器平滑
    速度剖面後，計算血管壁附近的剪切率張量，並轉換至血管壁座標系統。
    
    此方法可降低因速度梯度波動和壁面位置追蹤誤差所造成的 WSS 估計不穩定性。
    研究顯示，200 µm 的壁面偏差可能導致高達 60% 的誤差，但透過高幀率成像
    (可達 10 kHz) 配合對比劑可顯著改善壁面-流體邊界的辨識。

    Args:
        velocity_field: numpy.ndarray
            二維速度場，形狀為 (ny, nx, 2)，其中最後一維為 (u, v) 速度分量。
            單位：m/s。
        wall_coords: numpy.ndarray
            血管壁座標點，形狀為 (n_points, 2)，每點為 (x, y) 座標。
            單位：m。
        mu: float
            動態黏度 (dynamic viscosity)。對於血液，典型值為 0.004 kg/(m·s)。
            單位：kg/(m·s) 或 Pa·s。
        sg_order: int, optional
            Savitzky-Golay 濾波器的多項式階數。預設為 3。
            二階適用於拋物線流速剖面，三階提供更大的彈性以處理
            非拋物線的生理流速剖面（如股動脈的脈動流）。
        sg_window_ratio: float, optional
            Savitzky-Golay 濾波器視窗長度相對於血管直徑的比例。
            預設為 0.3（即 0.3D），可產生 8.6 ± 6.8% 的誤差。
            若使用 0.2，則適用於二階濾波器，誤差為 6.9 ± 5.8%。

    Returns:
        wss: numpy.ndarray
            沿血管壁的剪切應力分布，形狀為 (n_points,)。
            單位：Pa。
        wsr: numpy.ndarray
            沿血管壁的剪切率分布，形狀為 (n_points,)。
            單位：s⁻¹。

    Raises:
        ValueError: 若速度場維度不正確或濾波器參數超出有效範圍。

    Notes:
        WSS 計算基於以下公式：
        
        τ_w = μ * ε'₁₂
        
        其中：
        - τ_w: 壁面剪切應力 (Pa)
        - μ: 動態黏度 (Pa·s)
        - ε'₁₂: 壁面座標系中的切向剪切率張量分量 (s⁻¹)
        
        剪切率張量變換：
        
        ε'_mn = Σᵢ Σⱼ C_im * C_jn * ε_ij
        
        其中旋轉矩陣 C_ij 定義為：
        
        C_ij = [[cos(θ), sin(θ)],
                [-sin(θ), cos(θ)]]
        
        原始應變率張量：
        
        ε_ij = ∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ

    References:
        Leow, C. H., & Tang, M. X. (2018). Spatio-Temporal Flow and Wall 
        Shear Stress Mapping Based on Incoherent Ensemble-Correlation of 
        Ultrafast Contrast Enhanced Ultrasound Images. Ultrasound in Medicine 
        & Biology, 44(1), 134-152.
        https://doi.org/10.1016/j.ultrasmedbio.2017.08.930

    Examples:
        >>> import numpy as np
        >>> # 建立模擬的拋物線速度場
        >>> ny, nx = 100, 200
        >>> velocity_field = np.zeros((ny, nx, 2))
        >>> # 設定血液黏度
        >>> mu = 0.004  # Pa·s
        >>> # 定義血管壁座標
        >>> wall_coords = np.column_stack([np.linspace(0, 0.01, 50), 
        ...                                 np.zeros(50)])
        >>> wss, wsr = compute_wall_shear_stress(velocity_field, wall_coords, mu)
    """
    pass  # 實際實現


def compute_strain_rate_tensor(velocity_field, dx, dy):
    """計算二維應變率張量。
    
    根據速度場計算應變率張量的各個分量，用於後續的 WSS 估計。

    Args:
        velocity_field: numpy.ndarray
            二維速度場，形狀為 (ny, nx, 2)。
        dx: float
            x 方向的空間解析度。單位：m。
        dy: float
            y 方向的空間解析度。單位：m。

    Returns:
        strain_tensor: numpy.ndarray
            應變率張量，形狀為 (ny, nx, 2, 2)。
            ε_ij = [∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ]
            單位：s⁻¹。

    Notes:
        應變率張量定義為（公式 10）：
        
        ε_ij = ∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ
        
        其中 u = (u₁, u₂) 為速度向量，x = (x₁, x₂) 為空間座標。
    """
    pass  # 實際實現


def transform_to_wall_coordinates(strain_tensor, wall_angle):
    """將應變率張量從影像座標系轉換到血管壁座標系。
    
    使用旋轉變換矩陣將原始的笛卡爾座標應變率張量轉換至
    與血管壁切線方向對齊的座標系統。

    Args:
        strain_tensor: numpy.ndarray
            原始影像座標系中的應變率張量，形狀為 (2, 2) 或 (n_points, 2, 2)。
            單位：s⁻¹。
        wall_angle: float or numpy.ndarray
            血管壁切線與 x 軸的夾角 θ。
            若為 float，對所有點使用相同角度；
            若為 array，形狀為 (n_points,)，每點使用對應角度。
            單位：弧度 (radians)。

    Returns:
        transformed_tensor: numpy.ndarray
            血管壁座標系中的應變率張量，形狀與輸入相同。
            ε'₁₂ 分量即為壁面剪切率。
            單位：s⁻¹。

    Notes:
        張量變換公式（公式 9）：
        
        ε'_mn = Σᵢ₌₁² Σⱼ₌₁² C_im * C_jn * ε_ij
        
        旋轉變換矩陣（公式 11）：
        
        C_ij = [[cos(θ),  sin(θ)],
                [-sin(θ), cos(θ)]]
        
        其中 θ 為原始座標與血管壁座標之間的旋轉角度。
    """
    pass  # 實際實現


def smooth_velocity_profile_sg(velocity_profile, vessel_diameter, spatial_resolution, 
                                order=3, window_ratio=0.3):
    """使用 Savitzky-Golay 濾波器平滑速度剖面以降低 WSR 估計誤差。
    
    由於剪切率是速度梯度，對速度剖面中的小幅波動非常敏感。
    SG 濾波器透過最小平方法擬合低階多項式來平滑數據，
    可有效降低 WSR 測量的不確定性。

    Args:
        velocity_profile: numpy.ndarray
            沿血管徑向的速度剖面，形狀為 (n_points,)。
            單位：m/s。
        vessel_diameter: float
            血管直徑。單位：m。
        spatial_resolution: float
            影像的空間解析度 (dr)。單位：m。
        order: int, optional
            多項式階數。預設為 3。
            二階：適用於拋物線流速剖面，誤差較低但彈性較小。
            三階：適用於非拋物線的生理流速剖面，提供更大彈性。
            不建議使用更高階，因為高階多項式對小幅隨機誤差較敏感。
        window_ratio: float, optional
            濾波器視窗長度相對於血管直徑的比例 (n/D)。預設為 0.3。
            較大的比例可減少誤差和變異，但可能過度限制流速剖面形狀。
            建議選擇使平均誤差和變異都小於 10% 的值。

    Returns:
        smoothed_profile: numpy.ndarray
            平滑後的速度剖面，形狀與輸入相同。
            單位：m/s。

    Notes:
        濾波器參數選擇建議：
        - 二階 + 0.2D 視窗：誤差 6.9 ± 5.8%
        - 三階 + 0.3D 視窗：誤差 8.6 ± 6.8%
        
        過長的濾波器視窗會將流速剖面限制為特定多項式形狀，
        迫使剖面更符合遠離壁面區域的速度分布，而非近壁區域。

    References:
        Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation 
        of data by simplified least squares procedures. Analytical Chemistry, 
        36(8), 1627-1639.
    """
    # 計算視窗長度（必須為奇數）
    window_length = int(window_ratio * vessel_diameter / spatial_resolution)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, order + 2)  # 確保視窗長度 > 多項式階數
    
    return savgol_filter(velocity_profile, window_length, order)


def poiseuille_flow_velocity(r, R, v0):
    """計算穩態 Poiseuille 流的速度剖面。
    
    用於驗證和校準 UIV 速度估計的解析解。

    Args:
        r: float or numpy.ndarray
            距血管中心的徑向距離。單位：m。
        R: float
            血管半徑。單位：m。
        v0: float
            中心線速度（最大速度）。單位：m/s。

    Returns:
        v: float or numpy.ndarray
            該徑向位置的流速。單位：m/s。

    Notes:
        Poiseuille 流速公式（公式 1）：
        
        v(r) = v₀ * (1 - r²/R²)
        
        對應的壁面剪切應力（公式 2）：
        
        τ_wall = μ * (2v₀/R)

    Examples:
        >>> # 6mm 直徑血管，中心線速度 50 cm/s
        >>> R = 0.003  # 3mm 半徑
        >>> v0 = 0.5   # 50 cm/s
        >>> r = np.linspace(0, R, 100)
        >>> v = poiseuille_flow_velocity(r, R, v0)
    """
    return v0 * (1 - (r**2) / (R**2))


def poiseuille_wall_shear_stress(v0, R, mu):
    """計算穩態 Poiseuille 流的壁面剪切應力。
    
    Args:
        v0: float
            中心線速度。單位：m/s。
        R: float
            血管半徑。單位：m。
        mu: float
            動態黏度。單位：Pa·s。

    Returns:
        tau_wall: float
            壁面剪切應力。單位：Pa。

    Notes:
        公式 2：τ_wall = μ * (2v₀/R)
    """
    return mu * (2 * v0) / R


def womersley_velocity_profile(r, R, t, V0, Vn, phi_n, omega, alpha, n_harmonics=8):
    """計算 Womersley 脈動流的時變速度剖面。
    
    脈動流被視為穩態流分量與一系列振盪分量的疊加。
    用於模擬生理條件下的動脈血流（如頸動脈、股動脈）。

    Args:
        r: float or numpy.ndarray
            距血管中心的徑向距離。單位：m。
        R: float
            血管半徑。單位：m。
        t: float or numpy.ndarray
            時間。單位：s。
        V0: float
            平均速度的直流分量。單位：m/s。
        Vn: numpy.ndarray
            各諧波的速度振幅，形狀為 (n_harmonics,)。
            單位：m/s。
        phi_n: numpy.ndarray
            各諧波的相位，形狀為 (n_harmonics,)。
            單位：弧度。
        omega: float
            基頻角頻率 (ω = 2πf)。單位：rad/s。
        alpha: float
            Womersley 數，α = R * sqrt(ω/ν)，
            其中 ν 為運動黏度。無量綱。
        n_harmonics: int, optional
            使用的諧波數量。預設為 8。

    Returns:
        V: numpy.ndarray
            時變速度剖面。單位：m/s。

    Notes:
        Womersley 流速剖面（公式 7）：
        
        V(r/R, t) = 2V₀(1 - r²/R²) + Σₙ Vₙ|ψₙ|cos(nωt - φₙ + χₙ)
        
        其中 ψₙ 為與 Womersley 數相關的複數函數（公式 5）：
        
        ψₙ(r/R, τₙ) = [τJ₀(τₙ) - τJ₀((r/R)τₙ)] / [τJ₀(τₙ) - 2J₁(τₙ)]
        
        τ = j^(3/2) * α，Jₘ 為第一類 Bessel 函數。
        
        平均速度波形的 Fourier 分解（公式 6）：
        
        V̄(t) = V₀ + Σₙ Vₙ cos(nωt - φₙ)

    References:
        Womersley, J. R. (1955). Method for the calculation of velocity, 
        rate of flow and viscous drag in arteries when the pressure gradient 
        is known. J Physiol, 127, 553-563.
        
        Evans, D. H., & McDicken, W. N. (2000). Doppler ultrasound: Physics, 
        instrumentation and signal processing. John Wiley & Sons.
    """
    # 穩態拋物線分量
    V = 2 * V0 * (1 - (r/R)**2)
    
    # 各諧波振盪分量
    for n in range(1, n_harmonics + 1):
        tau_n = (1j)**(3/2) * alpha * np.sqrt(n)
        
        # 計算 ψₙ（公式 5）
        psi_n = (tau_n * jv(0, tau_n) - tau_n * jv(0, (r/R) * tau_n)) / \
                (tau_n * jv(0, tau_n) - 2 * jv(1, tau_n))
        
        # 添加振盪分量
        V = V + Vn[n-1] * np.abs(psi_n) * np.cos(n * omega * t - phi_n[n-1] + np.angle(psi_n))
    
    return V


def compute_mean_error(y_estimated, y_reference):
    """計算平均誤差 (Mean Error, ME)。
    
    用於評估 UIV 系統的估計精度。

    Args:
        y_estimated: numpy.ndarray
            UIV 估計值的時間序列。
        y_reference: numpy.ndarray
            參考（真實）值的時間序列。

    Returns:
        me: float
            平均誤差百分比 (%)。

    Notes:
        平均誤差公式（公式 12）：
        
        ME = [∫₀ᵗ y(t)dt - ∫₀ᵗ yₜ(t)dt] / ∫₀^∞ yₜ(t)dt × 100%
        
        其中 y 為估計值，yₜ 為參考值。
    """
    integral_estimated = np.trapz(y_estimated)
    integral_reference = np.trapz(y_reference)
    
    me = (integral_estimated - integral_reference) / integral_reference * 100
    return me


def compute_nrmse(y_estimated, y_reference):
    """計算正規化均方根誤差 (Normalized Root Mean Square Error, NRMSE)。
    
    用於評估 UIV 系統的估計精度。

    Args:
        y_estimated: numpy.ndarray
            UIV 估計值的時間序列。
        y_reference: numpy.ndarray
            參考（真實）值的時間序列。

    Returns:
        nrmse: float
            正規化均方根誤差百分比 (%)。

    Notes:
        NRMSE 公式（公式 13）：
        
        NRMSE = (1/yₜ₍ₘₐₓ₎) * sqrt[Σ(y - yₜ)²/n] × 100%
        
        其中 yₜ₍ₘₐₓ₎ 為參考值的最大值，n 為時間樣本數。
    """
    n = len(y_estimated)
    y_max = np.max(y_reference)
    
    mse = np.sum((y_estimated - y_reference)**2) / n
    nrmse = (1 / y_max) * np.sqrt(mse) * 100
    return nrmse