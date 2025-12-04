import numpy as np

def create_spatial_grid(x_size, z_size, dx):
    x = np.arange(-x_size/2, x_size/2 + dx, dx)
    z = np.arange(0, z_size + dx, dx)
    X, Z = np.meshgrid(x, z)
    return X, Z, x, z

def create_roi_masks(X, Z, roi_x_positions, vessel_cz, roi_size):
    half_size = roi_size / 2
    roi_masks = []

    for roi_x in roi_x_positions:
        mask = ((X >= roi_x - half_size) & (X <= roi_x + half_size) &
                (Z >= vessel_cz - half_size) & (Z <= vessel_cz + half_size))
        roi_masks.append(mask)

        if not mask.any():
            print(f"Warning: ROI at {roi_x} mm is empty")

    return roi_masks

def create_vertical_iw_line(X, Z, iw_config):
    """
    Generate a single vertical column of IWs at x = vessel_cx (perpendicular to flow).

    Only includes IWs where the center is inside the vessel (distance < R).
    This is optimized for X-direction flow measurement with Poiseuille profile.

    Parameters:
    -----------
    X, Z : ndarray (Nz, Nx)
        Spatial coordinate grids (mm)
    iw_config : InterrogationWindowConfig
        Configuration object

    Returns:
    --------
    iw_info : list of dict
        Each element: {
            'cx': float,       # Center X position (mm) - always vessel_cx
            'cz': float,       # Center Z position (mm)
            'mask': ndarray,   # Boolean mask (Nz, Nx)
            'index': tuple,    # (i, 0) - single column
            'inside_vessel': bool,  # Always True (filtered)
            'vessel_overlap': float  # Fraction of IW inside vessel [0, 1]
            'distance_to_center': float  # Distance from vessel center (mm)
        }
    n_iws : int
        Number of IWs in the vertical line
    """
    # Fixed X position at vessel center
    cx = iw_config.vessel_cx

    # Z range: from cz - R to cz + R
    z_min = iw_config.vessel_cz - iw_config.vessel_radius
    z_max = iw_config.vessel_cz + iw_config.vessel_radius

    # Generate Z centers with overlap
    z_start = z_min + iw_config.iw_size_z / 2
    z_centers = np.arange(z_start, z_max, iw_config.step_z)

    # 不過濾邊界，允許 IW 邊緣超出血管（中心在內即可）

    # Create IW metadata and masks
    iw_info = []
    half_x = iw_config.iw_size_x / 2
    half_z = iw_config.iw_size_z / 2

    for i, cz in enumerate(z_centers):
        # Check if IW center is inside vessel (1D check since cx = vessel_cx)
        dist_to_center = abs(cz - iw_config.vessel_cz)

        if dist_to_center >= iw_config.vessel_radius:
            continue  # Skip IWs outside vessel

        # Create rectangular mask
        mask = ((X >= cx - half_x) & (X <= cx + half_x) &
               (Z >= cz - half_z) & (Z <= cz + half_z))

        # Check if valid (enough pixels)
        if np.sum(mask) < iw_config.min_pixels_per_iw:
            continue

        # Calculate vessel overlap
        vessel_overlap = _estimate_vessel_overlap(
            cx, cz, iw_config.vessel_cx, iw_config.vessel_cz,
            iw_config.vessel_radius, half_x, half_z, mask, X, Z
        )

        iw_info.append({
            'cx': cx,
            'cz': cz,
            'mask': mask,
            'index': (i, 0),  # Single column
            'inside_vessel': True,  # Guaranteed by filter
            'vessel_overlap': vessel_overlap,
            'distance_to_center': dist_to_center
        })

    # Sort by distance to center (center IW first)
    iw_info.sort(key=lambda x: x['distance_to_center'])

    # Re-index after sorting
    for i, iw in enumerate(iw_info):
        iw['index'] = (i, 0)

    return iw_info, len(iw_info)


def create_iw_grid(X, Z, iw_config):
    """
    Generate 2D grid of Interrogation Windows with 50% overlap.

    Parameters:
    -----------
    X, Z : ndarray (Nz, Nx)
        Spatial coordinate grids (mm)
    iw_config : InterrogationWindowConfig
        Configuration object

    Returns:
    --------
    iw_info : list of dict
        Each element: {
            'cx': float,       # Center X position (mm)
            'cz': float,       # Center Z position (mm)
            'mask': ndarray,   # Boolean mask (Nz, Nx)
            'index': tuple,    # (i, j) grid index
            'inside_vessel': bool,  # Is center inside vessel?
            'vessel_overlap': float  # Fraction of IW inside vessel [0, 1]
        }
    grid_shape : tuple (n_z, n_x)
        Number of IWs in Z and X directions
    """
    # 1. Calculate grid coverage area
    # Coverage: vessel_center ± (vessel_radius + margin)
    x_min = iw_config.vessel_cx - (iw_config.vessel_radius + iw_config.coverage_margin)
    x_max = iw_config.vessel_cx + (iw_config.vessel_radius + iw_config.coverage_margin)
    z_min = iw_config.vessel_cz - (iw_config.vessel_radius + iw_config.coverage_margin)
    z_max = iw_config.vessel_cz + (iw_config.vessel_radius + iw_config.coverage_margin)

    # 2. Generate grid centers
    # Start from first IW center (offset by half IW size from boundary)
    x_start = x_min + iw_config.iw_size_x / 2
    z_start = z_min + iw_config.iw_size_z / 2

    x_centers = np.arange(x_start, x_max, iw_config.step_x)
    z_centers = np.arange(z_start, z_max, iw_config.step_z)

    # Ensure we don't exceed max boundary
    x_centers = x_centers[x_centers + iw_config.iw_size_x/2 <= x_max]
    z_centers = z_centers[z_centers + iw_config.iw_size_z/2 <= z_max]

    # 3. Create IW metadata and masks
    iw_info = []
    half_x = iw_config.iw_size_x / 2
    half_z = iw_config.iw_size_z / 2

    for i, cz in enumerate(z_centers):
        for j, cx in enumerate(x_centers):
            # Create rectangular mask
            mask = ((X >= cx - half_x) & (X <= cx + half_x) &
                   (Z >= cz - half_z) & (Z <= cz + half_z))

            # Check if valid (enough pixels)
            if np.sum(mask) < iw_config.min_pixels_per_iw:
                continue

            # Calculate vessel overlap
            dist_to_center = np.sqrt((cx - iw_config.vessel_cx)**2 +
                                    (cz - iw_config.vessel_cz)**2)
            inside_vessel = dist_to_center <= iw_config.vessel_radius

            # Estimate vessel overlap (approximate)
            vessel_overlap = _estimate_vessel_overlap(
                cx, cz, iw_config.vessel_cx, iw_config.vessel_cz,
                iw_config.vessel_radius, half_x, half_z, mask, X, Z
            )

            iw_info.append({
                'cx': cx,
                'cz': cz,
                'mask': mask,
                'index': (i, j),
                'inside_vessel': inside_vessel,
                'vessel_overlap': vessel_overlap
            })

    grid_shape = (len(z_centers), len(x_centers))

    return iw_info, grid_shape


def _estimate_vessel_overlap(cx, cz, vessel_cx, vessel_cz, vessel_radius,
                            half_x, half_z, mask, X, Z):
    """
    Estimate fraction of IW area that overlaps with vessel interior.

    Returns:
    --------
    overlap_ratio : float [0, 1]
        Fraction of IW pixels inside vessel
    """
    # Get IW pixels
    iw_pixels_x = X[mask]
    iw_pixels_z = Z[mask]

    # Check which pixels are inside vessel
    dist = np.sqrt((iw_pixels_x - vessel_cx)**2 + (iw_pixels_z - vessel_cz)**2)
    inside = dist <= vessel_radius

    overlap_ratio = np.sum(inside) / len(iw_pixels_x) if len(iw_pixels_x) > 0 else 0.0

    return overlap_ratio


def extract_iw_masks(iw_info):
    """
    Extract just the boolean masks from IW info for correlation.

    Parameters:
    -----------
    iw_info : list of dict
        Output from create_iw_grid()

    Returns:
    --------
    masks : list of ndarray
        Boolean masks for correlation functions
    """
    return [iw['mask'] for iw in iw_info]


def validate_iw_grid(iw_info, iw_config):
    """
    Validate IW grid and provide statistics.

    Returns:
    --------
    validation_report : dict
        Contains statistics and warnings
    """
    total_iws = len(iw_info)
    inside_vessel = sum(iw['inside_vessel'] for iw in iw_info)

    # Overlap statistics
    overlaps = [iw['vessel_overlap'] for iw in iw_info]
    avg_overlap = np.mean(overlaps)

    # Pixel counts
    pixel_counts = [np.sum(iw['mask']) for iw in iw_info]
    min_pixels = np.min(pixel_counts)
    max_pixels = np.max(pixel_counts)
    avg_pixels = np.mean(pixel_counts)

    # Expected IW size in pixels
    dx = 0.02  # mm/pixel (from GridConfig)
    expected_pixels = (iw_config.iw_size_x / dx) * (iw_config.iw_size_z / dx)

    report = {
        'total_iws': total_iws,
        'inside_vessel': inside_vessel,
        'outside_vessel': total_iws - inside_vessel,
        'avg_vessel_overlap': avg_overlap,
        'pixel_count_range': (min_pixels, max_pixels),
        'avg_pixel_count': avg_pixels,
        'expected_pixel_count': expected_pixels,
        'iw_size_mm': (iw_config.iw_size_x, iw_config.iw_size_z),
        'step_size_mm': (iw_config.step_x, iw_config.step_z),
        'coverage_area_mm': {
            'x_range': (min(iw['cx'] for iw in iw_info),
                       max(iw['cx'] for iw in iw_info)),
            'z_range': (min(iw['cz'] for iw in iw_info),
                       max(iw['cz'] for iw in iw_info))
        }
    }

    return report


def print_iw_validation_report(report):
    """Pretty print validation report"""
    print("\n" + "="*60)
    print("INTERROGATION WINDOW GRID VALIDATION")
    print("="*60)
    print(f"Total IWs: {report['total_iws']}")
    print(f"  Inside vessel: {report['inside_vessel']}")
    print(f"  Outside/partial: {report['outside_vessel']}")
    print(f"  Avg vessel overlap: {report['avg_vessel_overlap']:.2%}")
    print(f"\nIW Size: {report['iw_size_mm'][0]:.2f} mm × {report['iw_size_mm'][1]:.2f} mm")
    print(f"Step Size: {report['step_size_mm'][0]:.2f} mm × {report['step_size_mm'][1]:.2f} mm")
    print(f"\nPixel Count per IW:")
    print(f"  Expected: {report['expected_pixel_count']:.0f} pixels")
    print(f"  Actual: {report['avg_pixel_count']:.0f} ± {report['pixel_count_range'][1] - report['pixel_count_range'][0]:.0f} pixels")
    print(f"\nCoverage Area:")
    print(f"  X: [{report['coverage_area_mm']['x_range'][0]:.2f}, {report['coverage_area_mm']['x_range'][1]:.2f}] mm")
    print(f"  Z: [{report['coverage_area_mm']['z_range'][0]:.2f}, {report['coverage_area_mm']['z_range'][1]:.2f}] mm")
    print("="*60)
