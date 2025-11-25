close all;
% LaminarFlow_MaxCC_FullDiameter_Profile.m
% Blood Flow Velocity Profile Reconstruction using Multi-ROI NCC
% Complete Vessel Diameter Measurement: -5mm to +5mm (21 measurement points)

% Define simulation parameters
fx = 5e6;          % Center frequency: 5 MHz
c = 1540;          % Speed of sound in tissue (m/s)
wavelength = c / fx; 
gap = 0.01e-3;         
N_elements = 2;       
% ------------------------------
% index, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, material, width, height, dummy, dummy, dummy
% ------------------------------
% Using two rows (2 elements) data 
data = [... 
    1, -0.3e-3 - gap/2, (6e-3+gap)/2 - (2e-3) - gap, 0, -0.3e-3 - gap/2, (-6e-3-gap)/2, 0, -gap/2, (-6e-3-gap)/2, 0, -gap/2, (-6e-3-gap)/2 + (2e-3), 0, 1, 0.3e-3, 6e-3+gap, 0, 0, 0;...
    2, -0.3e-3 - gap/2, (6e-3+gap)/2, 0, -gap/2, (6e-3+gap)/2, 0, -gap/2, (-6e-3-gap)/2 + 2e-3 + gap, 0, -0.3e-3 - gap/2, (6e-3+gap)/2 - (2e-3), 0, 1, 0.3e-3, 6e-3+gap, 0, 0, 0];
% Calculate element centers (x, y, z) using four corners averaged
element_centers = zeros(N_elements, 3);
for i = 1:N_elements
    x_coords = [data(i,2), data(i,5), data(i,8), data(i,11)];
    y_coords = [data(i,3), data(i,6), data(i,9), data(i,12)];
    z_coords = [data(i,4), data(i,7), data(i,10), data(i,13)];
    element_centers(i,:) = [mean(x_coords), mean(y_coords), mean(z_coords)];
end
 
% Define odd and even element indices
odd_idx = 1:2:N_elements; 
even_idx = 2:2:N_elements;
fprintf('Initial odd element center(s):\n');
for i = odd_idx
    fprintf('  %.6f  %.6f  %.6f\n', element_centers(i,1), element_centers(i,2), element_centers(i,3));
end
fprintf('Initial even element center(s):\n');
for i = even_idx
    fprintf('  %.6f  %.6f  %.6f\n', element_centers(i,1), element_centers(i,2), element_centers(i,3));
end
% Calculate elevational pitch (only y)
odd_y = mean(element_centers(odd_idx,2)); 
even_y = mean(element_centers(even_idx,2));
y_distance = abs(odd_y - even_y) * 1e3; 
fprintf('Elevational pitch (y) = %.4f mm\n', y_distance);
% ------------------------------
% Grid definition (common for all iterations)
% ------------------------------
x_size = 10;        
z_size = 13;      
dx = 0.02;         
x = -x_size/2:dx:x_size/2;
z = 0:dx:z_size;
[X, Z] = meshgrid(x, z);

% ------------------------------
% Multi-ROI Definition: Full Diameter Coverage (-5mm to +5mm)
% ------------------------------
vessel_cx = 0;
vessel_cz = z_size / 2;
roi_size = 2.5;
half_size = roi_size / 2;

% Define 41 ROI positions across vessel diameter
roi_x_positions = -5:0.25:5;  % 41 positions from -5mm to +5mm, step 0.25mm
num_roi_positions = length(roi_x_positions);

% Create ROI masks for each position
roi_masks = cell(num_roi_positions, 1);
for roi_idx = 1:num_roi_positions
    roi_center_x = roi_x_positions(roi_idx);
    x_min = roi_center_x - half_size;
    x_max = roi_center_x + half_size;
    z_min = vessel_cz - half_size;
    z_max = vessel_cz + half_size;
    roi_masks{roi_idx} = (X >= x_min) & (X <= x_max) & (Z >= z_min) & (Z <= z_max);
    
    if ~any(roi_masks{roi_idx}(:))
        warning('ROI at position %.1f mm is empty. Check grid definition.', roi_center_x);
    end
end

fprintf('Multi-ROI System Setup Complete:\n');
fprintf('  -> Number of ROI positions: %d\n', num_roi_positions);
fprintf('  -> ROI coverage: %.1f mm to %.1f mm\n', roi_x_positions(1), roi_x_positions(end));
fprintf('  -> ROI spacing: %.1f mm\n', roi_x_positions(2) - roi_x_positions(1));

% ------------------------------
% Simulation parameters
% ------------------------------
num_scatterers = 8000;
y_size = 5;
sigma_y = 0.3;
p_comp = 1; 
R = 5;
Vmax = 10;
dt = 1 / 100;
num_frames = 100;
num_iterations = 10;  % Multiple iterations for statistical analysis

% Physical PSF sigma calculation (tunable parameters)
Fnum = 2;              % F-number (tuneable)
ncycles = 2;           % Pulse cycles (tuneable)
sigma_scale = 0.9;     % manual speckle scale (<1 smaller)
lambda_mm = wavelength * 1e3;                  % convert wavelength to mm

FWHM_lat = Fnum * lambda_mm;                  % lateral FWHM (mm)
sigma_x_phys = FWHM_lat / (2*sqrt(2*log(2))); % lateral sigma (mm)

pulse_len = ncycles * lambda_mm;             % pulse length (mm)
FWHM_ax = pulse_len / 2;                      % axial FWHM (mm)
sigma_z_phys = FWHM_ax / (2*sqrt(2*log(2)));  % axial sigma (mm)

% Apply manual scaling to speckle size
sigma_x_phys = sigma_x_phys * sigma_scale;
sigma_z_phys = sigma_z_phys * sigma_scale;

fprintf('Using physics-based sigma (scale=%.2f): sigma_x=%.3f mm, sigma_z=%.3f mm\n', sigma_scale, sigma_x_phys, sigma_z_phys);

% Data structures for multi-ROI measurements
all_roi_cc = zeros(num_roi_positions, num_iterations, num_frames);
measured_velocities_profile = zeros(num_roi_positions, num_iterations);

% ==========================================================
% Main loop for iterations
% ==========================================================
for iter = 1:num_iterations
    fprintf('\n=============================================\n');
    fprintf('Starting Multi-ROI Profile Measurement %d/%d\n', iter, num_iterations);
    fprintf('=============================================\n');
    
    x_scatter = x_size * rand(num_scatterers, 1) - x_size/2;
    z_scatter = z_size * rand(num_scatterers, 1);
    y_scatter = y_size * rand(num_scatterers, 1) - y_size/2;
    amplitudes = randn(num_scatterers, 1) .* (1 + 0.5*rand(num_scatterers, 1));
    
    r = sqrt((x_scatter - vessel_cx).^2 + (z_scatter - vessel_cz).^2);
    v_profile = Vmax * (1 - (r ./ R).^2);
    v_profile(r > R) = 0;
    y_move_speed = v_profile * dt * 1e-3;
    
    % Generate reference image (odd elements)
    ref_centers = element_centers;
    composite_image_ref = zeros(size(X));
    for tx = odd_idx
        for rx = odd_idx
            sub_image = zeros(size(X));
            for s = 1:num_scatterers
                tx_delay = sqrt((x_scatter(s) - ref_centers(tx,1))^2 + (y_scatter(s) - ref_centers(tx,2))^2 + (z_scatter(s) - ref_centers(tx,3))^2) / c;
                rx_delay = sqrt((x_scatter(s) - ref_centers(rx,1))^2 + (y_scatter(s) - ref_centers(rx,2))^2 + (z_scatter(s) - ref_centers(rx,3))^2) / c;
                total_delay = tx_delay + rx_delay;
                r_tx = [x_scatter(s) - ref_centers(tx,1), y_scatter(s) - ref_centers(tx,2), z_scatter(s) - ref_centers(tx,3)];
                r_rx = [x_scatter(s) - ref_centers(rx,1), y_scatter(s) - ref_centers(rx,2), z_scatter(s) - ref_centers(rx,3)];
                directivity_tx = r_tx(3) / norm(r_tx);
                directivity_rx = r_rx(3) / norm(r_rx);
                directivity_orig = directivity_tx * directivity_rx;
                directivity = directivity_orig^(1 - p_comp); 
                elev_weight = exp(- (y_scatter(s)^2) / (2 * sigma_y^2) );
                sigma_x = sigma_x_phys; sigma_z = sigma_z_phys;
                phase = 2*pi*fx*total_delay;
                psf = amplitudes(s) * directivity * elev_weight * exp(-(((X - x_scatter(s)).^2)/(2*sigma_x^2) + ((Z - z_scatter(s)).^2)/(2*sigma_z^2))) .* cos(2*pi*fx/c * (Z - z_scatter(s)) + phase);
                sub_image = sub_image + psf;
            end
            composite_image_ref = composite_image_ref + sub_image;
        end
    end
    env_ref = abs(hilbert(composite_image_ref));
    env_norm_ref = env_ref / max(env_ref(:));
    ref_image = max(20*log10(env_norm_ref), -30);
    % Rayleigh distribution validation for speckle (empirical vs theoretical)
    figure('Name','Envelope Rayleigh PDF','Position',[150,150,600,400]);
    env_vals = env_norm_ref(:);
    % Empirical PDF
    histogram(env_vals,50,'Normalization','pdf','FaceColor',[0.7 0.7 0.7]);
    hold on;
    % Estimate sigma from mean of Rayleigh distribution: mean = sigma*sqrt(pi/2)
    sigma_env = mean(env_vals) / sqrt(pi/2);
    % Theoretical Rayleigh PDF
    x_vals = linspace(0, max(env_vals), 100);
    y_theo = (x_vals./sigma_env^2) .* exp(-x_vals.^2/(2*sigma_env^2));
    plot(x_vals, y_theo, 'r-', 'LineWidth', 2);
    xlabel('Normalized Envelope Amplitude');
    ylabel('PDF');
    title('Speckle Envelope: Empirical vs Rayleigh');
    legend('Empirical','Rayleigh Fit');
    hold off;
    
    moving_centers = element_centers;
    
    % Frame-by-frame processing
    for frame = 1:num_frames
        y_scatter = y_scatter + y_move_speed;
        
        % Generate moving image (even elements)
        composite_image_even = zeros(size(X));
        for tx = even_idx
            for rx = even_idx
                sub_image = zeros(size(X));
                for s = 1:num_scatterers
                    tx_delay = sqrt((x_scatter(s) - moving_centers(tx,1))^2 + (y_scatter(s) - moving_centers(tx,2))^2 + (z_scatter(s) - moving_centers(tx,3))^2) / c;
                    rx_delay = sqrt((x_scatter(s) - moving_centers(rx,1))^2 + (y_scatter(s) - moving_centers(rx,2))^2 + (z_scatter(s) - moving_centers(rx,3))^2) / c;
                    total_delay = tx_delay + rx_delay;
                    r_tx = [x_scatter(s) - moving_centers(tx,1), y_scatter(s) - moving_centers(tx,2), z_scatter(s) - moving_centers(tx,3)];
                    r_rx = [x_scatter(s) - moving_centers(rx,1), y_scatter(s) - moving_centers(rx,2), z_scatter(s) - moving_centers(rx,3)];
                    directivity_tx = r_tx(3) / norm(r_tx);
                    directivity_rx = r_rx(3) / norm(r_rx);
                    directivity_orig = directivity_tx * directivity_rx;
                    directivity = directivity_orig^(1 - p_comp);
                    elev_weight = exp(- (y_scatter(s)^2) / (2 * sigma_y^2) );
                    sigma_x = sigma_x_phys; sigma_z = sigma_z_phys;
                    phase = 2*pi*fx*total_delay;
                    psf = amplitudes(s) * directivity * elev_weight * exp(-(((X - x_scatter(s)).^2)/(2*sigma_x^2) + ((Z - z_scatter(s)).^2)/(2*sigma_z^2))) .* cos(2*pi*fx/c * (Z - z_scatter(s)) + phase);
                    sub_image = sub_image + psf;
                end
                composite_image_even = composite_image_even + sub_image;
            end
        end
        
        env_even = abs(hilbert(composite_image_even));
        env_norm_even = env_even / max(env_even(:));
        even_image = max(20*log10(env_norm_even), -30);
        
        % Multi-ROI NCC calculation (perform before visualization)
        current_frame_cc = zeros(num_roi_positions, 1);
        for roi_idx = 1:num_roi_positions
            ref_pixels = ref_image(roi_masks{roi_idx});
            even_pixels = even_image(roi_masks{roi_idx});
            if ~isempty(ref_pixels) && ~isempty(even_pixels)
                C = corrcoef(ref_pixels, even_pixels);
                roi_cc = C(1, 2);
            else
                roi_cc = NaN;
            end
            all_roi_cc(roi_idx, iter, frame) = roi_cc;
            current_frame_cc(roi_idx) = roi_cc;
        end
        
        % Enhanced visualization with all ROIs and CC values
        if iter == 1
            if frame == 1
                v = VideoWriter('even_animation.avi');
                v.FrameRate = 10;
                open(v);
                fig_frames = figure('Name','Dynamic Even-element Image with Multi-ROI CC', 'Position', [100, 100, 800, 600]);
            end
            
            figure(fig_frames);
            imagesc(x, z, even_image);
            colormap(gray); colorbar;
            xlabel('Lateral Distance (mm)'); ylabel('Axial Distance (mm)');
            
            % Display key CC values in title
            center_idx = ceil(num_roi_positions/2);
            quarter_left_idx = ceil(num_roi_positions/4);
            quarter_right_idx = ceil(3*num_roi_positions/4);
            title(sprintf('Frame %d: CC Values - Center(%.1fmm):%.3f, Left(%.1fmm):%.3f, Right(%.1fmm):%.3f', ...
                         frame, roi_x_positions(center_idx), current_frame_cc(center_idx), ...
                         roi_x_positions(quarter_left_idx), current_frame_cc(quarter_left_idx), ...
                         roi_x_positions(quarter_right_idx), current_frame_cc(quarter_right_idx)));
            
            hold on;
            % Display vessel boundary
            vessel_position = [vessel_cx - R, vessel_cz - R, 2*R, 2*R];
            rectangle('Position', vessel_position, 'Curvature', [1,1], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '-');
            
            % Remove old colored ROI plotting and legend
            % ...existing code (old ROI loop and legend removed)...

            % New ROI visualization: center ROI rectangle and green dots for all positions
            hold on;
            center_idx = ceil(num_roi_positions/2);
            center_x = roi_x_positions(center_idx);
            center_z = vessel_cz;
            roi_rect_x = center_x - half_size;
            roi_rect_z = center_z - half_size;
            rectangle('Position',[roi_rect_x, roi_rect_z, roi_size, roi_size], 'EdgeColor','y','LineWidth',2,'LineStyle','--');
            for roi_idx = 1:num_roi_positions
                plot(roi_x_positions(roi_idx), vessel_cz, 'go', 'MarkerSize',6,'LineWidth',2, 'MarkerFaceColor','g');
            end
            legend('Vessel','Center ROI','Lateral Positions','Location','northeast');
            hold off;
            drawnow;
            frame_img = getframe(gcf);
            writeVideo(v, frame_img);
            
            % Print frame statistics for first iteration
            avg_cc = mean(current_frame_cc(~isnan(current_frame_cc)));
            max_cc = max(current_frame_cc(~isnan(current_frame_cc)));
            min_cc = min(current_frame_cc(~isnan(current_frame_cc)));
            fprintf('  [Iter 1] Frame %02d/%d: CC Stats - Avg:%.3f, Max:%.3f, Min:%.3f\n', ...
                    frame, num_frames, avg_cc, max_cc, min_cc);
        else
            % Print processing progress for other iterations
            if frame == 1 || mod(frame, 10) == 0
                fprintf('  Frame %02d/%d: Processing multi-ROI correlations...\n', frame, num_frames);
            end
        end
    end
    
    % Calculate velocities for each ROI position
    fprintf('\nCalculating velocities for each ROI position...\n');
    for roi_idx = 1:num_roi_positions
        current_roi_cc = squeeze(all_roi_cc(roi_idx, iter, :));
        
        % Peak detection and velocity calculation (same method as original)
        if num_frames > 1
            [~, temp_peak_frame] = max(current_roi_cc(2:end));
            temp_peak_frame = temp_peak_frame + 1;
        else
            [~, temp_peak_frame] = max(current_roi_cc);
        end
        
        if temp_peak_frame > 1 && temp_peak_frame < num_frames
            y1_t = current_roi_cc(temp_peak_frame - 1); 
            y2_t = current_roi_cc(temp_peak_frame); 
            y3_t = current_roi_cc(temp_peak_frame + 1);
            denominator_t = y1_t - 2*y2_t + y3_t;
            if abs(denominator_t) > 1e-9
                delta_t = 0.5 * (y1_t - y3_t) / denominator_t;
                precise_peak_frame_t = temp_peak_frame + delta_t;
            else
                precise_peak_frame_t = temp_peak_frame; 
            end
        else
            precise_peak_frame_t = temp_peak_frame; 
        end
        
        time_to_peak_t = precise_peak_frame_t * dt;
        measured_velocity_t = y_distance / time_to_peak_t;
        measured_velocities_profile(roi_idx, iter) = measured_velocity_t;
        
        if roi_idx == 1 || roi_idx == num_roi_positions || roi_idx == ceil(num_roi_positions/2)
            fprintf('  ROI %2d (%.1f mm): Peak at frame %.2f, Velocity = %.3f mm/s\n', ...
                    roi_idx, roi_x_positions(roi_idx), precise_peak_frame_t, measured_velocity_t);
        end
    end
    
    fprintf('Completed Multi-ROI Profile Measurement %d/%d.\n', iter, num_iterations);

end % Main iteration loop

% Close video writer if it was opened
if exist('v', 'var') && isvalid(v)
    close(v);
    fprintf('Video saved as: even_animation.avi\n');
end

% ==========================================================
% Analysis and Visualization
% ==========================================================

% Calculate average velocities and statistical measures across iterations
avg_measured_velocities = mean(measured_velocities_profile, 2);
std_measured_velocities = std(measured_velocities_profile, 0, 2);
cv_measured_velocities = std_measured_velocities ./ abs(avg_measured_velocities); % Coefficient of variation
cv_measured_velocities(avg_measured_velocities == 0) = 0; % Handle division by zero

% Calculate theoretical velocities for each position
theoretical_velocities = zeros(num_roi_positions, 1);
for roi_idx = 1:num_roi_positions
    r_pos = abs(roi_x_positions(roi_idx));
    if r_pos <= R
        theoretical_velocities(roi_idx) = Vmax * (1 - (r_pos / R)^2);
    else
        theoretical_velocities(roi_idx) = 0;
    end
end

% ==========================================================
% Blood Flow Velocity Profile Line Plot
% ==========================================================
figure('Name', 'Blood Flow Velocity Profile - Full Diameter Measurement', 'Position', [100, 100, 1200, 800]);

% Plot theoretical profile (smooth parabolic curve)
x_smooth = -5:0.1:5;
theoretical_smooth = zeros(size(x_smooth));
for i = 1:length(x_smooth)
    r_pos = abs(x_smooth(i));
    if r_pos <= R
        theoretical_smooth(i) = Vmax * (1 - (r_pos / R)^2);
    else
        theoretical_smooth(i) = 0;
    end
end

plot(x_smooth, theoretical_smooth, 'k-', 'LineWidth', 3, 'DisplayName', 'Theoretical Profile');
hold on;

% Plot measured profile (line connecting measurement points) with error bars
errorbar(roi_x_positions, avg_measured_velocities, std_measured_velocities, 'r-o', 'LineWidth', 2.5, 'MarkerSize', 6, ...
         'MarkerFaceColor', 'red', 'DisplayName', 'Measured Profile ± 1σ', 'CapSize', 4);

% Mark vessel boundaries
xline(-R, 'b--', 'LineWidth', 1.5, 'Alpha', 0.7, 'DisplayName', 'Vessel Boundary');
xline(R, 'b--', 'LineWidth', 1.5, 'Alpha', 0.7, 'HandleVisibility', 'off');
text(-R-0.3, Vmax*0.9, '-R', 'FontSize', 12, 'Color', 'blue', 'FontWeight', 'bold');
text(R+0.1, Vmax*0.9, '+R', 'FontSize', 12, 'Color', 'blue', 'FontWeight', 'bold');

% Mark center point
plot(0, Vmax, 'bs', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Theoretical Center');
center_measured = avg_measured_velocities(ceil(num_roi_positions/2));
plot(0, center_measured, 'rs', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Measured Center');

grid on;
xlabel('Radial Position (mm)', 'FontSize', 14);
ylabel('Velocity (mm/s)', 'FontSize', 14);
title(sprintf('Blood Flow Velocity Profile Measurement (Vmax=%.1f mm/s, R=%.1f mm)', Vmax, R), 'FontSize', 16);
legend('Location', 'northeast', 'FontSize', 12);

% Set axis limits
xlim([-6, 6]);
ylim([0, Vmax*1.1]);

% Add error statistics and stability analysis box
rmse_value = sqrt(mean((avg_measured_velocities - theoretical_velocities).^2));
mae_value = mean(abs(avg_measured_velocities - theoretical_velocities));
center_error = abs(center_measured - Vmax) / Vmax * 100;

% Stability analysis
mean_cv = mean(cv_measured_velocities(~isnan(cv_measured_velocities) & ~isinf(cv_measured_velocities)));
max_std = max(std_measured_velocities);

% text(-5.5, Vmax*0.4, sprintf('RMSE: %.3f mm/s\nMAE: %.3f mm/s\nCenter Error: %.2f%%\nMean CV: %.3f\nMax Std: %.3f mm/s\n(%d iterations)', ...
%      rmse_value, mae_value, center_error, mean_cv, max_std, num_iterations), ...
%      'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black', 'VerticalAlignment', 'top');

hold off;

% ==========================================================
% Comprehensive Text Statistics Output
% ==========================================================

fprintf('\n======================================================================\n');
fprintf('COMPLETE VESSEL DIAMETER VELOCITY PROFILE ANALYSIS\n');
fprintf('======================================================================\n');
fprintf('Measurement Parameters:\n');
fprintf('  -> Vessel Radius (R):           %.1f mm\n', R);
fprintf('  -> Maximum Velocity (Vmax):     %.1f mm/s\n', Vmax);
fprintf('  -> Number of ROI positions:     %d\n', num_roi_positions);
fprintf('  -> ROI coverage:                %.1f mm to %.1f mm\n', roi_x_positions(1), roi_x_positions(end));
fprintf('  -> Measurement resolution:      %.1f mm\n', roi_x_positions(2) - roi_x_positions(1));
fprintf('======================================================================\n');

fprintf('\nDETAILED VELOCITY COMPARISON TABLE (Multi-Iteration Statistics):\n');
fprintf('====================================================================================\n');
fprintf('Position | Theoretical | Measured  | Std Dev  |   CV    | Abs Error | Rel Error | Quality\n');
fprintf('  (mm)   |   (mm/s)    |  (mm/s)   | (mm/s)   |         |  (mm/s)   |    (%%)    |\n');
fprintf('---------|-------------|-----------|----------|---------|-----------|-----------|----------\n');

for roi_idx = 1:num_roi_positions
    pos = roi_x_positions(roi_idx);
    theo_vel = theoretical_velocities(roi_idx);
    meas_vel = avg_measured_velocities(roi_idx);
    std_vel = std_measured_velocities(roi_idx);
    cv_vel = cv_measured_velocities(roi_idx);
    abs_error = abs(meas_vel - theo_vel);
    
    if theo_vel > 0
        rel_error = (abs_error / theo_vel) * 100;
    else
        rel_error = 0;
    end
    
    % Quality indicator based on CV and relative error
    if abs(pos) <= R
        if cv_vel < 0.05 && rel_error < 5
            quality = 'Excellent';
        elseif cv_vel < 0.10 && rel_error < 15
            quality = 'Good';
        elseif cv_vel < 0.20 && rel_error < 25
            quality = 'Fair';
        else
            quality = 'Poor';
        end
    else
        quality = 'Outside';
    end
    
    fprintf('%7.1f  |   %8.2f  |  %7.2f  |  %6.3f  |  %6.3f  |   %6.3f  |   %6.2f  | %s\n', ...
            pos, theo_vel, meas_vel, std_vel, cv_vel, abs_error, rel_error, quality);
end

fprintf('---------|-------------|-----------|----------|---------|-----------|-----------|----------\n');

% Overall statistics
fprintf('\nOVERALL PERFORMANCE STATISTICS:\n');
fprintf('======================================================================\n');
fprintf('Root Mean Square Error (RMSE):     %.4f mm/s\n', rmse_value);
fprintf('Mean Absolute Error (MAE):         %.4f mm/s\n', mae_value);
fprintf('Maximum Absolute Error:            %.4f mm/s\n', max(abs(avg_measured_velocities - theoretical_velocities)));

% Center point analysis
center_idx = ceil(num_roi_positions/2);
fprintf('\nCENTER POINT ANALYSIS:\n');
fprintf('  -> Position:                     %.1f mm\n', roi_x_positions(center_idx));
fprintf('  -> Theoretical Center Velocity:  %.3f mm/s\n', theoretical_velocities(center_idx));
fprintf('  -> Measured Center Velocity:     %.3f mm/s\n', avg_measured_velocities(center_idx));
fprintf('  -> Center Absolute Error:        %.4f mm/s\n', abs(avg_measured_velocities(center_idx) - theoretical_velocities(center_idx)));
fprintf('  -> Center Relative Error:        %.2f%%\n', center_error);

% Vessel wall analysis
wall_positions = find(abs(roi_x_positions) <= R);
if ~isempty(wall_positions)
    vessel_rmse = sqrt(mean((avg_measured_velocities(wall_positions) - theoretical_velocities(wall_positions)).^2));
    vessel_mae = mean(abs(avg_measured_velocities(wall_positions) - theoretical_velocities(wall_positions)));
    fprintf('\nVESSEL INTERIOR ANALYSIS (|r| <= R):\n');
    fprintf('  -> Interior RMSE:                %.4f mm/s\n', vessel_rmse);
    fprintf('  -> Interior MAE:                 %.4f mm/s\n', vessel_mae);
    fprintf('  -> Number of interior points:   %d\n', length(wall_positions));
end

% Profile shape analysis
correlation_coeff = corrcoef(theoretical_velocities, avg_measured_velocities);
profile_correlation = correlation_coeff(1,2);

fprintf('\nPROFILE SHAPE ANALYSIS:\n');
fprintf('  -> Profile Correlation (R²):     %.4f\n', profile_correlation^2);
fprintf('  -> Profile Correlation (r):      %.4f\n', profile_correlation);

if profile_correlation^2 > 0.95
    profile_quality = 'Excellent parabolic fit';
elseif profile_correlation^2 > 0.85
    profile_quality = 'Good parabolic fit';
else
    profile_quality = 'Fair parabolic fit';
end
fprintf('  -> Profile Quality:              %s\n', profile_quality);

% Multi-iteration stability analysis
fprintf('\nMULTI-ITERATION STABILITY ANALYSIS:\n');
fprintf('======================================================================\n');
fprintf('  -> Number of iterations:         %d\n', num_iterations);
fprintf('  -> Overall Mean CV:              %.4f\n', mean_cv);
fprintf('  -> Overall Max Std:              %.4f mm/s\n', max_std);

% Find most and least stable positions
[~, most_stable_idx] = min(cv_measured_velocities(~isnan(cv_measured_velocities) & ~isinf(cv_measured_velocities)));
[~, least_stable_idx] = max(cv_measured_velocities(~isnan(cv_measured_velocities) & ~isinf(cv_measured_velocities)));
valid_indices = find(~isnan(cv_measured_velocities) & ~isinf(cv_measured_velocities));

if ~isempty(valid_indices)
    most_stable_global_idx = valid_indices(most_stable_idx);
    least_stable_global_idx = valid_indices(least_stable_idx);
    
    fprintf('  -> Most stable position:         %.1f mm (CV = %.4f)\n', ...
            roi_x_positions(most_stable_global_idx), cv_measured_velocities(most_stable_global_idx));
    fprintf('  -> Least stable position:        %.1f mm (CV = %.4f)\n', ...
            roi_x_positions(least_stable_global_idx), cv_measured_velocities(least_stable_global_idx));
end

% Quality distribution
vessel_indices = find(abs(roi_x_positions) <= R);
if ~isempty(vessel_indices)
    excellent_count = sum(cv_measured_velocities(vessel_indices) < 0.05);
    good_count = sum(cv_measured_velocities(vessel_indices) >= 0.05 & cv_measured_velocities(vessel_indices) < 0.10);
    fair_count = sum(cv_measured_velocities(vessel_indices) >= 0.10 & cv_measured_velocities(vessel_indices) < 0.20);
    poor_count = sum(cv_measured_velocities(vessel_indices) >= 0.20);
    
    fprintf('  -> Vessel interior quality distribution:\n');
    fprintf('     * Excellent (CV < 0.05):      %d positions\n', excellent_count);
    fprintf('     * Good (0.05 ≤ CV < 0.10):    %d positions\n', good_count);
    fprintf('     * Fair (0.10 ≤ CV < 0.20):    %d positions\n', fair_count);
    fprintf('     * Poor (CV ≥ 0.20):           %d positions\n', poor_count);
end

fprintf('======================================================================\n');

% Save complete results
frame_rate_hz = 1/dt;
filename = sprintf('FullDiameter_Profile_Vmax_%.1f_FrameRate_%dHz_ROI_%d_positions_results.mat', ...
                   Vmax, frame_rate_hz, num_roi_positions);
save(filename, 'roi_x_positions', 'theoretical_velocities', 'avg_measured_velocities', ...
     'std_measured_velocities', 'cv_measured_velocities', 'measured_velocities_profile', 'all_roi_cc', ...
     'Vmax', 'R', 'dt', 'y_distance', 'num_iterations', 'rmse_value', 'mae_value', 'center_error', ...
     'profile_correlation', 'mean_cv', 'max_std');
fprintf('Complete results saved to: %s\n', filename);
fprintf('======================================================================\n');