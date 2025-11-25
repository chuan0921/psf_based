import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def save_bmode_image(image, x, z, filename, title='B-mode Image'):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, extent=[x.min(), x.max(), z.max(), z.min()],
               cmap='gray', aspect='auto')
    plt.colorbar(label='dB')
    plt.xlabel('Lateral (mm)')
    plt.ylabel('Axial (mm)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_bmode_image(image, x, z, title='B-mode Image'):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, extent=[x.min(), x.max(), z.max(), z.min()],
               cmap='gray', aspect='auto')
    plt.colorbar(label='dB')
    plt.xlabel('Lateral (mm)')
    plt.ylabel('Axial (mm)')
    plt.title(title)
    plt.tight_layout()

def plot_rayleigh_distribution(envelope_data):
    env_vals = envelope_data.flatten()
    sigma_env = env_vals.mean() / np.sqrt(np.pi/2)

    plt.figure(figsize=(6, 4))
    plt.hist(env_vals, bins=50, density=True, color='gray', alpha=0.7, label='Empirical')

    x_vals = np.linspace(0, env_vals.max(), 100)
    y_theo = (x_vals / sigma_env**2) * np.exp(-x_vals**2 / (2*sigma_env**2))
    plt.plot(x_vals, y_theo, 'r-', linewidth=2, label='Rayleigh Fit')

    plt.xlabel('Normalized Envelope')
    plt.ylabel('PDF')
    plt.title('Speckle Envelope Distribution')
    plt.legend()
    plt.tight_layout()

def plot_velocity_profile(roi_x_positions, theoretical_vel, measured_vel, std_vel, R, Vmax):
    x_smooth = np.linspace(-5, 5, 100)
    r_smooth = np.abs(x_smooth)
    theo_smooth = np.where(r_smooth <= R, Vmax * (1 - (r_smooth/R)**2), 0)

    plt.figure(figsize=(12, 8))
    plt.plot(x_smooth, theo_smooth, 'k-', linewidth=3, label='Theoretical')
    plt.errorbar(roi_x_positions, measured_vel, yerr=std_vel,
                 fmt='ro-', linewidth=2.5, markersize=6, capsize=4,
                 label='Measured ± 1σ')

    plt.axvline(-R, color='b', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.axvline(R, color='b', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(-R-0.3, Vmax*0.9, '-R', fontsize=12, color='blue', fontweight='bold')
    plt.text(R+0.1, Vmax*0.9, '+R', fontsize=12, color='blue', fontweight='bold')

    plt.grid(True)
    plt.xlabel('Radial Position (mm)', fontsize=14)
    plt.ylabel('Velocity (mm/s)', fontsize=14)
    plt.title(f'Blood Flow Velocity Profile (Vmax={Vmax} mm/s, R={R} mm)', fontsize=16)
    plt.legend(fontsize=12)
    plt.xlim([-6, 6])
    plt.ylim([0, Vmax*1.1])
    plt.tight_layout()

def create_animation(frames, x, z, vessel_params, roi_params, filename='animation.gif'):
    import os
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame_idx):
        ax.clear()
        ax.imshow(frames[frame_idx], extent=[x.min(), x.max(), z.max(), z.min()],
                  cmap='gray', aspect='auto')
        ax.set_xlabel('Lateral (mm)')
        ax.set_ylabel('Axial (mm)')
        ax.set_title(f'Frame {frame_idx+1}')

        vessel_circle = plt.Circle((vessel_params['cx'], vessel_params['cz']),
                                   vessel_params['R'], color='r', fill=False, linewidth=2)
        ax.add_patch(vessel_circle)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100)

    try:
        anim.save(filename, writer='pillow', fps=10)
        print(f"Animation saved as: {filename}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Saving individual frames instead...")
        frame_dir = os.path.dirname(filename) if os.path.dirname(filename) else '.'
        for i, frame in enumerate(frames):
            frame_filename = os.path.join(frame_dir, f"frame_{i:03d}.png")
            save_bmode_image(frame, x, z, frame_filename, f'Frame {i}')

    plt.close()
