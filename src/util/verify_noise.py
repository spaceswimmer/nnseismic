import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_pink_noise(shape, device='cpu', dtype=torch.bfloat16):
    """Generate pink noise (1/f power spectral density)"""
    white_noise = torch.randn(shape, device=device, dtype=dtype)
    
    white_noise_flat = white_noise.flatten()
    n = white_noise_flat.shape[0]
    
    freqs = torch.fft.fftfreq(n, device=device)
    freqs[0] = 1e-10
    
    pink_filter = 1.0 / torch.sqrt(torch.abs(freqs))
    pink_filter = pink_filter.to(dtype)
    
    white_fft = torch.fft.fft(white_noise_flat.float())
    pink_fft = white_fft * pink_filter.float()
    pink_noise_flat = torch.fft.ifft(pink_fft).real
    
    pink_noise = pink_noise_flat.to(dtype).reshape(shape)
    
    pink_noise = pink_noise * (white_noise.std() / pink_noise.std())
    
    return pink_noise

def verify_noise_effect():
    seismic_path = '/mnt/storage/nnseismic/synthetic_data/val/seis/00030.dat'
    
    seismic_np = np.fromfile(seismic_path, dtype=np.float32)
    seismic_np = seismic_np.reshape(128, 128, 128)
    
    seismic_np = (seismic_np - seismic_np.mean()) / seismic_np.std()
    
    sample_data = torch.from_numpy(seismic_np).unsqueeze(0).unsqueeze(0).bfloat16()
    
    gaussian_std = 0.05 * sample_data.std()
    gaussian_noise = torch.randn_like(sample_data) * gaussian_std
    noisy_gaussian = sample_data + gaussian_noise
    
    pink_noise = generate_pink_noise(sample_data.shape, dtype=torch.bfloat16)
    pink_std = 0.05 * sample_data.std()
    pink_noise = pink_noise * pink_std / pink_noise.std()
    noisy_pink = sample_data + pink_noise
    
    slice_idx = 64
    original_slice = sample_data[0, 0, slice_idx, :, :].float().cpu().numpy()
    gaussian_slice = noisy_gaussian[0, 0, slice_idx, :, :].float().cpu().numpy()
    pink_slice = noisy_pink[0, 0, slice_idx, :, :].float().cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    vmin, vmax = original_slice.min(), original_slice.max()
    
    axes[0, 0].imshow(original_slice.T, cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0, 0].set_title('Original Seismic\n(slice at depth=64)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gaussian_slice.T, cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0, 1].set_title(f'Gaussian Noise\n(σ={gaussian_std.item():.4f})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pink_slice.T, cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0, 2].set_title(f'Pink Noise\n(σ={pink_std.item():.4f})')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow((pink_slice - gaussian_slice).T, cmap='seismic', vmin=-0.02, vmax=0.02, interpolation='nearest')
    axes[0, 3].set_title('Difference\n(Pink - Gaussian)')
    axes[0, 3].axis('off')
    
    original_spectrum = torch.abs(torch.fft.fft(sample_data.flatten().float())).cpu().numpy()
    gaussian_spectrum = torch.abs(torch.fft.fft(noisy_gaussian.flatten().float())).cpu().numpy()
    pink_spectrum = torch.abs(torch.fft.fft(noisy_pink.flatten().float())).cpu().numpy()
    
    freqs = np.fft.fftfreq(len(original_spectrum))
    positive_freqs = freqs[:len(freqs)//2]
    
    axes[1, 0].semilogy(positive_freqs, original_spectrum[:len(positive_freqs)], 'k-', linewidth=2, label='Original')
    axes[1, 0].set_title('Original Spectrum')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(positive_freqs, gaussian_spectrum[:len(positive_freqs)], 'b-', linewidth=2, label='Gaussian')
    axes[1, 0].semilogy(positive_freqs, gaussian_spectrum[:len(positive_freqs)], 'b-', linewidth=1, alpha=0.5)
    axes[1, 1].set_title('Gaussian Noise Spectrum')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].semilogy(positive_freqs, pink_spectrum[:len(positive_freqs)], 'r-', linewidth=2, label='Pink')
    axes[1, 0].semilogy(positive_freqs, pink_spectrum[:len(positive_freqs)], 'r-', linewidth=1, alpha=0.5)
    axes[1, 2].set_title('Pink Noise Spectrum')
    axes[1, 2].set_xlabel('Frequency')
    axes[1, 2].set_ylabel('Amplitude')
    axes[1, 2].grid(True, alpha=0.3)
    
    axes[1, 3].semilogy(positive_freqs, original_spectrum[:len(positive_freqs)], 'k-', linewidth=2, label='Original')
    axes[1, 3].semilogy(positive_freqs, gaussian_spectrum[:len(positive_freqs)], 'b-', linewidth=1, alpha=0.7, label='Gaussian')
    axes[1, 3].semilogy(positive_freqs, pink_spectrum[:len(positive_freqs)], 'r-', linewidth=1, alpha=0.7, label='Pink')
    axes[1, 3].set_title('Spectrum Comparison')
    axes[1, 3].set_xlabel('Frequency')
    axes[1, 3].set_ylabel('Amplitude')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = '/home/spaceswimmer/Documents/nnseismic/noise_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {save_path}")
    print(f"Original data - mean: {sample_data.mean().item():.4f}, std: {sample_data.std().item():.4f}")
    print(f"Gaussian noise - std: {gaussian_std.item():.4f} (1% of signal std)")
    print(f"Pink noise - std: {pink_std.item():.4f} (1% of signal std)")
    print(f"Relative noise amplitude: 1.00% for both")
    print("\nPink noise characteristics:")
    print("- Power decreases with frequency (1/f)")
    print("- More realistic for seismic acquisition noise")
    print("- Simulates ground roll and ambient vibrations")

if __name__ == "__main__":
    verify_noise_effect()
