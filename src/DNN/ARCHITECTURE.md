# Neural Network Architecture for Relative Geologic Time (RGT) Prediction

## Overview

This document describes a 3D U-Net based neural network designed for predicting Relative Geologic Time (RGT) from seismic volume data. The network takes seismic data as input and produces a continuous RGT volume as output.

## Core Principles

### Problem Definition

Relative Geologic Time (RGT) estimation is a key task in seismic interpretation. The goal is to assign a continuous time value to each point in a seismic volume, where points on the same geologic horizon share the same RGT value. This creates a structurally consistent temporal ordering of seismic events.

### Key Design Decisions

1. **3D Convolution**: Seismic data is inherently 3D (inline × crossline × time/depth), requiring volumetric convolutions to capture spatial relationships.

2. **U-Net Architecture**: The encoder-decoder structure with skip connections preserves both local details and global context, essential for tracking seismic horizons.

3. **Residual Learning**: ResBlocks with skip connections enable training of deep networks by addressing vanishing gradients.

4. **Multi-Scale SSIM Loss**: Structural Similarity Index Measure at multiple scales captures both fine details and coarse structural relationships.

5. **Mixed Precision Training**: BFloat16 precision reduces memory usage while maintaining numerical stability.

---

## Network Architecture: UNet3D

### Architecture Overview

```
Input: (batch, 1, D, H, W)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     ENCODER PATH                            │
│  enc1: 16 channels  → pool → enc2: 32 channels  → pool →   │
│  enc3: 64 channels  → pool → enc4: 128 channels → pool →   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │    BOTTLENECK       │
              │    256 channels     │
              └─────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     DECODER PATH                            │
│  upconv4 + enc4 → dec4: 128 channels                        │
│  upconv3 + enc3 → dec3: 64 channels                         │
│  upconv2 + enc2 → dec2: 32 channels                         │
│  upconv1 + enc1 → dec1: 16 channels                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              outconv: 1 channel
                          │
                          ▼
              smooth_conv: 1 channel (5×5×5 kernel)
                          │
                          ▼
Output: (batch, 1, D, H, W)
```

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `in_channels` | 1 | Single channel seismic input |
| `out_channels` | 1 | Single channel RGT output |
| `init_features` | 16 | Base number of feature maps |
| `smoothing_kernel_size` | 5 | Post-processing smoothing kernel |

---

## Detailed Layer Description

### Encoder Path

The encoder progressively downsamples the input while increasing feature depth.

#### Encoder Block 1 (enc1)
- **Input**: `(batch, 1, D, H, W)`
- **Output**: `(batch, 16, D, H, W)`
- **Operation**: ResBlock(1 → 16)
- Contains downsample branch since in_channels ≠ features

#### Pool 1
- **Operation**: `nn.AvgPool3d(kernel_size=2, stride=2)`
- **Output**: `(batch, 16, D/2, H/2, W/2)`

#### Encoder Block 2 (enc2)
- **Output**: `(batch, 32, D/2, H/2, W/2)`
- **Operation**: ResBlock(16 → 32)

#### Pool 2
- **Output**: `(batch, 32, D/4, H/4, W/4)`

#### Encoder Block 3 (enc3)
- **Output**: `(batch, 64, D/4, H/4, W/4)`
- **Operation**: ResBlock(32 → 64)

#### Pool 3
- **Output**: `(batch, 64, D/8, H/8, W/8)`

#### Encoder Block 4 (enc4)
- **Output**: `(batch, 128, D/8, H/8, W/8)`
- **Operation**: ResBlock(64 → 128)

#### Pool 4
- **Output**: `(batch, 128, D/16, H/16, W/16)`

### Bottleneck

The bottleneck captures the most abstract, global features.

#### Bottleneck Block
- **Input**: `(batch, 128, D/16, H/16, W/16)`
- **Output**: `(batch, 256, D/16, H/16, W/16)`
- **Operation**: ResBlock(128 → 256)
- Deepest point in the network with highest receptive field

### Decoder Path

The decoder upsamples features while concatenating with encoder outputs via skip connections.

#### Upconv 4
```
Sequential:
  - Upsample(scale_factor=2, mode='trilinear', align_corners=False)
  - Conv3d(256 → 128, kernel=3, padding=1, bias=False)
  - GroupNorm(8 groups, 128 channels)
  - ReLU(inplace=True)
```
- **Output after upsample**: `(batch, 128, D/8, H/8, W/8)`

#### Decoder Block 4 (dec4)
- **Concatenation**: `torch.cat([upconv4_output, enc4_output], dim=1)`
- **Input channels**: 128 + 128 = 256
- **Output**: `(batch, 128, D/8, H/8, W/8)`
- **Operation**: ResBlock(256 → 128)

#### Upconv 3
```
Sequential:
  - Upsample(scale_factor=2, mode='trilinear', align_corners=False)
  - Conv3d(128 → 64, kernel=3, padding=1, bias=False)
  - GroupNorm(8 groups, 64 channels)
  - ReLU(inplace=True)
```

#### Decoder Block 3 (dec3)
- **Concatenation**: upconv3 + enc3 (128 channels total)
- **Output**: `(batch, 64, D/4, H/4, W/4)`

#### Upconv 2
```
Sequential:
  - Upsample(scale_factor=2, mode='trilinear', align_corners=False)
  - Conv3d(64 → 32, kernel=3, padding=1, bias=False)
  - GroupNorm(8 groups, 32 channels)
  - ReLU(inplace=True)
```

#### Decoder Block 2 (dec2)
- **Concatenation**: upconv2 + enc2 (64 channels total)
- **Output**: `(batch, 32, D/2, H/2, W/2)`

#### Upconv 1
```
Sequential:
  - Upsample(scale_factor=2, mode='trilinear', align_corners=False)
  - Conv3d(32 → 16, kernel=3, padding=1, bias=False)
  - GroupNorm(8 groups, 16 channels)
  - ReLU(inplace=True)
```

#### Decoder Block 1 (dec1)
- **Concatenation**: upconv1 + enc1 (32 channels total)
- **Output**: `(batch, 16, D, H, W)`

### Output Layers

#### Output Convolution (outconv)
- **Operation**: `nn.Conv3d(16 → 1, kernel_size=1)`
- **Purpose**: Project features to single-channel RGT prediction
- **Output**: `(batch, 1, D, H, W)`

#### Smoothing Convolution (smooth_conv)
```
Sequential:
  - ReplicationPad3d(2)  # padding for kernel_size=5
  - Conv3d(1 → 1, kernel_size=5, padding=0, bias=False)
```
- **Purpose**: Spatial smoothing of RGT output for physically plausible continuous surfaces
- **Kernel size**: 5×5×5 (configurable via `smoothing_kernel_size`)
- **Padding**: Replication padding to handle boundaries
- **Note**: No activation function; output is continuous RGT values

---

## ResBlock Architecture

Each encoder/decoder block is a ResBlock with the following structure:

```
Input (in_channels)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     │
┌─────────────────┐                       │
│ Conv3d(3×3×3)   │ (bias=False)          │
│ out: features   │                       │
└────────┬────────┘                       │
         ▼                                │
┌─────────────────┐                       │
│ GroupNorm       │ (8 groups)            │
└────────┬────────┘                       │
         ▼                                │
┌─────────────────┐                       │
│ ReLU            │                        │
└────────┬────────┘                       │
         ▼                                │
┌─────────────────┐                       │
│ Conv3d(3×3×3)   │ (bias=False)          │
│ out: features   │                       │
└────────┬────────┘                       │
         ▼                                │
┌─────────────────┐                       │
│ GroupNorm       │ (8 groups)            │
└────────┬────────┘                       │
         │                                │
         ▼                                │
    (+ Add)◄──────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ ReLU            │
└─────────────────┘
         │
         ▼
Output (features)
```

### Skip Connection (Downsample Branch)

When `in_channels != features`, a projection shortcut is used:

```
Downsample = Sequential(
    Conv3d(in_channels → features, kernel_size=1, bias=False),
    GroupNorm(8 groups, features)
)
```

This ensures dimensional compatibility for the residual addition.

---

## Weight Initialization

The network uses **Kaiming Normal Initialization** (also known as He initialization) for convolution layers:

### Convolution Layers
```python
nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.1)
```

- **Mode**: `fan_out` - preserves magnitude of the variance in the backward pass
- **Nonlinearity**: `leaky_relu` with `a=0.1` (negative slope)
- **Bias**: Initialized to zero if present

### Normalization Layers (GroupNorm/BatchNorm)
```python
nn.init.constant_(m.weight, 1)  # gamma
nn.init.constant_(m.bias, 0)    # beta
```

### Rationale

Kaiming initialization is chosen because:
1. It accounts for the non-linearity (ReLU) used in the network
2. It prevents vanishing/exploding gradients in deep networks
3. `fan_out` mode is appropriate for convolution layers where output features matter

---

## Data Augmentation

Data augmentation is applied during training to improve generalization. Augmentations are applied on-the-fly during each training iteration.

### Geometric Transformations

#### Horizontal Flip 1 (Inline Direction)
```python
def HorizontalFlip1(dat):
    return torch.flip(dat, dims=[3])  # Flip along W axis (inline)
```

#### Horizontal Flip 2 (Crossline Direction)
```python
def HorizontalFlip2(dat):
    return torch.flip(dat, dims=[4])  # Flip along H axis (crossline)
```

#### Vertical Flip (Time/Depth Direction)
```python
def VerticalFlip(dat):
    return torch.flip(dat, dims=[2])  # Flip along D axis (time/depth)
```

For RGT labels, vertical flip requires special handling:
```python
def VerticalFlip_reverse(dat):
    return -torch.flip(dat, dims=[2])  # Negate after flip
```

**Why negate?** RGT values increase with depth/time. Flipping vertically reverses the temporal ordering, so values must be negated to maintain consistency.

### Noise Injection

Gaussian noise is added to seismic data for robustness:

```python
noise_std = 0.01 * seismic.std()
noise = torch.randn_like(seismic) * noise_std
seismic = seismic + noise
```

- **Standard deviation**: 1% of the data's standard deviation
- **Purpose**: Simulate acquisition noise and improve robustness

### Augmentation Pipeline

During training (when `data_augmentation=True`):

```python
# 4× batch expansion through augmentation
seismic = torch.cat([
    seismic,                    # Original
    HorizontalFlip1(seismic),   # Flip inline
    HorizontalFlip2(seismic),   # Flip crossline
    VerticalFlip(seismic)       # Flip depth
], dim=0)

rgt = torch.cat([
    rgt,                        # Original
    HorizontalFlip1(rgt),       # Flip inline
    HorizontalFlip2(rgt),       # Flip crossline
    VerticalFlip_reverse(rgt)   # Flip depth + negate
], dim=0)

# Add noise
seismic = seismic + noise
```

This effectively quadruples the batch size and introduces variability.

---

## Loss Function: Multi-Scale 3D SSIM

### Structural Similarity Index (SSIM)

SSIM measures perceptual similarity between two images/volumes by considering:
- **Luminance** (mean intensity)
- **Contrast** (variance)
- **Structure** (correlation)

For 3D volumes, SSIM is computed as:

$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Where:
- $\mu_x, \mu_y$: Local means (computed via 3D Gaussian convolution)
- $\sigma_x^2, \sigma_y^2$: Local variances
- $\sigma_{xy}$: Local covariance
- $C_1 = (k_1 \cdot L)^2$, $C_2 = (k_2 \cdot L)^2$: Stabilization constants

### Multi-Scale SSIM (MS-SSIM)

MS-SSIM computes SSIM at multiple resolution scales:

```python
weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # 5 scales
```

**Process**:
1. Compute SSIM and contrast-structure (CS) at each scale
2. Downsample by 2× using average pooling
3. Combine across scales:

$$\text{MS-SSIM} = \prod_{j=1}^{M-1} CS_j^{w_j} \cdot SSIM_M^{w_M}$$

### Implementation Details

#### 3D Gaussian Kernel
```python
def _fspecial_gaussian3d(size, channel, sigma):
    # Creates a 3D Gaussian kernel for SSIM computation
    # Default: size=7, sigma=1.5
    coords = -(coords**2) / (2.0 * sigma**2)
    grid = coords.view(1, -1, 1) + coords.view(-1, 1, 1) + coords.view(1, 1, -1)
    kernel = grid.softmax(-1)  # Normalized to sum to 1
```

#### SSIM3DLoss Class
```python
class SSIM3DLoss(nn.Module):
    def __init__(self, max_val=1.0):
        self.ssim = MultiScaleSSIMLoss3d(channel=1)
        self.max_val = max_val  # Maximum possible value in data

    def forward(self, output, target):
        return 1 - self.ssim(output, target, max_val=self.max_val)
```

**Training default**: `max_val=7.7236` (configured in training script)

### Why MS-SSIM for RGT?

1. **Structural coherence**: RGT should preserve seismic horizon structures
2. **Multi-scale**: Captures both local (fine horizons) and global (major surfaces) features
3. **Perceptual**: Better matches human interpretation than pixel-wise losses (MSE)
4. **Gradient-friendly**: Provides meaningful gradients even for small differences

---

## Training Process

### Optimizer

**Adam Optimizer**
```python
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

Default hyperparameters:
- Learning rate: `0.001` (from training script)
- Weight decay: `0.01`
- Betas: `(0.9, 0.999)` (default)

### Learning Rate Scheduling

**ReduceLROnPlateau**
```python
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)
```

- Reduces LR by factor of 0.5 after 5 epochs without validation improvement
- Monitors validation loss

### Gradient Accumulation

Supports effective batch sizes larger than GPU memory allows:

```python
# Accumulate gradients over multiple batches
loss = loss * actual_batch_size / target_batch_size
loss.backward()

if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

Default: `accumulation_steps=4`

### Gradient Clipping

Prevents exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

Default: `grad_clip=1.0`

### Mixed Precision

Model is cast to BFloat16:
```python
self.model = model.to(device).bfloat16()
```

Benefits:
- Reduced memory usage (~50% reduction)
- Faster computation on modern GPUs
- Better numerical stability than FP16

### Early Stopping

Monitors validation loss with patience:
```python
patience = 10  # Default (can be set to 20)
if val_loss < best_val_loss:
    patience_counter = 0
    save_best_model()
else:
    patience_counter += 1
    if patience_counter >= patience:
        early_stop()
```

### Checkpointing

- **Best model**: Saved when validation loss improves
- **Periodic checkpoints**: Every N epochs (configurable)
- **Final model**: Saved at training completion

Checkpoint format:
```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": train_loss,
    "val_loss": val_loss
}
```

---

## Dataset Handling

### Data Structure

```
dataroot/
├── seis/           # Seismic data files
│   ├── 0.dat
│   ├── 1.dat
│   └── ...
└── rgt/            # RGT labels (for training)
    ├── 0.dat
    ├── 1.dat
    └── ...
```

### Data Format

- **File format**: Raw binary (`.dat` files)
- **Data type**: `np.float32`
- **Shape**: `(D, H, W, C)` stored flattened, reshaped to `(C, W, H, D)` for PyTorch

### SeismicDataset Class

```python
class SeismicDataset(data.Dataset):
    def __init__(self, root_dir, list_IDs, shape=(128, 128, 128, 1), only_load_input=False):
        self.transform = transforms.Compose([
            Reshape(shape),
            ToTensor(),  # Transposes to (C, W, H, D)
        ])
    
    def __getitem__(self, index):
        # Load seismic
        X = np.fromfile(seis_path, dtype=np.float32)
        X = self.transform(X)
        X = mea_std_norm(X)  # Normalize
        X = torch.from_numpy(X).bfloat16()
        
        # Load RGT (if training)
        Y = np.fromfile(rgt_path, dtype=np.float32)
        Y = self.transform(Y)
        Y = mea_std_norm(Y)
        Y = torch.from_numpy(Y).bfloat16()
        
        return X, Y
```

### Normalization

Mean-Standard deviation normalization:
```python
def mea_std_norm(x):
    return (x - np.mean(x)) / np.std(x)
```

Applied to both seismic input and RGT labels independently per sample.

---

## Inference Pipeline

### Single Chunk Prediction

The `predict_chunk.py` script handles inference on pre-chunked volumes:

```python
model = UNet3D(in_channels=1, out_channels=1, init_features=16)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    prediction = model(seismic_data)
```

Output is saved as binary `.dat` files with `_rgt.dat` suffix.

### Large Volume Inference

The `VolumeMerger` class handles prediction on large seismic volumes that exceed GPU memory:

#### Workflow

1. **Slice Volume**: Divide into overlapping chunks
2. **Predict**: Run model on each chunk
3. **Merge**: Combine predictions with smooth blending

#### Chunk Slicing

```python
def slice_volume(self, volume):
    # Default: 128×128×128 chunks with 64-voxel stride (50% overlap)
    # Pads volume to ensure full coverage
```

#### Vertical Merging

Handles depth direction merging with:
- **Mean shift correction**: Aligns overlapping predictions
- **Linear blending**: Smooth transitions using weighted average

```python
# Weight for blending
weights = np.linspace(1, 0, overlap_size)  # Linear falloff
merged = merged * weights + new_chunk * (1 - weights)
```

#### Horizontal Merging

Handles inline/crossline directions with:
- **Tikhonov regularization**: Robust linear transformation fitting
- **Scale/bias correction**: Adjusts predictions to match overlapping regions

```python
def _fit_linear_tikhonov(x, y, lambda_reg=0.01):
    # Fit y = a*x + b with regularization
    # Returns (scale, bias) coefficients
```

#### Full Merge Process

```python
def merge_all_predictions(self, predictions, positions):
    # Step 1: Vertical merge (depth direction)
    chunks = vertical_merge(predictions, positions)
    
    # Step 2: Horizontal merge along axis 0 (inline)
    chunks = horizontal_merge(chunks, positions, axis=0)
    
    # Step 3: Horizontal merge along axis 1 (crossline)
    chunks = horizontal_merge(chunks, positions, axis=1)
    
    # Step 4: Crop to original shape
    return result[:original_shape]
```

---

## Model Parameters

With default `init_features=16`:

| Layer | Channels | Feature Map Size* |
|-------|----------|-------------------|
| Input | 1 | D×H×W |
| enc1 | 16 | D×H×W |
| enc2 | 32 | D/2×H/2×W/2 |
| enc3 | 64 | D/4×H/4×W/4 |
| enc4 | 128 | D/8×H/8×W/8 |
| bottleneck | 256 | D/16×H/16×W/16 |
| dec4 | 128 | D/8×H/8×W/8 |
| dec3 | 64 | D/4×H/4×W/4 |
| dec2 | 32 | D/2×H/2×W/2 |
| dec1 | 16 | D×H×W |
| Output | 1 | D×H×W |

*Assuming input size D×H×W

### Parameter Count

Total trainable parameters can be computed via:
```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

The network is designed to be lightweight while maintaining expressive power for seismic interpretation.

---

## Summary of Key Features

1. **3D U-Net with ResBlocks**: Encoder-decoder architecture for volumetric segmentation
2. **Skip Connections**: Preserve spatial details across resolution scales
3. **Kaiming Initialization**: Proper weight initialization for ReLU networks
4. **GroupNorm**: Stable normalization with 8 groups per layer
5. **Trilinear Upsampling**: Smooth interpolation in decoder
6. **Smoothing Convolution**: Post-processing for physically plausible RGT surfaces
7. **Multi-Scale SSIM Loss**: Perceptual loss capturing structural similarity
8. **Data Augmentation**: Geometric transforms + noise injection
9. **Mixed Precision**: BFloat16 for memory efficiency
10. **Gradient Accumulation**: Support for large effective batch sizes
11. **Large Volume Support**: Overlapping chunk prediction with smooth merging