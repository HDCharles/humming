# Benchmark Comparison Tables

This document explains what is included in each benchmark comparison table.

## Usage

### Generate tables for a specific benchmark directory:
```bash
python3 benchmarks/generate_comparison_tables.py \
    --result_dir benchmarks/results/h20/moe_e256_k8_n7168_k256_g128x128 \
    --output comparison_tables.md
```

### Generate tables for all benchmarks:
```bash
bash benchmarks/generate_all_tables.sh
```

## Table Categories

The script automatically groups benchmark results into separate tables based on:
1. **Activation precision** (16-bit, 8-bit, 4-bit)
2. **Weight type** (integer vs floating point)
3. **Special formats** (mxfp4, nvfp4)

### 1) Integer Weights with FP8/INT8 Activations (wNa8)

**Table Title**: "Integer Weights with FP8/INT8"

**Included Columns**:
- `triton-w8a8` - Triton baseline (if available)
- `cutlass-w8a8` - CUTLASS baseline (if available)
- `humming-w8a8` - Humming with 8-bit integer weights
- `humming-w7a8` - Humming with 7-bit integer weights
- `humming-w6a8` - Humming with 6-bit integer weights
- `humming-w5a8` - Humming with 5-bit integer weights
- `humming-w4a8` - Humming with 4-bit integer weights
- `humming-w3a8` - Humming with 3-bit integer weights
- `humming-w2a8` - Humming with 2-bit integer weights
- `humming-w1a8` - Humming with 1-bit integer weights

**Rows**: Batch sizes (shape_m): 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384

**Metric**: TFLOPS (Tera Floating Point Operations Per Second)

---

### 2) Floating Point Weights with FP16 Activations (wfpNa16)

**Table Title**: "Floating Point Weights with FP16"

**Included Columns**:
- `torch-w16a16` - PyTorch FP16 baseline (if available)
- `marlin-w8a16` - Marlin FP8 baseline (if available)
- `marlin-w4a16` - Marlin FP4 baseline (if available)
- `humming-w8a16` - Humming with 8-bit FP weights (e.g., float8e4m3)
- `humming-w7a16` - Humming with 7-bit FP weights (e.g., float7e4m2)
- `humming-w6a16` - Humming with 6-bit FP weights (e.g., float6e3m2)
- `humming-w5a16` - Humming with 5-bit FP weights (e.g., float5e2m2)
- `humming-w4a16` - Humming with 4-bit FP weights (e.g., float4e2m1)

**Special Formats** (if hardware supports):
- `humming-w4a16-mxfp4` - mxfp4 weights with FP16 activations (Hopper H100/H200/H20 or newer)
- `humming-w4a16-nvfp4` - nvfp4 weights with FP16 activations (Blackwell B100/B200 only)

**Rows**: Same batch sizes as above

**Metric**: TFLOPS

**Hardware Requirements**:
- **mxfp4**: Requires Hopper (SM90+) - H100, H200, H20
- **nvfp4**: Requires Blackwell (SM100+) - B100, B200, GB200

---

### 3) Floating Point Weights with FP8 Activations (wfpNa8)

**Table Title**: "Floating Point Weights with FP8/INT8"

**Included Columns**:
- `triton-w8a8` - Triton FP8 baseline (if available)
- `cutlass-w8a8` - CUTLASS FP8 baseline (if available)
- `humming-w8a8` - Humming with 8-bit FP weights (e.g., float8e4m3)
- `humming-w7a8` - Humming with 7-bit FP weights (e.g., float7e4m2)
- `humming-w6a8` - Humming with 6-bit FP weights (e.g., float6e3m2)
- `humming-w5a8` - Humming with 5-bit FP weights (e.g., float5e2m2)
- `humming-w4a8` - Humming with 4-bit FP weights (e.g., float4e2m1)

**Special Formats** (if hardware supports):
- `humming-w4a4-nvfp4` - nvfp4 weights with nvfp4 activations (Blackwell B100/B200 only)

**Rows**: Same batch sizes as above

**Metric**: TFLOPS

**Hardware Requirements**:
- **nvfp4**: Requires Blackwell (SM100+) - B100, B200, GB200

---

## Additional Table Information

Each table includes:

### Header Section:
- **Device name** (e.g., "NVIDIA H20", "NVIDIA A100")
- **Problem configuration**:
  - For MOE: shape, number of experts, top-k, block size
  - For Dense: shape dimensions
  
### Footer Section:
- **Configurations list**: Shows which JSON file each column corresponds to

### Hardware Compatibility:

The script automatically **skips** results for formats not supported by the hardware:

| Format | Required Hardware | Skipped On |
|--------|------------------|------------|
| mxfp4  | Hopper (SM90+): H100, H200, H20 | A100, 4090, L40, etc. |
| nvfp4  | Blackwell (SM100+): B100, B200, GB200 | All current-gen GPUs |

If you run the script on an A100 machine, any `*mxfp4*.json` or `*nvfp4*.json` files will be automatically excluded with a warning message.

## Example Output

```
# Benchmark Results: NVIDIA H20

## Problem Configuration

- Type: MoE
- Shape: n7168 k256 e256 topk8
- Block size: 128x128

## Integer Weights with FP8/INT8

**Activation Type**: FP8
**Metric**: TFLOPS

| shape_m | triton-w8a8 | humming-w8a8 | humming-w7a8 | ... |
|--------:|------------:|-------------:|-------------:|-----|
|       1 |        1.10 |         1.70 |         1.71 | ... |
|       2 |        1.35 |         2.17 |         2.19 | ... |
...
```

## Notes

- The script groups results **automatically** based on the filename pattern
- Hardware capability is checked using the device name in the JSON files
- Unsupported formats are skipped with a warning message
- All values are in TFLOPS for easy comparison
