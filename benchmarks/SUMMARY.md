# Benchmark Table Generation - Summary

## ✅ Fixed and Working!

All benchmark comparison tables have been successfully generated!

## What's Generated

The script creates **comparison tables** organized into your 3 requested categories:

### 1. **wNaFP4/8** - Integer Weights (w1-w8) with FP4/FP8 Activations
- Section title: `wNaFP8: Integer Weights (w1-w8) with FP8 Activations`
- Columns: `humming-w8a8`, `humming-w7a8`, ..., `humming-w1a8`
- Compares integer weight precisions (1-8 bits) with FP8 activations

### 2. **wfpNa16** - Floating Point Weights (w4-w8) with FP16 Activations
- Section title: `wfpNa16: Floating Point Weights (w4-w8) with FP16 Activations`
- Columns: `torch-w16a16`, `marlin-w8a16`, `marlin-w4a16`, `humming-w8a16`, ..., `humming-w4a16`
- **INCLUDES**: `mxfp4` and `nvfp4` (separate sections with hardware checking)
- Compares FP weight precisions with FP16 activations

### 3. **wfpNaFP4/8** - Floating Point Weights (w4-w8) with FP4/FP8 Activations
- Section title: `wfpNaFP8: Floating Point Weights (w4-w8) with FP8 Activations`
- Columns: `triton-w8a8`, `cutlass-w8a8`, `humming-w8a8`, ..., `humming-w4a8`  
- **INCLUDES**: `nvfp4` with nvfp4 activations (separate section, hardware checking)
- Compares FP weight precisions with FP8 activations

### Bonus: **wNa16** - Integer Weights (w1-w8) with FP16 Activations
- Section title: `wNa16: Integer Weights (w1-w8) with FP16 Activations`
- Columns: `marlin-w4a16`, `humming-w8a16`, ..., `humming-w1a16`
- Automatically included when such results are found

## Hardware Compatibility

The script **automatically checks** hardware capabilities:

| Format | Required Hardware | Example GPUs |
|--------|------------------|--------------|
| **mxfp4** | Hopper (SM90+) | H100, H200, H20 |
| **nvfp4** | Blackwell (SM100+) | B100, B200, GB200 |

Unsupported formats are **automatically skipped** with warning messages.

## Usage

### Generate for a specific directory:
```bash
python3 benchmarks/generate_comparison_tables.py \
    --result_dir benchmarks/results/h20/moe_e256_k8_n7168_k256_g128x128 \
    --output comparison_tables.md
```

### Generate for ALL benchmarks:
```bash
bash benchmarks/generate_all_tables.sh
```

## Output Locations

Tables are saved as `comparison_tables.md` in each result directory:

```
benchmarks/results/
├── 4090/
│   ├── dense_n8192_k8192_g0/
│   │   └── comparison_tables.md ✓
│   └── dense_n8192_k8192_g0_f16accum/
│       └── comparison_tables.md ✓
├── a800/
│   └── dense_n8192_k8192_g0/
│       └── comparison_tables.md ✓
├── h20/
│   ├── dense_n8192_k8192_g0/
│   │   └── comparison_tables.md ✓
│   ├── moe_e256_k8_n512_k7168_g128x128/
│   │   └── comparison_tables.md ✓
│   └── moe_e256_k8_n7168_k256_g128x128/
│       └── comparison_tables.md ✓
└── h200/
    └── dense_n8192_k8192_g0/
        └── comparison_tables.md ✓
```

## Example Table

The tables match the format from the vllm PR #34556:

| shape_m | triton-w8a8 | humming-w8a8 | humming-w7a8 | humming-w6a8 | ... |
|---------|-------------|--------------|--------------|--------------|-----|
| 1       | 1.10        | 1.70         | 1.71         | 1.72         | ... |
| 2       | 1.35        | 2.17         | 2.19         | 2.26         | ... |
| ...     | ...         | ...          | ...          | ...          | ... |

All values are in **TFLOPS** for easy performance comparison!

## Files Created

1. **`generate_comparison_tables.py`** - Main script with hardware checking
2. **`generate_all_tables.sh`** - Batch processor for all results
3. **`README_TABLES.md`** - Detailed documentation
4. **`SUMMARY.md`** - This file!

## Notes

- The script handles filenames like `torch_w16a16_float16.json` (3 parts) correctly
- Special formats (mxfp4, nvfp4) are detected and labeled appropriately
- Hardware compatibility is checked from device info in JSON files
- Results are automatically grouped by activation type and weight type
