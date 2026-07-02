# Benchmark Breakdown by Category

This document shows exactly what `run_all_benchmarks.sh` runs for each of the 5 categories.

---

## Category 1: WfpNa16 - Floating Point Weights with FP16 Activations

### Baselines (Reference Implementations)
1. **PyTorch FP16 Baseline**
   - `torch_w16a16_float16.json`
   - Command: `bench_torch_w16a16.py --dtype float16`
   - What: Native PyTorch FP16 x FP16 GEMM

2. **Marlin FP8 Baseline** (requires vLLM)
   - `marlin_w8a16_float8e4m3_float16.json`
   - Command: `bench_marlin.py --b_dtype float8e4m3 --a_dtype float16`
   - What: Marlin kernel with FP8 weights, FP16 activations

3. **Marlin INT4 Baseline** (requires vLLM)
   - `marlin_w4a16_int4_float16.json`
   - Command: `bench_marlin.py --b_dtype int4 --a_dtype float16`
   - What: Marlin kernel with INT4 weights, FP16 activations

### Humming Variants (w8 → w4)
4. **w8a16**: `humming_w8a16_float8e4m3_float16.json`
   - Weight: float8e4m3 (8-bit FP)
   - Activation: float16

5. **w7a16**: `humming_w7a16_float7e4m2_float16.json`
   - Weight: float7e4m2 (7-bit FP: 4-bit exponent, 2-bit mantissa)
   - Activation: float16

6. **w6a16**: `humming_w6a16_float6e3m2_float16.json`
   - Weight: float6e3m2 (6-bit FP: 3-bit exponent, 2-bit mantissa)
   - Activation: float16

7. **w5a16**: `humming_w5a16_float5e2m2_float16.json`
   - Weight: float5e2m2 (5-bit FP: 2-bit exponent, 2-bit mantissa)
   - Activation: float16

8. **w4a16**: `humming_w4a16_float4e2m1_float16.json`
   - Weight: float4e2m1 (4-bit FP: 2-bit exponent, 1-bit mantissa)
   - Activation: float16

**Total for WfpNa16**: 8 benchmarks (3 baselines + 5 humming variants)

---

## Category 2: WfpNAfp8 - Floating Point Weights with FP8 Activations

### Baselines
1. **CUTLASS FP8 Baseline**
   - `cutlass_w8a8_float8e4m3_float16.json`
   - Command: `bench_cutlass_w8a8.py --dtype float8e4m3`
   - What: NVIDIA CUTLASS library FP8 x FP8 GEMM

### Humming Variants (w8 → w4)
2. **w8a8**: `humming_w8a8_float8e4m3_float8e4m3.json`
   - Weight: float8e4m3 (8-bit FP)
   - Activation: float8e4m3 (8-bit FP)

3. **w7a8**: `humming_w7a8_float7e3m3_float8e4m3.json`
   - Weight: float7e3m3 (7-bit FP: 3-bit exponent, 3-bit mantissa)
   - Activation: float8e4m3

4. **w6a8**: `humming_w6a8_float6e2m3_float8e4m3.json`
   - Weight: float6e2m3 (6-bit FP: 2-bit exponent, 3-bit mantissa)
   - Activation: float8e4m3

5. **w5a8**: `humming_w5a8_float5e2m2_float8e4m3.json`
   - Weight: float5e2m2 (5-bit FP: 2-bit exponent, 2-bit mantissa)
   - Activation: float8e4m3

6. **w4a8**: `humming_w4a8_float4e2m1_float8e4m3.json`
   - Weight: float4e2m1 (4-bit FP: 2-bit exponent, 1-bit mantissa)
   - Activation: float8e4m3

**Total for WfpNAfp8**: 6 benchmarks (1 baseline + 5 humming variants)

---

## Category 3: WfpNafp4 - Floating Point Weights with FP4 Activations

**Status**: SKIPPED

**Reason**: Requires Blackwell architecture (SM120+) for FP4 tensor cores. Not available on:
- A100 (SM80)
- H100/H200/H20 (SM90)
- 4090 (SM89)

**Would require**: `--a_dtype float4e2m1`

**Total for WfpNafp4**: 0 benchmarks (not yet supported on current GPUs)

---

## Category 4: WintNAfp8 - Integer Weights with FP8 Activations

### Humming Variants (w5 → w1)
Note: w8a8 and w6a8 are skipped because those would be integer, and we already have the floating point versions in Category 2.

1. **w5a8**: `humming_w5a8_int5_float8e4m3.json`
   - Weight: int5 (5-bit integer, signed)
   - Activation: float8e4m3

2. **w4a8**: `humming_w4a8_int4_float8e4m3.json`
   - Weight: int4 (4-bit integer, signed)
   - Activation: float8e4m3

3. **w3a8**: `humming_w3a8_int3_float8e4m3.json`
   - Weight: int3 (3-bit integer, signed)
   - Activation: float8e4m3

4. **w2a8**: `humming_w2a8_int2_float8e4m3.json`
   - Weight: int2 (2-bit integer, signed)
   - Activation: float8e4m3

5. **w1a8**: `humming_w1a8_int1_float8e4m3.json`
   - Weight: int1 (1-bit integer, binary)
   - Activation: float8e4m3

**Total for WintNAfp8**: 5 benchmarks (all humming, low-bit integer weights)

---

## Category 5: WintNAfp4 - Integer Weights with FP4 Activations

### Humming Variants (w4 → w1)

1. **w4a4**: `humming_w4a4_int4_int4.json`
   - Weight: int4 (4-bit integer, signed)
   - Activation: int4 (4-bit integer)

2. **w3a4**: `humming_w3a4_int3_int4.json`
   - Weight: int3 (3-bit integer, signed)
   - Activation: int4

3. **w2a4**: `humming_w2a4_int2_int4.json`
   - Weight: int2 (2-bit integer, signed)
   - Activation: int4

4. **w1a4**: `humming_w1a4_int1_int4.json`
   - Weight: int1 (1-bit integer, binary)
   - Activation: int4

**Total for WintNAfp4**: 4 benchmarks (all humming, extreme low-bit)

---

## Summary

| Category | Description | # Baselines | # Humming | Total |
|----------|-------------|-------------|-----------|-------|
| **1. WfpNa16** | FP weights + FP16 act | 3 | 5 | **8** |
| **2. WfpNAfp8** | FP weights + FP8 act | 1 | 5 | **6** |
| **3. WfpNafp4** | FP weights + FP4 act | 0 | 0 | **0** (skipped) |
| **4. WintNAfp8** | INT weights + FP8 act | 0 | 5 | **5** |
| **5. WintNAfp4** | INT weights + FP4 act | 0 | 4 | **4** |
| **TOTAL** | | **4** | **19** | **23** |

**Note**: Baselines (torch, marlin, cutlass) may be skipped if vLLM is not installed. Humming benchmarks always run.

---

## Hardware Requirements by Precision

| Precision | Min SM | Hardware | Note |
|-----------|--------|----------|------|
| **int1-int3** | SM80 | A100+ | Software emulation via int8 |
| **int4** | SM80 | A100+ | Software emulation via int8 |
| **int5-int8** | SM75 | T4, 2080Ti+ | Native int8 tensor cores |
| **float4** | SM120 | Blackwell B100/B200 | Native FP4 tensor cores |
| **float5-float7** | SM80+ | A100+ | Custom FP formats, emulated |
| **float8** | SM89 | 4090, H100+ | Native FP8 tensor cores |
| **float16/bfloat16** | SM75 | All modern GPUs | Native FP16 tensor cores |

All humming benchmarks use **group size 128** for quantization scales.
