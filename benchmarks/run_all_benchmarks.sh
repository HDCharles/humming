#!/bin/bash
# Simple script to run all benchmarks - no arguments needed
# Just run: ./run_all_benchmarks.sh

# Don't exit on error - we want to skip failing benchmarks and continue
set +e

# Auto-detect GPU name and compute capability
DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0).replace(' ', '-').replace('/', '-'))" 2>/dev/null || echo "unknown")
SM_VERSION=$(python3 -c "import torch; major, minor = torch.cuda.get_device_capability(); print(major * 10 + minor)")
echo "Detected GPU: $DEVICE_NAME (SM${SM_VERSION})"

# Check FP8 support (requires SM89+)
HAS_FP8=false
if [ $SM_VERSION -ge 89 ]; then
    HAS_FP8=true
    echo "FP8 support: YES"
else
    echo "FP8 support: NO (requires SM89+, skipping FP8 benchmarks)"
fi

# Check FP4 support (requires SM100+ - Blackwell)
HAS_FP4=false
if [ $SM_VERSION -ge 100 ]; then
    HAS_FP4=true
    echo "FP4 support: YES"
else
    echo "FP4 support: NO (requires SM100+ Blackwell, skipping FP4 benchmarks)"
fi

# Default dense GEMM configuration
SHAPE_N=8192
SHAPE_K=8192
RESULT_DIR="benchmarks/results/${DEVICE_NAME}/dense_n${SHAPE_N}_k${SHAPE_K}_g0"

echo "Running benchmarks for dense GEMM: n=$SHAPE_N k=$SHAPE_K"
echo "Output: $RESULT_DIR"
echo

mkdir -p "$RESULT_DIR"

SHAPE_ARGS="--shape_n $SHAPE_N --shape_k $SHAPE_K"
GROUP_SIZE="--weight_scale_group_size 128 --input_scale_group_size 128"
GEMM_TYPE="--gemm_type dense"

# Check for vLLM (optional baselines)
HAS_VLLM=false
python3 -c "import vllm" 2>/dev/null && HAS_VLLM=true

# ============================================================================
# Category 1: WfpNa16 - Floating Point Weights with FP16 Activations
# ============================================================================
echo "===== WfpNa16: Floating Point Weights with FP16 ====="

python3 benchmarks/bench_torch_w16a16.py $SHAPE_ARGS --dtype float16 --output_file "$RESULT_DIR/torch_w16a16_float16.json" || echo "  Skipped torch_w16a16_float16 (failed)"

if [ "$HAS_VLLM" = true ]; then
    python3 benchmarks/bench_marlin.py $SHAPE_ARGS --a_dtype float16 --b_dtype float8e4m3 --bs_dtype float16 --c_dtype float16 --output_file "$RESULT_DIR/marlin_w8a16_float8e4m3_float16.json" 2>/dev/null || true
    python3 benchmarks/bench_marlin.py $SHAPE_ARGS --a_dtype float16 --b_dtype int4 --bs_dtype float16 --c_dtype float16 --output_file "$RESULT_DIR/marlin_w4a16_int4_float16.json" 2>/dev/null || true
fi

for W_BITS in 8 7 6 5 4; do
    if [ $W_BITS -eq 8 ]; then
        W_DTYPE="float8e4m3"
    elif [ $W_BITS -eq 7 ]; then
        W_DTYPE="float7e4m2"
    elif [ $W_BITS -eq 6 ]; then
        W_DTYPE="float6e3m2"
    elif [ $W_BITS -eq 5 ]; then
        W_DTYPE="float5e2m2"
    else
        W_DTYPE="float4e2m1"
    fi
    python3 benchmarks/bench_humming.py $SHAPE_ARGS --a_dtype float16 --b_dtype $W_DTYPE --bs_dtype float16 --c_dtype float16 $GROUP_SIZE $GEMM_TYPE --output_file "$RESULT_DIR/humming_w${W_BITS}a16_${W_DTYPE}_float16.json" || echo "  Skipped humming_w${W_BITS}a16 (failed)"
done

echo

# ============================================================================
# Category 2: WfpNAfp8 - Floating Point Weights with FP8 Activations
# ============================================================================
echo "===== WfpNAfp8: Floating Point Weights with FP8 ====="

if [ "$HAS_FP8" = true ]; then
    python3 benchmarks/bench_cutlass_w8a8.py $SHAPE_ARGS --dtype float8e4m3 --out_dtype float16 --output_file "$RESULT_DIR/cutlass_w8a8_float8e4m3_float16.json" 2>&1 || echo "  Skipped (cutlass not supported)"

    for W_BITS in 8 7 6 5 4; do
        if [ $W_BITS -eq 8 ]; then
            W_DTYPE="float8e4m3"
        elif [ $W_BITS -eq 7 ]; then
            W_DTYPE="float7e3m3"
        elif [ $W_BITS -eq 6 ]; then
            W_DTYPE="float6e2m3"
        elif [ $W_BITS -eq 5 ]; then
            W_DTYPE="float5e2m2"
        else
            W_DTYPE="float4e2m1"
        fi
        python3 benchmarks/bench_humming.py $SHAPE_ARGS --a_dtype float8e4m3 --b_dtype $W_DTYPE --bs_dtype float16 --c_dtype float16 $GROUP_SIZE $GEMM_TYPE --output_file "$RESULT_DIR/humming_w${W_BITS}a8_${W_DTYPE}_float8e4m3.json" || echo "  Skipped humming_w${W_BITS}a8 (failed)"
    done
else
    echo "  Skipped (GPU does not support FP8 - requires SM89+)"
fi

echo

# ============================================================================
# Category 3: WfpNafp4 - Floating Point Weights with FP4 Activations
# ============================================================================
echo "===== WfpNafp4: Floating Point Weights with FP4 ====="

if [ "$HAS_FP4" = true ]; then
    for W_BITS in 8 7 6 5 4; do
        if [ $W_BITS -eq 8 ]; then
            W_DTYPE="float8e4m3"
        elif [ $W_BITS -eq 7 ]; then
            W_DTYPE="float7e3m3"
        elif [ $W_BITS -eq 6 ]; then
            W_DTYPE="float6e2m3"
        elif [ $W_BITS -eq 5 ]; then
            W_DTYPE="float5e2m2"
        else
            W_DTYPE="float4e2m1"
        fi
        python3 benchmarks/bench_humming.py $SHAPE_ARGS --a_dtype float4e2m1 --b_dtype $W_DTYPE --bs_dtype float16 --c_dtype float16 $GROUP_SIZE $GEMM_TYPE --output_file "$RESULT_DIR/humming_w${W_BITS}a4_${W_DTYPE}_float4e2m1.json" || echo "  Skipped humming_w${W_BITS}a4 (failed)"
    done
else
    echo "  Skipped (GPU does not support FP4 - requires SM100+)"
fi

echo

# ============================================================================
# Category 4: WintNAfp8 - Integer Weights with FP8 Activations
# ============================================================================
echo "===== WintNAfp8: Integer Weights with FP8 ====="

if [ "$HAS_FP8" = true ]; then
    for W_BITS in 5 4 3 2 1; do
        python3 benchmarks/bench_humming.py $SHAPE_ARGS --a_dtype float8e4m3 --b_dtype int${W_BITS} --bs_dtype float16 --c_dtype float16 $GROUP_SIZE $GEMM_TYPE --output_file "$RESULT_DIR/humming_w${W_BITS}a8_int${W_BITS}_float8e4m3.json" || echo "  Skipped humming_w${W_BITS}a8_int${W_BITS} (failed)"
    done
else
    echo "  Skipped (GPU does not support FP8 - requires SM89+)"
fi

echo

# ============================================================================
# Category 5: WintNAint4 - Integer Weights with int4 Activations
# ============================================================================
echo "===== WintNAint4: Integer Weights with int4 ====="

for W_BITS in 4 3 2 1; do
    python3 benchmarks/bench_humming.py $SHAPE_ARGS --a_dtype int4 --b_dtype int${W_BITS} --bs_dtype float16 --c_dtype float16 $GROUP_SIZE $GEMM_TYPE --output_file "$RESULT_DIR/humming_w${W_BITS}a4_int${W_BITS}_int4.json" || echo "  Skipped humming_w${W_BITS}a4_int${W_BITS} (failed)"
done

echo
echo "========================================"
echo "All benchmarks complete!"
echo "Results: $RESULT_DIR"
echo
echo "Generate tables:"
echo "  python3 benchmarks/generate_comparison_tables.py --result_dir $RESULT_DIR"
echo "========================================"
