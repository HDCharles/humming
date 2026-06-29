#!/usr/bin/env python3
"""
Generate comparison tables from benchmark results.

Usage:
    python generate_comparison_tables.py --result_dir benchmarks/results/h20/moe_e256_k8_n7168_k256_g128x128
    python generate_comparison_tables.py --result_dir benchmarks/results/h20/dense_n8192_k8192_g0
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from tabulate import tabulate


def extract_config_from_filename(filename):
    """Extract framework, weight/activation precision from filename.

    Examples:
        triton_w8a8_float8e4m3_float16.json -> (triton, 8, 8, 'int', 'fp', None)
        humming_w4a16_int4_float16.json -> (humming, 4, 16, 'int', 'fp', None)
        humming_w7a8_float7e4m2_float8e4m3.json -> (humming, 7, 8, 'fp', 'fp', None)
        humming_w4a16_mxfp4_float16.json -> (humming, 4, 16, 'fp', 'fp', 'mxfp4')
        humming_w4a4_nvfp4_nvfp4.json -> (humming, 4, 4, 'fp', 'fp', 'nvfp4')
        torch_w16a16_float16.json -> (torch, 16, 16, 'fp', 'fp', None)
    """
    parts = filename.replace('.json', '').split('_')
    framework = parts[0]  # triton, humming, marlin, cutlass, torch, etc.

    # Extract wNaN pattern
    wa_part = parts[1]  # e.g., w8a8, w4a16, w16a16
    weight_bits = int(wa_part[1:].split('a')[0])
    activation_bits = int(wa_part.split('a')[1])

    # Handle torch_w16a16_float16.json (only 3 parts)
    if len(parts) == 3:
        # torch_w16a16_float16
        activation_dtype = parts[2]  # float16 or bfloat16
        weight_dtype = activation_dtype  # Same as activation
        special_format = None
    else:
        # Determine if weight is floating point or integer
        weight_dtype = parts[2]  # e.g., int4, float8e4m3, float7e4m2, mxfp4, nvfp4
        activation_dtype = parts[3]  # e.g., float16, float8e4m3, bfloat16, nvfp4

        # Check for special formats
        special_format = None
        if 'mxfp4' in weight_dtype or 'mxfp4' in activation_dtype:
            special_format = 'mxfp4'
        elif 'nvfp4' in weight_dtype or 'nvfp4' in activation_dtype:
            special_format = 'nvfp4'

    weight_type = 'fp' if (weight_dtype.startswith('float') or weight_dtype in ['mxfp4', 'nvfp4', 'bfloat16']) else 'int'
    activation_type = 'fp' if (activation_dtype.startswith(('float', 'bfloat')) or activation_dtype in ['mxfp4', 'nvfp4']) else 'int'

    # IMPORTANT: Distinguish between INT8 and FP8
    # activation_dtype will be 'int8' for INT8 and 'float8e4m3'/'float8e5m2' for FP8
    # We want to SKIP int8 files since WintNAfp8 means FP8, not INT8
    if activation_dtype == 'int8' or weight_dtype == 'int8':
        # Skip INT8 files - they're not FP8
        return None

    return framework, weight_bits, activation_bits, weight_type, activation_type, special_format


def check_hardware_support(device_info, special_format):
    """Check if hardware supports special formats like mxfp4, nvfp4.

    Requirements:
    - mxfp4: Requires Hopper (SM90+) or Blackwell (SM100+)
    - nvfp4: Requires Blackwell (SM100+)
    """
    if special_format is None:
        return True

    device_name = device_info.get('device_name', '').upper()

    # SM version mapping (approximate)
    sm_version = 0
    if 'H100' in device_name or 'H200' in device_name or 'H20' in device_name:
        sm_version = 90  # Hopper
    elif 'B100' in device_name or 'B200' in device_name or 'GB200' in device_name:
        sm_version = 100  # Blackwell
    elif 'A100' in device_name or 'A800' in device_name:
        sm_version = 80  # Ampere
    elif '4090' in device_name or 'L40' in device_name:
        sm_version = 89  # Ada

    if special_format == 'mxfp4':
        return sm_version >= 90  # Hopper or newer
    elif special_format == 'nvfp4':
        return sm_version >= 100  # Blackwell or newer

    return True


def load_benchmark_results(result_dir):
    """Load all JSON benchmark results from a directory."""
    result_dir = Path(result_dir)
    results = {}
    device_info = None

    for json_file in result_dir.glob('*.json'):
        try:
            result = extract_config_from_filename(json_file.name)
            if result is None:
                # Skip INT8 files
                continue
            framework, w_bits, a_bits, w_type, a_type, special_format = result

            with open(json_file) as f:
                data = json.load(f)

            # Cache device info from first file
            if device_info is None:
                device_info = data.get('device', {})

            # Check hardware support for special formats
            if not check_hardware_support(device_info, special_format):
                print(f"Skipping {json_file.name}: Hardware doesn't support {special_format}")
                continue

            # Extract compute_tops for each shape_m
            compute_data = {int(k): v["compute_tops"] for k, v in data["result"].items()}

            # Store with key that includes all info
            key = (framework, w_bits, a_bits, w_type, a_type, special_format, json_file.name)
            results[key] = {
                'data': compute_data,
                'problem': data.get('problem', {}),
                'device': data.get('device', {})
            }
        except Exception as e:
            print(f"Warning: Could not parse {json_file.name}: {e}")

    return results


def group_results_by_activation(results):
    """Group results by activation type (a16, a8, a4) and weight type."""
    groups = defaultdict(list)

    for (framework, w_bits, a_bits, w_type, a_type, special_format, filename), result_data in results.items():
        # Group key: (activation_bits, weight_type, special_format)
        # e.g., (16, 'int', None) for wNa16 integer weights
        #       (8, 'fp', None) for wfpNa8 floating point weights
        #       (16, 'fp', 'mxfp4') for mxfp4a16
        #       (4, 'fp', 'nvfp4') for nvfp4
        group_key = (a_bits, w_type, special_format)
        groups[group_key].append({
            'framework': framework,
            'w_bits': w_bits,
            'a_bits': a_bits,
            'w_type': w_type,
            'a_type': a_type,
            'special_format': special_format,
            'filename': filename,
            'data': result_data['data'],
            'problem': result_data['problem'],
            'device': result_data['device']
        })

    return groups


def create_comparison_table(group_data, metric='compute_tops'):
    """Create a comparison table for a group of results."""
    # Sort by framework (triton first if present), special format, and weight bits
    def sort_key(item):
        framework_order = {'triton': 0, 'cutlass': 1, 'marlin': 2, 'humming': 3, 'torch': 4}
        special_order = {None: 0, 'mxfp4': 1, 'nvfp4': 2}
        return (
            framework_order.get(item['framework'], 99),
            special_order.get(item.get('special_format'), 99),
            -item['w_bits']
        )

    sorted_data = sorted(group_data, key=sort_key)

    # Get all unique shape_m values
    all_shape_ms = set()
    for item in sorted_data:
        all_shape_ms.update(item['data'].keys())
    all_shape_ms = sorted(all_shape_ms)

    # Build table rows
    rows = []
    for shape_m in all_shape_ms:
        row = {'shape_m': shape_m}
        for item in sorted_data:
            # Create column name
            special_suffix = f"-{item['special_format']}" if item.get('special_format') else ""
            col_name = f"{item['framework']}-w{item['w_bits']}a{item['a_bits']}{special_suffix}"

            # Get value (TFLOPS)
            if shape_m in item['data']:
                row[col_name] = item['data'][shape_m]
            else:
                row[col_name] = None
        rows.append(row)

    return rows, sorted_data


def format_group_title(a_bits, w_type, special_format):
    """Format a readable title for each group.

    ONLY outputs the 5 user-requested categories:
    1) WfpNAfp8 - Floating point weights with FP8 activations
    2) WfpNafp4 - Floating point weights with FP4 activations
    3) WintNAfp8 - Integer weights with FP8 activations
    4) WintNAfp4 - Integer weights with FP4 activations
    5) WfpNa16 - Floating point weights with FP16 activations
    """
    # Handle special formats
    if special_format == 'mxfp4':
        return "WfpNa16 (mxfp4)"
    elif special_format == 'nvfp4':
        if a_bits == 16:
            return "WfpNa16 (nvfp4)"
        elif a_bits == 4:
            return "WfpNafp4 (nvfp4)"
        else:
            return None

    # Category 1: WfpNAfp8 - Floating point weights with FP8 activations
    if w_type == 'fp' and a_bits == 8:
        return "WfpNAfp8"

    # Category 2: WfpNafp4 - Floating point weights with FP4 activations
    elif w_type == 'fp' and a_bits == 4:
        return "WfpNafp4"

    # Category 3: WintNAfp8 - Integer weights with FP8 activations
    elif w_type == 'int' and a_bits == 8:
        return "WintNAfp8"

    # Category 4: WintNAfp4 - Integer weights with FP4 activations
    elif w_type == 'int' and a_bits == 4:
        return "WintNAfp4"

    # Category 5: WfpNa16 - Floating point weights with FP16 activations
    elif w_type == 'fp' and a_bits == 16:
        return "WfpNa16"

    # NOT REQUESTED - return None to skip this category (e.g., integer weights with FP16)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate comparison tables from benchmark results')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Directory containing benchmark JSON results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: print to stdout)')
    parser.add_argument('--metric', type=str, default='compute_tops',
                       choices=['compute_tops', 'memory_gbps', 'time'],
                       help='Metric to compare (default: compute_tops)')
    parser.add_argument('--format', type=str, default='github',
                       choices=['github', 'grid', 'simple', 'pipe'],
                       help='Table format (default: github)')

    args = parser.parse_args()

    # Load all results
    print(f"Loading results from {args.result_dir}...")
    results = load_benchmark_results(args.result_dir)

    if not results:
        print(f"No benchmark results found in {args.result_dir}")
        return

    print(f"Found {len(results)} benchmark results")

    # Group by activation type and weight type
    groups = group_results_by_activation(results)

    output_lines = []

    # Get device and problem info from first result
    first_result = next(iter(results.values()))
    device_name = first_result['device'].get('device_name', 'Unknown')
    problem = first_result['problem']

    # Create header
    output_lines.append(f"# Benchmark Results: {device_name}")
    output_lines.append("")

    # Add problem details
    if problem:
        output_lines.append("## Problem Configuration")
        output_lines.append("")
        if 'num_experts' in problem:
            output_lines.append(f"- Type: MoE")
            output_lines.append(f"- Shape: n{problem.get('shape_n')} k{problem.get('shape_k')} e{problem.get('num_experts')} topk{problem.get('top_k')}")
            if problem.get('weight_scale_group_size', 0) > 0:
                output_lines.append(f"- Block size: {problem['weight_scale_group_size']}x{problem['weight_scale_group_size']}")
        else:
            output_lines.append(f"- Type: Dense")
            output_lines.append(f"- Shape: n{problem.get('shape_n')} k{problem.get('shape_k')}")
        output_lines.append("")

    # Define the required table order
    required_tables = [
        ('WfpNa16', 16, 'fp', None, 'FP16'),
        ('WfpNAfp8', 8, 'fp', None, 'FP8'),
        ('WfpNafp4', 4, 'fp', None, 'FP4'),
        ('WintNAfp8', 8, 'int', None, 'FP8'),
        ('WintNAfp4', 4, 'int', None, 'FP4'),
    ]

    # Generate tables in the specified order
    for title, a_bits, w_type, special_format, act_label in required_tables:
        output_lines.append(f"## {title}")
        output_lines.append("")
        output_lines.append(f"**Activation Type**: {act_label}")
        output_lines.append(f"**Metric**: TFLOPS")
        output_lines.append("")

        # Find matching data
        group_key = (a_bits, w_type, special_format)
        group_data = groups.get(group_key, [])

        if not group_data:
            output_lines.append("*No data available for this configuration*")
            output_lines.append("")
            continue

        rows, sorted_data = create_comparison_table(group_data, args.metric)

        if rows:
            table = tabulate(rows, headers='keys', tablefmt=args.format,
                           numalign='right', floatfmt='.2f')
            output_lines.append(table)
            output_lines.append("")

            # Add summary of configurations
            output_lines.append("**Configurations:**")
            for item in sorted_data:
                special_suffix = f"-{item['special_format']}" if item.get('special_format') else ""
                col_name = f"{item['framework']}-w{item['w_bits']}a{item['a_bits']}{special_suffix}"
                output_lines.append(f"- `{col_name}`: {item['filename']}")
            output_lines.append("")

    # Handle special format tables (mxfp4, nvfp4) separately
    for (a_bits, w_type, special_format), group_data in sorted(groups.items()):
        if special_format in ['mxfp4', 'nvfp4']:
            title = format_group_title(a_bits, w_type, special_format)
            if title is None:
                continue

            output_lines.append(f"## {title}")
            output_lines.append("")

            if special_format == 'mxfp4':
                activation_dtype = 'FP16 (mxfp4 weights)'
            elif special_format == 'nvfp4':
                activation_dtype = 'nvfp4' if a_bits == 4 else 'FP16 (nvfp4 weights)'

            output_lines.append(f"**Activation Type**: {activation_dtype}")
            output_lines.append(f"**Metric**: TFLOPS")
            output_lines.append("")

            rows, sorted_data = create_comparison_table(group_data, args.metric)

            if rows:
                table = tabulate(rows, headers='keys', tablefmt=args.format,
                               numalign='right', floatfmt='.2f')
                output_lines.append(table)
                output_lines.append("")

                # Add summary of configurations
                output_lines.append("**Configurations:**")
                for item in sorted_data:
                    special_suffix = f"-{item['special_format']}" if item.get('special_format') else ""
                    col_name = f"{item['framework']}-w{item['w_bits']}a{item['a_bits']}{special_suffix}"
                    output_lines.append(f"- `{col_name}`: {item['filename']}")
                output_lines.append("")

    # Output
    output_text = '\n'.join(output_lines)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Results written to {args.output}")
    else:
        print(output_text)


if __name__ == '__main__':
    main()
