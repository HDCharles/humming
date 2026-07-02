#!/bin/bash
# Generate comparison tables for all benchmark results

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="$SCRIPT_DIR/results"

echo "Generating comparison tables for all benchmark results..."
echo ""

# Find all result directories
for device_dir in "$RESULTS_DIR"/*; do
    if [ -d "$device_dir" ]; then
        device_name=$(basename "$device_dir")
        echo "=== Device: $device_name ==="

        for result_dir in "$device_dir"/*; do
            if [ -d "$result_dir" ]; then
                benchmark_name=$(basename "$result_dir")
                echo ""
                echo "Benchmark: $benchmark_name"
                echo "---"

                # Generate markdown output
                output_file="$result_dir/comparison_tables.md"
                python3 "$SCRIPT_DIR/generate_comparison_tables.py" \
                    --result_dir "$result_dir" \
                    --output "$output_file" \
                    --format github

                if [ $? -eq 0 ]; then
                    echo "✓ Generated: $output_file"
                else
                    echo "✗ Failed to generate table for $result_dir"
                fi
            fi
        done
        echo ""
    fi
done

echo ""
echo "Done! Check the comparison_tables.md files in each result directory."
