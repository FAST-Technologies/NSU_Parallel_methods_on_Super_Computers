
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

BUILDS=(
    "float:build-float:f32"
    "double:build-double:f64"
)

BLOCK_SIZES=(2 4 8 16 32)

mkdir -p "$RESULTS_DIR"

echo "========================================================"
echo "  CUDA Matrix Multiplication Benchmark"
echo "  Precision: float + double"
echo "  Block sizes: ${BLOCK_SIZES[*]}"
echo "  Matrix sizes: config.hpp (MATRIX_SIZES)"
echo "  Warmup and repeats: handled by the C++ program"
echo "========================================================"

for BUILD in "${BUILDS[@]}"; do
    IFS=':' read -r PRECISION BUILD_DIR_NAME PRECISION_TAG <<< "$BUILD"

    BUILD_DIR="$SCRIPT_DIR/$BUILD_DIR_NAME"
    EXECUTABLE="$BUILD_DIR/Lab_03"

    echo
    echo "========================================================"
    echo "Running $PRECISION benchmark"
    echo "Build directory: $BUILD_DIR"
    echo "========================================================"

    if [[ ! -x "$EXECUTABLE" ]]; then
        echo "Error: executable not found: $EXECUTABLE" >&2
        exit 1
    fi

    PRECISION_RESULTS="$RESULTS_DIR/$PRECISION"
    mkdir -p "$PRECISION_RESULTS"

    for BLOCK in "${BLOCK_SIZES[@]}"; do

        CSV_NAME="cuda_lab3_${PRECISION_TAG}_b${BLOCK}.csv"
        CSV_SOURCE="$BUILD_DIR/$CSV_NAME"
        CSV_DESTINATION="$PRECISION_RESULTS/$CSV_NAME"

        echo
        echo "--------------------------------------------------------"
        echo "Precision: $PRECISION"
        echo "Block size: $BLOCK x $BLOCK"
        echo "Command: ./Lab_03 all $BLOCK"
        echo "--------------------------------------------------------"

        # Remove only the previous result for this configuration.
        rm -f "$CSV_SOURCE"

        (
            cd "$BUILD_DIR"
            ./Lab_03 all "$BLOCK"
        )

        if [[ ! -s "$CSV_SOURCE" ]]; then
            echo "Error: expected CSV was not created or is empty:" >&2
            echo "$CSV_SOURCE" >&2
            exit 1
        fi

        cp -f "$CSV_SOURCE" "$CSV_DESTINATION"

        echo "Saved: $CSV_DESTINATION"
    done
done

echo
echo "========================================================"
echo "All benchmarks completed successfully."
echo "Results directory: $RESULTS_DIR"
echo "========================================================"
