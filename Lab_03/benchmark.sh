
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

BUILDS=(
    "float:build-float"
    "double:build-double"
)

mkdir -p "$RESULTS_DIR"

echo "========================================================"
echo "  CUDA Matrix Multiplication Benchmark"
echo "  Precision: float + double"
echo "  Matrix sizes are taken from config.hpp (MATRIX_SIZES)"
echo "  Repeats and warmup are handled by the C++ program"
echo "========================================================"

for BUILD in "${BUILDS[@]}"; do
    PRECISION="${BUILD%%:*}"
    BUILD_DIR_NAME="${BUILD#*:}"

    BUILD_DIR="$SCRIPT_DIR/$BUILD_DIR_NAME"
    EXECUTABLE="$BUILD_DIR/Lab_03"
    CSV_SOURCE="$BUILD_DIR/cuda_lab3_results.csv"
    CSV_DESTINATION="$RESULTS_DIR/cuda_lab3_${PRECISION}_results.csv"

    echo
    echo "--------------------------------------------------------"
    echo "Running $PRECISION benchmark"
    echo "Executable: $EXECUTABLE"
    echo "--------------------------------------------------------"

    if [[ ! -x "$EXECUTABLE" ]]; then
        echo "Error: executable not found or not executable: $EXECUTABLE" >&2
        echo "Build the $PRECISION configuration first." >&2
        exit 1
    fi

    (
        cd "$BUILD_DIR"
        ./Lab_03
    )

    if [[ ! -s "$CSV_SOURCE" ]]; then
        echo "Error: results CSV was not created or is empty: $CSV_SOURCE" >&2
        exit 1
    fi

    cp -f "$CSV_SOURCE" "$CSV_DESTINATION"
    echo "Saved: $CSV_DESTINATION"
done

echo
echo "========================================================"
echo "All benchmarks completed successfully."
echo "Results directory: $RESULTS_DIR"
echo "========================================================"
