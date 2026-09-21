#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-production-latency}"
BIN="$BUILD_DIR/cuda-batch-benchmark"
OUT_DIR="$ROOT/logs/fastlite-profiler-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "Configuring CUDA build: $BUILD_DIR"
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building cuda-batch-benchmark..."
cmake --build "$BUILD_DIR" --target cuda-batch-benchmark -j"$(nproc)"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: benchmark binary was not produced: $BIN" >&2
    exit 1
fi

if [[ -n "${YERBAS_BENCH_GPUS:-}" ]]; then
    read -r -a GPUS <<< "$YERBAS_BENCH_GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
else
    GPUS=(0)
fi

VARIANTS=(fast lite)
KERNEL_REGEX="cryptonight_loop_stage_ttable4_coalesced"

echo "============================================================"
echo " YERBAS FAST/LITE PHASE-2 PROFILER"
echo "============================================================"
echo " Binary:      $BIN"
echo " GPUs:        ${GPUS[*]}"
echo " Batch:       3584"
echo " Variants:    ${VARIANTS[*]}"
echo " Kernel:      $KERNEL_REGEX"
echo " Output dir:  $OUT_DIR"
echo
echo "Goal: measure scheduler/warp stalls, occupancy and memory behavior"
echo "inside the production four-lane Fast/Lite phase-2 kernel."
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "GPU snapshot:"
    nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu \
        --format=csv,noheader 2>/dev/null || true
fi

if command -v ncu >/dev/null 2>&1; then
    echo
    echo "Profiler: Nsight Compute ($(ncu --version 2>/dev/null | tail -1 || true))"

    for gpu in "${GPUS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            base="$OUT_DIR/gpu${gpu}-${variant}"
            console="$base-console.log"

            echo
            echo "################################################################"
            echo "### GPU $gpu | CN-$variant | phase-2"
            echo "################################################################"

            # forced single-variant benchmark performs four warmup scans. Each
            # scan contains three matching CN phase-2 launches, so skip the 12
            # warmup launches and profile the first production launch from the
            # measured scan.
            (
                unset YERBAS_CN_PHASE_RETUNE || true
                unset YERBAS_CN_PHASE_THREADS || true
                unset YERBAS_CN_SETUP_THREADS || true
                unset YERBAS_CN_FINAL_THREADS || true
                unset YERBAS_DIAGNOSTICS || true
                unset YERBAS_CUDA_PROFILE || true
                unset YERBAS_CUDA_RETUNE || true
                unset YERBAS_CUDA_OVERLAP || true
                unset YERBAS_GPU_AUTOTUNE || true
                unset YERBAS_GPU_VARIANT_AUTOTUNE || true
                unset YERBAS_CN_MUL4_EXPERIMENT || true
                unset YERBAS_CN_PAIRLOAD_EXPERIMENT || true
                unset YERBAS_CN_2LANE_TTABLE_EXPERIMENT || true
                unset YERBAS_CN_READONLY_TTABLE_EXPERIMENT || true

                export YERBAS_CUDA_BENCH_RAW_BATCH=1
                export YERBAS_BENCH_FORCE_CN_VARIANT="$variant"

                ncu \
                    --target-processes all \
                    --kernel-name-base demangled \
                    --kernel-name "regex:$KERNEL_REGEX" \
                    --launch-skip 12 \
                    --launch-count 1 \
                    --section SpeedOfLight \
                    --section SchedulerStats \
                    --section WarpStateStats \
                    --section MemoryWorkloadAnalysis \
                    --section Occupancy \
                    --force-overwrite \
                    -o "$base" \
                    "$BIN" "$gpu" 3584
            ) 2>&1 | tee "$console"

            report="$base.ncu-rep"
            if [[ -f "$report" ]]; then
                ncu --import "$report" --page details --csv > "$base-details.csv" 2>/dev/null || true
                ncu --import "$report" --page raw --csv > "$base-raw.csv" 2>/dev/null || true
                echo "Saved: $report"
                echo "Saved: $base-details.csv"
                echo "Saved: $base-raw.csv"
            else
                echo "WARNING: Nsight Compute did not produce $report"
                echo "If the console reports ERR_NVGPUCTRPERM or unsupported GPU,"
                echo "upload $console and we will use the appropriate fallback."
            fi
        done
    done
elif command -v nvprof >/dev/null 2>&1; then
    echo
    echo "Nsight Compute not found; falling back to nvprof analysis metrics."

    for gpu in "${GPUS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            base="$OUT_DIR/gpu${gpu}-${variant}"
            (
                unset YERBAS_CN_PHASE_RETUNE || true
                unset YERBAS_DIAGNOSTICS || true
                unset YERBAS_CUDA_PROFILE || true
                unset YERBAS_CUDA_RETUNE || true
                unset YERBAS_CUDA_OVERLAP || true
                unset YERBAS_GPU_AUTOTUNE || true
                unset YERBAS_GPU_VARIANT_AUTOTUNE || true
                export YERBAS_CUDA_BENCH_RAW_BATCH=1
                export YERBAS_BENCH_FORCE_CN_VARIANT="$variant"

                nvprof \
                    --analysis-metrics \
                    --kernels "$KERNEL_REGEX" \
                    --log-file "$base-nvprof.log" \
                    "$BIN" "$gpu" 3584
            ) 2>&1 | tee "$base-console.log" || true
        done
    done
else
    echo
    echo "ERROR: neither ncu (Nsight Compute) nor nvprof is installed."
    echo "No source changes are needed. Upload this console output and the output of:"
    echo "  nvidia-smi"
    echo "  nvcc --version"
    exit 2
fi

echo
echo "============================================================"
echo " PROFILER PASS COMPLETE"
echo "============================================================"
echo "Upload the entire directory contents from:"
echo "  $OUT_DIR"
echo
echo "The first metrics we care about are:"
echo "  - long scoreboard / memory dependency stalls"
echo "  - eligible warps per scheduler"
echo "  - achieved vs theoretical occupancy"
echo "  - L1/L2 hit behavior and DRAM traffic"
echo "  - issue-slot utilization"
