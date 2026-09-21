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

PASCAL_SELECTED=0
if command -v nvidia-smi >/dev/null 2>&1; then
    for gpu in "${GPUS[@]}"; do
        cc="$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
        [[ "$cc" == 6.* ]] && PASCAL_SELECTED=1
    done
fi

NVPROF_BIN="$(command -v nvprof 2>/dev/null || true)"
if [[ -z "$NVPROF_BIN" ]]; then
    NVPROF_BIN="$(find /usr/local -maxdepth 3 -type f -name nvprof -path '*/cuda*/bin/nvprof' 2>/dev/null | sort -V | tail -1 || true)"
fi

if [[ "$PASCAL_SELECTED" -eq 0 ]] && command -v ncu >/dev/null 2>&1; then
    echo
    echo "Profiler: Nsight Compute ($(ncu --version 2>/dev/null | tail -1 || true))"

    NCU_RUN=(ncu)
    NCU_ELEVATED=0
    if [[ "${YERBAS_NCU_SUDO:-0}" != "0" ]]; then
        NCU_ELEVATED=1
        NCU_RUN=(sudo -E env
            "HOME=$HOME"
            "XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}"
            ncu)
        echo "Profiler counters: elevated via sudo (target only)"
    else
        echo "Profiler counters: normal user"
        echo "If NVIDIA reports ERR_NVGPUCTRPERM, rerun with:"
        echo "  YERBAS_NCU_SUDO=1 bash scripts/run-fastlite-phase2-profiler.sh"
    fi

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

                "${NCU_RUN[@]}" \
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
                if [[ "$NCU_ELEVATED" -eq 1 ]]; then
                    sudo chown "$(id -u):$(id -g)" "$report" 2>/dev/null || true
                fi
                ncu --import "$report" --page details --csv > "$base-details.csv" 2>/dev/null || true
                ncu --import "$report" --page raw --csv > "$base-raw.csv" 2>/dev/null || true
                echo "Saved: $report"
                echo "Saved: $base-details.csv"
                echo "Saved: $base-raw.csv"
            else
                echo "WARNING: Nsight Compute did not produce $report"
                echo "If the console reports ERR_NVGPUCTRPERM, rerun with:"
                echo "  YERBAS_NCU_SUDO=1 bash scripts/run-fastlite-phase2-profiler.sh"
                echo "If it reports unsupported GPU instead, upload $console."
            fi
        done
    done
elif [[ -n "$NVPROF_BIN" ]]; then
    echo
    echo "Pascal/legacy profiler path: nvprof ($NVPROF_BIN)"

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

                sudo env \
                    "HOME=$HOME" \
                    "XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}" \
                    "YERBAS_CUDA_BENCH_RAW_BATCH=1" \
                    "YERBAS_BENCH_FORCE_CN_VARIANT=$variant" \
                    "$NVPROF_BIN" \
                    --profile-api-trace none \
                    --replay-mode kernel \
                    --kernels "::.*$KERNEL_REGEX.*:13" \
                    --analysis-metrics \
                    --log-file "$base-nvprof.log" \
                    "$BIN" "$gpu" 3584
            ) 2>&1 | tee "$base-console.log" || true
        done
    done
else
    echo
    if [[ "$PASCAL_SELECTED" -eq 1 ]]; then
        echo "ERROR: Pascal GPU detected, but nvprof was not found."
        echo "Current Nsight Compute cannot profile Pascal performance counters."
        echo "Checked PATH and /usr/local/cuda-*/bin/nvprof."
        echo
        echo "Run and send me:"
        echo "  nvcc --version"
        echo "  ls -ld /usr/local/cuda* /usr/local/cuda*/bin/nvprof 2>/dev/null"
    else
        echo "ERROR: neither ncu nor nvprof is available."
    fi
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
