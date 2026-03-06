#!/usr/bin/env bash
set -euo pipefail

BIN="./build/bin/weight_loader_bench"
MODEL="resources/ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-3.1-8b-instruct-q4_0.gguf"
TENSOR_NAMES="bench/llama-3.1-tensor-names.txt"
OUTDIR="bench/results/weightloader"

mkdir -p "$OUTDIR"

# Clean old generated CSVs so merge step only sees this run's outputs
rm -f \
  "$OUTDIR"/*_latency.csv \
  "$OUTDIR"/*_telemetry.csv \
  "$OUTDIR"/all_latency.csv \
  "$OUTDIR"/all_telemetry.csv

LATENCY_HEADER="test,model,mode,cold_hot,latency_ms,major_faults_delta,minor_faults_delta"
TELEMETRY_HEADER="test,model,mode,cold_hot,tensor_idx,tensor_name,major_faults_delta,minor_faults_delta,rss_delta_bytes,rss_bytes"

run_cold() {
  local outfile="$1"
  local kind="$2"
  shift 2

  if command -v purge >/dev/null 2>&1; then
    echo "Purging caches before cold run..."
    sudo purge
  else
    echo "Warning: 'purge' not found; cold run will only be cold-ish."
  fi

  if [[ "$kind" == "latency" ]]; then
    echo "$LATENCY_HEADER" > "$outfile"
  else
    echo "$TELEMETRY_HEADER" > "$outfile"
  fi

  "$@" | tail -n +2 >> "$outfile"
}

run_hot() {
  local outfile="$1"
  local kind="$2"
  shift 2

  if [[ "$kind" == "latency" ]]; then
    echo "$LATENCY_HEADER" > "$outfile"
  else
    echo "$TELEMETRY_HEADER" > "$outfile"
  fi

  "$@" | tail -n +2 >> "$outfile"
}

run_latency_pair() {
  local mode="$1"
  local workload="$2"

  echo "=== LATENCY cold: mode=$mode workload=$workload ==="
  run_cold \
    "$OUTDIR/${mode}_${workload}_cold_latency.csv" \
    latency \
    "$BIN" \
      --model "$MODEL" \
      --tensor-names "$TENSOR_NAMES" \
      --mode "$mode" \
      --workload "$workload" \
      --measure latency \
      --cache-state cold

  echo "=== LATENCY hot: mode=$mode workload=$workload ==="
  run_hot \
    "$OUTDIR/${mode}_${workload}_hot_latency.csv" \
    latency \
    "$BIN" \
      --model "$MODEL" \
      --tensor-names "$TENSOR_NAMES" \
      --mode "$mode" \
      --workload "$workload" \
      --measure latency \
      --cache-state hot
}

run_telemetry() {
  local mode="$1"
  local workload="$2"

  echo "=== TELEMETRY cold: mode=$mode workload=$workload ==="
  run_cold \
    "$OUTDIR/${mode}_${workload}_cold_telemetry.csv" \
    telemetry \
    "$BIN" \
      --model "$MODEL" \
      --tensor-names "$TENSOR_NAMES" \
      --mode "$mode" \
      --workload "$workload" \
      --measure telemetry \
      --cache-state cold

  echo "=== TELEMETRY hot: mode=$mode workload=$workload ==="
  run_hot \
    "$OUTDIR/${mode}_${workload}_hot_telemetry.csv" \
    telemetry \
    "$BIN" \
      --model "$MODEL" \
      --tensor-names "$TENSOR_NAMES" \
      --mode "$mode" \
      --workload "$workload" \
      --measure telemetry \
      --cache-state hot
}

for mode in fully_resident mmap; do
  for workload in sequential layer random; do
    run_latency_pair "$mode" "$workload"
  done
done

for mode in fully_resident mmap; do
  for workload in sequential layer random; do
    run_telemetry "$mode" "$workload"
  done
done

# ---------- Combine latency CSVs ----------
LATENCY_COMBINED="$OUTDIR/all_latency.csv"
{
  echo "$LATENCY_HEADER"
  for f in "$OUTDIR"/*_latency.csv; do
    [[ -f "$f" ]] || continue
    [[ "$f" == "$LATENCY_COMBINED" ]] && continue
    tail -n +2 "$f"
  done
} > "$LATENCY_COMBINED"

# ---------- Combine telemetry CSVs ----------
TELEMETRY_COMBINED="$OUTDIR/all_telemetry.csv"
{
  echo "$TELEMETRY_HEADER"
  for f in "$OUTDIR"/*_telemetry.csv; do
    [[ -f "$f" ]] || continue
    [[ "$f" == "$TELEMETRY_COMBINED" ]] && continue
    tail -n +2 "$f"
  done
} > "$TELEMETRY_COMBINED"

echo "Combined latency CSV:   $LATENCY_COMBINED"
echo "Combined telemetry CSV: $TELEMETRY_COMBINED"
echo "All results written to $OUTDIR"