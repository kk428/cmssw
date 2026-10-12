#!/bin/bash
# S49: S44 base with the T5 AfterBuild shared-hit cut LST_AB_NMATCHED = 7 (default), 8, 9, 10, 11 (11 = off),
# every other cut at master defaults. Events 0-88 (-n 89 skips event 89, 1.6M T5s), real pLS, --jet, 4 streams.
# Binary: sparse worktree /mnt/data1/kk829/s44_worktree at 7a424adcef4 (S44 code; fix switches default off).
# Resumable: a point whose NumDen file exists is skipped. Ends with 5-way eff/fake/dup vs dR plots.
MAIN=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
WT=/mnt/data1/kk829/s44_worktree/RecoTracker/LSTCore/standalone

cd "$MAIN" && source setup.sh > /dev/null 2>&1
eval "$(cd /mnt/data1/kk829/CMSSW_16_1_1/src && scram runtime -sh 2> /dev/null)"  # = cmsenv
source setup.sh > /dev/null 2>&1
cd "$WT" && source setup.sh > /dev/null 2>&1       # worktree LST libs first on LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"                                # L40
cd "$MAIN"

NEV=89
declare -A TMO=([7]=7200 [8]=7200 [9]=14400 [10]=21600 [11]=21600)
for NM in 7 8 9 10 11; do
  TAG=s49_ABNM${NM}
  NTUPLE=Ntuple-files/LSTNtuple_${TAG}_${NEV}evt.root
  NUMDEN=NumDen-files/LSTNumDen_${TAG}_${NEV}evt.root
  RLOG=efficiency/.s44_runlogs/${TAG}_${NEV}evt.log
  [ -f "$NUMDEN" ] && { echo "$(date -Is) SKIP $TAG (done)"; continue; }
  [ -f "$NTUPLE" ] && mv "$NTUPLE" "$NTUPLE.stale_$(date +%s)"
  echo "$(date -Is) START $TAG  timeout=${TMO[$NM]}s"
  t0=$SECONDS
  env -u LST_PT5_HIGHPT_GATE -u LST_PLS_DISTINCT_HITS -u LST_PT5_SCORE_MODE -u LST_EXTEND_DUPT5 \
    LST_AB_NMATCHED=$NM \
    timeout ${TMO[$NM]} "$WT/bin/lst_cuda" -i trackingNtuple-100.root --jet -n $NEV -s 4 -v 1 -o "$NTUPLE" > "$RLOG" 2>&1
  rc=$?
  echo "$(date -Is) $TAG lst_cuda rc=$rc wall=$((SECONDS - t0))s"
  [ $rc -ne 0 ] && continue
  "$WT/efficiency/bin/createPerfNumDenHists" -i "$NTUPLE" -o "$NUMDEN" -J >> "$RLOG" 2>&1
  metrics=$(python3 efficiency/python/s44_eval_fixes.py "$NTUPLE" 2>> "$RLOG")
  echo "$(date -Is)  tag=$TAG  nevents=$NEV  env=LST_AB_NMATCHED=$NM  $metrics  wall_s=$((SECONDS - t0))" \
    >> efficiency/s44_fixes_log.txt
  echo "$(date -Is) DONE $TAG $metrics"
done

# Comparison plots of whichever points finished (default LST style; duplicate rate also in the *zoom plot)
cd "$MAIN" && source setup.sh > /dev/null 2>&1
files=(); labels=()
for NM in 7 8 9 10 11; do
  f=NumDen-files/LSTNumDen_s49_ABNM${NM}_${NEV}evt.root
  [ -f "$f" ] || continue
  files+=("$f"); [ $NM = 7 ] && labels+=("AB_NM=7_default") || { [ $NM = 11 ] && labels+=("AB_NM=11_off") || labels+=("AB_NM=$NM"); }
done
python3 "$MAIN/efficiency/python/lst_plot_performance.py" "${files[@]}" -L "$(IFS=,; echo "${labels[*]}")" \
  -t s49_ABNM_scan_${NEV}evt --compare -j \
  > efficiency/.s49_abnm_plot.log 2>&1
echo "$(date -Is) PLOTS rc=$? (${#files[@]} points) -> performance/s49_ABNM_scan_${NEV}evt_*/mtv/var/"
