#!/usr/bin/env bash
# Session 51: jet-core efficiency ceiling = current code (PR277) + PR286 fixes (LST_PT5_DEMOTE_SCORE=50 LST_PT5_UNSTALE=1)
# + synthetic truth pLS (--idealpls) + already-measured switches (Fix B, CrossCleanpLS embed scale, pT5 dedup key1).
# 1000 evt, --jet, -s 4. Resumable: tags already in the log are skipped.
# Stage 1 = runs 1-7 of the plan; the combined "ceiling" runs are added after stage 1 is inspected.
cd /mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone || exit 1
source setup.sh > /dev/null 2>&1
eval "$(cd /mnt/data1/kk829/CMSSW_16_1_1/src && scram runtime -sh 2> /dev/null)"
source setup.sh > /dev/null 2>&1
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
LOG=efficiency/s44_fixes_log.txt
NEV=${NEV:-1000}
PR286="LST_PT5_DEMOTE_SCORE=50 LST_PT5_UNSTALE=1"

run() {  # run <tag> [ENV=VAL ...]
  local T=s51_$1; shift
  local ENVS="$*"
  if grep -q " tag=$T  nevents=$NEV " $LOG 2> /dev/null; then echo "skip $T"; return; fi
  local N=Ntuple-files/LSTNtuple_${T}_${NEV}evt.root D=NumDen-files/LSTNumDen_${T}_${NEV}evt.root
  local R=efficiency/.s44_runlogs/${T}_${NEV}evt.log t0=$SECONDS
  [ -f "$N" ] && mv "$N" "$N.stale_$(date +%s)"
  echo "$(date -Is) START $T env=${ENVS:-none}"
  local rc try
  for try in 1 2 3; do  # retry the transient "There are only 0 devices!" CUDA failure
    env $ENVS timeout 7200 lst_cuda -i trackingNtuple-1000.root --jet --idealpls -n $NEV -s 4 -o $N > $R 2>&1
    rc=$?
    grep -q "only 0 devices" $R || break
    echo "$(date -Is) $T: no CUDA device (try $try), retrying in 60 s"; rm -f "$N"; sleep 60
  done
  if [ $rc -ne 0 ]; then
    echo "$(date -Is)  tag=$T  nevents=$NEV  env=${ENVS:-none}  RUN_FAILED(rc=$rc) (see $R)" >> $LOG
    echo "$(date -Is) FAILED $T rc=$rc"; return
  fi
  createPerfNumDenHists -i $N -o $D -J >> $R 2>&1 || echo "NumDen failed" >> $R
  local M=$(python3 efficiency/python/s44_eval_fixes.py $N 2>> $R)
  echo "$(date -Is)  tag=$T  nevents=$NEV  env=${ENVS:-none} idealpls  $M  wall_s=$((SECONDS - t0))" | tee -a $LOG
}

if [ "$1" = "smoke" ]; then
  SP=${2:?scratch dir}
  env $PR286 timeout 900 lst_cuda -i trackingNtuple-1000.root --jet --idealpls -n 5 -s 1 -v 1 -o $SP/s51_smoke_$(date +%s).root 2>&1 | tail -20
  exit
fi

run ideal_base
run ideal_pr286        $PR286
run ideal_pr286_rep1   $PR286
run ideal_pr286_B      $PR286 LST_PLS_DISTINCT_HITS=1
run ideal_pr286_emb05  $PR286 LST_PLS_T5EMBED_SCALE=0.5
run ideal_pr286_emb0   $PR286 LST_PLS_T5EMBED_SCALE=0
run ideal_pr286_k1     $PR286 LST_PT5_DEDUP_KEY=1
echo "$(date -Is) STAGE1 DONE"
# Stage 2: embedding-scale scan for a point within +1 pp dup of ideal_pr286
run ideal_pr286_emb075 $PR286 LST_PLS_T5EMBED_SCALE=0.75
run ideal_pr286_emb09  $PR286 LST_PLS_T5EMBED_SCALE=0.9
echo "$(date -Is) STAGE2 DONE"
