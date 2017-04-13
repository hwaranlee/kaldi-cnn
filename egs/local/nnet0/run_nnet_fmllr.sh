#!/bin/bash

train_stage=0
use_gpu=true
nj_train=1

#. ./cmd.sh
. ./path.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
#  parallel_opts="-l gpu=1"
  num_threads=1
  minibatch_size=512
  dir=exp/nnet-fbank/exp7_b_seed1
  else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  parallel_opts="-pe smp $num_threads"
  minibatch_size=128
  dir=exp/nnet_conv
fi

. ./cmd.sh
. utils/parse_options.sh

if [ ! -f $dir/.done ]; then

  steps/nnet0/train_conv_dropout_fmllr.sh --stage $train_stage --srand 1 \
   --samples-per-iter 400000 \
   --parallel-opts "$parallel_opts" \
   --num-threads "$num_threads" \
   --num-jobs-nnet "$nj_train" \
   --minibatch-size "$minibatch_size" \
   --learning-rate 0.005 \
   --max-epoch 20 \
   --keep-lr-epoch 0 \
   --halving-factor 0.5 \
   --max-stop-halving 6 \
   --cmd "$decode_cmd" \
   --egs-dir "exp/nnet-fmllr/egs" \
   --dropout-net "false" \
   --prob-relu "false" \
   data/train_nodup data/lang exp/tri4_ali_nodup $dir $dir/nnet.config  || exit 1
fi


acwt_val=0.1
mdl_id="final.test"

exit 0;

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config \
    --transform-dir exp/tri4_ali_nodup \
    exp/tri4/graph_sw1_tg data/eval2000 \
    $dir/decode_eval2000_sw1_tg_$mdl_id || exit 1;

has_fisher=true

  if $has_fisher; then
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_sw1_{tg,fsh_fg}_$mdl_id || exit 1;
  fi

wait;

# eval2000,
for x in $dir/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep Sum $x/score_*/*.ctm.filt.sys | utils/best_wer.sh; done 2>/dev/null
# swbd subset of eval2000,
for x in $dir/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep Sum $x/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh; done 2>/dev/null

#for x in $dir/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
