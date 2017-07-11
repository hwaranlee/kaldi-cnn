#!/bin/bash

train_stage=0
use_gpu=true
nj_train=1
dir=
lrate=0.005
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

  steps/nnet0/train_conv_dropout_wsj.sh --stage $train_stage --srand 1 \
   --samples-per-iter 400000 \
   --parallel-opts "$parallel_opts" \
   --num-threads "$num_threads" \
   --num-jobs-nnet "$nj_train" \
   --minibatch-size "$minibatch_size" \
   --learning-rate "$lrate" \
   --max-epoch 30 \
   --keep-lr-epoch 0 \
   --halving-factor 0.5 \
   --max-stop-halving 6 \
   --cmd "$decode_cmd" \
   --egs-dir "/home/hwaranlee/egs_aurora4" \
   --dropout-net "false" \
   --prob-relu "false" \
   data-fbank/train_si84_multi data/lang exp/tri2b_multi_ali_si84 data-fbank/dev_0330 exp/tri2b_multi_ali_dev_0330 $dir $dir/nnet.config  || exit 1
fi


acwt_val=0.1
mdl_id="final.test"
touch $dir/cmvn_opts

for test in $(seq -f "%02g" 01 14); do
  x=test_eval92_${test}

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
    --config conf/decode_dnn.config \
    exp/tri2b_multi/graph_tgpr_5k data-fbank/${x} \
    $dir/decode_tgpr_5k_${x} || exit 1;


wait;

done

for x in $dir/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

exit 0;
