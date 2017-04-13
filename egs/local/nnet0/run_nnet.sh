#!/bin/bash

train_stage=-2
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
  dir=exp/nnet-fbank/exp4_1
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

  steps/nnet0/train_conv_dropout.sh --stage $train_stage \
   --samples-per-iter 400000 \
   --parallel-opts "$parallel_opts" \
   --num-threads "$num_threads" \
   --num-jobs-nnet "$nj_train" \
   --minibatch-size "$minibatch_size" \
   --learning-rate 0.01 \
   --max-epoch 3000 \
   --keep-lr-epoch 0 \
   --halving-factor 0.5 \
   --max-stop-halving 6 \
   --cmd "$decode_cmd" \
   --egs-dir "exp/nnet-fbank/egs" \
   --dropout-net "false" \
   --prob-relu "false" \
   data/train_si284 data/lang exp/tri4b_ali_si284 \
   data/test_dev93 exp/tri4b_ali_dev93 \
   $dir $dir/nnet.config || exit 1
fi

acwt_val=0.1
steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 2 --iter "final.test" --acwt "$acwt_val"\
	--scoring-opts "--word-ins-penalty 0.0"\
    exp/tri4b/graph_bd_tgpr data/test_dev93 $dir/decode_bd_tgpr_dev93_$acwt_val

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 2 --iter "final.test" --acwt "$acwt_val"\
	--scoring-opts "--word-ins-penalty 0.0"\
	exp/tri4b/graph_bd_tgpr data/test_eval92 $dir/decode_bd_tgpr_eval92_$acwt_val

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 2 --iter "final.test" --acwt "$acwt_val"\
	--scoring-opts "--word-ins-penalty 0.0"\
    exp/tri4b/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93_$acwt_val

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 2 --iter "final.test" --acwt "$acwt_val"\
	--scoring-opts "--word-ins-penalty 0.0"\
	exp/tri4b/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92_$acwt_val


wait;

for x in $dir/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
