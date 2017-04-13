#!/bin/bash

# This is convolutional neural net training on top of adapted 40-dimensional features.

train_stage=-10
use_gpu=true
nj_train=4

num_hidden_layers=4

. cmd.sh
. ./path.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
  dir=exp/conv/temp
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

#dir="$dir"_"$num_hidden_layers"

if [ ! -f $dir/final.mdl ]; then
  if [ "$USER" == dpovey ]; then
     # spread the egs over various machines.  will help reduce overload of any
     # one machine.
     utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-pure/egs/wsj/s5/$dir/egs $dir/egs/storage
  fi

  steps/nnet0/train_conv.sh --stage $train_stage \
   --samples-per-iter 400000 \
   --parallel-opts "$parallel_opts" \
   --num-threads "$num_threads" \
   --num-jobs-nnet "$nj_train" \
   --minibatch-size "$minibatch_size" \
   --mix-up 8000 \
   --initial-learning-rate 0.0005 --final-learning-rate 0.0005 \
   --num-hidden-layers "$num_hidden_layers" \
   --kernel_height 40 --kernel-width 6 --group 2000 \
   --pool_h 1 --pool_w 2 --pool_c 10 \
   --pnorm_input_dim 400 --pnorm_output_dim 2000 --p 2 \
   --cmd "$decode_cmd" \
   --egs_dir exp/conv/egs \
   data/train_si284 data/lang exp/tri4b_ali_si284 $dir || exit 1
fi


#steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
#   --transform-dir exp/tri4b/decode_tgpr_dev93 \
#    exp/tri4b/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93

#steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
#  --transform-dir exp/tri4b/decode_tgpr_eval92 \
#    exp/tri4b/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
   --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
    exp/tri4b/graph_bd_tgpr data/test_dev93 $dir/decode_bd_tgpr_dev93

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
  --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
    exp/tri4b/graph_bd_tgpr data/test_eval92 $dir/decode_bd_tgpr_eval92


wait;


for x in $dir/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done


