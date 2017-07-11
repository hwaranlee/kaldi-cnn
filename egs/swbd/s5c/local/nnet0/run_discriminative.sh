#!/bin/bash


# This script demonstrates discriminative training of neural nets.  It's on top
# of run_5c_gpu.sh, which uses adapted 40-dimensional features.  This version of
# the script uses GPUs.  We distinguish it by putting "_gpu" at the end of the
# directory name.


gpu_opts="-l gpu=1"                   # This is suitable for the CLSP network,
                                      # you'll likely have to change it.  we'll
                                      # use it later on, in the training (it's
                                      # not used in denlat creation)
stage=0
train_stage=-100
dir=
nj=4


set -e # exit on error.

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh


# The denominator lattice creation currently doesn't use GPUs.

# Note: we specify 1G each for the mem_free and ram_free which, is per
# thread... it will likely be less than the default.  Increase the beam relative
# to the defaults; this is just for this RM setup, where the default beams will
# likely generate very thin lattices.  Note: the transform-dir is important to
# specify, since this system is on top of fMLLR features.

if [ $stage -le 0 ]; then

  cp exp/tri4/tree $dir/tree

  steps/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G" \
   --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "-pe smp 6" \
   data/train_nodup data/lang_nosp_sw1_tg $dir "$dir"_denlats
# --parallel-opts "-pe smp 6" \
 

fi


if [ $stage -le 1 ]; then
  steps/nnet2/align.sh  --cmd "$decode_cmd " --use-gpu yes \
    --nj $nj data/train_nodup data/lang_nosp_sw1_tg $dir "$dir"_ali
fi


if [ $stage -le 2 ]; then
  steps/nnet0/train_discriminative.sh --cmd "$decode_cmd"  --learning-rate 0.00002 \
    --stage $train_stage \
    --modify-learning-rates false \
    --num-epochs 4 \
    --cleanup false \
    --num-jobs-nnet 1 \
    --num-threads 1 \
    --degs-dir "/home/hwaranlee/${dir}_degs" \
     data/train_nodup data/lang_nosp_sw1_tg \
    "$dir"_ali "$dir"_denlats $dir/final.mdl "$dir"_smbr
 #--parallel-opts "$gpu_opts" \

fi

if [ $stage -le 3 ]; then
#  for epoch in 1 2 3 4; do 
   for epoch in 2 3 4 ; do
touch $dir/cmvn_opts

cp "${dir}_smbr/epoch$epoch".mdl ${dir}_smbr/final.mdl

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 --iter epoch$epoch \
    --config conf/decode.config \
    --acwt 0.05 \
    --stage 2 \
    exp/tri4/graph_sw1_tg data/eval2000 \
    ${dir}_smbr/decode_eval2000_sw1_tg_epoch$epoch || exit 1;

has_fisher=true
  if $has_fisher; then
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_sw1_{tg,fsh_fg} data/eval2000 \
      ${dir}_smbr/decode_eval2000_sw1_{tg,fsh_fg}_epoch$epoch || exit 1;
  fi

# eval2000
for x in "$dir"_smbr/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep Sum $x/score_*/*.ctm.filt.sys | utils/best_wer.sh; done 2>/dev/null
# swbd subset of eval2000,
for x in "$dir"_smbr/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep Sum $x/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh; done 2>/dev/null



  done
fi



# eval2000
for x in "$dir"_smbr/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep Sum $x/score_*/*.ctm.filt.sys | utils/best_wer.sh; done 2>/dev/null
# swbd subset of eval2000,
for x in "$dir"_smbr/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep Sum $x/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh; done 2>/dev/null


exit 0;
