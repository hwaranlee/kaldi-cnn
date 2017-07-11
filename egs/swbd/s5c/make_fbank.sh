#!/bin/bash

. ./cmd.sh 

mfccdir=fbank
#for x in train eval2000; do
for x in eval2000 train_nodup; do
  steps/make_fbank.sh --nj 50 --cmd "$train_cmd" \
    data/$x exp/make_fbank/$x $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_fbank/$x $mfccdir
  utils/fix_data_dir.sh data/$x
done
