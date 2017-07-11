#!/bin/bash
# Copyright 2013  Bagher BabaAli

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh

decode_nj=10
train_nj=30

make_data=false
make_mfcc=false
training=false
exp=exp/mono_61
ali_dir=exp/mono_61_ali
ali_dev_dir=exp/mono_61_ali_dev

if $make_data; then

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

timit=/data/export/LDC/LDC93S1/timit/TIMIT

local/timit_data_prep.sh $timit || exit 1;

local/timit_prepare_dict.sh || exit 1;

# Caution below: we insert optional-silence with probability 0.5, which is the
# default, but this is probably not appropriate for this setup, since silence
# appears also as a word in the dictionary and is scored.  We could stop this
# by using the option --sil-prob 0.0, but apparently this makes results worse.

#utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
# data/local/dict "sil" data/local/lang_tmp data/lang
utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "h#" data/local/lang_tmp data/lang || exit 1;

local/timit_format_data.sh || exit 1;

fi

if $make_mfcc; then
echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set           "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc
use_pitch=false
use_ffv=false

for x in train dev test; do 
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

fi


echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================

if $training; then

steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang $exp || exit 1;


utils/mkgraph.sh --mono data/lang_test_bg $exp $exp/graph || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --score_basic true --skip_scoring true \
 $exp/graph data/dev $exp/decode_dev || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --score_basic true --skip_scoring true \
 $exp/graph data/test $exp/decode_test || exit 1;
fi

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
 data/train data/lang $exp $ali_dir 
steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
 data/dev data/lang $exp $ali_dev_dir 

echo ============================================================================
echo "                    Getting Results [see RESULTS file]                    "
echo ============================================================================

for x in $exp/decode*; do
  [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh
done 

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0
