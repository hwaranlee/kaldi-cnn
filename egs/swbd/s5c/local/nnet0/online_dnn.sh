#!/bin/bash

. cmd.sh
. ./path.sh

exp_dir=online_demo

[ -f ./path.sh ] && . ./path.sh; # source the path.

# make glabal cmvn matrix
# matrix-sum scp:data/train_si284/cmvn.scp cmvn_stat

dir_dnn=exp/nnet-fbank/exp_maxout
graph=exp/tri4b/graph_bd_tgpr

#echo "utterance-id1 /data/hwaranlee/kaldi-trunk/egs/wsj/s5/../../../tools/sph2pipe_v2.5/sph2pipe -f wav 440c040a.wv1 | " > temp.scp
#echo "440 /data/hwaranlee/kaldi-trunk/egs/wsj/s5/fbank/cmvn_test_eval92.ark:4" > cmvn.scp
#echo "utterance-id1 440" > utt2spk

# example --wsj
#spk2utt_rsp="ark:echo utterance-id1 utterance-id1|"
#wav_rsp="scp:echo utterance-id1 temp.wav|"

# example
#spk2utt_rsp="ark:spk2utt_rsp"
#wav_rsp="scp:wav_rsp"

# real-time
spk2utt_rsp="ark:$exp_dir/spk2utt_rsp"
wav_rsp="scp,p:$exp_dir/wav_rsp"

clat_rsp=ark:/dev/null

. parse_options.sh || exit 1;

echo "DNN decoding"
#$cmd $exp_dir/log/dnn.log \
online2-wav-nnet2-latgen-faster --do-endpointing=false \
    --online=true \
    --config=conf/online_nnet0.conf \
    --max-active=7000 --beam=15.0 --lattice-beam=6.0 \
    --acoustic-scale=0.1 --word-symbol-table=$graph/words.txt \
    $dir_dnn/final.test.mdl $graph/HCLG.fst \
    "$spk2utt_rsp" "$wav_rsp" "$clat_rsp" "$exp_dir/decode_dnn"


