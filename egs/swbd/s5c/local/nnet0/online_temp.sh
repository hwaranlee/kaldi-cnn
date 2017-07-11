#!/bin/bash

. cmd.sh
. ./path.sh

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

# make glabal cmvn matrix
# matrix-sum scp:data/train_si284/cmvn.scp cmvn_stat


dir=exp/nnet-fbank/exp7_b_seed1
graph=exp/tri4b/graph_bd_tgpr

dir_dnn=exp/nnet-fbank/exp_maxout
dir_gmm=exp/tri4b

echo "utterance-id1 /data/hwaranlee/kaldi-trunk/egs/wsj/s5/../../../tools/sph2pipe_v2.5/sph2pipe -f wav 440c040a.wv1 | " > temp.scp
echo "440 /data/hwaranlee/kaldi-trunk/egs/wsj/s5/fbank/cmvn_test_eval92.ark:4" > cmvn.scp
echo "utterance-id1 440" > utt2spk


#spk2utt_rsp="ark:echo utterance-id1 utterance-id1|"
#wav_rsp="scp:echo utterance-id1 temp.wav|"
clat_rsp=ark:/dev/null

spk2utt_rsp="ark:spk2utt_rsp"
wav_rsp="scp:wav_rsp"

#echo "GMM decoding"
#online2-wav-gmm-latgen-faster --do-endpointing=true \
#    --online=true \
#    --config=conf/online_gmm.conf \
#    --max-active=7000 --beam=15.0 --lattice-beam=6.0 \
#    --acoustic-scale=0.1 --word-symbol-table=$graph/words.txt \
#    $dir_gmm/final.mdl $graph/HCLG.fst \
#    "$spk2utt_rsp" "$wav_rsp" $clat_rsp

#exit 0;


echo "CNN decoding"
# should be always "true" because iVectors are not used.
online2-wav-nnet2-latgen-faster --do-endpointing=true \
    --online=true \
    --config=conf/online_nnet0.conf \
    --max-active=7000 --beam=15.0 --lattice-beam=6.0 \
    --acoustic-scale=0.1 --word-symbol-table=$graph/words.txt \
    $dir/final.test.mdl $graph/HCLG.fst \
    "$spk2utt_rsp" "$wav_rsp" $clat_rsp


# ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |
# "ark,s,cs:wav-copy scp,p:temp.scp ark:- |" \
# "ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:data/test_eval92/utt2spk scp:cmvn.scp scp,p:temp.scp ark:- |" \



#exit 0;

echo "DNN decoding"
online2-wav-nnet2-latgen-faster --do-endpointing=true \
    --online=true \
    --config=conf/online_nnet0.conf \
    --max-active=7000 --beam=15.0 --lattice-beam=6.0 \
    --acoustic-scale=0.1 --word-symbol-table=$graph/words.txt \
    $dir_dnn/final.test.mdl $graph/HCLG.fst \
    "$spk2utt_rsp" "$wav_rsp" $clat_rsp

#dir_fisher=online_decoding_fisher
#echo "Fisher DB(1000hrs) + iVector + smbr"
#online2-wav-nnet2-latgen-faster --do-endpointing=true \
#    --online=true \
#    --config=$dir_fisher/nnet_a_gpu_online/conf/online_nnet2_decoding.conf \
#    --max-active=7000 --beam=15.0 --lattice-beam=6.0 \
#    --acoustic-scale=0.1 --word-symbol-table=$graph/words.txt \
#    $dir_fisher/nnet_a_gpu_online/smbr_epoch2.mdl $dir_fisher/graph/HCLG.fst \
#    "$spk2utt_rsp" "$wav_rsp" $clat_rsp



