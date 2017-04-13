#!/bin/bash

use_gpu=true
dir=

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

for mdl_id in "10" "15" "18" ; do

#acwt_val=0.1
#mdl_id="final.test"

touch $dir/cmvn_opts
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config \
    --iter $mdl_id \
    exp/tri4/graph_sw1_tg data/eval2000 \
    $dir/decode_eval2000_sw1_tg_$mdl_id || exit 1;

:<< comment
has_fisher=true
  if $has_fisher; then
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_sw1_{tg,fsh_fg}_$mdl_id || exit 1;
  fi
comment

done

wait;

# eval2000
for x in $dir/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep Sum $x/score_*/*.ctm.filt.sys | utils/best_wer.sh; done 2>/dev/null
# swbd subset of eval2000,
for x in $dir/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep Sum $x/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh; done 2>/dev/null


