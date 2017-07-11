#!/bin/bash

cmd=run.pl

dir=
egs_dir=
num_iters=380

stage=0
x=1
nj=4


if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <egs-dir> <exp-dir>"
  echo " e.g.: $0 exp/conv/egs exp/conv/maxout"
  echo "[opts] : x=1, num_iters=380 (default) "
exit 1;
fi

egs_dir=$1
dir=$2


while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ]; then

echo "nnet-compute-prob (pass $x)"

for n in `seq 1 $nj`; do

    # Set off jobs doing some diagnostics, in the background.
    $cmd $dir/log/compute_prob_valid.$x.$n.log \
      nnet-compute-prob $dir/$x.$n.mdl ark:$egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.$x.$n.log \
      nnet-compute-prob $dir/$x.$n.mdl ark:$egs_dir/train_diagnostic.egs
#    if [ $x -gt 0 ] && [ ! -f $dir/log/mix_up.$[$x-1].log ]; then
#      $cmd $dir/log/progress.$x.$n.log \
#        nnet-show-progress --use-gpu=no $dir/$[$x-1].mdl $dir/$x.$n.mdl \
#          ark:$egs_dir/train_diagnostic.egs '&&' \
#        nnet-am-info $dir/$x.$n.mdl &
#    fi

done

  fi
  x=$[$x+1]
done

