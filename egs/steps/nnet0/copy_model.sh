#!/bin/bash

cmd=run.pl

dir=

x=1
nj=0

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 [opts] <exp-dir>"
  echo " e.g.: $0 exp/conv/maxout"
  echo "[opts] : x=1, num_iters=1 (default) "
exit 1;
fi

dir=$1
num_iters=$x

#while [ $x -le $num_iters ]; do
#  if [ $x -ge 0 ]; then

echo "nnet-compute-prob (pass $x)"

 nnet-am-copy --binary=false $dir/$x.mdl $dir/$x.txt &
# nnet-am-copy --binary=true $dir/$x.txt $dir/$x.mdl &


#if [ $nj -gt 0 ]; then
#for n in `seq 1 $nj`; do
#     nnet-am-copy --binary=false $dir/$x.$n.mdl $dir/$x.$n.txt &
#done
#fi

#  fi
#  x=$[$x+1]
#done

exit 1;


