#!/bin/bash

# Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)
# Apache 2.0.

##  Basically based on logMel filterbank features

#./steps/nnet0/eval.sh data/train_si284 data/lang exp/tri4b_ali_si284 data/test_dev93 exp/tri4b_ali_dev93 $dir $dir/nnet.config

# Begin configuration section.
cmd=run.pl
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.   

# parallel options
num_threads=16
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
#combine_num_threads=8
#combine_parallel_opts="-pe smp 8"  # queue options for the "combine" stage.

# running options
egs_dir=exp/nnet-fbank/egs

# End configuration

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
  echo "Usage: $0 [opts] <data> <lang> <train-ali-dir> <valid-data> <valid-ali-dir> <exp-dir> <nnet-config>"
  echo " e.g.: $0 data/train data/lang exp/tri4b_ali_si284 data/test_dev93 exp/tri4b_ali_dev93 exp/dir nnet.config"
  echo ""
  echo "Main options (for others, see top of script file)"

  exit 1;
fi

data=$1
lang=$2
alidir=$3
data_valid=$4
alidir_valid=$5
dir=$6
nnet_config=$7


# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $alidir/tree $nnet_config; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
nj_valid=`cat $alidir_valid/num_jobs`

# Testing

logfile=$dir/test.log
test_mdl=$dir/final.test.mdl
if [ ! -f $dir/log/compute_prob_test.final.1.log ]; then
$cmd $parallel_opts JOB=1:$nj_valid $dir/log/compute_prob_test.final.JOB.log \
      nnet-compute-prob $test_mdl ark:$egs_dir/test_egs.JOB.ark || exit 1;
fi

validation=$(perl -e '($nj,$pat)=@ARGV; $num_frame=0; $logprob=0; $acc=0; $acc_num_frame=0; $this_loss=0; $this_acc=0;
	 for ($n=1;$n<=$nj;$n++) {
	 $fn = sprintf($pat,$n);
	 open(F, "<$fn") || die "Error opening log file $fn";
	 while (<F>) {
		 if (m/Saw (\S+) examples, average probability is (\S+) and accuracy is (\S+)/) {$num_frame=$1; $logprob=$2; $acc=$3}
		 }
	 close(F);
	 $this_loss=$this_loss + $num_frame * $logprob;
	 $this_acc=$this_acc + $num_frame * $acc;
	 $acc_num_frame=$acc_num_frame + $num_frame; }
	 $this_loss=$this_loss/$acc_num_frame;
	 $this_acc=$this_acc/$acc_num_frame;
	 print "$this_loss\n$this_acc"; ' $nj_valid $dir/log/compute_prob_test.final.%d.log) || exit 1;

x=0
while read -r line; do loss[$x]=$line; x=$[$x+1]; done <<< "$validation"

echo "EPOCH $epoch "
echo "CROSSVAL AVG.LOSS $(printf "%.4f" ${loss[0]}) ACC $(printf "%.4f" ${loss[1]}) "
(
echo "EPOCH $epoch"
echo "CROSSVAL AVG.LOSS $(printf "%.4f" ${loss[0]}) ACC $(printf "%.4f" ${loss[1]}) "
) >> $logfile



