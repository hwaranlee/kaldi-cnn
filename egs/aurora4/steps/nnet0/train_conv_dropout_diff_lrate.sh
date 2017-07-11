#!/bin/bash

# Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)
# Apache 2.0.

##  Basically based on logMel filterbank features

# Begin configuration section.
cmd=run.pl
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.   

stage=-5

# training options
learning_rate=0.008
learning_rate_scale=1:0.9:0.8:0.7:0.6:0.5
diff_lrate_scale=false

# data processing
splice_width=10 # meaning +- 4 frames on each side
minibatch_size=256
samples_per_iter=200000 # each iteration of training, see this many samples
                        # per job.  This option is passed to get_egs.sh
num_jobs_nnet=1   # Number of neural net jobs to run in parallel.  This option
                   # is passed to get_egs.sh.
get_egs_stage=0
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
                # (the point of this is to get data in different minibatches on different iterations,
                # since in the preconditioning method, 2 samples in the same minibatch can
                # affect each others' gradients.
shuffle_seed=
dropout_scales=1:1:1:0.5:0.5:1
dropout_net=false
prob_relu=false

# learning rate scheduling
max_epoch=20
# min_epoch=0 # keep training, disable weight rejection, start learn-rate halving as usual,
keep_lr_epoch=0 # fix learning rate for N initial epochs,
halving_factor=0.5
max_stop_halving=6
num_stop_halving=0

# parallel options
num_threads=16
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
#combine_num_threads=8
#combine_parallel_opts="-pe smp 8"  # queue options for the "combine" stage.

# running options
cleanup=true
egs_dir=
# lda_opts=
# lda_dim=
egs_opts=
# transform_dir=     # If supplied, overrides alidir
cmvn_opts=  # will be passed to get_lda.sh and get_egs.sh, if supplied.
            # only relevant for "raw" features, not lda.
feat_type="raw"  # Can be used to force "raw" features.

prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.

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

# Set some variables.
num_leaves=`tree-info $alidir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1
[ -z $num_leaves ] && echo "\$num_leaves is unset" && exit 1
[ "$num_leaves" -eq "0" ] && echo "\$num_leaves is 0" && exit 1

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj
utils/split_data.sh $data $nj

##############################
# Prepare data: nnet samples

extra_opts=()
[ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
[ ! -z "$feat_type" ] && extra_opts+=(--feat-type "$feat_type")
# [ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
# [ -z "$transform_dir" ] && transform_dir=$alidir
# extra_opts+=(--transform-dir $transform_dir)
extra_opts+=(--splice-width $splice_width) # for LDA

feat_dim=40  # 40 coefficient Mel filterbank features
echo $feat_dim > $dir/feat_dim

##  egs (add-deltas 부분 없음.. ㅜㅜ)
if [ $stage -le -3 ]; then
  echo "$0: calling get_egs0.sh"
  # training & valid data egs
  steps/nnet0/get_egs0.sh $egs_opts "${extra_opts[@]}" \
      --samples-per-iter $samples_per_iter \
      --num-jobs-nnet $num_jobs_nnet --stage $get_egs_stage \
      --cmd "$cmd" $egs_opts --io-opts "$io_opts" \
      $data $lang $alidir \
      $data_valid $alidir_valid \
      $egs_dir || exit 1;
fi

iters_per_epoch=`cat $egs_dir/iters_per_epoch`  || exit 1;
! [ $num_jobs_nnet -eq `cat $egs_dir/num_jobs_nnet` ] && \
  echo "$0: Warning: using --num-jobs-nnet=`cat $egs_dir/num_jobs_nnet` from $egs_dir"
num_jobs_nnet=`cat $egs_dir/num_jobs_nnet` || exit 1;

##############################
#Initializing NN and MDL

if [ $stage -le -2 ]; then
  echo "$0: initializing neural net";
    [ ! -f $dir/nnet.config ] && echo "$0: no such file $nnet_config" && exit 1; #Pre-defined nnet.config

if $diff_lrate_scale ; then
 [ ! -f  $dir/lrate_scale.config ] && echo "$0: no such file $dir/lrate_scale.config" && exit 1;
 learning_rate_scale=`cat $dir/lrate_scale.config`

 $cmd $dir/log/nnet_init.log \
    nnet-am-init $alidir/tree $lang/topo "nnet-init $dir/nnet.config -|" -\| \
    nnet-am-copy --learning-rate-factor="$learning_rate" --learning-rates="$learning_rate_scale" - $dir/0.mdl || exit 1;
else
 $cmd $dir/log/nnet_init.log \
	 nnet-am-init $alidir/tree $lang/topo "nnet-init $dir/nnet.config -|" -\| \
	 nnet-am-copy --learning-rate=$learning_rate - $dir/0.mdl || exit 1;
fi

test_mdl=$dir/0.test.mdl

fi

# nnet.config example:
# SpliceComponent input-dim=$ext_feat_dim left-context=$splice_width right-context=$splice_width const-component-dim=$ivector_dim
# ConvolutionComponent in-height=$ext_feat_dim in-width=$in_width in-channel=$in_channel kernel-height=$kernel_height kernel-width=$kernel_width stride=$stride group=$group out-height=$out_height out-width=$out_width learning-rate=$initial_learning_rate param-stddev=$conv_stddev bias-stddev=$bias_stddev weight-decay=0.0002 momentum=0.9
# MaxpoolComponent in-height=$out_height in-width=$out_width in-channel=$group pool-height-dim=$pool_h pool-width-dim=$pool_w pool-channel-dim=$pool_c
# RectifiedLinearComponent dim=$affine_input_dim
# NormalizeComponent dim=$affine_input_dim
# FullyConnectedComponent input-dim=$pnorm_input_dim output-dim=$num_leaves learning-rate=$initial_learning_rate param-stddev=0 bias-stddev=0 weight-decay=0.0002 momentum=0.9
# AffineComponentPreconditionedOnline input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
# SoftmaxComponent dim=$num_leaves

if [ $stage -le -1 ]; then
  echo "Training transition probabilities and setting priors"
  $cmd $dir/log/train_trans.log \
    nnet-train-transitions $dir/0.mdl "ark:gunzip -c $alidir/ali.*.gz|" $dir/0.mdl \
    || exit 1;
fi

##############################
#start training
if [ $num_threads -eq 1 ]; then
  parallel_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
  parallel_train_opts=
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
  fi
else
  parallel_suffix="-parallel"
  parallel_train_opts="--num-threads=$num_threads"
fi

# training
echo ""
echo "$0: Training NN : max epochs $max_epoch , keep-lr-epoch $keep_lr_epoch "
echo "$0: iteration per epoch $iters_per_epoch "

DATE=`date +\%y\%m\%d\%H\%M`
logfile=$dir/train_summary_$DATE.log

# cross-validation on original network
nj_valid=`cat $alidir_valid/num_jobs`

test_mdl=$dir/0.mdl
 
if $dropout_net ; then
 [ ! -f  $dir/dropout_scale.config ] && echo "$0: no such file $dir/dropout_scale.config" && exit 1;
 dropout_scale=`cat $dir/dropout_scale.config`
 nnet-am-copy --scales=$dropout_scale --remove-dropout="true" $dir/0.mdl $dir/0.test.mdl || exit 1;
 test_mdl=$dir/0.test.mdl
elif $prob_relu; then
 nnet-am-copy --expectation-probReLU="true" $dir/0.mdl $dir/0.test.mdl || exit 1;
 test_mdl=$dir/0.test.mdl
fi

if [ ! -f $dir/.init_valid ]; then
 $cmd $parallel_opts JOB=1:$nj_valid $dir/log/compute_prob_valid.0.JOB.log \
nnet-compute-prob $test_mdl ark:$egs_dir/valid_egs.JOB.ark || exit 1;
 touch $dir/.init_valid
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
	 print "$this_loss\n$this_acc"; ' $nj_valid $dir/log/compute_prob_valid.0.%d.log) || exit 1;

	x=0
	while read -r line; do loss_prev[$x]=$line; x=$[$x+1]; done <<< "$validation"

echo ""
echo "VALIDATION PRERUN AVG.LOSS $(printf "%.4f" ${loss_prev[0]}) $(printf "%.4f" ${loss_prev[1]})"
cat >$logfile << EOF
VALIDATION PRERUN AVG.LOSS $(printf "%.4f" ${loss_prev[0]}) ACC $(printf "%.4f" ${loss_prev[1]})
EOF

mdl_best=$dir/0.mdl

if [ ! -f $dir/lr ]; then
	echo "$learning_rate" > $dir/lr
	echo "${loss_prev[0]}" > $dir/loss_prev
	echo "$num_stop_halving" > $dir/stopping
else
	learning_rate=`cat $dir/lr`
	loss_prev[0]=`cat $dir/loss_prev`
	num_stop_halving=`cat $dir/stopping`
fi

for epoch in $(seq -w $max_epoch); do
  epoch=`echo $epoch|sed 's/^0*//'`
  echo -n "EPOCH $epoch: "

  mdl=$dir/$epoch.mdl
  
  # skip iteration if already done  
  [ -f $mdl ] && echo "skipping... " && continue
  if [ -f $dir/final.mdl ]; then mdl_best=$dir/final.mdl; fi

  #  mdl_iter=$dir/$[$epoch-1].mdl
  mdl_iter=$mdl_best

  learning_rate=`cat $dir/lr`
  loss_prev[0]=`cat $dir/loss_prev`
  num_stop_halving=`cat $dir/stopping`

  echo "lr : $learning_rate";

  tr_loss=0
  this_tr_loss=0
  
  for iters in $(seq -w $iters_per_epoch); do
   iters=`echo $iters|sed 's/^0*//'`

   [ -f $dir/$epoch.$iters.1.mdl ] && mdl_iter=$dir/$epoch.$iters.1.mdl && echo "EPOCH $epoch - ITER $iters : skipping" && continue # in case one model

   $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$epoch.$iters.JOB.log \
      nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$iters \
      ark:$egs_dir/egs.JOB.$[$iters-1].ark ark:- \| \
       nnet-train$parallel_suffix $parallel_train_opts \
        --minibatch-size=$minibatch_size --srand=$iters "$mdl_iter" \
        ark:- $dir/$epoch.$iters.JOB.mdl \
      || exit 1;

   if [ $num_jobs_nnet -gt 1 ]; then # multi-model and averaging
    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list="$nnets_list $dir/$epoch.$iters.$n.mdl"
    done

    $cmd $dir/log/average.$epoch.$iters.log \
        nnet-am-average $nnets_list - \| \
        nnet-am-copy --learning-rate=$learning_rate - $dir/$epoch.$iters.mdl || exit 1;

    mdl_iter=$dir/$epoch.$iters.mdl
   else
	mdl_iter=$dir/$epoch.$iters.1.mdl
   fi
   this_tr_loss=$(perl -e '($nj,$pat)=@ARGV; $this_tr_loss=0; $num_frame=1; $acc_num_frame=0;
	 for ($n=1;$n<=$nj;$n++) {
	 $fn = sprintf($pat,$n);
	 open(F, "<$fn") || die "Error opening log file $fn";
	 while (<F>) {
		 if (m/Did backprop on (\S+) examples, average log-prob per frame is (\S+)/) {$num_frame=$1; $logprob=$2;}
		 }
	 close(F); 
	 $this_tr_loss=$this_tr_loss + $num_frame * $logprob;
	 $acc_num_frame=$acc_num_frame + $num_frame; }
	 $this_tr_loss=$this_tr_loss/$acc_num_frame;
	 print "$this_tr_loss\n"; ' $num_jobs_nnet $dir/log/train.$epoch.$iters.%d.log) || exit 1;

	 echo "EPOCH $epoch - ITER $iters : $this_tr_loss" 
	 (
	 echo "EPOCH $epoch - ITER $iters : $this_tr_loss"
	 )>> $logfile
#	 tr_loss=`perl -e '($x,$y)=@ARGV; $ans=$x + $y; print "$ans"; ' $tr_loss $this_tr_loss`;
	 tr_loss=$(awk "BEGIN{print( $tr_loss+$this_tr_loss )}")
  done

  cp $mdl_iter $mdl || exit 1;
 
# After training one epoch, check validation performance

test_mdl=$dir/$epoch.mdl

if [ ! -f $dir/log/compute_prob_valid.$epoch.1.log ]; then

if $dropout_net ; then
 nnet-am-copy --scales=$dropout_scale --remove-dropout="true" $dir/$epoch.mdl $dir/$epoch.test.mdl || exit 1;
 test_mdl=$dir/$epoch.test.mdl
elif $prob_relu; then
 nnet-am-copy --expectation-probReLU="true"  $dir/$epoch.mdl $dir/$epoch.test.mdl || exit 1;
 test_mdl=$dir/$epoch.test.mdl
fi
   $cmd $parallel_opts JOB=1:$nj_valid $dir/log/compute_prob_valid.$epoch.JOB.log \
      nnet-compute-prob $test_mdl ark:$egs_dir/valid_egs.JOB.ark || exit 1;
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
	 print "$this_loss\n$this_acc"; ' $nj_valid $dir/log/compute_prob_valid.$epoch.%d.log) || exit 1;

	x=0
	while read -r line; do loss[$x]=$line; x=$[$x+1]; done <<< "$validation"

  rm $dir/$epoch.*.mdl || exit 1;

#tr_loss=`perl -e '($x,$y)=@ARGV; $ans=$x / $y; print "$ans";' $tr_loss $iters_per_epoch`;
tr_loss=$(awk "BEGIN{print( $tr_loss/$this_tr_loss )}")

echo "EPOCH $epoch "
echo "TRAIN AVG.LOSS : $(printf "%.4f" $tr_loss) "
echo "CROSSVAL AVG.LOSS $(printf "%.4f" ${loss[0]}) ACC $(printf "%.4f" ${loss[1]}) "
(
echo "EPOCH $epoch"
echo "TRAIN AVG.LOSS : $(printf "%.4f" $tr_loss) "
echo "CROSSVAL AVG.LOSS $(printf "%.4f" ${loss[0]}) ACC $(printf "%.4f" ${loss[1]}) "
) >> $logfile

# stopping criteria
  

echo "${loss[0]} ${loss_prev[0]}"
echo ""

if [ 1 == $(bc <<< "${loss[0]} < ${loss_prev[0]}" ) ]; then
	if [ "$epoch" -gt "$keep_lr_epoch" ]; then
	 num_stop_halving=$[$num_stop_halving+1]
	 learning_rate=$(awk "BEGIN{print($learning_rate*$halving_factor)}")
	 mdl_temp=$dir/temp.mdl
	 echo "learning rate : $learning_rate, having factor : $halving_factor [ $num_stop_halving / $max_stop_halving ] "
	  (echo "learning rate : $learning_rate, having factor : $halving_factor [ $num_stop_halving / $max_stop_halving ] "
	   ) >> $logfile
	   mdl_best=$dir/final.mdl

if $diff_lrate_scale ; then
 nnet-am-copy --learning-rate-factor=$learning_rate --learning-rates=$learning_rate_scale $mdl_best $mdl_temp || exit 1;
else
	nnet-am-copy --learning-rate=$learning_rate $mdl_best $mdl_temp || exit 1;
fi
	   cp $mdl_temp $dir/final.mdl
	   mdl_best=$dir/final.mdl

	 if [ $num_stop_halving -eq $max_stop_halving ]; then
	  mv $dir/log/train.$epoch.1.1.log $dir/log/train.$epoch.log || exit 1;
	  rm $dir/log/train.$epoch.*.*.log || exit 1
	   break;
	 fi
      fi
else
  mdl_best=$mdl
  loss_prev[0]=${loss[0]}  || exit 1;
  loss_prev[1]=${loss[1]}  || exit 1;
    cp $mdl_best $dir/final.mdl
fi

  mv $dir/log/train.$epoch.1.1.log $dir/log/train.$epoch.log || exit 1;
  rm $dir/log/train.$epoch.*.*.log || exit 1;

  echo "$learning_rate" > $dir/lr
  echo "${loss_prev[0]}" > $dir/loss_prev
  echo "$num_stop_halving" > $dir/stopping

done
rm $mdl_temp

if $dropout_net ; then
	nnet-am-copy --scales=$dropout_scale --remove-dropout="true" $mdl_best $dir/final.test.mdl || exit 1;
elif $prob_relu; then
	nnet-am-copy --expectation-probReLU="true" $mdl_best $dir/final.test.mdl || exit 1;
else
	cp $dir/final.mdl $dir/final.test.mdl
fi

touch $dir/.done


