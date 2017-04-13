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
test_mdl=


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
    $cmd $dir/log/nnet_init.log \
    nnet-am-init $alidir/tree $lang/topo "nnet-init $dir/nnet.config -|" -\| \
    nnet-am-copy --learning-rate=$learning_rate - $dir/0.mdl || exit 1;
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

# cross-validation on original network
nj_valid=`cat $alidir_valid/num_jobs`
 
 $cmd $parallel_opts JOB=1:$nj_valid $dir/log/test.JOB.log \
nnet-compute-prob $dir/$test_mdl ark:$egs_dir/valid_egs.JOB.ark || exit 1;


