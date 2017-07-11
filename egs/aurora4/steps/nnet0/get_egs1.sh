#!/bin/bash

# Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)  Apache 2.0.
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in separate archives.

# Begin configuration section.
cmd=run.pl
feat_type=
samples_per_iter=200000 # each iteration of training, see this many samples
                        # per job.  This is just a guideline; it will pick a number
                        # that divides the number of samples in the entire data.
num_jobs_nnet=16    # Number of neural net jobs to run in parallel
stage=0
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time. 
splice_width=4 # meaning +- 4 frames on each side
left_context=
right_context=
random_copy=false

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: steps/nnet2/get_egs.sh [opts] <data> <lang> <train-ali-dir> <valid-data> <valid-ali-dir> <egs-dir>"
  echo " e.g.: steps/nnet2/get_egs.sh data/train data/lang exp/4b_ali_si284 data/test_dev93 exp/4b_ali_dev93 exp/nnet-fbank/egs"
  echo "" 
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$6
data_valid=$4
alidir_valid=$5

[ -z "$left_context" ] && left_context=$splice_width
[ -z "$right_context" ] && right_context=$splice_width


# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $alidir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
for f in $data_valid/feats.scp $lang/L.fst $alidir_valid/ali.1.gz $alidir_valid/final.mdl $alidir_valid/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Set some variables.
oov=`cat $lang/oov.int`
num_leaves=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/number of pdfs/{print $NF}'` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj
utils/split_data.sh $data $nj

nj_valid=`cat $alidir_valid/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata_valid=$data_valid/split$nj_valid
utils/split_data.sh $data_valid $nj_valid

mkdir -p $dir/log
cp $alidir/tree $dir


[ -z "$transform_dir" ] && transform_dir=$alidir
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
cp $alidir/cmvn_opts $dir 2>/dev/null

#####################
# Set up features.
feat_type=raw
echo "$0: feature type is $feat_type"

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
    valid_feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_valid/JOB/utt2spk scp:$sdata_valid/JOB/cmvn.scp scp:$sdata_valid/JOB/feats.scp ark:- |"
   ;; 
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac


if [ $stage -le 0 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/num_frames
else
  num_frames=`cat $dir/num_frames` || exit 1;
fi

# Working out number of iterations per epoch.
iters_per_epoch=`perl -e "print int($num_frames/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
[ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
samples_per_iter_real=$[$num_frames/($num_jobs_nnet*$iters_per_epoch)]
echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations,"
echo "$0: giving samples-per-iteration of $samples_per_iter_real (you requested $samples_per_iter)."

# Making soft links to storage directories.  This is a no-up unless
# the subdirectory $dir/storage/ exists.
for x in `seq 1 $num_jobs_nnet`; do
  for y in `seq 0 $[$iters_per_epoch-1]`; do
    utils/create_data_link.pl $dir/egs.$x.$y.ark
    utils/create_data_link.pl $dir/egs_tmp.$x.$y.ark
  done
  for y in `seq 1 $nj`; do
    utils/create_data_link.pl $dir/egs_orig.$x.$y.ark
  done
done

remove () { for x in $*; do [ -L $x ] && rm $(readlink -f $x); rm $x; done }

nnet_context_opts="--left-context=$left_context --right-context=$right_context"
mkdir -p $dir/egs

if [ ! -z $spk_vecs_dir ]; then
  [ ! -f $spk_vecs_dir/vecs.1 ] && echo "No such file $spk_vecs_dir/vecs.1" && exit 1;
  spk_vecs_opt=("--spk-vecs=ark:cat $spk_vecs_dir/vecs.*|" "--utt2spk=ark:$data/utt2spk")
else
  spk_vecs_opt=()
fi

if [ $stage -le 1 ]; then
  mkdir -p $dir/temp

  # Other scripts might need to know the following info:
  echo $num_jobs_nnet >$dir/num_jobs_nnet
  echo $iters_per_epoch >$dir/iters_per_epoch
  echo $samples_per_iter_real >$dir/samples_per_iter

  echo "Creating training examples";
  # in $dir, create $num_jobs_nnet separate files with training examples.
  # The order is not randomized at this point.

  egs_list=
  for n in `seq 1 $num_jobs_nnet`; do
    egs_list="$egs_list ark:$dir/egs_orig.$n.JOB.ark"
  done
  echo "Generating training examples on disk"
  # The examples will go round-robin to egs_list.
  $cmd $io_opts JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet-get-egs $nnet_context_opts "${spk_vecs_opt[@]}" "$feats" \
    "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    nnet-copy-egs ark:- $egs_list || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: rearranging examples into parts for different parallel jobs"
  # combine all the "egs_orig.JOB.*.scp" (over the $nj splits of the data) and
  # then split into multiple parts egs.JOB.*.scp for different parts of the
  # data, 0 .. $iters_per_epoch-1.

  if [ $iters_per_epoch -eq 1 ]; then
    echo "$0: Since iters-per-epoch == 1, just concatenating the data."
    for n in `seq 1 $num_jobs_nnet`; do
      cat $dir/egs_orig.$n.*.ark > $dir/egs_tmp.$n.0.ark || exit 1;
      remove $dir/egs_orig.$n.*.ark 
    done
  else # We'll have to split it up using nnet-copy-egs.
    egs_list=
    for n in `seq 0 $[$iters_per_epoch-1]`; do
      egs_list="$egs_list ark:$dir/egs_tmp.JOB.$n.ark"
    done
    # note, the "|| true" below is a workaround for NFS bugs
    # we encountered running this script with Debian-7, NFS-v4.
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/split_egs.JOB.log \
      nnet-copy-egs --random=$random_copy --srand=JOB \
        "ark:cat $dir/egs_orig.JOB.*.ark|" $egs_list || exit 1;
    remove $dir/egs_orig.*.*.ark  2>/dev/null
  fi
fi

if [ $stage -le 3 ]; then
  # Next, shuffle the order of the examples in each of those files.
  # Each one should not be too large, so we can do this in memory.
  echo "Shuffling the order of training examples"
  echo "(in order to avoid stressing the disk, these won't all run at once)."


  # note, the "|| true" below is a workaround for NFS bugs
  # we encountered running this script with Debian-7, NFS-v4.
  for n in `seq 0 $[$iters_per_epoch-1]`; do
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/shuffle.$n.JOB.log \
      nnet-shuffle-egs "--srand=\$[JOB+($num_jobs_nnet*$n)]" \
      ark:$dir/egs_tmp.JOB.$n.ark ark:$dir/egs.JOB.$n.ark 
    remove $dir/egs_tmp.*.$n.ark
  done
fi


if [ $stage -le 4 ]; then
  echo "Creating validation examples."

  egs_list="ark:$dir/valid_egs.JOB.ark"

  # The examples will go round-robin to egs_list.
  $cmd $io_opts JOB=1:$nj_valid $dir/log/create_valid_egs.JOB.log \
    nnet-get-egs $nnet_context_opts "${spk_vecs_opt[@]}" "$valid_feats" \
    "ark,s,cs:gunzip -c $alidir_valid/ali.JOB.gz | ali-to-pdf $alidir_valid/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    nnet-copy-egs ark:- $egs_list || exit 1;
fi

echo "$0: Finished preparing training examples"
