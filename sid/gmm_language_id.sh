#!/bin/bash

# Copyright    2015  SST Lab
#              
# Apache 2.0.

# This script gets language-id information for a set of utterances.
# The output is a vld (voiced language detection) file that is similar to the vad outputs.
# The only difference is that instead of labeling 1 for voiced regions, the vld file
# labels the language ids in the voiced region. The language ids are 1 if it is the
# target language and a -ve integer if it is any other language. The sil regions
# are labeled as 0 as in vad outputs.

# Begin configuration section.
nj=10
cmd="run.pl"
stage=-4
num_gselect1=20 # Gaussian-selection using diagonal model: number of Gaussians to select
num_gselect2=3 # Gaussian-selection using full-covariance model.
tlang_prior=0.7
olang1_prior=0.25
olang2_prior=0.05
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: $0 <language-independent-ubm-dir> <targetlang-ubm-dir> <english-ubm-dir> <music-ubm-dir> <data> <exp-dir>"
  echo " "
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --tlang-prior <p|0.7>                            # Prior probability of target language"
  echo "  --olang1-prior <p|0.025>                         # Prior probability of other language 1"
  echo "  --olang2-prior <p|0.05>                          # Prior probability of other language 2"
  echo "  --cleanup <true,false|true>                      # If true, clean up temporary files"
  echo "  --num-processes <n|4>                            # Number of processes for each queue job (relates"
  echo "                                                   # to summing accs in memory)"
  echo "  --num-threads <n|4>                              # Number of threads for each process (can't be usefully"
  echo "                                                   # increased much above 4)"
  echo "  --stage <stage|-4>                               # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --sum-accs-opt <option|''>                       # Option e.g. '-l hostname=a15' to localize"
  echo "                                                   # sum-accs process to nfs server."
  exit 1;
fi

ubmdir=$1
tlang_ubmdir=$2
olang1_ubmdir=$3
olang2_ubmdir=$4
data=$5
dir=$6

echo -e "1=$1\n2=$2\n3=$3\n4=$4\n5=$5\n6=$6";

# $ubmdir/delta_opts should already be present as a result of running sid/train_full_ubm.sh
delta_opts=`cat $ubmdir/delta_opts 2>/dev/null`
#if [ -f $ubmdir/delta_opts ]; then
#  cp $ubmdir/delta_opts $tlang_ubmdir/ 2>/dev/null
#  cp $ubmdir/delta_opts $olang1_ubmdir/ 2>/dev/null
#  cp $ubmdir/delta_opts $olang2_ubmdir/ 2>/dev/null
#fi

for f in $ubmdir/final.ubm $tlang_ubmdir/final.ubm $olang1_ubmdir/final.ubm $olang2_ubmdir/final.ubm  $data/feats.scp $data/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log || exit 1;
sdata=$data/split$nj
utils/split_data.sh $data $nj || exit 1;

ng1=$(fgmm-global-info --print-args=false $ubmdir/final.ubm | grep gaussians | awk '{print $NF}')
ng2=$(fgmm-global-info --print-args=false $tlang_ubmdir/final.ubm | grep gaussians | awk '{print $NF}')
ng3=$(fgmm-global-info --print-args=false $olang1_ubmdir/final.ubm | grep gaussians | awk '{print $NF}')
ng4=$(fgmm-global-info --print-args=false $olang2_ubmdir/final.ubm | grep gaussians | awk '{print $NF}')
if ! [ $ng1 -eq $ng2 ] || ! [ $ng1 -eq $ng3 ] || ! [ $ng1 -eq $ng4 ]; then
  echo "$0:  Number of Gaussians mismatch between language-independent ($ng1), target language dependent ($ng2)"
  echo "$0:  and other language dependent UBMs ($ng3, $ng4)"
  exit 1;
fi


## Set up features.
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

if [ $stage -le -2 ]; then
  $cmd $dir/log/convert.log \
    fgmm-global-to-gmm $ubmdir/final.ubm $dir/final.dubm || exit 1;
fi

# Do Gaussian selection using diagonal form of model and then the full-covariance model.
# Even though this leads to, in some sense, less accurate likelihoods, I think it
# may improve the results for the same reason it sometimes helps to used fixed
# Gaussian posteriors rather than posteriors from the adapted model.

# Amit: Using diag ubm, select top $num_gselect1 mix components per frame. This is pre-selection of mixture components. 
# Then, using the full ubm, select the top $num_gselect2 mix components per frame but this time the top $num_gselect2 
# mix components are chosen from the limited set of mix components that were pre-selected using diag ubm.
if [ $stage -le -1 ]; then
  echo $nj > $dir/num_jobs
  echo "$0: doing Gaussian selection"
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect1 $dir/final.dubm "$feats" ark:- \| \
    fgmm-gselect --gselect=ark,s,cs:- --n=$num_gselect2 $ubmdir/final.ubm \
      "$feats" "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

if ! [ $nj -eq $(cat $dir/num_jobs) ]; then
  echo "Number of jobs mismatch" 
  exit 1;
fi

mkdir -p $dir/gmm

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/get_tlang_logprob.JOB.log \
    fgmm-global-get-frame-likes --average=false \
     "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $tlang_ubmdir/final.ubm \
      "$feats" ark,t:$dir/gmm/tlang_logprob.JOB || exit 1;
fi
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/get_olang1_logprob.JOB.log \
    fgmm-global-get-frame-likes --average=false \
     "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $olang1_ubmdir/final.ubm \
      "$feats" ark,t:$dir/gmm/olang1_logprob.JOB || exit 1;
fi
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/get_olang2_logprob.JOB.log \
    fgmm-global-get-frame-likes --average=false \
     "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $olang2_ubmdir/final.ubm \
      "$feats" ark,t:$dir/gmm/olang2_logprob.JOB || exit 1;
fi


if [ $stage -le 2 ]; then

  for j in $(seq $nj); do cat $dir/gmm/tlang_logprob.$j; done > $dir/gmm/tlang_logprob
  for j in $(seq $nj); do cat $dir/gmm/olang1_logprob.$j; done > $dir/gmm/olang1_logprob
  for j in $(seq $nj); do cat $dir/gmm/olang2_logprob.$j; done > $dir/gmm/olang2_logprob

  n1=$(cat $dir/gmm/tlang_logprob | wc -l)
  n2=$(cat $dir/gmm/olang1_logprob | wc -l)
  n3=$(cat $dir/gmm/olang2_logprob | wc -l)

  if [ $n1 -ne $n2 ] || [ $n1 -ne $n3 ]; then
    echo "Number of lines mismatch, target versus other language UBM probs: $n1 vs $n2 vs $n3"
    exit 1;
  fi
  
  sed -i 's:\[\|\]: :g' $dir/gmm/tlang_logprob
  sed -i 's:\[\|\]: :g' $dir/gmm/olang1_logprob
  sed -i 's:\[\|\]: :g' $dir/gmm/olang2_logprob
  
  #paste $dir/tlang_logprob $dir/olang1_logprob $dir/olang2_logprob | \
    #awk '{if ($1 != $3 || $1 != $5) { print >/dev/stderr "Sorting mismatch"; exit(1);  } print $1, $2, $4, $6;}' \
    #>$dir/logprob || exit 1;
  
  #cat $dir/logprob | \
   #awk -v ptlang=$tlang_prior -v polang1=$olang1_prior -v  polang2=$olang2_prior  '{ a1 = log(ptlang/polang1) + $2 - $3; a2 = log(ptlang/polang2) + $2 - $4; post_a = 1/(1 + exp(-a1) + exp(-a2));  b1 = log(polang1/ptlang) + $3 - $2;  b2 = log(polang1/polang2) + $3 - $4; post_b = 1/(1 + exp(-b1) + exp(-b2)); print $1, post_a, post_b, 1 - post_a - post_b}' \
    #>$dir/ratio || exit 1;     
  
  perl sid/get_best_langid_perframe.pl $dir/gmm/tlang_logprob $tlang_prior $dir/gmm/olang1_logprob $olang1_prior $dir/gmm/olang2_logprob $olang2_prior > $dir/gmm/frame_labels
   
  merge-vad-with-frame-labels "scp:$data/vad.scp" "ark,t:$dir/gmm/frame_labels" "ark,t:$dir/gmm/vld" || exit 1;
   
fi

exit 0;

