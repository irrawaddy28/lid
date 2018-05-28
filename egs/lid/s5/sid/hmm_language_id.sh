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
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 8 ]; then
  echo "Usage: $0 <language-independent-ubm-dir> <targetlang-ubm-dir> <english-ubm-dir> <music-ubm-dir> <data> <exp-dir>"
  echo " "
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"  
  echo "  --cleanup <true,false|true>                      # If true, clean up temporary files"
  echo "  --num-processes <n|4>                            # Number of processes for each queue job (relates"
  echo "                                                   # to summing accs in memory)"
  echo "  --num-threads <n|4>                              # Number of threads for each process (can't be usefully"
  echo "                                                   # increased much above 4)"
  echo "  --stage <stage|-4>                               # To control partial reruns"  
  exit 1;
fi

tlang_ubmdir=$1    #target language ubm dir
olang1_ubmdir=$2 # impostor lang 1 ubm dir
olang2_ubmdir=$3 # impostor lang 2 ubm dir
transmdl=$4  # transition model (HMM topo + log (trans probs))
tree=$5      # tree structure
data=$6
lang=$7
dir=$8

#echo -e "1=$1\n2=$2\n3=$3\n4=$4\n5=$5\n6=$6\n7=$7\n8=$8";

for f in ${tlang_ubmdir}/final.ubm ${olang1_ubmdir}/final.ubm ${olang2_ubmdir}/final.ubm $transmdl $tree $data/feats.scp $data/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
delta_opts=`cat ${tlang_ubmdir}/delta_opts 2>/dev/null`

mkdir -p $dir/log || exit 1;
nj=$(cat ${tlang_ubmdir}/num_jobs)
[[ -z $nj ]] && { echo "Either ${tlang_ubmdir}/num_jobs does not exist or is empty" ; exit 1; }
sdata=$data/split$nj

## Set up features.
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"
example_feats="`echo $feats | sed s/JOB/1/g`";
feat_dim=`feat-to-dim "$example_feats" - 2>/dev/null`

# Convert target language full cov ubm to diag cov ubm 
fgmm-global-to-gmm --binary=false ${tlang_ubmdir}/final.ubm ${tlang_ubmdir}/final.dubm 2>/dev/null || exit 1

# Merge: target lang diag ubm + other langs diag ubms + transition model = HMM
numgmms=3
hmm=$dir/hmm/final.mdl
rm -rf $dir/hmm 2>/dev/null
mkdir -p $dir/hmm
echo ${delta_opts} > $dir/hmm/delta_opts
echo "<DIMENSION> ${feat_dim} <NUMPDFS> $numgmms "|\
cat  $transmdl - ${tlang_ubmdir}/final.dubm ${olang1_ubmdir}/final.dubm ${olang2_ubmdir}/final.dubm > $hmm

# HCLG graph
local/mkgraph_HCLG.sh --mono --loop-scale 1 $lang $tree $hmm $dir/hmm

# Decode and generate alignments
local/decode.sh --nj "$nj" --cmd "$decode_cmd" --skip-scoring "true" \
                --lattice-beam 0.5 --determinize-lattice "false" $dir/hmm $data $dir/hmm/decode

# Convert ali to pdf                
$decode_cmd JOB=1:$nj $dir/log/ali2pdf.JOB.log \
	ali-to-pdf $hmm "ark,t:gunzip -c $dir/hmm/decode/ali.JOB.gz|" "ark,t:$dir/hmm/decode/alipdf.JOB" || exit 1;       

for j in $(seq $nj); do cat $dir/hmm/decode/alipdf.$j; done > $dir/hmm/decode/alipdf	

# Convert pdf to frame labels 
# 0 = taget, 1 = olang1, 2 = olang2 =>  1 = target, -1 = olang1, -2 = olang2
perl sid/pdf2label.pl --map-from-col 1 --map-to-col 2 conf/pdf2label.map $dir/hmm/decode/alipdf > $dir/hmm/frame_labels

# Merge vad with frame labels                
merge-vad-with-frame-labels "scp:$data/vad.scp" "ark,t:$dir/hmm/frame_labels" "ark,t:$dir/hmm/vld" || exit 1;

exit 0;

