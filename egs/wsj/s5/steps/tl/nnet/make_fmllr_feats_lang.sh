#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely),
#                 
# Apache 2.0.
#
# This script dumps fMLLR features in a new data directory, 
# which is later used for neural network training/testing.

# Begin configuration section.  
nj=4
cmd=run.pl
use_delta=false
use_transform=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <tgt-data-dir> <l2 config file> <gmm-dir-name> "
   echo "e.g.: $0 data-fmllr-lang/train conf/l2.conf tri3"
   echo ""
   echo "This script generates CMN + (delta+delta-delta | LDA+MLLT) or fMLLR features"
   echo "for each language present in the language config file"   
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."   
   echo "  --use-delta     <bool> 							# if set to true, will use mfcc + delta feats only, forcibly ignore fMLLR and LDA"
   echo "  --use-transform <bool>                           # if true, use language specific fMLLR transforms. If false, use lda"
   exit 1;
fi

dir=$1
langwts_config=$2
gmmdirtype=$3  # only name of dir, not the entire path

nlang=`cat $langwts_config| wc -l`;
i=0
while read line 
do
   str[i]=$line       
   lang[i]=$(echo ${str[i]}| cut -d' ' -f1)
   [[ ! -d ${lang[$i]} ]] && { echo "${lang[$i]} does not exist"; exit 1; }
   i=$(($i + 1))
done < $langwts_config

data_fmllr_base=$(dirname $dir)
for (( i=0; i < $nlang; i++ ))
do 	
	gmmdir=${lang[$i]}/exp/$gmmdirtype;
	data=${lang[$i]}/data/train;	
	data_fmllr_lang=${data_fmllr_base}_${i}/train
	
	$use_delta && use_transform=false
	$use_transform && transform_dir_opt="--transform-dir ${gmmdir}_ali" || transform_dir_opt=""
	
	steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$cmd" --use-delta $use_delta \
     $transform_dir_opt $data_fmllr_lang $data $gmmdir $data_fmllr_lang/log $data_fmllr_lang/data || exit 1		
done

utils/combine_data.sh $dir ${data_fmllr_base}_*/train || exit 1


exit 0;
