#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

echo "$0 $@"  # Print the command line for logging

stage=0 # resume training with --stage=N
max_nj_decode=10 
transform_dir=
num_trn_utt=
#precomp_feat_transform=
precomp_dbn=
post_fix=""
train_iters=20
l2_iters=2
use_delta=false
minibatch_size=256
# End of config.
[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;
#

if [ $# != 2 ]; then   
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <transform-dir>                  # where to find fMLLR transforms."
   echo "  --num-trn-utt <n>                                # number of utts in train set."
   echo "  --post-fix   <string>                            # add a post-fix string to exp/dnn dir" 
   echo "  --precomp-dbn <pre-computed dbn dir>             # pre-computed dbn dir that can be used for DNN training"
   echo "  --train-iters <N>                                # number of nnet training iterations for l1"
   echo "  --l2-iters <N>                                    # number of nnet training iterations for l2"
   echo "  --use-delta     <bool> 							# if set to true, will use mfcc + delta feats only, forcibly ignore transforms"
   echo "  --minibatch-size <N>                             # num of frames reqd. to perform parameter update in minibatch SGD"   
   exit 1;
fi

# Config:
langwts_config=$1
gmmdir=$2  #exp/tri3
data_fmllr=data-fmllr-$(basename $gmmdir)   #data-fmllr-tri3

[[ ! -z $post_fix ]] && post_fix="_$post_fix"
post_fix=$(basename $gmmdir)$post_fix
echo "user i/p fMMLR transform dir = $transform_dir";

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # test  
  dir=$data_fmllr/test
  [[ ! -z $transform_dir ]] && transform_dir_opt="--transform-dir $transform_dir/decode_test" || transform_dir_opt=""
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" --use-delta $use_delta \
     $transform_dir_opt \
     $dir data/test $gmmdir $dir/log $dir/data || exit 1
  # dev
  dir=$data_fmllr/dev
  [[ ! -z $transform_dir ]] && transform_dir_opt="--transform-dir $transform_dir/decode_dev" || transform_dir_opt=""
  steps/nnet/make_fmllr_feats.sh --nj 5 --cmd "$train_cmd" --use-delta $use_delta \
     $transform_dir_opt \
     $dir data/dev $gmmdir $dir/log $dir/data || exit 1
  # train (l1 - target lang)
  dir=$data_fmllr/train
  [[ ! -z $transform_dir ]] && transform_dir_opt="--transform-dir ${transform_dir}_ali" || transform_dir_opt=""
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" --use-delta $use_delta \
     $transform_dir_opt \
     $dir data/train${num_trn_utt} $gmmdir $dir/log $dir/data || exit 1
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
     
  # train (l2 - source lang)
  data_fmllr_lang=data-fmllr-lang
  dir=${data_fmllr_lang}/train  
  steps/tl/nnet/make_fmllr_feats_lang.sh --nj 10 --cmd "$train_cmd" --use-delta $use_delta \
    $dir $langwts_config "tri3" || exit 1 
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1      
fi

if [ $stage -le 1 ]; then
  if [[ -z ${precomp_dbn} ]]; then 
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=exp/dnn4_pretrain-dbn${post_fix}
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 20 $data_fmllr/train $dir || exit 1;
  # Use the feature transform from the dbn directory
  feature_transform=$dir/final.feature_transform
  feature_transform_opt=$(echo "--feature-transform $feature_transform")
  else
  [[ ! -d ${precomp_dbn} ]] && echo "pre-computed dbn directory ${precomp_dbn} does not exist"
  echo "using pre-computed dbn from ${precomp_dbn}"
  dir=exp/dnn4_pretrain-dbn${post_fix}
  mkdir -p $dir
  cp -r ${precomp_dbn}/* $dir
  feature_transform_opt=
  fi
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4_pretrain-dbn_dnn${post_fix}_l2seq
  ali=${gmmdir}_ali
  langali=$ali/langali
  cp $ali/{final.mdl,tree} $langali
  dbn=exp/dnn4_pretrain-dbn${post_fix}/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Step 1: Train NN using L2 data for few iterations
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --splice 5 --splice-step 1  $feature_transform_opt --feat-type "plain" --minibatch-size ${minibatch_size} \
	--nnet-binary "false" --train-iters ${l2_iters} \
    --dbn $dbn --hid-layers 0 --hid-dim 1024 --learn-rate 0.008 \
    ${data_fmllr_lang}/train_tr90 ${data_fmllr_lang}/train_cv10 data/lang $langali $langali $dir || exit 1;    
   mlp_init=$dir/final.nnet
   [ ! -e $mlp_init ] && { echo "$0: Could not find $mlp_init"; exit 1; }
    
  # Step 2: Starting with mlp generated from Step 1, train the NN using L1 data for many iterations
  dir=exp/dnn4_pretrain-dbn_dnn${post_fix}_l1seq
  ali=${gmmdir}_ali
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log  
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --mlp-init $mlp_init --splice 5 --splice-step 1  $feature_transform_opt --feat-type "plain" --minibatch-size ${minibatch_size} \
	--nnet-binary "false" --train-iters ${train_iters} \
    --dbn $dbn --hid-layers 0 --hid-dim 1024 --learn-rate 0.008 \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 data/lang $ali $ali $dir || exit 1;

  # Decode (reuse HCLG graph)
  nj_decode=$(cat conf/dev_spk.list |wc -l); [[ $nj_decode -gt  $max_nj_decode ]] && nj_decode=$max_nj_decode;  
  steps/nnet/decode.sh --nj $nj_decode --cmd "$decode_cmd" --use-gpu no --acwt 0.2 \
    $gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
  
  nj_decode=$(cat conf/test_spk.list |wc -l); [[ $nj_decode -gt  $max_nj_decode ]] && nj_decode=$max_nj_decode; 
  steps/nnet/decode.sh --nj $nj_decode --cmd "$decode_cmd" --use-gpu no --acwt 0.2 \
    $gmmdir/graph $data_fmllr/test $dir/decode_test || exit 1;  
fi


## Sequence training using sMBR criterion, we do Stochastic-GD 
## with per-utterance updates. We use usually good acwt 0.1
#dir=exp/dnn4_pretrain-dbn_dnn_smbr
#srcdir=exp/dnn4_pretrain-dbn_dnn
#acwt=0.2

#if [ $stage -le 3 ]; then
  ## First we generate lattices and alignments:
  #steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    #$data_fmllr/train data/lang $srcdir ${srcdir}_ali || exit 1;
  #steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --acwt $acwt \
    #--lattice-beam 10.0 --beam 18.0 \
    #$data_fmllr/train data/lang $srcdir ${srcdir}_denlats || exit 1;
#fi

#if [ $stage -le 4 ]; then
  ## Re-train the DNN by 6 iterations of sMBR 
  #steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt \
    #--do-smbr true --use-silphones true \
    #$data_fmllr/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  ## Decode
  #for ITER in 1 6; do
    #steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      #--nnet $dir/${ITER}.nnet --acwt $acwt \
      #$gmmdir/graph $data_fmllr/test $dir/decode_test_it${ITER} || exit 1
    #steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      #--nnet $dir/${ITER}.nnet --acwt $acwt \
      #$gmmdir/graph $data_fmllr/dev $dir/decode_dev_it${ITER} || exit 1
  #done 
#fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
