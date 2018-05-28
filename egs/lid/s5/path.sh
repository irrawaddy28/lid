#default settings for steps/nnet/{pretrain_dbn.sh, train.sh}
skip_cuda_check=false 
my_use_gpu=yes
corpus_dir=/ws/rz-cl-2/hasegawa/amitdas/corpus

#host dependendent settings 
if [ `hostname` = "ifp-48" ]; then
	export KALDI_ROOT=../../../
elif [ `hostname` = "ifp-30" ]; then
	export KALDI_ROOT=/media/data/workspace/gold/kaldi/kaldi-trunk
	skip_cuda_check=true
	my_use_gpu=no
else
	echo "Unidentified hostname `hostname`"; exit 1;
fi

export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$PWD:$PATH
export LC_ALL=C
export IRSTLM=$KALDI_ROOT/tools/irstlm
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
