#!/bin/bash
# language id script:
# target languages are all languages present in www.sbs.com.au/ directory
# the target language utterances are corrupted by english and music
# we seek to remove the english and music portions and generate
# segmented wav files containing only the target language


[[ -f path.sh ]] && . ./path.sh || { echo "path.sh does not exist"; exit 1; }
[[ -f cmd.sh ]] && . ./cmd.sh || { echo "cmd.sh does not exist"; exit 1; }

# settings that user can modify
max_seg_length=500 # max length, in number of frames, of the segmented wav files of the target language (500 frames = 5 secs for a skip rate of 10ms)
segwav_dir=$corpus_dir/www.sbs.com.au/seg2 # dir where you want to store the segmented wav files
tlang_prior=0.7    # prior prob. that a target lang speech is present in an utterance
olang1_prior=0.25  # prior prob. that english speech is present in an utterance
olang2_prior=0.05  # prior prob. that music is present in an utterance
# target languages: all languages present in www.sbs.com.au/ 
desired_lang="albanian  bangla   cookislands-maori  dinka     finnish   hebrew      italian   kurdish     malay      norwegian      punjabi   sinhalese  swahili   tongan
amharic   bosnian    croatian    dutch     french    hindi       japanese  lao         malayalam  pashto         romanian  slovak     swedish   turkish
arabic    bulgarian  czech    estonian  german    hmong       kannada   latvian     maltese    persian-farsi  russian   slovenian  tamil     ukrainian
armenian  burmese    danish   fijian    greek     hungarian   khmer     lithuanian  mandarin   polish         samoan    somali     thai      urdu
assyrian  cantonese  dari    filipino  gujarati  indonesian  korean    macedonian  nepali     portuguese     serbian   spanish    tigrinya  vietnamese"
#desired_lang="arabic"
# impostor languages: english and music
other_lang="english  music"

# the main script
set -e
stage=$1

sbs=$corpus_dir/www.sbs.com.au/wav
wsj0=$corpus_dir/wsj/wsj0
wsj1=$corpus_dir/wsj/wsj1
music=$corpus_dir/music/wav_mono

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


if [ $stage -eq 1 ]; then
# data prep

#local/make_fisher.sh /media/data/workspace/corpus/fisher/{LDC2004S13,LDC2004T19} data/fisher1
#local/make_fisher.sh /media/data/workspace/corpus/fisher/{LDC2005S13,LDC2005T19} data/fisher2

local/make_sbs.sh $sbs data/sbs
local/make_wsj.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.? data/english
local/make_music.sh $music data/music

utils/combine_data.sh data/train data/sbs data/english data/music
fi

if [ $stage -eq 2 ]; then
# Generate features
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/train exp/make_mfcc $mfccdir

sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
    data/train exp/make_vad $vaddir

# Note: to see the proportion of voiced frames you can do,
# grep Prop exp/make_vad/vad_*.1.log 
fi

if [ $stage -eq 3 ]; then
# Get language specific subsets of training data.
for l in `echo $other_lang $desired_lang`
do
	echo $l > tmp
	utils/subset_data_dir.sh --spk-list tmp data/train data/train_${l} 
done
rm tmp
fi


if [ $stage -eq 4 ]; then
# Get smaller subsets of training data for faster training.
utils/subset_data_dir.sh data/train 8000 data/train_8k
fi


if [ $stage -eq 5 ]; then
# init the diag ubm and from it generate the full ubm
# The recipe currently uses delta-window=3 and delta-order=2. However
# the accuracy is almost as good using delta-window=4 and delta-order=1
# and could be faster due to lower dimensional features.  Alternative
# delta options (e.g., --delta-window 4 --delta-order 1) can be provided to

nj=`echo $other_lang $desired_lang|tr ' ' '\n'|wc -l`
sid/train_diag_ubm.sh --nj $nj --cmd "$train_cmd" data/train_8k 2048 \
    exp/diag_ubm_2048

sid/train_full_ubm.sh --nj $nj --cmd "$train_cmd" data/train \
    exp/diag_ubm_2048 exp/full_ubm_2048
fi    

if [ $stage -eq 6 ]; then
# Get lang dependent versions of the UBM in one pass; make sure not to remove
# any Gaussians due to low counts (so they stay matched).  This will be 
# more convenient for language-id.

nj=`echo $other_lang $desired_lang|tr ' ' '\n'|wc -l`
for l in `echo $other_lang $desired_lang`
do
	echo "language ${l}: Building exp/full_ubm_2048_${l} from exp/full_ubm_2048 and data/train_${l} using Gaussian pruning ... ";
	# increase --num-gselect??? 20 by default. 
	sid/train_full_ubm.sh --nj 1 --remove-low-count-gaussians false \
		--num-iters 1 --cmd "$train_cmd" \
	data/train_${l} exp/full_ubm_2048 exp/full_ubm_2048_${l} &	
done	
wait
fi

if [ $stage -eq 7 ]; then
# Now do GMM based language-id.
vldtype=gmm 
for l in `echo $desired_lang`
do
{
	echo "Starting GMM based language id for language=$l";
	t1=$(date +"%s");
	sid/gmm_language_id.sh --cmd "$train_cmd" --nj 1 --tlang-prior $tlang_prior --olang1-prior $olang1_prior --olang2-prior $olang2_prior \
		exp/full_ubm_2048{,_${l},_english,_music} data/train_${l} exp/language_id_train_${l}
	t2=$(date +"%s");
	diff=$(($t2-$t1));
	echo "Done language id for language=$l after $(($diff / 60)) min $(($diff % 60)) secs";	
} &
done
wait
echo "Done all language ids"

# segment the wav files into chunks of contiguous frames of the target language id. 
for l in `echo $desired_lang`
do
local/extract_segments.sh --cmd "$train_cmd" --nj 1 --max-seg-length $max_seg_length data/train_${l} $segwav_dir exp/language_id_train_${l}/${vldtype}
done

fi

if [ $stage -eq 8 ]; then
# Now do HMM based language-id. 
vldtype=hmm
# Make graph LoG: copy everything from conf/lang to data/lang
conflang=conf/lang 
lang=data/lang
[[ -f $conflang/phones.txt ]] || { echo "$conflang/phones.txt does not exist "; exit 1; } 
[[ -f $conflang/words.txt ]] || { echo "$conflang/words.txt does not exist "; exit 1; } 
[[ -f $conflang/lexicon.txt ]] || { echo "$conflang/lexicon.txt does not exist "; exit 1; } 
[[ -f $conflang/grammar.txt ]] || { echo "$conflang/grammar.txt does not exist "; exit 1; } 
rm -rf $lang/* 2>/dev/null 
[[ ! -d $lang ]] && mkdir -p $lang 
cp -r $conflang/* $lang
local/mkgraph_LG.sh $lang

# Convert full ubms to diag ubms for the impostor langs. (Kaldi supports only diag ubms for modeling hmms)
for l in `echo $other_lang`
do	
	fgmm-global-to-gmm --binary=false exp/full_ubm_2048_${l}/final.ubm exp/full_ubm_2048_${l}/final.dubm 2>/dev/null || exit 1;
done

# Build the hmms and do lang classification
# hmm = taget lang diag ubm + other lang diag ubms + transition model 
transmdl=conf/mdl/trans.mdl
tree=conf/mdl/tree
for l in `echo $desired_lang`
do
{
	echo "Starting HMM based language id for language=$l";
	t1=$(date +"%s");	
	sid/hmm_language_id.sh --cmd "$train_cmd" \
		exp/full_ubm_2048{_${l},_english,_music} $transmdl $tree \
		data/train_${l} $lang exp/language_id_train_${l}
	t2=$(date +"%s");
	diff=$(($t2-$t1));
	echo "Done language id for language=$l after $(($diff / 60)) min $(($diff % 60)) secs";	
} &
done
wait
echo "Done all language ids"

# segment the wav files into chunks of contiguous frames of the target language id. 
for l in `echo $desired_lang`
do
local/extract_segments.sh --cmd "$train_cmd" --nj 1 --max-seg-length $max_seg_length data/train_${l} $segwav_dir exp/language_id_train_${l}/${vldtype}
done

fi

#if [ $stage -eq 9 ]; then
## segment the wav files into chunks where contiguous frames of the target language id has been detected. 
#for l in `echo $desired_lang`
#do
#local/extract_segments.sh --cmd "$train_cmd" --nj 1 --max-seg-length $max_seg_length data/train_${l} $segwav_dir exp/language_id_train_${l}/${vldtype}
#done  
#fi
