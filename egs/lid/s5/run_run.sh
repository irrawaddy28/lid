#!/bin/bash
# convert all files in www.sbs.com.au/ from mp3 -> wav
perl local/sbs_mp3towav.pl  /ws/ifp-48_1/hasegawa/amitdas/corpus/www.sbs.com.au/mp3 /ws/ifp-48_1/hasegawa/amitdas/corpus/www.sbs.com.au/wav

bash run.sh 1 # data prep
bash run.sh 2 # features
bash run.sh 3 # generate subsets of lang specific data 
bash run.sh 4 # generate smaller subsets of trn data 
bash run.sh 5 # create the lang-independent full ubm
bash run.sh 6 # lang dependent versions from the lang-independent ubm
bash run.sh 7 # gmm language id + wav file segmentation according to language id 
bash run.sh 8 # hmm language id + wav file segmentation according to language id
bash run.sh 9 # wav file segmentation according to language id decisions
