#!/bin/bash
#
# This script gets language-id information for a set of utterances.
# The output is a vld (voiced language detection) file that is similar to the vad output file.
# The only difference b/w vld and vad file is: In vad file, we label 1 for voiced regions. 
# In vld file, the labels are the language ids in the voiced region. The language ids are 1 if it is the
# target language and a -ve integer if it is any other language. The sil regions
# are labeled as 0 as in vad file.

# Begin configuration section.
cmd="run.pl"
nj=1;
max_seg_length=500
frame_offset=0.01  #10ms
frame_length=0.025  #25ms
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 "
  echo " e.g.: $0  "
  echo "main options (for others, see top of script file)"  
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"  
  echo "  --max-seg-length <n|500>                         # Max number of frames per segment"
  echo "  --frame-offset   <f|0.01>                        # Skip rate of each frame in milliseconds"  
  echo "  --frame-length   <f|0.025>                        # Frame length in milliseconds"
  exit 1;
fi

data=$1
segwav_dir=$2
dir=$3

echo -e "1=$1\n2=$2\n3=$3";

[[ ! -d $data ]] && echo "No such directory $data" && exit 1;
[[ ! -d $dir ]] && echo "No such directory $dir" && exit 1;

mkdir -p $segwav_dir

# create the segment file
# e.g. lines: 
# arabic_arabic_140912_360279-1 arabic_arabic_140912_360279 38508 39066
# arabic_arabic_140912_360279-2 arabic_arabic_140912_360279 89339 89863
create-split-from-vld --max-voiced=$max_seg_length "ark,t:$dir/vld" -| \
	awk -v frame_off=$frame_offset -v frame_len=$frame_length \
	'{
		start_time = ($3-1)*frame_off; 
		start_time = (start_time >= 0 ? start_time : 0); 
		end_time = ($4 - 1)*frame_off + frame_len; 
		print $1," ",$2," ",start_time," ",end_time
	 }' > $dir/segments

# create the output scp file from the segments file
# e.g. lines
# arabic_arabic_140912_360279-1   /media/data/workspace/corpus/www.sbs.com.au/seg/arabic/arabic_140912_360279-1.wav
# arabic_arabic_140912_360279-2   /media/data/workspace/corpus/www.sbs.com.au/seg/arabic/arabic_140912_360279-2.wav
export segwav_dir; 
awk '{print $1}' $dir/segments| \
perl -ne '$key=$lang=$_; chomp $lang; chomp $key; $lang =~ s/(.*?)_(.*)/$1/; $langdir = $ENV{segwav_dir}."/".$lang; +
		  unlink glob "$langdir/*.*"; mkdir $langdir; $fname = $langdir."/".$2.".wav"; print "$key   $fname\n";' > $data/seg.scp

# now generate the segments
$cmd JOB=1 $dir/log/extract-segments.JOB.log \
extract-segments "scp,p:$data/wav.scp"  $dir/segments  "scp:$data/seg.scp" || exit 1;

exit 0;

