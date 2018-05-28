#!/bin/bash
# 
# To do: This script does what??

# Begin configuration section.
cmd="run.pl"
nj=1; # this should be read from the num_jobs file in l2 align dir. 
	  # init it here anyways though it is not reqd.
cleanup=true
binary=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [[ -f path.sh ]]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 "
  echo " e.g.: $0  "
  echo "main options (for others, see top of script file)"  
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|1>                                      # Number of jobs (also see num-processes and num-threads)"  
  echo "  --binary (bool|true)                            # the accumulator files saved in binary or text mode."
  exit 1;
fi

langwts_config=$1; # conf/l2.conf
l1lang=$2  		   # data/lang dir of target language
alidir=$3          # locn of the alignment directory of the target language (l1). Ali for source (l2) languages must be in $alidir/langali/
x=$4 			   # iteration number of the force align-gmm restimation loop
realign=$5         # realign flag - either true or false
dir=$6			   # output directory

#echo -e "1=$1\n2=$2\n3=$3\n4=$4\n5=$5\n";
[[ ! -d $alidir/langali ]] && { echo "$alidir/langali does not exist"; exit 1; }

nlang=`cat $langwts_config| wc -l`;
i=0
while read line 
do
   str[i]=$line       
   lang[i]=$(echo ${str[i]}| cut -d' ' -f1)
   [[ ! -d ${lang[$i]} ]] && { echo "${lang[$i]} does not exist"; exit 1; }
   langwt[i]=$(echo ${str[i]}| cut -d' ' -f2)
   [[ -z ${langwt[$i]} ]] && { echo "2nd field rsvd for language weight is empty in $line"; exit 1; }
   #langmap[i]=$(echo ${str[i]}| cut -d' ' -f3)
   #[[ ! -f ${langmap[$i]} ]] && { echo "${langmap[$i]} does not exist"; exit 1; }
   #langmapcol[i]=$(echo ${str[i]}| cut -d' ' -f4)
   #[[ -z ${langmapcol[$i]} ]] && { echo "4th field rsvd for column number of ${langmap[$i]} is empty"; exit 1; }   
   i=$(($i + 1))
done < $langwts_config

mkdir -p $dir/langali
mkdir -p $dir/langacc
mkdir -p $dir/log/langali
mkdir -p $dir/log/langacc

for (( i=0; i < $nlang; i++ ))
do 	
	#alidir=${lang[$i]}/exp/$alitype;
	data=${lang[$i]}/data/train;	
	cp $alidir/langali/num_jobs $dir/langali/num_jobs # carry over for use in subsequent stages
	cp $alidir/langali/cmvn_opts $dir/langali/cmvn_opts # carry over for use in subsequent stages
	nj=`cat $dir/langali/num_jobs`;
	cmvn_opts=`cat $dir/langali/cmvn_opts 2>/dev/null`    	
	sdata=$data/split$nj
		
	#perl utils/l2l1phonemap_to_int.pl -col ${langmapcol[$i]} ${langmap[$i]} ${lang[$i]}/data/lang/phones.txt $l1lang/phones.txt > $dir/langali/lang${i}phonemapint.txt
	
	if $realign; then     
    echo "$0: Converting alignments from $alidir/langali/ali.lang${i}.*.gz using $dir/$x.mdl, $dir/tree"
    $cmd JOB=1:$nj $dir/log/langali/convert.lang${i}.${x}.JOB.log \
		convert-ali $alidir/final.mdl $dir/$x.mdl $dir/tree \
		"ark,t:gunzip -c $alidir/langali/ali.lang${i}.JOB.gz|" "ark,t:|gzip -c >$dir/langali/ali.lang${i}.JOB.gz" || exit 1;         
	fi	
    
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";
    echo "$0: Collecting stats for converted alignments $dir/langali/ali.lang${i}.*.gz in $dir/langacc/lang${i}.$x.*.acc"
    $cmd JOB=1:$nj $dir/log/langacc/acc.lang${i}.$x.JOB.log \
      gmm-acc-stats-ali  --binary=$binary $dir/$x.mdl "$feats" "ark,t:gunzip -c $dir/langali/ali.lang${i}.JOB.gz|" \
      $dir/langacc/lang${i}.$x.JOB.acc &>/dev/null || exit 1;    

    echo "$0: Summing the collected stats $dir/langacc/lang${i}.$x.*.acc in $dir/langacc/lang${i}.$x.acc"  
	$cmd JOB=1:1 $dir/log/langacc/sumacc.lang${i}.$x.JOB.log \
		gmm-sum-accs --binary=$binary $dir/langacc/lang${i}.$x.acc $dir/langacc/lang${i}.$x.*.acc || exit 1;
		
	echo "$0: Scaling (scale factor = ${langwt[i]}) the summed stats $dir/langacc/lang${i}.$x.acc in $dir/langacc/lang${i}.scaled.$x.acc"
	$cmd JOB=1:1 $dir/log/langacc/scaleacc.lang${i}.$x.JOB.log \
		gmm-scale-accs --binary=$binary ${langwt[$i]} $dir/langacc/lang${i}.$x.acc $dir/langacc/lang${i}.scaled.$x.acc || exit 1;

	rm $dir/langacc/lang${i}.$x.*.acc 2>/dev/null
done

echo "$0: Summing all langs scaled stats $dir/langacc/lang*.scaled.$x.acc in $dir/langacc/lang.sum.$x.acc"
$cmd JOB=1:1 $dir/log/langacc/sumlangsacc.$x.JOB.log \
	gmm-sum-accs --binary=$binary $dir/langacc/lang.sum.$x.acc $dir/langacc/lang*.scaled.$x.acc || exit 1; 

exit 0;
