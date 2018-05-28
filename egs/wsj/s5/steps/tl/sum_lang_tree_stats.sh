#!/bin/bash
# 
# To do: This script does what??

# Begin configuration section.
cmd="run.pl"
nj=1; # this should be read from the num_jobs file in l2 align dir. 
	  # init it here anyways though it is not reqd.
ci_phones=
cleanup=true
binary=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [[ -f path.sh ]]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 "
  echo " e.g.: $0  "
  echo "main options (for others, see top of script file)"  
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|1>                                      # Number of jobs (also see num-processes and num-threads)" 
  echo "  --ci-phones 									  context independent phone list separated by colon"
  echo "  --binary (bool|true)                            # the accumulator files saved in binary or text mode."
  exit 1;
fi

$langwts_config $lang $ciphonelist $alidir $dir

langwts_config=$1; # conf/langwts.list
#l1lang=$2  	   # data/lang dir of target language
alidir=$2          # alignment directory of the target (l1) language
dir=$3			   # output directory

#echo -e "1=$1\n2=$2\n3=$3\n";

nlang=`cat $langwts_config| wc -l`;
i=0
while read line 
do
   str[i]=$line       
   lang[i]=$(echo ${str[i]}| cut -d' ' -f1)
   [[ ! -d ${lang[$i]} ]] && { echo "${lang[$i]} does not exist"; exit 1; } 
   langwt[i]=$(echo ${str[i]}| cut -d' ' -f2)
   [[ -z ${langwt[$i]} ]] && { echo "2nd field rsvd for language weight is empty in $line"; exit 1; }  
   i=$(($i + 1))
done < $langwts_config

[[ ! -f ${alidir}/final.mdl ]] && { echo "$alidir/final.mdl does not exist"; exit 1; }
[[ ! -d ${alidir}/langali ]] && { echo "${alidir}/langali does not exist"; exit 1; }

mkdir -p $dir/langacc
mkdir -p $dir/log/langacc

for (( i=0; i < $nlang; i++ ))
do 	
	#alidir=${lang[$i]}/exp/$alitype;
	data=${lang[$i]}/data/train;
	nj=`cat $alidir/langali/num_jobs`;		
	sdata=$data/split$nj	 
    
    cmvn_opts=`cat $alidir/langali/cmvn_opts 2>/dev/null`    
    
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";
    
    echo "$0: Collecting tree stats for alignments $alidir/langali/ali.lang${i}.*.gz (from language ${lang[$i]}) and saving in $dir/langacc/lang${i}.*.treeacc"
	$cmd JOB=1:$nj $dir/log/langacc/tree_acc.lang${i}.JOB.log \
		acc-tree-stats  --ci-phones=$ci_phones $alidir/final.mdl "$feats" "ark:gunzip -c $alidir/langali/ali.lang${i}.JOB.gz|" $dir/langacc/lang${i}.JOB.treeacc || exit 1;
    
    echo "$0: Summing and scaling (scale factor = ${langwt[$i]}) tree stats in $dir/langacc/lang${i}.*.treeacc and saving in $dir/langacc/lang${i}.scaled.treeacc"
    $cmd JOB=1:1 $dir/log/langacc/scaleacc.lang${i}.JOB.log \
		sum-tree-stats --binary=$binary --scale=${langwt[$i]} $dir/langacc/lang${i}.scaled.treeacc $dir/langacc/lang${i}.*.treeacc || exit 1;  
            
    #$cmd JOB=1:$nj $dir/log/langacc/acc.lang${i}.$x.JOB.log \
    #  gmm-acc-stats-ali  --binary=$binary $dir/$x.mdl "$feats" "ark,t:gunzip -c $dir/langali/ali.lang${i}.JOB.gz|" \
    #  $dir/langacc/lang${i}.$x.JOB.acc || exit 1;
      
	
	#gmm-sum-accs --binary=$binary $dir/langacc/lang${i}.acc $dir/langacc/lang${i}.$x.*.acc 	
	#gmm-scale-accs --binary=$binary ${langwt[i]} $dir/langacc/lang${i}.acc $dir/langacc/lang${i}.scaled.acc
	#rm $dir/langacc/lang${i}.$x.*.acc 2>/dev/null
done

echo "$0: Sum all langs scaled tree stats $dir/langacc/lang*.scaled.treeacc and saving in $dir/langacc/lang.treeacc"
$cmd JOB=1:1 $dir/log/langacc/sumlangstreeacc.JOB.log \
	sum-tree-stats --binary=$binary --scale=1.0 $dir/langacc/lang.treeacc $dir/langacc/lang*.scaled.treeacc || exit 1;  

rm $dir/langacc/lang*.[0-9]*.treeacc $dir/langacc/lang*.scaled.treeacc 2>/dev/null
exit 0;
