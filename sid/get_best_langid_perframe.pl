#!/usr/bin/perl
#
# Given log likelihood scores for 3 classes p(x_t|lambda_1), p(x_t|lambda_2), p(x_t|lambda_3)
# evaluated for feature vector x_t at time t=1,...,T, this routine calculates the posteriors
# for each class and determines the best class in each frame.
# Best class at frame t = argmax p (lambda j | x_t) = argmax p (x_t|lambda j) p (j) 
#						  lambda					  lambda
# p(j) is the prior of class j and is an input argument to this routine.
# 
# The input files containing log likehood scores per frame are assumed to be in Kaldi format.
# Example:
# <L1 log like file>
# utt-id-1 -112.0 -114.0 -110.0 -115.0
#
# <L2 log like file>
# utt-id-1 -113.0 -111.0 -112.0 -107.0
#
# <L3 log like file>
# utt-id-1 -111.0 -112.0 -113.0 -108.0
#
# Assuming the target language is L1 (class label = 1) and other languages are "impostor"
# languages (class label = -1, -2), and if the priors are equal, then the 
# output of this routine for the above example will be
# utt-id-1 [4 -2 -1 1 -1 ]  (4 frames in utt-id-1, L3 most likely in frame 1, L2 in frame 2, L1 in frame 1, L2 in frame 4) 
#
# The log likelihood files can be obtained from the output of Kaldi's routine
# fgmm-global-get-frame-likes --average=false .... <loglike file>
# ====================================================================
# Revision History
# Date 				Author 					Description of Change
# 02/03/15			ad 						Created 
#
# ====================================================================

$USAGE = "USAGE\n: perl $0 <L1 log like file> <L1 prior> <L2 log like file> <L2 prior> <L3 log like file> <L3 prior>\n\n 
		  Example\n: perl $0 L1_loglikes_perframe 0.6 L2_loglikes_perframe 0.3  L3_loglikes_perframe 0.1 > best_langid_perframe.txt \n"; 
 
use File::Basename;

sub get_element_wise_max {
	
	$tlikes  =  $_[0]; # ref to array
	$tprior  =  $_[1];	
	
	$o1likes =  $_[2]; # ref to array
	$o1prior =  $_[3];	
	
	$o2likes =  $_[4]; # ref to array
	$o2prior =  $_[5];
	
	my $n=0;
	my @a=();
	my @b=();
	my @tpost=();
	my @o1post=();
	my @o2post=();
	my @maxind = ();
	my @maxval = ();
	
	foreach (@{$tlikes}) {		
		@a = (log($tprior/$o1prior) + $tlikes->[$n]  - $o1likes->[$n], log($tprior/$o2prior) + $tlikes->[$n]  - $o2likes->[$n]);
		@b = (log($o1prior/$tprior)  + $o1likes->[$n] - $tlikes->[$n], log($o1prior/$o2prior) + $o2likes->[$n] - $tlikes->[$n]);		
		
		$tpost[$n] = 1/(1 + exp(-$a[0]) + exp(-$a[1]));
		$o1post[$n] = 1/(1 + exp(-$b[0]) + exp(-$b[1]));
		$o2post[$n] = 1 - $tpost[$n] - $o1post[$n];	
		
		#print "n=$n, tlikes = $tlikes->[$n], o1likes = $o1likes->[$n], o2likes = $o2likes->[$n]\n";
		#print "n=$n, tpost = $tpost[$n], o1post = $o1post[$n], o2post = $o2post[$n]\n";
		
		if ( ($tpost[$n] > $o1post[$n]) &&  ($tpost[$n] > $o2post[$n]) ) {
			$maxval[$n] = $tpost[$n];
			$maxind[$n] = "1"; # target lang
		}
		elsif ( ($o1post[$n] >= $tpost[$n] ) &&  ( $o1post[$n] >= $o2post[$n]) ) {
			$maxval[$n] = $o1post[$n];
			$maxind[$n] = "-1";	# other lang 1		
		}
		else {
			$maxval[$n] = $o2post[$n];
			$maxind[$n] = "-2"; # other lang 2
		}		
		$n++;
	}
		
	return(\@maxval, \@maxind);	
}

sub get_uttid_and_likes {
	$tline  = $_[0];
	$o1line = $_[1];
	$o2line = $_[2];	
	
	$tline =~  s/\]//;  $tline =~  s/\[//;    ($tuttid,  @tlikes) = split(/\s+/,$tline);
	$o1line =~ s/\]//; $o1line =~  s/\[//;  ($o1uttid, @o1likes) = split(/\s+/,$o1line);
	$o2line =~ s/\]//; $o2line =~ s/\[//; ($o2uttid, @o2likes) = split(/\s+/,$o2line);
	
	chomp($tuttid); chomp($o1uttid); chomp($o2uttd);
	
	if ( ($tuttid ne $o1uttid) || ($tuttid ne $o2uttid) ) { 
		die "utterance ids do not match: $tuttid, $o1uttid, $o2uttid\n";
	}	
	
	#print "t = @tlikes[0..7]\n";
	#print "o1 = @o1likes[0..7]\n";
	#print "o2 = @o2likes[0..7]\n";
	
	return ($tuttid, \@tlikes, \@o1likes, \@o2likes);
}


if (@ARGV != 6) {
    die "$USAGE\n";
}

my ($tf, $tprior, $o1f, $o1prior, $o2f, $o2prior) = @ARGV; # target language file, other language 1 file, other language 2 file, destination dir


#unless (-e $dir) {
#		die "Unable to create $data\n";
#}

open(T, "<$tf") || die "Opening file list $tf";
open(O1, "<$o1f") || die "Opening file list $o1f";
open(O2, "<$o2f") || die "Opening file list $o2f";

@tlines = <T>;
@o1lines = <O1>;
@o2lines = <O2>;
my $n = 0;
my $nframes = 0;

foreach (@tlines) {		
	($uttid, $tlikes, $o1likes, $o2likes) = get_uttid_and_likes($tlines[$n],$o1lines[$n],$o2lines[$n]);
	#print "uttid = $uttid: @tlikes[0..$#tlikes]\n";
	#print "uttid = $uttid: @o1likes[0..$#o1likes]\n";
	#print "uttid = $uttid: @o2likes[0..$#o2likes]\n";
	($maxval, $maxind) = get_element_wise_max($tlikes, $tprior, $o1likes, $o1prior, $o2likes, $o2prior);
	$nframes = @{$maxval};
	print "$uttid   [ $nframes  ";
	for (my $i=0; $i < $nframes; $i++) {
		# print the language id that has the highest posterior
		# lang id: $maxind->[$i], posterior: $maxval->[$i]
		# print "$maxind->[$i]  $maxval->[$i]\n"; 
		print "$maxind->[$i]  ";
	}
	print " ] \n";    
    $n++;
}
close(T) || die "Closing input file $tf.";
close(O1) || die "Closing input file $o1f.";
close(O2) || die "Closing input file $o2f.";
