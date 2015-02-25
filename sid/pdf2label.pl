#!/usr/bin/perl
#
# Convert a sequence of pdf ids (hmm state sequence) to class labels
# using a map file that maps pdf ids to class labels.
#
# Can also be used to convert a sequence of transition ids to class labels
# if the map file maps trans ids to class labels.
#
# The input file containing pdf ids (or trans ids) is assumed to be in Kaldi format.
# Example:
# <alipdf file>
# utt-id-1 0 0 1 1 2 2 0 1
#
# The map file has pdf ids on col 1 and class labels on col 2.
# <map file>
# 0 1  
# 1 -1
# 2 -2
#
# Then the output of this routine for the above example will be
# utt-id-1 [8 1 1 -1 -1 -2 -2  1 -1]  (the first element is the number of pdf ids in utt-id-1) 
#
# The sequence of pdf ids can be obtained from Kaldi's ali-to-pdf program
# ====================================================================
# Revision History
# Date 				Author 					Description of Change
# 02/20/15			ad 						Created 
#
# ====================================================================

$USAGE = "\nUSAGE:\nperl $0 [--map-to-col m (default 1)]  [--map-to-col n (default 2)] <map file> <alipdf file>
Example:\nperl $0 --map-from-col 1 --map-to-col 2  pdf2label.map alipdf.1.txt \n\n"; 
 
use File::Basename;

sub get_uttid_and_pdfs {
	$line  = $_[0];				
	($uttid,  @pdfs) = split(/\s+/,$line);			
	chomp($uttid); 		
	# print "pdfs = @pdfs[0..7]\n";		
	return ($uttid, \@pdfs);
}

use Getopt::Long;
die "$USAGE" unless(@ARGV >= 2);
my $map_from_col = 1; # default, column 1 of the map file
my $map_to_col = 2;   # default, column 2 of the map file
GetOptions ("map-from-col=i" => \$map_from_col, "map-to-col=i" => \$map_to_col); 

# alignment file (o/p of Kaldi's ali-to-pdf), map file that maps pdf id to class label
my ($mapf, $alif) = @ARGV; 
my %TRANSFORM = ();

( $map_from_col > 0 ) || die "from-col map in $mapf must be positive. Current value is $map_from_col: $!";
( $map_to_col > 0 ) || die "to-col map in $mapf must be positive. Current value is $map_to_col: $!";

$map_from_col--;
$map_to_col--;
open(MF,"<$mapf") || die "Unable to read from $mapf: $!";
foreach $line (<MF>) {
	($line =~ /^\;/) && next;	
	my(@recs) = split(/\s+/,$line);
	#print "line-> $recs[0..$#recs]\n";
	if (!defined $TRANSFORM{$recs[$map_from_col]}) {
		#$TRANSFORM{$recs[0]} = $recs[1];
		$TRANSFORM{$recs[$map_from_col]} = $recs[$map_to_col];
	}					
}
close(MF);
#print "$_ $TRANSFORM{$_}\n" for sort keys %TRANSFORM;

open(AF, "<$alif") || die "Opening file list $alif";
@lines = <AF>;

my $n = 0;
my $nframes = 0;

foreach (@lines) {
	# get the uttid and pdfs		
	($uttid, $pdfs) = get_uttid_and_pdfs($lines[$n]);
	$nframes = @{$pdfs};
	#$nframes = 10;	
	#print "uttid = $uttid, nframes = $nframes, pdfs: @{$pdfs}[0..$nframes]\n";
	
	# map the pdfs to labels
	@labels = map { exists $TRANSFORM{$_} ? $TRANSFORM{$_} : () } @{$pdfs};		
	
	# print o/p to stdout	
	print "$uttid   [ $nframes  ";
	for (my $i=0; $i < $nframes; $i++) {	
		print "$labels[$i]  ";
	}
	print " ] \n";    
    $n++;    
}
close(AF) || die "Closing input file $alif.";
