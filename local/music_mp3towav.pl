#!/usr/bin/perl

use File::Basename;
use File::Find::Rule;

if (@ARGV != 2 ) {
  die "Usage: $0 <input mp3 dir> <output wav dir>\n" .
    "e.g.: $0 /corpus/www.sbs.com.au/mp3 /corpus/www.sbs.com.au/wav\n" .
    "will convert all mp3 files (under the input directory) to the corresponding wav files and " .
    "save them in the output directory. The tree structure within the output " .
    "directory will be the same as that of the input directory.\n";
}

($in_dir, $out_dir) = @ARGV;

my $avconv="/usr/bin/avconv";
unless (-e $avconv) {
 die "Could not find the avconv program at $avconv!";
}

my @avconv_opts = ();
$avconv_opts[0] = "pcm_u8";    # audio codec type: pcm, 8 bit, unsigned
$avconv_opts[1] = "16000";	   # sampling rate
$avconv_opts[2] = "1";		   # num channels, 1 for mono
$avconv_opts[3] = "0";		   # skip segment (in seconds) at the beginnig of input file
 

my $rule =  File::Find::Rule->new;
$rule->file;
$rule->name( '*.wav' );

my @in_files = $rule->in( $in_dir );  
my $f_index = 0;
my @out_files = ();
my $qin_dir = quotemeta($in_dir);

foreach (@in_files) {
	# prepare the output file: input_dir/turkish/abc.mp3 -> output_dir/turkish/abc.wav
	my $f = $_; 
	$f =~ s{$qin_dir}{$out_dir}; # literal string replacement. Replace $in_dir by $out_dir	
	#$f =~ s/\.mp3$/\.wav/; # replace extn .mp3 by .wav
	$out_files[$f_index] = $f;		
	
	# mkdir output dir
	my $dirname  = dirname($out_files[$f_index]);
	unless (-e $dirname) {
		( system("mkdir -p $dirname 2>/dev/null") == 0 ) or die "Unable to create $dirname\n";
	}
	
	# avconv mp3 to wav command
	my $avconvcli = join('  ', $avconv, '-i ', $in_files[$f_index], 
						 '-acodec ', $avconv_opts[0], 
						 '-ar ', $avconv_opts[1], 
						 '-ac ', $avconv_opts[2],
						 '-ss ', $avconv_opts[3],
						  $out_files[$f_index],
						  ' 2>/dev/null ');
	
	# send avconv cmd to system(). It should look sth like:
	# /usr/bin/avconv  -i  input.wav  -acodec   pcm_u8  -ar   16000  -ac   1  -ss   0 output.wav	2>/dev/null	
	system("$avconvcli");	
	print "Converted ", $f_index + 1 ,": $in_files[$f_index]\n ->$out_files[$f_index]\n";	
	$f_index++;
}
