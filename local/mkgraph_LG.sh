#!/bin/bash
lang=$1 

[[ -f $lang/phones.txt ]] || { echo "$lang/phones.txt does not exist "; exit 1; } 
[[ -f $lang/words.txt ]] || { echo "$lang/words.txt does not exist "; exit 1; } 
[[ -f $lang/lexicon.txt ]] || { echo "$lang/lexicon.txt does not exist "; exit 1; } 
[[ -f $lang/grammar.txt ]] || { echo "$lang/grammar.txt does not exist "; exit 1; } 

phone_disambig_symbol=$(grep \#0 $lang/phones.txt | awk '{print $2}')
word_disambig_symbol=$(grep \#0 $lang/words.txt | awk '{print $2}' )

# Build L.fst from lexicon.txt, sort on o/p label
utils/make_lexicon_fst.pl $lang/lexicon.txt |\
fstcompile --isymbols=$lang/phones.txt --osymbols=$lang/words.txt --keep_isymbols=false --keep_osymbols=false |\
fstaddselfloops "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" |\
fstarcsort --sort_type=olabel > $lang/L_disambig.fst

# Build G.fst from grammar.txt, sort on i/p label 
# Hack: grammar.txt is a hack to output a grammar fst using utils/make_lexicon_fst.pl.
# Grammar fst should look like the output of "arpa2fst <input-ngram-file>" 
# where <input-ngram-file> has no lm or backoff wts
utils/make_lexicon_fst.pl $lang/grammar.txt > $lang/grammar.fst
cat $lang/grammar.fst | utils/eps2disambig.pl | utils/s2eps.pl|\
fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $lang/G.fst
    
# visual checks
fstdraw --isymbols=$lang/phones.txt --osymbols=$lang/words.txt $lang/L_disambig.fst |\
dot -Tps  | ps2pdf -  $lang/L_disambig.pdf

fstdraw --isymbols=$lang/words.txt --osymbols=$lang/words.txt $lang/G.fst |\
dot -Tps  | ps2pdf -  $lang/G.pdf
