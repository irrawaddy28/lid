#!/bin/bash

##renew_nnet_softmax.sh
##Paul Hager

#Setting up $PATH
. ./path.sh

softmax_dim=
remove_last_components=2
# End of config.

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

usage="Usage: $0 <hmm model> <input_network> <output_network> > "
##e.g. renew_nnet_softmax.sh exp/tri3b/final.mdl nnet.old nnet.new

[[ $# -eq 3 ]] || { echo $usage; exit 1; }

mdl=$1    # hmm-gmm mdl file
oldnn=$2  # old dnn
newnn=$3  # new dnn

[[ -e $mdl ]] || { echo "$mdl does not exist"; exit 1; }
[[ -e $oldnn ]] || { echo "$oldnn does not exist"; exit 1; }

nndir=$(mktemp -d)
#trap "rm -rf $nndir;" EXIT
echo
#echo "Searching for number of PFIDs"
#Discover softmax dimensions from number of triphone tree leaves
[[ -z ${softmax_dim} ]] && softmax_dim=$(hmm-info $mdl | grep pdfs | awk '{ print $NF }')
echo "Number of PDFIDs: ${softmax_dim}"

echo "Stripping pre-trained network of softmax layer"
#Remove the softmax layer from pre-trained network which includes
# the last two components: <AffineTransform> and  <Softmax>
oldnnname=$(basename $oldnn)
nnet_stripped=$nndir/${oldnnname}_stripped.init
nnet-copy --binary=false --remove-last-components=${remove_last_components} $oldnn ${nnet_stripped}
echo "Done"
echo

echo "Protyping new softmax layer"
# Create new softmax layer, i.e., both create new <AffineTransform> and  <Softmax> components
##Note the number of hidden neurons must be greater than zero, 1 was arbitrarily chosen
tempnn="softmax"
oldnn_outdim=$(nnet-info ${nnet_stripped}|tail -n 1|awk '{print $NF}'|sed 's/\([0-9]\+\).*/\1/g')
echo "olddnn outdim = ${oldnn_outdim}"
echo "Output dim of last hidden layer = ${oldnn_outdim}"
[[ ! -z ${oldnn_outdim} ]] && [[ ${oldnn_outdim} -gt 0 ]] || { echo "output dim of the old nn is not valid = ${oldnn_outdim}"; exit 1; }
python utils/nnet/make_nnet_proto.py --no-proto-head 39 ${softmax_dim} 1 ${oldnn_outdim}|tail -n 3 > $nndir/${tempnn}.proto
echo "Done"
echo

echo "Initializing new softmax layer randomly"
#Initialize the softmax layer (if you do not specify a seed value, nnet-initialize uses the default seed
# everytime ... thereby generating the same initialized n/w no matter how many times you call nnet-initialize). 
nnet-initialize --binary=false $nndir/${tempnn}.proto $nndir/${tempnn}.init
echo "Done"
echo

echo "Concatenating two networks"
#Connect the two networks
#Make sure that the oldnn_outdim = softmax_indim dimensions 
nnet-concat ${nnet_stripped} $nndir/${tempnn}.init ${newnn}
echo "New network is stored at" ${newnn}
echo "Done"
echo

##Comment out these lines if there is no need to check work.
#Copy to ASCII for debugging newly created network to binary
#nnet-copy --binary=false $nndir/${newnn}.init $nndir/${newnn}_text.init
#echo "New network is stored at" $nndir/${newnn}_text.init
