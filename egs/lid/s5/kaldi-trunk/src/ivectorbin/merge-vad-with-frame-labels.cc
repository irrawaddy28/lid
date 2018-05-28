// ivectorbin/compute-vad.cc

// Copyright  2013  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "ivector/voice-activity-detection.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "This program reads the a) input vad decisions (1 = voiced, 0 = sil),\n"
    	"and b) the class labels for the voiced frames of the vad decisions.\n"
    	"It replaces the 1's in the vad decisions with the labels of the\n"
    	"voiced frames and outputs the resulting vector.\n"
    	"This program is useful when you have an utterance spoken\n"
    	"by multiple speakers (or spoken in multiple languages) and you know\n"
    	"(a), (b) but not (c). It will generate (c) to reveal\n"
    	"the timing information in the utterance when a particular class label\n"
    	"was active or not. This information can be used to segment the utterance\n"
    	"according to class labels.\n"
    	"a) the vad decisions (e.g. 0 0 1 1 1 0 0 1 1 0) \n"
    	"b) the class labels (spkr or lang id) of the voiced frames of the utterance\n"
    	"(e.g. 5 1 2 2 3 1) Note: the first element 5 is the no. of labels.\n"
    	"c) the positions of the class labels from the start of the utterance\n"
    	"(e.g. 0 0 1 2 2 0 0 3 1 0)\n"
    	"Usage: merge-vad-with-frame-labels <vad-rspecifier> <labels-rspecifier> <vad-wspecifier>\n"
    	"e.g. merge-vad-with-frame-labels scp:vad.scp ark:voiced_labels.ark ark,t:-\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string vad_rspecifier = po.GetArg(1);
    std::string frame_label_rspecifier = po.GetArg(2);
    std::string vad_wspecifier = po.GetArg(3);

    // bool binary = false;
    kaldi::int32 num_done = 0, num_err = 0;

    SequentialBaseFloatVectorReader vad_reader(vad_rspecifier);
    RandomAccessBaseFloatVectorReader frame_label_reader(frame_label_rspecifier);
    BaseFloatVectorWriter vad_writer(vad_wspecifier);

    for (;!vad_reader.Done(); vad_reader.Next()) {
    	std::string utt = vad_reader.Key();
    	const Vector<BaseFloat> &vad = vad_reader.Value();
    	// vad.Write(std::cout, false);
    	if (!frame_label_reader.HasKey(utt)) {
    		KALDI_WARN << "frame reader does not have key "
    		                       << utt << ", producing no output for this utterance";
    		num_err++;
    		continue;
    	}

    	// frame_labels is a vector of labels assigned to the voiced frames of vad
    	// with frame_labels(0) =  size(frame_labels(1:end)) = no. of labels
    	Vector<BaseFloat> frame_labels = frame_label_reader.Value(utt);
    	// frame_labels.Write(std::cout, binary);

    	// The no. of voiced frames in the vad vector
    	// and the no. of labels in the frame_labels vector must match
    	KALDI_ASSERT(vad.Sum() == frame_labels(0) && "The number of voiced frames do not match");
    	// std::cout << "utt = " << utt << ", vad nsegs = " << vad.Dim() << ", vad nvoiced = " << vad.Sum() << ", 0th ele =  " << frame_labels(0) << ", nvoiced + 1 = " << frame_labels.Dim() << std::endl;

    	// Remove frame_labels(0) since it is the count of no. of labels that follow
    	frame_labels.RemoveElement(0);

    	// Now merge the frame_labels into the output vad vector
    	Vector<BaseFloat> output_vad;
    	output_vad.Resize(vad.Dim(), kSetZero);
    	int32 i = 0;
    	for (int32 j = 0; j < vad.Dim(); j++) {
    		if (vad(j) != 0 ) {
    			output_vad(j) = frame_labels(i);
    			i++;
    		}
    		else {
    			output_vad(j) = vad(j);
    		}
    	}
    	//output_vad.Write(std::cout, binary);
    	vad_writer.Write(vad_reader.Key(), output_vad);
    	num_done++;
    }
    KALDI_LOG << "Merge vad voiced frames with frame labels for "
                  << num_done << " utterances successfully; "
                  << num_err << " utterance keys were not found";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
