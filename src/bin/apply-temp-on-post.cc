// bin/apply-temp-on-post.cc

// Copyright 2017-2022  University of Illinois (Author: Amit Das)

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
#include "hmm/posterior.h"
#include "matrix/kaldi-vector.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "This program applies a softmax, with temperature (T), on posteriors.\n"
        "T must be > 0 for softmax to be applied.\n"
        "Example, if input posteriors are [1 0.3 2 0.7] and\n"
    	"a) T = 1,  then output posteriors are [1 0.4013  2  0.5987] \n"
    	"b) T = -1, then output posteriors are [1 0.3  2 0.7] (no change) \n"
    	"c) T = 0.0001, then output posteriors are [1 0  2  1.0] (1-hot) \n"
        "\n"
        "Usage:  apply-temp-on-post [options] <posteriors-rspecifier> <posteriors-wspecifier>\n"
        "e.g.: apply-temp-on-post --T=1.0 ark:post.in ark:post.out \n";

    ParseOptions po(usage);

    BaseFloat T = 1.0;
    po.Register("T", &T, "Temperature in softmax");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string posteriors_rspecifier = po.GetArg(1),
        posteriors_wspecifier = po.GetArg(2);

    if (T <= 0)
      KALDI_WARN<< "Temperature = " << T << " must be > 0. Otherwise, softmax is not applied on posteriors.\n";

    int32 num_done = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      const kaldi::Posterior &posterior_in = posterior_reader.Value();
      int32 num_frames = static_cast<int32>(posterior_in.size());

      // posteriors for this utt
      kaldi::Posterior posterior_out;
      posterior_out.resize(num_frames);
      for (size_t i = 0; i < num_frames; i++) {
        size_t num_pdf = posterior_in[i].size();

        Vector<BaseFloat> probabilities(num_pdf);
    	for (size_t j = 0; j < num_pdf; j++) {
    	  if (T > 0) {
    	    probabilities(j) = posterior_in[i][j].second/T;
    	  } else if (T <= 0) {
    		probabilities(j) = posterior_in[i][j].second;
    	  }
    	}
    	if (T > 0)
    	  probabilities.ApplySoftMax();

    	// posteriors for this frame
    	for (size_t j = 0; j < num_pdf; j++) {
    	  posterior_out[i].push_back(std::make_pair(posterior_in[i][j].first, probabilities(j)));
    	} // end pdf
      } // end frame
      posterior_writer.Write(posterior_reader.Key(), posterior_out);
      num_done++;
    } // end utt

    KALDI_LOG << "Done apply temperature to posteriors for "
              << num_done << " utterances.";

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
