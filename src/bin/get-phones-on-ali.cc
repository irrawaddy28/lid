// bin/get-post-on-ali.cc

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)
//           2014  Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        " Given input posteriors of transition ids, (which could be multiple posteriors per frame) e.g. derived from lattice-to-post, \n"
    	" and a ground truth alignment typically derived from forced alignment of a known sequence of phones,\n"
    	" this program outputs the probability of the posterior of the trans id if the phone corresponding to that trans id \n"
    	" matches the phone corresponding of the trans id present in the ground truth alignment.\n"
    	" If match not found, the program outputs 0 and the frame is considered containing only noise.\n"
    	" If a match is found, the posterior value is considered a signal with the signal level set to\n"
    	" same as that of the posterior probability. E.g. If posterior file contains \n"
        " uttid_1 [ 1208 1 ] [ 1266 1 ] [ 1265 0.5896591 1310 0.4103409 ] [ 1310 0.4 1412 0.6] [ 1201 1]\n"
    	" and ground truth alignment file contains \n"
    	" uttid_1   1324       1266       1310			1265	1260\n"
    	" then the output confidence file will contain \n"
        " uttid_1   0          1          0.4103409		0		0 . \n"
    	" The label SNR = Summed Signal/Summed Noise = (1+0.4103409)/3 = 0.4701 ( Div by 3 since this is the # frames in confidence with 0 as output); \n"
    	" The Signal ratio = Summed Signal/All Frames = (1+0.4103409)/5 = 0.2821 ( Div by 5 since this is the total # frames in confidence); \n"
    	" Note: It is possible that the posteriors and alignments were generated by two different HMM models. Hence, pdf-id of one model need \n"
    	" not match the pdf-id of the other model. The only constraint is that the integers representing phone symbols must be identical in both those models.\n"
    	" This means both models must have identical lang/phones.txt.\n"
    	"\n"
        "See also: get-post-on-ali\n"
        "\n"
        "Usage:  get-phones-on-ali [options] <posteriors-rspecifier> <model file from which posteriors were generated> <ali-rspecifier> <model filename from which alignments were generated> <weights-wspecifier>\n"
        "e.g.: get-phones-on-ali ark:post.ark 1.mdl ark,s,cs:ali.ark 2.mdl ark:weights.ark\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string posteriors_rspecifier = po.GetArg(1),
    	posteriors_model_filename =  po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
		alignments_model_filename =  po.GetArg(4),
        confidences_wspecifier = po.GetArg(5);

    TransitionModel posteriors_trans_model, alignments_trans_model;
    ReadKaldiObject(posteriors_model_filename, &posteriors_trans_model);
    ReadKaldiObject(alignments_model_filename, &alignments_trans_model);


    int32 num_done = 0, num_no_alignment = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);
    BaseFloatVectorWriter confidences_writer(confidences_wspecifier);
    Vector<BaseFloat> snr, sigratio; // SNR and Signal Ratio of labels
    Vector<BaseFloat> mean_snr, mean_sigratio; // mean values
    
    snr.Resize(2); sigratio.Resize(2);
    mean_snr.Resize(2), mean_sigratio.Resize(2);    
    
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();
      snr.SetZero(); sigratio.SetZero();

      if (!alignments_reader.HasKey(key)) {
        num_no_alignment++;
      } else {
        //get the posterior
        const kaldi::Posterior &posterior = posterior_reader.Value();
        int32 num_frames = static_cast<int32>(posterior.size());
        //get the alignment
        const std::vector<int32> &alignment = alignments_reader.Value(key);
        //check the lengths match
        KALDI_ASSERT(num_frames == alignment.size());

        //fill the vector with posteriors on the alignment (under the alignment path)
        Vector<BaseFloat> confidence(num_frames);
        int32  num_good_frames = 0;

        for(int32 i = 0; i < num_frames; i++) {
          int32 alignment_phone = alignments_trans_model.TransitionIdToPhone(alignment[i]);
          // KALDI_LOG << "----frame = " << i << " ----- \n";
          // KALDI_LOG << "ali: tid = " << alignment[i] << ", ph = " << alignment_phone << "\n";
          BaseFloat post_i = 0.0;
          for(int32 j = 0; j < posterior[i].size(); j++) {
        	int32 posterior_phone = posteriors_trans_model.TransitionIdToPhone(posterior[i][j].first);
        	// KALDI_LOG << "post: tid = " << posterior[i][j].first << ", score = " << posterior[i][j].second << ", ph = " << posterior_phone << "\n";
            if(alignment_phone == posterior_phone) {
              post_i = posterior[i][j].second;
              num_good_frames++;
              // KALDI_LOG << "ali: tid = " << alignment[i] << ", ph = " << alignment_phone << "\n";
              // KALDI_LOG << "post: tid = " << posterior[i][j].first << ", ph = " << posterior_phone << "\n";
              break;
            }
          }
          confidence(i) = post_i;
        }
        snr(0) = confidence.Sum()/(num_frames - num_good_frames);
        snr(1) = (float)num_good_frames/(num_frames - num_good_frames);
        sigratio(0) = confidence.Sum()/num_frames;
        sigratio(1) = (float)num_good_frames/num_frames;
        KALDI_LOG << key << ": Confidence = " << confidence.Sum()
        		  << ", Frames With Signal  = " << num_good_frames
				  << ", Frames With Noise    = " << num_frames - num_good_frames
				  << ", Total Frames = " << num_frames
				  << ", SNR of labels (linear scale) = " << snr(0)
				  << ", Best SNR of labels (linear scale) = " << snr(1)
				  << ", Signal/Total = " << sigratio(0)
				  << ", Best Signal/Total = " << sigratio(1);
        //write the vector with confidences
        confidences_writer.Write(key,confidence);
        num_done++;
      }
      mean_snr.AddVec(1.0, snr);
      mean_sigratio.AddVec(1.0, sigratio);      
    }
    KALDI_LOG << "Done getting the posteriors under the alignment path for "
              << num_done << " utterances. " << num_no_alignment << " with missing alignments.";
    mean_snr.Scale(1.0/num_done);
    mean_sigratio.Scale(1.0/num_done);
    KALDI_LOG << "Mean SNR = " << mean_snr(0)
		      << ", Mean Best SNR = " << mean_snr(1)
			  << ", Mean Signal Ratio = " << mean_sigratio(0)
    		  << ", Mean Best Signal Ratio = " << mean_sigratio(1)
    		  << " over " << num_done << " utterances" ;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



