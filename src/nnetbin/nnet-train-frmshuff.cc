// nnetbin/nnet-train-frmshuff.cc

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "Usage:  a) Non Teacher-Student Loss,\n"
    	"        nnet-train-frmshuff [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n\n"
    	"        b) Teacher-Student Loss,\n"
    	"        nnet-train-frmshuff [options] --softmax-temperature=<T>0> --rho=<[0,1]> <feature-rspecifier> <targets-rspecifier1> <targets-rspecifier2> <model-in> [<model-out>]\n\n"
        "e.g.: \n"
        " a) nnet-train-frmshuff --objective-function=xent,500,1.0 scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n"
    	" b) nnet-train-frmshuff --objective-function=ts,500,1.0 --softmax-temperature=2 --rho=0.2 scp:feature.scp ark:1-hot-posterior.ark ark:soft-posterior.ark nnet.init nnet.iter1\n";


    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false,
         randomize = true,
		 teacherstudent = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");
    // po.Register("verbose", &verbose, "Print debug messages (Warning: Can be too many)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");
    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse|xentregmce|ts");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
    
    std::string tgt_interp_mode="none";
    po.Register("tgt-interp-mode", &tgt_interp_mode, "none|soft|hard, Modify ground truth target by interpolating it with nnet posterior or some function of nnet posterior");


    // po.Register("teacher-student", &teacherstudent, "Train using Teacher-Student loss");

    float softmax_temperature = 1.0;
    po.Register("softmax-temperature", &softmax_temperature, "Apply temperature to softmax; Used for Teacher-Student (T/S) training");

    float rho = 1.0; // rho = 1, retrieves the standard CE loss
    po.Register("rho", &rho, "rho has different meanings depending on the loss function. "
    		                "\nFor non Teacher-Student Loss, new ground truth target = rho*(ground truth target from corpus) + (1 - rho)*(function of nnet posterior)"
							"\nFor Teacher-Student Loss, Loss = rho*(CE w/ 1-hot labels) + (1-rho)*(CE w/ teacher labels)");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");

    
    po.Read(argc, argv);

    Xent xent;
    Mse mse;
    TS ts;
    MultiTaskLoss multitask;

    if (0 == objective_function.compare(0,2,"ts")) {
      teacherstudent = true;
    }

    if (0 == objective_function.compare(0,9,"multitask")) {
      // objective_function contains something like :
      // 'multitask,xent,2456,1.0,mse,440,0.001'
      //
      // the meaning is the following:
      // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
      multitask.InitFromString(objective_function);
      teacherstudent = multitask.IsTS();
    }

    if (!teacherstudent) {
      if (po.NumArgs() != 4-(crossvalidate?1:0)) {
        po.PrintUsage();
        exit(1);
      }
    } else {
      if (po.NumArgs() != 5-(crossvalidate?1:0)) {
    	po.PrintUsage();
    	exit(1);
      }
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
	  model_filename;

    std::string targets_rspecifier2 = po.GetArg(2); // valid only for teacher-student loss, init to fake now
        
    std::string target_model_filename;
    if (!teacherstudent) {
      model_filename = po.GetArg(3);
      if (!crossvalidate) {
        target_model_filename = po.GetArg(4);
      }
    } else { // teacher-student loss
      targets_rspecifier2 = po.GetArg(3);
      model_filename = po.GetArg(4);
      if (!crossvalidate) {
        target_model_filename = po.GetArg(5);
      }
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    if (dropout_retention > 0.0) {
      nnet_transf.SetDropoutRetention(dropout_retention);
      nnet.SetDropoutRetention(dropout_retention);
    }
    if (crossvalidate) {
      nnet_transf.SetDropoutRetention(1.0);
      nnet.SetDropoutRetention(1.0);
    }

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    RandomAccessPosteriorReader targets_reader2(targets_rspecifier2);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts), targets_randomizer2(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    
    CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_out2, obj_diff;
    KALDI_LOG << "Objective Function = " << objective_function << "\n";

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    while (!feature_reader.Done()) {
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer
      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        if (feature_randomizer.IsFull()) break; // suspend, keep utt for next loop
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << "Reading " << utt;
        // check that we have targets
        if (!targets_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          continue;
        }
        // check we have per-frame weights
        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        }
        // get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        Posterior targets = targets_reader.Value(utt),
                  targets2 = targets_reader2.Value(utt);

        // get per-frame weights
        Vector<BaseFloat> weights;
        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else { // all per-frame weights are 1.0
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }
        // correct small length mismatch ... or drop sentence
        {
          // add lengths to vector
          std::vector<int32> lenght;
          lenght.push_back(mat.NumRows());
          lenght.push_back(targets.size());
          lenght.push_back(weights.Dim());
          // find min, max
          int32 min = *std::min_element(lenght.begin(),lenght.end());
          int32 max = *std::max_element(lenght.begin(),lenght.end());
          // fix or drop ?
          if (max - min < length_tolerance) {
            if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if(targets.size() != min) targets.resize(min);
            if(weights.Dim() != min) weights.Resize(min, kCopyData);
          } else {
            KALDI_WARN << utt << ", length mismatch of targets " << targets.size()
                       << " and features " << mat.NumRows();
            num_other_error++;
            continue;
          }
        }
        // apply optional feature transform
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

        // pass data to randomizers
        KALDI_ASSERT(feats_transf.NumRows() == targets.size());
        feature_randomizer.AddData(feats_transf);
        targets_randomizer.AddData(targets);
        targets_randomizer2.AddData(targets2);
        weights_randomizer.AddData(weights);
        num_done++;
      
        // report the speed
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }
      }

      KALDI_LOG << "Finished filling randomizer. num done = " << num_done << ", num_no_tgt_mat = " << num_no_tgt_mat <<  ", num_no_frame_wts = " << num_other_error << "\n";

      // randomize
      if (!crossvalidate && randomize) {
        const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        targets_randomizer.Randomize(mask);
        targets_randomizer2.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }

      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next(),
										  targets_randomizer2.Next(),
                                          weights_randomizer.Next()) {
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const Posterior& nnet_tgt  = targets_randomizer.Value();
        const Posterior& nnet_tgt2 = targets_randomizer2.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        KALDI_VLOG(4) << "Mini-batch loop" << "\n";

        // forward pass
        nnet.Propagate(nnet_in, &nnet_out);

        KALDI_VLOG(4) << "nnet_in = " << nnet_in << "\n";
        KALDI_VLOG(4) << "nnet_out = " << nnet_out << "\n";

        // evaluate objective function we've chosen
        // obj_diff contains the error matrix. For e.g., in the case of MSE or XENT, obj_diff(t,k) = y(t,k) - d(t,k)
        if (objective_function == "xent") {
          // gradients re-scaled by weights in Eval,
          xent.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff); 
        } else if (objective_function == "mse") {
          // gradients re-scaled by weights in Eval,
          mse.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
        } else if (objective_function == "ts") {
          Component* last_component = nnet.GetComponent(nnet.NumComponents()-1).Copy(); // deep copy
          Component::ComponentType last_type = last_component->GetType();
          if (last_type == Component::kSoftmax) {
            // Set the temperature of softmax
        	last_component->SetTemperature(softmax_temperature);
        	// Get the logit values of the student nnet
        	const CuMatrix<BaseFloat> logit = nnet.GetPropagateBuf(nnet.NumComponents()-1);
        	// Feedforward the logit through the softmax
        	last_component->Propagate(logit, &nnet_out2);
            ts.SetTemperature(softmax_temperature);
            ts.SetRho(rho);
          } else {
        	KALDI_ERR << "Teacher-Student training not possible; Last component must be <Softmax>\n";
          }
          // gradients re-scaled by weights in Eval,
          KALDI_VLOG(4) << "Temperature of nnet1 =  " << nnet.GetComponent(nnet.NumComponents()-1).Info() << "\n";
          KALDI_VLOG(4) << "Temperature of nnet2 =  " << last_component->Info() << "\n";
          ts.Eval2(frm_weights, nnet_out, nnet_out2, nnet_tgt, nnet_tgt2, &obj_diff);
        } else if ("multitask" == objective_function.substr(0,9)) {
          // gradients re-scaled by weights in Eval,
          if (!teacherstudent) {
        	multitask.Set_Target_Interp(tgt_interp_mode, rho);
            multitask.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
          } else {
        	// Teacher-Student loss supported in the first softmax only. Not supported for other softmaxes
        	Component* last_component = nnet.GetComponent(nnet.NumComponents()-1).Copy();
        	Component::ComponentType last_type = last_component->GetType();
        	if (last_type == Component::kBlockSoftmax) {
        	  // Set the temperature of softmax
        	  last_component->SetTemperature(softmax_temperature);
        	  // Get the logit values of the student nnet
        	  const CuMatrix<BaseFloat> logit = nnet.GetPropagateBuf(nnet.NumComponents()-1);
        	  // Feedforward the logit through the softmax
        	  last_component->Propagate(logit, &nnet_out2);
        	  // Set the temperature and rho parameters for the loss class
        	  multitask.SetTemperature(softmax_temperature);
        	  multitask.SetRho(rho);
        	} else {
              KALDI_ERR << "Teacher-Student training not possible; Last component must be <BlockSoftmax>\n";
            }
        	KALDI_VLOG(4) << "Temperature of nnet1 =  " << nnet.GetComponent(nnet.NumComponents()-1).Info() << "\n";
        	KALDI_VLOG(4) << "Temperature of nnet2 =  " << last_component->Info() << "\n";
        	multitask.Eval2(frm_weights, nnet_out, nnet_out2, nnet_tgt, nnet_tgt2, &obj_diff);
          }
        } else {		  
          KALDI_ERR << "Unknown objective function code : " << objective_function;
        }

        KALDI_VLOG(4) << "grad = " << obj_diff << "\n";

        // backward pass
        if (!crossvalidate) {
          // backpropagate
          nnet.Backpropagate(obj_diff, NULL);
        }

        // 1st minibatch : show what happens in network 
        if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
          KALDI_VLOG(1) << "### After " << total_frames << " frames,";
          KALDI_VLOG(1) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(1) << nnet.InfoBackPropagate();
            KALDI_VLOG(1) << nnet.InfoGradient();
          }
        }
        
        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
          if ((total_frames/25000) != ((total_frames+nnet_in.NumRows())/25000)) { // print every 25k frames
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet.InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet.InfoGradient();
            }
          }
        }
        
        total_frames += nnet_in.NumRows();
      }
    }
    
    // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  

    if (objective_function == "xent") {
      KALDI_LOG << xent.Report();
    } else if (objective_function == "mse") {
      KALDI_LOG << mse.Report();
    } else if (objective_function == "ts") {
      KALDI_LOG << ts.Report();
    } else if (0 == objective_function.compare(0,9,"multitask")) {
      KALDI_LOG << multitask.Report();
    } else {
      KALDI_ERR << "Unknown objective function code : " << objective_function;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
