// nnet/nnet-loss.cc

// Copyright 2011-2015  Brno University of Technology (author: Karel Vesely)

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

#include "nnet/nnet-loss.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "hmm/posterior.h"

#include <sstream>
#include <iterator>

namespace kaldi {
namespace nnet1 {


/* Xent */

/**
 * Helper function of Xent::Eval,
 * calculates number of matching elemente in 'v1', 'v2' weighted by 'weights'.
 */
template <typename T>
inline void CountCorrectFramesWeighted(const CuArray<T> &v1, 
                                       const CuArray<T> &v2, 
                                       const VectorBase<BaseFloat> &weights, 
                                       double *correct) {
  KALDI_ASSERT(v1.Dim() == v2.Dim());
  KALDI_ASSERT(v1.Dim() == weights.Dim());
  int32 dim = v1.Dim();
  // Get GPU data to host,
  std::vector<T> v1_h(dim), v2_h(dim);
  v1.CopyToVec(&v1_h);
  v2.CopyToVec(&v2_h);
  // Get correct frame count (weighted),
  double corr = 0.0;
  for (int32 i=0; i<dim; i++) {
   corr += weights(i) * (v1_h[i] == v2_h[i] ? 1.0 : 0.0);
  }
  // Return,
  (*correct) = corr;
}


void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out, 
                const CuMatrixBase<BaseFloat> &target, 
                CuMatrix<BaseFloat> *diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target.Sum()));

  double num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0);

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

#if 0
  CuVector<BaseFloat> col_sum(net_out.NumRows());
  col_sum.AddColSumMat(1.0, net_out, 0.0);
  KALDI_LOG << " sum_j net_out(i, j), j = 1,...,  " << net_out.NumCols() << std::endl;
  for (int32 i = 0; i < col_sum.Dim(); i++)
    KALDI_LOG << col_sum(i) << " ";
  KALDI_LOG << std::endl;
#endif

  // evaluate the frame-level classification,
    double correct;
    net_out.FindRowMaxId(&max_id_out_); // find max in nn-output
    target.FindRowMaxId(&max_id_tgt_); // find max in targets
    CountCorrectFramesWeighted(max_id_out_, max_id_tgt_, frame_weights, &correct);
    
    CuMatrix<BaseFloat> target_interp(target, kNoTrans);
    int32 num_rows = net_out.NumRows(), num_cols = net_out.NumCols();

    // Linearly interpolate targets for the primary task if interpolation wt < 1.0
  	if (tgt_interp_mode_.compare("none") != 0) {
      target_interp.Scale(rho_);
  	  if (tgt_interp_mode_.compare("soft") == 0) {
  	    // soft interpolation: wt*t_k + (1 - wt)*y_k
        target_interp.AddMat(1 - rho_, net_out, kNoTrans);
  	  } else if (tgt_interp_mode_.compare("hard") == 0) {
  		// hard interpolation: wt*t_k + (1 - wt)*1_{max y_k}
  		Matrix<BaseFloat> net_out_hard(num_rows, num_cols, kSetZero);
  		std::vector<int32> max_id_out(num_rows);

  		max_id_out_.CopyToVec(&max_id_out);
  		// Set the "hard" matrix s.t. M(i, max_id_out(i)) = 1 and all other elements set to 0
  		for(int32 ri=0; ri<num_rows; ri++) {
  		  net_out_hard(ri, max_id_out[ri]) = 1.0;
  		}
  		CuMatrix<BaseFloat> net_out_hard_cu(net_out_hard, kNoTrans);
  		target_interp.AddMat(1 - rho_, net_out_hard_cu, kNoTrans);
  	  }
  	}

  // compute derivative wrt. activations of last layer of neurons,
  if (tgt_interp_mode_.compare("none") == 0 || tgt_interp_mode_.compare("hard") == 0) {
    *diff = net_out;
    //diff->AddMat(-1.0, target); // diff <-- (-1.0)*target + diff
    diff->AddMat(-1.0, target_interp);
  } else {
	// compute I(f, k) = -log y(f, k), where f = frame index, k = pdf index
	CuMatrix<BaseFloat> I(net_out);
	I.Add(1e-20); // avoid log(0)
	I.ApplyLog(); // log(y)
	I.Scale(-1);  // I = -log y

	// compute H(f) = - sum_k y(f, k) log y(f, k)
	CuMatrix<BaseFloat> H(num_rows, num_cols, kSetZero), H1(I), Ones(num_cols, num_cols, kSetZero);
	H1.MulElements(net_out); // -y*log(y)
	Ones.Set(1);
	H.AddMatMat(1.0, H1, kNoTrans, Ones, kNoTrans, 0.0); // H = 1.0*H1*Ones + 0.0*H

	// compute y(f,k)*(rho +  (1-rho)*( I(f,k) - H(f) ) ) - rho*target
	CuMatrix<BaseFloat> diff2(I);
	diff2.AddMat(-1.0, H); // diff2 = I(f,k) - H(f)
	diff2.Scale(1-rho_); // diff2 <-- (1-rho)*(I(f,k) - H(f))
	diff2.Add(rho_); // diff2 <-- rho + (1-rho)*(I(f,k) - H(f))
	diff2.MulElements(net_out); // diff2 <-- y(f,k).*diff2(f,k)
	diff2.AddMat(-rho_,target); // diff2 <-- -rho*t(f,k)
	*diff = diff2; // deep copy
  }
  diff->MulRowsVec(frame_weights_); // weighting,


  // calculate cross_entropy (in GPU),
  xentropy_aux_ = net_out; // y
  xentropy_aux_.Add(1e-20); // avoid log(0)
  xentropy_aux_.ApplyLog(); // log(y)
  //xentropy_aux_.MulElements(target); // t*log(y)
  xentropy_aux_.MulElements(target_interp); // t*log(y)
  xentropy_aux_.MulRowsVec(frame_weights_); // w*t*log(y) 
  double cross_entropy = -xentropy_aux_.Sum();
  
  // caluculate entropy (in GPU),
  entropy_aux_ = target; // t
  entropy_aux_.Add(1e-20); // avoid log(0)
  entropy_aux_.ApplyLog(); // log(t)
  entropy_aux_.MulElements(target); // t*log(t)
  entropy_aux_.MulRowsVec(frame_weights_); // w*t*log(t) 
  double entropy = -entropy_aux_.Sum();

  KALDI_ASSERT(KALDI_ISFINITE(cross_entropy));
  KALDI_ASSERT(KALDI_ISFINITE(entropy));

  KALDI_VLOG(4) << __PRETTY_FUNCTION__ << "\n";
  KALDI_VLOG(4) << "xentropy_aux_ vector = " << xentropy_aux_ << "\n";
  KALDI_VLOG(4) << "cross_entropy =  " << cross_entropy << "\n";
  KALDI_VLOG(4) << "entropy = "<< entropy << "\n";

  loss_ += cross_entropy;
  entropy_ += entropy;
  correct_ += correct;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100; // 1h
    frames_progress_ += num_frames;
    loss_progress_ += cross_entropy;
    entropy_progress_ += entropy;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last " 
                    << static_cast<int>(frames_progress_/100/3600) << "h of " 
                    << static_cast<int>(frames_/100/3600) << "h]: " 
                    << (loss_progress_-entropy_progress_)/frames_progress_ << " (Xent)";
      // store
      loss_vec_.push_back((loss_progress_-entropy_progress_)/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
      entropy_progress_ = 0.0;
    }
  }
}


void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out, 
                const Posterior &post, 
                CuMatrix<BaseFloat> *diff) {
  int32 num_frames = net_out.NumRows(),
    num_pdf = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_pdf, &tgt_mat_);

  // call the other eval function,
  Eval(frame_weights, net_out, tgt_mat_, diff);
}

void Xent::Set_Target_Interp(const std::string tgt_interp_mode="none", const BaseFloat rho=1.0) {
  if (rho_ < 0 || rho_ > 1.0) {
    KALDI_ERR << "rho = " << rho_ << " is outside the acceptable range [0, 1]\n";
  }

  tgt_interp_mode_ =   tgt_interp_mode;

  if (tgt_interp_mode.compare("none") == 0) {
    rho_ = 1.0;
  } else {
    rho_  = rho;
  }
}

std::string Xent::Report() {
  std::ostringstream oss;
  oss << "AvgLoss: " << (loss_-entropy_)/frames_ << " (Xent - AvgTgtEnt), "
      << "[AvgXent: " << loss_/frames_ 
	  << ", Target interpolation mode: " << tgt_interp_mode_
	  << ", rho:  " << rho_
      << ", AvgTargetEnt: " << entropy_/frames_ << "]" << std::endl;
  if (loss_vec_.size() > 0) {
     oss << "progress: [";
     std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
     oss << "]" << std::endl;
  }
  if (correct_ >= 0.0) {
    oss << "FRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<" << std::endl;
  }
  return oss.str(); 
}


/* Mse */

void Mse::Eval(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat>& net_out, 
               const CuMatrixBase<BaseFloat>& target, 
               CuMatrix<BaseFloat>* diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target.Sum()));

  int32 num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0);

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  //compute derivative w.r.t. neural nerwork outputs
  *diff = net_out; // y
  diff->AddMat(-1.0,target); // (y - t)
  diff->MulRowsVec(frame_weights_); // weighting,

  // Compute MeanSquareError loss of mini-batch
  diff_pow_2_ = *diff;
  diff_pow_2_.MulElements(diff_pow_2_); // (y - t)^2
  diff_pow_2_.MulRowsVec(frame_weights_); // w*(y - t)^2
  double mean_square_error = 0.5 * diff_pow_2_.Sum(); // sum the matrix,

  KALDI_ASSERT(KALDI_ISFINITE(mean_square_error));

  // accumulate
  loss_ += mean_square_error;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100; // 1h
    frames_progress_ += num_frames;
    loss_progress_ += mean_square_error;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last " 
                    << static_cast<int>(frames_progress_/100/3600) << "h of " 
                    << static_cast<int>(frames_/100/3600) << "h]: " 
                    << loss_progress_/frames_progress_ << " (Mse)";
      // store
      loss_vec_.push_back(loss_progress_/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
    }
  }
}


void Mse::Eval(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat>& net_out, 
               const Posterior& post, 
               CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_nn_outputs = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_nn_outputs, &tgt_mat_);

  // call the other eval function,
  Eval(frame_weights, net_out, tgt_mat_, diff);
}
 

std::string Mse::Report() {
  // compute root mean square,
  int32 num_tgt = diff_pow_2_.NumCols();
  BaseFloat root_mean_square = sqrt(loss_/frames_/num_tgt);
  // build the message,
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_/frames_ << " (Mse), " << "[RMS " << root_mean_square << "]" << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
  oss << "]" << std::endl;
  return oss.str();
}


void TS::Eval2(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat> &net_out1,
			   const CuMatrixBase<BaseFloat> &net_out2,
               const CuMatrixBase<BaseFloat> &target1,
			   const CuMatrixBase<BaseFloat> &target2,
               CuMatrix<BaseFloat> *diff) {
  // check inputs,
  KALDI_ASSERT(net_out1.NumCols() == target1.NumCols());
  KALDI_ASSERT(net_out1.NumRows() == target1.NumRows());
  KALDI_ASSERT(net_out2.NumCols() == target2.NumCols());
  KALDI_ASSERT(net_out2.NumRows() == target2.NumRows());

  KALDI_ASSERT(net_out1.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out1.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target1.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target2.Sum()));

  CuMatrix<BaseFloat> diff1, diff2;

  double num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0);

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  // evaluate the frame-level classification,
  double correct;
  net_out1.FindRowMaxId(&max_id_out_); // find max in nn-output
  target1.FindRowMaxId(&max_id_tgt_); // find max in targets
  CountCorrectFramesWeighted(max_id_out_, max_id_tgt_, frame_weights, &correct);

  // compute derivative wrt. activations of last layer of neurons,
  diff1 = net_out1;
  diff1.AddMat(-1.0, target1); // diff1 <-- (-1.0)*target + diff1
  diff1.Scale(rho_); // diff1 <-- rho*diff1
  diff2 = net_out2;  // diff2=net_out2
  diff2.AddMat(-1.0, target2);
  diff2.Scale(1.0/T_); // diff2 = (1/T)*(net_out1 - target2) = derivative of temperature softmax ... (eqn 1)
  diff2.Scale(T_*T_); // diff2 <-- (T^2)*diff2 (artificially scale diff2 by T^2 since eqn 1 is a function of 1/(T^2);
                      // This helps maintain the magnitude of the gradients of diff2 and diff1 close to each other
  diff2.Scale(1-rho_); // diff2 <-- (1-rho)*diff2
  *diff = diff1;
  diff->AddMat(1.0,diff2); // diff <-- (1.0)*diff2 + diff1
  diff->MulRowsVec(frame_weights_); // weighting,

  // calculate cross_entropy (CE1) for the hard labels (in GPU),
  xentropy_aux1_ = net_out1; // y=net_out1
  xentropy_aux1_.Add(1e-20); // avoid log(0)
  xentropy_aux1_.ApplyLog(); // log(y)
  xentropy_aux1_.MulElements(target1); // t*log(y)
  //xentropy_aux_.MulElements(target_interp); // t*log(y)
  xentropy_aux1_.MulRowsVec(frame_weights_); // w*t*log(y)

  // calculate cross_entropy (CE2) for the soft labels (temperature softmax) (in GPU),
  xentropy_aux2_ = net_out2; // y=net_out2
  xentropy_aux2_.Add(1e-20); // avoid log(0)
  xentropy_aux2_.ApplyLog(); // log(y)
  xentropy_aux2_.MulElements(target2); // t*log(y)
  xentropy_aux2_.MulRowsVec(frame_weights_); // w*t*log(y)
  // xentropy_aux2_.Scale(T_*T_); // To Do: Scale by T^2 to keep CE2 close to CE1 ??

  // Teacher-Student Loss = rho*CE1 + (1-rho)*CE2
  double cross_entropy1 = -(xentropy_aux1_.Sum());
  double cross_entropy2 = -(xentropy_aux2_.Sum());
  double cross_entropy = rho_*cross_entropy1 +  (1-rho_)*cross_entropy2;

  // calculate entropy (in GPU),
  entropy_aux_ = target1; // t
  entropy_aux_.Add(1e-20); // avoid log(0)
  entropy_aux_.ApplyLog(); // log(t)
  entropy_aux_.MulElements(target1); // t*log(t)
  entropy_aux_.MulRowsVec(frame_weights_); // w*t*log(t)
  double entropy = -entropy_aux_.Sum();

  KALDI_ASSERT(KALDI_ISFINITE(cross_entropy));
  KALDI_ASSERT(KALDI_ISFINITE(entropy));

  KALDI_VLOG(4) << __PRETTY_FUNCTION__ << "\n";
  KALDI_VLOG(4) << "xentropy_aux1_ vector = " << xentropy_aux1_ << "\n";
  KALDI_VLOG(4) << "cross_entropy1 (CE1) =  " << cross_entropy1 << "\n";
  KALDI_VLOG(4) << "xentropy_aux2_ vector = " << xentropy_aux2_ << "\n";
  KALDI_VLOG(4) << "cross_entropy2 (CE2) =  " << cross_entropy2 << "\n";
  KALDI_VLOG(4) << "T/S Loss (rho*CE1+(1-rho)*CE2) =  " << cross_entropy << "\n";
  KALDI_VLOG(4) << "entropy = "<< entropy << "\n";

  loss_  += cross_entropy;
  loss1_ += cross_entropy1;
  loss2_ += cross_entropy2;
  entropy_ += entropy;
  correct_ += correct;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100; // 1h
    frames_progress_ += num_frames;
    loss_progress_ += cross_entropy;
    entropy_progress_ += entropy;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last "
                    << static_cast<int>(frames_progress_/100/3600) << "h of "
                    << static_cast<int>(frames_/100/3600) << "h]: "
                    << (loss_progress_-entropy_progress_)/frames_progress_ << " (Xent)";
      // store
      loss_vec_.push_back((loss_progress_-entropy_progress_)/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
      entropy_progress_ = 0.0;
    }
  }
}


void TS::Eval2(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat> &net_out1,
			   const CuMatrixBase<BaseFloat> &net_out2,
               const Posterior &post1,
			   const Posterior &post2,
               CuMatrix<BaseFloat> *diff) {
  int32 num_frames = net_out1.NumRows(),
    num_pdf = net_out1.NumCols();

  KALDI_ASSERT(num_frames == net_out2.NumRows());
  KALDI_ASSERT(num_pdf == net_out2.NumCols());

  KALDI_ASSERT(num_frames == post1.size());
  KALDI_ASSERT(num_frames == post2.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post1, num_pdf, &tgt_mat1_);
  PosteriorToMatrix(post2, num_pdf, &tgt_mat2_);

  KALDI_VLOG(4) << "nnet_out1 = " << net_out1 << "\n";
  KALDI_VLOG(4) << "nnet_tgt = " << tgt_mat1_ << "\n";
  KALDI_VLOG(4) << "nnet_out2 = " << net_out2 << "\n";
  KALDI_VLOG(4) << "nnet_tgt2 = " << tgt_mat2_ << "\n";

  // call the other eval function,
  Eval2(frame_weights, net_out1, net_out2, tgt_mat1_, tgt_mat2_, diff);
  KALDI_VLOG(4) << "diff = " << (*diff) << "\n";
}


std::string TS::Report() {
  std::ostringstream oss;
  oss << "AvgLoss: " << (loss_-entropy_)/frames_ << " (AvgTS - AvgTargetEnt), "
      << "[AvgTS: " << loss_/frames_
	  << ", AvgXent (Hard):  " <<  loss1_/frames_
	  << ", AvgXent (Soft):  " <<  loss2_/frames_
	  << ", rho:     " << rho_
      << ", AvgTargetEnt: "    << entropy_/frames_ << "]" << std::endl;
  if (loss_vec_.size() > 0) {
     oss << "progress: [";
     std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
     oss << "]" << std::endl;
  }
  if (correct_ >= 0.0) {
    oss << "FRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<" << std::endl;
  }
  return oss.str();
}

void TS::SetTemperature(const BaseFloat temperature=1.0) {
  KALDI_ASSERT(temperature > 0);
  T_ = temperature;
}

void TS::SetRho(const BaseFloat rho=1.0) {
  KALDI_ASSERT(rho >= 0);
  KALDI_ASSERT(rho <= 1.0);
  rho_ = rho;
}

/* MultiTaskLoss */

void MultiTaskLoss::InitFromString(const std::string& s) {
  std::vector<std::string> v;
  SplitStringToVector(s, ",:" /* delimiter */, false, &v);

  KALDI_ASSERT((v.size()-1) % 3 == 0); // triplets,
  KALDI_ASSERT(v[0] == "multitask"); // header,

  // parse the definition of multitask loss,
  std::vector<std::string>::iterator it(v.begin()+1); // skip header,
  for ( ; it != v.end(); ++it) {
    // type,
    if (*it == "xent") {
      loss_vec_.push_back(new Xent());
    } else if (*it == "mse") {
      loss_vec_.push_back(new Mse());
    } else if (*it == "ts") {
      loss_vec_.push_back(new TS());
      teacherstudent_ = true;
    } else {
      KALDI_ERR << "Unknown objective function code : " << *it;
    }
    ++it;
    // dim,
    int32 dim;
    if (!ConvertStringToInteger(*it, &dim)) {
      KALDI_ERR << "Cannot convert 'dim' " << *it << " to integer!";
    }
    loss_dim_.push_back(dim);
    ++it;
    // weight,
    BaseFloat weight;
    if (!ConvertStringToReal(*it, &weight)) {
      KALDI_ERR << "Cannot convert 'weight' " << *it << " to integer!";
    }
    KALDI_ASSERT(weight >= 0.0);
    loss_weights_.push_back(weight);
  }

  // build vector with starting-point offsets,
  // e.g., if dims= 3:4:2, then offsets = 0:3:7:9 (meaning nodes 0-2 for task 1, nodes 3-6 for task 2, nodes 7-8 for task 3)
  loss_dim_offset_.resize(loss_dim_.size()+1, 0); // 1st zero stays,
  for (int32 i = 1; i <= loss_dim_.size(); i++) {
    loss_dim_offset_[i] = loss_dim_offset_[i-1] + loss_dim_[i-1];
  }

  // sanity check,
  KALDI_ASSERT(loss_vec_.size() > 0);
  KALDI_ASSERT(loss_vec_.size() == loss_dim_.size());
  KALDI_ASSERT(loss_vec_.size() == loss_weights_.size());
}

void MultiTaskLoss::Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const Posterior& post,
            CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_output = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());
  KALDI_ASSERT(num_output == loss_dim_offset_.back()); // sum of loss-dims,

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_output, &tgt_mat_);

  // allocate diff matrix,
  diff->Resize(num_frames, num_output);
  
  /// One vector of frame_weights per loss-function,
  /// The original frame weights are multiplied with
  /// a mask of `defined targets' according to the 'Posterior'.
  /// E.g. If the original frame weights are 0.9 and
  /// there are 10 frames with even and odd numbered frames
  /// belonging to task 1 and task 2 respectively, then
  /// the matrix frmwei_have_tgt will have the following values:
  ///  frames ->  0    1    2    3    4    5    6    7    8    9
  ///  task 1     0.9  0   0.9   0   0.9   0   0.9   0   0.9   0
  ///  task 2     0   0.9   0   0.9   0   0.9   0   0.9   0   0.9
  std::vector<Vector<BaseFloat> > frmwei_have_tgt;
  for (int32 l = 0; l < loss_vec_.size(); l++) {
    // copy original weights,
    frmwei_have_tgt.push_back(Vector<BaseFloat>(frame_weights));
    // We need to mask-out the frames for which the 'posterior' is not defined (= is empty):
    int32 loss_beg = loss_dim_offset_[l];   // first column of loss target,
    int32 loss_end = loss_dim_offset_[l+1]; // (last+1) column of loss target,
    for (int32 f = 0; f < num_frames; f++) {
      bool tgt_defined = false;
      for (int32 p = 0; p < post[f].size(); p++) {
        if (post[f][p].first >= loss_beg && post[f][p].first < loss_end) {
          tgt_defined = true;
          break;
        }
      }
      if (!tgt_defined) {
          frmwei_have_tgt[l](f) = 0.0; // set zero_weight for the frame with no targets!
      }
    }
  }

  // call the vector of loss functions,
  CuMatrix<BaseFloat> diff_aux;
  for (int32 l = 0; l < loss_vec_.size(); l++) {
	// target interpolation is only for the first softmax
    if (tgt_interp_mode_.compare("none") != 0 && l == 0) {
      loss_vec_[l]->Set_Target_Interp(tgt_interp_mode_, rho_);
      loss_vec_[l]->Eval(frmwei_have_tgt[l],
        net_out.ColRange(loss_dim_offset_[l], loss_dim_[l]),
        tgt_mat_.ColRange(loss_dim_offset_[l], loss_dim_[l]),
        &diff_aux);
	} else {
	  loss_vec_[l]->Set_Target_Interp("none", 1.0);
	  loss_vec_[l]->Eval(frmwei_have_tgt[l],
	    net_out.ColRange(loss_dim_offset_[l], loss_dim_[l]),
	    tgt_mat_.ColRange(loss_dim_offset_[l], loss_dim_[l]),
		&diff_aux);
	}

    // Scale the gradients,
    diff_aux.Scale(loss_weights_[l]);
    // Copy to diff,
    diff->ColRange(loss_dim_offset_[l], loss_dim_[l]).CopyFromMat(diff_aux);
  }

  /* 
  // call the vector of loss functions,
  CuMatrix<BaseFloat> diff_aux;
  for (int32 i = 0; i < loss_vec_.size(); i++) {
	//KALDI_LOG << "Evaluating Task  " << i + 1 << " , Weight = " << loss_weights_[i] << std::endl;
    loss_vec_[i]->Eval(frame_weights, 
      net_out.ColRange(loss_dim_offset_[i], loss_dim_[i]),
      tgt_mat_.ColRange(loss_dim_offset_[i], loss_dim_[i]),
      &diff_aux);
    // Scale the gradients,
    diff_aux.Scale(loss_weights_[i]);
    //KALDI_LOG << "loss_dim_offset_[ " << i << "] = " << loss_dim_offset_[i] << "\n";
    //KALDI_LOG << "loss_dim_[ " << i << "] = " << loss_dim_[i] << "\n";
    //KALDI_LOG << "net_out = " << net_out.ColRange(loss_dim_offset_[i], loss_dim_[i]) << "\n";
    //KALDI_LOG << "target mat = " << tgt_mat_.ColRange(loss_dim_offset_[i], loss_dim_[i]) << "\n";
    //KALDI_LOG << "diff aux = " << diff_aux << "\n";
    // Copy to diff,
    diff->ColRange(loss_dim_offset_[i], loss_dim_[i]).CopyFromMat(diff_aux);
  } */
}

void MultiTaskLoss::Eval2(const VectorBase<BaseFloat> &frame_weights,
                         const CuMatrixBase<BaseFloat> &net_out1,
			             const CuMatrixBase<BaseFloat> &net_out2,
                         const Posterior &post1,
			             const Posterior &post2,
                         CuMatrix<BaseFloat> *diff) {

	int32 num_frames = net_out1.NumRows(),
         num_output = net_out1.NumCols();

	KALDI_ASSERT(num_frames == net_out2.NumRows());
	KALDI_ASSERT(num_output == net_out2.NumCols());

	KALDI_ASSERT(num_frames == post1.size());
	KALDI_ASSERT(num_frames == post2.size());
	KALDI_ASSERT(num_output == loss_dim_offset_.back()); // sum of loss-dims,

	// convert posterior to matrix,
	PosteriorToMatrix(post1, num_output, &tgt_mat1_);
	PosteriorToMatrix(post2, num_output, &tgt_mat2_);

	KALDI_VLOG(4) << "nnet_out1 = " << net_out1 << "\n";
	KALDI_VLOG(4) << "nnet_tgt1 = " << tgt_mat1_ << "\n";
	KALDI_VLOG(4) << "nnet_out2 = " << net_out2 << "\n";
	KALDI_VLOG(4) << "nnet_tgt2 = " << tgt_mat2_ << "\n";

	// allocate diff matrix,
	diff->Resize(num_frames, num_output);

	/// One vector of frame_weights per loss-function,
	/// The original frame weights are multiplied with
	/// a mask of `defined targets' according to the 'Posterior'.
	std::vector<Vector<BaseFloat> > frmwei_have_tgt;
	for (int32 l = 0; l < loss_vec_.size(); l++) {
	  // copy original weights,
	  frmwei_have_tgt.push_back(Vector<BaseFloat>(frame_weights));
	  // We need to mask-out the frames for which the 'posterior' is not defined (= is empty):
	  int32 loss_beg = loss_dim_offset_[l];   // first column of loss target,
	  int32 loss_end = loss_dim_offset_[l+1]; // (last+1) column of loss target,
	  for (int32 f = 0; f < num_frames; f++) {
	    bool tgt_defined = false;
	    for (int32 p = 0; p < post1[f].size(); p++) {
	      if (post1[f][p].first >= loss_beg && post1[f][p].first < loss_end) {
	        tgt_defined = true;
	        break;
	      }
	    }
	    if (!tgt_defined) {
	        frmwei_have_tgt[l](f) = 0.0; // set zero_weight for the frame with no targets!
	    }
	  }
    }

	// call the vector of loss functions,
	CuMatrix<BaseFloat> diff_aux;
	for (int32 l = 0; l < loss_vec_.size(); l++) {
	  if (l == 0){
		loss_vec_[0]->SetTemperature(T_);
		loss_vec_[0]->SetRho(rho_);
		loss_vec_[0]->Eval2(frmwei_have_tgt[0],
		  net_out1.ColRange(loss_dim_offset_[0], loss_dim_[0]),
		  net_out2.ColRange(loss_dim_offset_[0], loss_dim_[0]),
		  tgt_mat1_.ColRange(loss_dim_offset_[0], loss_dim_[0]),
		  tgt_mat2_.ColRange(loss_dim_offset_[0], loss_dim_[0]),
		  &diff_aux);
	  } else {
	  loss_vec_[l]->Eval(frmwei_have_tgt[l],
	    net_out1.ColRange(loss_dim_offset_[l], loss_dim_[l]),
	    tgt_mat1_.ColRange(loss_dim_offset_[l], loss_dim_[l]),
	    &diff_aux);
      }

	  // Scale the gradients,
	  diff_aux.Scale(loss_weights_[l]);
	  // Copy to diff,
	  diff->ColRange(loss_dim_offset_[l], loss_dim_[l]).CopyFromMat(diff_aux);
    }
	KALDI_VLOG(4) << "diff = " << (*diff) << "\n";
}


void MultiTaskLoss::Set_Target_Interp(const std::string tgt_interp_mode="none", const BaseFloat rho=1.0) {
  if (rho_ < 0 || rho_ > 1.0) {
    KALDI_ERR << "rho = " << rho_ << " is outside the acceptable range [0, 1]\n";
  }

  tgt_interp_mode_ =   tgt_interp_mode;

  if (tgt_interp_mode.compare("none") == 0) {
    rho_ = 1.0;
  } else {
    rho_  = rho;
  }
}

bool MultiTaskLoss::IsTS() {
	return teacherstudent_ ;
}

void MultiTaskLoss::SetTemperature(const BaseFloat temperature=1.0) {
  KALDI_ASSERT(temperature > 0);
  T_ = temperature;
}

void MultiTaskLoss::SetRho(const BaseFloat rho=1.0) {
  KALDI_ASSERT(rho >= 0);
  KALDI_ASSERT(rho <= 1.0);
  rho_ = rho;
}

std::string MultiTaskLoss::Report() {
  // calculate overall loss (weighted),
  BaseFloat overall_loss = AvgLoss();
  // copy the loss-values into a vector,
  std::vector<BaseFloat> loss_values;
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    loss_values.push_back(loss_vec_[i]->AvgLoss());
  }

  // build the message,
  std::ostringstream oss;
  oss << "MultiTaskLoss, with " << loss_vec_.size() << " parallel loss functions." << std::endl;
  // individual loss reports first,
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    oss << "Loss " << i+1 << ", " << loss_vec_[i]->Report() << std::endl;
  }

  // overall loss is last,
  oss << "Loss (OVERALL), " 
      << "AvgLoss: " << overall_loss << " (MultiTaskLoss), "
      << "weights " << loss_weights_ << ", "
      << "values " << loss_values << std::endl;

  return oss.str();
}

BaseFloat MultiTaskLoss::AvgLoss() {
  BaseFloat ans(0.0);
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    ans += loss_weights_[i] * loss_vec_[i]->AvgLoss();
  }
  return ans;
}

} // namespace nnet1
} // namespace kaldi
