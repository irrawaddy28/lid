// nnet/nnet-loss.h

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

#ifndef KALDI_NNET_NNET_LOSS_H_
#define KALDI_NNET_NNET_LOSS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet1 {


class LossItf {
 public:
  LossItf() { }
  virtual ~LossItf() { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat> &net_out, 
            const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff) = 0;

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat> &net_out, 
            const Posterior &target,
            CuMatrix<BaseFloat> *diff) = 0;
  
  /// Evaluate Teacher-Student Loss (for details, refer the comments in class TS)
  virtual void Eval2(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out1,
    			const CuMatrixBase<BaseFloat> &net_out2,
                const Posterior &target1,
    			const Posterior &target2,
                CuMatrix<BaseFloat> *diff) {
    KALDI_ERR << "This is not supposed to be called as a base class method! Call the derived class method instead";
  }

  /// Evaluate Teacher-Student Loss (when targets are presented as matrix class form rather than Posterior class)
  virtual void Eval2(const VectorBase<BaseFloat> &frame_weights,
              const CuMatrixBase<BaseFloat> &net_out1,
  			  const CuMatrixBase<BaseFloat> &net_out2,
              const CuMatrixBase<BaseFloat> &target1,
  			  const CuMatrixBase<BaseFloat> &target2,
              CuMatrix<BaseFloat> *diff) {
    KALDI_ERR << "This is not supposed to be called as a base class method! Call the derived class method instead";
  }

  /// Set target interpolation mode and weight
  virtual void Set_Target_Interp(const std::string tgt_interp_mode,
  		    const BaseFloat rho) = 0;

  /// Set temperature for softmax (used in Teacher-Student training; for details, refer the comments in class TS)
  virtual void SetTemperature(const BaseFloat temperature) { }

  /// Set the weights for Teacher-Student loss (for details, refer the comments in class TS)
  virtual void SetRho(const BaseFloat rho) { }

  /// Generate string with error report,
  virtual std::string Report() = 0;

  /// Get loss value (frame average),
  virtual BaseFloat AvgLoss() = 0;

};


class Xent : public LossItf {
 public:
  Xent() : frames_(0.0), correct_(0.0), loss_(0.0), entropy_(0.0),
           tgt_interp_mode_("none"), rho_(1.0),
           frames_progress_(0.0), loss_progress_(0.0), entropy_progress_(0.0) { }
  ~Xent() { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat> &net_out, 
            const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff);

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat> &net_out, 
            const Posterior &target,
            CuMatrix<BaseFloat> *diff);
  
  /// Generate string with error report,
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
	if (frames_ == 0) return 0.0;
    return (loss_ - entropy_) / frames_;
  }

  // Set target interpolation mode and weight
  void Set_Target_Interp(const std::string tgt_interp_mode, const BaseFloat rho);


 private: 
  double frames_;
  double correct_;
  double loss_;
  double entropy_;
  std::string tgt_interp_mode_;
  BaseFloat rho_;

  // partial results during training
  double frames_progress_;
  double loss_progress_;
  double entropy_progress_;
  std::vector<float> loss_vec_;

  // weigting buffer,
  CuVector<BaseFloat> frame_weights_;

  // loss computation buffers
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> xentropy_aux_;
  CuMatrix<BaseFloat> entropy_aux_;

  // frame classification buffers, 
  CuArray<int32> max_id_out_;
  CuArray<int32> max_id_tgt_;
};

class Mse : public LossItf {
 public:
  Mse() : frames_(0.0), loss_(0.0), 
          frames_progress_(0.0), loss_progress_(0.0) { }
  ~Mse() { }

  /// Evaluate mean square error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const CuMatrixBase<BaseFloat>& target,
            CuMatrix<BaseFloat>* diff);

  /// Evaluate mean square error using target-posteior,
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const Posterior& target,
            CuMatrix<BaseFloat>* diff);
  
  /// Generate string with error report
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
	if (frames_ == 0) return 0.0;
    return loss_ / frames_;
  }

  // Set target interpolation mode and weight: No interpolation for MSE
  void Set_Target_Interp(const std::string tgt_interp_mode, const BaseFloat rho) { }

 private:
  double frames_;
  double loss_;
  
  double frames_progress_;
  double loss_progress_;
  std::vector<float> loss_vec_;

  CuVector<BaseFloat> frame_weights_;
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> diff_pow_2_;
};

/* Teacher-Student loss (aka Knowledge Distillation)
   C = rho*CE1 + (1-rho)*CE2
   CE1 = - sum_{k} d_k    log p_k(1)
   CE2 = - sum_{k} q_k(T) log p_k(T)
   where,
   d_k = ground truth labels (aka hard labels) from the corpus
   p_k(T) = softmax output of the student network parameterized by T
          = exp(z_k / T)/ ( sum_j exp(z_j / T) )  (z = logits of the student nnet)
   q_k(T) = softmax output of the teacher network parameterized by T (aka soft labels)
          = exp(v_k / T)/ ( sum_j exp(v_j / T) )  (v = logits of the teacher nnet)
*/
class TS : public LossItf {
 public:
   TS() : frames_(0.0), correct_(0.0), loss_(0.0), loss1_(0.0), loss2_(0.0), entropy_(0.0),
          frames_progress_(0.0), loss_progress_(0.0), entropy_progress_(0.0),
		  T_(1.0), rho_(1.0) { }
  ~TS() { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff) {
    KALDI_ERR << "This is not supposed to be called!";
  }

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const Posterior &target,
            CuMatrix<BaseFloat> *diff) {
    KALDI_ERR << "This is not supposed to be called!";
  }

  /// Evaluate cross entropy using 2 kinds of target-matrices (supports soft labels),
  void Eval2(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out1,
			const CuMatrixBase<BaseFloat> &net_out2,
            const CuMatrixBase<BaseFloat> &target1,
			const CuMatrixBase<BaseFloat> &target2,
            CuMatrix<BaseFloat> *diff);

  /// Evaluate cross entropy using 2 kinds of target-posteriors (supports soft labels),
  void Eval2(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out1  /* Output of student network with softmax temperature = 1 */,
			const CuMatrixBase<BaseFloat> &net_out2  /* Output of student network with softmax temperature = T > 0 */,
            const Posterior &target1  /* Ground truth targets, usually 1-hot, used to train the student network */,
			const Posterior &target2  /* Soft targets generated by a teacher network with softmax temperature = T, used to train the student network */,
            CuMatrix<BaseFloat> *diff);

  /// Generate string with error report,
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
	if (frames_ == 0) return 0.0;
    return (loss_ - entropy_) / frames_;
  }

  /// Set target interpolation mode and weight
  void Set_Target_Interp(const std::string tgt_interp_mode, const BaseFloat rho) { }

  /// Set temperature for softmax
  void SetTemperature(const BaseFloat temperature);

  // Set the weights for Teacher-Student loss
  void SetRho(const BaseFloat rho);

 private:
  double frames_;
  double correct_;
  double loss_, loss1_, loss2_;
  double entropy_;

  // partial results during training
  double frames_progress_;
  double loss_progress_;
  double entropy_progress_;
  std::vector<float> loss_vec_;

  // Temperature and rho (wt between the CE loss w/ hard targets and CE loss w/ soft targets)
  BaseFloat T_;
  BaseFloat rho_;

  // weigting buffer,
  CuVector<BaseFloat> frame_weights_;

  // loss computation buffers
  CuMatrix<BaseFloat> tgt_mat1_;
  CuMatrix<BaseFloat> tgt_mat2_;
  CuMatrix<BaseFloat> xentropy_aux1_;
  CuMatrix<BaseFloat> xentropy_aux2_;
  CuMatrix<BaseFloat> entropy_aux_;

  // frame classification buffers,
  CuArray<int32> max_id_out_;
  CuArray<int32> max_id_tgt_;
};



class MultiTaskLoss : public LossItf {
 public:
  MultiTaskLoss() : tgt_interp_mode_("none"),
  	  	  	  	  	teacherstudent_(false), T_(1.0),  rho_(1.0) { }
  ~MultiTaskLoss() {
    while (loss_vec_.size() > 0) {
      delete loss_vec_.back();
      loss_vec_.pop_back();
    }
  }

  /// Initialize from string, the format for string 's' is :
  /// 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
  ///
  /// Practically it can look like this :
  /// 'multitask,xent,2456,1.0,mse,440,0.001'
  void InitFromString(const std::string& s);

  /// Evaluate mean square error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const CuMatrixBase<BaseFloat>& target,
            CuMatrix<BaseFloat>* diff) {
    KALDI_ERR << "This is not supposed to be called!";
  }

  /// Evaluate mean square error using target-posteior,
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const Posterior& target,
            CuMatrix<BaseFloat>* diff);
  
  /// For Teacher-Student loss
  void Eval2(const VectorBase<BaseFloat> &frame_weights,
             const CuMatrixBase<BaseFloat> &net_out1,
   			 const CuMatrixBase<BaseFloat> &net_out2,
             const CuMatrixBase<BaseFloat> &target1,
   			 const CuMatrixBase<BaseFloat> &target2,
             CuMatrix<BaseFloat> *diff) {
     KALDI_ERR << "This is not supposed to be called!";
  }

  /// For Teacher-Student loss
  void Eval2(const VectorBase<BaseFloat> &frame_weights,
             const CuMatrixBase<BaseFloat> &net_out1 /* Output of student MTL network with softmax temperature = 1 */,
   			 const CuMatrixBase<BaseFloat> &net_out2 /* Output of student MTL network with softmax temperature = T > 0 */,
             const Posterior &target1 /* Ground truth targets, usually 1-hot, used to train the student network */,
  			 const Posterior &target2 /* Soft targets generated by a teacher network with softmax temperature = T, used to train the student network */,
             CuMatrix<BaseFloat> *diff);

  // Set target interpolation mode and weight
  void Set_Target_Interp(const std::string tgt_interp_mode, const BaseFloat rho);

  // Check if Teacher-Student training is enabled
  bool IsTS();

  // Set the softmax temperature; valid when Teacher-Student training is enabled
  void SetTemperature(const BaseFloat temperature);

  // Set rho; valid when Teacher-Student training is enabled
  void SetRho(const BaseFloat rho);

  /// Generate string with error report
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss();


 private:
  std::vector<LossItf*>  loss_vec_;
  std::vector<int32>     loss_dim_;
  std::vector<BaseFloat> loss_weights_;
  
  std::vector<int32>     loss_dim_offset_;

  CuMatrix<BaseFloat>    tgt_mat_, tgt_mat1_, tgt_mat2_;

  std::string tgt_interp_mode_;

  // Temperature and rho (wt between the CE loss w/ hard targets and CE loss w/ soft targets)
  bool teacherstudent_;
  BaseFloat T_;
  BaseFloat rho_;
};

} // namespace nnet1
} // namespace kaldi

#endif

