// nnet/nnet-activation.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_ACTIVATION_H_
#define KALDI_NNET_NNET_ACTIVATION_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet1 {

class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out),
	  T_(1.0)
  { }
  ~Softmax()
  { }

  Component* Copy() const { return new Softmax(*this); }
  ComponentType GetType() const { return kSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
	// y = e^(x_j/T)/sum_j(e^(x_j/T))
	CuMatrix<BaseFloat> inT(in.NumRows(), in.NumCols());
	inT.CopyFromMat(in);
	inT.Scale(1.0/T_);
	out->ApplySoftMaxPerRow(inT);  // out->ApplySoftMaxPerRow(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }


  void SetTemperature(const BaseFloat temperature=1.0) {
    KALDI_ASSERT(temperature > 0);
    T_ = temperature;    
  }

  std::string Info() const {
    return "\n  Temperature = " + ToString(T_);
  }

  private:
    BaseFloat T_;
};



class BlockSoftmax : public Component {
 public:
  BlockSoftmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out),
	  T_(1.0)
  { }
  ~BlockSoftmax()
  { }

  Component* Copy() const { return new BlockSoftmax(*this); }
  ComponentType GetType() const { return kBlockSoftmax; }
  
  void InitData(std::istream &is) {
    // parse config
    std::string token,
      dims_str;
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<BlockDims>") is >> dims_str;
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BlockDims)";
      is >> std::ws; // eat-up whitespace
    }
    // parse dims,
    if (!kaldi::SplitStringToIntegers(dims_str, ":", false, &block_dims))
      KALDI_ERR << "Invalid block-dims " << dims_str;
    // sanity check
    int32 sum = 0;
    for (int32 i=0; i<block_dims.size(); i++) {
      sum += block_dims[i];
    }
    KALDI_ASSERT(sum == OutputDim());

    // Initializing block_offset array early to enable unit testing
    // We repeat this initialization later in ReadData()
    block_offset.resize(block_dims.size()+1, 0);
    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_dims[i];
    }
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void ReadData(std::istream &is, bool binary) {
    ReadIntegerVector(is, binary, &block_dims);
    block_offset.resize(block_dims.size()+1, 0);
    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_dims[i];
    }
    // check
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteIntegerVector(os, binary, block_dims);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // perform softmax per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> in_bl = in.ColRange(block_offset[bl], block_dims[bl]);
      CuSubMatrix<BaseFloat> out_bl = out->ColRange(block_offset[bl], block_dims[bl]);
      // y = e^(x_j/T)/sum_j(e^(x_j/T))

      KALDI_VLOG(4) << " T = " << T_ << ", Block = " << bl << ", num_frames =  " << in_bl.NumRows() << ", num_pdf =  " << in_bl.NumCols() << "\n";
      out_bl.CopyFromMat(in_bl);
      if (T_ != 1) {
        KALDI_VLOG(4) << "in_bl  (before scale) = " << in_bl << "\n";
        KALDI_VLOG(4) << "out_bl (before scale) = " << out_bl << "\n";
      }
      if (bl == 0) out_bl.Scale(1.0/T_);  // Temperature only for the first softmax
      if (T_ != 1) {
        KALDI_VLOG(4) << "in_bl  (after scale) = " << in_bl << "\n";
        KALDI_VLOG(4) << "out_bl (after scale) = " << out_bl << "\n";
      }
      out_bl.ApplySoftMaxPerRow(out_bl);  //  out_bl.ApplySoftMaxPerRow(in_bl)
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // copy the error derivative:
    // (assuming we already got softmax-cross-entropy derivative in out_diff)
    in_diff->CopyFromMat(out_diff);
    
    // zero-out line-in-block, where sum different from zero,
    // process per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> diff_bl = in_diff->ColRange(block_offset[bl], block_dims[bl]);
      CuVector<BaseFloat> row_sum(diff_bl.NumRows());
      // std::cout << "block index = " << bl << ", offset = " << block_offset[bl] << "\n";
      // std::cout << "diff_bl (init) = " << diff_bl << "\n";
      row_sum.AddColSumMat(1.0, diff_bl, 0.0); // 0:keep, 1:zero-out
      // std::cout << "row_sum = " << row_sum << "\n";
      // we'll scale rows by 0/1 masks
      CuVector<BaseFloat> row_diff_mask(row_sum);
      row_diff_mask.Scale(-1.0); // 0:keep, -1:zero-out
      // std::cout << "row_diff_mask (scale) = " << row_diff_mask << "\n";
      row_diff_mask.Add(1.0); // 1:keep, 0:zero-out
      // std::cout << "row_diff_mask (add) = " << row_diff_mask << "\n";
      // here we should have only 0 and 1
      diff_bl.MulRowsVec(row_diff_mask);
      // std::cout << "diff_bl (final) = " << diff_bl << "\n";
    }
  }

  void SetTemperature(const BaseFloat temperature=1.0) {
    KALDI_ASSERT(temperature > 0);
    T_ = temperature; // Temperature only for the first softmax
  }

  std::string Info() const {
    return "\n  softmax-dims " + ToString(block_dims) + "\n  softmax-temperature (of 1st block)  " + ToString(T_);
  }

  std::vector<int32> block_dims;
  std::vector<int32> block_offset;
  BaseFloat T_;
};


class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Sigmoid()
  { }

  Component* Copy() const { return new Sigmoid(*this); }
  ComponentType GetType() const { return kSigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->DiffSigmoid(out, out_diff);
  }
};



class Tanh : public Component {
 public:
  Tanh(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Tanh()
  { }

  Component* Copy() const { return new Tanh(*this); }
  ComponentType GetType() const { return kTanh; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = (e^x - e^(-x)) / (e^x + e^(-x))
    out->Tanh(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = (1 - y^2)ex
    in_diff->DiffTanh(out, out_diff);
  }
};



class Dropout : public Component {
 public:
  Dropout(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out), dropout_retention_(0.5)
  { }
  ~Dropout()
  { }

  Component* Copy() const { return new Dropout(*this); }
  ComponentType GetType() const { return kDropout; }

  void InitData(std::istream &is) {
    is >> std::ws; // eat-up whitespace
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<DropoutRetention>") ReadBasicType(is, false, &dropout_retention_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (DropoutRetention)";
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void ReadData(std::istream &is, bool binary) {
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<DropoutRetention>");
      ReadBasicType(is, binary, &dropout_retention_);
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<DropoutRetention>");
    WriteBasicType(os, binary, dropout_retention_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    // switch off 50% of the inputs...
    dropout_mask_.Resize(out->NumRows(),out->NumCols());
    dropout_mask_.Set(dropout_retention_);
    rand_.BinarizeProbs(dropout_mask_,&dropout_mask_);
    out->MulElements(dropout_mask_);
    // rescale to keep same dynamic range as w/o dropout
    out->Scale(1.0/dropout_retention_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // use same mask on the error derivatives...
    in_diff->MulElements(dropout_mask_);
    // enlarge output to fit dynamic range w/o dropout
    in_diff->Scale(1.0/dropout_retention_);
  }
  
  BaseFloat GetDropoutRetention() {
    return dropout_retention_;
  }

  void SetDropoutRetention(BaseFloat dr) {
    dropout_retention_ = dr;
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

 private:
  CuRand<BaseFloat> rand_;
  CuMatrix<BaseFloat> dropout_mask_;
  BaseFloat dropout_retention_;
};



} // namespace nnet1
} // namespace kaldi

#endif

