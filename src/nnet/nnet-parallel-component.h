// nnet/nnet-parallel-component.h

// Copyright 2014  Brno University of Technology (Author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_PARALLEL_COMPONENT_H_
#define KALDI_NNET_NNET_PARALLEL_COMPONENT_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

#include <sstream>

namespace kaldi {
namespace nnet1 {

class ParallelComponent : public UpdatableComponent {
 public:
  ParallelComponent(int32 dim_in, int32 dim_out) 
    : UpdatableComponent(dim_in, dim_out)
  { }
  ~ParallelComponent()
  { }

  Component* Copy() const { return new ParallelComponent(*this); }
  ComponentType GetType() const { return kParallelComponent; }

  void InitData(std::istream &is) {
    // define options
    std::vector<std::string> nested_nnet_proto;
    std::vector<std::string> nested_nnet_filename;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<NestedNnetFilename>") {
        while(!is.eof()) {
          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</NestedNnetFilename>") break;
          nested_nnet_filename.push_back(file_or_end);
        }
      } else if (token == "<NestedNnetProto>") {
        while(!is.eof()) {
          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</NestedNnetProto>") break;
          nested_nnet_proto.push_back(file_or_end);
        }
      } else KALDI_ERR << "Unknown token " << token << ", typo in config?"
                       << " (NestedNnetFilename|NestedNnetProto)";
      is >> std::ws; // eat-up whitespace
    }
    // initialize
    KALDI_ASSERT((nested_nnet_proto.size() > 0) ^ (nested_nnet_filename.size() > 0)); //xor
    // read nnets from files
    if (nested_nnet_filename.size() > 0) {
      for (int32 i=0; i<nested_nnet_filename.size(); i++) {
        Nnet nnet;
        nnet.Read(nested_nnet_filename[i]);
        nnet_.push_back(nnet);
        KALDI_LOG << "Loaded nested <Nnet> from file : " << nested_nnet_filename[i];
      }
    }
    // initialize nnets from prototypes
    if (nested_nnet_proto.size() > 0) {
      for (int32 i=0; i<nested_nnet_proto.size(); i++) {
        Nnet nnet;
        nnet.Init(nested_nnet_proto[i]);
        nnet_.push_back(nnet);
        KALDI_LOG << "Initialized nested <Nnet> from prototype : " << nested_nnet_proto[i];
      }
    }
    // check dim-sum of nested nnets
    int32 nnet_input_sum = 0, nnet_output_sum = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      nnet_input_sum += nnet_[i].InputDim();
      nnet_output_sum += nnet_[i].OutputDim();
    }
    KALDI_ASSERT(InputDim() == nnet_input_sum);
    KALDI_ASSERT(OutputDim() == nnet_output_sum);
  }

  void ReadData(std::istream &is, bool binary) {
    // read
    ExpectToken(is, binary, "<NestedNnetCount>");
    int32 nnet_count;
    ReadBasicType(is, binary, &nnet_count);
    for (int32 i=0; i<nnet_count; i++) {
      ExpectToken(is, binary, "<NestedNnet>");
      int32 dummy;
      ReadBasicType(is, binary, &dummy);
      Nnet nnet;
      nnet.Read(is, binary);
      nnet_.push_back(nnet);
    }
    ExpectToken(is, binary, "</ParallelComponent>");

    // check dim-sum of nested nnets
    int32 nnet_input_sum = 0, nnet_output_sum = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      nnet_input_sum += nnet_[i].InputDim();
      nnet_output_sum += nnet_[i].OutputDim();
    }
    KALDI_ASSERT(InputDim() == nnet_input_sum);
    KALDI_ASSERT(OutputDim() == nnet_output_sum);
  }

  void WriteData(std::ostream &os, bool binary) const {
    // useful dims
    int32 nnet_count = nnet_.size();
    //
    WriteToken(os, binary, "<NestedNnetCount>");
    WriteBasicType(os, binary, nnet_count);
    for (int32 i=0; i<nnet_count; i++) {
      WriteToken(os, binary, "<NestedNnet>");
      WriteBasicType(os, binary, i+1);
      nnet_[i].Write(os, binary);
    }
    WriteToken(os, binary, "</ParallelComponent>");
  }

  Nnet& GetNestedNnet(int32 id) { return nnet_.at(id); }

  const Nnet& GetNestedNnet(int32 id) const { return nnet_.at(id); }

  int32 NumParams() const { 
    int32 num_params_sum = 0;
    for (int32 i=0; i<nnet_.size(); i++) 
      num_params_sum += nnet_[i].NumParams();
    return num_params_sum;
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const { 
    wei_copy->Resize(NumParams());
    int32 offset = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      Vector<BaseFloat> wei_aux;
      nnet_[i].GetParams(&wei_aux);
      wei_copy->Range(offset, wei_aux.Dim()).CopyFromVec(wei_aux);
      offset += wei_aux.Dim();
    }
    KALDI_ASSERT(offset == NumParams());
  }
    
  std::string Info() const { 
    std::ostringstream os;
    for (int32 i=0; i<nnet_.size(); i++) {
      os << "nested_network #" << i+1 << "{\n" << nnet_[i].Info() << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1); // removing last '\n'
    return s;
  }
                       
  std::string InfoGradient() const {
    std::ostringstream os;
    for (int32 i=0; i<nnet_.size(); i++) {
      os << "nested_gradient #" << i+1 << "{\n" << nnet_[i].InfoGradient() << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1); // removing last '\n'
    return s;
  }

  std::string InfoPropagate() const {
    std::ostringstream os;
    for (int32 i=0; i<nnet_.size(); i++) {
      os << "nested_propagate #" << i+1 << "{\n" << nnet_[i].InfoPropagate() << "}\n";
    }
    return os.str();
  }

  std::string InfoBackPropagate() const {
    std::ostringstream os;
    for (int32 i=0; i<nnet_.size(); i++) {
      os << "nested_backpropagate #" << i+1 << "{\n" << nnet_[i].InfoBackPropagate() << "}\n";
    }
    return os.str();
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    int32 input_offset = 0, output_offset = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      CuSubMatrix<BaseFloat> src(in.ColRange(input_offset, nnet_[i].InputDim()));
      CuSubMatrix<BaseFloat> tgt(out->ColRange(output_offset, nnet_[i].OutputDim()));
      //
      CuMatrix<BaseFloat> tgt_aux;
      nnet_[i].Propagate(src, &tgt_aux);
      tgt.CopyFromMat(tgt_aux);
      //
      input_offset += nnet_[i].InputDim();
      output_offset += nnet_[i].OutputDim();
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    int32 input_offset = 0, output_offset = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      CuSubMatrix<BaseFloat> src(out_diff.ColRange(output_offset, nnet_[i].OutputDim()));
      CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(input_offset, nnet_[i].InputDim()));
      // 
      CuMatrix<BaseFloat> tgt_aux;
      nnet_[i].Backpropagate(src, &tgt_aux);
      tgt.CopyFromMat(tgt_aux);
      //
      input_offset += nnet_[i].InputDim();
      output_offset += nnet_[i].OutputDim();
    }

    #if 0
	std::vector<Component*> c;
	for (int32 i=0; i<nnet_.size(); i++) {
	  int32 last_component = nnet_[i].NumComponents() - 1;
	  c.push_back(nnet_[i].GetComponent(last_component).Copy());
	}

	int32 input_offset = 0, output_offset = 0;
	float eps = 1e-06;
	std::vector<MatrixIndexT> softmax_indices;
	int32 nrows = out_diff.NumRows();
	int32 ncols = nnet_.size();
	Matrix<BaseFloat> row_diff_mask(nrows,ncols,kSetZero);

	// Set masks for those tasks whose last component is Softmax
	for (int32 i=0; i<ncols; i++) {
	  if (c[i]->GetType() == Component::kSoftmax) {
		softmax_indices.push_back(i);
	    CuSubMatrix<BaseFloat> src(out_diff.ColRange(output_offset, nnet_[i].OutputDim()));
	    CuVector<BaseFloat> row_sum(src.NumRows());
	    row_sum.AddColSumMat(1.0, src, 0.0); // 0:keep, 1:zero-out
	    // KALDI_LOG << "row_sum[task " << i << "] = "   << row_sum << "\n";
	    // Copy CuVector to Vector. Compute the masks in Vector class.
	    Vector<BaseFloat> r_sum(row_sum.Dim());
	    row_sum.CopyToVec(&r_sum);
	    r_sum.ApplyAbs();
	    r_sum.LowerThreshold(eps, 0);
	    // Copy Vector to the i^th col of CuMatrix
	    row_diff_mask.CopyColFromVec(r_sum, i);
	    // Compute the masks in CuMatrix
	    row_diff_mask.ColRange(i,1).Scale(-1.0); // 0:keep, -1:zero-out
	    row_diff_mask.ColRange(i,1).Add(1.0); // 1:keep, 0:zero-out
	    // KALDI_LOG << "row_diff_mask[" << i << "] = " << row_diff_mask << "\n";
	  }
	  output_offset += nnet_[i].OutputDim();
	}

	// Determine the masks for those tasks whose last component is not Softmax
	// Assume there is only ONE task with a non-Softmax component
	// Mask: 1 = Task active, 0 = Task inactive
	// Rule: If all the masks for a given frame were set to 0, then the frame must belong to
	// the task whose last component is not Softmax. Thus, set the mask for this task to 1.
	for (int32 i=0; i<ncols; i++) {
	  if (c[i]->GetType() != Component::kSoftmax) {
	    // check if mask for softmask based tasks are all 0's (0 = inactive)
		for (int32 j=0; j<nrows; j++) {
		  bool softmax_inactive = true;
		  for (int32 k=0; (k<softmax_indices.size()) && softmax_inactive; k++) {
		    if (!row_diff_mask.Range(j,1,softmax_indices.at(k),1).IsZero(eps))
		      softmax_inactive = false; // frame j belongs to a softmax task
		      break;
		  }
		  if (softmax_inactive) row_diff_mask.Range(j,1,i,1).Set(1.0);
	    }
	    // KALDI_LOG << "row_diff_mask[" << i << "] (non-softmax)= " << row_diff_mask << "\n";
	  }
    }

	// KALDI_LOG << "out_diff = " << out_diff << "\n";
	// KALDI_LOG << "row_diff_mask = " << row_diff_mask << "\n";
	input_offset = 0, output_offset = 0;
	CuMatrix<BaseFloat> row_diff_mask_cu(row_diff_mask, kNoTrans);
	for (int32 i=0; i<ncols; i++) {
	  CuSubMatrix<BaseFloat> src(out_diff.ColRange(output_offset, nnet_[i].OutputDim()));
	  CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(input_offset, nnet_[i].InputDim()));
	  CuMatrix<BaseFloat> src_masked(src, kNoTrans);
	  //
	  CuVector<BaseFloat> row_mask(nrows);
	  row_mask.CopyColFromMat(row_diff_mask_cu,i);
	  src_masked.MulRowsVec(row_mask);
	  CuMatrix<BaseFloat> tgt_aux;
	  nnet_[i].Backpropagate(src_masked, &tgt_aux);
	  tgt.CopyFromMat(tgt_aux);
	  //
	  input_offset += nnet_[i].InputDim();
	  output_offset += nnet_[i].OutputDim();
	  // KALDI_LOG << "out_diff[ " << i << "] = " << src << "\n";
  	  // KALDI_LOG << "out_diff[ " << i << "] (masked) = " << src_masked << "\n";
  	  // KALDI_LOG << "bprop o/p of task " << i << " = " << tgt << "\n";
	}
    #endif
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    ; // do nothing
  }
 
  void SetTrainOptions(const NnetTrainOptions &opts) {
    for (int32 i=0; i<nnet_.size(); i++) {
      nnet_[i].SetTrainOptions(opts);
    }
  }

 private:
  std::vector<Nnet> nnet_;
};

} // namespace nnet1
} // namespace kaldi

#endif