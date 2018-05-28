// nnet/nnet-component-test.cc
// Copyright 2014-2015  Brno University of Technology (author: Karel Vesely),
//                      The Johns Hopkins University (author: Sri Harish Mallidi)

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


#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-convolutional-component.h"
#include "nnet/nnet-convolutional-2d-component.h"
#include "nnet/nnet-max-pooling-component.h"
#include "nnet/nnet-max-pooling-2d-component.h"
#include "nnet/nnet-average-pooling-2d-component.h"
#include "nnet/nnet-loss.h"
#include "util/common-utils.h"

#include <sstream>
#include <fstream>
#include <algorithm>

namespace kaldi {
namespace nnet1 {

  /*
   * Helper functions
   */  
  template<typename Real>
  void ReadCuMatrixFromString(const std::string& s, CuMatrix<Real>* m) {
    std::istringstream is(s + "\n");
    m->Read(is, false); // false for ascii
  }

  Component* ReadComponentFromString(const std::string& s) {
    std::istringstream is(s + "\n");
    return Component::Read(is, false); // false for ascii
  }
  /*
   */

  void UnitTestConvolutionalComponentUnity() {
    // make 'identity' convolutional component,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 5 5 \
      <PatchDim> 1 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <MaxNorm> 0 \
      <Filters> [ 1 \
      ] <Bias> [ 0 ]"
    );
    
    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 2 3 4 5 ] ", &mat_in);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_in,mat_out);

    // backpropagate,
    CuMatrix<BaseFloat> mat_out_diff(mat_in), mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_out_diff " << mat_out_diff << " mat_in_diff " << mat_in_diff;
    AssertEqual(mat_out_diff,mat_in_diff);
    
    // clean,
    delete c;
  }

  void UnitTestConvolutionalComponent3x3() {
    // make 3x3 convolutional component, design such weights and input so output is zero,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 9 15 \
      <PatchDim> 3 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <MaxNorm> 0 \
      <Filters> [ -1 -2 -7   0 0 0   1 2 7 ; \
                  -1  0  1  -3 0 3  -2 2 0 ; \
                  -4  0  0  -3 0 3   4 0 0 ] \
      <Bias> [ -20 -20 -20 ]"
    );
    
    // prepare input, reference output,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 3 5 7 9  2 4 6 8 10  3 5 7 9 11 ]", &mat_in);
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 0 0 0  0 0 0  0 0 0 ]", &mat_out_ref);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_out, mat_out_ref);

    // prepare mat_out_diff, mat_in_diff_ref,
    CuMatrix<BaseFloat> mat_out_diff;
    ReadCuMatrixFromString("[ 1 0 0  1 1 0  1 1 1 ]", &mat_out_diff);
    CuMatrix<BaseFloat> mat_in_diff_ref; // hand-computed back-propagated values,
    ReadCuMatrixFromString("[ -1 -4 -15 -8 -6   0 -3 -6 3 6   1 1 14 11 7 ]", &mat_in_diff_ref);

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);
    
    // clean,
    delete c;
  }



  void UnitTestMaxPoolingComponent() {
    // make max-pooling component, assuming 4 conv. neurons, non-overlapping pool of size 3,
    Component* c = Component::Init("<MaxPoolingComponent> <InputDim> 24 <OutputDim> 8 \
                     <PoolSize> 3 <PoolStep> 3 <PoolStride> 4");

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 3 8 2 9 \
                              8 3 9 3 \
                              2 4 9 6 \
                              \
                              2 4 2 0 \
                              6 4 9 4 \
                              7 3 0 3;\
                              \
                              5 4 7 8 \
                              3 9 5 6 \
                              3 4 8 9 \
                              \
                              5 4 5 6 \
                              3 1 4 5 \
                              8 2 1 7 ]", &mat_in);

    // expected output (max values in columns),
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 8 8 9 9 \
                              7 4 9 4;\
                              5 9 8 9 \
                              8 4 5 7 ]", &mat_out_ref);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);

    // locations of max values will be shown,
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    mat_out_diff.Set(1);
    // expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref; // hand-computed back-propagated values,
    ReadCuMatrixFromString("[ 0 1 0 1 \
                              1 0 1 0 \
                              0 0 1 0 \
                              \
                              0 1 0 0 \
                              0 1 1 1 \
                              1 0 0 0;\
                              \
                              1 0 0 0 \
                              0 1 0 0 \
                              0 0 1 1 \
                              \
                              0 1 1 0 \
                              0 0 0 0 \
                              1 0 0 1 ]", &mat_in_diff_ref);
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestMaxPooling2DComponent() { /* Implemented by Harish Mallidi */
    // make max-pooling2d component
    Component* c = Component::Init("<MaxPooling2DComponent> <InputDim> 56 <OutputDim> 18 \
<FmapXLen> 4 <FmapYLen> 7 <PoolXLen> 2 <PoolYLen> 3 <PoolXStep> 1 <PoolYStep> 2");

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 21 21 22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);
    
    // expected output (max values in the patch)
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 9 9 11 11 13 13 16 16 18 18 20 20 23 23 25 25 27 27 ]", &mat_out_ref);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);


    // locations of max values will be shown
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]", &mat_out_diff);
    
    //expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref; //hand-computed back-propagated values,
    ReadCuMatrixFromString("[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0.25 0 0 1 1 0 0 0 0 0.75 0.75 0 0 1 1 0 0 2.5 2.5 0 0 0 0 3 3 0 0 3.5 3.5 0 0 8 8 ]", &mat_in_diff_ref);
    
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestAveragePooling2DComponent() { /* Implemented by Harish Mallidi */
    // make average-pooling2d component
    Component* c = Component::Init("<AveragePooling2DComponent> <InputDim> 56 <OutputDim> 18 \
<FmapXLen> 4 <FmapYLen> 7 <PoolXLen> 2 <PoolYLen> 3 <PoolXStep> 1 <PoolYStep> 2");

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 21 21 22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);
    
    // expected output (max values in the patch)
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 4.5 4.5 6.5 6.5 8.5 8.5 11.5 11.5 13.5 13.5 15.5 15.5 18.5 18.5 20.5 20.5 22.5 22.5 ]", &mat_out_ref);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);


    // locations of max values will be shown
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]", &mat_out_diff);
    
    // expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref; // hand-computed back-propagated values,
    ReadCuMatrixFromString("[  0 0 0 0 0.0833333 0.0833333 0.166667 0.166667 0.25 0.25 0.333333 0.333333 0.333333 0.333333 0.25 0.25 0.25 0.25 0.333333 0.333333 0.416667 0.416667 0.5 0.5 0.583333 0.583333 0.583333 0.583333 0.75 0.75 0.75 0.75 0.833333 0.833333 0.916667 0.916667 1 1 1.08333 1.08333 1.08333 1.08333 1 1 1 1 1.08333 1.08333 1.16667 1.16667 1.25 1.25 1.33333 1.33333 1.33333 1.33333 ]", &mat_in_diff_ref);
    
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }


  void UnitTestConvolutional2DComponent() { /* Implemented by Harish Mallidi */
    // Convolutional2D component 
    Component* c = ReadComponentFromString("<Convolutional2DComponent> 18 56 \
<LearnRateCoef> 0 <BiasLearnRateCoef> 0 <FmapXLen> 4 <FmapYLen> 7 <FiltXLen> 2 <FiltYLen> 3 <FiltXStep> 1 <FiltYStep> 2 <ConnectFmap> 1 <Filters> [ 0 0 1 1 2 2 3 3 4 4 5 5 ; 0 0 1 1 2 2 3 3 4 4 5 5 ] <Bias> [ 0 0 ]");
    
    // input matrix
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 21 21 22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 206 206 266 266 326 326 416 416 476 476 536 536 626 626 686 686 746 746 ]", &mat_out_ref);
    
    // propagate
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);

    // prepare mat_out_diff, mat_in_diff_ref,
    CuMatrix<BaseFloat> mat_out_diff;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]", &mat_out_diff);
    CuMatrix<BaseFloat> mat_in_diff_ref;
    ReadCuMatrixFromString("[ 0 0 0 0 0 0 2 2 2 2 4 4 8 8 0 0 3 3 4.5 4.5 8 8 9.5 9.5 13 13 20 20 9 9 18 18 19.5 19.5 23 23 24.5 24.5 28 28 41 41 36 36 48 48 51 51 56 56 59 59 64 64 80 80 ]", &mat_in_diff_ref);

    // backpropagate
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
	
  }

  void UnitTestParallelComponent() {

	// Create parallel component with 2 tasks in parallel: task 1(affine xform + softmax), task 2 (affine xform + softmax)
	std::vector<Component*> c;
	c.push_back(Component::Init(" <Splice> <InputDim> 5 <OutputDim> 10 <BuildVector> 0 0 </BuildVector> "));
	c.push_back(Component::Init(" <ParallelComponent> <InputDim> 10 <OutputDim> 6 <NestedNnetProto> xent_debug.proto xent_debug.proto </NestedNnetProto> "));
	// xent_debug.proto is a text file containing the template of an affine xform and softmax
	// <NnetProto>
	// <AffineTransform> <InputDim> 5 <OutputDim> 3 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.091537 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
	// <Softmax> <InputDim> 3 <OutputDim> 3
	// </NnetProto>

	c[0]->Write(std::cout, false);
	c[1]->Write(std::cout, false);

	// create feature matrix: row 1-2 = Gaussian(-10, 2), row 3-4 = Gaussian(0, 2), row 5-6 = Gaussian(10,2)
	// we'll pass the same feature matrix through both the components in the parallel component
	CuMatrix<BaseFloat> mat_in;
	ReadCuMatrixFromString("[ -9.2396   -7.4065    -13.1945  -8.7807   -9.5492   ;  \
							  -11.8494  -10.6132   -9.5155   -4.9394   -6.0834   ; \
							  -1.9090    4.2920     1.0259   -0.0892    1.0108   ; \
							  -0.2899   -0.1756     2.1068    1.9927    2.0042   ; \
                              10.9496    8.2924    11.0143   12.3055   10.6914   ; \
							  11.4633   11.0280     9.5709   10.4156    8.8866   ]", &mat_in);

	// create targets: assign a label for each row of feature matrix
	// Labels for task 1  = {0, 1, 2}. Labels for task 2  = {3, 4, 5}
	Posterior post(mat_in.NumRows());
	post[0].push_back(std::make_pair(0, 1.0)); // frame 0, target = 0 (task 1)
	post[1].push_back(std::make_pair(3, 1.0)); // frame 1, target = 3 (task 2)
	post[2].push_back(std::make_pair(1, 1.0)); // frame 2, target = 1 (task 1)
	post[3].push_back(std::make_pair(4, 1.0)); // frame 3, target = 4 (task 2)
	post[4].push_back(std::make_pair(2, 1.0)); // frame 4, target = 2 (task 1)
	post[5].push_back(std::make_pair(5, 1.0)); // frame 5, target = 5 (task 2)

	// create frame weights
	Vector<BaseFloat> frm_weights(mat_in.NumRows());
	frm_weights.Set(1);
	KALDI_LOG << "frame weights = " << frm_weights << "\n";

	// convert posterior to matrix
	CuMatrix<BaseFloat> mat_tgt;
	PosteriorToMatrix(post, c[1]->OutputDim(), &mat_tgt);
	KALDI_LOG << "targets = " << mat_tgt << "\n";

	// Feedforward
	std::vector< CuMatrix<BaseFloat> > mat_out(2);
	KALDI_LOG << "mat_in (i/p features) = " << mat_in << "\n";
	c[0]->Propagate(mat_in,&mat_out[0]);
	KALDI_LOG << "mat_out[0] (o/p after splicing) = " << mat_out[0] << "\n";
	c[1]->Propagate(mat_out[0],&mat_out[1]);
	KALDI_LOG << "mat_out[1] (softmax o/p of the 2 tasks) = " << mat_out[1] << "\n";

	// Backprop
	std::vector< CuMatrix<BaseFloat> > mat_diff(3);
	MultiTaskLoss multitask;
	multitask.InitFromString("multitask,xent,3,1.0,xent,3,1.0");
	multitask.Eval(frm_weights, mat_out[1], post, &mat_diff[0]);
	KALDI_LOG << "mat_diff[0] (xent loss = yk - tk) = " << mat_diff[0] << "\n";

	c[1]->Backpropagate(mat_out[0], mat_out[1], mat_diff[0], &mat_diff[1]);
	KALDI_LOG << "mat_diff[1] (bprop o/p of Affine xform) = " << mat_diff[1] << "\n";

	c[0]->Backpropagate(mat_in, mat_out[0], mat_diff[1], &mat_diff[2]);
	KALDI_LOG << "mat_diff[2] (bprop o/p of Splice component) = " << mat_diff[2] << "\n";


	// Hand-computed values
	CuMatrix<BaseFloat> mat_diff_ref;
	ReadCuMatrixFromString("[ -0.024376646869121  -0.115380682258626   0.001912529779371  -0.020029307208487   0.004024987781431; \
                               0.010389806419856   0.015550541562796  -0.012880139785396   0.005226754443175   0.006380661635522; \
                               0.048454612992152   0.132964133474791  -0.071722623065398   0.077128750497918   0.058262038289513; \
                              -0.009068313562302  -0.029714796515553  -0.078803348577301  -0.000694310749474  -0.052208305344111; \
                              -0.006137005324909  -0.006352205696433   0.016475084429451  -0.013829350433595  -0.014589783728879; \
                              -0.054733978728800  -0.056891377748837   0.207474964921216  -0.033531853562929   0.038703912370216; ]", &mat_diff_ref);
	CuMatrix<BaseFloat> err(mat_diff_ref);
	err.AddMat(-1.0, mat_diff[2]);
	KALDI_LOG << "error = " << err << "\n";
	KALDI_LOG << "Frobenius norm of err = " << err.FrobeniusNorm() << "\n";

  }

  void UnitTestParallelComponent_WithMSE(int32 config=1) {

    // "config" can take either 1 or 2
	KALDI_LOG << "Testing Config " << config << "\n";
	// Create parallel component with 3 tasks in parallel. Test with Config 1 or Config 2;
	MultiTaskLoss multitask;
  	std::vector<Component*> c;
  	c.push_back(Component::Init(" <Splice> <InputDim> 5 <OutputDim> 15 <BuildVector> 0 0 0 </BuildVector> "));
  	if (config == 1) {
  	  // Config 1:  task 1(affine xform + softmax), task 2 (affine xform + softmax), task 3 (affine xform),
  	  c.push_back(Component::Init(" <ParallelComponent> <InputDim> 15 <OutputDim> 11 <NestedNnetProto> xent_debug.proto xent_debug.proto mse.debug.proto </NestedNnetProto> "));
  	  multitask.InitFromString("multitask,xent,3,1.0,xent,3,1.0,mse,5,1.0");
  	} else if (config == 2) {
  	  // Config 2:  task 1(affine xform + softmax), task 2 (affine xform), task 3 (affine xform + softmax),
  	  c.push_back(Component::Init(" <ParallelComponent> <InputDim> 15 <OutputDim> 11 <NestedNnetProto> xent_debug.proto mse.debug.proto xent_debug.proto </NestedNnetProto> "));
  	multitask.InitFromString("multitask,xent,3,1.0,mse,5,1.0,xent,3,1.0");
  	}
  	// xent_debug.proto is a text file containing the template of an affine xform and softmax
  	// <NnetProto>
  	// <AffineTransform> <InputDim> 5 <OutputDim> 3 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.091537 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
  	// <Softmax> <InputDim> 3 <OutputDim> 3
  	// </NnetProto>
  	// mse_debug.proto is a text file containing the template of an affine xform
  	// <NnetProto>
  	// <AffineTransform> <InputDim> 5 <OutputDim> 5 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.091537 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
  	// </NnetProto>

  	c[0]->Write(std::cout, false);
  	c[1]->Write(std::cout, false);

  	// create feature matrix: row 1-3 = Gaussian(-10, 2), row 4-6 = Gaussian(0, 2), row 7-9 = Gaussian(10,2)
  	// we'll pass the same feature matrix through all the components in the Parallel component
  	CuMatrix<BaseFloat> mat_in;
  	ReadCuMatrixFromString("[   -9.2396   -7.4065  -13.1945   -8.7807   -9.5492 ; \
  	                            -11.8494  -10.6132   -9.5155   -4.9394   -6.0834 ; \
  	                            -11.9090   -5.7080   -8.9741  -10.0892   -8.9892 ; \
  	                            -0.2899   -0.1756    2.1068    1.9927    2.0042 ; \
  	                            0.9496   -1.7076    1.0143    2.3055    0.6914 ; \
  	                             1.4633    1.0280   -0.4291    0.4156   -1.1134 ; \
  	                           11.2564    8.3778    8.4884    8.8552    5.8362 ; \
  	                           12.0342   10.4599    8.9324   11.9379    7.5796 ; \
  	                           9.8554    9.6585   10.4514   10.4424    8.7769  ]", &mat_in);

  	// create targets: assign a label for each row of feature matrix
  	Posterior post(mat_in.NumRows());
  	if (config == 1) {
  	  // Config 1: Labels for task 1  = {0, 1, 2}. Labels for task 2  = {3, 4, 5}, Labels for task 3 = {target feature vector}
  	  post[0].push_back(std::make_pair(0, 1.0)); // frame 0, target = 0 (task 1)
  	  post[1].push_back(std::make_pair(3, 1.0)); // frame 1, target = 3 (task 2)
  	  for (int32 j=6;j<11;j++) post[2].push_back(std::make_pair(j, -10.0)); // frame 2, target = target feat vec (task 3)
  	  post[3].push_back(std::make_pair(1, 1.0)); // frame 3, target = 1 (task 1)
  	  post[4].push_back(std::make_pair(4, 1.0)); // frame 4, target = 4 (task 2)
  	  for (int32 j=6;j<11;j++) post[5].push_back(std::make_pair(j, 0.0)); // frame 5, target = target feat vec (task 3)
  	  post[6].push_back(std::make_pair(2, 1.0)); // frame 6, target = 2 (task 1)
  	  post[7].push_back(std::make_pair(5, 1.0)); // frame 7, target = 5 (task 2)
  	  for (int32 j=6;j<11;j++) post[8].push_back(std::make_pair(j, 10.0)); // frame 8, target = target feat vec (task 3)
  	} else if (config == 2) {
  	  // Config 2: Labels for task 1  = {0, 1, 2}. Labels for task 2  = {target feature vector}, Labels for task 3 = {8, 9, 10}
  	  post[0].push_back(std::make_pair(0, 1.0)); // frame 0, target = 0 (task 1)
  	  post[1].push_back(std::make_pair(8, 1.0)); // frame 1, target = 8 (task 3)
  	  for (int32 j=3;j<8;j++) post[2].push_back(std::make_pair(j, -10.0)); // frame 2, target = target feat vec (task 2)
  	  post[3].push_back(std::make_pair(1, 1.0)); // frame 3, target = 1 (task 1)
   	  post[4].push_back(std::make_pair(9, 1.0)); // frame 4, target = 9 (task 3)
  	  for (int32 j=3;j<8;j++) post[5].push_back(std::make_pair(j, 0.0)); // frame 5, target = target feat vec (task 2)
  	  post[6].push_back(std::make_pair(2, 1.0)); // frame 6, target = 2 (task 1)
  	  post[7].push_back(std::make_pair(10, 1.0)); // frame 7, target = 10 (task 3)
  	  for (int32 j=3;j<8;j++) post[8].push_back(std::make_pair(j, 10.0)); // frame 8, target = target feat vec (task 2)
  	}

  	// create frame weights
  	Vector<BaseFloat> frm_weights(mat_in.NumRows());
  	frm_weights.Set(1);
  	KALDI_LOG << "frame weights = " << frm_weights << "\n";

  	// convert posterior to matrix,
    CuMatrix<BaseFloat> mat_tgt;
  	PosteriorToMatrix(post, c[1]->OutputDim(), &mat_tgt);
  	KALDI_LOG << "targets = " << mat_tgt << "\n";

  	// Feedforward
  	std::vector< CuMatrix<BaseFloat> > mat_out(2);
  	KALDI_LOG << "mat_in (i/p features) = " << mat_in << "\n";
  	c[0]->Propagate(mat_in,&mat_out[0]);
  	KALDI_LOG << "mat_out[0] (o/p after splicing) = " << mat_out[0] << "\n";
  	c[1]->Propagate(mat_out[0],&mat_out[1]);
  	KALDI_LOG << "mat_out[1] (o/p of the 3 tasks) = " << mat_out[1] << "\n";

  	// Backprop
  	std::vector< CuMatrix<BaseFloat> > mat_diff(3);
  	multitask.Eval(frm_weights, mat_out[1], post, &mat_diff[0]);
  	KALDI_LOG << "mat_diff[0] = " << mat_diff[0] << "\n";

  	c[1]->Backpropagate(mat_out[0], mat_out[1], mat_diff[0], &mat_diff[1]);
  	KALDI_LOG << "mat_diff[1] (bprop o/p of Affine xform) = " << mat_diff[1] << "\n";

  	c[0]->Backpropagate(mat_in, mat_out[0], mat_diff[1], &mat_diff[2]);
  	KALDI_LOG << "mat_diff[2] (bprop o/p of Splice component) = " << mat_diff[2] << "\n";

  	CuMatrix<BaseFloat> mat_diff_ref;
  	if (config == 1) {
  	  // Hand-computed values
  	  ReadCuMatrixFromString("[ -0.0243767 -0.115381 0.00191256 -0.0200294 0.0040247; \
  		                         0.0103898 0.0155505 -0.0128801 0.00522674 0.00638065; \
  		                         0.375858 -1.0163 0.947881 2.48034 1.13847; \
  		                         0.0440399 0.118317 -0.0669723 0.071082 0.0546949; \
  			                    -0.00939555 -0.0326825 -0.0922197 -0.000265222 -0.0595685; \
  			                     0.0117809 0.00178596 -0.00331445 -0.00460904 0.00707515; \
  			                    -0.0113461 -0.0119387 0.0303217 -0.0254922 -0.0268396; \
  			                    -0.0500314 -0.0520923 0.189153 -0.0306296 0.0351218; \
  			                    -0.432694 1.05947 -0.941096 -2.51233 -1.16527; ]" , &mat_diff_ref);
  	} else if (config == 2) {
  	  // Hand-computed values
      ReadCuMatrixFromString("[ -0.0243767 -0.115381 0.00191256 -0.0200294 0.0040247; \
                                 0.0334785 0.00372078 -0.0284195 -0.0966428 -0.109535; \
                                -0.309105 -0.820697 -3.82111 -2.66235 -4.76586; \
                                 0.0440399 0.118317 -0.0669723 0.071082 0.0546949; \
                                -0.0663672 -0.0292561 0.0082099 -0.0503408 0.0463807; \
                                 0.0454691 0.0419295 -0.0980046 -0.00226191 -0.0548548; \
                                -0.0113461 -0.0119387 0.0303217 -0.0254922 -0.0268396; \
                                 0.0525795 0.0329999 0.0151 0.14848 0.0399066; \
                                 0.324932 0.883343 3.8653 2.70871 4.90017 ]",  &mat_diff_ref);
  	}

  	CuMatrix<BaseFloat> err(mat_diff_ref);
  	err.AddMat(-1.0, mat_diff[2]);
  	KALDI_LOG << "error = " << err << "\n";
  	KALDI_LOG << "Frobenius norm of err using Config " << config << " = " <<  err.FrobeniusNorm() << "\n";
  }

  void UnitTestBlockSoftmaxComponent() {

	// Create 2 tasks in parallel: task 1(affine xform + softmax), task 2 (affine xform + softmax)
	std::vector<Component*> c;
	c.push_back(Component::Init(" <AffineTransform> <InputDim> 5 <OutputDim> 6 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.091537 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000 "));
	c.push_back(Component::Init(" <BlockSoftmax> <InputDim> 6 <OutputDim> 6 <BlockDims> 3:3 "));

	c[0]->Write(std::cout, false);
	c[1]->Write(std::cout, false);

	// create feature matrix: frame 1-2: class 0, frame 3-4: class 1, frame 5-6: class 2
	// we'll pass the same feature matrix through both the components in the parallel component
	CuMatrix<BaseFloat> mat_in;
	ReadCuMatrixFromString("[ -9.2396   -7.4065    -13.1945  -8.7807   -9.5492   ;  \
							  -11.8494  -10.6132   -9.5155   -4.9394   -6.0834   ; \
							  -1.9090    4.2920     1.0259   -0.0892    1.0108   ; \
							  -0.2899   -0.1756     2.1068    1.9927    2.0042   ; \
	                          10.9496    8.2924    11.0143   12.3055   10.6914   ; \
						      11.4633   11.0280     9.5709   10.4156    8.8866   ]", &mat_in);

	// create targets: assign a label for each row of feature matrix
	// Labels for task 1  = {0, 1, 2}. Labels for task 2  = {3, 4, 5}
	Posterior post(mat_in.NumRows());
	post[0].push_back(std::make_pair(0, 1.0)); // frame 1, class = 0, node = 0 (task 1)
	post[1].push_back(std::make_pair(3, 1.0)); // frame 2, class = 0, node = 3 (task 2)
	post[2].push_back(std::make_pair(1, 1.0)); // frame 3, class = 1, node = 1 (task 1)
	post[3].push_back(std::make_pair(4, 1.0)); // frame 4, class = 1, node = 4 (task 2)
	post[4].push_back(std::make_pair(2, 1.0)); // frame 5, class = 2, node = 2 (task 1)
	post[5].push_back(std::make_pair(5, 1.0)); // frame 6, class = 2, node = 5 (task 2)



	// create frame weights
	Vector<BaseFloat> frm_weights(mat_in.NumRows());
	frm_weights.Set(1);
	KALDI_LOG << "frame weights = " << frm_weights << "\n";

	// convert posterior to matrix,
	CuMatrix<BaseFloat> mat_tgt;
	PosteriorToMatrix(post, c[1]->OutputDim(), &mat_tgt);
	KALDI_LOG << "targets = " << mat_tgt << "\n";

	// Feedforward
	std::vector< CuMatrix<BaseFloat> > mat_out(2);
	KALDI_LOG << "mat_in (i/p features) = " << mat_in << "\n";
	c[0]->Propagate(mat_in,&mat_out[0]);
	KALDI_LOG << "mat_out[0] (o/p of Affine xform) = " << mat_out[0] << "\n";
	c[1]->Propagate(mat_out[0],&mat_out[1]);
	KALDI_LOG << "mat_out[1] (yk = o/p of softmax) = " << mat_out[1] << "\n";

	// Backprop
	// convert posterior to matrix,
	CuMatrix<BaseFloat> tgt;
	PosteriorToMatrix(post, mat_out[1].NumCols(), &tgt);
	KALDI_LOG << "targets  (tk) = " << tgt << "\n";
	std::vector< CuMatrix<BaseFloat> > mat_diff(3);
	MultiTaskLoss multitask;
	multitask.InitFromString("multitask,xent,3,1.0,xent,3,1.0");
	multitask.Eval(frm_weights, mat_out[1], post, &mat_diff[0]);
	KALDI_LOG << "mat_diff[0] (xent loss = yk - tk) = " << mat_diff[0] << "\n";
	c[1]->Backpropagate(mat_out[0], mat_out[1], mat_diff[0], &mat_diff[1]);
	KALDI_LOG << "mat_diff[1] (masked yk - tk) = " << mat_diff[1] << "\n";
	c[0]->Backpropagate(mat_in, mat_out[0], mat_diff[1], &mat_diff[2]);
	KALDI_LOG << "mat_diff[1] (bprop o/p of Affine xform) = " << mat_diff[2] << "\n";

  }

  void UnitTestTargetInterpolation() {

	  MultiTaskLoss multitask;
	  std::string objective_function = "multitask,xent,5,1.0,xent,5,0.5";
	  std::string tgt_interp_mode = "soft"; // can be "soft"|"hard"|"none"
	  float tgt_interp_wt = 0.8; // number between 0 and 1

	  CuMatrix<BaseFloat> nnet_out, obj_diff;
	  ReadCuMatrixFromString("[   0.2   0.1   0.4   0.1   0.2   0.2   0.1   0.4   0.1   0.2 ;  \
	  							  0.4   0.3   0.0   0.2   0.1   0.4   0.3   0.0   0.2   0.1 ;  \
	  							  0.1   0.1   0.2   0.1   0.5   0.1   0.1   0.2   0.1   0.5 ;  \
	  							  0.2   0.1   0.3   0.4   0.0   0.2   0.1   0.3   0.4   0.0;   \
	  	                          0.3   0.1   0.2   0.2   0.2   0.3   0.1   0.2   0.2   0.2;   \
	  						      0.0   0.0   0.2   0.1   0.7   0.0   0.0   0.2   0.1   0.7]", &nnet_out); // keeping net_out equal for both tasks (only for debugging purposes!)
	  // Labels for task 1  = {0,..,4}. Labels for task 2  = {5,..,9}
	  Posterior nnet_tgt(nnet_out.NumRows());
	  nnet_tgt[0].push_back(std::make_pair(0, 1.0)); // frame 0, target = 0 (task 1)
	  nnet_tgt[1].push_back(std::make_pair(3, 1.0)); // frame 1, target = 3 (task 1)
	  nnet_tgt[2].push_back(std::make_pair(1, 1.0)); // frame 2, target = 1 (task 1)
	  nnet_tgt[3].push_back(std::make_pair(5, 1.0)); // frame 3, target = 5 (task 2)
	  nnet_tgt[4].push_back(std::make_pair(7, 1.0)); // frame 4, target = 7 (task 2)
	  nnet_tgt[5].push_back(std::make_pair(9, 1.0)); // frame 5, target = 9 (task 2)

	  // convert posterior to matrix,
	  CuMatrix<BaseFloat> mat_tgt;
	  PosteriorToMatrix(nnet_tgt, nnet_out.NumCols(), &mat_tgt);

	  // create frame weights
	  Vector<BaseFloat> frm_weights(nnet_out.NumRows());
	  frm_weights.Set(1);

	  KALDI_LOG << " Target Interp Mode = " << tgt_interp_mode << ", Target Interp Wt = " << tgt_interp_wt << "\n";
	  KALDI_LOG << "nnet_out = " << nnet_out << "\n";
	  KALDI_LOG << "targets = " << mat_tgt << "\n";
	  KALDI_LOG << "frame weights = " << frm_weights << "\n";

	  if (0 == objective_function.compare(0,9,"multitask")) {
	     // objective_function contains something like :
	     // 'multitask,xent,2456,1.0,mse,440,0.001'
	     //
	     // the meaning is following:
	     // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
	     multitask.InitFromString(objective_function);
	     multitask.Set_Target_Interp(tgt_interp_mode, tgt_interp_wt);
	     multitask.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
	  }
	  KALDI_LOG << "obj_diff = " << obj_diff << "\n";

	  // Hand-computed values
	  CuMatrix<BaseFloat> obj_diff_ref;
#if 0
	  if ((tgt_interp_mode.compare("soft") == 0) && (tgt_interp_wt == 0.8)) {
	    ReadCuMatrixFromString("[ -0.64  0.08 0.32 0.08 0.16 0 0 0 0 0; \
		    		               0.32  0.24 0   -0.64 0.08 0 0 0 0 0; \
		    		               0.08 -0.72 0.16 0.08 0.4  0 0 0 0 0; \
		    		               0  0 0 0  0 -0.4 0.05 0.15 0.2 0; \
		    			           0 0 0 0 0 0.15 0.05 -0.4 0.1 0.1; \
		    			           0 0 0 0 0 0 0 0.1 0.05 -0.15; ]" , &obj_diff_ref);
	  } else if ((tgt_interp_mode.compare("hard") == 0) && (tgt_interp_wt == 0.2)) {
		  ReadCuMatrixFromString("[ -0.6 0.1 0.2 0.1 0.2 0 0 0 0 0; \
                                     0.2 0.3 0 -0.6 0.1 0 0 0 0 0; \
                                     0.1 -0.7 0.2 0.1 0.3 0 0 0 0 0; \
                                     0 0 0 0 0 -0.4 0.05 0.15 0.2 0; \
                                     0 0 0 0 0 0.15 0.05 -0.4 0.1 0.1 \
                                     0 0 0 0 0 0 0 0.1 0.05 -0.15 ]" , &obj_diff_ref);
	  } else if (tgt_interp_mode.compare("none") == 0) {
		  ReadCuMatrixFromString("[ -0.8 0.1 0.4 0.1 0.2 0 0 0 0 0; \
		  		    		         0.4 0.3 0 -0.8 0.1 0 0 0 0 0; \
		  		    		         0.1 -0.9 0.2 0.1 0.5 0 0 0 0 0; \
		  		    		         0 0 0 0 0 -0.4 0.05 0.15 0.2 0; \
		  		    			     0 0 0 0 0 0.15 0.05 -0.4 0.1 0.1; \
		  		    			     0 0 0 0 0 0 0 0.1 0.05 -0.15; ]" , &obj_diff_ref);
	  }

	  KALDI_LOG << "obj_diff_ref = " << obj_diff_ref << "\n";
	  KALDI_LOG << " rows = " << obj_diff_ref.NumRows() << ", cols = " << obj_diff_ref.NumCols() << "\n";

	  CuMatrix<BaseFloat> err(obj_diff_ref);
	  err.AddMat(-1.0, obj_diff);
	  KALDI_LOG << " Target Interp Mode = " << tgt_interp_mode << ", Target Interp Wt = " << tgt_interp_wt << "\n";
	  KALDI_LOG << "error = " << err << "\n";
	  KALDI_LOG << "Frobenius norm of err " " = " <<  err.FrobeniusNorm() << "\n";
#endif
  }

  void UnitTestTemperedSoftmaxComponent() {

  	// Create 2 tasks in parallel: task 1(affine xform + softmax), task 2 (affine xform + softmax)
  	std::vector<Component*> c;
  	c.push_back(Component::Init(" <AffineTransform> <InputDim> 5 <OutputDim> 3 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.091537 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000 "));
  	c.push_back(Component::Init(" <TemperatureSoftmax> <InputDim> 3 <OutputDim> 3 <Temperature> 2.0 "));
  	// c.push_back(Component::Init(" <Softmax> <InputDim> 3 <OutputDim> 3 <Temperature> 1.0 "));

  	c[0]->Write(std::cout, false);
  	c[1]->Write(std::cout, false);

  	// create feature matrix: frame 1-2: class 0, frame 3-4: class 1, frame 5-6: class 2
  	// we'll pass the same feature matrix through both the components in the parallel component
  	CuMatrix<BaseFloat> mat_in;
  	ReadCuMatrixFromString("[ -9.2396   -7.4065    -13.1945  -8.7807   -9.5492   ;  \
  							  -11.8494  -10.6132   -9.5155   -4.9394   -6.0834   ; \
  							  -1.9090    4.2920     1.0259   -0.0892    1.0108   ; \
  							  -0.2899   -0.1756     2.1068    1.9927    2.0042   ; \
  	                          10.9496    8.2924    11.0143   12.3055   10.6914   ; \
  						      11.4633   11.0280     9.5709   10.4156    8.8866   ]", &mat_in);

  	// create targets: assign a label for each frame
  	Posterior post(mat_in.NumRows());
  	post[0].push_back(std::make_pair(0, 1.0)); // frame 1, class = 0
  	post[1].push_back(std::make_pair(0, 1.0)); // frame 2, class = 0
  	post[2].push_back(std::make_pair(1, 1.0)); // frame 3, class = 1
  	post[3].push_back(std::make_pair(1, 1.0)); // frame 4, class = 1
  	post[4].push_back(std::make_pair(2, 1.0)); // frame 5, class = 2
  	post[5].push_back(std::make_pair(2, 1.0)); // frame 6, class = 2

  	// create frame weights
  	Vector<BaseFloat> frm_weights(mat_in.NumRows());
  	frm_weights.Set(1);
  	KALDI_LOG << "frame weights = " << frm_weights << "\n";

  	// convert posterior to matrix,
  	CuMatrix<BaseFloat> mat_tgt;
  	PosteriorToMatrix(post, c[1]->OutputDim(), &mat_tgt);
  	KALDI_LOG << "targets = " << mat_tgt << "\n";

  	// Feedforward
  	std::vector< CuMatrix<BaseFloat> > mat_out(2);
  	KALDI_LOG << "mat_in (i/p features) = " << mat_in << "\n";
  	c[0]->Propagate(mat_in,&mat_out[0]);
  	KALDI_LOG << "mat_out[0] (o/p of Affine xform) = " << mat_out[0] << "\n";
  	KALDI_LOG << c[1]->Info() << "\n";
  	c[1]->Propagate(mat_out[0],&mat_out[1]);
  	KALDI_LOG << "mat_out[1] (yk = o/p of softmax) = " << mat_out[1] << "\n";
//  	c[1]->SetTemperature(3.0);
//  	KALDI_LOG << c[1]->Info() << "\n";
//  	c[1]->Propagate(mat_out[0],&mat_out[1]);
//  	KALDI_LOG << "mat_out[1] (yk = o/p of softmax) = " << mat_out[1] << "\n";

  	// Backprop
  	// convert posterior to matrix,
  	CuMatrix<BaseFloat> tgt;
  	PosteriorToMatrix(post, mat_out[1].NumCols(), &tgt);
  	KALDI_LOG << "targets  (tk) = " << tgt << "\n";
  	std::vector< CuMatrix<BaseFloat> > mat_diff(3);
  	Xent xent;
  	xent.Eval(frm_weights, mat_out[1], post, &mat_diff[0]);
  	KALDI_LOG << "mat_diff[0] (dE/dak = (1/Temperature)*(yk - tk) = " << mat_diff[0] << "\n"; // todo: add support for temp
  	c[1]->Backpropagate(mat_out[0], mat_out[1], mat_diff[0], &mat_diff[1]);
  	KALDI_LOG << "mat_diff[1] (copy) = " << mat_diff[1] << "\n";
  	c[0]->Backpropagate(mat_in, mat_out[0], mat_diff[1], &mat_diff[2]);
  	KALDI_LOG << "mat_diff[1] (bprop o/p of Affine xform) = " << mat_diff[2] << "\n";
  }

  void UnitTestBlockTemperedSoftmaxComponent() {

	// Create 2 tasks in parallel: task 1(affine xform + softmax), task 2 (affine xform + softmax)
	std::vector<Component*> c;
	c.push_back(Component::Init(" <AffineTransform> <InputDim> 5 <OutputDim> 6 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.091537 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000 "));
	c.push_back(Component::Init(" <BlockSoftmax> <InputDim> 6 <OutputDim> 6 <BlockDims> 3:3 "));

	c[0]->Write(std::cout, false);
	c[1]->Write(std::cout, false);

	// create feature matrix: frame 1-2: class 0, frame 3-4: class 1, frame 5-6: class 2
	// we'll pass the same feature matrix through both the components in the parallel component
	CuMatrix<BaseFloat> mat_in;
	ReadCuMatrixFromString("[ -9.2396   -7.4065    -13.1945  -8.7807   -9.5492   ;  \
							  -11.8494  -10.6132   -9.5155   -4.9394   -6.0834   ; \
							  -1.9090    4.2920     1.0259   -0.0892    1.0108   ; \
							  -0.2899   -0.1756     2.1068    1.9927    2.0042   ; \
	                          10.9496    8.2924    11.0143   12.3055   10.6914   ; \
						      11.4633   11.0280     9.5709   10.4156    8.8866   ]", &mat_in);

	// create targets: assign a label for each frame
	// Labels for task 1  = {0, 1, 2}. Labels for task 2  = {3, 4, 5}
	Posterior post(mat_in.NumRows());
	post[0].push_back(std::make_pair(0, 1.0)); // frame 1, class = 0, node = 0 (task 1)
	post[1].push_back(std::make_pair(3, 1.0)); // frame 2, class = 0, node = 3 (task 2)
	post[2].push_back(std::make_pair(1, 1.0)); // frame 3, class = 1, node = 1 (task 1)
	post[3].push_back(std::make_pair(4, 1.0)); // frame 4, class = 1, node = 4 (task 2)
	post[4].push_back(std::make_pair(2, 1.0)); // frame 5, class = 2, node = 2 (task 1)
	post[5].push_back(std::make_pair(5, 1.0)); // frame 6, class = 2, node = 5 (task 2)


	// create frame weights
	Vector<BaseFloat> frm_weights(mat_in.NumRows());
	frm_weights.Set(1);
	KALDI_LOG << "frame weights = " << frm_weights << "\n";

	// convert posterior to matrix,
	CuMatrix<BaseFloat> mat_tgt;
	PosteriorToMatrix(post, c[1]->OutputDim(), &mat_tgt);
	KALDI_LOG << "targets = " << mat_tgt << "\n";

	// Feedforward
	std::vector< CuMatrix<BaseFloat> > mat_out(2);
	KALDI_LOG << "mat_in (i/p features) = " << mat_in << "\n";
	c[0]->Propagate(mat_in,&mat_out[0]);
	KALDI_LOG << "mat_out[0] (o/p of Affine xform) = " << mat_out[0] << "\n";
	c[1]->Propagate(mat_out[0],&mat_out[1]);
	KALDI_LOG << "mat_out[1] (yk = o/p of softmax) = " << mat_out[1] << "\n";

	// Backprop
	// convert posterior to matrix,
	CuMatrix<BaseFloat> tgt;
	PosteriorToMatrix(post, mat_out[1].NumCols(), &tgt);
	KALDI_LOG << "targets  (tk) = " << tgt << "\n";
	std::vector< CuMatrix<BaseFloat> > mat_diff(3);
	MultiTaskLoss multitask;
	multitask.InitFromString("multitask,xent,3,1.0,xent,3,1.0");
	multitask.Eval(frm_weights, mat_out[1], post, &mat_diff[0]);
	KALDI_LOG << "mat_diff[0] (xent loss = yk - tk) = " << mat_diff[0] << "\n";
	c[1]->Backpropagate(mat_out[0], mat_out[1], mat_diff[0], &mat_diff[1]);
	KALDI_LOG << "mat_diff[1] (masked yk - tk) = " << mat_diff[1] << "\n";
	c[0]->Backpropagate(mat_in, mat_out[0], mat_diff[1], &mat_diff[2]);
	KALDI_LOG << "mat_diff[1] (bprop o/p of Affine xform) = " << mat_diff[2] << "\n";

  }

} // namespace nnet1
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet1;

  for (int32 loop = 0; loop < 1; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // use no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // use GPU when available
#endif
    // unit-tests :
#if 0
    UnitTestConvolutionalComponentUnity();
    UnitTestConvolutionalComponent3x3();
    UnitTestMaxPoolingComponent();
    UnitTestConvolutional2DComponent();
    UnitTestMaxPooling2DComponent();
    UnitTestAveragePooling2DComponent();
#endif
    // UnitTestParallelComponent();
    // UnitTestParallelComponent_WithMSE();
    // UnitTestParallelComponent_WithMSE(2);
    // UnitTestBlockSoftmaxComponent();
    // UnitTestTargetInterpolation();
    // UnitTestBlockTemperedSoftmaxComponent();
    UnitTestTemperedSoftmaxComponent();
    // end of unit-tests,
    if (loop == 0)
        KALDI_LOG << "Tests without GPU use succeeded.";
      else
        KALDI_LOG << "Tests with GPU use (if available) succeeded.";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0; 
}
