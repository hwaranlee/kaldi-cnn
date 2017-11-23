# kaldi-cnn

## Introduction
This Git repository is the CNN source code following the nnet2 (Dan's DNN implementation) in KALDI Speech Recognition Toolkit, which is the implementation of the paper:
Lee, Hwaran, et al. "Deep CNNs Along the Time Axis With Intermap Pooling for Robustness to Spectral Variations." IEEE Signal Processing Letters 23.10 (2016): 1310-1314. [[paper](https://arxiv.org/abs/1606.03207)] [[demo](https://www.youtube.com/watch?time_continue=3&v=wKbLlqVd524)]
 
We provide:
1. 2D Convolution layer (ConvoutionComponent)
2. 3D Maxpooling layer (MaxpoolComponent)
3. Fully connected layer (FullyConnectedComponent), which is plain version and different from the [['AffineComponentPreconditioned'](http://kaldi-asr.org/doc/classkaldi_1_1nnet2_1_1AffineComponentPreconditioned.html)] in nnet2.


## Install

0. Download and install the Kaldi Speech Recognition Toolkit from [[kaldi-git-trunk](https://github.com/kaldi-asr/kaldi)].

1. In the file "src/cudamatrix/cu-matrix.h", copy and paste the followings as member functions of class CuMatrixBase

		// Convolution 'this' with kernel => out
		// this matrix : row = num_chunks, col=in_height * in_width * in_channel

		void Conv2D(const CuMatrixBase<Real> &kernel,
			int32 in_height,
			int32 in_width,
			int32 in_channel,
			int32 kernel_height,
			int32 kernel_width,
			int32 group,
			CuMatrixBase<Real> *out,
			bool concat) const;

		// if vec = [1 2 3] and rep = 2 => vec2 = [ 1 1 2 2 3 3];
		// this = this * repmat(vec2, NumRows(), 1);
		void AddMatRepVec(const CuVectorBase<Real> &vec, int32 rep) const;

		// Flip 2D matrix. this [(kernel_height*kernel_width*in_channel) x group]
		// flip [(kernel_height*kernel_width*group) x in_channel]
		void FlipMat(int32 kernel_height, int32 kernel_width, int32 in_channel, int32 group, CuMatrix<Real> *flip) const;

		// zero padding along the edge
		// zero [ (NumRows() + pad_height*2) x (NumCols() + pad_width*2) ]
		void PaddingZero(int32 orig_height, int32 orig_width, int32 orig_channel, int32 kernel_height, int32 kernel_width, CuMatrix<Real> *padmat) const;

		void TpBlock(int32 in_channel, int32 block_size, CuMatrix<Real> *out) const;

		void TpInsideBlock(int32 group, int32 block_size, CuMatrix<Real> *out) const;

		void ModPermuteRow(int32 in_channel, int32 block_size, CuMatrix<Real> *out) const;
	
		void Maxpool_prop(int32 in_height, int32 in_width, int32 pool_height_dim, int32 pool_width_dim, int32 pool_channel_dim, CuMatrixBase<Real> *out) const;
		void Maxpool_backprop(const CuMatrixBase<Real> &out_value, const CuMatrixBase<Real> &out_deriv, CuMatrix<Real> *in_deriv,
												int32 in_height, int32 in_width, int32 pool_height_dim, int32 pool_width_dim, int32 pool_channel_dim) const;


2. Add new components in nnet0 into nnet2's header and source codes.

+ In the file "src/nnet2/nnet-component.cc"
	+ add: #include "nnet0/nnet-component-nnet0.h"
	+ add followings under "Component\* Component::NewComponentOfType(const std::string &component_type) "
<pre><code>
  } else if (component_type == "ConvolutionComponent") {
    ans = new cnsl::nnet0::ConvolutionComponent();
  } else if (component_type == "MaxpoolComponent") {
    ans = new cnsl::nnet0::MaxpoolComponent();
  } else if (component_type == "FullyConnectedComponent") { 
    ans = new cnsl::nnet0::FullyConnectedComponent();
  }
</code></pre>

+ In the file "src/nnet0/nnet-component-nnet0.h"
	+ Change the ChunkInfo's private variables to be "public"
	+ In the class NonlinearComponent, change 'UpdateStates' function to be "public"

3. Copy "src/cnslmat" folder into the kaldi trunk and make

4. Copy "src/nnet0" folder into the kaldi trunk and make 

5. In the file "src/nnet2bin/Makefile" add followings:
ADDLIBS ../nnet0/cnsl-nnet0.a ../cnslmat/cnsl-cnslmat.a

6. Make all source files
	cd ../src
	make

## Guide to run the library
1. To train CNN, run "local/nnet0/run_nnet.sh". 
Before you run the code, you need a network configuration file "nnet.conf" in your experiment directory. Also when the network includes dropout layers, "dropout_scale.config" file is required.


## Note
- Implemented by Hwaran Lee (Computational NeroSystems Labs, KAIST)
- under KALDI Revision 4510
- updated date : 2015. 05. 15.


