// nnet0/nnet-conv.h

// Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)

#ifndef CNSL_NNET0_NNET_CONV_H_
#define CNSL_NNET0_NNET_CONV_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "thread/kaldi-mutex.h"
#include "nnet2/nnet-component.h"

#include <iostream>

using namespace kaldi;
using namespace kaldi::nnet2;

namespace cnsl {
	namespace nnet0 {

		class ConvolutionComponent: public nnet2::UpdatableComponent {

		public:
			explicit ConvolutionComponent(const ConvolutionComponent &other);
			ConvolutionComponent(): is_gradient_(false), weight_decay_(0.0002), momentum_(0.9) { }; // use Init to really initialize.

			ConvolutionComponent(const CuMatrix<BaseFloat> &linear_params, const CuVector<BaseFloat> &bias_params,BaseFloat learning_rate,
				int32 in_height, int32 in_width, int32 in_channels,
				int32 in_pad_height, int32 in_pad_width,
				int32 kernel_height, int32 kernel_width, int32 stride,
				int32 group, int32 out_height, int32 out_width,
				BaseFloat weight_decay, BaseFloat momentum
				);

			virtual ~ConvolutionComponent() { };

			/**
			in = [num_chunks *  (in_height_ * in_width_ * in_channel_)]
			out = [num_chunks *  (out_height_ * out_width_ * group_)]
			linear_params_ = [(kernel_height_ * kernel_width_ * in_channel_; ) *  group_]
			bias_params_= [ group_ ]
			**/

			virtual int32 InputDim() const { return in_height_ * in_width_ * in_channel_; } 
			virtual int32 OutputDim() const { return out_height_ * out_width_ * group_; }
			inline int32 In_height() const { return in_height_ ; } 
			inline int32 In_width() const { return in_width_; } 
			inline int32 In_channels() const { return in_channel_; } 
			inline int32 Out_height() const { return out_height_ ; }
			inline int32 Out_width() const { return out_width_; }
			inline int32 Group() const { return group_; }

			inline int32 KernelDim() const { return kernel_height_ * kernel_width_ * in_channel_; }
			inline int32 Kernel_height() const { return kernel_height_ ; } 
			inline int32 Kernel_width() const { return kernel_width_; } 


			void Init(BaseFloat learning_rate,
				int32 in_height, int32 in_width, int32 in_channels,
				int32 in_pad_height, int32 in_pad_width,
				int32 kernel_height, int32 kernel_width, int32 stride,
				int32 group, int32 out_height, int32 out_width,
				BaseFloat param_stddev, BaseFloat bias_stddev,
				BaseFloat weight_decay, BaseFloat momentum);
			void Init(BaseFloat learning_rate,
				int32 in_height, int32 in_width, int32 in_channels,
				int32 in_pad_height, int32 in_pad_width,
				int32 kernel_height, int32 kernel_width, int32 stride,
				int32 group, int32 out_height, int32 out_width,
				BaseFloat weight_decay, BaseFloat momentum,
				std::string matrix_filename);

			virtual void InitFromString(std::string args);

			virtual std::string Info() const;
			virtual std::string Type() const { return "ConvolutionComponent"; }
			virtual bool BackpropNeedsInput() const { return true; }
			virtual bool BackpropNeedsOutput() const { return false; }
			using Component::Propagate; // to avoid name hiding
			virtual void Propagate(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in,
				CuMatrixBase<BaseFloat> *out) const;
			virtual void Scale(BaseFloat scale);
			virtual void Add(BaseFloat alpha, const UpdatableComponent &other); 
			virtual void Backprop(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in_value,
				const CuMatrixBase<BaseFloat> &out_value, // dummy
				const CuMatrixBase<BaseFloat> &out_deriv,
				Component *to_update, // may be identical to "this".
				CuMatrix<BaseFloat> *in_deriv) const;
			virtual void SetZero(bool treat_as_gradient);
			virtual void Read(std::istream &is, bool binary);
			virtual void Write(std::ostream &os, bool binary) const;
			virtual BaseFloat DotProduct(const UpdatableComponent &other) const; 
			virtual Component* Copy() const;
			virtual void PerturbParams(BaseFloat stddev);
			// This new function is used when mixing up:
			virtual void SetParams(const VectorBase<BaseFloat> &bias,
				const MatrixBase<BaseFloat> &linear);

			const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
			const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }

			virtual int32 GetParameterDim() const;
			virtual void Vectorize(VectorBase<BaseFloat> *params) const;
			virtual void UnVectorize(const VectorBase<BaseFloat> &params);

			void SetWeightDecay(BaseFloat weight_decay){ weight_decay_ = weight_decay; };
			void SetMomentum(BaseFloat momentum){ momentum_ = momentum; };

		protected:

			virtual void Update(
				const CuMatrixBase<BaseFloat> &in_value,
				const CuMatrixBase<BaseFloat> &out_deriv);  

			const ConvolutionComponent &operator = (const ConvolutionComponent &other); // Disallow.

			CuMatrix<BaseFloat> linear_params_; 
			CuVector<BaseFloat> bias_params_; // Each group shares a bias value

			bool is_gradient_; // If true, treat this as just a gradient.

			int32 in_height_;
			int32 in_width_;
			int32 in_channel_;

			int32 in_pad_height_;
			int32 in_pad_width_;

			int32 kernel_height_;
			int32 kernel_width_;
			int32 stride_;
			int32 group_;

			int32 out_height_;
			int32 out_width_;

			BaseFloat weight_decay_;
			BaseFloat momentum_;
			CuMatrix<BaseFloat> prev_grad_; // for momentum
		};

		class MaxpoolComponent: public nnet2::Component {

		public:
			void Init(int32 input_dim, int32 output_dim, int32 in_height, int32 in_width, int32 in_channel, int32 pool_height_dim, int32 pool_width_dim, int32 pool_channel_dim, bool overlap, bool overlap2D);
			explicit MaxpoolComponent(int32 input_dim, int32 output_dim, int32 in_height, int32 in_width, int32 in_channel, int32 pool_height_dim, int32 pool_width_dim, int32 pool_channel_dim, bool overlap, bool overlap2D) {
				Init(input_dim, output_dim, in_height, in_width, in_channel, pool_height_dim, pool_width_dim, pool_channel_dim, overlap, overlap2D);
			}
			MaxpoolComponent(): input_dim_(0), output_dim_(0), in_height_(0), in_width_(0), in_channel_(0), pool_height_dim_(0), pool_width_dim_(0), pool_channel_dim_(0), overlap_(false) , overlap2D_(false){ }
			virtual std::string Type() const { return "MaxpoolComponent"; }
			virtual void InitFromString(std::string args); 
			virtual int32 InputDim() const { return input_dim_; }
			virtual int32 OutputDim() const { return output_dim_; }
			using Component::Propagate; // to avoid name hiding
			virtual void Propagate(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in,
				CuMatrixBase<BaseFloat> *out) const;
			virtual void Backprop(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in_value,
				const CuMatrixBase<BaseFloat> &, // out_value
				const CuMatrixBase<BaseFloat> &out_deriv,
				Component *to_update, // may be identical to "this".
				CuMatrix<BaseFloat> *in_deriv) const;
			virtual bool BackpropNeedsInput() const { return true; }
			virtual bool BackpropNeedsOutput() const { return true; }
			virtual Component* Copy() const { return new MaxpoolComponent(input_dim_, output_dim_, in_height_, in_width_, in_channel_, pool_height_dim_, pool_width_dim_,	pool_channel_dim_, overlap_ , overlap2D_); }

			virtual void Read(std::istream &is, bool binary); // This Read function
			virtual void Write(std::ostream &os, bool binary) const; /// Write component to stream

			virtual std::string Info() const;
		protected:
			int32 input_dim_;
			int32 output_dim_;
			int32 in_height_;
			int32 in_width_;
			int32 in_channel_;
			int32 pool_height_dim_;
			int32 pool_width_dim_;
			int32 pool_channel_dim_;
			bool overlap_;
			bool overlap2D_;
		};

		class FullyConnectedComponent: public nnet2::AffineComponent {
		public:
			virtual std::string Type() const { return "FullyConnectedComponent"; }

			virtual void Read(std::istream &is, bool binary);
			virtual void Write(std::ostream &os, bool binary) const;
			void Init(BaseFloat learning_rate,
				int32 input_dim, int32 output_dim,
				BaseFloat param_stddev, BaseFloat bias_stddev,
				BaseFloat weight_decay, BaseFloat momentum);
			void Init(BaseFloat learning_rate, BaseFloat weight_decay,
				BaseFloat momentum, std::string matrix_filename);

			virtual void InitFromString(std::string args);
			virtual std::string Info() const;
			virtual Component* Copy() const;
			FullyConnectedComponent(): weight_decay_(0.0002), momentum_(0.9) { }

			void SetWeightDecay(BaseFloat weight_decay){ weight_decay_ = weight_decay; };
			void SetMomentum(BaseFloat momentum){ momentum_ = momentum; };

			virtual void Update(
				const CuMatrixBase<BaseFloat> &in_value,
				const CuMatrixBase<BaseFloat> &out_deriv){
					UpdateSimple(in_value, out_deriv);
			};

			virtual void UpdateSimple(
				const CuMatrixBase<BaseFloat> &in_value,
				const CuMatrixBase<BaseFloat> &out_deriv);

		protected:
			KALDI_DISALLOW_COPY_AND_ASSIGN(FullyConnectedComponent);

			BaseFloat weight_decay_;
			BaseFloat momentum_;
			CuMatrix<BaseFloat> prev_grad_; // for momentum


		};
		
		class ProbReLUComponent: public nnet2::NonlinearComponent {
		public:

			//explicit ProbReLUComponent(int32 dim, bool expectation): NonlinearComponent(dim) { expectation_= expectation; }
			//explicit ProbReLUComponent(int32 dim): NonlinearComponent(dim) {}
			explicit ProbReLUComponent(const ProbReLUComponent &other);

			void Init(int32 dim, bool expectation);
			ProbReLUComponent(int32 dim, bool expectation = false) {
				Init(dim, expectation);
			}
			ProbReLUComponent(): dim_(0), expectation_(false) { }
			virtual int32 InputDim() const { return dim_; }
			virtual int32 OutputDim() const { return dim_; }
			virtual void InitFromString(std::string args);

			virtual void Read(std::istream &is, bool binary);
			virtual void Write(std::ostream &os, bool binary) const;		

			virtual std::string Type() const { return "ProbReLUComponent"; }
			virtual Component* Copy() const;		
			virtual bool BackpropNeedsInput() const { return false; }
			virtual bool BackpropNeedsOutput() const { return true; }

			using Component::Propagate; // to avoid name hiding
			virtual void Propagate(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in,
				CuMatrixBase<BaseFloat> *out) const; 
			virtual void Backprop(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in_value,
				const CuMatrixBase<BaseFloat> &out_value,                        
				const CuMatrixBase<BaseFloat> &out_deriv,
				Component *to_update, // may be identical to "this".
				CuMatrix<BaseFloat> *in_deriv) const;

			void ResetGenerator() { random_generator_.SeedGpu(0); }
			void SetExpectation(bool boolExpectation) { expectation_ = boolExpectation; }
			virtual std::string Info() const;

		private:
			int32 dim_;  
			ProbReLUComponent &operator = (const ProbReLUComponent &other); // Disallow.
			CuRand<BaseFloat> random_generator_;
			bool expectation_;
		};
		
		class ConvolutionComponentContainer: public nnet2::UpdatableComponent{

		public:
			/*
			explicit ConvolutionComponentContainer(const ConvolutionComponent &other);
			ConvolutionComponentContainer(): is_gradient_(false), weight_decay_(0.0002), momentum_(0.9) { }; // use Init to really initialize.

			ConvolutionComponentContainer(const CuMatrix<BaseFloat> &linear_params, const CuVector<BaseFloat> &bias_params,BaseFloat learning_rate,
				int32 in_height, int32 in_width, int32 in_channels,
				int32 in_pad_height, int32 in_pad_width,
				int32 kernel_height, int32 kernel_width, int32 stride,
				int32 group, int32 out_height, int32 out_width,
				BaseFloat weight_decay, BaseFloat momentum
				);
				*/

			virtual std::string Type() const { return "ConvolutionComponentContainer"; };
			virtual void InitFromString(std::string args);
			/*void Init(BaseFloat learning_rate,
				int32 in_height, int32 in_width, int32 in_channels,
				int32 in_pad_height, int32 in_pad_width,
				int32 kernel_height, int32 kernel_width, int32 stride,
				int32 group, int32 out_height, int32 out_width,
				BaseFloat param_stddev, BaseFloat bias_stddev,
				BaseFloat weight_decay, BaseFloat momentum);
			void Init(BaseFloat learning_rate,
				int32 in_height, int32 in_width, int32 in_channels,
				int32 in_pad_height, int32 in_pad_width,
				int32 kernel_height, int32 kernel_width, int32 stride,
				int32 group, int32 out_height, int32 out_width,
				BaseFloat weight_decay, BaseFloat momentum,
				std::string matrix_filename);
*/

			virtual int32 InputDim() const { return in_height_ * in_width_ * in_channel_; };
			virtual int32 OutputDim() const { return group_ * out_height_ * out_width_; };

			virtual void Propagate(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in,
				CuMatrixBase<BaseFloat> *out) const;

			/// A non-virtual propagate function that first resizes output if necessary.
			void Propagate(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in,
				CuMatrix<BaseFloat> *out) const {
					if (out->NumRows() != out_info.NumRows() ||
						out->NumCols() != out_info.NumCols()) {
							out->Resize(out_info.NumRows(), out_info.NumCols());
					}

					// Cast to CuMatrixBase to use the virtual version of propagate function.
					Propagate(in_info, out_info, in,
						static_cast<CuMatrixBase<BaseFloat>*>(out));
			}

			virtual void Backprop(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in_value,
				const CuMatrixBase<BaseFloat> &out_value,                        
				const CuMatrixBase<BaseFloat> &out_deriv,
				Component *to_update, // may be identical to "this".
				CuMatrix<BaseFloat> *in_deriv) const;

			/// Copy component (deep copy).
			virtual Component* Copy() const;

			virtual void Read(std::istream &is, bool binary); // This Read function
			// requires that the Component has the correct type.

			/// Write component to stream
			virtual void Write(std::ostream &os, bool binary) const;

			virtual std::string Info() const;

			virtual ~ConvolutionComponentContainer() { };

			virtual void SetZero(bool treat_as_gradient);
			virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
			virtual void PerturbParams(BaseFloat stddev);
			virtual void Scale(BaseFloat scale);
			virtual void Add(BaseFloat alpha, const UpdatableComponent &other);


		private:
			std::vector<ConvolutionComponent*> ConvComps_;
			std::vector< std::vector<BaseFloat> > kernels_vec_;
			BaseFloat learning_rate_, weight_decay_, momentum_;

			int32 num_component_;
			//int32 indim_, outdim_;
			int32 in_height_, in_width_, in_channel_;
			int32 group_, out_height_, out_width_;
		};



	} // namespace nnet0
} // namespace cnsl



#endif


