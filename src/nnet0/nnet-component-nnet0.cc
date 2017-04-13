// nnet0/nnet-conv.cc

// Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)

#include <sstream>
//#include "util/text-utils.h"
//#include "util/kaldi-io.h"
#include "base/timer.h"
#include "util/common-utils.h"

#include "nnet0/nnet-component-nnet0.h"
#include "cnslmat/conv2D.cc"

using namespace kaldi;
using namespace kaldi::nnet2;

namespace cnsl {
	namespace nnet0 {

		// This is like ExpectToken but for two tokens, and it
		// will either accept token1 and then token2, or just token2.
		// This is useful in Read functions where the first token
		// may already have been consumed.
		static void ExpectOneOrTwoTokens(std::istream &is, bool binary,
			const std::string &token1,
			const std::string &token2)
		{
			KALDI_ASSERT(token1 != token2);
			std::string temp;
			ReadToken(is, binary, &temp);
			if (temp == token1) {
				ExpectToken(is, binary, token2);
			} else {
				if (temp != token2) {
					KALDI_ERR << "Expecting token " << token1 << " or " << token2
						<< " but got " << temp;
				}
			}
		}

		// static
		bool ParseFromString(const std::string &name, std::string *string,
			int32 *param) {
				std::vector<std::string> split_string;
				SplitStringToVector(*string, " \t", true,
					&split_string);
				std::string name_equals = name + "="; // the name and then the equals sign.
				size_t len = name_equals.length();

				for (size_t i = 0; i < split_string.size(); i++) {
					if (split_string[i].compare(0, len, name_equals) == 0) {
						if (!ConvertStringToInteger(split_string[i].substr(len), param))
							KALDI_ERR << "Bad option " << split_string[i];
						*string = "";
						// Set "string" to all the pieces but the one we used.
						for (size_t j = 0; j < split_string.size(); j++) {
							if (j != i) {
								if (!string->empty()) *string += " ";
								*string += split_string[j];
							}
						}
						return true;
					}
				}
				return false;
		}

		bool ParseFromString(const std::string &name, std::string *string,
			bool *param) {
				std::vector<std::string> split_string;
				SplitStringToVector(*string, " \t", true,
					&split_string);
				std::string name_equals = name + "="; // the name and then the equals sign.
				size_t len = name_equals.length();

				for (size_t i = 0; i < split_string.size(); i++) {
					if (split_string[i].compare(0, len, name_equals) == 0) {
						std::string b = split_string[i].substr(len);
						if (b.empty())
							KALDI_ERR << "Bad option " << split_string[i];
						if (b[0] == 'f' || b[0] == 'F') *param = false;
						else if (b[0] == 't' || b[0] == 'T') *param = true;
						else
							KALDI_ERR << "Bad option " << split_string[i];
						*string = "";
						// Set "string" to all the pieces but the one we used.
						for (size_t j = 0; j < split_string.size(); j++) {
							if (j != i) {
								if (!string->empty()) *string += " ";
								*string += split_string[j];
							}
						}
						return true;
					}
				}
				return false;
		}

		bool ParseFromString(const std::string &name, std::string *string,
			BaseFloat *param) {
				std::vector<std::string> split_string;
				SplitStringToVector(*string, " \t", true,
					&split_string);
				std::string name_equals = name + "="; // the name and then the equals sign.
				size_t len = name_equals.length();

				for (size_t i = 0; i < split_string.size(); i++) {
					if (split_string[i].compare(0, len, name_equals) == 0) {
						if (!ConvertStringToReal(split_string[i].substr(len), param))
							KALDI_ERR << "Bad option " << split_string[i];
						*string = "";
						// Set "string" to all the pieces but the one we used.
						for (size_t j = 0; j < split_string.size(); j++) {
							if (j != i) {
								if (!string->empty()) *string += " ";
								*string += split_string[j];
							}
						}
						return true;      
					}
				}
				return false;
		}

		bool ParseFromString(const std::string &name, std::string *string,
			std::string *param) {
				std::vector<std::string> split_string;
				SplitStringToVector(*string, " \t", true,
					&split_string);
				std::string name_equals = name + "="; // the name and then the equals sign.
				size_t len = name_equals.length();

				for (size_t i = 0; i < split_string.size(); i++) {
					if (split_string[i].compare(0, len, name_equals) == 0) {
						*param = split_string[i].substr(len);

						// Set "string" to all the pieces but the one we used.
						*string = "";
						for (size_t j = 0; j < split_string.size(); j++) {
							if (j != i) {
								if (!string->empty()) *string += " ";
								*string += split_string[j];
							}
						}
						return true;      
					}
				}
				return false;
		}

		bool ParseFromString(const std::string &name, std::string *string,
			std::vector<int32> *param) {
				std::vector<std::string> split_string;
				SplitStringToVector(*string, " \t", true,
					&split_string);
				std::string name_equals = name + "="; // the name and then the equals sign.
				size_t len = name_equals.length();

				for (size_t i = 0; i < split_string.size(); i++) {
					if (split_string[i].compare(0, len, name_equals) == 0) {
						if (!SplitStringToIntegers(split_string[i].substr(len), ":",
							false, param))
							KALDI_ERR << "Bad option " << split_string[i];
						*string = "";
						// Set "string" to all the pieces but the one we used.
						for (size_t j = 0; j < split_string.size(); j++) {
							if (j != i) {
								if (!string->empty()) *string += " ";
								*string += split_string[j];
							}
						}
						return true;
					}
				}
				return false;
		}

		ConvolutionComponent::ConvolutionComponent(const ConvolutionComponent &component):
			UpdatableComponent(component),
			linear_params_(component.linear_params_),
			bias_params_(component.bias_params_),
			is_gradient_(component.is_gradient_),
			in_height_(component.in_height_),
			in_width_(component.in_width_),
			in_channel_(component.in_channel_),
			in_pad_height_(component.in_pad_height_),
			in_pad_width_(component.in_pad_width_),
			kernel_height_(component.kernel_height_),
			kernel_width_(component.kernel_width_),
			stride_(component.stride_),
			group_(component.group_),
			out_height_(component.out_height_),
			out_width_(component.out_width_),
			weight_decay_(component.weight_decay_),
			momentum_(component.momentum_){ }



		ConvolutionComponent::ConvolutionComponent(const CuMatrix<BaseFloat> &linear_params,
			const CuVector<BaseFloat> &bias_params,
			BaseFloat learning_rate,
			int32 in_height, int32 in_width, int32 in_channels,
			int32 in_pad_height, int32 in_pad_width,
			int32 kernel_height, int32 kernel_width, int32 stride,
			int32 group, int32 out_height, int32 out_width,
			BaseFloat weight_decay, BaseFloat momentum
			):
		UpdatableComponent(learning_rate),
			linear_params_(linear_params),
			bias_params_(bias_params),
			in_height_(in_height),
			in_width_(in_width),
			in_channel_(in_channels),
			in_pad_height_(in_pad_height),
			in_pad_width_(in_pad_width),
			kernel_height_(kernel_height),
			kernel_width_(kernel_width),
			stride_(stride),
			group_(group),
			out_height_(out_height),
			out_width_(out_width),
			weight_decay_(weight_decay),
			momentum_(momentum)
		{
			KALDI_ASSERT(linear_params.NumCols() == bias_params.Dim()&&
				bias_params.Dim() != 0);
			is_gradient_ = false;

		}


		void ConvolutionComponent::Init(BaseFloat learning_rate,
			int32 in_height, int32 in_width, int32 in_channels,
			int32 in_pad_height, int32 in_pad_width,
			int32 kernel_height, int32 kernel_width, int32 stride,
			int32 group, int32 out_height, int32 out_width,
			BaseFloat param_stddev, BaseFloat bias_stddev,
			BaseFloat weight_decay, BaseFloat momentum) {

				this->in_height_=in_height;
				this->in_width_=in_width;
				this->in_channel_=in_channels;

				this->in_pad_height_=in_pad_height;
				this->in_pad_width_=in_pad_width;

				this->kernel_height_=kernel_height;
				this->kernel_width_=kernel_width;
				this->stride_=stride;
				this->group_=group;

				this->out_height_=out_height;
				this->out_width_=out_width;

				this->weight_decay_ = weight_decay;
				this->momentum_ = momentum;

				KALDI_ASSERT( in_pad_height_ >= 0 );
				KALDI_ASSERT( in_pad_width_ >= 0 );
				KALDI_ASSERT(out_height_ == 1 + (in_height + (2*in_pad_height) - kernel_height) / stride );
				KALDI_ASSERT(out_width_ == 1 + (in_width + (2*in_pad_width) - kernel_width) / stride );

				UpdatableComponent::Init(learning_rate);
				linear_params_.Resize(KernelDim(), group);
				bias_params_.Resize(group);
				prev_grad_.Resize(KernelDim(), group);

				KALDI_ASSERT( param_stddev >= 0.0 );
				linear_params_.SetRandn(); // sets to random normally distributed noise.
				linear_params_.Scale(param_stddev);
				prev_grad_.SetZero();
				bias_params_.SetRandn();
				bias_params_.Scale(bias_stddev);

		}

		void ConvolutionComponent::Init(BaseFloat learning_rate,
			int32 in_height, int32 in_width, int32 in_channels,
			int32 in_pad_height, int32 in_pad_width,
			int32 kernel_height, int32 kernel_width, int32 stride,
			int32 group, int32 out_height, int32 out_width,
			BaseFloat weight_decay, BaseFloat momentum,

			std::string matrix_filename) {

				this->in_height_=in_height;
				this->in_width_=in_width;
				this->in_channel_=in_channels;

				this->in_pad_height_=in_pad_height;
				this->in_pad_width_=in_pad_width;

				this->kernel_height_=kernel_height;
				this->kernel_width_=kernel_width;
				this->stride_=stride;
				this->group_=group;

				this->out_height_=out_height;
				this->out_width_=out_width;

				this->weight_decay_ = weight_decay;
				this->momentum_ = momentum;

				UpdatableComponent::Init(learning_rate);  
				CuMatrix<BaseFloat> mat;
				ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
				KALDI_ASSERT(mat.NumCols() >= 2);
				int32 num_group = mat.NumCols() , kernel_dim = mat.NumRows()-1;

				KALDI_ASSERT(num_group == Group() );
				KALDI_ASSERT(kernel_dim == KernelDim() );

				linear_params_.Resize(kernel_dim, num_group);
				bias_params_.Resize(kernel_dim);
				linear_params_.CopyFromMat(mat.Range(0, kernel_dim, 0, num_group));
				bias_params_.CopyColFromMat(mat, num_group);

				prev_grad_.Resize(KernelDim(), group);
				prev_grad_.SetZero();

		}

		void ConvolutionComponent::InitFromString(std::string args) {

			std::string orig_args(args);
			bool ok = true;
			BaseFloat learning_rate = learning_rate_;
			BaseFloat weight_decay = weight_decay_, momentum = momentum_;

			std::string matrix_filename;
			//int32 input_dim = -1, output_dim = -1;

			int32 in_height, in_width, in_channel, in_pad_height, in_pad_width, kernel_height, kernel_width, stride, group, out_height, out_width;
			in_pad_height=0;
			in_pad_width=0;

			ok=ok && ParseFromString("learning-rate", &args, &learning_rate); //optional
			ok=ok && ParseFromString("in-height", &args, &in_height); 
			ok=ok && ParseFromString("in-width", &args, &in_width); 
			ok=ok && ParseFromString("in-channel", &args, &in_channel);
			ParseFromString("in-pad-height", &args, &in_pad_height);
			ParseFromString("in-pad-width", &args, &in_pad_width);
			ok=ok && ParseFromString("kernel-height", &args, &kernel_height);
			ok=ok && ParseFromString("kernel-width", &args, &kernel_width);
			ok=ok && ParseFromString("stride", &args, &stride);
			ok=ok && ParseFromString("group", &args, &group);
			ok=ok && ParseFromString("out-height", &args, &out_height);
			ok=ok && ParseFromString("out-width", &args, &out_width);
			KALDI_ASSERT(out_height == 1 + (in_height + (2*in_pad_height) - kernel_height) / stride  &&
				"out_height_ == 1 + (in_height + (2*in_pad_height) - kernel_height) / stride ");
			KALDI_ASSERT(out_width == 1 + (in_width + (2*in_pad_width) - kernel_width) / stride  &&
				"out_width == 1 + (in_width + (2*in_pad_width) - kernel_width) / stride");
			KALDI_ASSERT( in_pad_height >= 0 && "in-pad-height should be positive");
			KALDI_ASSERT( in_pad_width >= 0 && "in-pad-width should be positive");
			
			if (ParseFromString("matrix", &args, &matrix_filename)) {    
				Init(learning_rate,
					in_height, in_width, in_channel,
					in_pad_height, in_pad_width, 
					kernel_height, kernel_width, stride,
					group, out_height, out_width,
					weight_decay, momentum,
					matrix_filename);
			} else {			
				BaseFloat param_stddev = 1.0 / std::sqrt(kernel_height * kernel_width),	bias_stddev = 1.0;
				ParseFromString("param-stddev", &args, &param_stddev);
				ParseFromString("bias-stddev", &args, &bias_stddev);
				Init(learning_rate,
					in_height, in_width, in_channel,
					in_pad_height, in_pad_width, 
					kernel_height,  kernel_width, stride,
					group,  out_height, out_width,
					param_stddev, bias_stddev,
					weight_decay, momentum);    
			}
			ParseFromString("weight-decay", &args, &weight_decay);
			ParseFromString("momentum", &args, &momentum);

			if (!args.empty())
				KALDI_ERR << "Could not process these elements in initializer: "
				<< args;
			if (!ok)
				KALDI_ERR << "Bad initializer " << orig_args;

		}

		std::string ConvolutionComponent::Info() const {
			std::stringstream stream;
			BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
				* static_cast<BaseFloat>(linear_params_.NumCols());
			BaseFloat linear_stddev =
				std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
				linear_params_size),
				bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
				bias_params_.Dim());
			stream << Type() << ", input-dim=" << InputDim()
				<< " ( in-height=" << In_height()
				<< ", in-width=" << In_width()
				<< ", in-channels=" << In_channels()

				<< "), output-dim=" << OutputDim()
				<< " ( out-height=" << Out_height()
				<< ", out-width=" << Out_width()
				<< ", group-num=" << Group()

				<< "), kernel-dim=" << KernelDim()
				<< " ( kernel-height=" << Kernel_height()
				<< ", kernel-width=" << Kernel_width()

				<< "), ( padding-height=" << in_pad_height_
				<< ", padding-width=" << in_pad_width_

				<< "), linear-params-stddev=" << linear_stddev
				<< ", bias-params-stddev=" << bias_stddev
				<< ", learning-rate=" << LearningRate()

				<< ", weight-decay=" << weight_decay_
				<< ", momentum=" << momentum_;

			return stream.str();
		}

		void ConvolutionComponent::Propagate(const ChunkInfo &in_info,
			const ChunkInfo &out_info,
			const CuMatrixBase<BaseFloat> &in,
			CuMatrixBase<BaseFloat> *out) const {
				//KALDI_LOG <<"conv::prop " << out;
				//KALDI_LOG << out->NumRows() << " " << out->NumCols();
				
				if(in_pad_height_>0 || in_pad_width_ >0){
					CuMatrix<BaseFloat> padded_input( in_info.NumChunks(), (in_height_+2*(in_pad_height_))*(in_width_+2*(in_pad_width_))*in_channel_);
					//KALDI_LOG << padded_input.NumCols();
					in.PaddingZero(in_height_, in_width_, in_channel_, in_pad_height_+1, in_pad_width_+1, &padded_input);
					//KALDI_LOG << padded_input.NumCols();
					padded_input.Conv2D(linear_params_, (in_height_+2*(in_pad_height_)), (in_width_+2*(in_pad_width_)), in_channel_, kernel_height_, kernel_width_, group_, out, true);
				}else{
					//KALDI_LOG << in.NumCols() << " " << in_height_<< " " <<in_width_ << " " << in_channel_ ;
					in.Conv2D(linear_params_, in_height_, in_width_, in_channel_, kernel_height_, kernel_width_, group_, out, true);
				}
				
				//KALDI_LOG << bias_params_.Dim() << " " << out_height_ * out_width_ << " , " << out->NumCols();
				
				out->AddMatRepVec(bias_params_, out_height_ * out_width_); 				// out = out + bias
				//KALDI_LOG << "conv prop: " << *out;

		}

		void ConvolutionComponent::Scale(BaseFloat scale) {
			linear_params_.Scale(scale);
			bias_params_.Scale(scale);
		}

		void ConvolutionComponent::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
			const ConvolutionComponent *other =
				dynamic_cast<const ConvolutionComponent*>(&other_in);
			KALDI_ASSERT(other != NULL);
			linear_params_.AddMat(alpha, other->linear_params_);
			bias_params_.AddVec(alpha, other->bias_params_);
		}

		void ConvolutionComponent::Backprop(const ChunkInfo &in_info,
			const ChunkInfo &out_info,
			const CuMatrixBase<BaseFloat> &in_value,
			const CuMatrixBase<BaseFloat> &,  // out_value
			const CuMatrixBase<BaseFloat> &out_deriv,
			Component *to_update_in,
			CuMatrix<BaseFloat> *in_deriv) const {

				int32 num_chunks = out_deriv.NumRows();
				ConvolutionComponent *to_update = dynamic_cast<ConvolutionComponent*>(to_update_in);

				//int32 in_height_temp = in_height_+2*(in_pad_height_), 
				//	in_width_temp = in_width_+2*(in_pad_width_);

				// Propagate the derivative back to the input.
				// full convolution

				bool flip_kernel=true;

				/*
				int32 pad_kernel_height = (kernel_height_ )  + 2 * (out_height_ - 2 *(in_pad_height_) -1),
					pad_kernel_width = (kernel_width_ )+ 2 * (out_width_  - 2 *(in_pad_width_)-1),
					pad_kernel_size = pad_kernel_height*pad_kernel_width,
					pad_out_deriv_height = (out_height_ - 2 *(in_pad_height_)) + 2*(kernel_height_-1),
					pad_out_deriv_width = (out_width_ - 2 *(in_pad_width_)) + 2*(kernel_width_-1),
					pad_out_size = pad_out_deriv_height*pad_out_deriv_width;
					*/

				int32 pad_kernel_height = (kernel_height_  )  + 2 * (out_height_ -in_pad_height_ -1),
					pad_kernel_width = (kernel_width_ )+ 2 * (out_width_ -in_pad_width_ -1),
					pad_kernel_size = pad_kernel_height*pad_kernel_width,
					pad_out_deriv_height = (out_height_ ) + 2*(kernel_height_-in_pad_height_-1),
					pad_out_deriv_width = (out_width_ ) + 2*(kernel_width_-in_pad_width_-1),
					pad_out_size = pad_out_deriv_height*pad_out_deriv_width;


				if( pad_kernel_size < pad_out_size ) flip_kernel = false;

				if( ! flip_kernel ){

					CuMatrix<BaseFloat> flip_out_deriv(out_height_*out_width_*group_,num_chunks); 
					{
						CuMatrix<BaseFloat> out_deriv_tp(out_height_*out_width_*out_deriv.NumRows() , group_ );
						// 1) transpose inside block
						out_deriv.TpInsideBlock(group_, out_height_*out_width_, &out_deriv_tp);
						// 2) flip
						out_deriv_tp.FlipMat(out_height_, out_width_, num_chunks, group_, &flip_out_deriv);
					}

					CuMatrix<BaseFloat> pad_kernel( in_channel_, pad_kernel_height *pad_kernel_width *group_);

					{
						CuMatrix<BaseFloat> linear_params_tp( group_, kernel_height_ *kernel_width_ *in_channel_);
						CuMatrix<BaseFloat> linear_params_tp2( in_channel_, kernel_height_ *kernel_width_ *group_);

						linear_params_tp.AddMat(1.0, linear_params_, kTrans);					
						linear_params_tp.TpBlock(in_channel_, kernel_height_*kernel_width_, &linear_params_tp2);

						linear_params_tp2.PaddingZero( kernel_height_, kernel_width_,group_, out_height_- in_pad_height_, out_width_- in_pad_width_, &pad_kernel);
					}

					CuMatrix<BaseFloat> in_deriv_tmp( in_channel_, in_height_ * in_width_ * num_chunks);
					//KALDI_LOG<<" back-prop1";
					pad_kernel.Conv2D( flip_out_deriv, pad_kernel_height, pad_kernel_width, group_, out_height_, out_width_,num_chunks, &in_deriv_tmp, true);
					in_deriv_tmp.TpBlock(num_chunks, in_height_*in_width_, in_deriv);
					//KALDI_LOG<<"done back-prop1";

				}
				else{
					CuMatrix<BaseFloat> pad_out_deriv(out_deriv.NumRows(), pad_out_deriv_height*pad_out_deriv_width*group_);
					CuMatrix<BaseFloat> flip_kernel((kernel_height_*kernel_width_*group_), in_channel_);

					// kw>=pw+1 check
					out_deriv.PaddingZero(out_height_, out_width_, group_, kernel_height_- in_pad_height_ , kernel_width_- in_pad_width_ , &pad_out_deriv);
					//out_deriv.PaddingZero(out_height_, out_width_, group_, kernel_height_- 2 *(in_pad_height_) , kernel_width_- 2 *(in_pad_width_) , &pad_out_deriv);
					linear_params_.FlipMat(kernel_height_, kernel_width_, in_channel_, group_, &flip_kernel);
					//KALDI_LOG<<"back-prop";
					pad_out_deriv.Conv2D( flip_kernel, pad_out_deriv_height, pad_out_deriv_width, group_, kernel_height_, kernel_width_, in_channel_, in_deriv, true);
					//KALDI_LOG<<"done back-prop";
				}
				if (to_update != NULL) {
						to_update->Update(in_value, out_deriv);
				}
		}

		void ConvolutionComponent::SetZero(bool treat_as_gradient) {
			if (treat_as_gradient) {
				SetLearningRate(1.0);
			}
			linear_params_.SetZero();
			bias_params_.SetZero();
			if (treat_as_gradient)
				is_gradient_ = true;
		}

		void ConvolutionComponent::Read(std::istream &is, bool binary) {
			std::ostringstream ostr_beg, ostr_end;
			ostr_beg << "<" << Type() << ">"; // e.g. "<ConvolutionComponent>"
			ostr_end << "</" << Type() << ">"; // e.g. "</ConvolutionComponent>"
			// might not see the "<ConvolutionComponent>" part because
			// of how ReadNew() works.

			ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<in_height>");
			ReadBasicType(is, binary, &in_height_);
			ExpectToken(is, binary, "<in_width>");
			ReadBasicType(is, binary, &in_width_);
			ExpectToken(is, binary, "<in_channel>");
			ReadBasicType(is, binary, &in_channel_);
			ExpectToken(is, binary, "<kernel_height>");
			ReadBasicType(is, binary, &kernel_height_);
			ExpectToken(is, binary, "<kernel_width>");
			ReadBasicType(is, binary, &kernel_width_);
			ExpectToken(is, binary, "<stride>");
			ReadBasicType(is, binary, &stride_); 
			ExpectToken(is, binary, "<padding_height>");
			ReadBasicType(is, binary, &in_pad_height_);
			ExpectToken(is, binary, "<padding_width>");
			ReadBasicType(is, binary, &in_pad_width_);
			ExpectToken(is, binary, "<group>");
			ReadBasicType(is, binary, &group_);
			ExpectToken(is, binary, "<out_height>");
			ReadBasicType(is, binary, &out_height_);
			ExpectToken(is, binary, "<out_width>");
			ReadBasicType(is, binary, &out_width_);

			ExpectToken(is, binary, "<LearningRate>");
			ReadBasicType(is, binary, &learning_rate_);
			ExpectToken(is, binary, "<WeightDecay>");
			ReadBasicType(is, binary, &weight_decay_);
			ExpectToken(is, binary, "<Momentum>");
			ReadBasicType(is, binary, &momentum_);

			ExpectToken(is, binary, "<LinearParams>");
			linear_params_.Read(is, binary);
			ExpectToken(is, binary, "<BiasParams>");
			bias_params_.Read(is, binary);
			ExpectToken(is, binary, "<PrevGrad>");
			prev_grad_.Read(is, binary);

			std::string tok;
			// back-compatibility code.  TODO: re-do this later.
			ReadToken(is, binary, &tok);
			if (tok == "<AvgInput>") { // discard the following.
				CuVector<BaseFloat> avg_input;
				avg_input.Read(is, binary);
				BaseFloat avg_input_count;
				ExpectToken(is, binary, "<AvgInputCount>");
				ReadBasicType(is, binary, &avg_input_count);
				ReadToken(is, binary, &tok);
			}
			if (tok == "<IsGradient>") {
				ReadBasicType(is, binary, &is_gradient_);
				ExpectToken(is, binary, ostr_end.str());
			} else {
				is_gradient_ = false;
				KALDI_ASSERT(tok == ostr_end.str());
			}

		}

		void ConvolutionComponent::Write(std::ostream &os, bool binary) const {
			std::ostringstream ostr_beg, ostr_end;
			ostr_beg << "<" << Type() << ">"; // e.g. "<ConvolutionComponent>"
			ostr_end << "</" << Type() << ">"; // e.g. "</ConvolutionComponent>"
			WriteToken(os, binary, ostr_beg.str());
			WriteToken(os, binary, "<in_height>");
			WriteBasicType(os, binary, in_height_);
			WriteToken(os, binary, "<in_width>");
			WriteBasicType(os, binary, in_width_);
			WriteToken(os, binary, "<in_channel>");
			WriteBasicType(os, binary, in_channel_);
			WriteToken(os, binary, "<kernel_height>");
			WriteBasicType(os, binary, kernel_height_);
			WriteToken(os, binary, "<kernel_width>");
			WriteBasicType(os, binary, kernel_width_);
			WriteToken(os, binary, "<stride>");
			WriteBasicType(os, binary, stride_);
			WriteToken(os, binary, "<padding_height>");
			WriteBasicType(os, binary, in_pad_height_);
			WriteToken(os, binary, "<padding_width>");
			WriteBasicType(os, binary, in_pad_width_);
			WriteToken(os, binary, "<group>");
			WriteBasicType(os, binary, group_);
			WriteToken(os, binary, "<out_height>");
			WriteBasicType(os, binary, out_height_);
			WriteToken(os, binary, "<out_width>");
			WriteBasicType(os, binary, out_width_);

			WriteToken(os, binary, "<LearningRate>");
			WriteBasicType(os, binary, learning_rate_);
			WriteToken(os, binary, "<WeightDecay>");
			WriteBasicType(os, binary, weight_decay_);
			WriteToken(os, binary, "<Momentum>");
			WriteBasicType(os, binary, momentum_);

			WriteToken(os, binary, "<LinearParams>");
			linear_params_.Write(os, binary);
			WriteToken(os, binary, "<BiasParams>");
			bias_params_.Write(os, binary);
			WriteToken(os, binary, "<PrevGrad>");
			prev_grad_.Write(os, binary);

			WriteToken(os, binary, "<IsGradient>");
			WriteBasicType(os, binary, is_gradient_);
			WriteToken(os, binary, ostr_end.str());
		}



		BaseFloat ConvolutionComponent::DotProduct(const kaldi::nnet2::UpdatableComponent &other_in) const {

			const ConvolutionComponent *other =	
				dynamic_cast<const ConvolutionComponent*>(&other_in);
			return TraceMatMat(linear_params_, other->linear_params_, kTrans)+ VecVec(bias_params_, other->bias_params_);			
		}



		kaldi::nnet2::Component* ConvolutionComponent::Copy() const {
			ConvolutionComponent *ans = new ConvolutionComponent();
			ans->learning_rate_ = learning_rate_;
			ans->linear_params_ = linear_params_;
			ans->bias_params_ = bias_params_;
			ans->is_gradient_ = is_gradient_;

			ans->in_height_=in_height_;
			ans->in_width_=in_width_;
			ans->in_channel_=in_channel_;

			ans->kernel_height_=kernel_height_;
			ans->kernel_width_=kernel_width_;
			ans->stride_=stride_;
			ans->in_pad_height_=in_pad_height_;
			ans->in_pad_width_=in_pad_width_;
			ans->group_=group_;

			ans->out_height_=out_height_;
			ans->out_width_=out_width_;

			ans->weight_decay_ = weight_decay_;
			ans->momentum_ = momentum_;
			ans-> prev_grad_ = prev_grad_;

			return ans;
		}
		void ConvolutionComponent::PerturbParams(BaseFloat stddev) {
			CuMatrix<BaseFloat> temp_linear_params(linear_params_);
			temp_linear_params.SetRandn();
			linear_params_.AddMat(stddev, temp_linear_params);

			CuVector<BaseFloat> temp_bias_params(bias_params_);
			temp_bias_params.SetRandn();
			bias_params_.AddVec(stddev, temp_bias_params);
		}

		void ConvolutionComponent::SetParams(const VectorBase<BaseFloat> &bias,
			const MatrixBase<BaseFloat> &linear) {
				bias_params_ = bias;
				linear_params_ = linear;
				KALDI_ASSERT(bias_params_.Dim() == linear_params_.NumRows());
		}

		int32 ConvolutionComponent::GetParameterDim() const {
			return (Group() + 1) * KernelDim();
		}

		void ConvolutionComponent::Vectorize(VectorBase<BaseFloat> *params) const {
			params->Range(0, Group() * KernelDim()).CopyRowsFromMat(linear_params_);
			params->Range(Group() * KernelDim(),
				KernelDim()).CopyFromVec(bias_params_);
		}
		void ConvolutionComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
			linear_params_.CopyRowsFromVec(params.Range(0, Group() * KernelDim()));
			bias_params_.CopyFromVec(params.Range(Group() * KernelDim(),
				KernelDim()));
		}

		void ConvolutionComponent::Update(const CuMatrixBase<BaseFloat> &in_value,
			const CuMatrixBase<BaseFloat> &out_deriv) {
				
				int32 num_sample = in_value.NumRows();
				int32 in_height= in_height_+2*(in_pad_height_), 
					in_width= in_width_+2*(in_pad_width_);

				CuMatrix<BaseFloat> in_value_tmp(in_channel_, num_sample * in_height*in_width);
				CuMatrix<BaseFloat> out_deriv_tmp(out_height_*out_width_ * num_sample, group_);

				CuMatrix<BaseFloat> linear_params_tmp(kernel_height_ * kernel_width_ * in_channel_ , group_);
				CuMatrix<BaseFloat> linear_params_grad(kernel_height_ * kernel_width_ * in_channel_ , group_);

				if(in_pad_height_>0 || in_pad_width_ >0){
					CuMatrix<BaseFloat> padded_input( num_sample, (in_height_+2*(in_pad_height_))*(in_width_+2*(in_pad_width_))*in_channel_);
					in_value.PaddingZero(in_height_ , in_width_, in_channel_, in_pad_height_+1, in_pad_width_+1, &padded_input);
					padded_input.TpBlock(in_channel_, in_height*in_width, &in_value_tmp);
				}else{			

					in_value.TpBlock(in_channel_, in_height*in_width, &in_value_tmp);
				}

				out_deriv.TpInsideBlock(group_, out_height_*out_width_, &out_deriv_tmp);
				//KALDI_LOG << in_pad_width_;
				//KALDI_LOG << in_value_tmp.NumCols() << " " << in_height<< " " <<in_width << " " << in_channel_ ;
				in_value_tmp.Conv2D(out_deriv_tmp, in_height, in_width, num_sample, out_height_, out_width_, group_, &linear_params_tmp, false);

				linear_params_tmp.ModPermuteRow(in_channel_, kernel_height_ * kernel_width_, &linear_params_grad );

				double learning_rate = learning_rate_ / num_sample;

				prev_grad_.Scale(momentum_);
				prev_grad_.AddMat(-1*learning_rate*weight_decay_, linear_params_, kNoTrans);
				prev_grad_.AddMat(learning_rate, linear_params_grad, kNoTrans);
				linear_params_.AddMat(1.0, prev_grad_, kNoTrans);

				//linear_params_.AddMat(learning_rate, linear_params_grad, kNoTrans);
				bias_params_.AddRowSumMat(learning_rate, out_deriv_tmp, 1.0);
				//KALDI_LOG << "update done";
		}

		void MaxpoolComponent::Init(int32 input_dim, int32 output_dim, int32 in_height, int32 in_width, int32 in_channel, int32 pool_height_dim, int32 pool_width_dim, int32 pool_channel_dim, bool overlap, bool overlap2D){
			input_dim_ = input_dim;
			output_dim_ = output_dim;
			in_height_ = in_height;
			in_width_ = in_width;
			in_channel_ = in_channel;
			pool_height_dim_ = pool_height_dim;
			pool_width_dim_ = pool_width_dim;
			pool_channel_dim_ = pool_channel_dim;
			overlap_=overlap;
			overlap2D_ = overlap2D;

			KALDI_ASSERT((in_height_ * in_width_ * in_channel_) == input_dim_ );
			KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0  && pool_height_dim_ > 0 && pool_width_dim_ > 0 && pool_channel_dim_ > 0);			
			KALDI_ASSERT(in_height_ % pool_height_dim_ == 0);
			KALDI_ASSERT(in_width_ % pool_width_dim_ == 0);
			KALDI_ASSERT( (overlap && overlap2D ) != true);

			if (overlap2D){
				//pooling region size = [ pool_channel_dim x pool_channel_dim ]
				KALDI_ASSERT(pool_height_dim_==1 && pool_width_dim_ ==1);
				int32 output_channel_ = output_dim_ / (in_height_ * in_width_);
				int32 expected_output_channel = pow((sqrt(in_channel_) - pool_channel_dim_ + 1),2);
				KALDI_ASSERT( output_channel_ == expected_output_channel);
			
			}else if (overlap){
				KALDI_ASSERT(pool_height_dim_==1 && pool_width_dim_ ==1);
				KALDI_ASSERT(input_dim_ / in_channel_ * (in_channel_ - pool_channel_dim_ +1) == output_dim_ );
			}else{
				KALDI_ASSERT(input_dim_ % output_dim_ == 0);
				KALDI_ASSERT(in_channel_ % pool_channel_dim_ == 0);
				KALDI_ASSERT(input_dim_ / (pool_height_dim_ * pool_width_dim_ * pool_channel_dim_) == output_dim_ );
			}
		}

		void MaxpoolComponent::InitFromString(std::string args) {
			std::string orig_args(args);
			int32 in_height = 1;
			int32 in_width = 1;
			int32 in_channel = 1;
			int32 pool_height_dim = 1;
			int32 pool_width_dim = 1;
			int32 pool_channel_dim = 1;
			bool overlap = false;
			bool overlap2D = false;

			bool ok = ParseFromString("in-height", &args, &in_height) &&
				ParseFromString("in-width", &args, &in_width) &&
				ParseFromString("in-channel", &args, &in_channel) &&
				ParseFromString("pool-height-dim", &args, &pool_height_dim) &&
				ParseFromString("pool-width-dim", &args, &pool_width_dim) &&
				ParseFromString("pool-channel-dim", &args, &pool_channel_dim);

			ParseFromString("overlap", &args, &overlap);			
			ParseFromString("overlap2D", &args, &overlap2D);			

			int32 input_dim = in_height*in_width*in_channel;
			int32 output_dim;

			if (overlap2D){
				int32 output_channel = pow((sqrt(in_channel) - pool_channel_dim + 1),2);
				output_dim = input_dim / in_channel * output_channel;			
			}
			else if (overlap){
				output_dim = input_dim / in_channel * (in_channel - pool_channel_dim +1);
			}
			else{
				output_dim = input_dim / (pool_height_dim * pool_width_dim * pool_channel_dim);
			}

			/*
			KALDI_LOG << " input-dim=" << input_dim
			<< " ( in-height=" << in_height
			<< ", in-width=" << in_width
			<< ", in-channels=" << in_channel

			<< "), output-dim=" << output_dim

			<< ", pool_height_dim_= " << pool_height_dim
			<< ", pool_width_dim_ = " << pool_width_dim 
			<< ", pool_channel_dim_ = " << pool_channel_dim << " " << ok;
			*/
			if (!ok || !args.empty() || output_dim <= 0)
				KALDI_ERR << "Invalid initializer for layer of type "
				<< Type() << ": \"" << orig_args << "\"";

			Init(input_dim, output_dim, in_height, in_width, in_channel, pool_height_dim, pool_width_dim, pool_channel_dim, overlap, overlap2D);

		}

		void MaxpoolComponent::Propagate(const ChunkInfo &in_info,
			const ChunkInfo &out_info,
			const CuMatrixBase<BaseFloat> &in,
			CuMatrixBase<BaseFloat> *out) const {

				in_info.CheckSize(in);
				out_info.CheckSize(*out);
				//CuMatrix<BaseFloat> out_temp(in_info.NumChunks(), output_dim_);
				//in.Maxpool_prop(in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_, overlap_, &out_temp);
				//out->CopyFromMat(out_temp, kNoTrans);
				in.Maxpool_prop(in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_, overlap_, overlap2D_, out);				
		}

		void MaxpoolComponent::Backprop(const ChunkInfo &in_info,
			const ChunkInfo &out_info,
			const CuMatrixBase<BaseFloat> &in_value,
			const CuMatrixBase<BaseFloat> &out_value,
			const CuMatrixBase<BaseFloat> &out_deriv,
			Component *to_update, // to_update
			CuMatrix<BaseFloat> *in_deriv) const {
				in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kSetZero);
				KALDI_ASSERT(output_dim_ == out_value.NumCols());				
				in_value.Maxpool_backprop(out_value, out_deriv, in_deriv, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_, overlap_, overlap2D_);
		}

		void MaxpoolComponent::Read(std::istream &is, bool binary) {
						std::ostringstream ostr_beg, ostr_end;
			ostr_beg << "<" << Type() << ">"; // e.g. "<ConvolutionComponent>"
			ostr_end << "</" << Type() << ">"; // e.g. "</ConvolutionComponent>"


			ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<InputDim>");
			ReadBasicType(is, binary, &input_dim_);
			ExpectToken(is, binary, "<in_height>");
			ReadBasicType(is, binary, &in_height_);
			ExpectToken(is, binary, "<in_width>");
			ReadBasicType(is, binary, &in_width_);
			ExpectToken(is, binary, "<in_channel>");
			ReadBasicType(is, binary, &in_channel_);
			ExpectToken(is, binary, "<OutputDim>");
			ReadBasicType(is, binary, &output_dim_);
			ExpectToken(is, binary, "<PoolHeightDim>");
			ReadBasicType(is, binary, &pool_height_dim_);
			ExpectToken(is, binary, "<PoolWidthDim>");
			ReadBasicType(is, binary, &pool_width_dim_);
			ExpectToken(is, binary, "<PoolChannelDim>");
			ReadBasicType(is, binary, &pool_channel_dim_);
			std::string tok;
			ReadToken(is, binary, &tok);
			if (tok == "<Overlap>") { 
				ReadBasicType(is, binary, &overlap_);
				ReadToken(is, binary, &tok);
				if (tok == "<Overlap2D>") { 
					ReadBasicType(is, binary, &overlap2D_);			
					ExpectToken(is, binary, "</MaxpoolComponent>");
				}
				else{
					overlap2D_=false;
					ExpectToken(is, binary, "</MaxpoolComponent>");
				}
			} else {
				overlap_=false;
				overlap2D_=false;
				KALDI_ASSERT(tok == ostr_end.str());
			}
		}

		void MaxpoolComponent::Write(std::ostream &os, bool binary) const {
			WriteToken(os, binary, "<MaxpoolComponent>");
			WriteToken(os, binary, "<InputDim>");
			WriteBasicType(os, binary, input_dim_);
			WriteToken(os, binary, "<in_height>");
			WriteBasicType(os, binary, in_height_);
			WriteToken(os, binary, "<in_width>");
			WriteBasicType(os, binary, in_width_);
			WriteToken(os, binary, "<in_channel>");
			WriteBasicType(os, binary, in_channel_);
			WriteToken(os, binary, "<OutputDim>");
			WriteBasicType(os, binary, output_dim_);
			WriteToken(os, binary, "<PoolHeightDim>");
			WriteBasicType(os, binary, pool_height_dim_);
			WriteToken(os, binary, "<PoolWidthDim>");
			WriteBasicType(os, binary, pool_width_dim_);
			WriteToken(os, binary, "<PoolChannelDim>");
			WriteBasicType(os, binary, pool_channel_dim_);			
			WriteToken(os, binary, "<Overlap>");
			WriteBasicType(os, binary, overlap_);
			WriteToken(os, binary, "<Overlap2D>");
			WriteBasicType(os, binary, overlap2D_);
			WriteToken(os, binary, "</MaxpoolComponent>");
		}

		std::string MaxpoolComponent::Info() const {
			std::stringstream stream;
			stream << Type()<< " input-dim=" << input_dim_
				<< " ( in-height=" << in_height_
				<< ", in-width=" << in_width_
				<< ", in-channels=" << in_channel_

				<< "), output-dim=" << output_dim_

				<< ", pool_height_dim_= " << pool_height_dim_
				<< ", pool_width_dim_ = " << pool_width_dim_
				<< ", pool_channel_dim_ = " << pool_channel_dim_
				
				<< ", max-pool-overlap_ = " << overlap_
				<< ", max-pool-overlap_2D = " << overlap2D_;

			return stream.str();
		}

		void FullyConnectedComponent::Read(std::istream &is, bool binary) {
			std::ostringstream ostr_beg, ostr_end;
			ostr_beg << "<" << Type() << ">"; // e.g. "<FullyConnectedComponent>"
			ostr_end << "</" << Type() << ">"; // e.g. "</FullyConnectedComponent>"
			// might not see the "<FullyConnectedComponent>" part because
			// of how ReadNew() works.
			ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
			ReadBasicType(is, binary, &learning_rate_);
			ExpectToken(is, binary, "<LinearParams>");
			linear_params_.Read(is, binary);
			ExpectToken(is, binary, "<BiasParams>");
			bias_params_.Read(is, binary);
			ExpectToken(is, binary, "<WeightDecay>");
			ReadBasicType(is, binary, &weight_decay_);
			ExpectToken(is, binary, "<Momentum>");
			ReadBasicType(is, binary, &momentum_);
			ExpectToken(is, binary, "<PrevGrad>");
			prev_grad_.Read(is, binary);
			ExpectToken(is, binary, ostr_end.str());
		}

		void FullyConnectedComponent::Write(std::ostream &os, bool binary) const {
			std::ostringstream ostr_beg, ostr_end;
			ostr_beg << "<" << Type() << ">"; // e.g. "<FullyConnectedComponent>"
			ostr_end << "</" << Type() << ">"; // e.g. "</FullyConnectedComponent>"
			WriteToken(os, binary, ostr_beg.str());
			WriteToken(os, binary, "<LearningRate>");
			WriteBasicType(os, binary, learning_rate_);
			WriteToken(os, binary, "<LinearParams>");
			linear_params_.Write(os, binary);
			WriteToken(os, binary, "<BiasParams>");
			bias_params_.Write(os, binary);
			WriteToken(os, binary, "<WeightDecay>");
			WriteBasicType(os, binary, weight_decay_);
			WriteToken(os, binary, "<Momentum>");
			WriteBasicType(os, binary, momentum_);
			WriteToken(os, binary, "<PrevGrad>");
			prev_grad_.Write(os, binary);
			WriteToken(os, binary, ostr_end.str());

		}

		void FullyConnectedComponent::Init(
			BaseFloat learning_rate,
			int32 input_dim, int32 output_dim,
			BaseFloat param_stddev, BaseFloat bias_stddev,
			BaseFloat weight_decay, BaseFloat momentum){
				UpdatableComponent::Init(learning_rate);
				KALDI_ASSERT(input_dim > 0 && output_dim > 0);
				linear_params_.Resize(output_dim, input_dim);
				bias_params_.Resize(output_dim);
				KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
				linear_params_.SetRandn(); // sets to random normally distributed noise.
				linear_params_.Scale(param_stddev);
				//bias_params_.SetRandn();
				//bias_params_.Scale(bias_stddev);
				bias_params_.SetZero();
				bias_params_.Add(bias_stddev);
				weight_decay_ = weight_decay;
				KALDI_ASSERT(weight_decay_ > 0.0);
				momentum_ = momentum;
				KALDI_ASSERT(momentum_ > 0.0);
				prev_grad_.Resize(output_dim, input_dim);
				prev_grad_.SetZero();

		}


		void FullyConnectedComponent::Init(BaseFloat learning_rate,
			BaseFloat weight_decay, BaseFloat momentum,
			std::string matrix_filename) {
				UpdatableComponent::Init(learning_rate);
				weight_decay_ = weight_decay;
				momentum_ = momentum;
				CuMatrix<BaseFloat> mat;
				ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
				KALDI_ASSERT(mat.NumCols() >= 2);
				int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
				linear_params_.Resize(output_dim, input_dim);
				bias_params_.Resize(output_dim);
				linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));				
				bias_params_.CopyColFromMat(mat, input_dim);
				prev_grad_.Resize(output_dim, input_dim);
				prev_grad_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
		}

		void FullyConnectedComponent::InitFromString(std::string args) {
			std::string orig_args(args);
			std::string matrix_filename;
			BaseFloat learning_rate = learning_rate_;
			BaseFloat weight_decay = weight_decay_, momentum = momentum_;
			int32 input_dim = -1, output_dim = -1;
			ParseFromString("learning-rate", &args, &learning_rate); // optional.
			ParseFromString("weight-decay", &args, &weight_decay);
			ParseFromString("momentum", &args, &momentum);

			if (ParseFromString("matrix", &args, &matrix_filename)) {
				Init(learning_rate, weight_decay, momentum, matrix_filename);
				if (ParseFromString("input-dim", &args, &input_dim))
					KALDI_ASSERT(input_dim == InputDim() &&
					"input-dim mismatch vs. matrix.");
				if (ParseFromString("output-dim", &args, &output_dim))
					KALDI_ASSERT(output_dim == OutputDim() &&
					"output-dim mismatch vs. matrix.");
			} else {
				bool ok = true;
				ok = ok && ParseFromString("input-dim", &args, &input_dim);
				ok = ok && ParseFromString("output-dim", &args, &output_dim);
				BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
					bias_stddev = 1.0;
				ParseFromString("param-stddev", &args, &param_stddev);
				ParseFromString("bias-stddev", &args, &bias_stddev);
				if (!ok)
					KALDI_ERR << "Bad initializer " << orig_args;
				Init(learning_rate, input_dim, output_dim, param_stddev,
					bias_stddev, weight_decay, momentum);
			}
			if (!args.empty())
				KALDI_ERR << "Could not process these elements in initializer: "
				<< args;
		}

		std::string FullyConnectedComponent::Info() const {
			std::stringstream stream;
			BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
				* static_cast<BaseFloat>(linear_params_.NumCols());
			BaseFloat linear_stddev =
				std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
				linear_params_size),
				bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
				bias_params_.Dim());
			stream << Type() << ", input-dim=" << InputDim()
				<< ", output-dim=" << OutputDim()
				<< ", linear-params-stddev=" << linear_stddev
				<< ", bias-params-stddev=" << bias_stddev
				<< ", learning-rate=" << LearningRate()
				<< ", weight-decay=" << weight_decay_
				<< ", momentum=" << momentum_;
			return stream.str();
		}

		Component* FullyConnectedComponent::Copy() const {
			FullyConnectedComponent *ans = new FullyConnectedComponent();
			ans->learning_rate_ = learning_rate_;
			ans->linear_params_ = linear_params_;
			ans->bias_params_ = bias_params_;
			ans->weight_decay_ = weight_decay_;
			ans->momentum_ = momentum_;
			ans->prev_grad_ = prev_grad_;
			ans->is_gradient_ = is_gradient_;
			return ans;
		}

		void FullyConnectedComponent::UpdateSimple(const CuMatrixBase<BaseFloat> &in_value,
			const CuMatrixBase<BaseFloat> &out_deriv) {
				int32 num_sample = in_value.NumRows();
				double learning_rate = learning_rate_ / num_sample;
				bias_params_.AddRowSumMat(learning_rate, out_deriv, 1.0);

				prev_grad_.Scale(momentum_);
				prev_grad_.AddMat(-1*learning_rate*weight_decay_, linear_params_, kNoTrans);
				prev_grad_.AddMatMat(learning_rate, out_deriv, kTrans, in_value, kNoTrans, 1.0);
				linear_params_.AddMat(1.0, prev_grad_, kNoTrans);


				/*
				linear_params_.AddMat(learning_rate*weight_decay_, linear_params_, kNoTrans);
				linear_params_.AddMatMat(learning_rate, out_deriv, kTrans, in_value, kNoTrans, 1.0);
				*/
				//KALDI_LOG << "weight_decay : " << weight_decay_;
		}

		std::string ProbReLUComponent::Info() const {
			std::stringstream stream;
			stream << Component::Info() << ", expectation = "
				<< expectation_ ;
			return stream.str();
		}

		void ProbReLUComponent::InitFromString(std::string args) {
			std::string orig_args(args);
			int32 dim;
			bool expectation = false;
			bool ok = ParseFromString("dim", &args, &dim);
			ParseFromString("expectation", &args, &expectation);

			if (!ok || !args.empty() || dim <= 0)
				KALDI_ERR << "Invalid initializer for layer of type ProbReLUComponent: \""
				<< orig_args << "\"";
			Init(dim, expectation);
		}

		void ProbReLUComponent::Read(std::istream &is, bool binary) {
			ExpectOneOrTwoTokens(is, binary, "<ProbReLUComponent>", "<Dim>");
			ReadBasicType(is, binary, &dim_);
			ExpectToken(is, binary, "<Expectation>");
			ReadBasicType(is, binary, &expectation_);
			ExpectToken(is, binary, "</ProbReLUComponent>");
		}

		void ProbReLUComponent::Write(std::ostream &os, bool binary) const {
			WriteToken(os, binary, "<ProbReLUComponent>");
			WriteToken(os, binary, "<Dim>");
			WriteBasicType(os, binary, dim_);
			WriteToken(os, binary, "<Expectation>");
			WriteBasicType(os, binary, expectation_);
			WriteToken(os, binary, "</ProbReLUComponent>");
		}

		void ProbReLUComponent::Init(int32 dim,	bool expectation){
				dim_ = dim;
				expectation_ = expectation;
		}

		Component* ProbReLUComponent::Copy() const {
			ProbReLUComponent *ans = new ProbReLUComponent();
			ans->dim_ = dim_;
			ans->expectation_ = expectation_;
			return ans;
		}

		void ProbReLUComponent::Propagate(const ChunkInfo &in_info,
			const ChunkInfo &out_info,
			const CuMatrixBase<BaseFloat> &in,
			CuMatrixBase<BaseFloat> *out) const  {
				// Apply rectified linear function (x >= 0 ? 1.0 : 0.0)
				CuMatrix<BaseFloat> prob(in.NumRows(), in.NumCols());				
				prob.CopyFromMat(in);
				prob.ApplyFloor(0.0); // ReLU
				prob.Add(1.0); //p=1-1/(in+1);
				prob.ApplyPow(-1.0);
				prob.Scale(-1.0);
				prob.Add(1.0);
				//KALDI_LOG << "in = " << in;
				//KALDI_LOG << "prob = " << prob;

				out_info.CheckSize(*out);
				KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());
				KALDI_ASSERT(in.NumCols() == this->InputDim());

				if (!expectation_){
					// This const_cast is only safe assuming you don't attempt
					// to use multi-threaded code with the GPU.
					const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(out);
					//KALDI_LOG << "out = " << *out;
					out->AddMat(-1.0, prob);
					out->Scale(-1.0);
					out->ApplyHeaviside(); // apply the function (x>0?1:0).  Now, a proportion "dp" will
					// be 1.0 and (1-dp) will be 0.0  
					//KALDI_LOG << "out_mask = " << *out;
					out->MulElements(in);
					//KALDI_LOG << "bernouilli";
				}else{
					//KALDI_LOG << "expectation ";
					out->CopyFromMat(prob);
					out->MulElements(in);
					//KALDI_LOG << "prob = " << prob;
					//KALDI_LOG << "out = " << *out;

				}

		}

		void ProbReLUComponent::Backprop(const ChunkInfo &,  //in_info,
			const ChunkInfo &,  //out_info,
			const CuMatrixBase<BaseFloat> &,  //in_value,
			const CuMatrixBase<BaseFloat> &out_value,
			const CuMatrixBase<BaseFloat> &out_deriv,
			Component *to_update, // may be identical to "this".
			CuMatrix<BaseFloat> *in_deriv) const  {

				in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols(),
					kUndefined);
				in_deriv->CopyFromMat(out_value);
				in_deriv->ApplyHeaviside();
				// Now in_deriv(i, j) equals (out_value(i, j) > 0.0 ? 1.0 : 0.0),
				// which is the derivative of the nonlinearity (well, except at zero
				// where it's undefined).
				if (to_update != NULL)
					dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
					in_deriv);
				
				// Updatestates -> public function
				in_deriv->MulElements(out_deriv);
		}

		void ConvolutionComponentContainer::InitFromString(std::string args) {
			std::string orig_args(args);
			bool ok = true;
			BaseFloat learning_rate, weight_decay, momentum;
			int32 in_height, in_width, in_channel, num_component, group, out_height, out_width;

			int32 stride=1;

			ok=ok && ParseFromString("learning-rate", &args, &learning_rate); this->learning_rate_=learning_rate;
			ok=ok && ParseFromString("in-height", &args, &in_height); this->in_height_=in_height;
			ok=ok && ParseFromString("in-width", &args, &in_width); this->in_width_=in_width;
			ok=ok && ParseFromString("in-channel", &args, &in_channel);this->in_channel_=in_channel;

			ok=ok && ParseFromString("num-component", &args, &num_component);this->num_component_=num_component;
			
			
			// kernel-height:kernel-width:groups:zero-padding-height:zp-width, ... 
			std::string kernels="";
			//std::vector< std::vector<BaseFloat> > kernels_vec;
			ok=ok && ParseFromString("kernels", &args, &kernels);
			if (kernels != "") {
				std::vector<std::string> kernels_str_vec;
				SplitStringToVector(kernels, ",", false, &kernels_str_vec);

				if (static_cast<int32>(kernels_str_vec.size()) != num_component){
						KALDI_ERR << "Expected --kernels option to be a comma(,) string with "
								<< num_component
								<< " elements, instead got "
								<< kernels;
				}

				std::vector<BaseFloat> kernels_vec_temp;
				for ( int i=0; i<num_component; i++ ){					
					if (!SplitStringToFloats(kernels_str_vec[i], ":", false, &kernels_vec_temp)
						|| static_cast<int32>(kernels_vec_temp.size()) != 5){
							KALDI_ERR << "Expected --kernels option to be a (:) string with 5 elements, instead got " << kernels_str_vec[i];
					}
					this->kernels_vec_.push_back(kernels_vec_temp);
				}
			}

			ok=ok && ParseFromString("group", &args, &group); this->group_=group;
			ok=ok && ParseFromString("out-height", &args, &out_height); this->out_height_=out_height;
			ok=ok && ParseFromString("out-width", &args, &out_width); this->out_width_=out_width;

			//BaseFloat param_stddev = 1.0 / std::sqrt(kernel_height * kernel_width),	bias_stddev = 1.0;
			BaseFloat param_stddev, bias_stddev;
			ParseFromString("param-stddev", &args, &param_stddev); 
			ParseFromString("bias-stddev", &args, &bias_stddev);
			ParseFromString("weight-decay", &args, &weight_decay); this->weight_decay_=weight_decay;
			ParseFromString("momentum", &args, &momentum); this->momentum_=momentum;

			// Initializing ConvComponents
			for ( int i=0; i<num_component; i++ ){					

				ConvolutionComponent *ConvComp = new ConvolutionComponent();

				ConvComp->Init(learning_rate,
					in_height, in_width, in_channel,
					kernels_vec_[i][3], kernels_vec_[i][4], 
					kernels_vec_[i][0], kernels_vec_[i][1], stride,
					kernels_vec_[i][2], out_height, out_width,
					param_stddev, bias_stddev,
					weight_decay, momentum);    

				KALDI_ASSERT(ConvComp != NULL);
				this->ConvComps_.push_back(ConvComp);				
			}

			if (!args.empty())
				KALDI_ERR << "Could not process these elements in initializer: "
				<< args;
			if (!ok)
				KALDI_ERR << "Bad initializer " << orig_args;

			//this->indim_ = in_height * in_width * in_channel;
			//this->outdim_ =group * out_height * out_width;
		}

		void ConvolutionComponentContainer::Propagate(const ChunkInfo &in_info,
			const ChunkInfo &out_info,
			const CuMatrixBase<BaseFloat> &in,
			CuMatrixBase<BaseFloat> *out) const{
				// Propagate
				bool fromCompToContainer = true;
				for ( int i=0; i<num_component_; i++ ){
					//KALDI_LOG << "here ";
					//KALDI_LOG << out_info.NumChunks() << ", " << out_info.NumCols()/num_component_;
					
					ChunkInfo out_info_temp(out_info.NumCols()/num_component_, out_info.NumChunks(), out_info.first_offset_ , out_info.last_offset_); 
					//ChunkInfo's private variables -> public
					CuMatrix<BaseFloat> out_comp(out_info.NumChunks(), (out_info.NumCols()/num_component_));
					//KALDI_LOG << "in_info : "<< in_info.first_offset_ <<" " << in_info.last_offset_;
					//KALDI_LOG << "out_info : "<< out_info.first_offset_ <<" " << out_info.last_offset_;

					//KALDI_LOG << out_vec.NumRows() << " " << out_vec.NumCols();
					//KALDI_LOG << &out_vec;
					//KALDI_LOG << out;
					
					ConvComps_[i]->Propagate(in_info, out_info_temp, in, &out_comp);			

					//re-ordering
					out_comp.ModPermuteChannel(i, num_component_, out_height_, out_width_, out, fromCompToContainer);
					//KALDI_LOG << "component = " << out_comp.RowRange(0, 10);
					//KALDI_LOG << "out = " <<  out->RowRange(0, 10);
				}
				//KALDI_LOG << "conv container prop: " << *out;
		}

		void ConvolutionComponentContainer::Backprop(const ChunkInfo &in_info,
				const ChunkInfo &out_info,
				const CuMatrixBase<BaseFloat> &in_value,
				const CuMatrixBase<BaseFloat> &out_value,                        
				const CuMatrixBase<BaseFloat> &out_deriv,
				Component *to_update, // may be identical to "this".
				CuMatrix<BaseFloat> *in_deriv) const{	

					bool fromCompToContainer = false;
					CuMatrix<BaseFloat> in_deriv_temp(in_info.NumChunks(), in_info.NumCols());
					CuMatrix<BaseFloat> out_value_temp(out_info.NumChunks(), out_info.NumCols());
					out_value_temp.CopyFromMat(out_value);
					CuMatrix<BaseFloat> out_deriv_temp(out_info.NumChunks(), out_info.NumCols());
					out_deriv_temp.CopyFromMat(out_deriv);

					// Backprop
					for ( int i=0; i<num_component_; i++ ){					
						// re-ordering index (out_value, out_deriv)
						ChunkInfo out_info_temp(out_info.NumCols()/num_component_, out_info.NumChunks(), out_info.first_offset_, out_info.last_offset_);
						CuMatrix<BaseFloat> out_comp(out_info.NumChunks(), (out_info.NumCols()/num_component_));
						CuMatrix<BaseFloat> out_deriv_comp(out_info.NumChunks(), (out_info.NumCols()/num_component_));

						out_comp.ModPermuteChannel(i, num_component_, out_height_, out_width_, &out_value_temp, fromCompToContainer);
						out_deriv_comp.ModPermuteChannel(i, num_component_, out_height_, out_width_, &out_deriv_temp, fromCompToContainer);

						CuMatrix<BaseFloat> in_deriv_comp(in_info.NumChunks(), in_info.NumCols());
						ConvComps_[i]->Backprop(in_info, out_info_temp, in_value, out_comp, out_deriv_comp, ConvComps_[i], &in_deriv_comp);

						// summation in_deriv
						in_deriv_temp.AddMat(1, in_deriv_comp, kNoTrans);
					}					
					in_deriv->CopyFromMat(in_deriv_temp);
		}

		kaldi::nnet2::Component* ConvolutionComponentContainer::Copy() const {
			ConvolutionComponentContainer *ans = new ConvolutionComponentContainer();
			ans->ConvComps_ = ConvComps_;
			ans->kernels_vec_ = kernels_vec_;
			ans->learning_rate_ = learning_rate_;
			ans->weight_decay_ = weight_decay_;
			ans->momentum_ = momentum_;
			ans->num_component_ = num_component_;
			//ans->indim_ = indim_;
			//ans->outdim_ = outdim_;
			ans->in_height_ = in_height_;
			ans->in_width_ = in_width_;
			ans->in_channel_ = in_channel_;
			ans->group_ = group_;
			ans->out_height_ = out_height_;
			ans->out_width_ = out_width_;
			
			return ans;
		}

		void ConvolutionComponentContainer::Read(std::istream &is, bool binary){
			std::ostringstream ostr_beg, ostr_end;
			ostr_beg << "<" << Type() << ">"; // e.g. "<ConvolutionComponentContainer>"
			ostr_end << "</" << Type() << ">"; // e.g. "</ConvolutionComponentContainer>"
			ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<in_height>");
			ReadBasicType(is, binary, &in_height_);
			ExpectToken(is, binary, "<in_width>");
			ReadBasicType(is, binary, &in_width_);
			ExpectToken(is, binary, "<in_channel>");
			ReadBasicType(is, binary, &in_channel_);
			ExpectToken(is, binary, "<num_component>");
			ReadBasicType(is, binary, &num_component_);
					
			ExpectToken(is, binary, "<kernels>");
			for ( int i=0; i<num_component_; i++ ){
				std::vector<BaseFloat> kernel;
				int32 temp;
				for (int j=0; j<5; j++){
					ExpectToken(is, binary, ":");
					ReadBasicType(is, binary, &temp);					
					kernel.push_back(temp);
				}
				kernels_vec_.push_back(kernel);
				ExpectToken(is, binary, ",");
			}

			ExpectToken(is, binary, "<group>");
			ReadBasicType(is, binary, &group_);
			ExpectToken(is, binary, "<out_height>");
			ReadBasicType(is, binary, &out_height_);
			ExpectToken(is, binary, "<out_width>");
			ReadBasicType(is, binary, &out_width_);

			ExpectToken(is, binary, "<LearningRate>");
			ReadBasicType(is, binary, &learning_rate_);
			ExpectToken(is, binary, "<WeightDecay>");
			ReadBasicType(is, binary, &weight_decay_);
			ExpectToken(is, binary, "<Momentum>");
			ReadBasicType(is, binary, &momentum_);

			for ( int i=0; i<num_component_; i++ ){
				//ConvComps_[i]->Read(is, binary);
				ConvolutionComponent *ConvComp = new ConvolutionComponent();
				ConvComp->Read(is, binary);
				KALDI_ASSERT(ConvComp != NULL);
				this->ConvComps_.push_back(ConvComp);
			}
			ExpectToken(is, binary, ostr_end.str());
		}

		void ConvolutionComponentContainer::Write(std::ostream &os, bool binary) const{
			std::ostringstream ostr_beg, ostr_end;
			ostr_beg << "<" << Type() << ">"; // e.g. "<ConvolutionComponentContainer>"
			ostr_end << "</" << Type() << ">"; // e.g. "</ConvolutionComponentContainer>"
			WriteToken(os, binary, ostr_beg.str());
			WriteToken(os, binary, "<in_height>");
			WriteBasicType(os, binary, in_height_);
			WriteToken(os, binary, "<in_width>");
			WriteBasicType(os, binary, in_width_);
			WriteToken(os, binary, "<in_channel>");
			WriteBasicType(os, binary, in_channel_);
			WriteToken(os, binary, "<num_component>");
			WriteBasicType(os, binary, num_component_);

			WriteToken(os, binary, "<kernels>");
			for ( int i=0; i<num_component_; i++ ){
				for (int j=0; j<5; j++){
					WriteToken(os, binary, ":");
					WriteBasicType(os, binary, static_cast<int32>(kernels_vec_[i][j]));					
				}
				WriteToken(os, binary, ",");
			}

			WriteToken(os, binary, "<group>");
			WriteBasicType(os, binary, group_);
			WriteToken(os, binary, "<out_height>");
			WriteBasicType(os, binary, out_height_);
			WriteToken(os, binary, "<out_width>");
			WriteBasicType(os, binary, out_width_);

			WriteToken(os, binary, "<LearningRate>");
			WriteBasicType(os, binary, learning_rate_);
			WriteToken(os, binary, "<WeightDecay>");
			WriteBasicType(os, binary, weight_decay_);
			WriteToken(os, binary, "<Momentum>");
			WriteBasicType(os, binary, momentum_);

			for ( int i=0; i<num_component_; i++ ){
				ConvComps_[i]->Write(os, binary);
			}
			WriteToken(os, binary, ostr_end.str());
		}

		std::string ConvolutionComponentContainer::Info() const{
			return 0;
		}

		// need check later (not used functions in training)
		void ConvolutionComponentContainer::SetZero(bool treat_as_gradient){
			if (treat_as_gradient) {
				SetLearningRate(1.0);
			}

			for ( int i=0; i<num_component_; i++ ){
				ConvComps_[i]->SetZero(treat_as_gradient);
			}

			//if (treat_as_gradient)
				//is_gradient_ = true;
		}
		// need check later (not used functions in training)
		BaseFloat ConvolutionComponentContainer::DotProduct(const UpdatableComponent &other) const{
			return 0;
		}
		// need check later (not used functions in training)
		void ConvolutionComponentContainer::PerturbParams(BaseFloat stddev){		
		}
		// need check later (not used functions in training)
		void ConvolutionComponentContainer::Scale(BaseFloat scale){		
		}
		// need check later (not used functions in training)
		void ConvolutionComponentContainer::Add(BaseFloat alpha, const UpdatableComponent &other){		
		}


	} // namespace nnet0
} // namespace cnsl
