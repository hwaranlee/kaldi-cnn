// cnslmat/cnsl-cu-kernels.cu

// Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)

#include <cfloat>

//#include "cu-kernels-ansi.h"
#include "cnslmat/cnsl-cu-kernels.h"

template<typename Real>
__global__
	static void _span_row_to_convmat(const Real *in, MatrixDim in_dim, Real *span, MatrixDim span_dim, 
	int in_height, int in_width, int in_channel, int kernel_height, int kernel_width, int row_offset){

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		int kernelsize = kernel_height*kernel_width,
			q = in_height-kernel_height+1;

		if (i < span_dim.rows && j < span_dim.cols) {

            int i_offset = i + row_offset,
				Ir = i_offset % in_dim.rows, 
				I = i_offset / in_dim.rows,
				Jr = j % kernelsize,
				J = j / kernelsize,
				Q = I % q + I / q * in_height,
				//P = (Jr % kernel_height) + (Jr/kernel_height) * (in_height-kernel_height) * in_height,
				P = (Jr % kernel_height) + (Jr/kernel_height) * in_height,

				index = Ir *in_dim.stride + Q+P+(J*in_height*in_width),
				span_idx = i * span_dim.stride + j;

			span[span_idx] = in[index];

		}
}



template<typename Real>
__global__
	static void _convmat_to_out(const Real *convMat, MatrixDim conv_dim, Real *out, MatrixDim out_dim, int out_height, int out_width, int num_sample){

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < conv_dim.rows && j < conv_dim.cols) {

			int Ir = i % num_sample, 
				I = i / num_sample,

				out_idx = Ir *out_dim.stride + (I + j *out_height*out_width),
				index= i * conv_dim.stride + j;

			out[out_idx] = convMat[index];
		}
}

template<typename Real>
__global__
	static void _add_mat_rep_vec(const Real *vec, int rep, Real *out, MatrixDim out_dim){

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < out_dim.rows && j < out_dim.cols) {

			int group = j / rep, 
				index= i * out_dim.stride + j;

			out[index] += vec[group];
		}
}


template<typename Real>
__global__
	static void _flip_mat(const Real *orig, MatrixDim orig_dim, int kernel_height, int kernel_width, int group, Real *flip, MatrixDim flip_dim){

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width
		int ksize = kernel_height*kernel_width;

		if (i < flip_dim.rows && j < flip_dim.cols) {
			
			int group_idx =  i / ksize,
				p = (group_idx + 1)*ksize -1 - i,
				m = p+j*ksize,
				orig_idx = m * orig_dim.stride + group_idx,
				flip_idx = i * flip_dim.stride + j;
			
			flip[flip_idx] = orig[orig_idx];

		}
}


template<typename Real>
__global__
	static void _pad_zero(const Real *orig, MatrixDim orig_dim, int orig_height, int orig_width, int kernel_height, int kernel_width, Real *padmat, MatrixDim padmat_dim){

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		int padmat_height = orig_height + 2*(kernel_height-1);
		int padmat_width = orig_width + 2*(kernel_width-1);
		int padmat_size = padmat_height*padmat_width;


		if (i < padmat_dim.rows && j < padmat_dim.cols) {
			
			int chan_idx = j / padmat_size,
				p = j % padmat_size,
				I = p % padmat_height,
				J = p/ padmat_height,
				padmat_idx = i*padmat_dim.stride +j;

			if ( (kernel_height-1) <= I && I < (kernel_height+orig_height-1) && (kernel_width-1) <= J && J < (kernel_width+orig_width-1)){

				int m = I-kernel_height+1,
					n = J-kernel_width+1,
					idx = (n*orig_height + m) + chan_idx*(orig_height*orig_width);

				padmat[padmat_idx] = orig[ i*orig_dim.stride + idx];
			}
			else{
				padmat[padmat_idx] = 0;
			}


		}
}



template<typename Real>
__global__
	static void _tp_block(const Real *in, MatrixDim in_dim, Real *out, MatrixDim out_dim, int block_size){
		// each block is vector
		// in = [ A B C ; D E F];
		// out = [A D ; B E ; C F];		
		// block_size = in_height * in_width

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		

		if (i < out_dim.rows && j < out_dim.cols) {

			int row = j/block_size,
				col = i*block_size + j%block_size,

				index = row *in_dim.stride + col,
				out_idx= i * out_dim.stride + j;

			out[out_idx] = in[index];
		}
}



template<typename Real>
__global__
	static void _tp_inside_block(const Real *in, MatrixDim in_dim, Real *out, MatrixDim out_dim, int block_size){
		// each block is vector
		// in = [ A B C ; D E F];
		// out = [A' B' C' ; D' E' F'];

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < out_dim.rows && j < out_dim.cols) {

			int row = i/block_size,
				col = j*block_size + i %block_size,

				index = row *in_dim.stride + col,
				out_idx= i * out_dim.stride + j;

			out[out_idx] = in[index];
		}
}



template<typename Real>
__global__
	static void _mod_permute_row(const Real *in, MatrixDim in_dim, Real *out, MatrixDim out_dim, int block_size, int in_channel){
		// for example in_channel = 2
		// in = [ r1; r2; r3; r4; r5; r6 ];
		// out = [ r1; r3; r5; r2; r4; r6];

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < out_dim.rows && j < out_dim.cols) {

			int chan_idx = i % in_channel, 
				pos_idx = i / in_channel,

				out_idx = (chan_idx *block_size + pos_idx )*out_dim.stride + j,

				index= i * in_dim.stride + j;

			out[out_idx] = in[index];
		}
}



template<typename Real>
__global__
	static void _copy_rows_at(const Real *src, MatrixDim src_dim, Real *dest, MatrixDim dest_dim, int row_offset){

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < src_dim.rows && j < src_dim.cols) {

			int dest_idx = (i+row_offset)*dest_dim.stride + j,
				src_idx = i * src_dim.stride + j;

			dest[dest_idx] = src[src_idx];
		}
}


template<typename Real>
__global__
	static void _maxpool_prop(const Real *src, MatrixDim src_dim, Real *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < pool_dim.rows && j < pool_dim.cols) {
			
			int out_height = in_height_ / pool_height_dim_,
				out_width = in_width_ / pool_width_dim_;

			int out_channel_idx = j / (out_height * out_width), 
				out_position_idx = j % (out_height * out_width),
				out_width_idx = out_position_idx / out_height,
				out_height_idx = out_position_idx % out_height;

			int startpoint = ( out_channel_idx * pool_channel_dim_ * in_height_ * in_width_ ) 
				+ ( out_width_idx * pool_width_dim_ * in_height_) + out_height_idx * pool_height_dim_;  

			Real val = -1e20;

			for (int c = 0; c  < pool_channel_dim_; c++){
				for (int w = 0; w < pool_width_dim_; w++){
					for (int h = 0; h < pool_height_dim_; h++){

						int idx = h + w * in_height_ + c * in_height_ * in_width_;
						Real src_val = src[ i * src_dim.stride + startpoint + idx ];
						
						if ( val < src_val ) val = src_val;
					}
				}
			}
			
			int dest_idx = i * pool_dim.stride + j;
			pool[dest_idx] = val;

		}
}

template<typename Real>
__global__
	static void _maxpool_backprop(const Real *in_val, MatrixDim in_val_dim, const Real *out_val, MatrixDim out_val_dim, const Real *out_deriv, MatrixDim out_deriv_dim, 
				Real *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < out_deriv_dim.rows && j < out_deriv_dim.cols) {

			int out_height = in_height_ / pool_height_dim_,
				out_width = in_width_ / pool_width_dim_;

			int out_channel_idx = j / (out_height * out_width), 
				out_position_idx = j % (out_height * out_width),
				out_width_idx = out_position_idx / out_height,
				out_height_idx = out_position_idx % out_height;


			int startpoint = ( out_channel_idx * pool_channel_dim_ * in_height_ * in_width_ ) + ( out_width_idx * pool_width_dim_ * in_height_) + out_height_idx * pool_height_dim_;  

			Real out_value = out_val[ i * out_val_dim.stride + j ] ,
				err = out_deriv[ i * out_val_dim.stride + j ];

			for (int c = 0; c  < pool_channel_dim_; c++){
				for (int w = 0; w < pool_width_dim_; w++){
					for (int h = 0; h < pool_height_dim_; h++){

						int idx = h + w * in_height_ + c * in_height_ * in_width_;
						Real in_value = in_val[ i * in_val_dim.stride + startpoint + idx ];
						
						if ( out_value == in_value ) 
							dest[i * dest_dim.stride + startpoint + idx ] = err;
					}
				}
			}
		}
}

template<typename Real>
__global__
	static void _maxpoolchannel_overlap_prop(const Real *src, MatrixDim src_dim, Real *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_channel_dim_){
		// pooling accross channel only
		// pool_height_dim_ = pool_width_dim_ = 1
		// stride = 1
		int pool_height_dim_ = 1;
		int pool_width_dim_ = 1;

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < pool_dim.rows && j < pool_dim.cols) {
			
			int out_height = in_height_ / pool_height_dim_,
				out_width = in_width_ / pool_width_dim_;

			int out_channel_idx = j / (out_height * out_width), 
				out_position_idx = j % (out_height * out_width),
				out_width_idx = out_position_idx / out_height,
				out_height_idx = out_position_idx % out_height;

			//int startpoint = ( out_channel_idx * pool_channel_dim_ * in_height_ * in_width_ ) 
			//	+ ( out_width_idx * pool_width_dim_ * in_height_) + out_height_idx * pool_height_dim_;  

			int startpoint = ( out_channel_idx * in_height_ * in_width_ ) 
				+ ( out_width_idx * pool_width_dim_ * in_height_) + out_height_idx * pool_height_dim_;  
			
			Real val = -1e20;

			for (int c = 0; c  < pool_channel_dim_; c++){
				for (int w = 0; w < pool_width_dim_; w++){
					for (int h = 0; h < pool_height_dim_; h++){

						int idx = h + w * in_height_ + c * in_height_ * in_width_;
						Real src_val = src[ i * src_dim.stride + startpoint + idx ];
						
						if ( val < src_val ) val = src_val;
					}
				}
			}
			
			int dest_idx = i * pool_dim.stride + j;
			pool[dest_idx] = val;

		}
}

template<typename Real>
__global__
	static void _maxpoolchannel_overlap_backprop(const Real *in_val, MatrixDim in_val_dim, const Real *out_val, MatrixDim out_val_dim, const Real *out_deriv, MatrixDim out_deriv_dim, 
				Real *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_channel_dim_){
		// pooling accross channel only
		// pool_height_dim_ = pool_width_dim_ = 1
		// stride = 1
		int pool_height_dim_ = 1;
		int pool_width_dim_ = 1;

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < out_deriv_dim.rows && j < out_deriv_dim.cols) {

			int out_height = in_height_ / pool_height_dim_,
				out_width = in_width_ / pool_width_dim_;

			int out_channel_idx = j / (out_height * out_width), 
				out_position_idx = j % (out_height * out_width),
				out_width_idx = out_position_idx / out_height,
				out_height_idx = out_position_idx % out_height;


			//int startpoint = ( out_channel_idx * pool_channel_dim_ * in_height_ * in_width_ ) + ( out_width_idx * pool_width_dim_ * in_height_) + out_height_idx * pool_height_dim_;  
			int startpoint = ( out_channel_idx * in_height_ * in_width_ ) + ( out_width_idx * pool_width_dim_ * in_height_) + out_height_idx * pool_height_dim_;  
			
			Real out_value = out_val[ i * out_val_dim.stride + j ] ,
				err = out_deriv[ i * out_val_dim.stride + j ];

			for (int c = 0; c  < pool_channel_dim_; c++){
				for (int w = 0; w < pool_width_dim_; w++){
					for (int h = 0; h < pool_height_dim_; h++){

						int idx = h + w * in_height_ + c * in_height_ * in_width_;
						Real in_value = in_val[ i * in_val_dim.stride + startpoint + idx ];
						
						if ( out_value == in_value ){
							Real prev_dest_val=dest[i * dest_dim.stride + startpoint + idx ];
							dest[i * dest_dim.stride + startpoint + idx ] = prev_dest_val+err;
						}
					}
				}
			}
		}
}

template<typename Real>
__global__
	static void _maxpoolchannel_overlap2D_prop(const Real *src, MatrixDim src_dim, Real *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_channel_dim_){
		// pooling accross channel only
		// pool_height_dim_ = pool_width_dim_ = 1
		// stride = 1

		// 2D channel pooling size = pool_channel_dim_ x pool_channel_dim_
		// pool_dim.cols / (out_height * out_width) = (out_2d_map)^2

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		int out_height = in_height_ ,
			out_width = in_width_ ,
			out_channel = pool_dim.cols / (out_height * out_width),
			out_2d_map = sqrt(out_channel), //2D-topological map size
			in_2d_map = out_2d_map + pool_channel_dim_ - 1;

		if (i < pool_dim.rows && j < pool_dim.cols) {
			
			int out_channel_idx = j / (out_height * out_width), 
				out_position_idx = j % (out_height * out_width);

			int channel_2d_x = out_channel_idx / out_2d_map,
				channel_2d_y = out_channel_idx % out_2d_map;
			
			Real val = -1e20;

			for (int cx = 0; cx  < pool_channel_dim_; cx++){
				for (int cy = 0; cy < pool_channel_dim_; cy++){
					
					int in_channel_2d_x = channel_2d_x + cx,
						in_channel_2d_y = channel_2d_y + cy,
						in_channel_idx = in_channel_2d_x * in_2d_map  + in_channel_2d_y;

					int idx = ( in_channel_idx * in_height_ * in_width_ ) + out_position_idx;
					Real src_val = src[ i * src_dim.stride + idx ];

					if ( val < src_val ) val = src_val;					
				}
			}

			int dest_idx = i * pool_dim.stride + j;
			pool[dest_idx] = val;

		}
}

template<typename Real>
__global__
	static void _maxpoolchannel_overlap2D_backprop(const Real *in_val, MatrixDim in_val_dim, const Real *out_val, MatrixDim out_val_dim, const Real *out_deriv, MatrixDim out_deriv_dim, 
	Real *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_channel_dim_){
		// pooling accross channel only
		// pool_height_dim_ = pool_width_dim_ = 1
		// stride = 1

		// 2D channel pooling size = pool_channel_dim_ x pool_channel_dim_
		// pool_dim.cols / (out_height * out_width) = (out_2d_map)^2

		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		int out_height = in_height_ ,
			out_width = in_width_ ,
			out_channel = out_deriv_dim.cols / (out_height * out_width),
			out_2d_map = sqrt(out_channel), //2D-topological map size
			in_2d_map = out_2d_map + pool_channel_dim_ - 1;

		if (i < out_deriv_dim.rows && j < out_deriv_dim.cols) {

			int out_channel_idx = j / (out_height * out_width), 
				out_position_idx = j % (out_height * out_width);

			int channel_2d_x = out_channel_idx / out_2d_map,
				channel_2d_y = out_channel_idx % out_2d_map;

			Real out_value = out_val[ i * out_val_dim.stride + j ] ,
				err = out_deriv[ i * out_val_dim.stride + j ];

			for (int cx = 0; cx  < pool_channel_dim_; cx++){
				for (int cy = 0; cy < pool_channel_dim_; cy++){

					int in_channel_2d_x = channel_2d_x + cx,
						in_channel_2d_y = channel_2d_y + cy,
						in_channel_idx = in_channel_2d_x * in_2d_map  + in_channel_2d_y;

					int idx = ( in_channel_idx * in_height_ * in_width_ ) + out_position_idx;

					Real in_value = in_val[ i * in_val_dim.stride + idx ];

					if ( out_value == in_value ){
						Real prev_dest_val=dest[i * dest_dim.stride + idx ];
						dest[i * dest_dim.stride + idx ] = prev_dest_val+err;
					}
				}
			}
		}
}

template<typename Real>
__global__
	static void _mod_permute_channels(Real *comp, MatrixDim comp_dim, Real *container, MatrixDim container_dim, int comp_idx, int num_component, int in_height, int in_width, bool fromCompToContainer){
		// For ConvolutionComponentContainer
		// in_height, in_width_ in_channels are of comp.


		int i = blockIdx.y * blockDim.y + threadIdx.y; // row index - height
		int j = blockIdx.x * blockDim.x + threadIdx.x; // column index - width

		if (i < comp_dim.rows && j < comp_dim.cols) {

			int chan_idx = j / (in_height * in_width), 
				pos_idx = j % (in_height * in_width),
				index = i * comp_dim.stride + j,
				out_chan_idx = chan_idx * num_component + comp_idx,
				out_idx = i * container_dim.stride + ( out_chan_idx * (in_height * in_width) + pos_idx );

			if (fromCompToContainer){
				container[out_idx] = comp[index];
			}else{
				comp[index] = container[out_idx];
			}			
		}
}



/*
* "float"
*/
void cudaF_span_row_to_convmat(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *span, MatrixDim span_dim, int in_height, int in_width, int in_channel, int kernel_height, int kernel_width, int row_offset){
	_span_row_to_convmat<<<Gr,Bl>>> (in, in_dim, span, span_dim, in_height, in_width, in_channel, kernel_height, kernel_width, row_offset);
}

void cudaF_convmat_to_out(dim3 Gr, dim3 Bl, const float *convMat, MatrixDim conv_dim, float *out, MatrixDim out_dim, int out_height, int out_width, int num_sample){
	_convmat_to_out<<<Gr,Bl>>>(convMat, conv_dim, out, out_dim, out_height, out_width, num_sample);
}

void cudaF_add_mat_rep_vec(dim3 Gr, dim3 Bl, const float *vec, int rep, float *out, MatrixDim out_dim){
	_add_mat_rep_vec<<<Gr, Bl>>>(vec, rep, out, out_dim);
}

void cudaF_flip_mat(dim3 Gr, dim3 Bl, const float *orig, MatrixDim orig_dim, int kernel_height, int kernel_width, int group, float *flip, MatrixDim flip_dim){
	_flip_mat<<<Gr, Bl>>>(orig, orig_dim, kernel_height, kernel_width, group, flip, flip_dim);
}

void cudaF_pad_zero(dim3 Gr, dim3 Bl, const float *orig, MatrixDim orig_dim, int orig_height, int orig_width, int kernel_height, int kernel_width, float *padmat, MatrixDim padmat_dim){
	_pad_zero<<<Gr, Bl>>>(orig, orig_dim, orig_height, orig_width, kernel_height, kernel_width, padmat, padmat_dim);
}

void cudaF_tp_block(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size){
    _tp_block<<<Gr, Bl>>>(in, in_dim, out, out_dim, block_size);
}

void cudaF_tp_inside_block(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size){
    _tp_inside_block<<<Gr, Bl>>>(in, in_dim, out, out_dim, block_size);
}

void cudaF_mod_permute_row(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size, int in_channel){
    _mod_permute_row<<<Gr, Bl>>>(in, in_dim, out, out_dim, block_size, in_channel);
}
void cudaF_copy_rows_at(dim3 Gr, dim3 Bl, const float *src, MatrixDim src_dim, float *dest, MatrixDim dest_dim, int row_offset){
    _copy_rows_at<<<Gr, Bl>>>(src, src_dim, dest, dest_dim, row_offset);
}

void cudaF_maxpool_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpool_prop<<<Gr, Bl>>>(src, src_dim, pool, pool_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
}
void cudaF_maxpool_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, 
				float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){

	_maxpool_backprop<<<Gr, Bl>>>(in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
}
void cudaF_maxpoolchannel_overlap_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpoolchannel_overlap_prop<<<Gr, Bl>>>(src, src_dim, pool, pool_dim, in_height_, in_width_, pool_channel_dim_);
}
void cudaF_maxpoolchannel_overlap_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, 
				float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpoolchannel_overlap_backprop<<<Gr, Bl>>>(in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_channel_dim_);
}
void cudaF_maxpoolchannel_overlap2D_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpoolchannel_overlap2D_prop<<<Gr, Bl>>>(src, src_dim, pool, pool_dim, in_height_, in_width_, pool_channel_dim_);
}
void cudaF_maxpoolchannel_overlap2D_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, 
				float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpoolchannel_overlap2D_backprop<<<Gr, Bl>>>(in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_channel_dim_);
}
void cudaF_mod_permute_channels(dim3 Gr, dim3 Bl, float *comp, MatrixDim comp_dim, float *container, MatrixDim container_dim, int comp_idx, int num_component, int in_height, int in_width, bool fromCompToContainer){
	_mod_permute_channels<<<Gr, Bl>>>(comp, comp_dim, container, container_dim, comp_idx, num_component, in_height, in_width, fromCompToContainer);
}

/*
* "double"
*/
void cudaD_span_row_to_convmat(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *span, MatrixDim span_dim, int in_height, int in_width, int in_channel, int kernel_height, int kernel_width, int row_offset){
	_span_row_to_convmat<<<Gr,Bl>>> (in, in_dim, span, span_dim, in_height, in_width, in_channel, kernel_height, kernel_width, row_offset);
}

void cudaD_convmat_to_out(dim3 Gr, dim3 Bl, const double *convMat, MatrixDim conv_dim, double *out, MatrixDim out_dim, int out_height, int out_width, int num_sample){
	_convmat_to_out<<<Gr,Bl>>>(convMat, conv_dim, out, out_dim, out_height, out_width, num_sample);
}

void cudaD_add_mat_rep_vec(dim3 Gr, dim3 Bl, const double *vec, int rep, double *out, MatrixDim out_dim){
	_add_mat_rep_vec<<<Gr, Bl>>>(vec, rep, out, out_dim);
}

void cudaD_flip_mat(dim3 Gr, dim3 Bl, const double *orig, MatrixDim orig_dim, int kernel_height, int kernel_width, int group, double *flip, MatrixDim flip_dim){
	_flip_mat<<<Gr, Bl>>>(orig, orig_dim, kernel_height, kernel_width, group, flip, flip_dim);
}

void cudaD_pad_zero(dim3 Gr, dim3 Bl, const double *orig, MatrixDim orig_dim, int orig_height, int orig_width, int kernel_height, int kernel_width, double *padmat, MatrixDim padmat_dim){
	_pad_zero<<<Gr, Bl>>>(orig, orig_dim, orig_height, orig_width, kernel_height, kernel_width, padmat, padmat_dim);
}

void cudaD_tp_block(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size){
    _tp_block<<<Gr, Bl>>>(in, in_dim, out, out_dim, block_size);
}

void cudaD_tp_inside_block(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size){
    _tp_inside_block<<<Gr, Bl>>>(in, in_dim, out, out_dim, block_size);
}

void cudaD_mod_permute_row(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size, int in_channel){
    _mod_permute_row<<<Gr, Bl>>>(in, in_dim, out, out_dim, block_size, in_channel);
}

void cudaD_copy_rows_at(dim3 Gr, dim3 Bl, const double *src, MatrixDim src_dim, double *dest, MatrixDim dest_dim, int row_offset){
    _copy_rows_at<<<Gr, Bl>>>(src, src_dim, dest, dest_dim, row_offset);
}

void cudaD_maxpool_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpool_prop<<<Gr, Bl>>>(src, src_dim, pool, pool_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
}
void cudaD_maxpool_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, 
				double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){

	_maxpool_backprop<<<Gr, Bl>>>(in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
}

void cudaD_maxpoolchannel_overlap_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpoolchannel_overlap_prop<<<Gr, Bl>>>(src, src_dim, pool, pool_dim, in_height_, in_width_, pool_channel_dim_);
}
void cudaD_maxpoolchannel_overlap_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, 
				double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpoolchannel_overlap_backprop<<<Gr, Bl>>>(in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_channel_dim_);
}
void cudaD_maxpoolchannel_overlap2D_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpoolchannel_overlap2D_prop<<<Gr, Bl>>>(src, src_dim, pool, pool_dim, in_height_, in_width_, pool_channel_dim_);
}
void cudaD_maxpoolchannel_overlap2D_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, 
				double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
	_maxpoolchannel_overlap2D_backprop<<<Gr, Bl>>>(in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_channel_dim_);
}

void cudaD_mod_permute_channels(dim3 Gr, dim3 Bl, double *comp, MatrixDim comp_dim, double *container, MatrixDim container_dim, int comp_idx, int num_component, int in_height, int in_width, bool fromCompToContainer){
	_mod_permute_channels<<<Gr, Bl>>>(comp, comp_dim, container, container_dim, comp_idx, num_component, in_height, in_width, fromCompToContainer);
}

