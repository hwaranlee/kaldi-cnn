// cnslmat/cnsl-cu-kernels.h

// Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)

#ifndef CNSL_CNSLMAT_CNSL_CU_KERNELS_H_
#define CNSL_CNSLMAT_CNSL_CU_KERNELS_H_



#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "cudamatrix/cu-matrixdim.h"

//#include "cudamatrix/cu-kernels-ansi.h"
/*
* In this file are C++ templated wrappers 
* of the ANSI-C CUDA kernels
*/

using namespace kaldi;
extern "C" {

	// float
	void cudaF_span_row_to_convmat(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *span, MatrixDim span_dim,
		int in_height, int in_width, int in_channel, int kernel_height, int kernel_width, int row_offset);

	void cudaF_convmat_to_out(dim3 Gr, dim3 Bl, const float *convMat, MatrixDim conv_dim, float *out, MatrixDim out_dim, int out_height, int out_width, int num_sample);	
	void cudaF_add_mat_rep_vec(dim3 Gr, dim3 Bl, const float *vec, int rep, float *out, MatrixDim out_dim);
	void cudaF_flip_mat(dim3 Gr, dim3 Bl, const float *orig, MatrixDim orig_dim, int kernel_height, int kernel_width, int group, float *flip, MatrixDim flip_dim);
	void cudaF_pad_zero(dim3 Gr, dim3 Bl, const float *orig, MatrixDim orig_dim, int orig_height, int orig_width, int kernel_height, int kernel_width, float *padmat, MatrixDim padmat_dim);
	void cudaF_tp_block(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size);
	void cudaF_tp_inside_block(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size);
	void cudaF_mod_permute_row(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size, int in_channel);
	void cudaF_copy_rows_at(dim3 Gr, dim3 Bl, const float *src, MatrixDim src_dim, float *dest, MatrixDim dest_dim, int row_offset);
	void cudaF_maxpool_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaF_maxpool_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, 
		float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaF_maxpoolchannel_overlap_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaF_maxpoolchannel_overlap_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, 
		float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaF_maxpoolchannel_overlap2D_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaF_maxpoolchannel_overlap2D_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, 
		float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaF_mod_permute_channels(dim3 Gr, dim3 Bl, float *comp, MatrixDim comp_dim, float *container, MatrixDim container_dim, int comp_idx, int num_component, int in_height, int in_width, bool fromCompToContainer);

	// double
	void cudaD_span_row_to_convmat(dim3 Gr, dim3 Bl, const double *in,  MatrixDim in_dim, double *span, MatrixDim span_dim,
		int in_height, int in_width, int in_channel, int kernel_height, int kernel_width, int row_offset);

	void cudaD_convmat_to_out(dim3 Gr, dim3 Bl, const double *convMat, MatrixDim conv_dim, double *out, MatrixDim out_dim, int out_height, int out_width, int num_sample);
	void cudaD_add_mat_rep_vec(dim3 Gr, dim3 Bl, const double *vec, int rep, double *out, MatrixDim out_dim);
	void cudaD_flip_mat(dim3 Gr, dim3 Bl, const double *orig, MatrixDim orig_dim, int kernel_height, int kernel_width, int group, double *flip, MatrixDim flip_dim);
	void cudaD_pad_zero(dim3 Gr, dim3 Bl, const double *orig, MatrixDim orig_dim, int orig_height, int orig_width, int kernel_height, int kernel_width, double *padmat, MatrixDim padmat_dim);
	void cudaD_tp_block(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size);
	void cudaD_tp_inside_block(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size);
	void cudaD_mod_permute_row(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size, int in_channel);
	void cudaD_copy_rows_at(dim3 Gr, dim3 Bl, const double *src, MatrixDim src_dim, double *dest, MatrixDim dest_dim, int row_offset);
	void cudaD_maxpool_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaD_maxpool_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, 
		double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaD_maxpoolchannel_overlap_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaD_maxpoolchannel_overlap_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, 
		double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaD_maxpoolchannel_overlap2D_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaD_maxpoolchannel_overlap2D_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, 
		double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_);
	void cudaD_mod_permute_channels(dim3 Gr, dim3 Bl, double *comp, MatrixDim comp_dim, double *container, MatrixDim container_dim, int comp_idx, int num_component, int in_height, int in_width, bool fromCompToContainer);

}


namespace cnsl {


	/*
	* "float"
	*/
	inline void cuda_span_row_to_convmat(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *span, MatrixDim span_dim,
		int in_height, int in_width, int in_channel, int kernel_height, int kernel_width, int row_offset){
			cudaF_span_row_to_convmat(Gr, Bl, in, in_dim, span, span_dim,
				in_height, in_width, in_channel, kernel_height, kernel_width, row_offset);
	}

	inline void cuda_convmat_to_out(dim3 Gr, dim3 Bl, const float *convMat, MatrixDim conv_dim, float *out, MatrixDim out_dim, int out_height, int out_width, int num_sample){
		cudaF_convmat_to_out(Gr, Bl, convMat, conv_dim, out, out_dim, out_height, out_width, num_sample);
	}

	inline void cuda_add_mat_rep_vec(dim3 Gr, dim3 Bl, const float *vec, int rep, float *out, MatrixDim out_dim){	
		cudaF_add_mat_rep_vec(Gr, Bl, vec, rep, out, out_dim);
	};

	inline void cuda_flip_mat(dim3 Gr, dim3 Bl, const float *orig, MatrixDim orig_dim, int kernel_height, int kernel_width, int group, float *flip, MatrixDim flip_dim){
		cudaF_flip_mat(Gr, Bl, orig, orig_dim, kernel_height, kernel_width, group, flip, flip_dim);
	};

	inline void cuda_pad_zero(dim3 Gr, dim3 Bl, const float *orig, MatrixDim orig_dim, int orig_height, int orig_width, int kernel_height, int kernel_width, float *padmat, MatrixDim padmat_dim){
		cudaF_pad_zero(Gr, Bl, orig, orig_dim, orig_height, orig_width, kernel_height, kernel_width, padmat, padmat_dim);
	};
	inline void cuda_tp_block(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size){
		cudaF_tp_block(Gr, Bl, in, in_dim, out, out_dim, block_size);
	};
	inline void cuda_tp_inside_block(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size){
		cudaF_tp_inside_block(Gr, Bl, in, in_dim, out, out_dim, block_size);
	};
	inline void cuda_mod_permute_row(dim3 Gr, dim3 Bl, const float *in, MatrixDim in_dim, float *out, MatrixDim out_dim, int block_size, int in_channel){
		cudaF_mod_permute_row(Gr, Bl, in, in_dim, out, out_dim, block_size, in_channel);
	};

	inline void cuda_copy_rows_at(dim3 Gr, dim3 Bl, const float *src, MatrixDim src_dim, float *dest, MatrixDim dest_dim, int row_offset){
		cudaF_copy_rows_at(Gr, Bl, src, src_dim, dest, dest_dim, row_offset);
	};

	inline void cuda_maxpool_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaF_maxpool_prop(Gr, Bl, src, src_dim, pool, pool_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};

	inline void cuda_maxpool_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaF_maxpool_backprop(Gr, Bl, 	in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};
	inline void cuda_maxpoolchannel_overlap_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaF_maxpoolchannel_overlap_prop(Gr, Bl, src, src_dim, pool, pool_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};
	inline void cuda_maxpoolchannel_overlap_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaF_maxpoolchannel_overlap_backprop(Gr, Bl, 	in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};
	inline void cuda_maxpoolchannel_overlap2D_prop(dim3 Gr, dim3 Bl,const float *src, MatrixDim src_dim, float *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaF_maxpoolchannel_overlap2D_prop(Gr, Bl, src, src_dim, pool, pool_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};
	inline void cuda_maxpoolchannel_overlap2D_backprop(dim3 Gr, dim3 Bl, const float *in_val, MatrixDim in_val_dim, const float *out_val, MatrixDim out_val_dim, const float *out_deriv, MatrixDim out_deriv_dim, float *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaF_maxpoolchannel_overlap2D_backprop(Gr, Bl, 	in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};

	inline void cuda_mod_permute_channels(dim3 Gr, dim3 Bl, float *comp, MatrixDim comp_dim, float *container, MatrixDim container_dim, int comp_idx, int num_component, int in_height, int in_width, bool fromCompToContainer){
		cudaF_mod_permute_channels(Gr, Bl, comp, comp_dim, container, container_dim, comp_idx, num_component, in_height, in_width, fromCompToContainer);
	};

	/*
	* "double"
	*/
	inline void cuda_span_row_to_convmat(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *span, MatrixDim span_dim,
		int in_height, int in_width, int in_channel, int kernel_height, int kernel_width, int row_offset){
			cudaD_span_row_to_convmat(Gr, Bl, in, in_dim, span, span_dim,
				in_height, in_width, in_channel, kernel_height, kernel_width, row_offset);
	}

	inline void cuda_convmat_to_out(dim3 Gr, dim3 Bl, const double *convMat, MatrixDim conv_dim, double *out, MatrixDim out_dim, int out_height, int out_width, int num_sample){
		cudaD_convmat_to_out(Gr, Bl, convMat, conv_dim, out, out_dim, out_height, out_width, num_sample);
	}

	inline void cuda_add_mat_rep_vec(dim3 Gr, dim3 Bl, const double *vec, int rep, double *out, MatrixDim out_dim){	
		cudaD_add_mat_rep_vec(Gr, Bl, vec, rep, out, out_dim);
	};	

	inline void cuda_flip_mat(dim3 Gr, dim3 Bl, const double *orig, MatrixDim orig_dim, int kernel_height, int kernel_width, int group, double *flip, MatrixDim flip_dim){
		cudaD_flip_mat(Gr, Bl, orig, orig_dim, kernel_height, kernel_width, group, flip, flip_dim);
	};

	inline void cuda_pad_zero(dim3 Gr, dim3 Bl, const double *orig, MatrixDim orig_dim, int orig_height, int orig_width, int kernel_height, int kernel_width, double *padmat, MatrixDim padmat_dim){
		cudaD_pad_zero(Gr, Bl, orig, orig_dim, orig_height, orig_width, kernel_height, kernel_width, padmat, padmat_dim);
	};
	inline void cuda_tp_block(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size){
		cudaD_tp_block(Gr, Bl, in, in_dim, out, out_dim, block_size);
	};
	inline void cuda_tp_inside_block(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size){
		cudaD_tp_inside_block(Gr, Bl, in, in_dim, out, out_dim, block_size);
	};
	inline void cuda_mod_permute_row(dim3 Gr, dim3 Bl, const double *in, MatrixDim in_dim, double *out, MatrixDim out_dim, int block_size, int in_channel){
		cudaD_mod_permute_row(Gr, Bl, in, in_dim, out, out_dim, block_size, in_channel);
	};
	inline void cuda_copy_rows_at(dim3 Gr, dim3 Bl, const double *src, MatrixDim src_dim, double *dest, MatrixDim dest_dim, int row_offset){
		cudaD_copy_rows_at(Gr, Bl, src, src_dim, dest, dest_dim, row_offset);
	};
	inline void cuda_maxpool_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaD_maxpool_prop(Gr, Bl, src, src_dim, pool, pool_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};

	inline void cuda_maxpool_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, 
		double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
			cudaD_maxpool_backprop(Gr, Bl, 	in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_	);
	};
	inline void cuda_maxpoolchannel_overlap_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaD_maxpoolchannel_overlap_prop(Gr, Bl, src, src_dim, pool, pool_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};
	inline void cuda_maxpoolchannel_overlap_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaD_maxpoolchannel_overlap_backprop(Gr, Bl, 	in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};
	inline void cuda_maxpoolchannel_overlap2D_prop(dim3 Gr, dim3 Bl,const double *src, MatrixDim src_dim, double *pool, MatrixDim pool_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaD_maxpoolchannel_overlap2D_prop(Gr, Bl, src, src_dim, pool, pool_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};
	inline void cuda_maxpoolchannel_overlap2D_backprop(dim3 Gr, dim3 Bl, const double *in_val, MatrixDim in_val_dim, const double *out_val, MatrixDim out_val_dim, const double *out_deriv, MatrixDim out_deriv_dim, double *dest, MatrixDim dest_dim, int in_height_, int in_width_, int pool_height_dim_, int pool_width_dim_, int pool_channel_dim_){
		cudaD_maxpoolchannel_overlap2D_backprop(Gr, Bl, 	in_val, in_val_dim, out_val, out_val_dim, out_deriv, out_deriv_dim, dest, dest_dim, in_height_, in_width_, pool_height_dim_, pool_width_dim_, pool_channel_dim_);
	};
	inline void cuda_mod_permute_channels(dim3 Gr, dim3 Bl, double *comp, MatrixDim comp_dim, double *container, MatrixDim container_dim, int comp_idx, int num_component, int in_height, int in_width, bool fromCompToContainer){
		cudaD_mod_permute_channels(Gr, Bl, comp, comp_dim, container, container_dim, comp_idx, num_component, in_height, in_width, fromCompToContainer);
	};
} // namespace cnsl



#endif // HAVE_CUDA

#endif
