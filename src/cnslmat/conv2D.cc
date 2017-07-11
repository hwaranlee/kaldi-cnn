// cnslmat/conv2D.cc

// Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)

#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "cudamatrix/cu-matrix.h"

#include <sstream>
#include <iostream>
#include "base/kaldi-common.h"
#include "base/timer.h"
#include "util/text-utils.h"
#include "util/kaldi-io.h"
#include "itf/options-itf.h"

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-randkernels.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-block-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

#include "cnslmat/cnsl-cu-kernels.h"



namespace kaldi {

	/**
	add following function inside cudamatrix/cu-matrix.h so that it become a CUMatrixBase public member function  
	this matrix : row = num_chunks, col=in_height * in_width * in_channel
	**/

	template<typename Real>
	void CuMatrixBase<Real>::Conv2D(const CuMatrixBase<Real> &kernel,
		int32 in_height,
		int32 in_width,
		int32 in_channel,
		int32 kernel_height,
		int32 kernel_width,
		int32 group,
		CuMatrixBase<Real> *out,
		bool concat) const {
			//KALDI_LOG << NumCols() << " " << in_height<< " " <<in_width << " " << in_channel ;

			KALDI_ASSERT( NumCols() == in_height*in_width*in_channel);
			KALDI_ASSERT( kernel.NumCols() == group);
			KALDI_ASSERT( kernel.NumRows() == kernel_height*kernel_width*in_channel);

			int32 out_height = in_height - kernel_height +1,
				out_width = in_width - kernel_width +1;

			KALDI_ASSERT(out != NULL);
			//out->Resize(NumRows(), out_height*out_width*group, kSetZero);

			int32 span_height_org = out_height*out_width*NumRows(),
				span_height,
				span_width = kernel_height*kernel_width*in_channel;
			
			int32 max_row = (0.5*1024*1024*1024) /(span_width * sizeof(Real));   // assume maximum memroy = 0.5GB

#if HAVE_CUDA == 1
			size_t free_mem;
			cudaMemGetInfo(&free_mem, NULL);
			max_row = free_mem / (span_width * sizeof(Real)) * 0.5;   
#endif

			if(span_height_org <= max_row) max_row = span_height_org;

			int32 split = span_height_org / max_row;

			CuMatrix<BaseFloat> convMat(out_height*out_width*NumRows(), group);
			std::vector<MatrixIndexT> indices;

			for (int32 split_idx=0 ; split_idx < split +1; split_idx++) {

				// 1. this -> spanThis.. ( kind of im2col)

				if (split_idx < split)
					span_height = max_row;
				else
					span_height = span_height_org - max_row*split;

				if (span_height == 0) break;


				CuMatrix<BaseFloat> spanThis(span_height, span_width);
				int32 row_offset= split_idx*max_row;

#if HAVE_CUDA == 1
				if (CuDevice::Instantiate().Enabled()) {
					Timer tim;
					dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
					dim3 dimGrid(n_blocks(span_width, CU2DBLOCK), n_blocks(span_height, CU2DBLOCK));

					cnsl::cuda_span_row_to_convmat(dimGrid, dimBlock, this->data_, this->Dim(), spanThis.Data(), spanThis.Dim(),
						in_height, in_width, in_channel, kernel_height, kernel_width, row_offset);

					CU_SAFE_CALL(cudaGetLastError());

					CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

				} else
#endif
				{
					MatrixDim in_dim = this->Dim();			

					int32 kernelsize = kernel_height*kernel_width,
						q = in_height-kernel_height+1;

					for (int32 i=0; i < span_height; i++){
						for (int32 j= 0 ; j< span_width; j++){

							int32 i_offset = i + row_offset,
								Ir = i_offset % in_dim.rows, 
								I = i_offset / in_dim.rows,
								Jr = j % kernelsize,
								J = j / kernelsize,
								Q = I % q + I / q * in_height,
								P = (Jr % kernel_height) + (Jr/kernel_height) * in_height;	

							(spanThis)(i,j) = (*this)(Ir, Q+P+(J*in_height*in_width));
						}
					}
				}

				// 2. convMat = spanThis * kernel 

				CuMatrix<BaseFloat> convMat_tmp(span_height, group);
				convMat_tmp.AddMatMat(1.0, spanThis, kNoTrans, kernel, kNoTrans, 1.0);

#if HAVE_CUDA == 1
				if (CuDevice::Instantiate().Enabled()) {
					Timer tim;

					MatrixDim convMat_tmp_dim = convMat_tmp.Dim();

					dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
					dim3 dimGrid(n_blocks(convMat_tmp_dim.cols, CU2DBLOCK), n_blocks(convMat_tmp_dim.rows, CU2DBLOCK));

					cnsl::cuda_copy_rows_at(dimGrid, dimBlock, convMat_tmp.Data(), convMat_tmp.Dim(), convMat.Data(), convMat.Dim(), row_offset);

					CU_SAFE_CALL(cudaGetLastError());

					CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

				} else
#endif
				{

					MatrixDim convMat_tmp_dim = convMat_tmp.Dim();				

					for (int32 i=0; i < convMat_tmp_dim.rows; i++){
						for (int32 j= 0 ; j< convMat_tmp_dim.cols; j++){							
							(convMat)(i+row_offset,j) = (convMat_tmp)(i, j);
						}
					}
				}
			}


			// 3. convMat -> out (.. kind of col2im)
			if (concat){
#if HAVE_CUDA == 1
				if (CuDevice::Instantiate().Enabled()) {
					Timer tim;
					MatrixDim conv_dim = convMat.Dim();				

					dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
					dim3 dimGrid(n_blocks(conv_dim.cols, CU2DBLOCK), n_blocks(conv_dim.rows, CU2DBLOCK));

					cnsl::cuda_convmat_to_out(dimGrid, dimBlock, convMat.Data(), convMat.Dim(), out->Data(), out->Dim(), out_height, out_width, NumRows());

					CU_SAFE_CALL(cudaGetLastError());
					CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
				} else
#endif
				{				
					MatrixDim conv_dim = convMat.Dim();				

					for (int32 i=0; i <  conv_dim.rows; i++){
						for (int32 j= 0 ; j< conv_dim.cols; j++){
							int Ir = i % NumRows(), 
								I = i / NumRows();
							(*out)(Ir, (I + j *out_height*out_width)) = convMat(i, j);
						}
					}
				}
			}else{
				out->CopyFromMat(convMat);
			}
	}


	/**
	add following function inside cudamatrix/cu-matrix.h so that it become a CUMatrixBase public member function  

	//if vec = [1 2 3] and rep = 2 => vec2 = [ 1 1 2 2 3 3];
	//this = this * repmat(vec2, NumRows(), 1);

	void AddMatRepVec(const CuVectorBase<Real> &vec, int32 rep) const;

	**/
	template<typename Real>
	void CuMatrixBase<Real>::AddMatRepVec(const CuVectorBase<Real> &vec, int32 rep) const {

		KALDI_ASSERT( vec.Dim() * rep == this->NumCols());

#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

			cnsl::cuda_add_mat_rep_vec(dimGrid, dimBlock, vec.Data(), rep, this->data_, this->Dim());
			CU_SAFE_CALL(cudaGetLastError());

			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

		} else
#endif
		{
			for (int32 i=0; i < this->NumRows(); i++){
				for (int32 j= 0 ; j< this->NumCols(); j++){

					int group = j / rep,
						index= i * this->Stride() + j;
					this->data_[index] += vec(group);
				}
			}
		}

	};

	template<typename Real>
	void CuMatrixBase<Real>::FlipMat(int32 kernel_height, int32 kernel_width, int32 in_channel, int32 group, CuMatrix<Real> *flip) const {

		KALDI_ASSERT(NumRows()== (kernel_height*kernel_width*in_channel));

		KALDI_ASSERT(flip != NULL);

		if( (flip->NumRows() != (kernel_height*kernel_width*group )) || ( flip->NumCols() != in_channel)) {
			flip->Resize((kernel_height*kernel_width*group), in_channel, kSetZero);
		}


#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(flip->NumCols(), CU2DBLOCK), n_blocks(flip->NumRows(), CU2DBLOCK));

			cnsl::cuda_flip_mat(dimGrid, dimBlock, this->data_, this->Dim(), kernel_height, kernel_width, group, flip->Data(), flip->Dim());

			CU_SAFE_CALL(cudaGetLastError());
			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

		} else
#endif
		{
			int32 ksize = kernel_height * kernel_width;

			for (int32 i=0; i < flip->NumRows(); i++){
				for (int32 j= 0 ; j< flip->NumCols(); j++){

					int group_idx = i / ksize,
						p = (group_idx + 1)*ksize -1 - i,
						m = p+j*ksize,
						n = i/ksize;
					(*flip)(i, j) = (*this)(m, n);
				}

			}

		}


	};

	template<typename Real>
	void CuMatrixBase<Real>::PaddingZero(int32 orig_height, int32 orig_width, int32 orig_channel, int32 kernel_height, int32 kernel_width, CuMatrix<Real> *padmat) const {

		KALDI_ASSERT(NumCols()== (orig_height*orig_width*orig_channel));
		KALDI_ASSERT(padmat != NULL);

		int32 padmat_height = orig_height + 2*(kernel_height-1),
			padmat_width = orig_width + 2*(kernel_width-1);

		if(padmat->NumRows() != NumRows() || padmat->NumCols() != padmat_height*padmat_width*orig_channel)
			padmat->Resize(NumRows(), padmat_height*padmat_width*orig_channel, kSetZero);


#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(padmat->NumCols(), CU2DBLOCK), n_blocks(padmat->NumRows(), CU2DBLOCK));


			cnsl::cuda_pad_zero(dimGrid, dimBlock, this->data_, this->Dim(), orig_height, orig_width, kernel_height, kernel_width, padmat->Data(), padmat->Dim() );

			CU_SAFE_CALL(cudaGetLastError());
			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

		} else
#endif
		{

			int32 padmat_size = padmat_height*padmat_width;

			for (int32 i=0; i < padmat->NumRows(); i++){
				for (int32 j= 0 ; j< padmat->NumCols(); j++){

					int32 chan_idx = j / padmat_size,
						p = j % padmat_size,
						I = p % padmat_height,
						J = p/ padmat_height;

					if ( (kernel_height-1) <= I && I < (kernel_height+orig_height-1) && (kernel_width-1) <= J && J < (kernel_width+orig_width-1)){

						int32 m = I-kernel_height+1,
							n = J-kernel_width+1,
							idx = (n*orig_height + m) + chan_idx*(orig_height*orig_width);

						(*padmat)(i, j) = (*this)(i, idx);
					}
					else{
						(*padmat)(i, j) = 0;
					}

				}
			}
		}

	};



	template<typename Real>
	void CuMatrixBase<Real>::TpBlock(int32 in_channel, int32 block_size, CuMatrix<Real> *out) const {
		//block_size = in_height*in_width;


		KALDI_ASSERT(this->NumCols()== block_size * in_channel);

		KALDI_ASSERT(out != NULL);

		if( (out->NumRows() != in_channel) || ( out->NumCols() != NumRows() * block_size)) {
			out->Resize(in_channel, NumRows() * block_size, kSetZero);
		}


#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(out->NumCols(), CU2DBLOCK), n_blocks(out->NumRows(), CU2DBLOCK));

			cnsl::cuda_tp_block(dimGrid, dimBlock, this->data_, this->Dim(), out->Data(), out->Dim(), block_size);

			CU_SAFE_CALL(cudaGetLastError());
			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

		} else
#endif
		{
			for (int32 i=0; i < out->NumRows(); i++){
				for (int32 j= 0 ; j< out->NumCols(); j++){

					int32 row = j/block_size,
						col = i*block_size + j%block_size;

					(*out)(i, j) = (*this)(row, col);
				}
			}
		}
	};

	template<typename Real>
	void CuMatrixBase<Real>::TpInsideBlock(int32 group, int32 block_size, CuMatrix<Real> *out) const {
		//block_size = out_derive_height*out_derive_width;


		KALDI_ASSERT(this->NumCols()== block_size * group);

		KALDI_ASSERT(out != NULL);

		if( (out->NumRows() != block_size*NumRows()) || ( out->NumCols() != group)) {
			out->Resize(block_size*NumRows(), group, kSetZero);
		}


#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(out->NumCols(), CU2DBLOCK), n_blocks(out->NumRows(), CU2DBLOCK));

			cnsl::cuda_tp_inside_block(dimGrid, dimBlock, this->data_, this->Dim(), out->Data(), out->Dim(), block_size);

			CU_SAFE_CALL(cudaGetLastError());
			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

		} else
#endif
		{
			for (int32 i=0; i < out->NumRows(); i++){
				for (int32 j= 0 ; j< out->NumCols(); j++){

					int32 row = i/block_size,
						col = j*block_size + i%block_size;

					(*out)(i, j) = (*this)(row, col);
				}
			}
		}
	};


	template<typename Real>
	void CuMatrixBase<Real>::ModPermuteRow(int32 in_channel, int32 block_size, CuMatrix<Real> *out) const {
		//block_size = out_derive_height*out_derive_width;

		KALDI_ASSERT(out != NULL);
		if( (out->NumRows() != NumRows()) || ( out->NumCols() != NumCols())) {
			out->Resize(NumRows(), NumCols(), kSetZero);
		}


#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(out->NumCols(), CU2DBLOCK), n_blocks(out->NumRows(), CU2DBLOCK));

			cnsl::cuda_mod_permute_row(dimGrid, dimBlock, this->data_, this->Dim(), out->Data(), out->Dim(), block_size, in_channel);

			CU_SAFE_CALL(cudaGetLastError());
			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

		} else
#endif
		{
			for (int32 i=0; i < out->NumRows(); i++){
				for (int32 j= 0 ; j< out->NumCols(); j++){

					int32 chan_idx = i % in_channel, 
						pos_idx = i / in_channel;

					(*out)((chan_idx *block_size + pos_idx ), j) = (*this)(i, j);
				}
			}
		}
	};

	template<typename Real>
	void CuMatrixBase<Real>::Maxpool_prop(int32 in_height, int32 in_width, int32 pool_height_dim, int32 pool_width_dim, int32 pool_channel_dim, bool overlap, bool overlap2D, CuMatrixBase<Real> *out) const {
		// max pooling 'this' matrix
		KALDI_ASSERT(out != NULL);
		int32 output_dim = out->NumCols();

		/*
		if(~overlap){
			output_dim = NumCols() / (pool_height_dim * pool_width_dim * pool_channel_dim);
		} else {
			int32 in_channel = NumCols() / in_height / in_width;
			output_dim = in_height * in_width * (in_channel - pool_channel_dim +1);
		}*/
		
#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(out->NumCols(), CU2DBLOCK), n_blocks(out->NumRows(), CU2DBLOCK));

			if (overlap){
				cnsl::cuda_maxpoolchannel_overlap_prop( dimGrid, dimBlock, this->data_, this->Dim(), out->Data(), out->Dim(), in_height, in_width, pool_height_dim, pool_width_dim, pool_channel_dim);
			} else if (overlap2D){
				cnsl::cuda_maxpoolchannel_overlap2D_prop( dimGrid, dimBlock, this->data_, this->Dim(), out->Data(), out->Dim(), in_height, in_width, pool_height_dim, pool_width_dim, pool_channel_dim);
			} else {
				cnsl::cuda_maxpool_prop( dimGrid, dimBlock, this->data_, this->Dim(), out->Data(), out->Dim(), in_height, in_width, pool_height_dim, pool_width_dim, pool_channel_dim);
			}			

			CU_SAFE_CALL(cudaGetLastError());
			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
		} else
#endif
		{

			int32 out_height = in_height / pool_height_dim,
				out_width = in_width / pool_width_dim;


			if (overlap2D){
				MatrixIndexT out_channel = out_deriv_dim.cols / (out_height * out_width),
					out_2d_map = sqrt(out_channel), //2D-topological map size
					in_2d_map = out_2d_map + pool_channel_dim_ - 1;

				for (MatrixIndexT j = 0; j < output_dim; j++) {
					CuSubMatrix<BaseFloat> pool(out->ColRange(j, 1));
					pool.Set(-1e20);

					MatrixIndexT out_channel_idx = j / (out_height * out_width), 
						out_position_idx = j % (out_height * out_width),
						out_width_idx = out_position_idx / out_height,
						out_height_idx = out_position_idx % out_height;

					for (int cx = 0; cx  < pool_channel_dim_; cx++){
						for (int cy = 0; cy < pool_channel_dim_; cy++){

							int in_channel_2d_x = channel_2d_x + cx,
								in_channel_2d_y = channel_2d_y + cy,
								in_channel_idx = in_channel_2d_x * in_2d_map  + in_channel_2d_y;

							MatrixIndexT idx = ( in_channel_idx * in_height_ * in_width_ ) + out_position_idx;
							pool.Max(this->ColRange( idx , 1));
						}
					}
				}
			}

			else{
				for (MatrixIndexT j = 0; j < output_dim; j++) {
					CuSubMatrix<BaseFloat> pool(out->ColRange(j, 1));
					pool.Set(-1e20);
					MatrixIndexT out_channel_idx = j / (out_height * out_width), 
						out_position_idx = j % (out_height * out_width),
						out_width_idx = out_position_idx / out_height,
						out_height_idx = out_position_idx % out_height;

					MatrixIndexT startpoint;
					if (overlap){
						startpoint = ( out_channel_idx * in_height * in_width ) + ( out_width_idx * pool_width_dim * in_height) + out_height_idx * pool_height_dim; 
					} else{
						startpoint = ( out_channel_idx * pool_channel_dim * in_height * in_width ) + ( out_width_idx * pool_width_dim * in_height) + out_height_idx * pool_height_dim;  
					}

					for (int32 c = 0; c  < pool_channel_dim; c++){
						for (int32 w = 0; w < pool_width_dim; w++){
							for (int32 h = 0; h < pool_height_dim; h++){

								MatrixIndexT idx = h + w * in_height + c * in_height * in_width;
								pool.Max(this->ColRange( startpoint + idx , 1));
							}
						}
					}
				}
			}
		}
	};





	template<typename Real>
	void CuMatrixBase<Real>::Maxpool_backprop(const CuMatrixBase<Real> &out_value, const CuMatrixBase<Real> &out_deriv, CuMatrix<Real> *in_deriv,
												int32 in_height, int32 in_width, int32 pool_height_dim, int32 pool_width_dim, int32 pool_channel_dim, bool overlap, bool overlap2D) const {
		// 'this' matrix is in_val, 'dest' matrix is in_deriv(backpropagated error matrix)

		KALDI_ASSERT(in_deriv != NULL);
		if( (in_deriv->NumRows() != NumRows()) || ( in_deriv->NumCols() != NumCols())) {
			in_deriv->Resize(NumRows(), NumCols(), kSetZero);
		}


#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(out_deriv.NumCols(), CU2DBLOCK), n_blocks(out_deriv.NumRows(), CU2DBLOCK));
			if (overlap){
				KALDI_ASSERT(pool_height_dim == 1 && pool_width_dim == 1  );
				cnsl::cuda_maxpoolchannel_overlap_backprop(dimGrid, dimBlock, this->data_, this->Dim(), out_value.data_, out_value.Dim(), out_deriv.data_, out_deriv.Dim(), in_deriv->data_, in_deriv->Dim(), in_height, in_width, pool_height_dim, pool_width_dim, pool_channel_dim);
			} if (overlap2D){
				KALDI_ASSERT(pool_height_dim == 1 && pool_width_dim == 1  );
				cnsl::cuda_maxpoolchannel_overlap2D_backprop(dimGrid, dimBlock, this->data_, this->Dim(), out_value.data_, out_value.Dim(), out_deriv.data_, out_deriv.Dim(), in_deriv->data_, in_deriv->Dim(), in_height, in_width, pool_height_dim, pool_width_dim, pool_channel_dim);
			}else {
				cnsl::cuda_maxpool_backprop(dimGrid, dimBlock, this->data_, this->Dim(), out_value.data_, out_value.Dim(), out_deriv.data_, out_deriv.Dim(), in_deriv->data_, in_deriv->Dim(), in_height, in_width, pool_height_dim, pool_width_dim, pool_channel_dim);
			}

			CU_SAFE_CALL(cudaGetLastError());
			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

		} else
#endif
		{

			int32 output_dim = out_value.NumCols(),
				out_height = in_height / pool_height_dim,
				out_width = in_width / pool_width_dim;


			if (overlap2D){
				MatrixIndexT out_channel = out_deriv_dim.cols / (out_height * out_width),
					out_2d_map = sqrt(out_channel), //2D-topological map size
					in_2d_map = out_2d_map + pool_channel_dim_ - 1;

				for (MatrixIndexT j = 0; j < output_dim; j++) {
					CuSubMatrix<BaseFloat> pool(out->ColRange(j, 1));
					pool.Set(-1e20);

					MatrixIndexT out_channel_idx = j / (out_height * out_width), 
						out_position_idx = j % (out_height * out_width),
						out_width_idx = out_position_idx / out_height,
						out_height_idx = out_position_idx % out_height;

					for (int cx = 0; cx  < pool_channel_dim_; cx++){
						for (int cy = 0; cy < pool_channel_dim_; cy++){

							int in_channel_2d_x = channel_2d_x + cx,
								in_channel_2d_y = channel_2d_y + cy,
								in_channel_idx = in_channel_2d_x * in_2d_map  + in_channel_2d_y;

							MatrixIndexT idx = ( in_channel_idx * in_height_ * in_width_ ) + out_position_idx;
							CuSubMatrix<BaseFloat> in_i(this->ColRange( idx , 1));
							CuSubMatrix<BaseFloat> in_deriv_i(in_deriv->ColRange( idx , 1));
							CuMatrix<BaseFloat> out_deriv_j(out_deriv.ColRange(j, 1));

							// Only the pool-inputs with 'max-values' are used to back-propagate into,
							// the rest of derivatives is zeroed-out by a mask.
							CuMatrix<BaseFloat> mask;
							in_i.EqualElementMask(out_j, &mask);
							out_deriv_j.MulElements(mask);
							in_deriv_i.AddMat(1.0, out_deriv_j); 
						}
					}
				}
			}
			else{
				if (overlap){
					KALDI_ASSERT(pool_height_dim == 1 && pool_width_dim == 1  );
				}


				for (MatrixIndexT j = 0; j < output_dim; j++) {
					CuSubMatrix<BaseFloat> out_j(out_value.ColRange(j, 1));

					MatrixIndexT out_channel_idx = j / (out_height * out_width), 
						out_position_idx = j % (out_height * out_width),
						out_width_idx = out_position_idx / out_height,
						out_height_idx = out_position_idx % out_height;

					MatrixIndexT startpoint;
					if (overlap){
						startpoint = ( out_channel_idx * in_height * in_width ) + ( out_width_idx * pool_width_dim * in_height) + out_height_idx * pool_height_dim;  
					} else {
						startpoint = ( out_channel_idx * pool_channel_dim * in_height * in_width ) + ( out_width_idx * pool_width_dim * in_height) + out_height_idx * pool_height_dim;  
					}

					for (int32 c = 0; c  < pool_channel_dim; c++){
						for (int32 w = 0; w < pool_width_dim; w++){
							for (int32 h = 0; h < pool_height_dim; h++){

								MatrixIndexT idx = h + w * in_height + c * in_height * in_width;

								CuSubMatrix<BaseFloat> in_i(this->ColRange( startpoint + idx , 1));
								CuSubMatrix<BaseFloat> in_deriv_i(in_deriv->ColRange( startpoint + idx , 1));
								CuMatrix<BaseFloat> out_deriv_j(out_deriv.ColRange(j, 1));

								// Only the pool-inputs with 'max-values' are used to back-propagate into,
								// the rest of derivatives is zeroed-out by a mask.
								CuMatrix<BaseFloat> mask;
								in_i.EqualElementMask(out_j, &mask);
								out_deriv_j.MulElements(mask);
								in_deriv_i.AddMat(1.0, out_deriv_j); 

							}
						}
					}
				}
			}
				}

	};

	
	template<typename Real>
	void CuMatrixBase<Real>::ModPermuteChannel(int32 comp_idx, int32 num_component, int32 in_height, int32 in_width, CuMatrixBase<Real> *container, bool fromCompToContainer) {

		KALDI_ASSERT(container != NULL);
/*		if( (out->NumRows() != NumRows()) || ( out->NumCols() != NumCols())) {
			out->Resize(NumRows(), NumCols(), kSetZero);
		}
		*/

#if HAVE_CUDA == 1
		if (CuDevice::Instantiate().Enabled()) {
			Timer tim;
			dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
			dim3 dimGrid(n_blocks(this->NumCols(), CU2DBLOCK), n_blocks(this->NumRows(), CU2DBLOCK));

			cnsl::cuda_mod_permute_channels(dimGrid, dimBlock, this->data_, this->Dim(), container->Data(), container->Dim(), comp_idx, num_component, in_height, in_width, fromCompToContainer);

			CU_SAFE_CALL(cudaGetLastError());
			CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

		} else
#endif
		{
			for (int32 i=0; i < this->NumRows(); i++){
				for (int32 j= 0 ; j< this->NumCols(); j++){

					int32 chan_idx = j / (in_height * in_width), 
						pos_idx = j % (in_height * in_width),
						//index = i * comp_dim.stride + j,
						out_chan_idx = chan_idx * num_component + comp_idx;
					//out_idx = i * container_dim.stride + ( out_chan_idx * (in_height * in_width) + pos_idx );

					if (fromCompToContainer){
						(*container)(i, ( out_chan_idx * (in_height * in_width) + pos_idx )) = (*this)(i, j);
					}else{
						(*this)(i, j) = (*container)(i, ( out_chan_idx * (in_height * in_width) + pos_idx ));
					}		
				}
			}
		}
	};
} // namespace kaldi


