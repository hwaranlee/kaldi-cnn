// nnet2/nnet-component-test.cc

// Copyright 2014-2015 Hwaran Lee (Computational NeroSystems Labs, KAIST)

#include "nnet2/nnet-component.h"
#include "nnet0/nnet-conv.h"
#include "util/common-utils.h"
#include <unistd.h> // for sleep().
#include "matrix/matrix-functions.h"

using namespace kaldi;
using namespace kaldi::nnet2;

namespace cnsl {
	namespace nnet0 {

		void UnitTestGenericComponentInternal(const Component &component) {
			int32 input_dim = component.InputDim(),
				output_dim = component.OutputDim();

			KALDI_LOG << component.Info();

			CuVector<BaseFloat> objf_vec(output_dim); // objective function is linear function of output.
			objf_vec.SetRandn(); // set to Gaussian noise.

			int32 num_egs = 2; // 10 + rand() % 5;
            KALDI_LOG << "num_egs " << num_egs;

			int32 rand_seed = 0; //rand();

			CuMatrix<BaseFloat> input(num_egs, input_dim),
				output(num_egs, output_dim);
			input.SetRandn();

			
			RandomComponent *rand_component =
				const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
			if (rand_component != NULL) {
				srand(rand_seed);
				rand_component->ResetGenerator();
			}

			component.Propagate(input, 1, &output);
			{
				bool binary = (rand() % 2 == 0);
				Output ko("tmpf", binary);
				component.Write(ko.Stream(), binary);
			}
            KALDI_LOG << "Propagation done.\n";

			Component *component_copy;
			{
				bool binary_in = (rand() % 2 == 0);
				Input ki("tmpf", &binary_in);
				component_copy = Component::ReadNew(ki.Stream(), binary_in);
			}


			
			{ // Test backward derivative is correct.
				CuVector<BaseFloat> output_objfs(num_egs);
				output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec, 0.0);
				BaseFloat objf = output_objfs.Sum();

				CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
				for (int32 i = 0; i < output_deriv.NumRows(); i++)
					output_deriv.Row(i).CopyFromVec(objf_vec);

    			CuMatrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());

				CuMatrix<BaseFloat> empty_mat;
				CuMatrix<BaseFloat> &input_ref =
					(component_copy->BackpropNeedsInput() ? input : empty_mat),
					&output_ref =
					(component_copy->BackpropNeedsOutput() ? output : empty_mat);
				int32 num_chunks = 1;


				component_copy->Backprop(input_ref, output_ref,
					output_deriv, num_chunks, NULL, &input_deriv);

            KALDI_LOG << "Backprop done.\n";
			
				int32 num_ok = 0, num_bad = 0, num_tries = 1;
				KALDI_LOG << "Comparing feature gradients " << num_tries << " times.";
				for (int32 i = 0; i < num_tries; i++) {
					CuMatrix<BaseFloat> perturbed_input(input.NumRows(), input.NumCols());
					{
						RandomComponent *rand_component =
							const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
						if (rand_component != NULL) {
							srand(rand_seed);
							rand_component->ResetGenerator();
						}
					}        
					perturbed_input.SetRandn();
					perturbed_input.Scale(1.0e-05); // scale by a small amount so it's like a delta.
					BaseFloat predicted_difference = TraceMatMat(perturbed_input,
						input_deriv, kTrans);
					perturbed_input.AddMat(1.0, input); // now it's the input + a delta.
					{ // Compute objf with perturbed input and make sure it matches
						// prediction.
						CuMatrix<BaseFloat> perturbed_output(output.NumRows(), output.NumCols());
						{
							RandomComponent *rand_component =
								const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
							if (rand_component != NULL) {
								srand(rand_seed);
								rand_component->ResetGenerator();
							}
						}        
						component.Propagate(perturbed_input, 1, &perturbed_output);
						CuVector<BaseFloat> perturbed_output_objfs(num_egs);
						perturbed_output_objfs.AddMatVec(1.0, perturbed_output, kNoTrans,
							objf_vec, 0.0);
						BaseFloat perturbed_objf = perturbed_output_objfs.Sum(),
							observed_difference = perturbed_objf - objf;
                        
						KALDI_LOG << "Input gradients: comparing " << predicted_difference
							<< " and " << observed_difference;
						if (fabs(predicted_difference - observed_difference) >
							0.15 * fabs((predicted_difference + observed_difference)/2) &&
							fabs(predicted_difference - observed_difference) > 1.0e-06) {
								KALDI_WARN << "Bad difference!";
								num_bad++;
						} else {
							num_ok++;
						}
					}
				}
				KALDI_LOG << "Succeeded for " << num_ok << " out of " << num_tries
					<< " tries.";
				//KALDI_ASSERT(num_ok > num_bad && "Feature-derivative check failed");
			}


            
			UpdatableComponent *ucomponent =
				dynamic_cast<UpdatableComponent*>(component_copy);
			
			if (ucomponent != NULL) { // Test parameter derivative is correct.

				int32 num_ok = 0, num_bad = 0, num_tries = 1;
				KALDI_LOG << "Comparing model gradients " << num_tries << " times.";
				for (int32 i = 0; i < num_tries; i++) {    
					UpdatableComponent *perturbed_ucomponent =
						dynamic_cast<UpdatableComponent*>(ucomponent->Copy()),
						*gradient_ucomponent =
						dynamic_cast<UpdatableComponent*>(ucomponent->Copy());
					KALDI_ASSERT(perturbed_ucomponent != NULL);
					gradient_ucomponent->SetZero(true); // set params to zero and treat as gradient.
					BaseFloat perturb_stddev = 5.0e-05;
					perturbed_ucomponent->PerturbParams(perturb_stddev);

					CuVector<BaseFloat> output_objfs(num_egs);
					output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec, 0.0);
					BaseFloat objf = output_objfs.Sum();

					CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
					for (int32 i = 0; i < output_deriv.NumRows(); i++)
						output_deriv.Row(i).CopyFromVec(objf_vec);
					CuMatrix<BaseFloat> input_deriv; // (input.NumRows(), input.NumCols());

					int32 num_chunks = 1;

					// This will compute the parameter gradient.
					ucomponent->Backprop(input, output, output_deriv, num_chunks,
						gradient_ucomponent, &input_deriv);

					// Now compute the perturbed objf.
					BaseFloat objf_perturbed;
					{
						CuMatrix<BaseFloat> output_perturbed; // (num_egs, output_dim);
						{
							RandomComponent *rand_component =
								const_cast<RandomComponent*>(dynamic_cast<const RandomComponent*>(&component));
							if (rand_component != NULL) {
								srand(rand_seed);
								rand_component->ResetGenerator();
							}
						}        
						perturbed_ucomponent->Propagate(input, 1, &output_perturbed);
						CuVector<BaseFloat> output_objfs_perturbed(num_egs);
						output_objfs_perturbed.AddMatVec(1.0, output_perturbed,
							kNoTrans, objf_vec, 0.0);
						objf_perturbed = output_objfs_perturbed.Sum();
					}

					BaseFloat delta_objf_observed = objf_perturbed - objf,
						delta_objf_predicted = (perturbed_ucomponent->DotProduct(*gradient_ucomponent) -
						ucomponent->DotProduct(*gradient_ucomponent));

					KALDI_LOG << "Model gradients: comparing " << delta_objf_observed
						<< " and " << delta_objf_predicted;
					if (fabs(delta_objf_predicted - delta_objf_observed) >
						0.05 * (fabs(delta_objf_predicted + delta_objf_observed)/2) &&
						fabs(delta_objf_predicted - delta_objf_observed) > 1.0e-06) {
							KALDI_WARN << "Bad difference!";
							num_bad++;
					} else {
						num_ok++;
					}
					delete perturbed_ucomponent;
					delete gradient_ucomponent;
				}
				//KALDI_ASSERT(num_ok >= num_bad && "model-derivative check failed");
					

			
			}
			delete component_copy; // No longer needed.
			
			
		}


		void UnitTestConvolutionComponent(){
			BaseFloat learning_rate = 0.01,
				param_stddev = 0.1, bias_stddev = 1.0;
			//int32 input_dim = 5 + rand() % 10, output_dim = 5 + rand() % 10;


			/**/
			int32 in_height = 4 , in_width = 6, in_channels =2,
			kernel_height=40,  kernel_width=4,  stride=1,
			group=300,  out_height=in_height-kernel_height+1,  out_width=in_width-kernel_width+1;
		/**/

			/*
			int32 in_height = 40 , in_width = 300, in_channels =1,
				kernel_height=40,  kernel_width=6,  stride=1,
				group=300,  out_height=in_height-kernel_height+1,  out_width=in_width-kernel_width+1;
				*/

			/*
			ConvolutionComponent component;
			component.Init(learning_rate,
				in_height, in_width, in_channels,
				kernel_height,  kernel_width,  stride,
				group,  out_height,  out_width,
				param_stddev,  bias_stddev);
				*/

			int32 input_dim = in_height * in_width * in_channels, 
				pool_height_dim =2 , pool_width_dim =2 , pool_channel_dim=1,
				output_dim = input_dim / (pool_height_dim * pool_width_dim *pool_channel_dim);


			MaxpoolComponent component;
			component.Init(input_dim, output_dim, in_height, in_width, in_channels, pool_height_dim, pool_width_dim, pool_channel_dim);

			/*				} else {
			Matrix<BaseFloat> mat(output_dim + 1, input_dim);
			mat.SetRandn();
			mat.Scale(param_stddev);
			WriteKaldiObject(mat, "tmpf", true);
			sleep(1);
			component.Init(learning_rate, "tmpf");
			}
			*/
			UnitTestGenericComponentInternal(component);
			/*			
			{
			const char *str = "learning-rate=0.01 input-dim=10 output-dim=15 param-stddev=0.1";
			ConvolutionComponent component;
			component.InitFromString(str);
			UnitTestGenericComponentInternal(component);
			}
			*/

		}
	} // namespace nnet0
} // namespace cnsl


using namespace kaldi;
using namespace kaldi::nnet2;

int main() {

	for (int32 loop = 0; loop < 2; loop++) {

#if HAVE_CUDA == 1
		if (loop == 0)
			CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
		else
			CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif
		for (int32 i = 0; i < 1; i++) {
			cnsl::nnet0::UnitTestConvolutionComponent();
			//kaldi::nnet2::UnitTestConvolutionComponent();
			if (loop == 0) KALDI_LOG << "Tests without GPU use succeeded.\n";
			else KALDI_LOG << "Tests with GPU use (if available) succeeded.\n";
		}
	}

#if HAVE_CUDA == 1
	CuDevice::Instantiate().PrintProfile();
#endif
	return 0;
}

