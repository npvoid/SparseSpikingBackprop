#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <THC/THCAtomics.cuh>

#include <cmath>
#include <limits>


namespace {
    template <typename scalar_t>
    __global__ void surr_grad_spike_kernel(
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> memth_out,
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_out,
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> ds_out
        ) {
      const int n = blockIdx.x * blockDim.x + threadIdx.x;
      if (n < ds_out.size(0)){
        const auto abs_val = 100.0 * abs(memth_out[n]) + 1.0;
        ds_out[n] = grad_out[n] / (abs_val*abs_val);
      }
    }

    template <typename scalar_t>
    __global__ void s3gd_w_backward_kernel(
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spk_trace,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> aout_b,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> aout_t,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> aout_i,
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> ds_out,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dW
        ) {
    //  const int i = blockIdx.x;  // Input neuron index
    //  const int n = blockIdx.y * blockDim.y + threadIdx.y;  // Active output neuron index
      const int i = blockIdx.y;  // Input neuron index
      const int n = blockIdx.x * blockDim.x + threadIdx.x;  // Active output neuron index
      if (n < ds_out.size(0)){
          const int b = aout_b[n];  // Batch
          const int t = aout_t[n];  // Timestep
          if (spk_trace[b][t][i]==0){
            return;
          }
          const int j = aout_i[n];  // Output neuron index
          const auto val = spk_trace[b][t][i] * ds_out[n];
          gpuAtomicAdd(&dW[i][j], val);
      }
    }

    template <typename scalar_t>
    __global__ void s3gd_s_backward_master_kernel(
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> ds_out,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> ts_out,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> bj_ends,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> aout_bj_freqs,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> ts_in,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> b_ends,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> ain_b_freqs,
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> alphas,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> ds,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> bM_freqs,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> b_in_unique_freqs,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> bj_out_unique_freqs,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> bj_out_unique,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> ain_bt_freqs,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> ain_bt_starts,
        const torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> ain_i,
        const int batch_size,
        const int nb_steps,
        const int nb_hidden,
        const int total_num_threads
//        torch::PackedTensorAccessor<int64_t,2,torch::RestrictPtrTraits,size_t> debug
        ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Batch and active input

        if (idx < total_num_threads){
            const int bj = bj_out_unique[idx];
            // b and j
            int b = bj / nb_hidden;
            int j = bj - b*nb_hidden;

            // Active output tensor start index
            int bj_out = bj_ends[bj] ; // Start counting from here back (COMPUTE)
            int nb_compute = aout_bj_freqs[bj];  // Count this number of times
            int t_out = bj_out==-1 ? -1 : ts_out[bj_out];  // Value of first t_out

            // Required input
            int b_in = b_ends[b] ;  // Start counting from here back (WRITE)
            int nb_writes = ain_b_freqs[b];
            int t_in = b_in==-1 ? -1 : ts_in[b_in]; // Value of first t_in

            // Vars
            float delta = 0.0;
            int t_last = t_in;  // Last time delta was updated (either t_in or t_out-1)
            int bt_tmp;
            int i;
            int i_start;
            int i_idx;
            float val;

//            int counter = 0;
            while (nb_writes>0){
                if (t_out==t_in+1){
//                    printf("%d : (b, j)=(%d, %d) EQ (t_in, t_out, t_last)=(%d, %d, %d) delta=%.4f \n",
//                           counter, b, j, t_in, t_out, t_last, delta);
                    // Update delta
                    delta = ds_out[bj_out] + delta*alphas[t_last-t_in];
                    t_last = t_in;
                    bj_out--;
                    nb_compute--;
                    t_out = nb_compute<=0 ? -1 : ts_out[bj_out];
                    // Write delta and store indices
                    if (delta!=0.0){
                        bt_tmp = b*nb_steps + t_in;
                        i_start = ain_bt_starts[bt_tmp];
                        for (i_idx=0;i_idx<ain_bt_freqs[bt_tmp];i_idx++){
                            i = ain_i[i_start+i_idx];
                            val = weight[i][j]* delta;
                            gpuAtomicAdd(&ds[b][t_in][i], val);
                        }
                    }

                    b_in--;
                    nb_writes--;
                    t_in = nb_writes<=0 ? -1 : ts_in[b_in];
                }
                else if(t_out>t_in+1){
                    // Update delta
                    delta = ds_out[bj_out] + delta*alphas[t_last-(t_out-1)];

                    t_last = t_out-1;
                    bj_out--;
                    nb_compute--;
                    t_out = nb_compute<=0 ? -1 : ts_out[bj_out];
                }
                else if(t_out<t_in+1){
                    // Update delta
                    delta = delta*alphas[t_last-t_in];
                    // Write delta and store index
                    if (delta!=0.0){
                        bt_tmp = b*nb_steps + t_in;
                        i_start = ain_bt_starts[bt_tmp];
                        for (i_idx=0;i_idx<ain_bt_freqs[bt_tmp];i_idx++){
                            i = ain_i[i_start+i_idx];
                            val = weight[i][j]* delta;
                            gpuAtomicAdd(&ds[b][t_in][i], val);
                        }
                    }

                    t_last = t_in;
                    b_in--;
                    nb_writes--;
                    t_in = nb_writes<=0 ? -1 : ts_in[b_in];
                }
//                counter++;
            }
        }
    }


} // namespace


// ================================================================================================================== //
// ================================================================================================================== //
// ================================================================================================================== //


torch::Tensor surr_grad_spike_cuda(
    torch::Tensor memth_out,
    torch::Tensor grad_out
    ) {
    auto ds_out = torch::zeros_like(memth_out);

    const int threads = 1024;
    const int blocks = (memth_out.size(0)+threads-1) / threads;

    AT_DISPATCH_FLOATING_TYPES(memth_out.type(), "surr_grad_spike_kernel", ([&] {
        surr_grad_spike_kernel<scalar_t><<<blocks, threads>>>(
        memth_out.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        grad_out.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        ds_out.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>()
        );
    }));

  return ds_out;
}

torch::Tensor s3gd_w_backward_cuda(torch::Tensor spk_trace,
                                 torch::Tensor aout_b,
                                 torch::Tensor aout_t,
                                 torch::Tensor aout_i,
                                 torch::Tensor ds_out,
                                 int nb_inputs,
                                 int nb_hidden){

    auto dW = torch::zeros({nb_inputs, nb_hidden}, torch::TensorOptions().device(torch::kCUDA));

    const int threads = 1024;
    const dim3 blocks((ds_out.size(0)+threads-1) / threads, nb_inputs) ;

    AT_DISPATCH_ALL_TYPES(ds_out.type(), "cuda_torch_test_abs", ([&] {
        s3gd_w_backward_kernel<scalar_t><<<blocks, threads>>>(
        spk_trace.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        aout_b.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        aout_t.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        aout_i.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        ds_out.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        dW.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
        );
    }));

    return dW;

}

torch::Tensor s3gd_s_backward_master_cuda(
                                   torch::Tensor ds_out,
                                   torch::Tensor ts_out,
                                   torch::Tensor bj_ends,
                                   torch::Tensor aout_bj_freqs,
                                   torch::Tensor ts_in,
                                   torch::Tensor b_ends,
                                   torch::Tensor ain_b_freqs,
                                   torch::Tensor alphas,
                                   torch::Tensor weight,
                                   torch::Tensor bM_freqs,
                                   torch::Tensor b_in_unique_freqs,
                                   torch::Tensor bj_out_unique_freqs,
                                   torch::Tensor bj_out_unique,
                                   torch::Tensor ain_bt_freqs,
                                   torch::Tensor ain_bt_starts,
                                   torch::Tensor ain_i,
                                   const int M,
                                   const int total_num_threads,  // len(bj_out_unique)
                                   const int batch_size,
                                   const int nb_steps,
                                   const int nb_inputs,
                                   const int nb_hidden){

    // Create output tensor of shape {batch_size, nb_steps, nb_inputs} filled with zeros
    auto ds = torch::zeros({batch_size, nb_steps, nb_inputs}, torch::TensorOptions().device(torch::kCUDA));

    const int threads = 1024;
    const int blocks = (total_num_threads + threads-1) / threads;

    AT_DISPATCH_ALL_TYPES(ds_out.type(), "cuda_torch_compute_deltas", ([&] {
        s3gd_s_backward_master_kernel<scalar_t><<<blocks, threads>>>(
        ds_out.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        ts_out.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        bj_ends.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        aout_bj_freqs.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        ts_in.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        b_ends.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        ain_b_freqs.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        alphas.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        ds.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        bM_freqs.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        b_in_unique_freqs.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        bj_out_unique_freqs.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        bj_out_unique.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        ain_bt_freqs.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        ain_bt_starts.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        ain_i.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        batch_size, nb_steps, nb_hidden, total_num_threads
        );
    }));
    return ds;
}

