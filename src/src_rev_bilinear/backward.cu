#include <torch/script.h>
#include <cuda.h>
#include <cuda_runtime.h>


//// blockIdx.x: num_points
//// blockIdx.y: 4
//// threadIdx.x: batch_size

__global__ void cal_pc_grad_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_grid_value,   //// (batch, channel, grid_size, grid_size)
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grid_value,   //// (batch, channel, grid_size, grid_size)
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc, //// (batch, 2, num_points)
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> weight_sum, //// (batch, grid_size, grid_size)
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc_value, //// (batch, channel, num_points)
    const torch::PackedTensorAccessor<int32_t,3,torch::RestrictPtrTraits,size_t> pc_grid_index,   //// (batch, 6, num_points)
    const int grid_size,
    const int num_channel,
    torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> grad_pc   //// (batch, 2, num_points)
    )
{
  float dx=1.0/(grid_size-1);
  float x = pc[threadIdx.x][0][blockIdx.x];
  float y = pc[threadIdx.x][1][blockIdx.x];
  int cell_x0 = pc_grid_index[threadIdx.x][0][blockIdx.x], cell_x1 = pc_grid_index[threadIdx.x][1][blockIdx.x];
  int cell_y0 = pc_grid_index[threadIdx.x][2][blockIdx.x], cell_y1 = pc_grid_index[threadIdx.x][3][blockIdx.x];
  float x0=cell_x0*dx, x1=cell_x1*dx, y0=cell_y0*dx, y1=cell_y1*dx;
  if(x0<x && x1>x && y0<y && y1>y){
    int cell_x=cell_x0, cell_y=cell_y0;
    float w1=-(y1-y),w2=-(x1-x);
    switch(blockIdx.y){
    case 0: {cell_x=cell_x0; cell_y=cell_y0; w1=-(y1-y); w2=-(x1-x); break;}
    case 1: {cell_x=cell_x0; cell_y=cell_y1; w1=-(y-y0); w2= (x1-x); break;}
    case 2: {cell_x=cell_x1; cell_y=cell_y0; w1= (y1-y); w2=-(x-x0); break;}
    case 3: {cell_x=cell_x1; cell_y=cell_y1; w1= (y-y0); w2= (x-x0); break;}
    default:break;
    }

    float dLdw=0;
    for(int channel_i=0;channel_i<num_channel;channel_i++){
      dLdw += grad_grid_value[threadIdx.x][channel_i][cell_x][cell_y] * (pc_value[threadIdx.x][channel_i][blockIdx.x]-grid_value[threadIdx.x][channel_i][cell_x][cell_y]);
    }
    dLdw /= weight_sum[threadIdx.x][cell_x][cell_y];
    atomicAdd(&(grad_pc[threadIdx.x][0][blockIdx.x]), dLdw*w1);
    atomicAdd(&(grad_pc[threadIdx.x][1][blockIdx.x]), dLdw*w2);
  }
}


torch::Tensor cal_pc_grad(torch::Tensor grad_grid_value, torch::Tensor grid_value, torch::Tensor pc, torch::Tensor weight_sum,
                                       torch::Tensor pc_value, torch::Tensor pc_grid_index, int grid_size)
{
  int batch_size = pc.size(0);
  int num_points = pc.size(2);
  int num_channel = grad_grid_value.size(1);
  auto grad_pc = torch::zeros({batch_size, 2, num_points}).to(pc);
  //pc = (pc + 1) / 2;

  const int threads = batch_size;
  const dim3 blocks(num_points, 4);

  cal_pc_grad_kernel<<<blocks, threads>>>(
        grad_grid_value.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        grid_value.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        weight_sum.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        pc_value.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        pc_grid_index.packed_accessor<int32_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_size,
        num_channel,
        grad_pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>()
        );

  return grad_pc;
}


//// blockIdx.x: num_points
//// blockIdx.y: num_channel
//// threadIdx.x: batch_size
__global__ void cal_pc_value_grad_kernel(
  const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_grid_value,   //// (batch, channel, grid_size, grid_size)
  const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc, //// (batch, 2, num_points)
  const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> weight_sum, //// (batch, grid_size, grid_size)
  const torch::PackedTensorAccessor<int32_t,3,torch::RestrictPtrTraits,size_t> pc_grid_index,   //// (batch, 6, num_points)
  const int grid_size,
  torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> grad_pc_value   //// (batch, num_channel, num_points)
  )
{
  float dx=1.0/(grid_size-1);
  float x = pc[threadIdx.x][0][blockIdx.x];
  float y = pc[threadIdx.x][1][blockIdx.x];
  int cell_x0 = pc_grid_index[threadIdx.x][0][blockIdx.x], cell_x1 = pc_grid_index[threadIdx.x][1][blockIdx.x];
  int cell_y0 = pc_grid_index[threadIdx.x][2][blockIdx.x], cell_y1 = pc_grid_index[threadIdx.x][3][blockIdx.x];
  float x0=cell_x0*dx, x1=cell_x1*dx, y0=cell_y0*dx, y1=cell_y1*dx;
  if(x0<x && x1>x && y0<y && y1>y){

    float w00=(x1-x) * (y1-y)/ weight_sum[threadIdx.x][cell_x0][cell_y0];
    float w01=(x1-x) * (y-y0)/ weight_sum[threadIdx.x][cell_x0][cell_y1];
    float w10=(x-x0) * (y1-y)/ weight_sum[threadIdx.x][cell_x1][cell_y0];
    float w11=(x-x0) * (y-y0)/ weight_sum[threadIdx.x][cell_x1][cell_y1];

    grad_pc_value[threadIdx.x][blockIdx.y][blockIdx.x] = w00*grad_grid_value[threadIdx.x][blockIdx.y][cell_x0][cell_y0]
      + w01*grad_grid_value[threadIdx.x][blockIdx.y][cell_x0][cell_y1]
      + w10*grad_grid_value[threadIdx.x][blockIdx.y][cell_x1][cell_y0]
      + w11*grad_grid_value[threadIdx.x][blockIdx.y][cell_x1][cell_y1];
    
  }
}


torch::Tensor cal_pc_value_grad(torch::Tensor grad_grid_value, torch::Tensor pc, torch::Tensor weight_sum, torch::Tensor pc_grid_index, int grid_size)
{
  int batch_size = pc.size(0);
  int num_points = pc.size(2);
  int num_channel = grad_grid_value.size(1);
  auto grad_pc_value = torch::zeros({batch_size, num_channel, num_points}).to(pc);
  //pc = (pc + 1) / 2;
  
  const int threads = batch_size;
  const dim3 blocks(num_points, num_channel);
  
  cal_pc_value_grad_kernel<<<blocks, threads>>>(
    grad_grid_value.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
    pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
    weight_sum.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
    pc_grid_index.packed_accessor<int32_t,3,torch::RestrictPtrTraits,size_t>(),
    grid_size,
    grad_pc_value.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>()
                                                );

  return grad_pc_value;
}
