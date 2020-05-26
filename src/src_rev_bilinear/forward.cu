#include <torch/script.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

//// threadIdx.x: batch_size
//// blockIdx.x: num_points
//// blockIdx.y: 4

__global__ void cal_pc_grid_index_kernel(
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc,   //// (batch, 2, num_points)  assume 0<=pc<=1
    const int grid_size,
    torch::PackedTensorAccessor<int32_t,3,torch::RestrictPtrTraits,size_t> pc_grid_index   //// (batch, 4, num_points)
    )
{
  float dx=1.0/(grid_size-1);
  float x = pc[threadIdx.x][0][blockIdx.x];
  float y = pc[threadIdx.x][1][blockIdx.x];
  int cell_x0 = __float2int_rd(x/dx), cell_x1 = cell_x0+1;
  int cell_y0 = __float2int_rd(y/dx), cell_y1 = cell_y0+1;
  cell_x1 = cell_x1 > (grid_size-1) ? (grid_size-1) : cell_x1;
  cell_y1 = cell_y1 > (grid_size-1) ? (grid_size-1) : cell_y1;
  cell_x0 = max(cell_x0, 0);
  cell_y0 = max(cell_y0, 0);
  cell_x1 = min(cell_x1, grid_size-1);
  cell_y1 = min(cell_y1, grid_size-1);
  pc_grid_index[threadIdx.x][0][blockIdx.x] = cell_x0;
  pc_grid_index[threadIdx.x][1][blockIdx.x] = cell_x1;
  pc_grid_index[threadIdx.x][2][blockIdx.x] = cell_y0;
  pc_grid_index[threadIdx.x][3][blockIdx.x] = cell_y1;
}

torch::Tensor cal_pc_grid_index(torch::Tensor pc, int grid_size)
{
  int batch_size = pc.size(0);
  int num_points = pc.size(2);
  torch::Tensor pc_grid_index = torch::zeros({batch_size, 4, num_points}).to(pc).to(at::kInt);
  ///pc = (pc + 1) / 2;
  
  const int threads = batch_size;
  const dim3 blocks(num_points, 1);

  cal_pc_grid_index_kernel<<<blocks, threads>>>(
        pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        grid_size,
        pc_grid_index.packed_accessor<int32_t,3,torch::RestrictPtrTraits,size_t>()
        );

  return pc_grid_index;

}

//// blockIdx.x: num_points
//// threadIdx.x: batch_size

__global__ void cal_weight_sum_kernel(
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc,   //// (batch, 2, num_points)  assume 0<=pc<=1
    const torch::PackedTensorAccessor<int32_t,3,torch::RestrictPtrTraits,size_t> pc_grid_index,   //// (batch, 4, num_points)
    const int grid_size,
    torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> weight_sum   //// (batch, grid_size, grid_size)
    )
{
  float dx=1.0/(grid_size-1);
  float x = pc[threadIdx.x][0][blockIdx.x];
  float y = pc[threadIdx.x][1][blockIdx.x];
  int cell_x0 = pc_grid_index[threadIdx.x][0][blockIdx.x], cell_x1 = pc_grid_index[threadIdx.x][1][blockIdx.x];
  int cell_y0 = pc_grid_index[threadIdx.x][2][blockIdx.x], cell_y1 = pc_grid_index[threadIdx.x][3][blockIdx.x];
  float x0=cell_x0*dx, x1=cell_x1*dx, y0=cell_y0*dx, y1=cell_y1*dx;
  if(x0<x && x1>x && y0<y && y1>y){
  float w00=(x1-x) * (y1-y);
  float w01=(x1-x) * (y-y0);
  float w10=(x-x0) * (y1-y);
  float w11=(x-x0) * (y-y0);

  atomicAdd(&(weight_sum[threadIdx.x][cell_x0][cell_y0]), w00);
  atomicAdd(&(weight_sum[threadIdx.x][cell_x0][cell_y1]), w01);
  atomicAdd(&(weight_sum[threadIdx.x][cell_x1][cell_y0]), w10);
  atomicAdd(&(weight_sum[threadIdx.x][cell_x1][cell_y1]), w11);
  }
}

torch::Tensor cal_weight_sum(torch::Tensor pc, torch::Tensor pc_grid_index, int grid_size)
{
  int batch_size = pc.size(0);
  int num_points = pc.size(2);
  torch::Tensor weight_sum = torch::zeros({batch_size, grid_size, grid_size}, torch::dtype(torch::kFloat32)).cuda();
  ///pc = (pc + 1) / 2;

  const int threads = batch_size;
  const dim3 blocks(num_points, 1);

  cal_weight_sum_kernel<<<blocks, threads>>>(
        pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        pc_grid_index.packed_accessor<int32_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_size,
        weight_sum.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>()
        );

  return weight_sum;
}

//// blockIdx.x: num_points
//// blockIdx.y: num_channel
//// threadIdx.x: batch_size

__global__ void cal_grid_value_kernel(
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc,   //// (batch, 2, num_points)  assume 0<=pc<=1
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> pc_value, //// (batch, channel, num_points)
    const torch::PackedTensorAccessor<int32_t,3,torch::RestrictPtrTraits,size_t> pc_grid_index,   //// (batch, 4, num_points)
    const int grid_size,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grid_value   //// (batch, channel, grid_size, grid_size)
    )
{
  float dx=1.0/(grid_size-1);
  float x = pc[threadIdx.x][0][blockIdx.x];
  float y = pc[threadIdx.x][1][blockIdx.x];
  int cell_x0 = pc_grid_index[threadIdx.x][0][blockIdx.x], cell_x1 = pc_grid_index[threadIdx.x][1][blockIdx.x];
  int cell_y0 = pc_grid_index[threadIdx.x][2][blockIdx.x], cell_y1 = pc_grid_index[threadIdx.x][3][blockIdx.x];
  float x0=cell_x0*dx, x1=cell_x1*dx, y0=cell_y0*dx, y1=cell_y1*dx;
  if(x0<x && x1>x && y0<y && y1>y){
    float w00=(x1-x) * (y1-y);
    float w01=(x1-x) * (y-y0);
    float w10=(x-x0) * (y1-y);
    float w11=(x-x0) * (y-y0);
    
    float value = pc_value[threadIdx.x][blockIdx.y][blockIdx.x];
    atomicAdd(&(grid_value[threadIdx.x][blockIdx.y][cell_x0][cell_y0]), w00*value);
    atomicAdd(&(grid_value[threadIdx.x][blockIdx.y][cell_x0][cell_y1]), w01*value);
    atomicAdd(&(grid_value[threadIdx.x][blockIdx.y][cell_x1][cell_y0]), w10*value);
    atomicAdd(&(grid_value[threadIdx.x][blockIdx.y][cell_x1][cell_y1]), w11*value);
    // debug print
    //printf("-------------------------\n");
    //printf("x0,x1,x:%f,%f,%f\n",x0,x1,x);
    //printf("cell_x0,cell_x1:%d,%d\n",cell_x0,cell_x1);
    //printf("y0,y1,y:%f,%f,%f\n",y0,y1,y);
    //printf("cell_y0,cell_y1:%d,%d\n",cell_y0,cell_y1);
    //printf("w00,w01,w10,w11:%f,%f,%f,%f\n",w00,w01,w10,w11);
    //printf("value:%f\n",value);
  }
}


torch::Tensor cal_grid_value(torch::Tensor pc, torch::Tensor pc_value, torch::Tensor pc_grid_index, int grid_size)
{
  int batch_size = pc.size(0);
  int num_points = pc.size(2);
  int num_channel = pc_value.size(1);
  torch::Tensor grid_value = torch::zeros({batch_size, num_channel, grid_size, grid_size}).to(pc);
  ///pc = (pc + 1) / 2;

  const int threads = batch_size;
  const dim3 blocks(num_points, num_channel);

  cal_grid_value_kernel<<<blocks, threads>>>(
        pc.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        pc_value.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        pc_grid_index.packed_accessor<int32_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_size,
        grid_value.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>()
        );

  return grid_value;
}
