#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}



__global__ void cube_select_sift_kernel(int b, int n,float radius, const float*__restrict__ xyz, int*__restrict__ idx_out) {
    int batch_idx = blockIdx.x;
    xyz += batch_idx * n * 3;
    idx_out += batch_idx * n * 8;
    float temp_dist[8];
    float judge_dist = radius * radius;
    for(int i = threadIdx.x; i < n;i += blockDim.x) {
        float x = xyz[i * 3];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        for(int j = 0;j < 8;j ++) {
            temp_dist[j] = 1e8;
            idx_out[i * 8 + j] = i; // if not found, just return itself..
        }
        for(int j = 0;j < n;j ++) {
            if(i == j) continue;
            float tx = xyz[j * 3];
            float ty = xyz[j * 3 + 1];
            float tz = xyz[j * 3 + 2];
            float dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
            if(dist > judge_dist) continue;
            int _x = (tx > x);
            int _y = (ty > y);
            int _z = (tz > z);
            int temp_idx = _x * 4 + _y * 2 + _z;
            if(dist < temp_dist[temp_idx]) {
                idx_out[i * 8 + temp_idx] = j;
                temp_dist[temp_idx] = dist;
            }
        }
    }
}

void cube_select_sift_kernel_wrapper(int b, int n, float radius,
                                     const float *xyz, int *idx_out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cube_select_sift_kernel<<<b, opt_n_threads(8), 0, stream>>>(
      b, n, radius, xyz, idx_out);

  CUDA_CHECK_ERRORS();
}
