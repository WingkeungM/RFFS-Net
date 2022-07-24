#include "ball_query.h"
#include "utils.h"

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);
void cube_select_sift_kernel_wrapper(int b, int n, float radius,
                                     const float *xyz, int *idx_out);

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.is_cuda()) {
    query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data_ptr<float>(),
                                    xyz.data_ptr<float>(), idx.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx;
}


at::Tensor cube_select_sift(at::Tensor xyz, const float radius) {
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(xyz);

  if (xyz.is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx_out =
      torch::zeros({xyz.size(0), xyz.size(1), 8},
                   at::device(xyz.device()).dtype(at::ScalarType::Int));

  if (xyz.is_cuda()) {
    cube_select_sift_kernel_wrapper(xyz.size(0), xyz.size(1), radius,
                                    xyz.data_ptr<float>(), idx_out.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx_out;
}
