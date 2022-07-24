#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
at::Tensor cube_select_sift(at::Tensor xyz, const float radius);
