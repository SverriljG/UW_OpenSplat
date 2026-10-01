#ifndef SMOOTH_DEPTH_LOSS_H
#define SMOOTH_DEPTH_LOSS_H

#include <torch/torch.h>

// Edge-aware total variation on rendered depth (SeaSplat eq. 8):
//   L_Zsmooth = sum_{i,j} ( exp(-∇x I) * |∇x Ẑ| + exp(-∇y I) * |∇y Ẑ| )
//
// Matches dxyang/seasplat deepseecolor/depth_losses.SmoothDepthLoss: channel-mean
// of signed RGB differences (not |∇I|), mean reduction over valid neighbors.
// rgb: [H,W,3] ground-truth image; depth: [H,W] rendered depth (gradients via depth).
torch::Tensor smoothDepthLoss(const torch::Tensor &rgb, const torch::Tensor &depth);

#endif
