#ifndef SMOOTH_DEPTH_LOSS_H
#define SMOOTH_DEPTH_LOSS_H

#include <torch/torch.h>

// Edge-aware total variation on rendered depth (SeaSplat eq. 8):
//   L_Zsmooth = mean( exp(-|∇x I|) * |∇x Ẑ| + exp(-|∇y I|) * |∇y Ẑ| )
//
// Faithful to the paper / SeaSplat's commented abs-gradient form. The active
// Python uses signed channel-mean RGB diffs; that makes exp(-∇I) > 1 on
// darkening edges and is a worse edge weight. We use |∇I| so weights stay in
// (0, 1]. Depth should be min-max normalized to ~[0,1] before this (SeaSplat
// default norm_depth_max=True); metric depth makes λ≈2 dominate the loss.
//
// LibTorch note: Tensor::diff(n=1, dim=-1) — never call .diff(1) intending
// "along dim 1"; that diffs the trailing axis and breaks [H,W,1] depth.
//
// rgb:   [H,W,3] ground-truth image (edge weights; no grad needed)
// depth: [H,W]   rendered depth (gradients flow here)
// alpha: optional [H,W] coverage; when defined, pairs involving low-alpha or
//        non-finite depth are excluded from the mean (avoids 1/α cliffs).
torch::Tensor smoothDepthLoss(const torch::Tensor &rgb,
                              const torch::Tensor &depth,
                              const torch::Tensor &alpha = torch::Tensor(),
                              float alphaThreshold = 1e-3f);

// CPU self-check: finite loss, abs-edge weights ≤ 1, normalized depth ⇒ O(1).
// Returns true on success; prints diagnostics and returns false on failure.
bool smoothDepthLossSanityCheck();

#endif
