#include "smooth_depth_loss.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace {

torch::Tensor neighborBothValid(const torch::Tensor &valid, int64_t dim){
    // valid: [H,W] bool. For diffs along dim, keep pairs where both sides are valid.
    return valid.slice(dim, 0, valid.size(dim) - 1) &
           valid.slice(dim, 1, valid.size(dim));
}

torch::Tensor maskedAbsMean(const torch::Tensor &values, const torch::Tensor &pairMask){
    // values: [H,W-1,1] or [H-1,W,1]; pairMask: matching [H,W-1] or [H-1,W]
    torch::Tensor m = pairMask.to(values.dtype()).unsqueeze(-1);
    torch::Tensor denom = m.sum().clamp_min(1.0f);
    return (values.abs() * m).sum() / denom;
}

} // namespace

torch::Tensor smoothDepthLoss(const torch::Tensor &rgb,
                              const torch::Tensor &depth,
                              const torch::Tensor &alpha,
                              float alphaThreshold){
    // Sanitize depth: NaN/Inf from α-normalization or empty tiles must not enter TV.
    torch::Tensor z = torch::nan_to_num(depth, /*nan=*/0.0f, /*posinf=*/0.0f, /*neginf=*/0.0f);
    z = torch::clamp_min(z, 0.0f).unsqueeze(-1); // [H,W,1]

    torch::Tensor depthDx = z.diff(1);
    torch::Tensor depthDy = z.diff(0);

    // Paper eq. 8 / edge-aware TV: weight by exp(-|∇I|), channel-mean of RGB diffs.
    torch::Tensor rgbDx = rgb.diff(1).mean(-1, true).abs();
    torch::Tensor rgbDy = rgb.diff(0).mean(-1, true).abs();

    // Clamp image grads so exp is well-behaved even if GT is outside [0,1].
    rgbDx = torch::clamp(rgbDx, 0.0f, 20.0f);
    rgbDy = torch::clamp(rgbDy, 0.0f, 20.0f);

    depthDx = depthDx * torch::exp(-rgbDx);
    depthDy = depthDy * torch::exp(-rgbDy);

    if (alpha.defined() && alpha.numel() == depth.numel()){
        torch::Tensor finite = torch::isfinite(depth);
        torch::Tensor covered = alpha.detach() > alphaThreshold;
        torch::Tensor valid = finite & covered;
        torch::Tensor lossX = maskedAbsMean(depthDx, neighborBothValid(valid, /*dim=*/1));
        torch::Tensor lossY = maskedAbsMean(depthDy, neighborBothValid(valid, /*dim=*/0));
        return lossX + lossY;
    }

    return depthDx.abs().mean() + depthDy.abs().mean();
}

bool smoothDepthLossSanityCheck(){
    torch::NoGradGuard noGrad;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Smooth ramp depth in [0,1] with a mild RGB edge — loss should be finite and small.
    const int64_t H = 32, W = 32;
    torch::Tensor rgb = torch::zeros({H, W, 3}, opts);
    rgb.index_put_({torch::indexing::Slice(), torch::indexing::Slice(W / 2, W), torch::indexing::Slice()}, 1.0f);

    torch::Tensor ys = torch::linspace(0.0f, 1.0f, H, opts).unsqueeze(1).expand({H, W});
    torch::Tensor depth = ys.clone();
    torch::Tensor alpha = torch::ones({H, W}, opts);

    torch::Tensor loss = smoothDepthLoss(rgb, depth, alpha);
    if (!torch::isfinite(loss).item<bool>()){
        std::cerr << "smoothDepthLossSanityCheck: non-finite loss on normalized depth" << std::endl;
        return false;
    }
    const float v = loss.item<float>();
    if (!(v >= 0.0f && v < 10.0f)){
        std::cerr << "smoothDepthLossSanityCheck: unexpected magnitude " << v
                  << " (expected O(1) on [0,1] depth)" << std::endl;
        return false;
    }

    // Metric-scale cliff without normalization would be huge; after caller-side
    // min-max normalize the same geometry must stay O(1).
    torch::Tensor metric = depth * 50.0f + 5.0f;
    torch::Tensor dMin = metric.min();
    torch::Tensor dMax = metric.max();
    torch::Tensor normed = (metric - dMin) / torch::clamp_min(dMax - dMin, 1e-3f);
    torch::Tensor lossNorm = smoothDepthLoss(rgb, normed, alpha);
    const float vn = lossNorm.item<float>();
    if (!std::isfinite(vn) || std::abs(vn - v) > 1e-3f){
        std::cerr << "smoothDepthLossSanityCheck: normalized metric depth mismatch "
                  << vn << " vs " << v << std::endl;
        return false;
    }

    // Low-alpha / NaN pixels must not poison the mean.
    torch::Tensor dirty = normed.clone();
    dirty.index_put_({0, 0}, std::numeric_limits<float>::quiet_NaN());
    torch::Tensor sparseAlpha = alpha.clone();
    sparseAlpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}, 0.0f);
    torch::Tensor lossMasked = smoothDepthLoss(rgb, dirty, sparseAlpha);
    if (!torch::isfinite(lossMasked).item<bool>()){
        std::cerr << "smoothDepthLossSanityCheck: mask failed to keep loss finite" << std::endl;
        return false;
    }

    std::cout << "smoothDepthLossSanityCheck: ok (loss=" << v << ")" << std::endl;
    return true;
}
