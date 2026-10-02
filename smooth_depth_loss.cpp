#include "smooth_depth_loss.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace {

using torch::indexing::Slice;

// LibTorch Tensor::diff(n=1, dim=-1): the first positional arg is n, NOT dim.
// Always pass both explicitly for spatial gradients on [H,W,*] tensors.
torch::Tensor spatialDiff(const torch::Tensor &t, int64_t dim){
    return t.diff(/*n=*/1, /*dim=*/dim);
}

torch::Tensor neighborBothValid(const torch::Tensor &valid, int64_t dim){
    // valid: [H,W] bool. For diffs along dim, keep pairs where both sides are valid.
    const auto len = valid.size(dim) - 1;
    return valid.narrow(dim, 0, len) & valid.narrow(dim, 1, len);
}

torch::Tensor maskedAbsMean(const torch::Tensor &values, const torch::Tensor &pairMask){
    // values: [H,W-1,1] or [H-1,W,1]; pairMask: matching [H,W-1] or [H-1,W]
    TORCH_CHECK(values.dim() == 3 && pairMask.dim() == 2,
                "maskedAbsMean expects values [*,*,1] and pairMask [*,*]");
    TORCH_CHECK(values.size(0) == pairMask.size(0) && values.size(1) == pairMask.size(1),
                "maskedAbsMean shape mismatch: values ", values.sizes(),
                " vs pairMask ", pairMask.sizes());
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

    // Spatial diffs: dim 1 = x (width), dim 0 = y (height). Do not call .diff(1) —
    // that sets n=1 with dim=-1 and diffs the trailing channel axis instead.
    torch::Tensor depthDx = spatialDiff(z, /*dim=*/1); // [H,W-1,1]
    torch::Tensor depthDy = spatialDiff(z, /*dim=*/0); // [H-1,W,1]

    // Paper eq. 8 / edge-aware TV: weight by exp(-|∇I|), channel-mean of RGB diffs.
    torch::Tensor rgbDx = spatialDiff(rgb, /*dim=*/1).mean(-1, true).abs(); // [H,W-1,1]
    torch::Tensor rgbDy = spatialDiff(rgb, /*dim=*/0).mean(-1, true).abs(); // [H-1,W,1]

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
    rgb.index_put_({Slice(), Slice(W / 2, W), Slice()}, 1.0f);

    torch::Tensor ys = torch::linspace(0.0f, 1.0f, H, opts).unsqueeze(1).expand({H, W});
    torch::Tensor depth = ys.clone();
    torch::Tensor alpha = torch::ones({H, W}, opts);

    // Shape contract: spatial dx/dy must be W-1 / H-1 (guards the LibTorch diff pitfall).
    torch::Tensor zCheck = depth.unsqueeze(-1);
    if (spatialDiff(zCheck, 1).size(1) != W - 1 || spatialDiff(zCheck, 0).size(0) != H - 1){
        std::cerr << "smoothDepthLossSanityCheck: spatialDiff ranks wrong; "
                  << "dx=" << spatialDiff(zCheck, 1).sizes()
                  << " dy=" << spatialDiff(zCheck, 0).sizes() << std::endl;
        return false;
    }

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
    sparseAlpha.index_put_({Slice(), Slice(0, 2)}, 0.0f);
    torch::Tensor lossMasked = smoothDepthLoss(rgb, dirty, sparseAlpha);
    if (!torch::isfinite(lossMasked).item<bool>()){
        std::cerr << "smoothDepthLossSanityCheck: mask failed to keep loss finite" << std::endl;
        return false;
    }

    std::cout << "smoothDepthLossSanityCheck: ok (loss=" << v << ")" << std::endl;
    return true;
}
