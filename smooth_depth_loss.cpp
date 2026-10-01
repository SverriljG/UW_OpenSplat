#include "smooth_depth_loss.hpp"

torch::Tensor smoothDepthLoss(const torch::Tensor &rgb, const torch::Tensor &depth){
    torch::Tensor z = depth.unsqueeze(-1);

    torch::Tensor depthDx = z.diff(1);
    torch::Tensor depthDy = z.diff(0);

    torch::Tensor rgbDx = rgb.diff(1).mean(-1, true);
    torch::Tensor rgbDy = rgb.diff(0).mean(-1, true);

    depthDx = depthDx * torch::exp(-rgbDx);
    depthDy = depthDy * torch::exp(-rgbDy);

    return depthDx.abs().mean() + depthDy.abs().mean();
}
