#ifndef MEDIUM_H
#define MEDIUM_H

#include <string>
#include <torch/torch.h>

// Underwater image formation model, following SeaThru as used by SeaSplat:
//
//   beta_d(z) = a * exp(-b * z)                 range dependent attenuation coefficient
//   A(z)      = exp(-beta_d(z) * z)             attenuation of the direct signal
//   B(z)      = B_inf * (1 - exp(-beta_b * z))  backscatter accumulated along the ray
//   I(z)      = J * A(z) + B(z)                 observed underwater image
//
// J is the medium-free color produced by the rasterizer, so the gaussians keep storing
// true scene color while the loss is evaluated against the hazy photograph.
//
// All coefficients are per color channel, which is what makes the model wavelength
// dependent: red attenuates fastest, blue-green survives longest.
//
// Positive-constrained parameters go through softplus rather than SeaSplat's clamp/relu.
// Clamping zeroes the gradient as soon as a parameter turns negative, stranding it there
// permanently; softplus keeps every parameter recoverable.
struct MediumModel{
    MediumModel(const torch::Device &device, float lr, bool useResidual);
    ~MediumModel();

    // depth is [H,W] metric depth, clean is [H,W,3]. Both return [H,W,3].
    torch::Tensor attenuation(const torch::Tensor &depth) const;
    torch::Tensor backscatter(const torch::Tensor &depth) const;
    torch::Tensor compose(const torch::Tensor &clean, const torch::Tensor &depth) const;

    void step(); // steps the optimizer and clears the gradients

    // Physically meaningful values, i.e. after softplus/sigmoid
    torch::Tensor effectiveAttenCoef() const;
    torch::Tensor effectiveAttenDecay() const;
    torch::Tensor effectiveBsCoef() const;
    torch::Tensor effectiveBInf() const;
    std::string summary() const;

    torch::Tensor attenCoef;    // a       [1,1,3]
    torch::Tensor attenDecay;   // b       [1,1,3]
    torch::Tensor bsCoef;       // beta_b  [1,1,3]
    torch::Tensor bInf;         // B_inf   [1,1,3]
    torch::Tensor residualCoef; // beta_d' [1,1,3]
    torch::Tensor jPrime;       // J'      [1,1,3]

    bool useResidual;
    torch::Device device;
    torch::optim::Adam *opt = nullptr;
};

#endif
