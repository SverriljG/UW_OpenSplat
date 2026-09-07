#include <iomanip>
#include <sstream>
#include "medium.hpp"

MediumModel::MediumModel(const torch::Device &device, float lr, bool useResidual) :
    useResidual(useResidual), device(device){

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // SeaSplat initializes every coefficient uniformly in [0,1]
    attenCoef = torch::rand({1, 1, 3}, opts).requires_grad_();
    attenDecay = torch::rand({1, 1, 3}, opts).requires_grad_();
    bsCoef = torch::rand({1, 1, 3}, opts).requires_grad_();
    bInf = torch::rand({1, 1, 3}, opts).requires_grad_();
    residualCoef = torch::rand({1, 1, 3}, opts).requires_grad_();
    jPrime = torch::rand({1, 1, 3}, opts).requires_grad_();

    std::vector<torch::Tensor> params = { attenCoef, attenDecay, bsCoef, bInf };
    if (useResidual){
        params.push_back(residualCoef);
        params.push_back(jPrime);
    }

    opt = new torch::optim::Adam(params, torch::optim::AdamOptions(lr));
}

MediumModel::~MediumModel(){
    if (opt != nullptr){
        delete opt;
        opt = nullptr;
    }
}

torch::Tensor MediumModel::effectiveAttenCoef() const { return torch::softplus(attenCoef); }
torch::Tensor MediumModel::effectiveAttenDecay() const { return torch::softplus(attenDecay); }
torch::Tensor MediumModel::effectiveBsCoef() const { return torch::softplus(bsCoef); }
torch::Tensor MediumModel::effectiveBInf() const { return torch::sigmoid(bInf); }

torch::Tensor MediumModel::attenuation(const torch::Tensor &depth) const{
    torch::Tensor z = depth.unsqueeze(-1);
    torch::Tensor betaD = effectiveAttenCoef() * torch::exp(-effectiveAttenDecay() * z);
    return torch::exp(-betaD * z);
}

torch::Tensor MediumModel::backscatter(const torch::Tensor &depth) const{
    torch::Tensor z = depth.unsqueeze(-1);
    torch::Tensor bs = effectiveBInf() * (1.0f - torch::exp(-effectiveBsCoef() * z));
    if (useResidual){
        bs = bs + torch::sigmoid(jPrime) * torch::exp(-torch::softplus(residualCoef) * z);
    }
    return bs;
}

torch::Tensor MediumModel::compose(const torch::Tensor &clean, const torch::Tensor &depth) const{
    return torch::clamp(clean * attenuation(depth) + backscatter(depth), 0.0f, 1.0f);
}

void MediumModel::step(){
    opt->step();
    opt->zero_grad();
}

static std::string formatRgb(const torch::Tensor &t){
    torch::Tensor c = t.detach().to(torch::kCPU).reshape({3}).contiguous();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4)
       << "R " << c[0].item<float>()
       << "  G " << c[1].item<float>()
       << "  B " << c[2].item<float>();
    return ss.str();
}

std::string MediumModel::summary() const{
    std::ostringstream ss;
    ss << "Learned medium parameters" << std::endl
       << "  attenuation a      : " << formatRgb(effectiveAttenCoef()) << std::endl
       << "  attenuation b      : " << formatRgb(effectiveAttenDecay()) << std::endl
       << "  backscatter beta_b : " << formatRgb(effectiveBsCoef()) << std::endl
       << "  backscatter B_inf  : " << formatRgb(effectiveBInf());
    if (useResidual){
        ss << std::endl
           << "  residual beta_d'   : " << formatRgb(torch::softplus(residualCoef)) << std::endl
           << "  residual J'        : " << formatRgb(torch::sigmoid(jPrime));
    }
    return ss.str();
}
