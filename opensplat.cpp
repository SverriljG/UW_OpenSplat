#include <filesystem>
#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "medium.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include "zip_utils.hpp"
#include <cxxopts.hpp>

#ifdef USE_VISUALIZATION
#include "visualizer.hpp"
#endif

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene (default: splat.ply next to the input)", cxxopts::value<std::string>()->default_value("splat.ply"))
        ("output-cameras", "Path where to save a cameras JSON file", cxxopts::value<std::string>()->default_value(""))
        ("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value("-1"))
        ("resume", "Resume training from this PLY file", cxxopts::value<std::string>()->default_value(""))
        ("val", "Withhold a camera shot for validating the scene loss")
        ("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value("random"))
        ("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(""))
        ("depth-render", "Path of the directory where to write rendered depth maps for every camera", cxxopts::value<std::string>()->default_value(""))
        ("depth-only", "Only render depth maps (requires --resume and --depth-render), then exit without training", cxxopts::value<bool>()->default_value("false"))

        ("underwater", "Train through an underwater image formation model (attenuation + backscatter). Gaussians keep medium-free color while the loss is scored against the hazy photograph", cxxopts::value<bool>()->default_value("false"))
        ("medium-from-iter", "Enable the underwater model after these many steps, giving geometry time to settle first (-1 = half of num-iters)", cxxopts::value<int>()->default_value("-1"))
        ("medium-lr", "Learning rate for the underwater medium parameters", cxxopts::value<float>()->default_value("0.01"))
        ("medium-residual", "Include the SeaThru residual backscatter term", cxxopts::value<bool>()->default_value("false"))
        ("medium-normalize-depth", "Min-max normalize depth per image before the medium model (reproduces SeaSplat, but discards metric range)", cxxopts::value<bool>()->default_value("false"))
        ("medium-detach-depth", "Do not backpropagate the medium loss into gaussian positions through depth (cheaper, and an ablation of the geometry claim)", cxxopts::value<bool>()->default_value("false"))
        ("medium-grayworld", "Weight of a gray-world prior on the recovered color, which discourages the medium from explaining the whole image", cxxopts::value<float>()->default_value("0.0"))
        ("center", "Center the model at the origin")
        ("cpu", "Force CPU execution")
        
        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        ("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value("1"))
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("0"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("refine-every", "Densify/prune gaussians every these many steps", cxxopts::value<int>()->default_value("500"))
        ("densify-from", "Start densifying gaussians after these many steps", cxxopts::value<int>()->default_value("500"))
        ("densify-until", "Stop densifying gaussians after these many steps (-1 = min(15000, half of num-iters))", cxxopts::value<int>()->default_value("-1"))
        ("loss-thresh", "High-error pixel threshold on the normalized L1 map for multi-view scoring", cxxopts::value<float>()->default_value("0.1"))
        ("no-edge-guidance", "Disable Canny edge weighting of the densification importance", cxxopts::value<bool>()->default_value("false"))
        ("max-gaussians", "Maximum number of gaussians (0 = unlimited)", cxxopts::value<int>()->default_value("5000000"))
        ("no-masks", "Ignore image masks even when present", cxxopts::value<bool>()->default_value("false"))
        ("no-gpu-cache", "Do not cache images/masks on the GPU (reduces VRAM usage, slower)", cxxopts::value<bool>()->default_value("false"))
#ifdef USE_VISUALIZATION
        ("has-visualization", "Show the visualization steps of training", cxxopts::value<bool>()->default_value("0"))
#endif
        ("h,help", "Print usage")
        ("version", "Print version")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[colmap/nerfstudio/opensfm/odx/openmvg project path or .zip archive]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("version")){
        std::cout << APP_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }


    const std::string projectRoot = result["input"].as<std::string>();
    std::string outputScene = result["output"].as<std::string>();
    if (result.count("output") == 0){
        // Output next to the input scene, for a .zip, next to the archive
        fs::path in = fs::absolute(fs::path(projectRoot));
        if (in.filename().empty()) in = in.parent_path();
        outputScene = (in.parent_path() / outputScene).string();
    }
    const std::string outputCameras = result["output-cameras"].as<std::string>();
    const int saveEvery = result["save-every"].as<int>();
    const std::string resume = result["resume"].as<std::string>();
    const bool validate = result.count("val") > 0 || result.count("val-render") > 0;
    const std::string valImage = result["val-image"].as<std::string>();
    const std::string valRender = result["val-render"].as<std::string>();
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    const std::string depthRender = result["depth-render"].as<std::string>();
    const bool depthOnly = result["depth-only"].as<bool>();
    if (!depthRender.empty() && !fs::exists(depthRender)) fs::create_directories(depthRender);
    if (depthOnly && depthRender.empty()){
        std::cerr << "--depth-only requires --depth-render <directory>" << std::endl;
        return EXIT_FAILURE;
    }
    if (depthOnly && result["resume"].as<std::string>().empty()){
        std::cerr << "--depth-only requires --resume <ply>" << std::endl;
        return EXIT_FAILURE;
    }
    const bool underwater = result["underwater"].as<bool>();
    int mediumFromIter = result["medium-from-iter"].as<int>();
    if (mediumFromIter < 0) mediumFromIter = result["num-iters"].as<int>() / 2;
    const float mediumLr = result["medium-lr"].as<float>();
    const bool mediumResidual = result["medium-residual"].as<bool>();
    const bool mediumNormalizeDepth = result["medium-normalize-depth"].as<bool>();
    const bool mediumDetachDepth = result["medium-detach-depth"].as<bool>();
    const float mediumGrayWorld = result["medium-grayworld"].as<float>();
    const bool keepCrs = result.count("center") == 0;
    const float downScaleFactor = (std::max)(result["downscale-factor"].as<float>(), 1.0f);
    const int numIters = result["num-iters"].as<int>();
    const int numDownscales = result["num-downscales"].as<int>();
    const int resolutionSchedule = result["resolution-schedule"].as<int>();
    const int shDegree = result["sh-degree"].as<int>();
    const int shDegreeInterval = result["sh-degree-interval"].as<int>();
    const float ssimWeight = result["ssim-weight"].as<float>();
    const int refineEvery = result["refine-every"].as<int>();
    const int densifyFrom = result["densify-from"].as<int>();
    int densifyUntil = result["densify-until"].as<int>();
    if (densifyUntil < 0) densifyUntil = (std::min)(15000, result["num-iters"].as<int>() / 2);
    const float lossThresh = result["loss-thresh"].as<float>();
    const int maxGaussians = result["max-gaussians"].as<int>();
    const bool noMasks = result["no-masks"].as<bool>();
    #ifdef USE_VISUALIZATION
        const bool hasVisualization = result["has-visualization"].as<bool>();
    #endif

    torch::Device device = torch::kCPU;
    int displayStep = 10;

    if (torch::hasCUDA() && result.count("cpu") == 0) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    } else if (torch::hasMPS() && result.count("cpu") == 0) {
        std::cout << "Using MPS" << std::endl;
        device = torch::kMPS;
    }else{
        std::cout << "Using CPU" << std::endl;
        displayStep = 1;
    }

#ifdef USE_VISUALIZATION
    Visualizer visualizer;
    if (hasVisualization)
        visualizer.Initialize(numIters);
#endif

    try{
        std::string projectPath = projectRoot;
        if (isZipArchive(projectRoot)) projectPath = extractZipToCache(projectRoot);
        InputData inputData = inputDataFromX(projectPath);

        int numMasks = 0;
        if (!noMasks){
            for (Camera &cam : inputData.cameras){
                cam.maskPath = findMaskPath(cam.filePath, projectPath);
                if (!cam.maskPath.empty()) numMasks++;
            }
        }
        if (numMasks > 0) std::cout << "Found " << numMasks << " masks" << std::endl;

        parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [&downScaleFactor](Camera &cam){
            cam.loadImage(downScaleFactor);
        });

        // Withhold a validation camera if necessary
        auto t = inputData.getCameras(validate, valImage);
        std::vector<Camera> cams = std::get<0>(t);
        Camera *valCam = std::get<1>(t);

        Model model(inputData,
                    cams.size(),
                    numDownscales, resolutionSchedule, shDegree, shDegreeInterval,
                    refineEvery, densifyFrom, densifyUntil, maxGaussians,
                    lossThresh,
                    numIters, keepCrs,
                    device);
        model.trainCams = &cams;
        model.edgeGuidance = !result["no-edge-guidance"].as<bool>();
        Camera::gpuCacheEnabled = !result["no-gpu-cache"].as<bool>();

        std::vector< size_t > camIndices( cams.size() );
        std::iota( camIndices.begin(), camIndices.end(), 0 );
        InfiniteRandomIterator<size_t> camsIter( camIndices );

        int imageSize = -1;
        size_t step = 1;

        if (resume != ""){
            step = model.loadPly(resume) + 1;
        }

        auto writeDepthMap = [&model, &device](Camera &c, int atStep, const std::string &dir){
            torch::NoGradGuard noGrad;
            torch::Tensor depth = model.renderDepth(c, atStep).detach().cpu().contiguous();
            const float dMin = depth.min().item<float>();
            const float dMax = depth.max().item<float>();
            torch::Tensor norm = (dMax - dMin) > 1e-8f ? (depth - dMin) / (dMax - dMin)
                                                       : torch::zeros_like(depth);
            torch::Tensor scaled = (norm * 255.0f).toType(torch::kU8).contiguous();

            cv::Mat gray(static_cast<int>(depth.size(0)), static_cast<int>(depth.size(1)), CV_8UC1);
            std::copy(scaled.data_ptr<uint8_t>(), scaled.data_ptr<uint8_t>() + scaled.numel(), gray.data);
            cv::Mat colored;
            cv::applyColorMap(gray, colored, cv::COLORMAP_JET);

            const std::string stem = fs::path(c.filePath).stem().string();
            cv::imwrite((fs::path(dir) / (stem + "_depth.png")).string(), colored);
            std::cout << stem << " depth range: " << dMin << " to " << dMax << std::endl;
        };

        std::unique_ptr<MediumModel> medium;
        if (underwater){
            if (device == torch::kCPU) throw std::runtime_error("--underwater requires a GPU backend (MPS/CUDA)");
            medium.reset(new MediumModel(device, mediumLr, mediumResidual));
            std::cout << "Underwater image formation model enabled from step " << mediumFromIter << std::endl;
        }

        // Renders the depth map the medium model consumes
        auto mediumDepth = [&model, mediumDetachDepth, mediumNormalizeDepth](Camera &c, int atStep){
            torch::Tensor depth;
            if (mediumDetachDepth){
                torch::NoGradGuard noGrad;
                depth = model.renderDepth(c, atStep).detach();
            }else{
                depth = model.renderDepth(c, atStep);
            }
            if (mediumNormalizeDepth){
                torch::Tensor dMin = depth.min().detach();
                torch::Tensor dMax = depth.max().detach();
                depth = (depth - dMin) / torch::clamp_min(dMax - dMin, 1e-6f);
            }
            return depth;
        };

        if (depthOnly){
            if (device == torch::kCPU) throw std::runtime_error("--depth-only requires a GPU backend (MPS/CUDA)");
            for (Camera &c : cams) writeDepthMap(c, numIters, depthRender);
            if (valCam != nullptr) writeDepthMap(*valCam, numIters, depthRender);
            std::cout << "Wrote depth maps to " << depthRender << std::endl;
            return EXIT_SUCCESS;
        }

        for (; step <= numIters; step++){
            Camera& cam = cams[ camsIter.next() ];

            torch::Tensor rgb = model.forward(cam, step);
            torch::Tensor gt = cam.getImageGpu(model.getDownscaleFactor(step), device);
            torch::Tensor mask = cam.getMaskGpu(model.getDownscaleFactor(step), device);

            // With the medium enabled the gaussians render medium-free color, which is
            // pushed through attenuation and backscatter before being compared to the photo
            const bool mediumActive = medium && static_cast<int>(step) > mediumFromIter;
            torch::Tensor rendered = mediumActive ? medium->compose(rgb, mediumDepth(cam, step))
                                                  : rgb;

            torch::Tensor mainLoss = model.mainLoss(rendered, gt, mask, ssimWeight);
            if (mediumActive && mediumGrayWorld > 0.0f){
                torch::Tensor channelMean = rgb.mean(std::vector<int64_t>{0, 1});
                mainLoss = mainLoss + mediumGrayWorld * (channelMean - channelMean.mean()).pow(2).sum();
            }
            mainLoss.backward();
            if (mediumActive) medium->step();

            if (step % displayStep == 0) {
                const float percentage = static_cast<float>(step) / numIters;
                std::cout << "Step " << step << ": " << mainLoss.item<float>() << " [" << floor(percentage * 100) << "%]" <<  std::endl;
            }

            model.afterTrain(step);
            model.optimizerStepCadence(step);
            model.schedulersStep(step);

            if (saveEvery > 0 && step % saveEvery == 0){
                fs::path p(outputScene);
                model.save(p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string(), step);
            }

            if (!valRender.empty() && step % 10 == 0){
                torch::Tensor rgb = model.forward(*valCam, step);
                cv::Mat image = tensorToImage(rgb.detach().cpu());
                cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                cv::imwrite((fs::path(valRender) / (std::to_string(step) + ".png")).string(), image);

                // Alongside the restored render, write what the medium turns it into
                if (mediumActive){
                    torch::NoGradGuard noGrad;
                    torch::Tensor uw = medium->compose(rgb.detach(), mediumDepth(*valCam, step));
                    cv::Mat uwImage = tensorToImage(uw.detach().cpu());
                    cv::cvtColor(uwImage, uwImage, cv::COLOR_RGB2BGR);
                    cv::imwrite((fs::path(valRender) / (std::to_string(step) + "_uw.png")).string(), uwImage);
                }
            }

#ifdef USE_VISUALIZATION
            if (hasVisualization) {
                visualizer.SetInitialGaussianNum(inputData.points.xyz.size(0));
                visualizer.SetLoss(step, mainLoss.item<float>());
                visualizer.SetGaussians(model.means, model.scales, model.featuresDc,
                                        model.opacities);
                visualizer.SetImage(rgb, gt);
                if (visualizer.QuitApp())
                    step = numIters + 1;
                visualizer.Draw();
            }
#endif
        }

        if (!outputCameras.empty()) inputData.saveCameras(outputCameras, keepCrs);
        model.save(outputScene, numIters);
        // model.saveDebugPly("debug.ply", numIters);

        if (!depthRender.empty() && device != torch::kCPU){
            for (Camera &c : cams) writeDepthMap(c, numIters, depthRender);
            if (valCam != nullptr) writeDepthMap(*valCam, numIters, depthRender);
        }

        if (medium){
            std::cout << medium->summary() << std::endl;

            auto toVec = [](const torch::Tensor &t){
                torch::Tensor c = t.detach().to(torch::kCPU).reshape({3}).contiguous();
                return std::vector<float>(c.data_ptr<float>(), c.data_ptr<float>() + 3);
            };
            nlohmann::json j;
            j["attenuation_a"] = toVec(medium->effectiveAttenCoef());
            j["attenuation_b"] = toVec(medium->effectiveAttenDecay());
            j["backscatter_beta"] = toVec(medium->effectiveBsCoef());
            j["backscatter_b_inf"] = toVec(medium->effectiveBInf());
            j["medium_from_iter"] = mediumFromIter;
            j["normalized_depth"] = mediumNormalizeDepth;
            j["detached_depth"] = mediumDetachDepth;

            fs::path mediumPath(outputScene);
            mediumPath.replace_extension(".medium.json");
            std::ofstream mediumOut(mediumPath.string());
            mediumOut << j.dump(4);
            mediumOut.close();
            std::cout << "Wrote " << mediumPath.string() << std::endl;
        }

        // Validate
        if (valCam != nullptr){
            torch::Tensor rgb = model.forward(*valCam, numIters);
            torch::Tensor gt = valCam->getImageGpu(model.getDownscaleFactor(numIters), device);
            torch::Tensor valMask = valCam->getMaskGpu(model.getDownscaleFactor(numIters), device);
            std::cout << valCam->filePath << " validation loss: " << model.mainLoss(rgb, gt, valMask, ssimWeight).item<float>() << std::endl;

            torch::Tensor mse;
            if (valMask.defined() && valMask.numel() > 0){
                mse = (valMask.unsqueeze(-1) * (rgb - gt).pow(2)).sum() / (valMask.sum() * gt.size(2) + 1e-8f);
            }else{
                mse = (rgb - gt).pow(2).mean();
            }
            std::cout << valCam->filePath << " validation PSNR: " << (10.0f * torch::log10(1.0f / mse)).item<float>() << std::endl;
        }
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}
