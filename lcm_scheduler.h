#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

class LCMScheduler {
public:
    LCMScheduler();
    
    void SetTimesteps(int num_inference_steps, int original_inference_steps = 50);
    
    std::pair<std::vector<float>, std::vector<float>> Step(
        const std::vector<float>& noise_pred,
        int timestep,
        const std::vector<float>& sample,
        int step_index = -1  // -1 means auto-detect, otherwise use provided index
    );
    
    void ResetStepIndex() { step_index_ = -1; }

    float GetInitNoiseSigma() const { return init_noise_sigma_; }
    const std::vector<int>& GetTimesteps() const { return timesteps_; }
    int GetOrder() const { return 1; }

private:
    float init_noise_sigma_;
    std::vector<int> timesteps_;
    int num_train_timesteps_;
    float beta_start_;
    float beta_end_;
    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alphas_cumprod_;
    float timestep_scaling_;  // Default: 10.0 for LCM
    float sigma_data_;  // Default: 0.5 for LCM
    int step_index_;  // Current step index in the timestep schedule
    int num_inference_steps_;  // Number of inference steps
    
    void GenerateBetas();
    std::pair<float, float> GetScalingsForBoundaryConditionDiscrete(int timestep);
    int FindTimestepIndex(int timestep) const;
};
