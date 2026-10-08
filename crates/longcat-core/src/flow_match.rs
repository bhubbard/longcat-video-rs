use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMatchStep {
    pub step_index: usize,
    pub timestep: f32,
    pub sigma: f32,
    pub dt: f32,
}

pub struct FlowMatchingEulerScheduler {
    pub num_train_timesteps: usize,
    pub shift: f32,
}

impl FlowMatchingEulerScheduler {
    pub fn new(shift: f32) -> Self {
        Self {
            num_train_timesteps: 1000,
            shift,
        }
    }

    /// Compute shifted time schedule for video flow matching
    pub fn get_schedule(&self, num_inference_steps: usize) -> Vec<FlowMatchStep> {
        let mut sigmas = Vec::with_capacity(num_inference_steps + 1);

        for i in 0..=num_inference_steps {
            let t = 1.0 - (i as f32 / num_inference_steps as f32);
            // Apply time shift function: sigma_shifted = (shift * t) / (1 + (shift - 1) * t)
            let shifted = if (1.0 + (self.shift - 1.0) * t).abs() > 1e-6 {
                (self.shift * t) / (1.0 + (self.shift - 1.0) * t)
            } else {
                t
            };
            sigmas.push(shifted);
        }

        let mut steps = Vec::with_capacity(num_inference_steps);
        for i in 0..num_inference_steps {
            let sigma = sigmas[i];
            let next_sigma = sigmas[i + 1];
            let dt = next_sigma - sigma;
            let timestep = sigma * self.num_train_timesteps as f32;

            steps.push(FlowMatchStep {
                step_index: i,
                timestep,
                sigma,
                dt,
            });
        }

        steps
    }

    /// Perform one Euler integration step along the flow field:
    /// x_{next} = x_curr + dt * velocity
    pub fn step(sample: &[f32], model_output: &[f32], dt: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(sample.len());
        for i in 0..sample.len() {
            let v = if i < model_output.len() { model_output[i] } else { 0.0 };
            out.push(sample[i] + dt * v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_match_schedule() {
        let scheduler = FlowMatchingEulerScheduler::new(3.0);
        let schedule = scheduler.get_schedule(30);

        assert_eq!(schedule.len(), 30);
        assert!((schedule[0].sigma - 1.0).abs() < 1e-5);
        assert!(schedule[29].sigma > 0.0);
        // Monotonically decreasing sigmas
        for w in schedule.windows(2) {
            assert!(w[0].sigma > w[1].sigma);
            assert!(w[0].dt < 0.0);
        }
    }

    #[test]
    fn test_euler_step() {
        let sample = vec![1.0, 2.0, 3.0];
        let velocity = vec![0.5, -0.5, 1.0];
        let next = FlowMatchingEulerScheduler::step(&sample, &velocity, -0.1);
        assert!((next[0] - 0.95).abs() < 1e-5);
        assert!((next[1] - 2.05).abs() < 1e-5);
        assert!((next[2] - 2.90).abs() < 1e-5);
    }
}
