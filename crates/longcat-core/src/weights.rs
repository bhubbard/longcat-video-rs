use anyhow::{Context, Result};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub num_elements: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSummary {
    pub path: PathBuf,
    pub tensor_count: usize,
    pub total_parameters: usize,
    pub tensors: Vec<TensorInfo>,
}

pub struct WeightsManager;

impl WeightsManager {
    /// Inspect a SafeTensors checkpoint file
    pub fn inspect_checkpoint<P: AsRef<Path>>(path: P) -> Result<CheckpointSummary> {
        let path_ref = path.as_ref();
        let mut file = File::open(path_ref)
            .with_context(|| format!("Failed to open SafeTensors file {:?}", path_ref))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let tensors = SafeTensors::deserialize(&buffer)
            .with_context(|| format!("Failed to parse SafeTensors metadata from {:?}", path_ref))?;

        let mut tensor_infos = Vec::new();
        let mut total_params = 0;

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let count: usize = shape.iter().product();
            total_params += count;

            tensor_infos.push(TensorInfo {
                name,
                shape,
                dtype: format!("{:?}", view.dtype()),
                num_elements: count,
            });
        }

        tensor_infos.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(CheckpointSummary {
            path: path_ref.to_path_buf(),
            tensor_count: tensor_infos.len(),
            total_parameters: total_params,
            tensors: tensor_infos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_serde() {
        let info = TensorInfo {
            name: "dit.blocks.0.attn.q_proj.weight".to_string(),
            shape: vec![3072, 3072],
            dtype: "BF16".to_string(),
            num_elements: 3072 * 3072,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("dit.blocks.0.attn.q_proj.weight"));
    }
}
