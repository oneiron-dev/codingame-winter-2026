use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

use serde::Deserialize;
use snakebot_engine::{GameState, PlayerAction};

use crate::config::{BotConfig, HybridConfig};
use crate::features::{encode_hybrid_position, policy_targets_for_action, HYBRID_GRID_CHANNELS, POLICY_ACTIONS_PER_BIRD, SCALAR_FEATURES};

const MIN_SCHEMA_VERSION: u32 = 1;
const MAX_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug)]
pub struct HybridPrediction {
    pub policy_logits: Vec<f32>,
    pub value: f32,
}

impl HybridPrediction {
    pub fn action_prior(&self, state: &GameState, owner: usize, action: &PlayerAction) -> f64 {
        let targets = policy_targets_for_action(state, owner, action);
        let mut total = 0.0_f64;
        for (slot_idx, target) in targets.into_iter().enumerate() {
            if target < 0 {
                continue;
            }
            let index = slot_idx * POLICY_ACTIONS_PER_BIRD + target as usize;
            if let Some(logit) = self.policy_logits.get(index) {
                total += f64::from(*logit);
            }
        }
        total
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ConvLayer {
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct LinearLayer {
    out_features: usize,
    in_features: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct TinyHybridWeights {
    version: u32,
    input_channels: usize,
    scalar_features: usize,
    board_height: usize,
    board_width: usize,
    conv1: ConvLayer,
    conv2: ConvLayer,
    #[serde(default)]
    conv3: Option<ConvLayer>,
    policy: LinearLayer,
    value: LinearLayer,
}

static MODEL_CACHE: OnceLock<RwLock<HashMap<String, Arc<TinyHybridWeights>>>> = OnceLock::new();

pub fn predict(state: &GameState, owner: usize, config: &BotConfig) -> Option<HybridPrediction> {
    let hybrid = config.hybrid.as_ref()?;
    if !hybrid.is_enabled() {
        return None;
    }
    let path = hybrid.weights_path.as_deref()?;
    let model = load_model(path).ok()?;
    let encoded = encode_hybrid_position(state, owner);
    if encoded.grid.len() != model.input_channels || encoded.scalars.len() != model.scalar_features {
        return None;
    }
    Some(model.forward(&encoded.grid, &encoded.scalars))
}

pub fn leaf_bonus(prediction: &HybridPrediction, hybrid: &HybridConfig) -> f64 {
    f64::from(prediction.value) * hybrid.value_scale
}

fn load_model(path: &str) -> Result<Arc<TinyHybridWeights>, Box<dyn std::error::Error>> {
    let resolved = Path::new(path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(path).to_path_buf());
    let key = resolved.to_string_lossy().into_owned();
    let cache = MODEL_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(existing) = cache.read().expect("cache read").get(&key) {
        return Ok(existing.clone());
    }

    let raw = fs::read_to_string(&resolved)?;
    let model: TinyHybridWeights = serde_json::from_str(&raw)?;
    model.validate()?;
    let model = Arc::new(model);
    cache
        .write()
        .expect("cache write")
        .insert(key, model.clone());
    Ok(model)
}

impl TinyHybridWeights {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.version < MIN_SCHEMA_VERSION || self.version > MAX_SCHEMA_VERSION {
            return Err(format!(
                "unsupported hybrid schema version {}, expected {}-{}",
                self.version, MIN_SCHEMA_VERSION, MAX_SCHEMA_VERSION
            )
            .into());
        }
        if self.input_channels != HYBRID_GRID_CHANNELS {
            return Err(format!("unexpected input channels {}", self.input_channels).into());
        }
        if self.scalar_features != SCALAR_FEATURES {
            return Err(format!("unexpected scalar features {}", self.scalar_features).into());
        }
        self.conv1.validate()?;
        self.conv2.validate()?;
        if let Some(ref conv3) = self.conv3 {
            conv3.validate()?;
        }
        self.policy.validate()?;
        self.value.validate()?;
        Ok(())
    }

    fn forward(&self, grid: &[Vec<Vec<f32>>], scalars: &[f32]) -> HybridPrediction {
        let height = grid.first().map(|channel| channel.len()).unwrap_or(0);
        let width = grid
            .first()
            .and_then(|channel| channel.first())
            .map(|row| row.len())
            .unwrap_or(0);
        // Flatten grid from [C][H][W] nested vecs to flat [C*H*W]
        let mut flat_input = Vec::with_capacity(grid.len() * height * width);
        for channel in grid {
            for row in channel {
                flat_input.extend_from_slice(row);
            }
        }
        let conv1_out = self.conv1.forward_flat(&flat_input, height, width);
        let conv2_out = self.conv2.forward_flat(&conv1_out, height, width);
        let (last_out, last_channels) = match self.conv3 {
            Some(ref conv3) => (conv3.forward_flat(&conv2_out, height, width), conv3.out_channels),
            None => (conv2_out, self.conv2.out_channels),
        };
        let pooled = global_average_pool_flat(&last_out, last_channels, height, width);
        let mut features = pooled;
        features.extend_from_slice(scalars);
        let policy_logits = self.policy.forward(&features);
        let value = self
            .value
            .forward(&features)
            .first()
            .copied()
            .unwrap_or_default()
            .tanh();
        HybridPrediction {
            policy_logits,
            value,
        }
    }
}

impl ConvLayer {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.kernel_size != 1 && self.kernel_size != 3 {
            return Err("only 1x1 and 3x3 conv kernels are supported".into());
        }
        if self.weights.len()
            != self.out_channels * self.in_channels * self.kernel_size * self.kernel_size
        {
            return Err("invalid conv weight shape".into());
        }
        if self.bias.len() != self.out_channels {
            return Err("invalid conv bias shape".into());
        }
        Ok(())
    }

    /// Forward pass on flat layout: input[channel * H * W + y * W + x].
    fn forward_flat(
        &self,
        input: &[f32],
        height: usize,
        width: usize,
    ) -> Vec<f32> {
        let hw = height * width;
        let mut output = vec![0.0_f32; self.out_channels * hw];
        let pad = (self.kernel_size / 2) as isize;
        for oc in 0..self.out_channels {
            for y in 0..height {
                for x in 0..width {
                    let mut acc = self.bias[oc];
                    for ic in 0..self.in_channels {
                        for ky in 0..self.kernel_size {
                            for kx in 0..self.kernel_size {
                                let iy = y as isize + ky as isize - pad;
                                let ix = x as isize + kx as isize - pad;
                                if iy < 0
                                    || ix < 0
                                    || iy >= height as isize
                                    || ix >= width as isize
                                {
                                    continue;
                                }
                                let w_idx = ((oc * self.in_channels + ic) * self.kernel_size + ky)
                                    * self.kernel_size
                                    + kx;
                                acc += input[ic * hw + iy as usize * width + ix as usize]
                                    * self.weights[w_idx];
                            }
                        }
                    }
                    output[oc * hw + y * width + x] = acc.max(0.0);
                }
            }
        }
        output
    }
}

impl LinearLayer {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.weights.len() != self.out_features * self.in_features {
            return Err("invalid linear weight shape".into());
        }
        if self.bias.len() != self.out_features {
            return Err("invalid linear bias shape".into());
        }
        Ok(())
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0_f32; self.out_features];
        for (out_index, slot) in output.iter_mut().enumerate() {
            let mut acc = self.bias[out_index];
            let base = out_index * self.in_features;
            for in_index in 0..self.in_features {
                acc += input[in_index] * self.weights[base + in_index];
            }
            *slot = acc;
        }
        output
    }
}

fn global_average_pool_flat(input: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    let hw = height * width;
    let norm = hw.max(1) as f32;
    (0..channels)
        .map(|c| {
            let start = c * hw;
            input[start..start + hw].iter().sum::<f32>() / norm
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::config::BotConfig;
    use crate::features::{HYBRID_GRID_CHANNELS, MAX_BIRDS_PER_PLAYER, POLICY_ACTIONS_PER_BIRD, SCALAR_FEATURES};

    use super::{load_model, predict};
    use snakebot_engine::initial_state_from_seed;

    #[test]
    fn loads_fixture_model_and_predicts() {
        let weights = temp_fixture_weights();
        let _ = load_model(weights.to_str().expect("fixture path")).expect("fixture should load");
        let mut config = BotConfig::embedded();
        config.hybrid = Some(crate::config::HybridConfig {
            weights_path: Some(weights.to_string_lossy().into_owned()),
            prior_mix: 0.25,
            leaf_mix: 0.5,
            value_scale: 48.0,
            prior_depth_limit: usize::MAX,
            leaf_depth_limit: usize::MAX,
        });
        let state = initial_state_from_seed(1, 4);
        let prediction = predict(&state, 0, &config).expect("prediction should be available");
        assert_eq!(
            prediction.policy_logits.len(),
            MAX_BIRDS_PER_PLAYER * POLICY_ACTIONS_PER_BIRD
        );
        let _ = fs::remove_file(weights);
    }

    #[test]
    fn loads_three_layer_fixture_and_predicts() {
        let weights = temp_fixture_weights_v2();
        let _ = load_model(weights.to_str().expect("fixture path")).expect("v2 fixture should load");
        let mut config = BotConfig::embedded();
        config.hybrid = Some(crate::config::HybridConfig {
            weights_path: Some(weights.to_string_lossy().into_owned()),
            prior_mix: 0.25,
            leaf_mix: 0.5,
            value_scale: 48.0,
            prior_depth_limit: usize::MAX,
            leaf_depth_limit: usize::MAX,
        });
        let state = initial_state_from_seed(1, 4);
        let prediction = predict(&state, 0, &config).expect("prediction should be available");
        assert_eq!(
            prediction.policy_logits.len(),
            MAX_BIRDS_PER_PLAYER * POLICY_ACTIONS_PER_BIRD
        );
        let _ = fs::remove_file(weights);
    }

    fn temp_fixture_weights() -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("snakebot-hybrid-{suffix}.json"));
        let json = serde_json::json!({
            "version": 1,
            "input_channels": HYBRID_GRID_CHANNELS,
            "scalar_features": SCALAR_FEATURES,
            "board_height": 23,
            "board_width": 42,
            "conv1": {
                "out_channels": 4,
                "in_channels": HYBRID_GRID_CHANNELS,
                "kernel_size": 3,
                "weights": vec![0.0_f32; 4 * HYBRID_GRID_CHANNELS * 3 * 3],
                "bias": vec![0.05_f32; 4],
            },
            "conv2": {
                "out_channels": 4,
                "in_channels": 4,
                "kernel_size": 3,
                "weights": vec![0.0_f32; 4 * 4 * 3 * 3],
                "bias": vec![0.02_f32; 4],
            },
            "policy": {
                "out_features": MAX_BIRDS_PER_PLAYER * POLICY_ACTIONS_PER_BIRD,
                "in_features": 4 + SCALAR_FEATURES,
                "weights": vec![0.01_f32; (MAX_BIRDS_PER_PLAYER * POLICY_ACTIONS_PER_BIRD) * (4 + SCALAR_FEATURES)],
                "bias": vec![0.0_f32; MAX_BIRDS_PER_PLAYER * POLICY_ACTIONS_PER_BIRD],
            },
            "value": {
                "out_features": 1,
                "in_features": 4 + SCALAR_FEATURES,
                "weights": vec![0.01_f32; 4 + SCALAR_FEATURES],
                "bias": vec![0.0_f32; 1],
            },
        });
        fs::write(&path, serde_json::to_vec_pretty(&json).expect("json bytes")).expect("fixture write");
        path
    }

    fn temp_fixture_weights_v2() -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("snakebot-hybrid-v2-{suffix}.json"));
        let json = serde_json::json!({
            "version": 2,
            "input_channels": HYBRID_GRID_CHANNELS,
            "scalar_features": SCALAR_FEATURES,
            "board_height": 23,
            "board_width": 42,
            "conv1": {
                "out_channels": 4,
                "in_channels": HYBRID_GRID_CHANNELS,
                "kernel_size": 3,
                "weights": vec![0.0_f32; 4 * HYBRID_GRID_CHANNELS * 3 * 3],
                "bias": vec![0.05_f32; 4],
            },
            "conv2": {
                "out_channels": 4,
                "in_channels": 4,
                "kernel_size": 3,
                "weights": vec![0.0_f32; 4 * 4 * 3 * 3],
                "bias": vec![0.02_f32; 4],
            },
            "conv3": {
                "out_channels": 4,
                "in_channels": 4,
                "kernel_size": 3,
                "weights": vec![0.0_f32; 4 * 4 * 3 * 3],
                "bias": vec![0.01_f32; 4],
            },
            "policy": {
                "out_features": MAX_BIRDS_PER_PLAYER * POLICY_ACTIONS_PER_BIRD,
                "in_features": 4 + SCALAR_FEATURES,
                "weights": vec![0.01_f32; (MAX_BIRDS_PER_PLAYER * POLICY_ACTIONS_PER_BIRD) * (4 + SCALAR_FEATURES)],
                "bias": vec![0.0_f32; MAX_BIRDS_PER_PLAYER * POLICY_ACTIONS_PER_BIRD],
            },
            "value": {
                "out_features": 1,
                "in_features": 4 + SCALAR_FEATURES,
                "weights": vec![0.01_f32; 4 + SCALAR_FEATURES],
                "bias": vec![0.0_f32; 1],
            },
        });
        fs::write(&path, serde_json::to_vec_pretty(&json).expect("json bytes")).expect("fixture write");
        path
    }
}
