use serde::Serialize;
use snakebot_engine::{Coord, FinalResult, GameState, OracleState, TileType};

use crate::search::SearchStats;

pub const GRID_CHANNELS: usize = 8;
pub const SCALAR_FEATURES: usize = 6;
const VALUE_SCALE: f32 = 12.0;

#[derive(Clone, Debug, Serialize)]
pub struct EncodedPosition {
    pub grid: Vec<Vec<Vec<f32>>>,
    pub scalars: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingRow {
    pub schema_version: u32,
    pub git_sha: String,
    pub config_artifact_hash: String,
    pub config_behavior_hash: String,
    pub seed: i64,
    pub game_id: String,
    pub turn: i32,
    pub owner: usize,
    pub raw_state_hash: String,
    pub encoded_view_hash: String,
    pub grid: Vec<Vec<Vec<f32>>>,
    pub scalars: Vec<f32>,
    pub value: f32,
    pub weight: f32,
    pub final_body_diff: i32,
    pub final_loss_diff: i32,
    pub winner: Option<usize>,
    pub chosen_action_id: usize,
    pub joint_action_count: usize,
    pub root_values: Vec<f32>,
    pub budget_type: String,
    pub budget_value: u64,
    pub search_stats: SearchStats,
}

#[derive(Clone, Debug)]
pub struct TrainingMetadata {
    pub schema_version: u32,
    pub git_sha: String,
    pub config_artifact_hash: String,
    pub config_behavior_hash: String,
    pub seed: i64,
    pub game_id: String,
    pub turn: i32,
    pub chosen_action_id: usize,
    pub joint_action_count: usize,
    pub root_values: Vec<f32>,
    pub budget_type: String,
    pub budget_value: u64,
    pub search_stats: SearchStats,
}

pub fn encode_position(state: &GameState, owner: usize) -> EncodedPosition {
    let width = state.grid.width as usize;
    let height = state.grid.height as usize;
    let mut grid = vec![vec![vec![0.0_f32; width]; height]; GRID_CHANNELS];

    for coord in state.grid.coords() {
        let view = canonical_coord(state, owner, coord);
        let (x, y) = (view.x as usize, view.y as usize);
        if state.grid.get(coord) == Some(TileType::Wall) {
            grid[0][y][x] = 1.0;
        }
    }

    for apple in &state.grid.apples {
        let view = canonical_coord(state, owner, *apple);
        grid[1][view.y as usize][view.x as usize] = 1.0;
    }

    for bird in state.birds.iter().filter(|bird| bird.alive) {
        let channels = if bird.owner == owner {
            (2, 3, 6)
        } else {
            (4, 5, 7)
        };
        let head = canonical_coord(state, owner, bird.head());
        grid[channels.0][head.y as usize][head.x as usize] = 1.0;
        for segment in bird.body.iter().skip(1) {
            let view = canonical_coord(state, owner, *segment);
            grid[channels.1][view.y as usize][view.x as usize] = 1.0;
        }
        for segment in bird.body.iter().copied() {
            if segment_has_support(state, bird.id, segment) {
                let view = canonical_coord(state, owner, segment);
                grid[channels.2][view.y as usize][view.x as usize] = 1.0;
            }
        }
    }

    EncodedPosition {
        grid,
        scalars: scalar_features(state, owner),
    }
}

pub fn encode_training_row(
    state: &GameState,
    owner: usize,
    final_result: &FinalResult,
    metadata: TrainingMetadata,
) -> TrainingRow {
    let encoded = encode_position(state, owner);
    let score_diff =
        (final_result.final_scores[owner] - final_result.final_scores[1 - owner]) as f32;
    let raw_state_hash = stable_hash(
        &serde_json::to_vec(&OracleState::from_game_state(state)).expect("oracle state hashable"),
    );
    let encoded_view_hash =
        stable_hash(&serde_json::to_vec(&encoded).expect("encoded position hashable"));
    TrainingRow {
        schema_version: metadata.schema_version,
        git_sha: metadata.git_sha,
        config_artifact_hash: metadata.config_artifact_hash,
        config_behavior_hash: metadata.config_behavior_hash,
        seed: metadata.seed,
        game_id: metadata.game_id,
        turn: metadata.turn,
        owner,
        raw_state_hash,
        encoded_view_hash,
        grid: encoded.grid,
        scalars: encoded.scalars,
        value: (score_diff / VALUE_SCALE).tanh(),
        weight: 1.0,
        final_body_diff: final_result.body_diff_for(owner),
        final_loss_diff: final_result.loss_diff_for(owner),
        winner: final_result
            .winner
            .map(|winner| if winner == owner { owner } else { 1 - owner }),
        chosen_action_id: metadata.chosen_action_id,
        joint_action_count: metadata.joint_action_count,
        root_values: metadata.root_values,
        budget_type: metadata.budget_type,
        budget_value: metadata.budget_value,
        search_stats: metadata.search_stats,
    }
}

fn stable_hash(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn scalar_features(state: &GameState, owner: usize) -> Vec<f32> {
    let body_scores = state.body_scores();
    let live_diff = live_bird_count(state, owner) as f32 - live_bird_count(state, 1 - owner) as f32;
    let breakpoint_diff =
        breakpoint_count(state, owner) as f32 - breakpoint_count(state, 1 - owner) as f32;
    let mobility_diff =
        mobility_count(state, owner) as f32 - mobility_count(state, 1 - owner) as f32;

    vec![
        (state.turn as f32 / 200.0).clamp(0.0, 1.0),
        ((body_scores[owner] - body_scores[1 - owner]) as f32 / 32.0).clamp(-1.0, 1.0),
        ((state.losses[1 - owner] - state.losses[owner]) as f32 / 32.0).clamp(-1.0, 1.0),
        (live_diff / 4.0).clamp(-1.0, 1.0),
        (breakpoint_diff / 4.0).clamp(-1.0, 1.0),
        (mobility_diff / 16.0).clamp(-1.0, 1.0),
    ]
}

fn canonical_coord(state: &GameState, owner: usize, coord: Coord) -> Coord {
    if owner == 0 {
        coord
    } else {
        state.grid.opposite(coord)
    }
}

fn live_bird_count(state: &GameState, owner: usize) -> usize {
    state
        .birds_for_player(owner)
        .filter(|bird| bird.alive)
        .count()
}

fn breakpoint_count(state: &GameState, owner: usize) -> usize {
    state
        .birds_for_player(owner)
        .filter(|bird| bird.alive && bird.length() >= 4)
        .count()
}

fn mobility_count(state: &GameState, owner: usize) -> usize {
    state
        .birds_for_player(owner)
        .filter(|bird| bird.alive)
        .map(|bird| state.legal_commands_for_bird(bird.id).len())
        .sum()
}

fn segment_has_support(state: &GameState, bird_id: i32, coord: Coord) -> bool {
    let below = coord.add(0, 1);
    if state.grid.get(below) == Some(TileType::Wall) || state.grid.apples.contains(&below) {
        return true;
    }
    state
        .birds
        .iter()
        .filter(|bird| bird.alive && bird.id != bird_id)
        .any(|bird| bird.body.contains(&below))
}

#[cfg(test)]
mod tests {
    use snakebot_engine::{Direction, Grid};

    use super::{encode_position, GRID_CHANNELS, SCALAR_FEATURES};
    use snakebot_engine::{Coord, GameState, TileType};

    #[test]
    fn encodes_expected_channel_counts() {
        let mut grid = Grid::new(4, 4);
        grid.set(Coord::new(0, 3), TileType::Wall);
        grid.apples.push(Coord::new(2, 2));
        let mut state = GameState::new(grid);
        state.add_bird(
            0,
            0,
            vec![Coord::new(1, 1), Coord::new(1, 2), Coord::new(1, 3)],
            Some(Direction::North),
        );
        let encoded = encode_position(&state, 0);
        assert_eq!(encoded.grid.len(), GRID_CHANNELS);
        assert_eq!(encoded.scalars.len(), SCALAR_FEATURES);
    }

    #[test]
    fn mirrors_player_one_view() {
        let grid = Grid::new(5, 4);
        let mut state = GameState::new(grid);
        state.add_bird(
            1,
            1,
            vec![Coord::new(4, 1), Coord::new(4, 2), Coord::new(4, 3)],
            Some(Direction::North),
        );
        let encoded = encode_position(&state, 1);
        assert_eq!(encoded.grid[2][1][0], 1.0);
    }
}
