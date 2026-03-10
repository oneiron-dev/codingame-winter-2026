use std::collections::BTreeSet;

use crate::{java_random::JavaRandom, Coord, GameState, Grid, TileType};

pub const MIN_GRID_HEIGHT: i32 = 10;
pub const MAX_GRID_HEIGHT: i32 = 24;
pub const ASPECT_RATIO: f32 = 1.8;
pub const SPAWN_HEIGHT: i32 = 3;
pub const DESIRED_SPAWNS: i32 = 4;

#[derive(Clone, Debug)]
pub struct GridMaker {
    random: JavaRandom,
    league_level: i32,
}

impl GridMaker {
    pub fn new(seed: i64, league_level: i32) -> Self {
        Self {
            random: JavaRandom::new(seed),
            league_level,
        }
    }

    pub fn make(&mut self) -> Grid {
        let skew = match self.league_level {
            1 => 2.0,
            2 => 1.0,
            3 => 0.8,
            _ => 0.3,
        };

        let rand = self.random.next_double();
        let height = MIN_GRID_HEIGHT
            + (f64::powf(rand, skew) * f64::from(MAX_GRID_HEIGHT - MIN_GRID_HEIGHT)).round() as i32;
        let mut width = ((height as f32) * ASPECT_RATIO).round() as i32;
        if width % 2 != 0 {
            width += 1;
        }
        let mut grid = Grid::new(width, height);

        let b = 5.0 + self.random.next_double() * 10.0;
        for x in 0..width {
            grid.set(Coord::new(x, height - 1), TileType::Wall);
        }

        for y in (0..(height - 1)).rev() {
            let y_norm = f64::from(height - 1 - y) / f64::from(height - 1);
            let block_chance = 1.0 / (y_norm + 0.1) / b;
            for x in 0..width {
                if self.random.next_double() < block_chance {
                    grid.set(Coord::new(x, y), TileType::Wall);
                }
            }
        }

        for coord in grid.coords() {
            let opp = grid.opposite(coord);
            if let Some(tile) = grid.get(coord) {
                grid.set(opp, tile);
            }
        }

        for island in grid.detect_air_pockets() {
            if island.len() < 10 {
                for coord in island {
                    grid.set(coord, TileType::Wall);
                }
            }
        }

        let coords = grid.coords();
        let mut something_destroyed = true;
        while something_destroyed {
            something_destroyed = false;
            for coord in coords.iter().copied() {
                if grid.get(coord) != Some(TileType::Empty) {
                    continue;
                }
                let neighbour_walls: Vec<_> = grid
                    .neighbours(coord, &Grid::ADJACENCY)
                    .into_iter()
                    .filter(|next| grid.get(*next) == Some(TileType::Wall))
                    .collect();
                if neighbour_walls.len() < 3 {
                    continue;
                }
                let mut destroyable: Vec<_> = neighbour_walls
                    .into_iter()
                    .filter(|next| next.y <= coord.y)
                    .collect();
                self.random.shuffle(&mut destroyable);
                if let Some(target) = destroyable.first().copied() {
                    grid.set(target, TileType::Empty);
                    grid.set(grid.opposite(target), TileType::Empty);
                    something_destroyed = true;
                }
            }
        }

        let island = grid.detect_lowest_island();
        let island_set: BTreeSet<_> = island.iter().copied().collect();
        let mut lower_by = 0;
        let mut can_lower = true;
        while can_lower {
            for x in 0..width {
                let coord = Coord::new(x, height - 1 - (lower_by + 1));
                if !island_set.contains(&coord) {
                    can_lower = false;
                    break;
                }
            }
            if can_lower {
                lower_by += 1;
            }
        }
        if lower_by >= 2 {
            lower_by = self.random.next_int_range(2, lower_by + 1);
        }

        for coord in island.iter().copied() {
            grid.set(coord, TileType::Empty);
            grid.set(grid.opposite(coord), TileType::Empty);
        }
        for coord in island.iter().copied() {
            let lowered = Coord::new(coord.x, coord.y + lower_by);
            if grid.is_valid(lowered) {
                grid.set(lowered, TileType::Wall);
                grid.set(grid.opposite(lowered), TileType::Wall);
            }
        }

        for y in 0..height {
            for x in 0..(width / 2) {
                let coord = Coord::new(x, y);
                if grid.get(coord) == Some(TileType::Empty) && self.random.next_double() < 0.025 {
                    grid.apples.push(coord);
                    grid.apples.push(grid.opposite(coord));
                }
            }
        }

        for coord in coords.iter().copied() {
            if grid.get(coord) != Some(TileType::Wall) {
                continue;
            }
            let neighbour_wall_count = grid
                .neighbours(coord, &Grid::ADJACENCY_8)
                .into_iter()
                .filter(|next| grid.get(*next) == Some(TileType::Wall))
                .count();
            if neighbour_wall_count == 0 {
                grid.set(coord, TileType::Empty);
                let opp = grid.opposite(coord);
                grid.set(opp, TileType::Empty);
                grid.apples.push(coord);
                grid.apples.push(opp);
            }
        }

        let mut potential_spawns: Vec<_> = coords
            .iter()
            .copied()
            .filter(|coord| {
                grid.get(*coord) == Some(TileType::Wall)
                    && grid.get_free_above(*coord, SPAWN_HEIGHT).len() == SPAWN_HEIGHT as usize
            })
            .collect();
        self.random.shuffle(&mut potential_spawns);
        let mut desired_spawns = DESIRED_SPAWNS;
        if height <= 15 {
            desired_spawns -= 1;
        }
        if height <= 10 {
            desired_spawns -= 1;
        }

        while desired_spawns > 0 && !potential_spawns.is_empty() {
            let spawn = potential_spawns.remove(0);
            let spawn_loc = grid.get_free_above(spawn, SPAWN_HEIGHT);
            let mut too_close = false;
            for coord in spawn_loc.iter().copied() {
                if coord.x == width / 2 - 1 || coord.x == width / 2 {
                    too_close = true;
                    break;
                }
                for neighbour in grid.neighbours(coord, &Grid::ADJACENCY_8) {
                    if grid.spawns.contains(&neighbour)
                        || grid.spawns.contains(&grid.opposite(neighbour))
                    {
                        too_close = true;
                        break;
                    }
                }
                if too_close {
                    break;
                }
            }
            if too_close {
                continue;
            }

            for coord in spawn_loc {
                grid.spawns.push(coord);
                let opp = grid.opposite(coord);
                grid.apples.retain(|apple| *apple != coord && *apple != opp);
            }
            desired_spawns -= 1;
        }

        grid.sorted_unique_apples();
        grid
    }
}

pub fn generate_map(seed: i64, league_level: i32) -> Grid {
    GridMaker::new(seed, league_level).make()
}

pub fn initial_state_from_seed(seed: i64, league_level: i32) -> GameState {
    let grid = generate_map(seed, league_level);
    let spawn_islands = grid.detect_spawn_islands();
    let mut state = GameState::new(grid);
    let mut next_bird_id = 0;

    for owner in 0..=1 {
        for island in spawn_islands.iter() {
            let mut body: Vec<_> = island.iter().copied().collect();
            body.sort();
            if owner == 1 {
                body = body
                    .into_iter()
                    .map(|coord| state.grid.opposite(coord))
                    .collect::<Vec<_>>();
            }
            state.add_bird(next_bird_id, owner, body, None);
            next_bird_id += 1;
        }
    }

    state
}

#[cfg(test)]
mod tests {
    use super::{generate_map, initial_state_from_seed};

    #[test]
    fn generated_map_is_symmetric() {
        let map = generate_map(42, 3);
        for coord in map.coords() {
            let opposite = map.opposite(coord);
            assert_eq!(map.get(coord), map.get(opposite));
        }
        assert_eq!(map.apples.len() % 2, 0);
    }

    #[test]
    fn initial_state_has_even_spawn_count() {
        let state = initial_state_from_seed(42, 3);
        assert_eq!(state.birds.len() % 2, 0);
    }
}
