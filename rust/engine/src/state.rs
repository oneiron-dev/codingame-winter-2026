use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::{Coord, Direction, Grid, TileType};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BirdState {
    pub id: i32,
    pub owner: usize,
    pub body: VecDeque<Coord>,
    pub alive: bool,
    pub direction: Option<Direction>,
}

impl BirdState {
    pub fn head(&self) -> Coord {
        *self.body.front().expect("bird has at least one segment")
    }

    pub fn facing(&self) -> Direction {
        if self.body.len() < 2 {
            Direction::Unset
        } else {
            let head = self.body[0];
            let neck = self.body[1];
            Direction::from_coord(Coord::new(head.x - neck.x, head.y - neck.y))
        }
    }

    pub fn length(&self) -> usize {
        self.body.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BirdCommand {
    Keep,
    Turn(Direction),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlayerAction {
    pub per_bird: BTreeMap<i32, BirdCommand>,
}

impl PlayerAction {
    pub fn command_for(&self, bird_id: i32) -> BirdCommand {
        self.per_bird
            .get(&bird_id)
            .copied()
            .unwrap_or(BirdCommand::Keep)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    pub grid: Grid,
    pub birds: Vec<BirdState>,
    pub losses: [i32; 2],
    pub turn: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepResult {
    pub game_over: bool,
    pub body_scores: [i32; 2],
    pub final_scores: [i32; 2],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VisibilityState {
    pub player_index: usize,
    pub my_ids: Vec<i32>,
    pub opp_ids: Vec<i32>,
    pub state: GameState,
}

impl GameState {
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            birds: Vec::new(),
            losses: [0, 0],
            turn: 0,
        }
    }

    pub fn add_bird(
        &mut self,
        id: i32,
        owner: usize,
        body: Vec<Coord>,
        direction: Option<Direction>,
    ) {
        self.birds.push(BirdState {
            id,
            owner,
            body: body.into_iter().collect(),
            alive: true,
            direction,
        });
        self.birds.sort_by_key(|bird| bird.id);
    }

    pub fn body_scores(&self) -> [i32; 2] {
        let mut scores = [0, 0];
        for bird in self.live_birds() {
            scores[bird.owner] += bird.body.len() as i32;
        }
        scores
    }

    pub fn final_scores(&self) -> [i32; 2] {
        let mut scores = self.body_scores();
        if scores[0] == scores[1] {
            scores[0] -= self.losses[0];
            scores[1] -= self.losses[1];
        }
        scores
    }

    pub fn live_birds(&self) -> impl Iterator<Item = &BirdState> {
        self.birds.iter().filter(|bird| bird.alive)
    }

    pub fn live_birds_mut(&mut self) -> impl Iterator<Item = &mut BirdState> {
        self.birds.iter_mut().filter(|bird| bird.alive)
    }

    pub fn birds_for_player(&self, owner: usize) -> impl Iterator<Item = &BirdState> {
        self.birds.iter().filter(move |bird| bird.owner == owner)
    }

    pub fn legal_commands_for_bird(&self, bird_id: i32) -> Vec<BirdCommand> {
        let Some(bird) = self
            .birds
            .iter()
            .find(|bird| bird.id == bird_id && bird.alive)
        else {
            return Vec::new();
        };
        let facing = bird.facing();
        let mut commands = vec![BirdCommand::Keep];
        for direction in Direction::ALL {
            if facing != Direction::Unset && direction == facing.opposite() {
                continue;
            }
            if direction == facing {
                continue;
            }
            commands.push(BirdCommand::Turn(direction));
        }
        commands
    }

    pub fn legal_joint_actions(&self, owner: usize) -> Vec<PlayerAction> {
        let birds: Vec<_> = self
            .birds_for_player(owner)
            .filter(|bird| bird.alive)
            .map(|bird| bird.id)
            .collect();
        let mut result = Vec::new();
        let mut current = PlayerAction::default();
        self.enumerate_actions(&birds, 0, &mut current, &mut result);
        if result.is_empty() {
            result.push(PlayerAction::default());
        }
        result
    }

    fn enumerate_actions(
        &self,
        birds: &[i32],
        idx: usize,
        current: &mut PlayerAction,
        output: &mut Vec<PlayerAction>,
    ) {
        if idx == birds.len() {
            output.push(current.clone());
            return;
        }

        let bird_id = birds[idx];
        for command in self.legal_commands_for_bird(bird_id) {
            match command {
                BirdCommand::Keep => {
                    current.per_bird.remove(&bird_id);
                }
                BirdCommand::Turn(direction) => {
                    current
                        .per_bird
                        .insert(bird_id, BirdCommand::Turn(direction));
                }
            }
            self.enumerate_actions(birds, idx + 1, current, output);
        }
        current.per_bird.remove(&bird_id);
    }

    pub fn encode_player_view(&self, player_index: usize) -> VisibilityState {
        let mut my_ids = self
            .birds_for_player(player_index)
            .map(|bird| bird.id)
            .collect::<Vec<_>>();
        my_ids.sort();
        let mut opp_ids = self
            .birds_for_player(1 - player_index)
            .map(|bird| bird.id)
            .collect::<Vec<_>>();
        opp_ids.sort();
        VisibilityState {
            player_index,
            my_ids,
            opp_ids,
            state: self.clone(),
        }
    }

    pub fn step(&mut self, p0: &PlayerAction, p1: &PlayerAction) -> StepResult {
        self.turn += 1;
        self.reset_turn_state();
        self.apply_moves(p0, p1);
        self.apply_eats();
        self.apply_beheadings();
        self.apply_falls();

        let body_scores = self.body_scores();
        let final_scores = self.final_scores();
        StepResult {
            game_over: self.is_game_over(),
            body_scores,
            final_scores,
        }
    }

    fn reset_turn_state(&mut self) {
        for bird in self.live_birds_mut() {
            bird.direction = None;
        }
    }

    fn apply_moves(&mut self, p0: &PlayerAction, p1: &PlayerAction) {
        for idx in 0..self.birds.len() {
            if !self.birds[idx].alive {
                continue;
            }
            let command = if self.birds[idx].owner == 0 {
                p0.command_for(self.birds[idx].id)
            } else {
                p1.command_for(self.birds[idx].id)
            };

            match command {
                BirdCommand::Keep => {
                    if self.birds[idx].direction.is_none()
                        || self.birds[idx].direction == Some(Direction::Unset)
                    {
                        self.birds[idx].direction = Some(self.birds[idx].facing());
                    }
                }
                BirdCommand::Turn(direction) => {
                    let facing = self.birds[idx].facing();
                    if facing != Direction::Unset && direction == facing.opposite() {
                        self.birds[idx].direction = Some(facing);
                    } else {
                        self.birds[idx].direction = Some(direction);
                    }
                }
            }

            let direction = self.birds[idx].direction.unwrap_or(Direction::Unset);
            let new_head = self.birds[idx].head().add_coord(direction.delta());
            let will_eat = self.grid.apples.contains(&new_head);
            if !will_eat {
                self.birds[idx].body.pop_back();
            }
            self.birds[idx].body.push_front(new_head);
        }
    }

    fn apply_eats(&mut self) {
        let mut apples_eaten = BTreeSet::new();
        for bird in self.live_birds() {
            if self.grid.apples.contains(&bird.head()) {
                apples_eaten.insert(bird.head());
            }
        }
        self.grid
            .apples
            .retain(|apple| !apples_eaten.contains(apple));
    }

    fn apply_beheadings(&mut self) {
        let alive_indices: Vec<_> = self
            .birds
            .iter()
            .enumerate()
            .filter_map(|(idx, bird)| bird.alive.then_some(idx))
            .collect();
        let mut to_behead = Vec::new();

        for idx in alive_indices {
            let head = self.birds[idx].head();
            let is_in_wall = self.grid.get(head) == Some(TileType::Wall);
            let intersecting: Vec<_> = self
                .birds
                .iter()
                .filter(|bird| bird.alive && bird.body.contains(&head))
                .collect();
            let is_in_bird = intersecting.iter().any(|other| {
                other.id != self.birds[idx].id
                    || other
                        .body
                        .iter()
                        .skip(1)
                        .any(|segment| *segment == other.head())
            });

            if is_in_wall || is_in_bird {
                to_behead.push(idx);
            }
        }

        for idx in to_behead {
            let owner = self.birds[idx].owner;
            if self.birds[idx].body.len() <= 3 {
                self.losses[owner] += self.birds[idx].body.len() as i32;
                self.birds[idx].alive = false;
            } else {
                self.birds[idx].body.pop_front();
                self.losses[owner] += 1;
            }
        }
    }

    fn apply_falls(&mut self) {
        let mut something_fell = true;
        while something_fell {
            something_fell = false;
            while self.apply_individual_falls() {
                something_fell = true;
            }
            if self.apply_intercoiled_falls() {
                something_fell = true;
            }
        }
    }

    fn apply_individual_falls(&mut self) -> bool {
        let mut moved = false;
        let alive_indices: Vec<_> = self
            .birds
            .iter()
            .enumerate()
            .filter_map(|(idx, bird)| bird.alive.then_some(idx))
            .collect();
        for idx in alive_indices {
            let body: Vec<_> = self.birds[idx].body.iter().copied().collect();
            let can_fall = body
                .iter()
                .all(|coord| !self.something_solid_under(*coord, &body));
            if can_fall {
                moved = true;
                self.shift_bird_down(idx);
                if self.birds[idx]
                    .body
                    .iter()
                    .all(|coord| coord.y >= self.grid.height + 1)
                {
                    self.birds[idx].alive = false;
                }
            }
        }
        moved
    }

    fn apply_intercoiled_falls(&mut self) -> bool {
        let groups = self.intercoiled_groups();
        let mut moved = false;
        for group in groups {
            let meta_body: Vec<_> = group
                .iter()
                .flat_map(|idx| self.birds[*idx].body.iter().copied())
                .collect();
            let can_fall = meta_body
                .iter()
                .all(|coord| !self.something_solid_under(*coord, &meta_body));
            if !can_fall {
                continue;
            }
            moved = true;
            for idx in group {
                self.shift_bird_down(idx);
                if self.birds[idx].head().y >= self.grid.height {
                    self.birds[idx].alive = false;
                }
            }
        }
        moved
    }

    fn intercoiled_groups(&self) -> Vec<Vec<usize>> {
        let alive_indices: Vec<_> = self
            .birds
            .iter()
            .enumerate()
            .filter_map(|(idx, bird)| bird.alive.then_some(idx))
            .collect();
        let mut groups = Vec::new();
        let mut seen = BTreeSet::new();
        for idx in alive_indices.iter().copied() {
            if seen.contains(&idx) {
                continue;
            }
            let mut group = Vec::new();
            let mut queue = VecDeque::from([idx]);
            while let Some(current) = queue.pop_front() {
                if !seen.insert(current) {
                    continue;
                }
                group.push(current);
                for other in alive_indices.iter().copied() {
                    if current == other || seen.contains(&other) {
                        continue;
                    }
                    if birds_are_touching(&self.birds[current], &self.birds[other]) {
                        queue.push_back(other);
                    }
                }
            }
            if group.len() > 1 {
                groups.push(group);
            }
        }
        groups
    }

    fn shift_bird_down(&mut self, idx: usize) {
        for coord in self.birds[idx].body.iter_mut() {
            coord.y += 1;
        }
    }

    fn something_solid_under(&self, coord: Coord, ignore_body: &[Coord]) -> bool {
        let below = coord.add(0, 1);
        if ignore_body.contains(&below) {
            return false;
        }
        if self.grid.get(below) == Some(TileType::Wall) {
            return true;
        }
        if self
            .birds
            .iter()
            .any(|bird| bird.alive && bird.body.contains(&below))
        {
            return true;
        }
        self.grid.apples.contains(&below)
    }

    pub fn is_game_over(&self) -> bool {
        self.grid.apples.is_empty()
            || (0..=1).any(|owner| self.birds_for_player(owner).all(|bird| !bird.alive))
    }
}

fn birds_are_touching(a: &BirdState, b: &BirdState) -> bool {
    a.body
        .iter()
        .any(|left| b.body.iter().any(|right| left.manhattan_to(*right) == 1))
}

#[cfg(test)]
mod tests {
    use super::{BirdCommand, GameState, PlayerAction};
    use crate::{Coord, Direction, Grid, TileType};

    fn action(entries: &[(i32, BirdCommand)]) -> PlayerAction {
        let mut action = PlayerAction::default();
        for (bird_id, command) in entries {
            action.per_bird.insert(*bird_id, *command);
        }
        action
    }

    #[test]
    fn shared_apple_is_eaten_by_multiple_heads() {
        let mut grid = Grid::new(7, 6);
        for x in 0..7 {
            grid.set(Coord::new(x, 5), TileType::Wall);
        }
        grid.apples.push(Coord::new(3, 2));
        let mut state = GameState::new(grid);
        state.add_bird(
            0,
            0,
            vec![Coord::new(2, 2), Coord::new(1, 2), Coord::new(0, 2)],
            Some(Direction::East),
        );
        state.add_bird(
            1,
            1,
            vec![Coord::new(4, 2), Coord::new(5, 2), Coord::new(6, 2)],
            Some(Direction::West),
        );

        state.step(&PlayerAction::default(), &PlayerAction::default());

        assert!(state.grid.apples.is_empty());
        assert_eq!(state.birds[0].length(), 3);
        assert_eq!(state.birds[1].length(), 3);
        assert_eq!(state.losses, [1, 1]);
    }

    #[test]
    fn eating_removes_support_before_fall() {
        let mut grid = Grid::new(5, 6);
        for x in 0..5 {
            grid.set(Coord::new(x, 5), TileType::Wall);
        }
        grid.apples.push(Coord::new(2, 3));
        let mut state = GameState::new(grid);
        state.add_bird(
            0,
            0,
            vec![Coord::new(2, 2), Coord::new(2, 1), Coord::new(2, 0)],
            Some(Direction::South),
        );

        state.step(&PlayerAction::default(), &PlayerAction::default());

        let body: Vec<_> = state.birds[0].body.iter().copied().collect();
        assert_eq!(
            body,
            vec![
                Coord::new(2, 4),
                Coord::new(2, 3),
                Coord::new(2, 2),
                Coord::new(2, 1)
            ]
        );
    }

    #[test]
    fn beheading_threshold_matches_referee() {
        let mut grid = Grid::new(5, 5);
        for x in 0..5 {
            grid.set(Coord::new(x, 4), TileType::Wall);
        }
        grid.set(Coord::new(3, 2), TileType::Wall);
        grid.set(Coord::new(3, 1), TileType::Wall);
        let mut state = GameState::new(grid);
        state.add_bird(
            0,
            0,
            vec![Coord::new(2, 2), Coord::new(1, 2), Coord::new(0, 2)],
            Some(Direction::East),
        );
        state.add_bird(
            1,
            1,
            vec![
                Coord::new(2, 1),
                Coord::new(1, 1),
                Coord::new(0, 1),
                Coord::new(0, 0),
            ],
            Some(Direction::East),
        );

        state.step(&PlayerAction::default(), &PlayerAction::default());

        assert!(!state.birds[0].alive);
        assert_eq!(state.losses[0], 3);
        assert!(state.birds[1].alive);
        assert_eq!(state.birds[1].length(), 3);
        assert_eq!(state.losses[1], 1);
    }

    #[test]
    fn self_collision_bug_is_preserved() {
        let mut grid = Grid::new(6, 6);
        for x in 0..6 {
            grid.set(Coord::new(x, 5), TileType::Wall);
        }
        let mut state = GameState::new(grid);
        state.add_bird(
            0,
            0,
            vec![
                Coord::new(2, 2),
                Coord::new(2, 3),
                Coord::new(1, 3),
                Coord::new(1, 2),
            ],
            Some(Direction::West),
        );

        state.step(
            &action(&[(0, BirdCommand::Turn(Direction::South))]),
            &PlayerAction::default(),
        );

        assert!(state.birds[0].alive);
    }

    #[test]
    fn tie_break_uses_losses() {
        let mut grid = Grid::new(4, 4);
        for x in 0..4 {
            grid.set(Coord::new(x, 3), TileType::Wall);
        }
        let mut state = GameState::new(grid);
        state.add_bird(
            0,
            0,
            vec![Coord::new(0, 0), Coord::new(0, 1), Coord::new(0, 2)],
            Some(Direction::North),
        );
        state.add_bird(
            1,
            1,
            vec![Coord::new(3, 0), Coord::new(3, 1), Coord::new(3, 2)],
            Some(Direction::North),
        );
        state.losses = [0, 2];
        let result = state.step(&PlayerAction::default(), &PlayerAction::default());
        assert_eq!(result.body_scores[0], result.body_scores[1]);
        assert!(result.final_scores[0] > result.final_scores[1]);
    }

    #[test]
    fn turn_direction_resets_to_facing_each_turn() {
        let mut grid = Grid::new(8, 8);
        for x in 0..8 {
            grid.set(Coord::new(x, 7), TileType::Wall);
        }
        grid.set(Coord::new(2, 5), TileType::Wall);
        grid.set(Coord::new(4, 5), TileType::Wall);
        grid.apples.push(Coord::new(3, 2));
        let mut state = GameState::new(grid);
        state.add_bird(
            0,
            0,
            vec![Coord::new(2, 2), Coord::new(2, 3), Coord::new(2, 4)],
            None,
        );
        state.add_bird(
            1,
            1,
            vec![Coord::new(4, 2), Coord::new(4, 3), Coord::new(4, 4)],
            None,
        );

        state.step(
            &action(&[(0, BirdCommand::Turn(Direction::East))]),
            &action(&[(1, BirdCommand::Turn(Direction::West))]),
        );

        assert_eq!(
            state.birds[0].body.iter().copied().collect::<Vec<_>>(),
            vec![Coord::new(2, 2), Coord::new(2, 3), Coord::new(2, 4)]
        );
        assert_eq!(
            state.birds[1].body.iter().copied().collect::<Vec<_>>(),
            vec![Coord::new(4, 2), Coord::new(4, 3), Coord::new(4, 4)]
        );

        state.step(&PlayerAction::default(), &PlayerAction::default());

        assert_eq!(
            state.birds[0].body.iter().copied().collect::<Vec<_>>(),
            vec![Coord::new(2, 2), Coord::new(2, 3), Coord::new(2, 4)]
        );
        assert_eq!(
            state.birds[1].body.iter().copied().collect::<Vec<_>>(),
            vec![Coord::new(4, 2), Coord::new(4, 3), Coord::new(4, 4)]
        );
        assert_eq!(state.losses, [1, 1]);
    }
}
