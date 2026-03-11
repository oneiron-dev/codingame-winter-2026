use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rayon::prelude::*;
use serde::Serialize;
use snakebot_bot::config::{artifact_hash_file, behavior_hash_file, BotConfig};
use snakebot_bot::search::{choose_action, live_budget_for_turn};
use snakebot_engine::initial_state_from_seed;

#[derive(Clone)]
struct ArenaConfig {
    bot_a: BotConfig,
    bot_b: BotConfig,
    bot_a_artifact_hash: String,
    bot_b_artifact_hash: String,
    bot_a_behavior_hash: String,
    bot_b_behavior_hash: String,
    suite_path: PathBuf,
    league: i32,
    jobs: usize,
    max_turns: i32,
}

#[derive(Clone)]
struct MatchTask {
    seed: i64,
    bot_a_is_player_zero: bool,
}

#[derive(Clone, Debug)]
struct MatchSummary {
    bot_a_is_player_zero: bool,
    body_diff_a: i32,
    winner_a: Option<bool>,
    tiebreak_win_a: bool,
    turns: i32,
    p0_opening_elapsed_ms: Option<u64>,
    p1_opening_elapsed_ms: Option<u64>,
    p0_later_elapsed_ms: Vec<u64>,
    p1_later_elapsed_ms: Vec<u64>,
    p0_nodes: Vec<u64>,
    p1_nodes: Vec<u64>,
}

#[derive(Serialize)]
struct SideMetrics {
    avg_nodes_per_move: f64,
    opening_move_max_ms: f64,
    opening_move_p95_ms: f64,
    later_move_p50_ms: f64,
    later_move_p95_ms: f64,
    later_move_p99_ms: f64,
}

#[derive(Serialize)]
struct ArenaSummary {
    suite: String,
    suite_name: String,
    league: i32,
    jobs: usize,
    matches: usize,
    bot_a_name: String,
    bot_b_name: String,
    bot_a_artifact_hash: String,
    bot_b_artifact_hash: String,
    bot_a_behavior_hash: String,
    bot_b_behavior_hash: String,
    average_body_diff: f64,
    wins: usize,
    draws: usize,
    losses: usize,
    tiebreak_win_rate: f64,
    average_turns: f64,
    side_a: SideMetrics,
    side_b: SideMetrics,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let seeds = load_suite(&config.suite_path)?;
    let tasks = seeds
        .into_iter()
        .flat_map(|seed| {
            [
                MatchTask {
                    seed,
                    bot_a_is_player_zero: true,
                },
                MatchTask {
                    seed,
                    bot_a_is_player_zero: false,
                },
            ]
        })
        .collect::<Vec<_>>();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.jobs)
        .build()?;
    let matches = pool.install(|| {
        tasks
            .par_iter()
            .map(|task| run_match(&config, task))
            .collect::<Vec<_>>()
    });
    let summary = summarize(&config, &matches);
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn parse_args() -> Result<ArenaConfig, Box<dyn Error>> {
    let mut bot_a = BotConfig::embedded();
    let mut bot_b = BotConfig::embedded();
    let mut bot_a_artifact_hash = BotConfig::embedded_artifact_hash().to_owned();
    let mut bot_b_artifact_hash = BotConfig::embedded_artifact_hash().to_owned();
    let mut bot_a_behavior_hash = BotConfig::embedded_behavior_hash().to_owned();
    let mut bot_b_behavior_hash = BotConfig::embedded_behavior_hash().to_owned();
    let mut suite_path = None;
    let mut league = 4_i32;
    let mut jobs = std::thread::available_parallelism()
        .map(|value| value.get().saturating_sub(2).max(1))
        .unwrap_or(1);
    let mut max_turns = 200_i32;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bot-a-config" => {
                let path = args.next().ok_or("missing value for --bot-a-config")?;
                bot_a_artifact_hash = artifact_hash_file(&path)?;
                bot_a_behavior_hash = behavior_hash_file(&path)?;
                bot_a = BotConfig::load(path)?
            }
            "--bot-b-config" => {
                let path = args.next().ok_or("missing value for --bot-b-config")?;
                bot_b_artifact_hash = artifact_hash_file(&path)?;
                bot_b_behavior_hash = behavior_hash_file(&path)?;
                bot_b = BotConfig::load(path)?
            }
            "--suite" => {
                suite_path = Some(PathBuf::from(
                    args.next().ok_or("missing value for --suite")?,
                ))
            }
            "--league" => league = args.next().ok_or("missing value for --league")?.parse()?,
            "--jobs" => jobs = args.next().ok_or("missing value for --jobs")?.parse()?,
            "--max-turns" => {
                max_turns = args
                    .next()
                    .ok_or("missing value for --max-turns")?
                    .parse()?
            }
            _ => return Err(format!("unknown arg: {arg}").into()),
        }
    }

    Ok(ArenaConfig {
        bot_a,
        bot_b,
        bot_a_artifact_hash,
        bot_b_artifact_hash,
        bot_a_behavior_hash,
        bot_b_behavior_hash,
        suite_path: suite_path.ok_or("missing required --suite")?,
        league,
        jobs: jobs.max(1),
        max_turns,
    })
}

fn load_suite(path: &PathBuf) -> Result<Vec<i64>, Box<dyn Error>> {
    let raw = fs::read_to_string(path)?;
    let mut seeds = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        seeds.push(line.parse()?);
    }
    if seeds.is_empty() {
        return Err(format!("suite has no seeds: {}", path.display()).into());
    }
    Ok(seeds)
}

fn run_match(config: &ArenaConfig, task: &MatchTask) -> MatchSummary {
    let mut state = initial_state_from_seed(task.seed, config.league);
    let mut p0_opening_elapsed_ms = None;
    let mut p1_opening_elapsed_ms = None;
    let mut p0_later_elapsed_ms = Vec::new();
    let mut p1_later_elapsed_ms = Vec::new();
    let mut p0_nodes = Vec::new();
    let mut p1_nodes = Vec::new();

    while !state.is_terminal(config.max_turns) {
        let turn = state.turn;
        let p0_config = if task.bot_a_is_player_zero {
            &config.bot_a
        } else {
            &config.bot_b
        };
        let p1_config = if task.bot_a_is_player_zero {
            &config.bot_b
        } else {
            &config.bot_a
        };

        let p0 = choose_action(
            &state,
            0,
            p0_config,
            live_budget_for_turn(p0_config, state.turn),
        );
        let p1 = choose_action(
            &state,
            1,
            p1_config,
            live_budget_for_turn(p1_config, state.turn),
        );
        if turn == 0 {
            p0_opening_elapsed_ms = Some(p0.stats.elapsed_ms);
            p1_opening_elapsed_ms = Some(p1.stats.elapsed_ms);
        } else {
            p0_later_elapsed_ms.push(p0.stats.elapsed_ms);
            p1_later_elapsed_ms.push(p1.stats.elapsed_ms);
        }
        p0_nodes.push(p0.stats.root_pairs + p0.stats.extra_nodes);
        p1_nodes.push(p1.stats.root_pairs + p1.stats.extra_nodes);
        state.step(&p0.action, &p1.action);
    }

    let result = state.final_result(config.max_turns);
    let body_diff_a = if task.bot_a_is_player_zero {
        result.body_scores[0] - result.body_scores[1]
    } else {
        result.body_scores[1] - result.body_scores[0]
    };
    let winner_a = result.winner.map(|winner| {
        if task.bot_a_is_player_zero {
            winner == 0
        } else {
            winner == 1
        }
    });
    let tiebreak_win_a = body_diff_a == 0 && winner_a == Some(true);

    MatchSummary {
        bot_a_is_player_zero: task.bot_a_is_player_zero,
        body_diff_a,
        winner_a,
        tiebreak_win_a,
        turns: state.turn,
        p0_opening_elapsed_ms,
        p1_opening_elapsed_ms,
        p0_later_elapsed_ms,
        p1_later_elapsed_ms,
        p0_nodes,
        p1_nodes,
    }
}

fn summarize(config: &ArenaConfig, matches: &[MatchSummary]) -> ArenaSummary {
    let mut wins = 0_usize;
    let mut draws = 0_usize;
    let mut losses = 0_usize;
    let mut tiebreak_opportunities = 0_usize;
    let mut tiebreak_wins = 0_usize;
    let mut total_body_diff = 0_i64;
    let mut total_turns = 0_i64;
    let mut side_a_opening_times = Vec::new();
    let mut side_b_opening_times = Vec::new();
    let mut side_a_later_times = Vec::new();
    let mut side_b_later_times = Vec::new();
    let mut side_a_nodes = Vec::new();
    let mut side_b_nodes = Vec::new();

    for summary in matches {
        total_body_diff += i64::from(summary.body_diff_a);
        total_turns += i64::from(summary.turns);
        match summary.winner_a {
            Some(true) => wins += 1,
            Some(false) => losses += 1,
            None => draws += 1,
        }
        if summary.body_diff_a == 0 {
            tiebreak_opportunities += 1;
            if summary.tiebreak_win_a {
                tiebreak_wins += 1;
            }
        }

        if summary.bot_a_is_player_zero {
            side_a_opening_times.extend(summary.p0_opening_elapsed_ms.iter().map(|value| *value as f64));
            side_b_opening_times.extend(summary.p1_opening_elapsed_ms.iter().map(|value| *value as f64));
            side_a_later_times.extend(summary.p0_later_elapsed_ms.iter().map(|value| *value as f64));
            side_b_later_times.extend(summary.p1_later_elapsed_ms.iter().map(|value| *value as f64));
            side_a_nodes.extend(summary.p0_nodes.iter().map(|value| *value as f64));
            side_b_nodes.extend(summary.p1_nodes.iter().map(|value| *value as f64));
        } else {
            side_a_opening_times.extend(summary.p1_opening_elapsed_ms.iter().map(|value| *value as f64));
            side_b_opening_times.extend(summary.p0_opening_elapsed_ms.iter().map(|value| *value as f64));
            side_a_later_times.extend(summary.p1_later_elapsed_ms.iter().map(|value| *value as f64));
            side_b_later_times.extend(summary.p0_later_elapsed_ms.iter().map(|value| *value as f64));
            side_a_nodes.extend(summary.p1_nodes.iter().map(|value| *value as f64));
            side_b_nodes.extend(summary.p0_nodes.iter().map(|value| *value as f64));
        }
    }

    ArenaSummary {
        suite: config.suite_path.display().to_string(),
        suite_name: config
            .suite_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("unknown_suite")
            .to_owned(),
        league: config.league,
        jobs: config.jobs,
        matches: matches.len(),
        bot_a_name: config.bot_a.name.clone(),
        bot_b_name: config.bot_b.name.clone(),
        bot_a_artifact_hash: config.bot_a_artifact_hash.clone(),
        bot_b_artifact_hash: config.bot_b_artifact_hash.clone(),
        bot_a_behavior_hash: config.bot_a_behavior_hash.clone(),
        bot_b_behavior_hash: config.bot_b_behavior_hash.clone(),
        average_body_diff: total_body_diff as f64 / matches.len().max(1) as f64,
        wins,
        draws,
        losses,
        tiebreak_win_rate: if tiebreak_opportunities == 0 {
            0.0
        } else {
            tiebreak_wins as f64 / tiebreak_opportunities as f64
        },
        average_turns: total_turns as f64 / matches.len().max(1) as f64,
        side_a: build_side_metrics(&side_a_opening_times, &side_a_later_times, &side_a_nodes),
        side_b: build_side_metrics(&side_b_opening_times, &side_b_later_times, &side_b_nodes),
    }
}

fn build_side_metrics(opening_times: &[f64], later_times: &[f64], nodes: &[f64]) -> SideMetrics {
    SideMetrics {
        avg_nodes_per_move: mean(nodes),
        opening_move_max_ms: max_value(opening_times),
        opening_move_p95_ms: percentile(opening_times, 0.95),
        later_move_p50_ms: percentile(later_times, 0.50),
        later_move_p95_ms: percentile(later_times, 0.95),
        later_move_p99_ms: percentile(later_times, 0.99),
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn max_value(values: &[f64]) -> f64 {
    values.iter().copied().max_by(|left, right| left.total_cmp(right)).unwrap_or(0.0)
}

fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let index = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[index]
}
