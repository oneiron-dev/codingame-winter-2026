use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use snakebot_bot::config::{artifact_hash_file, behavior_hash_file, BotConfig};
use snakebot_bot::features::{encode_training_row, TrainingMetadata};
use snakebot_bot::search::SearchBudget;
use snakebot_bot::selfplay::{play_and_collect, SelfPlayConfig};
use snakebot_engine::{initial_state_from_seed, load_dump_records};

const DEFAULT_SCHEMA_VERSION: u32 = 3;

struct ExportConfig {
    maps_path: Option<PathBuf>,
    seed_start: i64,
    seed_count: usize,
    league: i32,
    out_path: PathBuf,
    limit: usize,
    max_turns: usize,
    budget: SearchBudget,
    shard_index: usize,
    num_shards: usize,
    git_sha: String,
    bot_config: BotConfig,
    config_artifact_hash: String,
    config_behavior_hash: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let mut writer = BufWriter::new(File::create(&config.out_path)?);
    let mut emitted = 0_usize;
    let mut games = 0_usize;

    if let Some(maps_path) = &config.maps_path {
        let records = load_dump_records(maps_path)?;
        for (index, record) in records.into_iter().enumerate() {
            if index >= config.limit {
                break;
            }
            if index % config.num_shards != config.shard_index {
                continue;
            }
            let game_id = format!("seed-{}-league-{}", record.seed, record.league_level);
            emitted += export_game(
                &mut writer,
                &config,
                record.seed,
                game_id,
                record.state.to_game_state(),
            )?;
            games += 1;
        }
    } else {
        let total = config.limit.min(config.seed_count);
        for index in 0..total {
            if index % config.num_shards != config.shard_index {
                continue;
            }
            let seed = config.seed_start + index as i64;
            let game_id = format!("seed-{}-league-{}", seed, config.league);
            emitted += export_game(
                &mut writer,
                &config,
                seed,
                game_id,
                initial_state_from_seed(seed, config.league),
            )?;
            games += 1;
        }
    }
    writer.flush()?;

    eprintln!(
        "exported {emitted} samples from {games} game(s) to {}",
        config.out_path.display()
    );
    Ok(())
}

fn export_game(
    writer: &mut BufWriter<File>,
    config: &ExportConfig,
    seed: i64,
    game_id: String,
    initial_state: snakebot_engine::GameState,
) -> Result<usize, Box<dyn Error>> {
    let collected = play_and_collect(
        initial_state,
        &SelfPlayConfig {
            max_turns: config.max_turns,
            budget: config.budget,
            bot_config: config.bot_config.clone(),
        },
    );
    let mut emitted = 0_usize;
    for position in collected.positions {
        let row = encode_training_row(
            &position.state,
            position.owner,
            &collected.final_result,
            TrainingMetadata {
                schema_version: DEFAULT_SCHEMA_VERSION,
                git_sha: config.git_sha.clone(),
                config_artifact_hash: config.config_artifact_hash.clone(),
                config_behavior_hash: config.config_behavior_hash.clone(),
                seed,
                game_id: game_id.clone(),
                turn: position.turn,
                chosen_action_id: position.outcome.action_id,
                joint_action_count: position.outcome.action_count,
                root_values: position.outcome.root_values,
                budget_type: config.budget.budget_type().to_owned(),
                budget_value: config.budget.budget_value(),
                search_stats: position.outcome.stats,
            },
        );
        serde_json::to_writer(&mut *writer, &row)?;
        writer.write_all(b"\n")?;
        emitted += 1;
    }
    Ok(emitted)
}

fn parse_args() -> Result<ExportConfig, Box<dyn Error>> {
    let mut maps_path = None;
    let mut seed_start = 1_i64;
    let mut seed_count = 0_usize;
    let mut league = 4_i32;
    let mut out_path = None;
    let mut limit = usize::MAX;
    let mut max_turns = 200;
    let mut budget = SearchBudget::ExtraNodesAfterRoot(5_000);
    let mut shard_index = 0_usize;
    let mut num_shards = 1_usize;
    let mut git_sha = "unknown".to_owned();
    let mut bot_config = BotConfig::embedded();
    let mut config_artifact_hash = BotConfig::embedded_artifact_hash().to_owned();
    let mut config_behavior_hash = BotConfig::embedded_behavior_hash().to_owned();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--maps" => {
                maps_path = Some(PathBuf::from(
                    args.next().ok_or("missing value for --maps")?,
                ))
            }
            "--seed-start" => {
                seed_start = args
                    .next()
                    .ok_or("missing value for --seed-start")?
                    .parse()?
            }
            "--seed-count" => {
                seed_count = args
                    .next()
                    .ok_or("missing value for --seed-count")?
                    .parse()?
            }
            "--league" => league = args.next().ok_or("missing value for --league")?.parse()?,
            "--out" => {
                out_path = Some(PathBuf::from(args.next().ok_or("missing value for --out")?))
            }
            "--limit" => limit = args.next().ok_or("missing value for --limit")?.parse()?,
            "--max-turns" => {
                max_turns = args
                    .next()
                    .ok_or("missing value for --max-turns")?
                    .parse()?
            }
            "--search-ms" => {
                budget = SearchBudget::TimeMs(
                    args.next()
                        .ok_or("missing value for --search-ms")?
                        .parse()?,
                )
            }
            "--extra-nodes-after-root" => {
                budget = SearchBudget::ExtraNodesAfterRoot(
                    args.next()
                        .ok_or("missing value for --extra-nodes-after-root")?
                        .parse()?,
                )
            }
            "--shard-index" => {
                shard_index = args
                    .next()
                    .ok_or("missing value for --shard-index")?
                    .parse()?
            }
            "--num-shards" => {
                num_shards = args
                    .next()
                    .ok_or("missing value for --num-shards")?
                    .parse()?
            }
            "--git-sha" => git_sha = args.next().ok_or("missing value for --git-sha")?,
            "--config" => {
                let path = args.next().ok_or("missing value for --config")?;
                config_artifact_hash = artifact_hash_file(&path)?;
                config_behavior_hash = behavior_hash_file(&path)?;
                bot_config = BotConfig::load(path)?;
            }
            _ => return Err(format!("unknown arg: {arg}").into()),
        }
    }

    if num_shards == 0 {
        return Err("num shards must be positive".into());
    }
    if shard_index >= num_shards {
        return Err(
            format!("shard index {shard_index} out of range for {num_shards} shards").into(),
        );
    }
    if maps_path.is_none() && seed_count == 0 {
        return Err("either --maps or --seed-count must be provided".into());
    }

    Ok(ExportConfig {
        maps_path,
        seed_start,
        seed_count,
        league,
        out_path: out_path.ok_or("missing required --out")?,
        limit,
        max_turns,
        budget,
        shard_index,
        num_shards,
        git_sha,
        bot_config,
        config_artifact_hash,
        config_behavior_hash,
    })
}
