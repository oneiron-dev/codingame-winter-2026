use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use snakebot_bot::selfplay::{play_and_collect, SelfPlayConfig};
use snakebot_engine::load_dump_records;

struct ExportConfig {
    maps_path: PathBuf,
    out_path: PathBuf,
    limit: usize,
    max_turns: usize,
    search_time_ms: u64,
    shard_index: usize,
    num_shards: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let records = load_dump_records(&config.maps_path)?;
    let mut writer = BufWriter::new(File::create(&config.out_path)?);
    let mut emitted = 0_usize;
    let mut games = 0_usize;

    for (index, record) in records.into_iter().enumerate() {
        if index >= config.limit {
            break;
        }
        if index % config.num_shards != config.shard_index {
            continue;
        }
        let rows = play_and_collect(
            record.state.to_game_state(),
            SelfPlayConfig {
                max_turns: config.max_turns,
                search_time_ms: config.search_time_ms,
            },
        );
        for row in rows {
            serde_json::to_writer(&mut writer, &row)?;
            writer.write_all(b"\n")?;
            emitted += 1;
        }
        games += 1;
    }
    writer.flush()?;

    eprintln!(
        "exported {emitted} samples from {games} game(s) to {}",
        config.out_path.display()
    );
    Ok(())
}

fn parse_args() -> Result<ExportConfig, Box<dyn Error>> {
    let mut maps_path = None;
    let mut out_path = None;
    let mut limit = usize::MAX;
    let mut max_turns = 200;
    let mut search_time_ms = 0_u64;
    let mut shard_index = 0_usize;
    let mut num_shards = 1_usize;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--maps" => maps_path = Some(PathBuf::from(args.next().ok_or("missing value for --maps")?)),
            "--out" => out_path = Some(PathBuf::from(args.next().ok_or("missing value for --out")?)),
            "--limit" => limit = args.next().ok_or("missing value for --limit")?.parse()?,
            "--max-turns" => max_turns = args.next().ok_or("missing value for --max-turns")?.parse()?,
            "--search-ms" => {
                search_time_ms = args.next().ok_or("missing value for --search-ms")?.parse()?
            }
            "--shard-index" => {
                shard_index = args.next().ok_or("missing value for --shard-index")?.parse()?
            }
            "--num-shards" => {
                num_shards = args.next().ok_or("missing value for --num-shards")?.parse()?
            }
            _ => return Err(format!("unknown arg: {arg}").into()),
        }
    }

    if num_shards == 0 {
        return Err("num shards must be positive".into());
    }
    if shard_index >= num_shards {
        return Err(format!("shard index {shard_index} out of range for {num_shards} shards").into());
    }

    Ok(ExportConfig {
        maps_path: maps_path.ok_or("missing required --maps")?,
        out_path: out_path.ok_or("missing required --out")?,
        limit,
        max_turns,
        search_time_ms,
        shard_index,
        num_shards,
    })
}
