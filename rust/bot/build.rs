use std::env;
use std::fs;
use std::path::PathBuf;

use serde_json::{json, Value};

fn main() {
    println!("cargo:rerun-if-env-changed=SNAKEBOT_CONFIG_PATH");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let default_path = manifest_dir.join("configs/submission_current.json");
    println!("cargo:rerun-if-changed={}", default_path.display());

    let config_path = env::var_os("SNAKEBOT_CONFIG_PATH")
        .map(PathBuf::from)
        .unwrap_or(default_path);
    let config_path = if config_path.is_absolute() {
        config_path
    } else if config_path.exists() {
        config_path
    } else if manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .map(|workspace_root| workspace_root.join(&config_path).exists())
        .unwrap_or(false)
    {
        manifest_dir
            .parent()
            .and_then(|path| path.parent())
            .expect("workspace root")
            .join(config_path)
    } else {
        manifest_dir.join(config_path)
    };
    let config_path = config_path
        .canonicalize()
        .unwrap_or_else(|err| panic!("failed to resolve embedded config path: {err}"));
    println!("cargo:rerun-if-changed={}", config_path.display());

    let raw = fs::read(&config_path).unwrap_or_else(|err| {
        panic!(
            "failed to read embedded config {}: {err}",
            config_path.display()
        )
    });
    println!(
        "cargo:rustc-env=SNAKEBOT_EMBEDDED_CONFIG_PATH={}",
        config_path.display()
    );
    println!(
        "cargo:rustc-env=SNAKEBOT_EMBEDDED_CONFIG_ARTIFACT_HASH={}",
        hash_bytes(&raw)
    );
    println!(
        "cargo:rustc-env=SNAKEBOT_EMBEDDED_CONFIG_BEHAVIOR_HASH={}",
        behavior_hash(&raw)
    );
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn behavior_hash(raw: &[u8]) -> String {
    let parsed: Value =
        serde_json::from_slice(raw).expect("embedded config should be valid json for behavior hash");
    let eval = parsed
        .get("eval")
        .cloned()
        .expect("embedded config must contain eval");
    let search = parsed
        .get("search")
        .cloned()
        .expect("embedded config must contain search");
    let canonical = json!({
        "eval": eval,
        "search": search,
    });
    let bytes = canonical_json_bytes(&canonical);
    hash_bytes(&bytes)
}

fn canonical_json_bytes(value: &Value) -> Vec<u8> {
    let mut out = String::new();
    write_canonical_json(value, &mut out);
    out.into_bytes()
}

fn write_canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            out.push_str(
                &serde_json::to_string(value).expect("scalar json value should serialize"),
            );
        }
        Value::Array(values) => {
            out.push('[');
            for (idx, entry) in values.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_canonical_json(entry, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            for (idx, key) in keys.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                out.push_str(
                    &serde_json::to_string(key).expect("json object key should serialize"),
                );
                out.push(':');
                write_canonical_json(&map[key], out);
            }
            out.push('}');
        }
    }
}
