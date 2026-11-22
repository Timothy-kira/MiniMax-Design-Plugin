"""Local queue, official Colab transport, artifact hook, and independent watchdog."""
import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from queue import Empty, Queue
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import uuid

import managed_cli
from model_cards import (
    GPU_MEMORY, HERE, SHA256, build_lora, list_lora_registrations, load_cards, make_plan, read_json, sha256,
    validate_card,
)

LEGACY_ROOT = Path.home() / ".h3-colab"
DATA_DIR = LEGACY_ROOT if LEGACY_ROOT.exists() else Path.home() / ".minimax-design-colab"
ROOT = Path(os.environ.get("MINIMAX_DESIGN_COLAB_HOME") or os.environ.get("H3_COLAB_HOME") or DATA_DIR).resolve()
JOB_ID = re.compile(r"h3-[a-f0-9]{24}")
MARK = "H3COLAB "
REMOTE_SETUP_EVENT_FIELDS = {
    "gpu_verified": ("gpu", "vram_gib", "free_vram_gib", "ram_gib"),
    "preflight_ready": (),
    "environment_setup_started": ("shared_cache", "revision"),
    "environment_ready": ("cache_hit", "revision"),
    "setup_failed": ("error_kind",),
    "assets_ready": ("count", "shared_cache"),
    "prepared": ("asset_count", "environment_cache_hit"),
}
COLAB_ALLOCATION_TIMEOUT = 390
COLAB_ALLOCATION_SETTLE = 15
COLAB_PREFLIGHT_RESERVE = 45
CLI_TRANSPORTS = ("auto", "native", "configured")
RUNTIME_SHAPES = ("auto", "standard", "high_mem")
QUEUE_HOOK_MODES = ("events", "notify", "off")
TERMINAL = {"completed", "failed", "cancelled", "cleanup_pending", "review_pending"}
CAPACITY = re.compile(r"RESOURCE_EXHAUSTED|no available (?:runtime|accelerator|gpu)|"
                      r"(?:accelerator|gpu|runtime) (?:is )?unavailable|allocation (?:capacity|quota)|"
                      r"(?:not enough|insufficient) (?:resources|capacity)|no compatible runtime spec|"
                      r"^\[colab\] (?:Backend rejected accelerator '[A-Za-z0-9]+'\.|"
                      r"Allocation refused \(precondition failed\)\.)", re.I | re.M)
COLAB_BACKEND_HTTP_STATUS = {"Bad Gateway": 502, "Service Unavailable": 503, "Gateway Timeout": 504}
COLAB_ASSIGN_BACKEND_ERROR = re.compile(
    r"^ColabRequestError: Failed to issue request (?:GET|POST)\b[^\r\n]*"
    r"(?:\r?\n[^\r\n]+){0,3}?:[ \t]*(Bad Gateway|Service Unavailable|Gateway Timeout)[ \t\r]*$", re.M)
COLAB_ASSIGN_DISCONNECT = re.compile(
    r"^(?:requests\.exceptions\.)?ConnectionError:[ \t]+\('Connection aborted\.',\s*"
    r"RemoteDisconnected\('Remote end closed\s+connection without response'\)\)", re.M)
COLAB_SESSION_ROW = re.compile(r"\[([^\]]+)\] .+ \| Hardware: [^|]+ \| Shape: [^|]+ \| Variant: \S+")
COLAB_PRUNED_SESSIONS = re.compile(r"\[colab\] Pruned \d+ stale local session\(s\)\.")


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".new-", mode="w",
                                     encoding="utf-8", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        for delay in (0, 0.02, 0.1, 0.25, 0.5):
            time.sleep(delay)
            try:
                temp.replace(path)
                break
            except PermissionError:
                if delay == 0.5:
                    raise
    finally:
        temp.unlink(missing_ok=True)


def settings():
    path = ROOT / "settings.json"
    return read_json(path) if path.exists() else {}


def configure_runtime(**changes):
    config = settings()
    config.update(changes)
    if "queue_hook" in changes and config["queue_hook"] not in QUEUE_HOOK_MODES:
        raise ValueError("queue_hook must be events, notify, or off")
    gpus = config.get("gpus", [])
    if "gpus" in changes and (not gpus or len(gpus) != len(set(gpus)) or any(gpu not in GPU_MEMORY for gpu in gpus)):
        raise ValueError("Configure an ordered list of distinct Colab GPU names")
    for key in ("cache_dir", "wheelhouse"):
        if config.get(key) and (not PurePosixPath(config[key]).is_absolute() or ".." in PurePosixPath(config[key]).parts):
            raise ValueError(f"{key} must be an absolute path in the Linux Colab VM")
    drive = config.get("drive_cache_relative")
    if drive and (not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in drive.split("/"))
                  or config.get("cache_dir")):
        raise ValueError("Drive caching needs a relative folder name and is exclusive with cache_dir")
    if config.get("offline") and (drive or not config.get("cache_dir")):
        raise ValueError("Offline setup needs an already-mounted cache_dir with an executable environment")
    if config.get("design_executable") and not Path(config["design_executable"]).is_absolute():
        raise ValueError("Design launcher path must be absolute")
    if "asset_workers" in config and (type(config["asset_workers"]) is not int or not 1 <= config["asset_workers"] <= 6):
        raise ValueError("asset_workers must be an integer from 1 through 6")
    if changes:
        save(ROOT / "settings.json", config)
    return config


def backend():
    override = os.environ.get("H3_COLAB_RELAX_MCP")
    if override:
        roots = [Path(override)]
    else:
        cache = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "plugins/cache"
        def version(path):
            match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", path.name)
            return tuple(map(int, match.groups())) if match else (0, 0, 0)
        roots = [p / "mcp" for p in sorted(cache.glob("*/relax-agent/*"), key=version, reverse=True)]
        roots.append(Path.home() / ".gemini/config/plugins/relax-agent/mcp")
    folder = next((p.resolve() for p in roots if (p / "colab_transport.py").is_file()
                   and (p / "resource_pool.py").is_file()), None)
    if folder is None:
        raise RuntimeError("Install Relax Agent with the official Colab CLI transport")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import colab_accounts
    import colab_transport
    import resource_pool
    return colab_accounts, colab_transport, resource_pool


def directory(job_id):
    if not JOB_ID.fullmatch(job_id):
        raise ValueError("Invalid MiniMax Design Colab job ID")
    return ROOT / "jobs" / job_id


def status(job_id):
    job = directory(job_id)
    value = read_json(job / "status.json")
    if value.get("segments"):
        progress = job / "segment-outputs.json"
        value["segment_outputs"] = read_json(progress) if progress.exists() else []
    if value["phase"] == "review_pending":
        kept = job / "runtime-review-keep.json"
        if kept.exists():
            value["runtime_review"].update(read_json(kept))
        value["runtime_review"]["release_requested"] = (job / "runtime-review-release.json").exists()
    hook = job / "queue-hook.json"
    if hook.is_file():
        value.update(queue_hook=read_json(hook), queue_hook_path=str(hook))
    return {**value, "log_path": str(job / "run.log"), "events_path": str(job / "events.jsonl"),
            "cancel_requested": (job / "cancel").exists()}


def snapshot_profile(profile):
    keys = ("id", "verified_email", "credential_file", "transport", "config_path",
            "distribution", "executable", "network_route_sha256")
    data = [profile[k] for k in keys if k in profile]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def list_cards():
    return [{"id": card["id"], "mode": card.get("mode", "custom"), "native_h3": bool(card.get("native_h3")),
             "variants": [{"id": v["id"], "precision": v["precision"], "min_vram_gib": v["min_vram_gib"],
                           "min_ram_gib": v["min_ram_gib"],
                           "components": {key: asset["path"] for key, asset in v["assets"].items()}}
                          for v in card["variants"]],
             "loras": list(card.get("loras", {}))} for card in load_cards(ROOT).values()]


def register_card(path, replace=False):
    source = Path(path)
    if not source.is_absolute() or not source.is_file():
        raise ValueError("Provide the absolute path of a local model-card JSON file")
    card = validate_card(read_json(source))
    destination = ROOT / "cards" / (card["id"] + ".json")
    if card["id"] in load_cards(ROOT) and not replace:
        raise ValueError("A card with this ID already exists. Set replace only to update it.")
    save(destination, card)
    return {"id": card["id"], "path": str(destination)}


def list_loras():
    return list_lora_registrations(ROOT)


def register_lora(*, replace=False, **arguments):
    registration = build_lora(ROOT, **arguments)
    destination = ROOT / "loras" / registration["card_id"] / (registration["lora_id"] + ".json")
    if destination.exists() and replace is not True:
        raise ValueError("This model card already has the LoRA registered. Set replace only to update it.")
    save(destination, registration)
    return {**registration, "path": str(destination), "gpu_load_verified": False}


def account(action="list", **arguments):
    accounts, transport, _ = backend()
    options = {"list", "add", "auth", "verify", "use", "rename", "remove", "usage"}
    if action not in options:
        raise ValueError("Unknown Colab account action")
    if action == "add" and set(arguments) - {"name", "credential_file", "config_path",
            "expected_email", "transport", "distribution", "executable"}:
        raise ValueError("Provide ADC file paths only. Credential contents are never passed through MCP.")
    if action == "add" and arguments.get("transport", "native") == "native" and os.name == "nt":
        name = accounts._name(arguments.get("name"))
        profile_id = uuid.uuid4().hex
        shell_config = ROOT / "accounts" / profile_id
        credential = Path(arguments.get("credential_file") or shell_config / "application_default_credentials.json")
        expected = arguments.get("expected_email") or (name if "@" in name else "")
        if not credential.is_absolute() or (expected and not accounts.EMAIL_RE.fullmatch(expected)):
            raise ValueError("Supply an absolute ADC file path and a valid expected Google email")
        with accounts._transaction() as data:
            if name in data["accounts"] or any(Path(p["credential_file"]) == credential
                                               for p in data["accounts"].values()):
                raise ValueError("This Colab account name or ADC file is already registered")
            data["accounts"][name] = {"name": name, "id": profile_id, "credential_file": str(credential),
                                      "config_path": str(shell_config / "sessions.json"), "transport": "native",
                                      "distribution": "", "executable": "colab", "expected_email": expected.lower(),
                                      "verified_email": "", "network_route_spec": "", "network_route_sha256": ""}
            if not data.get("active"):
                data["active"] = name
        return {"status": "added", "account": name, "verified": False}
    if action in {"auth", "verify"}:
        profile = accounts.resolve(arguments.get("name", ""))
        if action == "verify" and managed_cli.supports(profile):
            version = managed_cli.latest() or managed_cli.BASELINE
            client = managed_cli.NativeCLI(ROOT, profile, version, transport)
            answer = client.run_colab(profile["name"], ["whoami"], timeout_seconds=28)
            if answer["exit_code"] == 0:
                google = accounts.parse_whoami(answer["stdout"])
                expected = profile.get("expected_email") or profile.get("verified_email")
                if expected and google["email"] != expected:
                    raise RuntimeError("ADC belongs to another Google identity")
                with accounts._transaction() as data:
                    current = data["accounts"].get(profile["name"])
                    if not current or snapshot_profile(current) != snapshot_profile(profile):
                        raise RuntimeError("Profile changed while its identity was checked")
                    if any(p["id"] != profile["id"] and p.get("verified_email") == google["email"]
                           for p in data["accounts"].values()):
                        raise RuntimeError("This Google identity is already registered")
                    if accounts._has_claim(profile["id"]) and current.get("verified_email") != google["email"]:
                        raise RuntimeError("An active job owns this profile")
                    from datetime import datetime, timezone
                    current.update(verified_email=google["email"], verified_at=datetime.now(timezone.utc).isoformat())
                return {"status": "verified", "account": profile["name"], **google}
        if action == "auth" and profile["transport"] == "native":
            config = ROOT / "accounts" / profile["id"]
            return {"account": profile["name"], "human_authorization_required": True,
                    "command_argv": ["gcloud", "auth", "application-default", "login",
                                     "--scopes=" + ",".join(accounts.SCOPES)],
                    "environment": {"CLOUDSDK_CONFIG": str(config)},
                    "generated_adc_file": str(config / "application_default_credentials.json"),
                    "expected_adc_file": profile["credential_file"]}
    if action == "usage":
        cli_transport = validate_cli_transport(arguments.get("cli_transport", "auto"))
        profile = accounts.resolve(arguments.get("name", ""))
        client, _ = select_cli(profile, accounts, transport, cli_transport=cli_transport)
        reply = client.run_colab(profile["name"], ["usage"], timeout_seconds=20)
        lines = [line for line in reply["stdout"].splitlines()
                 if line.startswith(("Current balance:", "Usage rate:", "Active assignments:"))]
        return {"account": profile["name"], "identity": profile.get("verified_email"), "usage": lines,
                "verified": reply["exit_code"] == 0 and bool(lines)}
    result = accounts.manage({"action": action, **arguments})
    if action == "list":
        result["accounts"] = [{key: value for key, value in profile.items()
                               if key in {"name", "transport", "verified_email", "verified_at"}}
                              for profile in result["accounts"]]
    if action == "add" and isinstance(result.get("account"), dict):
        result["account"] = result["account"]["name"]
    if "reason" in result:
        result["reason"] = transport.redact(result["reason"])[-2000:]
    return result


def validate_cli_transport(value):
    if value not in CLI_TRANSPORTS:
        raise ValueError("cli_transport must be auto, native, or configured")
    return value


def parse_cli_version(stdout):
    return next((line.partition(":")[2].strip() for line in stdout.splitlines()
                 if line.strip().startswith("Version:")), "")


def select_cli(profile, accounts, legacy, bound=None, *, cli_transport="auto"):
    validate_cli_transport(cli_transport)
    if ((cli_transport == "native" and bound == "legacy")
            or (cli_transport == "configured" and bound not in (None, "legacy"))):
        raise RuntimeError("The saved Colab CLI binding conflicts with the planned transport")
    if bound == "legacy":
        return legacy, "legacy"
    if bound:
        return managed_cli.NativeCLI(ROOT, profile, bound, legacy), bound
    if cli_transport == "configured" or (cli_transport == "auto" and profile.get("transport") == "wsl"):
        return legacy, "legacy"
    if managed_cli.supports(profile):
        version = managed_cli.latest()
        for candidate in dict.fromkeys(x for x in (version, managed_cli.BASELINE) if x):
            try:
                native = managed_cli.NativeCLI(ROOT, profile, candidate, legacy)
                reply = native.run_colab(profile["name"], ["whoami"], timeout_seconds=25)
                if (reply["exit_code"] == 0 and
                        accounts.parse_whoami(reply["stdout"])["email"] == profile["verified_email"]):
                    return native, candidate
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
    if cli_transport == "native" or (os.name == "nt" and profile["transport"] == "native"):
        raise RuntimeError("The native CLI could not verify this account. Check ADC and the official CLI version.")
    return legacy, "legacy"


def check_cli_update(name="", cli_transport="auto"):
    validate_cli_transport(cli_transport)
    accounts, legacy, _ = backend()
    profile = accounts.resolve(name)
    client, selected = select_cli(profile, accounts, legacy, cli_transport=cli_transport)
    reply = client.run_colab(profile["name"], ["version"], timeout_seconds=18)
    installed = parse_cli_version(reply["stdout"])
    result = {"version": installed, "latest": managed_cli.latest(), "native": selected != "legacy",
              "cli_transport": cli_transport, "exit_code": reply["exit_code"], "checked_at": time.time()}
    save(ROOT / "cli-update.json", result)
    return result


def doctor(name="", live=False, cli_transport="auto"):
    validate_cli_transport(cli_transport)
    accounts, transport, _ = backend()
    state = transport.doctor(name)
    keys = ("status", "account", "transport", "cli_installed", "adc_file_exists", "verified_email", "reason")
    result = {key: state[key] for key in keys if key in state}
    if live:
        profile = accounts.resolve(name)
        client, selected = select_cli(profile, accounts, transport, cli_transport=cli_transport)
        reply = client.run_colab(profile["name"], ["whoami"], timeout_seconds=20)
        result["cli_authenticated"] = reply["exit_code"] == 0 and (
            accounts.parse_whoami(reply["stdout"])["email"] == profile.get("verified_email"))
        result["active_cli"] = f"native {selected}" if selected != "legacy" else profile["transport"]
        if result["cli_authenticated"]:
            version_reply = client.run_colab(profile["name"], ["version"], timeout_seconds=15)
            cli_version = parse_cli_version(version_reply["stdout"]) if version_reply["exit_code"] == 0 else ""
            if cli_version or selected != "legacy":
                result["cli_version"] = cli_version or selected
            if selected != "legacy":
                result.update(status="ready", cli_installed=True, adc_file_exists=True)
            elif result.get("status") == "ready_for_identity_check":
                result["status"] = "ready"
    configuration = settings()
    result.update(home=str(ROOT), cold_start=not bool(configuration.get("cache_dir") or configuration.get("drive_cache_relative")),
                  cache_dir=configuration.get("cache_dir"), offline=configuration.get("offline", False))
    return result


def plan_video(**arguments):
    cli_transport = validate_cli_transport(arguments.get("cli_transport", "auto"))
    runtime_shape = arguments.get("runtime_shape", "auto")
    if runtime_shape not in RUNTIME_SHAPES:
        raise ValueError("runtime_shape must be auto, standard, or high_mem")
    config = settings()
    queue_hook_mode = arguments.get("queue_hook", config.get("queue_hook", "events"))
    if queue_hook_mode not in QUEUE_HOOK_MODES:
        raise ValueError("queue_hook must be events, notify, or off")
    plan = make_plan(ROOT, arguments, config)
    preflight_only = arguments.get("preflight_only", False)
    if type(preflight_only) is not bool:
        raise ValueError("preflight_only must be a boolean")
    hold_for_generation = arguments.get("hold_for_generation", False)
    if type(hold_for_generation) is not bool or preflight_only and hold_for_generation:
        raise ValueError("Choose either standalone preflight or preparation on a retained runtime")
    accounts, _, _ = backend()
    profile = accounts.resolve(arguments.get("account", ""))
    if not profile.get("verified_email"):
        raise ValueError("Verify the saved Colab account before planning a run")
    segment_count = len(plan.get("segments", ())) or 1
    default_timeout = (COLAB_ALLOCATION_TIMEOUT + COLAB_ALLOCATION_SETTLE + COLAB_PREFLIGHT_RESERVE
                       if preflight_only else min(10800, 1200 + 1500 * segment_count))
    timeout = arguments.get("timeout_seconds", default_timeout)
    setup = arguments.get("setup_timeout_seconds", min(timeout, 1200) if type(timeout) is int else 1200)
    if type(timeout) is not int or type(setup) is not int or not 60 <= setup <= timeout <= 10800:
        raise ValueError("Expected 60 ≤ setup_timeout_seconds ≤ timeout_seconds ≤ 10800")
    runtime_options = {k: config[k] for k in ("cache_dir", "offline", "wheelhouse", "asset_workers") if k in config}
    plan["drive_cache_relative"] = config.get("drive_cache_relative", "")
    for key in ("cache_dir", "wheelhouse"):
        if runtime_options.get(key) and not PurePosixPath(runtime_options[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute Colab filesystem path")
    requested_id = "h3-" + uuid.uuid4().hex[:24]
    job = directory(requested_id)
    plan.update(job_id=requested_id, account=profile["name"], profile_id=profile["id"],
                profile_fingerprint=snapshot_profile(profile), timeout_seconds=timeout,
                runner_sha256=sha256(HERE / "runtime_remote.py"),
                setup_timeout_seconds=setup, runtime_options=runtime_options, preflight_only=preflight_only,
                loras=arguments.get("loras", {}), cli_transport=cli_transport, runtime_shape=runtime_shape,
                hold_for_generation=hold_for_generation, queue_hook=queue_hook_mode)
    save(job / "plan.json", plan)
    digest = sha256(job / "plan.json")
    batch = {"segments": plan["segments"]} if "segments" in plan else {}
    save(job / "status.json", {"job_id": requested_id, "phase": "planned", "plan_sha256": digest,
                                "runtime_released": None, "updated_at": time.time(),
                                "preflight_only": preflight_only, "cli_transport": cli_transport,
                                "runtime_shape": runtime_shape, "hold_for_generation": hold_for_generation,
                                "audio_mode": plan["audio_mode"], "continuation": plan["continuation"],
                                "takes": plan["takes"], **batch})
    return {k: plan[k] for k in ("job_id", "card", "loras", "width", "height", "seconds", "account",
                                 "timeout_seconds", "setup_timeout_seconds", "preflight_only", "cli_transport",
                                 "runtime_shape", "hold_for_generation", "queue_hook", "audio_mode", "continuation",
                                 "takes")} | {
        "plan_sha256": digest,
        "candidates": [{"gpu": c["gpu"], "variant": c["variant"], "precision": c["precision"],
                        "total_model_bytes": sum(a["size_bytes"] for a in c["assets"]),
                        "upload_paths": [] if preflight_only else [a["local_path"] for a in c["uploads"]]}
                       for c in plan["candidates"]],
        "setup_mode": "drive_cache" if plan["drive_cache_relative"] else "offline_cache" if runtime_options.get("offline") else
                      "persistent_cache" if runtime_options.get("cache_dir") else "cold_boot",
        "charged": False, **batch}


class _DetachedProcess:
    """Minimal Popen-compatible handle for a WMI-spawned Windows process."""

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        if os.name != "nt":
            return None
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, self.pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return 1
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259:  # STILL_ACTIVE
                return None
            return int(code.value)
        finally:
            kernel32.CloseHandle(handle)


def _spawn_detached_via_wmi(argv, cwd, log_path):
    """Spawn a Windows background process via WMI so it escapes a restrictive parent Job Object."""
    exe = Path(argv[0])
    pythonw = exe.with_name("pythonw.exe")
    cmd_argv = [str(pythonw if exe.name.lower() == "python.exe" and pythonw.is_file() else exe),
                *argv[1:], "--log-file", str(log_path)]
    cmdline = subprocess.list2cmdline(cmd_argv)
    ps = (
        "$ProgressPreference='SilentlyContinue'\n"
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{"
        f"CommandLine={cmdline!r}; CurrentDirectory={str(cwd)!r}"
        "}\n"
        'Write-Output "$($r.ReturnValue) $($r.ProcessId)"\n'
    )
    encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
    reply = subprocess.run(
        ["powershell", "-NoProfile", "-EncodedCommand", encoded],
        capture_output=True, text=True, timeout=20,
    )
    parts = reply.stdout.strip().split()
    if reply.returncode == 0 and len(parts) == 2 and parts[0] == "0" and parts[1].isdigit() and int(parts[1]) > 0:
        return _DetachedProcess(int(parts[1]))
    raise OSError(f"Win32_Process.Create failed: rc={reply.returncode} out={reply.stdout.strip()!r}")


def background(command, job, log_name):
    argv = [sys.executable, "-B", str(Path(__file__).resolve()), command, job.name]
    log_path = job / log_name
    with log_path.open("ab") as log:
        if os.name != "nt":
            return subprocess.Popen(argv, cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    start_new_session=True)
        base_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        detached = base_flags | getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        try:
            return subprocess.Popen(argv, cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    creationflags=detached, close_fds=True)
        except OSError:
            try:
                return _spawn_detached_via_wmi(argv, HERE, log_path)
            except (OSError, subprocess.SubprocessError):
                return subprocess.Popen(argv, cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                        creationflags=base_flags, close_fds=True)


def start_video(job_id, plan_sha256, authorize_cloud):
    job = directory(job_id)
    if authorize_cloud is not True:
        raise ValueError("Cloud execution needs explicit authorization for the chosen plan and assets")
    if sha256(job / "plan.json") != plan_sha256:
        raise ValueError("Plan mismatch. Use its exact plan_sha256.")
    plan = read_json(job / "plan.json")
    with (job / "started").open("x", encoding="utf-8") as receipt:
        receipt.write(plan_sha256)
        receipt.flush()
        os.fsync(receipt.fileno())
    try:
        update(job, "queued", runtime_released=None)
        emit_queue_hook(job, plan, "queued")
        process = background("monitor", job, "monitor.log")
    except OSError:
        (job / "started").unlink(missing_ok=True)
        update(job, "planned")
        emit_queue_hook(job, plan, "queue_failed", kind="monitor_start")
        raise
    return {"job_id": job_id, "phase": "queued", "monitor_pid": process.pid,
            "status_path": str(job / "status.json"), "log_path": str(job / "run.log"),
            "queue_hook_path": str(job / "queue-hook.json") if plan.get("queue_hook", "events") != "off" else None}


def cancel_video(job_id):
    job = directory(job_id)
    value = status(job_id)
    if value["phase"] == "review_pending":
        raise ValueError("Use review_runtime with action=release and a reason to release this retained runtime")
    if value["phase"] in TERMINAL:
        return value
    if not (job / "started").exists():
        raise ValueError("This plan has no running job to cancel")
    (job / "cancel").touch(exist_ok=True)
    return {"job_id": job_id, "cancel_requested": True}


def replacement_candidate(original, replacement, gpu, variant):
    keys = ("account", "profile_id", "profile_fingerprint", "card", "comfy_revision")
    if replacement.get("preflight_only") or any(original[key] != replacement[key] for key in keys):
        raise ValueError("Use the same account, model card and ComfyUI build on the retained runtime")
    selected = []
    for plan in (original, replacement):
        candidate = next((item for item in plan["candidates"] if (item["gpu"], item["variant"]) == (gpu, variant)), None)
        if candidate is None:
            raise ValueError("The replacement plan must include the retained GPU and selected model variant")
        selected.append(candidate)
    before, after = selected
    old_assets = [entry for entry in before["assets"] if not entry["path"].startswith("loras/")]
    new_assets = [entry for entry in after["assets"] if not entry["path"].startswith("loras/")]
    old_models = [entry for entry in before["uploads"] if entry["name"].startswith("model-")]
    new_models = [entry for entry in after["uploads"] if entry["name"].startswith("model-")]
    if old_assets != new_assets or old_models != new_models:
        raise ValueError("Use the same base weights on the retained runtime")
    return after


def read_replacement(job_id, plan_sha256):
    path = directory(job_id) / "plan.json"
    if sha256(path) != plan_sha256:
        raise ValueError("Replacement plan digest does not match")
    return read_json(path)


def resume_video(job_id, plan_job_id, plan_sha256, authorize_cloud):
    if authorize_cloud is not True:
        raise ValueError("Cloud execution needs explicit authorization for the chosen plan and assets")
    job = directory(job_id)
    state = status(job_id)
    if state["phase"] != "awaiting_generation":
        raise ValueError("Wait for the retained runtime to finish preparing")
    original = read_json(job / "plan.json")
    replacement = read_replacement(plan_job_id, plan_sha256)
    replacement_candidate(original, replacement, state["gpu"], state["variant"])
    request = {"job_id": plan_job_id, "plan_sha256": plan_sha256}
    with (job / "resume.json").open("x", encoding="utf-8") as stream:
        json.dump(request, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"job_id": job_id, "replacement_plan_id": plan_job_id, "phase": "resume_requested",
            "session": state["session"]}


def update(job, phase, **fields):
    path = job / "status.json"
    old = read_json(path) if path.exists() else {"job_id": job.name}
    save(path, {**old, **fields, "phase": phase, "updated_at": time.time()})


def log_event(job, name, **data):
    with (job / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": time.time(), "event": name, **data}, ensure_ascii=True) + "\n")


def emit_queue_hook(job, plan, phase, **fields):
    mode = plan.get("queue_hook", "events")
    if mode == "off":
        return
    terminal = phase in {"allocated", "allocation_failed", "queue_failed", "queue_cancelled"}
    event = {"phase": phase, "terminal": terminal, **fields}
    log_event(job, "queue_hook", **event)
    save(job / "queue-hook.json", {"job_id": job.name, "at": time.time(), **event})
    if terminal and mode == "notify":
        notify(job, phase)


def record_remote_setup_event(job, session, marker, preflight_stage, seen):
    phase = marker["phase"]
    if phase not in REMOTE_SETUP_EVENT_FIELDS:
        return
    stage = marker.get("stage")
    if phase in ("gpu_verified", "preflight_ready"):
        if preflight_stage is None:
            return
        stage = preflight_stage
    stage = stage if isinstance(stage, str) else None
    event = (session, phase, stage)
    if event in seen:
        return
    seen.add(event)
    fields = {key: marker[key] for key in REMOTE_SETUP_EVENT_FIELDS[phase]
              if type(marker.get(key)) in (str, bool, int, float)}
    if stage is not None:
        fields["stage"] = stage
    if phase == "preflight_ready" and isinstance(marker.get("cached_asset_sha256"), list):
        fields["cached_asset_count"] = len(marker["cached_asset_sha256"])
    log_event(job, phase, session=session, **fields)


def notify(job, outcome):
    log_event(job, "notification", outcome=outcome)
    if os.name != "nt":
        message = f"{job.name}: {outcome}"
        command = (["osascript", "-e", f'display notification "{message}" with title "MiniMax Design"']
                   if sys.platform == "darwin" else ["notify-send", "MiniMax Design", message])
        if shutil.which(command[0]):
            try:
                subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=7, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        return
    # Native toast. Jobs continue to keep a durable event even when notifications are disabled.
    code = ("[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime] > $null\n"
            "$xml=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('ToastText02')\n"
            "$t=$xml.GetElementsByTagName('text'); $t.Item(0).AppendChild($xml.CreateTextNode('MiniMax Design')) > $null\n"
            f"$t.Item(1).AppendChild($xml.CreateTextNode('{job.name}: {outcome}')) > $null\n"
            "$n=[Windows.UI.Notifications.ToastNotification]::new($xml)\n"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Microsoft.Windows.PowerShell').Show($n)")
    try:
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", code],
                       stdin=subprocess.DEVNULL, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
                       timeout=7, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def identity(plan, job):
    accounts, transport, pool = backend()
    profile = accounts.resolve(plan["account"])
    if profile["id"] != plan["profile_id"] or snapshot_profile(profile) != plan["profile_fingerprint"]:
        raise RuntimeError("Saved Colab profile changed after planning")
    binding = job / "bound-cli.json"
    version = read_json(binding)["version"] if binding.exists() else None
    transport, selected = select_cli(profile, accounts, transport, version,
                                     cli_transport=plan.get("cli_transport", "auto"))
    who = transport.run_colab(profile["name"], ["whoami"], timeout_seconds=18)
    if who["exit_code"] or accounts.parse_whoami(who["stdout"])["email"] != profile["verified_email"]:
        raise RuntimeError("Colab CLI account identity could not be verified")
    if version is None:
        save(binding, {"version": selected})
    return accounts, transport, pool


def explicit_session_missing(reply, session):
    return (reply["exit_code"] == 0
            and f"[colab] Session '{session}' not found." in reply["stdout"].splitlines())


def named_session_state(reply, session):
    if reply["exit_code"] != 0:
        return ""
    for line in reply["stdout"].splitlines():
        if line.startswith(f"[{session}] ") and " | Status: " in line:
            status_text = line.partition(" | Status: ")[2].strip()
            return "BUSY" if status_text.startswith("BUSY") else status_text
    return ""


def named_session_idle(reply, session):
    return named_session_state(reply, session) == "IDLE"


def probe_remote_preparation_or_outputs(call, session, job_id, batch_ids):
    targets = [f"video-{segment_id}.mp4" for segment_id in batch_ids] if batch_ids else ["video.mp4"]
    probe = f"""
import hashlib, json
from pathlib import Path
root = Path("/content") / {job_id!r}
found = []
for name in {targets!r}:
    path = root / name
    if path.is_file() and path.stat().st_size > 0:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        found.append({{"name": name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}})
started_render = (root / "output").exists() or any(root.glob("video*.mp4"))
can_resume = root.is_dir() and not started_render
if len(found) == {len(targets)}:
    print("H3_PROBE " + json.dumps({{"outputs": found, "can_resume": False}}), flush=True)
else:
    print("H3_PREPARATION_RESUME " + str(can_resume), flush=True)
"""
    reply = call(["exec", "-s", session, "--timeout", "25"], 32, stdin=probe)
    if reply["exit_code"] != 0:
        return {"outputs": [], "can_resume": False}
    for line in reply["stdout"].splitlines():
        if line.startswith("H3_PROBE "):
            try:
                data = json.loads(line.removeprefix("H3_PROBE "))
                return {"outputs": data.get("outputs", []), "can_resume": bool(data.get("can_resume"))}
            except ValueError:
                break
        if line.startswith("H3_PREPARATION_RESUME "):
            return {"outputs": [], "can_resume": line.strip().endswith("True")}
    return {"outputs": [], "can_resume": False}


def exec_connection_lost(reply):
    text = reply.get("stderr", "") + "\n" + reply.get("stdout", "")
    return reply["exit_code"] != 0 and "RuntimeError: Connection was lost." in text


def disconnected_in_exec_prelude(reply):
    text = reply.get("stderr", "") + "\n" + reply.get("stdout", "")
    return (exec_connection_lost(reply) and re.search(r"execution\.py:\d+ in exec_command", text) is not None
            and "os.makedirs('/content', exist_ok=True)" in text)


PREPARING_GENERATION = {
    "gpu_verified", "environment_setup_started", "environment_ready", "asset_progress", "assets_ready",
}


def can_resume_preparation(call, session, job_id):
    if not named_session_idle(call(["status", "-s", session], 15), session):
        return False
    probe = f"""
from pathlib import Path
root = Path("/content") / {job_id!r}
started_render = (root / "output").exists() or any(root.glob("video*.mp4"))
print("H3_PREPARATION_RESUME " + str(root.is_dir() and not started_render), flush=True)
"""
    reply = call(["exec", "-s", session, "--timeout", "10"], 15, stdin=probe)
    return reply["exit_code"] == 0 and "H3_PREPARATION_RESUME True" in reply["stdout"].splitlines()


def provider_reports_empty(reply):
    # CLI 0.8 can print the empty list after catching the ADC path's SystemExit.
    return (reply["exit_code"] == 0 and "No valid default credentials found." not in clean_cli_text(reply.get("stderr"))
            and "[colab] No active sessions found on server." in clean_cli_text(reply["stdout"]).splitlines())


def provider_session_absence(reply, session):
    if provider_reports_empty(reply):
        return "no_active_server_sessions"
    if reply["exit_code"] != 0 or "No valid default credentials found." in clean_cli_text(reply.get("stderr")):
        return ""
    # CLI 0.8 lists every provider assignment, using [?] when the local name is unknown.
    lines = [line for line in clean_cli_text(reply["stdout"]).splitlines()
             if line and not COLAB_PRUNED_SESSIONS.fullmatch(line)]
    rows = [COLAB_SESSION_ROW.fullmatch(line) for line in lines]
    if rows and all(row and row[1] not in {"?", session} for row in rows):
        return "absent_from_named_server_sessions"
    return ""


def allocation_has_settled(lease):
    return (lease.get("new_outcome") == "returned"
            or (lease.get("new_outcome") == "unknown" and time.time() >= lease["allocation_deadline"]))


def clean_cli_text(text):
    """Strip embedded NUL bytes (e.g. UTF-16LE wsl.exe warnings) from CLI output."""
    return (text or "").replace("\x00", "")


def probe_session_exec(transport, account, session, timeout_seconds=15):
    """Verify an IDLE Colab session's kernel exec channel is actually responsive."""
    exec_timeout = max(5, min(int(timeout_seconds), 25))
    reply = transport.run_colab(
        account,
        ["exec", "-s", session, "--timeout", str(exec_timeout)],
        timeout_seconds=exec_timeout + 10,
        stdin="print('ok')",
    )
    stdout = clean_cli_text(reply.get("stdout", ""))
    stderr = clean_cli_text(reply.get("stderr", ""))
    reachable = reply.get("exit_code") == 0 and "ok" in stdout.splitlines()
    return {
        "exec_reachable": reachable,
        "exec_exit_code": reply.get("exit_code"),
        "stderr": stderr.strip(),
    }


def probe_session(account, session, *, timeout_seconds=15):
    """Check both provider status and kernel exec reachability for a retained session."""
    _, transport, _ = backend()
    receipt = {
        "session": session,
        "provider_state": "ambiguous",
        "session_idle": False,
        "exec_reachable": False,
        "exec_exit_code": None,
        "stderr": "",
    }
    reply = transport.run_colab(account, ["status", "-s", session], timeout_seconds=15)
    if reply["exit_code"] == 0:
        session_status = next(
            (line.partition(" | Status: ")[2] for line in clean_cli_text(reply["stdout"]).splitlines()
             if line.startswith(f"[{session}] ") and " | Status: " in line),
            "",
        )
        if session_status == "IDLE" or session_status.startswith("BUSY ("):
            receipt.update(provider_state="active", session_idle=session_status == "IDLE")
            if session_status == "IDLE":
                receipt.update(probe_session_exec(transport, account, session, timeout_seconds=timeout_seconds))
        elif explicit_session_missing(reply, session):
            sessions = transport.run_colab(account, ["sessions"], timeout_seconds=20)
            evidence = provider_session_absence(sessions, session)
            if evidence:
                receipt["provider_state"] = ("account_empty" if evidence == "no_active_server_sessions"
                                             else "session_absent")
    return receipt


def review_runtime(job_id, action, reason="", *, probe_exec=False):
    if action not in ("inspect", "keep", "release") or not isinstance(reason, str):
        raise ValueError("Choose inspect, keep or release and provide a text reason when needed")
    reason = reason.strip()
    job = directory(job_id)
    state = status(job_id)
    if state["phase"] != "review_pending":
        raise ValueError("This job is not pending a runtime review")
    lease = read_json(job / "lease.json")
    session = lease["session"]
    if lease["released"] or state["runtime_review"]["session"] != session:
        raise ValueError("The retained runtime no longer matches the review")
    if action == "inspect":
        plan = read_json(job / "plan.json")
        _, transport, _ = identity(plan, job)
        receipt = {"job_id": job_id, "phase": state["phase"], "session": session,
                   "provider_state": "ambiguous", "session_idle": False, "runtime_released": False}
        reply = transport.run_colab(plan["account"], ["status", "-s", session], timeout_seconds=15)
        if reply["exit_code"] == 0:
            session_status = next((line.partition(" | Status: ")[2] for line in clean_cli_text(reply["stdout"]).splitlines()
                                   if line.startswith(f"[{session}] ") and " | Status: " in line), "")
            if session_status == "IDLE" or session_status.startswith("BUSY ("):
                receipt.update(provider_state="active", session_idle=session_status == "IDLE")
                if probe_exec and session_status == "IDLE":
                    receipt.update(probe_session_exec(transport, plan["account"], session))
            elif explicit_session_missing(reply, session) and allocation_has_settled(lease):
                sessions = transport.run_colab(plan["account"], ["sessions"], timeout_seconds=20)
                evidence = provider_session_absence(sessions, session)
                if evidence:
                    receipt["provider_state"] = ("account_empty" if evidence == "no_active_server_sessions"
                                                 else "session_absent")
        log_event(job, "runtime_review_inspected", session=session, provider_state=receipt["provider_state"],
                  session_idle=receipt["session_idle"],
                  **({"exec_reachable": receipt["exec_reachable"]} if "exec_reachable" in receipt else {}))
        return receipt
    release_request = job / "runtime-review-release.json"
    if release_request.exists():
        raise ValueError("A release review has already been submitted for this runtime")
    decision = {"decision": action, "reason": reason, "session": session, "decided_at": time.time()}
    if action == "keep":
        save(job / "runtime-review-keep.json", decision)
        log_event(job, "runtime_review_kept", session=session, reason=reason)
        return status(job_id)
    if not reason:
        raise ValueError("Provide a reason before releasing the retained runtime")
    if not (job / "started").exists() or (job / "started").read_text() != sha256(job / "plan.json"):
        raise RuntimeError("Refusing to release an unapproved or modified plan")
    with release_request.open("x", encoding="utf-8") as stream:
        json.dump(decision, stream)
        stream.flush()
        os.fsync(stream.fileno())
    log_event(job, "runtime_review_release_requested", session=session, reason=reason)
    try:
        process = background("review-release", job, "monitor.log")
    except OSError:
        release_request.unlink(missing_ok=True)
        raise
    return {"job_id": job_id, "phase": "review_release_requested", "session": session,
            "monitor_pid": process.pid}


def recover_allocation(job, transport, plan, lease, timeout):
    session = lease["session"]
    try:
        reply = transport.run_colab(plan["account"], ["status", "-s", session], timeout_seconds=timeout)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        log_event(job, "allocation_status_error", session=session, error_type=type(error).__name__)
        return False
    if not named_session_idle(reply, session):
        return False
    lease["new_outcome"] = "returned"
    save(job / "lease.json", lease)
    log_event(job, "allocation_recovered", session=session, evidence="named_provider_status")
    return True


def confirm_unallocated(job, transport, plan, lease):
    session = lease["session"]
    try:
        reply = transport.run_colab(plan["account"], ["status", "-s", session], timeout_seconds=15)
        if not explicit_session_missing(reply, session):
            return False
        if lease.get("new_outcome") == "rejected":
            evidence = "allocation_rejected"
        else:
            evidence = (provider_session_absence(transport.run_colab(plan["account"], ["sessions"], timeout_seconds=20),
                                                 session) if allocation_has_settled(lease) else "")
            if not evidence:
                return False
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        log_event(job, "allocation_status_error", session=session, error_type=type(error).__name__)
        return False
    lease.update(released=True, release_evidence=evidence)
    save(job / "lease.json", lease)
    return True


def cleanup(job, transport, plan, lease):
    if lease["released"]:
        return True
    session = lease["session"]
    for attempt in range(3):
        stopped = absent = False
        evidence = ""
        try:
            answer = transport.run_colab(plan["account"], ["stop", "-s", session], timeout_seconds=20)
            stopped = answer["exit_code"] == 0 and "[colab] Session terminated." in answer["stdout"].splitlines()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            log_event(job, "release_transport_error", session=session, operation="stop", error_type=type(error).__name__)
        if not stopped:
            try:
                absent = explicit_session_missing(transport.run_colab(
                    plan["account"], ["status", "-s", session], timeout_seconds=15), session)
                if absent and allocation_has_settled(lease):
                    sessions = transport.run_colab(plan["account"], ["sessions"], timeout_seconds=20)
                    evidence = provider_session_absence(sessions, session)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                log_event(job, "release_transport_error", session=session, operation="status", error_type=type(error).__name__)
        log_event(job, "release_attempt", session=session, attempt=attempt + 1,
                  provider_stop_ack=stopped, provider_reports_missing=absent,
                  provider_account_empty=evidence == "no_active_server_sessions",
                  provider_session_absent=bool(evidence))
        if stopped:
            evidence = "stop_ack"
        elif absent and lease.get("new_outcome") == "rejected":
            evidence = "allocation_rejected"
        if evidence:
            lease.update(released=True, release_evidence=evidence)
            save(job / "lease.json", lease)
            return True
        if attempt < 2:
            time.sleep(2 ** attempt)
    return False


def run_worker(job, plan):
    runner = HERE / "runtime_remote.py"
    if sha256(runner) != plan["runner_sha256"]:
        raise RuntimeError("The remote runner changed after planning; create a new plan")
    source = runner.read_text(encoding="utf-8")
    accounts, transport, pool = identity(plan, job)
    probe_only = plan.get("preflight_only", False)
    if plan.get("drive_cache_relative") and not probe_only and not isinstance(transport, managed_cli.NativeCLI):
        raise RuntimeError("Automatic persistent Drive cache requires the verified managed Colab CLI")
    with accounts._transaction() as data:
        profile = data["accounts"].get(plan["account"])
        if not profile or snapshot_profile(profile) != plan["profile_fingerprint"]:
            raise RuntimeError("Colab account changed during admission")
        claim = pool.admit(provider="colab", account_id=plan["profile_id"], notebook_ref=job.name,
                           run_id=job.name, accelerator="gpu", thread_id=job.name)["claim"]
        save(job / "claim.json", claim)

    def call(arguments, timeout, **options):
        return transport.run_colab(plan["account"], arguments, timeout_seconds=timeout, **options)

    def require(reply, phase):
        if reply["exit_code"] != 0:
            raise RuntimeError(f"{phase} failed (Colab exit {reply['exit_code']}). Check run.log.")
        return reply

    def check_cancel():
        if (job / "cancel").exists():
            raise InterruptedError("Cancellation requested")

    deadline = time.time() + plan["timeout_seconds"]
    exhausted = set()
    allocation_retried = set()
    session_counter = 0
    for index, item in enumerate(plan["candidates"]):
        check_cancel()
        if item["gpu"] in exhausted or time.time() >= deadline - COLAB_PREFLIGHT_RESERVE:
            continue
        session_counter += 1
        session = f"{job.name}-{session_counter}"
        nonce = uuid.uuid4().hex
        output_events = Queue()
        preflight_stage = None
        seen_setup_events = set()
        awaiting_generation_reconnect = False
        def stream(line):
            nonlocal awaiting_generation_reconnect
            clean_line = line.replace("\x00", "")
            if clean_line:
                with (job / "run.log").open("a", encoding="utf-8") as log:
                    log.write(clean_line)
                marker_start = clean_line.find(MARK)
                if marker_start >= 0:
                    try:
                        marker = json.loads(clean_line[marker_start + len(MARK):])
                    except (TypeError, ValueError):
                        return
                    if (isinstance(marker, dict) and isinstance(marker.get("phase"), str)
                            and marker.get("job_id") == job.name and marker.get("nonce") == nonce):
                        marker_phase = marker["phase"]
                        if marker_phase == "sampling_heartbeat":
                            if isinstance(marker.get("step"), int) and isinstance(marker.get("max_steps"), int):
                                progress = {"step": marker["step"], "max_steps": marker["max_steps"]}
                                if "segment_id" in marker:
                                    progress["segment_id"] = marker["segment_id"]
                                update(job, "sampling", sampling_progress=progress)
                            return
                        markers[marker_phase] = marker
                        phase = "remote_failed" if marker_phase == "failed" else marker_phase
                        fields = {"remote_marker": {k: v for k, v in marker.items() if k != "nonce"}}
                        if awaiting_generation_reconnect:
                            awaiting_generation_reconnect = False
                            recovery = {"state": "recovered", "session": session, "remote_phase": marker_phase}
                            fields["generation_transport"] = recovery
                            log_event(job, "generation_transport_recovered", **recovery)
                        update(job, phase, **fields)
                        record_remote_setup_event(job, session, marker, preflight_stage, seen_setup_events)
                        if marker_phase == "output_ready":
                            output_events.put(marker)

        started = time.time()
        markers = {}
        remote_config = {**plan["runtime_options"], "job_id": job.name, "nonce": nonce, "gpu": item["gpu"],
                         "min_vram_gib": item["min_vram_gib"], "min_ram_gib": item["min_ram_gib"],
                         "comfy_revision": plan["comfy_revision"], "timeout_seconds": plan["timeout_seconds"],
                         "setup_timeout_seconds": plan["setup_timeout_seconds"],
                         "assets": item["assets"], "reference_uploads": item["uploads"],
                         "audio_mode": plan.get("audio_mode", "generated")}
        workflow_keys = ("segments",) if "segments" in item else ("graph", "output_node")
        remote_config.update({key: item[key] for key in workflow_keys})
        remote_config["reference_uploads"] = [{k: ref[k] for k in ("name", "sha256")}
                                                for ref in item["uploads"] if ref["name"].startswith("reference-")]
        def preflight_config():
            config = {key: value for key, value in remote_config.items()
                      if key not in {"graph", "output_node", "segments", "reference_uploads", "assets"}}
            config["assets"] = [{key: value for key, value in asset.items()
                                if key in {"path", "sha256", "size_bytes", "upload_name"}}
                                for asset in remote_config["assets"]]
            return {**config, "preflight_only": True}

        def remote_exec(config, budget):
            return call(["exec", "-s", session, "--timeout", str(budget)], budget + 8,
                        stdin="CONFIG = " + repr(config) + "\n" + source, on_output=stream)

        def require_remote(reply, phase):
            if failure := markers.get("failed"):
                kind = failure.get("error_kind") or "remote_failure"
                detail = failure.get("message") or "Check run.log."
                raise RuntimeError(f"{phase} failed ({kind}). {detail}")
            require(reply, phase)

        runtime_shape = plan.get("runtime_shape", "auto")
        high_memory = runtime_shape == "high_mem" or (runtime_shape == "auto" and item["min_ram_gib"] >= 20)
        memory_flag = ["--high-mem"] if high_memory else []
        rejected_gpu = False
        max_alloc_attempts = 2 if probe_only else 4
        for alloc_attempt in range(max_alloc_attempts):
            check_cancel()
            started = time.time()
            # Give CLI 0.8's assignment GET/POST and first same-hash reconciliation time to finish.
            alloc_timeout = min(COLAB_ALLOCATION_TIMEOUT, math.floor(deadline - started) - COLAB_PREFLIGHT_RESERVE)
            if alloc_timeout < 1:
                raise TimeoutError("Video job budget expired before Colab allocation")
            lease = {"session": session, "released": False, "new_outcome": "unknown",
                     "began_at": started, "allocation_deadline": started + alloc_timeout + COLAB_ALLOCATION_SETTLE}
            save(job / "lease.json", lease)
            if index == 0 and alloc_attempt == 0:
                pool.update(claim["id"], "submitting", job.name)
            update(job, "allocating", gpu=item["gpu"], variant=item["variant"],
                   precision=item["precision"], session=session, runtime_released=False)
            emit_queue_hook(job, plan, "allocating", session=session, gpu=item["gpu"], attempt=session_counter)
            reply = call(["new", "-s", session, "--gpu", item["gpu"], *memory_flag], alloc_timeout, on_output=stream)
            if reply["exit_code"] == 0:
                lease["new_outcome"] = "returned"
                save(job / "lease.json", lease)
                emit_queue_hook(job, plan, "allocated", session=session, gpu=item["gpu"], attempt=session_counter)
                break
            reason = reply["stdout"] + "\n" + reply["stderr"]
            timed_out = reply["exit_code"] == 124 or "requests.exceptions.ReadTimeout:" in reason
            transport_failure = None
            if "in _post_assignment" in reason and COLAB_ASSIGN_DISCONNECT.search(reason):
                transport_failure = {"kind": "transport_lost", "cli_exit_code": reply["exit_code"],
                                     "message": "Connection to Colab closed without a response to the allocation request."}
                log_event(job, "allocation_failed", session=session, **transport_failure)
            uncertain = timed_out or transport_failure is not None
            if not uncertain and (backend_error := COLAB_ASSIGN_BACKEND_ERROR.search(reason)):
                http_reason = backend_error[1]
                http_status = COLAB_BACKEND_HTTP_STATUS[http_reason]
                detail = f"Colab allocation backend is temporarily unavailable (HTTP {http_status} {http_reason})."
                failure = {"kind": "backend_unavailable", "http_status": http_status, "message": detail,
                           "cli_exit_code": reply["exit_code"]}
                update(job, "allocation_failed", allocation_failure=failure)
                log_event(job, "allocation_failed", session=session, **failure)
                retry_budget = COLAB_ALLOCATION_TIMEOUT + COLAB_ALLOCATION_SETTLE + COLAB_PREFLIGHT_RESERVE
                backoff = min(60, 15 * (2 ** alloc_attempt))
                if (not probe_only and alloc_attempt < max_alloc_attempts - 1
                        and deadline - time.time() >= retry_budget + backoff):
                    check_cancel()
                    sessions_reply = call(["sessions"], 20)
                    if evidence := provider_session_absence(sessions_reply, session):
                        log_event(job, "allocation_retry", gpu=item["gpu"], previous_session=session,
                                  reason=f"http_{http_status}", backoff_seconds=backoff, release_evidence=evidence)
                        emit_queue_hook(job, plan, "allocation_retry", session=session, gpu=item["gpu"])
                        time.sleep(backoff)
                        session_counter += 1
                        session = f"{job.name}-{session_counter}"
                        continue
                raise RuntimeError(detail)
            if not uncertain and "Assignment request failed; retrying" not in reason and CAPACITY.search(reason):
                lease["new_outcome"] = "rejected"
                save(job / "lease.json", lease)
                if confirm_unallocated(job, transport, plan, lease):
                    exhausted.add(item["gpu"])
                    log_event(job, "gpu_unavailable", gpu=item["gpu"])
                    emit_queue_hook(job, plan, "candidate_unavailable", session=session, gpu=item["gpu"])
                    rejected_gpu = True
                    break
            if uncertain:
                check_cancel()
                status_timeout = min(COLAB_ALLOCATION_SETTLE, math.floor(deadline - time.time()) - COLAB_PREFLIGHT_RESERVE)
                if status_timeout > 0 and recover_allocation(job, transport, plan, lease, status_timeout):
                    emit_queue_hook(job, plan, "allocated", session=session, gpu=item["gpu"], attempt=session_counter,
                                    recovered=True)
                    break
                retry_budget = COLAB_ALLOCATION_TIMEOUT + COLAB_ALLOCATION_SETTLE + COLAB_PREFLIGHT_RESERVE
                if (alloc_attempt < max_alloc_attempts - 1 and item["gpu"] not in allocation_retried
                        and deadline - max(time.time(), lease["allocation_deadline"]) >= retry_budget):
                    time.sleep(max(0, lease["allocation_deadline"] - time.time()))
                    if confirm_unallocated(job, transport, plan, lease) and deadline - time.time() >= retry_budget:
                        allocation_retried.add(item["gpu"])
                        log_event(job, "allocation_retry", gpu=item["gpu"], previous_session=session,
                                  release_evidence=lease.get("release_evidence"))
                        emit_queue_hook(job, plan, "allocation_retry", session=session, gpu=item["gpu"])
                        session_counter += 1
                        session = f"{job.name}-{session_counter}"
                        continue
            if transport_failure:
                update(job, "allocation_failed", allocation_failure=transport_failure)
                raise RuntimeError(transport_failure["message"])
            raise RuntimeError("Colab allocation failed or its outcome could not be verified. Check run.log.")
        if rejected_gpu:
            continue
        check_cancel()
        if time.time() > deadline - COLAB_PREFLIGHT_RESERVE:
            raise TimeoutError("Video job has insufficient time for Colab preflight after allocation")
        update(job, "preflight", session=session)
        preflight_stage = "hardware_storage"
        log_event(job, "preflight_started", session=session, stage=preflight_stage)
        initial = preflight_config()
        if plan.get("drive_cache_relative"):
            initial["assets"] = []
        result = remote_exec(initial, 35)
        if (result["exit_code"] != 0 and not markers.get("failed") and not markers.get("preflight_ready")
                and (result["exit_code"] == 124
                     or "Connection was lost" in (result["stdout"] + "\n" + result["stderr"]))
                and time.time() < deadline - 45):
            check_cancel()
            log_event(job, "preflight_retry", session=session, exit_code=result["exit_code"])
            time.sleep(2)
            result = remote_exec(initial, 35)
        require_remote(result, "Hardware and storage preflight")
        if not markers.get("gpu_verified") or not markers.get("preflight_ready"):
            raise RuntimeError("Colab did not report a completed hardware and storage preflight")
        if probe_only:
            receipt = {"session": session, "requested_gpu": item["gpu"], "variant": item["variant"],
                       "gpu": {key: value for key, value in markers["gpu_verified"].items()
                               if key not in {"job_id", "nonce", "phase"}},
                       "cache_checked": not bool(plan.get("drive_cache_relative"))}
            save(job / "preflight.json", receipt)
            log_event(job, "standalone_preflight_passed", gpu=receipt["gpu"])
            return
        if drive := plan.get("drive_cache_relative"):
            check_cancel()
            update(job, "mounting_model_cache")
            preflight_stage = "model_cache"
            log_event(job, "preflight_started", session=session, stage=preflight_stage)
            require(call(["drivemount", "-s", session, "/content/drive"], 75, on_output=stream),
                    "Authorized Drive cache mount")
            model_cache = f"/content/drive/MyDrive/{drive}"
            require(call(["exec", "-s", session, "--timeout", "20"], 28, stdin=(
                "from pathlib import Path\n"
                "drive = Path('/content/drive/MyDrive')\n"
                "if not drive.is_dir(): raise RuntimeError('Colab Drive mount is missing')\n"
                "Path(" + repr(model_cache) + ").mkdir(parents=True,exist_ok=True)\n")),
                "Initialize the configured Drive model folder")
            remote_config.update(cache_dir=model_cache, local_environment=True)
            markers.clear()
            require_remote(remote_exec(preflight_config(), 35), "Persistent model cache preflight")
            if not markers.get("preflight_ready"):
                raise RuntimeError("Drive model cache preflight did not report usable storage")
        cached = set(markers.get("preflight_ready", {}).get("cached_asset_sha256", []))
        remaining_uploads = [ref for ref in item["uploads"]
                             if not ref["name"].startswith("model-") or ref["sha256"] not in cached]
        def upload_inputs(refs):
            if not refs:
                return
            check_cancel()
            update(job, "uploading")
            remote = f"/content/{job.name}/input"
            mkdir_cmd = ["exec", "-s", session, "--timeout", "25"]
            mkdir_stdin = "from pathlib import Path\nPath(" + repr(remote) + ").mkdir(parents=True,exist_ok=True)\n"
            mkdir_reply = call(mkdir_cmd, 35, stdin=mkdir_stdin)
            if (mkdir_reply["exit_code"] == 124 or exec_connection_lost(mkdir_reply)) and time.time() < deadline - 45:
                check_cancel()
                log_event(job, "upload_mkdir_retry", session=session, exit_code=mkdir_reply["exit_code"])
                time.sleep(2)
                mkdir_reply = call(mkdir_cmd, 35, stdin=mkdir_stdin)
            require(mkdir_reply, "Create remote input directory")
            def upload_with_retry(local_path, remote_path, label):
                for attempt in range(2):
                    check_cancel()
                    remaining = min(600, math.floor(deadline - time.time()))
                    if remaining < 10:
                        raise TimeoutError("Video generation deadline reached during uploads")
                    reply = call(["upload", "-s", session, str(local_path), remote_path],
                                 remaining, on_output=stream)
                    if reply["exit_code"] == 0:
                        return
                    if attempt == 0 and (reply["exit_code"] == 124 or exec_connection_lost(reply)) and time.time() < deadline - 30:
                        log_event(job, "upload_retry", session=session, target=remote_path, exit_code=reply["exit_code"])
                        time.sleep(2)
                        continue
                    require(reply, label)

            for ref in refs:
                if sha256(ref["local_path"]) != ref["sha256"]:
                    raise ValueError(f"Local upload changed since planning: {Path(ref['local_path']).name}")
            if len(refs) == 1:
                ref = refs[0]
                upload_with_retry(ref["local_path"], f"{remote}/{ref['name']}", "Upload approved input")
                return
            bundle = job / f".upload-bundle-{uuid.uuid4().hex[:8]}.tar"
            try:
                with tarfile.open(bundle, "w") as archive:
                    for ref in refs:
                        archive.add(ref["local_path"], arcname=ref["name"])
                remote_bundle = f"{remote}/.upload-bundle.tar"
                upload_with_retry(bundle, remote_bundle, "Upload approved input bundle")
                extract_stdin = (
                    "import tarfile\nfrom pathlib import Path\n"
                    f"bundle = Path({remote_bundle!r})\n"
                    f"with tarfile.open(bundle, 'r') as tf:\n    tf.extractall({remote!r})\n"
                    "bundle.unlink(missing_ok=True)\n"
                )
                require(call(["exec", "-s", session, "--timeout", "45"], 55, stdin=extract_stdin),
                        "Extract approved input bundle")
            finally:
                bundle.unlink(missing_ok=True)

        upload_inputs(remaining_uploads)
        active_plan = plan
        if plan.get("hold_for_generation"):
            check_cancel()
            remaining = math.floor(deadline - time.time())
            if remaining < 20:
                raise TimeoutError("Video generation deadline reached before retained setup")
            preflight_stage = None
            markers.clear()
            setup_budget = min(remaining, plan["setup_timeout_seconds"])
            update(job, "preparing_runtime")
            setup_config = {key: value for key, value in remote_config.items()
                            if key not in ("graph", "segments", "output_node")}
            require_remote(remote_exec({**setup_config, "prepare_only": True,
                                        "timeout_seconds": setup_budget, "setup_timeout_seconds": setup_budget},
                                       setup_budget), "Prepare retained Colab runtime")
            if "prepared" not in markers:
                raise RuntimeError("The retained Colab runtime did not report preparation complete")
            update(job, "awaiting_generation")
            log_event(job, "awaiting_generation", session=session)
            request_path = job / "resume.json"
            while True:
                check_cancel()
                if time.time() >= deadline - 20:
                    raise TimeoutError("Retained Colab runtime reached its generation deadline")
                if request_path.is_file():
                    try:
                        request = read_json(request_path)
                    except json.JSONDecodeError:
                        pass
                    else:
                        break
                time.sleep(0.5)
            active_plan = read_replacement(request["job_id"], request["plan_sha256"])
            replacement = replacement_candidate(plan, active_plan, item["gpu"], item["variant"])
            previous_uploads = {ref["name"]: ref["sha256"] for ref in item["uploads"]}
            current_names = {ref["name"] for ref in replacement["uploads"]}
            removed_refs = [name for name in previous_uploads if not name.startswith("model-") and name not in current_names]
            if removed_refs:
                remote_input = f"/content/{job.name}/input"
                rm_stdin = "from pathlib import Path\n" + "\n".join(
                    f"Path({repr(remote_input + '/' + name)}).unlink(missing_ok=True)" for name in removed_refs
                ) + "\n"
                require(call(["exec", "-s", session, "--timeout", "20"], 30, stdin=rm_stdin),
                        "Remove stale reference inputs")
            changed_uploads = [ref for ref in replacement["uploads"] if previous_uploads.get(ref["name"]) != ref["sha256"]]
            upload_inputs(changed_uploads)
            for key in ("segments", "graph", "output_node"):
                remote_config.pop(key, None)
                if key in replacement:
                    remote_config[key] = replacement[key]
            remote_config["assets"] = replacement["assets"]
            remote_config["audio_mode"] = active_plan.get("audio_mode", "generated")
            update(job, "resume_accepted", replacement_plan_id=request["job_id"], loras=active_plan.get("loras", {}),
                   segments=active_plan.get("segments", []), audio_mode=remote_config["audio_mode"],
                   continuation=active_plan.get("continuation", "independent"), takes=active_plan.get("takes", 1))
            log_event(job, "generation_resumed", session=session, replacement_plan_id=request["job_id"])
        check_cancel()
        remaining = math.floor(deadline - time.time())
        if remaining < 20:
            raise TimeoutError("Video generation deadline reached before setup")
        update(job, "generating")
        preflight_stage = None
        markers.clear()
        result = {}
        def poll_detached_remote():
            remote_root = f"/content/{job.name}"
            offset = 0
            poll_failures = 0
            poll_code = (
                "from pathlib import Path\n"
                f"root = Path({remote_root!r})\n"
                "log = root / 'remote.log'\n"
                "ex = root / 'remote.exit'\n"
                "data = log.read_bytes() if log.is_file() else b''\n"
                "off = {offset}\n"
                "chunk = data[off:]\n"
                "print('__H3_OFF__:' + str(len(data)))\n"
                "if chunk:\n"
                "    print(chunk.decode('utf-8', errors='replace'), end='')\n"
                "if ex.is_file():\n"
                "    print('\\n__H3_EXIT__:' + ex.read_text().strip())\n"
            )
            while True:
                check_cancel()
                if time.time() >= deadline:
                    return {"exit_code": 124, "stdout": "", "stderr": "Detached generation deadline reached"}
                reply = call(["exec", "-s", session, "--timeout", "20"], 30, stdin=poll_code.format(offset=offset))
                if reply["exit_code"] != 0:
                    poll_failures += 1
                    log_event(job, "generation_poll_retry", session=session, attempt=poll_failures,
                              exit_code=reply["exit_code"])
                    if poll_failures >= 8:
                        return reply
                    time.sleep(5)
                    continue
                poll_failures = 0
                exit_rc = None
                for line in clean_cli_text(reply.get("stdout", "")).splitlines(keepends=True):
                    stripped = line.strip()
                    if stripped.startswith("__H3_OFF__:"):
                        try:
                            offset = int(stripped.split(":", 1)[1])
                        except ValueError:
                            pass
                    elif stripped.startswith("__H3_EXIT__:"):
                        try:
                            exit_rc = int(stripped.split(":", 1)[1])
                        except ValueError:
                            exit_rc = 1
                    else:
                        stream(line)
                if exit_rc is not None:
                    return {"exit_code": exit_rc, "stdout": "", "stderr": ""}
                time.sleep(min(12, max(1, deadline - time.time())))

        def run_generation(budget):
            inner_cfg = {**remote_config, "timeout_seconds": budget,
                         "setup_timeout_seconds": min(budget, plan["setup_timeout_seconds"])}
            if budget >= 600:
                runner_src = "CONFIG = " + repr(inner_cfg) + "\n" + source
                launch_cfg = {**inner_cfg, "detached": True,
                              "_runner_b64": base64.b64encode(runner_src.encode("utf-8")).decode("ascii")}
                reply = call(["exec", "-s", session, "--timeout", "60"], 70,
                             stdin="CONFIG = " + repr(launch_cfg) + "\n" + source, on_output=stream)
                if "detached_started" in markers:
                    return poll_detached_remote()
                return reply
            return call(["exec", "-s", session, "--timeout", str(budget)], budget + 8,
                        stdin="CONFIG = " + repr(inner_cfg) + "\n" + source, on_output=stream)

        def generate():
            nonlocal awaiting_generation_reconnect
            try:
                reply = run_generation(remaining)
                if not markers and disconnected_in_exec_prelude(reply) and deadline - time.time() >= 45:
                    check_cancel()
                    session_status = call(["status", "-s", session], 15)
                    if named_session_idle(session_status, session):
                        check_cancel()
                        awaiting_generation_reconnect = True
                        log_event(job, "generation_transport_retry", session=session, stage="cli_prelude")
                        reply = run_generation(math.floor(deadline - time.time()))
                if exec_connection_lost(reply) and "failed" not in markers:
                    last = status(job.name).get("remote_marker", {}).get("phase")
                    batch_list = [entry["id"] for entry in active_plan.get("segments", [])]
                    session_st = named_session_state(call(["status", "-s", session], 15), session)
                    if session_st == "BUSY" and deadline - time.time() >= 60:
                        log_event(job, "generation_transport_waiting", session=session, remote_phase=last)
                        while session_st == "BUSY" and deadline - time.time() >= 45:
                            check_cancel()
                            time.sleep(min(10, max(1, deadline - time.time() - 45)))
                            session_st = named_session_state(call(["status", "-s", session], 15), session)
                    if session_st == "IDLE" and deadline - time.time() >= 45:
                        outcome_probe = probe_remote_preparation_or_outputs(call, session, job.name, batch_list)
                        if outcome_probe["outputs"]:
                            already_done = {item["id"] for item in downloaded}
                            pending_outputs = [
                                item_out for item_out in outcome_probe["outputs"]
                                if not batch_list or item_out["name"].removeprefix("video-").removesuffix(".mp4") not in already_done
                            ]
                            log_event(job, "generation_transport_recovered", state="recovered",
                                      session=session, remote_phase="output_ready",
                                      recovered_outputs=len(pending_outputs))
                            for item_out in pending_outputs:
                                extra = {}
                                if batch_list:
                                    extra["segment_id"] = item_out["name"].removeprefix("video-").removesuffix(".mp4")
                                output_events.put({
                                    "job_id": job.name, "nonce": nonce, "phase": "output_ready",
                                    "sha256": item_out["sha256"], "size_bytes": item_out["size_bytes"], **extra,
                                })
                            result["response"] = {"exit_code": 0, "stdout": "", "stderr": ""}
                            return
                        if (last in PREPARING_GENERATION and "sampling" not in markers
                                and "output_ready" not in markers and deadline - time.time() >= 75
                                and outcome_probe["can_resume"]):
                            check_cancel()
                            awaiting_generation_reconnect = True
                            log_event(job, "generation_transport_retry", session=session, stage="preparation",
                                      remote_phase=last)
                            reply = run_generation(math.floor(deadline - time.time()))
                if exec_connection_lost(reply) and "failed" not in markers:
                    stage = "cli_prelude" if not markers and disconnected_in_exec_prelude(reply) else (
                        "script_dispatched" if markers else "dispatch_unconfirmed")
                    detail = ("Colab's Jupyter connection dropped before the generation script was sent."
                              if stage == "cli_prelude" else
                              "Lost the Colab Jupyter connection. Inspect the retained session for generation progress.")
                    failure = {"state": "interrupted", "session": session, "stage": stage,
                               "remote_phase": (status(job.name).get("remote_marker", {}).get("phase")
                                                if markers else None)}
                    log_event(job, "generation_transport_failed", **failure)
                    state = status(job.name)
                    update(job, state["phase"], generation_transport=failure)
                    raise RuntimeError(detail)
                result["response"] = reply
            except Exception as error:
                result["error"] = error
            finally:
                output_events.put(None)

        threading.Thread(target=generate, name="colab-render", daemon=True).start()
        batch_ids = [entry["id"] for entry in active_plan.get("segments", [])]
        downloaded = []
        while True:
            try:
                artifact = output_events.get(timeout=max(0, deadline + 16 - time.time()))
            except Empty as error:
                raise TimeoutError("Colab did not report a completed output before the deadline") from error
            if artifact is None:
                if "error" in result:
                    raise result["error"]
                all_verified = bool(batch_ids) and len(downloaded) == len(batch_ids)
                if all_verified and markers.get("failed", {}).get("error_kind") == "timeout":
                    log_event(job, "verified_outputs_after_timeout", segments=batch_ids)
                else:
                    require_remote(result["response"], "ComfyUI inference")
                if not all_verified:
                    raise RuntimeError("ComfyUI did not report all planned outputs")
                save(job / "artifact.json", {"segment_outputs": downloaded})
                return
            check_cancel()
            segment_id = artifact.get("segment_id")
            if batch_ids:
                if len(downloaded) >= len(batch_ids) or segment_id != batch_ids[len(downloaded)]:
                    raise RuntimeError("ComfyUI output did not match the next planned segment")
                stem = f"video-{segment_id}"
            else:
                if segment_id is not None:
                    raise RuntimeError("Unexpected segment ID in single video output")
                update(job, "downloading")
                stem = "video"
            receipt = download_output(job, session, call, stream, check_cancel, artifact, stem)
            if not batch_ids:
                save(job / "artifact.json", receipt)
                return
            downloaded.append({"id": segment_id, **receipt})
            save(job / "segment-outputs.json", downloaded)
            update(job, status(job.name)["phase"], segment_outputs=downloaded)
    raise RuntimeError("No compatible Colab GPU was allocated from the allowed candidates")


def download_output(job, session, call, stream, check_cancel, marker, stem):
    digest = marker.get("sha256")
    size = marker.get("size_bytes")
    if (not SHA256.fullmatch(str(digest or "")) or type(size) is not int or size <= 0):
        raise RuntimeError("ComfyUI output marker did not contain an integrity receipt")
    segment = {"segment_id": marker["segment_id"]} if "segment_id" in marker else {}
    log_event(job, "output_ready_hook", session=session, **segment)
    partial = job / f"{stem}.part"
    transfer_timeout = max(90, min(300, size // 100_000 + 60))
    try:
        for attempt in range(3):
            check_cancel()
            partial.unlink(missing_ok=True)
            try:
                reply = call(["download", "-s", session, f"/content/{job.name}/{stem}.mp4", str(partial)],
                             transfer_timeout, on_output=stream)
            except (OSError, subprocess.SubprocessError) as error:
                log_event(job, "download_transport_error", error_type=type(error).__name__, **segment)
                reply = None
            if (reply is not None and reply["exit_code"] == 0 and partial.is_file() and partial.stat().st_size == size
                    and sha256(partial) == digest):
                target = partial.replace(job / f"{stem}.mp4")
                log_event(job, "output_downloaded", sha256=digest, **segment)
                audio = {"audio_tracks": marker["audio_tracks"]} if "audio_tracks" in marker else {}
                return {"video_path": str(target), "sha256": digest, "size_bytes": size, **audio}
            log_event(job, "download_retry" if attempt < 2 else "download_failed", attempt=attempt + 1, **segment)
            if attempt < 2:
                time.sleep(2)
        raise RuntimeError("Video download failed integrity verification after 3 attempts")
    finally:
        partial.unlink(missing_ok=True)


def finish(job, plan, outcome, *, reviewed_session=None, **fields):
    lease_path = job / "lease.json"
    lease = read_json(lease_path) if lease_path.exists() else None
    claim_path = job / "claim.json"
    state_path = job / "status.json"
    last_phase = read_json(state_path)["phase"] if state_path.exists() else "initialization"
    hook_path = job / "queue-hook.json"
    if outcome in {"failed", "cancelled"} and hook_path.is_file() and not read_json(hook_path)["terminal"]:
        state = read_json(state_path) if state_path.exists() else {}
        failure = state.get("allocation_failure") or {}
        emit_queue_hook(job, plan, "queue_cancelled" if outcome == "cancelled" else "allocation_failed",
                        **{k: failure[k] for k in ("kind", "http_status") if k in failure})
    released = not lease or lease["released"]
    if outcome == "failed" and not released and reviewed_session != lease["session"]:
        update(job, "review_pending", runtime_released=False, outcome=outcome,
               runtime_review={"session": lease["session"], "last_phase": last_phase,
                               "new_outcome": lease["new_outcome"], "decision": "pending"}, **fields)
        log_event(job, "diagnostic_required", phase="review_pending", last_phase=last_phase,
                  runtime_released=False, session=lease["session"])
        notify(job, "review_pending")
        return
    try:
        if not released:
            update(job, "releasing")
            if lease["new_outcome"] == "unknown":
                time.sleep(max(0, lease["allocation_deadline"] - time.time()))
            _, transport, _ = identity(plan, job)
            reviewed_absence = reviewed_session == lease["session"] and confirm_unallocated(job, transport, plan, lease)
            released = reviewed_absence or cleanup(job, transport, plan, lease)
        if claim_path.exists():
            _, _, pool = backend()
            claim = read_json(claim_path)
            if released:
                final = outcome if lease else "not_submitted"
                evidence = {"confirmed": True, "source": "provider" if lease else "owner",
                            "reference": str(lease_path if lease else job / "status.json"), "observed_state": final,
                            "remote_request_issued": bool(lease)}
                pool.release(claim["id"], final, job.name, evidence=evidence)
            else:
                pool.update(claim["id"], "unknown", job.name)
    except Exception as error:
        log_event(job, "cleanup_error", detail=str(error))
        fields["cleanup_error"] = str(error)
    phase = outcome if released and "cleanup_error" not in fields else "cleanup_pending"
    update(job, phase, runtime_released=released, outcome=outcome,
           cleanup_error=fields.pop("cleanup_error", None), **fields)
    if phase in {"failed", "cleanup_pending"}:
        log_event(job, "diagnostic_required", phase=phase, last_phase=last_phase, runtime_released=released)
    notify(job, phase)


def release_reviewed_runtime(job, plan):
    state = status(job.name)
    lease = read_json(job / "lease.json")
    decision = read_json(job / "runtime-review-release.json")
    if (state["phase"] not in {"review_pending", "cleanup_pending", "releasing"}
            or state.get("outcome") != "failed" or decision.get("decision") != "release"
            or decision.get("session") != lease["session"] or not str(decision.get("reason", "")).strip()):
        raise RuntimeError("The release review does not authorize this retained runtime")
    finish(job, plan, "failed", reviewed_session=lease["session"],
           runtime_review={**state["runtime_review"], **decision})


def monitor(job, plan):
    log_event(job, "monitor_started", pid=os.getpid())
    worker = background("worker", job, "worker.log")
    expires = time.time() + plan["timeout_seconds"] + 160
    offsets = 0
    seen = set()
    last_observation = time.time()
    reason = ""
    while worker.poll() is None:
        (job / "monitor-heartbeat").touch()
        log_file = job / "run.log"
        if log_file.exists():
            with log_file.open("rb") as stream:
                stream.seek(offsets)
                body = stream.read(1048576).decode("utf-8", errors="replace")
                offsets = stream.tell()
            if body:
                last_observation = time.time()
            for keyword in (MARK, "Traceback", "out of memory", "RESOURCE_EXHAUSTED", "output_ready", "assets_ready"):
                if keyword.lower() in body.lower() and keyword not in seen:
                    seen.add(keyword)
                    log_event(job, "new_log_keyword", keyword=keyword)
        stalled = time.time() - last_observation > 300
        if stalled and "stalled" not in seen:
            log_event(job, "no_log_progress", duration_seconds=300)
            seen.add("stalled")
        cancelled = (job / "cancel").exists()
        if cancelled or time.time() > expires:
            reason = "cancelled" if cancelled else "deadline"
            worker.kill()
            worker.wait(timeout=12)
            break
        time.sleep(2)
    result = {"monitor_reason": reason} if reason else {}
    if (job / "worker-result.json").exists():
        result.update(read_json(job / "worker-result.json"))
    if (job / "artifact.json").exists():
        result.update(read_json(job / "artifact.json"))
        outcome = "completed"
    elif plan.get("preflight_only") and (job / "preflight.json").exists():
        result["preflight"] = read_json(job / "preflight.json")
        outcome = "completed"
    else:
        outcome = "cancelled" if reason == "cancelled" or result.get("cancelled") else "failed"
    finish(job, plan, outcome, **result)


def retry_cleanup(job_id):
    job = directory(job_id)
    state = status(job_id)
    if state["phase"] == "review_pending":
        raise ValueError("Use review_runtime with action=release and a reason before requesting cleanup")
    heartbeat = job / "monitor-heartbeat"
    stale = not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 180
    if state["phase"] in {"completed", "failed", "cancelled"} and state.get("runtime_released"):
        raise ValueError("This job already has provider-confirmed cleanup")
    if state["phase"] != "cleanup_pending" and (not stale or not (job / "started").exists()
                                                 or time.time() - state["updated_at"] < 180):
        raise ValueError("Cleanup is only available after an unresolved or interrupted job")
    with (job / "cleanup-retried").open("x", encoding="utf-8") as file:
        file.write(str(time.time()))
    try:
        process = background("recovery", job, "monitor.log")
    except OSError:
        (job / "cleanup-retried").unlink(missing_ok=True)
        raise
    return {"job_id": job_id, "phase": "cleanup_requested", "monitor_pid": process.pid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("monitor", "worker", "recovery", "review-release", "status"))
    parser.add_argument("job_id")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()
    if args.log_file:
        log_stream = open(args.log_file, "a", encoding="utf-8", errors="replace", buffering=1)
        try:
            os.dup2(log_stream.fileno(), 1)
            os.dup2(log_stream.fileno(), 2)
        except OSError:
            pass
        sys.stdout = log_stream
        sys.stderr = log_stream
    job = directory(args.job_id)
    if args.command == "status":
        print(json.dumps(status(args.job_id), ensure_ascii=True, indent=2))
        return
    plan = read_json(job / "plan.json")
    if (job / "started").read_text() != sha256(job / "plan.json"):
        raise RuntimeError("Refusing an unapproved or modified plan")
    if args.command == "monitor":
        try:
            monitor(job, plan)
        except Exception as error:
            finish(job, plan, "failed", error=str(error))
    elif args.command == "review-release":
        release_reviewed_runtime(job, plan)
    elif args.command == "recovery":
        if (job / "runtime-review-release.json").exists():
            release_reviewed_runtime(job, plan)
            return
        data = read_json(job / "artifact.json") if (job / "artifact.json").exists() else {}
        if plan.get("preflight_only") and (job / "preflight.json").exists():
            data["preflight"] = read_json(job / "preflight.json")
        state = status(job.name)
        outcome = "completed" if data else "cancelled" if state.get("outcome") == "cancelled" else "failed"
        finish(job, plan, outcome, **data)
    else:
        try:
            run_worker(job, plan)
        except Exception as error:
            save(job / "worker-result.json", {"error": str(error), "cancelled": isinstance(error, InterruptedError)})


if __name__ == "__main__":
    main()
