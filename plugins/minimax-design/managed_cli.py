"""Use the official versioned Colab CLI natively when a local ADC file is available."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
from urllib.request import urlopen

BASELINE = "0.8.0"
VERSION = re.compile(r"\d+\.\d+\.\d+")


def latest():
    try:
        with urlopen("https://pypi.org/pypi/google-colab-cli/json", timeout=5) as response:
            version = json.load(response)["info"]["version"]
        return version if VERSION.fullmatch(version) else None
    except (OSError, ValueError, KeyError):
        return None


def prefix(version):
    if not VERSION.fullmatch(version):
        raise ValueError("Invalid official CLI version")
    launcher = "uvw" if os.name == "nt" and shutil.which("uvw") else "uv"
    return [launcher, "run", "--isolated", "--no-project", "--no-python-downloads",
            "--with", "google-colab-cli==" + version, "colab"]


def supports(profile):
    return (not profile.get("network_route_spec") and bool(shutil.which("uv"))
            and Path(profile.get("credential_file") or "/missing").is_file())


class NativeCLI:
    def __init__(self, root, profile, version, legacy):
        self.root = root / "cli-sessions" / profile["id"]
        self.profile = profile
        self.version = version
        self.legacy = legacy

    def run_colab(self, account, argv, timeout_seconds=60, stdin=None, on_output=None):
        if account != self.profile["name"]:
            raise ValueError("Managed Colab transport is bound to one account")
        self.root.mkdir(parents=True, exist_ok=True)
        if argv[0] == "drivemount" and len(argv) == 4 and argv[1] == "-s" and argv[3] == "/content/drive":
            if not re.fullmatch(r"h3-[a-f0-9]{24}-[0-9]+", argv[2]) or stdin:
                raise ValueError("Drive mounting requires the exact owned MiniMax Design session")
            args = argv
        else:
            args = self.legacy._validate_argv({**self.profile, "transport": "native"}, argv, stdin)
        command = prefix(self.version) + ["--auth", "adc", "--config", str(self.root / "sessions.json"), *args]
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("GOOGLE_", "GCLOUD_", "CLOUDSDK_"))}
        env.update(GOOGLE_APPLICATION_CREDENTIALS=self.profile["credential_file"], PYTHONIOENCODING="utf-8",
                   PYTHONUNBUFFERED="1", NO_COLOR="1")
        process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=os.name != "nt",
                                   creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        buffers = {"stdout": "", "stderr": ""}
        failures = []

        def reader(key):
            def emit(value):
                buffers[key] = (buffers[key] + value)[-262144:]
                if on_output:
                    on_output(value)
            redactor = self.legacy._LineRedactor(emit, whoami=argv[0] == "whoami" and key == "stdout")
            try:
                with getattr(process, key) as stream:
                    while chunk := stream.read1(4096):
                        redactor.feed(chunk)
                redactor.feed(b"", final=True)
            except Exception:
                failures.append(key)

        def writer():
            try:
                with process.stdin:
                    process.stdin.write(stdin.encode("utf-8"))
            except OSError:
                pass

        readers = [threading.Thread(target=reader, args=(key,), daemon=True) for key in buffers]
        for task in readers:
            task.start()
        if stdin is not None:
            threading.Thread(target=writer, daemon=True).start()
        try:
            code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                try:
                    subprocess.run(["taskkill.exe", "/F", "/T", "/PID", str(process.pid)], check=False,
                                   stdin=subprocess.DEVNULL, capture_output=True,
                                   timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=8)
            code = 124
        for task in readers:
            task.join(timeout=2)
        if failures or any(task.is_alive() for task in readers):
            if code == 0:
                code = 70
        return {"exit_code": code, **buffers}
