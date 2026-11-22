"""Select the local platform's uv launcher and install this local Codex plugin."""
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parent
    command = "uvw" if os.name == "nt" else "uv"
    if shutil.which(command) is None:
        raise SystemExit(f"Install uv with the '{command}' launcher available on PATH.")
    manifest = root / "plugins/minimax-design/.mcp.json"
    config = json.loads(manifest.read_text(encoding="utf-8"))
    config["mcpServers"]["minimax-design-compute"]["command"] = command
    manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    output = subprocess.run(["codex", "plugin", "marketplace", "list"], capture_output=True,
                            encoding="utf-8", check=True).stdout
    entry = next((line for line in output.splitlines() if line.startswith("minimax-design-local ")), "")
    if not entry:
        subprocess.run(["codex", "plugin", "marketplace", "add", str(root)], check=True)
    else:
        configured_path = Path(entry.split(maxsplit=1)[1].strip()).resolve()
        if configured_path != root:
            raise SystemExit(f"Marketplace minimax-design-local is already configured at {configured_path}. "
                             "Update that source and run its install.py.")
    subprocess.run(["codex", "plugin", "add", "minimax-design@minimax-design-local", "--json"], check=True)


if __name__ == "__main__":
    main()
