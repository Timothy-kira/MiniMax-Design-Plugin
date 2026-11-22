"""Use the installed Design MCP's documented local project interfaces."""
from contextlib import contextmanager
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import controller
from controller import directory, read_json, save, settings, sha256

ENDPOINT = "http://127.0.0.1:18810/mcp"
_LAUNCH_LOCK = threading.Lock()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


def port_ready():
    try:
        with socket.create_connection(("127.0.0.1", 18810), timeout=0.5):
            return True
    except OSError:
        return False


def wake_design():
    with _LAUNCH_LOCK:
        if port_ready():
            return {"launched": False}
        explicit = settings().get("design_executable")
        if explicit:
            executable = Path(explicit)
            if not executable.is_file():
                raise ValueError("Configured MiniMax Design executable was not found")
            command = [str(executable)]
        elif os.name == "nt":
            local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
            executable = local / "Programs/MiniMax Design/current/MiniMax Design.exe"
            if not executable.is_file():
                raise RuntimeError("MiniMax Design is not installed at its standard Windows path")
            command = [str(executable)]
        elif sys.platform == "darwin":
            applications = (Path("/Applications"), Path.home() / "Applications")
            app = next((folder / "MiniMax Design.app" for folder in applications
                        if (folder / "MiniMax Design.app").is_dir()), None)
            if app is None:
                raise RuntimeError("MiniMax Design.app was not found")
            command = ["/usr/bin/open", "-a", str(app)]
        else:
            executable = shutil.which("minimax-design")
            if not executable:
                raise RuntimeError("Linux needs an installed MiniMax Design executable or a running local Design MCP")
            command = [executable]
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=os.name != "nt",
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        deadline = time.monotonic() + 22
        while time.monotonic() < deadline:
            if port_ready():
                return {"launched": True}
            time.sleep(0.35)
        raise RuntimeError("MiniMax Design was started, but its local MCP did not become reachable")


@contextmanager
def connect(wake=True, timeout=25):
    if wake:
        wake_design()
    opener = build_opener(ProxyHandler({}), NoRedirect())
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    index = 0

    def rpc(method, params, notification=False):
        nonlocal index
        index += 1
        packet = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            packet["id"] = index
        with opener.open(Request(ENDPOINT, json.dumps(packet).encode(), headers), timeout=timeout) as response:
            if session := response.headers.get("Mcp-Session-Id"):
                headers["Mcp-Session-Id"] = session
            if notification:
                return {}
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                for line in response:
                    if line.startswith(b"data:"):
                        result = json.loads(line[5:])
                        if result.get("id") == index:
                            break
                else:
                    raise RuntimeError("Design MCP closed before its response")
            else:
                result = json.load(response)
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result["result"]

    init = rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                              "clientInfo": {"name": "minimax-design-compute", "version": "0.1.21"}})
    headers["MCP-Protocol-Version"] = init["protocolVersion"]
    rpc("notifications/initialized", {}, notification=True)

    def call(name, **arguments):
        result = rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(str(result.get("structuredContent") or result.get("content")))
        if "structuredContent" in result:
            return result["structuredContent"]
        chunks = [block["text"] for block in result.get("content", []) if block.get("type") == "text"]
        return json.loads(chunks[0]) if len(chunks) == 1 else chunks

    try:
        yield call
    finally:
        if "Mcp-Session-Id" in headers:
            try:
                with opener.open(Request(ENDPOINT, headers=headers, method="DELETE"), timeout=3):
                    pass
            except OSError:
                pass


def list_projects():
    with connect() as call:
        return call("list_projects")


def ensure_design():
    launch = wake_design()
    with connect(wake=False) as call:
        return {**launch, "projects": call("list_projects")}


def project_tasks(project_id, limit=20):
    with connect() as call:
        call("list_projects")
        call("open_project", projectId=project_id)
        return call("task_query", mode="list", query={"limit": limit})


def continue_task(project_id, task_id, wait=False):
    if not project_id.strip() or not task_id.strip():
        raise ValueError("Provide the original Design project ID and task ID")
    key = hash_bytes(json.dumps([project_id, task_id]).encode()).hexdigest()
    binding = controller.ROOT / "design-task-handoffs" / f"{key}.json"
    record = {"provider": "design", "project_id": project_id, "task_id": task_id}
    if not binding.exists():
        save(binding, record)
    with connect(timeout=45 if wait else 25) as call:
        call("list_projects")
        call("open_project", projectId=project_id)
        if wait:
            result = call("task_wait", taskId=task_id)
        else:
            result = call("task_query", mode="get", taskId=task_id)
    task = result["task"]
    save(binding, {**record, "status": task.get("status"), "revision": task.get("revision")})
    return {**record, "task": task}


def list_handoffs(limit=20):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Limit must be an integer from 1 to 100")
    design = (controller.ROOT / "design-task-handoffs").glob("*.json")
    colab = (controller.ROOT / "jobs").glob("h3-*/status.json")
    candidates = [(path, "design") for path in design] + [(path, "colab") for path in colab]
    recent = sorted(candidates, key=lambda item: item[0].stat().st_mtime, reverse=True)[:limit]
    items = []
    for path, provider in recent:
        record = read_json(path)
        if provider == "colab":
            record = {"provider": provider, "task_id": path.parent.name, "status": record.get("phase"),
                      "runtime_released": record.get("runtime_released")}
        items.append(record)
    return {"items": items}


def read_workflow(project_id, source_node_id):
    with connect() as call:
        call("open_project", projectId=project_id)
        graph = call("get_comfyui_workflow", source_node_id=source_node_id)
        preflight = call("preflight_comfyui_workflow", source_node_id=source_node_id)
    return {"source_node_id": source_node_id, "workflow": graph, "model_dependencies": preflight}


def deliver_video(job_id, project_id, composition_id, name, segment_id=None):
    if not 1 <= len(name.strip()) <= 200:
        raise ValueError("Design resource name must be 1–200 characters")
    job = directory(job_id)
    if segment_id is None:
        receipt = read_json(job / "artifact.json")
        if "segment_outputs" in receipt:
            raise ValueError("Select a verified segment_id from video_status")
        stem = "video"
    else:
        receipts = read_json(job / "segment-outputs.json")
        receipt = next((item for item in receipts if item["id"] == segment_id), None)
        if receipt is None:
            raise ValueError("Select a verified segment_id from video_status")
        stem = f"video-{receipt['id']}"
    video = job / f"{stem}.mp4"
    if video.stat().st_size != receipt["size_bytes"] or sha256(video) != receipt["sha256"]:
        raise ValueError("Local video changed after checksum verification")
    if receipt["size_bytes"] > 50 * 1024 * 1024:
        raise ValueError("Design MCP's workspace-file delivery limit is 50 MiB. The original MP4 is available locally.")
    request_key = f"{project_id}/{composition_id}"
    if segment_id is not None:
        request_key += f"/{segment_id}"
    request_id = f"{job_id}-{hash_bytes(request_key.encode()).hexdigest()[:16]}"
    binding = job / f"design-{request_id}.json"
    if binding.exists():
        record = read_json(binding)
        if record["name"] != name:
            raise ValueError("An earlier request for this destination used a different resource name")
        if "receipt" in record:
            return record["receipt"]
    record = {"project_id": project_id, "composition_id": composition_id, "name": name}
    save(binding, record)
    with connect() as call:
        project = call("open_project", projectId=project_id)["project"]
        root = Path(project["folderPath"]).resolve(strict=True)
        suffix = f"-{segment_id}" if segment_id is not None else ""
        staged = root / f"H3-Colab-{job_id}{suffix}.mp4"
        if staged.is_symlink():
            raise ValueError("Design staging target is a symlink")
        if staged.exists():
            if sha256(staged) != receipt["sha256"]:
                raise ValueError("Existing Design staging file has different content")
        else:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".minimax-design-", delete=False) as output:
                temporary = Path(output.name)
            try:
                shutil.copyfile(video, temporary)
                if sha256(temporary) != receipt["sha256"]:
                    raise ValueError("Staged video failed integrity verification")
                temporary.replace(staged)
            finally:
                temporary.unlink(missing_ok=True)
        delivered = call("deliver", requestId=request_id, sources=[{"path": str(staged)}],
                         output={"resources": [{"mode": "create", "name": name}],
                                 "composition": {"compositionId": composition_id,
                                                 "placements": [{"role": "成片"}]}})
    save(binding, {**record, "receipt": delivered})
    return delivered
