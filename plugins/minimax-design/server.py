"""Codex MCP tools for local video jobs. The worker is independent of this process."""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import controller
import design_bridge
from runtime_remote import SEGMENT_ID

STR = {"type": "string"}
BOOL = {"type": "boolean"}
INT = {"type": "integer"}
NUM = {"type": "number"}
DICT = {"type": "object"}
CLI_TRANSPORT = {"type": "string", "enum": list(controller.CLI_TRANSPORTS)}
QUEUE_HOOK = {"type": "string", "enum": list(controller.QUEUE_HOOK_MODES)}


def mapping(value):
    return {"type": "object", "additionalProperties": value}


TOOLS = {}


def expose(name, handler, description, properties=None, required=()):
    TOOLS[name] = (handler, {
        "name": name, "description": description, "inputSchema": {
            "type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}})


expose("doctor", controller.doctor, "Check the local Colab transport, login and runtime configuration without starting cloud compute. Live checks use the selected cli_transport.",
       {"name": STR, "live": BOOL, "cli_transport": CLI_TRANSPORT})
expose("check_colab_cli_update", controller.check_cli_update, "Check the upstream official Colab CLI release and validate the best available version without allocating compute.",
       {"name": STR, "cli_transport": CLI_TRANSPORT})
expose("configure_runtime", controller.configure_runtime, "Read or update local-only runtime settings, including preferred GPUs and optional pre-mounted caches.",
       {"gpus": {"type": "array", "items": STR},
        "cache_dir": STR, "drive_cache_relative": STR, "wheelhouse": STR, "offline": BOOL, "asset_workers": INT,
        "design_executable": STR, "queue_hook": QUEUE_HOOK})
expose("colab_accounts", controller.account, "List, add, sign in, verify, select, rename or remove saved Colab accounts, or read usage. Use ADC file paths only; auth returns the official interactive sign-in command.",
       {"action": {"type": "string", "enum": ["list", "add", "auth", "verify", "use", "rename", "remove", "usage"]},
        "name": STR, "new_name": STR, "expected_email": STR, "credential_file": STR, "config_path": STR,
        "transport": {"enum": ["wsl", "native"]}, "cli_transport": CLI_TRANSPORT, "distribution": STR, "executable": STR})
expose("list_cards", controller.list_cards, "List available Colab model cards, weight variants and LoRAs.")
expose("register_card", controller.register_card, "Install a user-supplied local model-card JSON. No GPU is allocated.",
       {"path": STR, "replace": BOOL}, ["path"])
expose("list_loras", controller.list_loras, "List independently registered LoRAs with their compatible model cards and active variants.")
expose("register_lora", controller.register_lora, "Register a local or Hugging Face safetensors LoRA independently of an existing native H3 model card. Explicitly select compatible variants. No GPU is allocated and remote weights are not downloaded.",
       {"card_id": STR, "lora_id": STR, "variants": {"type": "array", "items": STR}, "replace": BOOL,
        "local_path": STR, "hf_repo": STR, "hf_revision": STR, "hf_file": STR, "steps": INT},
       ["card_id", "lora_id", "variants"])
expose("plan_video", controller.plan_video, "Freeze a local plan with assets, LoRAs and GPU candidates. Native H3 defaults to three takes per clip in one GPU session; set takes=1 or takes=2 to override. Each continuous take runs its own ordered sequence. cli_transport supports configured, native, and auto. preflight_only checks hardware and storage.",
       {"card": STR, "account": STR, "cli_transport": CLI_TRANSPORT,
        "runtime_shape": {"type": "string", "enum": list(controller.RUNTIME_SHAPES)},
        "audio_mode": {"type": "string", "enum": ["generated", "mute"]},
        "continuation": {"type": "string", "enum": ["independent", "previous_tail"]},
        "prompt": STR, "width": INT, "height": INT, "seconds": NUM, "seed": INT,
        "takes": {"type": "integer", "minimum": 1, "maximum": 3},
        "segments": {"type": "array", "minItems": 2, "maxItems": 8, "items": {
            "type": "object", "properties": {"id": {"type": "string", "pattern": SEGMENT_ID.pattern},
                                              "prompt": STR, "seconds": NUM, "seed": INT},
            "required": ["id", "prompt", "seconds"], "additionalProperties": False}},
        "gpus": {"type": "array", "items": {"type": "string", "enum": list(controller.GPU_MEMORY)}},
        "variant": STR, "loras": mapping(NUM), "references": mapping(STR), "bindings": DICT, "steps": INT,
        "preflight_only": BOOL, "hold_for_generation": BOOL, "queue_hook": QUEUE_HOOK,
        "timeout_seconds": INT, "setup_timeout_seconds": INT})
expose("start_video", controller.start_video, "Run a frozen plan in its own Colab GPU session. Up to two jobs per account can run independently, subject to Colab capacity. Download each verified MP4 and release only this job's session.",
       {"job_id": STR, "plan_sha256": STR, "authorize_cloud": BOOL}, ["job_id", "plan_sha256", "authorize_cloud"])
expose("video_status", controller.status, "Read the local persisted status, release receipt and verified paths, including already downloaded segments of an unfinished batch.",
       {"job_id": STR}, ["job_id"])
expose("cancel_video", controller.cancel_video, "Cancel one owned run and release its exact Colab session.",
       {"job_id": STR}, ["job_id"])
expose("review_runtime", controller.review_runtime, "Inspect a retained failed Colab runtime, record a decision to keep it, or release it after review.",
       {"job_id": STR, "action": {"type": "string", "enum": ["inspect", "keep", "release"]}, "reason": STR},
       ["job_id", "action"])
expose("resume_video", controller.resume_video, "Submit a frozen generation plan with a new LoRA or prompt to a runtime waiting after setup.",
       {"job_id": STR, "plan_job_id": STR, "plan_sha256": STR, "authorize_cloud": BOOL},
       ["job_id", "plan_job_id", "plan_sha256", "authorize_cloud"])
expose("retry_cleanup", controller.retry_cleanup, "Recover and release the exact owned session from an unresolved or interrupted job.",
       {"job_id": STR}, ["job_id"])
expose("design_projects", design_bridge.list_projects, "Read the local MiniMax Design project list.")
expose("ensure_design", design_bridge.ensure_design, "Start the installed local MiniMax Design app when needed, wait for its MCP, and verify list_projects.")
expose("list_task_handoffs", design_bridge.list_handoffs,
       "Find recent Design handoffs and Colab jobs from any MCP client using the same local state directory.",
       {"limit": INT})
expose("find_design_tasks", design_bridge.project_tasks,
       "Read existing tasks in a Design project, including tasks whose original client disconnected before registering a handoff.",
       {"project_id": STR, "limit": INT}, ["project_id"])
expose("continue_design_task", design_bridge.continue_task,
       "Record the original project and task IDs for handoff, then inspect the same task and its recovery actions. Set wait for one provider wait on that task.",
       {"project_id": STR, "task_id": STR, "wait": BOOL}, ["project_id", "task_id"])
expose("read_design_workflow", design_bridge.read_workflow, "Read a real Design ComfyUI node's draft and model preflight.",
       {"project_id": STR, "source_node_id": STR}, ["project_id", "source_node_id"])
expose("deliver_video_to_design", design_bridge.deliver_video, "Import a verified MP4 into Design. Supply segment_id when importing a specific batch segment.",
       {"job_id": STR, "project_id": STR, "composition_id": STR, "name": STR, "segment_id": STR},
       ["job_id", "project_id", "composition_id", "name"])


def dispatch(packet):
    method = packet.get("method")
    params = packet.get("params", {})
    if method == "initialize":
        requested = params.get("protocolVersion")
        versions = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
        return {"protocolVersion": requested if requested in versions else "2025-11-25",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "minimax-design-compute", "version": "0.1.21"}}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [schema for _, schema in TOOLS.values()]}
    if method != "tools/call":
        raise NotImplementedError(method)
    try:
        tool, schema = TOOLS[params["name"]]
        args = params.get("arguments", {})
        inputs = schema["inputSchema"]
        if not isinstance(args, dict) or set(args) - inputs["properties"].keys() or set(inputs["required"]) - set(args):
            raise ValueError("Invalid tool arguments")
        result = tool(**args)
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}
    except Exception as error:
        return {"isError": True, "content": [{"type": "text", "text": str(error)}]}


def is_local_read(packet):
    if packet.get("method") != "tools/call":
        return True
    params = packet.get("params") or {}
    if not isinstance(params, dict):
        return False
    name = params.get("name")
    return (name in {"video_status", "list_cards", "list_loras", "list_task_handoffs"}
            or name == "configure_runtime" and not params.get("arguments"))


def main():
    def check_update():
        try:
            controller.check_cli_update()
        except Exception as error:
            controller.save(controller.ROOT / "cli-update.json", {"error": str(error)})

    threading.Thread(target=check_update, daemon=True).start()
    output_lock = threading.Lock()

    def respond(packet):
        try:
            reply = {"jsonrpc": "2.0", "id": packet["id"], "result": dispatch(packet)}
        except NotImplementedError as error:
            reply = {"jsonrpc": "2.0", "id": packet["id"], "error": {"code": -32601, "message": str(error)}}
        except (TypeError, ValueError, AttributeError) as error:
            reply = {"jsonrpc": "2.0", "id": packet["id"], "error": {"code": -32600, "message": str(error)}}
        write(reply)

    def write(reply):
        with output_lock:
            sys.stdout.buffer.write(json.dumps(reply, ensure_ascii=True).encode() + b"\n")
            sys.stdout.buffer.flush()

    with ThreadPoolExecutor(max_workers=1) as operations:
        for line in sys.stdin.buffer:
            try:
                packet = json.loads(line)
                if not isinstance(packet, dict):
                    raise ValueError("MCP messages must be objects")
            except ValueError as error:
                write({"jsonrpc": "2.0", "id": None,
                       "error": {"code": -32600, "message": str(error)}})
                continue
            if "id" not in packet:
                continue
            if is_local_read(packet):
                respond(packet)
            else:
                operations.submit(respond, packet)


if __name__ == "__main__":
    main()
