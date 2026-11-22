import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import controller
import design_bridge
import server
from model_cards import make_plan, sha256


EXEC_PRELUDE_DISCONNECT = (
    "│ packages\\colab_cli\\commands\\execution.py:225 in exec_command │\n"
    "│ 226 │ \"import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')\" │\n"
    "RuntimeError: Connection was lost.\n"
)


ALLOCATION_DISCONNECT = (
    "│ /usr/local/lib/colab_cli/client.py:357 in _post_assignment                   │\n"
    "ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed \n"
    "connection without response'))\n"
)


class Pool:
    def __init__(self):
        self.releases = []

    def admit(self, **unused):
        return {"claim": {"id": "test-owned-claim"}}

    def update(self, *args):
        pass

    def release(self, claim, outcome, owner, **unused):
        self.releases.append((claim, outcome, owner))


class AccountStore:
    def __init__(self, profile):
        self.profile = profile

    @contextmanager
    def _transaction(self):
        yield {"accounts": {self.profile["name"]: self.profile}}

    def resolve(self, name):
        if name and name != self.profile["name"]:
            raise ValueError("Unknown fixture account")
        return self.profile

    def parse_whoami(self, stdout):
        return {"email": stdout.strip()}


class ColabFixture:
    def __init__(self, mismatched_first=False, corrupt_downloads=0, defer_exec_exit=False,
                 rejected_gpu=False, fail_segment=None, drop_preflight_once=False):
        self.calls = []
        self.live = set()
        self.mismatched_first = mismatched_first
        self.storage_errors = {}
        self.drop_preflight_once = drop_preflight_once
        self.corrupt_downloads = corrupt_downloads
        self.video = b"verified test video bytes"
        self.defer_exec_exit = defer_exec_exit
        self.download_started = threading.Event()
        self.received_download_before_exec_exit = None
        self.rejected_gpu = rejected_gpu
        self.preflights = []
        self.fail_segment = fail_segment
        self.timeout_after_outputs = False
        self.batch_video = {}
        self.first_segment_downloaded_before_next = None
        self.download_throws = False
        self.render_setup_markers = []
        self.generation_configs = []
        self.email = "test@example.com"
        self.cli_version = "0.8.0"

    def run_colab(self, account, argv, timeout_seconds, stdin=None, on_output=None):
        self.calls.append(argv[:])
        result = {"exit_code": 0, "stdout": "", "stderr": ""}
        if argv == ["whoami"]:
            return {**result, "stdout": self.email}
        if argv == ["version"]:
            return {**result, "stdout": f"Version: {self.cli_version}\n"}
        if argv == ["usage"]:
            return {**result, "stdout": "Current balance: 100.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n"}
        operation, _, session, *_ = argv
        if operation == "new":
            if self.rejected_gpu and session.endswith("-1"):
                return {**result, "exit_code": 1,
                        "stderr": "[colab] Backend rejected accelerator 'H100'. You may not have quota or entitlement for this accelerator on your account.\n"}
            self.live.add(session)
        if operation == "exec" and stdin.startswith("CONFIG"):
            conf = ast.literal_eval(stdin.splitlines()[0].partition("=")[2].strip())
            if conf.get("preflight_only"):
                self.preflights.append(conf)
            def send(phase, **data):
                on_output(controller.MARK + json.dumps({"job_id": conf["job_id"], "nonce": conf["nonce"],
                                                        "phase": phase, **data}) + "\n")
            if not conf.get("preflight_only"):
                for phase, fields in self.render_setup_markers:
                    send(phase, **fields)
                    if phase == "failed":
                        return result
            if self.drop_preflight_once and conf.get("preflight_only"):
                self.drop_preflight_once = False
                return {**result, "exit_code": 124, "stderr": "RuntimeError: Connection was lost.\n"}
            if self.mismatched_first and session.endswith("-1"):
                send("failed", error_kind="capacity_mismatch")
                result["exit_code"] = 1
            elif conf.get("preflight_only"):
                send("gpu_verified", gpu="NVIDIA A100-SXM4-80GB")
                if (exit_code := self.storage_errors.get(len(self.preflights))) is not None:
                    send("failed", error_kind="storage", message="Insufficient Colab disk space")
                    return {**result, "exit_code": exit_code}
                send("preflight_ready", cached_asset_sha256=[])
            elif conf.get("prepare_only"):
                send("prepared", asset_count=len(conf["assets"]), environment_cache_hit=False)
            elif "segments" in conf:
                self.generation_configs.append(conf)
                for index, segment in enumerate(conf["segments"]):
                    segment_id = segment["id"]
                    if index == 1:
                        self.first_segment_downloaded_before_next = self.download_started.wait(3)
                    if segment_id == self.fail_segment:
                        send("failed", error_kind="generation", message=f"Segment {segment_id} failed in ComfyUI",
                             segment_id=segment_id)
                        result["exit_code"] = 1
                        break
                    payload = self.video + b":" + segment_id.encode()
                    self.batch_video[f"video-{segment_id}.mp4"] = payload
                    send("output_ready", size_bytes=len(payload), sha256=hash_bytes(payload).hexdigest(),
                         segment_id=segment_id)
                if self.timeout_after_outputs:
                    send("failed", error_kind="timeout", message="timeout")
                    result["exit_code"] = 1
            else:
                send("output_ready", size_bytes=len(self.video), sha256=hash_bytes(self.video).hexdigest())
                if self.defer_exec_exit:
                    self.received_download_before_exec_exit = self.download_started.wait(3)
        if operation == "download":
            self.download_started.set()
            if self.download_throws:
                self.download_throws = False
                raise TimeoutError("test download transport timeout")
            remote_name = argv[-2].rsplit("/", 1)[-1]
            payload = self.batch_video.get(remote_name, self.video) if not self.corrupt_downloads else b"incomplete transfer"
            self.corrupt_downloads = max(0, self.corrupt_downloads - 1)
            Path(argv[-1]).write_bytes(payload)
        if operation == "stop":
            self.live.discard(session)
            result["stdout"] = "[colab] Session terminated.\n"
        if operation == "status":
            result["stdout"] = (f"[{session}] provider | Hardware: A100 | Shape: Standard | Variant: GPU | Status: IDLE\n"
                                if session in self.live else f"[colab] Session '{session}' not found.\n")
        return result


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        mocked = patch.object(controller, "ROOT", self.root)
        self.addCleanup(mocked.stop)
        mocked.start()
        self.job = controller.directory("h3-" + "a" * 24)
        self.job.mkdir(parents=True)
        self.profile = {"name": "test", "id": "b" * 32, "credential_file": str(self.root / "adc.json")}
        self.pool = Pool()
        self.accounts = AccountStore(self.profile)

    def plan(self, **args):
        request = {"card": "h3-t2v", "takes": 1, **args}
        if "segments" not in args:
            request.setdefault("prompt", "A sample sequence")
        value = make_plan(self.root, request, {})
        return {**value, "job_id": self.job.name, "account": "test", "profile_id": self.profile["id"],
                "profile_fingerprint": controller.snapshot_profile(self.profile),
                "timeout_seconds": 300, "setup_timeout_seconds": 100, "runtime_options": {},
                "runner_sha256": sha256(controller.HERE / "runtime_remote.py")}

    def test_muted_h3_batch_guides_next_shot_from_edit_end(self):
        ref = self.root / "ref.png"
        ref.write_bytes(b"reference upload")
        plan = self.plan(card="h3-r2v", references={"image_1": str(ref)},
                         audio_mode="mute", continuation="previous_tail",
                         segments=[{"id": "first", "prompt": "Open the door", "seconds": 11},
                                   {"id": "second", "prompt": "Walk forward", "seconds": 9}])
        first, second = plan["candidates"][0]["segments"]
        self.assertNotIn("previous_tail", first)
        self.assertEqual(plan["segments"][0]["frames"], 277)
        self.assertEqual(plan["segments"][0]["edit_frames"], 264)
        self.assertEqual(second["previous_tail"], {
            "file": "continuation-tail-1.mp4", "frames": 56, "end_frame": 264})
        graph = second["graph"]
        guide = next(node["inputs"] for node in graph.values() if node["class_type"] == "MiniMaxH3AddGuide")
        self.assertEqual(guide["frame_idx"], 0)
        self.assertEqual(graph[guide["image"][0]]["inputs"]["image"], "continuation-tail-1.png")
        condition = graph[guide["positive"][0]]["inputs"]
        self.assertIn("ref_videos.ref_video_0", condition)
        self.assertIn("Walk forward", condition["prompt"])
        for segment in (first, second):
            self.assertFalse(any(node["class_type"] == "VAEDecodeAudio" for node in segment["graph"].values()))

    def test_h3_single_clip_defaults_to_three_seeded_outputs(self):
        plan = make_plan(self.root, {"prompt": "Open the door", "seed": 42}, {})
        self.assertEqual(plan["takes"], 3)
        self.assertEqual([(part["segment_id"], part["take"]) for part in plan["segments"]],
                         [("clip", 1), ("clip", 2), ("clip", 3)])
        graphs = plan["candidates"][0]["segments"]
        seeds = [next(node["inputs"]["noise_seed"] for node in part["graph"].values()
                      if node["class_type"] == "RandomNoise") for part in graphs]
        self.assertEqual(seeds, [42, 43, 44])

    def test_h3_default_continuous_takes_only_use_their_own_tail(self):
        reference = self.root / "ref.png"
        reference.write_bytes(b"reference upload")
        request = {
            "card": "h3-r2v", "references": {"image_1": str(reference)},
            "continuation": "previous_tail",
            "segments": [{"id": "first", "prompt": "Open the door", "seconds": 11},
                         {"id": "second", "prompt": "Walk forward", "seconds": 9}],
        }
        plan = make_plan(self.root, request, {})
        self.assertEqual([(part["segment_id"], part["take"]) for part in plan["segments"]],
                         [("first", 1), ("second", 1), ("first", 2),
                          ("second", 2), ("first", 3), ("second", 3)])
        for take_start, second in zip(plan["candidates"][0]["segments"][::2],
                                      plan["candidates"][0]["segments"][1::2]):
            self.assertNotIn("previous_tail", take_start)
            self.assertEqual(second["previous_tail"]["end_frame"], 264)
        self.assertEqual(len({part["id"] for part in plan["segments"]}), 6)

    def exercise(self, plan, provider, expected, raises=False):
        with patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "notify"):
            if raises:
                with self.assertRaisesRegex(RuntimeError, raises if isinstance(raises, str) else "storage.*Insufficient Colab disk space"):
                    controller.run_worker(self.job, plan)
            else:
                controller.run_worker(self.job, plan)
            controller.finish(self.job, plan, expected)
        state = controller.status(self.job.name)
        if expected == "failed":
            self.assertTrue(provider.live)
            self.assertEqual(state["phase"], "review_pending")
            self.assertFalse(state["runtime_released"])
            self.assertEqual(self.pool.releases, [])
        else:
            self.assertFalse(provider.live)
            self.assertEqual(state["phase"], expected)
            self.assertTrue(state["runtime_released"])
            self.assertEqual(self.pool.releases, [("test-owned-claim", expected, self.job.name)])

    def approve_fixture_plan(self, plan):
        path = self.job / "plan.json"
        controller.save(path, plan)
        (self.job / "started").write_text(sha256(path))

    def test_first_precision_and_gpu_order(self):
        plan = self.plan(gpus=["L4", "A100", "H100"])
        self.assertEqual((plan["candidates"][0]["precision"], plan["candidates"][0]["gpu"]), ("bf16", "H100"))
        self.assertEqual(plan["frames"], 124)

    def test_standard_a100_request_keeps_the_model_ram_requirement(self):
        provider = ColabFixture()
        self.profile["verified_email"] = provider.email
        with patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)):
            review = controller.plan_video(card="h3-t2v", prompt="probe", gpus=["A100"],
                                           variant="pruned-bf16", runtime_shape="standard", takes=1)
        self.job = controller.directory(review["job_id"])
        plan = controller.read_json(self.job / "plan.json")
        self.assertEqual(review["runtime_shape"], "standard")
        self.exercise(plan, provider, "completed")
        allocation = next(args for args in provider.calls if args[0] == "new")
        self.assertEqual(allocation[3:], ["--gpu", "A100"])
        self.assertEqual(provider.preflights[0]["min_ram_gib"], 45)

    def test_existing_a100_plan_keeps_automatic_high_memory_flag(self):
        provider = ColabFixture()
        plan = self.plan(gpus=["A100"], variant="pruned-bf16")
        self.exercise(plan, provider, "completed")
        allocation = next(args for args in provider.calls if args[0] == "new")
        self.assertIn("--high-mem", allocation)

    def test_configured_cli_uses_frozen_saved_route_for_preflight_and_cleanup(self):
        provider = ColabFixture()
        self.profile.update(verified_email=provider.email, transport="wsl", distribution="Ubuntu-24.04",
                            executable="/opt/colab/bin/colab")
        with (patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller.managed_cli, "supports", return_value=True) as native_available,
              patch.object(controller, "notify")):
            reply = server.dispatch({"method": "tools/call", "params": {"name": "plan_video", "arguments": {
                "prompt": "A sample sequence", "preflight_only": True, "gpus": ["A100"], "cli_transport": "configured",
            }}})
            self.assertNotIn("isError", reply, reply)
            summary = json.loads(reply["content"][0]["text"])
            job = controller.directory(summary["job_id"])
            plan = controller.read_json(job / "plan.json")
            self.assertEqual(sha256(job / "plan.json"), summary["plan_sha256"])
            self.assertEqual(summary["cli_transport"], "configured")
            controller.run_worker(job, plan)
            controller.finish(job, plan, "completed")
            native_available.assert_not_called()
            self.assertEqual(controller.status(job.name)["cli_transport"], "configured")
        self.assertEqual([call[0] for call in provider.calls], ["whoami", "new", "exec", "whoami", "stop"])
        self.assertEqual(controller.read_json(job / "bound-cli.json"), {"version": "legacy"})
        self.assertFalse(provider.live)

    def test_required_native_cli_cannot_fall_back_after_account_mismatch(self):
        provider = ColabFixture()
        self.profile.update(verified_email=provider.email, transport="wsl")
        native = ColabFixture()
        native.email = "another@example.com"
        plan = self.plan() | {"cli_transport": "native"}
        with (patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller.managed_cli, "supports", return_value=True),
              patch.object(controller.managed_cli, "latest", return_value=controller.managed_cli.BASELINE),
              patch.object(controller.managed_cli, "NativeCLI", return_value=native)):
            with self.assertRaisesRegex(RuntimeError, "native CLI could not verify"):
                controller.run_worker(self.job, plan)
        self.assertEqual(native.calls, [["whoami"]])
        self.assertEqual(provider.calls, [])
        self.assertFalse((self.job / "bound-cli.json").exists())

    def test_configured_doctor_promotes_healthy_transport_after_matching_whoami(self):
        self.profile.update(verified_email="test@example.com", transport="wsl")
        for initial, email, expected in (
            ("ready_for_identity_check", "test@example.com", "ready"),
            ("ready_for_identity_check", "other@example.com", "ready_for_identity_check"),
            ("needs_setup", "test@example.com", "needs_setup"),
        ):
            with self.subTest(initial=initial, email=email):
                provider = ColabFixture()
                provider.email = email
                transport = SimpleNamespace(doctor=lambda _name: {"status": initial}, run_colab=provider.run_colab)
                with patch.object(controller, "backend", return_value=(self.accounts, transport, self.pool)):
                    result = controller.doctor("test", live=True, cli_transport="configured")
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["active_cli"], "wsl")
                self.assertEqual(result["cli_authenticated"], email == self.profile["verified_email"])
                if result["cli_authenticated"]:
                    self.assertEqual(result["cli_version"], "0.8.0")
                    self.assertEqual(provider.calls, [["whoami"], ["version"]])
                else:
                    self.assertNotIn("cli_version", result)
                    self.assertEqual(provider.calls, [["whoami"]])

    def test_auto_transport_keeps_configured_wsl_profile_without_native_hijack(self):
        provider = ColabFixture()
        self.profile.update(verified_email=provider.email, transport="wsl")
        with (patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller.managed_cli, "supports", return_value=True) as native_available):
            client, selected = controller.select_cli(self.profile, self.accounts, provider, cli_transport="auto")
            usage = controller.account(action="usage", name="test", cli_transport="configured")
            update_info = controller.check_cli_update(name="test", cli_transport="configured")
        self.assertIs(client, provider)
        self.assertEqual(selected, "legacy")
        native_available.assert_not_called()
        self.assertTrue(usage["verified"])
        self.assertEqual(update_info["version"], "0.8.0")
        self.assertFalse(update_info["native"])

    def test_preflight_retries_once_on_transient_kernel_drop(self):
        provider = ColabFixture(drop_preflight_once=True)
        plan = self.plan(gpus=["A100"]) | {"preflight_only": True}
        with patch.object(controller.time, "sleep"):
            self.exercise(plan, provider, "completed")
        self.assertEqual([call[0] for call in provider.calls], ["new", "exec", "exec", "stop"])
        events = [json.loads(line)["event"] for line in (self.job / "events.jsonl").read_text().splitlines()]
        self.assertIn("preflight_retry", events)

    def test_allocation_timeout_retries_once_after_confirmed_empty_release(self):
        self.exercise_allocation_retry(exit_code=124, stderr="")

    def test_allocation_disconnection_retries_once_after_confirmed_empty_release(self):
        events = self.exercise_allocation_retry(exit_code=1, stderr=ALLOCATION_DISCONNECT)
        failure = next(event for event in events if event["event"] == "allocation_failed")
        self.assertEqual(failure["kind"], "transport_lost")
        self.assertEqual(failure["cli_exit_code"], 1)

    def exercise_allocation_retry(self, exit_code, stderr):
        provider = ColabFixture()
        original = provider.run_colab
        now = [1000.0]

        def flaky(account, argv, timeout_seconds, stdin=None, on_output=None):
            if argv[0] == "new" and argv[2].endswith("-1"):
                provider.calls.append(argv[:])
                now[0] += timeout_seconds if exit_code == 124 else 60
                return {"exit_code": exit_code, "stdout": "", "stderr": stderr}
            if argv[0] == "sessions":
                lease = controller.read_json(self.job / "lease.json")
                self.assertEqual(lease["new_outcome"], "unknown")
                self.assertGreaterEqual(now[0], lease["allocation_deadline"])
                provider.calls.append(argv[:])
                return {"exit_code": 0, "stdout": "[colab] No active sessions found on server.\n", "stderr": ""}
            if argv[0] == "stop" and argv[2].endswith("-1"):
                provider.calls.append(argv[:])
                return {"exit_code": 0, "stdout": f"[colab] Session '{argv[2]}' not found.\n", "stderr": ""}
            return original(account, argv, timeout_seconds, stdin=stdin, on_output=on_output)

        def advance(seconds):
            now[0] += max(0, seconds)

        provider.run_colab = flaky
        plan = self.plan(gpus=["A100"]) | {"preflight_only": True, "timeout_seconds": 900}
        with patch.object(controller.time, "time", side_effect=lambda: now[0]), \
                patch.object(controller.time, "sleep", side_effect=advance):
            self.exercise(plan, provider, "completed")
        actions = [call[0] for call in provider.calls]
        self.assertEqual(actions, ["new", "status", "status", "sessions", "new", "exec", "stop"])
        self.assertNotEqual(provider.calls[0][2], provider.calls[4][2])
        events = [json.loads(line) for line in (self.job / "events.jsonl").read_text().splitlines()]
        retry = next(event for event in events if event["event"] == "allocation_retry")
        self.assertEqual(retry["release_evidence"], "no_active_server_sessions")
        return events

    def test_allocation_timeout_recovers_the_original_session_without_a_second_new(self):
        self.exercise_named_allocation_recovery(exit_code=124, stderr="")

    def test_allocation_disconnection_recovers_the_original_session_without_a_second_new(self):
        self.exercise_named_allocation_recovery(exit_code=1, stderr=ALLOCATION_DISCONNECT)
        self.assertNotIn("allocation_failure", controller.status(self.job.name))

    def exercise_named_allocation_recovery(self, exit_code, stderr):
        provider = ColabFixture()
        original = provider.run_colab
        allocation_timeouts = []

        def late(account, argv, timeout_seconds, stdin=None, on_output=None):
            if argv[0] == "new":
                allocation_timeouts.append(timeout_seconds)
                original(account, argv, timeout_seconds, stdin=stdin, on_output=on_output)
                return {"exit_code": exit_code, "stdout": "", "stderr": stderr}
            if argv[0] == "status" and argv[2] in provider.live:
                provider.calls.append(argv[:])
                return {"exit_code": 0, "stderr": "", "stdout": (
                    f"[{argv[2]}] test-endpoint | Hardware: A100 | Shape: High RAM | Variant: GPU | Status: IDLE\n")}
            return original(account, argv, timeout_seconds, stdin=stdin, on_output=on_output)

        provider.run_colab = late
        plan = self.plan(gpus=["A100"]) | {"preflight_only": True, "timeout_seconds": 900}
        self.exercise(plan, provider, "completed")

        self.assertEqual([call[0] for call in provider.calls], ["new", "status", "exec", "stop"])
        self.assertEqual(len({call[2] for call in provider.calls}), 1)
        self.assertGreaterEqual(allocation_timeouts[0], 120 + 120 + 10 + 120)
        events = [json.loads(line)["event"] for line in (self.job / "events.jsonl").read_text().splitlines()]
        self.assertIn("allocation_recovered", events)

    def test_allocation_disconnection_does_not_retry_without_confirmed_release(self):
        provider = ColabFixture()
        now = [1000.0]
        plan = self.plan(gpus=["A100"]) | {"preflight_only": True, "timeout_seconds": 900}

        def disconnected(_account, argv, timeout_seconds, stdin=None, on_output=None):
            provider.calls.append(argv[:])
            if argv[0] == "new":
                return {"exit_code": 1, "stdout": "", "stderr": ALLOCATION_DISCONNECT}
            if argv[0] == "sessions":
                return {"exit_code": 0, "stderr": "", "stdout": "[?] unconfirmed endpoint | Hardware: A100\n"}
            return {"exit_code": 0, "stderr": "", "stdout": f"[colab] Session '{argv[2]}' not found.\n"}

        def advance(seconds):
            now[0] += seconds

        provider.run_colab = disconnected
        with (patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "notify"), patch.object(controller.time, "time", side_effect=lambda: now[0]),
              patch.object(controller.time, "sleep", side_effect=advance)):
            with self.assertRaisesRegex(RuntimeError, "closed without a response"):
                controller.run_worker(self.job, plan)
            controller.finish(self.job, plan, "failed")

        self.assertEqual([call[0] for call in provider.calls].count("new"), 1)
        self.assertFalse(any(call[0] == "exec" for call in provider.calls))
        state = controller.status(self.job.name)
        self.assertEqual(state["phase"], "review_pending")
        self.assertFalse(any(call[0] == "stop" for call in provider.calls))
        self.assertFalse(state["runtime_released"])
        self.assertEqual(state["allocation_failure"]["kind"], "transport_lost")
        self.assertEqual(controller.read_json(self.job / "lease.json")["new_outcome"], "unknown")

    def test_allocation_backend_503_is_recorded_without_retrying_or_shortening_cleanup(self):
        provider = ColabFixture()
        now = [1000.0]
        plan = self.plan(gpus=["A100", "L4"]) | {"preflight_only": True, "timeout_seconds": 900}
        self.approve_fixture_plan(plan)

        def unavailable(_account, argv, timeout_seconds, stdin=None, on_output=None):
            provider.calls.append(argv[:])
            operation = argv[0]
            if operation == "new":
                now[0] += 60
                return {"exit_code": 1, "stdout": "", "stderr": (
                    "ColabRequestError: Failed to issue request POST \n"
                    "[REDACTED_URL]\n"
                    "notebook-fragment-secret&variant=GPU&accelerator=A100&shape=hm: Service Unavailable\n")}
            lease = controller.read_json(self.job / "lease.json")
            self.assertEqual(lease["new_outcome"], "unknown")
            self.assertEqual(lease["allocation_deadline"], 1000 + controller.COLAB_ALLOCATION_TIMEOUT
                             + controller.COLAB_ALLOCATION_SETTLE)
            if operation in ("stop", "status"):
                return {"exit_code": 0, "stderr": "", "stdout": f"[colab] Session '{argv[2]}' not found.\n"}
            self.assertEqual(operation, "sessions")
            self.assertGreaterEqual(now[0], lease["allocation_deadline"])
            return {"exit_code": 0, "stderr": "", "stdout": "[colab] No active sessions found on server.\n"}

        def run_background(command, job, _log):
            if command == "review-release":
                controller.release_reviewed_runtime(job, plan)
                return SimpleNamespace(pid=123)
            try:
                controller.run_worker(job, plan)
            except RuntimeError as error:
                controller.save(job / "worker-result.json", {"error": str(error)})
            return SimpleNamespace(poll=lambda: 0)

        def advance(seconds):
            now[0] += seconds

        provider.run_colab = unavailable
        with (patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "background", side_effect=run_background), patch.object(controller, "notify"),
              patch.object(controller.time, "time", side_effect=lambda: now[0]),
              patch.object(controller.time, "sleep", side_effect=advance)):
            controller.monitor(self.job, plan)
            self.assertEqual([call[0] for call in provider.calls], ["new"])
            self.assertEqual(controller.status(self.job.name)["phase"], "review_pending")
            controller.review_runtime(self.job.name, "release", reason="503 allocation ended")

        self.assertEqual([call[0] for call in provider.calls], ["new", "status", "sessions"])
        state = controller.status(self.job.name)
        self.assertEqual(state["phase"], "failed")
        self.assertTrue(state["runtime_released"])
        failure = state["allocation_failure"]
        self.assertEqual((failure["kind"], failure["http_status"], failure["cli_exit_code"]),
                         ("backend_unavailable", 503, 1))
        self.assertEqual(failure["message"], state["error"])
        self.assertIn("HTTP 503 Service Unavailable", state["error"])
        self.assertNotIn("notebook-fragment-secret", json.dumps(state))
        event_text = (self.job / "events.jsonl").read_text()
        self.assertNotIn("notebook-fragment-secret", event_text)
        event = next(event for line in event_text.splitlines() if (event := json.loads(line))["event"] == "allocation_failed")
        self.assertEqual(event["http_status"], 503)
        self.assertEqual(controller.read_json(self.job / "lease.json")["release_evidence"], "no_active_server_sessions")

    def test_old_plan_budget_never_starts_a_second_allocation_without_time(self):
        provider = ColabFixture()
        original = provider.run_colab
        now = [1000.0]
        allocation_timeouts = []

        def timed_out(account, argv, timeout_seconds, stdin=None, on_output=None):
            if argv[0] == "new":
                provider.calls.append(argv[:])
                allocation_timeouts.append(timeout_seconds)
                now[0] += timeout_seconds
                return {"exit_code": 124, "stdout": "", "stderr": ""}
            return original(account, argv, timeout_seconds, stdin=stdin, on_output=on_output)

        provider.run_colab = timed_out
        plan = self.plan(gpus=["A100"]) | {"preflight_only": True, "timeout_seconds": 240}

        def advance(seconds):
            now[0] += seconds

        with (patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "notify"), patch.object(controller.time, "time", side_effect=lambda: now[0]),
              patch.object(controller.time, "sleep", side_effect=advance)):
            with self.assertRaisesRegex(RuntimeError, "outcome could not be verified"):
                controller.run_worker(self.job, plan)
            controller.finish(self.job, plan, "failed")

        self.assertEqual(allocation_timeouts, [240 - controller.COLAB_PREFLIGHT_RESERVE])
        self.assertEqual([call[0] for call in provider.calls], ["new"])
        self.assertFalse(controller.status(self.job.name)["runtime_released"])

    def test_legacy_cli_binding_remains_pinned_and_checks_saved_profile(self):
        provider = ColabFixture()
        self.profile.update(verified_email=provider.email, transport="wsl", executable="/opt/colab/bin/colab")
        plan = self.plan()
        controller.save(self.job / "bound-cli.json", {"version": "legacy"})
        with (patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller.managed_cli, "supports", return_value=True) as native_available):
            self.assertIs(controller.identity(plan, self.job)[1], provider)
            self.profile["executable"] = "/different/bin/colab"
            with self.assertRaisesRegex(RuntimeError, "profile changed after planning"):
                controller.identity(plan, self.job)
            native_available.assert_not_called()
        self.assertEqual(provider.calls, [["whoami"]])

    def test_selected_turbo_loads_the_lora_and_steps(self):
        plan = self.plan(loras={"turbo-8step": 0.7})
        candidate = plan["candidates"][0]
        self.assertEqual(candidate["variant"], "pruned-int8")
        nodes = {n["class_type"]: n["inputs"] for n in candidate["graph"].values()}
        self.assertEqual(nodes["LoraLoaderModelOnly"]["strength_model"], 0.7)
        self.assertEqual(nodes["BasicScheduler"]["steps"], 8)

    def test_mcp_can_switch_adapters_and_cards_without_changing_existing_plan(self):
        def call(name, arguments=None):
            reply = server.dispatch({"method": "tools/call", "params": {"name": name, "arguments": arguments or {}}})
            self.assertNotIn("isError", reply, reply)
            return json.loads(reply["content"][0]["text"])

        header = b'{"adapter.lora_up.weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}'
        header += b" " * (-len(header) % 8)
        registrations = {}
        base_path = self.root / "custom-card.json"
        cards = controller.load_cards(self.root)
        base_path.write_text(json.dumps(cards["h3-t2v"] | {"id": "custom-h3"}), encoding="utf-8")
        call("register_card", {"path": str(base_path)})
        for name, card, weight in (("style-a", "h3-t2v", 1.0), ("style-b", "h3-t2v", 2.0),
                                   ("style-b", "custom-h3", 2.0)):
            path = self.root / (name + ".safetensors")
            path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", weight))
            registrations[card, name] = call("register_lora", {"card_id": card, "lora_id": name,
                                                 "variants": ["pruned-int8"], "local_path": str(path)})
        self.assertEqual(len(call("list_loras")), 3)
        self.assertEqual(len(call("list_cards")), len(cards) + 1)
        profile = self.profile | {"verified_email": "test@example.com"}
        accounts = SimpleNamespace(resolve=lambda name: profile)
        choices = (("h3-t2v", "style-a", 0.3), ("h3-t2v", "style-b", 0.7), ("custom-h3", "style-b", 0.8))
        saved = []
        with patch.object(controller, "backend", return_value=(accounts, None, None)):
            for card, name, strength in choices:
                summary = call("plan_video", {"card": card, "prompt": "Autumn leaves",
                                              "loras": {name: strength}, "takes": 1})
                plan_path = controller.directory(summary["job_id"]) / "plan.json"
                saved.append((plan_path, summary["plan_sha256"]))
                self.assertEqual((summary["card"], summary["loras"]), (card, {name: strength}))
                candidate = controller.read_json(plan_path)["candidates"][0]
                lora_node = next(n for n in candidate["graph"].values() if n["class_type"] == "LoraLoaderModelOnly")
                self.assertEqual(lora_node["inputs"]["strength_model"], strength)
                self.assertIn(registrations[card, name]["asset"]["sha256"], {a["sha256"] for a in candidate["assets"]})
        for path, checksum in saved:
            self.assertEqual(sha256(path), checksum)

    def test_reference_order_from_number_not_input_object(self):
        a, b = self.root / "first.jpg", self.root / "second.jpg"
        a.write_bytes(b"a")
        b.write_bytes(b"b")
        plan = self.plan(card="h3-r2v", references={"image_2": str(b), "image_1": str(a)})
        nodes = plan["candidates"][0]["graph"]
        inputs = next(n["inputs"] for n in nodes.values() if n["class_type"] == "MiniMaxH3ReferenceToVideo")
        first = nodes[inputs["ref_images.ref_image_0"][0]]["inputs"]["image"]
        second = nodes[inputs["ref_images.ref_image_1"][0]]["inputs"]["image"]
        self.assertEqual((first, second), ("reference-2.jpg", "reference-1.jpg"))

    def test_finish_after_verified_download_and_one_transfer_retry(self):
        provider = ColabFixture(corrupt_downloads=1)
        provider.render_setup_markers = [
            ("gpu_verified", {"gpu": "NVIDIA A100-SXM4-80GB"}),
            ("environment_setup_started", {"stage": "comfy_environment", "raw": "private detail"}),
            ("environment_setup_started", {"stage": "comfy_environment"}),
            ("asset_progress", {"asset": "private item", "transferred_bytes": 1024}),
            ("assets_ready", {"count": 5, "shared_cache": False}),
            ("environment_ready", {"cache_hit": True}),
        ]
        self.exercise(self.plan(), provider, "completed")
        self.assertEqual((self.job / "video.mp4").read_bytes(), provider.video)
        actions = [a[0] for a in provider.calls]
        self.assertEqual(actions.count("download"), 2)
        self.assertLess(actions.index("download"), actions.index("stop"))
        events = [json.loads(line) for line in (self.job / "events.jsonl").read_text().splitlines()]
        hooks = [row for row in events if row["event"] in controller.REMOTE_SETUP_EVENT_FIELDS
                 or row["event"] in ("preflight_started", "output_ready_hook")]
        self.assertEqual([row["event"] for row in hooks], ["preflight_started", "gpu_verified", "preflight_ready",
                         "environment_setup_started", "assets_ready", "environment_ready", "output_ready_hook"])
        self.assertEqual(hooks[2]["cached_asset_count"], 0)
        self.assertEqual(hooks[3]["stage"], "comfy_environment")
        self.assertFalse(hooks[4]["shared_cache"])
        self.assertEqual(hooks[4]["count"], 5)
        self.assertTrue(hooks[5]["cache_hit"])
        self.assertNotIn("nonce", json.dumps(events))
        self.assertNotIn("private", json.dumps(events))

    def test_remote_setup_failure_is_persisted_before_remote_exec_returns(self):
        provider = ColabFixture()
        provider.render_setup_markers = [
            ("environment_setup_started", {"stage": "comfy_environment"}),
            ("setup_failed", {"stage": "comfy_environment", "error_kind": "environment", "message": "private diagnostics"}),
            ("failed", {"error_kind": "environment", "message": "ComfyUI setup failed"}),
        ]
        original = provider.run_colab
        observed_failure = []

        def run_colab(account, argv, timeout_seconds, stdin=None, on_output=None):
            def watch(line):
                on_output(line)
                marker = json.loads(line.removeprefix(controller.MARK))
                if marker["phase"] == "setup_failed":
                    events = [json.loads(row) for row in (self.job / "events.jsonl").read_text().splitlines()]
                    observed_failure.extend(event for event in events if event["event"] == "setup_failed")

            return original(account, argv, timeout_seconds, stdin=stdin, on_output=watch if on_output else None)

        provider.run_colab = run_colab
        plan = self.plan()
        with (patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "notify")):
            with self.assertRaisesRegex(RuntimeError, "ComfyUI setup failed"):
                controller.run_worker(self.job, plan)
            controller.finish(self.job, plan, "failed")

        self.assertEqual([call[0] for call in provider.calls], ["new", "exec", "exec"])
        self.assertEqual(len(observed_failure), 1)
        self.assertEqual(observed_failure[0]["stage"], "comfy_environment")
        self.assertEqual(observed_failure[0]["error_kind"], "environment")
        self.assertNotIn("message", observed_failure[0])

    def test_output_event_triggers_download_before_exec_exits(self):
        provider = ColabFixture(defer_exec_exit=True)
        self.exercise(self.plan(), provider, "completed")
        self.assertTrue(provider.received_download_before_exec_exit)

    def test_queue_hook_is_bound_before_the_monitor_and_records_allocation(self):
        provider = ColabFixture()
        plan = self.plan(gpus=["A100"]) | {"queue_hook": "events"}
        path = self.job / "plan.json"
        controller.save(path, plan)
        hook = self.job / "queue-hook.json"

        def spawn(*args):
            self.assertEqual(controller.read_json(hook)["phase"], "queued")
            return SimpleNamespace(pid=123)

        with patch.object(controller, "background", side_effect=spawn):
            started = controller.start_video(self.job.name, sha256(path), authorize_cloud=True)
        self.assertEqual(started["queue_hook_path"], str(hook))
        self.exercise(plan, provider, "completed")
        events = [json.loads(line) for line in (self.job / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        lifecycle = [event["phase"] for event in events if event["event"] == "queue_hook"]
        self.assertEqual(lifecycle, ["queued", "allocating", "allocated"])
        entry = controller.status(self.job.name)["queue_hook"]
        self.assertEqual(entry["session"], self.job.name + "-1")
        self.assertTrue(entry["terminal"])

    def test_exec_prelude_disconnect_retries_only_on_the_same_idle_runtime(self):
        provider = ColabFixture()
        original = provider.run_colab
        disconnected = False

        def interrupt(account, argv, timeout_seconds, stdin=None, on_output=None):
            nonlocal disconnected
            if argv[0] == "exec" and stdin and stdin.startswith("CONFIG") and not disconnected:
                config = ast.literal_eval(stdin.splitlines()[0].partition("=")[2].strip())
                if not config.get("preflight_only"):
                    disconnected = True
                    provider.calls.append(argv[:])
                    return {"exit_code": 1, "stdout": "", "stderr": EXEC_PRELUDE_DISCONNECT}
            return original(account, argv, timeout_seconds, stdin, on_output)

        with patch.object(provider, "run_colab", side_effect=interrupt):
            self.exercise(self.plan(gpus=["A100"]), provider, "completed")
        self.assertEqual([c[0] for c in provider.calls], ["new", "exec", "exec", "status", "exec", "download", "stop"])
        self.assertEqual(len({c[2] for c in provider.calls}), 1)
        transport = controller.status(self.job.name)["generation_transport"]
        self.assertEqual(transport["state"], "recovered")
        self.assertEqual(transport["remote_phase"], "output_ready")

    def test_disconnect_during_download_resumes_on_same_runtime(self):
        self.exercise_download_disconnect(remote_safe=True)

    def test_disconnect_during_download_preserves_unconfirmed_runtime(self):
        self.exercise_download_disconnect(remote_safe=False)

    def exercise_download_disconnect(self, remote_safe):
        provider = ColabFixture()
        original = provider.run_colab
        interrupted = False

        def interrupt(account, argv, timeout_seconds, stdin=None, on_output=None):
            nonlocal interrupted
            if argv[0] == "exec" and stdin and "H3_PREPARATION_RESUME " in stdin:
                provider.calls.append(argv[:])
                return {"exit_code": 0, "stdout": f"H3_PREPARATION_RESUME {remote_safe}\n", "stderr": ""}
            if argv[0] == "exec" and stdin and stdin.startswith("CONFIG") and not interrupted:
                config = ast.literal_eval(stdin.splitlines()[0].partition("=")[2].strip())
                if not config.get("preflight_only"):
                    interrupted = True
                    provider.calls.append(argv[:])
                    for phase in ("asset_progress", "environment_ready", "asset_progress"):
                        on_output(controller.MARK + json.dumps({
                            "job_id": config["job_id"], "nonce": config["nonce"], "phase": phase}) + "\n")
                    return {"exit_code": 1, "stdout": "", "stderr": "RuntimeError: Connection was lost.\n"}
            return original(account, argv, timeout_seconds, stdin, on_output)

        with patch.object(provider, "run_colab", side_effect=interrupt):
            expected = "completed" if remote_safe else "failed"
            self.exercise(self.plan(gpus=["A100"]), provider, expected,
                          raises=False if remote_safe else "Inspect the retained session")
        calls = ["new", "exec", "exec", "status", "exec"]
        if remote_safe:
            calls += ["exec", "download", "stop"]
        self.assertEqual([call[0] for call in provider.calls], calls)
        self.assertEqual(len({call[2] for call in provider.calls}), 1)
        if not remote_safe:
            self.assertEqual(controller.status(self.job.name)["generation_transport"]["remote_phase"],
                             "asset_progress")

    def test_transport_loss_after_sampling_retains_remote_phase_without_duplicate_execution(self):
        provider = ColabFixture()
        original = provider.run_colab

        def interrupt(account, argv, timeout_seconds, stdin=None, on_output=None):
            if argv[0] == "exec" and stdin and stdin.startswith("CONFIG"):
                config = ast.literal_eval(stdin.splitlines()[0].partition("=")[2].strip())
                if not config.get("preflight_only"):
                    provider.calls.append(argv[:])
                    on_output(controller.MARK + json.dumps({
                        "job_id": config["job_id"], "nonce": config["nonce"], "phase": "sampling"}) + "\n")
                    return {"exit_code": 1, "stdout": "", "stderr": "RuntimeError: Connection was lost.\n"}
            return original(account, argv, timeout_seconds, stdin, on_output)

        with patch.object(provider, "run_colab", side_effect=interrupt):
            self.exercise(self.plan(gpus=["A100"]), provider, "failed", raises="Inspect the retained session")
        self.assertEqual([c[0] for c in provider.calls], ["new", "exec", "exec", "status", "exec"])
        state = controller.status(self.job.name)
        self.assertEqual(state["generation_transport"], {
            "state": "interrupted", "session": self.job.name + "-1", "stage": "script_dispatched",
            "remote_phase": "sampling"})
        self.assertEqual(state["remote_marker"]["phase"], "sampling")

    def test_batch_plans_distinct_outputs_on_shared_model_and_reference_graph(self):
        image = self.root / "reference.png"
        image.write_bytes(b"reference data")
        entries = [{"id": "opening", "prompt": "Opening shot", "seconds": 11},
                   {"id": "ending", "prompt": "Closing shot", "seconds": 9, "seed": 46}]
        profile = self.profile | {"verified_email": "test@example.com"}
        accounts = SimpleNamespace(resolve=lambda name: profile)
        with patch.object(controller, "backend", return_value=(accounts, None, None)):
            packet = server.dispatch({"method": "tools/call", "params": {"name": "plan_video", "arguments": {
                "card": "h3-r2v", "gpus": ["A100"], "references": {"image_1": str(image)},
                "segments": entries, "takes": 1}}})
        self.assertNotIn("isError", packet, packet)
        receipt = json.loads(packet["content"][0]["text"])
        path = controller.directory(receipt["job_id"]) / "plan.json"
        self.assertEqual(receipt["plan_sha256"], sha256(path))
        frozen = controller.read_json(path)
        self.assertEqual(frozen["frames"], 503)
        candidate = frozen["candidates"][0]
        self.assertEqual([item["name"] for item in candidate["uploads"]], ["reference-1.png"])
        graphs = candidate["segments"]
        self.assertEqual([item["id"] for item in receipt["segments"]], ["opening", "ending"])
        for segment, entry in zip(graphs, entries, strict=True):
            graph = segment["graph"]
            self.assertEqual(graph[segment["output_node"]]["inputs"]["filename_prefix"], "deliverable-" + entry["id"])
            generation = next(n["inputs"] for n in graph.values() if n["class_type"] == "MiniMaxH3ReferenceToVideo")
            self.assertEqual(generation["prompt"], entry["prompt"])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.plan(segments=[entries[0], {**entries[1], "id": "OPENING"}])
        with self.assertRaisesRegex(ValueError, "segment prompts"):
            self.plan(segments=entries, prompt="Ambiguous prompt")

    def test_batch_downloads_each_segment_and_releases_one_session_after_the_batch(self):
        provider = ColabFixture()
        provider.download_throws = True
        entries = [{"id": key, "prompt": key, "seconds": 5} for key in ("first", "second")]
        plan = self.plan(segments=entries)
        controller.update(self.job, "planned", segments=plan["segments"])
        self.exercise(plan, provider, "completed")
        actions = [args[0] for args in provider.calls]
        self.assertEqual(actions, ["new", "exec", "exec", "download", "download", "download", "stop"])
        self.assertEqual(len({args[2] for args in provider.calls}), 1)
        self.assertTrue(provider.first_segment_downloaded_before_next)
        self.assertNotIn("segments", provider.preflights[0])
        receipts = controller.status(self.job.name)["segment_outputs"]
        self.assertEqual([x["id"] for x in receipts], ["first", "second"])
        for item in receipts:
            path = Path(item["video_path"])
            self.assertEqual(path.read_bytes(), provider.batch_video[path.name])
        self.assertEqual(controller.read_json(self.job / "artifact.json")["segment_outputs"], receipts)

    def test_retained_batch_accepts_lora_plan_before_first_sample_on_same_lease(self):
        provider = ColabFixture()
        entries = [{"id": key, "prompt": key, "seconds": 5} for key in ("first", "second")]
        plan = self.plan(gpus=["A100"], segments=entries) | {"hold_for_generation": True}
        controller.save(self.job / "plan.json", plan)
        replacement = json.loads(json.dumps(plan))
        replacement["loras"] = {"example-style": 1.0}
        candidate = replacement["candidates"][0]
        adapter = {"path": "loras/example-style.safetensors", "url": "https://models.example.com/lora",
                   "sha256": "7" * 64, "size_bytes": 300}
        candidate["assets"].append(adapter)
        for entry in candidate["segments"]:
            entry["graph"]["1500"] = {"class_type": "LoraLoaderModelOnly",
                                       "inputs": {"lora_name": "example-style.safetensors", "strength_model": 1.0}}
        replacement_id = "h3-" + "c" * 24
        replacement_path = controller.directory(replacement_id) / "plan.json"
        controller.save(replacement_path, replacement | {"job_id": replacement_id})
        original_update = controller.update

        def submit_after_preparing(job, phase, **data):
            original_update(job, phase, **data)
            if phase == "awaiting_generation":
                self.assertEqual(provider.generation_configs, [])
                self.assertEqual(len(provider.live), 1)
                controller.resume_video(job.name, replacement_id, sha256(replacement_path), authorize_cloud=True)

        with patch.object(controller, "update", side_effect=submit_after_preparing):
            self.exercise(plan, provider, "completed")
        self.assertEqual([args[0] for args in provider.calls],
                         ["new", "exec", "exec", "exec", "download", "download", "stop"])
        self.assertEqual(len({args[2] for args in provider.calls}), 1)
        self.assertIn(adapter, provider.generation_configs[0]["assets"])
        self.assertTrue(all(entry["graph"]["1500"]["class_type"] == "LoraLoaderModelOnly"
                            for entry in provider.generation_configs[0]["segments"]))
        self.assertEqual(controller.status(self.job.name)["loras"], {"example-style": 1.0})

    def test_retained_batch_uploads_only_changed_references_on_resume(self):
        ref1 = self.root / "ref1.png"
        ref2_v1 = self.root / "ref2_v1.png"
        ref2_v2 = self.root / "ref2_v2.png"
        ref1.write_bytes(b"unchanged-ref-1")
        ref2_v1.write_bytes(b"ref-2-version-1")
        ref2_v2.write_bytes(b"ref-2-version-2")
        provider = ColabFixture()
        plan = self.plan(gpus=["A100"], segments=[
            {"id": "seg1", "prompt": "v1", "seconds": 5},
            {"id": "seg2", "prompt": "v2", "seconds": 5},
        ]) | {"hold_for_generation": True}
        plan["candidates"][0]["uploads"] = [
            {"name": "reference-1.png", "local_path": str(ref1), "sha256": sha256(ref1)},
            {"name": "reference-2.png", "local_path": str(ref2_v1), "sha256": sha256(ref2_v1)},
        ]
        controller.save(self.job / "plan.json", plan)
        replacement = json.loads(json.dumps(plan))
        replacement["candidates"][0]["uploads"] = [
            {"name": "reference-1.png", "local_path": str(ref1), "sha256": sha256(ref1)},
            {"name": "reference-2.png", "local_path": str(ref2_v2), "sha256": sha256(ref2_v2)},
        ]
        replacement_id = "h3-" + "d" * 24
        replacement_path = controller.directory(replacement_id) / "plan.json"
        controller.save(replacement_path, replacement | {"job_id": replacement_id})
        original_update = controller.update

        def submit_after_preparing(job, phase, **data):
            original_update(job, phase, **data)
            if phase == "awaiting_generation":
                controller.resume_video(job.name, replacement_id, sha256(replacement_path), authorize_cloud=True)

        with patch.object(controller, "update", side_effect=submit_after_preparing):
            self.exercise(plan, provider, "completed")
        upload_calls = [args for args in provider.calls if args[0] == "upload"]
        uploaded_names = [Path(args[4]).name for args in upload_calls]
        self.assertEqual(uploaded_names, [".upload-bundle.tar", "reference-2.png"])
        self.assertEqual(Path(upload_calls[-1][3]).name, "ref2_v2.png")

    def test_late_timeout_preserves_success_when_both_outputs_are_verified(self):
        provider = ColabFixture()
        provider.timeout_after_outputs = True
        plan = self.plan(segments=[{"id": key, "prompt": key, "seconds": 5}
                                   for key in ("opening", "ending")])
        controller.update(self.job, "planned", segments=plan["segments"])
        self.exercise(plan, provider, "completed")
        self.assertEqual([item["id"] for item in controller.status(self.job.name)["segment_outputs"]],
                         ["opening", "ending"])
        log = (self.job / "events.jsonl").read_text(encoding="utf-8")
        self.assertIn("verified_outputs_after_timeout", log)

    def test_failed_batch_keeps_first_verified_segment_but_is_not_completed(self):
        provider = ColabFixture(fail_segment="second")
        plan = self.plan(segments=[{"id": key, "prompt": key, "seconds": 5}
                                   for key in ("first", "second", "third")])
        controller.update(self.job, "planned", segments=plan["segments"])

        def run_background(_command, job, _log):
            try:
                controller.run_worker(job, plan)
            except RuntimeError as error:
                controller.save(job / "worker-result.json", {"error": str(error)})
            return SimpleNamespace(poll=lambda: 0)

        with patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "background", side_effect=run_background), \
                patch.object(controller, "notify"):
            controller.monitor(self.job, plan)
        self.assertEqual([args[0] for args in provider.calls], ["new", "exec", "exec", "download"])
        self.assertFalse((self.job / "artifact.json").exists())
        state = controller.status(self.job.name)
        self.assertEqual(state["phase"], "review_pending")
        self.assertIn("generation", state["error"])
        self.assertIn("Segment second failed in ComfyUI", state["error"])
        self.assertFalse(state["runtime_released"])
        self.assertEqual([x["id"] for x in state["segment_outputs"]], ["first"])
        self.assertTrue(provider.live)
        events = [json.loads(line) for line in (self.job / "events.jsonl").read_text().splitlines()]
        self.assertIn("diagnostic_required", [event["event"] for event in events])

    def test_batch_segments_have_distinct_idempotent_design_imports(self):
        payload = b"verified bytes"
        receipt = {"size_bytes": len(payload), "sha256": hash_bytes(payload).hexdigest()}
        controller.save(self.job / "segment-outputs.json", [{"id": key, **receipt} for key in ("first", "second")])
        for key in ("first", "second"):
            (self.job / f"video-{key}.mp4").write_bytes(payload)
        calls = []

        @contextmanager
        def connect():
            def call(method, **args):
                if method == "open_project":
                    return {"project": {"folderPath": str(self.root)}}
                calls.append(args)
                return {"requestId": args["requestId"]}
            yield call

        with patch.object(design_bridge, "connect", connect):
            first = design_bridge.deliver_video(self.job.name, "test-project", "test-cut", "First", segment_id="first")
            second = design_bridge.deliver_video(self.job.name, "test-project", "test-cut", "Second", segment_id="second")
            self.assertEqual(first, design_bridge.deliver_video(self.job.name, "test-project", "test-cut", "First",
                                                               segment_id="first"))
        self.assertNotEqual(first, second)
        self.assertEqual(len(calls), 2)
        self.assertEqual(Path(calls[0]["sources"][0]["path"]).read_bytes(), payload)

    def test_rejected_hardware_checks_absence_before_allocating_again(self):
        provider = ColabFixture(rejected_gpu=True)
        self.exercise(self.plan(gpus=["H100", "A100"]), provider, "completed")
        actions = [a[0] for a in provider.calls]
        self.assertEqual(actions.count("new"), 2)
        self.assertEqual(actions[:3], ["new", "status", "new"])

    def test_assigned_capacity_failure_keeps_runtime_pending_review(self):
        provider = ColabFixture(mismatched_first=True)
        plan = self.plan(gpus=["H100", "A100"])
        self.exercise(plan, provider, "failed", raises="capacity_mismatch")
        self.assertEqual([call[0] for call in provider.calls], ["new", "exec"])
        self.assertEqual(controller.status(self.job.name)["runtime_review"]["new_outcome"], "returned")

    def test_runtime_review_inspects_keeps_and_requires_approval_before_stopping(self):
        provider = ColabFixture(mismatched_first=True)
        plan = self.plan(gpus=["A100"])
        self.approve_fixture_plan(plan)
        self.exercise(plan, provider, "failed", raises="capacity_mismatch")
        with (patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)),
              patch.object(controller, "background", return_value=SimpleNamespace(pid=123)) as launch,
              patch.object(controller, "notify")):
            with self.assertRaisesRegex(ValueError, "review_runtime"):
                controller.retry_cleanup(self.job.name)
            with self.assertRaisesRegex(ValueError, "review_runtime"):
                controller.cancel_video(self.job.name)
            inspected = controller.review_runtime(self.job.name, "inspect")
            self.assertEqual(inspected["provider_state"], "active")
            self.assertTrue(inspected["session_idle"])
            kept = controller.review_runtime(self.job.name, "keep", reason="Investigate disk cache")
            self.assertEqual(kept["runtime_review"]["decision"], "keep")
            self.assertEqual(kept["runtime_review"]["reason"], "Investigate disk cache")
            with self.assertRaisesRegex(ValueError, "reason"):
                controller.review_runtime(self.job.name, "release")
            requested = controller.review_runtime(self.job.name, "release", reason="Review complete")
            self.assertEqual(requested["phase"], "review_release_requested")
            launch.assert_called_once_with("review-release", self.job, "monitor.log")
            self.assertEqual([call[0] for call in provider.calls], ["new", "exec", "status"])
            self.assertTrue(provider.live)
            with self.assertRaisesRegex(ValueError, "already been submitted"):
                controller.review_runtime(self.job.name, "release", reason="Duplicate")
            controller.release_reviewed_runtime(self.job, plan)
        final = controller.status(self.job.name)
        self.assertEqual(final["phase"], "failed")
        self.assertTrue(final["runtime_released"])
        self.assertEqual(final["runtime_review"]["reason"], "Review complete")
        self.assertEqual([call[0] for call in provider.calls], ["new", "exec", "status", "status", "stop"])
        self.assertFalse(provider.live)

    def test_probe_session_and_review_probe_exec_detect_hung_idle_session(self):
        provider = ColabFixture(mismatched_first=True)
        plan = self.plan(gpus=["A100"])
        self.approve_fixture_plan(plan)
        self.exercise(plan, provider, "failed", raises="capacity_mismatch")

        def hung_exec(account, argv, timeout_seconds=15, stdin="", **kwargs):
            if argv[0] == "status":
                return {"exit_code": 0, "stdout": f"[{argv[2]}] gpu-a100 | Status: IDLE\n", "stderr": ""}
            if argv[0] == "exec":
                return {"exit_code": 124, "stdout": "", "stderr": "w\x00s\x00l\x00:\x00 timeout\n"}
            raise AssertionError(f"Unexpected call: {argv}")

        fake_transport = SimpleNamespace(run_colab=hung_exec)
        with patch.object(controller, "identity", return_value=(self.accounts, fake_transport, self.pool)):
            inspected = controller.review_runtime(self.job.name, "inspect", probe_exec=True)
        self.assertEqual(inspected["provider_state"], "active")
        self.assertTrue(inspected["session_idle"])
        self.assertFalse(inspected["exec_reachable"])
        self.assertEqual(inspected["exec_exit_code"], 124)
        self.assertNotIn("\x00", inspected["stderr"])

        def healthy_exec(account, argv, timeout_seconds=15, stdin="", **kwargs):
            if argv[0] == "status":
                return {"exit_code": 0, "stdout": f"[{argv[2]}] gpu-a100 | Status: IDLE\n", "stderr": ""}
            if argv[0] == "exec":
                return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}
            raise AssertionError(f"Unexpected call: {argv}")

        with patch.object(controller, "backend", return_value=(self.accounts, SimpleNamespace(run_colab=healthy_exec), self.pool)):
            probed = controller.probe_session("acct", "sess-1")
        self.assertTrue(probed["exec_reachable"])
        self.assertEqual(probed["exec_exit_code"], 0)

    def test_background_uses_wmi_when_breakaway_is_denied_on_windows(self):
        with (patch.object(controller.os, "name", "nt"),
              patch.object(controller.subprocess, "Popen", side_effect=PermissionError(13, "denied")),
              patch.object(controller.subprocess, "run",
                           return_value=SimpleNamespace(returncode=0, stdout="0 43210\n", stderr="")) as wmi_run):
            proc = controller.background("monitor", self.job, "monitor.log")
        self.assertEqual(proc.pid, 43210)
        wmi_run.assert_called_once()
        self.assertEqual(wmi_run.call_args[0][0][:3], ["powershell", "-NoProfile", "-EncodedCommand"])

    def test_insufficient_disk_stops_without_upload(self):
        image = self.root / "test.png"
        image.write_bytes(b"test input bytes")
        provider = ColabFixture()
        provider.storage_errors[1] = 1
        plan = self.plan(card="h3-i2v", references={"first_frame": str(image)})
        self.exercise(plan, provider, "failed", raises=True)
        self.assertNotIn("upload", [args[0] for args in provider.calls])
        preflight = provider.preflights[0]
        self.assertNotIn("graph", preflight)
        self.assertNotIn("reference_uploads", preflight)
        self.assertTrue(all("url" not in asset for asset in preflight["assets"]))

    def test_preflight_failure_marker_overrides_zero_cli_exit(self):
        provider = ColabFixture()
        provider.storage_errors[1] = 0
        plan = self.plan(gpus=["A100"]) | {"preflight_only": True}
        self.exercise(plan, provider, "failed", raises=True)

        self.assertEqual([call[0] for call in provider.calls], ["new", "exec"])
        state = controller.status(self.job.name)
        self.assertEqual(state["remote_marker"]["error_kind"], "storage")
        self.assertFalse((self.job / "preflight.json").exists())

    def test_drive_preflight_failure_marker_overrides_zero_cli_exit(self):
        provider = ColabFixture()
        provider.storage_errors[2] = 0
        image = self.root / "test.png"
        image.write_bytes(b"test input bytes")
        plan = self.plan(card="h3-i2v", gpus=["A100"], references={"first_frame": str(image)})
        plan["drive_cache_relative"] = "Verified-Model-Cache"
        with patch.object(controller.managed_cli, "NativeCLI", ColabFixture):
            self.exercise(plan, provider, "failed", raises=True)

        self.assertEqual([call[0] for call in provider.calls], ["new", "exec", "drivemount", "exec", "exec"])
        self.assertEqual(provider.preflights[1]["cache_dir"], "/content/drive/MyDrive/Verified-Model-Cache")
        events = [json.loads(line) for line in (self.job / "events.jsonl").read_text().splitlines()]
        preflights = [(event["event"], event["stage"]) for event in events
                      if event["event"] in ("preflight_started", "gpu_verified", "preflight_ready")]
        self.assertEqual(preflights, [("preflight_started", "hardware_storage"), ("gpu_verified", "hardware_storage"),
                         ("preflight_ready", "hardware_storage"), ("preflight_started", "model_cache"),
                         ("gpu_verified", "model_cache")])

    def test_preflight_only_completes_and_releases_without_assets(self):
        plan = self.plan() | {"preflight_only": True}
        provider = ColabFixture()

        def run_background(_command, job, _log):
            controller.run_worker(job, plan)
            return SimpleNamespace(poll=lambda: 0)

        with patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "background", side_effect=run_background), \
                patch.object(controller, "notify"):
            controller.monitor(self.job, plan)
        self.assertEqual([call[0] for call in provider.calls], ["new", "exec", "stop"])
        self.assertFalse(provider.live)
        self.assertFalse((self.job / "video.mp4").exists())
        result = controller.status(self.job.name)
        self.assertEqual(result["phase"], "completed")
        self.assertTrue(result["runtime_released"])
        self.assertEqual(result["preflight"]["gpu"]["gpu"], "NVIDIA A100-SXM4-80GB")

    def test_cleanup_does_not_credit_a_vm_with_a_missing_cli_mapping(self):
        session = self.job.name + "-1"
        missing = {"stdout": f"[colab] Session '{session}' not found.\n", "exit_code": 0}
        provider = SimpleNamespace(run_colab=lambda *args, **kwargs: missing)
        lease = {"session": session, "released": False, "new_outcome": "returned"}
        with patch.object(controller.time, "sleep"):
            self.assertFalse(controller.cleanup(self.job, provider, {"account": "test"}, lease))
        self.assertFalse(lease["released"])

    def test_cleanup_confirms_pruned_returned_session_when_server_is_empty(self):
        session = self.job.name + "-1"
        responses = {
            "stop": f"[colab] Session '{session}' not found.\n",
            "status": f"[colab] Session '{session}' not found.\n",
            "sessions": "[colab] No active sessions found on server.\n",
        }
        provider = SimpleNamespace(run_colab=lambda _account, args, **kw: {
            "stdout": responses[args[0]], "exit_code": 0})
        lease = {"session": session, "released": False, "new_outcome": "returned"}
        with patch.object(controller.time, "sleep"):
            self.assertTrue(controller.cleanup(self.job, provider, {"account": "test"}, lease))
        self.assertTrue(lease["released"])
        self.assertEqual(lease["release_evidence"], "no_active_server_sessions")

    def test_two_workers_keep_named_colab_sessions_and_artifacts_independent(self):
        other = self.root / "jobs" / ("h3-" + "b" * 24)
        other.mkdir()
        jobs = [self.job, other]
        plans = [self.plan() | {"job_id": job.name} for job in jobs]
        provider = ColabFixture()
        original = provider.run_colab
        both_allocated = threading.Barrier(2)
        together = []

        def run_colab(account, argv, timeout_seconds, **options):
            reply = original(account, argv, timeout_seconds, **options)
            if argv[0] == "new":
                if both_allocated.wait(3) == 0:
                    together.extend(provider.live)
                both_allocated.wait(3)
            return reply

        def run(job, plan):
            controller.run_worker(job, plan)
            controller.finish(job, plan, "completed")

        provider.run_colab = run_colab
        with patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "notify"), ThreadPoolExecutor(max_workers=2) as workers:
            first, second = [workers.submit(run, job, plan) for job, plan in zip(jobs, plans)]
            first.result(timeout=10)
            second.result(timeout=10)
        self.assertCountEqual(together, [job.name + "-1" for job in jobs])
        self.assertCountEqual([argv[2] for argv in provider.calls if argv[0] == "stop"], together)
        for job in jobs:
            self.assertEqual((job / "video.mp4").read_bytes(), provider.video)
            result = controller.status(job.name)
            self.assertEqual(result["phase"], "completed")
            self.assertTrue(result["runtime_released"])

    def test_cleanup_confirmed_by_named_peer_sessions_but_not_orphan_assignments(self):
        session = self.job.name + "-1"
        peer = "h3-" + "b" * 24 + "-1"
        calls = []
        listed = {"name": "?"}

        def reply(_account, args, **unused):
            calls.append(args)
            stdout = (f"[{listed['name']}] provider | Hardware: A100 | Shape: Standard | Variant: GPU\n"
                      if args[0] == "sessions" else f"[colab] Session '{session}' not found.\n")
            return {"stdout": stdout, "exit_code": 0}

        provider = SimpleNamespace(run_colab=reply)
        lease = {"session": session, "released": False, "new_outcome": "returned"}
        with patch.object(controller.time, "sleep"):
            self.assertFalse(controller.cleanup(self.job, provider, {"account": "test"}, lease))
            listed["name"] = session
            self.assertFalse(controller.cleanup(self.job, provider, {"account": "test"}, lease))
            listed["name"] = peer
            self.assertTrue(controller.cleanup(self.job, provider, {"account": "test"}, lease))
        self.assertTrue(lease["released"])
        self.assertEqual(lease["release_evidence"], "absent_from_named_server_sessions")
        self.assertTrue(all(args == ["sessions"] or args[2] == session for args in calls))

    def test_ambiguous_allocation_needs_the_authoritative_empty_session_list(self):
        session = self.job.name + "-1"
        responses = {
            "stop": "session mapping missing",
            "status": f"[colab] Session '{session}' not found.\n",
            "sessions": "[colab] No active sessions found on server.\n",
        }
        sessions_stderr = {"value": ""}

        def reply(_account, args, **unused):
            return {"stdout": responses[args[0]], "exit_code": 0,
                    "stderr": sessions_stderr["value"] if args[0] == "sessions" else ""}

        provider = SimpleNamespace(run_colab=reply)
        lease = {"session": session, "released": False, "new_outcome": "unknown",
                 "allocation_deadline": time.time() + 60}
        with patch.object(controller.time, "sleep"):
            self.assertFalse(controller.cleanup(self.job, provider, {"account": "test"}, lease))
        lease["allocation_deadline"] = time.time() - 1
        sessions_stderr["value"] = "No valid default credentials found. To authenticate, run:\n"
        with patch.object(controller.time, "sleep"):
            self.assertFalse(controller.cleanup(self.job, provider, {"account": "test"}, lease))
        sessions_stderr["value"] = "warning: `--no-project` was provided, but no project was found\n"
        self.assertTrue(controller.cleanup(self.job, provider, {"account": "test"}, lease))
        self.assertEqual(lease["release_evidence"], "no_active_server_sessions")

    def test_design_wakeup_running_and_closed(self):
        with patch.object(design_bridge, "port_ready", return_value=True), \
                patch.object(design_bridge.subprocess, "Popen") as launch:
            self.assertEqual(design_bridge.wake_design(), {"launched": False})
            launch.assert_not_called()
        executable = self.root / "Design"
        executable.touch()
        with patch.object(design_bridge, "port_ready", side_effect=[False, True]), \
                patch.object(design_bridge, "settings", return_value={"design_executable": str(executable)}), \
                patch.object(design_bridge.subprocess, "Popen") as launch:
            self.assertEqual(design_bridge.wake_design(), {"launched": True})
            self.assertEqual(launch.call_args.args[0], [str(executable)])

    def test_disconnected_generation_waits_for_busy_session_and_recovers_remote_output(self):
        provider = ColabFixture()
        original = provider.run_colab
        status_polls = {"count": 0}
        digest = hash_bytes(provider.video).hexdigest()

        def flaky(account, argv, timeout_seconds, stdin=None, on_output=None):
            if argv[0] == "exec" and stdin and "CONFIG =" in stdin:
                conf = ast.literal_eval(stdin.splitlines()[0].partition("=")[2].strip())
                if not conf.get("preflight_only"):
                    provider.calls.append(argv[:])
                    on_output(controller.MARK + json.dumps({
                        "job_id": conf["job_id"], "nonce": conf["nonce"], "phase": "asset_progress",
                    }) + "\n")
                    return {"exit_code": 1, "stdout": "", "stderr": "RuntimeError: Connection was lost.\n"}
            if argv[0] == "status":
                provider.calls.append(argv[:])
                status_polls["count"] += 1
                state = "BUSY (1 executing)" if status_polls["count"] == 1 else "IDLE"
                return {"exit_code": 0, "stdout": f"[{argv[2]}] https://colab.example | Status: {state}\n", "stderr": ""}
            if argv[0] == "exec" and stdin and ("H3_REMOTE_OUTPUTS" in stdin or "H3_PROBE" in stdin):
                provider.calls.append(argv[:])
                payload = [{"name": "video.mp4", "size_bytes": len(provider.video), "sha256": digest}]
                return {"exit_code": 0, "stdout": "H3_PROBE " + json.dumps({"outputs": payload, "can_resume": False}) + "\n", "stderr": ""}
            return original(account, argv, timeout_seconds, stdin=stdin, on_output=on_output)

        provider.run_colab = flaky
        with patch.object(controller.time, "sleep"):
            self.exercise(self.plan(), provider, "completed")
        self.assertEqual((self.job / "video.mp4").read_bytes(), provider.video)
        events = [json.loads(line)["event"] for line in (self.job / "events.jsonl").read_text().splitlines()]
        self.assertIn("generation_transport_waiting", events)
        self.assertIn("generation_transport_recovered", events)

    def test_generation_job_retries_once_after_backend_503_when_account_is_empty(self):
        provider = ColabFixture()
        original = provider.run_colab
        failed_once = {"done": False}

        def flaky(account, argv, timeout_seconds, stdin=None, on_output=None):
            if argv[0] == "new" and not failed_once["done"]:
                failed_once["done"] = True
                provider.calls.append(argv[:])
                return {"exit_code": 1, "stdout": "", "stderr": (
                    "ColabRequestError: Failed to issue request POST \n"
                    "[REDACTED_URL]\n"
                    "notebook-fragment-secret&variant=GPU&accelerator=A100&shape=hm: Service Unavailable\n")}
            if argv[0] == "sessions":
                provider.calls.append(argv[:])
                return {"exit_code": 0, "stdout": "[colab] No active sessions found on server.\n", "stderr": ""}
            return original(account, argv, timeout_seconds, stdin=stdin, on_output=on_output)

        provider.run_colab = flaky
        plan = self.plan(gpus=["A100"]) | {"timeout_seconds": 1200}
        with patch.object(controller.time, "sleep"):
            self.exercise(plan, provider, "completed")
        actions = [call[0] for call in provider.calls]
        self.assertEqual(actions, ["new", "sessions", "new", "exec", "exec", "download", "stop"])
        events = [json.loads(line)["event"] for line in (self.job / "events.jsonl").read_text().splitlines()]
        self.assertIn("allocation_failed", events)
        self.assertIn("allocation_retry", events)

    def test_generation_job_retries_after_backend_503_with_named_peer_still_running(self):
        provider = ColabFixture()
        original = provider.run_colab
        peer = "h3-" + "b" * 24 + "-1"
        provider.live.add(peer)

        def flaky(account, argv, timeout_seconds, **options):
            if argv[:3] == ["new", "-s", self.job.name + "-1"]:
                return {"exit_code": 1, "stdout": "", "stderr": (
                    "ColabRequestError: Failed to issue request POST [REDACTED_URL]: Service Unavailable\n")}
            if argv == ["sessions"]:
                return {"exit_code": 0, "stdout":
                        f"[{peer}] provider | Hardware: A100 | Shape: Standard | Variant: GPU\n", "stderr": ""}
            return original(account, argv, timeout_seconds, **options)

        provider.run_colab = flaky
        plan = self.plan(gpus=["A100"]) | {"timeout_seconds": 1200}
        with patch.object(controller, "identity", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "backend", return_value=(self.accounts, provider, self.pool)), \
                patch.object(controller, "notify"), patch.object(controller.time, "sleep"):
            controller.run_worker(self.job, plan)
            controller.finish(self.job, plan, "completed")
        self.assertEqual(provider.live, {peer})
        self.assertEqual(controller.status(self.job.name)["phase"], "completed")
        events = [json.loads(line) for line in (self.job / "events.jsonl").read_text().splitlines()]
        retry = next(event for event in events if event["event"] == "allocation_retry")
        self.assertEqual(retry["release_evidence"], "absent_from_named_server_sessions")


if __name__ == "__main__":
    unittest.main()
