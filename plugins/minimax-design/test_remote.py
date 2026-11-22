"""Local checks for the Colab worker, including a loopback ComfyUI API fixture."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runtime_remote as worker


def weights(path, value=0):
    header = json.dumps({"fixture": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    body = struct.pack("<Q", len(header)) + header + struct.pack("<f", value)
    path.write_bytes(body)
    return {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}


def mp4_boxes(path):
    boxes = [(b"ftyp", b"test"), (b"free", b""), (b"mdat", b"test"), (b"moov", b"test")]
    path.write_bytes(b"".join(struct.pack(">I4s", 8 + len(payload), kind) + payload for kind, payload in boxes))


class PathAndGraphTests(unittest.TestCase):
    def test_comfy_bootstrap_uses_base_pip_without_running_ensurepip(self):
        directory = Path("managed-venv")
        deadline = time.monotonic() + 2
        with (patch.object(worker.importlib.metadata, "version", return_value="24.0"),
              patch.object(worker, "command") as run):
            executable, installer = worker.create_comfy_venv(directory, deadline, None)
        self.assertEqual(executable, directory / "bin" / "python")
        self.assertEqual(installer, [sys.executable, "-m", "pip", "--python", executable])
        self.assertEqual(run.call_count, 1)
        self.assertIn("--without-pip", run.call_args.args[0])

    def test_comfy_bootstrap_uses_venv_pip_with_older_base_pip(self):
        with (patch.object(worker.importlib.metadata, "version", return_value="22.2.2"),
              patch.object(worker, "command") as run):
            executable, installer = worker.create_comfy_venv(Path("managed-venv"), time.monotonic() + 2, None)
        self.assertEqual(installer, [executable, "-m", "pip"])
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("--without-pip", run.call_args.args[0])

    def test_relative_paths_confined(self):
        for value in (".", "../output", "a/../output", "a//output", "a\\output", "/etc/passwd", "C:foo"):
            with self.subTest(value=value), self.assertRaises(worker.RuntimeFailure):
                worker.safe_relative(value)

    def test_graph_only_executes_selected_save_video_dependencies(self):
        graph = {
            "1": {"class_type": "CreateVideo", "inputs": {"fps": 24}},
            "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}},
            "3": {"class_type": "UnrelatedNode", "inputs": {}},
        }
        self.assertEqual(set(worker.graph_for_output(graph, "2")), {"1", "2"})
        with self.assertRaises(worker.RuntimeFailure):
            worker.graph_for_output(graph, "3")

    def test_model_source_pinned_or_hashed(self):
        mutable = "https://huggingface.co/org/repo/resolve/main/example.safetensors"
        with self.assertRaises(worker.RuntimeFailure):
            worker.asset_url({"url": mutable})
        self.assertEqual(worker.asset_url({"url": mutable, "sha256": "1" * 64}), mutable)
        pinned = mutable.replace("/main/", "/" + "a" * 40 + "/")
        self.assertEqual(worker.asset_url({"url": pinned}), pinned)

    def test_successful_job_resolves_history_path_and_validates_container(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "preview.mp4"
            mp4_boxes(video)
            history = {"status": {"completed": True}, "outputs": {"2": {"images": [
                {"type": "output", "filename": "preview.mp4", "subfolder": ""},
            ]}}}
            output = worker.find_output(history, "2", root)
            self.assertEqual(output, video)
            with patch.object(worker.shutil, "which", return_value=None):
                worker.validate_mp4(output, time.monotonic() + 2)
                output.write_bytes(output.read_bytes()[:-1])
                with self.assertRaises(worker.RuntimeFailure):
                    worker.validate_mp4(output, time.monotonic() + 2)


FAKE_COMFY = r"""
import argparse, json, os, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import struct

parser = argparse.ArgumentParser()
parser.add_argument('--base-directory')
parser.add_argument('--port', type=int)
options, rest = parser.parse_known_args()
root = Path(options.base_directory)
history = {}

class Handler(BaseHTTPRequestHandler):
    def reply(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == '/object_info':
            self.reply({'CreateVideo': {}, 'SaveVideo': {}})
        elif self.path.startswith('/history/'):
            job = self.path.removeprefix('/history/')
            self.reply({job: history[job]} if job in history else {})
        else:
            self.reply({}, 404)

    def do_POST(self):
        self.enforce = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path != '/prompt' or set(self.enforce['prompt']) != {'1', '2'}:
            self.reply({}, 400)
            return
        job = str(uuid.uuid4())
        prefix = self.enforce['prompt']['2']['inputs'].get('filename_prefix', 'fixture')
        with (root / 'submissions.jsonl').open('a') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'prefix': prefix}) + '\n')
        if prefix == 'deliverable-fail':
            history[job] = {'status': {'status_str': 'error'}}
        else:
            video = root / 'output' / (prefix + '.mp4')
            payload = prefix.encode()
            video.write_bytes(b''.join(struct.pack('>I4s', 8 + len(payload), kind) + payload
                                       for kind in (b'ftyp', b'mdat', b'moov')))
            history[job] = {'status': {'completed': True}, 'outputs': {'2': {'images': [
                {'type': 'output', 'filename': video.name, 'subfolder': ''},
            ]}}}
        self.reply({'prompt_id': job})

ThreadingHTTPServer(('127.0.0.1', options.port), Handler).serve_forever()
"""


@unittest.skipUnless(os.name == "posix", "Colab worker uses Linux process groups and file locking")
class LinuxIntegrationTests(unittest.TestCase):
    def test_colab_child_failure_keeps_bootstrap_stage_and_stderr(self):
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / "bootstrap.log").open("a", encoding="utf-8") as log:
                with self.assertRaisesRegex(worker.RuntimeFailure, "创建 ComfyUI 虚拟环境.*退出码 1.*ensurepip missing"):
                    worker.command([sys.executable, "-c", "import sys; sys.exit('ensurepip missing')"],
                                   time.monotonic() + 5, log=log, stage="创建 ComfyUI 虚拟环境")

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_preflight_checks_hardware_and_disk_space(self):
        config = {"job_id": "h3-deadbeef", "nonce": "fixture", "preflight_only": True,
                  "timeout_seconds": 60, "setup_timeout_seconds": 60, "cache_dir": str(self.root),
                  "assets": [{"path": "diffusion_models/example.safetensors", "size_bytes": 10, "sha256": "a" * 64}]}
        with (patch.object(worker, "verify_gpu", return_value="GPU-test") as verify_gpu,
              patch.object(worker, "verify_torch") as verify_torch,
              patch.object(worker.shutil, "disk_usage", return_value=SimpleNamespace(free=10)) as disk_usage,
              patch.object(worker, "emit") as emit):
            worker.run(config)
            verify_gpu.assert_called_once()
            self.assertEqual(verify_torch.call_args.args[0], "GPU-test")
            emit.assert_called_once_with(config, "preflight_ready", cached_asset_sha256=[])
            disk_usage.return_value = SimpleNamespace(free=9)
            with self.assertRaises(worker.RuntimeFailure) as failed:
                worker.run(config)
            self.assertEqual(failed.exception.kind, "capacity_mismatch")
            self.assertEqual(emit.call_count, 1)

    def test_environment_hook_fires_before_slower_weights_finish(self):
        config = {"job_id": "h3-deadbeef", "nonce": "fixture", "timeout_seconds": 60,
                  "setup_timeout_seconds": 60, "comfy_revision": "a" * 40,
                  "assets": [{"path": "diffusion_models/example.safetensors"}]}
        notified = threading.Event()
        phases = []

        def emit(_, phase, **fields):
            phases.append(phase)
            if phase == "environment_ready":
                notified.set()

        def prepare_assets(*args):
            self.assertTrue(notified.wait(2), "environment hook must be observable before assets finish")

        with (patch.object(worker, "COLAB_CONTENT_ROOT", self.root),
              patch.object(worker, "verify_gpu", return_value="GPU-test"),
              patch.object(worker, "verify_torch"),
              patch.object(worker, "prepare_comfy", return_value=(self.root, Path(sys.executable), False)),
              patch.object(worker, "prepare_asset", side_effect=prepare_assets),
              patch.object(worker, "render") as render,
              patch.object(worker, "emit", side_effect=emit)):
            worker.run(config)
        self.assertEqual(phases, ["environment_setup_started", "environment_ready", "assets_ready"])
        render.assert_called_once()

    def test_prepare_only_reuses_verified_cache_and_can_swap_a_staged_lora_before_rendering(self):
        job_id = "h3-deadbeef"
        revision = "a" * 40
        run_root = self.root / job_id
        cache = run_root / "cache"
        inputs = run_root / "input"
        inputs.mkdir(parents=True)
        checkout = cache / "comfy" / revision
        checkout.mkdir(parents=True)
        (checkout / "main.py").write_text(FAKE_COMFY)
        (checkout / "requirements.txt").write_text("")
        reference = inputs / "reference.jpg"
        reference.write_bytes(b"reference fixture")
        base = {"path": "diffusion_models/base.safetensors", "upload_name": "base.safetensors",
                **weights(inputs / "base.safetensors")}
        lora = {"path": "loras/style.safetensors", "upload_name": "style.safetensors",
                **weights(inputs / "style.safetensors", value=1)}
        config = {"job_id": job_id, "nonce": "fixture", "comfy_revision": revision,
                  "timeout_seconds": 60, "setup_timeout_seconds": 20, "prepare_only": True,
                  "assets": [base, lora], "reference_uploads": [
                      {"name": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
                  ]}

        def create_venv(directory, deadline, log):
            executable = directory / "bin" / "python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            return executable, [executable, "-m", "pip"]

        def command(args, *unused, **options):
            return revision if args[:3] == ["git", "rev-parse", "HEAD"] else ""

        with (patch.object(worker, "COLAB_CONTENT_ROOT", self.root),
              patch.object(worker, "verify_gpu", return_value="GPU-test") as verify_gpu,
              patch.object(worker, "verify_torch"), patch.object(worker, "command", side_effect=command),
              patch.object(worker, "create_comfy_venv", side_effect=create_venv) as bootstrap,
              patch.object(worker, "dependency_version", return_value="fixture"),
              patch.object(worker.shutil, "which", return_value=None),
              patch.object(worker, "render", wraps=worker.render) as render,
              contextlib.redirect_stdout(io.StringIO()) as records):
            worker.run(config)
            render.assert_not_called()
            self.assertFalse((run_root / "output").exists())
            self.assertFalse((inputs / "base.safetensors").exists())
            base_target = (run_root / "models" / base["path"]).resolve()
            lora_link = run_root / "models" / lora["path"]
            self.assertEqual(hashlib.sha256(lora_link.read_bytes()).hexdigest(), lora["sha256"])
            updated_lora = {**lora, **weights(inputs / "style.safetensors", value=2)}
            worker.run({**config, **self.workflow(), "prepare_only": False, "assets": [base, updated_lora]})
            render.assert_called_once()
            bootstrap.assert_called_once()
            self.assertEqual(verify_gpu.call_count, 2)

        self.assertEqual((run_root / "models" / base["path"]).resolve(), base_target)
        self.assertEqual(hashlib.sha256(lora_link.read_bytes()).hexdigest(), updated_lora["sha256"])
        self.assertTrue((run_root / "video.mp4").is_file())
        markers = [json.loads(line.removeprefix("H3COLAB ")) for line in records.getvalue().splitlines()]
        prepared = next(index for index, marker in enumerate(markers) if marker["phase"] == "prepared")
        self.assertEqual(markers[prepared]["asset_count"], 2)
        self.assertFalse(markers[prepared]["environment_cache_hit"])
        self.assertNotIn("sampling", [marker["phase"] for marker in markers[:prepared]])
        self.assertTrue(any(marker["phase"] == "environment_ready" and marker["cache_hit"]
                            for marker in markers[prepared + 1:]))

    def test_uploaded_weight_verified_reused_and_bad_replacement_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, first, second = root / "cache", root / "first", root / "second"
            for run in (first, second):
                (run / "input").mkdir(parents=True)
            upload = first / "input" / "custom.safetensors"
            identity = weights(upload)
            asset = {"path": "loras/custom.safetensors", "upload_name": upload.name, **identity}
            end = time.monotonic() + 4
            worker.prepare_asset(asset, {"offline": True}, cache, first, end)
            worker.prepare_asset(asset, {"offline": True}, cache, second, end)
            self.assertFalse(upload.exists())
            self.assertEqual((first / "models" / asset["path"]).resolve(), (second / "models" / asset["path"]).resolve())
            bad = {**asset, "path": "loras/invalid.safetensors", "sha256": "0" * 64}
            weights(second / "input" / upload.name)
            with self.assertRaisesRegex(worker.RuntimeFailure, "SHA256"):
                worker.prepare_asset(bad, {"offline": True}, cache, second, end)

    def test_download_verifies_new_and_resumed_weights_without_rereading_final_file(self):
        fixture = self.root / "fixture.safetensors"
        asset = {"path": "diffusion_models/model.safetensors", "url": "https://models.example.com/model",
                 **weights(fixture)}
        body = fixture.read_bytes()
        split = len(body) // 2
        for case, status, prefix, response_body, should_publish in (
            ("new", 200, b"", body, True),
            ("resume", 206, body[:split], body[split:], True),
            ("restart", 200, b"x" * split, body, True),
            ("invalid", 200, b"", body[:-1] + b"x", False),
        ):
            with self.subTest(case=case):
                cache = self.root / case / "cache"
                run = self.root / case / "run"
                target = worker.asset_cache_path(cache, asset["sha256"])
                partial = target.with_suffix(".part")
                if prefix:
                    partial.parent.mkdir(parents=True)
                    partial.write_bytes(prefix)
                response = io.BytesIO(response_body)
                response.status = status
                response.headers = {"Content-Length": str(len(response_body))}
                if status == 206:
                    response.headers["Content-Range"] = f"bytes {split}-{len(body) - 1}/{len(body)}"
                opener = SimpleNamespace(open=lambda *args, **kwargs: response)
                with (patch.object(worker, "verify_public_resolution"),
                      patch.object(worker, "build_opener", return_value=opener),
                      patch.object(worker, "digest", side_effect=AssertionError("download reread"))):
                    if should_publish:
                        worker.prepare_asset(asset, {}, cache, run, time.monotonic() + 5)
                    else:
                        with self.assertRaisesRegex(worker.RuntimeFailure, "SHA256"):
                            worker.prepare_asset(asset, {}, cache, run, time.monotonic() + 5)
                self.assertFalse(partial.exists())
                self.assertEqual(target.exists(), should_publish)
                if should_publish:
                    self.assertEqual((run / "models" / asset["path"]).read_bytes(), body)

    def render_fixture(self, config):
        (self.root / "main.py").write_text(FAKE_COMFY)
        self.port = worker.local_port()
        self.records = io.StringIO()
        end = time.monotonic() + 8
        with (contextlib.redirect_stdout(self.records), patch.object(worker, "local_port", return_value=self.port),
              patch.object(worker.shutil, "which", return_value=None)):
            worker.render(config, self.root, sys.executable, self.root, "GPU-test", end, end)

    def workflow(self, prefix="fixture"):
        return {"output_node": "2", "graph": {
            "1": {"class_type": "CreateVideo", "inputs": {"fps": 24}},
            "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0], "filename_prefix": prefix}},
            "3": {"class_type": "UnrelatedNode", "inputs": {}},
        }}

    def markers(self):
        return [json.loads(line.removeprefix("H3COLAB ")) for line in self.records.getvalue().splitlines()]

    def submissions(self):
        return [json.loads(line) for line in (self.root / "submissions.jsonl").read_text().splitlines()]

    def assert_server_stopped(self):
        with self.assertRaises(OSError):
            worker.request_json(self.port, "GET", "/object_info", time.monotonic() + 1)

    def test_render_uses_loopback_api_and_cleans_up_server(self):
        self.render_fixture({"job_id": "h3-deadbeef", "nonce": "fixture", **self.workflow()})
        self.assertTrue((self.root / "video.mp4").is_file())
        self.assertEqual([marker["phase"] for marker in self.markers()], ["sampling", "output_ready"])
        self.assertNotIn("segment_id", self.markers()[1])
        self.assert_server_stopped()

    def test_render_batch_reuses_process_and_publishes_each_verified_video(self):
        ids = ["opening", "ending"]
        self.render_fixture({"job_id": "h3-deadbeef", "nonce": "fixture", "segments": [
            {"id": segment_id, **self.workflow("deliverable-" + segment_id)} for segment_id in ids
        ]})
        submissions = self.submissions()
        self.assertEqual([entry["prefix"] for entry in submissions], ["deliverable-" + item for item in ids])
        self.assertEqual(len({entry["pid"] for entry in submissions}), 1)
        self.assertEqual([(marker["segment_id"], marker["phase"]) for marker in self.markers()],
                         [(item, phase) for item in ids for phase in ("sampling", "output_ready")])
        for marker in self.markers()[1::2]:
            content = (self.root / f"video-{marker['segment_id']}.mp4").read_bytes()
            self.assertEqual(marker["size_bytes"], len(content))
            self.assertEqual(marker["sha256"], hashlib.sha256(content).hexdigest())
        self.assert_server_stopped()

    def test_render_batch_later_failure_preserves_first_video_without_running_remaining_segment(self):
        with self.assertRaisesRegex(worker.RuntimeFailure, "工作流执行失败"):
            self.render_fixture({"job_id": "h3-deadbeef", "nonce": "fixture", "segments": [
                {"id": item, **self.workflow("deliverable-" + item)} for item in ("opening", "fail", "ending")
            ]})
        self.assertEqual([entry["prefix"] for entry in self.submissions()], ["deliverable-opening", "deliverable-fail"])
        self.assertEqual([marker["segment_id"] for marker in self.markers() if marker["phase"] == "output_ready"], ["opening"])
        self.assertEqual([path.name for path in self.root.glob("video*.mp4")], ["video-opening.mp4"])
        self.assert_server_stopped()


if __name__ == "__main__":
    unittest.main()
