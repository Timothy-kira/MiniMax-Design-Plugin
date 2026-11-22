"""Execute one pinned ComfyUI video graph in a dedicated Colab runtime.

The local controller prepends CONFIG and owns provider allocation and release.
This file runs in the Colab kernel without importing any controller modules.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import base64
import hashlib
from http.client import HTTPException
import importlib.metadata
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
from threading import Event, Thread
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


COMFY_REPOSITORY = "https://github.com/Comfy-Org/ComfyUI.git"
COLAB_CONTENT_ROOT = Path("/content")
BUFFER_SIZE = 4 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
JOB_NAME = re.compile(r"h3-[0-9a-f]{8,64}\Z")
SEGMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
GPU_MODELS = {
    "T4": r"\bT4\b",
    "L4": r"\bL4\b",
    "A100": r"\bA100\b",
    "H100": r"\bH100\b",
    "G4": r"\bRTX PRO 6000\b.*\bBlackwell\b",
}


class RuntimeFailure(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def error_kind(failure):
    if isinstance(failure, RuntimeFailure):
        return failure.kind
    if isinstance(failure, subprocess.TimeoutExpired):
        return "timeout"
    return type(failure).__name__


class SetupBudget:
    def __init__(self, end, cancelled=None):
        self.end = end
        self.cancelled = cancelled if cancelled is not None else Event()


def emit(config, phase, **fields):
    marker = {"job_id": config["job_id"], "nonce": config["nonce"], "phase": phase, **fields}
    print("H3COLAB " + json.dumps(marker, ensure_ascii=True, separators=(",", ":")), flush=True)


def remaining(deadline, cap=None):
    if isinstance(deadline, SetupBudget):
        if deadline.cancelled.is_set():
            raise RuntimeFailure("setup_cancelled", "另一项并行准备失败，停止剩余准备。")
        deadline = deadline.end
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise RuntimeFailure("timeout", "运行阶段超时。")
    return min(seconds, cap) if cap else seconds


def within(deadline, seconds):
    limit = time.monotonic() + seconds
    if isinstance(deadline, SetupBudget):
        return SetupBudget(min(deadline.end, limit), deadline.cancelled)
    return min(deadline, limit)


def safe_relative(value):
    if not isinstance(value, str) or not value or any(char in value for char in "\\:\x00\n\r"):
        raise RuntimeFailure("configuration", "资源路径无效。")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or str(path) != value or any(part in (".", "..") for part in path.parts):
        raise RuntimeFailure("configuration", "资源只能使用目录内的规范相对路径。")
    return path


def safe_name(value):
    path = safe_relative(value)
    if len(path.parts) != 1:
        raise RuntimeFailure("configuration", "上传文件只能使用文件名。")
    return path.name


def rooted(root, relative):
    path = root / safe_relative(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeFailure("configuration", "资源路径超出任务目录。")
    return path


def file_hasher(path, deadline):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(BUFFER_SIZE):
            remaining(deadline)
            value.update(data)
    return value


def digest(path, deadline):
    return file_hasher(path, deadline).hexdigest()


def write_json(path, value):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=True)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def kill_group(process):
    # Reap the group even when its leader exited before its children did.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def command(args, deadline, *, cwd=None, log=None, env=None, stage=None):
    with subprocess.Popen(
        [str(arg) for arg in args], cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=log if log is not None else subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", start_new_session=True, env=env,
    ) as process:
        try:
            while True:
                try:
                    interval = remaining(deadline, 2 if isinstance(deadline, SetupBudget) else None)
                    output, _ = process.communicate(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    remaining(deadline)
        except BaseException:
            kill_group(process)
            raise
        if process.returncode:
            detail = ""
            if log is not None and getattr(log, "name", None) and Path(log.name).is_file():
                try:
                    log.flush()
                    detail = " " + log_excerpt(Path(log.name))
                except OSError:
                    detail = ""
            elif output:
                detail = " " + " ".join(output.split())[-400:]
            program = Path(args[0]).name
            label = f"{stage}（{program}）" if stage else program
            raise RuntimeFailure("subprocess", f"{label} 退出码 {process.returncode}。{detail}".strip())
        return (output or "").strip()


@contextmanager
def file_lock(path, deadline):
    # Locks are released by the OS even when a Colab kernel is interrupted.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(remaining(deadline, 0.2))
        yield


def verify_gpu(config, deadline):
    expected = GPU_MODELS.get(config["gpu"])
    if expected is None:
        raise RuntimeFailure("configuration", "未识别的 Colab GPU 类型。")
    try:
        output = command([
            "nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ], within(deadline, 12))
    except (OSError, RuntimeFailure, subprocess.TimeoutExpired) as error:
        raise RuntimeFailure("capacity_mismatch", "Colab 未提供可验证的 NVIDIA GPU。") from error
    minimum_vram = float(config["min_vram_gib"])
    minimum_ram = float(config["min_ram_gib"])
    if any(not math.isfinite(value) or value < 0 for value in (minimum_vram, minimum_ram)):
        raise RuntimeFailure("configuration", "GPU 和系统内存最低要求必须是有效非负数。")
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not re.fullmatch(r"GPU-[0-9A-Fa-f-]+", parts[0]):
            continue
        uuid, name, total, available = parts
        if total.isdecimal() and available.isdecimal() and re.search(expected, name, re.I):
            devices.append((int(available) / 1024, int(total) / 1024, name, uuid))
    ram_match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", Path("/proc/meminfo").read_text(), re.M)
    ram = int(ram_match.group(1)) / (1024 * 1024) if ram_match else 0.0
    limit_path = Path("/sys/fs/cgroup/memory.max")
    usage_path = Path("/sys/fs/cgroup/memory.current")
    if limit_path.is_file() and usage_path.is_file():
        limit, usage = limit_path.read_text().strip(), usage_path.read_text().strip()
        if limit.isdecimal() and usage.isdecimal():
            ram = min(ram, max(0.0, (int(limit) - int(usage)) / 1024**3))
    if not devices:
        raise RuntimeFailure("capacity_mismatch", f"实际 GPU 与申请的 {config['gpu']} 不符。")
    free, total, name, uuid = max(devices)
    if free < minimum_vram or ram < minimum_ram:
        raise RuntimeFailure("capacity_mismatch", f"{name} 可用显存 {free:.1f} GiB、系统内存 {ram:.1f} GiB，低于模型卡要求。")
    emit(config, "gpu_verified", gpu=name, vram_gib=round(total, 2), free_vram_gib=round(free, 2), ram_gib=round(ram, 2))
    return uuid


def verify_torch(device, deadline):
    code = "import torch; x = torch.ones(1, device='cuda'); x.add_(1).item()"
    try:
        command([sys.executable, "-c", code], within(deadline, 18),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": device})
    except (RuntimeFailure, OSError) as error:
        raise RuntimeFailure("environment", "Colab 基础 PyTorch 无法在已分配 GPU 上完成最小 CUDA 运算。") from error


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def validate_safetensors(path):
    with path.open("rb") as file:
        raw_size = file.read(8)
        if len(raw_size) != 8:
            raise RuntimeFailure("asset_integrity", "模型权重缺少 safetensors 文件头。")
        size = struct.unpack("<Q", raw_size)[0]
        if not 2 <= size <= min(64 * 1024 * 1024, path.stat().st_size - 8):
            raise RuntimeFailure("asset_integrity", "safetensors 模型权重的文件头长度无效。")
        try:
            header = json.loads(file.read(size))
            if not isinstance(header, dict) or not any(name != "__metadata__" for name in header):
                raise ValueError("no tensor")
        except (UnicodeError, ValueError) as error:
            raise RuntimeFailure("asset_integrity", "模型权重的 safetensors 元数据无效。") from error


def matches_asset(path, asset, deadline):
    if not path.is_file():
        return False
    expected_size, expected_hash = asset.get("size_bytes"), asset.get("sha256")
    stat = path.stat()
    if expected_size and stat.st_size != expected_size:
        return False
    marker = path.with_suffix(".verified.json")
    receipt = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "expected_sha256": expected_hash}
    if read_json(marker) == receipt:
        return True
    validate_safetensors(path)
    if expected_hash and digest(path, deadline) != expected_hash:
        return False
    write_json(marker, receipt)
    return True


def cached_receipt(path, asset, marker):
    if not path.is_file():
        return False
    stat = path.stat()
    size = asset.get("size_bytes")
    return (not size or stat.st_size == size) and read_json(marker) == {
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "expected_sha256": asset.get("sha256"),
    }


def asset_url(asset):
    url = asset.get("url")
    if not url:
        return None
    value = checked_url(url)
    if asset.get("sha256"):
        return url
    if value.netloc != "huggingface.co" or value.query:
        raise RuntimeFailure("configuration", "非固定 Hugging Face 文件需要指定 SHA256。")
    parts = safe_relative(unquote(value.path.removeprefix("/"))).parts
    if len(parts) < 5 or parts[2] != "resolve" or not GIT_SHA.fullmatch(parts[3]):
        raise RuntimeFailure("configuration", "模型源须固定 Hugging Face 提交版本或提供 SHA256。")
    return url


def checked_url(value):
    url = urlparse(value)
    host = url.hostname or ""
    try:
        port = url.port
    except ValueError as error:
        raise RuntimeFailure("asset_source", "模型下载链接的端口无效。") from error
    if url.scheme != "https" or not host or url.username or url.password or url.fragment or port not in (None, 443):
        raise RuntimeFailure("asset_source", "模型源与跳转只能使用无凭据的 HTTPS 下载链接。")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if "." in host and not host.lower().rstrip(".").endswith((".localhost", ".local", ".internal")):
            return url
    raise RuntimeFailure("asset_source", "模型下载地址须使用公网域名。")


def verify_public_resolution(url):
    host = checked_url(url).hostname
    addresses = {entry[4][0] for entry in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise RuntimeFailure("asset_source", "模型下载地址未解析到公网。")


class ModelRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        verify_public_resolution(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def download_model(url, partial, expected_size, deadline, progress, *, verify_hash=False):
    """Resume only when the server confirms the same byte range and total size."""
    offset = partial.stat().st_size if partial.exists() else 0
    hasher = None
    if verify_hash:
        hasher = file_hasher(partial, deadline) if offset else hashlib.sha256()
    headers = {"User-Agent": "minimax-design-colab/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    verify_public_resolution(url)
    opener = build_opener(ProxyHandler({}), ModelRedirect())
    with opener.open(Request(url, headers=headers), timeout=remaining(deadline, 12)) as response:
        range_header = response.headers.get("Content-Range", "")
        total = expected_size
        if response.status == 206:
            content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", range_header)
            if not content_range:
                raise RuntimeFailure("asset_integrity", "远端返回无法验证的下载范围。")
            start, end, total = map(int, content_range.groups())
            if start != offset or end + 1 != total or expected_size and total != expected_size:
                raise RuntimeFailure("asset_integrity", "远端返回的下载范围与已缓存文件不一致。")
        elif response.status == 200:
            if offset and hasher is not None:
                hasher = hashlib.sha256()
            offset = 0
            content_length = response.headers.get("Content-Length", "")
            if content_length.isdecimal():
                total = int(content_length)
                if expected_size and total != expected_size:
                    raise RuntimeFailure("asset_integrity", "远端权重大小与模型卡不一致。")
        else:
            raise RuntimeFailure("asset_source", f"模型源返回 HTTP {response.status}。")
        if total and shutil.disk_usage(partial.parent).free < total - offset:
            raise RuntimeFailure("storage", "模型缓存磁盘空间不足。")
        last_bytes, last_update = offset, time.monotonic()
        with partial.open("ab" if offset else "wb") as target:
            while data := response.read1(BUFFER_SIZE):
                remaining(deadline)
                offset += len(data)
                if total and offset > total:
                    raise RuntimeFailure("asset_integrity", "模型下载超过约定文件大小。")
                target.write(data)
                if hasher is not None:
                    hasher.update(data)
                now = time.monotonic()
                if offset - last_bytes >= 512 * 1024**2 or now - last_update >= 20:
                    progress(offset, total)
                    last_bytes, last_update = offset, now
            if not offset or total and offset != total:
                raise RuntimeFailure("asset_source", "模型下载提前中断。")
    return hasher.hexdigest() if hasher is not None else None


def copy_model(source, destination, deadline):
    if shutil.disk_usage(destination.parent).free < source.stat().st_size:
        raise RuntimeFailure("storage", "模型缓存磁盘空间不足。")
    with source.open("rb") as original, destination.open("wb") as staged:
        while data := original.read(BUFFER_SIZE):
            remaining(deadline)
            staged.write(data)


def asset_cache_path(cache, identity):
    return rooted(cache, f"weights/{identity[:2]}/{identity}.safetensors")


def verify_preseeded(seed, target, asset, deadline):
    stat = seed.stat()
    expected_hash, size = asset.get("sha256"), asset.get("size_bytes")
    receipt_path = target.with_suffix(".preseeded.json")
    receipt = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "expected_sha256": expected_hash}
    if size and stat.st_size != size:
        raise RuntimeFailure("asset_integrity", f"预置模型大小不符 {asset['path']}。")
    if read_json(receipt_path) != receipt:
        validate_safetensors(seed)
        if expected_hash and digest(seed, deadline) != expected_hash:
            raise RuntimeFailure("asset_integrity", f"预置模型 SHA256 不符 {asset['path']}。")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt_path, receipt)


def check_cache(config, cache, local, deadline):
    if config.get("cache_dir") and not cache.is_dir():
        raise RuntimeFailure("cache_miss", "配置的持久缓存目录未在当前运行时挂载。")
    hits = []
    writes = {}
    local_upload_bytes = 0
    for asset in config.get("assets", []):
        remaining(deadline)
        identity = asset.get("sha256")
        if not isinstance(identity, str) or not SHA256.fullmatch(identity) or identity in writes or identity in hits:
            continue
        target = asset_cache_path(cache, identity)
        if cached_receipt(target, asset, target.with_suffix(".verified.json")):
            hits.append(identity)
            continue
        seed = rooted(cache / "preseeded", asset["path"])
        if seed.is_file():
            if cached_receipt(seed, asset, target.with_suffix(".preseeded.json")):
                hits.append(identity)
            continue
        size = asset.get("size_bytes", 0)
        if target.is_file() and target.stat().st_size == size:
            if asset.get("upload_name"):
                local_upload_bytes += size
            continue
        partial = target.with_suffix(".part")
        if asset.get("upload_name"):
            local_upload_bytes += size
        writes[identity] = max(0, size - (partial.stat().st_size if partial.is_file() and not asset.get("upload_name") else 0))
    local_free = shutil.disk_usage(local).free
    cache_free = shutil.disk_usage(cache).free if cache.is_dir() else local_free
    size = sum(writes.values())
    if size > cache_free or local_upload_bytes > local_free:
        raise RuntimeFailure("capacity_mismatch", "当前 Colab 的可用磁盘空间容纳不了所选模型。")
    return hits


def prepare_asset(asset, config, cache, run_root, deadline):
    relative = safe_relative(asset["path"])
    if relative.suffix.lower() != ".safetensors" or len(relative.parts) < 2:
        raise RuntimeFailure("configuration", "模型与 LoRA 须使用 ComfyUI 子目录下的 safetensors 权重。")
    expected_hash, size = asset.get("sha256"), asset.get("size_bytes")
    if expected_hash is not None and not SHA256.fullmatch(expected_hash):
        raise RuntimeFailure("configuration", "模型 SHA256 必须是 64 位小写十六进制。")
    if size is not None and (type(size) is not int or size < 1):
        raise RuntimeFailure("configuration", "模型大小必须是正整数。")
    url, upload_name = asset_url(asset), asset.get("upload_name")
    if upload_name and (url or not expected_hash):
        raise RuntimeFailure("configuration", "本地权重上传必须带 SHA256 且与 URL 互斥。")
    stage_root = run_root / "models"
    if stage_root.is_symlink():
        raise RuntimeFailure("configuration", "模型目录必须位于当前任务内。")
    # Staged model links may point outside models, but only into this run's verified cache.
    destination = rooted(stage_root, relative.parent.as_posix()) / relative.name
    if destination.is_symlink():
        if not destination.resolve().is_relative_to(cache.resolve()):
            raise RuntimeFailure("configuration", "现有模型链接超出当前缓存目录。")
    elif destination.exists():
        raise RuntimeFailure("configuration", "模型路径已被其他文件占用。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    seed = rooted(cache / "preseeded", relative.as_posix())
    if not (expected_hash or url or seed.is_file()):
        raise RuntimeFailure("asset_source", f"未配置模型来源或预置缓存 {relative.as_posix()}。")
    identity = expected_hash or hashlib.sha256((url or relative.as_posix()).encode()).hexdigest()
    target = asset_cache_path(cache, identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(target.with_suffix(".lock"), deadline):
        source = None
        try:
            if matches_asset(target, asset, deadline):
                source = target
        except RuntimeFailure as error:
            if error.kind != "asset_integrity":
                raise
        if source is None and seed.is_file():
            verify_preseeded(seed, target, asset, deadline)
            source = seed
        if source is None:
            partial = target.with_suffix(".part")
            if upload_name:
                upload = rooted(run_root / "input", safe_name(upload_name))
                if not upload.is_file() or size and upload.stat().st_size != size:
                    raise RuntimeFailure("asset_integrity", f"上传权重缺失或大小不符 {relative.as_posix()}。")
                validate_safetensors(upload)
                if digest(upload, deadline) != expected_hash:
                    raise RuntimeFailure("asset_integrity", f"上传权重 SHA256 不符 {relative.as_posix()}。")
                if upload.stat().st_dev == partial.parent.stat().st_dev:
                    upload.replace(partial)
                else:
                    copy_model(upload, partial, deadline)
            else:
                if config.get("offline") or not url:
                    raise RuntimeFailure("cache_miss", f"离线缓存缺少模型 {relative.as_posix()}。")
                downloaded_sha = None
                if size and partial.exists() and partial.stat().st_size > size:
                    partial.unlink()
                if not size or not partial.is_file() or partial.stat().st_size < size:
                    for attempt in range(2):
                        try:
                            downloaded_sha = download_model(
                                url, partial, size, deadline,
                                lambda done, total: emit(config, "asset_progress", asset=relative.as_posix(),
                                                         transferred_bytes=done, total_bytes=total),
                                verify_hash=bool(expected_hash),
                            )
                            break
                        except (HTTPException, OSError, RuntimeFailure) as error:
                            if isinstance(error, RuntimeFailure) and error.kind != "asset_source":
                                raise
                            if isinstance(error, HTTPError) and error.code == 416 and attempt == 0:
                                partial.unlink(missing_ok=True)
                            elif attempt or isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                                raise RuntimeFailure("asset_source", f"无法获取模型 {relative.as_posix()}，请检查许可授权、源可达性和缓存。") from error
                            time.sleep(remaining(deadline, 1.0))
                try:
                    validate_safetensors(partial)
                    if expected_hash and (downloaded_sha or digest(partial, deadline)) != expected_hash:
                        raise RuntimeFailure("asset_integrity", f"模型 SHA256 不符 {relative.as_posix()}。")
                except RuntimeFailure as error:
                    if error.kind == "asset_integrity":
                        partial.unlink(missing_ok=True)
                    raise
            partial.replace(target)
            stat = target.stat()
            write_json(target.with_suffix(".verified.json"), {
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "expected_sha256": expected_hash,
            })
            source = target
        if not destination.is_symlink():
            destination.symlink_to(source)
        elif destination.resolve() != source.resolve():
            with TemporaryDirectory(dir=destination.parent, prefix=".asset-") as staged:
                link = Path(staged) / destination.name
                link.symlink_to(source)
                link.replace(destination)


def dependency_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeFailure("environment", f"Colab 基础环境缺少 {package}，需要兼容的运行时版本。") from error


def create_comfy_venv(directory, deadline, log):
    executable = directory / "bin" / "python"
    create = [sys.executable, "-m", "venv", "--clear", "--system-site-packages"]
    try:
        pip_version = re.match(r"(\d+)\.(\d+)", importlib.metadata.version("pip"))
    except importlib.metadata.PackageNotFoundError:
        pip_version = None
    if pip_version and tuple(map(int, pip_version.groups())) >= (22, 3):
        command([*create, "--without-pip", directory], deadline, log=log,
                stage="创建不自带 pip 的 ComfyUI 虚拟环境")
        installer = [sys.executable, "-m", "pip", "--python", executable]
    else:
        command([*create, directory], deadline, log=log, stage="创建 ComfyUI 虚拟环境")
        installer = [executable, "-m", "pip"]
    return executable, installer


def prepare_comfy(config, cache, run_root, deadline):
    revision = config["comfy_revision"]
    if not GIT_SHA.fullmatch(revision):
        raise RuntimeFailure("configuration", "ComfyUI 必须固定到完整的 Git 提交 SHA。")
    checkout_root = cache / "comfy"
    checkout = checkout_root / revision
    checkout_root.mkdir(parents=True, exist_ok=True)
    with file_lock(checkout_root / (revision + ".lock"), deadline), (run_root / "bootstrap.log").open("a", encoding="utf-8") as log:
        if not (checkout / "main.py").is_file():
            if config.get("offline"):
                raise RuntimeFailure("cache_miss", "离线缓存缺少固定版本的 ComfyUI 源码。")
            staging = checkout_root / (revision + "-" + config["job_id"])
            staging.mkdir(exist_ok=False)
            try:
                command(["git", "init", "--quiet", staging], deadline, log=log)
                command(["git", "fetch", "--quiet", "--depth=1", COMFY_REPOSITORY, revision], deadline, cwd=staging, log=log)
                command(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], deadline, cwd=staging, log=log)
                staging.replace(checkout)
            finally:
                if staging.is_dir():
                    shutil.rmtree(staging)
        head = command(["git", "rev-parse", "HEAD"], deadline, cwd=checkout)
        dirty = command(["git", "status", "--porcelain", "--untracked-files=no"], deadline, cwd=checkout)
        if head != revision or dirty:
            raise RuntimeFailure("environment", "ComfyUI 缓存版本或受跟踪文件与固定提交不一致。")
        versions = {package: dependency_version(package) for package in ("torch", "torchvision")}
        wheelhouse = config.get("wheelhouse") or ""
        identity = {
            "python": sys.version, "base_prefix": sys.base_prefix, "packages": versions,
            "requirements": digest(checkout / "requirements.txt", deadline),
            "comfy_revision": revision, "bootstrap_schema": 2,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        env_root = cache / "envs" / key
        env_root.mkdir(parents=True, exist_ok=True)
        executable = env_root / "venv" / "bin" / "python"
        receipt = env_root / "ready.json"
        if executable.is_file() and read_json(receipt) == identity:
            try:
                command([executable, "-c", "import torch, torchvision, av, aiohttp, safetensors"], within(deadline, 35), log=log)
                return checkout, executable, True
            except (RuntimeFailure, OSError, subprocess.TimeoutExpired):
                receipt.unlink(missing_ok=True)
        if config.get("offline") and not wheelhouse:
            raise RuntimeFailure("cache_miss", "离线缓存缺少兼容的 ComfyUI 环境或本地 wheelhouse。")
        if wheelhouse and (not Path(wheelhouse).is_absolute() or not Path(wheelhouse).is_dir()):
            raise RuntimeFailure("environment", "wheelhouse 路径必须是运行时可用的绝对目录。")
        executable, installer = create_comfy_venv(env_root / "venv", deadline, log)
        constraints = env_root / "constraints.txt"
        constraints.write_text("\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n", encoding="ascii")
        pip = [*installer, "install", "--disable-pip-version-check", "--no-input", "--timeout", "15", "--retries", "1", "--cache-dir", cache / "pip", "--constraint", constraints]
        if wheelhouse:
            pip.extend(["--no-index", "--find-links", wheelhouse])
        for attempt in range(2):
            try:
                command([*pip, "--requirement", checkout / "requirements.txt"], deadline, log=log,
                        stage="安装 ComfyUI Python 依赖")
                break
            except RuntimeFailure:
                if attempt or wheelhouse or config.get("offline"):
                    raise
                time.sleep(remaining(deadline, 2.0))
        command([executable, "-c", "import torch, torchvision, av, aiohttp, safetensors"], within(deadline, 35),
                log=log, stage="导入 ComfyUI Python 依赖")
        write_json(receipt, identity)
        return checkout, executable, False


def local_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def request_json(port, method, route, deadline, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(f"http://127.0.0.1:{port}{route}", data, {"Content-Type": "application/json"}, method=method)
    with build_opener(ProxyHandler({})).open(request, timeout=remaining(deadline, 5)) as response:
        return json.load(response)


def graph_for_output(graph, selected):
    if not isinstance(graph, dict) or not isinstance(graph.get(selected), dict) or graph[selected].get("class_type") != "SaveVideo":
        raise RuntimeFailure("graph", "指定输出节点须是 ComfyUI SaveVideo。")
    compiled = {}
    pending = [selected]
    while pending:
        node_id = pending.pop()
        if node_id in compiled:
            continue
        node = graph[node_id]
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict) or not isinstance(node.get("class_type"), str):
            raise RuntimeFailure("graph", f"节点 {node_id} 缺少 class_type 或 inputs。")
        compiled[node_id] = {**node, "inputs": dict(node["inputs"])}
        for item in node["inputs"].values():
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], int) and item[0] in graph:
                pending.append(item[0])
    return compiled


def log_excerpt(path):
    with path.open("rb") as log:
        log.seek(max(0, log.seek(0, os.SEEK_END) - 1400))
        tail = log.read().decode("utf-8", errors="replace")
    tail = re.sub(r"https?://\S+|(?i:bearer\s+\S+)", "[已隐藏]", tail)
    return " ".join(tail.split())[-700:]


def find_output(history, selected, output_dir):
    status = history.get("status", {})
    if status.get("status_str") == "error":
        messages = status.get("messages", [])
        for kind, details in reversed(messages):
            if kind == "execution_error" and isinstance(details, dict):
                raise RuntimeFailure("generation", f"ComfyUI 节点 {details.get('node_id', '?')} 执行失败（{details.get('exception_type', 'error')}）。")
        raise RuntimeFailure("generation", "ComfyUI 视频工作流执行失败。")
    if not status.get("completed"):
        return None
    outputs = history.get("outputs", {}).get(selected, {})
    paths = []
    for entries in outputs.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "output":
                continue
            name = safe_name(entry.get("filename", ""))
            if not name.lower().endswith(".mp4"):
                continue
            folder = entry.get("subfolder", "")
            candidate = rooted(output_dir, f"{folder}/{name}" if folder else name)
            if candidate.is_file() and candidate not in paths:
                paths.append(candidate)
    if len(paths) != 1:
        raise RuntimeFailure("artifact", "选定的视频保存节点必须产出一个 MP4 文件。")
    return paths[0]


def validate_mp4(path, deadline):
    """Read ISO-BMFF top-level boxes and use ffprobe when installed."""
    size = path.stat().st_size
    kinds = set()
    with path.open("rb") as file:
        position = 0
        while position < size:
            remaining(deadline)
            file.seek(position)
            header = file.read(8)
            if len(header) != 8:
                break
            box_size, kind = struct.unpack(">I4s", header)
            header_length = 8
            if box_size == 1:
                extended = file.read(8)
                if len(extended) != 8:
                    break
                box_size = struct.unpack(">Q", extended)[0]
                header_length = 16
            elif box_size == 0:
                box_size = size - position
            if box_size < header_length or position + box_size > size:
                break
            if box_size > header_length:
                kinds.add(kind)
            position += box_size
    if position != size or not {b"ftyp", b"moov", b"mdat"} <= kinds:
        raise RuntimeFailure("artifact", "成片 MP4 缺少完整的容器结构。")
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        media = json.loads(command([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_type,width,height:format=duration", "-of", "json", path,
        ], within(deadline, 20)))
        if not media.get("streams") or float(media.get("format", {}).get("duration", 0)) <= 0:
            raise RuntimeFailure("artifact", "成片无可识别的视频流或时长。")


# Use PyAV from ComfyUI's own environment for frame-exact trimming and audio removal.
MEDIA_PREP = r"""
from fractions import Fraction
from pathlib import Path
import sys

import av

operation, source, target = sys.argv[1:4]
if operation == "tail":
    start, end = map(int, sys.argv[4:6])
    with av.open(source) as input_video, av.open(target, "w") as output_video:
        source_stream = input_video.streams.video[0]
        stream = output_video.add_stream("libx264", rate=24, options={"preset": "fast", "crf": "8"})
        stream.width, stream.height = source_stream.width, source_stream.height
        stream.pix_fmt = "yuv420p"
        count = 0
        for index, frame in enumerate(input_video.decode(source_stream)):
            if index >= end:
                break
            if index < start:
                continue
            if index == end - 1:
                frame.to_image().save(Path(target).with_suffix(".png"))
            frame.pts, frame.time_base = count, Fraction(1, 24)
            output_video.mux(stream.encode(frame))
            count += 1
        output_video.mux(stream.encode())
    if count != end - start:
        raise ValueError(f"Continuation needs {end-start} frames, got {count}")
else:
    with av.open(source) as input_video:
        if input_video.streams.audio:
            with av.open(target, "w", options={"movflags": "+faststart"}) as output_video:
                original = input_video.streams.video[0]
                stream = output_video.add_stream_from_template(original)
                for packet in input_video.demux(original):
                    if packet.dts is not None:
                        packet.stream = stream
                        output_video.mux(packet)
    clean = target if Path(target).is_file() else source
    with av.open(clean) as output_video:
        if not output_video.streams.video or output_video.streams.audio:
            raise ValueError("Silent export needs a video stream and zero audio streams")
"""


def prepare_media(executable, operation, source, target, deadline, *frame_range):
    command([executable, "-c", MEDIA_PREP, operation, source, target, *frame_range],
            within(deadline, 60), stage="artifact")


COMFY_GUARD = """
import os, signal, sys, time
from pathlib import Path

pid, expires = int(sys.argv[1]), float(sys.argv[2])
state = Path(f'/proc/{pid}/stat')
def identity():
    return state.read_text().rsplit(') ', 1)[1].split()[19]
try:
    original = identity()
    while time.monotonic() < expires:
        if identity() != original:
            break
        time.sleep(min(0.5, max(0, expires - time.monotonic())))
    else:
        if identity() == original and os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
except (FileNotFoundError, ProcessLookupError):
    pass
"""


def render_workflows(config):
    if "segments" not in config:
        selected = str(config["output_node"])
        return [(None, selected, graph_for_output(config["graph"], selected), None)]
    segments = config["segments"]
    if not isinstance(segments, list) or not segments:
        raise RuntimeFailure("configuration", "分段工作流须是非空列表。")
    seen = set()
    workflows = []
    for segment in segments:
        segment_id = segment.get("id") if isinstance(segment, dict) else None
        if not isinstance(segment_id, str) or not SEGMENT_ID.fullmatch(segment_id) or segment_id.lower() in seen:
            raise RuntimeFailure("configuration", "分段 ID 无效或重复。")
        seen.add(segment_id.lower())
        selected = str(segment.get("output_node"))
        workflows.append((segment_id, selected, graph_for_output(segment.get("graph"), selected),
                          segment.get("previous_tail")))
    return workflows


def _watch_ws_progress(port, client_id, stop_event, state):
    """Best-effort RFC 6455 WebSocket listener for ComfyUI step progress."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            req = (
                f"GET /ws?clientId={client_id} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            )
            sock.sendall(req.encode("ascii"))
            buf = b""
            while b"\r\n\r\n" not in buf and not stop_event.is_set():
                chunk = sock.recv(4096)
                if not chunk:
                    return
                buf += chunk
            _, _, rest = buf.partition(b"\r\n\r\n")
            sock.settimeout(1.0)
            while not stop_event.is_set():
                while len(rest) < 2:
                    try:
                        chunk = sock.recv(4096)
                    except TimeoutError:
                        if stop_event.is_set():
                            return
                        continue
                    if not chunk:
                        return
                    rest += chunk
                opcode = rest[0] & 0x0F
                masked = (rest[1] & 0x80) != 0
                length = rest[1] & 0x7F
                offset = 2
                if length == 126:
                    while len(rest) < offset + 2:
                        rest += sock.recv(4096)
                    length = struct.unpack("!H", rest[offset:offset + 2])[0]
                    offset += 2
                elif length == 127:
                    while len(rest) < offset + 8:
                        rest += sock.recv(4096)
                    length = struct.unpack("!Q", rest[offset:offset + 8])[0]
                    offset += 8
                mask_len = 4 if masked else 0
                while len(rest) < offset + mask_len + length:
                    rest += sock.recv(4096)
                payload = rest[offset + mask_len:offset + mask_len + length]
                rest = rest[offset + mask_len + length:]
                if opcode == 0x8:
                    return
                if opcode == 0x1:
                    msg = json.loads(payload.decode("utf-8", errors="replace"))
                    if msg.get("type") == "progress":
                        data = msg.get("data") or {}
                        val, mx = data.get("value"), data.get("max")
                        if isinstance(val, int) and isinstance(mx, int) and mx > 0:
                            state["step"] = val
                            state["max_steps"] = mx
    except (OSError, ValueError, RuntimeError):
        return


def render_prompt(config, segment_fields, graph, selected, port, process, output, log_path, deadline):
    client_id = uuid.uuid4().hex
    progress_state = {}
    stop_ws = Event()
    ws_thread = Thread(target=_watch_ws_progress, args=(port, client_id, stop_ws, progress_state), daemon=True)
    ws_thread.start()
    try:
        try:
            submitted = request_json(port, "POST", "/prompt", deadline, {"prompt": graph, "client_id": client_id})
        except HTTPError as error:
            try:
                details = json.loads(error.read(65536))
                problem_nodes = sorted(details.get("node_errors", {}))
                error_kind = details.get("error", {}).get("type", "unknown")
            except (ValueError, TypeError, AttributeError):
                problem_nodes, error_kind = [], "unknown"
            raise RuntimeFailure("graph", f"ComfyUI 校验拒绝 HTTP {error.code}，类型 {error_kind}，节点 {', '.join(problem_nodes) or '未知'}。") from error
        prompt_id = submitted.get("prompt_id", "")
        if not re.fullmatch(r"[0-9a-f-]{36}", prompt_id):
            raise RuntimeFailure("graph", "ComfyUI 未返回明确的工作流任务 ID。")
        emit(config, "sampling", prompt_id=prompt_id, **segment_fields)
        started_at = time.monotonic()
        last_heartbeat = started_at
        while True:
            if process.poll() is not None:
                raise RuntimeFailure("generation", f"ComfyUI 在采样时提前退出。{log_excerpt(log_path)}")
            try:
                response = request_json(port, "GET", "/history/" + prompt_id, deadline)
                result = find_output(response.get(prompt_id, {}), selected, output)
                if result:
                    return result
            except (URLError, TimeoutError, ConnectionError):
                pass
            now = time.monotonic()
            if now - last_heartbeat >= 30:
                extra = {k: progress_state[k] for k in ("step", "max_steps") if k in progress_state}
                emit(config, "sampling_heartbeat", elapsed_seconds=int(now - started_at), **extra, **segment_fields)
                last_heartbeat = now
            time.sleep(remaining(deadline, 0.5))
    finally:
        stop_ws.set()


def render(config, checkout, executable, run_root, device, setup_deadline, deadline):
    workflows = render_workflows(config)
    port = local_port()
    output = run_root / "output"
    log_path = run_root / "comfy.log"
    output.mkdir()
    (run_root / "user").mkdir()
    (run_root / "models").mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": device, "PYTHONUNBUFFERED": "1"}
    args = [executable, checkout / "main.py", "--base-directory", run_root, "--listen", "127.0.0.1", "--port", str(port), "--disable-all-custom-nodes", "--offline", "--disable-metadata", "--disable-assets"]
    with log_path.open("w", encoding="utf-8") as log, subprocess.Popen(
        [str(arg) for arg in args], cwd=checkout, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, env=env, start_new_session=True,
    ) as process:
        guard = None
        try:
            guard = subprocess.Popen([sys.executable, "-c", COMFY_GUARD, str(process.pid), str(deadline)],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
            while True:
                if process.poll() is not None:
                    raise RuntimeFailure("environment", f"ComfyUI 未能启动。{log_excerpt(log_path)}")
                try:
                    info = request_json(port, "GET", "/object_info", setup_deadline)
                    break
                except (URLError, TimeoutError, ConnectionError):
                    time.sleep(remaining(setup_deadline, 0.5))
            node_types = {node["class_type"] for _, _, graph, _ in workflows for node in graph.values()}
            unsupported = sorted(node_types - info.keys())
            if unsupported:
                raise RuntimeFailure("graph", "固定 ComfyUI 环境缺少这些节点 " + ", ".join(unsupported))
            previous_output = None
            mute = config.get("audio_mode") == "mute"
            for segment_id, selected, graph, tail in workflows:
                segment_fields = {} if segment_id is None else {"segment_id": segment_id}
                if tail is not None:
                    end = tail["end_frame"]
                    start = end - tail["frames"]
                    target = run_root / "input" / tail["file"]
                    prepare_media(executable, "tail", previous_output, target, deadline, start, end)
                    emit(config, "continuation_ready", start_frame=start, end_frame=end, **segment_fields)
                source = render_prompt(config, segment_fields, graph, selected, port, process, output, log_path, deadline)
                if mute:
                    silent = source.with_name(source.stem + "-muted.mp4")
                    prepare_media(executable, "mute", source, silent, deadline)
                    if silent.is_file():
                        silent.replace(source)
                    segment_fields["audio_tracks"] = 0
                validate_mp4(source, deadline)
                sha = digest(source, deadline)
                size = source.stat().st_size
                filename = "video.mp4" if segment_id is None else f"video-{segment_id}.mp4"
                previous_output = source.replace(run_root / filename)
                emit(config, "output_ready", sha256=sha, size_bytes=size, **segment_fields)
        finally:
            try:
                kill_group(process)
            finally:
                if guard is not None:
                    if guard.poll() is None:
                        guard.terminate()
                    guard.wait(timeout=3)


def run(config):
    name = config["job_id"]
    if not JOB_NAME.fullmatch(name):
        raise RuntimeFailure("configuration", "运行任务 ID 无效。")
    prepare_only = config.get("prepare_only", False)
    if type(prepare_only) is not bool or (prepare_only and config.get("preflight_only")):
        raise RuntimeFailure("configuration", "准备模式须为布尔值，且与预检模式互斥。")
    timeout = int(config["timeout_seconds"])
    setup_timeout = int(config["setup_timeout_seconds"])
    if not 1 <= timeout <= 43200 or not 1 <= setup_timeout <= timeout:
        raise RuntimeFailure("configuration", "运行与准备阶段的时间预算无效。")
    started = time.monotonic()
    deadline = started + timeout
    if config.get("preflight_only"):
        setup_timeout = min(setup_timeout, 30)
    setup_deadline = SetupBudget(started + setup_timeout)
    device = verify_gpu(config, setup_deadline)
    verify_torch(device, setup_deadline)
    content_root = COLAB_CONTENT_ROOT
    run_root = content_root / name
    configured_cache = config.get("cache_dir")
    cache = Path(configured_cache or run_root / "cache")
    if not cache.is_absolute() or ".." in cache.parts or cache == Path("/"):
        raise RuntimeFailure("configuration", "缓存位置必须是规范的绝对目录。")
    if config.get("preflight_only"):
        hits = check_cache(config, cache, content_root, setup_deadline)
        emit(config, "preflight_ready", cached_asset_sha256=hits)
        return
    if run_root.is_symlink() or (run_root.exists() and not run_root.is_dir()):
        raise RuntimeFailure("configuration", "任务目录无法安全建立。")
    run_root.mkdir(exist_ok=True)
    if configured_cache and not cache.is_dir():
        raise RuntimeFailure("cache_miss", "配置的持久缓存目录未在当前运行时挂载。")
    cache.mkdir(exist_ok=True)
    inputs = run_root / "input"
    inputs.mkdir(exist_ok=True)
    if inputs.is_symlink():
        raise RuntimeFailure("configuration", "输入目录必须位于当前任务内。")
    for reference in config.get("reference_uploads", []):
        path = rooted(inputs, safe_name(reference["name"]))
        if not path.is_file() or digest(path, setup_deadline) != reference["sha256"]:
            raise RuntimeFailure("asset_integrity", "任务参考文件 SHA256 不符。")
    assets = config["assets"]
    if len({asset["path"] for asset in assets}) != len(assets):
        raise RuntimeFailure("configuration", "模型卡有重复的模型目标路径。")
    workers = max(1, min(6, int(config.get("asset_workers", 3))))
    with ThreadPoolExecutor(max_workers=workers + 1) as pool:
        environment_cache = run_root / "runtime-env" if config.get("local_environment") else cache
        emit(config, "environment_setup_started", stage="environment", revision=config["comfy_revision"],
             shared_cache=environment_cache == cache and bool(configured_cache))
        boot = pool.submit(prepare_comfy, config, environment_cache, run_root, setup_deadline)
        model_tasks = [pool.submit(prepare_asset, asset, config, cache, run_root, setup_deadline) for asset in assets]
        tasks = [boot, *model_tasks]
        pending_assets = len(model_tasks)
        if not pending_assets:
            emit(config, "assets_ready", stage="assets", count=0, shared_cache=bool(configured_cache))
        try:
            for task in as_completed(tasks):
                stage = "environment" if task is boot else "assets"
                try:
                    result = task.result()
                except BaseException as failure:
                    emit(config, "setup_failed", stage=stage, error_kind=error_kind(failure))
                    raise
                if task is boot:
                    checkout, python, cache_hit = result
                    emit(config, "environment_ready", stage=stage, cache_hit=cache_hit,
                         revision=config["comfy_revision"])
                else:
                    pending_assets -= 1
                    if not pending_assets:
                        emit(config, "assets_ready", stage=stage, count=len(assets),
                             shared_cache=bool(configured_cache))
        except BaseException:
            setup_deadline.cancelled.set()
            for task in tasks:
                task.cancel()
            raise
    if prepare_only:
        emit(config, "prepared", asset_count=len(assets), environment_cache_hit=cache_hit)
        return
    render(config, checkout, python, run_root, device, setup_deadline, deadline)


def launch_detached(config):
    run_root = Path("/content") / safe_name(config["job_id"])
    run_root.mkdir(parents=True, exist_ok=True)
    runner_path = run_root / "remote_runner.py"
    log_path = run_root / "remote.log"
    exit_path = run_root / "remote.exit"
    exit_path.unlink(missing_ok=True)
    log_path.write_text("", encoding="utf-8")
    inner_code = base64.b64decode(config["_runner_b64"].encode("ascii")).decode("utf-8")
    runner_path.write_text(inner_code, encoding="utf-8")
    wrapper = (
        "import subprocess, sys\n"
        f"with open({str(log_path)!r}, 'ab', buffering=0) as log:\n"
        f"    rc = subprocess.call([sys.executable, '-u', {str(runner_path)!r}], stdout=log, stderr=log, stdin=subprocess.DEVNULL)\n"
        f"open({str(exit_path)!r}, 'w').write(str(rc))\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", wrapper],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    emit(config, "detached_started", pid=proc.pid)


if "CONFIG" in globals():
    try:
        if CONFIG.get("detached") and "_runner_b64" in CONFIG:
            launch_detached(CONFIG)
        else:
            run(CONFIG)
    except BaseException as failure:
        kind = error_kind(failure)
        text = str(failure) if isinstance(failure, RuntimeFailure) else kind
        emit(CONFIG, "failed", error_kind=kind, message=text)
        raise RuntimeError(text) from None
