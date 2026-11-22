"""Compile local model cards into explicit, immutable Colab inputs."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener

from runtime_remote import SEGMENT_ID, NoRedirect, RuntimeFailure, validate_safetensors

HERE = Path(__file__).resolve().parent
KEY = re.compile(r"[a-z][a-z0-9_-]{0,63}")
SHA256 = re.compile(r"[a-f0-9]{64}")
REVISION = "afcb984e76fe9f870bf7746cf394766e1f690563"
# Conservative usable memory estimates, followed by a runtime hardware probe.
GPU_MEMORY = {"G4": 95, "H100": 78, "A100": 39, "L4": 22, "T4": 14}
PRECISION = {"fp32": 80, "bf16": 70, "fp16": 70, "fp8": 50, "int8": 50,
             "w6a8": 40, "int4": 30, "nvfp4": 30}
ASSET_DIRS = {"diffusion_models", "checkpoints", "text_encoders", "vae", "loras",
              "embeddings", "model_patches", "controlnet", "clip_vision", "upscale_models"}
MEDIA_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".webm",
               ".mp3", ".wav", ".flac", ".m4a"}
IMAGE_TYPES = {".png", ".jpeg", ".jpg", ".webp"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def storage_name(asset):
    expect = str(asset["path"])
    target = PurePosixPath(expect)
    if (target.is_absolute() or not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part)
                                       for part in target.parts)
            or len(target.parts) < 2 or target.parts[0] not in ASSET_DIRS
            or target.suffix != ".safetensors" or target.as_posix() != expect):
        raise ValueError("Model paths must be relative models/<folder> .safetensors paths")
    return "/".join(target.parts[1:])


def validate_asset(asset):
    storage_name(asset)
    if not SHA256.fullmatch(str(asset.get("sha256", ""))):
        raise ValueError("Every model and LoRA needs a SHA-256")
    if type(asset.get("size_bytes")) is not int or asset["size_bytes"] <= 0:
        raise ValueError("Every model and LoRA needs its exact byte size")
    if asset.get("url"):
        parsed = urlsplit(asset["url"])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Model URLs must be HTTPS and contain no username or password")
    if asset.get("url") and asset.get("local_path"):
        raise ValueError("Use one model source per asset")
    if asset.get("local_path") and not Path(asset["local_path"]).is_absolute():
        raise ValueError("Local model paths must be absolute")


def validate_card(card):
    if not KEY.fullmatch(card.get("id", "")) or not card.get("variants"):
        raise ValueError("Model card needs a valid id and at least one variant")
    if not re.fullmatch(r"[a-f0-9]{40}", card.get("comfy_revision", REVISION)):
        raise ValueError("comfy_revision must be a pinned Git SHA")
    if not card.get("native_h3"):
        graph = card.get("graph")
        validate_graph(graph, str(card.get("output_node", "")))
    if card.get("native_h3") and card.get("mode") not in {"t2v", "i2v", "r2v-images"}:
        raise ValueError("Unknown native H3 card mode")
    ids = set()
    for variant in card["variants"]:
        name = variant.get("id", "")
        if not KEY.fullmatch(name) or name in ids or variant.get("precision") not in PRECISION:
            raise ValueError("Variant ids must be unique and each precision must be recognized")
        ids.add(name)
        if not variant.get("assets") or not isinstance(variant["assets"], dict):
            raise ValueError("Variant needs an asset map")
        for field in ("min_vram_gib", "min_ram_gib"):
            if type(variant.get(field)) not in (int, float) or not 1 <= variant[field] <= 1024:
                raise ValueError(f"Invalid {field}")
        if any(gpu not in GPU_MEMORY for gpu in variant.get("gpus", [])):
            raise ValueError("Variant contains an unsupported GPU")
        for asset in variant["assets"].values():
            validate_asset(asset)
        if card.get("native_h3") and not {"unet", "clip", "video_vae", "audio_vae"} <= variant["assets"].keys():
            raise ValueError("Native H3 needs the diffusion model, encoder and two VAEs")
    for key, lora in card.get("loras", {}).items():
        if not KEY.fullmatch(key) or not set(lora.get("variants", ids)) <= ids:
            raise ValueError("LoRA points to an unknown variant")
        validate_asset(lora["asset"])
        if not lora["asset"]["path"].startswith("loras/"):
            raise ValueError("LoRA files belong under models/loras")
    return card


def validate_graph(graph, output_node):
    if (not isinstance(graph, dict) or not graph
            or any(not isinstance(node, dict) or not isinstance(node.get("class_type"), str)
                   or not isinstance(node.get("inputs"), dict) for node in graph.values())):
        raise ValueError("Use an exported ComfyUI API-format graph")
    if graph.get(output_node, {}).get("class_type") != "SaveVideo":
        raise ValueError("output_node must point to a native SaveVideo node")


def load_model_cards(root):
    cards = {c["id"]: validate_card(c) for c in read_json(HERE / "default_cards.json")}
    for manifest in sorted((root / "cards").glob("*.json")):
        card = validate_card(read_json(manifest))
        cards[card["id"]] = card
    return cards


def check_lora_registration(registration, card_id, lora_id):
    if (not isinstance(registration, dict) or not KEY.fullmatch(card_id) or not KEY.fullmatch(lora_id)
            or registration.get("card_id") != card_id or registration.get("lora_id") != lora_id):
        raise ValueError("LoRA registration IDs must match the registry path")
    variants = registration.get("variants")
    hashes = registration.get("unet_sha256")
    if (not isinstance(variants, list) or not variants
            or any(not isinstance(item, str) or not KEY.fullmatch(item) for item in variants)
            or len(variants) != len(set(variants)) or not isinstance(hashes, dict) or set(hashes) != set(variants)
            or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in hashes.values())):
        raise ValueError("Every registered variant must be bound to its native H3 diffusion model SHA-256")
    steps = registration.get("steps")
    if steps is not None and (type(steps) is not int or not 1 <= steps <= 100):
        raise ValueError("The LoRA's sampling steps must be an integer from 1 to 100")
    asset = registration.get("asset")
    if (not isinstance(asset, dict) or asset.get("path") != f"loras/{lora_id}.safetensors"
            or not any(asset.get(key) for key in ("url", "local_path"))
            or any(asset.get(key) is not None and not isinstance(asset[key], str) for key in ("url", "local_path"))):
        raise ValueError("A registered LoRA requires an explicit source and its own destination")
    validate_asset(asset)


def lora_registry(root, cards):
    for path in sorted((root / "loras").glob("*/*.json")):
        card_id, lora_id = path.parent.name, path.stem
        record = {"card_id": card_id, "lora_id": lora_id, "path": str(path)}
        try:
            registration = read_json(path)
            check_lora_registration(registration, card_id, lora_id)
        except (OSError, ValueError, TypeError) as error:
            yield record | {"status": "invalid", "active_variants": [], "inactive_variants": {}, "reason": str(error)}
            continue
        card = cards.get(card_id, {})
        targets = ({variant["id"]: variant["assets"] for variant in card["variants"]}
                   if card.get("native_h3") is True else {})
        target = registration["asset"]["path"]
        inline_loras = card.get("loras", {})
        collision = lora_id in inline_loras or any(item["asset"]["path"] == target for item in inline_loras.values())
        active, inactive = [], {}
        for variant in registration["variants"]:
            assets = targets.get(variant)
            if collision or (assets and any(item["path"] == target for item in assets.values())):
                inactive[variant] = "destination_conflict"
            elif not assets:
                inactive[variant] = "base_variant_unavailable"
            elif assets["unet"]["sha256"] != registration["unet_sha256"][variant]:
                inactive[variant] = "base_weight_changed"
            else:
                active.append(variant)
        status = "incompatible" if not active else "partial" if inactive else "ready"
        yield {**registration, **record, "status": status, "active_variants": active, "inactive_variants": inactive}


def list_lora_registrations(root):
    return list(lora_registry(root, load_model_cards(root)))


def load_cards(root):
    cards = load_model_cards(root)
    for item in list(lora_registry(root, cards)):
        if item["active_variants"]:
            selection = {"asset": item["asset"], "variants": item["active_variants"]}
            if item.get("steps") is not None:
                selection["steps"] = item["steps"]
            cards[item["card_id"]].setdefault("loras", {})[item["lora_id"]] = selection
    return cards


def hugging_face_lora_metadata(repo, revision, file):
    """Look up one public model file and pin its downloadable URL to a commit."""
    repo_part = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
    if not isinstance(repo, str) or not re.fullmatch(f"{repo_part}/{repo_part}", repo):
        raise ValueError("Use a Hugging Face model repository in owner/name form")
    if not isinstance(revision, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", revision):
        raise ValueError("Specify a Hugging Face commit, branch or tag")
    if (not isinstance(file, str) or not file.endswith(".safetensors") or "\\" in file
            or any(part in {"", ".", ".."} for part in file.split("/"))
            or any(ord(character) < 32 or ord(character) == 127 for character in file)):
        raise ValueError("Specify an exact repository-relative .safetensors file")
    source = f"https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe='')}?blobs=true"
    try:
        with build_opener(NoRedirect()).open(Request(source, headers={"Accept": "application/json"}), timeout=15) as response:
            limit = 8 * 1024 * 1024
            body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("repository metadata is too large")
        metadata = json.loads(body)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read Hugging Face model metadata for {repo}@{revision}") from error
    if not isinstance(metadata, dict) or metadata.get("gated") or metadata.get("private") or metadata.get("disabled"):
        raise ValueError("The Hugging Face model repository must be publicly downloadable")
    commit = metadata.get("sha", "")
    siblings = metadata.get("siblings")
    if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit) or not isinstance(siblings, list):
        raise ValueError("Hugging Face did not return a commit and file list")
    asset = next((item for item in siblings if isinstance(item, dict) and item.get("rfilename") == file), None)
    if asset is None:
        raise ValueError("The exact LoRA file was not found at the selected revision")
    lfs = asset.get("lfs")
    asset_sha = lfs.get("sha256") if isinstance(lfs, dict) else None
    size = asset.get("size")
    if not isinstance(asset_sha, str) or not SHA256.fullmatch(asset_sha) or type(size) is not int or size < 1:
        raise ValueError("Hugging Face did not publish the LoRA file's SHA-256 and exact size")
    return {"url": f"https://huggingface.co/{repo}/resolve/{commit}/{quote(file, safe='/')}",
            "size_bytes": size, "sha256": asset_sha}


def build_lora(root, *, card_id, lora_id, variants, local_path=None, hf_repo=None, hf_revision=None, hf_file=None,
               steps=None):
    """Build a standalone LoRA registration pinned to compatible native H3 weights."""
    cards = load_model_cards(root)
    base = cards.get(card_id) if isinstance(card_id, str) else None
    if not base or base.get("native_h3") is not True:
        raise ValueError("Select an existing native H3 model card")
    if not isinstance(lora_id, str) or not KEY.fullmatch(lora_id) or lora_id in base.get("loras", {}):
        raise ValueError("Give the LoRA a new, valid ID on the base card")
    known = {variant["id"]: variant for variant in base["variants"]}
    if (not isinstance(variants, list) or not variants or any(not isinstance(item, str) or item not in known for item in variants)
            or len(variants) != len(set(variants))):
        raise ValueError("Select one or more unique compatible variants from the base H3 card")
    if steps is not None and (type(steps) is not int or not 1 <= steps <= 100):
        raise ValueError("The LoRA's documented sampling steps must be an integer from 1 to 100")
    target = f"loras/{lora_id}.safetensors"
    if any(item["path"] == target for variant in variants for item in known[variant]["assets"].values()):
        raise ValueError("The new LoRA destination collides with a model weight on a selected variant")
    if any(item["asset"]["path"] == target for item in base.get("loras", {}).values()):
        raise ValueError("The new LoRA destination collides with an existing LoRA")
    if local_path is not None and all(item is None for item in (hf_repo, hf_revision, hf_file)):
        path = Path(local_path)
        if not path.is_absolute() or path.suffix.lower() != ".safetensors" or not path.is_file():
            raise ValueError("Provide an existing absolute .safetensors path")
        path = path.resolve()
        try:
            stat = path.stat()
            validate_safetensors(path)
            source = {"local_path": str(path), "size_bytes": stat.st_size, "sha256": sha256(path)}
            after = path.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("The local LoRA changed during validation")
        except (OSError, RuntimeFailure) as error:
            raise ValueError(f"Could not validate local safetensors LoRA: {error}") from error
    elif local_path is None and all(item is not None for item in (hf_repo, hf_revision, hf_file)):
        source = hugging_face_lora_metadata(hf_repo, hf_revision, hf_file)
    else:
        raise ValueError("Use one local file or provide the Hugging Face repository, revision and file together")
    lora = {"card_id": card_id, "lora_id": lora_id, "variants": list(variants), "asset": {"path": target, **source},
            "unet_sha256": {variant: known[variant]["assets"]["unet"]["sha256"] for variant in variants}}
    if steps is not None:
        lora["steps"] = steps
    check_lora_registration(lora, card_id, lora_id)
    return lora


CONTINUATION_FRAMES = 56


def native_graph(mode, width, height, frames, seed, prompt, names, loras, steps, refs,
                 *, mute=False, tail_file=None):
    graph = {}

    def add(node, **inputs):
        key = str(len(graph) + 1)
        graph[key] = {"class_type": node, "inputs": inputs}
        return [key, 0]

    unet = add("UNETLoader", unet_name=names["unet"], weight_dtype="default")
    clip = add("CLIPLoader", clip_name=names["clip"], type="minimax", device="default")
    video_vae = add("VAELoader", vae_name=names["video_vae"])
    audio_vae = add("VAELoader", vae_name=names["audio_vae"]) if not mute else None
    for name, strength in loras:
        unet = add("LoraLoaderModelOnly", model=unet, lora_name=name, strength_model=strength)
    inputs = {}
    if mode == "i2v":
        inputs["first_frame"] = add("LoadImage", image=refs["first_frame"])
        if "last_frame" in refs:
            inputs["last_frame"] = add("LoadImage", image=refs["last_frame"])
    if mode == "r2v-images":
        for i in range(len(refs)):
            inputs[f"ref_images.ref_image_{i}"] = add("LoadImage", image=refs[f"image_{i+1}"])
        if tail_file:
            previous = add("LoadVideo", file=tail_file)
            inputs["ref_videos.ref_video_0"] = add("GetVideoComponents", video=previous)
            prompt = ("Continue directly after the final frame of <Video 1>. Preserve the final pose, screen positions, "
                      "wardrobe state, props, camera direction and lighting. Continue the action already in progress.\n"
                      + prompt)
        condition = add("MiniMaxH3ReferenceToVideo", clip=clip, vae=video_vae,
                        prompt=prompt, width=width, height=height, length=frames, ref_image_size="match", **inputs)
    else:
        condition = add("MiniMaxH3ImageToVideo", clip=clip, vae=video_vae, prompt=prompt, width=width,
                        height=height, length=frames, **inputs)
    conditioning = condition
    if tail_file:
        last_frame = add("LoadImage", image=str(PurePosixPath(tail_file).with_suffix(".png")))
        conditioning = add("MiniMaxH3AddGuide", positive=condition, vae=video_vae,
                           latent=[condition[0], 1], image=last_frame, frame_idx=0)
    guider = add("BasicGuider", model=unet, conditioning=conditioning)
    sampler = add("KSamplerSelect", sampler_name="res_multistep")
    sigmas = add("BasicScheduler", model=unet, scheduler="simple", steps=steps, denoise=1)
    noise = add("RandomNoise", noise_seed=seed)
    latent = add("SamplerCustomAdvanced", guider=guider, sampler=sampler, sigmas=sigmas,
                 noise=noise, latent_image=[condition[0], 1])
    images = add("VAEDecode", samples=latent, vae=video_vae)
    video_inputs = {"images": images, "fps": 24}
    if not mute:
        video_inputs["audio"] = add("VAEDecodeAudio", samples=latent, vae=audio_vae)
    video = add("CreateVideo", **video_inputs)
    output = add("SaveVideo", video=video, filename_prefix="deliverable", format="mp4", codec="h264")
    return graph, output[0]


def bindings_used(value):
    if isinstance(value, dict):
        if set(value) == {"$bind"}:
            return {value["$bind"]}
        return set().union(*(bindings_used(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(bindings_used(child) for child in value))
    return set()


def bind(value, values):
    if isinstance(value, dict):
        if set(value) == {"$bind"}:
            return values[value["$bind"]]
        return {key: bind(child, values) for key, child in value.items()}
    if isinstance(value, list):
        return [bind(child, values) for child in value]
    return value


def make_plan(root, args, settings):
    card = load_cards(root)[args.get("card", "h3-t2v")]
    ranges = card.get("limits", {"max_seconds": 15, "max_pixels": 768 * 1360})
    width, height = args.get("width", 480), args.get("height", 864)
    if any(type(x) is not int or x < 32 or x % 32 for x in (width, height)):
        raise ValueError("Width and height must be positive multiples of 32")
    if width * height > ranges["max_pixels"]:
        raise ValueError("Duration or resolution exceeds the selected card's limits")
    audio_mode = args.get("audio_mode", "generated")
    if audio_mode not in ("generated", "mute"):
        raise ValueError("audio_mode must be generated or mute")
    is_batch = "segments" in args
    preflight = args.get("preflight_only", False)
    takes = args.get("takes", 3 if card.get("native_h3") and not preflight else 1)
    if type(takes) is not int or not 1 <= takes <= 3:
        raise ValueError("Takes must be an integer from 1 to 3")
    if takes != 1 and (preflight or not card.get("native_h3")):
        raise ValueError("Multiple takes require a native H3 generation")
    continuation = args.get("continuation", "independent")
    if continuation not in ("independent", "previous_tail"):
        raise ValueError("continuation must be independent or previous_tail")
    if continuation == "previous_tail" and not (is_batch and card.get("native_h3") and card["mode"] == "r2v-images"):
        raise ValueError("previous_tail requires a batch on the native H3 reference-to-video card")
    if is_batch:
        if {"prompt", "seconds"}.intersection(args):
            raise ValueError("Use segment prompts and durations for a batch")
        if args.get("preflight_only"):
            raise ValueError("Use a single plan for a hardware-only preflight")
        entries = args["segments"]
        if not isinstance(entries, list) or not 2 <= len(entries) <= 8:
            raise ValueError("A video batch must contain 2 to 8 segments")
    else:
        entries = [{"prompt": args.get("prompt", ""), "seconds": args.get("seconds", 5)}]
    shots = []
    segment_ids = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.keys() - {"id", "prompt", "seconds", "seed"}:
            raise ValueError("Each segment accepts id, prompt, seconds and optional seed")
        if is_batch:
            segment_id = entry.get("id")
            if (not isinstance(segment_id, str) or not SEGMENT_ID.fullmatch(segment_id)
                    or segment_id.lower() in segment_ids):
                raise ValueError("Segment IDs must be unique, path-safe strings of 1 to 32 characters")
            segment_ids.add(segment_id.lower())
        seconds = entry.get("seconds")
        if type(seconds) not in (int, float) or not 1 <= seconds <= ranges["max_seconds"]:
            raise ValueError("Duration or resolution exceeds the selected card's limits")
        seed = entry.get("seed", args.get("seed", 1))
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("Seed is outside ComfyUI's range")
        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or len(prompt) > 30000 or not prompt.strip():
            raise ValueError("Enter a nonempty video prompt of up to 30000 characters")
        edit_frames = round(seconds * 24)
        if continuation == "previous_tail" and index < len(entries) - 1 and edit_frames < CONTINUATION_FRAMES:
            raise ValueError("Segments feeding a continuation need at least 56 requested frames")
        frames = max(5, edit_frames)
        frames += (5 - frames) % 17
        shots.append({"id": entry.get("id"), "prompt": prompt, "frames": frames, "edit_frames": edit_frames,
                      "seed": seed, "continue_previous": continuation == "previous_tail" and index > 0})
    if takes > 1:
        # Keep each take's sequence adjacent so previous_tail can only read within that take.
        shots = [
            {**shot, "id": f"t{take}-s{index}-{(shot['id'] or 'clip')[:26]}",
             "segment_id": shot["id"] or "clip", "take": take,
             "seed": (shot["seed"] + take - 1) % (2**64)}
            for take in range(1, takes + 1)
            for index, shot in enumerate(shots, 1)
        ]
        is_batch = True
    refs, uploads = {}, {}
    for index, (key, source) in enumerate(args.get("references", {}).items()):
        path = Path(source)
        if not path.is_absolute() or not path.is_file() or path.suffix.lower() not in MEDIA_TYPES:
            raise ValueError("References must be existing local media files with absolute paths")
        name = f"reference-{index + 1}{path.suffix.lower()}"
        refs[key] = name
        uploads[name] = {"local_path": str(path.resolve()), "sha256": sha256(path), "name": name}
    if card.get("native_h3"):
        if card["mode"] == "t2v" and refs:
            raise ValueError("Choose the native i2v card or a custom reference graph for media")
        if card["mode"] == "i2v" and ("first_frame" not in refs or set(refs) - {"first_frame", "last_frame"}
                or any(PurePosixPath(name).suffix not in IMAGE_TYPES for name in refs.values())):
            raise ValueError("I2V needs first_frame and accepts an optional last_frame image")
        if card["mode"] == "r2v-images" and (not 1 <= len(refs) <= 9
                or set(refs) != {f"image_{i+1}" for i in range(len(refs))}
                or any(PurePosixPath(name).suffix not in IMAGE_TYPES for name in refs.values())):
            raise ValueError("Reference video requires one to nine images numbered image_1, image_2, …")
    gpus = args.get("gpus", settings.get("gpus", ["H100", "G4", "A100", "L4"]))
    if not gpus or len(gpus) != len(set(gpus)) or any(g not in GPU_MEMORY for g in gpus):
        raise ValueError("Provide ordered, unique supported Colab GPU names")
    requested_loras = args.get("loras", {})
    missing_loras = set(requested_loras).difference(card.get("loras", {}))
    if missing_loras:
        stale = missing_loras.intersection(item["lora_id"] for item in list_lora_registrations(root)
                                          if item["card_id"] == card["id"])
        if stale:
            raise ValueError("Registered LoRA has no compatible active variant: " + ", ".join(sorted(stale)))
        raise ValueError("A requested LoRA is not on this model card")
    if any(type(s) not in (int, float) or not -4 <= s <= 4 for s in requested_loras.values()):
        raise ValueError("LoRA strength must be a number between -4 and 4")
    chosen = args.get("variant")
    if chosen and chosen not in {v["id"] for v in card["variants"]}:
        raise ValueError("Unknown model variant")
    candidates = []
    variants = sorted(card["variants"], key=lambda v: (PRECISION[v["precision"]], v.get("quality_rank", 0)), reverse=True)
    for variant in variants:
        if chosen and chosen != variant["id"]:
            continue
        if any(variant["id"] not in card["loras"][name].get("variants", [variant["id"]]) for name in requested_loras):
            continue
        eligible = [gpu for gpu in gpus if GPU_MEMORY[gpu] >= variant["min_vram_gib"]
                    and gpu in variant.get("gpus", gpus)]
        if not eligible:
            continue
        assets = copy.deepcopy(variant["assets"])
        for name in requested_loras:
            assets[f"lora.{name}"] = copy.deepcopy(card["loras"][name]["asset"])
        names = {key: storage_name(asset) for key, asset in assets.items()}
        if len({a["path"] for a in assets.values()}) != len(assets):
            raise ValueError("Model assets must have distinct target paths")
        values = {**args.get("bindings", {}), "width": width, "height": height}
        values.update({f"asset.{k}": v for k, v in names.items()})
        values.update({f"ref.{k}": v for k, v in refs.items()})
        values.update({f"strength.{k}": v for k, v in requested_loras.items()})
        configured_steps = {card["loras"][name]["steps"] for name in requested_loras
                            if "steps" in card["loras"][name]}
        if len(configured_steps) > 1:
            raise ValueError("Selected LoRAs require conflicting sampling schedules")
        preset_steps = next(iter(configured_steps), variant.get("steps", 20))
        steps = args.get("steps", preset_steps)
        if configured_steps and steps != preset_steps:
            raise ValueError("Selected LoRA requires its preset sampling steps")
        if type(steps) is not int or not 1 <= steps <= 100:
            raise ValueError("Sampling steps must be an integer from 1 to 100")
        values["steps"] = steps
        graphs = []
        for index, shot in enumerate(shots):
            tail_file = f"continuation-tail-{index}.mp4" if shot["continue_previous"] else None
            inputs = {**values, **{key: shot[key] for key in ("prompt", "frames", "seed")}}
            if card.get("native_h3"):
                if args.get("bindings"):
                    raise ValueError("Custom graph bindings require a custom card")
                graph, output_node = native_graph(
                    card["mode"], width, height, shot["frames"], shot["seed"], shot["prompt"], names,
                    [(names[f"lora.{k}"], s) for k, s in requested_loras.items()], steps, refs,
                    mute=audio_mode == "mute", tail_file=tail_file)
            else:
                template = card["graph"]
                declared = bindings_used(template)
                unused = ({f"ref.{k}" for k in refs} | {f"asset.lora.{k}" for k in requested_loras}
                          | set(args.get("bindings", {}))) - declared
                if unused:
                    raise ValueError("Inputs without graph bindings: " + ", ".join(sorted(unused)))
                missing = declared - inputs.keys()
                if missing:
                    raise ValueError("Missing graph bindings: " + ", ".join(sorted(missing)))
                graph, output_node = bind(template, inputs), str(card["output_node"])
            validate_graph(graph, output_node)
            prefix = f"deliverable-{shot['id']}" if is_batch else "deliverable"
            graph[output_node]["inputs"].update(filename_prefix=prefix, format="mp4", codec="h264")
            graph_entry = {"id": shot["id"], "graph": graph, "output_node": output_node}
            if tail_file:
                graph_entry["previous_tail"] = {"file": tail_file, "frames": CONTINUATION_FRAMES,
                                                "end_frame": shots[index - 1]["edit_frames"]}
            graphs.append(graph_entry)
        workflows = {"segments": graphs} if is_batch else {k: graphs[0][k] for k in ("graph", "output_node")}
        model_uploads = dict(uploads)
        for asset in assets.values():
            if local := asset.pop("local_path", None):
                path = Path(local)
                if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
                    raise ValueError(f"Local model file is missing or has the wrong size: {path}")
                name = f"model-{asset['sha256']}.safetensors"
                asset["upload_name"] = name
                model_uploads[name] = {"name": name, "local_path": str(path.resolve()), "sha256": asset["sha256"]}
        for gpu in eligible:
            candidates.append({"variant": variant["id"], "precision": variant["precision"], "gpu": gpu,
                               "min_vram_gib": variant["min_vram_gib"], "min_ram_gib": variant["min_ram_gib"],
                               "assets": list(assets.values()), "uploads": list(model_uploads.values()), **workflows})
    if not candidates:
        raise ValueError("No selected GPU fits the compatible model and LoRA variants")
    frame_count = sum(shot["frames"] for shot in shots)
    plan = {"card": card["id"], "frames": frame_count,
            "comfy_revision": card.get("comfy_revision", REVISION), "seconds": frame_count / 24,
            "width": width, "height": height, "candidates": candidates,
            "audio_mode": audio_mode, "continuation": continuation, "takes": takes}
    if is_batch:
        plan["segments"] = [{"id": shot["id"], "frames": shot["frames"], "seconds": shot["frames"] / 24,
                             "edit_frames": shot["edit_frames"],
                             **({"segment_id": shot["segment_id"], "take": shot["take"]} if takes > 1 else {})}
                            for shot in shots]
    return plan
