import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from model_cards import build_lora, list_lora_registrations, load_cards, make_plan


class LoraImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "adapter.safetensors"
        header = json.dumps({"adapter.lora_up.weight": {"dtype": "F32", "shape": [1, 1],
                                                         "data_offsets": [0, 4]}}).encode()
        header += b" " * (-len(header) % 8)
        self.path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))
        self.options = {"card_id": "h3-t2v", "lora_id": "my-style", "variants": ["pruned-int8"]}

    def register(self, entry):
        path = self.root / "loras" / entry["card_id"] / (entry["lora_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry), encoding="utf-8")
        return path

    def test_local_adapter_hash_and_variant_reach_the_real_graph(self):
        entry = build_lora(self.root, local_path=str(self.path), steps=18, **self.options)
        asset = entry["asset"]
        self.assertEqual(asset["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(asset["size_bytes"], self.path.stat().st_size)
        self.assertNotIn("my-style", load_cards(self.root)["h3-t2v"]["loras"])
        self.assertFalse((self.root / "loras").exists())
        self.register(entry)
        self.assertEqual(list_lora_registrations(self.root)[0]["status"], "ready")
        args = {"card": entry["card_id"], "prompt": "Two flowers sway in a quiet garden.", "takes": 1,
                "loras": {"my-style": 0.65}, "gpus": ["A100"]}
        plan = make_plan(self.root, args, {})
        self.assertEqual([candidate["variant"] for candidate in plan["candidates"]], ["pruned-int8"])
        candidate = plan["candidates"][0]
        nodes = {node["class_type"]: node["inputs"] for node in candidate["graph"].values()}
        self.assertEqual(nodes["LoraLoaderModelOnly"]["lora_name"], "my-style.safetensors")
        self.assertEqual(nodes["LoraLoaderModelOnly"]["strength_model"], 0.65)
        self.assertEqual(nodes["BasicScheduler"]["steps"], 18)
        self.assertIn({"name": f"model-{asset['sha256']}.safetensors", "local_path": str(self.path.resolve()),
                       "sha256": asset["sha256"]}, candidate["uploads"])
        with self.assertRaisesRegex(ValueError, "compatible model and LoRA variants"):
            make_plan(self.root, {**args, "variant": "full-int8"}, {})

    def test_local_invalid_safetensors_is_rejected_before_card_creation(self):
        self.path.write_bytes(b"plain text")
        with self.assertRaisesRegex(ValueError, "validate local safetensors"):
            build_lora(self.root, local_path=str(self.path), **self.options)
        self.assertFalse((self.root / "loras").exists())

    def test_explicit_variants_and_unused_builtin_ids_are_required(self):
        for fields, message in (({"card_id": "unknown-card"}, "existing native H3 model card"),
                                ({"lora_id": "turbo-8step"}, "new, valid ID"),
                                ({"variants": []}, "compatible variants"),
                                ({"variants": ["made-up"]}, "compatible variants"),
                                ({"variants": ["pruned-int8", "pruned-int8"]}, "compatible variants")):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, message):
                build_lora(self.root, local_path=str(self.path), **(self.options | fields))

    def test_base_weight_changes_disable_only_the_changed_variant(self):
        entry = build_lora(self.root, local_path=str(self.path),
                           **(self.options | {"variants": ["full-int8", "pruned-int8"]}))
        self.register(entry)
        card = load_cards(self.root)["h3-t2v"]
        card["loras"].pop("my-style")
        variants = {variant["id"]: variant for variant in card["variants"]}
        variants["full-int8"]["assets"]["unet"]["sha256"] = "e" * 64
        cards = self.root / "cards"
        cards.mkdir()
        card_path = cards / "h3-t2v.json"
        card_path.write_text(json.dumps(card), encoding="utf-8")
        record = list_lora_registrations(self.root)[0]
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["active_variants"], ["pruned-int8"])
        self.assertEqual(record["inactive_variants"], {"full-int8": "base_weight_changed"})
        args = {"card": "h3-t2v", "prompt": "A quiet garden.", "loras": {"my-style": 0.65}, "gpus": ["A100"]}
        self.assertEqual([c["variant"] for c in make_plan(self.root, args, {})["candidates"]], ["pruned-int8"])
        variants["pruned-int8"]["assets"]["unet"]["sha256"] = "f" * 64
        card_path.write_text(json.dumps(card), encoding="utf-8")
        self.assertEqual(list_lora_registrations(self.root)[0]["status"], "incompatible")
        with self.assertRaisesRegex(ValueError, "Registered LoRA has no compatible active variant"):
            make_plan(self.root, args, {})

    def test_invalid_registry_stays_visible_without_breaking_a_valid_registration(self):
        entry = build_lora(self.root, local_path=str(self.path), **self.options)
        self.register(entry)
        self.register(entry | {"lora_id": "bad-record"}).write_text("not json", encoding="utf-8")
        statuses = {item["lora_id"]: item["status"] for item in list_lora_registrations(self.root)}
        self.assertEqual(statuses, {"bad-record": "invalid", "my-style": "ready"})
        self.assertIn("my-style", load_cards(self.root)["h3-t2v"]["loras"])

    def test_hugging_face_lookup_pins_commit_and_never_fetches_weights(self):
        commit, digest = "a" * 40, "b" * 64
        file = "nested/My LoRA.safetensors"
        body = json.dumps({"sha": commit, "siblings": [{"rfilename": file, "size": 33,
                                                       "lfs": {"sha256": digest}}]}).encode()
        with patch("model_cards.build_opener") as factory:
            loader = factory.return_value.open
            loader.return_value = io.BytesIO(body)
            entry = build_lora(self.root, hf_repo="artist/model", hf_revision="refs/pr/3", hf_file=file,
                               **self.options)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0].full_url,
                         "https://huggingface.co/api/models/artist/model/revision/refs%2Fpr%2F3?blobs=true")
        asset = entry["asset"]
        self.assertEqual(asset["url"], f"https://huggingface.co/artist/model/resolve/{commit}/nested/My%20LoRA.safetensors")
        self.assertEqual((asset["size_bytes"], asset["sha256"]), (33, digest))

    def test_hugging_face_missing_lfs_sha_fails_closed(self):
        body = json.dumps({"sha": "a" * 40, "siblings": [{"rfilename": "model.safetensors", "size": 33,
                                                       "blobId": "b" * 40}]}).encode()
        with patch("model_cards.build_opener") as factory, self.assertRaisesRegex(ValueError, "SHA-256"):
            factory.return_value.open.return_value = io.BytesIO(body)
            build_lora(self.root, hf_repo="artist/model", hf_revision="main", hf_file="model.safetensors",
                       **self.options)


if __name__ == "__main__":
    unittest.main()
