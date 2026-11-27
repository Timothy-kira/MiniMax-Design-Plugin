"""Integration tests for the bundled fuse-bead CLI; use synthetic images only."""
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

PLUGIN = Path(__file__).resolve().parent
SCRIPT = PLUGIN / "skills/fuse-bead-pattern/scripts/render_pattern.py"


class FuseBeadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "out"

    def run_cli(self, *args, out=None, succeeds=True):
        result = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out or self.out),
                                 *map(str, args)], capture_output=True, text=True)
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def verify_outputs(self, out, n, i):
        sub = out / f"{i:02}"
        with (sub / "grid.csv").open(newline="") as f:
            grid = list(csv.reader(f))
        self.assertEqual(len(grid), n)
        self.assertTrue(all(len(row) == n for row in grid))
        grid_counts = Counter(code for row in grid for code in row if code)
        with (sub / "palette.csv").open(newline="") as f:
            self.assertEqual(grid_counts, {row["code"]:int(row["count"]) for row in csv.DictReader(f)})
        with Image.open(sub / "preview.png") as im:
            for y, row in enumerate(grid):
                for x, code in enumerate(row):
                    self.assertEqual(im.getpixel((12*x+5, 12*y+5))[3], 255 if code else 0)
        with Image.open(sub / "chart.png") as im:
            im.verify()
        return grid

    def test_batch_shared_palette_and_determinism(self):
        sources = []
        for i in range(2):
            im = Image.new("RGBA", (100, 100))
            d = ImageDraw.Draw(im)
            d.rectangle((20, 20, 79, 80), fill="navy")
            d.rectangle((44, 45, 55, 55), fill="white" if i == 0 else "orange")
            sources.append(self.root / f"{i}.png")
            im.save(sources[-1])
        self.run_cli(*sources)
        other = self.root / "other"
        self.run_cli(*sources, out=other)
        stats = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(stats["board"], 29)
        for index in (1, 2):
            grid = self.verify_outputs(self.out, 87, index)
            self.assertEqual(grid, self.verify_outputs(other, 87, index))
            self.assertEqual(grid[0][0], "")
            self.assertEqual(stats["palette"][grid[30][30]], "#000080")
        self.assertEqual((self.out / "summary.json").read_bytes(), (other / "summary.json").read_bytes())
        self.run_cli(*sources, succeeds=False)

    def test_explicit_brand_codes_and_disconnected_detail(self):
        art = Image.new("RGBA", (100, 100))
        draw = ImageDraw.Draw(art)
        draw.rectangle((15, 15, 65, 80), fill="black")
        draw.rectangle((27, 30, 45, 52), fill=(0, 0, 0, 0))
        draw.rectangle((75, 45, 86, 55), fill="red")
        source = self.root / "cutout.png"
        art.save(source)
        palette = self.root / "colors.csv"
        palette.write_text("code,hex\nBLK,#000000\nR-12,#FF0000\n")
        self.run_cli(source, "--palette", palette, "--size", "29")
        grid = self.verify_outputs(self.out, 29, 1)
        self.assertEqual({c for row in grid for c in row}, {"", "BLK", "R-12"})
        self.assertIn("", grid[12][8:14])

    def test_opaque_crop_and_transparent_input_error(self):
        original = Image.new("RGB", (40, 20), "white")
        source = self.root / "opaque.png"
        original.save(source)
        self.run_cli(source, "--crop", "0,0,20,20", "--size", "29")
        self.assertTrue(all(all(row) for row in self.verify_outputs(self.out, 29, 1)))
        empty = self.root / "empty.png"
        Image.new("RGBA", (20, 20)).save(empty)
        result = self.run_cli(empty, out=self.root / "never", succeeds=False)
        self.assertIn("no visible pixels", result.stderr)
        self.assertFalse((self.root / "never").exists())

    def test_all_manifests_package_skill(self):
        for name in ("plugin.json", ".codex-plugin/plugin.json", ".minimax-plugin/plugin.json"):
            skills = json.loads((PLUGIN / name).read_text(encoding="utf-8"))["skills"]
            if isinstance(skills, str):
                self.assertTrue((PLUGIN / skills / "fuse-bead-pattern/SKILL.md").is_file())
            else:
                self.assertIn("skills/fuse-bead-pattern/SKILL.md", skills)


if __name__ == "__main__":
    unittest.main()
