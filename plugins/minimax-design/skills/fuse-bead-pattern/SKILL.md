---
name: fuse-bead-pattern
description: Turn user-provided photos, character art, sprites, or reference images into buildable fuse-bead patterns (拼豆、拼豆图纸、熔珠、Perler or Hama charts). Inspect each source first; use an image-generation tool to make a recognizable chibi (Q 版) cutout when a realistic subject needs simplification, or retain clean existing art. Use deterministic local code for grids, palette counts, and bead previews.
---

# 拼豆图纸

1. **逐张分流。** 打开实际图像，不依据文件名或扩展名判断。真人照片、写实渲染和逼真的电影截图，默认先用宿主可用的图片生成工具做 Q 版；已有 Q 版、像素图、轮廓清晰的平涂插画直接走代码，不能为了统一风格而重画。混合批次逐张判断，同框双人保留在同一张图。用户指定路线优先；如果明确禁用生成模型且照片需要简化，说明像素化不等于 Q 版。没有可用图像工具时，不把直接像素化冒充 Q 版；先完成其它可处理的图，指出缺哪几张底稿。
2. **做 Q 版底稿。** 每张用自己的原图作参考。保留人物数量、身份特征、表情、发型、衣服、动作、相对位置和关键饰物。简化成大头、清晰外轮廓、连续平涂色块、有限阴影的 Q 版插画；用透明背景，不加字、网格、颗粒、额外道具或珠子。先查看结果，确认透明区和关键元素正确，再进入制图；生成服务只负责底稿，不能负责画格子。默认交付原稿供比对。
3. **确定取景。** 以正方形为默认，靠近主体而不是用整幅背景。优先保留头、手和辨识度高的道具，不拉伸主体；透明素材用 alpha 边界等比缩放并留约 3 格，默认保留和主体分离的小饰物及空洞。已有不透明插画先选择能保留主体的裁切矩形，必要时做显式透明蒙版；不能把所有近白像素当背景。没有蒙版的不透明素材会视为每格都要摆豆。用户给了尺寸则按该尺寸。
4. **本地生成。** 用随包脚本统一量化、编号和渲染。默认 87×87，一格一颗，每 29 格粗线分板，批内共享最多 26 色。默认色号是这批图的内部编号，不声称等于品牌料号。要使用实物色号，输入用户确认的品牌 RGB/HEX 对照表，而不是猜映射。多个底稿在同一次运行输入以共用编号；逐个保存原稿，不覆盖原文件。

   ```bash
   uv run --script scripts/render_pattern.py --out ./beads a-chibi.png b-illustration.png
   # 为单张不透明插画指定裁切：左、上、右、下（EXIF 转正后的像素坐标）。
   uv run --script scripts/render_pattern.py --out ./beads --crop 80,140,980,1040 art.png
   ```

   不用 uv 时，使用已装 Pillow 和 NumPy 的 Python 3.11+ 运行。可用 `--size`、`--board`、`--colors` 调整格数和调色盘大小；`--palette FILE` 接受已有的 `code,hex` CSV（每个色号 1–4 个字母、数字或连字符），并按表保留色号。输出里 `chart.png` 是带坐标和用量图例的摆放图，`preview.png` 是同一矩阵的透明实物效果预览，`grid.csv` 是精确格阵（空白为不摆），`palette.csv` 是本图用量和 RGB 对照，`summary.json` 是批次汇总。每张位于编号的独立子目录。
5. **复核后交付。** 实际打开每张预览和图纸；确认左右人物、眉眼、手势、饰物、红叉/绿勾等小符号没有消失或变形，轮廓无背景噪点，图例和计数可读。必要时回到 Q 版底稿修正，或在代码输入中用裁切/透明蒙版保关键元素后重跑，不手改 PNG 图纸。每图的非空格数必须等于用量总和，预览和图纸必须来自同一格阵。逐张交付 Q 版原稿（如生成过）、图纸、预览；用户需要备料或二次排版时一起给 CSV。不要把单次素材、真实人像、产物、日志或账户信息提交到公开仓库。
