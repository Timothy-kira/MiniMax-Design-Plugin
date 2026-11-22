---
name: design-character-lookup
description: Resolve unfamiliar character nicknames, aliases, or shorthand against MiniMax Design canvas character assets before writing or revising storyboards, prompts, or reference assignments, regardless of the video model.
---

# 查找画布角色与素材

1. 用户使用了尚未确认英文名的角色代称时，先用 `ensure_design` 核对连接，打开对应项目。以用户的原话搜索画布角色资产。完整代称无结果时，再用有辨识度的共同词干搜索。优先核对角色标注中的「外号」「别名」和角色名称。
2. 对匹配的角色读取 Composition 详情，用画布标注确定代称与英文名的对应关系，并取出本次实际需要的角色参考版本。存在多个同名结果时，用项目、来源和角色关系核对。画布仍无法消歧时，只询问尚未确定的对应关系。
3. 分镜、提示词和素材职责统一使用已核实的英文名。按实际引用顺序给素材连续编号，每张图注明负责哪个角色或场景、参考哪些特征。沿用用户已明确指定的角色或素材版本。
