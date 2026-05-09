# muav-core 目录说明

## 当前状态

本目录是从现有工作区中按 `core` 边界复制出来的底层工程子集。

注意：

- 这里是复制，不是搬移。
- 原工作区内容保持不变。
- 当前复制内容主要用于梳理仓库边界，还不是一个已经完全去耦的独立仓库。

## 已复制内容

### `src/`

- `planner/`
- `realflight_modules/`
- `uav_simulator/`
- `utils/`
- `CMakeLists.txt`

### 运行与辅助目录

- `shfiles/`
- `firmware/`

## 已排除内容

以下上层模块没有放入 `core`：

- `adv_module`
- `guide_module`
- `langguide`

## 当前已知情况

- 复制时排除了 `__pycache__` 一类缓存目录。
- `start/` 中的脚本已经并入 `shfiles/`。
- 部分脚本目前仍保留对原工作区结构或旧模块的引用。
- 例如某些启动脚本仍可能引用 `langguide`，这说明它们还没有完成按三仓结构重组。

## 下一步建议

后续可以继续做两类整理：

1. 保持复制内容不变，只补充文档，明确哪些目录和脚本属于 `core`
2. 在 `muav/core` 内对脚本和启动链做第二轮清理，使其更接近未来的 `muav-core` 独立仓库