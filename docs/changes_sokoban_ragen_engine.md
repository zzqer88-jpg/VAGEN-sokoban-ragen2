# 改动记录：Sokoban 引擎替换为 RAGEN-2 实现

- 日期：2026-09-08
- 来源：RAGEN（github.com/mll-lab-nu/RAGEN，`ragen/env/sokoban/`，main @ d97bb32，RAGEN-2 版本）
- 方案：**保留 VAGEN 外壳，只换引擎**（房间生成 + 文本渲染）。奖励、提示词/解析、state reward、异步接口、registry、trainer 均未改动，训练代码零改动。

## 背景与动机

两边底层都是 `gym_sokoban.envs.sokoban_env.SokobanEnv`（相同的 room_state 编码、相同的动作整数 1-4=上下左右）。RAGEN-2 引擎相对原 `PatchedSokobanEnv`（调 `gym_sokoban.envs.room_utils.generate_room`）的本质区别：

1. **房间生成器**：反向搜索深度可配（`search_depth`，RAGEN 默认 300 vs gym_sokoban 硬编码 100），且生成后调用 `add_random_player_movement` —— 修复"玩家总是贴着箱子"的伪影（RAGEN-2 特有）。
2. **观测格式**：新增 `coord`（纯坐标列表）与 `grid_coord`（坐标 + 紧凑网格）文本格式。

## 文件清单

### 新增
| 文件 | 内容 |
|---|---|
| `vagen/envs/sokoban/ragen_engine/utils.py` | RAGEN vendored：`generate_room`（含 `search_depth`、`add_random_player_movement`）、`reverse_playing`/`depth_first_search`/`reverse_move` 等反向搜索、`collect_entity_coordinates`/`format_coordinate_render` 坐标渲染。已剥离 matplotlib 依赖与重复 BFS。 |
| `vagen/envs/sokoban/ragen_engine/env.py` | `RagenSokobanEngine(gym_sokoban SokobanEnv)`：RAGEN 生成器 + `grid/coord/grid_coord` 渲染，叠加从旧 patch 原样迁入的 LCG 种子重试（`_next_retry_seed`）、sha256 地图划分（`_room_partition_bucket`/`_room_matches_partition`）、BFS 难度门控（`get_shortest_action_path`）。 |
| `vagen/envs/sokoban/ragen_engine/__init__.py` | 包导出。 |
| `tests/test_sokoban_ragen_engine.py` | 8 个测试：同 seed 确定性、房间可解、玩家-箱子非邻接（验证随机移动生效）、难度带、三种文本格式渲染、`search_depth` 透传。 |
| `tools/regen_sokoban_val_seeds.py` | 验证集种子清单重建工具（见下）。 |

### 修改
| 文件 | 改动 |
|---|---|
| `vagen/envs/sokoban/sokoban_env.py` | import 换为 `RagenSokobanEngine`；`SokobanEnvConfig` 新增 `search_depth: int = 300`、`observation_format: str = "grid"`（校验 ∈ {grid, coord, grid_coord}）；构造引擎透传 `search_depth`；`_render_async` 文本分支按 `observation_format` 分派（默认 `grid` 保持 VAGEN 带空格格式，与提示词/已训模型兼容）；`__main__` 冒烟 CLI 加 `search_depth` 参数。 |
| `tests/test_sokoban_seed_retries.py` | import 路径 → `ragen_engine`（LCG 数值断言不变）。 |
| `tests/test_sokoban_map_partition.py` | `generate_room` 改从 vendored 包导入（调用加 `search_depth=300`，解包 4 元组）；helpers import 路径同步改。 |
| `examples/train/sokoban/val_sokoban_vision.yaml`<br>`examples/train/sokoban/val_sokoban_vision_sr.yaml` | `seed_list` 重建（两文件清单一致，测试要求）。 |
| `docs/configuration.md` | sokoban 配置表新增 `search_depth`、`observation_format` 两行及生成器说明。 |

### 删除
- `vagen/envs/sokoban/patch_sokoban_env.py`（被 `ragen_engine` 取代；全仓库引用仅 3 处，均已更新）

## 种子清单重建

RAGEN 生成器产出不同地图，原 256-seed 验证清单失效。已用工具重建：

```bash
python tools/regen_sokoban_val_seeds.py \
    examples/train/sokoban/val_sokoban_vision.yaml \
    examples/train/sokoban/val_sokoban_vision_sr.yaml
# Accepted 256 seeds after trying 1714 candidates ([10002 .. 11714])
```

入选条件：**首次生成**（无重试走查）即满足 BFS 最优解 ∈ `min_solution_steps` 带（[1,5]）、sha256 落入 eval bucket（mod 4, bucket 0）、房间唯一。以后改动任何生成相关配置（`search_depth`/`num_boxes`/`dim_room`/`min_solution_steps`/partition 参数）后需重跑此工具。

## 与 RAGEN 原版的有意差异（已写入代码注释）

1. **种子重试**：用确定性 LCG `_next_retry_seed` 而非 RAGEN 的 `abs(hash(str(seed))) % 2**32` —— Python hash 有盐，跨 Ray worker 不可复现（本仓库为此专门修过）。
2. **播种**：用 `vagen/envs/sokoban/utils/seeding.set_seed`（seed `random` + `numpy.random`，即 `generate_room` 实际用到的两个 RNG），而非 `ragen.utils.all_seed`（多 seed torch）。
3. **grid 文本格式**：默认观测仍用 VAGEN 带空格版（` #  P  X ...`）；RAGEN 的紧凑无空格版只在 `render("grid")`/`grid_coord` 内嵌网格中出现。
4. **奖励**：保持 VAGEN 稀疏 success（+1.0）+ format（+0.1/回合），继续丢弃 gym_sokoban 每步原生 shaping（RAGEN 保留它）——与已有实验可比。

## 验证结果（2026-09-08，Windows / Anaconda base）

- `tests/test_sokoban_ragen_engine.py`：8 passed
- `tests/test_sokoban_seed_retries.py`：2 passed
- `tests/test_sokoban_map_partition.py`：11 passed（含 256-seed manifest 校验）
- 端到端冒烟：外层 `Sokoban` 用 BFS 最优解驱动 3 个 seed 全部通关，总奖励精确 = 1.0 + 0.1×回合数；`coord`/`grid_coord`/`grid` 三种观测格式输出正确
- 全量套件：678 passed / 37 failed，失败全部源于本机缺 torch/verl/ray/vllm（训练栈，与本次改动无关）
- 注意：本机跑测试需 `PYTHONUTF8=1`（GBK locale 会导致读 UTF-8 文件假失败）
