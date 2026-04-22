# ABB_IRB 项目记录

## 1. 项目定位

本目录用于承接原 `xArm` 版本工程在 `ABB_IRB` 机械臂上的迁移、重建与扩展工作。

当前研究重点不是直接复现真实产线控制器，而是围绕以下主线建立一套完整、可验证、可扩展的研究链路：

1. 机械臂运动学建模
2. 正运动学 FK 与雅可比建模
3. 基于神经网络的逆运动学近似求解
4. Newton-Raphson / DLS 等数值法校正与对比
5. 开放空间与固定障碍场景下的碰撞检测和轨迹验证
6. 后续面向列车车底受限环境的避障与可视化扩展

本文件作为当前 ABB 工程的总记录页，后续所有关键思路、参数确认、代码实现、实验命令、运行结果和结论，统一追加在本文件中，便于后续上传 GitHub 后直接查看项目进展。

---

## 2. 当前目录约定

后续所有与 ABB 机械臂相关的代码、数据、配置和实验产物，统一保存在：

`E:\CSU\毕业设计\ABB_Arm_Control`

当前已建立的配置文档：

- [docs/ABB_IRB_当前配置草案.md](E:\CSU\毕业设计\ABB_Arm_Control\docs\ABB_IRB_当前配置草案.md)

说明：

1. `README.md` 负责记录整体思路、阶段性结论、命令、结果与路线。
2. `docs/` 负责保存参考资料、参数草案、说明文档等。
3. 后续如需新建代码子目录，将在本目录下按功能划分建立，例如 `kinematics/`、`ik/`、`training/`、`artifacts/` 等。

---

## 3. 当前已确认的基础配置

### 3.1 机器人型号

- 工程内部统一命名：`ABB_IRB`
- 当前对应具体型号：`ABB IRB 1200-7/0.7`

说明：

1. 工程命名层面统一写作 `ABB_IRB`，避免未来文件名和代码变量过长。
2. 资料核对、参数来源和尺寸验证阶段仍然明确对应 `IRB 1200-7/0.7`。

### 3.2 末端工具假设

当前阶段默认不挂载额外工具，末端执行器视为法兰中心，采用：

$$
{}^{\text{flange}}\mathbf{T}_{\text{tool}}=\mathbf{I}
$$

因此当前研究中的末端位姿等价于法兰位姿。

### 3.3 位姿表达形式

当前保持与原 xArm 工程一致，采用 6 维位姿表达：

$$
\mathbf{x}=[x,\ y,\ z,\ \phi,\ \theta,\ \psi]^\top
$$

其中：

1. 位置单位为 `mm`
2. 姿态采用 `ZYX Euler`
3. 姿态角单位为 `rad`

---

## 4. 关节范围与当前建模范围

依据 ABB 官方规格书，当前采用的关节范围为：

- `q1 ∈ [-170°, 170°]`
- `q2 ∈ [-100°, 135°]`
- `q3 ∈ [-200°, 70°]`
- `q4 ∈ [-270°, 270°]`
- `q5 ∈ [-130°, 130°]`
- `q6` 官方默认范围为 `[-400°, 400°]`

当前工程为了保持训练与分类系统简洁，暂定将第 6 轴建模范围压缩为：

- `q6 ∈ [-180°, 180°]`

说明：

1. 这不会影响当前阶段的 FK、IK、NN、NR、DLS、碰撞检测流程建立。
2. 后续若研究连续回转、最短转角或电缆绕线，再恢复更大的轴 6 范围。

---

## 5. 当前采用的运动学建模结论

### 5.1 建模方式

当前建议冻结为：

- `Standard DH`
- 长度单位：`mm`
- 外部关节输入单位：`deg`
- 三角函数内部计算单位：`rad`

选择原因：

1. 与原 `ref22_reproduction` 项目结构兼容性最高。
2. 后续迁移 FK、Jacobian、NR、DLS、训练与推理脚本时改动最小。
3. 更适合在当前阶段快速验证整条工程链路。

### 5.2 当前采用的 Standard DH 参数

当前以 `IRB 1200-7/0.7` 标准版为对象，建议采用如下参数作为第一版运动学基线：

| 关节 i | a_i (mm) | alpha_i (deg) | d_i (mm) | theta_i |
|---|---:|---:|---:|---|
| 1 | 0 | -90 | 399.1 | q1 |
| 2 | 350 | 0 | 0 | q2 - 90° |
| 3 | 42 | -90 | 0 | q3 |
| 4 | 0 | 90 | 351.0 | q4 |
| 5 | 0 | -90 | 0 | q5 |
| 6 | 0 | 0 | 82.0 | q6 |

对应的内部偏置写法为：

$$
\theta_1=q_1,\quad
\theta_2=q_2-90^\circ,\quad
\theta_3=q_3,\quad
\theta_4=q_4,\quad
\theta_5=q_5,\quad
\theta_6=q_6
$$

---

## 6. 关于 `theta2_offset = -90°` 的当前理解

这是当前建模中最关键的知识点之一。

### 6.1 它不表示关节 2 的机械限位变化

这里必须区分两类角度：

1. `q_i`：机械臂物理关节角，也是训练、控制、限位判断时使用的角
2. `theta_i`：DH 变换矩阵中实际使用的建模角

因此：

$$
\theta_2=q_2-90^\circ
$$

并不意味着关节 2 的物理零位被改掉了，而是说明在当前这套 `Standard DH` 建系方式下，DH 第 2 轴参考角与机械零位之间存在一个固定偏置。

### 6.2 为什么需要这个偏置

如果直接令：

$$
\theta_2=q_2
$$

则当所有关节角都取零时，机械臂的几何姿态会偏离 ABB 官方给出的典型 `L` 型零位，说明当前坐标系定义与实际机械零位不一致。

若采用：

$$
\theta_2=q_2-90^\circ
$$

则零位姿态下的骨架结构与 ABB 官方尺寸图和工作范围图能够保持一致。

### 6.3 数值校核结论

当前已根据 ABB 官方工作范围图对 `机械腕中心` 位置进行了数值反校验。

采用上述 `Standard DH + theta2_offset=-90°` 时，模型计算结果与官方表格的典型位置误差约为亚毫米量级，说明该建模方案当前是自洽的。

已核对的典型点包括：

1. `Pos0`
2. `Pos3`
3. `Pos6`
4. `Pos7`
5. `Pos9`

当前判断：

- `a/d` 数值可信
- `theta2_offset=-90°` 合理且必要
- 可作为 ABB 工程第一阶段的 FK 核心配置

---

## 7. 参数来源与资料使用说明

### 7.1 当前重点参考资料

- `docs/IRB1200产品规格.pdf`
- `docs/IRB1200数据表.pdf`
- `docs/IRB1200产品概述.pdf`
- `docs/DH参数参考.txt`
- `docs/结构参数图1.png`
- `docs/结构参数图2.png`
- `docs/改进IRB1200DH参数.png`

### 7.2 当前对资料可信度的判断

1. `IRB1200产品规格.pdf` 与 `IRB1200数据表.pdf` 为官方资料，优先级最高。
2. `IRB1200产品概述.pdf` 可用于辅助确认尺寸量级。
3. `DH参数参考.txt` 与 `改进IRB1200DH参数.png` 主要用于辅助构建 DH 表和理解建系方式，不能替代官方资料。
4. `结构参数图2.png` 对应 Hygienic 版本，末端尺寸与标准版存在差异，不能直接拿来替换当前 `d6=82 mm`。

---

## 8. 当前建议采用的子空间划分方案

为了尽量平滑迁移原 xArm 工程结构，当前先保留两套 ABB 子空间配置。

### 8.1 `abb_simplified`

- `q1`: 2 段
- `q2`: 2 段
- `q3`: 2 段
- `q4`: 2 段
- `q5`: 3 段
- `q6`: 2 段

总数：

$$
2\times2\times2\times2\times3\times2=96
$$

### 8.2 `abb_strict`

- `q1`: 2 段
- `q2`: 2 段
- `q3`: 2 段
- `q4`: 4 段
- `q5`: 3 段
- `q6`: 2 段

总数：

$$
2\times2\times2\times4\times3\times2=192
$$

当前建议：

1. 第一阶段优先使用 `abb_simplified`
2. 待 FK、数据链路和推理链路跑通后，再视精度决定是否升级到 `abb_strict`

---

## 9. 当前迁移策略

当前不建议直接在原 `ref22_reproduction` 上硬改。

建议策略：

1. 保留 `ref22_reproduction` 作为 xArm 已完成版本
2. 在 `ABB_Arm_Control` 中单独建立 ABB 工程
3. 先实现 ABB 版运动学核心
4. 再迁移训练、推理、对比和避障模块

原因：

1. 可以保留原工程作为可回溯基线
2. 避免 xArm 与 ABB 参数混用
3. 便于后续在 GitHub 上清晰展示“平台迁移”的过程

---

## 10. 当前建议的实施顺序

### 阶段 1：FK 核心建立与验证

目标：

1. 实现 ABB 版 `fk_model`
2. 实现关节骨架点输出
3. 实现位姿转换、雅可比计算等基础功能
4. 验证零位姿态与典型工作点

这是当前最优先的工作，因为如果 FK 错了，后续数据集、NN、NR、DLS 和碰撞检测都会整体失效。

### 阶段 2：关节分段系统建立

目标：

1. 实现 `abb_simplified`
2. 实现 `abb_strict`
3. 建立子空间编号、解码、采样逻辑

### 阶段 3：数据生成与训练链路迁移

目标：

1. 生成 ABB 数据集
2. 训练 prediction system
3. 训练 classification system
4. 跑通 `predict_ik + NR`

### 阶段 4：传统数值法与基准对比

目标：

1. 实现 DLS 或其他数值法
2. 对比 `NN`、`NN+NR`、`DLS`
3. 统计时间、误差、成功率

### 阶段 5：避障与可视化迁移

目标：

1. 在开放空间中验证 ABB 机械臂避障
2. 建立固定障碍场景
3. 输出碰撞/无碰撞轨迹图像与视频
4. 逐步过渡到车底受限环境建模

---

## 11. 后续 README 记录规范

从现在开始，本文件将持续追加以下内容：

1. 关键参数变更
2. 代码目录结构变更
3. 核心脚本功能说明
4. 运行命令
5. 运行结果
6. 图表、误差分析与阶段性结论

建议记录格式：

- 日期
- 操作内容
- 命令
- 输出结果
- 结论
- 下一步

---

## 12. 当前阶段性结论

截至目前，ABB 版本已经完成“参数与建模前提确认 + FK 核心第一版实现 + 官方工作点校核 + 子空间模块 + 数据生成链路 + prediction system smoke test + classification system smoke test + `predict_ik + NR` 端到端 smoke test”，已经正式进入代码实现阶段。

当前可以认为已经基本冻结的内容如下：

1. 工程命名：`ABB_IRB`
2. 对应型号：`IRB 1200-7/0.7`
3. 建模方式：`Standard DH`
4. 关节范围：
   - `q1 [-170,170]`
   - `q2 [-100,135]`
   - `q3 [-200,70]`
   - `q4 [-270,270]`
   - `q5 [-130,130]`
   - `q6 [-180,180]`（当前项目建模范围）
5. DH 参数：
   - `a = [0, 350, 42, 0, 0, 0]`
   - `alpha = [-90, 0, -90, 90, -90, 0]`
   - `d = [399.1, 0, 0, 351, 0, 82]`
   - `theta2_offset = -90°`
6. 位姿表达：`[x, y, z, phi, theta, psi]`
7. 子空间方案：
   - `abb_simplified = 96`
   - `abb_strict = 192`

---

## 13. 当前待执行的下一步

下一步优先做以下内容：

1. 进行一轮更合理规模的 `prediction + classification` 训练
2. 用正式训练产物做第一次可参考的 ABB 推理实验
3. 再决定先做 benchmark 还是先做避障迁移
4. 之后再考虑将 DLS/L-BFGS-B 等传统数值法迁移到 ABB 版本

当前不建议跳过数据链路验证而直接开始大规模训练。

---

## 14. 实验记录

### 2026-04-22

#### 已完成

1. 确定工程统一命名为 `ABB_IRB`
2. 确定研究对象为 `IRB 1200-7/0.7`
3. 确定当前不挂工具，末端视为法兰中心
4. 整理并记录当前配置草案文档
5. 基于官方资料与尺寸图确认第一版 `a/d` 参数
6. 通过工作范围典型点反校验，确认 `theta2_offset=-90°` 当前成立

#### 当前结论

可以进入代码实现阶段，但必须先做 FK 核心验证，再进行数据生成与训练。

#### 备注

后续每次代码实现、运行命令、图像输出、训练结果和分析结论，统一继续追加在本节之后。

### 2026-04-22 - FK 核心第一版实现

#### 本轮新增代码

1. `robot_config.py`
   - 统一保存 ABB_IRB 当前阶段的机器人参数
   - 包括 DH 参数、关节限位、`theta2_offset` 和官方腕中心参考点
2. `fk_model.py`
   - 实现 `numpy` 版 FK
   - 实现关节骨架点输出
   - 实现 `pose6` 输出
   - 实现 `torch` 批量 FK 接口
   - 实现数值雅可比接口
3. `scripts/validate_fk_model.py`
   - 对零位骨架点进行导出
   - 对 ABB 官方工作范围图中的典型腕中心点做误差校核
   - 比较不同 `theta2_offset` 假设的误差

#### 当前目录结构

当前 ABB 工程最小骨架如下：

```text
ABB_Arm_Control/
├─ docs/
├─ scripts/
│  └─ validate_fk_model.py
├─ robot_config.py
├─ fk_model.py
└─ README.md
```

#### 本轮运行环境

- 虚拟环境：`arm_nn`
- 工作目录：`E:\CSU\毕业设计\ABB_Arm_Control`

#### 本轮运行命令

```powershell
conda activate arm_nn
python -X utf8 scripts/validate_fk_model.py
```

#### 本轮输出结果

生成报告：

- `artifacts/fk_validation/fk_validation_report.json`

终端关键输出：

```text
theta2_offset_deg=-90.0
mean_xz_error_mm=0.413227
max_xz_error_mm=0.575172
[Pos0] err=0.100000 mm
[Pos3] err=0.502328 mm
[Pos6] err=0.351964 mm
[Pos7] err=0.536669 mm
[Pos9] err=0.575172 mm
```

不同 `theta2_offset` 假设对比：

| theta2_offset (deg) | mean_xz_error_mm | max_xz_error_mm |
|---:|---:|---:|
| -90 | 0.413227 | 0.575172 |
| 0 | 697.198521 | 994.681593 |
| 90 | 985.806420 | 1406.503915 |

零位关节骨架点结果：

| 点位 | 坐标 (mm) |
|---|---|
| base | `(0.0, 0.0, 0.0)` |
| joint1 | `(0.0, 0.0, 399.1)` |
| joint2 | `(0.0, 0.0, 749.1)` |
| joint3 | `(0.0, 0.0, 791.1)` |
| joint4 / wrist center | `(351.0, 0.0, 791.1)` |
| joint5 | `(351.0, 0.0, 791.1)` |
| joint6 / end-effector | `(433.0, 0.0, 791.1)` |

#### 本轮结论

1. 当前 `Standard DH` 参数组已经可用于 ABB 工程第一阶段实现。
2. `theta2_offset = -90°` 不是经验性修补项，而是当前坐标系定义下的必要偏置。
3. 用 ABB 官方工作范围图的腕中心点进行验证后，当前模型误差已降至亚毫米量级。
4. ABB 版本现在可以继续向“子空间划分 -> 数据生成 -> 小样本训练链路验证”推进。

### 2026-04-22 - 子空间模块与数据生成链路

#### 本轮新增代码

1. `abb_nn/__init__.py`
2. `abb_nn/subspace.py`
   - 新增 `abb_simplified`
   - 新增 `abb_strict`
   - 支持子空间计数、编码、解码、采样
3. `naming.py`
4. `naming_config.json`
5. `generate_dataset.py`
   - 支持 ABB 随机关节采样
   - 支持 CUDA 批量 FK 生成
   - 保存 `csv / npz / meta`
6. `scripts/validate_subspaces.py`
   - 输出子空间 profile 检查结果

#### 本轮运行命令

```powershell
conda activate arm_nn
python -X utf8 scripts/validate_subspaces.py
python -X utf8 generate_dataset.py --n_samples 512 --seed 2026 --out_dir data --feature_batch_size 512 --overwrite
```

#### 本轮输出结果

子空间校核输出：

```text
[abb_simplified] bins=2 x 2 x 2 x 2 x 3 x 2 => subspaces=96
[abb_strict] bins=2 x 2 x 2 x 4 x 3 x 2 => subspaces=192
```

对应文件：

- `artifacts/subspace_validation/subspace_profiles.json`

数据生成输出：

```text
Robot: ABB_IRB (IRB 1200-7/0.7)
Base name: abb_irb_fk_uniform_random_512_seed2026
Generated samples: 512
Feature device: cuda
```

生成文件：

- `data/abb_irb_fk_uniform_random_512_seed2026_full.csv`
- `data/abb_irb_fk_uniform_random_512_seed2026_full.npz`
- `data/abb_irb_fk_uniform_random_512_seed2026_full_meta.json`

生成数据结构：

| 键 | 形状 | 类型 |
|---|---|---|
| `q_deg` | `(512, 6)` | `float32` |
| `position_mm` | `(512, 3)` | `float32` |
| `rotation` | `(512, 3, 3)` | `float32` |
| `euler_rad_zyx` | `(512, 3)` | `float32` |
| `T06` | `(512, 4, 4)` | `float32` |

该数据集的抽样范围已覆盖当前项目关节限位，例如：

- `q_min ≈ [-168.83, -99.67, -198.54, -269.05, -129.53, -179.79]`
- `q_max ≈ [169.47, 134.99, 69.85, 269.22, 129.94, 179.78]`

#### 本轮结论

1. ABB 版子空间划分已经正式落地，并与先前讨论的 `96 / 192` 设计一致。
2. ABB 版数据生成链路已经可用，且已成功调用 CUDA 做批量 FK。
3. 当前可以继续进入 prediction system 的小规模训练验证。

### 2026-04-22 - Prediction System Smoke Test

#### 本轮新增代码

1. `abb_nn/data_utils.py`
2. `abb_nn/models.py`
3. `train_prediction_models.py`
   - 已迁移为 ABB 版 segmented prediction training 脚本
   - 当前支持：
     - `abb_simplified / abb_strict`
     - 子空间采样训练
     - 全局 normalizer
     - `q1-5` 与 `q6` 分开建模
     - `e_max` 与测试位姿位置误差统计

#### 本轮运行命令

```powershell
conda activate arm_nn
python -X utf8 train_prediction_models.py --segment_profile abb_simplified --subspaces 0,95 --samples_per_subspace 64 --epochs 5 --batch_size 32 --hidden_layers 2 --neurons_per_layer 16 --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --normalizer_samples 512 --feature_batch_size 256 --num_workers 0 --out_dir artifacts/prediction_system_smoke
```

#### 本轮输出结果

```text
[subspace 000] train=44 val=9 mse(q1-5)=12870.0508 mse(q6)=11913.7305 e_max=1109.802368
[subspace 095] train=44 val=9 mse(q1-5)=10525.0938 mse(q6)=13942.5225 e_max=1187.886963
Saved metadata: artifacts\prediction_system_smoke\metadata.json
Trained subspaces: 2
```

对应文件：

- `artifacts/prediction_system_smoke/metadata.json`
- `artifacts/prediction_system_smoke/subspace_models/subspace_000.pt`
- `artifacts/prediction_system_smoke/subspace_models/subspace_095.pt`

当前 smoke test 元数据已正确记录：

1. `segment_profile = abb_simplified`
2. `subspace_count = 96`
3. `theta_offsets_deg = [0, -90, 0, 0, 0, 0]`
4. 已训练子空间数为 `2`
5. normalizer 与超参数均已写入 `metadata.json`

#### 本轮结果解释

1. 当前数值很大，这本身不代表脚本有问题。
2. 这是一个故意压缩到极小规模的 smoke test：
   - 每个子空间只有 `64` 个样本
   - 只有 `5` 个 epoch
   - 只训练了 `2` 个子空间
3. 因此这里的意义不是追求精度，而是验证：
   - 采样是否正常
   - FK 特征构建是否正常
   - normalizer 是否正常
   - 子模型训练与保存是否正常
   - 元数据格式是否可供后续推理脚本继续使用

#### 本轮结论

1. ABB 版 prediction training 链路已经最小闭环跑通。
2. 当前可以继续迁移 classification system。
3. 在 classification 与 inference 脚本迁移完成前，不建议直接开展大规模精度实验。

### 2026-04-22 - Classification System 设计与 Smoke Test

#### 设计思路

当前 ABB 版 `classification system` 的目标不是直接输出关节角，而是先回答：

> 给定一个目标末端位姿，它更可能属于哪一个关节子空间？

这一步的作用是为后续 `prediction system` 提供候选子空间，从而避免在推理时遍历全部 `96` 或 `192` 个子空间模型。

当前采用的分类数据构造方式为：

1. 在全局关节范围内均匀采样关节角 `q`
2. 通过 FK 计算对应位姿 `x = [x, y, z, phi, theta, psi]`
3. 通过子空间划分规则，将 `q` 映射为类别标签 `y = subspace_id`
4. 用样本对 `(x, y)` 训练分类器

因此分类器学习的是：

$$
\mathbf{x} \rightarrow s
$$

其中：

- `\mathbf{x}` 为 6 维末端位姿特征
- `s` 为子空间编号

#### 当前分类器架构

当前沿用之前 Ref[22] 复刻版中的 3 种分类器变体：

1. `v1`
   - 普通深层 MLP
   - 宽度 `35`
   - 深度 `6`
2. `v2`
   - 更深的残差式 MLP
   - 宽度 `35`
   - 深度 `20`
3. `v3`
   - 更深的残差式 MLP + BatchNorm
   - 宽度 `35`
   - 深度 `30`

在 `abb_simplified` 配置下：

- 输入维度：`6`
- 输出类别数：`96`

在 `abb_strict` 配置下：

- 输入维度：`6`
- 输出类别数：`192`

#### 本轮新增代码

1. `abb_nn/models.py`
   - 补充 `ResidualBlock`
   - 补充 `ClassifierMLP`
   - 补充 `build_classifier_variant`
2. `abb_nn/__init__.py`
   - 导出分类模型与数据工具
3. `train_classification_models.py`
   - 已迁移为 ABB 版 classification training 脚本
   - 支持 `abb_simplified / abb_strict`
   - 支持三种分类器版本的训练与保存

#### 本轮运行命令

```powershell
conda activate arm_nn
python -X utf8 train_classification_models.py --segment_profile abb_simplified --trainset_v1 256 --trainset_v2 384 --trainset_v3 320 --val_samples 128 --epochs 3 --batch_size 64 --feature_batch_size 256 --num_workers 0 --out_dir artifacts/classification_system_smoke
```

#### 本轮输出结果

```text
[classifier v1] train=256 val=128 val_loss=4.578125 val_acc=0.0156
[classifier v2] train=384 val=128 val_loss=4.550781 val_acc=0.0078
[classifier v3] train=320 val=128 val_loss=4.751953 val_acc=0.0000
Saved metadata: artifacts\classification_system_smoke\metadata.json
```

生成文件：

- `artifacts/classification_system_smoke/classifier_v1.pt`
- `artifacts/classification_system_smoke/classifier_v2.pt`
- `artifacts/classification_system_smoke/classifier_v3.pt`
- `artifacts/classification_system_smoke/metadata.json`

当前 `metadata.json` 已正确记录：

1. `segment_profile = abb_simplified`
2. `num_classes = 96`
3. `theta_offsets_deg = [0, -90, 0, 0, 0, 0]`
4. 3 个分类器版本对应的训练样本数、验证损失和验证精度
5. global normalizer

#### 本轮结果解释

1. 当前精度极低，这本身是符合预期的。
2. 原因不是脚本错误，而是当前 smoke test 被刻意压缩得非常小：
   - `trainset_v1 = 256`
   - `trainset_v2 = 384`
   - `trainset_v3 = 320`
   - `val_samples = 128`
   - `epochs = 3`
3. 对于 `96` 类问题，这种样本量和训练轮数几乎只够验证代码链路是否打通，不足以评价模型性能。
4. 这一步的意义在于确认：
   - 全局随机采样正常
   - FK 特征构造正常
   - `pose6 -> subspace_id` 标签生成正常
   - 3 个分类器版本都能训练和保存
   - metadata 格式可供后续推理脚本直接使用

#### 本轮结论

1. ABB 版 classification system 已经最小闭环跑通。
2. 现在 ABB 工程已经具备：
   - FK 核心
   - 子空间系统
   - 数据生成
   - prediction training
   - classification training
3. 下一步可以进入 `predict_ik.py + NR` 迁移，建立 ABB 版第一次完整推理闭环。

### 2026-04-22 - `predict_ik + NR` 端到端 Smoke Test

#### 设计思路

当前 ABB 版推理流程沿用之前 Ref[22] 复刻版的总体结构：

1. 输入目标末端位姿 `pose6`
2. 使用 `classification system` 预测候选子空间
3. 在候选子空间中调用对应的 `prediction system` 子模型
4. 用末端位置误差 `position_l2_mm` 选择更优初值
5. 若需要，则再用 `Newton-Raphson / DLS-style` 局部校正

即：

$$
\mathbf{x}_{target}
\xrightarrow{\text{classification}}
\{s_1,s_2,\dots\}
\xrightarrow{\text{prediction}}
\mathbf{q}_0
\xrightarrow{\text{NR refine}}
\mathbf{q}^{*}
$$

#### 本轮新增代码

1. `abb_nn/optimization.py`
   - 新增 `NROptions`
   - 新增 `newton_raphson_refine(...)`
2. `predict_ik.py`
   - 新增 ABB 版推理入口
   - 支持：
     - classification 候选子空间筛选
     - prediction 子模型装载与初值生成
     - `e_max` 基础筛选
     - NR 校正
     - 计时统计
     - 可选 `--out_json`
3. `abb_nn/__init__.py`
   - 导出优化模块接口

#### 当前推理脚本的额外处理

由于当前 smoke 阶段 prediction system 只训练了部分子空间，而 classification 可能预测到未训练子空间，因此推理脚本增加了一个务实回退策略：

1. 若分类器预测的子空间中没有任何一个已训练 prediction 子模型
2. 则自动回退到“所有已训练子空间”
3. 这样可以避免 smoke 阶段因为训练覆盖不完整而直接报错

此外，在当前 Windows + conda 环境中，推理阶段出现过一次 `OpenMP duplicate runtime` 问题，因此在 `predict_ik.py` 中加入了最小兼容处理：

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

这属于当前本地运行环境兼容措施，不改变推理算法本身。

#### 本轮 smoke test 所用目标位姿

本轮测试位姿来自一组手工指定关节角的 FK 输出：

- `q_deg = [-80, 0, -100, -120, -80, -90]`

对应目标位姿为：

```text
57.581909,76.177643,1108.508537,-1.926516,-0.150956,-1.308327
```

#### 本轮运行命令

```powershell
conda activate arm_nn
python -X utf8 predict_ik.py --pose=57.581909,76.177643,1108.508537,-1.926516,-0.150956,-1.308327 --pred_meta artifacts/prediction_system_smoke/metadata.json --cls_meta artifacts/classification_system_smoke/metadata.json --enable_nr --out_json artifacts/inference_smoke/predict_ik_smoke_result.json
```

#### 本轮输出结果

结果文件：

- `artifacts/inference_smoke/predict_ik_smoke_result.json`

关键输出摘要：

```text
candidate_subspaces = [0]
candidate_source = classification_predictions
initial position_l2_mm = 495.7419
nr_iters = 40
nr_converged = false
final_pos_err_mm = 57.0463
final_ori_err_rad = 0.7198
ik_solve_time_ms ≈ 220.78
```

时间分解：

- `classification_ms ≈ 114.62`
- `initial_selection_ms ≈ 7.90`
- `nr_refinement_ms ≈ 93.01`
- `total_ms ≈ 220.78`

#### 本轮结果解释

1. 这次端到端推理已经成功完成，说明 ABB 版完整闭环已经打通：
   - classification
   - prediction
   - NR refinement
2. 当前初始误差较大，原因很直接：
   - prediction system 只训练了 `2` 个子空间
   - 每个子空间仅 `64` 个样本
   - 只训练了 `5` 个 epoch
3. 在这种极弱训练配置下，NR 仍然把位置误差从约 `495.74 mm` 压低到约 `57.05 mm`，说明：
   - 当前 FK/雅可比/NR 数值链路是有效的
   - 但 prediction 初值质量仍然远不足以支撑正式实验
4. `nr_converged = false` 也是符合预期的，因为当前 smoke model 的初值本身太差。

#### 本轮结论

1. ABB 版第一次完整逆解推理闭环已经建立完成。
2. 当前 ABB 工程已经具备：
   - FK
   - 子空间系统
   - 数据生成
   - prediction training
   - classification training
   - inference + NR refinement
3. 接下来不再缺“功能模块”，而是进入“提升训练质量与实验可信度”的阶段。

---

## 15. 云端正式训练命令

本节用于记录当前 ABB 工程在云平台上可直接执行的正式训练命令。

当前推荐优先训练方案：

1. 先使用 `abb_simplified`
2. 先完整跑完 `prediction system formal`
3. 再完整跑完 `classification system formal`
4. 最后再进行逆解推理与 NR 校正验证

说明：

1. `abb_simplified` 对应 `96` 个子空间，适合作为 ABB 版本的第一版正式实验方案。
2. 当前不建议先上 `abb_strict = 192`，因为训练成本更高，且需要先观察 `96` 子空间结果。
3. 正式训练完成后，才建议使用 `predict_ik.py` 做推理精度与时间测试。

### 15.1 云端环境建议

推荐在云端按如下顺序执行：

```powershell
conda activate arm_nn
cd /path/to/ABB_Arm_Control
```

如果云端是 Linux，也可直接执行：

```bash
source activate arm_nn
cd /path/to/ABB_Arm_Control
```

其中 `/path/to/ABB_Arm_Control` 需要替换为你在云平台上的实际项目路径。

### 15.2 Prediction System Formal 命令

该命令会对 `abb_simplified` 下的全部 `96` 个子空间进行正式训练。

```powershell
python -X utf8 train_prediction_models.py --segment_profile abb_simplified --samples_per_subspace 80000 --epochs 300 --batch_size 1024 --hidden_layers 3 --neurons_per_layer 20 --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --normalizer_samples 200000 --feature_batch_size 8192 --out_dir artifacts/prediction_system_formal
```

命令含义：

1. `--segment_profile abb_simplified`
   - 使用 ABB 简化版分段
   - 共 `96` 个子空间
2. `--samples_per_subspace 80000`
   - 每个子空间采样 `80000` 条样本
3. `--epochs 300`
   - 每个子空间训练 `300` 个 epoch
4. `--batch_size 1024`
   - 回归网络训练 batch size
5. `--hidden_layers 3`
   - 子空间回归网络隐层数为 `3`
6. `--neurons_per_layer 20`
   - 每层 `20` 个神经元
7. `--train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15`
   - 每个子空间内部按 `70/15/15` 划分训练、验证、测试集
8. `--normalizer_samples 200000`
   - 全局 normalizer 使用 `200000` 个关节样本拟合
9. `--feature_batch_size 8192`
   - FK 批量特征生成的 batch 大小
10. `--out_dir artifacts/prediction_system_formal`
   - 输出目录

### 15.3 Classification System Formal 命令

该命令会基于 ABB 全局关节空间采样结果，训练 3 个分类器版本。

```powershell
python -X utf8 train_classification_models.py --segment_profile abb_simplified --trainset_v1 300000 --trainset_v2 500000 --trainset_v3 400000 --val_samples 3000 --epochs 40 --batch_size 4096 --feature_batch_size 8192 --out_dir artifacts/classification_system_formal
```

命令含义：

1. `--segment_profile abb_simplified`
   - 分类类别数为 `96`
2. `--trainset_v1 300000`
   - 分类器 `v1` 训练样本数 `300000`
3. `--trainset_v2 500000`
   - 分类器 `v2` 训练样本数 `500000`
4. `--trainset_v3 400000`
   - 分类器 `v3` 训练样本数 `400000`
5. `--val_samples 3000`
   - 分类验证样本数 `3000`
6. `--epochs 40`
   - 分类器训练轮数 `40`
7. `--batch_size 4096`
   - 分类训练 batch size
8. `--feature_batch_size 8192`
   - FK 批量特征生成 batch 大小
9. `--out_dir artifacts/classification_system_formal`
   - 输出目录

### 15.4 正式训练完成后的推理命令模板

在上述两条正式训练命令都执行完成后，可使用如下命令进行推理：

```powershell
python -X utf8 predict_ik.py --pose=x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad --pred_meta artifacts/prediction_system_formal/metadata.json --cls_meta artifacts/classification_system_formal/metadata.json --enable_nr --out_json artifacts/inference_formal/result.json
```

示例：

```powershell
python -X utf8 predict_ik.py --pose=100,200,800,0.1,-0.2,0.3 --pred_meta artifacts/prediction_system_formal/metadata.json --cls_meta artifacts/classification_system_formal/metadata.json --enable_nr --out_json artifacts/inference_formal/test_pose_001.json
```

### 15.5 云端运行注意事项

1. `prediction system formal` 的训练时间会显著长于 `classification system formal`。
2. `prediction system formal` 只有在全部目标子空间训练完成后，才会生成最终 `metadata.json`。
3. 如果云平台显存不足，可优先减小：
   - `--feature_batch_size`
   - `--batch_size`
4. 如果云平台算力足够，建议先完整跑 `prediction_system_formal`，不要与其他大型任务同时抢占显存。
5. 正式推理时，`pred_meta` 和 `cls_meta` 必须来自同一分段配置，当前即：
   - `abb_simplified`

### 15.6 当前推荐执行顺序

建议严格按如下顺序执行：

1. `train_prediction_models.py` formal
2. `train_classification_models.py` formal
3. `predict_ik.py --enable_nr`

当前这三步完成后，ABB 版才算进入“正式实验结果可分析”的阶段。
