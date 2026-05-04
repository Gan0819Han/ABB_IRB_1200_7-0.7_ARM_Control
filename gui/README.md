# GUI 工程

## 当前版本说明
当前 GUI 为第一版可用工具界面，采用 `tkinter` 实现，不依赖额外 GUI 库。

当前版本新增：
- 常用 `JSON` 路径输入框的“选择...”按钮
- 常用路径/目录输入框新增“设为默认”按钮，支持跨重启记忆默认路径
- 右侧“结果总结 + 原始日志”双输出区
- 主体区支持拖拽分栏，左侧页面区与右侧输出区宽度可调
- 右侧输出区支持上下拖拽，结果总结与原始日志高度可调
- `pose6 / q_start / center_mm / size_mm` 改为分框输入，便于直观编辑
- 说明文字改为随窗口宽度自动换行，避免缩窄时被遮挡
- 图表页左侧参数区与右侧避障图预览区支持单独拖拽调宽，默认预览区更大
- Unity 页中的“避障结果转 Unity 回放”
- 图表页的详细用途说明
- 图表页的图像输出目录与数据输出目录可自定义
- 图表页可为三类绘图分别指定数据来源
- 避障页内置首个 AABB 障碍物编辑器
- 避障页障碍物编辑器已支持多障碍物切换、新增与删除
- 避障页支持纵向滚动，参数较多时可直接滚动查看下半部分
- 避障成功后自动生成避障图，并在图表页右侧显示预览

启动命令：

```bash
conda activate arm_nn
python gui\app.py
```

默认路径持久化说明：

- 点击某个路径或目录输入框右侧的 `设为默认` 后
- 当前值会写入本地配置文件：
  - `gui/gui_defaults.json`
- 下次重新打开 GUI 时，会优先使用该文件中的默认值
- 这只影响 GUI 初始路径，不改动算法脚本逻辑

## 页面结构
### 1. 推理页
用于运行 `predict_ik.py`。

输入项：
- 目标位姿 `pose6`
- `prediction metadata`
- `branch metadata`
- `fine metadata`
- 输出 `JSON`
- 推理超参数：
  - `topk_shoulder`
  - `topk_elbow`
  - `topk_wrist`
  - `max_branch_candidates`
  - `fine_topk_per_branch`
  - `max_subspace_candidates`
  - `enable_nr`
  - `nr_max_iters`
  - `nr_tol_pos_mm`
  - `nr_tol_ori_rad`
  - `nr_damping`
  - `nr_step_scale`

说明：
- 本页只负责逆解推理
- `q_start` 不参与本页的 `predict_ik`
- 本页 `pose6` 与“避障”页共用同一输入框变量，任一页面修改后另一页会同步更新
- `pose6` 已拆分为 `x / y / z / phi / theta / psi` 六个输入框
- 其中 `x,y,z` 单位为 `mm`，`phi,theta,psi` 单位为 `rad`
- `prediction / branch / fine metadata` 与输出 `JSON` 都可通过“选择...”按钮直接选取路径
- 运行成功后，GUI 会自动尝试把最终关节解同步到 Unity 页的 `q_goal` 与 `FK参考导出` 的 `q`

超参数含义：
- `topk_shoulder / topk_elbow / topk_wrist`
  - 第一层分支分类器在三个头部分别保留多少个候选
- `max_branch_candidates`
  - 三个头组合后，最多保留多少个粗分支候选
- `fine_topk_per_branch`
  - 每个粗分支下，第二层细分类器最多保留多少个候选
- `max_subspace_candidates`
  - 最终进入子空间回归评估的候选总数上限
- `enable_nr`
  - 是否启用 Newton-Raphson 校正
- `nr_max_iters`
  - NR 最大迭代次数
- `nr_tol_pos_mm`
  - NR 位置收敛阈值，单位 `mm`
- `nr_tol_ori_rad`
  - NR 姿态收敛阈值，单位 `rad`
- `nr_damping`
  - NR 阻尼系数
- `nr_step_scale`
  - NR 步长缩放因子

### 2. 避障页
用于运行 `scripts/plan_collision_free_ik.py`。

输入项：
- 目标位姿 `pose6`
- 起始关节 `q_start`
- 场景文件 `scene_json`
- `prediction metadata`
- `branch metadata`
- `fine metadata`
- 输出 `JSON`
- 避障常用超参数：
  - `topk_shoulder`
  - `topk_elbow`
  - `topk_wrist`
  - `max_branch_candidates`
  - `fine_topk_per_branch`
  - `max_subspace_candidates`
  - `max_evaluated_candidates`
  - `nr_max_iters`
  - `nr_tol_pos_mm`
  - `nr_tol_ori_rad`
  - `nr_damping`
  - `nr_step_scale`
  - `trajectory_steps`
  - `dedupe_tol_deg`

说明：
- 本页会同时做候选逆解评估、轨迹碰撞检测与自动换解
- 当前轨迹评估不再只检查单段直达轨迹，也会同时比较若干 waypoint 两段式轨迹模板
- 本页 `pose6` 与“推理”页共用同一输入框变量，任一页面修改后另一页会同步更新
- `pose6` 已拆分为 `x / y / z / phi / theta / psi` 六个输入框
- `q_start` 已拆分为 `q1 ~ q6` 六个输入框，并在下方显示项目关节限位范围
- `q_start` 会参与整条运动轨迹的碰撞分析
- 当前已开放避障阶段常用超参数调节
- 场景文件、元数据和输出路径均支持通过“选择...”按钮指定
- 运行成功后，GUI 会自动把选中的最终关节解同步到 Unity 页的 `q_goal` 与 `FK参考导出` 的 `q`

避障超参数含义：

- `topk_shoulder / topk_elbow / topk_wrist`
  - 第一层粗分类三个头分别保留多少个候选
- `max_branch_candidates`
  - 粗分类组合后最多保留多少个 branch
- `fine_topk_per_branch`
  - 每个 branch 下第二层细分类最多保留多少个候选
- `max_subspace_candidates`
  - 最终进入子空间候选池的上限
- `max_evaluated_candidates`
  - 真正进入 `NR + 碰撞检测 + 代价排序` 的候选数量上限
- `nr_max_iters / nr_tol_pos_mm / nr_tol_ori_rad / nr_damping / nr_step_scale`
  - 控制 Newton-Raphson 修正过程
- `trajectory_steps`
  - 轨迹离散采样步数，越大碰撞检测越细，但耗时更高
- `dedupe_tol_deg`
  - 候选解去重容差，越小通常保留的不同候选更多

结果摘要说明：

- `是否找到无碰撞候选`
  - 表示所有已评估候选里，是否至少存在一条无碰撞轨迹
- `选中轨迹是否无碰撞`
  - 表示最终被选中并导出到 Unity 的轨迹是否无碰撞
- `选中轨迹模式`
  - `direct` 表示单段直达
  - 其他如 `goal_biased / midpoint / lift_elbow` 等表示 waypoint 两段式轨迹

#### 2.1 障碍物编辑器
当前“避障”页新增一个“障碍物编辑（AABB）”区域。

可编辑项：
- `当前障碍物`
- `obstacle name`
- `center_mm`
- `size_mm`

说明：
- 当前支持编辑 `scene_json` 中的多个障碍物
- 障碍物默认按 `AABB` 长方体处理
- `center_mm` 已拆分为 `x / y / z`
- `size_mm` 已拆分为 `dx / dy / dz`
- `center_mm` 表示障碍物中心位置
- `size_mm` 表示长方体尺寸
- 单位均为 `mm`

按钮功能：
- `从 scene_json 读取`
  - 将当前 `scene_json` 中所选障碍物参数读入 GUI
- `写回 scene_json`
  - 将当前 GUI 中所选障碍物参数写回 `scene_json`
- `恢复默认值`
  - 将障碍物参数恢复为工程最初默认值
  - 当前默认值为：
    - `name = demo_box_1`
    - `center_mm = [221.7, 274.53, 493.57]`
    - `size_mm = [127.62, 90.45, 175.06]`
- `新增障碍物`
  - 在当前 `scene_json` 的 `obstacles` 列表末尾新增一个 AABB
  - 当前默认新增参数为：
    - `center_mm = [360.0, -180.0, 540.0]`
    - `size_mm = [110.0, 90.0, 160.0]`
- `删除当前障碍物`
  - 删除当前选中的障碍物
  - 当前程序要求至少保留 1 个障碍物，不支持通过 GUI 删除到空列表

联动规则：
- 点击 `运行 plan_collision_free_ik` 前，GUI 会先自动执行一次“写回 scene_json”
- 因此你只要改完 `center_mm / size_mm`，直接运行避障即可生效
- 若你想回到最初场景，只需先点 `恢复默认值`，再运行避障或手动点 `写回 scene_json`

### 3. Unity 页
分三部分：
- `FK` 参考导出
- 轨迹 `JSON` 导出
- 避障结果转 Unity 回放

当前支持自定义：
- `FK` 的关节角 `q`
- 轨迹的 `q_start / q_goal / steps / duration / name / out_json`
- 避障规划结果的 `plan_json / demo_name / out_json`

说明：
- Unity 页不会重新做逆解或避障规划
- 它只负责把 Python 侧已经算好的结果整理成 Unity 直接可读的 `JSON`
- 其中“避障结果转 Unity 回放”对应脚本为 `scripts/export_unity_obstacle_avoidance_demo.py`
- 该脚本的输入应来自“避障”页生成的规划结果 `JSON`
- `FK` 输出、轨迹输出、避障回放输出路径均支持通过“选择...”按钮指定
- 若前面已经运行过“推理”或“避障”，则：
  - `轨迹导出` 中的 `q_goal` 会被自动更新为最新求得的终解
  - `FK参考导出` 中的 `q` 也会被同步为同一组终解

### 4. 图表页
用于调用现有绘图脚本，当前分为“输出路径配置 + 三个动作”。

当前图表页右侧新增一个“避障图预览”区域，用于显示最近一次生成的避障图结果。

#### 4.0 输出路径
当前可配置：
- `图像输出目录`
- `数据输出目录`

说明：
- 若不修改，则默认输出到：
  - `figure/figures/`
  - `figure/data/`
- 若修改，则 GUI 会把目录路径传给绘图脚本
- `避障图` 主要写入图像输出目录
- `核心图表` 与 `工作空间图` 会同时写入图像输出目录和数据输出目录

#### 4.1 数据来源控制
当前图表页支持分别指定三类图的输入来源：

- `单案例 IK JSON`
  - 控制核心图表中的“单案例时间分解与误差图”
  - 适合你切换不同目标位姿后的 `predict_ik` 输出进行对比
- `参考样本目录`
  - 控制工作空间图读取哪一批 `subspace_*_reference.npz`
- `避障规划 JSON`
  - 控制避障图读取哪一份 `plan_collision_free_ik` 结果
  - 适合你切换不同起始关节、目标位姿、障碍物场景后的规划结果进行对比

#### 4.2 生成核心图表
作用：
- 汇总当前工程的核心论文图表
- 包括 `FK` 偏置验证、子空间划分对比、预测误差、分类器表现、benchmark 相关图

数据来源：
- `artifacts/fk_validation/fk_validation_report.json`
  - 用于 `FK` 中 `theta2_offset` 偏置验证图
- `artifacts/subspace_validation/subspace_profiles.json`
  - 用于子空间划分方案对比图
- `artifacts/prediction_system_formal/metadata.json`
  - 用于 `192` 个子空间预测误差统计图
- `artifacts/classification_system_formal/metadata.json`
  - 用于 `192` 类平铺分类器精度图
- `artifacts/branch_classification_system/metadata.json`
  - 用于第一层粗分类器精度图
- `artifacts/fine_classification_system/metadata.json`
  - 用于第二层细分类器精度图
- `artifacts/fine_classification_system/test_pose_001_full_ik.json`
  - 用于单案例 `IK` 时间分解与误差图

说明：
- 该按钮读取的是“实验汇总结果文件”
- 不依赖当前 GUI 页面的 `pose6` 或 `q_start`
- 但其中“单案例时间/误差图”会读取上面的 `单案例 IK JSON`

适用场景：
- 写论文主体实验章节
- 更新 `figure/figures/` 下的综合图

#### 4.3 生成工作空间图
作用：
- 根据保存的参考样本绘制三视图投影和三维样本可达空间图

依赖：
- `data/subspace_reference_abb_strict_samples512_seed2026/`

数据来源：
- `data/subspace_reference_abb_strict_samples512_seed2026/subspace_*_reference.npz`
  - 每个 `npz` 中读取 `pose6`
  - 只使用其中的末端位置 `x,y,z`
  - 并结合 `subspace_id -> branch_label` 的映射进行着色与分组

说明：
- 该按钮读取的是“参考样本点云”
- 不依赖当前 GUI 页面的 `pose6`、`q_start` 或避障结果

适用场景：
- 说明样本覆盖范围
- 展示 ABB_IRB 的工作空间分布

#### 4.4 生成避障图
作用：
- 根据固定障碍物规划结果生成碰撞/无碰撞轨迹对比图
- 生成当前论文风格的避障示意图

依赖：
- `artifacts/obstacle_avoidance/open_space_reselect_demo_plan.json`

当前该规划结果内部包含：
- 初始关节：`q_start_deg`
- 目标位姿：`target_pose6`
- 障碍物场景：`scene`
- 候选轨迹评估结果：`evaluated_candidates`

说明：
- 该按钮读取的是“单次避障规划结果文件”
- 你可以在图表页手动改成任意一份 `plan_json`
- 若你是在 GUI 的“避障”页刚运行成功，当前程序也会自动把输出路径同步到这里
- 并且当前会自动执行一次“生成避障图”
- 程序会自动切换到“图表”页，并刷新右侧预览图

适用场景：
- 说明为什么系统需要进行候选重选
- 展示障碍物与目标点之间的空间关系

## 输出区
右侧拆分为两个部分：

### 结果总结
显示整理后的关键信息，例如：
- 是否收敛
- 位置误差
- 姿态误差
- 迭代次数
- 输出文件路径

当前为规则型总结，不依赖 LLM。
后续可接入 DeepSeek API 生成解释性分析。

### 原始日志
显示完整命令、标准输出、错误输出和退出码。
用于排查脚本运行问题。

### 分栏调节
当前 GUI 主体区已改为可拖拽布局：

1. 左右分隔线
   - 可左右拖动
   - 用于调整左侧主工作区与右侧输出区占比
2. 右侧上下分隔线
   - 可上下拖动
   - 用于调整“结果总结”和“原始日志”的显示高度

## 当前实现原则
- 不改现有算法脚本
- GUI 只做命令调度与结果展示
- 结果文件继续保存到原有 `artifacts/`、`figure/`、Unity 目录

## 当前推荐使用顺序
### 1. 仅做逆解
1. 在“推理”页填写目标位姿
2. 运行 `predict_ik`
3. 查看右侧“结果总结”与“原始日志”

### 2. 做避障规划
1. 在“避障”页填写：
   - `pose6`
   - `q_start`
   - `scene_json`
2. 运行 `plan_collision_free_ik`
3. 得到规划结果 `JSON`
4. GUI 会自动：
   - 同步 `图表 -> 避障规划 JSON`
   - 切换到 `图表` 页
   - 运行一次 `生成避障图`
   - 在右侧“避障图预览”区显示最新结果

### 3. 导入 Unity 做障碍物回放
1. 在“避障”页先生成规划结果，例如：
   - `artifacts/obstacle_avoidance/gui_plan.json`
2. 切到“Unity”页的“避障结果转 Unity 回放”
3. 填写：
   - `plan_json`：上一步的规划结果
   - `demo_name`：演示名称
   - `输出 JSON`：建议写到 Unity 工程 `Assets/PlanningData/`
4. 点击“导出避障回放 JSON”
5. 在 Unity 中把该 `JSON` 指给 `AbbObstacleAvoidanceDemo` 组件使用

### 4. 生成论文图表
1. 切到“图表”页
2. 如有需要，先设置：
   - `图像输出目录`
   - `数据输出目录`
   - `单案例 IK JSON`
   - `参考样本目录`
   - `避障规划 JSON`
3. 根据需要选择：
   - `生成核心图表`
   - `生成工作空间图`
   - `生成避障图`
4. 运行完成后到你设置的目录查看结果；若未修改，则仍查看：
   - `figure/figures/`
   - `figure/data/`

## 下一步可扩展内容
- 参数合法性校验
- 训练页整合
- DeepSeek API 分析页
- 任务历史记录
