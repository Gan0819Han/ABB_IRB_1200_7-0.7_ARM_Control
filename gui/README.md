# GUI 工程

## 当前版本说明
当前 GUI 为第一版可用工具界面，采用 `tkinter` 实现，不依赖额外 GUI 库。

当前版本新增：
- 常用 `JSON` 路径输入框的“选择...”按钮
- 右侧“结果总结 + 原始日志”双输出区
- Unity 页中的“避障结果转 Unity 回放”
- 图表页的详细用途说明

启动命令：

```bash
conda activate arm_nn
python gui\app.py
```

## 页面结构
### 1. 推理页
用于运行 `predict_ik.py`。

输入项：
- 目标位姿 `pose6`
- `prediction metadata`
- `branch metadata`
- `fine metadata`
- 输出 `JSON`

说明：
- 本页只负责逆解推理
- `q_start` 不参与本页的 `predict_ik`
- `prediction / branch / fine metadata` 与输出 `JSON` 都可通过“选择...”按钮直接选取路径
- 运行成功后，GUI 会自动尝试把最终关节解同步到 Unity 页的 `q_goal` 与 `FK参考导出` 的 `q`

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

说明：
- 本页会同时做候选逆解评估、轨迹碰撞检测与自动换解
- `q_start` 会参与整条运动轨迹的碰撞分析
- 场景文件、元数据和输出路径均支持通过“选择...”按钮指定
- 运行成功后，GUI 会自动把选中的最终关节解同步到 Unity 页的 `q_goal` 与 `FK参考导出` 的 `q`

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
用于调用现有绘图脚本，当前分为三个动作：

#### 4.1 生成核心图表
作用：
- 汇总当前工程的核心论文图表
- 包括 `FK` 偏置验证、子空间划分对比、预测误差、分类器表现、benchmark 相关图

适用场景：
- 写论文主体实验章节
- 更新 `figure/figures/` 下的综合图

#### 4.2 生成工作空间图
作用：
- 根据保存的参考样本绘制三视图投影和三维样本可达空间图

依赖：
- `data/subspace_reference_abb_strict_samples512_seed2026/`

适用场景：
- 说明样本覆盖范围
- 展示 ABB_IRB 的工作空间分布

#### 4.3 生成避障图
作用：
- 根据固定障碍物规划结果生成碰撞/无碰撞轨迹对比图
- 生成当前论文风格的避障示意图

依赖：
- `artifacts/obstacle_avoidance/open_space_reselect_demo_plan.json`

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
2. 根据需要选择：
   - `生成核心图表`
   - `生成工作空间图`
   - `生成避障图`
3. 运行完成后到以下目录查看结果：
   - `figure/figures/`
   - `figure/data/`

## 下一步可扩展内容
- 参数合法性校验
- 训练页整合
- DeepSeek API 分析页
- 任务历史记录
