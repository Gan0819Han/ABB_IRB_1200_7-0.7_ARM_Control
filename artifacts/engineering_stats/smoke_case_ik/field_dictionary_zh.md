## 字段中文说明

| field_key | field_name_zh | unit | meaning | module |
| --- | --- | --- | --- | --- |
| method_id | 方法ID | - | 脚本内部使用的方法标识 | ik_single, obstacle_single |
| label | 方法名称 | - | 用于界面和论文展示的中文/英文方法名 | ik_single, obstacle_single |
| solve_time_ms | 求解时间 | ms | 普通逆解阶段总耗时 | ik_single |
| final_pos_err_mm | 最终位置误差 | mm | 末端位姿解的位置误差 | ik_single, obstacle_single |
| final_ori_err_rad | 最终姿态误差 | rad | 末端位姿解的姿态误差 | ik_single, obstacle_single |
| iters | 迭代次数 | - | 数值修正或优化器迭代次数 | ik_single, obstacle_single |
| converged | 是否收敛 | - | 求解过程是否满足收敛条件 | ik_single, obstacle_single |
| within_joint_limits | 满足关节限位 | - | 结果关节角是否位于机械臂限位范围内 | ik_single |
| planning_time_ms | 总规划时间 | ms | 避障单解从末端逆解到轨迹筛选的总耗时 | obstacle_single |
| ik_time_ms | 末端逆解时间 | ms | 避障链路里仅求出末端关节解的时间 | obstacle_single |
| trajectory_generation_time_ms | 轨迹生成时间 | ms | 构造候选关节轨迹的时间 | obstacle_single |
| trajectory_evaluation_time_ms | 轨迹评估时间 | ms | 碰撞检测与净空评估耗时 | obstacle_single |
| selection_time_ms | 方案筛选时间 | ms | 候选轨迹计算代价并排序的时间 | obstacle_single |
| selected_solution_collision_free | 是否无碰撞 | - | 最终选中的轨迹是否无碰撞 | obstacle_single |
| collision_frame_count | 碰撞帧数 | 帧 | 轨迹离散帧中发生碰撞的帧数 | obstacle_single |
| min_clearance_mm | 最小净空 | mm | 轨迹全过程的最小障碍物净空 | obstacle_single |
| joint_path_length_deg | 路径长度 | deg | 六个关节累计路径长度 | obstacle_single |
| max_joint_step_deg | 最大单步关节变化 | deg | 离散轨迹中单步最大的关节变化量 | obstacle_single |
| trajectory_mode | 轨迹模式 | - | 选中的轨迹模式，例如 direct 或 via_waypoint | obstacle_single |
| initial_guess_count | 初值数量 | - | 数值法尝试的初始关节解数量 | obstacle_single |
| unique_goal_candidate_count | 唯一终点候选数 | - | 数值法去重后的终点关节候选数量 | obstacle_single |
