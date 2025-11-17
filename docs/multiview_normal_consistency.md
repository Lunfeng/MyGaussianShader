# 多视角法线一致性约束的可微对齐流程（次创新点详解）

本节展开描述“基于 Procrustes/球面平均的可微对齐流程”这一次创新点，说明其作用场景、数据流与可插拔实现方式，并给出注释丰富的伪代码示例，便于直接落地到当前训练循环中。

## 设计目标
- **服务主创新点**：为“多视角法线一致性”提供一个稳定、可微、易调权的算法模块，使同一三维点/像素在不同视角下的法线方向保持一致或只相差一个刚性旋转。
- **稳健性**：利用可见性/alpha/深度置信度过滤掉遮挡与噪声区域；法线在对齐前先单位化，避免数值不稳定。
- **可插拔**：以独立损失项形式与现有 `predicted_normal_loss` 协同开启，按迭代或视差自适应调整权重 `λ_mv`。

## 数据流与主要符号
- `normal_map_per_view[v]`：渲染得到的第 `v` 个视角的预测法线贴图 `(3, H, W)`。
- `depth_map_per_view[v]`：对应视角的深度贴图，用于反投影到三维空间。
- `camera_to_world_rot[v]`：将法线从相机系旋转到世界系的旋转矩阵 `R_v`（外参的旋转部分）。
- `visibility_mask[v]`：可见性或 alpha 掩码，筛掉遮挡/背景。
- `ref_view`：选作参考的主视角索引（可固定或根据视差动态挑选）。
- `point_world`：由主视角反投影得到的世界坐标点，用来在其他视角重投影并收集对应法线。

## 对齐与损失的数学表达
1. **坐标统一**：将各视角法线从相机系旋转到世界系并单位化：`n_v = normalize(R_v * n_v_cam)`。
2. **参考选择**：取参考法线 `n_ref`（来自 `ref_view` 的同一空间点）。
3. **刚性对齐**：对每个视角法线 `n_v` 求一个最优旋转 `Q_v`，使得 `Q_v * n_v` 与 `n_ref` 夹角最小。若使用 Procrustes/SVD，可写为 `Q_v = argmin_Q ||Q n_v - n_ref||_2^2`，闭式解为 `Q_v = U V^T`，其中 `U Σ V^T = n_ref n_v^T` 的 SVD。
4. **一致性损失**：
   - 余弦形式：`L_v = 1 - cos(Q_v * n_v, n_ref)`。
   - 或欧氏形式：`L_v = ||Q_v * n_v - n_ref||_2^2`。
   - 总损失：`L_mv = Σ_v w_v * L_v`，权重 `w_v` 可由可见性、视差或置信度决定。

## 伪代码（变量命名更直观，注释更细）
```python
# 假设以下输入在训练循环中已经可用：
# normal_map_per_view[v]: 第 v 个视角的预测法线 (3, H, W)
# depth_map_per_view[v]:  第 v 个视角的深度 (1, H, W)
# camera_to_world_rot[v]: 外参旋转矩阵 R_v (3, 3)
# visibility_mask[v]:     可见性/alpha 掩码 (1, H, W)
# pixel_sampler():        采样可重投影像素的函数（可基于前景掩码与深度稳定性）

lambda_mv = schedule_mv_weight(iteration)  # 迭代相关的权重调度
ref_view = choose_reference_view()         # 例如固定主视角或依据视差选择
sampled_pixels = pixel_sampler(ref_view, visibility_mask[ref_view])

loss_mv = 0.0
for (y, x) in sampled_pixels:
    # 1) 主视角反投影得到世界坐标点（用于在其他视角重投影）
    point_world = backproject(depth_map_per_view[ref_view][0, y, x], y, x, ref_view)

    # 2) 收集所有视角的法线，并统一到世界坐标系
    normals_world = []
    view_weights = []
    for v in all_views:
        # 使用可见性/遮挡过滤
        if visibility_mask[v][0, y, x] < visibility_threshold:
            continue

        # 将法线从相机系旋转到世界系并单位化
        normal_cam = normal_map_per_view[v][:, y, x]
        normal_world = normalize(camera_to_world_rot[v] @ normal_cam)
        normals_world.append(normal_world)

        # 视差/置信度权重（可用深度一致性或光度误差估计）
        weight_v = compute_confidence(point_world, depth_map_per_view[v], v)
        view_weights.append(weight_v)

    if len(normals_world) < 2:
        continue  # 没有足够视角就跳过

    # 3) 选择参考法线（来自 ref_view）
    normal_ref = normals_world[0]

    # 4) 对每个视角求刚性对齐旋转并累加损失
    for n_v, w_v in zip(normals_world[1:], view_weights[1:]):
        # Procrustes/SVD 求最优旋转 Q_v，使 Q_v * n_v 逼近 normal_ref
        Q_v = procrustes_optimal_rotation(source=n_v, target=normal_ref)

        # 余弦距离或欧氏距离均可；以下用余弦距离
        aligned_normal = Q_v @ n_v
        loss_v = 1.0 - cosine_similarity(aligned_normal, normal_ref)

        # 按置信度加权并累加
        loss_mv += w_v * loss_v

# 5) 与其它损失合并，控制开启时机与权重
loss_total = loss_rgb + lambda_pred_normal * loss_pred_normal + lambda_mv * loss_mv
loss_total.backward()
```

### 实现细节与落地建议
- **采样策略**：`pixel_sampler` 可优先选择前景高 alpha、深度梯度小的像素，减少遮挡误对齐；也可在高斯点层级按可见视角列表采样。
- **置信度权重**：`compute_confidence` 可结合深度一致性（重投影误差小则权重大）、视差（大视差更有约束力）、光度残差等信息。
- **旋转求解**：`procrustes_optimal_rotation` 可用 SVD 的闭式解；若想更简单可直接使用余弦距离而不显式求 `Q_v`，等价于假设跨视角法线仅受轻微噪声。
- **调度策略**：在 `predicted_normal_loss` 启动后延迟开启 `λ_mv`（例如迭代 > 10k），或按“先小后大”的权重曲线平滑引入，避免早期噪声。
- **可微性**：上述运算均可在 PyTorch 中实现；SVD 与矩阵乘法在 autograd 下可正常反传。
