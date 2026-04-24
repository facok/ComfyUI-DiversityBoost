# ComfyUI-DiversityBoost

恢复蒸馏扩散模型的构图多样性。免训练、单步执行、零模型修改。

> [English README](README.md)

## 问题

步数蒸馏模型（FLUX2.[Klein]、z-image-turbo 等）能在极少步数内生成高质量图像，但存在**构图坍缩**：不同 seed 生成几乎相同的布局。人像提示词总是把主体放在正中间，风景提示词总是同一条地平线。

根因：蒸馏冻结了 token norm 的空间分布，无论初始噪声是什么，模型都被锁定在一个"平均"构图上。

## 解决方案（V3）

DiversityBoost V3 通过单个 post-CFG 钩子应用**多项式频率调制**和 **DCT 构图推动**：

1. **多项式频率调制** — 平滑、连续地衰减高频振幅。基于 token 网格归一化（与分辨率无关）。近 DC 低频受保护，防止亮度/颜色偏移。
2. **DCT 构图推动** — 施加一个随机低频空间场，重新分配潜空间的能量分布，引导模型朝不同构图方向重建。

后续步骤中，模型在每个 seed 的噪声驱动下自由重建连贯的细节，产生不同的构图。

零模型修改。零训练。一个节点。

## 工作原理

1. 将模型预测转换到原始潜空间
2. **多项式频率调制** — 在频域中平滑衰减高频，通过 DiT patch_size 进行 token 网格归一化。DC 归零以获得多样性；近 DC 低频受保护以保留结构。
3. **DCT 空间场** — 合成随机 4x4 低频场（零 DC、pink/white/blue 噪声加权），归一化到单位标准差，按 strength 缩放
4. **乘法推动** — `调制结果 * (1 + field)`，钳位防止死区
5. 转换回原空间

主要效果在第 0 步。可选的渐进衰减（`linear`/`cosine` 调度）将高频衰减延伸到早期步骤以获得更强的多样性。

## 节点

提供两个节点以保持向后兼容：

| 节点 | 类类型 | 说明 |
|------|--------|------|
| **Diversity Boost (V3)** | `DiversityBoostCoreV3` | 多项式频率调制，token 网格归一化，近 DC 保护。推荐使用。 |
| **Diversity Boost** | `DiversityBoostCore` | 旧版 Butterworth LPF（`n_periods` 参数）。保留给旧工作流。 |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/facok/ComfyUI-DiversityBoost.git
```

无额外依赖——只需要 PyTorch（ComfyUI 自带）。

## 使用

```
MODEL -> [Diversity Boost (V3)] -> MODEL -> KSampler
```

1. 添加 **Diversity Boost (V3)** 节点（分类：`sampling`）
2. 将模型连接到输入，输出连接到 KSampler
3. 用不同 seed 生成——构图会产生变化

## 参数（V3）

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| **strength** | 2.0 | 0.0 – 2.0 | 构图推动幅度。0 = 仅清理，1.0 = 适中，2.0 = 强烈。 |
| **clamp** | 0.5 | 0.1 – 3.0 | 乘法缩放因子的上限。scale 被钳位到 [0.1, 1+clamp]。 |
| **noise_type** | pink | pink / white / blue | 随机 DCT 系数的频谱类型。pink 增强低频构图模式。 |
| **dc_preserve** | 0.0 | 0.0 – 1.0 | DC 振幅保留（仅第 0 步）。0 = 最大多样性。1 = 保留原始亮度。 |
| **energy_compensate** | False | — | 将输出 RMS 缩放至与原始预测一致。默认关闭。 |
| **hf_factor** | 1.0 | 0.0 – 1.0 | 高频衰减强度。1.0 = 完全归零高频。 |
| **lf_factor** | 0.3 | 0.0 – 1.0 | 低频放大。1.0 = +50% 提升。 |
| **transition** | 2.0 | 0.5 – 4.0 | 多项式过渡形状。0.5 = 陡峭，1.0 = 线性，2.0 = 平滑，4.0 = 非常平滑。 |
| **schedule** | linear | flat / linear / cosine | 时间步调度。`flat` = 仅第 0 步（所有采样器安全）。`linear`/`cosine` = 渐进衰减。 |

### schedule 参考

| 模式 | 效果 | 采样器兼容性 |
|------|------|-------------|
| **flat** | 仅在第 0 步进行频率调制 + DCT 推动。模型在剩余所有步骤中恢复。 | 所有采样器（一阶和二阶） |
| **linear** | 高频衰减在前 ~3 步线性衰减。DCT 推动仍在第 0 步。 | 二阶采样器（res_2m、heunpp2） |
| **cosine** | 高频衰减按余弦曲线衰减。比 linear 更平滑。 | 二阶采样器 |

**注意：** 一阶采样器（euler）对第 1+ 步的 denoised 修改敏感。一阶采样器请使用 `flat` 调度。

### strength 参考

| 值 | 效果 |
|----|------|
| 0.0 | 仅高频清理（无构图推动） |
| 0.5 | 微妙的构图变化 |
| 1.0 | 适中的构图变化 |
| 2.0 | 强烈的构图变化（默认） |

### hf_factor 参考

| 值 | 效果 |
|----|------|
| 0.0 | 无高频衰减（仅清理） |
| 0.5 | 适中高频衰减（高频 ~50%） |
| 0.7 | 强烈高频衰减（高频 ~30%） |
| 1.0 | 完全归零高频（默认） |

## 旧版节点（V2）

旧的 `Diversity Boost` 节点（类类型 `DiversityBoostCore`）保留用于向后兼容。它使用原始的 Butterworth LPF 和 `n_periods` 参数。

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| **strength** | 0.5 | 0.0 – 2.0 | 构图推动幅度 |
| **clamp** | 1.0 | 0.1 – 3.0 | 缩放因子上限 |
| **noise_type** | pink | pink / white / blue | DCT 系数频谱 |
| **n_periods** | 2 | 1 – 10 | Butterworth 截止频率——保留的空间周期数 |
| **dc_preserve** | 0.0 | 0.0 – 1.0 | DC 振幅保留 |
| **energy_compensate** | False | — | 输出 RMS 重缩放 |

## 使用建议

- **从 V3 默认值开始** — 它们针对强多样性和最小副作用进行了调优
- **一阶采样器（euler）？** 使用 `schedule=flat` 以避免降噪不完全
- **二阶采样器（res_2m）？** `schedule=linear` 适合渐进式高频释放
- **想要更多多样性？** 提高 `strength` 或 `hf_factor`
- **只需清理？** 设置 `strength=0` — 纯高频衰减，无构图推动
- **兼容**其他模型补丁（ControlNet 等）——使用不同的钩子，互不干扰

## 已测试模型

| 模型 | 状态 |
|------|------|
| FLUX2.[Klein] 9B | 已测试 |
| z-image-turbo | 已测试 |

欢迎反馈其他蒸馏模型的测试结果。

## 许可证

MIT
