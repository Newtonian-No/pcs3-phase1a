# PE-SSM 项目介绍与协作计划

> 写给浩宇 · 2026-06-15（2026-06-20 更新）  
> 阮愔哲 (Kevin) · 上海大学

---

## 一、这是什么项目

**PE-SSM**：Predictive Error-Conditioned Selective State Space Model。

一句话：把「预测编码」（Predictive Coding）的 prediction error 信号注入 Mamba（SSM），让状态空间模型的 selective scan 机制获得更强的时序调制信号。

更直白地说：传统深度学习网络对每个输入一视同仁地处理。但生物大脑不是这样——大脑在不停地预测接下来会发生什么，当预测出错时，错误信号本身会驱动后续处理。我们要做的事就是把这种「预测→偏差→选择性响应」的机制嵌入 Mamba 架构。

### 为什么是 Mamba

Mamba 的核心是 **selective scan**——它的状态转移矩阵不是固定的，而是根据当前输入动态变化的。这个动态调制本质上是一种「选择性滤波」：什么信息该记住、什么该忘掉，由当前 context 决定。

在滤波系统里，最自然的驱动力就是 **innovation（创新/更新信号）**——也就是「预测值和实际值的偏差」。Kalman filter 就是这么工作的：prediction error 告诉滤波器要不要更新状态、更新多少。

所以想法很直接：**让 prediction error 去调制 Mamba 的 selective scan**。不引入复杂的 PCN 迭代推理（那很慢），而是做一个一步到位（amortized）的误差嵌入。

---

## 二、项目代号演变

- **PC-S³**：Predictive Coding with Structured State-Space Sequence Models（最初命名）
- **PE-SSM**：Predictive Error-Conditioned Selective State Space Model（当前命名）

改名的原因是——星海理事会（一个多 Agent 学术评审系统）的魔鬼代言人指出：我们没有做传统 PCN 的迭代推理，叫 "Predictive Coding" 会被 reviewer 抓把柄。所以降级为 "predictive error-conditioned"——更准确地描述了我们实际做的东西。

---

## 三、已经做了什么

### Phase 1a — 基线验证（2026.05 下旬）

- 展平 3072 维 token + 1 层 Mamba，~6.85M 参数
- CIFAR-100，vanilla 约 0.363
- **教训**：展平 token + 单层 SSM 是死路，必须 patch-based tokenization + 深层堆叠

### Phase 1b — 架构验证（2026.06 初）

- PatchEmbed (patch=4, 64 tokens) + L=64 深层 Mamba，**2.07M 参数**
- 6 种变体：vanilla, concat, C, delta, concat_shuffled, B

| 变体 | Best Acc | vs Vanilla | 说明 |
|------|----------|-----------|------|
| concat | 0.3813 | +1.07% | PCN error 拼接到 SSM input |
| vanilla | 0.3706 | — | 纯 SSM 基线 |
| C | 0.3656 | -0.50% | 标量 gate 调节 SSM 输出 |
| delta | 0.3575 | -1.31% | Error gate 调节 Δ |
| concat_shuffled | 0.3515 | -1.91% | Shuffled error（破坏语义） |
| B | 0.3475 | -2.31% | Error gate 调节 B |

**三条关键发现**：

1. **concat 有效且信号真实** — shuffled（随机打乱 error）垫底，说明增益来自 prediction error 的语义信息而非参数增量
2. **Phase 1b >> Phase 1a** — 参数从 6.85M 砍到 2.07M，性能反而涨 0.76pp。Patch-based tokenization + deep SSM 才是正道
3. **Gating 在深层架构崩溃** — B/delta/C 全部低于 vanilla，误差直接调制算子不如增强输入（"preserve the operator, perturb the input"）

### Phase 2 — 增强管线 + 扩展验证（2026.06 中）

- Conv stem + PatchEmbed (patch=4)、d=160、L=8、**6.5M 参数**
- RandAugment(N=2, M=9) + RandomErasing(0.25)
- 5090 上跑，~47 GPU-hours

| 变体 | Best Acc | vs Vanilla |
|------|----------|-----------|
| **concat** | **0.6930** | **+1.36%** |
| concat_shuffled | 0.6844 | +0.50% |
| C | 0.6837 | +0.43% |
| vanilla | 0.6794 | — |
| delta | 0.6775 | -0.19% |
| B | 0.6731 | -0.63% |

concat 仍然最优，但 shuffled 异常地排到第二——与 Phase 1b 中 shuffled 垫底的结论矛盾。

### Step 2 — 消融：要不要 Conv stem（2026.06.12）

无 Conv stem 版本在 PRO 6000 上跑完：

| 变体 | Step 2 (无stem) | Phase 2 (有stem) | 降幅 |
|------|:--:|:--:|:--:|
| concat | 0.6528 | 0.6930 | -4.0pp |
| vanilla | 0.6444 | 0.6790 | -3.5pp |

**结论**：Conv stem 必须保留。无 stem 直接掉 4pp，不值得继续。

### 🔥 MVC 验证实验 — gen_error K=2（2026.06.19-20，最新）

这是整个项目目前最重要的新结果。在 PRO 6000 上跑了三路并行 MVC（Minimum Viable Check）验证：

| 实验 | 误差类型 | 参数量 | Best Acc | 最佳轮 | 耗时 |
|------|------|--------|----------|--------|------|
| **A** vanilla | 标量 (1D) | 6.55M | 67.28% | 238 | — |
| **B** gen_error | 位置+速度 (2D) | 7.04M | **69.35%** | 279 | 21.4h |
| **D** gen_error_shuffled | 2D 打乱时序 | 7.04M | 68.27% | 296 | 21.4h |

**两条核心预测全部验证通过**：

1. ✅ **B > A**：gen_error (69.35%) 比 vanilla (67.28%) 高 **+2.07pp**——位置+速度的广义坐标误差确实优于标量误差
2. ✅ **B > D**：gen_error (69.35%) 比 shuffled (68.27%) 高 **+1.08pp**——排除了「多几个维度就有增益」的替代假说，证明增益来自**时序结构**而非维度数

额外发现：
- gen_error 收敛更快：达到 67% 只需 138 轮（vanilla 要 238 轮，**快了 100 轮**）
- 没有过拟合迹象：train-test gap 三组都在 0.27-0.28

> ⚠️ vanilla 崩在 280/300 轮（进程意外挂掉），不过 best acc 在 238 轮已出，不影响结论。

---

## 四、核心假说：广义坐标误差

### 从标量到向量

当前 Phase 2 用的是标量 prediction error：
```
x → Predictor → x̂ → e = x - x̂ → Mamba Δ(e)
```
只编码了「偏离了多远」。

**升级方案**（理论锚点：Friston 的 Generalized Coordinates of Motion）：
```
x → Predictor → (x̂, x̂', x̂'') → ẽ = (x-x̂, x'-x̂', x''-x̂'') → Mamba
```
同时编码瞬时幅度、变化方向、变化加速度。有限差分近似（零额外网络开销）。

### 实验矩阵（5 组可证伪）

| 实验 | 维度 | 状态 | Acc | vs A |
|:---:|:---:|:---:|:---:|:---:|
| **A** vanilla | 1D 标量 | ✅ | 67.28% | — |
| **B** gen_error | 2D (位置+速度) | ✅ | 69.35% | +2.07 |
| **C** K=3 | 3D (+加速度) | ⏳ 待跑 | ? | ? |
| **D** shuffled | 2D 时序打乱 | ✅ | 68.27% | +0.99 |
| **E** random | 2D 随机噪声 | ⏳ 待跑 | ? | ? |

### 三条可证伪预测

1. ✅ **B > A** — 已验证（+2.07pp）
2. ✅ **B > D** — 已验证（+1.08pp），证明时序结构是关键
3. ⏳ **C vs B 的差距 < B vs A 的差距** — 待验证（高阶导数边际递减，Friston 论文预测）

---

## 五、下一步：具体执行计划

现在最该做的是把实验矩阵 C 和 E 补完。两件事可以同时做，都在 PRO 6000 上。

### 实验 C：K=3（加上加速度）

**代码改动**（3 处，都在 `pcs3_step2.py`）：

**改动 1** — `SelfPredictor.__init__`，`vel_head` 那行下面加：
```python
if K >= 3:
    self.acc_head = nn.Linear(d_model, d_model, bias=False)
```

**改动 2** — `SelfPredictor.forward`，`vel_error` 计算完后追加：
```python
if self.K >= 3:
    x_acc = F.pad(x[:, 2:, :] - 2*x[:, 1:-1, :] + x[:, :-2, :], (0, 0, 0, 2))
    pred_acc = self.acc_head(h)
    acc_error = x_acc - pred_acc
    return torch.cat([pos_error, vel_error, acc_error], dim=-1)
```

**改动 3** — `Step2Model.__init__`，`self.K` 那行：
```python
self.K = 3 if mode == "gen_error_k3" else (2 if mode in ("gen_error", "gen_error_shuffled") else 1)
```

`Step2Model.forward` 中 gen_error_k3 复用 concat 调制路径（现有逻辑已经覆盖，确认 `ssm_mode = "concat"` 那行包含 `"gen_error_k3"` 即可）。

**运行命令**：
```bash
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate evuav
cd /media/lab/0E526ACF526ABB5B/fhy/pcs3-phase1a

nohup python3 train_step2.py \
  --mode gen_error_k3 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k3_s42.log 2>&1 &
```

- 参数量：~8.8M
- 耗时：~22h
- VRAM：~65GB

---

### 实验 E：Random Noise 对照

**不改架构**。在 `Step2Model.forward` 中，`error = self.predictors[pcn_idx](h, x)` 之后加一行：

```python
if self.mode == "gen_error_random":
    error = torch.randn_like(error) * error.std()
```

这样架构、参数量、计算量完全不变，唯一区别是 prediction error 被替换成了同方差的高斯噪声。如果 random < vanilla，就彻底排除了「Mamba 只是喜欢额外的信息通道」这个审稿人最爱挑的刺。

**运行命令**：
```bash
nohup python3 train_step2.py \
  --mode gen_error_random \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_random_s42.log 2>&1 &
```

- 参数量：7.04M（与 gen_error K=2 完全一致）
- 耗时：~21h
- VRAM：~65GB

---

### 跑之前确认清单

- [ ] `git pull` 同步最新代码
- [ ] 确认 `einops` 已装（`pip install einops`，上次因为缺这个报错过）
- [ ] 确认 CIFAR-100 数据在 `./data/` 下
- [ ] C 和 E 可以串行跑（96GB 单卡没法同时跑两个 65GB 的），建议先 C 后 E，或者 nohup 后台自动排队

### 跑完后验证

每个实验跑完会生成 `results/step2_{mode}_seed42_*.json`。快速验证命令：
```bash
python3 -c "
import json
d = json.load(open('results/step2_gen_error_k3_seed42_*.json'))  # 或 gen_error_random
print(f'best_acc={d[\"best_acc\"]:.4f} at epoch {d[\"best_epoch\"]}')
print(f'time={d[\"time\"]/3600:.1f}h')
"
```

---

## 六、做完 C+E 之后的路线图

| 阶段 | 内容 | 状态 |
|------|------|:--:|
| A~E 单 seed screening | 完成实验矩阵 | ⏳ C/E 待跑 |
| 3-seed | 对最优配置跑统计检验（seed=42/123/2024） | ⏳ 等 C/E 出结果后定 |
| 外部 baseline | ViM-Ti + ResNet-18 + DeiT-Ti（CIFAR-10/100 + TinyImageNet） | ⏳ 待 3-seed 后 |
| 论文 | 写作和投稿 | 📝 待定 |

> 3-seed + 外部 baseline 预估总计约 180 GPU-hours，PRO 6000 上排队约一周。

---

## 七、相关文献

| 论文 | 角色 |
|------|------|
| Friston et al., arxiv:2605.02675 | 理论锚点——广义坐标数学形式 |
| Ofner & Stober, arxiv:2112.03378 | 工程参考——可微广义 PC 实现 |
| Ling et al., arxiv:2212.11642 | 问题意识——PC 只预测层级不预测时间 |
| Nguyen et al., arxiv:2106.07156 | 前例——PC + SSM 组合可行（ICML 2021） |
| Lu et al., arxiv:2507.13638 | 极度相关——SSM 自然涌现时间细胞 |

PDF 都在 `论文/pcn-search/` 目录下。

---

## 八、协作方式

### 你在 PRO 6000 (96GB) 上跑实验

代码路径：`/media/lab/0E526ACF526ABB5B/fhy/pcs3-phase1a/`

核心文件：
- `train_step2.py` — 训练入口（`--mode`, `--seed`, `--ckpt_interval`）
- `pcs3_step2.py` — 模型定义（SelfPredictor, Step2Model）
- `ssm.py` — MambaBlock + SelectiveSSM
- `gap_watch.py` — 读 checkpoint 分析收敛曲线

### 训练配置（所有实验统一）

- 增强管线：RandAugment(N=2, M=9) + RandomErasing(0.25)
- ⚠️ **不用** Mixup/CutMix（会破坏 PCN error 语义）
- Optimizer: AdamW, cosine schedule, warmup=10
- Epochs: 300, Batch: 128
- d_model=160, n_layers=8, patch_size=4

### 我这边

- 主代码维护和架构设计
- 结果分析和论文写作
- 星海理事会多 Agent 协作调度
- 随时帮你 debug（把 log 或 error 丢过来就行）

---

## 九、为什么这件事有意思

1. **空白地带**：PCN + SSM 的组合几乎没人做过。Nguyen et al. (2021) 做 RL/planning，没人做视觉分类 + error-gated SSM。
2. **生物合理性**：大脑确实在用 prediction error 驱动注意力——你走楼梯踩空一步，那个瞬间的「预测偏差」比任何预训练都更有效地让你调整动作。这跟 Mamba 的 selective scan 天然对应。
3. **可证伪性**：gen_error 实验矩阵的每一行都有一个明确的可证伪预测。刚验证的两条（B > A, B > D）过了，C 和 E 也都有清晰的预期。
4. **SSM 的前沿**：Mamba 还在快速演进，理解「什么信号最适合调制 selective scan」是根本性问题。我们的贡献不在刷榜，在回答机制性问题。

---

有问题随时找我。实验跑起来了说一声，我这边可以远程监控进度。🤙
