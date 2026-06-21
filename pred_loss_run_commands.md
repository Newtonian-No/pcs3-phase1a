# K=1/K=2 Prediction Loss 实验命令

本文列出 6 个新增 `pred_loss` 实验。旧的 CE-only 对照已经跑过，因此这里不重复列 `gen_error_k1` 和 `gen_error_k2`。

---

## 统一配置

所有实验保持之前 Step 2 配置不变：

```text
epochs=300
batch_size=128
d_model=160
n_layers=8
patch_size=4
seed=42
optimizer=AdamW
schedule=cosine
augmentation=RandAugment(N=2,M=9) + RandomErasing(0.25)
```

新增参数：

```text
--lambda_pred : prediction loss 总权重
--lambda_vel  : K=2 中速度预测损失权重，默认 0.5
```

6 个实验名已经内置默认 `lambda_pred`，例如：

```bash
--mode gen_error_k1_pred001
```

会自动使用：

```text
lambda_pred=0.001
```

如果需要手动覆盖，也可以显式传入：

```bash
--lambda_pred 0.003
```

---

## 损失函数表

| 实验名 | K | lambda_pred | 损失函数 |
|---|---:|---:|---|
| `gen_error_k1_pred001` | 1 | 0.001 | `L = L_cls + 0.001 * L_pos` |
| `gen_error_k1_pred003` | 1 | 0.003 | `L = L_cls + 0.003 * L_pos` |
| `gen_error_k1_pred01` | 1 | 0.01 | `L = L_cls + 0.01 * L_pos` |
| `gen_error_k2_pred001` | 2 | 0.001 | `L = L_cls + 0.001 * (L_pos + 0.5 * L_vel)` |
| `gen_error_k2_pred003` | 2 | 0.003 | `L = L_cls + 0.003 * (L_pos + 0.5 * L_vel)` |
| `gen_error_k2_pred01` | 2 | 0.01 | `L = L_cls + 0.01 * (L_pos + 0.5 * L_vel)` |

其中：

```text
L_cls = CrossEntropy(logits, y)
L_pos = SmoothL1(X_hat, X.detach())
L_vel = SmoothL1(V_hat, V.detach())
V     = X 的一阶 token 差分
```

---

## 推荐先短跑检查

先用 30 epoch 检查代码、loss 曲线和 `error_norm`，不要直接烧 300 epoch。

```bash
python3 train_step2.py --mode gen_error_k1_pred001 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4 --ckpt_interval 5

python3 train_step2.py --mode gen_error_k1_pred003 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4 --ckpt_interval 5

python3 train_step2.py --mode gen_error_k1_pred01 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4 --ckpt_interval 5

python3 train_step2.py --mode gen_error_k2_pred001 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4 --ckpt_interval 5

python3 train_step2.py --mode gen_error_k2_pred003 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4 --ckpt_interval 5

python3 train_step2.py --mode gen_error_k2_pred01 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4 --ckpt_interval 5
```

观察重点：

```text
train_pred_loss 是否下降
train_pos_loss 是否下降
train_vel_loss 是否下降
train_error_norm 是否快速塌缩到接近 0
test_acc 是否明显低于 CE-only 对照
```

---

## 300 epoch 正式命令

建议串行跑，避免单卡显存和调度冲突。

先进入原 md 文档指定的实验环境：

```bash
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate evuav
cd /media/lab/0E526ACF526ABB5B/fhy/pcs3-phase1a
```

先确认日志目录存在：

```bash
mkdir -p logs
```

```bash
nohup python3 train_step2.py \
  --mode gen_error_k1_pred001 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k1_pred001_s42.log 2>&1 &
```

```bash
nohup python3 train_step2.py \
  --mode gen_error_k1_pred003 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k1_pred003_s42.log 2>&1 &
```

```bash
nohup python3 train_step2.py \
  --mode gen_error_k1_pred01 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k1_pred01_s42.log 2>&1 &
```

```bash
nohup python3 train_step2.py \
  --mode gen_error_k2_pred001 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k2_pred001_s42.log 2>&1 &
```

```bash
nohup python3 train_step2.py \
  --mode gen_error_k2_pred003 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k2_pred003_s42.log 2>&1 &
```

```bash
nohup python3 train_step2.py \
  --mode gen_error_k2_pred01 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k2_pred01_s42.log 2>&1 &
```

---

## 手动改损失权重示例

如果不想新建 mode 名，可以用同一个 mode 手动覆盖：

```bash
python3 train_step2.py \
  --mode gen_error_k2_pred001 \
  --lambda_pred 0.003 \
  --lambda_vel 1.0 \
  --epochs 30 \
  --seed 42 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4
```

这条命令实际使用：

```text
L = L_cls + 0.003 * (L_pos + 1.0 * L_vel)
```

---

## 结果文件

每个实验会输出：

```text
results/step2_{mode}_seed42_*.json
results/step2_{mode}_seed42.checkpoint.json
```

JSON 中会保存：

```text
lambda_pred
lambda_vel
train_pred_loss
train_pos_loss
train_vel_loss
train_error_norm
train_vel_error_norm
best_acc
best_epoch
```
