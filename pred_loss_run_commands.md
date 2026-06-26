# K=3 Prediction Loss 实验命令

本文只列当前需要跑的 3 个 K=3 实验：

```text
gen_error_k3_pred003
gen_error_k3_pred01
gen_error_k3_shuffled_pred003
```

K=1、K=2 的相关实验不在本文重复列出。

---

## 统一配置

保持之前 Step 2 配置不变：

```text
epochs=300
batch_size=128
d_model=160
n_layers=8
patch_size=4
seed=42
augmentation=RandAugment(N=2,M=9) + RandomErasing(0.25)
optimizer=AdamW
schedule=cosine
```

K=3 prediction loss 使用：

```text
L = L_cls + lambda_pred * (L_pos + 0.5 * L_vel + 0.25 * L_acc)
```

其中：

```text
L_cls = CrossEntropy(logits, y)
L_pos = SmoothL1(X_hat, X.detach())
L_vel = SmoothL1(V_hat, V.detach())
L_acc = SmoothL1(A_hat, A.detach())
V     = X 的一阶 token 差分
A     = X 的二阶 token 差分
```

默认权重：

```text
lambda_vel = 0.5
lambda_acc = 0.25
```

---

## 实验表

| 实验名 | K | lambda_pred | 是否 shuffled | 损失函数 |
|---|---:|---:|---|---|
| `gen_error_k3_pred003` | 3 | 0.003 | 否 | `L = L_cls + 0.003 * (L_pos + 0.5 * L_vel + 0.25 * L_acc)` |
| `gen_error_k3_pred01` | 3 | 0.01 | 否 | `L = L_cls + 0.01 * (L_pos + 0.5 * L_vel + 0.25 * L_acc)` |
| `gen_error_k3_shuffled_pred003` | 3 | 0.003 | 是 | `L = L_cls + 0.003 * (L_pos + 0.5 * L_vel + 0.25 * L_acc)` |

`gen_error_k3_shuffled_pred003` 会打乱 error 和样本的对应关系，用来对照：

```text
K=3 pred003 的提升是否来自真实 error 结构，而不是额外通道或正则化。
```

---

## 推荐先短跑检查

正式 300 epoch 前，建议先各跑 30 epoch：

```bash
python3 train_step2.py --mode gen_error_k3_pred003 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4

python3 train_step2.py --mode gen_error_k3_pred01 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4

python3 train_step2.py --mode gen_error_k3_shuffled_pred003 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4
```

观察重点：

```text
train_pred_loss 是否下降
train_pos_loss 是否下降
train_vel_loss 是否下降
train_acc_loss 是否下降
train_error_norm 是否塌缩到接近 0
test_acc 是否明显低于已有 CE-only / K=2 对照
```

---

## 300 epoch 正式命令

先进入原 md 文档指定环境：

```bash
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate evuav
cd /media/lab/0E526ACF526ABB5B/fhy/pcs3-phase1a
mkdir -p logs
```

### gen_error_k3_pred003

```bash
nohup python3 train_step2.py \
  --mode gen_error_k3_pred003 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k3_pred003_s42.log 2>&1 &
```

### gen_error_k3_pred01

```bash
nohup python3 train_step2.py \
  --mode gen_error_k3_pred01 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k3_pred01_s42.log 2>&1 &
```

### gen_error_k3_shuffled_pred003

```bash
nohup python3 train_step2.py \
  --mode gen_error_k3_shuffled_pred003 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k3_shuffled_pred003_s42.log 2>&1 &
```

---

## 手动改权重

如需覆盖默认权重，可以显式传参：

```bash
python3 train_step2.py \
  --mode gen_error_k3_pred003 \
  --lambda_pred 0.003 \
  --lambda_vel 0.5 \
  --lambda_acc 0.25 \
  --epochs 30 \
  --seed 42 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4
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
lambda_acc
train_pred_loss
train_pos_loss
train_vel_loss
train_acc_loss
train_error_norm
train_vel_error_norm
train_acc_error_norm
best_acc
best_epoch
```

