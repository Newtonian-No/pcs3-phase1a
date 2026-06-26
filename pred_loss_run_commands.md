# K=1/K=2/K=3 Prediction Loss 实验命令

本文列出 `pred_loss` 相关实验命令。K=1 和 K=2 的 CE-only 对照如果已经跑过，可以不重复；K=3 需要先跑 `gen_error_k3` 作为 CE-only 对照，再跑三个 pred-loss 权重。

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

新增可配置参数：

```text
--lambda_pred : prediction loss 总权重
--lambda_vel  : K=2/K=3 中速度预测损失权重，默认 0.5
--lambda_acc  : K=3 中加速度预测损失权重，默认 0.25
```

实验名会自动设置默认 `lambda_pred`：

```text
*_pred001 -> lambda_pred=0.001
*_pred003 -> lambda_pred=0.003
*_pred01  -> lambda_pred=0.01
```

也可以手动覆盖，例如：

```bash
python3 train_step2.py --mode gen_error_k3_pred001 --lambda_pred 0.003 --lambda_vel 1.0 --lambda_acc 0.5
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
| `gen_error_k3` | 3 | 0 | `L = L_cls` |
| `gen_error_k3_pred001` | 3 | 0.001 | `L = L_cls + 0.001 * (L_pos + 0.5 * L_vel + 0.25 * L_acc)` |
| `gen_error_k3_pred003` | 3 | 0.003 | `L = L_cls + 0.003 * (L_pos + 0.5 * L_vel + 0.25 * L_acc)` |
| `gen_error_k3_pred01` | 3 | 0.01 | `L = L_cls + 0.01 * (L_pos + 0.5 * L_vel + 0.25 * L_acc)` |

其中：

```text
L_cls = CrossEntropy(logits, y)
L_pos = SmoothL1(X_hat, X.detach())
L_vel = SmoothL1(V_hat, V.detach())
L_acc = SmoothL1(A_hat, A.detach())
V     = X 的一阶 token 差分
A     = X 的二阶 token 差分
```

---

## 推荐先短跑检查

先用 30 epoch 检查代码、loss 曲线和 error norm，再跑 300 epoch。

```bash
python3 train_step2.py --mode gen_error_k3 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4

python3 train_step2.py --mode gen_error_k3_pred001 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4

python3 train_step2.py --mode gen_error_k3_pred003 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4

python3 train_step2.py --mode gen_error_k3_pred01 --epochs 30 --seed 42 --batch_size 128 --d_model 160 --n_layers 8 --patch_size 4
```

观察重点：

```text
train_pred_loss 是否下降
train_pos_loss 是否下降
train_vel_loss 是否下降
train_acc_loss 是否下降
train_error_norm 是否塌缩到接近 0
test_acc 是否明显低于 CE-only 对照
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

### K=3 CE-only 对照

```bash
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

### K=3 pred-loss sweep

```bash
nohup python3 train_step2.py \
  --mode gen_error_k3_pred001 \
  --seed 42 \
  --epochs 300 \
  --batch_size 128 \
  --d_model 160 \
  --n_layers 8 \
  --patch_size 4 \
  >> logs/gen_error_k3_pred001_s42.log 2>&1 &
```

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

