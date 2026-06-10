import torch, time
from pcs3_step2 import Step2Model, count_params

for d in [160, 192, 224]:
    for n in [8, 10, 12]:
        try:
            m = Step2Model(d_model=d, n_layers=n, patch_size=4).cuda()
            p = count_params(m)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
            x = torch.randn(128, 3, 32, 32).cuda()
            y = torch.randint(0, 100, (128,)).cuda()
            for _ in range(3): m(x).sum().backward(); opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                out = m(x); loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward(); opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            t1 = time.time()
            per_iter = (t1-t0)/10
            per_epoch = per_iter * 50000/128 / 60
            print(f"d={d:3d} L={n:2d} p=4  {p:>12,} params  {per_iter*1000:4.0f}ms/iter  {per_epoch:.1f}min/ep  300ep={per_epoch*300/60:.1f}h")
            del m, opt; torch.cuda.empty_cache()
        except Exception as e:
            msg = "OOM" if "OOM" in str(e) else str(e)[:60]
            print(f"d={d:3d} L={n:2d} p=4  {msg}")
