import torch
from llama import Transformer, LlamaModelConfig
import tiktoken
import time
from loader import DataLoaderLite

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps"

enc = tiktoken.get_encoding("gpt2")

config = LlamaModelConfig(
    dim=1024,
    n_layers=12,
    n_heads=8,
    device=torch.device(DEVICE),
    vocab_size=enc.n_vocab,
)
model = Transformer(config).to(config.device)
# model = torch.compile(model)

BATCH_SIZE = 16

train_loader = DataLoaderLite(B=BATCH_SIZE, T=512, enc=enc)


# fused is later update in pytorch adamw implementation with kernel fusion
# can betas here silar to nangorad (0.9, 0.95), esp=1e-8
# can also implement weight decay
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # fused=True
if DEVICE == "mps":
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
elif DEVICE == "cuda":
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
else:
    pass

for i in range(5000):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(config.device), y.to(config.device)
    optimizer.zero_grad()
    if DEVICE == "mps":
        output, loss = model(x, y)
    elif DEVICE == "cuda":
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            output, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # also could use consine lr scheduler
    optimizer.step()
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()
    else:
        pass
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_second = (train_loader.B * train_loader.T) / (t1 - t0)
    mfu = model.estimate_mfu(BATCH_SIZE, dt)
    print(
        f"step {i}, loss {loss.item()} in {dt:.2f}ms, norm {norm:.4f}, tokens/s {tokens_per_second:.2f}, mfu {mfu*100:.2f}%"
    )
