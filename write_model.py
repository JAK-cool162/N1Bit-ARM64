import os

# We will read n1bit/model.py, replace the transpositions with standard run_matmul order,
# or we can rewrite the NumPyBitRNNLM train_step function completely to be 100% standard!
with open("n1bit/model.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace the transposed run_matmul blocks with standard mathematically correct dot layouts
code = code.replace(
    'proj_xh = self.vulkan.run_matmul(x_emb[t], W_xh_f)',
    'proj_xh = self.vulkan.run_matmul(W_xh_f, x_emb[t])'
)
code = code.replace(
    'proj_hh = self.vulkan.run_matmul(h[t-1], W_hh_f)',
    'proj_hh = self.vulkan.run_matmul(W_hh_f, h[t-1])'
)
code = code.replace(
    'logits_t = self.vulkan.run_matmul(h[t], W_hy_f)',
    'logits_t = self.vulkan.run_matmul(W_hy_f, h[t])'
)
code = code.replace(
    'dW_hy += self.vulkan.run_matmul(h[t], dy.T).T',
    'dW_hy += self.vulkan.run_matmul(dy, h[t].T)'
)
code = code.replace(
    'dh = self.vulkan.run_matmul(dy.T, W_hy_f).T + dh_next',
    'dh = self.vulkan.run_matmul(W_hy_f.T, dy) + dh_next'
)
code = code.replace(
    'dW_xh += self.vulkan.run_matmul(x_emb[t], da.T).T',
    'dW_xh += self.vulkan.run_matmul(da, x_emb[t].T)'
)
code = code.replace(
    'dW_hh += self.vulkan.run_matmul(h[t-1], da.T).T',
    'dW_hh += self.vulkan.run_matmul(da, h[t-1].T)'
)
code = code.replace(
    'dx = self.vulkan.run_matmul(da.T, W_xh_f).T',
    'dx = self.vulkan.run_matmul(W_xh_f.T, da)'
)
code = code.replace(
    'dh_next = self.vulkan.run_matmul(da.T, W_hh_f).T',
    'dh_next = self.vulkan.run_matmul(W_hh_f.T, da)'
)

with open("n1bit/model.py", "w", encoding="utf-8") as f:
    f.write(code)

print("SUCCESS")
