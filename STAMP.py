import torch  # 如果你使用PyTorch

model = torch.load('final.ckpt')

model.eval()
with torch.no_grad():
    outputs = model(your_data)




