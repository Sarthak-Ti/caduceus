#have to input data like the model and all that

import torch
from src.models.sequence.conv import GraphRegConvNet
# from src.dataloaders.datasets.graphreg_dataset import GraphRegDataset as d
# dataset = d.GraphRegDataset('test', 100_000)

input_shape = [4,100_000]

model = GraphRegConvNet().cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
while True:
    try:
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward + Backward Pass
        output = model(dummy_input)
        loss = output.sum()  # Dummy loss
        loss.backward()
        optimizer.step()
        
        # Report GPU memory usage
        memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Batch size {batch_size}: {memory_used:.2f} MB")
        
        # Increment batch size
        batch_size += 1
    except RuntimeError as e:
        print(e)
        print(f"OOM at batch size {batch_size}")
        break
