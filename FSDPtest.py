import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as L
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.demos import Transformer, WikiText2


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(  # 1B parameters
            vocab_size=vocab_size,
            nlayers=32,
            nhid=4096,
            ninp=1024,
            nhead=64,
        )

    def training_step(self, batch):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)
    
    def on_after_backward(self):
        if self.global_step % 100 == 0:  # Print every 100 steps
            print(f"Step {self.global_step}:")
            print_gpu_memory_usage()


def print_gpu_memory_usage():
    print("\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"GPU {i}: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")

L.seed_everything(42)

# Data
dataset = WikiText2()
train_dataloader = DataLoader(dataset, batch_size=32) #this will show the tqdm and we'll see if it shoudl be reduced or not
print('num samples:', len(train_dataloader))

# Model
model = LanguageModel(vocab_size=dataset.vocab_size)

# Trainer
# trainer = L.Trainer(accelerator="cuda", devices=1, strategy='auto')
trainer = L.Trainer(accelerator="cuda", devices=2, strategy=FSDPStrategy(sharding_strategy='FULL_SHARD', cpu_offload=True)) #this is the optimal one for lowest memory

#with full shard we get a total of 933 steps, we see 4 GB on GPU 0 and 14 max memory allocated. but it's weird because it's split and shows it twice
#by epoch 100 it goes up to 8 used and 19 as the max, but that seems fine
#.97 it/s

#let's set shard to NO_SHARD and see what happens
#still 933 steps, at step 0 see 8 and 12 GB for memory and max. it's a bit faster too
#for epoch 100 it goes up to 16 and 23, so we definitely are seeing some savings!
#1.09 it/s

#finally let's see what happens if we do ddp with 2 devices. still 933 steps, 1.5 it/s
#see a good speedup, and see that at epoch 0 it is 12 and 15
#for the epoch 100 it is 20 and 28. fully duplicated

#so still ddp but now just 1 device. weird to do ddp, but now is 1865 steps, 1.85 it/s so actually slower, similar to fsdp
#memory usage is 12 and 15 then 20 and 28, 
#but yeah it shows double the steps, still ddp makes it take more memory
#if we do strategy auto, we see 8 and 11 at first, then 16 and 23 later on. This is edentical ish to fsdp with no sharding... huh
#still 1865 steps

#key takeaways
#the main thing is when doing fsdp we see a good memory reduction, but it also divides the number of steps by 2, so maybe my fsdp is working??
#but we go any larger than what hyena can handle on one device and it stops working? that's really strange and I have no clue why?

#final is 2 devices with full shard and cpu offload, let's see the memory savings
#starts at 0 and 14 again, but the amount used is near 0. 

trainer.fit(model, train_dataloader)
trainer.print(torch.cuda.memory_summary())