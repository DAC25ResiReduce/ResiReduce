from resir_lib import Memory
import torch
import sys


class MemoryTradeoff(Memory):
    def __init__(self, beta=1.0, gamma=1.0, percent=0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma

        self.percent = percent
        # sys.getsizeof(self.residuals)
        self.memory_size = -1
        self.partial_indices = {}

        torch.manual_seed(0)


    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
    
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + self.gamma * tensor
        return tensor
    

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed

        # 将 residual 中 percent 的元素置零
        if self.percent == 1:
            # 全部置零，相当于none EF
            residual.zero_()
        elif self.percent != 0:
            numel = residual.numel()
            if name not in self.partial_indices:
                num_zeros = int(numel * self.percent)
                self.partial_indices[name] = torch.randperm(numel, device=tensor.device)[:num_zeros].cuda()
                # self.partial_indices[name] = torch.randperm(numel)[:num_zeros]
                # indices = torch.randint(0, numel, (num_zeros,)).cuda()
            indices = self.partial_indices[name]

            # num_zeros = int(numel * self.percent)
            # indices = torch.randperm(numel, device=tensor.device)[:num_zeros].cuda()

            residual.view(-1)[indices] = 0
        
        self.residuals[name] = residual


    def get_memory_usage(self):
        total_memory_usage = 0
        for name, tensor in self.residuals.items():
            element_size = tensor.element_size()  # 每个元素占用的字节数
            numel = tensor.numel()  # 张量中元素的总数
            memory_usage = element_size * numel  # 张量占用的总字节数
            total_memory_usage += memory_usage
        self.memory_size =  total_memory_usage
    
    
    def clear_partial_indices(self):
        self.partial_indices = {}


    def print_ef(self, epoch, iter):
        
        print("len(self.residuals): ", len(self.residuals))

        for name, res in self.residuals.items():
            print("name: ", name, ", size: ", res.size(), ", numel: ", res.numel())
            
        total_memory_usage = 0
        for name, tensor in self.residuals.items():
            element_size = tensor.element_size()  # 每个元素占用的字节数
            numel = tensor.numel()  # 张量中元素的总数
            memory_usage = element_size * numel  # 张量占用的总字节数
            total_memory_usage += memory_usage
            print("name : ", name, "numel : ", self.partial_indices[name].numel(), "index : ", self.partial_indices[name])

        print("Memory size: ", total_memory_usage, "B")
        print("Memory size: ", total_memory_usage / (1024 * 1024), "MB")
        print("Residuals size: ", sys.getsizeof(total_memory_usage), "B")

        return 1    
