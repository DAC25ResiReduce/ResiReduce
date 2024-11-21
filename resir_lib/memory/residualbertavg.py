from resir_lib import Memory
import re
import sys

class ResidualBertavgMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma
        self.groupName_res = {}


    def cmpt_reuse_name(self, name):
        # 定义正则表达式模式
        # pattern = r"transformer\.h\.(\d+)\.(.+)"
        pattern = r"bert\.encoder\.layer\.(\d+)\.(.+)"
        match = re.match(pattern, name)

        # 无需重用EF内存
        if not match:
            reuse_name = name
            layer = -1
        # 重用EF内存的tensor，计算reuse_name
        else:
            layer = int(match.group(1))
            # 4*4 groups in total: G0-., ..., G3-..
            reuse_name = 'G' + str(int(layer / 3)) + '-' + match.group(2)
        
        return layer, reuse_name


    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        ### residuals为空，直接返回
        if not self.residuals:
            return tensor
        
        ### 存在name，直接compensate
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + self.gamma * tensor
            return tensor
        
        ### compute reuse name
        layer, reuse_name = self.cmpt_reuse_name(name)

        ### 不进行EF
        if reuse_name == 'none EF':
            return tensor

        ### 使用上一次迭代的avg_residual进行compensate
        if reuse_name in self.residuals:
            tensor = self.beta * self.residuals[reuse_name] + self.gamma * tensor

        return tensor  
    

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        ### compute residual
        # tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        # residual = tensor - tensor_decompressed

        ### judge reuse name
        layer, reuse_name = self.cmpt_reuse_name(name)

        ### 不进行EF
        if reuse_name == 'none EF':
            return 
        
        # 无需重用，直接update
        if reuse_name == name and layer == -1:
            self.residuals[reuse_name] = tensor - compressor.decompress(tensor_compressed, ctx)
            return
        

        ### 每层都更新groupName_res，累加residual
        residual = tensor - compressor.decompress(tensor_compressed, ctx)
        if (reuse_name not in self.groupName_res):
            # (sum)
            self.groupName_res[reuse_name] = 0
        self.groupName_res[reuse_name] += residual


        ### 反向传播至group第一层时，使用groupName_res更新avg_residual（residuals）
        if (layer % 3) == 0:
            self.residuals[reuse_name] = self.groupName_res[reuse_name] / 3
            self.groupName_res[reuse_name] = 0
        

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
        print("Memory size: ", total_memory_usage, "B")
        print("Memory size: ", total_memory_usage / (1024 * 1024), "MB")
        print("Residuals size: ", sys.getsizeof(total_memory_usage), "B")

        return 1    
