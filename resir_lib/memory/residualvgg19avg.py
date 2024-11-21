from resir_lib import Memory
import re
import sys

class ResidualVGG19avgMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma
        self.groupName_res = {}
        # VGG19重用组
        self.group = {'17': 'G1', '20': 'G1', '23': 'G1', 
                      '30': 'G2', '33': 'G2', '36': 'G2', 
                      '40': 'G3', '43': 'G3', '46': 'G3', '49': 'G3'}

    def cmpt_reuse_name(self, name):
        # 定义正则表达式模式
        pattern = r"features\.(\d+)\.weight"
        match = re.match(pattern, name)

        # 无需重用EF内存
        if not match:
            reuse_name = name
            fidx = 0
        # 重用EF内存的tensor，计算reuse_name
        else:
            fidx = match.group(1)
            
            # 不属于group，无需重用
            if (fidx not in self.group):
                reuse_name = name
                fidx = 0
            # reuse_name: Gx
            else:
                reuse_name = self.group[fidx]
        
        return fidx, reuse_name


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
        fidx, reuse_name = self.cmpt_reuse_name(name)

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
        fidx, reuse_name = self.cmpt_reuse_name(name)

        ### 不进行EF
        if reuse_name == 'none EF':
            return 
        
        # 无需重用，直接update
        if reuse_name == name and fidx == 0:
            self.residuals[reuse_name] = tensor - compressor.decompress(tensor_compressed, ctx)
            return
        
        ### 每层都更新groupName_res，累加residual
        residual = tensor - compressor.decompress(tensor_compressed, ctx)
        if (reuse_name not in self.groupName_res):
            # (sum)
            self.groupName_res[reuse_name] = 0
        self.groupName_res[reuse_name] += residual

        ## 计算重用数量
        block_num = 0
        if fidx == '17' or fidx == '30':
            block_num = 3
        elif fidx == '40':
            block_num = 4

        ### 反向传播至group第一层时，使用groupName_res更新avg_residual（residuals）
        if fidx == '17' or fidx == '30' or fidx == '40':
            self.residuals[reuse_name] = self.groupName_res[reuse_name] / block_num
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
