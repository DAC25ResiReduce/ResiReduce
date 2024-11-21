from resir_lib import Memory
import torch
import sys


class DimCompression(Memory):
    def __init__(self, beta=1.0, gamma=1.0, ratio=0.5, hrank=0):
        self.residuals = {}
        # residuals = {
        #     'name1': ( tensor([row1, row2]), tensor([col1, col2]), tensor([[value1, value2], [value3, value4]]) )
        #     'name2': ( tensor([row1, row2]), tensor([col1, col2]), tensor([[value1, value2], [value3, value4]]) )
        # }

        self.beta = beta
        self.gamma = gamma

        # 层内EF压缩率
        self.ratio = 0.7
        # sys.getsizeof(self.residuals)
        self.memory_size = -1

        # hvd.rank()用于初始化随机种子
        self.hrank = hrank
        torch.manual_seed(0)


    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
    
        if name in self.residuals:
            # 使用行列坐标，进行层内EF压缩
            if self.is_dim_compress(name):
                row_idx = self.residuals[name][0]
                col_idx = self.residuals[name][1]
                tensor[row_idx.unsqueeze(1), col_idx] += self.residuals[name][2]
                
            else:
                tensor = self.beta * self.residuals[name] + self.gamma * tensor
        return tensor
    

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        residual = tensor - compressor.decompress(tensor_compressed, ctx)
        numel, shape = ctx

        # 不进行层内EF压缩
        if not self.is_dim_compress(name):
            self.residuals[name] = residual
            return
        
        
        row_ratio = 0.5
        col_ratio = 0.5
        
        
        row_num = int(shape[0] * row_ratio)
        col_num = int(shape[1] * col_ratio)

        # 计算第一维张量的L_n范数, 其中n是范数的阶数
        n = 1
        # n = float('inf')
        
        # tensor.dim() == 2:
        row_ln_norms = torch.norm(tensor, p=n, dim=1)
        col_ln_norms = torch.norm(tensor, p=n, dim=0)

        # 使用softmax来将最大值转换为概率分布，softmax 函数可以确保总概率为1
        row_probabilities = torch.softmax(row_ln_norms, dim=0)
        col_probabilities = torch.softmax(col_ln_norms, dim=0)
        
        row = torch.multinomial(row_probabilities, row_num, replacement=False)
        col = torch.multinomial(col_probabilities, col_num, replacement=False)
        
        
        # # 创建一个布尔掩码，初始假设所有行都保留
        # mask = torch.ones(residual.size(0), dtype=torch.bool)
        # # 将需要排除的行的掩码设置为False
        # mask[row] = False
        # # 使用掩码来索引原始张量，获取除了指定行之外的所有行
        # # result_tensor = residual[mask]
        # self.residuals[name] = (row, col, residual[mask][:, col])

        self.residuals[name] = (row, col, residual[row, :][:, col])


    def get_memory_usage(self):
        total_memory_usage = 0
        for name, tensor in self.residuals.items():
            element_size = tensor.element_size()  # 每个元素占用的字节数
            numel = tensor.numel()  # 张量中元素的总数
            memory_usage = element_size * numel  # 张量占用的总字节数
            total_memory_usage += memory_usage
        self.memory_size =  total_memory_usage


    def is_dim_compress(self, name):
        # 全部压缩
        # return True
    
        # VGG19: 全连接层
        # if 'classifier' in name and 'weight' in name:
        if name == 'classifier.0.weight':
        # if 'classifier.0.weight' in name or 'classifier.3.weight' in name or 'classifier.6.weight' in name:
            return True
        
        # Bert: WTE, WPE
        # if 'bert.pooler.dense_act.weight' in name or 'word_embeddings' in name or 'position_embeddings' in name:
        if 'word_embeddings' in name :
            return True
        
        # GPT2: WTE, WPE
        # if 'wpe' in name or 'wte' in name:
        if 'wte' in name:
            return True

        return False

    

    def init_seed(self):
        torch.manual_seed(0)


    def print_ef(self, epoch, iter):
        
        print("len(self.residuals): ", len(self.residuals))

        for name, res in self.residuals.items():
            if self.is_dim_compress(name):
                print(f"name: {name}, len: {len(res)}, row: {res[0].numel()}, col: {res[1].numel()}")
                # print(f"hrank: {self.hrank}, name: {name}, dim: {res[0]}")
            else:    
                print("name: ", name, ", size: ", res.size(), ", numel: ", res.numel())
            
        total_memory_usage = 0
        for name, tensor in self.residuals.items():
            # 不进行层内EF压缩
            if not self.is_dim_compress(name):
                element_size = tensor.element_size()  # 每个元素占用的字节数
                numel = tensor.numel()  # 张量中元素的总数
                memory_usage = element_size * numel  # 张量占用的总字节数
            # 进行层内EF压缩
            else:
                # print("dim : ", tensor[0])
                # print("tensor_dim : ", tensor[1])
                memory_usage = tensor[2].element_size() * tensor[2].numel()
                

            total_memory_usage += memory_usage
        
        print("Memory size: ", total_memory_usage, "B")
        print("Memory size: ", total_memory_usage / (1024 * 1024), "MB")
        print("Residuals size: ", sys.getsizeof(total_memory_usage), "B")

        return 1   
