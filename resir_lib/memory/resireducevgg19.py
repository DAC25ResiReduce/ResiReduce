from resir_lib import Memory
import re
import sys
import torch
import numpy as np

class ResiReduceVGG19(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma
        self.groupName_res = {}
        # VGG19重用组
        self.group = {'17': 'G1', '20': 'G1', '23': 'G1', 
                      '30': 'G2', '33': 'G2', '36': 'G2', 
                      '40': 'G3', '43': 'G3', '46': 'G3', '49': 'G3'}
        self.groupName_weight = {}
        torch.manual_seed(0)

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
            ############# ResiReduce-2d: 使用行列坐标，进行层内EF压缩
            if self.is_dim_compress(name):
                row_idx = self.residuals[name][0]
                col_idx = self.residuals[name][1]
                tensor[row_idx.unsqueeze(1), col_idx] += self.residuals[name][2]
            else:
                tensor = self.beta * self.residuals[name] + self.gamma * tensor
            return tensor
        
        ############# ResiReduce-w: 
        ### compute reuse name
        fidx, reuse_name = self.cmpt_reuse_name(name)


        ### 使用上一次迭代的 weighted avg residual 进行compensate
        if reuse_name in self.residuals:
            tensor = self.beta * self.residuals[reuse_name] + self.gamma * tensor

        return tensor  
    

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        ### compute residual
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed

        ### judge reuse name
        fidx, reuse_name = self.cmpt_reuse_name(name)

        
        # 无需重用，直接update
        if reuse_name == name and fidx == 0:
            # 不进行层内EF压缩，直接update
            if not self.is_dim_compress(name):
                self.residuals[reuse_name] = residual
            
            ############# ResiReduce-2d: 
            else:
                # numel, shape = ctx
                shape = tensor.size()
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
                self.residuals[name] = (row, col, residual[row, :][:, col])
            return
        
        
        ############# ResiReduce-w: 
        ### 每层都更新groupName_res: residual * weight
        residual = tensor - compressor.decompress(tensor_compressed, ctx)
        
        weight_l1 = torch.norm(tensor, p=1) / tensor.numel()
        weight_l2 = torch.norm(tensor, p=2) / np.sqrt(tensor.numel())
        weight = (weight_l1 * weight_l2) / 2
        
        if (reuse_name not in self.groupName_res):
            # (sum)
            self.groupName_res[reuse_name] = 0
            self.groupName_weight[reuse_name] = 0
        self.groupName_res[reuse_name] += residual * weight
        self.groupName_weight[reuse_name] += weight

        ### 反向传播至group第一层时，使用groupName_res更新avg_residual（residuals）
        if fidx == '17' or fidx == '30' or fidx == '40':
            self.residuals[reuse_name] = self.groupName_res[reuse_name] / self.groupName_weight[reuse_name]
            self.groupName_res[reuse_name] = 0
            self.groupName_weight[reuse_name] = 0
    
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
