from resir_lib import Memory
import re
import sys
import numpy as np
import os

class ResidualRN50avgMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma
        
        # 用于update group的residual
        self.cnt_res = 0
        self.sum_res = 0.0
        self.groupName_res = {}
        self.OR = {}
        self.epoch = 0
        
    
    def set_epoch(self, epoch):
        self.epoch = epoch


    def cmpt_reuse_name(self, name):
        # 定义正则表达式模式
        pattern = r"conv(\d+)_x\.(\d+)\.residual_function\.(0|3|6)\.weight" 
        match = re.match(pattern, name)
        
        # 无需重用EF内存
        if not match:
            reuse_name = name
            name_ctx = None
        # 重用EF内存的tensor，计算reuse_name
        else:
            conv_num = match.group(1)
            x_num = match.group(2)
            function_num = match.group(3)
            name_ctx = conv_num, x_num, function_num
            
            # 特殊处理group第1个tensor
            if x_num == '0' and function_num == '0':
                reuse_name = name
                name_ctx = None
            
            # reuse_name: conv_(2|3|4|5).residual_function_(0|3|6)
            else:
                reuse_name = 'conv_' + conv_num + '.residual_function_' + function_num
        
        return reuse_name, name_ctx
    
    # 判断resue_name是否进行了重用
    def is_reuse(self, reuse_name):
        # 定义正则表达式模式
        pattern = r"conv_(\d+)\.residual_function_(0|3|6)" 
        match = re.match(pattern, reuse_name)
        if match:
            return True
        return False


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
        reuse_name, name_ctx = self.cmpt_reuse_name(name)

        ### 不进行EF
        if reuse_name == 'none EF':
            return tensor
        
        ### 使用上一次迭代的avg_residual进行compensate
        if reuse_name in self.residuals:
            tensor = self.beta * self.residuals[reuse_name] + self.gamma * tensor

        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""

        ### judge reuse name
        reuse_name, name_ctx = self.cmpt_reuse_name(name)

        ### 不进行EF
        if reuse_name == 'none EF':
            return

        ### 不在重用group中，直接update
        if not name_ctx:
            self.residuals[reuse_name] = tensor - compressor.decompress(tensor_compressed, ctx)
            return
        
        ### 每层都更新groupName_res，累加residual
        residual = tensor - compressor.decompress(tensor_compressed, ctx)
        if (reuse_name not in self.groupName_res):
            # (sum)
            self.groupName_res[reuse_name] = 0
        self.groupName_res[reuse_name] += residual
        
        # 保存原始残差OR，用于打印输出
        # self.OR[name] = residual

        ### 反向传播至group第一层时，使用groupName_res更新avg_residual（residuals）
        conv_num, x_num, function_num = name_ctx
        
        block_num = 0
        if conv_num == '5' or conv_num == '2':
            block_num = 3
        elif conv_num == '3':
            block_num = 4
        elif conv_num == '4':
            block_num = 6
        
        ## 特殊处理func0
        if function_num == '0':
            if x_num == '1':
                self.residuals[reuse_name] = self.groupName_res[reuse_name] / (block_num - 1)
                self.groupName_res[reuse_name] = 0
        elif x_num == '0':
                if (block_num == 0):
                    print("ERROR! block_num == 0!")
                else:
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



        print("OK!")
        return 1    
