import torch
from resir_lib import Compressor

class RandomKCompressor(Compressor):
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    def __init__(self, compress_ratio, rank):
        super().__init__()
        self.global_step = 0
        self.compress_ratio = compress_ratio
        
        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]

        self.tensor_flatten_np_arr=[]
        self.values_flatten_global_np_arr=[]
        self.values_flatten_channel_np_arr=[]

    def sparsify(self, tensor, compress_ratio):
        compress_ratio=0.01
        
        
        tensor = tensor.flatten()
        numel = tensor.numel()
        k = max(1, int(numel * compress_ratio))
        indices = torch.randperm(numel, device=tensor.device)[:k]
        values = tensor[indices]
        return indices, values
    
    def compress(self, tensor, name):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        indices, values = self.sparsify(tensor, self.compress_ratio)

        ctx = indices, tensor.numel(), tensor.size()
        return [values], ctx

    def decompress(self, tensors, ctx, name):
        if ctx==None:
            tensor, = tensors
            return tensor
        
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, numel, shape = ctx
        values, = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
