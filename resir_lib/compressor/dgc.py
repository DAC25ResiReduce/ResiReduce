import torch

from resir_lib import Compressor


class DgcCompressor(Compressor):

    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=True)
        self.compress_ratio = compress_ratio
        self.compress_ratio = 0.01

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()
        # compress_ratio=0.001
        compress_ratio=0.01
        # compress_ratio=0.05

        sample_shape = [max(1, int(numel * compress_ratio))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        k = max(1, int(numel * self.compress_ratio * compress_ratio))
        vals, indices = torch.topk(sample_tensor.abs(), k)

        thr = vals.min()
        # thr = vals.max()
        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(2):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        indices, = torch.where(mask)
        values = tensor[indices]

        tensor_compressed = values, indices
        # ctx = shape, mask, numel

        ctx = shape, numel
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        if ctx==None:
            tensor, = tensor_compressed
            return tensor
        values, indices = tensor_compressed
        # shape, _, numel = ctx
        shape, numel = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
    
    def decompress_add(self, tensors, ctx, name):
        # numel, shape = ctx
        # shape, _, numel = ctx
        shape, numel = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values
        # 返回一个形状为为size,类型为torch.dtype,里面的每一个值都是0的tensor
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        # 填充稀疏值
        # if hvd.rank() == 0:
        #     print('values: ', values, 'indices: ', indices)
        # [a,b,    c,d]  [0,1,    0,2]
        # [c, b ,d ][a+c, b,d ]
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)
