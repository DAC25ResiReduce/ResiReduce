import torch

from resir_lib import Compressor

class QSGDCompressor(Compressor):

    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        ctx = tensor.numel(), tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm(p=float('inf'))
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        numel, shape = ctx
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed

    def decompress_add(self, tensor_compressed, ctx, name):
        numel, shape = ctx
        tensor_compressed, norm = tensor_compressed
        
        tensor_chunk = torch.chunk(tensor_compressed, 8)
        norm_chunk = torch.chunk(norm, 8)
        tensor_decompressed = torch.zeros(shape, dtype=torch.float32).cuda()

        for i, tensor_compressed in enumerate(tensor_chunk):    
            decode_output = tensor_compressed.type(torch.float32)
            tensor = norm_chunk[i] / self.quantum_num * decode_output
            tensor_decompressed += tensor.view(shape)
        
        return tensor_decompressed