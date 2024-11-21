# Copyright (c) 2020, King Abdullah University of Science and Technology (KAUST)  

from abc import ABC, abstractmethod
from .optimizer import DistributedOptimizer
# from .optimizer_reuse_avg import DistributedOptimizer

class Memory(ABC):
    @abstractmethod
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

    def gUpdate(self, tensor_agg, name):
        """Update the global residuals."""
        pass


class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average

        self.tensors_size_are_same = tensors_size_are_same
        # self.tensors_size_are_same = False

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)