from resir_lib import Memory


class NoneMemory(Memory):
    
    
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

    def gUpdate(self, tensor_agg, name):
        """Update the global residuals."""
        pass

    

