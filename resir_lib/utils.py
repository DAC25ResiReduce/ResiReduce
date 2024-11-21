import horovod.torch as hvd

def find_duplicates(lst):
    seen = set()
    dups = set()
    for el in lst:
        if el in seen:
            dups.add(el)
        seen.add(el)
    return dups

# return communication mode: allreduce or allgather
def get_comm(params):
    comm_name = params.get('comm_mode', 'allreduce')
    return comm_name

def get_compressor(params):
    compress_name = params.get('compressor', 'none')
    
    if compress_name == 'none':
        from resir_lib.compressor.none import NoneCompressor
        compressor = NoneCompressor()

    elif compress_name == 'dgc':
        from resir_lib.compressor.dgc import DgcCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = DgcCompressor(compress_ratio)
    
    elif compress_name == 'topk':
        from resir_lib.compressor.topk import TopKCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = TopKCompressor(compress_ratio)
    
    elif compress_name == 'randomk':
        from resir_lib.compressor.randomk import RandomKCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = RandomKCompressor(compress_ratio,rank=hvd.rank())
        
    elif compress_name == 'qsgd':
        from resir_lib.compressor.qsgd import QSGDCompressor
        compressor = QSGDCompressor(127)
        
    elif compress_name == 'fp16':
        from resir_lib.compressor.fp16 import FP16Compressor
        compressor = FP16Compressor()

    else:
        raise NotImplementedError(compressor)

    return compressor


def get_memory(params):
    memory_name = params.get('memory', 'none')

    if memory_name == 'none':
        from resir_lib.memory.none import NoneMemory
        memory = NoneMemory()

    elif memory_name == 'residual':
        from resir_lib.memory.residual import ResidualMemory
        memory = ResidualMemory()
    
    elif memory_name == 'tradeoff':
        from resir_lib.memory.tradeoff import MemoryTradeoff
        percent = params.get('percent', 0)
        memory = MemoryTradeoff(percent=percent)
    
    
    
############## ResiReduce-a, w
    
    ### ResNet50
    elif memory_name == 'residualrn50avg2':
        from resir_lib.memory.residualrn50avg2 import ResidualRN50avgMemory
        memory = ResidualRN50avgMemory()
    
    elif memory_name == 'residualrn50avg2weight':
        from resir_lib.memory.residualrn50avg2weight import ResidualRN50avgMemory
        memory = ResidualRN50avgMemory()

    
    ### VGG19
    
    elif memory_name == 'residualvgg19avg':
        from resir_lib.memory.residualvgg19avg import ResidualVGG19avgMemory
        memory = ResidualVGG19avgMemory()
    
    elif memory_name == 'residualvgg19avgweight':
        from resir_lib.memory.residualvgg19avgweight import ResidualVGG19avgMemory
        memory = ResidualVGG19avgMemory()
    
    
    ### GPT-2
    
    elif memory_name == 'residualgpt2savg':
        from resir_lib.memory.residualgpt2savg import ResidualGPT2SavgMemory
        memory = ResidualGPT2SavgMemory()
    
    elif memory_name == 'residualgpt2savgweight':
        from resir_lib.memory.residualgpt2savgweight import ResidualGPT2SavgMemory
        memory = ResidualGPT2SavgMemory()

    
    ### BERT
    elif memory_name == 'residualbertavg':
        from resir_lib.memory.residualbertavg import ResidualBertavgMemory
        memory = ResidualBertavgMemory()

    elif memory_name == 'residualbertweight':
        from resir_lib.memory.residualbertweight import ResidualBertavgMemory
        memory = ResidualBertavgMemory()
        
    
############## ResiReduce-d, 2d

    elif memory_name == 'dimcprs':
        from resir_lib.memory.dimcprs import DimCompression
        ratio = params.get('ratio', 0.5)
        hrank = params.get('hrank', 0)
        memory = DimCompression(ratio=ratio, hrank=hrank)
    
    elif memory_name == 'dimcprs2d':
        from resir_lib.memory.dimcprs2d import DimCompression
        ratio = params.get('ratio', 0.5)
        hrank = params.get('hrank', 0)
        memory = DimCompression(ratio=ratio, hrank=hrank)
    
############## ResiReduce-(w+2d)

    elif memory_name == 'resireducern50':
        from resir_lib.memory.resireducern50 import ResiReduceResNet50
        memory = ResiReduceResNet50()
        
    elif memory_name == 'resireducevgg19':
        from resir_lib.memory.resireducevgg19 import ResiReduceVGG19
        memory = ResiReduceVGG19()
    
    elif memory_name == 'resireducegpt2':
        from resir_lib.memory.resireducegpt2 import ResiReduceGPT2
        memory = ResiReduceGPT2()
        
    elif memory_name == 'resireducebert':
        from resir_lib.memory.resireducebert import ResiReduceBERT
        memory = ResiReduceBERT()
        
    
    
    else:
        raise NotImplementedError(memory)

    return memory


def get_config(params):
    send_size_aresame = params.get('send_size_aresame', True)
    return send_size_aresame


# Special case:
# All dim==1 or numel<10000 tensor should not be compressed
def check_not_compress(params, name, tensor):
    
    if tensor.dim() == 1 or tensor.numel() < 10000:
        return True
    if 'features.0' in name:
        return True
    if 'rnn.weight_hh' in name:
        return True
    if name == 'fc.weight':
        return True

    return False


def check_not_ef(params, name, tensor):

    compressor_name = params.get('compressor', 'none')    
    # VGG19:
    # if 'classifier.0.weight' in name:
    #     return True
    # if 'classifier.3.weight' in name:
    #     return True
    # if 'classifier.6.weight' in name:
    #     return True
    
    
    # Bert: WTE, WPE
    # if 'bert.pooler.dense_act.weight' in name or 'word_embeddings' in name or 'position_embeddings' in name:
    # if 'word_embeddings' in name or 'position_embeddings' in name:
    # if 'word_embeddings' in name :
        # return True
    
    # GPT2: WTE, WPE
    # if 'wpe' in name or 'wte' in name:
    # if 'wte' in name:
    #     return True

    return False 