import os,sys
from pathlib import Path
DEFAULT_ARTIFACT=Path(__file__).resolve().parent / 'artifacts' / 'lm_head_int8_rowwise.safetensors'
def register():
    if os.environ.get('INT8_LM_HEAD_VEC4_ENABLE')!='1': return
    import torch
    from safetensors.torch import load_file
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    root=Path(__file__).resolve().parent; sys.path.insert(0,str(root/'build'))
    import int8_gemv_vec4_serving_ext
    artifact=Path(os.environ.get('INT8_LM_HEAD_ARTIFACT',str(DEFAULT_ARTIFACT)))
    if not artifact.exists(): raise RuntimeError(f'INT8 artifact not found: {artifact}')
    original=LogitsProcessor._apply_head
    if getattr(original,'_int8_vec4_experimental',False): return
    trace_enabled=os.environ.get('INT8_LM_HEAD_VEC4_TRACE')=='1'; trace={'custom':0,'fallback':0}
    def patched(self,lm_head,hidden_states,embedding_bias):
        eligible=(hidden_states.is_cuda and hidden_states.dtype==torch.bfloat16 and hidden_states.shape[-1]==4096 and hidden_states.numel()==4096 and tuple(lm_head.weight.shape)==(151936,4096) and lm_head.tp_size==1 and embedding_bias is None and isinstance(lm_head.quant_method,UnquantizedEmbeddingMethod) and (self.head_dtype is None or self.head_dtype==hidden_states.dtype))
        if not eligible:
            if trace_enabled: trace['fallback']+=1
            return original(self,lm_head,hidden_states,embedding_bias)
        cache=getattr(lm_head,'_experimental_int8_vec4_cache',None)
        if cache is None:
            tensors=load_file(str(artifact),device='cpu'); Wq=tensors['weight_q'].to(hidden_states.device); scales=tensors['scales'].to(hidden_states.device)
            if tuple(Wq.shape)!=(151936,4096): raise RuntimeError(f'Unexpected Wq shape: {tuple(Wq.shape)}')
            if tuple(scales.shape)!=(151936,): raise RuntimeError(f'Unexpected scales shape: {tuple(scales.shape)}')
            cache=(Wq.contiguous(),scales.contiguous()); lm_head._experimental_int8_vec4_cache=cache
            print('[INT8 LM HEAD VEC4] Quantized weights loaded; char4+float4 CUDA path enabled',flush=True)
        Wq,scales=cache
        x=hidden_states.reshape(-1).float().contiguous(); y=int8_gemv_vec4_serving_ext.forward(x,Wq,scales)
        if trace_enabled:
            trace['custom']+=1; n=trace['custom']
            if n==1 or n%64==0: print(f"[INT8 VEC4 TRACE] custom_forward={n} fallback={trace['fallback']} shape={tuple(hidden_states.shape)}",flush=True)
        return y.reshape(*hidden_states.shape[:-1],151936).to(hidden_states.dtype)
    patched._int8_vec4_experimental=True; LogitsProcessor._apply_head=patched
    print('[INT8 LM HEAD VEC4] Plugin registered',flush=True)
