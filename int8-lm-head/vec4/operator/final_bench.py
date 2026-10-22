#!/usr/bin/env python3
import argparse, json, os, statistics, subprocess, sys, time, hashlib
from pathlib import Path


def pct(xs,q):
    ys=sorted(xs); pos=(len(ys)-1)*q; lo=int(pos); hi=min(lo+1,len(ys)-1); f=pos-lo
    return ys[lo]*(1-f)+ys[hi]*f

def shasum(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def smi():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        return p.stdout.strip()
    except Exception as e:
        return f'ERROR:{e}'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bf16-shard',type=Path,default=None)
    ap.add_argument('--hidden',type=Path,default=None)
    ap.add_argument('--rounds',type=int,default=12)
    ap.add_argument('--iters',type=int,default=64)
    ap.add_argument('--heater-iters',type=int,default=256)
    a=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES','')!='0': raise SystemExit('set CUDA_VISIBLE_DEVICES=0')
    import torch
    from safetensors.torch import load_file
    from safetensors import safe_open
    from torch.utils.cpp_extension import load
    torch.cuda.set_device(0)
    if '3090' not in torch.cuda.get_device_name(0): raise RuntimeError(torch.cuda.get_device_name(0))
    pkg=Path(__file__).resolve().parent
    a.out.mkdir(parents=True,exist_ok=True)
    print('[ENV]',torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),flush=True)
    print('[ENV before]',smi(),flush=True)
    data=load_file(str(a.artifact),device='cpu')
    Wq=data['weight_q']; scales=data['scales']
    assert tuple(Wq.shape)==(151936,4096) and Wq.dtype==torch.int8
    Wq=Wq.cuda().contiguous(); scales=scales.cuda().contiguous(); del data
    if a.hidden and a.hidden.is_file():
        h=load_file(str(a.hidden),device='cpu')['hidden_states']
        assert h.ndim==2 and h.shape[1]==4096
        x_bf=h[0].cuda().to(torch.bfloat16).contiguous(); activation='captured_hidden_0'
    else:
        torch.manual_seed(42); x_bf=(torch.randn(4096,device='cuda')*0.03).to(torch.bfloat16).contiguous(); activation='synthetic_bf16_rounded'
    x=x_bf.float().contiguous(); assert x.data_ptr()%16==0
    build=a.out/'jit_build_final'; build.mkdir(exist_ok=True)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6'); os.environ.setdefault('MAX_JOBS','2')
    ext=load(name='int8_final_probe_v4',sources=[str(pkg/'ncu_int8_probe.cpp'),str(pkg/'ncu_int8_probe.cu')],extra_include_paths=[str(pkg)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(build),verbose=False)
    yv2=torch.empty(151936,device='cuda',dtype=torch.float32); yvec=torch.empty_like(yv2)
    def v2(): ext.v2_into(x,Wq,scales,yv2)
    def vec4(): ext.vec4_into(x,Wq,scales,yvec)
    funcs={'v2':v2,'vec4':vec4}
    Wbf=None; ybf=None
    if a.bf16_shard:
        print('[LOAD] BF16 lm_head from',a.bf16_shard,flush=True)
        with safe_open(str(a.bf16_shard),framework='pt',device='cpu') as f:
            sl=f.get_slice('lm_head.weight'); assert sl.get_shape()==[151936,4096] and sl.get_dtype()=='BF16'
            Wbf=torch.empty((151936,4096),device='cuda',dtype=torch.bfloat16)
            for st in range(0,151936,2048):
                en=min(st+2048,151936); Wbf[st:en].copy_(sl[st:en,:].to('cuda',dtype=torch.bfloat16))
        ybf=torch.empty((1,151936),device='cuda',dtype=torch.float32)
        xt=x_bf.reshape(1,4096)
        wt=Wbf.t()
        try:
            torch.mm(xt,wt,out=ybf,out_dtype=torch.float32); torch.cuda.synchronize()
            def bf16(): torch.mm(xt,wt,out=ybf,out_dtype=torch.float32)
            bf_mode='preallocated_out'
        except Exception:
            def bf16():
                nonlocal ybf
                ybf=torch.mm(xt,wt,out_dtype=torch.float32)
            bf_mode='allocating_output_fallback'
        funcs['bf16']=bf16
        print('[LOAD] BF16 ready mode=',bf_mode,flush=True)
    # correctness, including multiple activations for v2 vs vec4
    max_abs=0.0; max_rel=0.0; top1_all=True
    torch.manual_seed(123)
    test_xs=[]
    if a.hidden and a.hidden.is_file():
        hh=load_file(str(a.hidden),device='cpu')['hidden_states']
        for i in range(min(8,hh.shape[0])): test_xs.append(hh[i].cuda().to(torch.bfloat16).float().contiguous())
    while len(test_xs)<8:
        test_xs.append((torch.randn(4096,device='cuda')*0.03).to(torch.bfloat16).float().contiguous())
    orig_x=x
    for tx in test_xs:
        x.copy_(tx); v2(); vec4(); torch.cuda.synchronize()
        d=(yvec-yv2).abs(); max_abs=max(max_abs,d.max().item()); max_rel=max(max_rel,(torch.linalg.vector_norm(yvec-yv2)/torch.linalg.vector_norm(yv2).clamp_min(1e-12)).item()); top1_all &= yvec.argmax().item()==yv2.argmax().item()
    x.copy_(orig_x); v2(); vec4(); torch.cuda.synchronize()
    torch.testing.assert_close(yvec,yv2,atol=5e-3,rtol=1e-3)
    print('[CHECK]',{'vectors':len(test_xs),'max_abs':max_abs,'max_relative_l2':max_rel,'top1_all':top1_all},flush=True)
    # initial warmup all cases
    for _ in range(40):
        v2(); vec4()
        if 'bf16' in funcs: funcs['bf16']()
    torch.cuda.synchronize()
    cases=list(funcs)
    rec={k:[] for k in cases}; clocks=[]
    start=torch.cuda.Event(enable_timing=True); stop=torch.cuda.Event(enable_timing=True)
    # each measured group is preceded by the SAME V2 heater, to normalize P-state/clock history
    orders=[]
    if 'bf16' in funcs:
        perms=[['bf16','v2','vec4'],['v2','vec4','bf16'],['vec4','bf16','v2'],['vec4','v2','bf16'],['v2','bf16','vec4'],['bf16','vec4','v2']]
        for r in range(a.rounds): orders.append(perms[r%len(perms)])
    else:
        for r in range(a.rounds): orders.append(['v2','vec4'] if r%2==0 else ['vec4','v2'])
    for r,order in enumerate(orders):
        for key in order:
            for _ in range(a.heater_iters): v2()
            torch.cuda.synchronize()
            start.record()
            for _ in range(a.iters): funcs[key]()
            stop.record(); stop.synchronize()
            ms=start.elapsed_time(stop)/a.iters; rec[key].append(ms)
            clk=smi(); clocks.append({'round':r,'case':key,'ms':ms,'smi_after':clk})
            print(f'[MEASURE] round={r+1:02d} case={key:5s} {ms:.6f} ms | {clk.splitlines()[0] if clk else ""}',flush=True)
    stats={}
    for k,xs in rec.items():
        stats[k]={'median_ms':statistics.median(xs),'p25_ms':pct(xs,.25),'p75_ms':pct(xs,.75),'min_ms':min(xs),'max_ms':max(xs),'n':len(xs)}
    out={'artifact_sha256':shasum(a.artifact),'baseline_sha256':shasum(pkg/'baseline_int8_gemv_v2.cu'),'activation':activation,'rounds':a.rounds,'iters':a.iters,'heater_iters':a.heater_iters,'stats':stats,'times_ms':rec,'clock_log':clocks,'correctness':{'vectors':len(test_xs),'max_abs':max_abs,'max_relative_l2':max_rel,'top1_all':top1_all}}
    if 'vec4' in stats and 'v2' in stats: out['v2_div_vec4']=stats['v2']['median_ms']/stats['vec4']['median_ms']
    if 'bf16' in stats:
        out['bf16_div_v2']=stats['bf16']['median_ms']/stats['v2']['median_ms']; out['bf16_div_vec4']=stats['bf16']['median_ms']/stats['vec4']['median_ms']
    p=a.out/'final_bench_results.json'; p.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print('[RESULT]',json.dumps({'stats':stats,'v2_div_vec4':out.get('v2_div_vec4'),'bf16_div_v2':out.get('bf16_div_v2'),'bf16_div_vec4':out.get('bf16_div_vec4')},ensure_ascii=False),flush=True)
    print('[WROTE]',p,flush=True)

if __name__=='__main__': main()
