import os,csv,json,time,signal,socket,hashlib,subprocess,statistics
from pathlib import Path
from datetime import datetime
import requests
ROOT=Path(__file__).resolve().parents[1]; MODEL=Path(os.environ['QWEN_MODEL_DIR']); ARTIFACT=Path(os.environ['INT8_LM_HEAD_ARTIFACT'])
OUT=ROOT/'results'/('vec4_serving_abba_'+datetime.now().strftime('%Y%m%d_%H%M%S')); OUT.mkdir(parents=True,exist_ok=True)
PORT=8002; URL=f'http://127.0.0.1:{PORT}/v1/chat/completions'
STAGES=[('A1_BF16',False),('B1_VEC4',True),('B2_VEC4',True),('A2_BF16',False)]
PROMPTS=['详细解释 CUDA Warp、Block 和 GPU 内存优化。','介绍 CUDA 全局内存合并访问和共享内存 bank conflict。','解释 LLM Decode 中 KV Cache 的访存与带宽瓶颈。']
REQUESTS_PER_STAGE=int(os.environ.get('ABBA_REQUESTS_PER_STAGE','8')); WARMUP_PER_STAGE=int(os.environ.get('ABBA_WARMUP_PER_STAGE','2')); MAX_TEMPERATURE=int(os.environ.get('ABBA_MAX_TEMPERATURE','79'))
CMD=['vllm','serve',str(MODEL),'--served-model-name','qwen-int8-vec4','--host','127.0.0.1','--port',str(PORT),'--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len','2048','--max-num-batched-tokens','2048','--max-num-seqs','8','--gpu-memory-utilization','0.85']
def sha256(p):
 h=hashlib.sha256(); f=open(p,'rb')
 try:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 finally: f.close()
 return h.hexdigest()
def port_busy():
 s=socket.socket(); s.settimeout(1)
 try:return s.connect_ex(('127.0.0.1',PORT))==0
 finally:s.close()
def gpu_status(): return subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
def check_temperature():
 status=gpu_status(); temp=int(float(status.split(',')[0].strip()))
 if temp>=MAX_TEMPERATURE: raise RuntimeError(f'GPU0 temperature {temp}C >= {MAX_TEMPERATURE}C')
 return status
def stop_server(proc):
 if proc is None:return
 if proc.poll() is None:
  try: os.killpg(proc.pid,signal.SIGINT); proc.wait(timeout=30)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try: proc.wait(timeout=15)
   except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=10)
 for _ in range(45):
  if not port_busy():return
  time.sleep(1)
 raise RuntimeError('Port 8002 was not released')
def start_server(stage,vec4):
 if port_busy(): raise RuntimeError('Port 8002 occupied; stop old vLLM server')
 env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':'0','VLLM_WSL2_ENABLE_PIN_MEMORY':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_PLUGINS':'qwen_int8_vec4_head','INT8_LM_HEAD_VEC4_ENABLE':'1' if vec4 else '0','INT8_LM_HEAD_VEC4_TRACE':'1' if vec4 else '0','INT8_LM_HEAD_ARTIFACT':str(ARTIFACT),'HF_HUB_OFFLINE':'1'})
 log_path=OUT/f'{stage}.log'; log_file=open(log_path,'w',buffering=1)
 print(f'\n========== START {stage} ==========',flush=True); print('[PRE-STAGE GPU]',check_temperature(),flush=True)
 proc=subprocess.Popen(CMD,cwd=str(ROOT),env=env,stdout=log_file,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  ready=False
  for _ in range(420):
   if proc.poll() is not None: raise RuntimeError(f'{stage}: vLLM exited; inspect {log_path}')
   try:
    r=requests.get(f'http://127.0.0.1:{PORT}/health',timeout=2)
    if r.status_code==200: ready=True; break
   except requests.RequestException: pass
   time.sleep(1)
  if not ready: raise RuntimeError(f'{stage}: startup timeout; inspect {log_path}')
  time.sleep(2); text=log_path.read_text(errors='replace')
  if 'Graph capturing finished' not in text: raise RuntimeError(f'{stage}: CUDA Graph capture not confirmed')
  print(f'{stage}: server ready, CUDA Graph confirmed'); return proc,log_file,log_path
 except BaseException:
  stop_server(proc); log_file.close(); raise
def request_once(prompt):
 payload={'model':'qwen-int8-vec4','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':128,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
 t0=time.perf_counter(); first=last=done=None; tokens=None; chunks=0
 with requests.post(URL,json=payload,stream=True,timeout=180) as response:
  response.raise_for_status()
  for line in response.iter_lines():
   if not line or not line.startswith(b'data: '): continue
   raw=line[6:]
   if raw==b'[DONE]': done=time.perf_counter(); break
   data=json.loads(raw); now=time.perf_counter(); usage=data.get('usage')
   if usage: tokens=usage.get('completion_tokens')
   for choice in data.get('choices',[]):
    content=choice.get('delta',{}).get('content')
    if content:
     chunks+=1
     if first is None:first=now
     last=now
 if first is None or last is None or tokens!=128: raise RuntimeError(f'Invalid response: tokens={tokens}, chunks={chunks}')
 if done is None:done=last
 return {'tokens':tokens,'chunks':chunks,'ttft_ms':(first-t0)*1000,'tpot_ms':(last-first)*1000/(tokens-1),'e2e_ms':(done-t0)*1000}
def describe(v): return {'mean_ms':statistics.mean(v),'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v),'n':len(v)}
if port_busy(): raise SystemExit('ERROR: port 8002 busy')
if not MODEL.exists(): raise SystemExit(f'ERROR: model not found: {MODEL}')
if not ARTIFACT.exists(): raise SystemExit(f'ERROR: artifact not found: {ARTIFACT}')
print('========== VEC4 SERVING SMALL ABBA =========='); print('Output:',OUT); print('Model:',MODEL); print('Artifact:',ARTIFACT); print('Artifact sha256:',sha256(ARTIFACT)); print('Stages:',[x[0] for x in STAGES]); print('Requests/stage:',REQUESTS_PER_STAGE); print('Warmup/stage:',WARMUP_PER_STAGE); print('Total measured:',REQUESTS_PER_STAGE*len(STAGES)); print('GPU:',check_temperature(),flush=True)
(OUT/'experiment.json').write_text(json.dumps({'model':str(MODEL),'artifact':str(ARTIFACT),'artifact_sha256':sha256(ARTIFACT),'stages':STAGES,'requests_per_stage':REQUESTS_PER_STAGE,'warmup_per_stage':WARMUP_PER_STAGE,'cmd':CMD},indent=2,ensure_ascii=False))
results=[]; csv_path=OUT/'results.csv'
with open(csv_path,'w',newline='') as csv_file:
 writer=csv.DictWriter(csv_file,fieldnames=['stage','mode','request','prompt_id','tokens','chunks','ttft_ms','tpot_ms','e2e_ms','gpu_status']); writer.writeheader()
 for stage,vec4 in STAGES:
  proc=log_file=log_path=None
  try:
   proc,log_file,log_path=start_server(stage,vec4); print(f'\n========== WARMUP {stage} ==========')
   for i in range(WARMUP_PER_STAGE): check_temperature(); request_once(PROMPTS[i%len(PROMPTS)])
   print('Warmup complete',flush=True); text=log_path.read_text(errors='replace'); loaded='[INT8 LM HEAD VEC4] Quantized weights loaded' in text; called='[INT8 VEC4 TRACE] custom_forward=' in text
   if vec4 and not (loaded and called): raise RuntimeError(f'{stage}: vec4 path not confirmed in {log_path}')
   if not vec4 and (loaded or called): raise RuntimeError(f'{stage}: unexpected vec4 path in BF16 stage')
   stage_vals=[]
   for i in range(REQUESTS_PER_STAGE):
    check_temperature(); pid=i%len(PROMPTS); sample=request_once(PROMPTS[pid]); status=check_temperature(); stage_vals.append(sample['tpot_ms'])
    row={'stage':stage,'mode':'VEC4' if vec4 else 'BF16','request':i+1,'prompt_id':pid,'tokens':sample['tokens'],'chunks':sample['chunks'],'ttft_ms':round(sample['ttft_ms'],4),'tpot_ms':round(sample['tpot_ms'],4),'e2e_ms':round(sample['e2e_ms'],4),'gpu_status':status}; writer.writerow(row); csv_file.flush(); results.append(row)
    print(f"{stage} [{i+1:02d}/{REQUESTS_PER_STAGE}] TPOT={sample['tpot_ms']:.3f}ms TTFT={sample['ttft_ms']:.2f}ms GPU={status}",flush=True)
   print(f'{stage}: median={statistics.median(stage_vals):.4f}ms mean={statistics.mean(stage_vals):.4f}ms',flush=True)
  finally:
   stop_server(proc)
   if log_file: log_file.close()
expected=len(STAGES)*REQUESTS_PER_STAGE
if len(results)!=expected: raise RuntimeError(f'Incomplete benchmark: {len(results)} != {expected}')
summary={}
for stage,vec4 in STAGES:
 rows=[r for r in results if r['stage']==stage]; summary[stage]={'mode':'VEC4' if vec4 else 'BF16','tpot':describe([r['tpot_ms'] for r in rows]),'ttft':describe([r['ttft_ms'] for r in rows]),'e2e':describe([r['e2e_ms'] for r in rows])}
bf16=[r['tpot_ms'] for r in results if r['mode']=='BF16']; vec4=[r['tpot_ms'] for r in results if r['mode']=='VEC4']; bm=statistics.mean(bf16); vm=statistics.mean(vec4)
report={'stages':summary,'pooled':{'bf16_tpot_mean_ms':bm,'vec4_tpot_mean_ms':vm,'bf16_tpot_median_ms':statistics.median(bf16),'vec4_tpot_median_ms':statistics.median(vec4),'vec4_minus_bf16_mean_ms':vm-bm,'speed_ratio_mean':bm/vm,'tpot_change_mean_pct':100*(vm/bm-1),'n_bf16':len(bf16),'n_vec4':len(vec4)},'interpretation':'Descriptive small ABBA only. Client streaming chunk timing is approximate TPOT. No statistical significance claim.','results_csv':str(csv_path)}
summary_path=OUT/'summary.json'; summary_path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
print('\n========== FINAL SMALL ABBA SUMMARY ==========')
for stage in [x[0] for x in STAGES]: s=summary[stage]['tpot']; print(f"{stage}: median={s['median_ms']:.4f}ms mean={s['mean_ms']:.4f}ms n={s['n']}")
print('\n========== BF16 VS VEC4 =========='); print(f'BF16 pooled mean TPOT: {bm:.4f} ms'); print(f'VEC4 pooled mean TPOT: {vm:.4f} ms'); print(f'VEC4 - BF16:          {vm-bm:+.4f} ms'); print(f'Speed ratio:          {bm/vm:.4f}x'); print(f'TPOT change:          {100*(vm/bm-1):+.3f}%'); print('\nNOTE: descriptive small ABBA only.'); print('No statistical significance claim is made.'); print('Client streaming chunk timing is approximate TPOT.'); print('\nResults:',csv_path); print('Summary:',summary_path); print('Logs:',OUT); print('VEC4 SERVING SMALL ABBA COMPLETE')
