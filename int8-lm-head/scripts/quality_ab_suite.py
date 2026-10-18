import hashlib
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

SERVERS = {
    "BF16": (8001, "qwen-baseline"),
    "INT8": (8002, "qwen-int8"),
}

MAX_TOKENS = 384
TASKS = []

def closed(category, question, *answers):
    TASKS.append({
        "category": category,
        "kind": "closed",
        "question": question,
        "accepted": list(answers),
    })

def opened(category, question, criteria):
    TASKS.append({
        "category": category,
        "kind": "open",
        "question": question,
        "criteria": criteria,
    })

# ========== 1. 基本语言能力 ==========

C = "基本语言能力"

opened(C,
    "把这句话准确翻译成英文：算子变快不一定意味着端到端请求延迟下降。",
    "保留“不一定”的逻辑含义；准确翻译算子及端到端延迟。")

opened(C,
    "某系统低并发时延迟很低，高并发时吞吐提升，但排队使尾延迟增加。"
    "请用不超过25个汉字概括主要现象。",
    "语言简洁；准确保留吞吐提升与尾延迟增加。")

opened(C,
    "把“我们需要在接下来的后续阶段进一步继续检查这个问题”"
    "改写成简洁、自然的书面中文。",
    "减少冗余；不改变原句意图；表达自然。")

opened(C,
    "Use exactly two English sentences to explain the difference "
    "between latency and throughput.",
    "恰好两句英文；清楚区分单请求耗时与单位时间处理量。")

opened(C,
    "严格输出一个合法 JSON 对象，只有 title 和 risks 两个字段。"
    "title 为字符串“CUDA优化”，risks 为恰好两项字符串数组，"
    "列出两种常见优化风险。不准添加 Markdown 代码块。",
    "JSON 语法合法；字段准确；两项合理风险；没有多余内容。")

# ========== 2. 基本计算能力 ==========

C = "基本计算能力"

closed(C, "计算 37×24，结果是多少？", "888")
closed(C, "1250 的 8% 是多少？", "100")
closed(C, "汽车以48公里/小时行驶45分钟，行驶多少公里？", "36")
closed(C, "计算 7/8 - 1/3，结果写成最简分数。", "13/24")
closed(C, "计算 0.125×240。", "30")

# ========== 3. 基本推理思考能力 ==========

C = "基本推理思考能力"

closed(C,
    "如果部署失败，系统一定报警。现在确定系统没有报警，"
    "且规则没有例外。能推断部署没有失败吗？",
    "是", "能", "可以")

closed(C,
    "如果缓存命中，就会打印hit日志。现在看到hit日志，"
    "一定能反推出缓存命中吗？",
    "否", "不能", "不可以")

closed(C,
    "甲比乙高，乙比丙高。三人中谁最高？",
    "甲")

closed(C,
    "三个任务X、Y、Z，X必须在Y之前，Y必须在Z之前。"
    "最后执行哪个任务？",
    "Z", "任务Z")

closed(C,
    "每个Thread属于一个Warp，每个Warp属于一个Block。"
    "已知Thread T属于Warp W，Warp W属于Block B，"
    "能推出T属于B吗？",
    "是", "能", "可以")

# ========== 4. 进阶解题能力 ==========

C = "进阶解题能力"

closed(C,
    "无向带权图边为 A-B:4、A-C:2、C-B:1、"
    "B-D:5、C-D:9。A到D的最短路径总权重是多少？",
    "8")

closed(C,
    "0/1背包容量为6，三个物品重量分别为2、3、4，"
    "价值分别为4、5、10。最大可获得价值是多少？",
    "14")

closed(C,
    "有五个左闭右开时间区间：[1,3)、[2,5)、"
    "[3,4)、[4,7)、[5,6)。"
    "最多可以选出几个互不重叠的区间？",
    "3")

opened(C,
    "编写Python函数 unique_keep_first(items)，"
    "输入字符串列表，忽略大小写去重，但保留每个字符串首次出现时的原始写法。"
    "例如 ['GPU','cpu','gpu','CPU','Cuda'] 应返回 "
    "['GPU','cpu','Cuda']。请给出代码和时间复杂度。",
    "正确处理大小写；保留首次顺序；代码可运行；通常为O(n)。")

opened(C,
    "有SQL表 events(user_id,event_id,event_time)。"
    "请写SQL，返回每位用户最新的三条事件。"
    "如果event_time相同，使用event_id降序决定先后。"
    "不足三条则全部返回。",
    "使用分组窗口排序或等价方案；处理时间相同；每用户最多三条。")

# ========== 5. 进阶计算能力 ==========

C = "进阶计算能力"

closed(C, "计算组合数 C(12,4)。", "495")
closed(C, "计算 2^20 除以17的余数。", "16")
closed(C,
    "袋子里有3个红球、2个蓝球，不放回连续抽2球。"
    "两球都是红球的概率是多少？写成最简分数。",
    "3/10", "0.3")

closed(C,
    "计算定积分：从0到2，对函数 3x^2+2x 积分。",
    "12")

closed(C,
    "解方程组：3x+2y=19，x-y=3。"
    "结果按照 x=数字,y=数字 的形式给出。",
    "x=5,y=2", "x=5，y=2")

# ========== 6. 进阶推理能力 ==========

C = "进阶推理能力"

closed(C,
    "甲乙丙三人中恰好只有一人说真话。"
    "甲说：“乙和丙都说假话。”"
    "乙说：“甲说假话。”"
    "丙说：“乙说假话。”"
    "谁说真话？",
    "乙", "B")

closed(C,
    "桌上四张卡片可见面分别为A、D、4、7，"
    "每张卡片一面字母另一面数字。"
    "要验证“如果字母是元音，则背面数字为偶数”，"
    "最少必须翻哪两张卡？"
    "用卡片可见面标识回答。",
    "A和7", "A、7", "A,7", "7和A", "7、A", "7,A")

closed(C,
    "四个任务A、B、C、D，每个恰好执行一次。"
    "A必须早于B和C，C必须早于B，B必须早于D。"
    "写出唯一合法的执行顺序，只用四个大写字母。",
    "ACBD")

closed(C,
    "有三个箱子，标签分别写着“苹果”“橙子”“混合”，"
    "三个标签全部贴错。每次可以从某个箱子抽出一个水果观察。"
    "最少抽几个水果就能确定全部箱子的内容？",
    "1", "一个")

opened(C,
    "实验中GPU0运行BF16，GPU1运行INT8。"
    "测得INT8单次请求快8%，但输出Token数也少5%。"
    "请分析为什么不能直接认定INT8带来了8%的端到端加速，"
    "并设计一个更公平的对照实验。",
    "识别Token数混杂因素；讨论GPU差异；固定工作量；多轮测量与交叉实验。")

# ========== 7. 专业术语与准确性 ==========

C = "专业术语与准确性"

opened(C,
    "解释CUDA中的Warp、Block、SM Occupancy。"
    "说明为何Occupancy高不必然意味着Kernel速度快。",
    "Warp 32线程；Block组织线程；Occupancy定义正确；"
    "解释访存、指令和带宽等性能限制。")

opened(C,
    "解释LLM Serving中的Prefill、Decode、TTFT和TPOT。"
    "为什么TTFT降低不代表TPOT一定降低？",
    "四个概念准确；区分首Token处理和后续Token生成。")

opened(C,
    "一个LM Head的权重形状为[151936,4096]。"
    "解释逐输出行对称INT8量化时scale的含义、"
    "近似反量化公式，以及量化误差主要来自什么。",
    "权重维度准确；每行scale；W≈q×scale；"
    "舍入与截断误差；不能保证Token完全一致。")

opened(C,
    "区分UVA、Pinned Memory与Mapped Pinned Memory。"
    "解释为什么在WSL2下设置VLLM_WSL2_ENABLE_PIN_MEMORY=1"
    "不等于打开GPU硬件UVA能力。",
    "区分地址能力、页锁定和映射；识别WSL2限制；"
    "不混淆环境开关与硬件开关。")

opened(C,
    "解释Tensor Parallel和Pipeline Parallel的区别。"
    "再解释为什么针对单个4096维输入向量编写的INT8 GEMV"
    "不能直接替代所有Prefill阶段的矩阵乘法。",
    "并行切分方式正确；区分GEMV和GEMM；"
    "考虑多Token输入、并行通信与形状要求。")


def normalize(value):
    value = value.strip().lower()
    return re.sub(
        r"[\s`*.,，。:：;；!！、→\[\]（）()]",
        "", value
    )


def extract_answer(text):
    matches = re.findall(
        r"答案\s*[:：]\s*([^\r\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    return matches[-1].strip() if matches else None


def call_model(tag, prompt):
    port, model = SERVERS[tag]

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()

    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)

    elapsed = time.perf_counter() - start
    choice = data["choices"][0]

    return {
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "output_tokens": data.get("usage", {}).get("completion_tokens"),
        "elapsed_s": elapsed,
    }


def evaluate(task, result):
    if task["kind"] != "closed":
        return None

    answer = extract_answer(result["text"])
    if answer is None or result["finish_reason"] == "length":
        return None

    return normalize(answer) in {
        normalize(a) for a in task["accepted"]
    }


outdir = Path("artifacts/quality_ab")
outdir.mkdir(parents=True, exist_ok=True)

for tag, (port, _) in SERVERS.items():
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/health", timeout=10
    ) as response:
        print(f"{tag} health: HTTP {response.status}")

results = []

print(f"\nTOTAL TASKS: {len(TASKS)}")
print(f"TOTAL REQUESTS: {len(TASKS) * 2}", flush=True)

for i, task in enumerate(TASKS):
    task = dict(task)
    task["id"] = f"Q{i+1:03d}"

    if task["kind"] == "closed":
        prompt = (
            task["question"]
            + "\n请简要说明思路，最后单独一行严格写"
            "“答案：结果”，冒号后只填最终结果，不附加解释。"
        )
    else:
        prompt = task["question"]

    order = (
        ["BF16", "INT8"]
        if i % 2 == 0
        else ["INT8", "BF16"]
    )

    answers = {}
    for tag in order:
        result = call_model(tag, prompt)
        result["auto_correct"] = evaluate(task, result)
        answers[tag] = result

    record = {
        **task,
        "actual_prompt": prompt,
        "answers": answers,
        "exact_text_match": (
            answers["BF16"]["text"] == answers["INT8"]["text"]
        ),
    }

    results.append(record)

    # 每题保存一次，中途报错不会丢失已有测试结果
    (outdir / "full_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    b = answers["BF16"]["auto_correct"]
    q = answers["INT8"]["auto_correct"]

    if task["kind"] == "closed":
        print(
            f'{task["id"]} {task["category"]} '
            f'BF16={b} INT8={q}',
            flush=True,
        )
    else:
        print(
            f'{task["id"]} {task["category"]} '
            f'OPEN REVIEW',
            flush=True,
        )

print("\n========== OBJECTIVE SUMMARY ==========")

summary = {}

for category in dict.fromkeys(t["category"] for t in TASKS):
    subset = [
        r for r in results
        if r["category"] == category
        and r["kind"] == "closed"
    ]

    if not subset:
        continue

    entry = {"total": len(subset)}

    for tag in SERVERS:
        scores = [
            r["answers"][tag]["auto_correct"]
            for r in subset
        ]
        entry[tag] = {
            "correct": sum(v is True for v in scores),
            "unparsed": sum(v is None for v in scores),
        }

    summary[category] = entry

    print(
        category,
        f"BF16={entry['BF16']['correct']}/{len(subset)}",
        f"INT8={entry['INT8']['correct']}/{len(subset)}",
        f"unparsed(BF16/INT8)="
        f"{entry['BF16']['unparsed']}/{entry['INT8']['unparsed']}",
    )

regressions = [
    r["id"]
    for r in results
    if r["kind"] == "closed"
    and r["answers"]["BF16"]["auto_correct"] is True
    and r["answers"]["INT8"]["auto_correct"] is False
]

print("BF16 correct -> INT8 incorrect:", regressions)

# 创建匿名盲评文件：X/Y不标明模型版本
blind = []

for r in results:
    if r["kind"] != "open":
        continue

    flip = int(
        hashlib.sha256(r["id"].encode()).hexdigest(), 16
    ) % 2

    tags = ["BF16", "INT8"]
    if flip:
        tags.reverse()

    blind.append({
        "id": r["id"],
        "category": r["category"],
        "question": r["question"],
        "criteria": r["criteria"],
        "X": r["answers"][tags[0]]["text"],
        "Y": r["answers"][tags[1]]["text"],
        "score_X_0_to_4": None,
        "score_Y_0_to_4": None,
        "notes": "",
    })

(outdir / "blind_review.json").write_text(
    json.dumps(blind, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

(outdir / "objective_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nSaved:", outdir.resolve())
print("QUALITY A/B COMPLETE")
