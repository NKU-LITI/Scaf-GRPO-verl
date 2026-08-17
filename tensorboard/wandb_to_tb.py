import math 
import numbers 
from pathlib import Path 
import wandb 
from torch.utils.tensorboard import SummaryWriter 
# from tensorboardX import SummaryWriter
# from tensorboard.summary.writer import SummaryWriter
# ============================================================ 
# 要转换的 online W&B run 
# ============================================================ 
RUN_PATH = ( 
    "1196113942-nankai-university/" 
    "scaf-grpo-expert-sft/" 
    "y52m6blw" 
) 
# 本地 TensorBoard 输出目录 
OUTPUT_DIR = Path("./tensorboard/qwen25_math7b_y52m6blw_grpo_baseline") 
# W&B 默认图表横坐标 "Step" 对应 _step 
STEP_KEY = "_step" 
# ============================================================ 
# 连接在线 W&B 
# ============================================================ 
api = wandb.Api() 
run = api.run(RUN_PATH) 
print("=" * 70) 
print("W&B run") 
print("name :", run.name) 
print("id :", run.id) 
print("state:", run.state) 
print("=" * 70) 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 
# ============================================================ 
# 创建 TensorBoard event writer 
# ============================================================ 
writer = SummaryWriter(log_dir=str(OUTPUT_DIR)) 
num_rows = 0 
num_scalars = 0 
metric_names = set() 
# ============================================================ 
# 在线 W&B history -> 本地 TensorBoard 
# ============================================================ 
for row in run.scan_history(page_size=1000): 
    num_rows += 1 
    step = row.get(STEP_KEY) 
    if step is None: 
        continue 
    try: 
        step = int(step) 
    except (TypeError, ValueError): 
        continue 
    # W&B 的真实时间戳 
    walltime = row.get("_timestamp") 
    for key, value in row.items(): 
        # 跳过 W&B 内部字段 
        if key.startswith("_"): 
            continue 
        # 只处理 scalar 
        if not isinstance(value, numbers.Number): 
            continue 
        if isinstance(value, bool): 
            continue 
        value = float(value) 
        # TensorBoard 不写 nan / inf 
        if not math.isfinite(value): 
            continue 
        kwargs = { 
            "tag": key, 
            "scalar_value": value, 
            "global_step": step, 
        } 
        if isinstance(walltime, numbers.Number): 
            kwargs["walltime"] = float(walltime) 
        writer.add_scalar(**kwargs) 
        metric_names.add(key) 
        num_scalars += 1 
writer.flush() 
writer.close() 
# ============================================================ 
# 输出结果 
# ============================================================ 
print() 
print("=" * 70) 
print("Conversion finished") 
print("=" * 70) 
print(f"History rows : {num_rows}") 
print(f"Scalar points : {num_scalars}") 
print(f"Scalar metrics : {len(metric_names)}") 
print(f"Output dir : {OUTPUT_DIR.resolve()}") 
print() 
print("Metrics:") 
for name in sorted(metric_names): 
    print(" ", name)