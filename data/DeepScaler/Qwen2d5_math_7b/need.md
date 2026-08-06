我现在需要构造一个小的训练集做算法的快速验证。针对qwen2.5-math-7b模型，我需要从"data/DeepScaleR/Qwen2.5_math_7b/solution_breakdown_cot_answer/Qwen2.5-Math-7B.solution_breakdown_cot_answer.reward1.parquet"数据集中选择出800条数据，根据数据相对模型能力的难度划分：

- 高难度：success_rate@k=0，400条 
- 中难度：0 < success_rate@k <= 0.5，200条 
- 低难度：1 > success_rate@k > 0.5，200条

验证集：分别从训练集中随机选取100条、50条、50条，构成验证集。

k取8即可

我的目的就是为了验证hint guide和专家轨迹注入以及额外添加的sft loss能帮助模型解决之前无法解决的难题。

