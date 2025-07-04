---
author: "Hongyao Tang"
title: "LLM Principles Optimization"
date: "2025-07-04"
tags: [
    "AI Optim",
]
ShowToc: true
weight: 1
draft: false
---

Orgnized by the way of thinking, this page shows issues and solutions in each phase of DL


## Variable value size impacts stability
- 渐变
- threashhold

### LR
#### S - 在训练开始时使用较大的LR更新，损失剧烈波动，模型在训练过程不稳定
T - 学习率预热(learning rate warmup)

- 逐步将学习率从一个非常低的初始值(initial_lr)提升到用户设定的最大值(peak_lr)
![alt text](images/lr-warm.png)


A

- 计划训练一个大语言模型 15 轮，开始时设定的初始学习率为 0.0001，随后将其提升至最大学习率 0.01
- 预热步骤的数量通常设置为总步骤数的 0.1% 到 20%. 这段代码将输出 27，这意味着在前 27 个训练步骤中，我们将控制学习率从 0.0001 逐步提高到 0.01。
```py
n_epochs = 15
initial_lr = 0.0001
peak_lr = 0.01


total_steps = len(train_loader) * n_epochs
warmup_steps = int(0.2 * total_steps) # 20% warmup
print(warmup_steps)
# 27


lr_increment = (peak_lr - initial_lr) / warmup_steps

global_step = -1
track_lrs = []

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1
    
        if global_step < warmup_steps:
            lr = initial_lr + global_step * lr_increment
        else:
            lr = peak_lr
        
        # Apply the calculated learning rate to the optimizer
        # 将计算后的学习率应用到优化器上
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 记录lr后续plot
        track_lrs.append(optimizer.param_groups[0]["lr"])
    
        # Calculate loss and update weights
        # ...


import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.ylabel("Learning rate")
plt.xlabel("Step")
total_training_steps = len(train_loader) * n_epochs
plt.plot(range(total_training_steps), track_lrs)
plt.tight_layout(); plt.savefig("1.pdf")
plt.show()
```


R

- 在训练开始时使用较小的权重更新，有助于降低模型在训练过程中遭遇大幅度、不稳定更新的风险。稳定复杂模型（如大语言模型）的训练过程


#### S - 固定的学习率可能导致模型在训练后期收敛不稳定。

T - 余弦退火/衰减(cosine annealing/decay)

- 调节学习率，使其在预热阶段后呈现余弦曲线的变化，逐渐降低
![alt text](images/lr-decay.png)


A

```py
import math

min_lr = 0.1 * initial_lr
track_lrs = []

lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1
    
        # Adjust the learning rate based on the current phase (warmup or cosine annealing)
        if global_step < warmup_steps:
            # Linear warmup
            lr = initial_lr + global_step * lr_increment  
        else:
            # Cosine annealing after warmup
            progress = ((global_step - warmup_steps) / 
                        (total_training_steps - warmup_steps))
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress))
        
        # Apply the calculated learning rate to the optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])
    
        # Calculate loss and update weights

plt.figure(figsize=(5, 3))
plt.ylabel("Learning rate")
plt.xlabel("Step")
plt.plot(range(total_training_steps), track_lrs)
plt.tight_layout(); plt.savefig("2.pdf")
plt.show()
```


R

旨在减缓模型更新权重的速度，有助于降低训练过程中超过损失最小值的风险，从而确保后期训练的稳定性。





### Gradient
#### S - Gradient过大，梯度在反向传播中爆炸，导致训练不稳定
- 在深度网络或RNN中，梯度过大可能在反向传播中爆炸，导致训练不稳定

T - 梯度裁剪(gradient clipping)

- 该方法涉及设定一个阈值，超过该阈值的梯度会被缩放到预定的最大值


<u>范数Norm</u>
线性代数中的一个概念，用来衡量一个向量的“长度”
![alt text](images/norm.png)


```py
# Get gradient
for param in model.parameters():
    if param.grad is not None:
        grad_values = param.grad.data.flatten()
```


A

```py
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()
# 当调用 .backward() 方法时，PyTorch 会计算损失的梯度，并将其存储在每个模型权重（参数）张量的 .grad 属性中


# 查看最大梯度
def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad

print(find_highest_gradient(model))
# tensor(0.0411)


# 裁剪梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(find_highest_gradient(model))
tensor(0.0185)
```


R

这种做法可以确保在反向传播过程中，对模型参数的更新保持在一个可控的范围内。


#### Summary
```py
from previous_chapters import evaluate_model, generate_and_print_sample

# train_model_simple ->
def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment  
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate

            # Calculate and backpropagate the loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()


            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            tokens_seen += input_batch.numel()

            # Periodically evaluate the model on the training and validation sets
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # Print the current losses
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        # Generate and print a sample from the model to monitor progress
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs
```

---

## Param quantity impacts resource consumption
- Decompose large structure into smaller pieces

### Finetune
#### S - 微调中参数的数量巨大
T - LoRA减少微调中参数的数量

- Rank秩
  - ![alt text](images/rank.png)
  - 第 2 行是第 1 行的 2 倍，第 3 行是第 1 行的 3 倍。所以这三行其实只包含了一行独立信息。因此，这个矩阵的 Rank = 1。
  - Rank秩 理解为：一个矩阵中有多少行或列是“真正有用的”，也就是线性无关的行或列的数量。
- Low Rank低秩
  - 如果一个权重矩阵的秩很低，说明它的“信息复杂度”不高。
  -  我们可以用两个小得多的矩阵相乘来近似它（低秩分解），从而减少参数量。
- Adapatation自适应
  - 使预训练模型更好地适应特定且通常较小的数据集的技术

<u>W_delta = AB</u>
- r, rank矢，内部维度，可调的超参数。LoRA 的两个小矩阵之所以“特别小”，是因为它们的秩 r很低（通常是 4、8、16 这样的数）,在模型的适应性和效率之间建立平衡
- W_delta = 4096×4096=16,777,216 个参数
- r 是一个很小的秩，比如 r = 8
  - A 有 8×4096=32,768 个参数
  - B有 4096×8=32,768 个参数
  - BA is (4096, 4096)
  - LoRA 参数总数=32,768+32,768=65,536, 只用了不到 0.4% 的参数量！
![alt text](images/lora.png)


A

初始化一个 LoRA 层，它创建了矩阵 A 和 B，并设置了rank(r)和alpha缩放因子
```py
import math

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
```

创建一个 LinearWithLoRA 层,整合原始线性层的权重
```py
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x) #xW + xW_delta
```

replace_linear_ with_lora函数, 在 LoRA中，典型的目标是替换现有的线性层，从而允许权重更新直接应用于已有的预训练权重
- for name, module in model.named_children():
  - 遍历模型的所有直接子模块（不包括孙子模块）
  - name 是子模块的名字（字符串），module 是对应的子模块对象
- setattr(object, name, value)
  - setattr 是 Python 的内置函数，用来动态地给对象设置属性。
  - object  对象  要设置属性的对象（比如一个类的实例）
  - name  字符串  要设置的属性名（作为字符串传入）
  - value   任意类型  要赋给该属性的值
- setattr(model, name, LinearWithLoRA(module, rank, alpha))
  - 把model对象中名为name字符串的子模块，替换成一个新的 LinearWithLoRA 实例。

```py
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)

# 冻结原始模型的参数
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")

for param in model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")
# Total trainable parameters before: 124,441,346
# Total trainable parameters after: 0


replace_linear_with_lora(model, rank=16, alpha=16)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")
# Total trainable LoRA parameters: 2,666,528
```


---

## DL method impacts task domain
- Change to pure RL

 
### Train
#### S - SFT+RLHF训练出来的模型，推理能力不佳
- 多步推理的复杂任务（如解谜题、数学推导和解决复杂的编程问题）上
- 传统的SFT + RLHF 表现不佳



T - Pure RL

- Pure RL(Without previous SFT) 不依赖人类标注，模型通过试错学习推理策略，reward：leetcode，其他的
- SFT(Cold start data) + RL -> SFT(CoT data) + RL(HF-like with verification & human preferecen reward)
- Pure SFT(Without later RL) + distillation(based on small model)


<u>Concepts</u>
推理
- 复杂问题
- 需要多步骤生成并包含中间过程/思维过程
- 解答这些问题的过程叫推理

推理模型
- 具备推理能力的模型
- 例如，回答像“法国的首都是哪里？”这种事实性的问题并不涉及推理。但如果回答像“如果一列火车以每小时60英里的速度行驶 3 小时，它能行驶多远？”这样的问题，就需要一些简单的推理，因为模型需要先识别“距离=速度 × 时间”的关系，才能得出正确答案

推理模型的中间推理步骤主要以两种方式呈现
- 在内部进行多次迭代，但不会向用户展示推理过程
- 直接体现在回答中，让用户看到完整的推理过程


##### DeepSeek R1的训练流程/推理模型的构建和优化4种策略

<u>Inference-time scaling推理时间扩展</u>
What

-  提升大语言模型推理能力的一种方法
-  指的是增加推理时的计算资源来提高模型输出的质量

How

- 一种简单的推理时间扩展方法是提示词工程(prompt engineering), 其中最典型的例子是思维链(chain-of-thought，CoT)提示
  - 输入提示词(prompt)中加入类似“一步步思考”(think step by step)这样的短语，鼓励模型先生成中间推理步骤，而不是直接输出最终答案
  - 上述思维链方法可以被看作一种推理时间扩展，因为它通过生成更多的输出词元来增加推理的计算成本
  - ![alt text](images/cot.png)
- 另一种推理时间扩展方法是使用投票和搜索算法
  - 一个简单的例子是多数投票法，即让大语言模型生成多个答案，然后我们通过多数投票来选出最有可能的正确答案
  - 同样，我们还可以使用束搜索(beam search)或其他搜索算法来生成更优质的答案


<u>Pure RL纯强化学习</u>
RL强化学习
- Agent（智能体）学习如何行动的主体
- Environment（环境）	智能体所处的世界，它会对智能体的行为做出反应
- State（状态）	当前环境的描述
- Policy（策略）	智能体根据状态选择动作的规则
- Action（动作）	智能体在某个状态下可以选择的行为
- Reward（奖励）	环境对智能体行为的反馈，用于指导学习
- 
- Value Function（价值函数）	评估某个状态或状态-动作对的长期收益
- Q-Function（动作价值函数）	评估在某状态下采取某动作的期望回报
- 
- 强化     通过“奖励”来强化某些行为，从而让智能体学会更好的决策策略。
- 强化学习（Reinforcement Learning, RL）是一种机器学习方法，它让智能体（agent）通过与环境交互来学习如何做决策，以最大化长期奖励。

DeepSeek 团队发现推理能力作为一种行为可以通过纯强化学习自发涌现
- Pure跳过了监督微调阶段
  - 与典型的强化学习流程不同（通常在强化学习之前会先进行监督微调）
  - DeepSeek-R1-Zero 完全通过强化学习进行训练，没有经历初始的监督微调阶段
- 采用了两种奖励方式
  - 准确性奖励：通过使用 LeetCode 编译器来验证代码答案的正确性，并通过一个确定性的系统来评估数学答案的准确性。
  - 格式奖励：依赖大语言模型来确保回答遵循预期的格式，比如在<think>标签内放置推理步骤。
- 令人惊讶的是，仅凭这种方法，大语言模型就已经具备了基本的推理能力,证实了使用纯强化学习开发推理模型是可行的

![alt text](images/purerl.png)


<u>SFT + RL</u>
相较于 DeepSeek-R1-Zero，它通过额外的监督微调和强化学习进一步提升了推理性能
- DeepSeek 团队使用 DeepSeek-R1-Zero 生成了他们所称的“冷启动”监督微调数据。“冷启动”是指这些数据是由 DeepSeek-R1-Zero 模型生成的，而该模型本身并未接受任何监督微调数据的训练
- 在获得这些“冷启动”监督微调数据后，DeepSeek 团队对模型进行了指令微调(instruction fine-tuning)
- 随后又进行了一个强化学习阶段。这个强化学习阶段沿用了DeepSeek-R1-Zero中的奖励机制，包括准确性奖励（验证数学和代码问题的正确性）和格式奖励（确保输出符合预期格式）。除此之外，他们还新增了一个一致性奖励，以避免模型在回答中混用多种语言的问题
- 在强化学习阶段之后，他们进行了新一轮的监督微调数据收集。在这一阶段中，他们使用最新的模型检查点(checkpoint)生成了60 万条思维链监督微调样本，同时还基于DeepSeek-V3 基础模型生成了20 万条基于知识的监督微调样本
- 随后，这 80(20+60)万条监督微调数据被用于指令微调 DeepSeek-V3 基础模型
- 然后又进行最后一轮的强化学习训练。在这一阶段，他们继续使用基于规则的方法对数学和编程问题的答案给予准确性奖励，而对其他类型的问题引入了基于人类偏好标签的奖励机制。

![alt text](images/sftrl.png)


<u>Pure SFT and distillation纯监督微调与蒸馏</u>

传统的知识蒸馏中
- 较小的“学生模型”会在较大的“教师模型”的 logits 和target上进行训练，学说话。

通过蒸馏过程训练小型模型
- DeepSeek的蒸馏方法是通过使用 DeepSeek-V3 和 DeepSeek-R1的中间检查点生成的监督微调数据集(与训练 DeepSeek-R1 时使用的数据集完全相同)，来对较小的大语言模型（如参数量为 80 亿或 700 亿的 Llama 模型以及参数量为 5 亿~320 亿的 Qwen 2.5 模型）进行指令微调
- 为什么 DeepSeek 团队要开发这些蒸馏模型？我认为有两个主要原因。
  - 小型模型具有更高的效率。这意味着它们运行成本更低，同时还可以在低端硬件上运行，这对许多研究人员和像我这样的技术爱好者来说特别具有吸引力。
  - 纯监督微调的案例研究。这些蒸馏模型作为一个有趣的基准，展示了在没有强化学习的情况下，纯监督微调能将模型提升到什么程度。



R 

推理行为是从纯强化学习中**涌现**出来的