# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install unsloth
# !pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
#

from huggingface_hub import login
from dotenv import load_dotenv
import os
import wandb
# unsloth是一个优化的语言模型加载和训练库
from unsloth import FastLanguageModel
from datasets import load_dataset


load_dotenv('.env')
hf_token = os.getenv('HF_TOKEN')
login(hf_token)

wb_token = os.getenv('WB_TOKEN')
project = os.getenv('WB_PROJECT')
model_name = os.getenv('MODEL_NAME')

wandb.login(key=wb_token)
run = wandb.init(
    project=project,    # 设置项目名称 - 这里是用于SQL分析的DeepSeek模型微调项目
    job_type="training",
    anonymous="allow"# 允许匿名访问 # "allow"表示即使没有wandb账号的用户也能查看这个项目
)

# 从unsloth库中导入FastLanguageModel类


# 设置模型参数
# 最大序列长度，即模型能处理的最大token数量
max_seq_length = 2048
dtype = None # 数据类型设置为None，让模型自动选择合适的数据类型
load_in_4bit = True # 启用4bit量化加载 # 4bit量化可以显著减少模型内存占用，但可能略微影响模型性能

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,     # 设置最大序列长度
    dtype = dtype,# 设置数据类型
    load_in_4bit = load_in_4bit,  # 启用4bit量化加载
    token = hf_token, # 使用Hugging Face的访问令牌
)


train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a SQL query that appropriately answers the request.
Before writing the query, think carefully about the requirements and create a step-by-step plan to ensure a logical and accurate SQL solution.

### Instruction:
You are a SQL expert who can convert natural language questions into SQL queries. Given the database schema and a business question, generate the appropriate SQL query to retrieve the requested information.

### Query:
{}

### Response:
<think>
{}
</think>
{}"""



# 獲取當前腳本所在目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
# 構建數據文件的完整路徑
data_file_path = os.path.join(current_dir, "train.json")

dataset = load_dataset("json", data_files=data_file_path, split="train[0:500]")
EOS_TOKEN = tokenizer.eos_token

def switch_and_format_prompt(examples):
    inputs = examples["input"] # 使用 SQL 作為輸入
    context = examples["instruction"] 
    outputs = examples["output"] # 使用英文描述作為輸出
    texts = []
    for input, context, output in zip(inputs, context, outputs):
        text = train_prompt_style.format(input, context, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }

# 應用轉換
dataset = dataset.map(switch_and_format_prompt, batched=True)


model = FastLanguageModel.get_peft_model(
    model,
    r=16,   # LoRA的秩(rank)值，决定了低秩矩阵的维度，较大的r值(如16)可以提供更强的模型表达能力，但会增加参数量和计算开销，较小的r值(如4或8)则会减少参数量，但可能影响模型性能，通常在4-16之间选择,需要在性能和效率之间权衡
    target_modules=[ #指定需要应用LoRA微调的模块列表，q_proj, k_proj, v_proj: 注意力机制中的查询、键、值投影层
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj", #注意力输出投影层
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16, #缩放参数，用于控制LoRA更新的强度，通常设置为与r相同的值，较大的alpha会增加LoRA的影响力，较小的alpha则会减弱LoRA的影响
    lora_dropout=0, #LoRA层的dropout率，0表示不使用dropout，增加dropout可以帮助防止过拟合，但可能影响训练稳定性，在微调时通常设为0或很小的值
    bias="none", #是否微调偏置项，"none"表示不微调偏置参数，也可以设置为"all"或"lora_only"来微调不同范围的偏置
    use_gradient_checkpointing="unsloth",  # 梯度检查点策略，"unsloth"是一种优化的检查点策略，适用于长上下文可以显著减少显存使用，但会略微增加计算时间对处理长文本特别有用
    random_state=3407, #随机数种子，控制初始化的随机性，固定种子可以确保实验可重复性
    use_rslora=False, #是否使用RSLoRA(Rank-Stabilized LoRA) False表示使用标准LoRARSLoRA是一种改进的LoRA变体，可以提供更稳定的训练
    loftq_config=None, #LoftQ配置None表示不使用LoftQ量化LoftQ是一种用于模型量化的技术，可以减少模型大小
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    ## 训练参数配置
    args=TrainingArguments(
        # 批处理相关
        per_device_train_batch_size=2, # 每个设备（GPU）的训练批次大小
        gradient_accumulation_steps=4,# 梯度累积步数，用于模拟更大的批次大小
         # 训练步数和预热
        warmup_steps=5,# 学习率预热步数，逐步增加学习率
        max_steps=60,# 最大训练步数
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(), # 如果不支持 bfloat16，则使用 float16
        bf16=is_bfloat16_supported(),# 如果支持则使用 bfloat16，通常在新型 GPU 上性能更好
        logging_steps=10,# 每10步记录一次日志
        optim="adamw_8bit", # 使用8位精度的 AdamW 优化器
        weight_decay=0.01,# 权重衰减率，用于防止过拟合
        lr_scheduler_type="linear",# 学习率调度器类型，使用线性衰减
        seed=3407,# 随机种子，确保实验可重复性
        output_dir="outputs", # 模型和检查点的输出目录
    ),
)

trainer_stats = trainer.train()

# 这个模板包含了指导模型如何理解和解释SQL查询的结构化提示
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a SQL query that appropriately answers the request.
Before writing the query, think carefully about the requirements and create a step-by-step plan to ensure a logical and accurate SQL solution.

### Instruction:
You are a SQL expert who can convert natural language questions into SQL queries. Given the database schema and a business question, generate the appropriate SQL query to retrieve the requested information.

### Query:
{}

### Response:
<think>{}"""

# 定义测试用的SQL查询
# 这是一个复杂的客户分析查询，用于测试模型的理解能力
query1 = """
維修材料交易記錄中 交易日期在2023/1月到3月之間 所屬段的倉庫 的資料筆數分布
"""

# 将模型设置为推理模式
FastLanguageModel.for_inference(model)
# 准备输入数据
# 使用提示模板格式化查询，并转换为模型可处理的张量格式
inputs = tokenizer([prompt_style.format(query1, "")], return_tensors="pt").to("cuda")

# 生成响应
outputs = model.generate(
    input_ids=inputs.input_ids, # 输入的标记ID
    attention_mask=inputs.attention_mask, # 注意力掩码，用于处理填充
    max_new_tokens=1200, # 最大生成的新标记数
    use_cache=True,  # 使用缓存以提高生成速度
)
# 解码模型输出
# 使用分词器将输出转换回文本，并提取Response部分
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])

#微调后：解释更加具体和准确 能更好地捕捉查询的完整上下文 例如在示例中，微调后的模型特别指出了"top 10 customers with the highest total spent"这个关键细节，这在微调前的响应中是缺失的

local_path="deepseek_sql_model"

model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)

# Save merged model
# model.save_pretrained_merged(local_path, tokenizer, save_method="merged_16bit")

# model.push_to_hub(local_path)
# tokenizer.push_to_hub(local_path)

# from google.colab import drive
# drive.mount('/content/drive')
# drive_path = "/content/drive/MyDrive/deepseek_model"
# model.save_pretrained(drive_path)
# tokenizer.save_pretrained(drive_path)

