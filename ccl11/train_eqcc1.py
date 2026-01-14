# jwsong 2025/3
import functools
import os
import random
import json
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

sys.setrecursionlimit(8000)
import math

from transformers.loss.loss_utils import ForCausalLMLoss


import time

os.environ["HF_HUB_OFFLINE"] = '1'

from accelerate import Accelerator

from datasets import load_dataset, concatenate_datasets
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple
import os
from transformers import GenerationConfig, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from datetime import datetime
from peft import (
    LoraConfig, get_peft_model, PeftModel,
)
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
import torch



# 超参数

@dataclass
class HyperParameters:
    epoch: Optional[int] = field(default=100)
    train_batch_size: Optional[int] = field(default=4)
    eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=2)
    lr: Optional[float] = field(default=1.8e-5)
    weight_decay: Optional[float] = field(default=1e-2)
    gradient_checkpointing: Optional[bool] = field(default=True)
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=512)
    max_grad_norm: Optional[float] = field(default=0.5)
    output_dir: Optional[str] = field(default="./saved_lora")
    log_step: Optional[int] = field(default=5)
    eval_step: Optional[int] = field(default=10)
    nef_tune: Optional[bool] = field(default=False)
    noise_alpha: Optional[int] = field(default=5)
    bf16: Optional[bool] = field(default=True)
    save_log: Optional[bool] = field(default=False)
    unsloth: Optional[bool] = field(default=False)
    max_pixels: Optional[int] = field(default=512 * 28 * 28)
    # 选择对应任务
    task: Optional[str] = field(default="task2") #或者task2


config = HyperParameters()
#from unsloth import FastVisionModel

if config.unsloth:
    config.gradient_checkpointing = False

# 训练集
DATA_task1="./data/task1_train.json"
DATA_task2="./data/task2_train.json"

CUTOFF_LEN = 8000

# 模型路径
model_path = "./Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_path,use_fast=True)
#print(processor.tokenizer("）\n"))
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss,apply_liger_kernel_to_qwen2_5_vl

# LigerKernel优化，这是一个用于加速大语言模型推理和训练的自定义核函数库
apply_liger_kernel_to_qwen2_5_vl(
    rope=True,
    swiglu=True,
    cross_entropy=False,
    fused_linear_cross_entropy=False,
    rms_norm=True
)

# 自定义前向传播
def forward_fused_ce(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **loss_kwargs,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        # token转embedding
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            # 对齐精度
            pixel_values = pixel_values.type(self.visual.dtype)
            # 图像embedding
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            # <image>数量
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            # 图像向量数
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            # 找到<image>位置
            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            # 用图片向量替换掉<image>
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    # 基础理解结果
    hidden_states = outputs[0]
    loss = None
    logits = None
    if self.training and (labels is not None):
        shift_hidden_states = hidden_states[..., :-1, :].contiguous().to(self.lm_head.weight.device)
        # print(self.lm_head.weight.device)
        shift_labels = labels[..., 1:].contiguous().to(self.lm_head.weight.device)

        # Flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
        num_items_in_batch = loss_kwargs.get('num_items_in_batch', 1)
        # print(num_items_in_batch)
        # print(shift_hidden_states)
        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels) / int(num_items_in_batch)
        # print(loss)
    else:
        logits = self.lm_head(hidden_states)
        if labels is not None:
            loss = ForCausalLMLoss(logits, labels, self.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )

Qwen2_5_VLForConditionalGeneration.forward=forward_fused_ce

def getModel(lora_r, lora_alpha):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                                                    trust_remote_code=True,
                                                    torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2",
                                                    # load_in_4bit=True,
                                                    # torch_dtype="auto",
                                                    # load_in_4bit=True,
                                                    device_map="auto")

    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.08,
        # use_rslora=True,
        #modules_to_save=['merger'],
        # inference_mode=False,
        bias="none",
        # target_modules="all-linear",
        target_modules=['down_proj', 'gate_proj', 'o_proj', 'up_proj', 'q_proj', 'k_proj', 'v_proj'],
        task_type="CAUSAL_LM",
    )
    # model = get_peft_model(model, lora_config,autocast_adapter_dtype=True)

    LORA_WEIGHTS = "/root/shared-nvme/ccl_11/saved_lora/04-08-23-02task2_eval_{'ROUGE-1'_ 0.7209, 'ROUGE-2'_ 0.6118, 'ROUGE-L'_ 0.6914, 'score'_ 0.6764}"
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        autocast_adapter_dtype=False
    )
    #model = model.merge_and_unload()
    # for name,param  in model.named_parameters():
    #     if 'visual' in name:
    #         param.requires_grad = False

    for n, p in model.named_parameters():
        print(n, p.dtype)
    return model


#cache_1 = torch.empty((1024 * 1024 * 2500,), device=1)

# batch内容转向量
class CustomCollateFn:
    def __init__(self, mode):
        self.mode = mode
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.image_base_path = "./data/Task1/Train/huizong" if config.task=="task1" else "./data/Task2/Train/img"
        self.test_image_base_path = "./data/Task1/Test/img" if config.task=="task1" else "./data/Task2/Test/img"

        if self.mode == "train":
            self._caller=self.trainFn
        elif self.mode == "eval":
            self._caller = self.evalFn
        elif self.mode == "test":
            self._caller = self.testFn

    def __call__(self, examples):
        return self._caller(examples)

    def trainFn(self, examples):
        texts = []
        images = []
        batch_size = len(examples)

        def find_last_match_indices(row, target):
            # 从右到左查找目标片段的起始位置
            row_len = row.size(0)
            target_len = target.size(0)
            for i in range(row_len - target_len, -1, -1):
                if torch.equal(row[i:i + target_len], target):
                    return i
            return -1

        for example in examples:
            # 目的replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            instruction = self.replace_placeholder(example["instruction"])
            messages = [
                {
                    "role": "system",
                    "content": example["system"],
                },
                {
                    "role": "user",
                    "content": instruction,
                },
                {
                    "role": "assistant",
                    "content": example["output"],
                },
            ]
            # 拼接
            message = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ).rstrip("\n")
            image = self.get_image_info(example["image"])
            texts.append(message)
            images.append(image)
            if image is None:
                images = None
        self.processor.tokenizer.padding_side="right"
        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        labels = batch["input_ids"].clone()
        target = torch.tensor([151644, 77091, 198])

        start_indices = []
        for i in range(batch_size):
            label = labels[i]
            start_idx = find_last_match_indices(label, target)
            start_indices.append(start_idx)

        # 创建mask并设置mask
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i in range(batch_size):
            start_idx = start_indices[i]
            if start_idx >= 0:
                end_idx = start_idx + len(target)
                mask[i, :end_idx] = True  # 包括target本身

        labels[mask] = -100
        labels[labels==self.tokenizer.pad_token_id]=-100
        batch["labels"] = labels
        assert len(batch["labels"][0]) == len(batch["input_ids"][0])
        return batch

    def evalFn(self, examples):

        start_time=time.time()
        texts = []
        images = []

        for example in examples:
            instruction = self.replace_placeholder(example["instruction"])
            messages = [
                {
                    "role": "system",
                    "content": example["system"],
                },
                {
                    "role": "user",
                    "content": instruction,
                }
            ]
            self.processor.tokenizer.padding_side = "left"
            message = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image = self.get_image_info(example["image"])
            texts.append(message)
            images.append(image)
            if image is None:
                images = None

        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        labels = [example["output"] for example in examples]
        batch["labels"] = self.tokenizer(labels, padding=True, return_tensors="pt",
                                         return_attention_mask=False).input_ids
        end_time=time.time()
        elapsed_time = end_time - start_time
        #print(elapsed_time)
        return batch

    def testFn(self, examples):
        texts = []
        images = []
        self.processor.tokenizer.padding_side = "left"
        ids=[]
        for example in examples:
            instruction = self.replace_placeholder(example["instruction"])
            ids.append(example['image'][0])
            messages = [
                {
                    "role": "system",
                    "content": example["system"],
                },
                {
                    "role": "user",
                    "content": instruction,
                },
            ]
            message = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image = self.get_image_info(example["image"], infer=True)
            texts.append(message)
            images.append(image)
            if image is None:
                images = None

        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        batch['ids']=ids
        return batch

    def get_image_info(self, image_list, min_pixel=256 * 28 * 28, max_pixel=config.max_pixels, infer=False,image_base_path=None):
        if image_base_path is None:
            image_base_path=self.image_base_path if not infer else self.test_image_base_path
        if len(image_list) < 1:
            return None
        if len(image_list) == 1:
            content = [{
                "type": "image",
                "image": os.path.join(image_base_path, image_list[0]),
                #"min_pixel": min_pixel,
                #"max_pixel": max_pixel
            }]
        elif len(image_list) > 1:
            content = [{
                "type": "image",
                "image": os.path.join(image_base_path, item),
                #"min_pixel": min_pixel,
                #"#max_pixel": max_pixel
            } for item in image_list]

        messages = [
            {"role": "user",
             "content": content
             }
        ]
        images, _ = process_vision_info(messages)
        return images

    def replace_placeholder(self, prompt):
        return prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")


def getDataLoader(train_batch_size, eval_batch_size):
    # 使用 train_test_split 将数据划分为训练集和验证集。
    # task1：大验证集（300）+ 高种子（3407）→ 适合大规模数据，强调稳定性。
    # task2：小验证集（60）+ 经典种子（42）→ 适合小规模数据，兼顾训练效率。
    if config.task=="task1":
        data = load_dataset("json", data_files=DATA_task1, keep_in_memory=True)
        data_image_split = data["train"].train_test_split(test_size=300, seed=3407, keep_in_memory=True)
    else:
        data = load_dataset("json", data_files=DATA_task2, keep_in_memory=True)
        data_image_split = data["train"].train_test_split(test_size=60, seed=42, keep_in_memory=True)

    # 从拆分后的数据中提取训练集（train）和验证集（test）。
    train_data=data_image_split['train']
    val_data=data_image_split['test']

    # 定义数据整理函数（Collate Function）
    # 作用：自定义数据整理逻辑（如填充、批处理等），分别用于训练和验证模式。
    # CustomCollateFn 是一个用户定义的类，可能包含以下功能：
        # 对输入数据进行预处理（如文本分词、图像归一化）。
        # 动态填充（padding）以保证批次内数据长度一致。
        # 根据模式（train/eval）调整数据增强策略。
    train_collate_fn = CustomCollateFn(mode="train")
    val_collate_fn = CustomCollateFn(mode="eval")
    # val_data=val_data[:100]

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=train_batch_size,
                                               collate_fn=train_collate_fn,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4,
                                               drop_last=False)
    eval_loader = torch.utils.data.DataLoader(dataset=val_data,
                                              batch_size=eval_batch_size,
                                              collate_fn=val_collate_fn,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=False)
    return train_loader, eval_loader


import csv
from torch.utils.tensorboard import SummaryWriter
from rouge_chinese import Rouge
from torch.autograd import grad_mode


class ccl_Trainer:
    def __init__(self, model, optimizer, tokenizer, dataloader, config,**kwargs):
        self.model = model

        self.optimizer, self.scheduler = optimizer
        self.train_loader, self.eval_loader = dataloader
        self.tokenizer = tokenizer
        self.config = config
        # print(self.model.base_model.model.__class__.__name__)
        self.best_metric = 0.0
        self.eval_times = 0
        if self.config.save_log:
            self.writer = SummaryWriter("cail_log/instruction")
        if self.config.task=="task1":
            self.eval_metric_func=self.compute_f1
        else:
            self.eval_metric_func=self.compute_rouge

    @staticmethod
    def compute_f1(predictions:List[str], true_labels:List[str]):
        """
        评估函数，计算每个类别的 Precision, Recall 和 F1，
        同时计算 macro-average 和 micro-average 指标。

        参数:
            predictions: list[str]，预测标签列表，元素为"优秀"、"中等"、"不合格"
            true_labels: list[str]，真实标签列表，元素为"优秀"、"中等"、"不合格"

        返回:
            metrics: dict，包含每个类别的 P, R, F1 值
            averages: dict，包含 macro 和 micro 平均值指标
        """
        classes = ["优秀", "中等", "不合格"]
        metrics = {}
        # 用于计算 micro 级别指标的累加
        total_tp, total_fp, total_fn = 0, 0, 0

        for cls in classes:
            # 统计该类别下的真阳性、假阳性、假阴性数量
            tp = sum(1 for pred, true in zip(predictions, true_labels) if pred == cls and true == cls)
            fp = sum(1 for pred, true in zip(predictions, true_labels) if pred == cls and true != cls)
            fn = sum(1 for pred, true in zip(predictions, true_labels) if pred != cls and true == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[cls] = {"P": precision, "R": recall, "F1": f1}

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # macro 平均：各类别指标的简单平均
        macro_precision = sum(metrics[c]["P"] for c in classes) / len(classes)
        macro_recall = sum(metrics[c]["R"] for c in classes) / len(classes)
        macro_f1 = sum(metrics[c]["F1"] for c in classes) / len(classes)

        # micro 平均：将所有类别样本合并后计算指标
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                        micro_precision + micro_recall) > 0 else 0.0

        averages = {
            "macro": {"P": round(macro_precision,4), "R": round(macro_recall,4), "F1": round(macro_f1,4)},
            "micro": {"P": round(micro_precision,4), "R": round(micro_recall,4), "F1": round(micro_f1,4)}
        }
        for cls in classes:
            metrics[cls]={k:round(v,4) for k,v in metrics[cls].items()}
        tqdm.write(str(metrics))
        return averages

    @staticmethod
    def compute_rouge(predictions, references):
        assert len(predictions) == len(references), "predictions 与 references 长度不一致。"
        predictions=[' '.join(x) for x in predictions]
        references=[' '.join(x) for x in references]
        rouge = Rouge()

        scores = rouge.get_scores(predictions, references, avg=True)
        # scores 的结构示例：
        # {
        #   'rouge-1': {'r': 0.9, 'p': 0.85, 'f': 0.87},
        #   'rouge-2': {'r': 0.7, 'p': 0.65, 'f': 0.67},
        #   'rouge-l': {'r': 0.88, 'p': 0.83, 'f': 0.85}
        # }

        rouge_1 = scores['rouge-1']['f']
        rouge_2 = scores['rouge-2']['f']
        rouge_l = scores['rouge-l']['f']

        final_score = 0.4 * rouge_l + 0.3 * rouge_2 + 0.3 * rouge_1

        metrics = {
            'ROUGE-1': round(rouge_1,4),
            'ROUGE-2': round(rouge_2,4),
            'ROUGE-L': round(rouge_l,4),
            'score': round(final_score,4)
        }
        return metrics

    @functools.lru_cache(maxsize=512)
    def get_original_text(self, inputs) -> List[str]:
        inputs[inputs == -100] = self.tokenizer.pad_token_id
        return self.tokenizer.batch_decode(inputs, skip_special_tokens=True)

    def evaluation(self, model, eval_loader)-> dict:

        torch.cuda.empty_cache()
        model.eval()
        #model.config.pad_token_id = self.tokenizer.pad_token_id
        #model.config.eos_token_id = self.tokenizer.eos_token_id
        model.config.use_cache = True
        generation_config = GenerationConfig(
            # temperature=0.5,
            # top_p = 0.85,
            # num_beams=3,
            do_sample=False,
            # repetition_penalty=2.0,
            max_new_tokens=256,
            eos_token_id=[self.tokenizer.eos_token_id],
        )
        eval_bar = tqdm(colour="yellow", desc=f"Evaluation(eval_batch_size={config.eval_batch_size})",
                        total=len(eval_loader), dynamic_ncols=True)
        predictions = []
        true_labels = []
        start_time = time.time()
        for step, batch in enumerate(eval_loader):
            model.eval()

            labels = self.get_original_text(batch.pop("labels"))

            #labels=[item.split("\n标签：")[1] for item in labels]
            true_labels.extend(labels)

            with grad_mode.inference_mode():
                # if len(batch['input_ids']) > 1:
                generate_ids = model.generate(**batch, generation_config=generation_config, )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(batch.input_ids, generate_ids)
            ]
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # tqdm.write(str(output))
            #output=[item.split("\n")[0].strip() for item in output]
            #tqdm.write(str(output))
            predictions.extend(output)
            eval_bar.update(1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        tqdm.write(f"总时间:{elapsed_time}s,平均时间:{elapsed_time / len(eval_loader)}s")

        model.config.use_cache = False

        eval_bar.close()
        metrics = self.eval_metric_func(predictions, true_labels)
        self.eval_times += 1
        if self.config.save_log:
            pass
        return metrics

    def _save(self, model, eval_result:dict):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m-%d-%H-%M")

        file_name = formatted_time + self.config.task + "_eval_" + str(eval_result)
        output_dir = os.path.join(self.config.output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(
            output_dir, safe_serialization=False
        )
        tqdm.write(f"评估突破了阈值，lora权重保存至{output_dir}")

    def infer(self, model, batch_size=1,write_path="./data/Task2/Test/task22-answer-submit.json"):
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        fw = open(write_path, mode='w', newline='', encoding="utf-8")
        writer = csv.writer(fw)

        data_test=load_dataset("json", data_files="./data/task2_test.json", keep_in_memory=True)["train"]

        accelerator = Accelerator()
        test_collate_fn = CustomCollateFn(mode="test")
        test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                                  batch_size=batch_size,
                                                  collate_fn=test_collate_fn,
                                                  num_workers=4,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  drop_last=False)
        model, test_loader, eval_loader = accelerator.prepare(
            model, test_loader, self.eval_loader
        )
        eval_rusult = self.evaluation(model, eval_loader)
        tqdm.write(str(eval_rusult))

        torch.cuda.empty_cache()
        accelerator = Accelerator()
        model, test_loader = accelerator.prepare(
            model, test_loader
        )
        output_json = []
        model.eval()
        model.config.use_cache = True
        generation_config = GenerationConfig(
            # temperature=0.5,
            # top_p = 0.85,
            # num_beams=3,
            do_sample=False,
            # repetition_penalty=2.0,
            max_new_tokens=512,
            eos_token_id=[self.tokenizer.eos_token_id],
        )
        eval_bar = tqdm(colour="yellow", desc=f"Evaluation",
                        total=len(test_loader), dynamic_ncols=True)
        total = 0
        # writer.writerow(["id", "predict"])
        output_json= {}
        for step, batch in enumerate(test_loader):
            eval_bar.update(1)
            ids = batch.pop("ids")

            with grad_mode.inference_mode():
                generate_ids = model.generate(**batch, generation_config=generation_config, )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(batch.input_ids, generate_ids)
            ]
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            tqdm.write(str(output))
            #output = [item.split("\n标签：")[1] if "\n标签：" in item else "" for item in output]
            for out, id_ in zip(output, ids):
                output_json[id_]=out
        json.dump(output_json, fw, ensure_ascii=False, indent=2)
        fw.close()
        eval_bar.close()

    @staticmethod
    def get_batch_samples(epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
                # print(num_items_in_batch)
            except (TypeError, AttributeError):
                pass

        return batch_samples, num_items_in_batch

    def train(self):
        torch.cuda.empty_cache()

        accelerator = Accelerator()
        model, self.optimizer, train_loader, eval_loader = accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.eval_loader
        )

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        if self.config.nef_tune:
            orig_embed = model.get_input_embeddings()
        total_batched_samples = 0
        total_length = len(train_loader) * self.config.epoch // self.config.gradient_accumulation_steps
        progress_bar = tqdm(colour="blue", desc=f"Training", total=total_length, dynamic_ncols=True)
        for epoch in range(self.config.epoch):
            total_loss = 0.0
            model.config.use_cache = False
            epoch_dataloader = train_loader
            epoch_iterator = iter(epoch_dataloader)
            remainder = len(train_loader.dataset) % self.config.gradient_accumulation_steps
            if remainder == 0:
                remainder = self.config.gradient_accumulation_steps
            # if total_batched_samples % 2000==0:
            #     torch.cuda.empty_cache()
            update_step = -1
            total_updates = len(train_loader) // self.config.gradient_accumulation_steps + 1
            mini_step = 0
            for _ in range(total_updates):
                update_step += 1
                num_batches = self.config.gradient_accumulation_steps if update_step != (
                            total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for step, batch in enumerate(batch_samples):
                    mini_step += 1
                    # batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, 4)
                    total_batched_samples += 1
                    model.train()
                    # if self.config.nef_tune:
                    #     embed_init = orig_embed(batch.input_ids)
                    #     dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                    #     mag_norm = self.config.noise_alpha / torch.sqrt(dims)
                    #     batch['inputs_embeds'] = embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
                    #     batch.pop('input_ids')

                    loss_kwargs={"num_items_in_batch": num_items_in_batch.item()}
                    batch = {**batch, **loss_kwargs}
                    # print(batch)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.bf16):
                        loss = model(**batch).loss
                        # total_loss += loss.detach().item()/self.config.gradient_accumulation_steps
                        total_loss += loss.detach().item()
                        if self.config.save_log:
                            if total_batched_samples % 2 == 0:
                                self.writer.add_scalar('Loss/train', loss.detach().item(),
                                                       math.ceil(total_batched_samples / 2))
                        # accelerator.backward(scaler.scale(loss))
                    accelerator.backward(loss)
                    del batch
                    if total_batched_samples % self.config.gradient_accumulation_steps == 0:
                        # print(total_batched_samples)
                        accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        model.zero_grad()
                        progress_bar.update(1)

                    if total_batched_samples % (self.config.log_step * self.config.gradient_accumulation_steps) == 0:
                        log_loss = total_loss / self.config.log_step
                        total_loss -= total_loss
                        tqdm.write("%s loss；%f lr: %s, epoch: %f, step: %d" % (os.environ['CUDA_VISIBLE_DEVICES'],
                                                                               log_loss,
                                                                               format(self.scheduler.get_last_lr()[0],
                                                                                      'e'), round(epoch + mini_step / len(
                            epoch_dataloader),2), total_batched_samples // self.config.gradient_accumulation_steps))
                    # if mini_step==len(epoch_dataloader)-1 and epoch<=10:
                    #     #self.infer(model,write_path=f"12-26-epoch-{epoch}.csv")
                    #     eval_rusult = self.evaluation(model, eval_loader)
                    #     # eval_rusult=eval_rusult["acc"]
                    #     self._save(model,f"epoch{epoch+1}_{eval_rusult}")
                    # 评估并保存检查点
                    # 可以通过设置total_batched_samples，训练超过一定步数后再开始评估，节约时间
                    if total_batched_samples % (
                            self.config.eval_step * self.config.gradient_accumulation_steps) == 0 and total_batched_samples > 50:
                        eval_result = self.evaluation(model, eval_loader)
                        tqdm.write("%s epoch: %f" % (
                            str(eval_result), round(epoch + mini_step / len(epoch_dataloader),2))
                                   )
                        torch.cuda.empty_cache()
                        if self.config.task=="task1":
                            if (score:=eval_result["macro"]['F1']) >= 0.9 and score>self.best_metric:
                                self.best_metric = score
                                unwrap_model = accelerator.unwrap_model(model)
                                self._save(unwrap_model, eval_result)
                        elif self.config.task=="task2":
                            if (score:=eval_result["score"]) >= 0.59 and score>self.best_metric:
                                self.best_metric = score
                                unwrap_model = accelerator.unwrap_model(model)
                                self._save(unwrap_model, eval_result)



from bitsandbytes.optim import AdamW8bit,PagedAdamW8bit

model = getModel(config.lora_r, config.lora_alpha)

# optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,weight_decay=config.weight_decay)

optimizer = AdamW8bit(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 800, eta_min=1e-6, last_epoch=-1)

trainer = ccl_Trainer(model=model,
                       optimizer=(optimizer, scheduler),
                       dataloader=getDataLoader(config.train_batch_size, config.eval_batch_size),
                       tokenizer=processor.tokenizer,
                       config=config
                       )

# trainer.train()
trainer.infer(model)

# trainer.train()

