"""One constrained plan, then PrimeBuddy validates and executes it."""

import copy, json, time, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from bootstrap import ROOT


class LocalChat:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "model", local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            ROOT / "model",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        ).eval()
        self.calls = []

    @staticmethod
    def schema(registry):
        variants = []
        ids = [f"t{i}" for i in range(1, 9)]
        for skill, data in registry.skills.items():
            for action, definition in data["actions"].items():
                fields = {
                    k: {
                        p: v
                        for p, v in rule.items()
                        if p in ("type", "enum", "default")
                    }
                    for k, rule in definition["parameters"].items()
                }
                variants.append(
                    {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "enum": ids},
                            "skill": {"type": "string", "enum": [skill]},
                            "action": {"type": "string", "enum": [action]},
                            "args": {
                                "type": "object",
                                "properties": fields,
                                "required": list(fields),
                                "additionalProperties": False,
                            },
                            "after": {
                                "type": "array",
                                "items": {"type": "string", "enum": ids},
                                "maxItems": 7,
                            },
                        },
                        "required": ["id", "skill", "action", "args", "after"],
                        "additionalProperties": False,
                    }
                )
        return {
            "type": "object",
            "properties": {
                "reply": {"type": "string", "maxLength": 200},
                "tasks": {"type": "array", "items": {"anyOf": variants}, "maxItems": 8},
            },
            "required": ["reply", "tasks"],
            "additionalProperties": False,
        }

    @torch.inference_mode()
    def propose(self, text, registry, feedback=None, previous=None):
        catalog = [
            {"skill": name, **registry.read_file(name)} for name in registry.skills
        ]
        schema = self.schema(registry)
        system = (
            "你是机器人任务规划器。输出一个JSON计划，不执行动作。只使用已安装技能。\n"
            "任务格式为id、skill、action、args、after。id用t1、t2等。先A后B：A的after=[]，B的after=[A的id]。\n"
            "同一robot.body资源不能并行。即使用户没有强调顺序，多个身体动作也要排好先后。\n"
            "可执行请求：reply为空，tasks列出任务。含糊请求：tasks=[]，reply简短追问。\n"
            "请求目录外能力：tasks=[]，reply说明当前不支持；不要询问该能力的细节。\n"
            "下面是程序加载的全部技能说明与精确动作定义。不得编造其他能力：\n"
            + json.dumps(catalog, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        if feedback:
            messages += [
                {"role": "assistant", "content": previous},
                {
                    "role": "user",
                    "content": "计划尚未执行。程序校验错误："
                    + feedback
                    + "。请修正计划，保持原始请求。",
                },
            ]
        return self.generate(messages, schema, "plan")

    def decide(self, text, registry):
        catalog = registry.catalog()
        schema = {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["execute", "clarify", "unsupported"],
                },
                "reply": {"type": "string", "maxLength": 100},
            },
            "required": ["decision", "reply"],
            "additionalProperties": False,
        }
        system = (
            "判断用户是否明确要求机器人执行已安装动作。只分类，不安排任务。\n"
            "execute：明确要求执行，且动作种类能从原话确定，reply留空。\n"
            "clarify：动作种类不明确，或者只是询问能力而未要求执行。reply简短追问，不能自行选择动作。\n"
            "unsupported：明确要求了目录外能力。reply说明目前不支持，不追问该能力的执行细节。\n"
            "多个动作组成的明确命令也属于execute；不要把明确的命令改成确认问题。只有原话缺少动作种类时才clarify。\n"
            "根据原话判断，不要因为目录里存在技能就替用户挑一个。输出严格JSON，包含decision和reply。已安装技能："
            + json.dumps(catalog, ensure_ascii=False)
        )
        return self.generate(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            schema,
            "intent",
        )

    @torch.inference_mode()
    def generate(self, messages, schema, kind):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(kind == "intent"),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prefix = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer, JsonSchemaParser(schema)
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.model.generate(
            **inputs,
            max_new_tokens=1536 if kind == "intent" else 768,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=None if kind == "intent" else prefix,
        )
        torch.cuda.synchronize()
        ids = out[0, inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        record = {
            "input_tokens": inputs["input_ids"].shape[1],
            "output_tokens": len(ids),
            "generation_s": time.perf_counter() - start,
            "raw": raw,
            "messages": copy.deepcopy(messages),
            "schema": schema,
            "kind": kind,
        }
        self.calls.append(record)
        if ids[-1].item() != self.tokenizer.eos_token_id:
            raise ValueError("Generation truncated; no execution")
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        if answer.startswith("```json") and answer.endswith("```"):
            answer = answer[7:-3].strip()
        plan = json.loads(answer)
        if kind == "intent" and (
            set(plan) != {"decision", "reply"}
            or plan["decision"] not in {"execute", "clarify", "unsupported"}
            or not isinstance(plan["reply"], str)
        ):
            raise ValueError("Invalid intent response; no execution")
        record["plan"] = plan
        return plan
