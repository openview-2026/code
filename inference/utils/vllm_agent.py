from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import base64

from .parser import parser
from .prompts import (
    CAPTION_SYSTEM, CAPTION_USER_TEMPLATE, # caption
    SUMMARIZE_SYSTEM, SUMMARIZE_WITH_LABEL_SYSTEM, SUMMARIZE_USER_TEMPLATE, # summarize
    BASE_PROPOSAL_SYSTEM, PROPOSAL_USER_TEMPLATE, # proposal
    CONTEXTUAL_SYSTEM, DIRECTIONAL_SYSTEM,
    CORRECT_SYSTEM # correct
)
from .utils import views_block_str, out_size_from_fov, spec_check

PROMPT_DICT = {
    "contextual": CONTEXTUAL_SYSTEM,
    "directional": DIRECTIONAL_SYSTEM,
}

STAGE_PIXEL_PRESETS = {
    "captions": {
        "min_pixels": 224 * 224,
        "max_pixels": 768 * 28 * 28,
    },
    "summary": {
        "min_pixels": 224 * 224,
        "max_pixels": 256 * 28 * 28,
    },
    "proposals": {
        "min_pixels": 224 * 224,
        "max_pixels": 768 * 28 * 28,
    },
    "default": {
        "min_pixels": 224 * 224,
        "max_pixels": 1024 * 28 * 28,
        "total_pixels": 20480 * 28 * 28,
    }
}

def encode_image(paths: list[str]) -> str:
    if paths is None: return None
    images = []
    for path in paths:
        with open(path, "rb") as f:
            images.append(base64.b64encode(f.read()).decode("utf-8"))
    return images

def manual_chat_template(history):
    prompt = ""
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # Flatten if content is a list of dicts (for multimodal input)
        if isinstance(content, list):
            pieces = []
            for c in content:
                t = c.get("type", "text")
                v = c.get(t, c.get("text", ""))
                if t == "text":
                    pieces.append(v)
                else:
                    pieces.append(f"<{t}>")
            content_text = " ".join(pieces)
        else:
            content_text = content
        if role == "system":
            prompt += f"[SYSTEM] {content_text}\n"
        elif role == "user":
            prompt += f"[USER] {content_text}\n"
        elif role == "assistant":
            prompt += f"[ASSISTANT] {content_text}\n"
        else:
            prompt += f"{content_text}\n"
    return prompt


class AgentMLLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = LLM(
            model=self.cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"),
            max_model_len=self.cfg.get("max_model_len", 32768),
            tensor_parallel_size=self.cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.cfg.get("gpu_memory_utilization", 0.95),
            limit_mm_per_prompt=self.cfg.get("limit_mm_per_prompt", {"image": 1}),
            enforce_eager=self.cfg.get("enforce_eager", False),
            dtype=self.cfg.get("dtype", "bfloat16"),
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"))
        self.default_sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.4,
            top_p=0.8,
            repetition_penalty=1.05,
        )
        self.backend = self.cfg.get("model_id", "Qwen2.5-VL-3B-Instruct").split("/")[-1]

    def parse_input(self, query=None, imgs=None, vid=None, stage=None):
        # pure-text question
        if imgs is None and vid is None:
            return [{"role": "user", "content": query}]

        # multimodal question
        content = []
        settings = STAGE_PIXEL_PRESETS.get(stage or "default")

        if imgs is not None:
            if not isinstance(imgs, list):
                imgs = [imgs]
            for img in imgs:
                content.append({"type": "image", "image": img, **settings})

        if vid is not None:
            content.append({"type": "video", "video": vid, **settings})


        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]

    def get_sampling_params(self, stage=None):
        
        temp_cfg = self.cfg.get(f"{stage}_cfg", None) if stage else None

        if temp_cfg:
            sampling_params = SamplingParams(
                temperature=temp_cfg[0],
                top_p=temp_cfg[1],
                max_tokens=temp_cfg[2],
                repetition_penalty=temp_cfg[3],
            )
        else:
            sampling_params = self.default_sampling_params
        return sampling_params

    def build_chat_template(self, imgs, system_prompt, query, stage=None, history=None):
        if history is None:
            history = [{"role": "system", "content": system_prompt}]
        else:
            history.extend([{"role": "system", "content": system_prompt}])
        history.extend(self.parse_input(query, imgs, stage=stage))

        image_inputs = None
        try:
            prompt_text = self.processor.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, _ = process_vision_info(history)
        except Exception:
            prompt_text = manual_chat_template(history)
            image_inputs = encode_image(imgs)

        mm_data = {}
        if image_inputs:
            mm_data["image"] = image_inputs

        if mm_data:
            return {"prompt": prompt_text, "multi_modal_data": mm_data}
        else:
            return {"prompt": prompt_text}

    def parse_response(self, raw_response, list_obj=False):
        n = len(raw_response)
        responses = [[{'error': '_ERROR_'}] for _ in range(n)]
        error_indices = []
        error_responses = []
        errors = []

        for i, item in enumerate(raw_response):
            try:
                text = getattr(item.outputs[0], "text", None)
                if text is None:
                    raise ValueError("[ERROR] Missing text in model response.")
                # Attempt to parse this one item
                parsed_text = parser(text)
                if not parsed_text:
                    print(f"[WARNING] Empty text in model response")
                    continue
                if not list_obj and not isinstance(parsed_text, dict):
                    print(f"[ERROR] Parsing error: output not in JSON format: {text}")
                    continue
                
                if list_obj and not isinstance(parsed_text, list):
                    parsed_text = [parsed_text]
                responses[i] = parsed_text
            except Exception as e:
                # print(f"[ERROR] First round parsing error: {e}")
                error_indices.append(i)
                raw_text = text if text is not None else repr(item)
                error_responses.append(raw_text)
                errors.append(e)

        if error_indices:
            print(f"[ERROR] Parsing errors in indices {error_indices} in batch size = {n}")
            corrected = self.correct(error_responses, errors)
        
            for idx, fixed in zip(error_indices, corrected):
                if not fixed:
                    print(f"[ERROR] Unable to fix at index {idx}")
                    continue
                print(f"[FIXED] Fixed at index {idx}")
                if list_obj and not isinstance(fixed, list):
                    fixed = [fixed]
                responses[idx] = fixed

        return responses

    def caption(self, batch_path):
        sampling_params = self.get_sampling_params("caption")

        batch_inputs = []
        for image_path in batch_path:
            batch_inputs.append(
                self.build_chat_template(image_path, CAPTION_SYSTEM, CAPTION_USER_TEMPLATE, "captions")
                )
        raw_response = self.model.generate(batch_inputs, sampling_params)
        responses = self.parse_response(raw_response)
        return responses

    def summarize(self, batch_captions, batch_view_paths, batch_pano_label=None):
        sampling_params = self.get_sampling_params("summary")

        batch_inputs = []
        for idx, captions in enumerate(batch_captions):
            if batch_pano_label[idx] == "unknown" or batch_pano_label[idx] is None:
                system_prompt = SUMMARIZE_WITH_LABEL_SYSTEM
            else:
                system_prompt = SUMMARIZE_SYSTEM
                
            batch_inputs.append(
                self.build_chat_template(
                    batch_view_paths[idx], system_prompt, 
                    SUMMARIZE_USER_TEMPLATE.format(captions=views_block_str(captions)), 
                    "summary"
                    )
                )
        raw_response = self.model.generate(batch_inputs, sampling_params)
        responses = self.parse_response(raw_response)
        return responses

    def propose(self, batch_summary, batch_captions, batch_pano_path, k=5, question_type="scene_understanding"):
        sampling_params = self.get_sampling_params("propose")

        batch_inputs = []
        for idx in range(len(batch_captions)):
            batch_inputs.append(
                self.build_chat_template(
                    batch_pano_path[idx], 
                    BASE_PROPOSAL_SYSTEM + PROMPT_DICT[question_type], 
                    PROPOSAL_USER_TEMPLATE.format(
                        label=batch_summary[idx].label,
                        global_summary=batch_summary[idx].summary,
                        captions=views_block_str(batch_captions[idx]),
                        k=k,
                    ), 
                    "proposals"
                    )
                )

        raw_response = self.model.generate(batch_inputs, sampling_params)
        responses = self.parse_response(raw_response, list_obj=True)

        # add extra fields to response
        for idx, response in enumerate(responses):
            for r in response.copy():
                if not spec_check(r):
                    response.remove(r)
                    continue
                
                if question_type != "other_types":
                    r["question_type"] = [question_type]
                r["pano_size"] = batch_summary[idx].pano_size
                r["pano_path"] = batch_summary[idx].pano_path
                r["category"] = batch_summary[idx].label
                r["outdoor"] = batch_summary[idx].outdoor
                r["image_size"] = out_size_from_fov(
                    batch_summary[idx].pano_size, 
                    (None, None, float(r["diag_fov"])), 
                    aspect_ratio=r["aspect_ratio"]
                    )[0]

        return responses

    def correct(self, raw_contents, error_msgs):
        sampling_params = self.get_sampling_params("correct")

        batch_inputs = []
        for raw_content, error_msg in zip(raw_contents, error_msgs):
            batch_inputs.append(
                self.build_chat_template(imgs=None, system_prompt=CORRECT_SYSTEM, query=f"Raw message: {raw_content}\nError message: {error_msg}")
                )
        raw_response = self.model.generate(batch_inputs, sampling_params)

        responses = []
        for response in raw_response:
            try:
                # Check if outputs list is empty before accessing
                if not response.outputs:
                    print(f"[ERROR] Empty outputs in response, skipping")
                    responses.append(None)
                    continue
                
                text = getattr(response.outputs[0], "text", None)
                if text is None:
                    print(f"[ERROR] Missing text in model response")
                    responses.append(None)
                    continue
                if text == "[]":
                    print(f"[ERROR] Empty text in model response")
                    responses.append(None)
                    continue
                    
                responses.append(parser(text))
            except Exception as e:
                print(f"[ERROR] Correcting error: {e}, Corrected text: {text}, return None")
                responses.append(None)
        return responses

def build_vllm_agent(cfg):
    return AgentMLLM(cfg.get("agent_config", {}) or {})