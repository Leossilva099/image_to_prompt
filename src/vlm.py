import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def load_vlm(model_id = "Qwen/Qwen3-VL-2B-Instruct"):
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    vlm.eval()
    return vlm, processor


def _generate_one_candidate(
    image,
    vlm,
    processor,
    temperature = 0.9,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at writing LCM image generation prompts. "
                "Analyse the given image and write a single prompt that, when rendered with an LCM diffusion model, reproduces it as closely as possible. "
                "Focus on: subject, style, lighting, colours, composition, mood, and quality tags. "
                "Return ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "Write a single image generation prompt that reproduces this image. Return ONLY the prompt.",
                },
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs=inputs.to(vlm.device)

    with torch.no_grad():
        generated_ids = vlm.generate(
            **inputs,
            max_new_tokens=128,
            temperature=temperature,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    prompt = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
        )[0].strip()

    return prompt


def generate_initial_candidates(
    target_path,
    vlm,
    processor,
    n_candidates = 10,
    temperature = 0.9,
):
    image = Image.open(target_path).convert("RGB")
    candidates = []
    for i in range(n_candidates):
        prompt = _generate_one_candidate(image, vlm, processor, temperature)
        candidates.append(prompt)
        print(f"  [{i+1:02d}/{n_candidates}] {prompt}")

    print(f" {len(candidates)} generated candidates")
    return candidates


def unload_vlm(vlm, processor):
    del vlm, processor
    torch.cuda.empty_cache()