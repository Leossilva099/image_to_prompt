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
    temperature=0.9,
    hint="",
    dimension="",
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at writing LCM image generation prompts. "
                "Analyse the given image and write a single prompt that, when rendered with an LCM diffusion model, reproduces it as closely as possible.\n\n"
                f"STRICT FOCUS - {hint}\n\n"
                "The prompt must be short, complete, and under 70 tokens. "
                "No incomplete sentences. Return ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Write a prompt focused EXCLUSIVELY on {dimension}. "
                        "Under 70 tokens, complete. Return ONLY the prompt text."
                    ),
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
            max_new_tokens=70,
            temperature=temperature,
            do_sample=True,
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

STYLE_HINTS = [
    (
        "Describe ONLY the main subject: its shape, colour, and physical details. "
        "Do NOT mention lighting, style tags, background, or composition.",
        "subject"
    ),
    (
        "Describe ONLY the lighting, shadows, and atmosphere. "
        "Do NOT describe the subject in detail, add style tags, or mention composition.",
        "lighting"
    ),
    (
        "Describe ONLY the colour palette, dominant tones, and colour relationships. "
        "Do NOT describe the subject, lighting, or composition.",
        "colours"
    ),
    (
        "Describe ONLY the composition, framing, and spatial arrangement. "
        "Do NOT describe colours, mood, or artistic style.",
        "composition"
    ),
    (
        "Describe ONLY the mood, emotion, and narrative feeling of the scene. "
        "Do NOT describe physical details, lighting specifics, or technical tags.",
        "mood"
    ),
    (
        "Describe ONLY the artistic style, rendering quality, and visual technique "
        "(e.g. painterly, cinematic, photorealistic, 8K). "
        "Do NOT describe the subject, lighting, or composition.",
        "style"
    ),
    (
        "Describe ONLY the textures and surface materials visible in the image. "
        "Do NOT mention mood, lighting, or style tags.",
        "textures"
    ),
    (
        "Describe ONLY the background, setting, and environment. "
        "Do NOT describe the main subject or foreground elements.",
        "background"
    ),
    (
        "Describe ONLY the photographic technique: lens, depth of field, focal length, "
        "and camera angle. Do NOT describe colours, mood, or subject details.",
        "photography"
    ),
    (
        "Describe ONLY the contrast, saturation, and overall visual impact. "
        "Do NOT describe subject details, composition, or artistic style.",
        "contrast"
    ),
]


def generate_initial_candidates(
    target_path,
    vlm,
    processor,
    n_candidates=10,
    temperature=0.9,
):
    image = Image.open(target_path).convert("RGB")
    candidates = []
    for i in range(n_candidates):
        hint, dimension = STYLE_HINTS[i % len(STYLE_HINTS)]
        prompt = _generate_one_candidate(image, vlm, processor, temperature, hint, dimension)
        candidates.append(prompt)
        print(f"  [{i+1:02d}/{n_candidates}] [{dimension}] {prompt}")

    print(f" {len(candidates)} generated candidates")
    return candidates


def unload_vlm(vlm, processor):
    del vlm, processor
    torch.cuda.empty_cache()