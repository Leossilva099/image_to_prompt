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
                "Analyse the given image and write a single, complete prompt that, when rendered with an LCM diffusion model, reproduces it as closely as possible.\n\n"
                f"EMPHASIS GUIDE: {hint}\n\n"
                "The prompt must cover all key visual aspects (subject, setting, lighting, style, composition) "
                "but give particular weight to the indicated emphasis. "
                "Under 70 tokens, no incomplete sentences. Return ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Write a complete image generation prompt with particular emphasis on {dimension}. "
                        f"Cover all visual aspects but make {dimension} the strongest element. "
                        "Under 70 tokens. Return ONLY the prompt text."
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
        "Write a complete prompt with particular emphasis on the main subject: "
        "its shape, colour, material, and physical details. Also include setting, lighting, and style.",
        "subject"
    ),
    (
        "Write a complete prompt with particular emphasis on lighting, shadows, and atmosphere. "
        "Also describe the subject, setting, and overall style.",
        "lighting"
    ),
    (
        "Write a complete prompt with particular emphasis on the colour palette and dominant tones. "
        "Also describe the subject, composition, and mood.",
        "colours"
    ),
    (
        "Write a complete prompt with particular emphasis on composition, framing, and spatial arrangement. "
        "Also describe the subject, lighting, and style.",
        "composition"
    ),
    (
        "Write a complete prompt with particular emphasis on the mood, emotion, and narrative feeling. "
        "Also describe the subject, setting, and visual style.",
        "mood"
    ),
    (
        "Write a complete prompt with particular emphasis on artistic style and rendering quality "
        "(e.g. painterly, cinematic, photorealistic, 8K). Also describe subject and lighting.",
        "style"
    ),
    (
        "Write a complete prompt with particular emphasis on textures and surface materials. "
        "Also describe the subject, background, and lighting.",
        "textures"
    ),
    (
        "Write a complete prompt with particular emphasis on the background and environment. "
        "Also describe the main subject, lighting, and overall style.",
        "background"
    ),
    (
        "Write a complete prompt with particular emphasis on photographic technique: "
        "lens, depth of field, focal length, camera angle. Also describe subject and lighting.",
        "photography"
    ),
    (
        "Write a complete prompt with particular emphasis on contrast, saturation, and visual impact. "
        "Also describe the subject, composition, and style.",
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