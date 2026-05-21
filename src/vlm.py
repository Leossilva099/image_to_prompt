import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def load_vlm(model_id = "Qwen/Qwen3-VL-8B-Instruct"):
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    vlm.eval()
    return vlm, processor


def generate_one_candidate(
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
                f"STRUCTURE GUIDE: {hint}\n\n"
                "The prompt must cover all key visual aspects (subject, setting, lighting, style, composition) "
                "but follow the structure prescribed above. "
                "CRITICAL: maximum 70 tokens. Every prompt must be complete. "
                "**VERY IMPORTANT** - NEVER cut mid-sentence. "
                "Return ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Follow the structure guide and write a complete prompt that starts as instructed and emphasises {dimension}. "
                        "Maximum 70 tokens. End with a complete sentence. **NEVER cut mid-sentence**. Return ONLY the prompt text."
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
            max_new_tokens=72,
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
        "START with the main subject: its shape, colour, material, and physical details. "
        "Then add setting, lighting, and style.",
        "subject"
    ),
    (
        "START with the lighting conditions: quality, direction, colour temperature, and shadows. "
        "Then describe the subject, setting, and overall atmosphere.",
        "lighting"
    ),
    (
        "START with the dominant colour palette and tones. "
        "Then describe the subject, composition, and mood.",
        "colours"
    ),
    (
        "START with the framing and spatial arrangement: perspective, depth of field, and focal point. "
        "Then describe the subject, lighting, and style.",
        "composition"
    ),
    (
        "START with the emotional atmosphere and narrative feeling. "
        "Then describe the subject, setting, and visual style.",
        "mood"
    ),
    (
        "START with the artistic style and rendering quality: medium, technique, and visual treatment. "
        "Then describe the subject, lighting, and composition.",
        "style"
    ),
    (
        "START with the surface textures and material details of the key objects. "
        "Then describe the subject, background, and lighting.",
        "textures"
    ),
    (
        "START with the background and environment: its colour, texture, and depth. "
        "Then describe the main subject and lighting.",
        "background"
    ),
    (
        "START with camera and lens parameters: focal length, aperture, depth of field, and camera angle. "
        "Then describe subject, lighting, and colour grading.",
        "photography"
    ),
    (
        "START with the tonal contrast: bright highlights versus deep shadows. "
        "Then describe the subject, composition, and saturation.",
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
        prompt = generate_one_candidate(image, vlm, processor, temperature, hint, dimension)
        candidates.append(prompt)
        print(f"  [{i+1:02d}/{n_candidates}] {prompt}")

    print(f" {len(candidates)} generated candidates")
    return candidates


def unload_vlm(vlm, processor):
    del vlm, processor
    torch.cuda.empty_cache()