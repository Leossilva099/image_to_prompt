import random
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

DIMENSIONS = ["subject", "lighting", "style", "composition", "mood"]
DIVERSITY_THRESHOLD = 0.85

def load_llm(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    llm.eval()
    return llm, processor


def unload_llm(llm, processor):
    del llm, processor
    torch.cuda.empty_cache()


def cosine_sim_to_pool(prompt, pool_prompts, clip_model, clip_processor):
    if not pool_prompts:
        return 0.0
    texts = pool_prompts + [prompt]
    inputs = clip_processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(clip_model.device)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    features = torch.nn.functional.normalize(features, dim=-1)
    sims = (features[-1:] @ features[:-1].T).squeeze(0)
    return float(sims.max().item())


def is_diverse_enough(prompt, population, clip_model, clip_processor, threshold=DIVERSITY_THRESHOLD):
    return cosine_sim_to_pool(prompt, [c["prompt"] for c in population], clip_model, clip_processor) < threshold


def format_top_k(candidates, top_k=5):
    weights = [c["fitness"] for c in candidates]
    selected = random.choices(candidates, weights=weights, k=top_k)
    return "\n".join(f"Score {c['fitness']:.3f}: {c['prompt'].strip()}" for c in selected)


def generate_one_dimension(image, candidates, llm, processor, dimension, temperature):
    top_refs = format_top_k(candidates, top_k=5)

    DIMENSION_GUIDE = {
        "subject": "subject — its appearance, material, colour, and details",
        "lighting": "lighting — quality, direction, colour temperature, and shadows",
        "style": "artistic style and rendering quality",
        "composition": "framing, perspective, depth of field, and spatial arrangement",
        "mood": "emotional tone and narrative atmosphere",
    }

    emphasis = DIMENSION_GUIDE[dimension]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive the target image and the top-scoring prompts found so far.\n"
                "Your task: write ONE new, complete prompt that reproduces the target image "
                f"and particularly excels at **{emphasis}**.\n\n"
                "Rules:\n"
                " - Cover all visual aspects (subject, lighting, style, composition, mood).\n"
                " - Maximum 70 tokens. Every sentence must be complete. **NEVER cut mid-sentence**.\n"
                " - Use a different structure and opening from all references. Recombine and improve.\n"
                " - Output ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"TOP REFERENCES:\n{top_refs}\n\n"
                        f"Write a complete prompt that excels at **{emphasis}**. "
                        "Maximum 70 tokens. End with a complete sentence. **NEVER cut mid-sentence**. Output ONLY the prompt text."
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
    ).to(llm.device)

    torch.cuda.empty_cache()
    with torch.no_grad():
        generated_ids = llm.generate(
            **inputs,
            max_new_tokens=72,
            temperature=temperature,
            do_sample=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def generate_initial_candidates(
    image_path,
    llm,
    processor,
    candidates,
    clip_model,
    clip_processor,
    n_candidates=5,
    temperature=0.9,
    max_retries=3,
):
    image = Image.open(image_path).convert("RGB")
    results = []
    for i in range(n_candidates):
        dimension = DIMENSIONS[i % len(DIMENSIONS)]
        existing = [c["prompt"] for c in candidates] + results
        diverse = False
        for attempt in range(max_retries):
            prompt = generate_one_dimension(image, candidates, llm, processor, dimension, temperature)
            if cosine_sim_to_pool(prompt, existing, clip_model, clip_processor) < DIVERSITY_THRESHOLD:
                diverse = True
                break
        if not diverse:
            print(f" [{i+1:02d}/{n_candidates}] skipped (diversity not achieved after {max_retries} attempts)")
            continue
        results.append(prompt)
        print(f"  [{i+1:02d}/{n_candidates}] {prompt}")
    print(f" {len(results)} generated candidates")
    return results
