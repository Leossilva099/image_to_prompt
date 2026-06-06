import random
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DIVERSITY_THRESHOLD = 0.85


def load_llm(model_id=MODEL_ID):
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    vlm.eval()
    return vlm, processor


def unload_llm(vlm, processor):
    del vlm, processor
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


def _vlm_chat(vlm, processor, messages, temperature=0.9, max_new_tokens=72):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vlm.device)
    with torch.no_grad():
        generated_ids = vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def rank_select(population):
    ranked = sorted(population, key=lambda x: x["fitness"])
    n = len(ranked)
    weights = list(range(1, n + 1))
    total = sum(weights)
    pick = random.uniform(0, total)
    cumulative = 0.0
    for candidate, w in zip(ranked, weights):
        cumulative += w
        if cumulative >= pick:
            return candidate
    return ranked[-1]


def _format_spatial_map(grid_fitness):
    labels = [["Top-left", "Top-right"], ["Bottom-left", "Bottom-right"]]
    lines = []
    for r in range(2):
        for c in range(2):
            lines.append(f"  - {labels[r][c]:12s}: {grid_fitness[r][c]:.2f}")
    return "\n".join(lines)


def crossover(vlm, processor, parent_a, parent_b, target_image, temperature=0.7):
    map_a = _format_spatial_map(parent_a["grid_fitness"])
    map_b = _format_spatial_map(parent_b["grid_fitness"])
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will see the target image, two parent-generated images, and their 2×2 spatial fitness maps.\n"
                "Each score shows how closely that region matches the target (higher = better, max 1.0).\n\n"
                "Your goal is to write a NEW prompt that scores HIGHER than both parents by:\n"
                "1. Carefully studying the TARGET image — it is the ground truth.\n"
                "2. Identifying which regions score LOW in both parents and describing them more precisely based on what you see in the target.\n"
                "3. Keeping what already works well from whichever parent scores higher in each region.\n\n"
                "Rules:\n"
                " - Be specific and precise about visual details you observe directly in the target image.\n"
                " - Cover all visual aspects: subject, lighting, style, composition, mood.\n"
                " - Use a different sentence structure and opening words than both parents.\n"
                " - Maximum 70 tokens. Every sentence must be complete. NEVER cut mid-sentence.\n"
                " - Output ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text",  "text": "TARGET IMAGE (ground truth):"},
                {"type": "image", "image": target_image},
                {"type": "text",  "text": (
                    f"\nPARENT A — global fitness {parent_a['fitness']:.3f}\n"
                    f"Prompt: {parent_a['prompt'].strip()}\n"
                    f"Spatial fitness map:\n{map_a}"
                )},
                {"type": "image", "image": parent_a["image"]},
                {"type": "text",  "text": (
                    f"\nPARENT B — global fitness {parent_b['fitness']:.3f}\n"
                    f"Prompt: {parent_b['prompt'].strip()}\n"
                    f"Spatial fitness map:\n{map_b}"
                )},
                {"type": "image", "image": parent_b["image"]},
                {"type": "text",  "text": (
                    "\nStudy the target carefully. Identify the weak regions in both parents "
                    "and describe them more precisely from what you observe in the target. "
                    "Inherit what works well from each parent's strong regions. "
                    "Use a different structure and opening than both parents. "
                    "Maximum 70 tokens. Complete sentences only. Output ONLY the prompt."
                )},
            ],
        },
    ]
    return _vlm_chat(vlm, processor, messages, temperature=temperature)


def mutate(vlm, processor, prompt, temperature=0.9):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive a prompt. Change exactly ONE adjective or short descriptor "
                "(a lighting word, texture word, colour word, or mood word) to a different but plausible alternative. "
                "Everything else must remain word-for-word identical.\n\n"
                "Rules:\n"
                " - Change only ONE word or short phrase.\n"
                " - Do NOT change the subject, objects, or overall scene.\n"
                " - Output ONLY the modified prompt, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt:\n{prompt.strip()}\n\n"
                "Change one adjective or descriptor. Output ONLY the modified prompt."
            ),
        },
    ]
    return _vlm_chat(vlm, processor, messages, temperature=temperature)


def evolve(
    vlm, processor, population, clip_model, clip_processor,
    n_candidates=5, iteration=1, base_mutation_rate=0.3, temperature=0.9,
    threshold=DIVERSITY_THRESHOLD,
    render_fn=None, target_image=None, grid_fitness_fn=None,
):
    new_prompts = []
    for i in range(n_candidates):
        parent_a = rank_select(population)
        parent_b = rank_select(population)
        retries = 0
        while parent_b["prompt"] == parent_a["prompt"] and len(population) > 1 and retries < 5:
            parent_b = rank_select(population)
            retries += 1

        img_a = render_fn(parent_a["prompt"])
        img_b = render_fn(parent_b["prompt"])
        grid_a = grid_fitness_fn(img_a)
        grid_b = grid_fitness_fn(img_b)

        pa = {**parent_a, "image": img_a, "grid_fitness": grid_a}
        pb = {**parent_b, "image": img_b, "grid_fitness": grid_b}
        child = crossover(vlm, processor, pa, pb, target_image, temperature=0.7)

        flag = 0
        if random.random() < base_mutation_rate:
            flag = 1
            child = mutate(vlm, processor, child, temperature)

        pool = [c["prompt"] for c in population] + new_prompts
        if cosine_sim_to_pool(child, pool, clip_model, clip_processor) >= threshold:
            print(f"MUT-{flag}-[{i+1:02d}/{n_candidates}] skipped (too similar)")
            continue

        new_prompts.append(child)
        print(f"MUT-{flag}-[{i+1:02d}/{n_candidates}] {child}")

    print(f" {len(new_prompts)} evolved candidates")
    return new_prompts
