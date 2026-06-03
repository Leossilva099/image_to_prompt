import random
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DIVERSITY_THRESHOLD = 0.85


def load_llm(model_id=MODEL_ID):
    llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
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


def vlm_chat(llm, processor, messages, temperature=0.9, max_new_tokens=72):
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
            max_new_tokens=max_new_tokens,
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


def crossover(llm, processor, parent_a, parent_b, target_image_path, temperature=0.9):
    target_image = Image.open(target_image_path).convert("RGB")
    img_a = Image.open(parent_a["image_path"]).convert("RGB")
    img_b = Image.open(parent_b["image_path"]).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive a target image and two generated images with their prompts.\n"
                "Your task: write ONE new prompt that takes the strongest visual elements "
                "from both generated images to better match the target.\n\n"
                "Rules:\n"
                " - Cover all visual aspects (subject shape, lighting, style, composition, mood).\n"
                " - Use a different sentence structure and opening words than both parents.\n"
                " - Maximum 70 tokens. Every sentence must be complete. **NEVER cut mid-sentence.**\n"
                " - Output ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Target image:"},
                {"type": "image", "image": target_image},
                {"type": "text", "text": f"Generated image A (fitness {parent_a['fitness']:.3f}):"},
                {"type": "image", "image": img_a},
                {"type": "text", "text": f"Prompt A: {parent_a['prompt'].strip()}"},
                {"type": "text", "text": f"Generated image B (fitness {parent_b['fitness']:.3f}):"},
                {"type": "image", "image": img_b},
                {"type": "text", "text": f"Prompt B: {parent_b['prompt'].strip()}"},
                {
                    "type": "text",
                    "text": (
                        "Combine the best visual elements from both to match the target. "
                        "Different structure and opening than both parents. "
                        "Maximum 70 tokens. End with a complete sentence. Output ONLY the prompt text."
                    ),
                },
            ],
        },
    ]
    return vlm_chat(llm, processor, messages, temperature=temperature)


def mutate(llm, processor, prompt, reference_image_path, target_image_path, temperature=0.9):
    target_image = Image.open(target_image_path).convert("RGB")
    ref_image = Image.open(reference_image_path).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive a target image, a reference image showing the current visual style, and a prompt.\n"
                "Identify ONE specific visual gap between the reference and the target "
                "(e.g. subject shape, composition, where colours appear, pose), "
                "then make the minimal change to the prompt to address it.\n\n"
                "Rules:\n"
                " - Change only ONE word or short phrase.\n"
                " - The change must address a real, visible gap between reference and target.\n"
                " - Output ONLY the modified prompt, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Target image:"},
                {"type": "image", "image": target_image},
                {"type": "text", "text": "Reference image (current style):"},
                {"type": "image", "image": ref_image},
                {
                    "type": "text",
                    "text": (
                        f"Prompt:\n{prompt.strip()}\n\n"
                        "Change one word or short phrase to close the most important visual gap. "
                        "Output ONLY the modified prompt."
                    ),
                },
            ],
        },
    ]
    return vlm_chat(llm, processor, messages, temperature=temperature)


def evolve(llm, processor, population, clip_model, clip_processor, target_image_path,
           n_candidates=5, base_mutation_rate=0.3, temperature=0.9, threshold=DIVERSITY_THRESHOLD):
    new_prompts = []
    for i in range(n_candidates):
        parent_a = rank_select(population)
        parent_b = rank_select(population)
        retries = 0
        while parent_b["prompt"] == parent_a["prompt"] and len(population) > 1 and retries < 5:
            parent_b = rank_select(population)
            retries += 1

        child = crossover(llm, processor, parent_a, parent_b, target_image_path, temperature)
        flag = 0
        if random.random() < base_mutation_rate:
            flag = 1
            child = mutate(llm, processor, child, parent_a["image_path"], target_image_path, temperature)

        pool = [c["prompt"] for c in population] + new_prompts
        if cosine_sim_to_pool(child, pool, clip_model, clip_processor) >= threshold:
            print(f"MUT-{flag}-[{i+1:02d}/{n_candidates}] skipped (too similar, threshold={threshold:.2f})")
            continue

        new_prompts.append(child)
        print(f"MUT-{flag}-[{i+1:02d}/{n_candidates}] {child}")

    print(f" {len(new_prompts)} evolved candidates")
    return new_prompts
