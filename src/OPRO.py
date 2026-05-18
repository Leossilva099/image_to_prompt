import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(model_id = "Qwen/Qwen3-4B-Instruct-2507"):
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer= AutoTokenizer.from_pretrained(model_id)
    llm.eval()
    return llm, tokenizer

def format_opro_history(candidates):
    sorted_candidates = sorted(candidates, key=lambda x: x['fitness'], reverse=False)
    
    formatted_text = "PREVIOUS ATTEMPTS HISTORY\n\n"
    for idx, c in enumerate(sorted_candidates, 1):
        prompt = c['prompt'].strip()
            
        formatted_text += f"Candidate {idx}\n"
        formatted_text += f"Prompt: {prompt}\n"
        formatted_text += f"Score: {c['fitness']:.3f}\n"
        formatted_text += "-------------------\n"
    return formatted_text


def generate_candidates_opro(
    candidates,
    llm,
    tokenizer,
    temperature = 0.9,
):
    
    prompts_history=format_opro_history(candidates)
    messages = [
    {
        "role": "system",
        "content": (
            "You are an expert at optimizing image generation prompts for LCM diffusion models. "
            "You will receive a list of prompts sorted by score from worst to best (0 to 1). "
            "Your goal is to analyse the highest-scoring prompts and generate a NEW, improved prompt "
            "that builds on what worked best. "
            "Output ONLY the new prompt text, nothing else."
        ),
    },
    {
        "role": "user",
        "content": (
            "Below are previous prompts sorted by score from worst to best.\n"
            "The score ranges from 0 to 1. The last candidate has the highest score.\n\n"
            f"{prompts_history}\n\n"
            "Analyse the highest-scoring prompts carefully and generate a NEW prompt "
            "that is different from all the prompts above and scores higher than all of them. "
            "Output ONLY the prompt text, nothing else."
        ),
    },
    ]
    
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)

    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=50,
        temperature=temperature
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    return content


def generate_initial_candidates(
    llm,
    tokenizer,
    candidates,
    n_candidates = 5,
    temperature = 0.9,
):
    candidates_opro=[]
    for i in range(n_candidates):
        prompt = generate_candidates_opro(candidates,llm,tokenizer,temperature)
        candidates_opro.append(prompt)
        print(f"  [{i+1:02d}/{n_candidates}] {prompt}")

    print(f" {len(candidates_opro)} generated candidates")
    return candidates_opro






