import numpy as np
import torch
import lpips
from transformers import CLIPModel, CLIPProcessor


def load_clip(device):
    model=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    return model,processor


def clip_image_similarity(candidate, target, model, processor, device):
    inputs = processor(images=[target.convert("RGB"), candidate.convert("RGB")], return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    features_n = torch.nn.functional.normalize(features, dim=-1)
    return float((features_n[0] * features_n[1]).sum().item())



def load_lpips(device):
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def lpips_distance(candidate, target, loss_fn, device):
    def to_tensor(img):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        result = loss_fn(to_tensor(target), to_tensor(candidate))
    return float(result.item())


def mse_distance(candidate, target):
    candidate_arr = np.asarray(candidate.convert("RGB"), dtype=np.float32) / 255.0
    target_arr    = np.asarray(target.convert("RGB"),    dtype=np.float32) / 255.0
    return float(np.mean((target_arr - candidate_arr) ** 2))


def rmse_distance(candidate, target):
    return float(np.sqrt(mse_distance(candidate, target)))


def compute_fitness(
    candidate,
    target,
    clip_model,
    clip_processor,
    lpips_fn,
    device,
    w_clip = 0.45,
    w_lpips = 0.45,
    w_rmse = 0.1,
):
    
    clip_score  = clip_image_similarity(candidate, target, clip_model, clip_processor, device)
    lpips_score = lpips_distance(candidate, target, lpips_fn, device)
    rmse_score   = rmse_distance(candidate, target)

    fitness = (
        w_clip  * clip_score
        + w_lpips * (1.0 / (lpips_score + 1.0))
        + w_rmse   * (1.0 / (rmse_score   + 1.0))
    )

    return {
        "fitness": fitness,
        "clip":    clip_score,
        "lpips":   lpips_score,
        "rmse":     rmse_score,
    }