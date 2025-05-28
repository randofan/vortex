import torch


@torch.no_grad()
def rollout(vit_model, x, discard_ratio: float = 0.0):
    """Attention rollout for timm ViT models (needs model patched to return attn)."""
    vit_model.eval()
    _, attn_list = vit_model.get_attention_map(x, return_all=True)
    result = torch.eye(attn_list[0].size(-1), device=x.device)
    for attn in attn_list:
        attn = attn.mean(1)  # avg heads
        if discard_ratio:
            flat = attn.view(attn.size(0), -1)
            v, _ = flat.topk(int(flat.size(1) * discard_ratio), dim=1, largest=False)
            mask = (flat <= v[:, -1].unsqueeze(-1)).view_as(attn)
            attn = attn.masked_fill(mask, 0)
        attn += torch.eye(attn.size(-1), device=x.device)
        attn /= attn.sum(-1, keepdim=True)
        result = attn @ result
    mask = result[:, 0, 1:]  # drop CLS
    s = int(mask.size(-1) ** 0.5)
    mask = mask.view(-1, 1, s, s)
    return torch.nn.functional.interpolate(
        mask, scale_factor=16, mode="bilinear", align_corners=False
    )
