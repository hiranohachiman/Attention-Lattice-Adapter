import torch


def predict_one_shot(model, img, device="cuda"):
    """
    input: img: torch.tensor, shape: (b, 3, 384, 384)
    output: label: int, attn: torch.tensor, shape: (b, 1, 384, 384)
    """
    model.eval()
    with torch.no_grad():
        img = torch.tensor(img).unsqueeze(0)
        img = img.to(device)
        logit, _, attn = model(img)
        label = torch.sigmoid(logit)
        label = torch.argmax(label, dim=1)
        label = int(label.cpu().numpy())
        return label, attn
