import os
import torch


def save_adapter_and_head(model, path):
    os.makedirs(path, exist_ok=True)

    adapter_state = {
        k: v.cpu()
        for k, v in model.bert_model.encoder.state_dict().items()
        if "adapter" in k
    }
    torch.save(adapter_state, os.path.join(path, "adapter.pt"))

    torch.save(model.classifier_head.state_dict(), os.path.join(path, "classifier_head.pt"))

    print(f"Saved adapter and classifier head to {path}")


def load_adapter_and_head(model, path, device="cpu"):
    adapter_weights = torch.load(os.path.join(path, "adapter.pt"), map_location=device)
    model.bert_model.encoder.load_state_dict(adapter_weights, strict=False)

    classifier_weights = torch.load(os.path.join(path, "classifier_head.pt"), map_location=device)
    model.classifier_head.load_state_dict(classifier_weights)

    print(f"Loaded adapter and classifier head from {path}")