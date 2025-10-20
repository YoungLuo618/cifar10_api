import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

_preprocess = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

def load_image_to_tensor(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    x = _preprocess(img).unsqueeze(0)
    return x

def logits_to_pred(logits: torch.Tensor):
    probs = F.softmax(logits, dim=1)
    conf, pred_idx = probs.max(dim=1)
    return pred_idx.item(), conf.item(), probs.squeeze(0).tolist()

