"""
Flickr8k Dataset Loader - Loads first 200 images and their captions.
Data is cached in the data/ folder.
Caption tokenization uses OPT tokenizer (facebook/opt-125m).
"""
import os
import shutil
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer


KAGGLEHUB_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "kagglehub",
    "datasets", "adityajn105", "flickr8k", "versions", "1"
)
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NUM_IMAGES = 200


_tokenizer = None


def get_tokenizer(model_name="facebook/opt-125m"):
    """Lazy-load the OPT tokenizer (singleton)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        # OPT uses </s> (id=2) as both BOS and EOS; <pad> is id=1
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def tokenize_captions(captions, tokenizer, max_length=32):
    """Tokenize a list of caption strings into input_ids and attention_mask.

    Each caption is tokenized as:
        [BOS] token1 token2 ... tokenN [EOS] [PAD] [PAD] ...

    The EOS token teaches the model when to stop generating.
    Labels for LM loss are identical to input_ids (shift happens inside the model).

    Returns dict with keys: input_ids, attention_mask (both tensors of shape [B, max_length]).
    """
    # Step 1: tokenize without adding any special tokens
    tokens = tokenizer(captions, add_special_tokens=False)

    bos_id = tokenizer.bos_token_id  # 2
    eos_id = tokenizer.eos_token_id  # 2 (same as BOS in OPT)
    pad_id = tokenizer.pad_token_id  # 1

    input_ids_list = []
    attention_mask_list = []

    for ids in tokens["input_ids"]:
        # Reserve 1 slot for BOS prefix; if the caption alone is too long, truncate
        ids = ids[:max_length - 2]
        # Build: [BOS] + tokens + [EOS] + [PAD]...
        ids = [bos_id] + ids + [eos_id]
        mask = [1] * len(ids)

        # Pad to max_length
        padding_len = max_length - len(ids)
        ids += [pad_id] * padding_len
        mask += [0] * padding_len

        input_ids_list.append(ids)
        attention_mask_list.append(mask)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
    }


def prepare_data():
    """Copy first NUM_IMAGES images from kagglehub cache to local data/ folder."""
    local_images_dir = os.path.join(LOCAL_DATA_DIR, "Images")
    local_captions_path = os.path.join(LOCAL_DATA_DIR, "captions.txt")

    if os.path.exists(local_images_dir) and os.path.exists(local_captions_path):
        return local_images_dir, local_captions_path

    cache_images_dir = os.path.join(KAGGLEHUB_CACHE, "Images")
    cache_captions_path = os.path.join(KAGGLEHUB_CACHE, "captions.txt")

    if not os.path.exists(cache_captions_path):
        raise FileNotFoundError(
            f"Kagglehub cache not found at {KAGGLEHUB_CACHE}. "
            "Run: python -c \"import kagglehub; kagglehub.dataset_download('adityajn105/flickr8k')\""
        )

    with open(cache_captions_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    # Collect unique image names in order of first appearance
    seen = set()
    selected_images = []
    for line in lines[1:]:  # skip header
        img_name = line.split(',', 1)[0].strip()
        if img_name not in seen:
            seen.add(img_name)
            selected_images.append(img_name)
            if len(selected_images) == NUM_IMAGES:
                break

    os.makedirs(local_images_dir, exist_ok=True)

    # Copy selected images
    for img_name in selected_images:
        src = os.path.join(cache_images_dir, img_name)
        dst = os.path.join(local_images_dir, img_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Write filtered captions file (only captions for selected images)
    selected_set = set(selected_images)
    with open(local_captions_path, 'w', encoding='utf-8') as f_out:
        f_out.write(lines[0] + '\n')  # header
        for line in lines[1:]:
            img_name = line.split(',', 1)[0].strip()
            if img_name in selected_set:
                f_out.write(line + '\n')

    return local_images_dir, local_captions_path


def parse_captions(captions_path, num_images=NUM_IMAGES):
    """Parse captions file, return list of (image_name, caption) pairs for first num_images."""
    with open(captions_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    captions_dict = defaultdict(list)
    for line in lines[1:]:
        parts = line.split(',', 1)
        if len(parts) == 2:
            img_name = parts[0].strip()
            caption = parts[1].strip().strip('"')
            captions_dict[img_name].append(caption)

    all_images = sorted(captions_dict.keys())[:num_images]
    data = []
    for img_name in all_images:
        for caption in captions_dict[img_name]:
            data.append((img_name, caption))
    return data


class Flickr8kDataset(Dataset):
    """PyTorch Dataset for Flickr8k image captioning (first 200 images, 5 captions each)."""

    def __init__(self, data, images_dir, transform=None):
        self.data = data
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption


def create_dataloader(batch_size=16, num_workers=0, transform=None, shuffle=True,
                      tokenizer=None, max_caption_length=32):
    """Create a DataLoader for the Flickr8k first-200 dataset.

    Returns (dataloader, dataset, tokenizer).
    The tokenizer is loaded lazily if not provided.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    images_dir, captions_path = prepare_data()
    data = parse_captions(captions_path)
    dataset = Flickr8kDataset(data, images_dir, transform=transform)

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        tokens = tokenize_captions(captions, tokenizer, max_length=max_caption_length)
        return images, captions, tokens["input_ids"], tokens["attention_mask"]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dataloader, dataset, tokenizer


def get_clip_transform():
    """Return torchvision transforms matching CLIP ViT-B/32 preprocessing.

    Pipeline: Resize(shortest=224) → CenterCrop(224) → ToTensor → Normalize(CLIP stats)
    This mirrors what CLIPImageProcessor does internally.
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(224),              # resize shortest edge to 224
        transforms.CenterCrop(224),          # crop center 224x224
        transforms.ToTensor(),               # [0,255] → [0,1]
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


if __name__ == "__main__":
    transform = get_clip_transform()

    dataloader, dataset, tokenizer = create_dataloader(
        batch_size=8, transform=transform, shuffle=False
    )

    print(f"Unique images: {len(set(img for img, _ in dataset.data))}")
    print(f"Total samples (images x captions): {len(dataset)}")
    print(f"Batches: {len(dataloader)}")

    images, captions, input_ids, attention_mask = next(iter(dataloader))
    print(f"Image batch shape: {images.shape}")           # [B, 3, 224, 224]
    print(f"Captions in batch: {len(captions)}")
    print(f"input_ids shape: {input_ids.shape}")           # [B, max_length]
    print(f"attention_mask shape: {attention_mask.shape}") # [B, max_length]
    print(f"Sample caption 0: {captions[0]}")
    print(f"Sample token ids 0: {input_ids[0]}")
    print(f"Sample decoded: {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
