import torch
from model import MiniBLIP2
from GetData import create_dataloader, get_clip_transform, get_tokenizer


def generate(model, image, tokenizer, max_len=32, device="cpu"):
    """对单张图片生成 caption（greedy decoding）。

    image: [1, 3, 224, 224]
    """
    model.eval()

    with torch.no_grad():
        # ① 图像 → projected_queries
        image_embeds = model.vision_encoder(image)       # [1, 50, 768]
        query_output = model.qformer(image_embeds)        # [1, 32, 768]
        projected = model.projection(query_output)        # [1, 32, 768]

        embed = model.language_decoder.opt_model.get_input_embeddings()
        opt = model.language_decoder.opt_model

        # ② 逐 token 生成
        cur_ids = [tokenizer.bos_token_id]
        for _ in range(max_len):
            # 当前已生成 token 的 embedding
            cur_tensor = torch.tensor([cur_ids], device=device)
            cur_embeds = embed(cur_tensor)                # [1, cur_len, 768]

            # visual prefix + 当前序列
            inputs_embeds = torch.cat([projected, cur_embeds], dim=1)
            attn_mask = torch.ones(1, inputs_embeds.shape[1], device=device)

            logits = opt(inputs_embeds=inputs_embeds, attention_mask=attn_mask).logits
            next_token = logits[:, -1, :].argmax(dim=-1).item()

            if next_token == tokenizer.eos_token_id:
                break
            cur_ids.append(next_token)

    # ③ 解码（跳过 BOS）
    caption = tokenizer.decode(cur_ids[1:], skip_special_tokens=True).strip()
    return caption


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    _, dataset, _ = create_dataloader(
        batch_size=16, transform=get_clip_transform(), shuffle=False
    )
    tokenizer = get_tokenizer()

    # 加载 checkpoint
    model = MiniBLIP2().to(device)
    ckpt = torch.load("checkpoint.pth", map_location=device)
    model.qformer.load_state_dict(ckpt["qformer"])
    model.projection.load_state_dict(ckpt["projection"])
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})\n")

    # 展示 5 张不同图片
    shown = set()
    for idx in range(len(dataset)):
        image, caption = dataset[idx]
        img_name = dataset.data[idx][0]

        if img_name in shown:
            continue
        shown.add(img_name)

        img_tensor = image.unsqueeze(0).to(device)
        generated = generate(model, img_tensor, tokenizer, device=device)

        print(f"[{img_name}]")
        print(f"  Real: {caption}")
        print(f"  Gen:  {generated}\n")

        if len(shown) >= 5:
            break
