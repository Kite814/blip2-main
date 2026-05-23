import torch
import torch.nn as nn

from vision_encoder import FrozenVisionEncoder
from qformer import MiniQFormer
from projection import ProjectionLayer
from language_decoder import FrozenLanguageDecoder


class MiniBLIP2(nn.Module):
    """Mini-BLIP2：串联 Vision Encoder → Q-Former → Projection → Language Decoder"""

    def __init__(self):
        super().__init__()
        self.vision_encoder = FrozenVisionEncoder()
        self.qformer = MiniQFormer()
        self.projection = ProjectionLayer()
        self.language_decoder = FrozenLanguageDecoder()

    def forward(self, images, caption_input_ids, caption_attention_mask):
        # images:                 [B, 3, 224, 224]
        # caption_input_ids:      [B, max_len]
        # caption_attention_mask: [B, max_len]

        # ① 图像编码
        image_embeds = self.vision_encoder(images)        # [B, 50, 768]

        # ② Q-Former 提取语义
        query_output = self.qformer(image_embeds)          # [B, 32, 768]

        # ③ 投影到语言空间
        projected = self.projection(query_output)          # [B, 32, 768]

        # ④ 语言解码器计算 loss + logits
        loss, logits = self.language_decoder(
            projected, caption_input_ids, caption_attention_mask
        )
        return loss, logits


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from GetData import create_dataloader, get_clip_transform

    # 取一个 batch
    dl, _, _ = create_dataloader(batch_size=4, transform=get_clip_transform(), shuffle=True)
    images, _, caption_ids, caption_mask = next(iter(dl))

    print(f"Images:            {images.shape}")       # [4, 3, 224, 224]
    print(f"Caption ids:       {caption_ids.shape}")   # [4, 32]
    print(f"Caption mask:      {caption_mask.shape}")  # [4, 32]

    model = MiniBLIP2()
    loss, logits = model(images, caption_ids, caption_mask)

    print(f"Logits:            {logits.shape}")         # [4, 64, 50272]
    print(f"Loss:              {loss.item():.4f}")

    # 统计可训练 vs 冻结参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable:,}  |  Frozen: {frozen:,}")
