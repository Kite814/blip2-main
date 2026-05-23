import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class FrozenVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载 CLIP 的视觉部分
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        # 冻结：不训练这个模型
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, images):
        # images: [B, 3, 224, 224]
        with torch.no_grad():
            output = self.model(images)
        # 返回所有 token（CLS + 49 patch），Q-Former 需要对它们做 cross-attention
        return output.last_hidden_state


if __name__ == "__main__":
    # 测试脚本思路
    encoder = FrozenVisionEncoder()
    encoder = encoder.to("cuda")

    # 用 GetData.py 的 transform 预处理一张图
    from GetData import get_clip_transform, create_dataloader
    dataloader, _, _ = create_dataloader(batch_size=4, transform=get_clip_transform())
    images, _, _, _ = next(iter(dataloader))

    images = images.to("cuda")
    with torch.no_grad():
        output = encoder(images)

    print(output.shape)  # 应该输出 torch.Size([4, 50, 768])

    # 验证冻结
    for name, param in encoder.named_parameters():
        assert not param.requires_grad, f"{name} should be frozen!"
    print("All parameters frozen: OK")