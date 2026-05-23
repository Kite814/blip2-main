import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class FrozenLanguageDecoder(nn.Module):
    """Frozen OPT-125m：接收 projected queries + caption tokens，输出 caption logits。

    关键点：不传 input_ids，直接传 inputs_embeds（visual prefix + caption embeddings 拼接）。
    """

    def __init__(self, model_name="facebook/opt-125m"):
        super().__init__()
        # 加载 OPT-125m，显式指定 float32 避免 dtype 不匹配
        self.opt_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 冻结 OPT 所有参数（Vision Encoder 一样的三连）
        for param in self.opt_model.parameters():
            param.requires_grad = False
        self.opt_model.eval()

        self.hidden_size = self.opt_model.config.hidden_size  # 768
        self.num_queries = None  # 由 forward 时 projected_queries 的 shape 决定

    def forward(self, projected_queries, caption_input_ids, caption_attention_mask):
        # projected_queries:      [B, num_queries, hidden_size]  -- from ProjectionLayer
        # caption_input_ids:      [B, caption_len]                -- tokenized captions
        # caption_attention_mask: [B, caption_len]                -- 1=有效token, 0=PAD

        B, num_queries, _ = projected_queries.shape
        device = projected_queries.device

        # --- Step 1: 获取 caption 的 embeddings ---
        # 用 OPT 自己的 embedding 层查表（保证语言端不变）
        embed = self.opt_model.get_input_embeddings()
        caption_embeds = embed(caption_input_ids.to(device))
        # shape: [B, caption_len, hidden_size]

        # --- Step 2: 拼接 visual prefix + caption embeddings ---
        inputs_embeds = torch.cat([projected_queries, caption_embeds], dim=1)
        # shape: [B, num_queries + caption_len, hidden_size]

        # --- Step 3: 构造 attention mask ---
        # visual prefix 全部有效（1），后面接 caption 的 mask
        prefix_mask = torch.ones(B, num_queries, device=device, dtype=caption_attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, caption_attention_mask.to(device)], dim=1)
        # shape: [B, num_queries + caption_len]

        # --- Step 4: 构造 labels ---
        # 规则：visual prefix 位置填 -100（不算 loss），caption PAD 也填 -100
        prefix_labels = torch.full((B, num_queries), -100, device=device, dtype=torch.long)
        caption_labels = caption_input_ids.clone().to(device)
        # PAD 位置用 -100 标记（CrossEntropyLoss 的 ignore_index 默认就是 -100）
        caption_labels[caption_attention_mask.to(device) == 0] = -100
        labels = torch.cat([prefix_labels, caption_labels], dim=1)
        # shape: [B, num_queries + caption_len]

        # --- Step 5: OPT forward ---
        # OPT 内部会自动把 labels 右移做 next-token prediction
        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from GetData import get_tokenizer, tokenize_captions

    # 模拟输入
    B, N_query, hidden, caption_len = 2, 32, 768, 32
    fake_projected = torch.randn(B, N_query, hidden)

    # 用真实 tokenizer 构造一批假 caption
    tokenizer = get_tokenizer()
    sample_captions = ["a dog running on grass", "a child playing in the park"]
    tokens = tokenize_captions(sample_captions, tokenizer, max_length=caption_len)
    caption_ids = tokens["input_ids"]
    caption_mask = tokens["attention_mask"]

    decoder = FrozenLanguageDecoder()
    loss, logits = decoder(fake_projected, caption_ids, caption_mask)

    print(f"Projected queries:  {fake_projected.shape}")  # [2, 32, 768]
    print(f"Caption ids:        {caption_ids.shape}")      # [2, 32]
    print(f"Logits:             {logits.shape}")            # [2, 64, 50272]
    print(f"Loss:               {loss.item():.4f}")

    # 验证冻结
    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Trainable params:   {trainable}")               # 应为 0
