import torch
import torch.nn as nn


class QFormerBlock(nn.Module):
    """单层 Q-Former: Self-Attention → Cross-Attention → FFN，各带残差 + LayerNorm"""

    def __init__(self, hidden_size=768, num_heads=8, ffn_hidden_size=3072):
        super().__init__()
        # Self-Attention：query 之间互相交流，避免 32 个 query 都关注同一个东西
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        # Cross-Attention：query 去图像特征中"检索"信息
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        # FFN: 标准 Transformer 前馈网络，hidden_size → 4倍 → hidden_size
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Linear(ffn_hidden_size, hidden_size),
        )
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, query_tokens, image_embeds):
        # query_tokens: [B, num_queries, hidden_size]
        # image_embeds: [B, num_patches, hidden_size]

        # --- Self-Attention ---
        # Q=query_tokens, K=query_tokens, V=query_tokens
        residual = query_tokens
        query_tokens, _ = self.self_attn(query_tokens, query_tokens, query_tokens)
        query_tokens = self.norm1(residual + query_tokens)

        # --- Cross-Attention ---
        # Q=query_tokens, K=image_embeds, V=image_embeds
        residual = query_tokens
        query_tokens, _ = self.cross_attn(query_tokens, image_embeds, image_embeds)
        query_tokens = self.norm2(residual + query_tokens)

        # --- FFN ---
        residual = query_tokens
        query_tokens = self.ffn(query_tokens)
        query_tokens = self.norm3(residual + query_tokens)

        return query_tokens


class MiniQFormer(nn.Module):
    """Mini Q-Former: 多层 QFormerBlock + 可学习 query tokens"""

    def __init__(self, hidden_size=768, num_queries=32, num_layers=2, num_heads=8):
        super().__init__()
        # 可学习的 query tokens，训练开始前是随机向量，训练中学会从图像提取信息
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))

        # 堆叠多层 QFormerBlock
        self.layers = nn.ModuleList([
            QFormerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, image_embeds):
        # image_embeds: [B, num_patches, hidden_size]  -- from vision encoder

        # 把 query_tokens 扩展到当前 batch 大小（所有样本共享同一组 query）
        B = image_embeds.shape[0]
        query = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, hidden_size]

        # 逐层传递：每层的 query 输出作为下一层的 query 输入
        for layer in self.layers:
            query = layer(query, image_embeds)

        return query  # [B, num_queries, hidden_size]


if __name__ == "__main__":
    # 测试：模拟 vision encoder 输出
    B, N_patch, N_query, hidden = 2, 50, 32, 768
    fake_image_embeds = torch.randn(B, N_patch, hidden)

    qformer = MiniQFormer()
    out = qformer(fake_image_embeds)

    print(f"Input:  {fake_image_embeds.shape}")   # [2, 50, 768]
    print(f"Output: {out.shape}")                  # [2, 32, 768]

    # 统计参数量
    total = sum(p.numel() for p in qformer.parameters())
    trainable = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    print(f"Params: {total:,} total, {trainable:,} trainable")
