import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """将 Q-Former 输出从视觉空间映射到 OPT-125m 的语言 embedding 空间。

    两者 hidden_size 相同（都是 768），但语义空间不同。
    这一层学习"坐标系转换"，让 projected queries 可以作为 OPT 的 prefix embeddings。
    """

    def __init__(self, hidden_size=768):
        super().__init__()
        # TODO: 最简单的做法：一个 Linear(768 → 768)
        # TODO: 稍复杂的做法：两层 MLP（Linear → GELU → Linear）
        # 两种都可以，效果差异不大

    def forward(self, query_output):
        # query_output: [B, num_queries, hidden_size]  -- from Q-Former
        # TODO: 应用 projection
        # 返回: [B, num_queries, hidden_size]
        pass


if __name__ == "__main__":
    # 模拟 Q-Former 输出
    B, N_query, hidden = 2, 32, 768
    fake_query_output = torch.randn(B, N_query, hidden)

    proj = ProjectionLayer()
    out = proj(fake_query_output)

    print(f"Input:  {fake_query_output.shape}")   # [2, 32, 768]
    print(f"Output: {out.shape}")                   # [2, 32, 768]
    print(f"Params: {sum(p.numel() for p in proj.parameters()):,} trainable")
