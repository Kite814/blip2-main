# Mini-BLIP2 图像描述生成复现实验报告

## 1. 论文信息

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597

## 2. 任务说明

本实验复现的任务是图像描述生成 Image Captioning。

输入：图片  
输出：英文 caption

## 3. 数据集

- 数据集名称：Flickr8k
- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k
- 实际使用数据量：前 200 张图片

## 4. 模型结构

```text
Image → Frozen Vision Encoder → Mini Q-Former → Projection Layer → Frozen Language Decoder → Caption
```

### 4.1 Vision Encoder

`openai/clip-vit-base-patch32`

### 4.2 Mini Q-Former

说明自己实现的 Mini Q-Former：

- query token 数量：32
- hidden size：768
- Transformer 层数：2
- 是否使用 cross-attention：是

### 4.3 Language Decoder

`facebook/opt-125m`

## 5. 训练设置

请填写：

- 训练数据量：1000样本（200张图 × 5 captions）
- epoch：15
- batch size：16
- learning rate：1e-4
- optimizer：AdamW
- loss function：Cross Entropy Loss
- 冻结的模块：Vision Encoder + Language Decoder
- 训练的模块：Mini Q-Former + Projection Layer

## 6. 训练过程

粘贴训练日志或 loss 变化截图。

| Epoch | Train Loss |
| ----- | ---------- |
| 1     | 3.3416     |
| 2     | 2.8215     |
| 3     | 2.6674     |
| 4     | 2.5357     |
| 5     | 2.4105     |
| 6     | 2.2718     |
| 7     | 2.1409     |
| 8     | 2.0349     |
| 9     | 1.9004     |
| 10    | 1.8158     |
| 11    | 1.7108     |
| 12    | 1.6469     |
| 13    | 1.5818     |
| 14    | 1.5285     |
| 15    | 1.4765     |

## 7. 生成结果展示

| #    | 图片                  | 真实 Caption                                                 | 模型生成 Caption                                             |
| ---- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | 1000268201_693b08cb0e | A child in a pink dress is climbing up a set of stairs in an entry way . | A little girl in a pink dress climbing into a wooden wooden wooden cabin . |
| 2    | 1001773457_577c3a7d70 | A black dog and a spotted dog are fighting                   | Two dogs are fighting over a black dog .                     |
| 3    | 1002674143_1b742ab4b8 | A little girl covered in paint sits in front of a painted rainbow with her hands in a bowl . | A little girl sits in front of a rainbow in front of a painting . |
| 4    | 1003163366_44323f5815 | A man lays on a bench while his dog sits by him .            | A man lays on a bench with a dog .                           |
| 5    | 1007129816_e794419615 | A man in an orange hat starring at something .               | A man with glasses is wearing a hat and glasses .            |



## 8. 总结

请简要说明：

- 是否成功跑通训练；成功

- 生成效果如何；基本正确，部分出现幻觉

- 遇到了什么问题；

  1. OPT默认dtype float16，和Q-Former输出的 float32 不匹配，导致layer_norm报错
  2. 爬 HuggingFace 网页无法访问（网络限制）
  3. 文件路径歧义

- 如果继续改进，可以怎么做。

  更多训练数据/epoch、使用beam search替代 greedy decoding、增大batch size。

## 9. AI 对话过程记录

请填写本次复现过程中与 AI 工具的对话记录（对应 requirements.md 第 9.1 节）。

- 录制工具：entire.io
- 对话链接：[Sessions · Kite814/blip2-main · Entire](https://entire.io/gh/Kite814/blip2-main/sessions)
- 使用的 AI 模型： Claude code+deepseekv4 pro
- 累计对话时长 / 会话数：约 6-8 小时，2-3 次会话

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议（2—4 句话即可）：

```text
AI帮助搭建了完整的模型框架（从数据加载、tokenization、Q-Former架构到训练和推理），讲解了 BLIP2 各组件原理及参数选择理由。其中Q-Former的cross-attention设计、OPT embedding拼接逻辑、dtype不匹配的排查是AI解决的关键技术卡点。vision_encoder.py和qformer.py从模板填空完成，model.py和 generate.py的索引bug是自己调试修复的。训练参数和生成结果的分析判断由自己完成。
```

## 10. Git 提交记录

请填写本次复现的代码仓库与提交历史（对应 requirements.md 第 9.2 节）。

- 仓库地址：[Kite814/blip2-main](https://github.com/Kite814/blip2-main)
- 总 commit 数：17

粘贴 `git log --oneline` 输出（或截图）：

```text
（在这里粘贴 git log --oneline）
fix:删除data,添加,claude和.entire
fix: add .claude/ and .entire/ back to gitignore
997dbe85 chore: trigger entire session checkpoint
0bbc9db7 补充loss曲线
fb27f612 fix: gitignore training outputs and claude config to avoid large files
1a4bd02a docs: 补充实验报告与 loss 曲线
17ded923 generate
dcd76957 训练
e84a4d43 fix: 删除多余data
b6cce292 模型整合
f43691ac feat: 接入 frozen OPT-125m 作为语言解码器
86355b0a feat: 添加 projection layer 对齐到 OPT 词向量空间
116a6c87 feat: 实现 Mini Q-Former 模块（含 learnable queries）
dc613fa7 feat: 接入 CLIP ViT-B/32 作为 frozen vision encoder
fa49db7e Preprocessing
d66bb487 GetData
021f0521 first commit
```
