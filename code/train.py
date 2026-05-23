import torch
import torch.optim as optim
from model import MiniBLIP2
from GetData import create_dataloader, get_clip_transform


def train(num_epochs=10, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ① 数据
    train_loader, _, _ = create_dataloader(
        batch_size=batch_size,
        transform=get_clip_transform(),
        shuffle=True,
    )

    # ② 模型
    model = MiniBLIP2().to(device)

    # ③ 优化器：只优化 requires_grad=True 的参数
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # ④ 训练循环
    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, _, caption_ids, caption_mask in train_loader:
            images = images.to(device)
            caption_ids = caption_ids.to(device)
            caption_mask = caption_mask.to(device)

            loss, _ = model(images, caption_ids, caption_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1:2d}/{num_epochs}  |  Avg Loss: {avg_loss:.4f}")

    # ⑤ 保存 checkpoint（只存可训练权重）
    torch.save({
        "qformer": model.qformer.state_dict(),
        "projection": model.projection.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": num_epochs,
        "loss": avg_loss,
    }, "checkpoint.pth")
    print(f"\nCheckpoint saved to checkpoint.pth")

    # ⑥ loss 曲线
    try:
        import matplotlib.pyplot as plt
        plt.plot(range(1, num_epochs + 1), history, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig("loss_curve.png")
        print("Loss curve saved to loss_curve.png")
    except ImportError:
        pass

    return history


if __name__ == "__main__":
    train(num_epochs=15, batch_size=16, lr=1e-4)
