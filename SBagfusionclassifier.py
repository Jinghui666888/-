
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report

# 1. 跑通论文代码
# =========================
# 2. 定义跨模态注意力模块
# =========================
class CrossModalAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(dim_q, dim_out)
        self.key_proj = nn.Linear(dim_kv, dim_out)
        self.value_proj = nn.Linear(dim_kv, dim_out)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key_value):
        Q = self.query_proj(query).unsqueeze(1)     # [B, 1, d]
        K = self.key_proj(key_value).unsqueeze(1)   # [B, 1, d]
        V = self.value_proj(key_value).unsqueeze(1) # [B, 1, d]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attn_weights = self.softmax(attn_scores)
        attended = torch.bmm(attn_weights, V)
        return attended.squeeze(1)  # [B, d]

# =========================
# 3. 定义融合分类器
# =========================
class FusionClassifier(nn.Module):
    def __init__(self, input_dim, attention_module):
        super(FusionClassifier, self).__init__()
        self.attention = attention_module
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, sbag_feat, bert_feat, senti_feat, topic_feat):
        fused_senti_topic = self.attention(senti_feat, topic_feat)
        x = torch.cat([sbag_feat, bert_feat, fused_senti_topic], dim=1)
        return self.fc(x)

# 初始化注意力模块与模型
attention = CrossModalAttention(dim_q=16, dim_kv=10, dim_out=16)
model = FusionClassifier(input_dim=128 + 32 + 16, attention_module=attention)

# =========================
# 4. 训练过程
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0

    for sbag_x, bert_x, senti_x, topic_x, label_y in dataloader:
        logits = model(sbag_x, bert_x, senti_x, topic_x)
        loss = criterion(logits, label_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sbag_x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == label_y).sum().item()

    avg_loss = total_loss / len(dataset)
    acc = correct / len(dataset)
    train_losses.append(avg_loss)
    train_accuracies.append(acc)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

# =========================
# 5. 可视化训练曲线
# =========================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# =========================
# 6. 保存模型和预测结果
# =========================
torch.save(model.state_dict(), "fusion_classifier.pth")
print("✅ 模型已保存为 fusion_classifier.pth")

# 模拟所有输入（一次性前向）
model.eval()
with torch.no_grad():
    fused_preds = model(sbag_features, bert_features, sentiment_features, topic_features)
    all_pred = torch.argmax(fused_preds, dim=1)

    np.save("predictions.npy", all_pred.numpy())
    np.save("labels.npy", labels.numpy())
    print("✅ 预测结果和标签已保存为 predictions.npy 和 labels.npy")

# =========================
# 7. 加载模型并评估
# =========================
# 重新加载模型
loaded_model = FusionClassifier(input_dim=176, attention_module=attention)
loaded_model.load_state_dict(torch.load("fusion_classifier.pth"))
loaded_model.eval()
print("✅ 加载模型成功")

# 加载结果
predictions = np.load("predictions.npy")
labels_loaded = np.load("labels.npy")
accuracy = np.mean(predictions == labels_loaded)
print(f"📊 加载后预测准确率：{accuracy:.4f}")

# 混淆矩阵 & 分类报告
print("\n🎯 混淆矩阵：")
print(confusion_matrix(labels_loaded, predictions))

print("\n📋 分类报告：")
print(classification_report(labels_loaded, predictions, target_names=["非谣言", "谣言"]))
