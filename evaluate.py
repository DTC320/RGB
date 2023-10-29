from dataprocess import MyDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from HSCNN import HSCNN
from Resnet import ResidualBlock, ResNet
from dataprocess import MyDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from HSCNND import HSCNN_D


model_path = '/root/code/3rdCODE/HSCNNtWith_best.pth'
#model_path = '/root/code/3rdCODE/ResnetWith_best.pth'
model = HSCNN()
#model = HSCNN_D()
#model = ResNet(ResidualBlock, [2], num_channels=31)
model.load_state_dict(torch.load(model_path))



model.eval()

test_dataset = MyDataset(
    rgb_folder='/root/autodl-tmp/ZERO/RGB_test', 
    mat_folder='/root/autodl-tmp/ZERO/HS_test',
    patch_size=128
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print("the length of test:", len(test_dataset))

reconstructed = []
original_hs = []
original_rgb = []
with torch.no_grad():
    for rgb_images, hs_images in test_loader:
        outputs = model(rgb_images)
        reconstructed.append(outputs.cpu().numpy())
        original_hs.append(hs_images.cpu().numpy())
        original_rgb.append(rgb_images.cpu().numpy())
        


reconstructed = np.concatenate(reconstructed)
original_hs = np.concatenate(original_hs)
original_rgb = np.concatenate(original_rgb)

# 反归一化
stats_hs = torch.load('/root/autodl-tmp/ZERO/dataset_stats_hs.pt')
mean_hs = stats_hs['mean'].numpy()
std_hs = stats_hs['std'].numpy()

stats_rgb = torch.load('/root/autodl-tmp/ZERO/dataset_stats_rgb.pt')
mean_rgb = stats_rgb['mean'].numpy()
std_rgb = stats_rgb['std'].numpy()

std_hs = std_hs[:, np.newaxis, np.newaxis]
mean_hs = mean_hs[:, np.newaxis, np.newaxis]
std_rgb = std_rgb[:, np.newaxis, np.newaxis]
mean_rgb = mean_rgb[:, np.newaxis, np.newaxis]


reconstructed_denorm = reconstructed * std_hs + mean_hs
original_hs_denorm = original_hs * std_hs + mean_hs
original_rgb_denorm = original_rgb * std_rgb + mean_rgb



#  评估
def RMSE(true, pred):
    return np.sqrt(((true - pred) ** 2).mean())

def SAM(true, pred):
    # 计算SAM
    dot_product = np.sum(true * pred, axis=1)
    norm_true = np.linalg.norm(true, axis=1)
    norm_pred = np.linalg.norm(pred, axis=1)
    cosine_similarity = dot_product / (norm_true * norm_pred)
    # 防止数值问题，将cosine_similarity限制在[-1, 1]范围内
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    return np.arccos(cosine_similarity)  # 返回角度

rmse_values = [RMSE(t, p) for t, p in zip(original_hs_denorm, reconstructed_denorm)]
sam_values = [SAM(t, p) for t, p in zip(original_hs_denorm, reconstructed_denorm)]

avg_rmse = np.mean(rmse_values)
avg_sam = np.mean(sam_values)

print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average SAM: {avg_sam:.4f}")

# 保存评价指标到文本文件
#
with open("/root/code/3rdCODE/comparison_results/HSCNN/with/evaluation_metrics.txt", "w") as file:
    file.write(f"Average RMSE: {avg_rmse:.4f}\n")
    file.write(f"Average SAM: {avg_sam:.4f}\n")

print("Evaluation metrics saved to evaluation_metrics.txt")


# 可视化展示并保存对比结果
output_dir = '/root/code/3rdCODE/comparison_results/HSCNN/with'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

band = 15  # 选择一个波段，例如15

for idx in range(len(reconstructed)):
    plt.figure(figsize=(18, 6))
    
    # 原始RGB图像
    #plt.subplot(1, 3, 1)
    #plt.imshow(np.transpose(original_rgb_denorm[idx], (1, 2, 0)))
    #plt.title('Original RGB')
    
    # 原始HS图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_hs_denorm[idx, band], cmap='gray')
    plt.title('Original  HS')
    
    # 重建的HS图像
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_denorm[idx, band], cmap='gray')
    plt.title('Reconstructed HS')
    
    plt.savefig(os.path.join(output_dir, f'comparison_{idx}.png'))
    plt.close()

print(f"Comparison results saved in {output_dir}")