# Project 1: NumPy 实现的 OpenCLIP ViT-B-32 图像编码器

用纯 NumPy 复现 OpenCLIP ViT-B-32 的**图像编码器前向计算**，输出与给定 FP32 参考
数值一致。该编码器即为期末项目 `szu_python_openclip` 的图像塔。

## 运行环境

- Python 3.9+
- NumPy >= 1.24, < 3
- Pillow >= 9.2, < 13
- 无需 GPU / PyTorch / OpenCLIP / 网络

```bash
python -m pip install -r requirements.txt
```

## 复现结果

```bash
# 分阶段数值校验（全部 PASS）
python check.py --stage all

# 对附件自带 10 张图片生成输出 token
python run.py --output output_tokens.npz

# 对任意本地图片生成输出 token
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 数值校验结果（`check.py --stage all`）

| 阶段 | 形状 | 最大绝对误差 | 平均绝对误差 | 容差 (atol) |
| --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0.0 | 0.0 | 2e-6 |
| patches | (2,49,3072) | 0.0 | 0.0 | 2e-4 |
| tokens | (2,50,768) | 0.0 | 0.0 | 2e-4 |
| block0 | (2,50,768) | 7.540e-06 | 2.274e-07 | 2e-4 |
| last_tokens | (2,50,768) | 4.387e-05 | 2.406e-06 | 2e-4 |
| output | (10,512) | 6.780e-07 | 8.215e-08 | 2e-5 |
| batch | (3,512) | 4.619e-07 | 8.133e-08 | 2e-5 |

（所有阶段 rtol = 2e-4）

## 实现要点

- **patchify**：`[B,3,224,224] -> [B,3,7,32,7,32] -> transpose(0,2,4,1,3,5) -> [B,49,3072]`，
  保持“先左后右、先上后下”的 patch 顺序与 C,H,W 展平顺序。
- **多头注意力**：`[B,N,768] -> [B,12,N,64]`，真实 k 为 4D，仅交换最后两轴。
- **线性层**：注意力/MLP 权重为 `[out,in]` 布局，矩阵乘前转置；`visual.proj` 已是
  `[in,out]=[768,512]`，直接使用。
- **两个残差**：注意力更新与 MLP 更新各自与输入残差相加。
- **输出头**：取 CLS token -> post-LayerNorm -> 投影 -> L2 归一化。

## 提交内容

- `image_encoder.py`：完成 TODO-01 ~ TODO-06 的实现（本项目核心，唯一需要编辑的文件）。
- `helpers.py`：提供的数值辅助（LayerNorm / 精确 GELU / softmax / L2 缩放）。
- `check.py`、`run.py`、`runtime.py`：提供的加载、命令行与参考校验。
- 报告见 Overleaf 模板 `report/main.tex`（配套插图 `report/figs/image_encoder_pipeline.pdf`）。
- `weights/`、`assets/` 等大文件与参考数据按作业要求未提交，可从课程附件获取。

## 注意事项

macOS 上 NumPy 使用 Accelerate/vecLib 后端做 float32 矩阵乘法时，可能打印
`divide by zero / invalid value encountered in matmul` 一类**误报警告**，结果仍有限且正确，
可忽略。
