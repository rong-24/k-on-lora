# k-on-lora
前段时间二刷《轻音少女》，感觉很有意思，正好最近尝试实现 AI 绘画，遂建此项目。
<div align="center">
  <img src="https://github.com/user-attachments/assets/ad3b93c9-b65c-4acf-a13e-3620eaf607d5" width="30%" />
  <img src="https://github.com/user-attachments/assets/f48df1b7-f1ba-4ee9-ade9-df2b6458b7d1" width="31%" />
  <img src="https://github.com/user-attachments/assets/f0a5c5bc-e37a-4119-8467-a64cb23b8729" width="23%" />
</div>

下面以 K-ON 中**平泽唯**角色为例，整体步骤按照如下进行：

## 1.准备数据集
**目标**：打造一份**平泽唯**独有的数据集

**过程**：笔者从淘宝购入了一份平泽唯官方原图集，通过**裁切**以及**填白**的方式，将每张图片缩减到只有**平泽唯**一人，随后以相同的序号为每张图编写txt标签

**试错**：笔者有尝试使用**WD14/DeepDanbooru**等方式添加统一标签，但受限于学习成本，最后还是采用了**人工方式**

**结果**：生成了**188**组图片与标签

## 2.采用kohya_ss进行训练
**目标**：做一个**角色 LoRA**，让模型认识具体人物

**过程**：笔者在这里租用了一份AutoDL上的**3090显卡**，通过**kohya_ss**来进行自动训练
### 2.1 加载底模
笔者采用的是**AnyLoRA**底模作为大模型的“底座”，该特点是**风格**足够中性，对各种LoRA的兼容度非常高

笔者后续也会对**NAI Animefull/AbyssOrangeMix/Counterfeit**等底模分别展开尝试。

>**NAI Animefull（NovelAI模型） – （基于SD1.5的Anime模型）**
>这是NovelAI团队训练的动漫风模型，在动画人物方面表现出色，也是许多后续动漫模型的基础。
>用它作为底模训练LoRA具有广泛的兼容性：因为社区里流行的Anime模型大多或多或少融合了NAI模型的权重。
>换言之，在NAI上训练的LoRA几乎可以无缝应用于绝大部分动漫衍生模型。
>如果用户希望LoRA在不同模型上通用，这是首选底模。其文件名称常为animefull-final-pruned.ckpt等。

>**Anything V3 / AnyLoRA – （通用二次元模型）**
>Anything系列模型是由NAI模型融合改进而来的动漫模型。其中V3作为经典版本，以风格中性著称，能较忠实地还原创作者画风。
>AnyLoRA则是社区专门发布的一个LoRA友好模型，特点是风格足够中性，对各种LoRA的兼容度非常高。
>其初衷正是“确保未来的LoRA训练与新的模型兼容”并保持风格不过度偏颇，适合作为LoRA训练的基石模型。
>使用这些模型作为底模，可以更好地迁移LoRA到其他Anime模型而不引入奇怪风格偏差。

>**AbyssOrangeMix / Counterfeit – （动漫混合模型）**
>这些是国内外常用的高质量动漫模型，如AOM系列融合了多个Anime模型，画风细腻，兼顾写实与二次元；Counterfeit偏插画风，细节丰富。
>这些模型本身在角色细节上建模很好，也常被用作LoRA应用的基础模型。
>如果希望生成时自带一些这些模型的画面风格，可以在训练时直接以它们为底模。
>不过需要注意，不同底模的风格可能会稍微“染”到LoRA中。因此若以AOM等为底模训练LoRA，今后最好也在相近风格的模型上调用该LoRA，以得到最理想的效果。

将**底模**放置于**models**文件夹下，标注后的**训练集**放置于**datasets**文件夹下
```
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
path = hf_hub_download(
    repo_id="Lykon/AnyLoRA",
    filename="AnyLoRA_bakedVae_blessed_fp16.safetensors",
    local_dir="models",
    local_dir_use_symlinks=False
)
print("Downloaded file to:", path)
PY
```
### 2.2 读取图-文对
特别注意，笔者在这里对训练集的命名为**10_yuihirasawa**，具体格式为/mnt/.../datasets/10_yuihirasawa/001.jpg……

因为sd-scripts（DreamBooth模式）要**带重复次数的子文件夹名**，这里前面的10指的是这一整个文件夹的**重复次数**

对每张 jpg 都找同名的 txt 当作这张图的**标签/提示词（caption）**
### 2.3 前向+损失
用 caption 作为条件，让模型去**还原**这张图

预测与真实图像的**差距**就是损失（Loss）
### 2.4 只训练 LoRA 小模块
大模型不动；只更新插在 UNet/Text Encoder 上的 LoRA 低秩矩阵（很小的“补丁”）

这就是**微调**
### 2.5 保存 LoRA 权重
跑到设定的步数，就在 output_dir 里产出一个 **.safetensors（这就是角色 LoRA）**

生成时**底模 + LoRA**一起作用
### 2.6 训练参数设置
在确保数据集准备就绪后，就可以配置LoRA训练的**具体参数**。

这里提供一些针对**角色识别**和**动作/场景泛化**的参数建议，以便平衡模型对角色特征的记忆和对不同场景的泛化能力：
>**网络结构**
>选择LoRA模块并设置秩（rank）。一般使用Kohya的train_network.py脚本，指定--network_module=networks.lora即可使用LoRA
>网络维度network_dim建议设置为16或32，对于角色细节较丰富时可用32（会增加LoRA文件大小，但捕捉特征更充分）
>Alpha一般与dim相同或略低，例如16维配α=16，32维配α=16或32，确保LoRA权重不会过强

>**学习率**
>针对UNet部分的学习率可设在1e-4到2e-4左右。这通常是LoRA微调的默认量级
>Text Encoder（文本编码器）部分如要一起训练，可以用更低的学习率（例如1e-5）以免破坏预训练的词向量
>Kohya脚本允许分别设置两个优化器或者用联合优化器配合比例。初次尝试可以启用--train_text_encoder让文本编码器一起微调，以更好地学习“平泽唯”这个新词汇的语义映射

>**训练批次与迭代**
>3090显存较充裕，可适当提高batch_size，比如设置批大小为4或8（具体取决于显存占用），批次大可以稳定训练，但注意显存不足时应调小
>最大训练迭代可以通过--max_train_steps或--epoch控制
>假设有1000张图，1个epoch即1000步。经验来看，大概训练2~3个epoch（2000～3000步）即可取得良好效果，如过度训练可能导致过拟合（生成图像都穿同样制服、表情固定等）
>因此建议先以较少epoch试训，观察效果再决定是否增加。很多教程针对少量数据建议20个epoch左右，但那通常对应几十张图的数据集；对于我们的大数据集，无需那么多轮。

>**正则与其他参数**
>使用8-bit Adam优化器（--optimizer_type AdamW8bit）以降低显存占用，这在kohya_ss中是默认选项（GUI中为“Use 8bit Adam”勾选）
>权重衰减(weight decay)通常可以设为0或极小，LoRA训练主要关注特征拟合。学习率调度可以使用cosine或linear等，让学习率随训练逐渐降低，有助于后期收敛平滑
>图像增强方面，可以开启水平翻转等轻量增强（如--flip_aug），利用素材左右对称性扩充数据，有助于模型学习更多姿态
>分辨率设定：使用--resolution 512,512，脚本会自动对非正方形图片做随机裁剪或缩放（bucketing机制）以适配512尺寸。这样不同宽高的图都能参与训练而不失真

>**角色识别 vs 泛化**
>为了强化角色识别，可确保每张图都有角色名触发词，给予模型反复学习角色独特模式（发型、脸型等)
>为了动作和场景的泛化，则要避免过拟合：一方面通过多样化数据（不同服装背景）和详实标签引导，让模型知道哪些元素是可变的；
>另一方面在训练量上适可而止，监控损失下降曲线，在损失收敛且样本生成效果良好时提早停止训练
>必要时可以引入噪声偏移(noise offset)等高级选项来防止模型过分偏向训练集分布，但通常良好的标签和适度训练周期已经足够
>总之，参数选择上兼顾让模型充分学习到平泽唯的特征，又不过度记忆单一场景，才能在生成时既认得出角色又能听从提示变化场景

#### 推荐参数汇总：
| 训练参数 | 详细配置 | 说明 |
|----------|----------|------|
| **网络结构** |||
| LoRA维度 | 32 | alpha=16 |
| **优化设置** | | |
| 学习率 | UNet: 1e-4<br>TextEncoder: 1e-5 | 分别设置 |
| 优化器 | AdamW 8-bit | beta1=0.9, beta2=0.999 |
| LR调度器 | Cosine | warmup ratio 0.1 |
| **训练配置** | | |
| Batch大小 | 4 | 根据显存调整 |
| 训练轮数 | 2 epochs | ≈2000 steps，可调范围1-3 |
| 分辨率 | 512 | 固定尺寸 |
| **数据增强** | | |
| 随机裁剪 | 开启 | bucketing策略 |
| 水平翻转 | 开启 | 数据增强 |
```
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

python /root/autodl-tmp/lora/train/sd-scripts/train_network.py \
  --pretrained_model_name_or_path /root/autodl-tmp/lora/train/models/AnyLoRA_bakedVae_blessed_fp16.safetensors \
  --train_data_dir /root/autodl-tmp/lora/train/datasets \
  --output_dir /root/autodl-tmp/lora/train/output \
  --output_name lora_yui_v1_dim32a16 \
  --save_model_as safetensors --save_precision fp16 \
  --network_module networks.lora \
  --network_dim 32 --network_alpha 16 \
  --resolution 512,512 \
  --enable_bucket \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --max_train_steps 2000 \
  --learning_rate 1e-4 \
  --text_encoder_lr 1e-5 \
  --stop_text_encoder_training 400 \
  --optimizer_type AdamW8bit \
  --optimizer_args "weight_decay=0.01" "eps=1e-8" \
  --lr_scheduler cosine --lr_warmup_steps 200 \
  --clip_skip 2 \
  --shuffle_caption \
  --caption_extension .txt \
  --min_snr_gamma 5 \
  --xformers \
  --gradient_checkpointing \
  --cache_latents \
  --mixed_precision fp16 \
  --max_data_loader_n_workers 4 \
  --persistent_data_loader_workers \
  --sample_every_n_steps 200 \
  --sample_prompts /root/autodl-tmp/lora/train/prompts.txt \
  --save_every_n_steps 500 \
  --seed 42 \
  --logging_dir /root/autodl-tmp/lora/train/logs
```
**试错**：笔者在这里遇到了数据集无法读取，显存不足，Numpy版本不匹配等问题

**结果**：训练出了一份.safetensors格式文件
## 3.推理与出图
**目的**：依据提示词生成想要的图片

**过程**：首先放置底模`/mnt/.../stable-diffusion-webui/models/Stable-diffusion/AnyLoRA_bakedVae_blessed_fp16.safetensors`

然后放置所有训练输出的lora版本`/mnt/.../stable-diffusion-webui/models/Lora/`

这里需要注意，webui有些情况下并不支持root权限进入，故这里需要新建用户

使用`su weuiuser`切换用户,最后`python launch.py --listen --port 7860 --xformers`进入即可

**试错**：笔者遇到了root权限无法进入，台球乱飞，git库权限不足，git克隆失败等问题

**结果**：生成图像
## 4.后续工作
笔者将会继续调整lora和prompt等问题，争取早日生成符合预期的图像~
