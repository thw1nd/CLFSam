from Image_Encoder import Hiera
# from .freq_free_fusion import MultiScaleWaveletFusion
from Fusion_Block import MultiScaleFreqSEFusionV2
import torch.nn as nn
from einops import rearrange


from timm.models.layers import DropPath, trunc_normal_


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.05):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0.08, attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

# 假定你工程已有：Conv, ConvBN, ConvBNReLU, SeparableConvBN

class BaseSMoERefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64, use_hp_inject=True):
        super().__init__()
        self.eps = 1e-8
        self.use_hp_inject = use_hp_inject
        C = decode_channels

        # 1) 对齐 & 融合：x↑ 与 res 一起作为 base 的输入（你建议的做法）
        self.pre_conv  = Conv(in_channels, C, kernel_size=1)      # 对齐 res 通道
        self.post_conv = ConvBNReLU(C, C, kernel_size=3)
        self.fuse_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))  # softplus -> 正权重 -> 归一

        # 2) SMoE 专家（极简 3 个：local / context / bottleneck）
        self.exp_local   = SeparableConvBN(C, C, kernel_size=3)                # 细粒度
        self.exp_context = SeparableConvBN(C, C, kernel_size=3, dilation=2)    # 上下文
        self.exp_bottl   = ConvBN(C, C, kernel_size=1)                         # 瓶颈/重排
        # 逐像素门控：base_in->logits(K=3)->softmax
        self.gate = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(C, 3, kernel_size=1, bias=True)
        )
        # 让门控初始更“平均”，最后一层零初始化
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        # 3) 高频注入（可选）：hp = res' - avg3x3(res')
        self.res_align = Conv(in_channels, C, kernel_size=1)
        self.blur_dw   = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        with torch.no_grad():
            k = torch.ones(1, 1, 3, 3) / 9.0
            self.blur_dw.weight.copy_(k.repeat(C, 1, 1, 1))
        self.blur_dw.weight.requires_grad_(False)
        self.gamma = nn.Parameter(torch.zeros(1, C, 1, 1))  # 逐通道注入系数，初始=0

        # 4) 轻投影残差（与你原版一致）
        self.shortcut = ConvBN(C, C, kernel_size=1)
        self.proj     = SeparableConvBN(C, C, kernel_size=3)
        self.act      = nn.ReLU6()

    def forward(self, x, res):
        # --- 融合成 base_in ---
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        res_p = self.pre_conv(res)
        w = F.softplus(self.fuse_logits)                             # 正权重
        fuse_w = w / (torch.sum(w, dim=0, keepdim=True) + self.eps)  # 归一
        base_in = fuse_w[0] * res_p + fuse_w[1] * x_up
        base = self.post_conv(base_in)

        # --- SMoE：逐像素门控 + 专家混合（以 base 为主体）---
        logits = self.gate(base)                 # [B,3,H,W]
        mix = F.softmax(logits, dim=1)           # 每像素和=1
        e_local   = self.exp_local(base)
        e_context = self.exp_context(base)
        e_bottl   = self.exp_bottl(base)
        moe_out = (mix[:, 0:1] * e_local +
                   mix[:, 1:2] * e_context +
                   mix[:, 2:3] * e_bottl)

        out = base + moe_out

        # --- res 高频“锦上添花”（可选，线性注入）---
        if self.use_hp_inject:
            res_q = self.res_align(res)
            hp = res_q - self.blur_dw(res_q)
            out = out + self.gamma * hp          # γ 初始=0，是否注入/注入多少由学习决定

        # --- 轻投影残差 & 激活 ---
        shortcut = self.shortcut(out)
        out = self.proj(out) + shortcut
        return self.act(out)


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=[96, 192, 384, 768],
                 decode_channels=64,
                 dropout=0.05,
                 window_size=4,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = BaseSMoERefinementHead(
            in_channels=encoder_channels[-4],
            decode_channels=decode_channels,
            )

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FreASam(nn.Module):
    def __init__(self,
                 embed_dim=96,
                 n_class=6,
                 lora_r=8,
                 lora_alpha=8,
                 k_rank_lora=2,
                 k_lora_alpha=2,
                 v_rank_lora=2,
                 v_lora_alpha=2,
                 decoder_chans=64,
                 num_last_blocks_no_detach = 3,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seg_classes = n_class
        self.k_rank_lora = k_rank_lora
        self.k_lora_alpha = k_lora_alpha
        self.v_rank_lora = v_rank_lora
        self.v_lora_alpha = v_lora_alpha
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.no_detach = num_last_blocks_no_detach


        # self.postconv = nn.Conv2d(in_channels=1024, out_channels=encoder_chans[3], kernel_size=1)

        #---- encoder tiny -------
        self.encoder = Hiera(
            embed_dim=self.embed_dim,
            num_heads=1,
            stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9],
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            k_lora_r=self.k_rank_lora,
            k_lora_alpha=self.k_lora_alpha,
            v_lora_r=self.v_rank_lora,
            v_lora_alpha=self.v_lora_alpha) # if you want change params, please change them in image encoder2.py

        self.fusion_module = MultiScaleFreqSEFusionV2(
            in_channels_list=[self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8],
            highfreq_top_k=2,
            p_modality_drop=0.15,
            freeze_hp_epochs=2
        )

        # please pay attrntion at parameter img_size, if the input image size is not same, this parameter need change

        self.decoder = Decoder(num_classes=self.seg_classes, decode_channels=decoder_chans)

    def forward(self, rgb, aux1):
        h, w = rgb.size()[-2:]

        x_list, aux1_list = self.encoder(rgb, aux1)
        # fuse4x, fuse8x, fuse16x, fuse32x = self.fusion_module(x_list, aux1_list, aux2_list)

        fuse_list = self.fusion_module(x_list, aux1_list)

        result = self.decoder(fuse_list[0], fuse_list[1], fuse_list[2], fuse_list[3], h, w)

        return result

# ==================== FPS测试代码 ====================
# ==================== 完整性能测试代码 ====================
import time
import numpy as np
from thop import profile, clever_format


def test_model_performance(model, device='cuda'):
    """
    完整模型性能测试：参数量、FLOPs、FPS
    """
    print("=" * 60)
    print("开始完整模型性能测试")
    print("=" * 60)

    model = model.to(device)
    model.eval()

    # 1. 测试参数量
    print("\n1. 模型参数量测试:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
    print(f"不可训练参数量: {(total_params - trainable_params) / 1e6:.2f} M")

    # 2. 测试FLOPs
    print("\n2. 模型FLOPs测试:")
    rgb_input = torch.randn(1, 3, 512, 512).to(device)
    sar_input = torch.randn(1, 3, 512, 512).to(device)

    try:
        flops, params = profile(model, inputs=(rgb_input, sar_input), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        print(f"FLOPs: {flops_formatted}")
        print(f"参数量 (thop): {params_formatted}")
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        flops = total_params  # 备用值

    # 3. 测试FPS
    print("\n3. 模型FPS测试:")
    fps_results = test_fps_comprehensive(model, device=device)

    # 4. 性能总结
    print("\n" + "=" * 60)
    print("性能测试总结")
    print("=" * 60)
    print(f"参数量: {total_params / 1e6:.2f} M")
    if 'flops_formatted' in locals():
        print(f"FLOPs: {flops_formatted}")
    print(f"推理FPS: {fps_results['inference_fps']:.2f}")
    print(f"端到端FPS: {fps_results['end_to_end_fps']:.2f}")
    print(f"每帧时间: {fps_results['inference_time_per_frame']:.2f} ms")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops if 'flops' in locals() else None,
        'fps': fps_results
    }


def test_fps_comprehensive(model, device='cuda', num_warmup=100, num_test=500):
    """
    综合FPS测试：包含推理时间和端到端时间
    """
    model = model.to(device)
    model.eval()

    # 创建测试数据
    rgb = torch.randn(1, 3, 512, 512).to(device)
    sar = torch.randn(1, 3, 512, 512).to(device)

    print("输入尺寸:")
    print(f"  RGB: {rgb.shape}")
    print(f"  SAR: {sar.shape}")

    # 测试1: 纯推理FPS（不包含数据加载和后处理）
    print("\n[测试1] 纯推理FPS测试...")

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(rgb, sar)

    if device == 'cuda':
        torch.cuda.synchronize()

    # 推理时间测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_test):
            _ = model(rgb, sar)

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    inference_time = end_time - start_time
    inference_fps = num_test / inference_time
    inference_time_per_frame = inference_time / num_test * 1000

    print(f"纯推理FPS: {inference_fps:.2f}")
    print(f"纯推理每帧时间: {inference_time_per_frame:.2f} ms")

    # 测试2: 端到端FPS（模拟真实场景，包含数据预处理）
    print("\n[测试2] 端到端FPS测试...")

    def simulate_data_loading():
        """模拟数据加载和预处理"""
        time.sleep(0.001)  # 模拟1ms的数据加载时间
        return rgb.clone(), sar.clone()

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup // 2):
            rgb_test, sar_test = simulate_data_loading()
            _ = model(rgb_test, sar_test)

    if device == 'cuda':
        torch.cuda.synchronize()

    # 端到端时间测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_test // 2):
            rgb_test, sar_test = simulate_data_loading()
            _ = model(rgb_test, sar_test)

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    end_to_end_time = end_time - start_time
    end_to_end_fps = (num_test // 2) / end_to_end_time
    end_to_end_time_per_frame = end_to_end_time / (num_test // 2) * 1000

    print(f"端到端FPS: {end_to_end_fps:.2f}")
    print(f"端到端每帧时间: {end_to_end_time_per_frame:.2f} ms")

    return {
        'inference_fps': inference_fps,
        'inference_time_per_frame': inference_time_per_frame,
        'end_to_end_fps': end_to_end_fps,
        'end_to_end_time_per_frame': end_to_end_time_per_frame
    }


def test_different_batch_sizes(model, batch_sizes=[1, 2, 4, 8], device='cuda', num_iterations=100):
    """
    测试不同batch size下的性能
    """
    print("\n" + "=" * 50)
    print("不同Batch Size性能测试")
    print("=" * 50)

    model = model.to(device)
    model.eval()

    print("Batch Size\tFPS\t\t每帧时间(ms)\t总吞吐量")
    print("-" * 55)

    results = {}

    for batch_size in batch_sizes:
        rgb = torch.randn(batch_size, 3, 512, 512).to(device)
        sar = torch.randn(batch_size, 3, 512, 512).to(device)

        # 预热
        with torch.no_grad():
            for _ in range(20):
                _ = model(rgb, sar)

        if device == 'cuda':
            torch.cuda.synchronize()

        # 测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(rgb, sar)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        total_frames = batch_size * num_iterations
        fps = total_frames / total_time
        time_per_frame = total_time / total_frames * 1000
        throughput = total_frames / total_time

        results[batch_size] = {
            'fps': fps,
            'time_per_frame': time_per_frame,
            'throughput': throughput
        }

        print(f"{batch_size}\t\t{fps:.2f}\t\t{time_per_frame:.2f}\t\t{throughput:.2f} fps")

    return results


def test_different_resolutions(model, resolutions=[512], device='cuda', num_iterations=50):
    """
    测试不同输入分辨率下的性能
    """
    print("\n" + "=" * 50)
    print("不同分辨率性能测试")
    print("=" * 50)

    model = model.to(device)
    model.eval()

    print("分辨率\t\tFPS\t\t每帧时间(ms)")
    print("-" * 40)

    results = {}

    for resolution in resolutions:
        rgb = torch.randn(1, 3, resolution, resolution).to(device)
        sar = torch.randn(1, 3, resolution, resolution).to(device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(rgb, sar)

        if device == 'cuda':
            torch.cuda.synchronize()

        # 测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(rgb, sar)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        fps = num_iterations / total_time
        time_per_frame = total_time / num_iterations * 1000

        results[resolution] = {
            'fps': fps,
            'time_per_frame': time_per_frame
        }

        print(f"{resolution}x{resolution}\t{fps:.2f}\t\t{time_per_frame:.2f}")

    return results

def measure_fps(model, device="cuda", warmup=10, iters=100, B=1, H=512, W=512):
    model.eval().to(device)
    x = torch.randn(B, 3, H, W, device=device)
    y = torch.randn(B, 3, H, W, device=device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x,y)
    torch.cuda.synchronize()

    # 计时
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x,y)
        torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000.0 / iters
    fps = (iters * B) / (t1 - t0)
    print(f"avg latency: {avg_ms:.2f} ms  |  FPS: {fps:.2f}")
# 如果直接运行这个文件，进行完整性能测试


# 如果直接运行这个文件，进行完整性能测试
if __name__ == "__main__":
    # 初始化网络
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    net = FreASam(n_class=6)
    x = torch.randn(1, 3, 512, 512).to(device)
    y = torch.randn(1, 3, 512, 512).to(device)

    measure_fps(model=net, device=device)

    # 完整性能测试
    performance_results = test_model_performance(net, device=device)

    # 不同batch size测试
    # batch_results = test_different_batch_sizes(net, device=device)

    # 不同分辨率测试
    # resolution_results = test_different_resolutions(net, device=device)

    print("\n🎯 测试完成！")
