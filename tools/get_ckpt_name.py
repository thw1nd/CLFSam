import argparse
import os
import sys
from collections import OrderedDict

import torch

def is_state_dict_like(obj) -> bool:
    """判断一个对象是否像 state_dict：键多为字符串、值多为 Tensor。"""
    if not isinstance(obj, (dict, OrderedDict)):
        return False
    if not obj:
        return False
    str_keys = all(isinstance(k, str) for k in obj.keys())
    tensor_vals = sum(1 for v in obj.values() if torch.is_tensor(v))
    return str_keys and tensor_vals >= max(1, len(obj) // 2)

CANDIDATE_KEYS = [
    "state_dict",
    "model",
    "model_state_dict",
    "net",
    "module",
    "ema_state_dict",
    "weights",
    "params",
    "model_ema",
]

def unwrap_state_dict(obj):
    """
    递归地从各种 checkpoint 包装里剥离，返回一个真正的 state_dict（name->Tensor）。
    """
    # 直接就是 state_dict
    if is_state_dict_like(obj):
        return obj

    # 常见字典包装
    if isinstance(obj, (dict, OrderedDict)):
        # 先尝试常见键
        for k in CANDIDATE_KEYS:
            if k in obj:
                sd = unwrap_state_dict(obj[k])
                if sd is not None:
                    return sd
        # 尝试所有 value
        for v in obj.values():
            sd = unwrap_state_dict(v)
            if sd is not None:
                return sd

    # 列表/元组包装
    if isinstance(obj, (list, tuple)):
        for it in obj:
            sd = unwrap_state_dict(it)
            if sd is not None:
                return sd

    # torch.nn.Module（极少数直接保存）
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        try:
            return obj.state_dict()
        except Exception:
            pass

    return None

def maybe_strip_prefix(state_dict, prefixes=("module.", "model.")):
    """
    如果权重名统一带有前缀（如 DataParallel 的 'module.'），去掉它。
    """
    if not is_state_dict_like(state_dict):
        return state_dict
    keys = list(state_dict.keys())
    need_strip = all(any(k.startswith(p) for p in prefixes) for k in keys)
    if not need_strip:
        return state_dict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd

def load_checkpoint(path):
    print(f"==> Loading checkpoint: {path}")
    obj = torch.load(path, map_location="cpu")
    print(f"Top-level type: {type(obj)}")
    if isinstance(obj, dict):
        print(f"Top-level keys: {list(obj.keys())[:20]}{' ...' if len(obj) > 20 else ''}")
    return obj

def summarize_state_dict(sd):
    n_params = len(sd)
    n_tensors = sum(1 for v in sd.values() if torch.is_tensor(v))
    total_elems = 0
    for v in sd.values():
        if torch.is_tensor(v):
            total_elems += v.numel()
    print(f"\n==> State dict summary:")
    print(f"Params (keys): {n_params}")
    print(f"Tensors       : {n_tensors}")
    print(f"Total elements: {total_elems:,}")

def print_state_dict(sd, limit=None, to_file=None):
    lines = []
    for name, tensor in sd.items():
        if torch.is_tensor(tensor):
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype).replace("torch.", "")
            lines.append(f"{name:60s}  {str(shape):>20s}  {dtype}")
        else:
            # 偶尔会有非 Tensor（比如 None/标量等），也打印出来方便排查
            lines.append(f"{name:60s}  {type(tensor)}")

    if limit is not None:
        print("\n==> First {} entries:".format(limit))
        for line in lines[:limit]:
            print(line)
        if len(lines) > limit:
            print(f"... ({len(lines) - limit} more)")
    else:
        print("\n==> All entries:")
        for line in lines:
            print(line)

    if to_file:
        with open(to_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nSaved full list to: {to_file}")

def main():
    parser = argparse.ArgumentParser(description="Inspect SAM2 checkpoint and list parameter names.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Print only the first N entries (0 = all)")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the full list as a .txt")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: file not found: {args.ckpt}")
        sys.exit(1)

    obj = load_checkpoint(args.ckpt)
    sd = unwrap_state_dict(obj)
    if sd is None:
        print("\nERROR: Could not find a state_dict inside this checkpoint.")
        print("Tip: This file may use a custom format. Try printing the loaded object structure manually.")
        sys.exit(2)

    # 去除常见 'module.' 前缀
    sd = maybe_strip_prefix(sd)

    summarize_state_dict(sd)
    save_path = args.save if args.save else None
    limit = args.limit if args.limit and args.limit > 0 else None
    print_state_dict(sd, limit=limit, to_file=save_path)

if __name__ == "__main__":
    main()
