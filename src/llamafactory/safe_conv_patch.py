import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def apply_patch():
    """
    Monkey patch for Conv3d to fallback to CPU when CUDA fails.
    """
    original_conv_forward = torch.nn.modules.conv.Conv3d._conv_forward

    def safe_conv3d_forward(self, input, weight, bias):
        try:
            return original_conv_forward(self, input, weight, bias)
        except RuntimeError as e:
            if "too many resources" in str(e) or "launch failed" in str(e):
                # 仅在第一次触发时打印日志，防止刷屏
                if not getattr(safe_conv3d_forward, "_logged", False):
                    print(f"\n[Patch] WARNING: CUDA Conv3d failed ({str(e)}). Falling back to CPU for this layer.", file=sys.stderr)
                    safe_conv3d_forward._logged = True
                
                # Move to CPU
                input_cpu = input.cpu()
                weight_cpu = weight.cpu()
                bias_cpu = bias.cpu() if bias is not None else None
                
                # Compute on CPU
                output_cpu = F.conv3d(
                    input_cpu, weight_cpu, bias_cpu, 
                    self.stride, self.padding, self.dilation, self.groups
                )
                
                # Move back to original device
                return output_cpu.to(input.device)
            raise e

    torch.nn.modules.conv.Conv3d._conv_forward = safe_conv3d_forward
    print(">>> [Patch] Safe Conv3d fallback enabled.")

