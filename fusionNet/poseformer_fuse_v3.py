import torch
from torch import nn
from einops import rearrange
from fusionNet.model_poseformer import PoseTransformer
from fusionNet.module.denseFC import DenseFC
from common.arguments_edited import parse_args
import time

args = parse_args()


class FusionNet(nn.Module):
    def __init__(self, num_frame, num_joints, hidden_chanel, out_chanel):
        super().__init__()

        self.hidden_chanel = hidden_chanel
        self.out_chanel = out_chanel

        self.feature_extractor = PoseTransformer(num_frame=num_frame, num_joints=num_joints, in_chans=3,
                                                 out_chans=hidden_chanel, embed_dim_ratio=32,
                                                 depth=4, num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None,
                                                 drop_path_rate=0.1)
        # DenseFC
        self.regression_head = DenseFC(
            in_channel=hidden_chanel * args.num_proposals,
            out_channel=out_chanel,
            linear_size=1024,  # 2048,
            num_stage=2,  # 2
            p_dropout=0.01
        )

    def forward(self, x):
        # feature extractor (pose former)
        B, H, F, J, _ = x.shape
        out_fea_ext = []
        for i in range(args.num_proposals):
            output_extractor = self.feature_extractor(x[:, i])
            out_fea_ext.append(output_extractor)

        out_fea_ext = torch.cat(out_fea_ext, dim=3)  # → (B, F, J, H*C)

        # out_fea_ext = out_fea_ext.reshape(B, F, J * H * self.hidden_chanel)  # (B, F, J * H * C)

        # # Regression Head
        final_output = self.regression_head(out_fea_ext)
        # final_output = final_output.reshape(B, F, J, self.out_chanel)

        return final_output


# if __name__ == '__main__':
#     frame = 243
#     c = 2
#     num_joint = 17
#
#     encoder = FusionNet(num_frame=243, num_joints=17, hidden_chanel=24, out_chanel=3).cuda()
#     # norm_1 = nn.LayerNorm(frame*5)
#
#     input = torch.randn(2, 5, 243, 17, 3, dtype=torch.float32).cuda()
#     out_put = encoder(input)
#     print('Done!')
