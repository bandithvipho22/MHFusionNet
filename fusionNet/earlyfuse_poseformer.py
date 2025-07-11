import torch
from torch import nn
from einops import rearrange
from fusionNet.model_poseformer import PoseTransformer
from common.arguments import parse_args
import time

args = parse_args()


class FusionNet(nn.Module):
    def __init__(self, num_frame, num_joints, out_chanel):
        super().__init__()

        # self.hidden_chanel = hidden_chanel
        self.out_chanel = out_chanel

        self.pose_former = PoseTransformer(num_frame=num_frame, num_joints=num_joints, in_chans=3*args.num_proposals,
                                                 out_chans=out_chanel, embed_dim_ratio=32,
                                                 depth=2, num_heads=4, mlp_ratio=2.0, qkv_bias=True, qk_scale=None,
                                                 drop_path_rate=0.1)  # embed_dim_ratio=32(good), 64

    def forward(self, x):
        B, H, F, J, C = x.shape

        # # version 1
        # x = x.permute(0, 2, 3, 1, 4)  # [B, F, J, H, C]
        # out_fea_ext = x.reshape(B, F, J, H * C)
        #
        # # # Regression Head
        # final_output = self.pose_former(out_fea_ext)

        # # version 2
        out_data = []

        for i in range(args.num_proposals):
            hypo_i = x[:, i]
            out_data.append(hypo_i)

        fused_input = torch.cat(out_data, dim=3)

        final_output = self.pose_former(fused_input)

        return final_output


# if __name__ == '__main__':
#     frame = 243
#     c = 3
#     num_joint = 17
#
#     encoder = FusionNet(num_frame=243, num_joints=17, out_chanel=3).cuda()
#     norm_1 = nn.LayerNorm(frame*5)
#
#     input = torch.randn(10, 5, 243, 17, 3, dtype=torch.float32).cuda()
#     out_put = encoder(input)
#     print('Done!')

#
