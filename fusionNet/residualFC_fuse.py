import torch
from torch import nn
from einops import rearrange
from fusionNet.model_poseformer import PoseTransformer
from fusionNet.module.residualFC import ResidualFC
from common.arguments_edited import parse_args
import time

args = parse_args()


class ResidualFuse(nn.Module):
    def __init__(self, num_frame, num_joints, hidden_chanel, out_chanel):
        super().__init__()

        self.hidden_chanel = hidden_chanel
        self.out_chanel = out_chanel

        # ResidualFC feature extractor
        self.feature_extractor = ResidualFC(in_channel=3, out_channel=hidden_chanel, linear_size=1024, num_stage=2)

        # self.regression_head = nn.Sequential(
        #     nn.Linear(in_features=hidden_chanel * args.num_proposals, out_features=128),
        #     nn.ReLU(),
        #
        #     nn.Linear(in_features=128, out_features=64),
        #     nn.ReLU(),
        #
        #     nn.Linear(in_features=64, out_features=out_chanel)
        # )

        # self.regression_head = nn.Sequential(
        #     nn.Linear(in_features=hidden_chanel * args.num_proposals, out_features=64),
        #     nn.ReLU(),
        #
        #     nn.Linear(in_features=64, out_features=32),
        #     nn.ReLU(),
        #
        #     nn.Linear(in_features=32, out_features=out_chanel)
        # )

        # # ResidualFC
        self.regression_head = ResidualFC(
            in_channel=hidden_chanel * args.num_proposals,
            out_channel=out_chanel,
            linear_size=1024,  # 2048,
            num_stage=3,  # 2
            # p_dropout=0.1
        )

    def forward(self, x):
        # feature extractor (pose former)
        B, H, F, J, _ = x.shape
        out_fea_ext = []
        for i in range(args.num_proposals):
            output_extractor = self.feature_extractor(x[:, i])
            out_fea_ext.append(output_extractor)

        out_fea_ext = torch.cat(out_fea_ext, dim=3)  # → (B, F, J, H*C)

        out_fea_ext = out_fea_ext.reshape(B, F, J, H * self.hidden_chanel)  # (B, F, J * H * C)

        # # Regression Head
        final_output = self.regression_head(out_fea_ext)
        final_output = final_output.reshape(B, F, J, self.out_chanel)

        return final_output


# if __name__ == '__main__':
#     frame = 243
#     c = 2
#     num_joint = 17
#
#     encoder = ResidualFuse(num_frame=243, num_joints=17, hidden_chanel=24, out_chanel=3).cuda()
#     # norm_1 = nn.LayerNorm(frame*5)
#
#     input = torch.randn(2, 5, 243, 17, 3, dtype=torch.float32).cuda()
#     out_put = encoder(input)
#     print('Done!')
