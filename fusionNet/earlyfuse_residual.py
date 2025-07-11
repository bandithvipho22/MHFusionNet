import torch
from torch import nn
from einops import rearrange
# from fusionNet.model_poseformer import PoseTransformer
from common.arguments import parse_args
import time

args = parse_args()


class ResidualFC(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 linear_size=2048,
                 num_stage=3,
                 p_dropout=0.5,
                 ):
        super(ResidualFC, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints + depth
        self.input_size = in_channel
        # 3d joints
        self.output_size = out_channel

        self.dense_in = nn.Sequential(
            nn.Linear(self.input_size, self.linear_size),
            nn.BatchNorm2d(243),
            # nn.LayerNorm([243, 17, self.linear_size]),
            # nn.LayerNorm([243, 17, 128]),
            nn.PReLU(),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(0.25), # cant
            # nn.Dropout(self.p_dropout),
        )

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.dense_out = nn.Sequential(
            nn.Linear(self.linear_size, self.output_size),
        )

    def forward(self, input):

        output = self.dense_in(input)

        for i in range(self.num_stage):
            output = self.linear_stages[i](output)

        output = self.dense_out(output)

        return output

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
            # nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            pass
        return 0


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()

        self.l_size = linear_size
        self.p_dropout = p_dropout
        # self.linear_size = 512  # 128

        self.relu = nn.LeakyReLU(inplace=True)
        # self.relu = nn.PReLU()
        self.dropout = nn.Dropout(self.p_dropout)

        self.dense_1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm2d(243)
        # self.layer_norm1 = nn.LayerNorm([243, 17, self.linear_size])

        self.dense_2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm2d(243)
        # self.layer_norm2 = nn.LayerNorm([243, 17, self.linear_size])

    def forward(self, x):
        y = self.dense_1(x)
        y = self.batch_norm1(y)
        # y = self.layer_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.dense_2(y)
        y = self.batch_norm2(y)
        # y = self.layer_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class FusionNet(nn.Module):
    def __init__(self, num_frame, num_joints, out_chanel):
        super(FusionNet, self).__init__()

        # self.hidden_chanel = hidden_chanel
        self.out_chanel = out_chanel

        # ResidualFC
        self.regression_head = ResidualFC(
            in_channel=3 * args.num_proposals,
            out_channel=out_chanel,
            linear_size=1024,  # 2048, 128
            num_stage=3,  # 2
            # p_dropout=0.25
        )

    def forward(self, x):
        B, H, F, J, C = x.shape

        # # version 1
        # x = x.permute(0, 2, 3, 1, 4)  # [B, F, J, H, C]
        # out_fea_ext = x.reshape(B, F, J, H * C)
        # # out_fea_ext = torch.cat(x, dim=3)
        #
        # # Regression Head (expects input shape [B, F, J, H*C])
        # final_output = self.regression_head(out_fea_ext)

        # version 2
        out_data = []

        for i in range(args.num_proposals):
            hypo_i = x[:, i]
            out_data.append(hypo_i)

        fused_input = torch.cat(out_data, dim=3)

        final_output = self.regression_head(fused_input)

        return final_output


# if __name__ == '__main__':
#     frame = 243
#     c = 2
#     num_joint = 17
#     h = 5
#
#     encoder = FusionNet(num_frame=frame, num_joints=17, out_chanel=3).cuda()
#     # norm_1 = nn.LayerNorm(frame*5)
#
#     input = torch.randn(2, h, frame, 17, 3, dtype=torch.float32).cuda()
#     out_put = encoder(input)
#     print('Done!')
