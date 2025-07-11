import torch.nn as nn
# import torch
# from timm.models.layers import DropPath
# from Modulated_GCN.GCN_conv import ModulatedGraphConv


class ResidualFC(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 linear_size=2048,
                 num_stage=3,  # original 2
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
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
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

        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        self.dense_1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm2d(243)

        self.dense_2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm2d(243)

    def forward(self, x):
        y = self.dense_1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.dense_2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

# # Remove dropout
# import torch.nn as nn
# import torch
# # from timm.models.layers import DropPath
# # from Modulated_GCN.GCN_conv import ModulatedGraphConv
#
#
# class ResidualFC(nn.Module):
#     def __init__(self,
#                  in_channel,
#                  out_channel,
#                  linear_size=2048,
#                  num_stage=3,  # original 2
#                  # p_dropout=0.5,
#                  ):
#         super(ResidualFC, self).__init__()
#
#         self.linear_size = linear_size
#         # self.p_dropout = p_dropout
#         self.num_stage = num_stage
#
#         # 2d joints + depth
#         self.input_size = in_channel
#         # 3d joints
#         self.output_size = out_channel
#
#         self.dense_in = nn.Sequential(
#             nn.Linear(self.input_size, self.linear_size),
#             nn.BatchNorm2d(243),
#             nn.LeakyReLU(),
#             # nn.Dropout(self.p_dropout),
#         )
#
#         self.linear_stages = []
#         for l in range(num_stage):
#             # self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
#             self.linear_stages.append(Linear(self.linear_size))
#         self.linear_stages = nn.ModuleList(self.linear_stages)
#
#         self.dense_out = nn.Sequential(
#             nn.Linear(self.linear_size, self.output_size),
#         )
#
#     def forward(self, input):
#
#         output = self.dense_in(input)
#
#         for i in range(self.num_stage):
#             output = self.linear_stages[i](output)
#
#         output = self.dense_out(output)
#
#         return output
#
#     def init_weights(m):
#         if isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
#             # nn.init.normal_(m.weight, std=1e-3)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0.0)
#         elif isinstance(m, nn.BatchNorm1d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
#         else:
#             pass
#         return 0
#
#
# class Linear(nn.Module):
#     # def __init__(self, linear_size, p_dropout=0.5):
#     def __init__(self, linear_size):
#         super(Linear, self).__init__()
#
#         self.l_size = linear_size
#         # self.p_dropout = p_dropout
#
#         self.relu = nn.LeakyReLU(inplace=True)
#         # self.dropout = nn.Dropout(self.p_dropout)
#
#         self.dense_1 = nn.Linear(self.l_size, self.l_size)
#         self.batch_norm1 = nn.BatchNorm2d(243)
#
#         self.dense_2 = nn.Linear(self.l_size, self.l_size)
#         self.batch_norm2 = nn.BatchNorm2d(243)
#
#     def forward(self, x):
#         y = self.dense_1(x)
#         y = self.batch_norm1(y)
#         y = self.relu(y)
#         # y = self.dropout(y)
#
#         y = self.dense_2(y)
#         y = self.batch_norm2(y)
#         y = self.relu(y)
#         # y = self.dropout(y)
#
#         out = x + y
#
#         return out
