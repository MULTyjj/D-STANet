import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial

class MLP(nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class DFTLayer(nn.Module):
    def __init__(self, in_channels, delay, num_heads=1, mlp_hidden_units=64):
        super(DFTLayer, self).__init__()
        self.in_channels = in_channels
        self.delay = delay
        self.num_heads = num_heads
        
        # 每个头的特征维度
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        # 为每个头分别定义线性变换层
        self.fc_layers = nn.ModuleList([
            nn.Linear(self.head_dim * (delay + 1), self.head_dim) for _ in range(num_heads)
        ])
        
        # 定义 MLP 处理延迟特征
        self.mlp = MLP(in_features=self.head_dim * (delay + 1), hidden_units=mlp_hidden_units, out_features=self.head_dim)
        
        # 非线性激活函数
        self.activation = nn.ReLU() 

    def forward(self, x):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        
        # 创建延迟的时间序列
        x_delayed = [x]
        for i in range(1, self.delay + 1):
            zero_padding = torch.zeros(batch_size, num_of_vertices, num_of_features, i, device=x.device)
            x_delayed.append(torch.cat([zero_padding, x[:, :, :, :-i]], dim=-1))
        
        x_delayed = torch.cat(x_delayed, dim=-1)  # 直接拼接延迟的序列
        x_delayed = x_delayed.reshape(batch_size, num_of_vertices, num_of_timesteps, -1)
        
        # 将输入数据分成多个头
        x_heads = x_delayed.view(batch_size, num_of_vertices, num_of_timesteps, self.num_heads, -1)
        x_heads = x_heads.permute(3, 0, 1, 2, 4)  # 变换维度顺序为 (num_heads, batch_size, num_of_vertices, num_of_timesteps, head_dim * (delay + 1))
        
        # 对每个头分别应用 MLP
        head_outputs = []
        for i in range(self.num_heads):
            head_output = self.mlp(x_heads[i].view(-1, x_heads[i].size(-1)))  # 展开输入以适配 MLP
            head_output = head_output.view(x_heads[i].size(0), x_heads[i].size(1), -1)  # 恢复原始维度
            head_outputs.append(head_output)
        
        # 将所有头的输出合并
        x_transformed = torch.cat(head_outputs, dim=-1)
        x_transformed = x_transformed.view(batch_size, num_of_vertices, num_of_timesteps, -1).permute(0, 1, 3, 2)
        
        return x_transformed





class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps,ratio=16):
        super(Spatial_Attention_layer, self).__init__()
        self.channel_attention = ChannelAttention_layer(num_of_vertices,
                                                  ratio) 
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        """
        x = self.channel_attention(x)
        # x = x* channel_out

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized

class ChannelAttention_layer(nn.Module):
    def __init__(self, in_channels, ratio=170):
        super(ChannelAttention_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max_out = self.max_pool(x).squeeze(-1).squeeze(-1)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.fc(combined)
        out = self.sigmoid(out).unsqueeze(2).unsqueeze(3)
        return x * out


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels

        # Ensure embed_dim is equal to in_channels
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        
        # Output channels need to match after concatenation
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x shape: (batch_size, num_of_vertices, in_channels)
        batch_size, num_of_vertices, in_channels = x.size()
        
        # Transpose to match MultiheadAttention input shape
        x = x.permute(1, 0, 2)  # (N, batch_size, in_channels)

        # Apply attention
        attn_output, _ = self.attention(x, x, x)  # (N, batch_size, in_channels)

        # Transpose back
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, N, in_channels)

        # Apply the final linear transformation
        output = self.fc(attn_output)  # (batch_size, N, out_channels)
        
        return F.relu(output)

class ChebConv(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super(ChebConv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)]
        )

    def forward(self, x):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = torch.zeros(batch_size, num_of_vertices, self.out_channels, num_of_timesteps).to(self.DEVICE)
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                theta_k = self.Theta[k]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output += rhs.matmul(theta_k)
            outputs[:, :, :, time_step] = F.relu(output)
        return outputs

class ChebConvWithGAT(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels, num_heads=1):
        super(ChebConvWithGAT, self).__init__()
        self.cheb_conv = ChebConv(K, cheb_polynomials, in_channels, out_channels)
        self.gat = GATLayer(out_channels, out_channels, num_heads=num_heads)

    def forward(self, x, spatial_attention):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        cheb_conv_output = self.cheb_conv(x)  # Apply Chebyshev convolution
        spatial_attention = spatial_attention.to(x.device)  # Move to same device as x

        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = cheb_conv_output[:, :, :, time_step]  # (batch_size, num_of_vertices, out_channels)
            gat_output = self.gat(graph_signal, spatial_attention)  # Apply GAT
            outputs.append(gat_output.unsqueeze(-1))  # (batch_size, num_of_vertices, out_channels, 1)

        return torch.cat(outputs, dim=-1)  # (batch_size, num_of_vertices, out_channels, num_of_timesteps)



class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
     
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized



class STANet_with_DFT(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                 num_of_vertices, num_of_timesteps, delay):
        super(STANet_with_DFT, self).__init__()
        self.DFT = DFTLayer(in_channels, delay)
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = ChebConvWithGAT(
    K=K,
    cheb_polynomials=cheb_polynomials,
    in_channels=in_channels,
    out_channels=nb_chev_filter,
    num_heads=1
)  # 这里 num_heads 可以根据需求设置
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.gru = nn.GRU(input_size=nb_time_filter, hidden_size=nb_time_filter, num_layers=1, batch_first=True)
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        x = self.DFT(x)  # 应用DFT层
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size,
                                                                                               num_of_vertices,
                                                                                               num_of_features,
                                                                                               num_of_timesteps)

        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)

        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))

        # 在这里添加GRU层
        x_residual = x_residual.permute(0, 2, 3, 1)  # 调整维度为 (batch_size, num_of_vertices, num_timesteps, features)
        x_residual = x_residual.reshape(batch_size * num_of_vertices, num_of_timesteps, -1)  # 调整为 (batch_size*num_of_vertices, num_timesteps, features)
        x_residual, _ = self.gru(x_residual)
        x_residual = x_residual.reshape(batch_size, num_of_vertices, num_of_timesteps, -1)  # 调整回 (batch_size, num_of_vertices, num_timesteps, features)
        x_residual = x_residual.permute(0, 3, 1, 2)  # 调整回原始维度
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        return x_residual





class DSTANet(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                 num_for_predict, len_input, num_of_vertices, delay):
        super(DSTANet, self).__init__()

        self.BlockList = nn.ModuleList([STANet_with_DFT(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                                              time_strides, cheb_polynomials, num_of_vertices, len_input, delay)])

        self.BlockList.extend([STANet_with_DFT(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                                                     cheb_polynomials, num_of_vertices, len_input // time_strides, delay) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict,
               len_input, num_of_vertices, delay):
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    
    model = DSTANet(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                                      cheb_polynomials, num_for_predict, len_input, num_of_vertices, delay)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model

