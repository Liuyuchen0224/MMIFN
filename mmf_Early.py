
import torch.nn as nn
import torch
import einops
from torch.nn import functional as F
from ptflops import get_model_complexity_info

class depthwise_conv_block(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3),
                stride=(1, 1), 
                padding=(1, 1), 
                dilation=(1, 1),
                groups=None, 
                norm_mriype='bn',
                activation=True, 
                use_bias=True,
                pointwise=False, 
                ):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_mriype
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation, 
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features, 
                                        out_features, 
                                        kernel_size=(1, 1), 
                                        stride=(1, 1), 
                                        padding=(0, 0),
                                        dilation=(1, 1), 
                                        bias=use_bias)

        self.norm_mriype = norm_mriype
        self.act = activation

        if self.norm_mriype == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_mriype == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_mriype is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x
    
class depthwise_projection(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                groups,
                kernel_size=(1, 1), 
                padding=(0, 0), 
                norm_mriype=None, 
                activation=False, 
                pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features, 
                                        out_features=out_features, 
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        groups=groups,
                                        pointwise=pointwise, 
                                        norm_mriype=norm_mriype,
                                        activation=activation)
                            
    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P) 
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')      
        return x


class PoolEmbedding2d(nn.Module):
    def __init__(self,
                patch,
                ) -> None:
        super().__init__()
        self.projection = nn.AdaptiveAvgPool2d(output_size=(patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B C (H W)')    
        return x
    
class PoolEmbedding3d(nn.Module):
    def __init__(self,
                patch,
                ) -> None:
        super().__init__()
        self.projection = nn.AdaptiveAvgPool3d(output_size=(patch, patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C D H W -> B C (D H W)')        
        return x

class Layernorm(nn.Module):
    def __init__(self, features, eps=1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(features, eps=eps)                                                  
    def forward(self, x):
        H = x.shape[2]
        x = einops.rearrange(x, 'B C H W -> B (H W) C')        
        x = self.norm(x)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=H) 
        return x       

class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()


        channels = in_channels
        self.out_channels = channels//2

        self.meg_key = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm1d(self.out_channels), nn.ReLU(),
        )
        self.meg_query = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm1d(self.out_channels), nn.ReLU(),
        )
        self.meg_value = nn.Conv1d(in_channels=channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.meg_W = nn.Conv1d(in_channels=self.out_channels, out_channels=channels,
                           kernel_size=1, stride=1, padding=0)

        self.mri_key = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm1d(self.out_channels), nn.ReLU(),
        )
        self.mri_query = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm1d(self.out_channels), nn.ReLU(),
        )
        self.mri_value = nn.Conv1d(in_channels=channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.T_W = nn.Conv1d(in_channels=self.out_channels, out_channels=channels,
                           kernel_size=1, stride=1, padding=0)

        self.gate_meg = nn.Conv1d(channels * 2, 1, kernel_size=1, bias=True)
        self.gate_mri = nn.Conv1d(channels * 2, 1, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, meg, mri):

        # SCA Block
        adapt_channels = 2 * self.out_channels
        batch_size = meg.size(0)
        meg_query = self.meg_query(meg).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        meg_key = self.meg_key(meg).view(batch_size, adapt_channels, -1)
        meg_value = self.meg_value(meg).view(batch_size, adapt_channels, -1).permute(0, 2, 1)

        mri_query = self.mri_query(mri).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        mri_key = self.mri_key(mri).view(batch_size, adapt_channels, -1)
        mri_value = self.mri_value(mri).view(batch_size, adapt_channels, -1).permute(0, 2, 1)

        meg_sim_map = torch.matmul(mri_query, meg_key)
        meg_sim_map = (adapt_channels ** -.5) * meg_sim_map
        meg_sim_map = F.softmax(meg_sim_map, dim=-1)
        meg_context = torch.matmul(meg_sim_map, meg_value)
        meg_context = meg_context.permute(0, 2, 1).contiguous()
        meg_context = meg_context.view(batch_size, self.out_channels,  *meg.size()[2:])
        meg_context = self.meg_W(meg_context)

        mri_sim_map = torch.matmul(meg_query, mri_key)
        mri_sim_map = (adapt_channels ** -.5) * mri_sim_map
        mri_sim_map = F.softmax(mri_sim_map, dim=-1)
        mri_context = torch.matmul(mri_sim_map, mri_value)
        mri_context = mri_context.permute(0, 2, 1).contiguous()
        mri_context = mri_context.view(batch_size, self.out_channels, *mri.size()[2:])
        mri_context = self.T_W(mri_context)


        # CFA Block
        cat_fea = torch.cat([mri_context, meg_context], dim=1)
        attention_vector_meg = self.gate_meg(cat_fea)
        attention_vector_mri = self.gate_mri(cat_fea)

        attention_vector = torch.cat([attention_vector_meg, attention_vector_mri], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_meg, attention_vector_mri = attention_vector[:, 0:1, :], attention_vector[:, 1:2, :]
        new_fusion = meg * attention_vector_meg + mri * attention_vector_mri

        return new_fusion

                


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, dim =2,kernel_size=3, stride=1,padding=1, **kwargs):
        super(BasicBlock, self).__init__()
        if dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif dim == 3:
            conv = nn.Conv3d
            bn = nn.InstanceNorm3d
        self.conv0 = conv(in_channels=in_channel, out_channels=out_channel,
                    kernel_size=1, stride=stride,  bias=False)  
        self.conv1 = conv(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=in_channel)
        self.bn1 = bn(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = conv(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = bn(out_channel)
        
        if stride != 1:
            self.downsample = nn.Sequential(
                    conv(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                    bn(out_channel))
        else:
            self.downsample = None
            
    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = self.conv0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        
        return out
    
    
class MMFNet(nn.Module):

    def __init__(self,num_classes=2):
        super(MMFNet, self).__init__()
        
        self.meg_encoder = nn.Sequential(
            BasicBlock(2, 32,dim =2,kernel_size=(1,15), stride=(1,4),padding=(0,7)),
            BasicBlock(32, 32,dim =2,kernel_size=(3,3), stride=1,padding=1),    
            BasicBlock(32, 64, dim =2,kernel_size=(1,15), stride=(1,4),padding=(0,7)),
            BasicBlock(64, 64, dim =2,kernel_size=(3,3), stride=1,padding=1),
            BasicBlock(64, 128, dim =2,kernel_size=(1,15), stride=(1,4),padding=(0,7)),
            BasicBlock(128, 128, dim =2,kernel_size=(3,3), stride=1,padding=1),
            BasicBlock(128, 256, dim =2,kernel_size=(1,15), stride=(1,4),padding=(0,7)),
            BasicBlock(256, 256, dim =2,kernel_size=(3,3), stride=1,padding=1)  
            )
        self.mri_encoder = nn.Sequential(
            BasicBlock(1, 32,dim =3,kernel_size=(7,7,7), stride=2,padding=3),
            BasicBlock(32, 32,dim =3,kernel_size=(7,7,7), stride=1,padding=3),     
            BasicBlock(32, 64, dim =3,kernel_size=(7,7,7), stride=2,padding=3),
            BasicBlock(64, 64,dim =3,kernel_size=(7,7,7), stride=1,padding=3), 
            BasicBlock(64, 128, dim =3,kernel_size=(7,7,7), stride=2,padding=3),
            BasicBlock(128, 128,dim =3,kernel_size=(7,7,7), stride=1,padding=3), 
            BasicBlock(128, 256, dim =3,kernel_size=(7,7,7), stride=2,padding=3),
            BasicBlock(256, 256,dim =3,kernel_size=(7,7,7), stride=1,padding=3),            
        )
        patch = 96
        self.projection3d = nn.AdaptiveAvgPool3d(output_size=(patch, patch, patch))
        self.projection2d = nn.AdaptiveAvgPool2d(output_size=(patch, patch*patch))        


        # self.megPoolEmbedding = PoolEmbedding2d(64)
        
        self.classifier = nn.Sequential(
            nn.Linear(256*96*36, num_classes),
            nn.ReLU()   
        )
        
    def forward(self, meg, mri):
        mri = self.projection3d(mri).view(mri.size(0),mri.size(1), 96,-1)
        
        meg = self.projection2d(meg).view(meg.size(0),meg.size(1), 96,-1)
        
        
        x = torch.cat([meg,mri],dim=1)
        x = self.meg_encoder(x)
        # x = self.megPoolEmbedding(x)
        
        
        x = x.view(x.size(0), -1)        
        out = self.classifier(x)
        return out

if __name__ == "__main__":
    meg = torch.rand([1,1,102,8192])
    mri = torch.rand([1,1,192,192,192])    
    model = MMFNet(num_classes=2)
    # out = model(meg,mri)
    # print(out.shape)
    from thop import profile
    from thop import clever_format
    # 使用thop分析模型的运算量和参数量
    flops, params = profile(model, inputs=(meg, mri))

    # 将结果转换为更易于阅读的格式
    flops, params = clever_format([flops, params], '%.3f')

    print(f"运算量：{flops}, 参数量：{params}")