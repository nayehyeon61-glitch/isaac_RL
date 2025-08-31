import os
# from turtle import forward 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
# print(parent_dir)
# print(current_dir)
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from neuralop.models import FNO
#import pykan.kan.KANLayer as kn
#import KANU_Net.src.kan_unet as ku
#import kan
#import matplotlib.pyplot as plt
#import numpy as np
#import torch
#from kan import KAN, create_dataset
#from sklearn.datasets import make_classification, load_iris
#from sklearn.model_selection import train_test_split
def Tonumpy(data):
    if data.ref().ndim == 1 :
        return np.array(data.ref())
    else :
        numpy_data = np.empty((data.rows(),data.cols()))
        numpy_data[:data.rows(),:data.cols()]=data.ref()
        return numpy_data
class NormalNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.fckan= kn.KANLayer(in_dim=input_size,out_dim=output_size)
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,output_size)
#        self.fc5 = nn.Linear(hidden_size,hidden_size)
#        self.fc6 = nn.Linear(hidden_size,output_size)

    def forward(self,input_data):
        input_data = input_data.flatten(-2)
        out = self.fc1(F.elu(input_data))
        out = self.fc2(F.elu(out))
        out = self.fc3(F.elu(out))
#        out4 = self.fc4(F.elu(out3))
#        out5 = self.fc4(F.elu(out4))
        return self.fc4(out)

    def set_normalization(self,mu,std):
        self.mu = mu
        self.std = std

    def normalize(self,t):
        return (t-self.mu)/self.std

    def denormalize(self,t):
        return t*self.std + self.mu

class Encoder2(nn.Module):
    def __init__(self,input_size,latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = 256
        self.num_layers = 1

        #self.device="cuda"
        real_input = self.input_size+35
        #self.gru1 = nn.GRU(real_input,self.hidden_size,batch_first=True)
        #self.gru(self.hidden_size+self.input_size,
        self.fc1 = nn.Linear(real_input,self.hidden_size)
        self.fc2 = nn.Linear(input_size+self.hidden_size,self.hidden_size)
        self.mu  = nn.Linear(input_size+self.hidden_size,latent_size)
        self.var = nn.Linear(input_size+self.hidden_size,latent_size)
    
    def reparameterize(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self,input,condition_input):
        out1 = F.elu(self.fc1(torch.cat((input,condition_input),dim=1)))
        out2 = F.elu(self.fc2(torch.cat((input,out1),dim=1)))
        out3 = torch.cat((input,out2),dim=1)
        return self.mu(out3) , self.var(out3)

    def forward(self,input, condition_input):
        mu , var = self.encode(input,condition_input)
        z = self.reparameterize(mu,var)
        return z,mu,var
class Encoder3(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = 256
        self.num_layers = 1
        
        # Input includes both the input size and the condition input size (+35)
        real_input = self.input_size + 35
        
        # Define GRU layer to process the concatenated inputs
        self.gru = nn.GRU(real_input, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # Linear layers to generate mean and variance from GRU outputs
        self.mu = nn.Linear(self.hidden_size, latent_size)
        self.var = nn.Linear(self.hidden_size, latent_size)
    
    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, input, condition_input):
        # Concatenate input and condition_input
        combined_input = torch.cat((input, condition_input), dim=1)  # dim=2 for GRU (batch, seq_len, features)
        
        # Pass through GRU
        gru_output, _ = self.gru(combined_input)  # Output shape: (batch_size, seq_len, hidden_size)
        
        # Use the output of the last time step
        out_last = gru_output[:, -1]  # Shape: (batch_size, hidden_size)
        print(out_last.shape)
        # Compute mean and variance
        mu = self.mu(out_last)  # Shape: (batch_size, latent_size)
        var = self.var(out_last)  # Shape: (batch_size, latent_size)
        
        return mu, var
    
    def forward(self, input, condition_input):
        # Encode the input and condition input
        mu, var = self.encode(input, condition_input)
        
        # Reparameterize to get latent variable z
        z = self.reparameterize(mu, var)
        
        return z, mu, var


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = 256
        self.num_layers = 1

        real_input = self.input_size + 35
        
        # GRU layer
        self.gru = nn.GRU(input_size=real_input, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # Fully connected layers
        self.fc2 = nn.Linear(latent_size + self.hidden_size, self.hidden_size)
        self.mu = nn.Linear(latent_size + self.hidden_size, 35)
        self.var = nn.Linear(latent_size + self.hidden_size, 35)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, input, condition_input):
        # Concatenate input and condition_input along the last dimension
        #print(input.shape)
        gru_input = torch.cat((input, condition_input), dim=-1)
        
        # Pass through GRU
        gru_output, _ = self.gru(gru_input)
        
        # Take the last hidden state output
        #print(gru_output.shape)
        
        # Pass through fully connected layers
        out2 = F.elu(self.fc2(torch.cat(( gru_output,condition_input), dim=-1)))
        out3 = torch.cat((out2,condition_input), dim=1)
        
        return self.mu(out3), self.var(out3)

    def forward(self, input, condition_input):
        mu, var = self.encode(input, condition_input)
        z = self.reparameterize(mu, var)
        return z, mu, var

class Decoder(nn.Module):
    def __init__(self,input_size,latent_size,num_experts,output_size):
        super().__init__()
        self.input_size = input_size+latent_size
        output_size = output_size
        hidden_size = 256
        self.fc1 = nn.Linear(self.input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size+latent_size,hidden_size)
        self.fc3 = nn.Linear(latent_size+hidden_size,output_size)

    def forward(self,z,condition_input):
        # print("latent_shape : ",z.shape)
        out4 = F.elu(self.fc1(torch.cat((z,condition_input),dim=1)))
        out5 = F.elu(self.fc2(torch.cat((z,out4),dim=1)))
        return self.fc3(torch.cat((z,out5),dim=1))

class Decoder2(nn.Module):
    def __init__(self,input_size,latent_size,num_experts,output_size):
        super().__init__()
        self.input_size = input_size+latent_size
        output_size = output_size
        hidden_size = 256
        self.fc1 = nn.Linear(self.input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size+input_size,hidden_size)
        self.fc3 = nn.Linear(input_size+hidden_size,output_size)

    def forward(self,z,condition_input):
        # print("latent_shape : ",z.shape)
        out4 = F.elu(self.fc1(torch.cat((z,condition_input),dim=1)))
        out5 = F.elu(self.fc2(torch.cat((z,out4),dim=1)))
        return self.fc3(torch.cat((z,out5),dim=1))
class MixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z ):
        coefficients = F.softmax(self.gate(z), dim=1)
        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = z.unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out



class VAE(nn.Module):
    def __init__(self,input_size,latent_size,num_experts,output_size):
        super().__init__()
        self.encoder = Encoder(input_size,latent_size)
        #self.decoder = MixedDecoder(input_size,latent_size,256,1,1,num_experts)
        self.decoder = Decoder(input_size,latent_size,num_experts,output_size)
        
        ############################change initialization orer##########################3
        self.data_std = 0
        self.data_avg = 0
        ############################change initialization orer##########################3
        self.latent_list = []

    def encode(self,x,c):
        z,mu,logvar = self.encoder(x,c)
        return z,mu,logvar
    def forward(self,x,c):
        z,mu,logvar = self.encoder(x,c)
        return self.decoder(z,c),mu,logvar
    def sample(self,z,c):
        return self.decoder(z,c)
    def set_normalization(self,std,avg):
        self.data_std=std
        self.data_avg=avg
    def set_latent_list(self,latent_vectors):
        self.latent_list = latent_vectors

    #######################
    def normalize(self, t):
        return (t - self.data_avg) / self.data_std
    def denormalize(self, t):
        return t * self.data_std + self.data_avg
    #######################

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Encoder4(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder4, self).__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class Decoder4(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder4, self).__init__()
        self.fc = nn.Linear(35, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        return self.relu(self.fc(z))

class DeFeeNetWithEncoderDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(DeFeeNetWithEncoderDecoder, self).__init__()
        self.encoder = Encoder(input_dim, 35)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc =nn.Linear(hidden_dim+input_dim,hidden_dim)
        self.gelu = nn.GELU()
        self.decoder = Decoder(35,35,num_experts=6,output_size=output_dim)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_var_layer = nn.Linear(hidden_dim, output_dim)
        self.data_std = 0
        self.data_avg = 0


    def forward(self, obs, prev_pred, prev_obs,future):
        """
        :param obs: 현재 관찰 \(X_r\) [batch_size, N, input_dim]
        :param prev_pred: 이전 예측 [batch_size, T, input_dim]
        :param prev_obs: 이전 관찰 [batch_size, T, input_dim]
        """
        # 오차 계산
        velocity_obs = prev_obs- prev_pred
        velocity_pred = future - prev_pred
        deviation = velocity_obs - velocity_pred
        
        # 현재 관찰 데이터를 인코딩
        encoded_obs,mean,var = self.encoder(obs,future)
        
        # GRU로 오차 학습
        gru_out, _ = self.gru(deviation)
        
        # 잠재 공간 결합
        combined_latent = torch.cat((encoded_obs, gru_out), dim=-1)
         # 디코딩하여 최종 예측 생성
        hidden_state = self.fc(combined_latent)
        hidden_state = self.gelu(hidden_state)

        # Get mean and log variance for Gaussian distribution
        mean = self.mean_layer(hidden_state)
        log_var = self.log_var_layer(hidden_state)
        std = torch.exp(0.5 * log_var)
        
        
        # Sample using reparameterization trick
        eps = torch.randn_like(std)
        sampled_output = mean + eps * std
        
        # Decode the sampled output
        output = self.decoder(sampled_output,future)
        
        return output,mean,log_var
    def encode(self,x,c):
        z,mu,logvar = self.encoder(x,c)
        return z,mu,logvar
    def sample(self,z,c):
        return self.decoder(z,c)
    def set_normalization(self,std,avg):
        self.data_std=std
        self.data_avg=avg
    def set_latent_list(self,latent_vectors):
        self.latent_list = latent_vectors
     #######################
    def normalize(self, t):
        return (t - self.data_avg) / self.data_std
    def denormalize(self, t):
        return t * self.data_std + self.data_avg
    #######################
class BetaDerivatives():
    def __init__(self,time_steps,beta_start,beta_end):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = time_steps
        
        #Sself.betas = self.prepare_noise_schedule().to(device="cpu")
        
        self.betas = self.prepare_noise_schedule().to(device="cuda")
        self.alpha = 1-self.betas
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start,self.beta_end,self.time_steps)

    def sample_timesteps(self,n):
        return torch.randint(low=1,high=self.time_steps-1,size=(n,))

    def gather(self,a,t):
        return torch.gather(a,1,t)


class GaussianDiffusion():
    def __init__(self,input_size,noise_step,output_size):
        #self.device = "cpu" 
        self.device = "cuda"
        self.input_size = input_size
        self.output_size = output_size
        self.noise_step = noise_step
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.betaderivative = BetaDerivatives(noise_step,self.beta_start,self.beta_end)
        
        self.beta = self.betaderivative.prepare_noise_schedule().to(self.device)
        self.alpha = self.betaderivative.alpha
        self.alpha_hat = self.betaderivative.alpha_hat

    def q_sample(self,x_0,t,noise=None):
        if noise is None:
            noise = torch.randn((t.shape[0],x_0.shape[0]))
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])
        return sqrt_alpha_hat*x_0 + sqrt_one_minus_alpha_hat*noise,noise.to(self.device)

class TimeEmbedding(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.n = n
        self.dim= n//2
        self.fc1 = nn.Linear(n,n)
        self.fc2 = nn.Linear(n,self.dim)
        self.device="cuda"
        #self.device="cpu"

    def activation(self,x):
        return x*F.elu(x)

    def forward(self,t):
        half_dim = self.n//2
        emb = torch.log(torch.tensor(10000.0,device=self.device)/(half_dim-1)).to(device=self.device)
        emb = torch.exp(torch.arange(half_dim,device=self.device)*-emb).to(device=self.device)
        emb = t*emb
        emb = torch.cat((emb.sin(),emb.cos()),dim=1)
        emb = self.activation(self.fc1(emb))
        emb = self.fc2(emb)
        return emb

import torch
from torch import nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-2)
        v = v / v.size(-2)

        context = torch.einsum('b h d j, b h d i -> b h i j', v, k)
        out = torch.einsum('b h i j, b h d j -> b h d i', context, q)

        # Since 'i' = 1, we can remove that dimension by squeezing it
        out = out.squeeze(-1)  # Remove the dimension where 'i' = 1

        # Now we can safely rearrange to (h d, 1, 1) for spatial dimensions
        out = rearrange(out, "b h d -> b (h d) 1 1")

        return self.to_out(out)
# Denoising Diffusion Model using UNet and GaussianDiffusion



class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
class DownsamplingBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

        # Transformer component
        self.transformer = nn.Transformer(
            d_model=out_channels,  # Matches the out_channels of conv1d
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, x):
        # Downsampling part
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # Preparing the input for the transformer
        # Transformer expects (sequence length, batch size, embedding dimension)
        # Permuting the tensor to have shape (sequence_length, batch_size, embedding_dim)
        x = x.permute(2, 0, 1)

        # Creating a dummy target sequence for the decoder part of the transformer
        target = torch.zeros_like(x)

        # Transformer forward pass
        x = self.transformer(x, target)

        # Permuting back to the original shape (batch_size, out_channels, sequence_length)
        x = x.permute(1, 2, 0)

        return x
# Upsampling block
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UpsamplingBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

        # Transformer component
        self.transformer = nn.Transformer(
            d_model=out_channels,  # Matches the out_channels of conv1d
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, x):
        # Upsampling part
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # Preparing the input for the transformer
        # Transformer expects (sequence length, batch size, embedding dimension)
        # Permuting the tensor to have shape (sequence_length, batch_size, embedding_dim)
        x = x.permute(2, 0, 1)

        # Creating a dummy target sequence for the decoder part of the transformer
        target = torch.zeros_like(x)

        # Transformer forward pass
        x = self.transformer(x, target)

        # Permuting back to the original shape (batch_size, out_channels, sequence_length)
        x = x.permute(1, 2, 0)

        return x
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.nn6= nn.Linear(64,out_channels)
        
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        u2 = u2.unsqueeze(-1)
        u2= self.attn5(u2)
        u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3
class UNetO(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.opertor1 = FNO(n_modes=(1,1),in_channels=64,out_channels=64,hidden_channels=64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.opertor2 = FNO(n_modes=(1,1),in_channels=128,out_channels=128,hidden_channels=64)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.opertor3 = FNO(n_modes=(1,1),in_channels=64,out_channels=64,hidden_channels=64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.opertor4 = FNO(n_modes=(1,1),in_channels=128,out_channels=128,hidden_channels=64)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.opertor5 = FNO(n_modes=(1,1),in_channels=64,out_channels=64,hidden_channels=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.opertor6= FNO(n_modes=(1,1),in_channels=128,out_channels=128,hidden_channels=64)
        self.nn6= nn.Linear(64,out_channels)
        
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)
       # print(d1.shape)
        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        
        d1 = self.attn1(d1)
        d1=self.opertor1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        
        
        d2 = self.attn2(d2)
        d2=self.opertor2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        
        
        d3=self.attn0(d3)
        d3=self.opertor3(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = self.opertor4(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 = self.opertor5(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        u2 = u2.unsqueeze(-1)
        u2= self.attn5(u2)
        u2 = self.opertor6(u2)
        u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3
class UNetTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(99, 99)
        self.res1 = ResNetBlock(99,99)
        self.attn1 = Attention(dim=99)
        self.nn2=nn.Linear(99,128)
        self.down2 = DownsamplingBlock(163, 163)
        self.res2 = ResNetBlock(163,163)
        self.attn2 =Attention(dim=163)
        self.nn3= nn.Linear(163,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=70)
        self.up0 = UpsamplingBlock(70,70)
        self.res0 = ResNetBlock(70,70)
        self.nn4 = nn.Linear(70,128)
        self.mid1 =nn.Linear(291,128)
        #self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128,128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(99, 99)
        self.res6 = ResNetBlock(99,99)
        self.attn4 =Attention(dim=99)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.nn6= nn.Linear(128,out_channels)
        self.nn7 = nn.Linear(198,128)
        self.enc = Encoder(35,35)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.trans = nn.Transformer(d_model=35,nhead=5,num_encoder_layers=5,num_decoder_layers=5)

    def forward(self, x,t,p,cur,cur_n):
        # Downsampling
        xx = x
        x_trans,_,_=self.enc(x,p)
        x_t=self.trans(cur_n,cur)
        d0=self.nn1(x)
        d0=torch.cat((d0,x_trans),dim=-1)
        d0=d0.unsqueeze(-1)
        d1 = self.down1(d0)
        
        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1=torch.cat((d1,x_t),dim=-1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        #d2=torch.cat((d2,x_trans),dim=-1)

        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        #d3=torch.cat((d3,x_trans),dim=-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t
        up0=torch.cat((x,x_trans),dim=-1)
        up0=up0.unsqueeze(-1)
        up0=self.up0(up0)
        up0=self.res0(up0)
        x1=up0.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        
        ax1=ax1.unsqueeze(-1)


        


        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)

        u1= torch.cat((u1,x_t),dim=-1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2= self.nn7(u2)
        #u2 = torch.cat((u2,x_trans),dim=-1)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3 = u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        u3=u3.squeeze(-1)

        #u2=self.mid2(u2)
        u3 = self.nn6(u3)
        u3 = u3.unsqueeze(-1)

        # Final output
        u4=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u4


class PUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.mid0 = nn.Linear(70,35)
        self.midd = nn.Linear(35,70)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(70,70)
        self.res0 = ResNetBlock(70,70)
        self.nn4 = nn.Linear(70,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.upd1= UpsamplingBlock(64,64)
        self.updres1= ResNetBlock(64,64)
        self.nupd1 = nn.Linear(64,out_channels)
        self.upd2 = UpsamplingBlock(128,128)
        self.updres2 =ResNetBlock(128,128)
        self.ud2n = nn.Linear(128,64)
        self.updd2 =UpsamplingBlock(64,64)
        self. upddres2 =ResNetBlock(64,64)
        self.nupd2= nn.Linear(128,64)
        self.nupdd2 = nn.Linear(64,out_channels)
        self.nn6= nn.Linear(64,out_channels)
        self.nn7 = nn.Linear(3*out_channels,out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)
        

        d1 = self.res1(d1)
       
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        upd1=d1
        d1 = d1.squeeze(-1)
        d11=d1
        
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        
        d2 = self.res2(d2)


        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        upd2=d2
        d2 = d2.squeeze(-1)
        d22=d2
        
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=torch.cat((x,t),dim=-1)

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x = x.squeeze(-1)
        x=self.mid0(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        ax1= self.midd(ax1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        #u1 = u1.unsqueeze(-1)
        #u1 = self.attn3(u1)
        #u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        #u2 = u2.unsqueeze(-1)
        #u2 = self.attn4(u2)
        #u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        #u2 = u2.unsqueeze(-1)
        #u2= self.attn5(u2)
        #u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        upd1=self.upd1(upd1)
        upd1=self.updres1(upd1)
        upd1=upd1.squeeze(-1)
        upd1=self.nupd1(upd1)
        
        upd2=self.upd2(upd2)
        upd2=self.updres2(upd2)
        upd2=upd2.squeeze(-1)
        upd2=self.nupd2(upd2)
        upd2=upd2.unsqueeze(-1)
        upd2=self.updd2(upd2)
        upd2=self.upddres2(upd2)
        upd2=upd2.squeeze(-1)
        upd2=self.nupdd2(upd2)
        u2=u2.squeeze(-1)

        u2=torch.cat((u2,upd1),dim=-1)
        u2=torch.cat((u2,upd2),dim=-1)
        u2=self.nn7(u2)
        u2=u2.unsqueeze(-1)
        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3
class UNet0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.mid0 = nn.Linear(70,35)
        self.midd = nn.Linear(35,70)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(70,70)
        self.res0 = ResNetBlock(70,70)
        self.nn4 = nn.Linear(70,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.nn6= nn.Linear(64,out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=torch.cat((x,t),dim=-1)

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x = x.squeeze(-1)
        x=self.mid0(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        ax1= self.midd(ax1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        #u1 = u1.unsqueeze(-1)
        #u1 = self.attn3(u1)
        #u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        #u2 = u2.unsqueeze(-1)
        #u2 = self.attn4(u2)
        #u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        #u2 = u2.unsqueeze(-1)
        #u2= self.attn5(u2)
        #u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3
class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.Lattn1 =LinearAttention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.Lattn2 =LinearAttention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.Lattn3 =LinearAttention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.Lattn4 = LinearAttention(dim=64)
        self.nn6= nn.Linear(64,out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = self.Lattn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2 = self.Lattn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = self.Lattn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 = self.Lattn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3

class UNet3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)



        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3

class UNet3_ITER(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)



        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3

class UNet4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.n0 = nn.Linear(35,175)
        self.n00 = nn.Linear(175,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        x=d4.squeeze(-1)
        x=self.n0(x)



        #print(x.shape)
        x=x+t


        x=self.n00(x)
        #x=self.n00(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3


class UNet5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,256)
        self.down22 =DownsamplingBlock(256,256)
        self.res22 = ResNetBlock(256,256)
        self.attn22 =Attention(dim=256)
        self.nn22 = nn.Linear(256,512)
        self.down33 =DownsamplingBlock(512,512)
        self.res33 = ResNetBlock(512,512)
        self.attn34 =Attention(dim=512)
        self.nn33 =nn.Linear(512,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.attn01=Attention(35)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.n0 = nn.Linear(35,175)
        self.n00 = nn.Linear(175,35)
        self.attn = Attention(dim=70)
        self.mid11 =nn.Linear(70,35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,256)
        self.up22 = UpsamplingBlock(256,256)
        self.res22 =ResNetBlock(256,256)
        self.attn33 =Attention(256)
        self.nn55 = nn.Linear(256,512)
        self.up33 =UpsamplingBlock(512,512)
        self.res66 =ResNetBlock(512,512)
        self.attn44 =Attention(dim=512)
        self.nn44 =nn.Linear(1024,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        #xx=x
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down22(d2)
        d3 = self.res22(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn22(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn22(d3)
        d33=d3
        d3=d3.unsqueeze(-1)
        d4=self.down33(d3)
        d4=self.res33(d4)
        d4=d4.unsqueeze(-1)
        #print(d4.shape)
        d4=self.attn34(d4)
        d4=d4.squeeze(-1)

        #x=d4.squeeze(-1)
        #print(d4.shape)
        d4=d4.squeeze(-1)
        d55= d4
        d5=self.nn33(d4)
        d5=d5.unsqueeze(-1)
        d5=self.down3(d5)
        d5=self.res3(d5)
        d5=d5.unsqueeze(-1)
        d5=self.attn0(d5)
        d5=d5.squeeze(-1)
        d5=d5.squeeze(-1)
        d6=self.nn0(d5)
        d6=d6.unsqueeze(-1)
        d6=self.down4(d6)
        d6=self.res4(d6)
        d6=d6.unsqueeze(-1)
        d6=self.attn01(d6)
        d6=d6.squeeze(-1)
        x=d6.squeeze(-1)
        #x=self.n0(x)


        #print(x.shape)
        x=torch.cat((x,t),dim=1)



        #x=self.n00(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=ax1.squeeze(-1)
        ax1=self.mid11(ax1)
        ax1=ax1.unsqueeze(-1)

        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        u12 = self.up22(u1)
        u12 = self.res22(u12)
        u12 = u12.unsqueeze(-1)
        u12 = self.attn33(u12)
        u12 =u12.squeeze(-1)
        u12 = u12.squeeze(-1)
        u12 = self.nn55(u12)
        u12 = u12.unsqueeze(-1)
        u13 = self.up33(u12)
        u13 = self.res66(u13)
        u13= u13.unsqueeze(-1)
        u13 = self.attn44(u13)
        u13 = u13.squeeze(-1)
        u13 = u13.squeeze(-1)
        u13= torch.cat((u13,d33),dim=1)
        u13 = self.nn44(u13)
        u13 = u13.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u13)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        #u2 = torch.cat((u2,xx),dim=1)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3

class UNet6(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,256)
        self.down22 =DownsamplingBlock(256,256)
        self.res22 = ResNetBlock(256,256)
        self.attn22 =Attention(dim=256)
        self.nn22 = nn.Linear(256,512)
        self.down33 =DownsamplingBlock(512,512)
        self.res33 = ResNetBlock(512,512)
        self.attn34 =Attention(dim=512)
        self.nn33 =nn.Linear(512,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.attn01=Attention(35)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.n0 = nn.Linear(35,175)
        self.n00 = nn.Linear(175,35)
        self.attn = Attention(dim=70)
        self.mid11 =nn.Linear(70,35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,256)
        self.up22 = UpsamplingBlock(256,256)
        self.res22 =ResNetBlock(256,256)
        self.attn33 =Attention(256)
        self.nn55 = nn.Linear(512,512)
        self.up33 =UpsamplingBlock(512,512)
        self.res66 =ResNetBlock(512,512)
        self.attn44 =Attention(dim=512)
        self.nn44 =nn.Linear(1024,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(128,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.mid3 = nn.Linear(out_channels,64)
        self.nn7 = nn.Linear(128,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)
        xx=x
        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down22(d2)
        d3 = self.res22(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn22(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d31=d3
        d3=self.nn22(d3)
        d33=d3
        d3=d3.unsqueeze(-1)
        d4=self.down33(d3)
        d4=self.res33(d4)
        d4=d4.unsqueeze(-1)
        #print(d4.shape)
        d4=self.attn34(d4)
        d4=d4.squeeze(-1)

        #x=d4.squeeze(-1)
        #print(d4.shape)
        d4=d4.squeeze(-1)
        d55= d4
        d5=self.nn33(d4)
        d5=d5.unsqueeze(-1)
        d5=self.down3(d5)
        d5=self.res3(d5)
        d5=d5.unsqueeze(-1)
        d5=self.attn0(d5)
        d5=d5.squeeze(-1)
        d5=d5.squeeze(-1)
        d6=self.nn0(d5)
        d6=d6.unsqueeze(-1)
        d6=self.down4(d6)
        d6=self.res4(d6)
        d6=d6.unsqueeze(-1)
        d6=self.attn01(d6)
        d6=d6.squeeze(-1)
        x=d6.squeeze(-1)
        #x=self.n0(x)


        #print(x.shape)
        x=torch.cat((x,t),dim=1)



        #x=self.n00(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=ax1.squeeze(-1)
        ax1=self.mid11(ax1)
        ax1=ax1.unsqueeze(-1)

        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        u12 = self.up22(u1)
        u12 = self.res22(u12)
        u12 = u12.unsqueeze(-1)
        u12 = self.attn33(u12)
        u12 =u12.squeeze(-1)
        u12 = u12.squeeze(-1)
        u12=torch.cat((u12,d31),dim=1)
        u12 = self.nn55(u12)
        u12 = u12.unsqueeze(-1)
        u13 = self.up33(u12)
        u13 = self.res66(u13)
        u13= u13.unsqueeze(-1)
        u13 = self.attn44(u13)
        u13 = u13.squeeze(-1)
        u13 = u13.squeeze(-1)
        u13= torch.cat((u13,d33),dim=1)
        u13 = self.nn44(u13)
        u13 = u13.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u13)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)

        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3 = u3.squeeze(-1)
        u3 = self.mid3(u3)
        u3 = torch.cat((u3,xx),dim=1)
        u3 = self.nn7(u3)
        u3 = u3.unsqueeze(-1)

        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3
class TransformerModel(nn.Module):
    def __init__(self, in_channels, nhead=5, num_encoder_layers=2, dim_feedforward=64, dropout=0.1):
        super(TransformerModel, self).__init__()
        # Transformer U-net
        # Define the transformer layer
        self.transformer_layer = nn.Transformer(
            d_model=in_channels,  # input size (same as in_channels in Linear layer)
            nhead=nhead,  # number of attention heads
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,  # For encoder-decoder architecture, specify decoder layers
            dim_feedforward=dim_feedforward,  # corresponds to the hidden dimension in feedforward layers
            dropout=dropout
        )
        self.transformer_layer2 = nn.Transformer(
            d_model=in_channels,  # input size (same as in_channels in Linear layer)
            nhead=nhead,  # number of attention heads
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,  # For encoder-decoder architecture, specify decoder layers
            dim_feedforward=dim_feedforward,  # corresponds to the hidden dimension in feedforward layers
            dropout=dropout
        )

       

        self.transformer_layer3 = nn.Transformer(
            d_model=in_channels*2,  # input size (same as in_channels in Linear layer)
            nhead=nhead,  # number of attention heads
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,  # For encoder-decoder architecture, specify decoder layers
            dim_feedforward=dim_feedforward,  # corresponds to the hidden dimension in feedforward layers
            dropout=dropout
        )
       
        self.transformer_layer4 = nn.Transformer(
            d_model=in_channels*3,  # input size (same as in_channels in Linear layer)
            nhead=nhead,  # number of attention heads
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,  # For encoder-decoder architecture, specify decoder layers
            dim_feedforward=dim_feedforward,  # corresponds to the hidden dimension in feedforward layers
            dropout=dropout
        )
 
        # You may still need to apply a final linear projection
        self.fc_out = nn.Linear(in_channels, 35)  # Final linear layer to reduce to 64 dims

    def forward(self, src, tgt):
        # src is the input sequence
        # tgt is the target sequence (can be None if only using the encoder)
        output = self.transformer_layer(src,tgt)
        #down1= output
        #output = self.transformer_layer2(output,condition)
        #down2 =output
        #output = output+tgt
        #output =torch.cat((output,down1),dim=-1)
        #c=torch.cat((v,c),dim=-1)
        #output = self.transformer_layer3(output,c)
        #output = torch.cat((output,down2),dim=-1)

        #output2 = self.transformer_layer2(src2,tgt)
        #output3 = self.transformer_layer3(src3,tgt)
        #Final_output = torch.cat((output,output2),dim=-1)
        #Final_output = torch.cat((Final_output,output3),dim=-1)
        #Final_output=  self.transformer_layer4(output,c)
        # Final projection (if needed)
        output = self.fc_out(output)
        return output
class UNet7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans1=TransformerModel(in_channels=35)
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,256)
        self.down22 =DownsamplingBlock(256,256)
        self.res22 = ResNetBlock(256,256)
        self.attn22 =Attention(dim=256)
        self.nn22 = nn.Linear(256,512)
        self.down33 =DownsamplingBlock(512,512)
        self.res33 = ResNetBlock(512,512)
        self.attn34 =Attention(dim=512)
        self.nn33 =nn.Linear(512,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.attn0=Attention(64)
        self.attn01=Attention(35)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.n0 = nn.Linear(35,175)
        self.n00 = nn.Linear(175,35)
        self.attn = Attention(dim=70)
        self.mid11 =nn.Linear(70,35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.attn00 = Attention(dim=35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,256)
        self.up22 = UpsamplingBlock(256,256)
        self.res22 =ResNetBlock(256,256)
        self.attn33 =Attention(256)
        self.nn55 = nn.Linear(256,512)
        self.up33 =UpsamplingBlock(512,512)
        self.res66 =ResNetBlock(512,512)
        self.attn44 =Attention(dim=512)
        self.nn44 =nn.Linear(1024,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.nn6= nn.Linear(64,out_channels)

        self.up3 =UpsamplingBlock(out_channels,out_channels)
        self.res7=ResNetBlock(out_channels,out_channels)
        self.attn5 =Attention(dim=out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t):
        # Downsampling
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)

        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down22(d2)
        d3 = self.res22(d3)
        d3=d3.unsqueeze(-1)
        d3=self.attn22(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn22(d3)
        d33=d3
        d3=d3.unsqueeze(-1)
        d4=self.down33(d3)
        d4=self.res33(d4)
        d4=d4.unsqueeze(-1)
        #print(d4.shape)
        d4=self.attn34(d4)
        d4=d4.squeeze(-1)

        #x=d4.squeeze(-1)
        #print(d4.shape)
        d4=d4.squeeze(-1)
        d55= d4
        d5=self.nn33(d4)
        d5=d5.unsqueeze(-1)
        d5=self.down3(d5)
        d5=self.res3(d5)
        d5=d5.unsqueeze(-1)
        d5=self.attn0(d5)
        d5=d5.squeeze(-1)
        d5=d5.squeeze(-1)
        d6=self.nn0(d5)
        d6=d6.unsqueeze(-1)
        d6=self.down4(d6)
        d6=self.res4(d6)
        d6=d6.unsqueeze(-1)
        d6=self.attn01(d6)
        d6=d6.squeeze(-1)
        x=d6.squeeze(-1)
        #x=self.n0(x)


        #print(x.shape)
        x=torch.cat((x,t),dim=1)



        #x=self.n00(x)
        x=x.unsqueeze(-1)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1=ax1.squeeze(-1)
        ax1=self.mid11(ax1)
        ax1=ax1.unsqueeze(-1)

        ax1=self.up0(ax1)
        ax1=self.res0(ax1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        u12 = self.up22(u1)
        u12 = self.res22(u12)
        u12 = u12.unsqueeze(-1)
        u12 = self.attn33(u12)
        u12 =u12.squeeze(-1)
        u12 = u12.squeeze(-1)
        u12 = self.nn55(u12)
        u12 = u12.unsqueeze(-1)
        u13 = self.up33(u12)
        u13 = self.res66(u13)
        u13= u13.unsqueeze(-1)
        u13 = self.attn44(u13)
        u13 = u13.squeeze(-1)
        u13 = u13.squeeze(-1)
        u13= torch.cat((u13,d33),dim=1)
        u13 = self.nn44(u13)
        u13 = u13.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u13)
        u2 = self.res6(u2)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)
        u3= self.up3(u2)
        u3= self.res7(u3)
        u3=u3.unsqueeze(-1)
        u3= self.attn5(u3)
        u3=u3.squeeze(-1)
        # Final output
        u3=self.final_conv(u3).squeeze(-1)
        #print(u3.shape)
        return u3
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        #print(x.shape)
        #print(residual.shape)
        return x + residual
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, out_channels)
        self.gcn2 = GCNConv(out_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        residual = x
        x = self.gcn1(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.gcn2(x, edge_index)
        return x + residual

class UNet8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn1=nn.Linear(in_channels,64)
        self.down1 = DownsamplingBlock(64, 64)
        self.res1 = ResNetBlock(64,64)
        self.gcn1 =GCNBlock(64,64)
        self.attn1 = Attention(dim=64)
        self.nn2=nn.Linear(64,128)
        self.down2 = DownsamplingBlock(128, 128)
        self.res2 = ResNetBlock(128,128)
        self.gcn2 = GCNBlock(128,128)
        self.attn2 =Attention(dim=128)
        self.nn3= nn.Linear(128,64)
        self.down3 =DownsamplingBlock(64,64)
        self.res3 = ResNetBlock(64,64)
        self.gcn3 = GCNBlock(64,64)
        self.attn0=Attention(64)
        self.nn0 = nn.Linear(64,35)
        self.down4 = DownsamplingBlock(35, 35)
        self.res4= ResNetBlock(35,35)
        self.gcn4 = GCNBlock(35,35)
        self.attn = Attention(dim=35)
        self.up0 = UpsamplingBlock(35,35)
        self.res0 = ResNetBlock(35,35)
        self.gcn0 = GCNBlock(35,35)
        self.nn4 = nn.Linear(35,128)
        self.mid1 =nn.Linear(256,128)
        self.mid2 =nn.Linear(128,64)
        self.up1 = UpsamplingBlock(128, 128)
        self.res5 = ResNetBlock(128,128)
        self.gcn5 = GCNBlock(128,128)
        self.attn3 = Attention(dim=128)
        self.nn5 = nn.Linear(128,64)
        self.up2 = UpsamplingBlock(64, 64)
        self.res6 = ResNetBlock(64,64)
        self.gcn6 = GCNBlock(64,64)
        self.attn4 =Attention(dim=64)
        self.up3 = UpsamplingBlock(128,128)
        self.res7 = ResNetBlock(128,128)
        self.gcn7 = GCNBlock(128,128)
        self.attn5 = Attention(dim=128)
        self.nn6= nn.Linear(64,out_channels)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x,t,edge):
        # Downsampling
        xx = x
        x=self.nn1(x)

        x=x.unsqueeze(-1)
        d1 = self.down1(x)

        d1 = self.res1(d1)
        d1= self.gcn1(d1,edge)
        d1 = d1.unsqueeze(-1)
        d1 = self.attn1(d1)
        d1 = d1.squeeze(-1)
        d1 = d1.squeeze(-1)
        d11=d1
        d1 = self.nn2(d1)
        d1 = d1.unsqueeze(-1)
        d2 = self.down2(d1)
        d2 = self.res2(d2)
        d2= self.gcn2(d2,edge)
        d2 = d2.unsqueeze(-1)
        d2 = self.attn2(d2)
        d2=d2.squeeze(-1)
        d2 = d2.squeeze(-1)
        d22=d2
        d2 = self.nn3(d2)
        d2 = d2.unsqueeze(-1)
        d3 = self.down3(d2)
        d3 = self.res3(d3)
        d3 = self.gcn3(d3,edge)
        d3=d3.unsqueeze(-1)
        d3=self.attn0(d3)
        d3=d3.squeeze(-1)
        d3=d3.squeeze(-1)
        d3=self.nn0(d3)
        d3=d3.unsqueeze(-1)
        d4=self.down4(d3)
        d4=self.res4(d4)
        d4= self.gcn4(d4,edge)
        x=d4.squeeze(-1)


        #print(x.shape)
        x=x+t

        x=x.unsqueeze(-1)
        x=self.up0(x)
        x=self.res0(x)
        x1=x.unsqueeze(-1)
        ax = self.attn(x1)
        #print(ax.shape)
        #print(x.shape)

        # Upsampling
        ax1=ax.squeeze(-1)
        ax1 = ax1.squeeze(-1)
        #x=x.squeeze(-1)
        ax1 = self.nn4(ax1)

        ax1 = ax1.unsqueeze(-1)
        ax1 = ax1.squeeze(-1)

        ax1 = torch.cat((ax1,d22),dim=1)
        ax1 = self.mid1(ax1)
        ax1=ax1.unsqueeze(-1)



        u1 = self.up1(ax1)
        u1 = self.res5(u1)
        u1 = self.gcn5(u1,edge)
        u1 = u1.unsqueeze(-1)
        u1 = self.attn3(u1)
        u1 = u1.squeeze(-1)
        u1 = u1.squeeze(-1)
        u1 = self.nn5(u1)
        u1 = u1.unsqueeze(-1)
        #print(u1.shape)
        u2 = self.up2(u1)
        u2 = self.res6(u2)
        u2 = self.gcn6(u2,edge)
        u2 = u2.unsqueeze(-1)
        u2 = self.attn4(u2)
        u2 =u2.squeeze(-1)

        u2 = u2.squeeze(-1)
        u2= torch.cat((u2,d11),dim=1)
        u2 = u2.unsqueeze(-1)
        u2= self.up3(u2)
        u2= self.res7(u2)
        u2 = self.gcn7(u2,edge)
        u2 = u2.unsqueeze(-1)
        u2= self.attn5(u2)
        u2=u2.squeeze(-1)
        u2=u2.squeeze(-1)

        u2=self.mid2(u2)
        u2 = self.nn6(u2)
        u2 = u2.unsqueeze(-1)

        # Final output
        u3=self.final_conv(u2).squeeze(-1)
        #print(u3.shape)
        return u3


# Denoising Diffusion Model
class DenoiseDiffusion(nn.Module):
    def __init__(self,input_size,output_size,noise_steps):
        super().__init__()
        #self.input_size = input_size
        self.input_size= input_size
        self.output_size = output_size
        self.noise_steps = noise_steps
        self.hidden_size = 512
        self.time_dim=70
        self.time_dim2=35
        #self.time_dim2=self.time_sim
        #self.time_dim = self.hidden_size
        self.gaussiandiffusion = GaussianDiffusion(self.input_size,self.noise_steps,self.output_size)
        self.timeembedding = TimeEmbedding(self.time_dim)
        self.betas = self.gaussiandiffusion.beta
        self.alpha = 1-self.betas
        self.alpha_bar = torch.cumprod(self.alpha,dim=0)
        #self.d1 = DownsamplingBlock2(self.input_size,self.input_size)
        #print(self.input_size)
        #self.nn1 =nn.Linear(self.output_size,self.output_size*5)
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.time_dim2,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.time_dim2,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
        self.fc6 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
        #self.fc6 = nn.Linear(self.hidden_size + self.time_dim2, self.hidden_size)
        self.fc7 = nn.Linear(self.hidden_size+self.time_dim2,self.output_size)
        #self.up1 = UpsamplingBlock2(self.input_size,self.input_size)
        self.mean=0
        self.var=0

    def q_xt_x0(self,x0,t):
        mean = self.gaussiandiffusion.alpha_hat
        mean = self.gaussiandiffusion.alpha_hat[t]**0.5*x0
        var = 1-self.gaussiandiffusion.alpha_hat[t]
        self.mean=mean
        self.var=var
        return mean , var

    def q_sample(self,x0,t,eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean,var = self.q_xt_x0(x0,t)
        return mean
        #return mean+(var**0.5)*eps

    def p_sample(self,xt,t):
    #def p_sample(self,xt,t):
        #eps_theta=self.forward(xt,t)
        eps_theta = self.forward(xt,t)
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[t]
        alpha = self.gaussiandiffusion.alpha[t]

        eps_coef = (1-alpha)/(1-alpha_hat)**0.5
        #eps_coef2=eps_coef.repeat(1,5)
        #alpha2=alpha.repeat(1,5)
        mean = 1/(alpha**0.5)*(xt-eps_coef*eps_theta)
        var = self.gaussiandiffusion.beta[t]
        #var2=var.repeat(1,5)
        eps = torch.randn_like(xt)

        return mean + (var**0.5)*eps

    def forward(self,xt,t):

        t = self.timeembedding(t)
        #xt=xt.unsqueeze(-1)
        #xt=self.d1(xt)
        #xt=self.up1(xt)
        #xt=xt.squeeze(-1)

        emb = self.fc1(xt)
        emb = torch.cat((emb,t),dim=1)
        emb = F.elu(self.fc2(emb))
        emb = torch.cat((emb,t),dim=1)
        emb = F.elu(self.fc3(emb))
        emb = torch.cat((emb,t),dim=1)
        emb = F.elu(self.fc4(emb))
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc5(emb))
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc6(emb))
        emb = torch.cat((emb, t), dim=1)
        emb = F.elu(self.fc7(emb))
        return emb
class DenoiseDiffusion2(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0,t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_= self.forward(xt,t)
        tt=torch.tensor(0).long().to(device="cpu")
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):

        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        mean,var=self.q_xt_x0(x,t)
        return x,mean,var

class DenoiseDiffusion4(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var


class DenoiseDiffusion5(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion6(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet2(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var

class DenoiseDiffusion7(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion8(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)
        self.mean=0
        self.logvar=0

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var

class DenoiseDiffusion77(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)

    def q_xt_x0(self, x0, t):
        #gaussian 을  t 에 맟추어서 분포시
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion9(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet4(175, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean + (var ** 0.5) * eps

    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var


from tqdm import tqdm
class DenoiseDiffusion10(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet4(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(t).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)

        return mean + (var ** 0.5) * eps


    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion11(nn.Module):
    def __init__(self, input_size, noise_steps, output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet5(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t):
        eps_theta,_,_ = self.forward(xt,t)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)

        return mean + (var ** 0.5) * eps


    def forward(self, x,t):
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        return x,mean,var
class DenoiseDiffusion12(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet4(175, 175)  # Assuming 3 input and 3 output channels for RGB
        self.encoder=Encoder(input_size,175)
        self.decoder=Decoder(350,350,5,output_size*5)
        self.time_embed = TimeEmbedding(350)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.nn1= nn.Linear(175,output_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)

        x=mean + (var ** 0.5) * eps
        x=self.nn1(x)
        return x


    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)

        return self.decoder(zc,c),mean,var

class DenoiseDiffusion13(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.trans=TransformerModel(in_channels=input_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)

        return mean + (var ** 0.5) * eps


    def forward(self, x,t,c):
        xc=self.trans(x,c)
        z,mu,logvar=self.encoder(xc,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)

        return self.decoder(zc,c),mean,var

class DenoiseDiffusion13_ITER(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.trans=TransformerModel(in_channels=input_size)
        self.unet = UNet4(35, 175)  # Assuming 3 input and 3 output channels for RGB
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,210,5,5*output_size)
        self.decoder2 = Decoder(35,175,5,output_size)
        self.time_embed = TimeEmbedding(350)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.n= nn.Linear(35,input_size)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        z = mean+(var**0.5)*eps
        return self.decoder2(z,c)
        #return mean + (var ** 0.5) * eps


    def forward(self, x,t,c):
        #xc=self.trans(x,c)
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)

        return self.decoder(zc,c),mean,var


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class FeatureGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        self.gcn1 = GCNConv(in_channels=35, out_channels=35)
        self.gcn2 = GCNConv(in_channels=35, out_channels=35)

    def forward(self, x,M):  # x: [batch_size, num_features]
        batch_size, num_features = x.size()
        edge_index, edge_weight = dense_to_sparse(M)
        x = x.T  # [num_features, batch_size]
        x = x.unsqueeze(-1)  # [num_features, batch_size, 1]
        #x = x.reshape(num_features, -1)  # [num_features, batch_size]
        x = x.reshape(batch_size, num_features)
        #x=x.T
       # print(x.shape)
      #  print(edge_index.shape)
        # Treat each sample's feature as a "node" in a graph
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = x.T  # [batch_size, num_features]
        x=x.T
        return x
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # width 축 절반 (2048 -> 1024)

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # (1024 -> 512)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # (512 -> 256)
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((35, 35)), # 출력 shape을 (35,35)로 맞춤
        )

        self.final_conv = nn.Conv2d(128, 1, kernel_size=(1,1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x
class GNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # bias를 포함하지 않는 basic linear transformation
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
 
    def forward(self, x, adjacency):
        
        # (1) Linear trasnformation
        x = self.linear(x)
        
        # (2) Multiplication with the adjacency matrix A
        x = torch.sparse.mm(adjacency, x)
        return x
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # width 축 절반 (2048 -> 1024)

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # (1024 -> 512)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # (512 -> 256)
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((35, 35)), # 출력 shape을 (35,35)로 맞춤
        )

        self.final_conv = nn.Conv2d(128, 1, kernel_size=(1,1))

    def forward(self, x):
        x = x.to(next(self.parameters()).device) 
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x

class Encoder5(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Encoder5, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        #print(x.shape)
        #x_a=torch.cat((x,c),dim=1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
class Decoder5(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder5, self).__init__()
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
class VAE2(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=35, latent_dim=20):
        super(VAE2, self).__init__()
        self.input=input_dim
        self.encoder = Encoder5(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder5(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1,self.input))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, graclus, max_pool, max_pool_x

class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return F.relu(x)

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.block = GraphConvBlock(in_channels, out_channels)
        self.pool = TopKPooling(out_channels, ratio=0.8)

    def forward(self, x, edge_index, batch):
        x = self.block(x, edge_index)
        x, edge_index, _, batch, perm, _ = self.pool(x, edge_index, None, batch)
        return x, edge_index, batch, perm

class GraphDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphDecoder, self).__init__()
        self.block = GraphConvBlock(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.block(x, edge_index)

class GraphVNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphVNet, self).__init__()
        self.encoder1 = GraphEncoder(in_channels, hidden_channels)
        self.encoder2 = GraphEncoder(hidden_channels, hidden_channels * 2)

        self.middle = GraphConvBlock(hidden_channels * 2, hidden_channels * 2)

        self.decoder1 = GraphDecoder(hidden_channels * 4, hidden_channels)
        self.decoder2 = GraphDecoder(hidden_channels * 2, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encoder
        x1, edge_index1, batch1, perm1 = self.encoder1(x, edge_index, batch)
        x2, edge_index2, batch2, perm2 = self.encoder2(x1, edge_index1, batch1)

        # Bottleneck
        xm = self.middle(x2, edge_index2)

        # Decoder
        x = torch.cat([xm, x2], dim=1)  # skip connection
        x = self.decoder1(x, edge_index2)

        x = torch.cat([x, x1], dim=1)   # skip connection
        x = self.decoder2(x, edge_index1)

        return F.log_softmax(x, dim=1)
class FeatureGraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.adj_weight = nn.Parameter(torch.eye(in_features))  # Learnable adjoint matrix M ∈ ℝ^{F×F}

    def forward(self, x):
        # x ∈ ℝ^{N×F}, M ∈ ℝ^{F×F}
        x = torch.matmul(x, self.adj_weight)  # Feature interaction
        x = self.linear(x)                    # Linear projection
        return F.relu(x)
class FeatureVNet(nn.Module):
    def __init__(self, in_features=35, hidden_features=64, out_features=35):
        super().__init__()
        self.enc1 = FeatureGraphConv(in_features, hidden_features)
        self.enc2 = FeatureGraphConv(hidden_features, hidden_features)

        self.middle = FeatureGraphConv(hidden_features, hidden_features)

        self.dec1 = FeatureGraphConv(hidden_features * 2, hidden_features)
        self.dec2 = FeatureGraphConv(hidden_features * 2, out_features)

    def forward(self, x):
        x1 = self.enc1(x)  # [2048, 64]
        x2 = self.enc2(x1) # [2048, 64]

        xm = self.middle(x2)

        x = torch.cat([xm, x2], dim=1)   # [2048, 128]
        x = self.dec1(x)                 # [2048, 64]

        x = torch.cat([x, x1], dim=1)    # [2048, 128]
        x = self.dec2(x)                 # [2048, 35]

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
class GraphSelfAttentionLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.1):
        super().__init__(aggr='add')  # aggregation: sum/add
        self.heads = heads
        self.out_dim = out_dim

        self.lin_query = nn.Linear(in_dim, heads * out_dim)
        self.lin_key   = nn.Linear(in_dim, heads * out_dim)
        self.lin_value = nn.Linear(in_dim, heads * out_dim)

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(heads * out_dim, out_dim)

    def forward(self, x, edge_index):
        # x: [N, in_dim]
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        Q = self.lin_query(x).view(-1, self.heads, self.out_dim)
        K = self.lin_key(x).view(-1, self.heads, self.out_dim)
        V = self.lin_value(x).view(-1, self.heads, self.out_dim)
        return self.propagate(edge_index, Q=Q, K=K, V=V, num_nodes=x.size(0))

    def message(self, Q_i, K_j, V_j, index, ptr, num_nodes):
        score = (Q_i * K_j).sum(dim=-1) / (self.out_dim ** 0.5)
        attn = softmax(score, index, num_nodes=num_nodes)
        attn = self.dropout(attn)
        return attn.unsqueeze(-1) * V_j

    def update(self, aggr_out):
        return self.final_linear(aggr_out)

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphSelfAttentionLayer(hidden_dim, hidden_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for attn_layer, norm in zip(self.layers, self.norms):
            residual = x
            #print(residual.shape)
            x = attn_layer(x, edge_index)
             
            x = norm(x + residual)
        return self.mlp(x)
    

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GraphTransformer2(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, out_dim=1, heads=4, dropout=0.1):
        super(GraphTransformer2, self).__init__()
        self.embedding = nn.Linear(1, hidden_dim)  # since x is scalar per node
        self.conv1 = TransformerConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.readout = nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x, edge_index):
        # x: [2048, 35] -> reshape for each graph
        batch_size, num_nodes = x.size()
        x = x.view(-1, 1)  # -> [2048 * 35, 1]
        x = self.embedding(x)  # -> [2048 * 35, hidden_dim]

        # Assume repeated edge_index for each sample (using batch)
        x = self.conv1(x, edge_index.repeat(1, batch_size))
        x = self.conv2(x, edge_index.repeat(1, batch_size))

        x = self.readout(x)  # -> [2048 * 35, out_dim]
        x = x.view(batch_size, num_nodes, -1)  # -> [2048, 35, out_dim]
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse
class GraphMessagePassingNet(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphMessagePassingNet, self).__init__(aggr='add')  # 'add', 'mean', or 'max'
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [num_nodes (==35), in_channels], edge_index: [2, num_edges]
        return self.propagate(edge_index, x=self.lin(x))

    def message(self, x_j):
        return x_j  # 여기서 message는 간단히 전달값으로 설정

    def update(self, aggr_out):
        return F.relu(aggr_out)  # 활성화 적용

# 3. 배치별 메시지 패싱
class BatchedGraphProcessor(nn.Module):
    def __init__(self, feature_dim):
        super(BatchedGraphProcessor, self).__init__()
        self.gmp = GraphMessagePassingNet(feature_dim, feature_dim)

    def forward(self, x, edge_index):
        # x: [batch_size, feature_dim]
        # PyG는 개별 그래프를 [num_nodes, dim]로 처리하므로 배치 분리 필요
        out = []
        for i in range(x.size(0)):
            xi = x[i]  # shape: [35]
            xi = xi.unsqueeze(0).repeat(35, 1)  # 모사: 35개 노드 모두 동일 feature
            xo = self.gmp(xi, edge_index)  # [35, 35]
            xo = xo[0]  # 중심 노드만 뽑기 (혹은 mean(xo))
            out.append(xo)
        return torch.stack(out, dim=0)  # [2048, 35]

class DenoiseDiffusion14_2(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.create=VAE2(input_dim=35)
        self.vnet=FeatureVNet(in_features=35,out_features=35,hidden_features=64)
        self.nn_g=nn.Linear(70,35,bias=False)
        self.nn_g2=nn.Linear(105,35,bias=False)
        self.msg=BatchedGraphProcessor(feature_dim=35)
        self.msg2=BatchedGraphProcessor(feature_dim=35)
        self.g_encoder=Encoder2(input_size=35,latent_size=35)
        self.g_decoder = Decoder2(input_size=35,latent_size=35,num_experts=3,output_size=35)
       # self.gt=GraphTransformer2(in_channels=1)
        #self.operator = FNO(n_modes=(1,1),hidden_channels=64,in_channels=35,out_channels=35)
        #self.decoder2 = Decoder(35,70,5,output_size)
        #self.trans =TransformerModel(in_channels=35)
        self.gnn=FeatureGCN(in_channels=35,hidden_channels=128,out_channels=35)
        self.cnn=CNNNet()
        self.unet2 = UNet(35,35)
        #self.gnn=GNNLayer(dim_in=35,dim_out=35)
        self.gnn2=GNNLayer(dim_in=35,dim_out=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0

        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)

    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)



    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def small_world(self,x,M,t,c,cur):
        #x=torch.cat((x,c),dim=-1)
        #x=self.nn_g(x)
        x=self.dct(x)
        x,mu_s,log_s=self.g_encoder(x,c)
        x=self.vnet(x)
        adjoint_Matrix = torch.randint(0, 2, (35, 35))  # Binary adjacency matrix
        edge_index, _ = dense_to_sparse(adjoint_Matrix)
        sx=self.gnn(x,M)
        sx=self.msg(sx,edge_index)
        sx2=self.gnn2(M,sx)
        sx2=self.msg2(sx2,edge_index)
        sx2=self.vnet(sx2)
      #  t_emb=self.time_embed(t)
        sx2=self.g_decoder(sx2+cur,c)
        return self.idct(sx2),mu_s,log_s
    
    def p_sample(self, xt,t,c,g,train,ground,currpose):
        eps_theta,_,_,_,_,_,_,_,_ = self.forward(xt,t,c,g,train,ground,currpose)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        if train:
            return mean + (var ** 0.5) * eps
        else:
            return mean
            #return mean
        #return val*mean+(var**0.5)
        #return eps_theta
        #return mean  
        #in this project just using the mane(they want about robust in noise then noise zero is need!)
    def forward(self, x,t,c,g,train,ground,currpose):
        #x_n=x.unsqueeze(-1)
        #x_n=x_n.unsqueeze(-1)
        #xo=self.operator(x_n)
        #xo=xo.squeeze(-1)
        #xo=xo.squeeze(-1)
        
        #print(x.shape)
        c2=g.unsqueeze(-1)
        c2=c2.unsqueeze(-1)
        c2=c2.permute(3,2,1,0)
        c2=self.cnn(c2)
        c2=c2.permute(3,2,1,0)
        c2=c2.squeeze(-1)
        c2=c2.squeeze(-1)

        M,mu2,logvar2,=self.create(c2)
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        #print(x.shape)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        #print(tt.shape)
        #mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        if train:
         
        #   c3=self.nn_g(c3)
           c3=self.dct(ground)
           c3,mu_g,log_g=self.g_encoder(c3,c)
           #c3=ground
           c3=self.vnet(c3)
           ground=torch.cat((c3,currpose),dim=-1)
           ground=self.nn_g(ground)
           adjoint_Matrix = torch.randint(0, 2, (35, 35))  # Binary adjacency matrix
           edge_index, _ = dense_to_sparse(adjoint_Matrix)
           
           x2=self.gnn(ground,M)

           x2=self.msg(x2,edge_index)
           #x2=torch.cat((x2,zc),dim=-1)
           #x2=self.nn_g2(x2)
           #x2=self.vnet(x2)
           x2=self.gnn2(M,x2)

           x2=self.msg2(x2,edge_index)
           x2=self.vnet(x2)
           #x2=self.unet(x2,t_emb)
           #adj = torch.randint(0, 2, (2048, 2048))
           #edge_index=M.nonzero(as_tuple=False).T
           
           #edge_index, edge_attr = dense_to_sparse(M) 
           #print(edge_index.shape)
           #data=Data(x=x2,edge_index=edge_index)
           x2=x2+currpose
           x2=self.g_decoder(x2,currpose)
           x2=self.idct(x2)
           
          # x2=self.gt(x2,edge_index)
        else:
            x2=x
            mu_g=x
            log_g=x
        return self.decoder(zc,c),mu,logvar,x2,mu2,logvar2,M,mu_g,log_g


class TEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden2logv = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        embedded = self.embedding(x)               # [batch, seq_len, embed_dim]
        _, h = self.rnn(embedded)                  # [1, batch, hidden]
        h = h.squeeze(0)
        mu = self.hidden2mean(h)
        log_var = self.hidden2logv(h)
        return mu, log_var


class TDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim, hidden_dim):
        super().__init__()
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.outputs2vocab = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, seq_len, input_tokens=None):
        # z: [batch, latent_dim]
        h = self.latent2hidden(z).unsqueeze(0)  # [1, batch, hidden]
        inputs = torch.zeros((z.size(0), seq_len), dtype=torch.long, device=z.device)
        if input_tokens is not None:
            inputs = input_tokens
        embedded = self.embedding(inputs)
        outputs, _ = self.rnn(embedded, h)
        logits = self.outputs2vocab(outputs)
        return logits


class TextVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.encoder = TEncoder(vocab_size, embedding_dim, hidden_dim, latent_dim)
        self.decoder = TDecoder(vocab_size, embedding_dim, latent_dim, hidden_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, seq_len):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        logits = self.decoder(z, seq_len, input_tokens=x)
        return logits, mu, log_var
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.utils.prune as prune
class DummyVAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.templates = [
            "A person is walking.",
            "A person is sitting.",
            "A person is jumping.",
            "A person is turning left.",
            "A person is raising their hand.",
        ]

    def forward(self, x):  # x: [2048, 35]
        # 실제로는 transformer decoder를 사용하여 text 생성해야 함
        batch_size = x.size(0)
        return [self.templates[i % len(self.templates)] for i in range(batch_size)]
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GPT2Tokenizer, GPT2LMHeadModel
)

try:
    import torch.nn.utils.prune as prune
    PRUNE_AVAILABLE = True
except Exception:
    PRUNE_AVAILABLE = False

# LoRA/QLoRA (optional)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class GPTConfig:
    model_id: str = "distilgpt2"   # e.g. "distilgpt2", "EleutherAI/pythia-14m"
    device: str = "cpu"                      # "cuda" or "cpu"
    max_length: int = 40
    do_sample: bool = False
    padding_side: str = "left"
    prune_ratio: Optional[float] = 0.75      # e.g. 0.75 → L1 unstructured pruning
    peft_adapter_path: Optional[str] = None  # e.g. "./lora-out" (adapter dir)
    torch_dtype: Optional[Union[str, torch.dtype]] = None  # e.g. torch.float16


# -----------------------------
# 2) Tokenizer/Model Factory
# -----------------------------
class ModelFactory:
    @staticmethod
    def load_tokenizer(model_id: str, padding_side: str = "left"):
        # AutoTokenizer가 대부분 호환됨. GPT-2 계열도 OK
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            # GPT 계열은 보통 pad_token이 없음 → eos로 세팅
            tok.pad_token = tok.eos_token
        tok.padding_side = padding_side
        return tok

    @staticmethod
    def load_model(model_id: str, device: str = "cpu",
                   torch_dtype: Optional[Union[str, torch.dtype]] = None):
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return model.to(device)


# -----------------------------
# 3) Optional Pruner
# -----------------------------
class Pruner:
    @staticmethod
    def l1_unstructured(model: torch.nn.Module, amount: float = 0.5):
        """
        모델의 Linear 레이어 weight를 L1 기준으로 비가중치 프루닝.
        amount=0.5 → 50% 연결 제거.
        """
        if not PRUNE_AVAILABLE:
            print("[Pruner] torch.nn.utils.prune 미설치. 프루닝을 건너뜁니다.")
            return model

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
                # 프루닝 마스크를 영구 반영하려면 다음 줄을 활성화:
                # prune.remove(module, "weight")
        return model


# -----------------------------
# 4) Optional PEFT Adapter Loader
# -----------------------------
class PEFTLoader:
    @staticmethod
    def attach_adapter_if_available(model: torch.nn.Module, adapter_dir: Optional[str] = None):
        if adapter_dir is None:
            return model
        if not PEFT_AVAILABLE:
            print("[PEFT] peft 미설치. 어댑터 로딩을 건너뜁니다.")
            return model
        # LoRA 어댑터 디렉토리를 붙여서 추론/학습 가능
        model = PeftModel.from_pretrained(model, adapter_dir)
        return model


# -----------------------------
# 5) Generator (추론 유틸)
# -----------------------------
class Generator:
    def __init__(self,*,tokenizer, model, max_length: int = 40, do_sample: bool = False):
        self.tok = tokenizer
        self.model = model
        self.max_length = max_length
        self.do_sample = do_sample

    @torch.inference_mode()
    def __call__(self, texts: List[str]) -> List[str]:
        inputs = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_length=self.max_length,
            pad_token_id=self.tok.pad_token_id,
            do_sample=self.do_sample
        )
        return self.tok.batch_decode(outputs, skip_special_tokens=True)

class GPTModule:
    """
    기존 API 유지: __call__(texts: List[str]) -> List[str]
    Config만 바꿔서 모델 교체/프루닝/LoRA 연결 등을 제어할 수 있음.
    """
    def __init__(self, config: Optional[GPTConfig] = None):
        self.config = config or GPTConfig()
        # 1) 토크나이저/모델 로드
        self.tokenizer = ModelFactory.load_tokenizer(
            model_id=self.config.model_id,
            padding_side=self.config.padding_side
        )
        self.model = ModelFactory.load_model(
            model_id=self.config.model_id,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype
        )
        # 2) (옵션) LoRA/QLoRA 어댑터 붙이기
        self.model = PEFTLoader.attach_adapter_if_available(
            self.model, self.config.peft_adapter_path
        )
        # 3) (옵션) 프루닝
        if self.config.prune_ratio is not None:
            self.model = Pruner.l1_unstructured(self.model, amount=self.config.prune_ratio)

        # 4) 제너레이터 준비
        self.generator = Generator(
            tokenizer=self.tokenizer,
            model=self.model,
            max_length=self.config.max_length,
            do_sample=self.config.do_sample
        )

    def __call__(self, texts: List[str]) -> List[str]:
        return self.generator(texts)
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
class GPTModule2:
    def __init__(self, device="cuda", max_input_len=40, max_new_tokens=24,
                 do_sample=False, temperature=1.0, top_p=0.9,
                 enforce_token_format=True):
        """
        - max_input_len: 인풋을 이 길이로 잘라서 넘깁니다 (context blowup 방지)
        - max_new_tokens: 새로 생성할 토큰 개수 (max_length 대신 사용)
        - do_sample=False로 두면 결정론적(그리디); True면 샘플링
        - enforce_token_format: M2T: <Tddd> 패턴만 남기고 정리
        """
        use_cuda = torch.cuda.is_available() and (device == "cuda")
        self.device = "cuda" if use_cuda else "cpu"

        # 토크나이저/모델
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        # pad_token이 없는 모델이라 eos를 pad로 지정
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(self.device)
        self.model.eval()

        # 설정값
        self.max_input_len = max_input_len
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.enforce_token_format = enforce_token_format

        # M2T 토큰 정규식
        self._token_re = re.compile(r"<T(\d{3})>")

    @torch.no_grad()
    def __call__(self, texts):
        """
        texts: list[str] (예: ["M2T: <T042> <T155> ...", ...])
        returns: list[str] (GPT 후처리된 문장)
        """
        # 인풋 길이 제어: 너무 긴 프롬프트는 마지막 max_input_len 토큰만 사용
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_len
        ).to(self.device)

        gen_kwargs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            gen_kwargs.update(dict(temperature=self.temperature, top_p=self.top_p))

        outputs = self.model.generate(**gen_kwargs)

        # 원문 + 신생성 토큰 전체를 디코딩해도 되지만,
        # 아래처럼 '생성분만' 추출해서 후처리하면 더 깔끔합니다.
        # (원본 길이 = 각 배치의 input_ids 길이)
        decoded = []
        for i in range(outputs.size(0)):
            input_len_i = enc["input_ids"][i].ne(self.tokenizer.pad_token_id).sum().item()
            # 생성 토큰만 떼기
            new_tokens_i = outputs[i, input_len_i:]
            text_i = self.tokenizer.decode(new_tokens_i, skip_special_tokens=True)

            if self.enforce_token_format:
                # "M2T: " 프리픽스 유지 + <Tddd> 토큰만 남기기
                toks = self._token_re.findall(text_i)
                toks = [f"<T{int(t):03d}>" for t in toks]
                text_i = "M2T: " + " ".join(toks) if toks else "M2T: <T000>"

            decoded.append(text_i)

        return decoded
import requests, os, time
from typing import List
import os, time, requests
from typing import List
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# -------------------------
# 1) Dummy VAE Decoder (vector -> text)
#    실제로는 여러분의 VAE 텍스트 디코더로 교체하세요.
# -------------------------
class DummyVAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.templates = [
            "The person is walking forward.",
            "The person is sitting down.",
            "The person is turning to the left.",
            "The person is raising the right arm.",
            "The person is jumping lightly."
        ]

    def forward(self, x):  # x: [batch, 35]
        b = x.size(0)
        return [self.templates[i % len(self.templates)] for i in range(b)]

class APIGPTModule_HF:
    """
    - endpoint_url: Hugging Face Inference API 모델 엔드포인트
    - hf_token: 없으면 환경변수 HF_TOKEN을 자동 사용
    - batch_size: 2048개 텍스트를 나눠서 호출
    """
    def __init__(self,
                 endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
                 hf_token=None,
                 batch_size=16,
                 max_new_tokens=64,
                 temperature=0.0,
                 system_prompt=None):
        self.url = endpoint_url
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

    def _build_input(self, user_text: str) -> str:
        # 간단한 프롬프트 래핑 (원하면 system_prompt 포함)
        if self.system_prompt:
            return f"{self.system_prompt}\n\nUser: {user_text}\nAssistant:"
        return user_text

    def _call_once(self, texts: List[str]) -> List[str]:
        outs = []
        for t in texts:
            payload = {
                "inputs": self._build_input(t),
                "parameters": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "return_full_text": False
                }
            }
            r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            # 쿼터/대기 등으로 503 가능 → 재시도는 바깥에서
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                outs.append(data[0]["generated_text"].strip())
            else:
                outs.append(t)  # 실패 시 원문 유지
            time.sleep(0.02)  # rate limit 방지용 소량 대기
        return outs

    def __call__(self, texts: List[str]) -> List[str]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i+self.batch_size]
            for attempt in range(3):
                try:
                    results.extend(self._call_once(chunk))
                    break
                except Exception:
                    # 간단한 지수적 백오프
                    if attempt == 2:
                        results.extend(chunk)  # 마지막 실패 시 원문 통과
                    else:
                        time.sleep(1.0 * (attempt + 1))
        return results


class TextEncoder(nn.Module):
    def __init__(self, output_dim=35, device="cuda"):
        super().__init__()
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        self.linear = nn.Linear(384, output_dim)
        self.device = device

    def forward(self, text_batch):
        with torch.no_grad():
            embeddings = self.model.encode(text_batch, convert_to_tensor=True, device=self.device)
        return self.linear(embeddings)  # [2048, 35]
class FullPipeline:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.vae_decoder = DummyVAEDecoder2(device=self.device).to(self.device)
        # 온라인 LLM: 학습 없음, 호출만 함
        self.gpt = GPTModule2()
        self.text_encoder = TextEncoder2(device=self.device).to(self.device)

    def __call__(self, x: torch.Tensor):
        # x: [batch, 35]
        texts = self.vae_decoder(x)            # list[str], 길이=batch
        gpt_texts = self.gpt(texts)            # list[str]
        x_recon = self.text_encoder(gpt_texts) # [batch, 35]
        return x_recon, texts, gpt_texts

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DummyVAEDecoder2(nn.Module):
    """
    Motion(35-dim) -> Text(list[str]) 변환기
    - codebook_size: 토큰 종류 수 (예: 256개 -> <T000> ~ <T255>)
    - code_len: 문장 길이(토큰 수). 길이가 길수록 표현력↑
    - mode='argmax'이면 결정론적, 'sample'이면 Gumbel-sample 기반 확률적 생성
    """
    def __init__(
        self,
        dim_in: int = 35,
        codebook_size: int = 256,
        code_len: int = 8,
        temperature: float = 1.0,
        mode: str = "argmax",  # 'argmax' | 'sample'
        device: str = "cpu",
    ):
        super().__init__()
        self.dim_in = dim_in
        self.codebook_size = codebook_size
        self.code_len = code_len
        self.temperature = temperature
        self.mode = mode
        self.device = device

        # 간단한 "인코더": 연속 35D -> (code_len * codebook_size) 로짓
        # 필요 시 더 깊게 쌓아도 됨.
        hidden = max(64, dim_in * 2)
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, code_len * codebook_size),
        )

        # 코드북(토큰 임베딩). 여기서는 텍스트를 만들 때 직접 쓰진 않지만
        # 학습을 할 때는 codebook을 정규화/참조하는 식으로 확장 가능.
        self.codebook = nn.Parameter(torch.randn(codebook_size, 32) * 0.02)

        # 프리픽스/서픽스 등 문장 스타일
        self.prefix = "M2T:"
        self.sep = " "
        self.suffix = ""

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[str]:
        """
        x: [batch, 35]
        returns: list[str] 길이=batch (각 원소는 공백으로 구분된 토큰 시퀀스 문장)
        """
        assert x.dim() == 2 and x.size(1) == self.dim_in, \
            f"Expected [B,{self.dim_in}], got {tuple(x.shape)}"

        x = x.to(next(self.parameters()).device)
        logits = self.encoder(x)                             # [B, code_len*codebook_size]
        logits = logits.view(-1, self.code_len, self.codebook_size)  # [B, L, K]

        if self.mode == "sample":
            # Gumbel sampling
            gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
            sampled = torch.argmax((logits + gumbel)/max(1e-6, self.temperature), dim=-1)  # [B, L]
            token_ids = sampled
        else:
            # deterministic
            token_ids = torch.argmax(logits, dim=-1)  # [B, L]

        texts: List[str] = []
        for row in token_ids.tolist():
            tokens = [f"<T{tid:03d}>" for tid in row]
            sentence = self.prefix + self.sep + self.sep.join(tokens) + self.suffix
            texts.append(sentence)
        return texts

    # --- (선택) 학습용 유틸: 토큰 ID로부터 임베딩/로스 계산 등에 사용 가능 ---
    def ids_to_embeddings(self, ids: torch.LongTensor) -> torch.Tensor:
        """
        ids: [B, L]  -> returns [B, L, D]
        """
        return F.embedding(ids, self.codebook)

    def set_sampling(self, mode: str = "sample", temperature: float = 1.0):
        assert mode in ("sample", "argmax")
        self.mode = mode
        self.temperature = temperature


# (선택) Text2Motion Decoder 예시
# 이미 프로젝트에 TextEncoder가 있다면 아래는 건너뛰셔도 됩니다.
class TextEncoder2(nn.Module):
    """
    list[str] -> [batch, 35]
    - 단순 파서: "<T###>" 토큰을 숫자 ID로 파싱 -> 임베딩 평균 -> 35D로 사상
    - 프로젝트에 이미 같은 이름의 클래스가 있다면 이 파트는 사용하지 마세요.
    """
    def __init__(self, device="cpu", codebook_size: int = 256, dim_out: int = 35, emb_dim: int = 64):
        super().__init__()
        self.device = device
        self.codebook_size = codebook_size

        self.token_embed = nn.Embedding(codebook_size, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, dim_out),
        )

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        texts: ["M2T: <T042> <T155> ...", ...]
        returns: [B, 35]
        """
        token_id_batches = []
        for s in texts:
            # 간단한 파싱: "<Tddd>" 패턴만 수집
            ids = []
            for tok in s.replace("M2T:", " ").split():
                if tok.startswith("<T") and tok.endswith(">"):
                    try:
                        idx = int(tok[2:-1])
                        if 0 <= idx < self.codebook_size:
                            ids.append(idx)
                    except:
                        pass
            if len(ids) == 0:
                ids = [0]
            token_id_batches.append(torch.tensor(ids, dtype=torch.long))

        # 가변 길이 -> 패딩 없이 각 문장 평균 임베딩
        embs = []
        for ids in token_id_batches:
            ids = ids.to(next(self.parameters()).device)
            e = self.token_embed(ids)          # [L, D]
            embs.append(e.mean(dim=0))         # [D]
        embs = torch.stack(embs, dim=0)        # [B, D]
        out = self.head(embs)                  # [B, 35]
        return out
class GPTModule3:
    def __init__(self, device="cuda"):
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.tokenizer.padding_side = "left"  # ✅ 추가
        self.tokenizer.pad_token = self.tokenizer.eos_token  # ✅ 필수
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
        self.device = device

    def __call__(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=40,
                pad_token_id=self.tokenizer.pad_token_id,  # ✅ 명시적으로 설정
                do_sample=False
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
class FullPipeline2:
    def __init__(self, device="cuda"):
        self.vae_decoder = DummyVAEDecoder()
        self.gpt = GPTModule3(device=device)
        self.text_encoder = TextEncoder(device=device).to(device)

    def __call__(self, x):  # x: [2048, 35]
        text_batch = self.vae_decoder(x)                         # → list of 2048 texts
        gpt_outputs = self.gpt(text_batch)                       # → transformed texts
        x_recon = self.text_encoder(gpt_outputs)                 # → [2048, 35]
        return x_recon, text_batch, gpt_outputs
# (선택) 전체 VAE 래퍼: encode()는 Motion->Text, dec
class DenoiseDiffusion14_3(nn.Module):
    def __init__(self, input_size, noise_steps, latent_size, output_size):
        super().__init__()
        self.timesteps = noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder = Encoder(input_size, 35)
        self.decoder = Decoder(35, 70, 5, output_size)
        self.create = VAE2(input_dim=35)
        self.vnet = FeatureVNet(in_features=35, out_features=35, hidden_features=64)
        self.nn_g = nn.Linear(70, 35, bias=False)
        self.nn_g2 = nn.Linear(105, 35, bias=False)
        self.msg = BatchedGraphProcessor(feature_dim=35)
        self.msg2 = BatchedGraphProcessor(feature_dim=35)
        self.g_encoder = Encoder2(input_size=35, latent_size=35)
        self.g_decoder = Decoder2(input_size=35, latent_size=35, num_experts=3, output_size=35)
        #self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to('cpu')
        #self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        self.world = FullPipeline2()
       
       # self.gt=GraphTransformer2(in_channels=1)
        #self.operator = FNO(n_modes=(1,1),hidden_channels=64,in_channels=35,out_channels=35)
        #self.decoder2 = Decoder(35,70,5,output_size)
        #self.trans =TransformerModel(in_channels=35)
        self.gnn = FeatureGCN(in_channels=35, hidden_channels=128, out_channels=35)
        self.cnn = CNNNet()
        self.unet2 = UNet(35, 35)
        #self.gnn=GNNLayer(dim_in=35,dim_out=35)
        self.gnn2 = GNNLayer(dim_in=35, dim_out=35)
        self.time_embed = TimeEmbedding(70)
        self.beta = self.gaussiandiffusion.beta
        self.mean = 0
        self.logvar = 0

        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)
    def run_gpt(self,texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = self.gpt_model.generate(**inputs, max_length=50)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def dct(self, x):
        # Apply DCT transformation
        self.dct_basis = self.dct_basis.to(x.device)
        return torch.matmul(x, self.dct_basis)

    def idct(self, x):
        # Apply IDCT transformation
        self.idct_basis = self.idct_basis.to(x.device)
        return torch.matmul(x, self.idct_basis)



    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def small_world(self, x, M, t, c, cur):
        #x=torch.cat((x,c),dim=-1)
        #x=self.nn_g(x)
        x = self.dct(x)
        x, mu_s, log_s = self.g_encoder(x, c)
       
      #  t_emb=self.time_embed(t)
        sx2 = self.g_decoder(x + cur, c)
        sx2, _, _ = self.world(sx2)
        return self.idct(sx2), mu_s, log_s
    
    def p_sample(self, xt,t,c,g,train,ground,currpose):
        eps_theta,_,_,_,_,_,_,_,_ = self.forward(xt,t,c,g,train,ground,currpose)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        if train:
            return mean + (var ** 0.5) * eps
        else:
            return mean
            #return mean
        #return val*mean+(var**0.5)
        #return eps_theta
        #return mean  
        #in this project just using the mane(they want about robust in noise then noise zero is need!)
    def forward(self, x, t, c, g, train, ground, currpose):
        #x_n=x.unsqueeze(-1)
        #x_n=x_n.unsqueeze(-1)
        #xo=self.operator(x_n)
        #xo=xo.squeeze(-1)
        #xo=xo.squeeze(-1)
        
        #print(x.shape)
        c2 = g.unsqueeze(-1)
        c2 = c2.unsqueeze(-1)
        c2 = c2.permute(3, 2, 1, 0)
        c2 = self.cnn(c2)
        c2 = c2.permute(3, 2, 1, 0)
        c2 = c2.squeeze(-1)
        c2 = c2.squeeze(-1)

        M, mu2, logvar2 = self.create(c2)
        z, mu, logvar = self.encoder(x, c)
        t_emb = self.time_embed(t)
        x = self.unet(z, t_emb).to(device="cuda")
        #print(x.shape)
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cuda")
        #print(tt.shape)
        #mean,var=self.q_xt_x0(x,tt)
        zc = torch.cat((x, z), dim=-1)
        if train:
         
        #   c3=self.nn_g(c3)
           c3 = self.dct(ground)
           c3, mu_g, log_g = self.g_encoder(c3, ground)
           c2 = self.g_decoder(c3, ground)
           x2, _, _ = self.world(c3)
           x2 = self.idct(x2)
           
          # x2=self.gt(x2,edge_index)
        else:
            x2 = x
            mu_g = x
            log_g = x
        return self.decoder(zc, c), mu, logvar, x2, mu2, logvar2, M, mu_g, log_g
    

class DenoiseDiffusion14stgcn(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.create=VAE2(input_dim=35)
        self.vnet=FeatureVNet(in_features=35,out_features=35,hidden_features=64)
        self.nn_g=nn.Linear(70,35,bias=False)
        self.nn_g2=nn.Linear(105,35,bias=False)
        self.msg=BatchedGraphProcessor(feature_dim=35)
        self.msg2=BatchedGraphProcessor(feature_dim=35)
        self.g_encoder=Encoder2(input_size=35,latent_size=35)
        self.g_decoder = Decoder2(input_size=35,latent_size=35,num_experts=3,output_size=35)
       # self.gt=GraphTransformer2(in_channels=1)
        #self.operator = FNO(n_modes=(1,1),hidden_channels=64,in_channels=35,out_channels=35)
        #self.decoder2 = Decoder(35,70,5,output_size)
        #self.trans =TransformerModel(in_channels=35)
        self.gnn=FeatureGCN(in_channels=35,hidden_channels=128,out_channels=35)
        self.cnn=CNNNet()
        self.unet2 = UNet(35,35)
        #self.gnn=GNNLayer(dim_in=35,dim_out=35)
        self.gnn2=GNNLayer(dim_in=35,dim_out=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.cross_fusion=BipartiteFusion(70)
        self.l1=nn.Linear(70,35)
        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)

    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)



    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def small_world(self,x,M,t,c,cur):
        #x=torch.cat((x,c),dim=-1)
        #x=self.nn_g(x)
        x=self.dct(x)
        x,mu_s,log_s=self.g_encoder(x,c)
        x=self.vnet(x)
        adjoint_Matrix = torch.randint(0, 2, (35, 35))  # Binary adjacency matrix
        edge_index, _ = dense_to_sparse(adjoint_Matrix)
        sx=self.gnn(x,M)
        sx=self.msg(sx,edge_index)
        sx2=self.gnn2(M,sx)
        sx2=self.msg2(sx2,edge_index)
        sx2=self.vnet(sx2)
      #  t_emb=self.time_embed(t)
        sx2=self.g_decoder(sx2+cur,c)
        return self.idct(sx2),mu_s,log_s
    
    def p_sample(self, xt,t,c,g,train,ground,currpose):
        eps_theta,_,_,_,_,_,_,_,_ = self.forward(xt,t,c,g,train,ground,currpose)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        if train:
            return mean + (var ** 0.5) * eps
        else:
            return mean
            #return mean
        #return val*mean+(var**0.5)
        #return eps_theta
        #return mean  
        #in this project just using the mane(they want about robust in noise then noise zero is need!)
    def forward(self, x,t,c,g,train,ground,currpose):
        #x_n=x.unsqueeze(-1)
        #x_n=x_n.unsqueeze(-1)
        #xo=self.operator(x_n)
        #xo=xo.squeeze(-1)
        #xo=xo.squeeze(-1)
        
        #print(x.shape)
        c2=g.unsqueeze(-1)
        c2=c2.unsqueeze(-1)
        c2=c2.permute(3,2,1,0)
        c2=self.cnn(c2)
        c2=c2.permute(3,2,1,0)
        c2=c2.squeeze(-1)
        c2=c2.squeeze(-1)

        M,mu2,logvar2,=self.create(c2)
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        #print(x.shape)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        #print(tt.shape)
        #mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        if train:
         
        #   c3=self.nn_g(c3)
           c3=self.dct(ground)
           c3,mu_g,log_g=self.g_encoder(c3,c)
           #c3=ground
           c3=self.vnet(c3)
           ground=torch.cat((c3,currpose),dim=-1)
           ground=self.nn_g(ground)
           adjoint_Matrix = torch.randint(0, 2, (35, 35))  # Binary adjacency matrix
           edge_index, _ = dense_to_sparse(adjoint_Matrix)
           
           x2=self.gnn(ground,M)

           x2=self.msg(x2,edge_index)
           #x2=torch.cat((x2,zc),dim=-1)
           #x2=self.nn_g2(x2)
           #x2=self.vnet(x2)
           x2=self.gnn2(M,x2)

           x2=self.msg2(x2,edge_index)
           x2=self.vnet(x2)
           #x2=self.unet(x2,t_emb)
           #adj = torch.randint(0, 2, (2048, 2048))
           #edge_index=M.nonzero(as_tuple=False).T
           
           #edge_index, edge_attr = dense_to_sparse(M) 
           #print(edge_index.shape)
           #data=Data(x=x2,edge_index=edge_index)
           x2=x2+currpose
           x2=self.g_decoder(x2,currpose)
           x2=self.idct(x2)
           
          # x2=self.gt(x2,edge_index)
           x3=self.decoder(zc,c)
           fused=self.cross_fusion(x2,x3)
           
           return self.l1(fused),mu,logvar,x2,mu2,logvar2,M,mu_g,log_g
        else:
            x2=x
            mu_g=x
            log_g=x
            return self.decoder(zc,c),mu,logvar,x2,mu2,logvar2,M,mu_g,log_g
from scipy.fftpack import dct
# ---------- Anchor Sampler ----------
class AnchorSampler(nn.Module):
    def __init__(self, num_spatial, num_temporal, latent_dim):
        super().__init__()
        self.spatial = nn.Parameter(torch.randn(num_spatial, latent_dim))
        self.temporal = nn.Parameter(torch.randn(num_temporal, latent_dim))
        self.latent_dim = latent_dim

    def forward(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim).to(self.spatial.device)  # noise
        anchors = []
        for s in self.spatial:
            for t in self.temporal:
                anchors.append(s + t)
        anchors = torch.stack(anchors)  # [K, latent_dim]
        return anchors, z  # z: [B, latent_dim]


# ---------- STGCN Layer ----------
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, V, F):
        super().__init__()
        self.V = V
        self.F = F
        self.linear = nn.Linear(in_channels, out_channels)

        self.A_spatial = nn.Parameter(torch.eye(V))    # shape: [V, V]
        self.A_temporal = nn.Parameter(torch.eye(F))   # shape: [F, F]

    def forward(self, x):  # x: [B, F, V, C]
        B, F, V, C = x.shape
        x = torch.einsum("bfvc,vu->bfuc", x, self.A_spatial)  # spatial
        x = torch.einsum("bfuc,ff->bfuc", x, self.A_temporal)  # temporal
        x = self.linear(x)
        return x


# ---------- IE-STGCN ----------
class IE_STGCN(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, num_layers=4, V=1, F=10):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            self.layers.append(STGCNLayer(in_ch, hidden_dim, V, F))
        self.out_proj = nn.Linear(hidden_dim, 3)

    def forward(self, x):  # x: [B, F, V, C]
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x)  # [B, F, V, 3]


# ---------- STARS + IE-STGCN ----------
class STARSModel(nn.Module):
    def __init__(self, num_spatial=5, num_temporal=6, latent_dim=128, V=1, F=10):
        super().__init__()
        self.anchor_sampler = AnchorSampler(num_spatial, num_temporal, latent_dim)
        self.encoder = nn.Linear(30, latent_dim)  # input [B, 30] → latent
        self.backbone = IE_STGCN(in_channels=latent_dim, V=V, F=F)
        self.CLayer=nn.Linear(35,30)
        self.DLayer=nn.Linear(900,35)
        self.K = num_spatial * num_temporal
        self.V = V
        self.F = F
    def create_dct_matrix(self, N):
        dct_mat = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_mat[k, n] = 1.0 / torch.sqrt(torch.tensor(N, dtype=torch.float32))
                else:
                    angle = torch.tensor(torch.pi * (n + 0.5) * k / N, dtype=torch.float32)
                    dct_mat[k, n] = (torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(angle))
        return dct_mat
    def dct_torch(self,x, dim=1):
        N = x.shape[dim]
        dct_mat = self.create_dct_matrix(N)
        x_perm = x.transpose(dim, -1)  # bring time to last
        shape = x_perm.shape
        x_flat = x_perm.reshape(-1, N)  # [..., T]
        x_dct = torch.matmul(x_flat, dct_mat.T)
        x_dct = x_dct.reshape(*shape).transpose(dim, -1)
        return x_dct
    def forward(self, x,train):  # x: [B, 30]
        x=self.CLayer(x)
        B = x.size(0)

        # [B, 30] → [B, 10, 1, 3]  (T_h=10, V=1, C=3)
        x = x.view(B, 10, self.V, 3)

        # DCT 변환: 시간축 (dim=1) 기준
        x_freq = self.dct_torch(x,dim=1)  # [B, F=10, V=1, C=3]

        # 인코딩
        x_encoded = self.encoder(x.view(B, -1)).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, latent]
        x_encoded = x_encoded.expand(-1, self.F, self.V, -1)  # [B, F, V, latent]

        # anchor + noise
        anchors, z = self.anchor_sampler(B)  # [K, latent], [B, latent]
        outputs = []

        for anchor in anchors:  # for each anchor
            combined = x_encoded + anchor.view(1, 1, 1, -1)  # broadcast: [B, F, V, latent]
            y_pred = self.backbone(combined)  # [B, F, V, 3]
            outputs.append(y_pred)

        t=torch.stack(outputs, dim=0)  # [K, B, F, V, 3]
        t=t.permute(1,0,2,3,4) # [B, K, F, V, C]
        if train:
            gt=t
            t=t.reshape(2048,-1)
            
        else:
            t=t.reshape(1,-1)
        return gt,self.DLayer(t)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Encoder9(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)

    def forward(self, x):
        x = self.fc(x)  # [B, Tp, D]
        x = self.pe(x)
        return x

class SpatialAnchorEncoder(nn.Module):
    def __init__(self, pose_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(pose_dim, hidden_dim)

    def forward(self, x):  # x: [B, J×3] → reference pose
        return self.fc(x)  # [B, D]

class TemporalAnchorEncoder(nn.Module):
    def __init__(self, seq_len, pose_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=pose_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):  # x: [B, T, pose_dim]
        _, h = self.gru(x)
        return h.squeeze(0)  # [B, D]

class MotionSampler(nn.Module):
    def __init__(self, context_dim, z_dim):
        super().__init__()
        self.fc_mu = nn.Linear(context_dim, z_dim)
        self.fc_logvar = nn.Linear(context_dim, z_dim)

    def forward(self, context):
        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class Decoder9(nn.Module):
    def __init__(self, z_dim, context_dim, output_dim, T):
        super().__init__()
        self.output_dim=output_dim
        self.fc1 = nn.Linear(z_dim + context_dim, 256)
        self.fc2 = nn.Linear(256, T * output_dim)

    def forward(self, z, context):
        x = torch.cat([z, context], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), -1, self.output_dim)  # [B, T, output_dim]

class STARS(nn.Module):
    def __init__(self, Tp, Tf, pose_dim, hidden_dim=128, z_dim=35):
        super().__init__()
        self.encoder = Encoder9(pose_dim, hidden_dim)
        self.spatial_anchor = SpatialAnchorEncoder(pose_dim, hidden_dim)
        self.temporal_anchor = TemporalAnchorEncoder(Tp, pose_dim, hidden_dim)
        self.sampler = MotionSampler(context_dim=3*hidden_dim, z_dim=z_dim)
        self.decoder = Decoder9(z_dim, context_dim=3*hidden_dim, output_dim=pose_dim, T=Tf)
        self.l1=nn.Linear(70,35)
    def forward(self, x,train=True):
        # x: [B, Tp, pose_dim]
        context_enc = self.encoder(x)
        sa = self.spatial_anchor(x[:, -1])           # [B, D]
        ta = self.temporal_anchor(x)                # [B, D]
        context_vec = torch.cat([context_enc[:, -1], sa, ta], dim=-1)  # [B, 3D]

        z, mu, logvar = self.sampler(context_vec)   # VAE-style sampling
        out = self.decoder(z, context_vec) 
        if train: 
            out=out.reshape(2048,70)        # [B, Tf, pose_dim]
        else:
            out=out.reshape(1,70)
        return self.l1(out), mu, logvar
    
    def denormalize(self, t,mu,logvar):
        return t * mu + logvar


import torch
import torch.nn as nn
from torch_dct import dct, idct

class GNNLayer2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer2, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        # x: [N, F], edge_index: [2, E]
        row, col = edge_index
        x_j = x[col]  # neighbor features
        messages = self.linear(x_j)

        # aggregate messages for each node (sum)
        agg_messages = torch.zeros_like(x)
        agg_messages.index_add_(0, row, messages)

        # update
        out = F.relu(agg_messages)
        return out
class GCNLayerAdj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayerAdj, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):  # x: [N, F], adj: [N, N]
        agg = torch.matmul(x,adj)  # message passing
        out = self.linear(agg)
        return F.relu(out)
class MessagePassingGNNAdj(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MessagePassingGNNAdj, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(GCNLayerAdj(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):  # adj: [N, N]
        for layer in self.layers:
            x = layer(x, adj)
        return x
from torch_dct import dct, idct
class DCT1D(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT1D, self).__init__()
        self.norm = norm

    def forward(self, x):  # x: [B, L]
        N = x.size(-1)
        v = torch.cat([x, x.flip(dims=[-1])], dim=-1)  # even extension
        Vc = torch.fft.fft(v, dim=-1)
        D = Vc[..., :N].real  # only the first half is used
        if self.norm == 'ortho':
            D[..., 0] = D[..., 0] / math.sqrt(N)
            D[..., 1:] = D[..., 1:] * math.sqrt(2 / N)
        return D
class IDCT1D(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT1D, self).__init__()
        self.norm = norm

    def forward(self, X):  # X: [B, L]
        N = X.size(-1)
        X = X.clone()
        if self.norm == 'ortho':
            X[..., 0] = X[..., 0] * math.sqrt(N)
            X[..., 1:] = X[..., 1:] / math.sqrt(2 / N)

        V = torch.zeros(X.shape[:-1] + (2 * N,), dtype=torch.complex64, device=X.device)
        V[..., :N] = X
        V[..., N:] = torch.flip(X, dims=[-1])
        v = torch.fft.ifft(V, dim=-1).real
        return v[..., :N]
class SPGSN2(nn.Module):
    def __init__(self,input_dim,dct=30,hidden_dim=256):
        super(SPGSN2,self).__init__()
        self.l1=nn.Linear(35,30)
        self.spliter1=GNNLayer(input_dim,input_dim)
        self.spliter2=GNNLayer(15,15)
        self.spliter3=GNNLayer(15,15)
        self.vae1=VAE2(input_dim=30)
        self.vae2=VAE2(input_dim=15)
        self.vae3=VAE2(input_dim=15)
        self.vae4=VAE2(input_dim=60)
        self.message=MessagePassingGNNAdj(input_dim=60,hidden_dim=60,output_dim=60,num_layers=3)
        self.l2 = nn.Linear(60,35)
        self.dct=DCT1D()
        self.idct=IDCT1D()
    def forward(self,x):
        x=self.l1(x)
        x = self.dct(x)
        I1 = torch.eye(30)
        I2 = torch.eye(15)
        I3 = torch.eye(60)
        m1,mu1,logvar1=self.vae1(I1)
        m2,mu2,logvar2 = self.vae2(I2)
        m3,mu3,logvar3=self.vae3(I2)
        m4,mu4,logvar4 = self.vae4(I3)
        feat=x
        x2,x3=torch.split(x,15,dim=-1)
        feat1=self.spliter1(m1,feat)
        feat2=self.spliter2(m2,x2)
        feat3=self.spliter3(m3,x3)
        feat_m=torch.cat((feat1,feat2),dim=-1)
        feat_m=torch.cat((feat_m,feat3),dim=-1)
        feat_m=self.message(feat_m,m4)
        feat_m=self.idct(feat_m)
        feat_m=self.l2(feat_m)
        return feat_m,mu1,logvar1,mu2,logvar2,mu3,logvar3,mu4,logvar4


class SPGSN(nn.Module):
    def __init__(self, input_dim, dct_channels, num_mpgsb=1, hidden_dim=256, K=2, L=2):
        super(SPGSN, self).__init__()
        self.l1=nn.Linear(35,30)
        self.dct_channels = dct_channels
        self.num_joints = input_dim
        self.input_dim = input_dim 

        self.mpgsb_blocks = nn.ModuleList([
            MPGSB(self.input_dim, dct_channels, hidden_dim, K=K, L=L)
            for _ in range(num_mpgsb)
        ])

        self.output_proj = nn.Linear(dct_channels, dct_channels)
        self.l2=nn.Linear(30,35)
        self.l3=nn.Linear(30,15)
    def forward(self, x):  # x: [B, T, J, 3]
        x=self.l1(x)
        x=x.unsqueeze(1)
        feat = x
        x1, x2 = torch.split(x, 15, dim=-1)  # 또는 dim=2
        for mpgsb in self.mpgsb_blocks:
            feat = feat + mpgsb(feat,x1,x2)
        out = self.output_proj(feat)
        out=out.squeeze(1)  # [B, D]
        return self.l2(out)

class MPGSB(nn.Module):
    def __init__(self, input_dim, dct_channels, hidden_dim, K=2, L=2):
        super(MPGSB, self).__init__()
        self.scattering_upper = AdaptiveGraphScattering(input_dim//2, dct_channels, hidden_dim, K, L)
        self.scattering_lower = AdaptiveGraphScattering(input_dim//2, dct_channels, hidden_dim, K, L)
        #self.scattering_whole = AdaptiveGraphScattering2(input_dim, dct_channels, hidden_dim, K, L)
        self.n1=nn.Linear(60,30)
        self.n2= nn.Linear(128,15)
        self.n3= nn.Linear(128,15)
        #self.n4= nn .Linear(128,30)
        self.n5 = nn.Linear(60,30)
        self.cross_fusion = BipartiteFusion(dct_channels)
        self.cross_fusion2 = BipartiteFusion(dct_channels*2)
    def forward(self, x,y,z):
        upper_feat = self.scattering_upper(y)
        lower_feat = self.scattering_lower(z)
        #whole_feat = self.scattering_whole(x)
        upper_feat=self.n2(upper_feat)
        lower_feat=self.n3(lower_feat)
        #whole_feat=self.n4(whole_feat)
        fused=self.cross_fusion(upper_feat, lower_feat)
        #print(fused.shape)
        #whole=torch.cat([fused,x],dim=1)
        #print(x.shape)
        #xt=x.reshape(2048,30)
        #fused=self.cross_fusion2(fused,xt)
       # print(whole.shape)
       # print(fused.shape)
        return fused
class AdaptiveGraphScattering(nn.Module):
    def __init__(self, input_dim, dct_channels, hidden_dim, K, L):
        super(AdaptiveGraphScattering, self).__init__()
        self.K = K
        self.L = L
        self.hidden_dim = hidden_dim
        
        self.adj = nn.Parameter(torch.eye(input_dim))
        self.weights = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range((K+1)**L)
        ])

        self.f1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, C, D]
        B, C, D = x.shape
        #print(self.adj.shape)
        A = 0.5 * (self.adj + torch.eye(D, device=x.device))  # normalized
        #print(x.shape)
        H = []
        x=x.reshape(2048,15)
        for k in range(self.K + 1):
           # print(A.matrix_power(2*k+1).shape)
            h_k = torch.matmul(x,A.matrix_power(2*k+1))
           
            H.append(torch.tanh(self.weights[k](h_k)))

        # Spectrum attention aggregation
        H_cat = torch.stack(H, dim=1)  # [B, K+1, D, C']
        H_sum = H_cat.sum(dim=1)
        H_sp = torch.relu(H_sum / (self.K + 1))

        scores = []
        for i in range(self.K + 1):
            h_concat = torch.cat([H_sp, H_cat[:, i]], dim=-1)
            s = self.f2(torch.tanh(self.f1(h_concat)))
            scores.append(s)

        scores = torch.softmax(torch.stack(scores, dim=1), dim=1)  # [B, K+1, D, 1]
        H_agg = (H_cat * scores).sum(dim=1)  # [B, D, C']
        return H_agg
class AdaptiveGraphScattering2(nn.Module):
    def __init__(self, input_dim, dct_channels, hidden_dim, K, L):
        super(AdaptiveGraphScattering2, self).__init__()
        self.K = K
        self.L = L
        self.hidden_dim = hidden_dim
        
        self.adj = nn.Parameter(torch.eye(input_dim))
        self.weights = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range((K+1)**L)
        ])

        self.f1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, C, D]
        B, C, D = x.shape
        #print(self.adj.shape)
        A = 0.5 * (self.adj + torch.eye(D, device=x.device))  # normalized
        #print(x.shape)
        H = []
        print(x.shape)
        x=x.reshape(2048,30)
        for k in range(self.K + 1):
           # print(A.matrix_power(2*k+1).shape)
            h_k = torch.matmul(x,A.matrix_power(2*k+1))
           
            H.append(torch.tanh(self.weights[k](h_k)))

        # Spectrum attention aggregation
        H_cat = torch.stack(H, dim=1)  # [B, K+1, D, C']
        H_sum = H_cat.sum(dim=1)
        H_sp = torch.relu(H_sum / (self.K + 1))

        scores = []
        for i in range(self.K + 1):
            h_concat = torch.cat([H_sp, H_cat[:, i]], dim=-1)
            s = self.f2(torch.tanh(self.f1(h_concat)))
            scores.append(s)

        scores = torch.softmax(torch.stack(scores, dim=1), dim=1)  # [B, K+1, D, 1]
        H_agg = (H_cat * scores).sum(dim=1)  # [B, D, C']
        return H_agg
class BipartiteFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(BipartiteFusion, self).__init__()
        self.f_upper = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.f_lower = nn.Linear(hidden_dim//2, hidden_dim//2)
        #self.f_whole = nn.Linear(hidden_dim,hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_upper, h_lower):
        A_u2l = torch.softmax(self.f_upper(h_upper) @ self.f_lower(h_lower).transpose(-1, -2), dim=-1)
        A_l2u = torch.softmax(self.f_lower(h_lower) @ self.f_upper(h_upper).transpose(-1, -2), dim=-1)
        #A_w = torch.softmax(self.f_whole(h_whole) @ self.f_whole(h_upper).transpose(-1, -2), dim=-1)
        #print(A_u2l.shape)
        h_upper_new = h_upper + torch.matmul(A_l2u, h_lower)
        h_lower_new = h_lower + torch.matmul(A_u2l, h_upper)
        
        fused = torch.cat([h_upper_new, h_lower_new], dim=1)  # concat joints
        return self.output_proj(fused)
class EqMotion(nn.Module):
    def __init__(self, Tp, Tf, n, C=16, D=64, K=1, L=1):
        super(EqMotion, self).__init__()
        self.Tp, self.Tf, self.n = Tp, Tf, n
        self.C, self.D, self.K, self.L = C, D, K, L

        # Feature Initialization
        self.init_geom = nn.Linear(Tp, C)  # ϕinit_g
        self.init_pattern = nn.Sequential(  # ϕinit_h
            nn.Linear(2 * Tp, 128),
            nn.ReLU(),
            nn.Linear(128, D)
        )

        # Invariant Reasoning MLPs
        self.reason_mlp1 = nn.Linear(2 * D + C, 128)
        self.reason_mlp2 = nn.Linear(128 + D, 128)
        self.reason_mlp3 = nn.Linear(3 * D + C, K)

        # EGFL blocks
        self.attention_layers = nn.ModuleList([nn.Linear(D, C) for _ in range(L)])
        self.agg_mlps = nn.ModuleList([
            nn.ModuleList([nn.Sequential(nn.Linear(2 * D + C, C), nn.ReLU()) for _ in range(K)])
            for _ in range(L)
        ])
        self.query_k_layers = nn.ModuleList([nn.Linear(C, C) for _ in range(L)])

        # IPFL blocks
        self.message_mlps = nn.ModuleList([nn.Sequential(nn.Linear(2 * D + C, D), nn.ReLU()) for _ in range(L)])
        self.update_mlps = nn.ModuleList([nn.Sequential(nn.Linear(2 * D, D), nn.ReLU()) for _ in range(L)])

        # Output layer
        self.output_layer = nn.Linear(C, Tf)

    def forward(self, X):  # X: [M, Tp, n]
        M = X.shape[0]
        mean_X = X.mean(dim=(0, 1))  # [n]
        X_centered = X - mean_X  # remove translation

        # === Feature Initialization ===
        delta_X = X[:, 1:] - X[:, :-1]
        speed = torch.norm(delta_X, dim=2)
        angle = torch.atan2(delta_X[:, 1:, 1], delta_X[:, 1:, 0])  # assuming 2D
        angle = F.pad(angle, (0, self.Tp - angle.shape[1]))  # pad to Tp
        speed = F.pad(speed, (0, self.Tp - speed.shape[1]))
        h = self.init_pattern(torch.cat([speed, angle], dim=1))  # [M, D]
        g = self.init_geom(X_centered.transpose(1, 2)).transpose(1, 2) + mean_X  # [M, C, n]

        # === Invariant Reasoning ===
        c = torch.zeros(M, M, self.K).to(X.device)
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                gij = torch.norm(g[i] - g[j], dim=-1)  # [C]
                m_ij = torch.cat([h[i], h[j], gij], dim=-1)
                m1 = self.reason_mlp1(m_ij)
                m2 = self.reason_mlp2(torch.cat([m1, h[i]], dim=-1))
                logits = self.reason_mlp3(torch.cat([m2, h[j], gij], dim=-1))
                c[i, j] = F.softmax(logits, dim=-1)

        # === Feature Learning Loop ===
        for l in range(self.L):
            # -- EGFL inner attention --
            att_weight = self.attention_layers[l](h).unsqueeze(-1)  # [M, C, 1]
            g = att_weight * (g - g.mean(dim=0, keepdim=True)) + g.mean(dim=0, keepdim=True)

            # -- Inter-agent aggregation --
            new_g = g.clone()
            for i in range(M):
                agg = 0
                for j in range(M):
                    if i == j: continue
                    gij = torch.norm(g[i] - g[j], dim=-1)
                    for k in range(self.K):
                        edge_feat = torch.cat([h[i], h[j], gij], dim=-1)
                        w_ij = c[i, j, k] * self.agg_mlps[l][k](edge_feat)
                        agg += w_ij * (g[j] - g[i])
                new_g[i] += agg
            g = new_g

            # -- Non-linear equivariant transform (simplified) --
            g = torch.tanh(g)

            # -- IPFL --
            new_h = h.clone()
            for i in range(M):
                agg = 0
                for j in range(M):
                    if i == j: continue
                    gij = torch.norm(g[i] - g[j], dim=-1)
                    m_ij = self.message_mlps[l](torch.cat([h[i], h[j], gij], dim=-1))
                    agg += m_ij
                h_update = self.update_mlps[l](torch.cat([h[i], agg], dim=-1))
                new_h[i] = h_update
            h = new_h

        # === Output ===
        g_mean = g.mean(dim=0, keepdim=True)
        y_hat = self.output_layer((g - g_mean)) + g_mean  # [M, Tf, n]
        return y_hat

class MiniEqMotion(nn.Module):
    def __init__(self, Tp, Tf, n, C=8, D=32, K=1, L=1):
        super().__init__()
        self.Tp, self.Tf, self.n = Tp, Tf, n
        self.C, self.D, self.K, self.L = C, D, K, L

        self.init_geom = nn.Linear(Tp, C)
        self.init_pattern = nn.Sequential(
            nn.Linear(2 * Tp, 64),
            nn.ReLU(),
            nn.Linear(64, D)
        )

        self.reason_mlp1 = nn.Linear(2 * D + C, 64)
        self.reason_mlp2 = nn.Linear(64 + D, 64)
        self.reason_mlp3 = nn.Linear(3 * D + C, K)

        self.attention_layers = nn.ModuleList([nn.Linear(D, C) for _ in range(L)])
        self.agg_mlps = nn.ModuleList([
            nn.ModuleList([nn.Sequential(nn.Linear(2 * D + C, C), nn.ReLU()) for _ in range(K)])
            for _ in range(L)
        ])
        self.query_k_layers = nn.ModuleList([nn.Linear(C, C) for _ in range(L)])

        self.message_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * D + C, D), nn.ReLU()) for _ in range(L)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * D, D), nn.ReLU()) for _ in range(L)
        ])

        self.output_layer = nn.Linear(C, Tf)

    def forward(self, X):  # [M, Tp, n]
        M = X.shape[0]
        mean_X = X.mean(dim=(0, 1))
        X_centered = X - mean_X

        delta_X = X[:, 1:] - X[:, :-1]
        speed = torch.norm(delta_X, dim=2)
        angle = torch.atan2(delta_X[:, 1:, 1], delta_X[:, 1:, 0])  # assuming 2D
        Tp = self.Tp
        speed = F.pad(speed, (0, Tp - speed.shape[1]))
        angle = F.pad(angle, (0, Tp - angle.shape[1]))
        h = self.init_pattern(torch.cat([speed, angle], dim=1))  # [M, D]
        g = self.init_geom(X_centered.transpose(1, 2)).transpose(1, 2) + mean_X  # [M, C, n]

        c = torch.zeros(M, M, self.K).to(X.device)
        for i in range(M):
            for j in range(M):
                if i == j: continue
                gij = torch.norm(g[i] - g[j], dim=-1)
                m_ij = torch.cat([h[i], h[j], gij], dim=-1)
                m1 = self.reason_mlp1(m_ij)
                m2 = self.reason_mlp2(torch.cat([m1, h[i]], dim=-1))
                logits = self.reason_mlp3(torch.cat([m2, h[j], gij], dim=-1))
                c[i, j] = F.softmax(logits, dim=-1)

        for l in range(self.L):
            att_weight = self.attention_layers[l](h).unsqueeze(-1)
            g = att_weight * (g - g.mean(dim=0, keepdim=True)) + g.mean(dim=0, keepdim=True)

            new_g = g.clone()
            for i in range(M):
                agg = 0
                for j in range(M):
                    if i == j: continue
                    gij = torch.norm(g[i] - g[j], dim=-1)
                    for k in range(self.K):
                        edge_feat = torch.cat([h[i], h[j], gij], dim=-1)
                        w_ij = c[i, j, k] * self.agg_mlps[l][k](edge_feat)
                        agg += w_ij * (g[j] - g[i])
                new_g[i] += agg
            g = torch.tanh(new_g)

            new_h = h.clone()
            for i in range(M):
                agg = 0
                for j in range(M):
                    if i == j: continue
                    gij = torch.norm(g[i] - g[j], dim=-1)
                    m_ij = self.message_mlps[l](torch.cat([h[i], h[j], gij], dim=-1))
                    agg += m_ij
                h_update = self.update_mlps[l](torch.cat([h[i], agg], dim=-1))
                new_h[i] = h_update
            h = new_h

        g_mean = g.mean(dim=0, keepdim=True)
        y_hat = self.output_layer((g - g_mean)) + g_mean
        return y_hat

class DenoiseDiffusion14(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2=Decoder(35,35,5,output_size)
        self.ode = NeuralODE(dim=35)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.integral = nn.Conv1d(in_channels=35,out_channels=35,kernel_size=1)
        #self.intgru= nn.GRU(input_size=35,hidden_size=35,num_layers=1,batch_first=True)
        #self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean
        #return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
       # return mean
    def forward(self, x,t,c):
        time_grid = torch.linspace(0, 1, steps=10)
       # time_grid2 = torch.linspace(0,1,steps=2)
        #x_n=x
        #z_n,mu_n,logvar_n=self.encoder(x_n,c)
    
        #c is anchor
        #cur_int,mu_x,logvar_x=self.encoder(x,c)
        #cur_int=cur_int.unsqueeze(-1)
        #y0 = self.integral(cur_int) # Extend v to shape [2048, 35]
        #y0=y0.squeeze(-1)
        #yint,h1=self.intgru(y0)
        #yint=self.decoder2(yint,c)
        #full box of neural ik in this space !
        #print(h1.shape)
        #than this pipe line is just using ik solver that calculate the jerk and velocity!
        #Hmm what happen to you ?
        #OK than we can thinking about the minimum about this problem maybe its just a problem that happen in this file 
        #then we can make depender in this system!
        #print(yint)
       # vt=self.ode(yint,time_grid)
               
        z,mu,logvar=self.encoder(x,c) 
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(zc,c),mean,var

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ActNorm(nn.Module):
    """Activation Normalization Layer."""
    def __init__(self, num_channels):
        super().__init__()
        self.bias = Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = Parameter(torch.ones(1, num_channels, 1, 1))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                std = x.std(dim=(0, 2, 3), keepdim=True)
                self.bias.data.copy_(-mean)
                self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized = True
        return self.scale * (x + self.bias)

class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 Convolution."""
    def __init__(self, num_channels):
        super().__init__()
        weight = torch.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = Parameter(weight)
    
    def forward(self, x, reverse=False):
        b, c, h, w = x.size()
        log_det = torch.slogdet(self.weight)[1] * h * w
        if reverse:
            weight_inv = torch.inverse(self.weight)
            return F.conv2d(x, weight_inv.unsqueeze(-1).unsqueeze(-1)), -log_det
        else:
            return F.conv2d(x, self.weight.unsqueeze(-1).unsqueeze(-1)), log_det

class AffineCoupling(nn.Module):
    """Affine Coupling Layer."""
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        params = self.net(x1)
        scale, shift = params.chunk(2, dim=1)
        scale = torch.sigmoid(scale + 2.0)  # Scale stabilization
        if reverse:
            x2 = (x2 - shift) / scale
        else:
            x2 = scale * x2 + shift
        return torch.cat([x1, x2], dim=1)

class GlowStep(nn.Module):
    """One step of Glow."""
    def __init__(self, num_channels):
        super().__init__()
        self.actnorm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.coupling = AffineCoupling(num_channels)

    def forward(self, x, reverse=False):
        if reverse:
            x = self.coupling(x, reverse=True)
            x, _ = self.inv_conv(x, reverse=True)
            x = self.actnorm(x)
        else:
            x = self.actnorm(x)
            x, log_det1 = self.inv_conv(x)
            x = self.coupling(x)
        return x

class Glow(nn.Module):
    """Glow Model."""
    def __init__(self, num_channels, num_levels, num_steps):
        super().__init__()
        self.levels = nn.ModuleList()
        for _ in range(num_levels):
            steps = nn.ModuleList([GlowStep(num_channels) for _ in range(num_steps)])
            self.levels.append(steps)
    
    def forward(self, x, reverse=False):
        log_det = 0
        if reverse:
            for steps in reversed(self.levels):
                for step in reversed(steps):
                    x = step(x, reverse=True)
        else:
            for steps in self.levels:
                for step in steps:
                    x = step(x)
        return x
# Define the ODE Function

class DenoiseDiffusion14_mul(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size*2,35)
        self.encoder2 =Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2=Decoder(35,70,5,output_size)
        self.flow = nn.Linear(70,64)
        self.flow2 =nn.Linear(64,35)
        self.plane1=nn.Linear(105,100)
        self.plane2=nn.Linear(100,70)
        self.plane3=nn.Linear(70,35)
        self.h0 = torch.ones(2,2048, 35)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,v,x0,x1,x2,t,c):
        eps_theta,_,_ = self.forward(xt,v,x0,x1,x2,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,v,x0,x1,x2,t,c):
        flow=torch.cat((x,v),dim=-1)
        flow_box=self.flow(flow)
        flow_box=self.flow2(flow_box)
        plane=torch.cat((x0,x1),dim=-1)
        plane=torch.cat((plane,x2),dim=-1)
        plane=self.plane1(plane)
        plane=self.plane2(plane)
        plane=self.plane3(plane)
        flow_op=torch.cat((flow_box,plane),dim=-1)

        z,mu,logvar=self.encoder(flow_op,c)
        #z2=self.trans(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #x2 = self.unet(z2,t_emb).to(device="cpu") 
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #zc2=torch.cat((x2,z),dim=-1)
        return self.decoder(zc,c),mean,var

import torch
import torch.nn as nn
import torchdiffeq  # Library for ODE solvers
from copy import deepcopy
import torch
import torch_dct as dct
#from torchtnt.utils.flops import FlopTensorDispatchMode
#from neuralop.models import FNO
# Define ODE function
class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t,h):
        return self.fc(h)

# Neural ODE model
class NeuralODE(nn.Module):
    def __init__(self, dim):
        super(NeuralODE, self).__init__()
        self.ode_func = ODEfunc(dim)
    
    def forward(self, h0, t):
        # Solve ODE using odeint
        h = torchdiffeq.odeint(self.ode_func, h0, t,options={'min_step': 1e-6})
        return h

class DenoiseDiffusion14_mul_vec(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNetTransformer(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size*3,35)
        self.encoder2 =Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2=Decoder(70,35,5,output_size)
        self.flow = nn.Linear(70,64)
        self.flow2 =nn.Linear(64,35)
        self.plane1=nn.Linear(70,50)
        self.plane2=nn.Linear(50,35)
        #self.plane3=nn.Linear(70,35)
        self.h0 = torch.ones(2,2048, 35)
        self.func=ODEfunc(dim=35)
        self.ode = NeuralODE(dim=35)
        self.attn = Attention(dim=35)
        self.attn2 = Attention(dim=35)
        #self.fno = FNO(n_models=(35,35),hidden_channels=64,in_channels=1,out_channels=1,projection_channel_ration=1)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)

    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var
    def get_normal(self,x1,x2,x3):
        v1=x1-x2
        v2=x1-x3
        normal=torch.cross(v1,v2,dim=-1)
        if torch.norm(normal)<1e-4:
            eps=torch.randn(2048,35)
            normal=normal+eps
        return normal
    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,v,x1,x2,t,c):
        eps_theta,_,_,_,_= self.forward(xt,v,x1,x2,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,v,x1,x2,t,c):
        time_grid = torch.linspace(0, 1, steps=10)
        y0 = v # Extend v to shape [2048, 35]
        #y0=self.dct(y0)
        vt=self.ode(y0,time_grid)
        #vt=self.idct(vt)
        #self.v=self.fno(v)
        vtx=vt[9,::]-vt[0,::]
        
                 # Time grid
        flow=torch.cat((x,v),dim=-1)
        #flow=torch.cat((flow,tim),dim=-1)

        flow_box=self.flow(flow)
        flow_box=self.flow2(flow_box)
        
        #plane=torch.cat((x0,x1),dim=-1)
        plane=torch.cat((x1,x2),dim=-1)
        plane=self.plane1(plane)
        plane=self.plane2(plane)
        
        #plane=self.plane3(plane)

        #eps=torch.randn(2048,35)
        #eps=eps*0.0000000001
        #if torch.norm(plane)<1e-9:
        #    plane=plane+eps
        
        #plane=self.get_normal(x0,x1,x2)
        #plane=self.plane3(plane)
        flow_op=torch.cat((flow_box,plane),dim=-1)
        flow_box_t=torch.cat((flow_op,vtx),dim=-1)


        z,mu,logvar=self.encoder(flow_box_t,c)
        #z2=self.trans(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb,flow_box).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #x2 = self.unet(z2,t_emb).to(device="cpu") 
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #zc2=torch.cat((x2,z),dim=-1)
        return self.decoder(zc,c),mu,logvar,vt[0,::],vt[9,::]
class ComplexConv1d(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.real_conv = torch.nn.Conv1d(*args, **kwargs)
        self.imag_conv = torch.nn.Conv1d(*args, **kwargs)

    def forward(self, x):
        real = x.real
        imag = x.imag

        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)

        return torch.complex(real_out, imag_out)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x,condition):
        # Initialize hidden state
        #h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)  # [num_layers, batch_size, hidden_size]
        out, _ = self.rnn(x, condition)  # RNN forward pass
        out = self.fc(out[:, -1, :])  # Fully connected on the last time step's output
        return out
#general method in world and weak solution in probability!
class DenoiseDiffusion14_mul_vec_n_atten(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNetTransformer(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size*3,35)
        self.encoder2 =Encoder(input_size,35)
        self.encoder3 = Encoder(input_size,35)
        self.encoder4 = Encoder(input_size,35)
        self.encoder5 = Encoder(input_size,35)
        self.encoder6 = Encoder2(input_size,35)
        self.world_enc = Encoder2(input_size,35)
        self.world_dec = Decoder2(35,35,5,output_size)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2=Decoder(35,70,5,output_size)
        self.decoder3 = Decoder(35,70,5,output_size)
        self.decoder4 = Decoder(35,35,5,output_size)
        self.decoder5 = Decoder(35,70,5,output_size)
        #self.world_gru = nn.GRU(input_size=35,hidden_size=35,num_layers=1,batch_first=True)
        #if so delicated than function overfitting! 
        #in this sense we can thinking about light version about model!
        self.flow = nn.Linear(105,64)
        self.flow2 =nn.Linear(64,35)
        self.pos_net=nn.Linear(70,35)
        #self.flowgru = nn.GRU(input_size=105,hidden_size=35,num_layers=2,batch_first=False)
        self.plane1=nn.Linear(70,50)
        self.plane2=nn.Linear(50,35)
        self.intgru= nn.GRU(input_size=35,hidden_size=35,num_layers=1,batch_first=True)
        

        #self.gru =nn.GRU(input_size=105,hidden_size=35,num_layers=2,batch_first=True)
        self.h0 = torch.ones(2,2048, 35)
        self.func=ODEfunc(dim=35)
        self.ode = NeuralODE(dim=35)
        self.ode2 = NeuralODE(dim=35)
        self.ode3 = NeuralODE(dim=35)
        self.attn = Attention(dim=35)
        self.attn2 = Attention(dim=35)
        self.conv = ComplexConv1d(in_channels=1, out_channels=1, kernel_size=1)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.hidden_flow=nn.Linear(70,35)
        #self.trans =nn.Transformer(d_model=35,nhead=5,num_encoder_layers=5,num_decoder_layers=5)
        self.unet2 =UNet0(35,35)
        self.unet3 = UNet0(35,35)
        #World VAE model in this sense!
       # self.unet4 = UNet0(35,35)
        self.time_embed = TimeEmbedding(70)
        self.integral = nn.Conv1d(in_channels=35,out_channels=35,kernel_size=1)
        self.integral2 = nn.Conv1d(in_channels=105,out_channels=105,kernel_size=1)
        self.su_int = nn.Linear(35,35)
        self.beta =self.gaussiandiffusion.beta
        self.final_layer=nn.Linear(35,35)
        self.mean=0
        self.logvar=0
        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)
        self.world1 = nn.Linear(35,70)
        self.world2 = nn.Linear(105,35)
        #self.world3 = nn.Linear(35,35)
        #self.world4 = nn.Linear(35,35)
        #self.world5 = nn.Linear(35,35)
        self.world3 = nn.Linear(105,105)       
        self.world6 = nn.Linear(105,35)
        
        
        self.normal_sampls =nn.Linear(35,35)
    

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis
    
    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)
    def positional_encoding(self,seq_len, dim, device='cpu'):
       """
    Generate a positional encoding matrix for a Transformer.
    
    Args:
        seq_len (int): The sequence length (number of tokens).
        dim (int): The dimension of the embedding vector.
        device (str): The device to store the tensor ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: A tensor of shape (seq_len, dim) containing the positional encodings.
       """
    # Create a matrix of positions (seq_len x dim)
       positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # Shape: [seq_len, 1]
       div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * 
                         (-math.log(10000.0) / dim))  # Shape: [dim/2]
    
       # Apply sine to even indices and cosine to odd indices
       pos_enc = torch.zeros((seq_len, dim), device=device)
       pos_enc[:, 0::2] = torch.sin(positions * div_term)  # Even indices
       pos_enc[:, 1::2] = torch.cos(positions * div_term)  # Odd indices
    
       return pos_enc
    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var
    def get_normal(self,x1,x2,x3):
        v1=x1-x2
        v2=x1-x3
        normal=torch.cross(v1,v2,dim=-1)
        if torch.norm(normal)<1e-4:
            eps=torch.randn(2048,35)
            normal=normal+eps
        return normal
    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    # real manifold vs data manifold
    def world(self,curr,cur1,cur2,cur3,condition,w,t,noise):
        world_cur=self.q_sample(curr,t,noise)
        world = self.world1(world_cur)
        cur=torch.cat((cur1,cur2),dim=-1)
        cur=torch.cat((cur,cur3),dim=-1)
        sub_world_t=self.world3(cur)
        #sub_world=self.world3(cur1)
        #sub_world2=self.world4(cur2)
        #sub_world3=self.world5(cur3)
        #sub_world_t=torch.cat((sub_world,sub_world2),dim=-1)
        #sub_world_t=torch.cat((sub_world_t,sub_world3),dim=-1)
        #Model->Vison->Memory->ODE-> World position!
        #World Embedding and making encoder and decoder and taking the ode!
        #Using Anchor Condition!  
        #sub_world_t,world_mu,world_logvar=self.encoder(sub_world_t,condition)
        
        world_E,w_m,w_l=self.world_enc(world_cur,condition)
        #V model and M model and C model is clear!
        #world_x=self.unet2(world_E,t)
        #world_E=torch.cat((world_E,),dim=-1)
        world_E=self.world_dec(world_E,condition)
        #sub_world_t=torch.cat((sub_world_t,world_E),dim=-1)
       # than what happen in this field when using vae ....... 
       # some over fitting is here!
        sub_world_t=self.world6(sub_world_t)
        #sub_world_t,s_h = self.world_gru(sub_world_t)
        #No reason using more gru ok stop in this time!
    
        #sub_world_t=self.decoder(sub_world_t,condition)
        time_grid = torch.linspace(0, 1, steps=10)
        #sub_world_t=self.dct(sub_world_t)
        #sub_world_t=sub_world_t.unsqueeze(-1)
       # sub_world_t=self.integral(sub_world_t)
        #sub_world_t=sub_world_t.squeeze(-1)
        sw=self.ode(sub_world_t,time_grid)
        #in flow mapping in world model we can thinking about the flow in manifold!
        world_flow=sw[9,::]-sw[0,::]
        #world_flow=self.idct(world_flow)
        world=torch.cat((world_flow,world),dim=-1)
        return  self.world2(world),w_m,w_l

    def p_sample(self, xt,v,x1,x2,t,c,cur,vec):
        eps_theta,_,_,_,_,_,_,_= self.forward(xt,v,x1,x2,t,c,cur,vec)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        return mean+(var**0.5)*eps
        #return mean
        #return val*mean+(var**0.5)
        #return eps_theta
        #return mean
    def pd_sample(self, xt,v,x1,x2,t,c,cur,vec):
        eps_theta,_,_,_,_,_,_,_= self.forward(xt,v,x1,x2,t,c,cur,vec)
        tt=torch.tensor(0).long()
        eps_theta2 = eps_theta.repeat(1,5)
        #dual sampling in this space!
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        return mean-(var**0.5)*eps
        #dual sense in this sampling!
        #return mean
        #return val*mean+(var**0.5)
        #return eps_theta
        #return mean
    #def normal_sample(self,x1,x2):
    #    normal=x1+x2
        #normal=torch.cat((normal,x1),dim=-1)
        
        #normal=torch.cat((normal,x2),dim=-1)
        #normal=self.normal_sample1(normal)
        #normal=self.normal_sample2(normal)
    #    return self.normal_sampls(normal)
    def forward(self, x,v,x1,x2,t,c,cur,vec):
        time_grid = torch.linspace(0, 1, steps=10)
        time_grid2 = torch.linspace(0,1,steps=2)
        x_n=x
        z_n,mu_n,logvar_n=self.encoder6(x_n,c)
        cur_int=self.dct(x)
        #c is anchor
        cur_int,mu_x,logvar_x=self.encoder5(cur_int,c)
        cur_int=cur_int.unsqueeze(-1)
        y0 = self.integral(cur_int) # Extend v to shape [2048, 35]
        y0=y0.squeeze(-1)
        yint,h1=self.intgru(y0)
        #print(yint)
        yint=self.decoder4(yint,c)
        #full box of neural ik in this space !
        #print(h1.shape)
        #than this pipe line is just using ik solver that calculate the jerk and velocity!
        #Hmm what happen to you ?
        #OK than we can thinking about the minimum about this problem maybe its just a problem that happen in this file 
        #then we can make depender in this system!
        #print(yint)
        vt=self.ode(yint,time_grid)
               
        vtt=vt[9,::]-vt[0,::]
        
        #vt=self.idct(vt)
        #c is anchor
        v_dct=self.dct(v)
        v_dct,mu_dct,logvar_dct=self.encoder5(v_dct,c)
        vt2 = self.ode(v_dct,time_grid)
        vtx=vt2[9,::]-vt2[0,::]
        vtx=self.decoder4(vtx,c)
        vtx = self.idct(vtx)
        x_i=self.dct(x)
        x_i,mu_i,logvar_i=self.encoder5(x_i,c)
        x_i=x_i.unsqueeze(-1)
        #x_int=self.integral(x_i)
        x_int=x_i.squeeze(-1)
        #it's same work in this gru beacuse the ode learning in ik solver!
        x_int,h_int=self.intgru(x_int)
        x_int=self.decoder4(x_int,c)
        x_int=self.idct(x_int)
        riman=self.ode(x_int,time_grid2)
        riman_d=riman[1,::].mul(riman[0,::])
                 # Time grid
        flow=torch.cat((x,v),dim=-1)
        flow=torch.cat((flow,riman_d),dim=-1)
        #flow= torch.cat((flow,vtx),dim=-1)
        #flow_box,hidden_flow=self.flowgru(flow)
        flow_box=self.flow(flow)
        flow_box=self.flow2(flow_box)

        #flow_box=flow_box.unsqueeze(-1)
        #flow_box=flow_box.unsqueeze(-1)
        #flow_box =self.attn(flow_box)
        #flow_box=flow_box.squeeze(-1)
        #flow_box=flow_box.squeeze(-1)
        #plane=torch.cat((x0,x1),dim=-1)
        plane=torch.cat((x1,x2),dim=-1)
        plane=self.plane1(plane)
        plane=self.plane2(plane)
        #plane,hidden_plane=self.planegru(plane)
        #plane=plane.unsqueeze(-1)
        #plane=plane.unsqueeze(-1)
        #plane = self.attn2(plane)
        #plane=plane.squeeze(-1)
        #plane=plane.squeeze(-1)
        #plane=self.plane3(plane)

        #eps=torch.randn(2048,35)
        #eps=eps*0.0000000001
        #if torch.norm(plane)<1e-9:
        #    plane=plane+eps
        
        #plane=self.get_normal(x0,x1,x2)
        #plane=self.plane3(plane)
        flow_op=torch.cat((flow_box,plane),dim=-1)
        flow_box_t=torch.cat((flow_op,vtx),dim=-1)
        #flow_box_t,hidden_state= self.gru(flow_box_t)
        #print(flow_box_t.shape)
        # I want make noise that can be walk better~~~

        z,mu,logvar=self.encoder(flow_box_t,c)
        pos_enc=self.positional_encoding(2048,70)# positional encoding that using linear equations 
        pos_enc=self.pos_net(pos_enc)
        z_cur,mu_cur,logvar_cur=self.encoder2(cur,vec) #embedding method in vec and cur 
        x_noise,mu_noise,logvar_noise = self.encoder3(x,v)
        #cur_z,cur_mu,cur_logvar=self.encoder4(x,cur)
        #z2=self.trans(x,c)
        t_emb = self.time_embed(t)
        #t_noise=self.unet2(x,t_emb)
        #cur_pos=cur+pos_enc
        #semi MOE in KL divergence
        x= self.unet(z,t_emb,flow_box,z_cur,x_noise).to(device="cpu")
        mu = self.unet2(mu,t_emb)
        logvar = self.unet2(logvar,t_emb)
    
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #z2=self.q_sample(x,tt,t_noise)
        #x2 = self.unet(z2,t_emb,flow_box,cur).to(device="cpu") 
        zc=torch.cat((x,z),dim=-1)
        #n_oise=self.unet3(x_n,t_emb)
        #zc_cur=torch.cat((x_cur,cur_z),dim=-1)
        #zc2=torch.cat((x2,z),dim=-1)\
        #e1=self.decoder(zc,c)
        n_oise=self.unet3(z_n,t_emb)
        n_oise=torch.cat((n_oise,z_n),dim=-1)
        #noise two space in this model!(like gan property!)
        #if in this page we can think about the noise that can make gan effect in the RL but when diversity is high than the quality is down in this sense how solve this problem?
        #when my first try is to check the layer 2->1 ...
        #Moe=torch.cat((e1,e2),dim=-1)
        #return self.final_layer(self.decoder(zc,c)+self.decoder3(zc_cur,cur)),mu,logvar,self.idct(vt[0,::]),self.idct(vt[5,::]),self.idct(vt[9,::]),self.decoder2(zc,x),vtt
        #latent space of decoder space !\
        #output=self.decoder(zc,c)+cur
        return self.decoder(zc,c),mu,logvar,self.idct(vt[0,::]),self.idct(vt[5,::]),self.idct(vt[9,::]),self.decoder5(n_oise,c),vtt
import torch

import torch.nn as nn
import math

class DenoiseDiffusion14_dct(nn.Module):
    def __init__(self, input_size, noise_steps, latent_size, output_size):
        super().__init__()
        self.timesteps = noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming input and output dimensions of 35 for U-Net
        self.encoder = Encoder(input_size, 35)
        self.decoder = Decoder(35, 70, 5, output_size)
        self.trans = TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta = self.gaussiandiffusion.beta
        self.mean = 0
        self.logvar = 0

        # DCT and IDCT Basis
        self.dct_basis = self.create_dct_basis(35)  # Dimension 35
        self.idct_basis = self.create_idct_basis(35)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)

    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)

    def generate(self, z, c):
        z = torch.cat((z, c), dim=-1)
        return self.decoder(z, c)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt, t, c):
        eps_theta, _, _,_ = self.forward(xt, t, c)
        tt = torch.tensor(0).long()
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean

    def forward(self, x, t, c):
        # DCT Transformation
        z, mu, logvar = self.encoder(x, c)
        z_dct = self.dct(z)  # Apply DCT to latent representation
        noise = torch.randn_like(z)
        # Time Embedding
        t_emb = self.time_embed(t)
        z_noise = self.q_sample(z_dct,t,noise)
        # Pass through U-Net

        # IDCT Transformation
        unet_output_idct = self.idct(z_noise)  # Apply IDCT after U-Net
        unet_output = self.unet(unet_output_idct, t_emb).to(device="cpu")

      

        # Decoder
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean, var = self.q_xt_x0(unet_output_idct, tt)
        zc = torch.cat((unet_output, z), dim=-1)
        return self.decoder(zc, c), mean, var,z
class DenoiseDiffusion16(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean+(var**0.5)
        #return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        for i in range(1,self.timesteps):
            x = self.unet(x,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(zc,c),mean,var
class DenoiseDiffusion14_Loop(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet3(35, 35)  # Assuming 3 input and 3 output channels for RGB
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,idx,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        if idx == 0:
            return mean
        else:
            return mean + (var ** 0.5) * eps
    def p_sample_loop(self,xt,t,c):
        b=xt
        img=torch.randn(35)
        imgs=[]
        for i in tqdm(reversed(range(0, self.timesteps)), desc = 'sampling loop time step', total=self.timesteps):
            img=self.p_sample(xt,t, i,c)
            #print(img.shape)
            imgs.append(img)
        return imgs
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        mean,var=self.q_xt_x0(x,tt)
        #z=self.trans(z,c)
        zc=torch.cat((x,z),dim=-1)

        return self.decoder(zc,c),mean,var


class DenoiseDiffusion15(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        #self.mixdecoder = MixedDecoder()
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2 = self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(z, zc), mean, var
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var

class DenoiseDiffusion18(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(175, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,175)
        self.decoder=Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(350)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var
    def reduce_mean_to_35(self, mean_175):
        # 175차원을 5개의 그룹으로 나누고, 각 그룹의 평균을 구해 35차원으로 축소
        mean_35 = mean_175.view(mean_175.size(0), 5, -1, *mean_175.size()[2:]).mean(dim=1)
        return mean_35

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        #z2,mu2,logvar2 = self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(z, zc), mean, var
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var
class DenoiseDiffusion0(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        #z,mu,logvar=self.encoder(x,c)
        #z2,mu2,logvar2 = self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(x,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        #zc=torch.cat((x,z),dim=-1)
        #return self.decoder(z, zc), mean, var
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var
        return x,mean,var

class Conv1dModel(nn.Module):
    def __init__(self):
        super(Conv1dModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Generator(nn.Module):
    def __init__(self,input,output):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,output),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self,input,output):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
class DenoiseDiffusion20(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder2(input_size,35)
        self.encoder2 = Encoder2(input_size,35)
        self.encoder3 = Encoder2(input_size,35)
        #self.encoder4 = Encoder(105,35)
        self.cnn=Conv1dModel()
        self.attn = Attention(dim=35)
        self.decoder=Decoder(35,70,5,output_size)
        #self.generator =  Generator(input,3*input)
        #self.discriminator= Discriminator(3*input,input)
        #self.mixdecoder =MixedDecoder(35,70,output_size,10,10,10)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=105)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,xt2,xt3,t,c,c2,c3):
        eps_theta,_,_,_= self.forward(xt,xt2,xt3,t,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,x2,x3,t,c,c2,c3):
        #cc=torch.cat((c,c2),dim=-1)
        #cc=torch.cat((cc,c3),dim=-1)
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2 = self.encoder2(x2,c2)
        z3,mu3,logvar3 = self.encoder3(x3,c3)
        #z=z+z2
        #z=torch.cat((z,z2),dim=-1)
        z=torch.cat((z,z2),dim=-1)
        z=torch.cat((z,z3),dim=-1)
        tgt=torch.cat((c,c2),dim=-1)
        tgt=torch.cat((tgt,c3),dim=-1)
        z=self.trans(z,tgt)
        z_noise=z
        #z,mu4,logvar4= self.encoder4(z,c)
        #z=z.unsqueeze(-1)
        #z=z.permute(0,2,1) 
        #z=self.cnn(z)
        #z=z.squeeze(1)
        #z=z.unsqueeze(-1)
        #z=z.unsqueeze(-1)
        #z=self.attn(z)
        #z=z.squeeze(-1)
        #z=z.squeeze(-1)
        t_emb = self.time_embed(t)
        #t_e = t_emb.repeat(1,2)
        #x= self.trans(z,t_emb)
        #z=self.trans(z,t_e)

        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(zc,c),mean,var,z_noise

class Belfusion(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.encoder2 = Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
    
    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var
    def reduce_mean_to_35(self, mean_175):
        # 175차원을 5개의 그룹으로 나누고, 각 그룹의 평균을 구해 35차원으로 축소
        mean_35 = mean_175.view(mean_175.size(0), 5, -1, *mean_175.size()[2:]).mean(dim=1)
        return mean_35

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c):
        eps_theta,_,_,_,output = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return eps_theta
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
        #return output
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        #z2,mu2,logvar2 = self.encoder2(x,c)
        #z2,mu2,logvar2 = self.encoder(x,c)
        noise = torch.randn_like(z)
        z_noise = self.q_sample(z,t,noise)
        t_emb = self.time_embed(t)
        x = self.unet(z_noise,t_emb).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        zc=torch.cat((x,z),dim=-1)
        return x, z,mean, var,self.decoder(zc, z)
                                                            
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var
    
class Dif_gan(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet0(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.unet2 = UNet0(35,35)
        self.encoder=Encoder(input_size,35)
        self.encoder2=Encoder(input_size,35)
        self.encoder2 = Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.decoder2 = Decoder (35,70,5,output_size)

        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        N = 35  # 사용되는 차원 수
        self.dct_basis = self.create_dct_basis(N)
        self.idct_basis = self.create_idct_basis(N)

    
    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder2(z,c)

    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def reduce_mean_to_35(self, mean_175):
        # 175차원을 5개의 그룹으로 나누고, 각 그룹의 평균을 구해 35차원으로 축소
        mean_35 = mean_175.view(mean_175.size(0), 5, -1, *mean_175.size()[2:]).mean(dim=1)
        return mean_35
    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis
    
    def apply_dct(self, x):
        return self.dct(x)
    def apply_idct(self, y):
        return self.idct(y)
    

    def dct(self, x):
        # Apply DCT transformation
        return torch.matmul(x, self.dct_basis)
    def idct(self, x):
        # Apply IDCT transformation
        return torch.matmul(x, self.idct_basis)
    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    
    def p_sample(self, xt,t,c):
        eps_theta,_,out,_,_,_,_,output,_= self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return eps_theta
        #return mean+(var**0.5)*eps
        return mean
         
        #return self.generate(eps_theta,out)
        #return self.generate(eps_theta,c)
        #return mean + (var ** 0.5) * output
        #return output
    def forward(self, x,t,c):
    
        z,mu,logvar=self.encoder(x,c)
        
        #z2,mu2,logvar2 = self.encoder2(x,c)
        #z2,mu2,logvar2 = self.encoder(x,c)
        #z_noise = self.apply_dct(z)
        #noise = torch.randn_like(z)
        #z_noise = self.q_sample(z,t,noise)
        #x_noise= self.q_sample(x,t,noise)
        #z2,mu2,logvar2 = self.encoder2(x_noise,x)
        #noise = torch.randn_like(z)
        #z_noise = self.q_sample(z,t,noise)
        #z_noise = self.apply_idct(z_noise)
        t_emb = self.time_embed(t)
        x1 = self.unet(z,t_emb).to(device="cpu")
        x2 = self.unet2(z,t_emb).to(device="cpu")
         
        #tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,t)
        zc=torch.cat((x1,z),dim=-1)
        zc2=torch.cat((x2,z),dim=-1)
        z_noise2=self.apply_dct(x1)
        z_noise2 = self.q_sample(x1,t,x2)
        z_noise2 = self.apply_idct(z_noise2)
        x3=self.unet(z_noise2,t_emb).to(device="cpu")
        x4=self.unet2(z_noise2,t_emb).to(device="cpu")
        zc3=torch.cat((x3,z),dim=-1)
        zc4 = torch.cat((x4,z),dim=-1)

        return x1,x3,z,mean, var,self.decoder(zc, z),self.decoder2(zc2,z),self.decoder(zc3,z),self.decoder2(zc4,z)
                                                            
        #return self.decoder(zc,z),mean,var
        #return self.decoder(zc,c),mean,var
class DenoiseDiffusion20_Trans(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder2(input_size,35)
        self.encoder2 = Encoder2(input_size,35)
        self.encoder3 = Encoder2(input_size,35)
        #self.encoder4 = Encoder(105,35)
        self.cnn=Conv1dModel()
        self.attn = Attention(dim=35)
        self.decoder=Decoder2(70,35,5,output_size)
        #self.generator =  Generator(input,3*input)
        #self.discriminator= Discriminator(3*input,input)
        #self.mixdecoder =MixedDecoder(35,70,output_size,10,10,10)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,xt2,xt3,t,c,c2,c3):
        eps_theta,_,_= self.forward(xt,xt2,xt3,t,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,x2,x3,t,c,c2,c3):
        cc=torch.cat((c,c2),dim=-1)
        cc=torch.cat((cc,c3),dim=-1)
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2 = self.encoder2(x2,c2)
        z3,mu3,logvar3 = self.encoder3(x3,c3)
        #z=z+z2
        #z=torch.cat((x,x2),dim=-1)
       
        #noise = torch.randn_like(z)
        zt=torch.cat((z,z2),dim=-1)
        zt=torch.cat((zt,z3),dim=-1)
        #z=torch.cat((z,x3),dim=-1)
        tgt=torch.cat((c,c2),dim=-1)
        tgts=torch.cat((tgt,c3),dim=-1)
        #z=self.trans(zt,tgt)
        #z_noise=z
        #z,mu4,logvar4= self.encoder4(z,c)
        #z=z.unsqueeze(-1)
        #z=z.permute(0,2,1) 
        #z=self.cnn(z)
        #z=z.squeeze(1)
        #z=z.unsqueeze(-1)
        #z=z.unsqueeze(-1)
        #z=self.attn(z)
        #z=z.squeeze(-1)
        #z=z.squeeze(-1)
        #t_emb = self.time_embed(t)
        #tgt=torch.cat((tgt,t_emb),dim=-1)
        #t_e = t_emb.repeat(1,2)
        #x= self.trans(z,t_emb)
        #z=self.trans(z,t_e)
        #print(t_emb.shape)
        #t3=t.repeat(1,3)
        
       
        #z_noise = self.q_sample(z,t,noise)
        #z_noise = self.q_sample(z_noise,t,noise)
        #t3_emb=self.time_embed(t3)
        t_emb=self.time_embed(t)
        #z=torch.cat((z,t_emb),dim=-1)
        x = self.trans(z,z2,z3,t_emb,c,tgt,tgts).to(device="cpu")
        # may be ok when using t_emb upper !
        #x = self.unet(x,t_emb).to(device="cpu")
        #x = self.trans(z_noise,t_emb).to(device="cpu")
        #x = self.trans(x,t_emb)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #return x ,mean,var
        return self.decoder(zc,c),mean,var

class DenoiseDiffusion21(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder2(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        self.encoder2 = Encoder2(input_size,35)
        self.decoder2=Decoder(35,70,5,output_size)
        #self.decoder2 = Decoder(35,70,5,output_size)
        #self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,c,c2,c3):
        eps_theta,_,_ = self.forward(xt,t,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,t,c,c2,c3):
        #cc=torch.cat((c,c2),dim=-1)
        #cc=torch.cat((cc,c3),dim=-1)
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        #x = self.unet(x,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        #zcc=torch.cat((zc,c3),dim=-1)
        x2=self.decoder(zc,c)
        z2,mu2,logvar2=self.encoder2(x2,c3)
        x2=self.unet(z2,t_emb).to(device="cpu")
        zcc=torch.cat((x2,z2),dim=-1)
        return self.decoder2(zcc,c3),mean,var
        #return self.decoder(zcc,c),mean,var

class DenoiseDiffusion22(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder2(input_size,35)
        self.encoder=Encoder2(input_size,35)
        self.decoder=Decoder(35,105,5,output_size)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=70)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,xt2,t,c,c2,c3):
        eps_theta,_,_ = self.forward(xt,t,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,x2,t,c,c2,c3):
        #cc=torch.cat((c,c2),dim=-1)
        #cc=torch.cat((cc,c3),dim=-1)
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2= self.encoder2(x2,c3)
        tgt=torch.cat((c,c3),dim=-1)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        x2= self.unet(z2,t_emb).to(device="cpu")
        #x = self.unet(x,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        zcc=torch.cat((x2,z2),dim=-1)
        zt=torch.cat((zc,zcc),dim=-1)
        return self.trans(zt,tgt),mean,var
import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim, bias = bias)
        self.kw = nn.Linear(dim, dim, bias = bias)
        self.vw = nn.Linear(dim, dim, bias = bias)

        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))

        self.ow = nn.Linear(dim, dim, bias = bias)

    def forward(self, x):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        B, L, D = q.shape
        q = torch.reshape(q, [B, L, self.n_heads, -1])
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.reshape(k, [B, L, self.n_heads, -1])
        k = torch.permute(k, [0, 2, 3, 1])
        v = torch.reshape(v, [B, L, self.n_heads, -1])
        v = torch.permute(v, [0, 2, 3, 1])
        k = torch.matmul(k, self.E[:L, :])

        v = torch.matmul(v, self.F[:L, :])
        v = torch.permute(v, [0, 1, 3, 2])

        qk = torch.matmul(q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v)
        v_attn = torch.permute(v_attn, [0, 2, 1, 3])
        v_attn = torch.reshape(v_attn, [B, L, D])

        x = self.ow(v_attn)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransformerBlock(nn.Module):
    def __init__(self, seq_len, dim, heads, mlp_dim, k, rate=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LinformerAttention(seq_len, dim, heads, k)
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)

        nn.init.zeros_(self.gamma_1.weight)
        nn.init.zeros_(self.beta_1.weight)
        nn.init.zeros_(self.gamma_1.bias)
        nn.init.zeros_(self.beta_1.bias)  

        nn.init.zeros_(self.gamma_2.weight)
        nn.init.zeros_(self.beta_2.weight)
        nn.init.zeros_(self.gamma_2.bias)
        nn.init.zeros_(self.beta_2.bias)  

        nn.init.zeros_(self.scale_1.weight)
        nn.init.zeros_(self.scale_2.weight)
        nn.init.zeros_(self.scale_1.bias)
        nn.init.zeros_(self.scale_2.bias)  

    def forward(self, x, c):
        #c = self.ln_act(c)
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(1)
        x = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)) * gate_msa + x
        return self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)) * gate_mlp + x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        # Zero-out output layers:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)        

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    def __init__(self, img_size, dim=64, patch_size=35,
                 depth=3, heads=7, mlp_dim=512, k=64, in_channels=3):
        super(DiT, self).__init__()
        self.dim = dim
        self.n_patches = (img_size // patch_size)**2 
        self.depth = depth
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, dim))
        self.patches = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, 
                      stride=patch_size, padding=0, bias=False),
        )
        
        self.transformer = nn.ModuleList()
        for i in range(self.depth):
            self.transformer.append(
                TransformerBlock(
                    self.n_patches, dim, heads, mlp_dim, k)
            )

        self.emb = nn.Sequential(
            TimeEmbedding(70),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.final = FinalLayer(dim, patch_size, in_channels)
        self.ps = nn.PixelShuffle(patch_size)

    def forward(self, x, t):
        t = self.emb(t)
        x=x.unsqueeze(-1)
        x = self.patches(x)
        B, C, H, W = x.shape
        x = x.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        x += self.pos_embedding
        for layer in self.transformer:
            x = layer(x, t)

        x = self.final(x, t).permute([0, 2, 1])
        x = x.reshape([B, -1, H, W])
        x = self.ps(x)
        x=x.squeeze(-1)
        return x
class DenoiseDiffusion23(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet8(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        
        self.trans =TransformerModel(in_channels=105)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt,t,edge,c,c2,c3):
        eps_theta,_,_,_ = self.forward(xt,t,edge,c,c2,c3)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,edge,c,c2,c3):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")
        for i in range(1,self.timesteps):
            x = self.unet(x,t_emb,edge).to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.decoder(zc,c),mean,var


class DenoiseDiffusion24(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.encoder2=Encoder(input_size,35)
        self.encoder3 = Encoder(input_size,35)
        self.encoder4= Encoder(input_size,35)
        self.decoder=Decoder(35,140,5,output_size)
        #self.decoder2 = Decoder(35,70,5,output_size)
        self.trans =TransformerModel(in_channels=105)
        self.nn =nn.Linear(140,70)
        self.nn2 = nn.Linear(70,35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0
        N = 35  # 사용되는 차원 수
        self.dct_basis = self.create_dct_basis(N)
        self.idct_basis = self.create_idct_basis(N)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis

    def generate(self,z,c):
        z=torch.cat((z,c),dim=-1)
        return self.decoder(z,c)
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def apply_dct(self, x):
        return self.dct(x)
    def apply_idct(self, y):
        return self.idct(y)
    def p_sample(self, xt,t,c):
        eps_theta,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #return mean+(var**0.5)
        #return self.decoder(val,c)
        #return mean + (var ** 0.5) * eps
        #return val*mean+(var**0.5)
        #return eps_theta
        return mean
    def forward(self, x,x2,x3,x4,t,c):
        z,mu,logvar=self.encoder(x,c)
        z2,mu2,logvar2 = self.encoder2(x2,c)
        z3,mu3,logvar3 = self.encoder3(x3,c)
        z4,mu4,logvar4 = self.encoder4(x4,c)
        #zk=torch.cat((z,z2),dim=-1)
        #zk=torch.cat((zk,z3),dim=-1)
        #zk=torch.cat((zk,z4),dim=-1)
        zk=z+z2+z3+z4
        zk=self.apply_dct(zk)
        #zkk=self.nn(zk)
        #zkk=self.nn2(zkk)
        t_emb = self.time_embed(t)
        x = self.unet(zk,t_emb).to(device="cpu")
        x=self.apply_idct(x)
        tt = torch.tensor(t_emb).long().to(device="cpu")
        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,zk),dim=-1)
        return self.decoder(zc,c),mean,var
import torch
class SEBlock(nn.Module):
    def __init__(self, input_dim, output,reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        b, c = x.shape
        y = self.global_avg_pool(x.view(b, c, 1)).view(b, c)
        # Excitation: Fully Connected layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Scale the input features
        y = y.view(b, c)

        return x * y
class SETransformerBlock(nn.Module):
    def __init__(self, input_dim, output,num_heads, reduction=16):
        super(SETransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.se_block = SEBlock(input_dim,output, reduction)

    def forward(self, x):
        # Multihead Attention + Residual Connection
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        # Feed Forward Network + Residual Connection
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)

        # SEBlock 적용
        x = self.se_block(x)

        return x
import math
import torch.fft
class TransFusion(nn.Module):
    def __init__(self, input_size, noise_steps, latent_size, output_size):
        super().__init__()
        self.timesteps = noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)
        self.encoder = Encoder(input_size, 35)
        self.decoder = Decoder(35, 70, 5, output_size)
        #self.layer = nn.ModuleList([
        #    SETransformerBlock(input_size,output=output_size, num_heads=7, reduction=4)
        #    for _ in range(2)
        #])
        self.down1 = SETransformerBlock(input_size,output=output_size,num_heads=7,reduction=4)
        self.down2 = SETransformerBlock(input_size,output=output_size,num_heads=7,reduction=4)
        self.nn1 =nn.Linear(2*input_size,input_size)
        self.up1 = SETransformerBlock(input_size,output=output_size,num_heads=7,reduction=4)
        self.nn2 = nn.Linear(2*input_size,input_size)
        self.up2 = SETransformerBlock(input_size,output=output_size,num_heads=7,reduction=4)
        self.embedding= nn.Linear(70,35)
        self.trans = TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta = self.gaussiandiffusion.beta
        self.mean = 0
        self.logvar = 0

        # DCT와 IDCT 행렬 미리 생성 후 캐싱
        N = 35  # 사용되는 차원 수
        self.dct_basis = self.create_dct_basis(N)
        self.idct_basis = self.create_idct_basis(N)

    def create_dct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        dct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)
        return dct_basis

    def create_idct_basis(self, N):
        k = torch.arange(N).view(1, -1)
        n = torch.arange(N).view(-1, 1)
        idct_basis = torch.sqrt(torch.tensor(2.0/N)) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)
        return idct_basis
    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def apply_dct(self, x):
        return self.dct(x)
    def apply_idct(self, y):
        return self.idct(y)
    def p_sample(self, xt,t,c):
        eps_theta,_,_,_= self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        #mean = self.apply_idct(mean)
        #return mean+(var**0.5)
        return mean
        #return mean + (var ** 0.5) * eps
    def dct(self, x):
        return torch.matmul(x, self.dct_basis.to(x.device))

    def idct(self, x):
        return torch.matmul(x, self.idct_basis.to(x.device))
    def forward(self, x, t, c):
        # 인코딩 부분
        z, mu, logvar = self.encoder(x, c)
        z2 = self.apply_dct(x)
        # 시간 임베딩 처리
        t_emb = self.time_embed(t)

        # DCT 적용 + noise 적용
        z_noise = self.apply_dct(z)
        noise = torch.randn_like(z)
        z_noise = self.q_sample(z_noise,t,noise)
        combined_input= torch.cat((z2,z_noise),dim=-1)
        combined_input=self.embedding(combined_input)
        # Transformer Layer 적용
        
        x = self.down1(combined_input)
        xd1= x
        x = self.down2(x)
        xd2 = x
        x=torch.cat((x,xd1),dim=-1)
        x = self.nn1(x)
        x = self.up1(x)
        x = torch.cat((x,xd2),dim=-1)
        x = self.nn2(x)
        x = self.up2(x)

        # Inference용 계산 (p_sample에서 사용됨)
        mean, var = self.q_xt_x0(x, t)

        # Concatenate encoded state
        zc = torch.cat((x, z), dim=-1)

        # 최종 IDCT 적용
        return self.apply_idct(x), mean, var,noise
class DenoiseDiffusion_Trans(nn.Module):
    def __init__(self, input_size, noise_steps,latent_size ,output_size):
        super().__init__()
        self.timesteps=noise_steps
        self.gaussiandiffusion = GaussianDiffusion(input_size, noise_steps, output_size)
        self.unet = UNet(35, 35)  # Assuming 3 input and 3 output channels for RGB can changin u-net
        self.encoder=Encoder(input_size,35)
        self.decoder=Decoder(35,70,5,output_size)
        
        self.trans =TransformerModel(in_channels=35)
        self.time_embed = TimeEmbedding(70)
        self.beta =self.gaussiandiffusion.beta
        self.mean=0
        self.logvar=0


    def q_xt_x0(self, x0, t):
        mean = self.gaussiandiffusion.alpha_hat[t] ** 0.5 * x0
        var = 1 - self.gaussiandiffusion.alpha_hat[t]
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    def dct(self,x):
        N = x.size(-1)  # Last dimension size
        device = x.device

        # Create DCT basis matrix
        k = torch.arange(N, device=device).view(1, -1)
        n = torch.arange(N, device=device).view(-1, 1)
        dct_basis = torch.sqrt(2 / N) * torch.cos(math.pi / N * (n + 0.5) * k)
        dct_basis[0] /= math.sqrt(2)  # Normalize the first row

        # Apply DCT basis to input data
        return torch.matmul(x, dct_basis)
    def idct(self,x):
        N = x.size(-1)  # Last dimension size
        device = x.device

        # Create IDCT basis matrix
        k = torch.arange(N, device=device).view(1, -1)
        n = torch.arange(N, device=device).view(-1, 1)
        idct_basis = torch.sqrt(2 / N) * torch.cos(math.pi / N * (k + 0.5) * n)
        idct_basis[0] /= math.sqrt(2)  # Normalize the first column

        # Apply IDCT basis to input data
        return torch.matmul(x, idct_basis)
    def p_sample(self, xt,t,c):
        eps_theta,_,_,_ = self.forward(xt,t,c)
        tt=torch.tensor(0).long()
        #eps_theta2 = eps_theta.repeat(1,5)
        alpha_hat = self.gaussiandiffusion.alpha_hat[tt]
        alpha = self.gaussiandiffusion.alpha[tt]
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** 0.5
        #eps_coef2 = eps_coef.repeat(1,5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gaussiandiffusion.beta[tt]
        eps = torch.randn_like(xt)
        return mean+(var**0.5)
        #return mean
        #return mean + (var ** 0.5) * eps
    def forward(self, x,t,c):
        z,mu,logvar=self.encoder(x,c)
        t_emb = self.time_embed(t)
        z_noise=self.dct(z)
        x=self.trans(z,z_noise)

        #x = self.unet(z,t_emb).to(device="cpu")
        tt = torch.tensor(t_emb).long().to(device="cpu")

        #tt = torch.tensor(t_emb).long().to(device="cuda")
        mean,var=self.q_xt_x0(x,tt)
        zc=torch.cat((x,z),dim=-1)
        return self.idct(x),mean,var
class DanceEncoder10(nn.Module):
    def __init__(self,pose_size,hidden_size,latent_size):
        super().__init__()
        self.input_size = pose_size*10
        self.pose_size = pose_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size,self.hidden_size)
        self.mu = nn.Linear(self.hidden_size,self.latent_size)
        self.std = nn.Linear(self.hidden_size,self.latent_size)


    def encode(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        data = torch.cat((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)
        out1 = self.fc1(F.elu(data))
        out2 = self.fc2(F.elu(out1))
        out3 = self.fc3(F.elu(out2))
        out4 = self.fc4(F.elu(out3))
        return self.mu(out4),self.std(out4) 

    def reparameterize(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu+std*eps

    def forward(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        mu , var = self.encode(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        z = self.reparameterize(mu,var)
        return z,mu,var

class DanceDecoder10(nn.Module):
    def __init__(self,latent_size,pose_size,hidden_size,output_size):
        super().__init__()
        self.latent_size = latent_size
        self.pose_size = pose_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.pose_size*5+self.latent_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.pose_size*5,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.pose_size*5,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+self.pose_size*5,self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size+self.pose_size*5,self.output_size)

    def forward(self,z,t1,t2,t3,t4,t5):
        out1 = self.fc1(F.elu(torch.cat((z,t1,t2,t3,t4,t5),dim=1)))
        out2 = self.fc2(F.elu(torch.cat((out1,t1,t2,t3,t4,t5),dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2,t1,t2,t3,t4,t5),dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3,t1,t2,t3,t4,t5),dim=1)))
        return self.fc5(torch.cat((out4,t1,t2,t3,t4,t5),dim=1))

class DanceVAE10(nn.Module):
    def __init__(self,pose_size,encode_hidden_size,latent_size,decode_hidden_size,output_size):
        super().__init__()
        self.encoder = DanceEncoder10(pose_size,encode_hidden_size,latent_size)
        self.decoder = DanceDecoder10(latent_size,pose_size,decode_hidden_size,output_size)
        self.pose_data_mu = 0
        self.pose_data_std = 0
    
    def encode(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        z,mu,logvar = self.encoder(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        return z,mu,logvar

    def forward(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        z,mu,logvar= self.encoder(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        return self.decoder(z,t1,t3,t5,t7,t9),mu,logvar

    def sample(self,z,t1,t3,t5,t7,t9):
        return self.decoder(z,t1,t3,t5,t7,t9)

    def set_normalize(self,pose_mu,pose_std):
        self.pose_data_mu = pose_mu
        self.pose_data_std = pose_std

    def normalize_pose(self,x):
        return (x-self.pose_data_mu)/self.pose_data_std
    
    def denormalize_pose(self,x):
        return x*self.pose_data_std+self.pose_data_mu

class TrackerEncoder(nn.Module):
    def __init__(self,tracker_size,hidden_size,latent_size):
        super().__init__()
        self.input_size = tracker_size*10
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.mu = nn.Linear(self.hidden_size+self.input_size,self.latent_size)
        self.std = nn.Linear(self.hidden_size+self.input_size,self.latent_size)

    def encode(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        data = torch.cat((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)
        out1 = self.fc1(F.elu(data))
        out2 = self.fc2(F.elu(torch.cat((out1,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)))
        return self.mu(torch.cat((out4,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)),self.std(torch.cat((out4,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10),dim=1)) 

    def reparameterize(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu+std*eps

    def forward(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        mu , var = self.encode(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        z = self.reparameterize(mu,var)
        return z,mu,var

class TrackerDecoder(nn.Module):
    def __init__(self,latent_size,tracker_size,hidden_size,output_size):
        super().__init__()
        self.tracker_size = tracker_size
        self.input_size = latent_size+self.tracker_size*5
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+latent_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+latent_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+latent_size,self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size+latent_size,self.output_size)

    def forward(self,z,t1,t3,t5,t7,t9):
        out1 = self.fc1(F.elu(torch.cat((z,t1,t3,t5,t7,t9),dim=1)))
        out2 = self.fc2(F.elu(torch.cat((out1,z),dim=1)))
        out3 = self.fc3(F.elu(torch.cat((out2,z),dim=1)))
        out4 = self.fc4(F.elu(torch.cat((out3,z),dim=1)))
        return self.fc5(torch.cat((out4,z),dim=1))

class TrackerVAE(nn.Module):
    def __init__(self,tracker_size,encode_hidden_size,latent_size,decode_hidden_size,output_size):
        super().__init__()
        self.encoder = TrackerEncoder(tracker_size,encode_hidden_size,latent_size)
        self.decoder = TrackerDecoder(latent_size,tracker_size,decode_hidden_size,output_size)
    
    def encode(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        z,mu,logvar = self.encoder(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        return z,mu,logvar

    def forward(self,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
        z,mu,logvar= self.encoder(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10)
        return self.decoder(z,t1,t3,t5,t7,t9),mu,logvar

    def sample(self,z,t1,t3,t5,t7,t9):
        return self.decoder(z,t1,t3,t5,t7,t9)


class TrackerAutoEncoder(nn.Module):
    def __init__(self,tracker_size,num_condition_frames,hidden_size,latent_size):
        super().__init__()
        self.input_size = tracker_size*num_condition_frames
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+self.input_size,self.latent_size)

    def encode(self,tracker_data):
        data = tracker_data.flatten(-2)
        out = F.elu(self.fc1(data))
        out = F.elu(self.fc2(torch.cat((out,data),dim=1)))
        out = F.elu(self.fc3(torch.cat((out,data),dim=1)))
        out = self.fc4((torch.cat((out,data),dim=1)))
        return out

    def forward(self,tracker_data):
        latent = self.encode(tracker_data)
        return latent 

class TrackerAutoDecoder(nn.Module):
    def __init__(self,latent_size,tracker_size,num_condition_frames,hidden_size,output_size):
        super().__init__()
        self.input_size = latent_size
        self.tracker_size = tracker_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size+self.input_size,self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size+self.input_size,self.output_size)

    def forward(self,latent,tracker):
        out = F.elu(self.fc1(latent))
        out = F.elu(self.fc2(torch.cat((out,latent),dim=1)))
        out = F.elu(self.fc3(torch.cat((out,latent),dim=1)))
        out = self.fc4(torch.cat((out,latent),dim=1))
        return out

class TrackerAuto(nn.Module):
    def __init__(self,tracker_size,num_condition_frames,encoder_hidden_size,latent_size,decoder_hidden_size,output_size):
        super().__init__()
        self.encoder = TrackerAutoEncoder(tracker_size,num_condition_frames,encoder_hidden_size,latent_size)
        self.decoder = TrackerAutoDecoder(latent_size,tracker_size,num_condition_frames,decoder_hidden_size,output_size)
        self.num_condition_frames = num_condition_frames
        #self.decoder = MixedDecoder(35,latent_size,decoder_hidden_size,0,1,2)

    def forward(self,tracker_data):
        z = self.encoder(tracker_data)
        return self.decoder(z,tracker_data[:,int(self.num_condition_frames/2-1),:])
    
class CNN(nn.Module):
    def __init__(self,tracker_size,condition_size,output_size):
        super().__init__()
        self.tracker_size = tracker_size*condition_size
        self.output_size = output_size
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,512,(1,3),stride=(1,3)),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512,32,1),
            torch.nn.ELU(),
        )
        self.fc1 = torch.nn.Linear(19200,1024)
        self.fc2 = torch.nn.Linear(1024,output_size)


    def forward(self,history):
        history = history.unsqueeze(1)
        #print(history.shape)
        out = self.layer1(history)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.shape[0],-1)
        #print(out.shape)
        out = F.elu(self.fc1(out))
        out = self.fc2(out)
        #print(out.shape)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderOnly(nn.Module):
    def __init__(self, input_size, condition_size, latent_size, hidden_size=512):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        real_input = self.input_size + condition_size
        self.fc1 = nn.Linear(real_input, self.hidden_size)
        self.fc2 = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.fc5 = nn.Linear(input_size + self.hidden_size, latent_size)

    def encode(self, input, condition_input):
        out1 = F.elu(self.fc1(torch.cat((input, condition_input), dim=1)))
        out2 = F.elu(self.fc2(torch.cat((input, out1), dim=1)))
        out3 = F.elu(self.fc3(torch.cat((input, out2), dim=1)))
        out4 = F.elu(self.fc4(torch.cat((input, out3), dim=1)))
        out5 = torch.cat((input, out4), dim=1)
        return self.fc5(out5)

    def forward(self, input, condition_input):
        output = self.encode(input, condition_input)
        return output


class VectorQuantizer2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer2, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        z = z.view(-1, self.embedding_dim)
        distances = (torch.sum(z ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(z, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = torch.index_select(self.embeddings.weight, dim=0, index=encoding_indices.squeeze()).view_as(z)

        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()

        return z_q, loss


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class MoEDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, hidden_dim):
        super(MoEDecoder, self).__init__()
        self.num_experts = num_experts

        # Experts: 각 전문가에 대해 별도의 네트워크를 생성
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        ) for _ in range(num_experts)])

        # Gating network: 각 전문가의 가중치를 결정
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Gating network를 통해 각 전문가의 가중치 계산
        gate_values = F.softmax(self.gating_network(x), dim=-1)

        # 각 전문가의 출력을 계산하고, 가중치를 곱한 후 합산
        output = sum(gate_values[:, i].unsqueeze(1) * self.experts[i](x) for i in range(self.num_experts))

        return output

class ACTEncoder(nn.Module):
    def __init__(self, pose_dim, action_dim, latent_dim, hidden_dim=256, use_transformer=False):
        super().__init__()
        input_dim = pose_dim + action_dim
        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.temporal_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, X, a):
        # X: [B, N, pose_dim], a: [B, action_dim]
        B, N, _ = X.shape
        a_expand = a.unsqueeze(1).expand(-1, N, -1)
        x_in = torch.cat([X, a_expand], dim=-1)  # [B, N, pose_dim + action_dim]
        if self.use_transformer:
            x_enc = self.temporal_encoder(x_in)
            x_enc = x_enc.mean(dim=1)
        else:
            _, x_enc = self.temporal_encoder(x_in)
            x_enc = x_enc.squeeze(0)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        return mu, logvar

class PriorNet(nn.Module):
    def __init__(self, pose_dim, action_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(pose_dim + action_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, X, a):
        # Use only last frame of X for conditioning
        x_last = X[:, -1, :]
        x_in = torch.cat([x_last, a], dim=-1)
        h = F.relu(self.fc1(x_in))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

class ACTDecoder(nn.Module):
    def __init__(self, pose_dim, action_dim, latent_dim, hidden_dim=256, use_transformer=False):
        super().__init__()
        input_dim = pose_dim + action_dim + latent_dim
        self.use_transformer = use_transformer
        if use_transformer:
            decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=4)
            self.temporal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        else:
            self.temporal_decoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, pose_dim)

    def forward(self, X, a, z, T_future):
        B, N, _ = X.shape
        a_expand = a.unsqueeze(1).expand(-1, T_future, -1)
        z_expand = z.unsqueeze(1).expand(-1, T_future, -1)
        x_last = X[:, -1, :].unsqueeze(1).expand(-1, T_future, -1)
        dec_input = torch.cat([x_last, a_expand, z_expand], dim=-1)  # [B, T_future, D]
        if self.use_transformer:
            tgt = dec_input
            memory = None  # Simple version (can modify)
            out = self.temporal_decoder(tgt, memory)
        else:
            out, _ = self.temporal_decoder(dec_input)
        Y_hat = self.fc_out(out)
        return Y_hat

class ActionVAE(nn.Module):
    def __init__(self, pose_dim, action_dim, latent_dim, hidden_dim=256, use_transformer=False):
        super().__init__()
        self.encoder = ACTEncoder(pose_dim, action_dim, latent_dim, hidden_dim, use_transformer)
        self.prior = PriorNet(pose_dim, action_dim, latent_dim, hidden_dim)
        self.decoder = ACTDecoder(pose_dim, action_dim, latent_dim, hidden_dim, use_transformer)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X, a, Y=None, T_future=None, training=True):
        if training:
            mu_post, logvar_post = self.encoder(X, a)
            z = self.reparameterize(mu_post, logvar_post)
            mu_prior, logvar_prior = self.prior(X, a)
            recon_Y = self.decoder(X, a, z, Y.shape[1])
            return recon_Y, mu_post, logvar_post, mu_prior, logvar_prior
        else:
            mu_prior, logvar_prior = self.prior(X, a)
            z = self.reparameterize(mu_prior, logvar_prior)
            pred_Y = self.decoder(X, a, z, T_future)
            return pred_Y
class VQVAE(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_size, output_size,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, input_frame=1, expert_num=8,
                 input_mult_frame=False, output_mult_frame=False):

        super(VQVAE, self).__init__()
        self.encoder = EncoderOnly(input_size * input_frame, input_size, latent_size, hidden_dim)

        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer2(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = MoEDecoder(embedding_dim + input_size, latent_size, expert_num, hidden_dim)

        if output_mult_frame == False:
            self.decoder2 = MoEDecoder(latent_size + input_size, output_size, expert_num, hidden_dim)
        else:
            self.decoder2 = MoEDecoder(latent_size + input_size, output_size * input_frame, expert_num, hidden_dim)

        self.data_std = 0
        self.data_avg = 0
        self.config = {'model_name': 'VQVAE',
                       'input': input_size,
                       'latent': latent_size,
                       'output_size': output_size,
                       'embeddings': n_embeddings,
                       'embedding_dim': embedding_dim,
                       'beta': beta,
                       'input_frame': input_frame,
                       'isRecursive': False
                       }

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def get_param(self):
        return self.config

    def set_normalization(self, std, avg):
        self.data_std = std
        self.data_avg = avg

    def normalize(self, t):
        return (t - self.data_avg) / self.data_std

    def denormalize(self, t):

        return t * self.data_std + self.data_avg

    def forward(self, x, condition2, verbose=False):

        z_e = self.encoder(x, condition2)
        # pdb.set_trace()
        # z_e = self.pre_quantization_conv(z_e)
        z_q, loss = self.vector_quantization(
            z_e)
        # x_hat = self.decoder(z_q, condition2)
        x_hat = self.decoder(torch.cat((z_q, condition2), dim=1))
        x_hat = self.decoder2(torch.cat((x_hat, condition2), dim=1))

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return loss, x_hat