
import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F

import global_v as glv


class TAID_decoding(nn.Module):
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=T, out_channels=T,
                              kernel_size=kernel_size_t, padding='same', bias=False)
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding='same', bias=False)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        x_seq = x_seq.permute(4, 0, 1, 2, 3)
        x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        x_c = x.permute(0, 2, 1)
        # print('x', x.shape)  # B,T,C
        # print('x_c', x_c.shape)  # B,C,T
        conv_t_out = self.conv(x).permute(1, 0, 2)

        # print('conv_t_out', conv_t_out.shape)
        # print('conv_c_out', conv_c_out.shape)
        out = self.sigmoid(conv_t_out)
        y_seq = x_seq * out[:, :, :, None, None]
        y_seq = torch.sum(y_seq, dim=0)
        return y_seq



class FSVAE_TAID(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike(),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)

        self.posterior = PosteriorBernoulliSTBP(self.k)

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=LIFSpike())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=LIFSpike()),
            tdConvTranspose(hidden_dims[-1],
                            out_channels=glv.network_config['in_channels'],
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = TAID_decoding(1,4,16,1)

        self.psp = PSP()

    def forward(self, x, scheduled=False):
        sampled_z, q_z, p_z = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z)
        return x_recon, q_z, p_z, sampled_z

    def encode(self, x, scheduled=False):
        x = self.encoder(x)  # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x)  # (N,latent_dim,T)
        sampled_z, q_z = self.posterior(latent_x)  # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)

        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def decode(self, z):
        result = self.decoder_input(z)  # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)
        result = self.decoder(result)  # (N,C,H,W,T)
        result = self.final_layer(result)  # (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))
        return out

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z

    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2)  # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2)  # (N, latent_dim, T)

        # kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber) - self.psp(p_z_ber)) ** 2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': mmd_loss}

    def loss_function_kld(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        prob_q = torch.mean(q_z, dim=2)  # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2)  # (N, latent_dim, T)

        kld_loss = prob_q * torch.log((prob_q + 1e-2) / (prob_p + 1e-2)) + (1 - prob_q) * torch.log(
            (1 - prob_q + 1e-2) / (1 - prob_p + 1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1, 2)))

        loss = recons_loss + 1e-4 * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': kld_loss}

    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4, 4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p - init_p) * epoch / max_epoch + init_p


class FSVAELarge_TAID(FSVAE_TAID):
    def __init__(self):
        super(FSVAE_TAID, self).__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)

        self.posterior = PosteriorBernoulliSTBP(self.k)

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=LIFSpike())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=LIFSpike()),
            tdConvTranspose(hidden_dims[-1],
                            out_channels=glv.network_config['in_channels'],
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = TAID_decoding(1,4,16,1)

        self.psp = PSP()
import global_v as glv

from .HHneuron import *
class HH(nn.Module):
    def __init__(self):
        super(HH, self).__init__()
        self.HH = HHNode(tau=0.25, v_threshold=0.2, v_reset=0.)

    def forward(self, x):
        # print(x.shape)
        # print(x[...,1].shape)
        #         for step in range(x.size(-1)):
        #             # print(self.HH(x[...,step]).shape)
        #             x[..., step] = self.HH(x[...,step])

        out = self.HH(x)
        return out


class FSVAE(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=HH(),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=HH())

        self.prior = PriorBernoulliSTBP(self.k)

        self.posterior = PosteriorBernoulliSTBP(self.k)

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=HH())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=HH())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=HH()),
            tdConvTranspose(hidden_dims[-1],
                            out_channels=glv.network_config['in_channels'],
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

    def forward(self, x, scheduled=False):
        sampled_z, q_z, p_z = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z)
        return x_recon, q_z, p_z, sampled_z

    def encode(self, x, scheduled=False):
        x = self.encoder(x)  # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x)  # (N,latent_dim,T)
        sampled_z, q_z = self.posterior(latent_x)  # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)

        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def decode(self, z):
        result = self.decoder_input(z)  # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)
        result = self.decoder(result)  # (N,C,H,W,T)
        result = self.final_layer(result)  # (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))
        return out

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z

    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2)  # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2)  # (N, latent_dim, T)

        # kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber) - self.psp(p_z_ber)) ** 2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': mmd_loss}

    def loss_function_kld(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        prob_q = torch.mean(q_z, dim=2)  # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2)  # (N, latent_dim, T)

        kld_loss = prob_q * torch.log((prob_q + 1e-2) / (prob_p + 1e-2)) + (1 - prob_q) * torch.log(
            (1 - prob_q + 1e-2) / (1 - prob_p + 1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1, 2)))

        loss = recons_loss + 1e-4 * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': kld_loss}

    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4, 4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p - init_p) * epoch / max_epoch + init_p


class FSVAELarge(FSVAE):
    def __init__(self):
        super(FSVAE, self).__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=HH())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=HH())

        self.prior = PriorBernoulliSTBP(self.k)

        self.posterior = PosteriorBernoulliSTBP(self.k)

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=HH())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=HH())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=HH()),
            tdConvTranspose(hidden_dims[-1],
                            out_channels=glv.network_config['in_channels'],
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()