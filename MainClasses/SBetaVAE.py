from torch import nn
import torch


class LinearAttention(nn.Module):
    def __init__(self, n_latents, activation=nn.Softmax):
        super(LinearAttention, self).__init__()
        self.att_layer = nn.Sequential(
            nn.Linear(n_latents, n_latents * 4),
            nn.ReLU(),
            nn.Linear(n_latents * 4, n_latents),
            activation()
        )

    def forward(self, inputs):
        alphas = self.att_layer(inputs)
        x = alphas * inputs
        return x, alphas


class Encoder(nn.Module):
    def __init__(self, ts, feature_dim, in_dim, latents_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=128,
                            num_layers=3,
                            batch_first=True,
                            dropout=0.3)
        self.linear_1 = nn.Linear(128, latents_dim)
        self.linear_2 = nn.Linear(128, latents_dim)

        self.ts = ts
        self.feature_dim = feature_dim
    def sampling(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, inputs):
        inputs = inputs.view(-1, self.ts, self.feature_dim)
        x, c = self.lstm(inputs, None)
        z_mean = self.linear_1(x[:, -1, :])
        z_log_var = self.linear_2(x[:, -1, :])
        z = self.sampling(z_mean, z_log_var)
        return z, (z_mean, z_log_var)


class Decoder(nn.Module):
    def __init__(self, ts, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=128,
                            num_layers=3,
                            batch_first=True,
                            dropout=0.3)
        self.linear_1 = nn.Linear(128, 256)
        self.out = nn.Linear(256, out_dim)
        self.ts = ts

    def forward(self, inputs):
        x = torch.relu(inputs)
        x = torch.repeat_interleave(x, self.ts, dim=1).reshape(-1, self.ts, x.shape[-1])
        x, _ = self.lstm(x, None)
        x = self.linear_1(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        return x


class Detector(nn.Module):
    def __init__(self, n_latents):
        super(Detector, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.linear_att = LinearAttention(n_latents=n_latents)
        self.linear_1 = nn.Linear(n_latents, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, inputs):
        x = torch.relu(inputs)
        z_star, alpha = self.linear_att(x)
        x = self.linear_1(z_star)
        h = self.dropout(x)
        x = torch.relu(h)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x, z_star, alpha, h


class SBetaVAE(nn.Module):
    def __init__(self, options):
        super(SBetaVAE, self).__init__()
        self.name = ''
        self.op = options
        self.K = self.op.num_events
        self.return_h = options.return_h
        inputs_dim = self.op.feature_dim
        if self.K == 5 or self.K == 6:
            latents_dim = 15
        elif self.K == 10:
            latents_dim = 30
        elif self.K == 15:
            latents_dim = 45
        elif self.K == 20:
            latents_dim = 60
        else:
            return
        ts = self.op.time_step
        feature_dim = self.op.feature_dim

        self.encoder = Encoder(ts, feature_dim, inputs_dim, latents_dim)
        self.decoder = Decoder(ts, latents_dim, inputs_dim)
        self.att_layer = LinearAttention(latents_dim)
        self.detectors = nn.ModuleList()
        for k in range(self.K):
            self.detectors.add_module('e_' + str(k) + '_detector', Detector(n_latents=latents_dim))

    def forward(self, inputs):
        detectors_out = torch.Tensor().cuda()
        z_stars = torch.Tensor().cuda()
        alphas = torch.Tensor().cuda()
        bottleneck_f = torch.Tensor().cuda()
        z, (mu, log_var) = self.encoder(inputs)
        dec_out = self.decoder(z)

        for detector in self.detectors:
            detec_out, z_star, alpha, h = detector(z)
            detectors_out = torch.cat([detectors_out, detec_out], dim=-1)
            z_stars = torch.cat([z_stars, z_star.unsqueeze(-1)], dim=-1)
            alphas = torch.cat([alphas, alpha.unsqueeze(-1)], dim=-1)
            bottleneck_f = torch.cat([bottleneck_f, h.unsqueeze(-1)], dim=-1)

        if self.return_h:
            return dec_out, detectors_out, z_stars, alphas, (mu, log_var), bottleneck_f
        else:
            return dec_out, detectors_out, z_stars, alphas, (mu, log_var)