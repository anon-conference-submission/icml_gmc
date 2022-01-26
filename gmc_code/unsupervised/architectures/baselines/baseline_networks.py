import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.autograd import Variable

###########################
#           MHD           #
###########################


mhd_image_data_size = torch.Size([1, 28, 28])
mhd_sound_data_size = torch.Size([1, 32, 128])
mhd_trajectory_data_size = torch.Size([200])



class MHDImageEncoder(nn.Module):
    """
    @param latent_dim: integer
                      number of latent dimensions
    """

    def __init__(self, latent_dim):
        super(MHDImageEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish())
        self.feature_extractor = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            Swish(),
            nn.Linear(512, latent_dim))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        return x


class MHDSoundEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDSoundEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Properties
        self.sound_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc_out = nn.Linear(2048, latent_dim)


    def forward(self, x):
        h = self.sound_features(x)
        h = h.view(h.size(0), -1)
        return self.fc_out(h)


class MHDTrajectoryEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDTrajectoryEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 512),
            Swish(),
        )

        self.fc_out = nn.Linear(512, latent_dim)


    def forward(self, x):
        h = self.trajectory_features(x)
        h = self.feature_extractor(h)
        return self.fc_out(h)


class MHDLabelEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDLabelEncoder, self).__init__()

        self.label_features = nn.Sequential(
            nn.Linear(10, 512),
            Swish(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 512),
            Swish()
        )

        self.fc_out = nn.Linear(512, latent_dim)


    def forward(self, x):
        h = self.label_features(x)
        h = h.view(h.size(0), -1)
        h = self.feature_extractor(h)
        return self.fc_out(h)


# Decoders

class MHDImageDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim):
        super(MHDImageDecoder, self).__init__()
        self.n_latents = latent_dim
        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 128 * 7 * 7),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 128, 7, 7)
        out = self.hallucinate(z)
        return out


class MHDSoundDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDSoundDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False))


    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 256, 8, 1)
        out = self.decoder(z)
        return F.sigmoid(out)



class MHDTrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDTrajectoryDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 200)
        )


    def forward(self, z):
        out = self.decoder(z)
        return F.sigmoid(out)



class MHDLabelDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDLabelDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 10))

    def forward(self, z):
        return self.decoder(z)


class MHDImageGaussianEncoder(nn.Module):

    def __init__(self, latent_dim):
        super(MHDImageGaussianEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.image_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            Swish(),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.image_features(x)
        h = h.view(h.size(0), -1)
        h = self.feature_extractor(h)
        return self.fc_mu(h), self.fc_logvar(h)



class MHDImageLaplacianEncoder(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim):
        super(MHDImageLaplacianEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            Swish())

        self.latent_dim = latent_dim

        self.fc_1 = nn.Linear(512, latent_dim)
        self.fc_2 = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        h = self.classifier(h)

        lv = self.fc_2(h)
        return self.fc_1(h), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class MHDSoundGaussianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDSoundGaussianEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Properties
        self.sound_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

    def forward(self, x):
        h = self.sound_features(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class MHDSoundLaplacianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDSoundLaplacianEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Properties
        self.sound_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )


        self.fc_1 = nn.Linear(2048, latent_dim)
        self.fc_2 = nn.Linear(2048, latent_dim)

    def forward(self, x):
        h = self.sound_features(x)
        h = h.view(h.size(0), -1)
        lv = self.fc_2(h)
        return self.fc_1(h), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class MHDTrajectoryGaussianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDTrajectoryGaussianEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 512),
            Swish(),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)


    def forward(self, x):
        h = self.trajectory_features(x)
        h = self.feature_extractor(h)
        return self.fc_mu(h), self.fc_logvar(h)


class MHDTrajectoryLaplacianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDTrajectoryLaplacianEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 512),
            Swish(),
        )

        self.fc_1 = nn.Linear(512, latent_dim)
        self.fc_2 = nn.Linear(512, latent_dim)


    def forward(self, x):
        h = self.trajectory_features(x)
        h = self.feature_extractor(h)
        lv = self.fc_2(h)
        return self.fc_1(h), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class MHDLabelGaussianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDLabelGaussianEncoder, self).__init__()

        self.label_features = nn.Sequential(
            nn.Linear(10, 512),
            Swish(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 512),
            Swish()
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)


    def forward(self, x):
        h = self.label_features(x)
        h = h.view(h.size(0), -1)
        h = self.feature_extractor(h)
        return self.fc_mu(h), self.fc_logvar(h)


class MHDLabelMMVAEGaussianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDLabelMMVAEGaussianEncoder, self).__init__()
        self.fc1   = nn.Linear(10, 64)
        self.fc2   = nn.Linear(64, 64)
        self.fc31  = nn.Linear(64, latent_dim)
        self.fc32  = nn.Linear(64, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc31(h), F.softplus(self.fc32(h)) + Constants.eta



class MHDImageGaussianDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDImageGaussianDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 128 * 7 * 7),
            Swish())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 128, 7, 7)
        out = self.decoder(z)
        return out


class MHDImageLaplacianDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim):
        super(MHDImageLaplacianDecoder, self).__init__()
        self.n_latents = latent_dim
        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 128 * 7 * 7),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False))

    def forward(self, z):
        h = self.upsampler(z)
        h = h.view(-1, 128, 7, 7)
        out = self.hallucinate(h)
        d = torch.sigmoid(out.view(*z.size()[:-1], *mhd_image_data_size))
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)



class MHDSoundGaussianDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDSoundGaussianDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False))


    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 256, 8, 1)
        out = self.decoder(z)
        return F.sigmoid(out)

class MHDSoundLaplacianDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDSoundLaplacianDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False))


    def forward(self, z):

        h = self.upsampler(z)
        h = h.view(-1, 256, 8, 1)
        out = self.decoder(h)
        d = torch.sigmoid(out.view(*z.size()[:-1], *mhd_sound_data_size))
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)


class MHDTrajectoryGaussianDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDTrajectoryGaussianDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 200)
        )


    def forward(self, z):
        out = self.decoder(z)
        return F.sigmoid(out)


class MHDTrajectoryLaplacianDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDTrajectoryLaplacianDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 200)
        )

    def forward(self, z):
        out = self.decoder(z)
        d = torch.sigmoid(out.view(*z.size()[:-1], *mhd_trajectory_data_size))
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)



class MHDLabelGaussianDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MHDLabelGaussianDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 10))

    def forward(self, z):
        return self.decoder(z)



class MHDMuseTopEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MHDMuseTopEncoder, self).__init__()
        self.fc1   = nn.Linear(input_dim, 128)
        self.fc2   = nn.Linear(128, 128)
        self.fc31  = nn.Linear(128, latent_dim)
        self.fc32  = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class MHDMuseTopDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super(MHDMuseTopDecoder, self).__init__()
        self.fc1   = nn.Linear(latent_dim, 128)
        self.fc2   = nn.Linear(128, 128)
        self.fc3   = nn.Linear(128, out_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        return self.fc3(h)



class MHDMessageEncoder(nn.Module):
    def __init__(self, input_dim, message_dim):
        super(MHDMessageEncoder, self).__init__()
        self.fc   = nn.Linear(input_dim, message_dim)

    def forward(self, x):
        return self.fc(x)


class MHDNexusEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MHDNexusEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MHDNexusDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super(MHDNexusDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )

        self.fc_out = nn.Linear(512, out_dim)

    def forward(self, z):
        h = self.decoder(z)
        return self.fc_out(h)


# Intermediate
class MHDIntermediate(nn.Module):
    def __init__(self, fusion_dim, latent_dim):
        super(MHDIntermediate, self).__init__()
        self.fc1   = nn.Linear(fusion_dim, latent_dim)
        self.fc2   = nn.Linear(latent_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        return self.relu(self.fc2(h))


# Fusion
class MHDFusion(nn.Module):
    def __init__(self, fusion_dim, latent_dim):
        super(MHDFusion, self).__init__()
        self.fc1   = nn.Linear(fusion_dim, 2*latent_dim)
        self.fc2   = nn.Linear(2*latent_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        return self.relu(self.fc2(h))


# Head
class MHDHead(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(MHDHead, self).__init__()
        self.fc1   = nn.Linear(latent_dim, 20)
        self.fc2   = nn.Linear(20, n_classes)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        return self.relu(self.fc2(h))




"""


Extra components


"""

class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2*math.pi)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar)
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


def logvar_to_std(logvar):
    return torch.exp(0.5 * logvar)

def KLD_gaussian(q_mu, q_logvar, p_mu, p_logvar, eps=1E-10):
    """
    Symmetric KL (q(x) || p(x))
    :param p_mu
    :param p_logvar
    :param q_mu
    :param q_logvar
    :return:
    """

    q_dist = Normal(loc=q_mu, scale=logvar_to_std(q_logvar))
    p_dist = Normal(loc=p_mu, scale=logvar_to_std(p_logvar))

    kl = kl_divergence(q_dist, p_dist)

    return torch.sum(kl, dim=-1)


def sym_KLD_gaussian(p_mu, p_logvar, q_mu, q_logvar):
    """
    KL (p(x) || q(x))
    :param p_mu
    :param p_logvar
    :param q_mu
    :param q_logvar
    :return:
    """
    p_var = torch.exp(p_logvar)
    q_var = torch.exp(q_logvar)

    mu_diff = torch.pow(p_mu - q_mu, 2)
    first_term = 0.5 * (mu_diff + q_var) / p_var
    second_term = 0.5 * (mu_diff + p_var) / q_var

    return torch.sum(first_term + second_term - 1, dim=-1)