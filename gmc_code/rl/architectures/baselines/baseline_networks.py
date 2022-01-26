import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.autograd import Variable


###########################
#         Pendulum        #
###########################


pendulum_image_data_size = torch.Size([1, 28, 28])
pendulum_sound_data_size = torch.Size([1, 32, 128])



# Pendulum
class PendulumJointEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumJointEncoder, self).__init__()
        # Variables
        self.latent_dim = latent_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = self.n_stack * self.sound_channels * self.sound_length

        self.img_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish())

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish())

        self.classifier = nn.Sequential(
            nn.Linear(14400 + 50, 128),
            Swish())

        self.fc_out = nn.Linear(128, latent_dim)

    def forward(self, x):

        x_img, x_snd = x[0], x[1]

        if x_img is None:
            x_img = torch.zeros([x_snd.size(0), x_snd.size(1), 60, 60]).to(x_snd.device)
        if x_snd is None:
            x_snd = torch.zeros([x_img.size(0), x_img.size(1), 3, 2]).to(x_img.device)

        x_img = self.img_features(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        x = self.classifier(torch.cat((x_img, x_snd), dim=-1))
        return self.fc_out(x)


class PendulumImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(14400, 128),
            Swish())

        self.latent_dim = latent_dim

        self.fc_out = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.fc_out(x)


class PendulumSoundEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumSoundEncoder, self).__init__()

        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = self.n_stack*self.sound_channels*self.sound_length

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish())

        self.fc_out = nn.Linear(50, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.snd_features(x)
        return self.fc_out(h)



class PendulumJointGaussianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumJointGaussianEncoder, self).__init__()
        # Variables
        self.latent_dim = latent_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = self.n_stack * self.sound_channels * self.sound_length

        self.img_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish())

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish())

        self.classifier = nn.Sequential(
            nn.Linear(14400 + 50, 128),
            Swish())

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):

        x_img, x_snd = x[0], x[1]

        if x_img is None:
            x_img = torch.zeros([x_snd.size(0), x_snd.size(1), 60, 60]).to(x_snd.device)
        if x_snd is None:
            x_snd = torch.zeros([x_img.size(0), x_img.size(1), 3, 2]).to(x_img.device)

        x_img = self.img_features(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        x = self.classifier(torch.cat((x_img, x_snd), dim=-1))
        return self.fc_mu(x), self.fc_logvar(x)


class PendulumImageGaussianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumImageGaussianEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(14400, 128),
            Swish())

        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.fc_mu(x), self.fc_logvar(x)



class PendulumImageLaplacianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumImageLaplacianEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(14400, 128),
            Swish())

        self.latent_dim = latent_dim

        self.fc_1 = nn.Linear(128, latent_dim)
        self.fc_2 = nn.Linear(128, latent_dim)

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        h = self.features(x)
        h = h.view(h.size(0), -1)
        h = self.classifier(h)
        lv = self.fc_2(h)
        return self.fc_1(h), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class PendulumSoundGaussianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumSoundGaussianEncoder, self).__init__()

        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = self.n_stack*self.sound_channels*self.sound_length

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish())

        self.fc_mu = nn.Linear(50, latent_dim)
        self.fc_logvar = nn.Linear(50, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.snd_features(x)
        return self.fc_mu(h), self.fc_mu(h)


class PendulumSoundLaplacianEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumSoundLaplacianEncoder, self).__init__()

        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = self.n_stack * self.sound_channels * self.sound_length

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish())

        self.fc_1 = nn.Linear(50, latent_dim)
        self.fc_2 = nn.Linear(50, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.snd_features(x)
        lv = self.fc_2(h)
        return self.fc_1(h), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta



class PendulumJointDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumJointDecoder, self).__init__()

        # Variables
        self.latent_dim = latent_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = self.n_stack * self.sound_channels * self.sound_length

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 128),
            Swish(),
            nn.Linear(128, 14400 + 50),
            Swish())

        self.img_hallucinate = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(32, 2, 4, 2, 1, bias=False),
            nn.Sigmoid())

        self.snd_hallucinate = nn.Sequential(
            nn.Linear(50, 50),
            Swish(),
            nn.Linear(50, self.n_stack * self.sound_channels * self.sound_length),
        )

    def forward(self, z):
        z = self.upsampler(z)
        z_img = z[:, :14400].view(-1, 64, 15, 15)
        z_snd = z[:, 14400:]
        out_img = self.img_hallucinate(z_img)
        out_snd = self.snd_hallucinate(z_snd)
        out_snd = out_snd.view(-1, self.n_stack, self.sound_channels, self.sound_length)
        return out_img, torch.sigmoid(out_snd)


class PendulumImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumImageDecoder, self).__init__()
        self.n_latents = latent_dim
        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 128),
            Swish(),
            nn.Linear(128, 14400),
            Swish())

        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(32, 2, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 64, 15, 15)
        out = self.hallucinate(z)
        return out



class PendulumLaplacianImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumLaplacianImageDecoder, self).__init__()
        self.n_latents = latent_dim
        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 128),
            Swish(),
            nn.Linear(128, 14400),
            Swish())

        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(32, 2, 4, 2, 1, bias=False))

    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 64, 15, 15)
        out = self.hallucinate(z)
        d = torch.sigmoid(out.view(*z.size()[:-1], *pendulum_image_data_size))
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)


class PendulumSoundDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumSoundDecoder, self).__init__()

        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish(),
            nn.Linear(50, self.n_stack * self.sound_channels * self.sound_length))

    def forward(self, z):
        h = self.upsampler(z)
        h = h.view(-1, self.n_stack, self.sound_channels, self.sound_length)
        return torch.sigmoid(h)





class PendulumSoundLaplacianDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumSoundLaplacianDecoder, self).__init__()

        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2

        self.upsampler = nn.Sequential(
            nn.Linear(latent_dim, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish(),
            nn.Linear(50, self.n_stack * self.sound_channels * self.sound_length))

    def forward(self, z):
        h = self.upsampler(z)
        h = h.view(-1, self.n_stack, self.sound_channels, self.sound_length)
        d = torch.sigmoid(h.view(*z.size()[:-1], *pendulum_sound_data_size))
        d = d.clamp(Constants.eta, 1 - Constants.eta)
        return d, torch.tensor(0.75).to(z.device)





# Sound
class PendulumMUSESoundEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PendulumMUSESoundEncoder, self).__init__()

        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2

        self.unrolled_sound_input = self.n_stack*self.sound_channels*self.sound_length

        self.fc1   = nn.Linear(self.unrolled_sound_input, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2   = nn.Linear(50, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc31  = nn.Linear(50, latent_dim)
        self.fc32  = nn.Linear(50, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        return self.fc31(h), self.fc32(h)


class PendulumMUSESoundDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim):
        super(PendulumMUSESoundDecoder, self).__init__()

        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2

        # Net
        self.fc1   = nn.Linear(latent_dim, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2   = nn.Linear(50, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3   = nn.Linear(50, self.n_stack*self.sound_channels*self.sound_length)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.bn1(self.fc1(z)))
        h = self.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        h = h.view(-1, self.n_stack, self.sound_channels, self.sound_length)
        return torch.sigmoid(h)




class PendulumMuseTopEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PendulumMuseTopEncoder, self).__init__()
        self.fc1   = nn.Linear(input_dim, 256)
        self.fc2   = nn.Linear(256, 256)
        self.fc3   = nn.Linear(256, 256)
        self.fc31  = nn.Linear(256, latent_dim)
        self.fc32  = nn.Linear(256, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc31(h), self.fc32(h)


class PendulumMuseTopDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, latent_dim, out_dim):
        super(PendulumMuseTopDecoder, self).__init__()
        self.fc1   = nn.Linear(latent_dim, 256)
        self.fc2   = nn.Linear(256, 256)
        self.fc3   = nn.Linear(256, 256)
        self.fc4   = nn.Linear(256, out_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc4(h)



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



"""


Approximation stuff - Maybe there's a better place for this....


"""



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