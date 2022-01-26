import torch.distributions as dist
from pytorch_lightning import LightningModule
from gmc_code.unsupervised.architectures.baselines.baseline_networks import *


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


class MMVAE(LightningModule):
    def __init__(self, name, latent_dim):

        super(MMVAE, self).__init__()

        # Variables
        self.name = name
        self.latent_dim = latent_dim
        self.encoders = []
        self.decoders = []
        self.post_dists = []
        self.prior_dist = None
        self.like_dists = []
        self._pz_params = None

    def encode(self, x, sample=False):
        zss = []

        for i in range(len(x)):
            if x[i] is not None:
                mod_qz_x_params = self.encoders[i](x[i])
                qz_x = self.post_dists[i](*mod_qz_x_params)
                if sample:
                    zss.append(qz_x.rsample())
                else:
                    zss.append(mod_qz_x_params[0])

        moe_z = torch.mean(torch.stack(zss, dim=0), dim=0)
        return moe_z


    def loss_function(self, data, qz_xs, px_zs_params, zss, qz_params, train_params):
        return

    def training_step(self, data, train_params):

        # Forward pass through the encoders
        qz_xs, px_zs_params, zss, qz_params = self.forward(data, n_samples=train_params['n_training_samples'])
        loss = self.loss_function(data, qz_xs, px_zs_params, zss, qz_params, train_params)

        tqdm_dict = {'loss': loss}

        return loss, tqdm_dict

    def validation_step(self, data, train_params):

        # Forward pass through the encoders
        qz_xs, px_zs_params, zss, qz_params = self.forward(data, n_samples=train_params['n_training_samples'])
        loss = self.loss_function(data, qz_xs, px_zs_params, zss, qz_params, train_params)

        tqdm_dict = {'loss': loss}

        return tqdm_dict

    def forward(self, data, n_samples=1):
        qz_params = []
        qz_xs, zss = [], []
        px_zs_params = [[None for _ in range(len(self.encoders))] for _ in range(len(self.encoders))]

        for i in range(len(data)):
            mod_qz_x_params = self.encoders[i](data[i])
            qz_x = self.post_dists[i](*mod_qz_x_params)
            zs = qz_x.rsample(torch.Size([n_samples]))
            px_z = self.decoders[i](zs)

            qz_params.append(mod_qz_x_params)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs_params[i][i] = px_z

        for i in range(len(data)):
            for j in range(len(data)):
                if i != j:
                    px_zs_params[i][j] = self.decoders[j](zss[i])

        return qz_xs, px_zs_params, zss, qz_params


class MhdMMVAE(MMVAE):
    def __init__(self, name, latent_dim):
        super(MhdMMVAE, self).__init__(name, latent_dim)
        self.name = name
        self.latent_dim = latent_dim

        # Encoders
        self.image_encoder = MHDImageLaplacianEncoder(latent_dim)
        self.sound_encoder = MHDSoundLaplacianEncoder(latent_dim)
        self.trajectory_encoder = MHDTrajectoryLaplacianEncoder(latent_dim)
        self.label_encoder = MHDLabelMMVAEGaussianEncoder(latent_dim)

        # Decoders
        self.image_decoder = MHDImageLaplacianDecoder(latent_dim)
        self.sound_decoder = MHDSoundLaplacianDecoder(latent_dim)
        self.trajectory_decoder = MHDTrajectoryLaplacianDecoder(latent_dim)
        self.label_decoder = MHDLabelGaussianDecoder(latent_dim)

        self.encoders = [self.image_encoder, self.sound_encoder, self.trajectory_encoder, self.label_encoder]
        self.decoders = [self.image_decoder, self.sound_decoder, self.trajectory_decoder, self.label_decoder]

        # Dists
        self.post_dists = [dist.Laplace, dist.Laplace, dist.Laplace, dist.Normal]
        self.prior_dist = [dist.Laplace, dist.Laplace, dist.Laplace, dist.Normal]
        self.like_dists = [dist.Laplace, dist.Laplace, dist.Laplace, None]

        self._mod_0_pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=True)
        ])

        self._mod_1_pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=True)
        ])

        self._mod_2_pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=True)
        ])

        self._mod_3_pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=True)
        ])

        self.pz_params = [self._mod_0_pz_params, self._mod_1_pz_params, self._mod_2_pz_params, self._mod_3_pz_params]

    def get_pz_params(self, mod):
        if mod == 0:
            return self._mod_0_pz_params[0], F.softmax(self._mod_0_pz_params[1], dim=1) * self._mod_0_pz_params[1].size(-1)
        elif mod == 1:
            return self._mod_1_pz_params[0], F.softmax(self._mod_1_pz_params[1], dim=1) * self._mod_1_pz_params[1].size(-1)
        elif mod == 2:
            return self._mod_2_pz_params[0], F.softmax(self._mod_2_pz_params[1], dim=1) * self._mod_2_pz_params[1].size(-1)
        elif mod == 3:
            return self._mod_3_pz_params[0], F.softplus(self._mod_3_pz_params[1]) + Constants.eta
        else:
            raise ValueError("Wrong mod selected for pz params of MMVAE model in MNIST dataset")



    def loss_function(self, data, qz_xs, px_zs_params, zss, qz_params, train_params):

        # We employ the looser DREG loss function as employed in Shi et al. (2019)

        qz_xs_ = []

        for post_dist, qz_pr in zip(self.post_dists, qz_params):
            qz_xs_.append(post_dist(*[p.detach() for p in qz_pr]))

        lws = []

        for i in range(len(data)):
            lpz = self.prior_dist[i](*self.get_pz_params(mod=i)).log_prob(zss[i]).sum(-1)
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[i]).sum(-1) for qz_x in qz_xs_]))

            lpx_z = []
            for j in range(len(px_zs_params[i])):

                # Image likelihood
                if j == 0 :
                    px_z = self.like_dists[j](*px_zs_params[i][j])

                    loss = px_z.log_prob(data[j]).mul(train_params['coefs'][j]).sum(-1).sum(-1).sum(-1)
                    lpx_z.append(loss)

                # Sound likelihood
                elif j == 1:
                    px_z = self.like_dists[j](*px_zs_params[i][j])

                    loss = px_z.log_prob(data[j]).mul(train_params['coefs'][j]).sum(-1).sum(-1).sum(-1)
                    lpx_z.append(loss)

                # Trajectory likelihood
                elif j == 2:
                    px_z = self.like_dists[j](*px_zs_params[i][j])

                    loss = px_z.log_prob(data[j]).mul(train_params['coefs'][j]).sum(-1)
                    lpx_z.append(loss)

                # Label likelihood - Categorical distribution
                else:
                    _, targets = data[j].max(dim=1)
                    loss = -F.cross_entropy(input=px_zs_params[i][j].view(-1, px_zs_params[i][j].size(-1)),
                                             target=targets.expand(px_zs_params[i][j].size()[:-1]).long().view(-1),
                                             reduction='none',
                                             ignore_index=0).mul(train_params['coefs'][j])

                    loss = loss.view(*px_zs_params[i][j].shape[:-1])
                    lpx_z.append(loss)

            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)

        lw = torch.cat(lws, 0)
        return -lw.mean(0).mean(-1)