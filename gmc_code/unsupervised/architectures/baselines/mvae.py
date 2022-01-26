from pytorch_lightning import LightningModule
from gmc_code.unsupervised.architectures.baselines.baseline_networks import *


class MVAE(LightningModule):
    def __init__(self, name, latent_dim):

        super(MVAE, self).__init__()

        # Variables
        self.name = name
        self.latent_dim = latent_dim
        self.encoders = []
        self.decoders = []
        self.poe = ProductOfExperts()

    def encode(self, x, sample=False):
        mu, logvar = self.infer(x)
        # Return a latent sample
        if sample:
            return self.reparametrize(mu, logvar)
        else:
            return mu

    def infer(self, data):

        for mod_data in data:
            if mod_data is not None:
                batch_size = mod_data.size(0)
                break
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA

        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.latent_dim), use_cuda=use_cuda)

        # Encode modality-specific data
        for mod_data, mod_encoder in zip(data, self.encoders):
            if mod_data is not None:
                mod_mu, mod_logvar = mod_encoder(mod_data)
                mu = torch.cat((mu, mod_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, mod_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.poe(mu, logvar)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        # Sample epsilon from a random gaussian with 0 mean and 1 variance
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        # Check if cuda is selected
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()

        # z = std * epsilon + mu
        return mu.addcmul(std, epsilon)

    def loss_function(self, recon_data, data, mu, logvar, train_params):
        return

    def training_step(self, data, train_params):

        # Forward pass through the encoders
        recon, mu, logvar = self.forward(data)
        f_loss = self.loss_function(recon, data, mu, logvar, train_params)

        # # Subsampling
        mod_losses = []
        for i in range(len(data)):
            ss_data = [None] * len(data)
            ss_data[i] = data[i]
            mod_recon, mod_mu, mod_logvar = self.__call__(ss_data)
            mod_loss = self.loss_function(mod_recon, ss_data, mod_mu, mod_logvar, train_params)
            mod_losses.append(mod_loss)

        loss = f_loss + sum(mod_losses)

        tqdm_dict = {'loss': loss}

        return loss, tqdm_dict

    def validation_step(self, data, train_params):

        ## Forward pass through the encoders
        recon, mu, logvar = self.forward(data)
        f_loss = self.loss_function(recon, data, mu, logvar, train_params)

        # Subsampling
        mod_losses = []
        for i in range(len(data)):
            ss_data = [None] * len(data)
            ss_data[i] = data[i]
            mod_recon, mod_mu, mod_logvar = self.__call__(ss_data)
            mod_loss = self.loss_function(mod_recon, ss_data, mod_mu, mod_logvar, train_params)
            mod_losses.append(mod_loss)

        loss = f_loss + sum(mod_losses)

        tqdm_dict = {'loss': loss}

        return tqdm_dict

    def forward(self, data):
        recons = []

        # Encode all available modalities
        mu, logvar = self.infer(data)
        z = self.reparametrize(mu, logvar)

        # Decode for each modality from the joint encoder
        for mod_decoder in self.decoders:
            recons.append(mod_decoder(z))

        return recons, mu, logvar


class MhdMVAE(MVAE):
    def __init__(self, name, latent_dim):
        super(MhdMVAE, self).__init__(name, latent_dim)
        self.name = name
        self.latent_dim = latent_dim

        self.image_encoder = MHDImageGaussianEncoder(latent_dim)
        self.sound_encoder = MHDSoundGaussianEncoder(latent_dim)
        self.trajectory_encoder = MHDTrajectoryGaussianEncoder(latent_dim)
        self.label_encoder = MHDLabelGaussianEncoder(latent_dim)

        self.image_decoder = MHDImageGaussianDecoder(latent_dim)
        self.sound_decoder = MHDSoundGaussianDecoder(latent_dim)
        self.trajectory_decoder = MHDTrajectoryGaussianDecoder(latent_dim)
        self.label_decoder = MHDLabelGaussianDecoder(latent_dim)

        self.encoders = [self.image_encoder, self.sound_encoder, self.trajectory_encoder, self.label_encoder]
        self.decoders = [self.image_decoder, self.sound_decoder, self.trajectory_decoder, self.label_decoder]

    def loss_function(self, recon_data, data, mu, logvar, train_params):

        if data[0] is not None:
            img_recon_loss = train_params['lambda_x0'] * torch.sum(F.mse_loss(recon_data[0].view(recon_data[0].size(0), -1),
                                                                            data[0].view(data[0].size(0), -1),
                                                                            reduction='none'), dim=-1)
        else:
            img_recon_loss = 0.


        if data[1] is not None:
            sound_recon_loss = train_params['lambda_x1'] * torch.sum(F.mse_loss(recon_data[1].view(recon_data[1].size(0), -1),
                                                                            data[1].view(data[1].size(0), -1),
                                                                            reduction='none'), dim=-1)
        else:
            sound_recon_loss = 0.


        if data[2] is not None:
            traj_recon_loss = train_params['lambda_x2'] * torch.sum(F.mse_loss(recon_data[2].view(recon_data[2].size(0), -1),
                                                                            data[2].view(data[2].size(0), -1),
                                                                            reduction='none'), dim=-1)
        else:
            traj_recon_loss = 0.


        if data[3] is not None:
            _, targets = data[3].max(dim=1)
            label_recon_loss = train_params['lambda_x3'] * F.cross_entropy(recon_data[3], targets, reduction='none')
        else:
            label_recon_loss = 0.

        prior_loss = train_params['beta'] * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = torch.mean(img_recon_loss + sound_recon_loss + traj_recon_loss + label_recon_loss + prior_loss)

        return loss