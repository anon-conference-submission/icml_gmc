from pytorch_lightning import LightningModule
from gmc_code.unsupervised.architectures.baselines.baseline_networks import *

class MUSE(LightningModule):
    def __init__(self, name, modality_dims, latent_dim):

        super(MUSE, self).__init__()

        # Structure
        self.name = name
        self.latent_dim = latent_dim
        self.modality_dims = modality_dims

        # Bottom-level
        self.modality_encoders = []
        self.modality_decoders = []

        self.top_encoders = []
        self.top_decoders = []

        self.poe = ProductOfExperts()

    def encode(self, x, sample=False):
        with torch.no_grad():

            zs = []

            # Bottom encoding
            for mod_data, mod_encoder in zip(x, self.modality_encoders):
                if mod_data is not None:
                    mod_mu, _ = mod_encoder(mod_data)
                    zs.append(mod_mu)
                else:
                    zs.append(None)

            # Top encoding
            top_mu, top_logvar = self.infer(zs)

            if sample:
                return self.reparametrize(top_mu, top_logvar)
            else:
                return top_mu

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

    def infer(self, zs):

        for z in zs:
            if z is not None:
                batch_size = z.size(0)
                break
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA

        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.latent_dim), use_cuda=use_cuda)

        # Encode modality-specific data
        for z, top_encoder in zip(zs, self.top_encoders):
            if z is not None:
                top_mu, top_logvar = top_encoder(z)
                mu = torch.cat((mu, top_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, top_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.poe(mu, logvar)
        return mu, logvar


    def loss_function(self, recon_data, data, mod_mus, mod_logvars, mod_zs, mod_top_zs, mu, logvar, alma_mus, alma_logvars, train_params):
        return

    def training_step(self, data, train_params):

        # Forward pass through the encoders
        recon, mod_mus, mod_logvars, mod_zs, mod_top_zs, mu, logvar, alma_mus, alma_logvars = self.forward(data)
        loss = self.loss_function(recon, data, mod_mus, mod_logvars, mod_zs, mod_top_zs, mu, logvar, alma_mus, alma_logvars, train_params)

        tqdm_dict = {'loss': loss}

        return loss, tqdm_dict

    def validation_step(self, data, train_params):

        ## Forward pass through the encoders
        recon, mod_mus, mod_logvars, mod_zs, mod_top_zs, mu, logvar, alma_mus, alma_logvars = self.forward(data)
        loss = self.loss_function(recon, data, mod_mus, mod_logvars, mod_zs, mod_top_zs, mu, logvar, alma_mus, alma_logvars, train_params)

        tqdm_dict = {'loss': loss}

        return tqdm_dict


    def forward(self, data):

        recons = []
        mod_mus, mod_logvars, mod_zs, mod_top_zs = [], [], [], []
        mod_zs_detach, alma_mus, alma_logvars = [], [], []

        # Bottom-level encoding/decoding
        for mod_data, mod_encoder, mod_decoder in zip(data, self.modality_encoders, self.modality_decoders):
            mod_mu, mod_logvar = mod_encoder(mod_data)
            mod_z = self.reparametrize(mod_mu, mod_logvar)
            mod_recon = mod_decoder(mod_z)

            mod_mus.append(mod_mu)
            mod_logvars.append(mod_logvar)
            mod_zs.append(mod_z)
            mod_zs_detach.append(mod_z.clone().detach())
            recons.append(mod_recon)

        # Top level encoding/decoding
        top_mu, top_logvar = self.infer(mod_zs_detach)
        top_z = self.reparametrize(top_mu, top_logvar)

        for top_decoder in self.top_decoders:
            mod_top_z = top_decoder(top_z)
            mod_top_zs.append(mod_top_z)


        # ALMA terms
        for i in range(len(mod_zs_detach)):
            alma_input = [None]*len(mod_zs_detach)
            alma_input[i] = mod_zs_detach[i].clone().detach()
            alma_mu, alma_logvar = self.infer(alma_input)

            alma_mus.append(alma_mu)
            alma_logvars.append(alma_logvar)


        return recons, mod_mus, mod_logvars, mod_zs, mod_top_zs, top_mu, top_logvar, alma_mus, alma_logvars




class MhdMUSE(MUSE):
    def __init__(self, name, modality_dims, latent_dim):
        super(MhdMUSE, self).__init__(name, modality_dims=modality_dims, latent_dim=latent_dim)

        self.name = name
        self.latent_dim = latent_dim
        self.modality_dims = modality_dims

        self.image_encoder = MHDImageGaussianEncoder(modality_dims[0])
        self.sound_encoder = MHDSoundGaussianEncoder(modality_dims[1])
        self.trajectory_encoder = MHDTrajectoryGaussianEncoder(modality_dims[2])
        self.label_encoder = MHDLabelGaussianEncoder(modality_dims[3])

        self.image_decoder = MHDImageGaussianDecoder(modality_dims[0])
        self.sound_decoder = MHDSoundGaussianDecoder(modality_dims[1])
        self.trajectory_decoder = MHDTrajectoryGaussianDecoder(modality_dims[2])
        self.label_decoder = MHDLabelGaussianDecoder(modality_dims[3])

        self.image_top_encoder = MHDMuseTopEncoder(modality_dims[0], latent_dim)
        self.sound_top_encoder = MHDMuseTopEncoder(modality_dims[1], latent_dim)
        self.trajectory_top_encoder = MHDMuseTopEncoder(modality_dims[2], latent_dim)
        self.label_top_encoder = MHDMuseTopEncoder(modality_dims[3], latent_dim)

        self.image_top_decoder = MHDMuseTopDecoder(latent_dim=latent_dim, out_dim=modality_dims[0])
        self.sound_top_decoder = MHDMuseTopDecoder(latent_dim=latent_dim, out_dim=modality_dims[1])
        self.trajectory_top_decoder = MHDMuseTopDecoder(latent_dim=latent_dim, out_dim=modality_dims[2])
        self.label_top_decoder = MHDMuseTopDecoder(latent_dim=latent_dim, out_dim=modality_dims[3])

        self.modality_encoders = [self.image_encoder, self.sound_encoder, self.trajectory_encoder, self.label_encoder]
        self.modality_decoders = [self.image_decoder, self.sound_decoder, self.trajectory_decoder, self.label_decoder]

        self.top_encoders = [self.image_top_encoder, self.sound_top_encoder, self.trajectory_top_encoder, self.label_top_encoder]
        self.top_decoders = [self.image_top_decoder, self.sound_top_decoder, self.trajectory_top_decoder, self.label_top_decoder]


    def loss_function(self, recon_data, data, mod_mus, mod_logvars, mod_zs, mod_top_zs, mu, logvar, alma_mus, alma_logvars, train_params):


        # Bottom loss terms
        image_recon_loss = train_params['lambdas'][0] * torch.sum(F.mse_loss(recon_data[0].view(recon_data[0].size(0), -1),
                                                                          data[0].view(data[0].size(0), -1),
                                                                          reduction='none'), dim=-1)
        image_prior_loss = train_params['betas'][0] * (-0.5 * torch.sum(1 + mod_logvars[0] - mod_mus[0].pow(2) - mod_logvars[0].exp(), dim=1))

        sound_recon_loss = train_params['lambdas'][1] * torch.sum(F.mse_loss(recon_data[1].view(recon_data[1].size(0), -1),
                                                                          data[1].view(data[1].size(0), -1),
                                                                          reduction='none'), dim=-1)
        sound_prior_loss = train_params['betas'][1] * (-0.5 * torch.sum(1 + mod_logvars[1] - mod_mus[1].pow(2) - mod_logvars[1].exp(), dim=1))

        trajectory_recon_loss = train_params['lambdas'][2] * torch.sum(F.mse_loss(recon_data[2].view(recon_data[2].size(0), -1),
                                                                                 data[2].view(data[2].size(0), -1),
                                                                                 reduction='none'), dim=-1)
        trajectory_prior_loss = train_params['betas'][2] * (-0.5 * torch.sum(1 + mod_logvars[2] - mod_mus[2].pow(2) - mod_logvars[2].exp(), dim=1))

        _, targets = data[3].max(dim=1)
        label_recon_loss = train_params['lambdas'][3] * F.cross_entropy(recon_data[3], targets, reduction='none')
        label_prior_loss = train_params['betas'][3] * (-0.5 * torch.sum(1 + mod_logvars[3] - mod_mus[3].pow(2) - mod_logvars[3].exp(), dim=1))


        # Top loss terms
        image_top_recon = train_params['gammas'][0] * torch.sum(F.mse_loss(mod_top_zs[0], mod_zs[0].clone().detach(), reduction='none'), dim=-1)
        sound_top_recon = train_params['gammas'][1] * torch.sum(F.mse_loss(mod_top_zs[1], mod_zs[1].clone().detach(), reduction='none'), dim=-1)
        trajectories_top_recon = train_params['gammas'][2] * torch.sum(F.mse_loss(mod_top_zs[2], mod_zs[2].clone().detach(), reduction='none'), dim=-1)
        label_top_recon = train_params['gammas'][3] * torch.sum(F.mse_loss(mod_top_zs[3], mod_zs[3].clone().detach(), reduction='none'), dim=-1)
        top_prior_loss = train_params['beta_top'] * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Alma Terms
        top_fpa = 0.0
        for i in range(len(alma_mus)):
            top_fpa += train_params['alpha'] * sym_KLD_gaussian(q_mu=mu, q_logvar=logvar, p_mu=alma_mus[i], p_logvar=alma_logvars[i])
        top_fpa /= len(alma_mus)

        # Total loss
        loss = torch.mean(image_recon_loss + image_prior_loss
                          + sound_recon_loss + sound_prior_loss
                          + trajectory_recon_loss + trajectory_prior_loss
                          + label_recon_loss + label_prior_loss
                          + image_top_recon + sound_top_recon + trajectories_top_recon + label_top_recon
                          + top_prior_loss + top_fpa)

        return loss