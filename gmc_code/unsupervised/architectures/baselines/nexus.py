import numpy as np
from torch.distributions import Bernoulli
from pytorch_lightning import LightningModule
from gmc_code.unsupervised.architectures.baselines.baseline_networks import *

class Nexus(LightningModule):
    def __init__(self, name, modality_dims, message_dim, latent_dim):

        super(Nexus, self).__init__()

        # Structure
        self.name = name
        self.latent_dim = latent_dim
        self.modality_dims = modality_dims
        self.message_dim = message_dim

        # Bottom-level
        self.modality_encoders = []
        self.modality_decoders = []

        self.message_encoders = []
        self.nexus_encoder = None
        self.nexus_decoders = []

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

            # Message encoding
            mod_messages = []
            for z, message_encoder in zip(zs, self.message_encoders):
                if z is not None:
                    mod_messages.append(message_encoder(z))
                else:
                    mod_messages.append(None)

            # Aggregate
            aggregator_msg = self.aggregate(mod_messages)

            # Encode Nexus
            nx_mu, nx_logvar = self.nexus_encoder(aggregator_msg)

            if sample:
                return self.reparametrize(nx_mu, nx_logvar)
            else:
                return nx_mu

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

    def aggregate(self, mod_messages):

        # For single messages no need to aggregate
        if len([x for x in mod_messages if x is not None]) == 1:
            for mod_msg in mod_messages:
                if mod_msg is not None:
                    return mod_msg

        # Concatenate existing messages
        for mod_msg in mod_messages:
            if mod_msg is not None:
                comp_msg = torch.zeros(size=mod_msg.size()).unsqueeze(0).to(mod_msg.device)
                break

        for mod_msg in mod_messages:
            if mod_msg is not None:
                comp_msg = torch.cat([comp_msg, mod_msg.unsqueeze(0)], dim=0)

        comp_msg = self.mean_drop(comp_msg[1:], mod_messages)
        return comp_msg

    def mean_drop(self, mean_msg, mod_msgs):

        # Compute mean message
        mean_msg = torch.mean(mean_msg, dim=0)

        if not self.training:
            return mean_msg
        else:
            # For each entry in batch: (During training we have all modalities available)
            for i in range(mean_msg.size(0)):

                drop_mask = Bernoulli(torch.tensor([0.2])).sample()

                # If there is no dropout, we continue
                if torch.sum(drop_mask).item() == 0:
                    continue

                # If there is dropout, we randomly select the number and type of modalities to drop
                else:
                    n_mods_to_drop = torch.randint(low=1, high=len(mod_msgs), size=(1,)).item()
                    mods_to_drop = np.random.choice(range(len(mod_msgs)), size=n_mods_to_drop, replace=False)

                    prune_msg = torch.zeros(mod_msgs[0].size(-1))
                    prune_msg = prune_msg.unsqueeze(0).to(mod_msgs[0].device)

                    for j in range(len(mod_msgs)):
                        if j in mods_to_drop:
                            continue
                        else:
                            prune_msg = torch.cat([prune_msg, mod_msgs[j][i].unsqueeze(0)], dim=0)
                    prune_msg = prune_msg[1:]
                    mean_msg[i] = torch.mean(prune_msg, dim=0)

            return mean_msg

    def loss_function(self, recon_data, data, mod_mus, mod_logvars, mod_zs, mod_nexus_zs, mu, logvar, train_params):
        return

    def training_step(self, data, train_params):

        # Forward pass through the encoders
        recons, mod_mus, mod_logvars, mod_zs, mod_nexus_zs, nx_mu, nx_logvar = self.forward(data)
        loss = self.loss_function(recons, data, mod_mus, mod_logvars, mod_zs, mod_nexus_zs, nx_mu, nx_logvar, train_params)

        tqdm_dict = {'loss': loss}

        return loss, tqdm_dict

    def validation_step(self, data, train_params):

        ## Forward pass through the encoders
        recons, mod_mus, mod_logvars, mod_zs, mod_nexus_zs, nx_mu, nx_logvar = self.forward(data)
        loss = self.loss_function(recons, data, mod_mus, mod_logvars, mod_zs, mod_nexus_zs, nx_mu, nx_logvar, train_params)

        tqdm_dict = {'loss': loss}

        return tqdm_dict


    def forward(self, data):

        recons = []
        mod_mus, mod_logvars, mod_zs, mod_zs_detach, mod_nexus_zs = [], [], [], [], []
        mod_messages = []

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

        # Message encoding
        for z, message_encoder in zip(mod_zs_detach, self.message_encoders):
            mod_messages.append(message_encoder(z))

        # Aggregate
        aggregator_msg = self.aggregate(mod_messages)

        # Encode Nexus
        nx_mu, nx_logvar = self.nexus_encoder(aggregator_msg)
        nx_z = self.reparametrize(nx_mu, nx_logvar)

        # Decode Nexus
        for nx_decoder in self.nexus_decoders:
            mod_nx_z = nx_decoder(nx_z)

            mod_nexus_zs.append(mod_nx_z)


        return recons, mod_mus, mod_logvars, mod_zs, mod_nexus_zs, nx_mu, nx_logvar



class MhdMexus(Nexus):
    def __init__(self,  name, modality_dims, message_dim, latent_dim):
        super(MhdMexus, self).__init__(name, modality_dims, message_dim, latent_dim)

        self.name = name
        self.latent_dim = latent_dim
        self.modality_dims = modality_dims
        self.message_dim = message_dim

        self.image_encoder = MHDImageGaussianEncoder(modality_dims[0])
        self.sound_encoder = MHDSoundGaussianEncoder(modality_dims[1])
        self.trajectory_encoder = MHDTrajectoryGaussianEncoder(modality_dims[2])
        self.label_encoder = MHDLabelGaussianEncoder(modality_dims[3])

        self.image_decoder = MHDImageGaussianDecoder(modality_dims[0])
        self.sound_decoder = MHDSoundGaussianDecoder(modality_dims[1])
        self.trajectory_decoder = MHDTrajectoryGaussianDecoder(modality_dims[2])
        self.label_decoder = MHDLabelGaussianDecoder(modality_dims[3])

        self.image_msg_encoder = MHDMessageEncoder(modality_dims[0],message_dim)
        self.sound_msg_encoder = MHDMessageEncoder(modality_dims[1], message_dim)
        self.trajectory_msg_encoder = MHDMessageEncoder(modality_dims[2], message_dim)
        self.label_msg_encoder = MHDMessageEncoder(modality_dims[3], message_dim)

        self.nexus_encoder = MHDNexusEncoder(message_dim, latent_dim)

        self.image_nx_decoder = MHDNexusDecoder(latent_dim, modality_dims[0])
        self.sound_nx_decoder = MHDNexusDecoder(latent_dim, modality_dims[1])
        self.trajectory_nx_decoder = MHDNexusDecoder(latent_dim, modality_dims[2])
        self.label_nx_decoder = MHDNexusDecoder(latent_dim, modality_dims[3])

        self.modality_encoders = [self.image_encoder, self.sound_encoder, self.trajectory_encoder, self.label_encoder]
        self.modality_decoders = [self.image_decoder, self.sound_decoder, self.trajectory_decoder, self.label_decoder]

        self.message_encoders = [self.image_msg_encoder,
                                 self.sound_msg_encoder,
                                 self.trajectory_msg_encoder,
                                 self.label_msg_encoder]

        self.nexus_decoders = [self.image_nx_decoder, self.sound_nx_decoder, self.trajectory_nx_decoder, self.label_nx_decoder]


    def loss_function(self, recon_data, data, mod_mus, mod_logvars, mod_zs, mod_nexus_zs, mu, logvar, train_params):


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
        image_nx_recon = train_params['gammas'][0] * torch.sum(F.mse_loss(mod_nexus_zs[0], mod_zs[0].clone().detach(), reduction='none'), dim=-1)
        sound_nx_recon = train_params['gammas'][1] * torch.sum(F.mse_loss(mod_nexus_zs[1], mod_zs[1].clone().detach(), reduction='none'), dim=-1)
        trajectory_nx_recon = train_params['gammas'][2] * torch.sum(F.mse_loss(mod_nexus_zs[2], mod_zs[2].clone().detach(), reduction='none'), dim=-1)
        label_nx_recon = train_params['gammas'][3] * torch.sum(F.mse_loss(mod_nexus_zs[3], mod_zs[3].clone().detach(), reduction='none'), dim=-1)

        nx_prior_loss = train_params['beta_nexus'] * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


        # Total loss
        loss = torch.mean(image_recon_loss + image_prior_loss
                          + sound_recon_loss + sound_prior_loss
                          + trajectory_recon_loss + trajectory_prior_loss
                          + label_recon_loss + label_prior_loss
                          + image_nx_recon + sound_nx_recon + trajectory_nx_recon + label_nx_recon
                          + nx_prior_loss)

        return loss

