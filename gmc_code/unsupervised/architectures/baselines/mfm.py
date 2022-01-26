from pytorch_lightning import LightningModule
from gmc_code.unsupervised.architectures.baselines.baseline_networks import *


class MFM(LightningModule):
    def __init__(self, name, fusion_dim, latent_dim):

        super(MFM, self).__init__()

        # Variables
        self.name = name
        self.fusion_dim = fusion_dim
        self.latent_dim = latent_dim
        self.encoders = []
        self.decoders = []
        self.intermediates = []
        self.fusion = None
        self.head = None

    def encode(self, x, sample=False):
        with torch.no_grad():
            # Get Device
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            # Get batch-size
            for i in range(len(x)):
                if x[i] is not None:
                    batch_size = x[i].size(0)
                    break

            # Crop data to encoder list length
            x = x[:len(self.encoders)]


            # Encode Representation
            mod_rep = []
            for i in range(len(x)):
                if x[i] is not None:
                    mod_rep.append(self.encoders[i](x[i]))
                else:
                    mod_rep.append(torch.zeros([batch_size, self.fusion_dim]).to(dev))

            # Concat representations
            concat_rep = torch.cat(mod_rep, dim=-1)

            # Fusion concat representations
            fusion_rep = self.fusion(concat_rep)
            return fusion_rep



    def loss_function(self, recon_data, data, pred, train_params):
        return

    def training_step(self, data, train_params):

        # Forward pass through the encoders
        recon, pred = self.forward(data)
        loss = self.loss_function(recon, data, pred, train_params)

        tqdm_dict = {'loss': loss}

        return loss, tqdm_dict

    def validation_step(self, data, train_params):
        recon, pred = self.forward(data)
        loss = self.loss_function(recon, data, pred, train_params)

        tqdm_dict = {'loss': loss}

        return tqdm_dict

    def forward(self, data):

        # Crop data to encoder list length
        data = data[:len(self.encoders)]

        # Encode Representation
        mod_rep = []
        for i in range(len(data)):
            mod_rep.append(self.encoders[i](data[i]))

        # Encode intermediate representations
        inter_rep = []
        for i in range(len(data)):
            inter_rep.append(self.intermediates[i](mod_rep[i]))

        # Concat representations
        concat_rep = torch.cat(mod_rep, dim=-1)

        # Fusion concat representations
        fusion_rep = self.fusion(concat_rep)

        # Class output
        pred = self.head(fusion_rep)

        # Decode representations
        recons = []
        for i in range(len(data)):
            recons.append(self.decoders[i](torch.cat([inter_rep[i], fusion_rep], dim=-1)))

        return recons, pred



class MhdMFM(MFM):
    def __init__(self, name, fusion_dim, latent_dim):
        super(MhdMFM, self).__init__(name, fusion_dim, latent_dim)
        self.name = name
        self.latent_dim = latent_dim
        self.fusion_dim = fusion_dim

        # Encoders
        self.image_encoder = MHDImageEncoder(fusion_dim)
        self.sound_encoder = MHDSoundEncoder(fusion_dim)
        self.trajectory_encoder = MHDTrajectoryEncoder(fusion_dim)
        self.label_encoder = MHDLabelEncoder(fusion_dim)

        # Decoders
        self.image_decoder = MHDImageDecoder(2*latent_dim)
        self.sound_decoder = MHDSoundDecoder(2 * latent_dim)
        self.trajectory_decoder = MHDTrajectoryDecoder(2 * latent_dim)
        self.label_decoder = MHDLabelDecoder(2*latent_dim)

        # Intermediates
        self.image_intermediate = MHDIntermediate(fusion_dim, latent_dim)
        self.sound_intermediate = MHDIntermediate(fusion_dim, latent_dim)
        self.trajectory_intermediate = MHDIntermediate(fusion_dim, latent_dim)
        self.label_intermediate = MHDIntermediate(fusion_dim, latent_dim)

        # Fusion
        self.fusion = MHDFusion(4*fusion_dim, latent_dim)

        # Head
        self.head = MHDHead(latent_dim, 10)

        self.encoders = [self.image_encoder, self.sound_encoder, self.trajectory_encoder, self.label_encoder]
        self.decoders = [self.image_decoder, self.sound_decoder, self.trajectory_decoder, self.label_decoder]
        self.intermediates = [self.image_intermediate, self.sound_intermediate, self.trajectory_intermediate, self.label_intermediate]

    def loss_function(self, recon_data, data, pred, train_params):

        # Image Loss
        img_recon_loss = train_params['lambda_x0'] * torch.sum(F.mse_loss(recon_data[0].view(recon_data[0].size(0), -1),
                                                                            data[0].view(data[0].size(0), -1),
                                                                            reduction='none'), dim=-1)

        # Sound Loss
        sound_recon_loss = train_params['lambda_x1'] * torch.sum(F.mse_loss(recon_data[1].view(recon_data[1].size(0), -1),
                                                                          data[1].view(data[1].size(0), -1),
                                                                          reduction='none'), dim=-1)

        # Trajectory Loss
        traj_recon_loss = train_params['lambda_x2'] * torch.sum(F.mse_loss(recon_data[2].view(recon_data[2].size(0), -1),
                                                                          data[2].view(data[2].size(0), -1),
                                                                          reduction='none'), dim=-1)

        # Label Loss
        _, targets = data[3].max(dim=1)
        label_recon_loss = train_params['lambda_x3'] * F.cross_entropy(recon_data[3], targets, reduction='none')

        # Supervised Loss
        ce_loss = train_params['alpha'] * F.cross_entropy(pred, targets, reduction='none')

        return torch.mean(img_recon_loss + sound_recon_loss + traj_recon_loss + label_recon_loss + ce_loss)