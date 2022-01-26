import sacred


###########################
#        Model            #
###########################

model_ingredient = sacred.Ingredient("model")


@model_ingredient.config
def gmc_mhd():
    model = "gmc"
    common_dim = 64
    latent_dim = 64
    loss_type = "infonce"  # "joints_as_negatives"


@model_ingredient.config
def mvae_mhd():
    model = "mvae"
    latent_dim = 64


@model_ingredient.config
def mfm_mhd():
    model = "mfm"
    fusion_dim = 64
    latent_dim = 64


@model_ingredient.config
def mmvae_mhd():
    model = "mmvae"
    latent_dim = 64


@model_ingredient.config
def muse_mhd():
    model = "muse"
    modality_dims = [64, 128, 32, 4]
    latent_dim = 64

@model_ingredient.config
def nexus_mhd():
    model = "nexus"
    modality_dims = [64, 128, 16, 5]
    message_dim = 512
    latent_dim = 64


##############################
#       Model  Train         #
##############################


model_train_ingredient = sacred.Ingredient("model_train")


@model_train_ingredient.named_config
def gmc_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 64
    num_workers = 8

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None
    temperature = 0.1


@model_train_ingredient.named_config
def mvae_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 64
    num_workers = 8

    # Training Hyperparameters
    epochs = 1
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    lambda_x0 = 1.0
    lambda_x1 = 1.0
    lambda_x2 = 50.0
    lambda_x3 = 50.0
    beta = 1.0


@model_train_ingredient.named_config
def mfm_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 64
    num_workers = 8

    # Training Hyperparameters
    epochs = 1
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    lambda_x0 = 1.0
    lambda_x1 = 1.0
    lambda_x2 = 50.0
    lambda_x3 = 50.0
    alpha = 2.0


@model_train_ingredient.named_config
def mmvae_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 64
    num_workers = 8
    n_training_samples = 1

    # Training Hyperparameters
    epochs = 1
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    coefs = [1.0, 1.0, 50.0, 50.0]
    beta = 1.0


@model_train_ingredient.named_config
def muse_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 64
    num_workers = 8
    n_training_samples = 1

    # Training Hyperparameters
    epochs = 1
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    lambdas = [1.0, 1.0, 50.0, 50.0]
    betas = [1.0, 1.0, 1.0, 1.0]
    gammas = [10, 10, 10, 10]
    beta_top = 1.0
    alpha = 1.0

@model_train_ingredient.named_config
def nexus_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 64
    num_workers = 8
    n_training_samples = 1

    # Training Hyperparameters
    epochs = 1
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    lambdas = [1.0, 1.0, 50.0, 50.0]
    betas = [1.0, 1.0, 1.0, 1.0]
    gammas = [1, 1, 50, 50]
    beta_nexus = 1.0