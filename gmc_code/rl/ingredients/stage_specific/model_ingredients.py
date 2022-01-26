import sacred


###########################
#        Model            #
###########################

model_ingredient = sacred.Ingredient("model")

# Pendulum

@model_ingredient.config
def gmc_pendulum():
    model = "gmc"
    common_dim = 64
    latent_dim = 10
    loss_type = "prepared_for_ablation"  # "joints_as_negatives"


@model_ingredient.config
def mvae_pendulum():
    model = "mvae"
    latent_dim = 10


@model_ingredient.config
def muse_pendulum():
    model = "muse"
    modality_dims = [16, 8]
    latent_dim = 10




##############################
#       Model  Train         #
##############################


model_train_ingredient = sacred.Ingredient("model_train")


@model_train_ingredient.named_config
def gmc_pendulum_train():
    # Dataset parameters
    batch_size = 128
    num_workers = 8

    # Training Hyperparameters
    epochs = 500
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    temperature = 0.3


@model_train_ingredient.named_config
def mvae_pendulum_train():
    # Dataset parameters
    data_dir = "./dataset/rl/"
    batch_size = 128
    num_workers = 8

    # Training Hyperparameters
    epochs = 500
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    lambda_x0 = 1.0
    lambda_x1 = 1.0
    beta = 1.0



@model_train_ingredient.named_config
def muse_pendulum_train():
    # Dataset parameters
    data_dir = "./dataset/rl"
    batch_size = 128
    num_workers = 8
    n_training_samples = 1

    # Training Hyperparameters
    epochs = 500
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    lambdas = [1.0, 100.0]
    betas = [1.0, 1.0]
    gammas = [10, 10]
    beta_top = 1.0
    alpha = 1.0