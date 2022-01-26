import sacred


###########################
#        Model            #
###########################

model_ingredient = sacred.Ingredient("model")

# MOSI
@model_ingredient.config
def gmc_mosi():
    model = "gmc"
    common_dim = 60
    latent_dim = 60
    loss_type = "infonce"  


@model_ingredient.config
def multimodal_transformer_mosi():
    model = "multimodal_transformer"

# MOSEI
@model_ingredient.config
def gmc_mosei():
    model = "gmc"
    common_dim = 60
    latent_dim = 60
    loss_type = "infonce"  

@model_ingredient.config
def multimodal_transformer_mosei():
    model = "multimodal_transformer"



##############################
#       Model  Train         #
##############################


model_train_ingredient = sacred.Ingredient("model_train")

# MOSI
@model_train_ingredient.named_config
def gmc_mosi_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 24
    num_workers = 8

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None
    temperature = 0.3

@model_train_ingredient.named_config
def multimodal_transformer_mosi_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 24
    num_workers = 8

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None



# MOSEI
@model_train_ingredient.named_config
def gmc_mosei_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 24
    num_workers = 8

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None
    temperature = 0.3


@model_train_ingredient.named_config
def multimodal_transformer_mosei_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 24
    num_workers = 8

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None