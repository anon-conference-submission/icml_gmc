import sacred
import gmc_code.unsupervised.ingredients.stage_specific.model_ingredients as sacred_model
import gmc_code.unsupervised.ingredients.stage_specific.scenario_ingredients as sacred_scenario
import gmc_code.unsupervised.ingredients.stage_specific.down_ingredients as sacred_down
import gmc_code.unsupervised.ingredients.stage_specific.dca_evaluation_ingredients as sacred_dca


########################
#     Experiment       #
########################

exp_ingredient = sacred.Ingredient("experiment")


@exp_ingredient.config
def exp_config():

    # Experiment setup
    scenario = "mhd"
    model = "gmc"
    seed = 0
    cuda = True

    # Experiment id (for checkpoints)
    exp_id = None

    # Stages
    # Model Training         - 'train_model'
    # DCA Evaluation         - 'evaluate_dca'
    # Classifier Training    - 'train_downstream_classfier',
    # Classifier Evaluation  - 'evaluate_downstream_classifier'

    stage = "evaluate_model_dca"
    evaluation_mods = [0,1,2,3]

    # Load model and scenario specific ingredients
    if scenario == "mhd":

        scenario_config = sacred_scenario.mhd()
        down_train_config = sacred_down.mhd()
        dca_evaluation_config = sacred_dca.mhd()

        if model == "gmc":
            model_config = sacred_model.gmc_mhd()
            model_train_config = sacred_model.gmc_mhd_train()

        elif model == "mvae":
            model_config = sacred_model.mvae_mhd()
            model_train_config = sacred_model.mvae_mhd_train()

        elif model == "mfm":
            model_config = sacred_model.mfm_mhd()
            model_train_config = sacred_model.mfm_mhd_train()

        elif model == "mmvae":
            model_config = sacred_model.mmvae_mhd()
            model_train_config = sacred_model.mmvae_mhd_train()

        elif model == "muse":
            model_config = sacred_model.muse_mhd()
            model_train_config = sacred_model.muse_mhd_train()

        elif model == "nexus":
            model_config = sacred_model.nexus_mhd()
            model_train_config = sacred_model.nexus_mhd_train()

        else:
            raise ValueError("[Exp Ingredient] Model not yet implemented for MHD environment : " + model)

    else:
        raise ValueError("[Exp Ingredient] Scenario not yet implemented : " + scenario)

