# GMC

## Creating conda environment and setting up the environment
```bash
conda env create -f gmc.yml
conda activate GMC
poetry install
```


## Seting up DCA evaluation requiriements

We refer to the instructions given by [the authors of DCA on their github repository](https://github.com/anonymous-researcherID1893ar/DelaunayComponentAnalysis). In particular, clone the repository containing [Delaunay approximation algorithm](https://github.com/vlpolyansky/voronoi-boundary-classifier/tree/testing) and run:
```bash
cd voronoi-boundary-classifier
git checkout testing
mkdir build && cd build
cmake ..
make VoronoiClassifier_cl
```
Then copy `cpp/VoronoiClassifier_cl` to the `gmc_code/DelaunayComponentAnalysis/` folder and check that the executable file was built successfully by running `gmc_code/DelaunayComponentAnalysis/VoronoiClassifier_cl` in the terminal. Make sure you see the following output
```bash
VoronoiClassifier_cl: <path>/voronoi-boundary-classifier/cpp/main_vc.cpp:51: void run_classification(int, char**): Assertion `argc >= 3' failed.
Aborted (core dumped)
```
For troubleshooting, we refer to the [original implementation.](https://github.com/anonymous-researcherID1893ar/DelaunayComponentAnalysis)


## Reproducing experiments

### Unsupervised learning problem

To reproduce the results reported in the paper, download the datasets and pretrained models first:

```bash
cd gmc_code/
bash download_unsupervised_dataset.sh
bash download_unsupervised_pretrain_models.sh
cd unsupervised/
```

Then set the correct path of your local machine in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to 
```bash
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-pf-pwd-here"
```

You can then evaluate a pretrained GMC model on the downstream classification task, for example, on image modality:

```bash
python main_unsupervised.py -f with experiment.model="gmc" experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_classifier"
```

To evaluate on other modalities, choose between `[0], [2], [3]` or `[0,1,2,3]` for complete observations in the `experiment.evaluation_mods` argument in the above code snipped.  To run DCA evaluation, use 

```bash
python main_unsupervised.py -f with experiment.model="gmc" experiment.stage="evaluate_dca"
```

The results will appear in the `evaluation/gmc_mhd/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and image representations are given in the `joint_m1/DCA_results_version0.log` file. Similarly, you can evaluate other pretrained models by setting `experiment.model` argument to `mfm, mmvae, muse, mvae` or `nexus`. If you wish to train your own models and downstream classifiers, run

```bash
model="gmc"
echo "** Train representation model"
python main_unsupervised.py -f with experiment.model=$model experiment.stage="train_model" 

echo "** Train classifier"
python main_unsupervised.py -f with experiment.model=$model experiment.stage="train_downstream_classfier"
```
After training the representation model or the classifier, please move the files `$model_mhd_model.pth.tar` and `down_$model_mhd_model.pth.tar` (available in the `evaluation`folder) to the `trained_models` folder.

### Supervised learning problem

To reproduce the results reported in the paper, download the datasets and pretrained models first:

```bash
cd gmc_code/
bash download_supervised_dataset.sh
bash download_supervised_pretrain_models.sh
cd supervised/
```

Then set the correct path of your local machine in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to 
```bash
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-pf-pwd-here"
```

You can then evaluate a pretrained GMC model, for example on the `mosei` downstream classification task on text modality:

```bash
python main_supervised.py -f with experiment.model="gmc" experiment.scenario="mosei" experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_classifier"
```

To evaluate on other modalities, choose between `[0], [2]` or `[0,1,2]` for complete observations in the `experiment.evaluation_mods` argument in the above code snipped.  To run DCA evaluation, use 

```bash
python main_supervised.py -f with experiment.model="gmc" experiment.scenario="mosei" experiment.stage="evaluate_dca"
```

The results will appear in the `evaluation/gmc_mosei/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and text representations are given in the `joint_m1/DCA_results_version0.log` file. Similarly, you can evaluate Multimodal Transformer by setting `experiment.model="multimodal_transformer"` or use CMU-MOSI dataset by setting `experiment.scenario="mosi"`

If you wish to train your own models, run

```bash
model="gmc"
scenario="mosei"
echo "** Train representation model"
python main_supervised.py -f with experiment.model=$model experiment.scenario=$scenario experiment.stage="train_model" 
```
After training the model, please move the file `$model_$scenario_model.pth.tar`(available in the `evaluation`folder) to the `trained_models` folder.


### Reinforcement Learning: Pendulum

To reproduce the results reported in the paper, download the datasets and pretrained models first:

```bash
cd gmc_code/
bash download_rl_dataset.sh
bash download_rl_pretrain_models.sh
cd rl/
```

Set the correct path of your local machine in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to 
```bash
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-pf-pwd-here"
```

You can then evaluate a pretrained GMC model with the downstream controller on sound

```bash
python main_rl.py -f with experiment.model="gmc"  experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_controller"
```

To evaluate on other modalities, choose between `[0]` or `[0,1]` for complete observations in the `experiment.evaluation_mods` argument in the above code snipped.  To run DCA evaluation, use 

```bash
python main_rl.py -f with experiment.model="gmc" experiment.stage="evaluate_dca"
```

The results will appear in the `evaluation/gmc_pendulum/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and text representations are given in the `joint_m1/DCA_results_version0.log` file. Similarly, you can evaluate other pretrained models by setting `experiment.model` argument to `mvae` or `muse`.

If you wish to train your own models and downstream classifiers, run

```bash
model="gmc"
echo "** Train representation model"
python main_rl.py -f with experiment.model=$model_name experiment.stage="train_model" 

echo "** Train controller"
python main_rl.py -f with experiment.model=$model_name experiment.stage="train_downstream_controller" 
```

After training the representation model or the controller, please move the files `$model_pendulum_model.pth.tar` and `down_$model_pendulum_model.pth.tar` (available in the `evaluation`folder) to the `trained_models` folder.


