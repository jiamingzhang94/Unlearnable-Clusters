# Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples

https://arxiv.org/abs/2301.01217
Code relative to "[Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples](https://arxiv.org/abs/2301.01217)"


## Preprocess
First, run preprocess.py to rename the dataset directory
```
python preprocess.py --config config/preprocess.yaml -f rename
```

## K-means initial
Second, run preprocess.py to conduct K-means initial.
```
python preprocess.py --config config/preprocess.yaml -f cluster_pets_rn50_cluster10
# the cluster file will be stored in {output_dir}/{dataset}_{surrogate_model}_cluster{num_clusters}.pth
# e.g., output/cluster/pets_rn50_cluster10.pth
```

## Genearte the cluster-wise perturbations and the corresponding UEs
Third, run main.py --stage 1 to generate cluster-wise perturbations.
```
python main.py --config config/stage_1.yaml -e {experiment} --stage 1 
# {experiment} is uc_{dataset}_{surrogate_model}
# e.g., python main.py --config config/stage_1.yaml -e uc_pets_cliprn50 --stage 1
```

## Train target model
Finally, run main.py --stage 2 to train target models.
```
python main.py --config config/stage_2.yaml -e {experiment} --stage 2 
# {experiment} is uc_{dataset}_{surrogate_model}_{target_model}
# e.g., python main.py --config config/stage_2.yaml -e uc_pets_rn50_rn18 --stage 2
```
