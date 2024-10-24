# SwissDINO

Official implementation of our work **SwissDINO**, published in [IROS2024](https://arxiv.org/abs/2407.07541).
In this paper, we present a one-shot personal object search method based on the recent DINOv2 transformer model. Swiss DINO handles challenging on-device personalized scene understanding requirements and does not require any adaptation training.

![image](https://github.com/user-attachments/assets/335eba04-46e4-4dbc-ab93-15611f75cb40)

## Install requirements

Install conda environment with
```
$ conda env create -f swiss_dino_env.yml
```

## Download datasets

We use two datasets for evaluation of the method: PerSeg (https://github.com/ZrrSkywalker/Personalize-SAM#preparation) and ICubWorld (https://robotology.github.io/iCubWorld/#icubworld-transformations-modal).

Download and extract a chosen dataset, and set `$DATA_DIR` to the root dataset path.

## Run evaluation

To run evaluation on PerSeg dataset:
```
python swiss_dino_evaluation.py --dataset_name perseg --data_dir $DATA_DIR --fe_model_type vit_s --verbose
```

To run evaluation on ICubWorld dataset:
```
python swiss_dino_evaluation.py --dataset_name icubworld --data_dir $DATA_DIR --fe_model_type vit_s --verbose
```

The evaluation script generates `log.txt` file with per-class metrics.

## Run inference

The inference is not supported yet.

## Cite us

If you use this repository, please cite our work

       @article{paramonov2024swiss,
        title={Swiss DINO: Efficient and Versatile Vision Framework for On-device Personal Object Search},
        author={Paramonov, Kirill and Zhong, Jia-Xing and Michieli, Umberto and Moon, Jijoong and Ozay, Mete},
        journal={IROS},
        year={2024}
        }
