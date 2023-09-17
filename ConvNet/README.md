## Training a model

The network file already has 2 models usable. In order to add more, you can just extend 'ImageClassificationBase'. This way you just have to add an $_init_$ and forward function.

## Model Parameters
Most hyper parameters and folder path are taken from the config file. Here is a complete list :

| **Parameters** | **Details**                                                    |   |   |   |
|----------------|----------------------------------------------------------------|---|---|---|
| _epoch_        | number of epochs                                               |   |   |   |
| _lr_           | learning rate                                                  |   |   |   |
| _batch_size_   | batch size                                                     |   |   |   |
| _savefolder_   | where the weights of each epoch will be saved                  |   |   |   |
| _dataset_      | folder where all your classes can be found (cf data/README.md) |   |   |   |

## External parameters
Some parameters, mostly related to the use of a GPU, can be changed from the terminal. A complete list :

| **Parameters** | **Details**                         | **Default** | **Type** |
|----------------|-------------------------------------|-------------|----------|
| --config       | config file in the yaml format      | config.yaml | string   |
| --use_GPU      | do you want to use GPU, cf train.sh | False       | none     |
| --GPU_id       | number of the GPU to use            | 0           | int      |