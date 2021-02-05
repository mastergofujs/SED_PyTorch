## Experiments Document



### SOURCE CODE STRUCTURE

The root path `SED_PyTorch/` of source code is made up of three folders: `aed_data/`, `MainClasse/`, and `task_scripts/`. The detailed introduction about them is given in the following. 

* `SED_PyTorch/aed_data/`

This folder contains two database directories of the datasets (i.e., Freesound and TUT), together with the experiment results on these two datasets. Under the database directories of `freesound/​` and `tut_data/​`, there are two folders ​`mfccs/​` and ​`result/​` respectively. Where, ​`mfccs/` provides the input raw data, and ​`result/` saves the result of each experiment on the corresponding database, and also contains the weights of the proposed model in the $K$-folds evaluation processing. The detailed folder structure of ​`aed_data/​` is described in the following.

   * `freesound/`
     * `ae_dataset/`: saves the original Freesound dataset.
     * `audio_target/`: saves the discriminative audio signal which are selected from the original Freesound dataset.
     * `audio/`: saves the short sound segments cut from `audio_target/` via Audacity (a free software). Here, the labels of short sound segments are the same as the label of the long audio signal that they are cut from.
     * `mfccs/`: 
      * `datas/`: contains MFCC features (`*.pkl` file), named as `mfcc_[events number]_.pkl`.
      * `labels/`: contains pickle files which are corresponding to feature files.
     * `noise/`: contains three types of environment noise, which will be used when generating sub-set with different polyphonic level.
     * `result/`: saves the model weights in the training stage and the results of cross evaluation as table files (`*.csv`).
   * `tut_data/`
     * `train/`: saves the data related to the training set, which includes annotation files, original signal files and MFCC features.
     * `test/`: saves the testing set.
     * `result/`: saves the model weights and experiment results.

* `SED_PyTorch/MainClasses/`

This folder contains several **core files** (Python classes files), which define the data processing, models initialization, the training and evaluation process. The details about each class are given below.

  * `DataHandler.py`: This Python script defines the class of parameter initialization and data processing, which contain several methods for data preprocessing, such as `enframe()`, `windowing()`, etc. Besides, it provides all the raw input data for each task script by `load_data()` method. Note that, the most important parameter of `load_data()` is `fold`, which is used to generate the training and validation set. It is worth mentioning that the method `mix_data()` is essential for the Freesound dataset, because it is used to generate the sub-dataset with various polyphonic levels. Here is the usage of mixing the polyphonic sound for Freesound dataset:
    1.Load `DataHandler` class and create an object:
    
    ```python
    from MainClasses.DataHandler import DataHandler
    dh = DataHandler(dataset='Freesound')
    ```

  2.Generate `n` mixture sound with `k` categories, and here  `k` denotes polyphonic level: 

```python
dt.mix_data(nevents=k, nsamples=n)
```

  By using `mix_data()`, we can get a `.pkl` file which contains the MFCCs features extracted from the mixed sound. 

  * `TorchDataset.py`: This Python script defines the class of data packing, which extending the super class Dataset imported from PyTorch. As the data in array form can not be loaded directly for the PyTorch module, we should convert the form of data into Tensor and pack them into the model. To achieve the data packing, we re-write the initialization and `get_item()` methods in the class TorchDataset.
  
  * `CustmerMetrics.py` and `CustmerLosses.py`: These two scripts give the definition of the loss function and metrics using in our experiments. In the class of `CustmerMetrics`, we define the most two important metrics proposed by the DCASE Challenge 2017, named as segment-based F1 score and segment-based Error Rate (ER). Besides, in order to make training stage more clearly, we also define the R square and binary accurate in this class. And the proposed novel disentangling loss function is defined in the `CustmerLosses.py`. Below are the usage about this two scripts:

  ```python
  from MainClasses.CustmerLosses import DisentLoss
  from MainClasses.CustmerMetrics import segment_metrics, r_square, binary_accurate
  loss_fcn = DisentLoss(K=num_events, beta=beta) # K denotes the number of event category
  ```

  * `SBetaVAE.py`: This Python script defines the classes while implementing our proposed method, which extends the PyTorch super class `nn.Module` by rewriting the `init()` and `forward()` function. Specifically, `SBetaVAE.py` contains four classes: `LinearAttention`, `Encoder`, `Decoder` and `SBetaVAE`, of which the first three classes define the key layers of the proposed SBetaVAE model are defined in `SBetaVAE` class. It is simple to define a new model by using the codes below:
  
    ```python
    from MainClasses.SBetaVAE import SBetaVAE
    sb_vae = SBetaVAE(options)
    ```
  
* `SED_PyTorch/task_scripts/`: This folder contains Python scripts provided for the major experiments implemented in our paper. **It is important to review the code carefully for the replicating work.** We provide five script files for various tasks mentioned in our paper, which correspond to the evaluation experiment on the feature distribution in Fig. 3, the disentanglement evaluation in Fig. 4, the evaluation experiment on DCASE2017 challenge TUT dataset in Table 3, the evaluation experiment on Freesound dataset with various event polyphonic levels in Table 4 and the evaluation experiment on data augmentation in Table 5, respectively. To make these script files easier to read, we reseal each script with at least two functions: `setup_args()​` for arguments setup and `running()` for preparing input data, building model and executing training/evaluation process. Besides, some of the scripts contain `validation()​` and ​`test()​` functions which will not be execute individually but will be called in `running()​` function. All the task scripts can be implemented using the following shell command:

   ```shell
   cd SED_PyTorch/tast_scripts/
   python [any scripts].py [args] # -h for help
   ```
   Where the optional arguments can be found in `setup_args()​` or `-h​` in terminal and note that, for all experiments, this shell command will implement all the training, validation and testing stage. If you want to execute testing stage only, just set the arguments `e​` (epoch) to 0, which means do not train the model.
   
   Although such command can be used to repeat the experiments, it is necessary to give a detailed introduction for each task script. **It is important to emphasize that we need execute these task scripts in the order listed below since their results are dependent with each other.**

   - `tb4_various_events.py`: This script evaluates the performance of our methods on data of various polyphonic levels made from Freesound as shown in Table 4 in the accepted paper. In this script, there are many optional arguments which will determine the structure of the model. We can conduct this experiment after we choose suitable value for each optional argument. For example, if we want to train our model with the samples which contain 10 event categories,  and in this case, the parameter `k​` should be set as 10 and the other hyper-parameters $\beta$, $\lambda$ and `latent_dim​` will be set as 4, 2 and 30 automatically. Therefore we just need to call the shell command below to conduct this evaluation experiment:

     ```shell
     python tb4_various_events.py -k 10 # here "-k" is short of "--num_events"
     ```
     
   Here, It's important to note that the argument `-m` denotes whether we need generate new data, and it is set as 1 as default if you want generate new data, and when you do not need to generate new data, set as 0. **As we have uploaded the sub-dataset for this experiments, it needn't to generate them again.**
   
  * `fig4_disentanglement_visualization.py`: This script is used to evaluate the disentanglement performance, listed as in Fig. 4 in our paper . To qualitatively show event-specific disentangled factors learned by supervised $\beta$-VAE, we need to call the shell command below, which contains five major steps:
    
    ```python
    python fig4_disentanglement_visualization.py
    ```
  1. Create model and load weights;
     
       ```python
       sb_vae = SBetaVAE(options).cuda()
       sb_vae.load_state_dict(torch.load(options.result_path + sb_vae.name + '/fold_' + str(fold) + '_cp_weight.h5')
      ```
     
  2. Give `n` samples of input data `x` , and extract `z*` for some specific sound event categories;
     
       ```python
       ..., z_stars, ... = sb_vae(torch.from_numpy(test_data).float().cuda())
      ```
     
  3. Adjust one latent variable while fixing others in `z_star` and visualize the corresponding changes in the generated data. Take `Children` (shown in Fig.4. in our paper) as an instance, we adjust the value of the 14-th dimension of `z_star` and fixing others.
     
  4. Define `delta()` function, mentioned in Section 4.5 of our paper, and calculate the difference among generated data;
     
   5. Visualize the differences calculated above using hot figure.


   - `tb5_data_augmentation.py`: This script gives the method to evaluate the data generation ability of the proposed method. 

   1. We first make up the unbalanced dataset from Freesound using the class `DataHandler` mentioned before:
  
        ```python
        from DataHandler import *
        dh = DataHandler('freesound')
        ```
  
   2. Then call `mix_data()` method with the argument `isUnbalanced=True`, by which we can limit the number of the samples in the unbalanced dataset for a specific event category. 
  
        ```python
        dh.mix_data(nevents=5, nsamples=2000, isUnbalanced=True)
        ```
  
   3. After getting the unbalanced dataset, we train and evaluate the model:
  
        ```python
        model = SBetaVAE(options)
        outputs = model(inputs)
        for e in range(epoch):
          loss = DisentLoss(outputs, targets)
          loss.backward()
        # here we get the original results
        f1_score, error_rate = test(model, test_data, test_labels, supervised=True)
        ```
  
   4. Next, we generate the samples for the event category with insufficient samples by decoding the specific latent factors `z*`:
  
        ```python
        # generate new data
        with torch.no_grad():
          x_augmented = torch.Tensor()
          for n_sample, (x_data, y_data) in enumerate(train_loader):
            dec_out, detectors_out,z_stars,alphas,(mu,log_var)=sb_vae(x_data.float().cuda())
            dec_x = sb_vae.decoder(z_stars[:, :, 0]) # we generate the first event defaultly.
            x_augmented = torch.cat([x_augmented, dec_x])
        ```
  
5. At last, we extend the unbalanced dataset with the generated data to retrain the model, and evaluate it again, which will improve the performance of the augmented event category.

  - `tb3_dcase17_ours.py`: This script evaluates the performance of the proposed method on DCASE 2017 SED challenge TUT dataset as shown in Table 3. In `tb3_dcase17.py​`, `running()​` function defines the process of building model, loading weights, training, evaluating and testing models among the 4-folds cross-validation. Since the training and testing set are selected by the DCASE challenge, it is not necessary to provide more arguments than $b$ (batchsize), $e$ (epoch), and $lr$ (learning rate) when calling the shell command below:
  
    ```python
    python tb3_dcase17.py -b 128 -e 50 -lr 0.0003
    ```
  
  - `fig3_feature_distribution.py`: This script is used to visualize the distributions of the features learned by the proposed model, listed as in Fig. 3. The visualization procedure is dependent on $t$-SNE which should be imported by `sklearn.manifold`. In order to make the replicating work simpler, we have put all the steps in each of algorithm into `running()​` function.  Finally, to show the figure of feature distribution, call the command below and the figure file (`.png​`) will be saved at ​`aed_data/tut_data/result/distribution.png​`:
  
    ```python
    python fig3_feature_distribution.py
    ```
  
    Below are the details about this experiment. 

   1. Create model and load weights:
  
       ```python
       sb_vae = SBetaVAE(options).cuda()
       sb_vae.load_state_dict(torch.load('/path/of/weight'))
       ```
  
   2. Give `n` samples as the input data of `x` , and extract the output of hidden layers(`m_th`):
  
       ```python
       for i in range(options.num_events):
         n_ = 0
         while n_ < num_to_plot:
           item = random.randint(0, len(test_dataset) - 1)
           if test_dataset[item][1][i] == 1:  # ensure the test data contains the i-th type of event
             n_ += 1
             x_data, y_data = test_dataset[item]
             with torch.no_grad():
               ..., bottleneck_features, ... = sb_vae(torch.from_numpy(x_data)                                                                                                  .float().cuda())
               h_out = torch.cat([h_out, torch.relu(bottleneck_features)[:, :, i]])
       ```
  
   3. Create $t$-SNE object and initialize it with `PCA`:
  
       ```python
       from sklearn.manifold import TSNE
       tsne = TSNE(n_components=target_dim, init='pca')
       ```
  
   4. Train and transform the high-level features into 2 dimensions:
  
       ```python
       tsne_datas = tsne.fit_transform(datas)
       ```
  
   5. Using `matplotlib` to plot the transformed data with colorful legend labels:
  
       ```python
       nums_each_ev = int(len(datas) / n_events)
       for i in range(n_events):
         plt.scatter(tsne_datas[nums_each_ev * i:nums_each_ev * (i + 1), 0],
                     tsne_datas[nums_each_ev * i:nums_each_ev * (i + 1), 1],
                     c=plt.cm.Set1((i + 1) / 10.),
                     alpha=0.6, marker='o')
         plt.legend(label_list, loc='lower right', ncol=2)
         plt.title('Distribution of features learned by {}.'.format(model.name))
         plt.show() 
       ```

   In order to simplify the procedure, in `fig3_feature_distribution.py`, we has put all the five steps into `running()` function. For more details, you need read the code comments carefully.

### TIME SPENDING

The running time of the main task scripts executed in our GPU server is shown below:

`tb4_various_events_ours.py` with various event categories:

5 events with 2000 samples: 26s * 80 epochs * 4 folds;
10 events with 3000 samples: 44s * 80 epochs * 4 folds;
15 events with 4000 samples: 50s * 80 epochs * 4 folds;
20 events with 5000 samples: 73s * 80 epochs * 4 folds;

`tb3_dcase17_ours.py`:
dcase: 179s * 200 epochs * 4 folds

The CPU of our server is Intel(R) Core(TM) i9-9820X CPU @ 3.30GHz and the GPU is NVIDIA RTX 2080Ti.

### PRETRAINED WEIGHTS

At last, in order to make the reproducibility work more efficiently, we provide all the weights of the model using in all the five experiments. The weights file ($*\_weights.h5$) are saved in the directory of `results`, so if one wants to skip the training stag, he/she needs to put the weights file into the corresponding dataset directories first, then **run each task scripts motioned above, with the parameter $e$ set as 0.**

