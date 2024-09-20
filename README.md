# DSTANet

D-STANet（Delay-aware Spatio-Temporal Attention Network with GAT and Conv）

<img src="fig/model.pdf" alt="model.pdf" style="zoom:100%;" />

This is a Pytorch implementation of DSTANet . The pytorch version of DSTANet released here only consists of the  recent component, since the other two components have the same network architecture. 

# Reference

```latex
还没发
```

# Configuration

Step 1: The loss function and metrics can be set in the configuration file in ./configurations



# Datasets

Step 1: Download datasets provided by (https://drive.google.com/drive/folders/1VTj9CFY_5-N_X3nsOagB3GhD5HLemdxQ?usp=drive_link). 

Step 2: Process dataset

- on PEMS04 dataset

  ```shell
  python prepareData.py --config configurations/PEMS04_DSTANet.conf
  ```
- on PEMS07 dataset

  ```shell
  python prepareData.py --config configurations/PEMS07_DSTANet.conf
  ```
- on PEMS08 dataset

  ```shell
  python prepareData.py --config configurations/PEMS08_DSTANet.conf
  ```
- on HZME_INFLOW dataset

  ```shell
  python prepareData.py --config configurations/HZME_INFLOW._DSTANet.conf
  ```
  - on HZME_OUTFLOW dataset

  ```shell
  python prepareData.py --config configurations/HZME_OUTFLOW_DSTANet.conf
  ```
# Train and Test

- on PEMS04 dataset

  ```shell
  python train_D-STANet.py --config configurations/PEMS04_DSTANet.conf
  ```
- on PEMS07 dataset

  ```shell
  python train_D-STANet.py --config configurations/PEMS07_DSTANet.conf
  ```
- on PEMS08 dataset

  ```shell
  python train_D-STANet.py --config configurations/PEMS08_DSTANet.conf
  ```
- on HZME_INFLOW dataset

  ```shell
  python train_D-STANet.py --config configurations/HZME_INFLOW._DSTANet.conf
  ```
  - on HZME_OUTFLOW dataset

  ```shell
  python train_D-STANet.py --config configurations/HZME_OUTFLOW_DSTANet.conf
  ```

  



