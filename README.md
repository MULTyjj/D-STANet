# DSTANet

D-STANet（Delay-aware Spatio-Temporal Attention Network with GAT and Conv）

<img src="fig/模型.tif" alt="image-20200103164326338" style="zoom:50%;" />

This is a Pytorch implementation of DSTANet . The pytorch version of DSTANet released here only consists of the  recent component, since the other two components have the same network architecture. 

# Reference

```latex
还没发
```

# Configuration

Step 1: The loss function and metrics can be set in the configuration file in ./configurations



# Datasets

Step 1: Download datasets provided by (https://github.com/). 

Step 2: Process dataset

- on PEMS04 dataset

  ```shell
  python prepareData.py --config configurations/PEMS04_DSTANet.conf
  ```

- on PEMS08 dataset

  ```shell

  ```
  python prepareData.py --config configurations/PEMS08_DSTANet.conf
  ````

# Train and Test

- on PEMS04 dataset

  ```shell
  python train_D-STANet.py --config configurations/PEMS04_DSTANet.conf
  ```

- on PEMS08 dataset

  ```shell
  python train_D-STANet.py --config configurations/PEMS08_DSTANet.conf
  ```

  

  



