# D-STANet

**D-STANet: Delay-aware Spatio-Temporal Attention Network with GAT and Conv**

![Model Architecture](fig/model.pdf)

This repository provides the **PyTorch implementation** of **D-STANet**, a spatio-temporal prediction model that integrates **Graph Attention Networks (GAT)** and **Convolutional Networks**, while incorporating **delay-aware mechanisms** for improved traffic forecasting accuracy.

> 🔍 Only the recent component is included here, as the other two components follow the same architecture.

---

## 📄 Paper

Our paper has been accepted by **Knowledge-Based Systems (KBS)**.

> 📌 The BibTeX citation will be added here once available.

```latex
% Coming soon


# ⚙️ Configuration

The model settings, loss function, and metrics can be customized in the configuration files under ./configurations.



# 📂 Datasets

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
# 🏃‍♂️ Train and Test

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

  



