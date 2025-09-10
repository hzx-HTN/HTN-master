## Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset
We use two open-source datasets:

+ **CWE Dataset**: [https://github.com/CGCL-codes/VulCNN](https://github.com/CGCL-codes/VulCNN)
+ **CodeXGLUE Dataset**: [https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF](https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF)

## Dataset Reconstruction
You can run `CWEutils.py` and `CodeXGLUEutils.py` to reconstruct these two datasets.

For the CodeXGLUE dataset, you also need to download [https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/dataset), which is the partitioning method provided by CodeXGLUE.

## Train
```bash
python trainXXX.py
```

Where "XXX" is the model name.

+ Set model parameters in `parse.py`
+ Configure dataset options and parameters in `CWEutils.py` and `CodeXGLUEutils.py`

## Analysis
If you need to export attention matrices, you need to run:

```bash
python trainHTN_Attr.py
```

This appears to be a specific training script for HTN (Hierarchical Transformer Network) that includes attention matrix export functionality, which is commonly used for analyzing what the model focuses on during prediction.



You can perform model performance and resource usage analysis by running:
```bash
python ModelProfiler.py
```

This script appears to be designed for profiling the computational performance, memory usage, and other resource consumption metrics of the trained models.
