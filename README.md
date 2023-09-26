# Cluster-based Short-term Wind Speed Interval Prediction with Multi-objective Ensemble Learning

This paper proposes a cluster-based short-term wind speed interval prediction with multi-objective ensemble learning. This method integrates uncertainties, extracted via clustering from the original time series, into the point prediction. Simultaneously, the point prediction methodology is enhanced by a multi-objective ensemble learning progress.

## FrameWork

First we extract statistical information from the original wind speed sequence

Next, we decompose the original sequence by VMD and obtain HE values for the subsequences to match the model training.

After that, we use the multi-objective algorithm MOEA/D to find the optimal superposition weights of each subsequence to get the point prediction results.

Finally, K-means determines which group the point prediction belongs to, and adds the corresponding estimated width $\lambda$ to get the final interval prediction.

<img src=".\images\img.png" style="zoom: 80%;" />



## Requirements

```
numpy~=1.23.1
pandas~=2.0.3
tensorflow~=2.13.0
scikit-learn~=1.3.0
matplotlib~=3.7.2
geatpy~=2.7.0
pmdarima~=2.0.3
```

## Dataset

The wind speed data is from [https://www.nrel.gov/wind/data-tools.html](National Renewable Energy Laboratory), 
you can download it from [https://pan.baidu.com/s/1XJf4pzF--bIv3iVlFFh4Gw?pwd=sker ](here)

## Run

### Perform one-step prediction
```
python main.py
```

### Perform three-steps prediction
```
python three_main.py
```

### Perform five-steps prediction
```
python five_main.py
```

## Acknowledgement
Thanks to those who provided help with this paper.