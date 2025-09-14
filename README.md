# Comparing-Mamba-and-Transformer-on-Text-Classification
This is my college project demonstration.  
We mainly focus on how the Mamba architecture work.  
We compare Mamba and Transformer by two different text classification dataset.  
And the cmoparisons include the accuracy and the resouce (GPU usage , training time , inference time ...)  
## Comparison Result  
|           | Mb-IMDb-1 |   TF-IMDb-1   |  Gain of Mamba  |
| --------- | --------- | ------------- | --------------- |
| 模型參數量 |  354,018  |393,121  |9.9%|
|Best accuracy |   86.3%   | 87.8%  |-1.5%|
| GPU 記憶體最高使用量(MB) |   244  | 558 |52.3%|
|訓練時間(s) |26.48|72.69|63.5%|
|推理時間(s) |1.55|3.57|56.6%|
  
IMDb 資料集中，參數量30~40萬之間的最佳實驗結果  

|           | Mb-IMDb-2 |   TF-IMDb-2   |  Gain of Mamba  |
| --------- | --------- | ------------- | --------------- |
| 模型參數量 |  740,290|663,201|-11.6%|
|Best accuracy |87.6%|87.7%|-0.1%|
| GPU 記憶體最高使用量(MB) |304|560|45.7%|
|訓練時間(s) |32.22|37.95|15.1%|
|推理時間(s) |1.69|1.83|7.6%%|
  
IMDb 資料集中，參數量60~80萬之間的最佳實驗結果  
