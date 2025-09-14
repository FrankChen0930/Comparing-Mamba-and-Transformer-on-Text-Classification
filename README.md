# Comparing-Mamba-and-Transformer-on-Text-Classification
This is my college project demonstration.  
We mainly focus on how the Mamba architecture work.  
We compare Mamba and Transformer by two different text classification dataset.  
And the cmoparisons include the accuracy and the resouce (GPU usage , training time , inference time ...)  
## Comparison Result  
Below is the equation that calculate how much does Mamba better than Transformer on accuracy.  
*Gain of Mamba=Mamba accuracy−Transformer accuracy*  
  
Below is the equation that calculate how much does Mamba better than Transformer other metrics.    
*Gain of Mamba = (Transformer value − Mamba value) / Transformer value × 100%*  

### IMDb 資料集中，參數量30~40萬之間的最佳實驗結果
|           | Mb-IMDb-1 |   TF-IMDb-1   |  Gain of Mamba  |
| --------- | --------- | ------------- | --------------- |
| 模型參數量 |  354,018  |393,121  |9.9%|
|Best accuracy |   86.3%   | 87.8%  |-1.5%|
| GPU 記憶體最高使用量(MB) |   244  | 558 |52.3%|
|訓練時間(s) |26.48|72.69|63.5%|
|推理時間(s) |1.55|3.57|56.6%|

    
### IMDb 資料集中，參數量60~80萬之間的最佳實驗結果  
|           | Mb-IMDb-2 |   TF-IMDb-2   |  Gain of Mamba  |
| --------- | --------- | ------------- | --------------- |
| 模型參數量 |  740,290|663,201|-11.6%|
|Best accuracy |87.6%|87.7%|-0.1%|
| GPU 記憶體最高使用量(MB) |304|560|45.7%|
|訓練時間(s) |32.22|37.95|15.1%|
|推理時間(s) |1.69|1.83|7.6%|
  

### IMDb 資料集實驗中最佳實驗結果之參數組合  
|           | Mb-IMDb-2 |   TF-IMDb-2   | Mb-IMDb-2 |   TF-IMDb-2   |
| --------- | --------- | ------------- | ---------- | ------------ |
| vocab_size | 10000 | 10000 | 10000| 20000 |
|max length | 500 | 200 | 500 | 200 |
| batch size | 32 | 64 | 32 | 64 |
|embed dim | 32 | 32 | 32 | 64 |
| d_state | 32 | N/A | 16 | N/A |
|conv_size | 3 | N/A | 4 | N/A |
|nhead | N/A | 4 | N/A | 2 |
|ff_dim | N/A | 1024 | N/A | 256 |


### News Category 資料集中，參數量30~40萬之間的最佳實驗結果
|           | Mb-News-1 |   TF-News-1   |  Gain of Mamba  |
| --------- | --------- | ------------- | --------------- |
| 模型參數量 | 332,395  |360,234  |7.73%|
|Best accuracy |  46.3%   | 49.7%  |-3.4%|
| GPU 記憶體最高使用量(MB) |   1565  | 5865 |73.3%|
|訓練時間(s) |88.44|193.1|54.2%|
|推理時間(s) |0.68|1.45|53.1%|

    
### News Category資料集中，參數量60~80萬之間的最佳實驗結果 
|           | Mb-News-2 |   TF-News-2   |  Gain of Mamba  |
| --------- | --------- | ------------- | --------------- |
| 模型參數量 |  682,251|726,666|6.1%|
|Best accuracy |51.3%|52.1%|-0.8%|
| GPU 記憶體最高使用量(MB) |2979|7849|62.0%|
|訓練時間(s) |160.48|218.22|26.5%|
|推理時間(s) |1.04|1.66|37.3%|
  

### News Category 資料集實驗中最佳實驗結果之參數組合
|           | Mb-News-2 |   TF-News-2   | Mb-News-2 |   TF-News-2   |
| --------- | --------- | ------------- | ---------- | ------------ |
| vocab_size | 10000 | 10000 | 10000| 10000 |
|max length | 100 | 250 | 100 | 250 |
| batch size | 512 | 256 | 512 | 256 |
|embed dim | 32 | 32 | 64 | 64 |
| d_state | 16 | N/A | 32 | N/A |
|conv_size | 2 | N/A | 2 | N/A |
|nhead | N/A | 4 | N/A | 4 |
|ff_dim | N/A | 512 | N/A | 512 |
