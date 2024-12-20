#root
```
expt_name"tsp10" 
gpu_id"0"
train_filepath"./data/tsp10_train_concorde.txt" #訓練
val_filepath"./data/tsp10_val_concorde.txt" #驗證
test_filepath"./data/tsp10_test_concorde.txt" #測試數據
num_nodes12 #節點數量:在問題中的位置數
num_neighbors1 #車輛在每次選擇行駛路線時可以考慮的鄰近路徑數
node_dim3 #F
#用於圖神經網絡（GNN）來對車輛和路徑進行編碼和解碼:節點表示送貨地址，邊表示車輛行駛的路徑
voc_nodes_in2
voc_nodes_out2
voc_edges_in3
voc_edges_out2
q_edges_out1
#搜索寬度為 1280，表示在解碼過程中最多保留 1280 條候選路徑:可以幫助找出多種潛在的最優路徑
beam_size1280
#控制神經網絡的深度和複雜度，從而影響模型的表現
hidden_dim2
num_layers20
mlp_layers3
#將鄰居節點的特徵平均值作為每個節點的特徵聚合方式。這種聚合方式可以幫助車輛在考慮路徑時綜合鄰近的路徑特徵
aggregation"mean"
max_epochs1500 #模型訓練的時間長短
val_every5 #每次驗證和測試的頻率
test_every100 #每次驗證和測試的頻率
batch_size1
batches_per_epoch1000 
accumulation_steps1 
learning_rate0.001 #模型學習率
decay_rate1.01 #衰減率
```
