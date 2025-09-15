# Mamba Architecture  
## SSM  
由SSM (State Space Model) 的概念來自訊號系統中的 Continuous-Time Linear State-Space ODE 。  
具體公式如下： x(t)：系統狀態；A、B、C、D：系統參數矩陣；u(t)：訊號輸入；y(t)：狀態輸出  
$$\frac{d\mathbf{x}(t)}{dt} = A\mathbf{x}(t) + B u(t)$$ ----(1)  
$$y\mathbf{x}(t) = C\mathbf{x}(t) + D u(t)$$ ----(2)  

為了將這個描述連續狀態的微分方程改為描述一個離散狀態的公式，引入了 step size Δ 作為離散化參數。  
並且透過 Bilinear Transform 對 Eq.(1) 的A和B進行離散化，公式如下： Ad：離散化A；Bd：離散化B  
$$A_d = \left( I - \frac{\Delta}{2} A \right)^{-1} \left( I + \frac{\Delta}{2} A \right)$$ ----(3)  
$$B_d = \left( I - \frac{\Delta}{2} A \right)^{-1} \, \Delta B$$ ----(4)  

除了將 A 和 B 進行離散化處理之外，系統狀態 x(t) 也會變成離散化的狀態，將不再需要如 Eq.(1) 使用微分的形式進行表示。  
而系統參數矩陣 D 則相當於 Residual 的功能，對 SSM 的系統狀態並沒有影響，所以將D從系統參數矩陣中移除，至此 SSM的核心公式如下：  
$$\mathbf{x}_t = A_d \mathbf{x}_{t-1} + B_d u_t$$ ----(5)  
$$\mathbf{y}_t = C \mathbf{x}_t$$ ----(6)  

在SSM的核心公式 Eq.(5)、(6) 中的 Ad 和 Bd 並非是SSM的訓練目標，SSM中可訓練的參數是 A、B、C、D、step size Δ 。  
關於SSM如何處理資料以及訓練時是如何執行backpropagation (藍色箭頭) 如圖一所示，黑色箭頭則表示正常進行forward pass 時，系統參數矩陣 A、B、C、D、step size Δ 的運作方式。
