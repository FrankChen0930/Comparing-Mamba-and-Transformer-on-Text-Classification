# Mamba Architecture  
## SSM  
由SSM (State Space Model) 的概念來自訊號系統中的 Continuous-Time Linear State-Space ODE 。  
具體公式如下： x(t)：系統狀態；A、B、C、D：系統參數矩陣；u(t)：訊號輸入；y(t)：狀態輸出  
<p align="center">
$\frac{d\mathbf{x}(t)}{dt} = A\mathbf{x}(t) + B u(t)$ ----(1)
</p>

<p align="center">
$y(t) = C\mathbf{x}(t) + D u(t)$ ----(2)
</p>

為了將這個描述連續狀態的微分方程改為描述一個離散狀態的公式，引入了 step size Δ 作為離散化參數。  
並且透過 Bilinear Transform 對 Eq.(1) 的A和B進行離散化，公式如下： Ad：離散化A；Bd：離散化B  
<p align="center">
$A_d = \left( I - \frac{\Delta}{2} A \right)^{-1} \left( I + \frac{\Delta}{2} A \right)$ ----(3)
</p>

<p align="center">
$B_d = \left( I - \frac{\Delta}{2} A \right)^{-1} \, \Delta B$ ----(4)

除了將 A 和 B 進行離散化處理之外，系統狀態 x(t) 也會變成離散化的狀態，將不再需要如 Eq.(1) 使用微分的形式進行表示。  
而系統參數矩陣 D 則相當於 Residual 的功能，對 SSM 的系統狀態並沒有影響，所以將D從系統參數矩陣中移除，至此 SSM的核心公式如下：
<p align="center">
$\mathbf{x}_t = A_d \mathbf{x}_{t-1} + B_d u_t$ ----(5)  
</p>

<p align="center">
$$\mathbf{y}_t = C \mathbf{x}_t$$ ----(6)
</p>

在SSM的核心公式 Eq.(5)、(6) 中的 Ad 和 Bd 並非是SSM的訓練目標，SSM中可訓練的參數是 A、B、C、D、step size Δ 。  
關於SSM如何處理資料以及訓練時是如何執行backpropagation (藍色箭頭) 如圖一所示，黑色箭頭則表示正常進行forward pass 時，系統參數矩陣 A、B、C、D、step size Δ 的運作方式。
![圖一：SSM架構](https://github.com/FrankChen0930/Comparing-Mamba-and-Transformer-on-Text-Classification/blob/main/What%20is%20Mamba/SSM.png)  
圖一：SSM架構  

### 1.CNN-like  
SSM的核心公式Eq. (5)、(6)不難發現 SSM 是一個 RNN-like 的模型，而 RNN-like 模型最大的問題便是訓練速度緩慢，但SSM與一般 RNN 模型最大的不同便是 SSM 具有 CNN-like 的訓練方式。  
假設將 x-1 設為0，並將Eq. (5)、(6)攤開，可以得到結果如下：  
$$ \mathbf{x}_0 = B_d u_0 $$
$$ \mathbf{y}_0 = C B_d u_0 $$

$$ \mathbf{x}_1 = A_d B_d u_0 + B_d u_1 $$
$$ \mathbf{y}_1 = C A_d B_d u_0 + C B_d u_1 $$

$$ \mathbf{x}_2 = A_d^2 B_d u_0 + A_d B_d u_1 + B_d u_2 $$
$$ \mathbf{y}_2 = C A_d^2 B_d u_0 + C A_d B_d u_1 + C B_d u_2 $$

透過上面的式子，能夠整理出下面的規律：
$$ \mathbf{y}_t = C A_d^t B_d u_0 + C A_d^{t-1} B_d u_1 + \cdots + C A_d B_d u_{t-1} + C B_d u_t $$ ----(7)

定義一個 K 作為kernel，則得出：
$$ K_t = (C A_d^t B_d, \cdots , C A_d B_d, C B_d) $$ ----(8)

再將Kt 與Eq.(7) 進行公式整理之後便會得到下面的式子：
<p align="center">
$$\mathbf{y}_t = K \mathbf{x}_t × U \mathbf{x}_t$$ ----(9)
</p>
Ut代表[u1, u2, … , ut]，對於輸入 token 數量為 t 時，能夠直接根據Ad、Bd、C、t計算出 Kt ，再將Ut中的每個token 對應到Kt 的每一項。因為 kernel Kt 的存在，賦予了 SSM CNN-like的特性，不論 t 的大小都能用一個公式直接計算出 SSM 的輸出 yt ，大大降低了訓練所需的時間。

### 2.HiPPO Matrix  
從Eq. (3)、(5)中可知系統參數矩陣 A 會負責處理系統狀態 xt 的保留。 透過Eq.(6) 可知SSM 的輸出yt 與系統參數矩陣有密切的關係，因此系統參數矩陣 A 可以說是決定狀態系統 x 記憶長短與穩定的關鍵。  
為了保證系統狀態 xt 的穩定，引入了 HiPPO matrix 對系統參數矩陣 A 進行初始化。HiPPO matrix透過 Legendre 多項式的係數對矩陣進行設定，進而達到讓系統參數矩陣 A 從訓練的一開始便具有記憶能力，維持系統狀態 xt 的穩定。

### 3.Calculation Issue  
在離散化的過程中 Eq. (3)、(4) 出現了反矩陣的運算，再加上 Eq. (7) 中系統參數矩陣 A 的多次連乘都會導致計算上的不穩定，尤其在大量的運算上容易出現錯誤，致使整個系統不穩定，輸出結果也會無法信任。  
在Structured State Space Model [4]中給出了解決辦法，透過專門為SSM設計的計算演算法，包含iFFT, NPLR, DPLR ,Woodbury Identity, Cauchy kernel, FFT保證了計算的穩定性。  

## Selctive
即使在Structured State Space Model(S4) 的研究中已經解決了 SSM 在實作上的問題，但仍然不足以讓 SSM成為 Transformer 的挑戰者，因為 SSM 並沒有像Transformer 中的 Attention 機制能對輸入進行篩選。  

相較於 Attention 直接考慮整個序列，並挑選出重要資訊進行加權的做法， Selective 的做法是對輸入的資訊進行挑選並記憶，進而達到在有限的系統狀態 xt 內盡可能儲存重要資訊。具體Selective的作法如圖二所示，由三個部分組成，分別是映射矩陣、1D-Convolution、gated機制 。  

映射矩陣是兩個可學習的矩陣，一個負責將input 投影成gate所需的資訊(如圖二中的Gate_Proj)，另一個矩陣則負責將input 投影成後續1D-Convolution所需的矩陣(如圖二中的CNN_Proj)。  

1D-Convolution 中的1D是指在一段序列中每個 token 的每個維度所出現的順序或者說時間關係，如圖二中 token 的 t, t-1, t-2, t-3 的關係。而Convolution 的輸入則是來自 In_Proj 的映射結果。具體這個 1D-Convolution 的 window 大小則作為超參數用以根據目標與任務進行調整。而 1D-Convolution 的輸出則會交由 SSM 進行後續運算。

在 SSM 輸出之後會進入一個gate ，而這個gate會決定 SSM 的輸出yt 中每個元素應該保留或是遺忘的程度。而gate中決定保留或是遺忘的程度的矩陣由Gate_Proj 經過SiLU activation function 所產生。  
![圖一：SSM架構](https://github.com/FrankChen0930/Comparing-Mamba-and-Transformer-on-Text-Classification/blob/main/What%20is%20Mamba/mamba.png)  
圖二：Mamba架構  

## Mamba Block  
主要由 Selective 和 SSM 所組成，但並不完整，仍需要一點調整與技術才能完整實現整個Mamba Block (如圖二所示)  


以 RMS Norm (Root Mean Square Layer Normalization) 對輸入序列進行 normalization ，不僅能契合 SSM 的性質，相較其他normalization也有更高的計算效率，對於計算量龐大的 Mamba 來說非常適合。  

接著是 1D-Convolution 和 SSM 之間的對接。已知 SSM 需要的輸入包括 A、B、C、u、Δ，而1D-Convolution 的輸出結果會切割成B、C、u 傳進 SSM ，A 則是由 HiPPO matrix 直接進行初始化，最後剩下的 Δ 則是將1D-Convolution的輸出結果透過一個可學習的映射矩陣 Proj_Δ (如圖二所示) 投影得到。  

在 SSM 完成計算輸出 yt 並經過gated計算之後由 Proj_Out 進行維度調整再加上input sequence 提供 Residual 機制。經過上述的組合才能夠完整的呈現一個如圖二所示的 Mamba Block。  
