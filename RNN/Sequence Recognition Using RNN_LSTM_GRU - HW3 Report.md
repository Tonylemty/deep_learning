---
title: Sequence Recognition Using RNN/LSTM/GRU - HW3 Report

---

# Sequence Recognition Using RNN/LSTM/GRU - HW3 Report

**Student：** 蔡東廷
**Student ID：** 411221315
**Kaggle Username：** tonylemty
**Kaggle ID：** 411221315

---

## 1. Introduction（前言）
在這次 Kaggle 競賽中，我們的目標是將已經轉為數字序列的文字資料，分類到五個類別（Category_A 到 Category_E）。雖然本質上是文字分類任務，但資料不像一般文本那樣直觀——它已經被預處理成數字序列，因此在模型設計時，必須特別留意序列長度的差異、語意如何從數字中表達，以及不同類別之間的特徵區別。

其實這次作業的重點，不完全是追求最高的準確率，更是希望透過競賽的完整流程，實際走一遍資料預處理、模型設計、訓練與驗證，一直到最後提交結果上 Kaggle 評測的過程。我們也可以藉此嘗試不同的模型架構和訓練策略，觀察它們對效能的影響，再一步步調整與改進。

我自己這次主要採用深度學習模型，並且在訓練集中切出一部分作為驗證集，避免模型過度擬合導致準確度下降。接下來，我會從資料處理、模型架構、設計上的幾個亮點與策略，以及實驗結果這幾個部分來逐一說明。

## 2. Data Processing（資料處理）
在此次競賽中教授所提供的資料已完成初步處理，包含訓練與測試樣本對應的 tokenized sequences，所以不需要再進行原始文字清洗或斷詞。然而，在實際使用這些資料前，仍需進行以下幾個重要的處理步驟。

### 2.1 Loading Tokenized Sequences
首先，透過 `train_sequences.json` 與 `test_sequences.json` 載入每筆樣本的 token 序列。同時，我們也載入 `train.csv` 中的標籤資訊，方便之後做數值編碼。然後由於每個序列的長度不會完全相同，所以為了能夠輸入至神經網路模型中，必須將所有序列統一為固定長度。

```python=
import json
import pandas as pd

# 讀取訓練標籤資料
train_df = pd.read_csv("train.csv")

# 載入已 tokenized 的訓練序列
with open("train_sequences.json", "r") as f:
    train_sequences = json.load(f)

# 載入已 tokenized 的測試序列
with open("test_sequences.json", "r") as f:
    test_sequences = json.load(f)
```

### 2.2 Sequence Padding and Truncation
而在實作中我採用 padding 的方式，將較短的序列補齊至指定長度，而過長的序列則進行截斷，以避免序列過長造成訓練時間過久。此外，我使用 post-padding / post-truncation 的方式來保留序列前面的資訊，並將序列長度固定為 `MAX_LEN`。

```python=
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 設定最大序列長度
MAX_LEN = 200

# 對訓練資料進行 padding 與 truncation
X_train = pad_sequences(
    train_sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

# 對測試資料進行 padding 與 truncation
X_test = pad_sequences(
    test_sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)
```

### 2.3 Label Encoding
接著，根據 `train.csv` 中提供的標籤資訊，將類別文字（Category_A 至 Category_E）轉換為對應的數值編碼。接著，在模型預測完成後，再將數值結果轉換回原始類別的名稱，作為最終的提交格式。

```python=
# 定義類別文字與數值標籤的對應關係
label_mapping = {
    "Category_A": 0,
    "Category_B": 1,
    "Category_C": 2,
    "Category_D": 3,
    "Category_E": 4
}

# 將文字標籤轉換為數值
y_train = train_df["category"].map(label_mapping).values
```

### 2.4 Train / Validation Split
此外，我為了評估模型在未知資料上的泛化能力，就從訓練資料中切分出一部分作為驗證集，透過驗證集準確率來調整模型參數與訓練策略，避免出現只對訓練資料表現良好但在測試資料上效果不佳的情況。

```python=
from sklearn.model_selection import train_test_split

# 將資料切分為訓練集與驗證集
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)
```

## 3. Model Architecture（模型架構）

![image](https://hackmd.io/_uploads/SyhAvf94-g.png =700x)


在模型設計方面，這次我主要採用深度學習的方法來處理這個數字序列分類問題。

一開始，我先用了一層 Embedding 層。這層的目的是把那些原本沒有直接意義的數字 token，轉成一個低維度、而且能蘊含語意關係的向量。這樣一來，模型在後續處理時，比較有機會去理解不同數字背後可能代表的關聯性。

接著，為了捕捉這些數字序列中的前後關聯，我使用了像 LSTM 這類的序列模型。我們知道，單純把每個數字分開看是沒意義的，必須看它們整段序列的組合。LSTM 的記憶單元設計，對於抓住長序列裡的依賴關係特別有幫助，這也是文字分類任務中常用的手法。為了防止模型發生過擬合，我在模型裡也加入了 Dropout，隨機忽略一部分神經元的輸出，強迫模型學得更穩健。

等到序列特徵抓得差不多了，我再用全連接層把這些高維特徵「壓縮」成五個類別的對應分數，最後透過 Softmax 輸出每個類別的機率。整個訓練過程，我主要是看分類準確率來評估模型好壞，並且用交叉熵損失函數來計算預測和真實標籤之間的差距，讓模型可以朝著對的方向進行調整。

## 4. Methodology & Novelty（設計亮點與策略）


### A. Train / Validation Split Strategy（訓練與驗證策略）
在訓練模型的時候，我沒有把所有的訓練資料一次用完。而是先將原始資料切分成兩部分：Training Set 和 Validation Set。大部分資料用來訓練模型，剩下的一小部分則留下來，當作驗證集。

這樣做的好處是，我可以在訓練過程中，隨時用驗證集檢查模型在「沒看過的資料」上表現如何。如果看到訓練準確率一直上升，但驗證準確率卻卡住甚至往下掉，那很可能就是模型開始過擬合了——也就是學得太拘泥於訓練資料的細節，反而失去泛化能力。

因為這次任務是五類分類，而且每個類別的樣本數量也算平均，用這種隨機切分的方式來評估模型的泛化能力，算是直觀也相對可靠的做法。後續調整模型參數或訓練策略時，驗證集上的表現就是我最主要的參考依據。

### B. Optimizer Strategy（優化策略）
在模型訓練這部分，我選擇用 Adam 作為主要的最佳化方法。Adam 的好處是，它會根據每個參數的情況自動調整學習率，這樣在訓練初期，模型能比較快地往對的方向更新；到了訓練後期，步伐也會自然放緩，讓收斂更穩定，不用手動一直調學習率。

至於損失函數，我用的是 Cross-Entropy Loss，它很適合像這次的五類別分類任務。這個函數能清楚衡量模型預測出來的機率分布，和真實標籤之間有多大差距，並且讓模型朝著縮小這個差距的方向去學習，慢慢提高分類的準確率。

### C. Regularization Strategy（正規化策略）
為了防止模型過度擬合訓練資料，我在模型裡加入了在課堂中所學到的 Dropout 機制。它的做法是在訓練過程中，隨機丟棄一部分神經元的輸出，這樣可以強迫模型不要只依賴少數幾個特徵來做判斷。

透過這個正規化的方式，模型對於不同的輸入資料會更有適應力，也比較不會死背訓練集的樣本。從結果來看，加入 Dropout 之後，模型在驗證集上的表現通常更穩定，整體的泛化能力也比較好。

## 5. Experimental Results（實驗結果）

### 5.1 Cross Validation Performance
下表為各 Fold 在驗證集上的最佳表現：

| Fold | Best Validation Accuracy | Early Stopping Epoch |
| :--: | :----------------------: | :------------------: |
| 1    | 0.9975                   | 4                    |
| 2    | 1.0000                   | 4                    |
| 3    | 0.9988                   | 4                    |
| 4    | 1.0000                   | 6                    |
| 5    | 0.9988                   | 5                    |

由上表來看，實驗結果顯示了每個 Fold 的驗證準確率都接近于 1.0，表示模型在不同的資料切分下，皆能保持穩定且出色的分類表現。另外，Early Stopping 大多在第 4 到 6 個 Epoch 進行觸發，這樣的結果說明了模型在訓練初期能夠快速收斂的特性。

### 5.2 Overall Performance（OOF Accuracy）
| Metric       | Value  |
| :----------: | :----: |
| OOF Accuracy | 0.9990 |

此結果顯示模型對於未參與訓練的資料具有良好的泛化能力，且整體預測表現相當穩定。

### 5.3 Result Summary
從交叉驗證的實驗結果來看，五個 Fold 的驗證準確率都相當高，幾乎接近 1.0。這代表即使資料被切成不同組合來訓練，模型的表現依然穩定且出色，沒有因為資料切分方式不同而出現明顯波動。

另一個觀察是，Early Stopping 大多在第 4 到第 6 個 Epoch 就啟動了，這說明模型在訓練初期就能快速學習到有效的特徵，收斂速度很快，不需要經過太多輪訓練。

此外，整體的 OOF（Out-of-Fold）準確率達到 0.9990，這表示模型在「沒有參與訓練的資料」上表現依然非常好，泛化能力很強，預測結果一致且可靠，也顯示過擬合的情況被有效控制住了。

## 6. Difficulties Encountered（遇到的困難）
1. **理解資料格式與序列處理**：
這是我第一次使用 tokenized sequences 作為模型輸入，一開始不太清楚什麼是 token、為什麼需要 padding 或 truncation。當我嘗試把不同長度的序列送進模型時，程式會報錯，這才理解到神經網路要求輸入資料必須統一長度。透過查閱文件與範例，我學會了如何使用 pad_sequences 進行補齊，並理解了 padding 對模型訓練的影響。

2. **模型架構與層次理解**：
初次設計深度學習模型時，我對 Embedding、LSTM、Dropout 這些層次的作用不太熟悉。一開始不知道為什麼訓練集準確率可以很快上升，但驗證集表現不理想。後來學到 Dropout 的作用是降低過擬合，理解了序列模型如何捕捉文字前後文關係，也知道 Softmax 是用來輸出類別機率的。

3. **訓練與除錯過程耗時**：
由於模型訓練需要一定時間，一旦程式有錯誤，就必須重新訓練，耗費大量時間。初期我經常因小錯誤而必須重跑整個 Fold，才發現問題出在哪裡。為了改善，我學會了先用小批次資料測試程式流程，再進行完整訓練，這讓我體會到良好除錯習慣的重要性。

4. **初次使用深度學習工具與函式庫**：
這是我第一次完整使用 TensorFlow/Keras 建立模型並進行訓練。對於模型 compile、fit、EarlyStopping、optimizer 設定等功能都不熟悉，一開始容易弄混參數。透過查詢官方文件與範例程式，我慢慢理解各個參數的用途與對模型訓練的影響。

## 7. Conclusion & Learning（心得與學習）
這次的作業是我第一次真正動手實作一個深度學習的文字分類模型。從資料讀取、序列處理、模型設計、訓練到驗證，一步一步跟著我在網路查找的相關資料、文獻或是影片，同時搭配 AI 工具幫忙整理、釐清之後，再跟著一起實作。

在模型設計與訓練過程中，我除了學會了各個層的實際作用，也加深了在課程中所學習到的專業知識。例如 Embedding 層可以將 token 映射成向量表示，捕捉文字間的語意關係，或是 LSTM 層可以學習序列中的上下文資訊，甚至是 Dropout 能有效降低過擬合的風險。另外，我也實際使用了 Adam 優化器與 Cross-Entropy Loss，並透過 Early Stopping 監控驗證集表現。這些技巧雖然不是全部自己從零發想，但一邊參考、一邊修改、一邊觀察結果，讓我對整個流程的掌握度提高了不少。

此外，我也體會到調參的重要性。從設定最大序列長度、調整 Dropout 比例，到選擇適當的學習率和 batch size，每個設定都需要反覆的嘗試。再這樣調整的過程中，我也慢慢的開始理解如何判斷模型是否過擬合，並透過調整策略來提升表現。這個過程雖然花時間，但讓我對實作深度學習模型的流程與細節有了更進一步的認識。

最後，這次作業也讓我學到如何有效率地結合各種資源來解決問題。不管是透過 AI 解釋程式錯誤、查閱技術文章理解模型架構，還是觀看影片學習實作技巧，都漸漸讓我摸索出一套自己的學習與實作方法。雖然一開始常常卡關，但當模型終於順利運行、並且得到不錯的驗證結果時，讓我感到很有成就感，也讓我對之後對之後更複雜的專題增加了不少信心。

---