---
title: Music Scale Recognition Using Mel-cepstrum - HW2 Report

---

# Music Scale Recognition Using Mel-cepstrum - HW2 Report

**Student：** 蔡東廷
**Student ID：** 411221315
**Kaggle Username：** tonylemty
**Kaggle ID：** 411221315

---

## 1. Introduction（前言）
本次作業主要是利用卷積神經網路 (CNN) 解決音樂音階辨識問題。藉由實際去撰寫解決音樂音階辨識的程式碼，深入了解卷積神經網路的相關知識，以及音樂音階辨識其對於音樂資訊檢索的基礎任務、自動記譜以及樂器教學等應用的實用價值。

老師所給予的輸入資料為音訊訊號經處理後轉換成的 **Mel-cepstrum (梅爾倒頻譜)** 二維特徵圖，其中這種表示方法結合了人耳聽覺特性與頻譜資訊，讓音訊特徵得以圖像形式呈現。因此，本任務實質上是一個圖像分類問題，需要將每個梅爾倒頻譜圖像分類至對應的音階，總共 **88 個類別**，分別對應到鋼琴的 88 個琴鍵。

接下來，我將分成資料處理、模型架構、設計亮點與策略和實驗結果這幾個部分，說明本次模型的建置過程與表現。


## 2. Data Processing（資料處理）

### 2.1 預處理步驟

#### 尺寸統一化（Resize）
將所有圖片統一縮放至 `128x128` 像素。
* 原因：原始圖像的尺寸皆不相同，統一尺寸可確保每一次的訓練能夠順利進行，同時在能夠保留足夠時頻特徵的前提下降低計算的負擔。
* 影響：可能損失部分高頻細節，但在本任務中已能捕捉主要音階特徵，並大幅加速訓練。

#### 標準化（Normalization）
使用 ImageNet 的統計值進行像素值標準化：
```text=
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```
* 原因：採用 ImageNet 統計值有助於：
    * 若使用預訓練模型，能購更好地進行遷移學習
    * 穩定梯度更新，加速模型收斂
    * 將像素值縮放到相近範圍，避免數值不穩定

#### 資料增強（Data Augmentation）

| 增強方法 | 參數設定 | 目的 |
| :---: | :---: | :---: |
| **ColorJitter** | `brightness=0.2, contrast=0.2, saturation=0.1` | 模擬錄音音量、設備差異造成的亮度與對比變化 |
| **RandomHorizontalFlip** | `p=0.5` | 時間軸翻轉不會影響音高標籤，能夠增加序列不變性 |
| **RandomRotation** | `degrees=±8` | 模擬頻率微小偏移或顫音效果，提升對音高波動的魯棒性 |


### 2.2 程式碼實作

#### 訓練集轉換流程
```python=
# 定義影像增強 (Data Augmentation) 的流程
def get_train_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)), # 1. 強制縮放
        # 2. 隨機色彩抖動：亮度、對比、飽和度
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),  # 3. 隨機水平翻轉
        transforms.RandomRotation(8),            # 4. 隨機旋轉 (+-8度)
        transforms.ToTensor(),                   # 5. 轉成 Tensor 格式
        # 6. 正規化 (數值減去平均除以標準差)
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
```

#### 驗證 / 測試集轉換流程
```python=
def get_val_test_transforms(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),      # 僅縮放
        transforms.ToTensor(),                        # 轉為 Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )                                             # 標準化（與訓練集一致）
    ])
```

## 3. Model Architecture（模型架構）

### 3.1 骨幹選擇
本次作業我選用 ResNet18 作為核心骨幹網路。ResNet 透過殘差連接 解決深層網路梯度消失問題，在 ImageNet 等大規模數據集上具有優秀性能。

#### 選擇原因
1. 計算效率：由於 Kaggle 的計算資源有限，ResNet18 訓練速度快，適合快速迭代。
2. 深度適中：88 類分類任務不需要極深的網路，18 層就已經可以學習足夠的特徵。
3. 遷移學習潛力：雖然本次是從頭訓練，但若需使用預訓練權重，ResNet 系列的支援更加完善。

### 3.2 模型修改與適配

#### 輸出層調整
```python=
def build_model(num_classes=NUM_CLASSES, pretrained=False):
    # 1. 載入 ResNet18 架構
    model = models.resnet18(pretrained=pretrained)
    
    # 2. 取得最後一層的輸入特徵維度
    in_features = model.fc.in_features
    
    # 3. 替換輸出層：1000類 → 88類
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
```

#### 調整原因說明
由於本次任務需要對 88 個音階 進行分類，而 ResNet18 的預設輸出層為 1000 類（ImageNet），直接使用會造成多餘神經元，浪費計算資源並可能影響梯度更新。因此，我將最後的全連接層改為 512 → 88，每個神經元對應一個音階。

1. 任務適配性
    * 類別數量一致，輸出對齊任務需求，避免多餘機率值影響收斂。
2. 參數效率與過擬合風險降低
    * 原始輸出層有約 512,000 個參數，修改後僅 45,000 個參數，減少 91%。
    * 減少不必要的參數可以降低過擬合風險，特別是在中等規模資料集上更有效。
3. 梯度傳播考量
    * 新的全連接層權重隨機初始化，梯度大小與任務匹配，有利於從頭訓練模型。


## 4. Methodology & Novelty（設計亮點與策略）

### A. K-Fold Cross Validation（K-折交叉驗證）
我沒有使用單一的 Train/Validation 切分，而是使用了 **Stratified 5-Fold Cross Validation**。
* **作法：** 將資料分成 5 份，輪流取一份當驗證集，其餘四份當訓練集，總共訓練 5 個模型。
* **優點：** 確保每一筆資料都被測試過，能更客觀地評估模型效能，且利用 5 個模型的預測結果進行投票 (Ensemble)，能大幅提升準確度。

由於本任務有 88 類細分類問題，且部分音階樣本數較少。因此，使用 Stratified K-Fold 能夠避免某些音階在驗證集中缺失，提升評估的穩定性。

### B. Optimizer Strategy（優化策略）
* **Optimizer：** 使用 `AdamW`，相比傳統 Adam，它對權重衰減 (Weight Decay) 的處理更好，能提升泛化能力。
* **Scheduler：** 使用 `CosineAnnealingLR`，讓 Learning Rate 隨著訓練過程呈現餘弦下降。這有助於模型在初期快速收斂，後期細微調整找到全域最優解。

### C. Test Time Augmentation（TTA, 測試時增強）
在預測階段（Inference），我不只預測原始圖片，
我還對測試圖片進行了水平翻轉，並將兩次預測的機率取平均。

對於 Mel-cepstrum 而言，時間軸的左右翻轉不會改變音高本質，
因此 TTA 不會破壞標籤語意，適合用於穩定模型輸出。

* **效果：** 這種做法可以消除模型對於方向的敏感度，通常能穩定提升 1%~2% 的準確率。

### D. Novelty / 相比 baseline
* 與單純使用 CNN 或單一訓練 / 驗證集相比：
    1. Stratified K-Fold + 投票提升了小類別樣本準確率。
    2. TTA 與資料增強增加模型魯棒性。
    3. AdamW + CosineAnnealingLR 提升收斂穩定性。
* 這些策略組合是本作業相對於傳統 CNN 訓練的創新之處。

## 5. Experimental Results（實驗結果）

| Fold | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 0.9987 | **1.0000** | 0.1074 | 0.0430 |
| **Fold 1** | 0.9996 | **1.0000** | 0.0920 | 0.0341 |
| **Fold 2** | 0.9974 | **1.0000** | 0.1073 | 0.0375 |
| **Fold 3** | 1.0000 | **1.0000** | 0.1248 | 0.0438 |
| **Fold 4** | 0.9974 | **1.0000** | 0.1246 | 0.0478 |
| --- | --- | --- | --- | --- |
| **平均 (Avg)** | **0.9986** | **1.0000** | **0.1112** | **0.0412** |

從以上的實驗結果可以觀察到，五個 Fold 的驗證準確率皆達到 100%。推測原因在於 Mel-cepstrum 特徵本身對音高具有高度辨識性，使得不同音階在特徵空間中具有明顯區隔。此外，透過 K-Fold 交叉驗證與資料增強，有效降低了過擬合的風險，使模型在各個 Fold 上能夠穩定收斂。

## 6. Difficulties Encountered（遇到的困難）
1.  **資料路徑讀取問題：**
在實作初期，我常常因為 Kaggle 與本地端的檔案路徑設定方式不同而遇到問題，導致程式明明在本地端可以正常執行，但是到 Kaggle 上卻讀不到資料。所以我為了避免一直手動修改路徑，我後來就採用在 GPT 的建議，改用 `pathlib` 搭配 `iterdir` 撰寫自動偵測資料位置的程式碼，讓程式可以自行搜尋訓練標註檔（`train_truth.csv`）以及對應的影像資料夾。透過這樣的方式，讓我不論是在本地端還是在 Kaggle 環境中執行，都能夠找到正確的資料路徑，同時也減少了因環境差異所造成的錯誤。

2. **本地開發環境與 Kaggle Notebook 相容性問題：**
在本地端使用 VSCode 進行開發與測試時，程式碼可以正常執行，但直接將同一份 Python 程式搬移至 Kaggle 上執行時，常會因為資料路徑結構、執行方式不同而導致錯誤。因此，為了解決此問題，我參考了 Kaggle 官方文件以網路上相關社群的資料，除不必要的命令列參數設定，將原本偏向本地端執行的程式碼修改為可直接在 Kaggle 上執行的形式。

## 7. Conclusion & Learning（心得與學習）

這次實作 CNN 的作業，真的讓我學到很多課本上沒教的東西。課堂上聽老師講模型架構、講怎麼訓練，都覺得大概懂了，但自己動手做完全是另一回事。真的要自己處理資料、寫訓練流程、跑測試，才發現很多細節之前根本沒想過，像資料要怎麼切、怎麼做 augmentation 才能真的幫到模型，這些都是實際做過才會有感覺。

整個過程其實蠻痛苦的，寫程式就是一直撞牆。常常程式跑下去不是出現一堆 錯誤，就是 loss 根本不會動，不然就是模型隨便亂猜。很多時間都在 debug，看錯誤訊息、查資料、上論壇爬文，有時候一個小 bug 可以卡好幾個小時。最煩的是，就算程式能跑了，模型表現還是很爛，這時候就要回頭調參數、改架構、換訓練方式，一直試一直試。

這次作業掛在 Kaggle 上做，多了競賽排名的壓力，但也讓我更認真去優化模型。看到自己調一調，準確率從 80% 慢慢爬到 90% 以上，甚至最後衝到接近 100%，雖然過程很燒腦，但真的蠻有成就感的。我也開始會去看別人的 public notebook，學別人怎麼做特徵工程、用什麼模型，這些都是實戰才會累積的經驗。

總而言之，這次作業讓我對 CNN 和整個深度學習流程更有手感了。現在回頭看一些熱門的模型像 ResNet、VGG，比較能看懂它們為什麼要這樣設計，也大概知道怎麼根據任務選模型。雖然做作業很累，但能實際做出一個會動、而且效能不錯的模型，感覺還是很值得的。

---