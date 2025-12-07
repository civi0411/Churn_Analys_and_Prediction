# 🔄 Customer Churn Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Tổng quan (Overview)

Dự án này không chỉ là một bài toán phân loại Machine Learning thông thường. Đây là một hệ thống **Software Engineering for Data Science** hoàn chỉnh, giải quyết bài toán dự đoán khách hàng rời bỏ (Customer Churn) cho lĩnh vực Thương mại điện tử (E-Commerce).

> 💡 Khác với việc chạy code trên Jupyter Notebook rời rạc, hệ thống này được xây dựng thành một **Pipeline khép kín**, có khả năng tái sử dụng (reproducible), dễ dàng mở rộng (scalable) và tích hợp sẵn quy trình **MLOps tự xây dựng** (Custom MLOps).

### 💼 Giá Trị Kinh Doanh (Business Value)

| Giá trị | Mô tả |
|---------|-------|
| 🎯 **Sàng lọc sớm** | Nhận diện khách hàng có nguy cơ rời bỏ với độ chính xác cao (F1-Score > 0.90) |
| 🔍 **Hiểu hành vi** | Sử dụng SHAP để giải thích lý do khách hàng rời bỏ (VD: Do thời gian giao hàng, hay do ít nhận được ưu đãi) |
| 💰 **Tối ưu chi phí** | Giúp bộ phận Marketing khoanh vùng đúng đối tượng để gửi voucher giữ chân, tránh lãng phí ngân sách |

---

## 🏗️ Kiến trúc hệ thống (System Architecture)

### Pipeline Flow

```mermaid
flowchart TB
    subgraph INPUT["📥 INPUT"]
        A[("RAW DATA<br/>Excel/CSV")]
    end

    subgraph STAGE1["📊 STAGE 1: EDA"]
        B["Load & Inspect<br/>• Missing value analysis<br/>• Correlation heatmaps<br/>• Distribution plots"]
    end

    subgraph STAGE2["🧹 STAGE 2: CLEANING"]
        C["Clean Data<br/>• Remove duplicates<br/>• Standardize categories<br/>• Basic validation"]
    end

    subgraph STAGE3["✂️ STAGE 3: SPLIT"]
        D["Stratified Split<br/>80% Train / 20% Test"]
        E["TRAIN SET"]
        F["TEST SET"]
    end

    subgraph STAGE4["⚙️ STAGE 4: TRANSFORM"]
        G["FIT on Train<br/>• Imputation<br/>• Outlier clipping<br/>• Feature engineering<br/>• Encoding<br/>• Scaling"]
        H[("LEARNED PARAMS<br/>Scaler, Imputer,<br/>Encoder, Bounds")]
        I["PROCESSED<br/>TRAIN"]
        J["TRANSFORM<br/>TEST"]
    end

    subgraph STAGE5["🎓 STAGE 5: TRAINING"]
        K["Model Training<br/>• SMOTE injection<br/>• Multiple models<br/>• Hyperparameter tuning<br/>• Cross-validation"]
        L["BEST MODEL<br/>RF / XGBoost / LR"]
    end

    subgraph STAGE6["📈 STAGE 6: EVALUATION"]
        M["Evaluation<br/>• Confusion matrix<br/>• ROC-AUC curves<br/>• Feature importance<br/>• SHAP explainability"]
    end

    subgraph STAGE7["📦 STAGE 7: REGISTRY"]
        N["Model Registry<br/>• Save best model<br/>• Version tracking<br/>• Metadata logging"]
    end

    subgraph STAGE8["👁️ STAGE 8: MONITORING"]
        O["Monitoring<br/>• Performance logging<br/>• Drift detection<br/>• Health checks"]
    end

    A --> B --> C --> D
    D --> E & F
    E --> G
    G --> H
    H --> I
    H -.->|"Apply learned params"| J
    F --> J
    I & J --> K --> L --> M --> N --> O

    style INPUT fill:#e1f5fe
    style STAGE1 fill:#fff3e0
    style STAGE2 fill:#e8f5e9
    style STAGE3 fill:#fce4ec
    style STAGE4 fill:#f3e5f5
    style STAGE5 fill:#e0f2f1
    style STAGE6 fill:#fff8e1
    style STAGE7 fill:#e8eaf6
    style STAGE8 fill:#fbe9e7
```

### 🔐 Key Principle: NO DATA LEAKAGE

```mermaid
flowchart LR
    subgraph TRAIN["🎯 TRAIN SET"]
        A["FIT + TRANSFORM"]
    end

    subgraph PARAMS["📊 LEARNED PARAMS"]
        B["Scaler params<br/>Imputer values<br/>Encoder mappings<br/>IQR bounds"]
    end
    
    subgraph TEST["🧪 TEST SET"]
        C["TRANSFORM ONLY"]
    end
    
    A -->|"Learn"| B
    B -->|"Apply"| C

    style TRAIN fill:#c8e6c9
    style PARAMS fill:#fff9c4
    style TEST fill:#ffcdd2
```

---

## 🧩 Logic xử lý (Logical Components)

```mermaid
graph TB
    subgraph PREPROCESSING["🔧 Preprocessing Module"]
        P1["DataCleaner"]
        P2["FeatureEngineer"]
        P3["Encoder"]
        P4["Scaler"]
        P5["FeatureSelector"]
    end
    
    subgraph MODELING["🤖 Modeling Module"]
        M1["BaseModel"]
        M2["RandomForest"]
        M3["XGBoost"]
        M4["LogisticRegression"]
        M5["HyperparamTuner"]
    end
    
    subgraph OPS["⚡ MLOps Module"]
        O1["ExperimentTracker"]
        O2["ModelRegistry"]
        O3["PerformanceMonitor"]
        O4["AlertSystem"]
    end
    
    subgraph UTILS["🛠️ Utilities"]
        U1["ConfigLoader"]
        U2["Logger"]
        U3["DataIO"]
    end
    
    PREPROCESSING --> MODELING
    MODELING --> OPS
    UTILS -.->|"Support"| PREPROCESSING
    UTILS -.->|"Support"| MODELING
    UTILS -.->|"Support"| OPS
```

---

## 📁 Project Structure

```
📦 Churn_Analysis_and_Predict/
├── 📂 config/
│   └── 📄 config.yaml                    # Central configuration file
│
├── 📂 src/                               # Source code modules
│   ├── 📄 __init__.py
│   ├── 📄 utils.py                       # Logging, IO, config handling
│   ├── 📄 preprocessing.py               # Data preprocessing pipeline
│   ├── 📄 modeling.py                    # Model training & evaluation
│   ├── 📄 visualization.py               # Plotting & visualization
│   ├── 📄 ops.py                         # MLOps: tracking, registry, monitoring
│   └── 📄 pipeline.py                    # Main pipeline orchestrator
│
├── 📂 data/                              # [WORKSPACE] Latest working files
│   ├── 📂 raw/
│   │   └── 📄 E_Commerce.xlsx            # Raw input data
│   ├── 📂 processed/
│   │   └── 📄 E_Commerce_cleaned.parquet # Cleaned data
│   └── 📂 train_test/
│       ├── 📄 E_Commerce_train.parquet   # Latest train split
│       └── 📄 E_Commerce_test.parquet    # Latest test split
│
├── 📂 artifacts/                         # [ARCHIVE] Historical results
│   ├── 📂 experiments/                   # Run-specific snapshots
│   │   └── 📂 20251205_203000_FULL/      # Example run ID
│   │       ├── 📄 config_snapshot.yaml
│   │       ├── 📄 params.json
│   │       ├── 📄 metrics.json
│   │       ├── 📄 run.log
│   │       ├── 📂 figures/
│   │       │   ├── 📂 eda/
│   │       │   └── 📂 evaluation/
│   │       ├── 📂 models/
│   │       │   ├── 📄 preprocessor.joblib
│   │       │   └── 📄 xgboost.joblib
│   │       └── 📂 data/
│   │           ├── 📄 processed.parquet
│   │           ├── 📄 train.parquet
│   │           └── 📄 test.parquet
│   │
│   ├── 📂 model_registry/                # Production models
│   │   ├── 📄 registry.json
│   │   └── 📄 xgboost_v1_20251205.joblib
│   │
│   ├── 📂 versions/                      # Data lineage tracking
│   │   └── 📄 versions.json
│   │
│   ├── 📂 monitoring/                    # Model monitoring data
│   │   ├── 📄 performance_log.csv
│   │   └── 📄 alerts_log.csv
│   │
│   └── 📂 logs/
│       └── 📄 MAIN_20251205.log
│
├── 📄 main.py                            # CLI entry point
├── 📄 requirements.txt                   # Python dependencies
└── 📄 README.md                          # This file
```

---

## 🚀 Hướng dẫn sử dụng (Usage Instructions)

### 1️⃣ Clone repository

```bash
git clone https://github.com/civi0411/Churn_Analysis_and_Predict.git
cd Churn_Analysis_and_Predict
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure settings

Edit `config/config.yaml` to set paths, model parameters, etc.

### 5️⃣ Usage Modes

#### 🔄 Full Pipeline (Recommended)

Tự động thực hiện: Clean → Split → Train → Optimize → Visualize → Save Artifacts

```bash
python main.py --mode full --model xgboost --optimize
```

#### 🔧 Step-by-Step Execution

```bash
# Exploratory Data Analysis
python main.py --mode eda

# Preprocessing
python main.py --mode preprocess

# Train specific model
python main.py --mode train --model xgboost

# Full pipeline with all models
python main.py --mode full

# Visualization only
python main.py --mode visualize
```

### 6️⃣ Chạy kiểm thử (Testing)

```bash
# Chạy toàn bộ test
pytest

# Chạy riêng phần Unit Test (Test hàm lẻ)
pytest tests/test_module

# Chạy riêng phần Integration Test (Test luồng)
pytest tests/test_flow
```

---

## 📊 Model Comparison

```mermaid
xychart-beta
    title "Model Performance Comparison"
    x-axis ["Random Forest", "XGBoost", "Logistic Reg"]
    y-axis "Score" 0 --> 1
    bar [0.89, 0.92, 0.85]
    line [0.87, 0.90, 0.83]
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/civi0411">civi0411</a>
</p>

