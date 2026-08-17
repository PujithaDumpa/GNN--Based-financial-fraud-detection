# GNN-Based Financial Fraud Detection Using Graph Attention Networks

A graph-based deep learning system for detecting potentially fraudulent cryptocurrency transactions using **Graph Neural Networks (GNNs)** and **Graph Attention Networks (GATs)**.

## 📌 Overview

Traditional machine learning models treat transactions as independent records. However, financial transactions are naturally connected: transactions can interact with other transactions through shared addresses, entities, or transaction relationships.

This project represents cryptocurrency transactions as a **graph** and uses a **Graph Attention Network (GAT)** to learn both:

* Transaction-level features
* Relationships between connected transactions

The model classifies transactions into legitimate and fraudulent/suspicious categories and provides prediction confidence and graph-based explainability.

## 🎯 Objectives

* Detect potentially fraudulent cryptocurrency transactions.
* Represent transactions as a graph instead of independent tabular records.
* Apply Graph Neural Networks to exploit relationships between transactions.
* Use Graph Attention Networks to identify influential neighboring transactions.
* Evaluate the model using precision, recall, F1-score, accuracy, and ROC-AUC.
* Provide an interactive interface for transaction-level fraud prediction and explanation.

## 🧠 Why GNNs?

Financial fraud often involves relationships between multiple transactions.

For example:

```text
Transaction A ─── Transaction B ─── Transaction C
       │                  │
       └──────── D ───────┘
```

A traditional classifier mainly learns from the features of one transaction.

A GNN can additionally aggregate information from neighboring transactions:

```text
Target Transaction
       ↓
Neighbor Transactions
       ↓
Graph Message Passing
       ↓
Learned Representation
       ↓
Fraud Prediction
```

This allows the model to capture patterns that may not be visible from individual transaction features alone.

## 📊 Dataset

This project uses the **Elliptic Bitcoin Transaction Dataset**.

The dataset contains Bitcoin transaction information represented as a temporal transaction graph.

Each transaction contains:

* Transaction-level features
* Connections to other transactions
* Class labels for supervised learning

The graph used in the project contains approximately:

* **203,769 transactions (nodes)**
* **234,355 relationships (edges)**
* **166 input features per transaction**

The dataset contains labeled and unlabeled transactions, with the labeled transactions used for model training and evaluation.

## 🏗️ System Architecture

```text
                    Elliptic Dataset
                           │
                           ▼
                 Data Preprocessing
                           │
                           ▼
              Graph Construction
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Node Features                Edge Information
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    Graph Data Object
                           │
                           ▼
                Graph Attention Network
                           │
                           ▼
                  Message Passing
                           │
                           ▼
                 Attention Mechanism
                           │
                           ▼
                   Fraud Prediction
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Classification              Explainability
             │                           │
             ▼                           ▼
      Fraud / Legitimate        Important Neighbors
                                  & Features
```

## 🔬 Model Architecture

The main model is based on **Graph Attention Networks (GAT)**.

The architecture consists of multiple graph attention layers.

```text
Input Features
      │
      ▼
   GAT Layer
      │
      ▼
 Batch Normalization
      │
      ▼
   GAT Layer
      │
      ▼
   Output Layer
      │
      ▼
 Fraud Classification
```

### Graph Attention

Unlike standard GCNs, GAT learns different importance weights for neighboring nodes.

For a target transaction:

```text
              Neighbor A
                 ↓
                 ↓ α₁
Neighbor B → Target Transaction ← Neighbor C
                 ↑
                 ↑ α₃
              Neighbor D
```

The attention mechanism learns which neighboring transactions are more important for making the final prediction.

## ⚙️ Technologies Used

| Technology        | Purpose                             |
| ----------------- | ----------------------------------- |
| Python            | Programming language                |
| PyTorch           | Deep learning framework             |
| PyTorch Geometric | Graph Neural Network implementation |
| Pandas            | Data preprocessing                  |
| NumPy             | Numerical operations                |
| Scikit-learn      | Evaluation metrics                  |
| Matplotlib        | Visualization                       |
| Streamlit         | Interactive application             |
| Hugging Face Hub  | Model/data hosting                  |

## 🧪 Training

The model is trained using supervised learning on labeled transaction nodes.

The training process consists of:

1. Loading the transaction dataset.
2. Constructing the graph.
3. Preparing node features and labels.
4. Splitting labeled nodes into training, validation, and test sets.
5. Performing graph message passing through GAT layers.
6. Computing classification loss.
7. Updating model parameters using Adam.
8. Evaluating the model on unseen transactions.

### Example Hyperparameters

```text
Hidden Channels : 128
Attention Heads : 8
Learning Rate   : 0.005
Weight Decay    : 0.0005
Optimizer       : Adam
Scheduler       : ReduceLROnPlateau
Early Stopping  : Enabled
```

These values can be modified during experimentation.

## 📈 Model Performance

The GAT model achieved strong classification performance on the evaluation data.

Example evaluation results:

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        |   ~98% |
| ROC-AUC         | ~0.982 |
| Fraud Precision |  ~0.88 |
| Fraud Recall    |  ~0.87 |
| Fraud F1-Score  |  ~0.87 |

> Results may vary depending on the train/validation/test split, preprocessing, model configuration, and hyperparameters.

## 🔍 Explainability

A major component of the project is understanding **why the model predicts a transaction as fraudulent**.

The system explores graph-based explanations such as:

### 1. Important Neighboring Transactions

The GAT attention mechanism can be used to identify neighboring transactions that receive higher attention weights.

```text
Target Transaction
       │
       ├── Neighbor 1 → Attention: 0.41
       ├── Neighbor 2 → Attention: 0.27
       ├── Neighbor 3 → Attention: 0.18
       └── Neighbor 4 → Attention: 0.14
```

This helps identify which connected transactions influenced the prediction.

### 2. Important Features

GNN explainability techniques can also identify features that contribute strongly to a transaction's prediction.

### 3. Transaction-Level Explanation

For a selected transaction, the application can display:

* Predicted class
* Prediction confidence
* Important neighboring transactions
* Attention weights
* Important features

This makes the model more interpretable than simply returning a fraud/legitimate label.

## 🖥️ Streamlit Application

The project includes an interactive Streamlit interface.

The user can enter a transaction/node identifier and obtain a prediction.

Example workflow:

```text
Enter Transaction ID
        ↓
Load Transaction
        ↓
Run GAT Model
        ↓
Generate Prediction
        ↓
Display Confidence
        ↓
Show Important Neighbors
        ↓
Show Explanation
```

Example output:

```text
Transaction: 1234

Prediction: Suspicious

Confidence: 78.16%

Important Neighbors:
1. Node XXXX
2. Node XXXX
3. Node XXXX
```

## 📁 Project Structure

```text
gnn-fraud-detection/
│
├── data/
│   └── README.md
│
├── models/
│   └── gat_model.py
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── gcn_training.ipynb
│   └── gat_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── graph_construction.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
cd gnn-fraud-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧩 Key Features

* Graph-based fraud detection
* Graph Attention Network architecture
* Multi-head attention
* Transaction-level prediction
* Prediction confidence
* Attention-based neighbor analysis
* GNN explainability
* Interactive Streamlit interface
* Model evaluation using multiple classification metrics

## 🔮 Future Improvements

Possible extensions include:

* Temporal GNNs for better modeling of transaction time.
* GraphSAGE and other GNN architectures for comparison.
* Graph-level fraud pattern detection.
* Advanced explainability using GNNExplainer.
* Hyperparameter optimization.
* Class-imbalance handling using focal loss or weighted loss.
* Real-time transaction monitoring.
* Anomaly detection for previously unseen fraud patterns.
* Improved visualization of suspicious transaction subgraphs.
* Integration with a real-time fraud detection pipeline.

## 📚 References

* Elliptic Bitcoin Dataset
* Graph Attention Networks — Veličković et al.
* PyTorch Geometric documentation
* PyTorch documentation

## 👩‍💻 Author

**Pujitha Dumpa**

B.Tech — Computer Science and Engineering

---

⭐ If you find this project useful, consider giving the repository a star.
