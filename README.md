# Presentation-Advisor-Recommender-System
Project Description

The Presentation Advisor Recommender System is a multi-model framework that enhances users’ presentation skills through personalized article recommendations. It implements and compares five approaches, ranging from traditional filtering methods to advanced deep learning models.

The system uses user interactions, presentation metadata, and article features to generate tailored recommendations based on user type, audience, presentation style, past issues, and temporal behavior. It also handles sparse data and cold-start problems using feature-based modeling and time-aware weighting, prioritizing recent activity.

The pipeline includes a hybrid content-based and collaborative filtering baseline, a denoising autoencoder for learning latent preferences, and a Deep Q-Network (DQN) for sequential decision-making. The main contribution is a hybrid multi-tower neural network that processes multiple feature groups independently before combining them for prediction, further improved using pre-trained embeddings.

All models are evaluated using MAE, MSE, and RMSE, with EDA supporting model design. Overall, the project focuses on building a scalable, personalized, and high-performance recommendation system for presentation skill improvement.

Objective
This project builds and evaluates a complete recommender system pipeline for a Presentation Advisor application:

* Implement five recommendation models of increasing complexity
* Compare all models on MAE, MSE, and RMSE across a shared test set
* Analyze the effect of temporal encoding, problem sequence weighting, and utility-based scoring
* Validate that deep hybrid architectures outperform traditional and RL-based approaches on this domain

Repository Structure
```text
Presentation-Advisor-Recommender-System/
├── Model_1/                        # Baseline Hybrid Approach
│   ├── model1.ipynb                # CBF & Collaborative Filtering
│   ├── cbf_cf_metrics.json         # Performance metrics
│   ├── cbf_cf_predictions.csv      # Sample output predictions
│   └── cbf_cf_results.png          # Visualization of baseline
│
├── model_2_auto_encoders/          # Neural Network Recommenders
│   ├── autoencodersmodel2.ipynb    # Deep learning autoencoder
│   ├── best_autoencoder.keras      # Saved model weights
│   ├── autoencoder_metrics.json    # Accuracy and loss logs
│   └── autoencoder_results.png     # Training history plots
│
├── model_3/                        # Reinforcement Learning Agent
│   ├── model-3-dqn.ipynb           # Deep Q-Network implementation
│   ├── dqn_model.keras             # Saved RL agent
│   ├── dqn_metrics.json            # Reward tracking
│   └── dqn_results.png             # Agent performance charts
│
├── Model_4/                        # Optimized Hybrid Model
│   ├── model-4.ipynb               # Refined hybrid architecture
│   ├── hybrid_best.keras           # Best iteration weights
│   ├── hybrid_metrics.json         # Evaluation results
│   └── hybrid_results.png          # Comparison graphs
│
├── model_5/                        # Embedding-Based Hybrid
│   ├── model-5.ipynb               # Advanced embeddings (User/Item)
│   ├── user_encoder.keras          # Serialized user embedding
│   ├── item_encoder.keras          # Serialized item embedding
│   └── hybrid_emb_results.png      # Embedding space visualizations
│
├── comparison_results/             # Cross-Model Evaluation
│   ├── comparison.ipynb            # Benchmarking all 5 models
│   └── all_models_comparison.png   # Final performance leaderboard
├── plots/                          # EDA Plota
│   ├── all_models_comparison.png
│   ├── eda_article_distribution.jpg
│   ├── eda_ratings.jpg
│   ├── eda_temporal_trends.jpg
│   ├── eda_user_preferences.jpg
│   └── eda_user_types.jpg
│
└── presentation_data/              # Core Datasets
    ├── presentations.csv           # Content metadata
    ├── interaction_data.csv        # User-item interactions
    ├── ratings_matrix.csv          # Pivot table for CF
    └── user_profiles.csv           # User preference data
```
📊 Dataset Overview
The dataset contains 25,000 user-article interactions across 50 users and 500 articles, with ratings from 2.0 to 5.0 and an average of ~3.09.
| File Name | Shape | Description |
| :--- | :--- | :--- |
| `interaction_data.csv` | 25,000 × 75 | Full interaction data with all user and item features |
| `ratings_matrix.csv` | 3,000 × 5 | Clean ratings for CF-based models |
| `recommendations.csv` | 100 × 26 | Article content features and metadata |
| `user_profiles.csv` | 30 × 5 | User type, location, and preferences |
| `presentations.csv` | 78 × 15 | Presentation issues flagged per user |

Feature groups:
| Group | Count | Examples |
| :--- | :--- | :--- |
| **User preferences** | 18 | readability, graphics, agenda, text-heavy |
| **User types** | 7 | business, teacher, student, researcher |
| **Presentation types** | 6 | formal, creative, technical, educational |
| **Audience types** | 12 | academic, executive, kids, general |
| **Article issues** | 12 | bullets, consistency, images, positioning |
| **Time features** | 5 | hour_sin, hour_cos, day_sin, day_cos, hours_since_first |
| **Problem weights** | 12 | exponential decay weighted issue flags |

## 🤖 Model Architectures

---

###  Model 1: Hybrid Baseline (CBF + CF)
A hybrid approach combining collaborative filtering with content-based similarity.

- 🔹 **CF Component:** User-user cosine similarity on rating matrix  
- 🔹 **CBF Component:** Item-item cosine similarity on article features  

**Prediction Formula:**

Score = alpha * CF_score + (1 - alpha) * CBF_score
alpha = 0.5


---

###  Model 2: Denoising Collaborative Filtering Autoencoder
Learns latent user preferences from sparse rating data and reconstructs missing values.

**Architecture:**

Input(500)
→ Dropout(0.2)
→ Dense(256) → BatchNorm
→ Dense(128)
→ Latent(32)
→ Dense(128)
→ Dense(256)
→ Output(500)


---

###  Model 3: Reinforcement Learning (DQN)
Models recommendation as a sequential decision-making problem using Q-learning.

**Architecture:**

Dense(256)
→ BatchNorm
→ Dropout(0.2)
→ Dense(128)
→ BatchNorm
→ Output(Q-values)


---

###  Model 4: Hybrid Multi-Tower Neural Network (Primary)
Processes multiple feature groups independently before fusion.

####  Time Encoding

hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)


####  Problem Weighting

weight = exp(-0.01 * days_since_interaction)


---

###  Model 5: Hybrid + Pre-Trained Embeddings
Enhances Model 4 by incorporating pre-trained user and item embeddings using denoising autoencoders.

---

##  Key Takeaways
- Combines **traditional + deep learning models**
- Handles **cold-start + sparse data**
- Uses **temporal + behavioral signals**
- Demonstrates **performance improvement with hybrid architectures**

## 📊 Results

All models are evaluated on a 70/15/15 train/validation/test split with a held-out 15% test set.

### 🔹 Model Performance Comparison

| Model                     | MAE     | MSE     | RMSE    | Paper MAE | Beats Paper |
|--------------------------|--------|--------|--------|-----------|-------------|
| CBF + CF (Hybrid)        | 0.5255 | 0.3908 | 0.6252 | 1.13      |  Yes       |
| Autoencoder              | 0.7861 | 0.7964 | 0.8924 | 3.05      |  Yes       |
| DQN                      | 1.4968 | 3.3036 | 1.8176 | 3.22      |  Yes       |
| Hybrid Multi-Tower       | 0.0734 | 0.0259 | 0.1611 | 0.11      |  Yes       |
| Hybrid + Embeddings      | 0.3324 | 0.2514 | 0.5014 | 0.49      |  Yes       |

---

### ⚡ Key Observations

-  **Hybrid Multi-Tower model performs best** with the lowest MAE (0.0734)
-  Deep learning models significantly outperform traditional methods
-  Autoencoder improves over baseline but is less effective than hybrid models
-  DQN underperforms due to complexity and reward sparsity
-  All models outperform the research paper benchmarks

---

### ⏱️ Training Summary

| Model                | Training Time | Epochs / Steps |
|---------------------|--------------|----------------|
| CBF + CF            | ~1.1 sec     | —              |
| Autoencoder         | 17 sec       | 43 epochs      |
| DQN                 | 4 min        | 9000 steps     |
| Hybrid Multi-Tower  | 7 min        | 200 epochs     |
| Hybrid + Embeddings | 2 min        | 68 epochs      |

---

## 📊 Exploratory Data Analysis (EDA)

A detailed exploratory analysis was performed to understand user behavior, interaction patterns, and feature distributions. These insights directly influenced model design choices such as hybrid architectures, temporal encoding, and feature engineering.

---

### 📉 Rating Distribution

The rating values range from **2.0 to 5.0**, with a mean around **3.0**, indicating a realistic and moderately positive feedback distribution. The data shows a slight right skew, suggesting users tend to give average-to-good ratings rather than extreme values.

<p align="center">
  <img src="plots/eda_ratings.jpg" width="650"/>
</p>

**Insights:**
- No extreme rating bias → suitable for regression-based models  
- Mid-range clustering → harder prediction problem (less trivial patterns)  
- Justifies use of deep learning over simple heuristics  

---

### 📊 Article Interaction Distribution

Most articles receive a **uniform number of ratings**, while a small subset shows slightly higher engagement.

<p align="center">
  <img src="plots/eda_article_distribution.jpg" width="650"/>
</p>

**Insights:**
- Dataset is relatively balanced → reduces popularity bias  
- No heavy long-tail problem → CF models remain effective  
- Ensures fair learning across items  

---

### 🎯 User Preferences Analysis

User preference features highlight the most common presentation challenges such as **readability, graphics, and agenda structuring**.

<p align="center">
  <img src="plots/eda_user_preferences.jpg" width="650"/>
</p>

**Insights:**
- Strong signal for **problem-driven recommendations**  
- Justifies inclusion of a dedicated **problem feature tower**  
- Helps model generate actionable and personalized suggestions  

---

### 👥 User Types & Presentation Styles

The dataset includes multiple user types (business, student, technical, etc.) and presentation styles (formal, creative, educational).

<p align="center">
  <img src="plots/eda_user_types.jpg" width="650"/>
</p>

**Insights:**
- High diversity → rules out one-size-fits-all recommendation  
- Supports need for **personalized embeddings**  
- Validates multi-tower architecture (separating user & item signals)  

---

### ⏱️ Temporal Analysis

User interactions span across a long time range, with noticeable fluctuations in activity and rating patterns.

<p align="center">
  <img src="plots/eda_temporal_trends.jpg" width="650"/>
</p>

**Insights:**
- User preferences evolve over time → static models are insufficient  
- Justifies **time-aware features (sin/cos encoding)**  
- Supports use of **exponential decay weighting** for recent interactions  

---

## 🔍 Key Takeaways

-  Ratings are moderately distributed → non-trivial prediction task  
-  User behavior is **multi-dimensional (type + preference + time)**  
-  Temporal dynamics are critical → recent interactions matter more  
-  Problem-specific features strongly influence recommendations  
-  Dataset structure supports advanced hybrid models over simple CF  

---

## 🚀 Impact on Model Design

Based on EDA findings:

- Hybrid models were used to combine **collaborative + content signals**  
- Multi-tower architecture was designed to **avoid feature interference**  
- Temporal encoding and decay were added to capture **behavior evolution**  
- Problem-weighted features were introduced to ensure **relevance**  

## 🔍 Critical Analysis

### 🚀 Why Deep Models Outperform Traditional Approaches
The dataset contains rich contextual features such as user type, audience type, presentation style, temporal patterns, and problem history. Traditional methods like collaborative filtering (CF) and content-based filtering (CBF) fail to fully capture these multi-dimensional signals.

The hybrid multi-tower architecture addresses this by processing each feature group independently, preventing high-cardinality features from dominating simpler binary signals during training.

---

### ⚠️ Factors Affecting Performance Metrics

- **Limited dataset size:** 25,000 interactions across only 50 users  
- **Compressed rating scale:** Ratings range from 2–5 (no extreme values)  
- **Weak collaborative signal:** Small user base limits CF effectiveness  
- **Feature dependency:** Models rely more on content and contextual features  

---

### 📈 Future Improvements

- Increase number of users and articles to strengthen collaborative learning  
- Incorporate real user feedback for tuning utility-based scoring  
- Apply sequential modeling (e.g., Transformers) on user interaction history  
- Use contrastive learning for better representation of presentation issues  

---

## ⚙️ Installation

```bash id="u3r7xp"
git clone https://github.com/yourusername/presentation-advisor-recommender.git
cd presentation-advisor-recommender
pip install -r requirements.txt 
📦 Requirements
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.64.0
```
▶️ How to Run
```
# Step 1 — EDA
jupyter notebook notebooks/01_EDA.ipynb

# Step 2 — Model 1: CBF + CF
jupyter notebook notebooks/02_Model1_CBF_CF.ipynb

# Step 3 — Model 2: Autoencoder
jupyter notebook notebooks/03_Model2_Autoencoder.ipynb

# Step 4 — Model 3: DQN
jupyter notebook notebooks/04_Model3_DQN.ipynb

# Step 5 — Model 4: Hybrid Multi-Tower
jupyter notebook notebooks/05_Model4_Hybrid.ipynb

# Step 6 — Model 5: Hybrid + Embeddings
jupyter notebook notebooks/06_Model5_Hybrid_Emb.ipynb

# Step 7 — Final Comparison
jupyter notebook notebooks/07_Final_Comparison.ipynb
```
📥 Dataset (Kaggle)
```
kaggle datasets download janvijain96/presentation-advisor-data
```
Each notebook automatically saves metrics to saved_models/.
The final comparison notebook loads all saved JSON files to generate results without retraining.


## 👩‍💻 Author

Janvi Jain
BTech – Data Science

## 📚 Reference

Vlahova-Takova, M., & Lazarova, M. (2025).
A Recommender System Model for Presentation Advisor Application Based on Multi-Tower Neural Network and Utility-Based Scoring.
Electronics, 14(13), 2528.
https://doi.org/10.3390/electronics14132528
