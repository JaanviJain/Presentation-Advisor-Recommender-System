# Presentation-Advisor-Recommender-System
📌Project Description

The Presentation Advisor Recommender System is a multi-model framework that enhances users’ presentation skills through personalized article recommendations. It implements and compares five approaches, ranging from traditional filtering methods to advanced deep learning models.

The system uses user interactions, presentation metadata, and article features to generate tailored recommendations based on user type, audience, presentation style, past issues, and temporal behavior. It also handles sparse data and cold-start problems using feature-based modeling and time-aware weighting, prioritizing recent activity.

The pipeline includes a hybrid content-based and collaborative filtering baseline, a denoising autoencoder for learning latent preferences, and a Deep Q-Network (DQN) for sequential decision-making. The main contribution is a hybrid multi-tower neural network that processes multiple feature groups independently before combining them for prediction, further improved using pre-trained embeddings.

All models are evaluated using MAE, MSE, and RMSE, with EDA supporting model design. Overall, the project focuses on building a scalable, personalized, and high-performance recommendation system for presentation skill improvement.

🎯Objective
This project builds and evaluates a complete recommender system pipeline for a Presentation Advisor application:

* Implement five recommendation models of increasing complexity
* Compare all models on MAE, MSE, and RMSE across a shared test set
* Analyze the effect of temporal encoding, problem sequence weighting, and utility-based scoring
* Validate that deep hybrid architectures outperform traditional and RL-based approaches on this domain

Repository Structure
Presentation-Advisor-Recommender-System/
├── Model_1/                        # Baseline Hybrid Approach
│   ├── model1.ipynb                # CBF & Collaborative Filtering implementation
│   ├── cbf_cf_metrics.json         # Performance metrics
│   ├── cbf_cf_predictions.csv      # Sample output predictions
│   └── cbf_cf_results.png          # Visualization of baseline performance
│
├── model_2_auto_encoders/          # Neural Network Recommenders
│   ├── autoencodersmodel2.ipynb    # Deep learning autoencoder implementation
│   ├── best_autoencoder.keras      # Saved model weights (HDF5 format)
│   ├── autoencoder_metrics.json    # Accuracy and loss logs
│   └── autoencoder_results.png     # Training history plots
│
├── model_3/                        # Reinforcement Learning Agent
│   ├── model-3-dqn.ipynb           # Deep Q-Network implementation
│   ├── dqn_model.keras             # Saved RL agent
│   ├── dqn_metrics.json            # Reward and Q-value tracking
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
│   ├── user_encoder.keras          # Serialized user embedding layer
│   ├── item_encoder.keras          # Serialized item embedding layer
│   └── hybrid_emb_results.png      # Embedding space visualizations
│
├── comparison_results/             # Cross-Model Evaluation
│   ├── comparison.ipynb            # Benchmarking all 5 models
│   └── all_models_comparison.png   # Final performance leaderboard
│
└── presentation_data/              # Core Datasets
    ├── presentations.csv           # Content metadata
    ├── interaction_data.csv        # User-item interactions
    ├── ratings_matrix.csv          # Pivot table for CF
    └── user_profiles.csv           # User preference data
