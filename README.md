# Presentation-Advisor-Recommender-System
📌 Project Description

The Presentation Advisor Recommender System is a multi-model framework that enhances users’ presentation skills through personalized article recommendations. It implements and compares five approaches, ranging from traditional filtering methods to advanced deep learning models.

The system uses user interactions, presentation metadata, and article features to generate tailored recommendations based on user type, audience, presentation style, past issues, and temporal behavior. It also handles sparse data and cold-start problems using feature-based modeling and time-aware weighting, prioritizing recent activity.

The pipeline includes a hybrid content-based and collaborative filtering baseline, a denoising autoencoder for learning latent preferences, and a Deep Q-Network (DQN) for sequential decision-making. The main contribution is a hybrid multi-tower neural network that processes multiple feature groups independently before combining them for prediction, further improved using pre-trained embeddings.

All models are evaluated using MAE, MSE, and RMSE, with EDA supporting model design. Overall, the project focuses on building a scalable, personalized, and high-performance recommendation system for presentation skill improvement.

🎯 Objective
Implement five recommendation models with increasing complexity
Compare model performance using MAE, MSE, and RMSE on a common test set
Analyze the impact of temporal encoding, problem-sequence weighting, and utility-based scoring
Demonstrate that hybrid deep learning architectures outperform traditional and reinforcement learning approaches in this domain
