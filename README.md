# Presentation-Advisor-Recommender-System
📌 Project Description

The Presentation Advisor Recommender System is a multi-model recommendation framework designed to enhance users’ presentation skills through personalized article suggestions. Instead of relying on a single approach, this project systematically implements and compares five distinct recommendation techniques, ranging from traditional filtering methods to advanced deep learning architectures.

The system leverages a rich dataset consisting of user interactions, presentation metadata, and article features. It analyzes key factors such as user type, audience type, presentation style, historical issues, and temporal behavior to generate highly tailored recommendations. To address real-world challenges like sparse data and cold-start scenarios, the system incorporates feature-based modeling and time-aware weighting, where recent user behavior is prioritized using exponential decay.

The modeling pipeline begins with a hybrid baseline combining content-based filtering and collaborative filtering. It then progresses to a denoising autoencoder that captures latent user preferences from sparse interaction data. A Deep Q-Network (DQN) is also implemented to model recommendations as a sequential decision-making process. The core contribution of this project is a hybrid multi-tower neural network, where separate towers independently process user, item, temporal, and problem-specific features before being combined for final prediction. This architecture is further enhanced using pre-trained embeddings to improve representation learning and overall performance.

All models are rigorously evaluated using standard metrics such as MAE, MSE, and RMSE on a shared test set, enabling a fair and direct comparison. Additionally, detailed exploratory data analysis (EDA) is performed to uncover behavioral patterns and inform model design decisions.

Overall, this project presents a structured, research-driven approach to recommender systems, with a strong focus on personalization, adaptability, and performance optimization in the context of presentation skill development.

🎯 Objective
Implement five recommendation models with increasing complexity
Compare model performance using MAE, MSE, and RMSE on a common test set
Analyze the impact of temporal encoding, problem-sequence weighting, and utility-based scoring
Demonstrate that hybrid deep learning architectures outperform traditional and reinforcement learning approaches in this domain
