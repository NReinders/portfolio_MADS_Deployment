# Project Text - Kadaster


# Project Goal and Architectural Strategy

The challenge during this hackathon was centered on optimizing the classification of Dutch Cadastre (Kadaster) deed descriptions. Our team opted for an ambitious, Dual-Path architectural strategy involving three interdependent pipelines: a RegEx pipeline, a Neural Network (NN) for bulk classification (my primary focus), and an LLM-Runner for the rare edge cases.

# Deep Dive: Neural Network Optimization

My role focused on developing and optimizing the core NN pipeline. The process was iterative, emphasizing stability and feature utilization:

1. Semantic Boost and Model Selection

The initial step was to drastically improve the quality of the text embeddings. We moved away from the default, low-context models:

Initial Benchmarking: Starting with bert-tiny to establish a baseline.

Performance Scaling: Transitioning to larger BERT variants immediately yielded significant gains.

Final Text Vectorizer: We settled on pdelobelle/robbert-v2-dutch-base (a RoBERTa model trained on Dutch data), which, despite having a 512-token context limit, provided superior semantic representation essential for notary texts.

We attempted to integrate the Nebius API to test modern 8k and 32k context models, but this complex integration proved too time-consuming for the one-day hackathon and had to be aborted. This was a missed opportunity to assess the value of ultra-long context embeddings.

2. Architectural Upgrade (Dual-Path ResNet)

The simplistic initial architecture (direct concatenation) was replaced with a sophisticated Dual-Path Hybrid Classifier :

Feature Separation: The architecture was split to process Text Embeddings (dense, continuous) and RegEx Features (sparse, binary) in parallel paths.

Residual Connections: The Text Path was enhanced with Residual Blocks (Skip Connections). This design stabilized the training of the deeper network and mitigated vanishing gradients, crucial for effective feature learning.

Deep Regularization: We added higher Dropout rates (up to 0.4) and strategic Batch Normalization (BN) layers throughout the architecture to aggressively combat the fast overfitting observed during initial tests.

RegEx Path Deepening: The RegEx path was expanded to multiple layers with BN, allowing the network to better weigh and interpret the complex, rule-based binary features before fusion.

3. Data Challenges and Filtering

A key learning was the severity of the data imbalance. The 'long tail' (labels with a count ≤10) led to rapid overfitting. While we developed the logic in the Trainer to hardcode the removal of this long tail, a technical obstacle in the CLI parameter passing prevented this crucial filter from being executed during the final runs. This failure to implement the necessary data pre-filtering was one of the biggest setbacks.

# Conclusion and Reflection

Despite the technical setbacks—the inability to finalize the RegEx and LLM integrations, and the issue with data filtering—our optimized NN pipeline achieved a strong F1 micro score of 85-86% on the bulk of the classification tasks.

The day provided an intense and invaluable learning experience. Being fully focused on a single task as a team highlighted the difficulty of managing architectural complexity, external API dependencies, and robust debugging under severe time constraints. While I was disappointed that the full vision was not realized, the process was a significant win, showcasing the technical depth required to transition an ML concept into a deployable solution. The entire experience reinforced the value of modular design and aggressive iteration in machine learning development.


# Repo:
 https://github.com/MaxNollet/mads-2025-hackathon.git


[Go back to Homepage](../README.md)
