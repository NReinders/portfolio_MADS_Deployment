Hypertuning Report – Flowers Dataset

1. Introduction

The goal of this experiment was to apply the scientific method to the process of hyperparameter tuning using a convolutional neural network (CNN) on the Flowers dataset. The exercise focused on systematically improving model performance through iterative experimentation, guided by hypotheses derived from theory. The experiment also served to understand the trade-offs between model complexity, computational cost, and performance, and to experience the practical challenges of hypertuning in a limited hardware environment.

The dataset used was the Flowers dataset from torchvision, consisting of 5 categories: daisy, dandelion, rose, sunflower, and tulip. Images were resized to 224×224, and training was done locally on an Apple MPS GPU backend, which accelerated single runs effectively but imposed strict limitations on parallelism.

⸻

2. Hypothesis and Setup

The initial hypothesis was that a deeper CNN with more filters and proper regularization (Batch Normalization and Dropout) would lead to significantly higher accuracy than a simple baseline network. Additionally, it was expected that tuning learning rate and architectural depth could further refine performance.

The experiments were structured around an iterative process:
	1.	Start with a simple baseline CNN to establish reference performance and explore search space boundaries.
	2.	Gradually increase model depth and number of filters to identify patterns in accuracy and overfitting.
	3.	Apply BatchNorm and Dropout to stabilize and regularize the model.
	4.	Use Ray Tune to automate hypertuning and compare with manual search results.
	5.	Finally, test Transfer Learning as a benchmark for efficiency and performance improvement.

The model was implemented in a configurable way so that all parameters, such as the number of convolutional layers, filters, dense units, dropout rate, and learning rate could easily be adjusted through configuration objects.

⸻

3. Experimental Process

Manual Search and Search Space Exploration

The first phase focused on understanding the dataset and establishing a baseline.
A CNN with three convolutional layers, 64 filters, and 3 epochs achieved an accuracy of only 55%, which was lower than expected. Increasing the number of epochs to ten provided only a small improvement to around 61%, indicating underfitting.

Next, several manual experiments explored the effect of architectural parameters:
	•	Doubling the filters from 64 to 128 did not improve accuracy, it even decreased to 51%.
	•	Increasing the number of layers from three to five showed the first real improvement, with a model reaching 66% accuracy, suggesting that moderate depth (around five layers) was beneficial for this dataset.

These early tests helped define a realistic search space for later tuning:
layers between 3–6, filters between 32–140, dropout between 0.0–0.3, and learning rates between 0.0005–0.003.

⸻

Attempt with Ray Tune (Parallel Hypertuning)

After the initial exploration, the next goal was to automate the process using Ray Tune. The intention was to perform multiple trials in parallel and visualize interactions between layers and filters. However, this step became the most technically challenging part of the assignment.

Running Ray Tune on the MPS GPU was not supported for parallel trials, forcing all runs onto the CPU, which made image-based models extremely slow. Even with aggressive optimizations, reducing image size from 224×224 to 32×32, limiting trials to three, and reducing epochs, execution remained extremely heavy.

In practice, each trial consumed excessive memory (over 60 GB), and the laptop frequently froze or crashed. Several fixes were tested, such as limiting thread usage, controlling memory allocation, and dynamically adjusting image sizes. Despite two full days of troubleshooting, Ray Tune could not be made to run efficiently in this environment.

This experience clearly illustrated one of the core lessons of hypertuning: computational resources are often the main bottleneck, not the theoretical setup.

⸻

Manual Grid and Random Search

Given the limitations of Ray Tune, the process was continued with manual Grid Search and Random Search, running sequentially on the MPS GPU. This approach proved stable and reproducible.

The first grid search tested models between 3–7 layers and 32–128 filters. The best configuration achieved 62.9% accuracy (L3, F128), showing that deeper networks did not necessarily help beyond a certain point.
The random search that followed, focusing on 3–6 layers and 55–150 filters, performed significantly better: L5 with 126 filters reached 67% accuracy in about eight minutes of training time.

This combination, a 5-layer network with around 126 filters, consistently emerged as the optimal balance between capacity, training time, and generalization. Further refinements with smaller search steps confirmed this optimum across multiple runs, indicating a local performance plateau around 67–68%.

⸻

Regularization and Learning Rate Tuning

Once the architectural optimum was identified, the next focus was on reducing overfitting through regularization. Batch Normalization (BN) and Dropout were systematically tested in all combinations.

BatchNorm alone slightly improved performance, but the combination of BatchNorm and Dropout yielded major gains. With a dropout rate of 0.25, the model reached 77.4% accuracy, a 14.5% improvement over the original baseline. Dropout alone, without BatchNorm, caused instability and significantly worse results, confirming the theoretical expectation that normalization should precede stochastic regularization.

Learning rate experiments showed that Adam’s default (0.001) remained optimal. Higher learning rates (0.002–0.003) led to overshooting, while lower ones (0.0005) slowed convergence. Attempts with cosine warmup and decay schedules did not outperform the simple ReduceLROnPlateau scheduler.

⸻

Transfer Learning

Finally, out of curiosity, transfer learning was tested using a pretrained ResNet18 model. Training this model took only 1.5 minutes per run and immediately achieved 91% accuracy without hypertuning.
After limited hypertuning — adjusting dropout (0.05–0.25) and unfreezing blocks 3 and 4 — accuracy increased to 95%, with the optimal dropout at 0.15.

This result was a strong contrast to the CNN built from scratch:
in just one hour of experimentation, transfer learning achieved results that had taken several days of manual tuning to even approach.

⸻

4. Discussion

This exercise vividly demonstrated both the power and the limitations of hypertuning. The manual search process was labor-intensive and required significant computational time, but it offered valuable intuition about how different parameters interact. The Ray Tune attempt, while theoretically elegant, revealed the practical constraints of local hardware when training image models.

Batch normalization and dropout proved to be the most impactful parameters for improving generalization, while architectural complexity had clear diminishing returns. Moreover, random search turned out to be more effective than wide grid search for discovering promising configurations.

The transfer learning experiment served as a key insight: in modern image classification tasks, pretrained architectures not only save time but often outperform custom-designed networks by large margins. This suggests that for future projects, starting from a pretrained backbone should be the default approach.

⸻

5. Conclusions
	•	The best custom CNN achieved 77.4% accuracy with the configuration:
5 layers, 126 filters, Dropout=0.25, BatchNorm=True, LR=0.001.
	•	The ResNet18 transfer learning model achieved 95% accuracy with a dropout of 0.15, training in under two minutes per run.
	•	Batch Normalization is essential for stable learning and effective dropout.
	•	Random Search > Grid Search for exploration.
	•	Ray Tune is powerful but impractical on CPU-only setups with image data.
	•	Transfer Learning provides the best balance between performance and efficiency.

⸻

6. Lessons Learned

This exercise emphasized that hypertuning is as much an engineering challenge as a statistical one. The process reinforced several lessons:
	•	Always start with exploratory manual runs to understand model behavior.
	•	Avoid overcomplicating searches, focus on well-chosen ranges and meaningful parameters.
	•	Computational feasibility should guide methodological choices.
	•	Pretrained models dramatically reduce both training time and search effort.

In the future, for image-related projects, beginning with a pretrained model and using targeted hypertuning around regularization and learning rate will likely yield the best return on effort.
