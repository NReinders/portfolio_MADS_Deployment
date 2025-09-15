1. Assignment
The goal of this assignment was to build and train a Neural Network (NN). task is a classification problem, where the model learns to map input images (28×28 pixels) to one of 10 classes.
The focus was on experimenting with the architecture and hyperparameters to see how these influence model performance.

2. How the Neural Network Works
Input: Each Fashion image is 28×28 pixels. The image is flattened into a vector of 784 values (each pixel value between 0 and 255, often normalized to [0,1]).
Hidden layers (units and layers):
A layer is a set of neurons.
Each neuron computes a weighted sum of its inputs, applies an activation function (like ReLU), and passes the result forward.
The number of units in a layer determines its capacity: more units → more patterns the network can represent.
Adding more layers lets the network capture increasingly complex patterns.
Output layer: Has 10 units (one per clothing class). The activation function is usually softmax, giving probabilities across the 10 categories.
In short: pixels → flattened vector → hidden layers (feature extraction) → output (clothing prediction).

3. Key Hyperparameters
These are the parameters we tuned and observed:

Learning rate (lr)
Controls the step size of weight updates.
Too high: unstable training. Too low: very slow convergence.

Epochs
One pass through the entire dataset.
More epochs → more training, but also higher risk of overfitting.

Batch size
Number of samples processed before updating weights.
Small batch → noisier updates, sometimes better generalization.
Large batch → faster and more stable training, but can overfit.

Units per layer
Determines the capacity of each layer.
More units can model more complex patterns but increase risk of overfitting.

Number of layers
Depth of the network.
Shallow (1–2 layers) → simpler models.
Deeper (3–4 layers) → more expressive but harder to train.

4. How These Parameters Interact
Increasing units/layers usually increases accuracy, but at the cost of longer training and possible overfitting.
Learning rate is critical: it balances speed of learning with stability.
Batch size affects both speed and generalization.
Epochs let the model refine its understanding, but only until it starts to overfit.

5. Experiments & Findings

Setup
I systematically changed one hyperparameter at a time compared to a baseline:
Baseline (H0): 2 hidden layers (units ∈ {64,128,256}), epochs=3, batch=64, Adam lr=1e-3.
Best result: val-loss 0.3566 @ epoch 2, val-acc 0.8726 → still undertrained (curves trending down).
I then varied epochs, units, batch size, extra layers, and learning rate, and finally tested a “best expected setup.”
Results per Hypothesis

H1 – More epochs (3→10)
Expectation: higher validation accuracy, lower loss, no major overfitting within 10 epochs.
Result: best val-loss 0.3289, val-acc 0.8861 (around epoch 8).
Conclusion: confirmed — more epochs helped; slight overfitting towards the end.

H2 – More units (512/256/128, 3 epochs)
Expectation: lower val-loss / higher acc than baseline, but slower.
Result: best val-loss 0.3445, best val-acc 0.8765.
Conclusion: confirmed — small gain, noticeably slower runtime.

H3 – Larger batch (64→128, 3 epochs)
Expectation: smoother curves, possibly faster convergence.
Result: best val-loss 0.3664, val-acc ≈ 0.8726.
Conclusion: refuted — batch size 128 did not improve validation metrics, only smoothed the curves.

H4 – Extra hidden layer (3 layers, 3 epochs)
Expectation: slightly better performance, risk of fluctuations.
Result: val-acc between 0.854–0.873, e.g., train ≈ 0.3345, val-loss ≈ 0.3650, val-acc ≈ 0.8733.
Conclusion: marginal improvement in some runs, but no structural advantage over baseline.

H5 – Higher learning rate (1e-2, 3 epochs)
Expectation: faster updates, more fluctuation, worse generalization.
Result: best val-loss ≈ 0.4093, val-acc ≈ 0.8546.
Conclusion: confirmed — unstable, worse generalization compared to baseline.

H6 – Lower learning rate (1e-5, 3 epochs)
Expectation: slower but more stable convergence, possibly better validation metrics.
Result: val-loss ≈ 0.70–1.97, val-acc ≈ 0.55–0.74 (e.g., 0.7078 / 0.7429).
Conclusion: refuted — underfitting; too few epochs for such a small learning rate.

H7 – Slower learning + more capacity (3 layers, 10 epochs, lr=1e-4, units∈{512,256,128,64})
Expectation: better convergence, higher validation accuracy, but slower.
Result: run aborted (too slow).
Conclusion: not practical in this setup; runtime/complexity is a key constraint.

H8/H9 – Best expected setup
Setup: 2 hidden layers, units1=256, units2=128, epochs=10, batch=64, Adam lr=1e-3, weight_decay=1e-5, ReduceLROnPlateau.
Result (H9): train loss ≈ 0.24, best val-loss 0.3256, max val-acc 0.8907 (best around epoch 8; slight overfitting afterwards).
Conclusion: sweet spot in terms of speed, stability, and accuracy on Fashion-MNIST without convolutional layers (~89% accuracy).

Key Takeaways
Epochs ↑ → clear improvement up to ~8–10 epochs, then risk of overfitting.
Units/layers ↑ → slight gain, but requires more epochs or regularization to show real benefits.
Batch size ↑ → smoother updates, but not better validation performance at fixed lr/epochs.

Learning rate was decisive:
Too high (1e-2) → overshooting, instability, worse validation.
Too low (1e-5) → underfitting with only 3 epochs.
1e-3 was the sweet spot.

Best balance: 2 layers (256→128), lr=1e-3, batch=64, ~10 epochs, light weight decay, scheduler, early stopping around epoch 8.

Why does validation sometimes get worse after later epochs?
Due to overfitting (train loss keeps dropping while val-loss increases) and validation noise (minor fluctuations per epoch). A scheduler or early stopping prevents this.


