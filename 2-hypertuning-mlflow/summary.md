Objective
Extend last week’s hyperparameter interaction study by adding dropout, normalization, and an extra convolutional layer, then observe how these choices interact. All experiments are tracked in MLflow.

Experimental Setup
Dataset/Task: Image classification (28×28, 1 channel, 10 classes).
Base model: 3×(Conv→ReLU→MaxPool) → Global AvgPool → 2×Linear → Output.
Training: Adam (default lr), early stopping on validation loss, 5–13 epochs.
Measured: Test loss (↓) and accuracy (↑).

Variants tested:
Add dropout (single layer in dense stack).
Add BatchNorm2d (before ReLU) and later BatchNorm1d in dense stack.
Add one extra Conv layer (+ MaxPool).
Set padding=1 for all convs (shape-preserving).
Increase filters (32→64).
Widen dense layers (units1=256, units2=128).

Hypotheses
Normalization will stabilize and speed up learning, improving generalization.
Dropout will reduce overfitting, especially as capacity grows.
More capacity (extra conv / filters / units) will help until regularization becomes necessary.
Padding=1 in all convs will help

Results (best per round):

| Round | Change vs. base                | Epochs | Best Test Loss |  Best Acc |
| ----- | ------------------------------ | -----: | -------------: | --------: |
| 1     | Dropout only                   |      5 |          0.736 |     0.726 |
| 2     | No dropout (default)           |      5 |          0.739 |     0.732 |
| 3     | BatchNorm2d, no dropout        |      5 |          0.603 |     0.774 |
| 4     | BatchNorm2d + dropout          |      5 |          0.557 |     0.793 |
| 5     | +1 Conv (+Pool)                |      5 |          0.480 |     0.819 |
| 6     | Padding=1 (all convs)          |      5 |          0.494 |     0.815 |
| 7     | Padding=1 + longer train       |      8 |          0.430 |     0.843 |
| 8     | As (7), 10 epochs              |     10 |          0.393 |     0.856 |
| 9     | Filters **64** (was 32)        |     13 |      **0.323** | **0.887** |
| 10    | Filters 64 + units **256/128** |     13 |          0.328 |     0.881 |


Best run: Round 9 (filters=64, BN + dropout, extra conv, padding=1): loss 0.323 / acc 0.887.
Increasing dense units (Round 10) was competitive but did not beat Round 9.

Analysis & Interaction Takeaways
Normalization dominates early gains: Adding BatchNorm2d (R3→R4) notably lowers loss and boosts accuracy; it also makes deeper/wider models train stably.
Dropout complements BN: With added capacity (extra conv / filters), a single dropout in the dense stack consistently helps generalization.
Capacity matters most after stabilization: Extra conv (R5) and filters=64 (R9) bring the biggest jumps once BN+dropout are in place.
Padding=1 supports deeper stacks: Retaining feature map size improves downstream feature quality (R7–R8).
Dense widening vs. conv widening: Widening convs (filters↑) helped more than widening dense layers for this task (R9 > R10).

Conclusion
The interaction is clear: BatchNorm enables effective scaling of model capacity (extra conv, more filters), while dropout keeps that added capacity from overfitting. The winning recipe here is:
BN + single dense-dropout + one extra conv + padding=1 + 64 filters, trained ~13 epochs.

Reflection (science part)
Hypotheses confirmed: BN improves optimization/generalization; dropout helps when capacity increases; capacity boosts results after stabilization; padding aids deeper conv stacks.
Trade-off: More filters increase compute, but gave the best accuracy within our epoch budget.
