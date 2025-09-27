# Gradient Descent (GD):
It’s an optimization algorithm used to minimize a loss function by iteratively updating parameters in the opposite direction of the gradient. Core idea: move step-by-step towards the global or local minimum.

## 1. Batch Gradient Descent

Definition: Uses the entire training dataset to compute gradients for each update.

Update frequency: One update per epoch → stable convergence.

Pros: Converges smoothly; guaranteed to approach optimal solution for convex problems.

Cons: Computationally expensive for large datasets; slow because it processes all samples before one update.

## 2. Stochastic Gradient Descent (SGD)

Definition: Updates parameters using a single randomly chosen sample at a time.

Update frequency: One update per training sample → very frequent updates.

Pros: Fast; introduces randomness that helps escape local minima; good for online learning.

Cons: Highly noisy path to convergence; requires careful learning rate tuning; may oscillate around minima.

## 3. Mini-Batch Gradient Descent

Definition: A compromise — splits dataset into small batches (e.g., 32–256 samples) and updates per batch.

Update frequency: More frequent than batch GD, less noisy than pure SGD.

Pros: Efficient with vectorization on GPUs; balances convergence stability and speed; most commonly used in deep learning.

Cons: Requires batch size tuning; still some variance in updates; too small/large batch can hurt performance.

### Now Lets discuss , what I have covered in these files,
1. First I learnt about the concept
2.  then it's mathematical funtioning
3.  Using with the help scikit
4.  creating my own class for each of gradient by gradully improving in the code
5.  Finding and improving my code to the optimat solution ( I have used ChatGPt in my final phase to improve my code )
