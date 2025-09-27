# Gradient Descent (GD):
It’s an optimization algorithm used to minimize a loss function by iteratively updating parameters in the opposite direction of the gradient. Core idea: move step-by-step towards the global or local minimum.

**Class for simple 2D dataset**
```python
lass MySGD:
    def __init__(self,m = 78.35,b= 0 ):
        self.m = 78.35
        self.b = 0
        self.epochs = None
        self.lr = None

    def fit(self,X,y,epochs = 10, lr =0.1):
        self.epochs = epochs
        self.lr =lr
        b_arr = []
        y_pred_arr = []
                
        for i in range(epochs):
            loss_slope =-2*np.sum(y -self.m*X.ravel() - self.b)
            step_size = loss_slope * lr
            self.b = self.b - step_size
            
            b_arr.append(self.b) 
            y_pred = ((self.m*X) +self.b).ravel()
            y_pred_arr.append(y_pred)
        print("Final SGD b:", self.b)
        print("Final SGD m:", self.m)
```

## 1. Batch Gradient Descent

**Definition**: Uses the entire training dataset to compute gradients for each update.

**Update frequency**: One update per epoch → stable convergence.

**Pros**: Converges smoothly; guaranteed to approach optimal solution for convex problems.

**Cons**: Computationally expensive for large datasets; slow because it processes all samples before one update.
**Class**:
```python
class BatchGD:
    def __init__(self):
        self.B0 = None
        self.B = None

    def fit(self, X_train, y_train,epochs = 100, lr = 0.01):
        self.epochs = epochs
        self.lr = lr
        
        # adding 1 in first row
        n = X_train.shape[0]       

        # making a matrix of 1  for Beta (B)
        # B = [B1, B2, ....Bn ]
        self.B0 = 0
        self.B = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            # For the B0 (intercept) value
            y_cap = np.dot(X_train, self.B) + self.B0
            der_B0 = -2 * np.mean(y_train - y_cap)                           
            self.B0 = self.B0 - self.lr * der_B0

            # For the coef_ values (B)
            der_B = -2 * np.dot((y_train - y_cap) ,X_train)/X_train.shape[0]
            self.B = self.B - self.lr * der_B
        print(self.B0, self.B)

    def pred(self,X_test):
        return np.dot(X_test, self.B) + self.B0
```


## 2. Stochastic Gradient Descent (SGD)

**Definition**: Updates parameters using a single randomly chosen sample at a time.

**Update frequency**: One update per training sample → very frequent updates.

**Pros**: Fast; introduces randomness that helps escape local minima; good for online learning.

**Cons**: Highly noisy path to convergence; requires careful learning rate tuning; may oscillate around minima.
**Class**:
```python
class StochasticGD:
    def __init__(self):
        self.B0 = None
        self.B = None

    def fit(self, X_train, y_train,epochs = 100, lr = 0.01):
        self.epochs = epochs
        self.lr = lr
        
        # making a matrix of 1  for Beta (B)
        # B = [B1, B2, ....Bn ]
        self.B0 = 0
        self.B = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                
                # Picking random value for picking random row
                idx = np.random.randint(0,X_train.shape[0])
                
                # For the B0 (intercept) value
                y_cap = np.dot(X_train[idx], self.B) + self.B0
                der_B0 = -2 * (y_train[idx] - y_cap)
                self.B0 = self.B0 - self.lr * der_B0

                # For the coef_ values (B)
                der_B = -2 * np.dot((y_train[idx] - y_cap) ,X_train[idx])
                self.B = self.B - self.lr * der_B
            print(self.B0,self.B)
        print("Final B0 :", self.B0)
        print("Final B :" ,self.B)
    def pred(self,X_test):
        return np.dot(X_test, self.B) + self.B0

```

## 3. Mini-Batch Gradient Descent

**Definition**: A compromise — splits dataset into small batches (e.g., 32–256 samples) and updates per batch.

**Update frequency**: More frequent than batch GD, less noisy than pure SGD.

**Pros**: Efficient with vectorization on GPUs; balances convergence stability and speed; most commonly used in deep learning.

**Cons**: Requires batch size tuning; still some variance in updates; too small/large batch can hurt performance.
**Class**:
```python
class MiniBatchGD:
    def __init__(self):
        self.B0 = None
        self.B = None

    def fit(self, X_train, y_train,batch_size = 10, epochs = 100, lr = 0.1):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        # making a matrix of 1  for Beta (B)
        # B = [B1, B2, ....Bn ]
        self.B0 = 0
        self.B = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(int(X_train.shape[0]/self.batch_size)):
                
                # Picking random sample of values from rows
                idx = np.random.choice(range(X_train.shape[0]),self.batch_size)
                
                # For the B0 (intercept) value
                y_cap = np.dot(X_train[idx], self.B) + self.B0
                der_B0 = -2 * np.mean(y_train[idx] - y_cap)
                self.B0 = self.B0 - self.lr * der_B0

                # For the coef_ values (B)
                der_B = -2 * np.dot((y_train[idx] - y_cap) ,X_train[idx])
                self.B = self.B - self.lr * der_B
            print(self.B0,self.B)
        print("Final B0 :", self.B0)
        print("Final B :" ,self.B)
    def pred(self,X_test):
        return np.dot(X_test, self.B) + self.B0
```
**Class 2 with some improvments**
```python
class ImprovedMiniBatchGD1:
    def __init__(self):
        self.B0 = None
        self.B = None

    def fit(self, X_train, y_train,batch_size = 10, epochs = 40, lr = 0.05):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        # making a matrix of 1  for Beta (B)
        # B = [B1, B2, ....Bn ]
        self.B0 = 0
        self.B = np.ones(X_train.shape[1])

        # Learning Schedule : varying learning rate
        '''t0,t1 = 5,50
        def learning_rate(t):
            return t0/(t1+t)
        '''
        

        for i in range(self.epochs):
            for j in range(int(X_train.shape[0]/self.batch_size)):
               # lr = learning_rate(i * X_train.shape[0] + j)
                # Picking random sample of values from rows
                idx = np.random.choice(range(X_train.shape[0]),self.batch_size)
                
                # For the B0 (intercept) value
                y_cap = np.dot(X_train[idx], self.B) + self.B0
                der_B0 = -2 * np.mean(y_train[idx] - y_cap)
                self.B0 = self.B0 - self.lr * der_B0

                # For the coef_ values (B)
                der_B = -2 * np.dot((y_train[idx] - y_cap) ,X_train[idx])/batch_size # taking mean
                self.B = self.B - self.lr * der_B

                # Track loss
                loss = np.mean((y_train - (np.dot(X_train, self.B) + self.B0))**2)
            print("loss:", loss)
            print(" value B0:" ,self.B0,"value B:",self.B)
        print("Final B0 :", self.B0)
        print("Final B :" ,self.B)
    def pred(self,X_test):
        return np.dot(X_test, self.B) + self.B0
```

**Note :** Here I have tried to add Learning Schedule, but after this my r2 score decrease .
I tried lots of combinations but still , I am not able to implemented it properly

### Now Lets discuss , what I have covered in these files,
1. First I learnt about the concept
2.  then it's mathematical funtioning
3.  Using with the help scikit
4.  creating my own class for each of gradient by gradully improving in the code
5.  Finding and improving my code to the optimat solution ( I have used ChatGPt in my final phase to improve my code )
