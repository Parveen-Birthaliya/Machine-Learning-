## Simple Linear Regression (SLR)

**Definition**:Predicts a dependent variable y from a single independent variable x by fitting a straight line:
y = B0 + B1*x.

**Goal / Objective**:Minimize the difference between predicted and actual values, typically using Mean Squared Error (MSE):
MSE = mean((y - y_pred)^2).

**Parameters & Interpretation**:

B0 → intercept (value of y when x=0)

B1 → slope (change in y per unit change in x)

These represent the strength and direction of the linear relationship.

**Pros & Cons**:

Pros: Simple, interpretable, easy to compute analytically or with gradient descent.

Cons: Can only capture linear relationships; sensitive to outliers.

**Class**:
```python
class MyLr:
    def _init_(self):
        self.m = None
        self.b = None

    
    def fit(self,X_train,y_train):
        num = 0
        den = 0
        
        for i in range(X_train.shape[0]):
            num = num + ((X_train[i]-X_train.mean())*(y_train[i]-y_train.mean()))
            den = den + ((X_train[i]-X_train.mean())*(X_train[i]-X_train.mean()))

        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)

    def predict(self,X_test):
        print(X_test)
        return self.m * X_test + self.b
```

## Multiple Linear Regression (MLR)

**Definition**:
Extends SLR to multiple independent variables x1, x2, ..., xn:
y = B0 + B1*x1 + B2*x2 + ... + Bn*xn.

**Goal / Objective**:
Same as SLR → minimize the prediction error (MSE) across all features. Gradient descent or closed-form (Normal Equation) can be used.

**Parameters & Interpretation**:

B0 → intercept

Bi → coefficient for each feature xi, representing the expected change in y per unit change in xi when other features are constant.

**Pros & Cons:**

Pros: Can model multiple factors simultaneously; captures more complex linear relationships.

**Class**:
```python
lass MyMLS:
    def _init_(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X_train, y_train):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
            
        X_train = np.insert(X_train,0,1,axis = 1)
        beta = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        

    def pred(self,X_test):
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
            
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred
```



Cons: Assumes linearity; multicollinearity between features can destabilize coefficients; still sensitive to outliers.
