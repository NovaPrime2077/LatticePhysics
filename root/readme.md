# Lattice Physics 
The model encompasses 2 lattice physics parameters
1. **k -inf** : The infinite multiplication factor  
2. **PPPF** : Pin Power Peaking Factor


Please refer for further information on the reactor - Huu Tiep, N. (2024). Lattice-physics (PWR fuel assembly neutronics simulation results) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BK64.
 



## Approach
 
1. **Assigning Features and Target Variables**
2. **Z Score Normalisation**
3. **Linear Regression for k-inf**
4. **Polynomial Regression for PPPF**


### Assigning Features and Target Variables 
The features are arranged in 41 columns out of which 2 are target variables and 39 input features. To seperate features and target variables we evaluate mean, mode, median for every individual column. The value of **k-inf** lies in range **(0.9,1.5)**  and **PPPF** lies between **(1.5,2.2**) for a *stable reactor*. If the mean, mode , median of a column lies exactly between these ranges those columns correspond to the targeted variables.

### Z Score Normalisation

The unnormalised data varies between a greater range which might hamper effective learning of the model since convergence is reached in a much more stable fashion when data is normalised. 
The Normalisation technique used for this model is known as Z-Score Normalisation.

The normalisation technique works using the following: $$X_{\text norm} = \frac{X - \mu}{\sigma}$$

where $$\mu = \frac{\sum_{i=1}^n x_i}{n}, \quad \sigma = \sqrt{\frac{\sum_{i=1}^n (x_i - \mu)^2}{n}}$$

### Linear Regression for k-inf

The Linear regression works on the principle of reducing the squared error cost function. This is achieved by gradient descent which reduces the cost at every iteration.

The cost function is given below:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w}(x^{(i)}) - y^{(i)} \right)^2$$
The gradient descent is given below:
$$w_j := w_j - \frac{\alpha}{m} \sum_{i=1}^{m} \left( f_{w}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
$$b:= b - \frac{\alpha}{m} \sum_{i=1}^{m} \left( f_{w}(x^{(i)}) - y^{(i)} \right)$$

here *m* = total number of sample, *alpha* = the learning rate. 


### Polynomial Regression for PPPF
The variance of data for PPPF with the fetaures isn't linear that's why Polynomial regression comes into play. This is fairly observable in the R^2 scores. The k-inf locks itself at 0.98 which is extremely close to 1 while PPPF is a negative real number implying failure of the linear model. The polynomial regression is accompanied by regularization to ensure that overfitting is avoided. 


### Possible Improvements
The model can be imporved if we introduce neural network for the model. It is fairly observable that model can be improved especially for PPPF factor.

### Author
NovaPrime2077



