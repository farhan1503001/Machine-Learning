import numpy as np
import numpy as np
import pandas as pd  
import seaborn as sns
class LinearRegressor():
    def __init__(self) -> None:
        self.learning_rate=0.001
        self.theta=None
        self.epochs=10000
        self.costs=[]
        self.X=None
        self.y=None
    def cost_function(self,X,y):
        m=len(y)
        y_pred=np.dot(X,self.theta)
        error=(y_pred-y)**2
        return 1/(2*m)*np.sum(error)
    def __gradient_descent(self,X,y,theta,alpha,epochs):
        m=len(y)
        cost=[]
        for i in range(epochs):
            y_pred=np.dot(X,theta)
            error=np.dot(X.traspose(),(y_pred-y))
            theta-=alpha*(1/m)*error
            cost.append(self.cost_function(X,y))
        return cost,theta
    
    def fit_data(self,X,y,epochs,learning_rate):
        m=len(y)
        feature_len=X.shape[1]
        self.X=np.append(np.ones((m,1)),X.reshape(m,feature_len),axis=1)
        self.y=y.reshape((m,1))
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.theta=np.zeros((feature_len+1,1))
        
        self.cost,self.theta=self.__gradient_descent(self.X,self.y,self.theta,self.learning_rate)
    def get_cost(self):
        return self.cost
    def get_theta(self):
        return self.theta
    


