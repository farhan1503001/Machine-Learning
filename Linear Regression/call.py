import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from linear_regression import LinearRegressor

if __name__ == '__main__':
    dataset=pd.read_csv('bike_sharing_data.txt')
    dataset.head()
    ax=sns.scatterplot(x='Population',y='Profit',data=dataset)
    ax.set_title("Profit vs Population in respectively $100000 and 10000 scale")
    model=LinearRegressor()