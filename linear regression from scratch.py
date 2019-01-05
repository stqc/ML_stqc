from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


x=np.array([1,2,3,4,5,6],dtype=np.float64)
y=np.array([5,4,6,5,6,7],dtype=np.float64)


def best_fit_slope_and_intercept(a,b):
    m =((mean(a)* mean(b))-mean(a*b))/(mean(a)**2-mean(a**2))
    c=mean(b)-m*mean(a)
    return m,c

def squared_error(y_orig,y_line):
    return sum((y_line-y_orig)**2)

def coeff_deter(y_orig,y_line):
    y_mean_line=[mean(y_orig) for s in y_orig]
    squared_error_regr=squared_error(y_orig,y_line)
    squared_error_ymean=squared_error(y_orig,y_mean_line)
    return 1-(squared_error_regr/squared_error_ymean)


m,b=best_fit_slope_and_intercept(x,y)

print(m,b)

regression_line =[(m*s)+b for s in x ]
r_squared=coeff_deter(y,regression_line)


print(regression_line)
print(r_squared)
plt.scatter(x,y)
plt.plot(x,regression_line)
plt.show()
    
