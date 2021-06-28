import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rcParams['figure.figsize'] = (14, 7)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False


class SimpleLinearRegression:
    '''
    A class which implements simple linear regression model.
    '''

    def __init__(self):
        self.b0 = None
        self.b1 = None

    def fit(self, X, y):
        '''
        Used to calculate slope and intercept coefficients.

        :param X: array, single feature
        :param y: array, true values
        :return: None
        '''
        numerator = np.sum((X - np.mean(X)) * (y - np.mean(y)))
        denominator = np.sum((X - np.mean(X)) ** 2)
        self.b1 = numerator / denominator
        self.b0 = np.mean(y) - self.b1 * np.mean(X)

    def predict(self, X):
        '''
        Makes predictions using the simple line equation.

        :param X: array, single feature
        :return: None
        '''
        if not self.b0 or not self.b1:
            raise Exception('Please call `SimpleLinearRegression.fit(X, y)` before making predictions.')
        return self.b0 + self.b1 * X

    def __str__(self):
        '''
        Representation string of the class
        '''
        return f'({self.b0}, {self.b1})'


def rmse(y, y_pred):
    '''
    Function to return the Root Mean Squared Error
    '''
    return np.sqrt(mean_squared_error(y, y_pred))


if __name__ == '__main__':
    # Creating a training dataset
    X = np.arange(start=1, stop=301)
    y = np.random.normal(loc=X, scale=20)

    plt.scatter(X, y, s=200, c='#087E8B', alpha=0.65, label='Source data')
    plt.title('Best fit line', size=20)
    plt.xlabel('X', size=14)
    plt.ylabel('Y', size=14)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Creating an instance of the class
    model = SimpleLinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    plt.plot(X_test, preds, color='#000000', lw=3, label=f'Best fit line > B0 = {model.b0:.2f}, B1 = {model.b1:.2f}')
    plt.legend()
    plt.show()

    print(model)
    print(rmse(y_test, preds))
