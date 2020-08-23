from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class RecalculateNullGrades():    
    def fit(self, X, y=None):
        return self
    # Retorna um novo dataframe com valor nulo igual a média das demais notas do aluno  
    def transform(self, X):
        data = X.copy()
        notas = data.loc[:,'NOTA_DE':'NOTA_GO']
        notas = notas.T.fillna(notas.mean(axis=1)).T
        data.loc[:,'NOTA_DE':'NOTA_GO'] = notas
        return data

class RemoveNull():
    def fit (self, X, y=None):
        return self

    def transform(self, X, y):
        bool_index = X.isnull().any(axis=1)
        return X.drop(X[bool_index].index), y.drop(y[bool_index].index)