from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import svm


class SvmClassifier:
    def __init__(self):
        self.model = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def set_train_test(self, X, y, test_size):
        self.train_x,self.test_x, self.train_y,self.test_y = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    def train_model(self):
        self.model = svm.SVC(kernel="rbf", gamma="auto", C=0.1)
        self.model.fit(self.train_x, self.train_y)

    def result_model(self):
        y_pred = self.model.predict(self.test_x)
        poly_accuracy = accuracy_score(self.test_y, y_pred)
        poly_f1 = f1_score(self.test_y, y_pred, average='weighted')
        print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy * 100))
        print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1 * 100))


