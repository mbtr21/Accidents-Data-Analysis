from sklearn.model_selection import train_test_split  # Import train_test_split to divide data into training and testing sets
from sklearn.metrics import accuracy_score  # Import accuracy_score to evaluate model accuracy
from sklearn.metrics import f1_score  # Import f1_score to evaluate model's F1 score
from sklearn import svm  # Import svm for Support Vector Machine classifier

class SvmClassifier:
    def __init__(self):
        self.model = None  # Initialize model attribute to None
        self.train_x = None  # Placeholder for training data features
        self.train_y = None  # Placeholder for training data labels
        self.test_x = None  # Placeholder for testing data features
        self.test_y = None  # Placeholder for testing data labels

    def set_train_test(self, X, y, test_size, random_state):
        # Split the dataset into training and testing sets
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train_model(self, kernel, C=0.4):
        # Train the SVM model with the specified kernel
        self.model = svm.SVC(kernel=kernel, gamma="auto", C=C)  # Create SVM model with specific parameters
        self.model.fit(self.train_x, self.train_y)  # Fit model to the training data

    def result_model(self):
        # Predict on the test set and print the accuracy and F1 score
        y_pred = self.model.predict(self.test_x)  # Predict labels for test data
        poly_accuracy = accuracy_score(self.test_y, y_pred)  # Calculate accuracy
        poly_f1 = f1_score(self.test_y, y_pred, average='weighted')  # Calculate F1 score
        print('Accuracy : ', "%.2f" % (poly_accuracy * 100))  # Print accuracy as a percentage
        print('F1 : ', "%.2f" % (poly_f1 * 100))  # Print F1 score as a percentage
        return poly_accuracy * 100, poly_f1 * 100