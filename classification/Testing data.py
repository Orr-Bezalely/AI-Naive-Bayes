import numpy as np
from keras.datasets import mnist



# This skeleton code simply classifies every input as ham
#
# Here you can see there is a parameter k that is unused, the
# point is to show you how you could set up your own. You can
# also see a train method that does nothing here
# but your classifier would probably do the main work here.
# Modify this code as much as you like so long as the
# accuracy test in the cell below runs

class MyClassifier:
    def __init__(self, num_of_classes, alpha=1.0):
        self.num_of_classes = num_of_classes
        self.alpha = alpha

    #     def train(self, train_data, train_labels):
    #         pass

    def estimate_log_class_priors(self, data):
        """
        Given a data set with binary response variable (0s and 1s),
        calculate the logarithm of the empirical class priors,
        that is, the logarithm of the proportions of 0s and 1s:
            log(p(C=0)) and log(p(C=1))

        :param data: a numpy array of length n_samples
                     that contains the binary response (coded as 0s and 1s).

        :return log_class_priors: a numpy array of length two
        """
        ### YOUR CODE HERE...
        log_class_priors = np.zeros(self.num_of_classes)
        for class_i in range(self.num_of_classes):
            log_class_priors[class_i] = np.count_nonzero(data == class_i)
        log_class_priors = np.log(log_class_priors / data.shape[0])
        return log_class_priors

    def estimate_log_class_conditional_likelihoods(self, input_data, labels):
        """
        Given input_data of binary features (words) and labels
        (binary response variable (0s and 1s)), calculate the logarithm
        of the empirical class-conditional likelihoods, that is,
        log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

        Assume a multinomial feature distribution and use Laplace smoothing
        if alpha > 0.

        :param input_data: a two-dimensional numpy-array with shape = [n_samples, n_features]
                           contains binary features (words)
        :param labels: a numpy array of length n_samples
                       contains response variable

        :return theta:
            a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging
            to class j.
        """
        ### YOUR CODE HERE...
        n_datas, n_features = input_data.shape
        theta = np.zeros((self.num_of_classes, n_features))
        for class_i in range(self.num_of_classes):
            feature_c = np.sum(input_data[labels == class_i], axis=0)
            n_c = np.sum(feature_c)
            theta[class_i] = (feature_c + self.alpha) / (n_c + n_features * self.alpha)
        return np.log(theta)

    def train(self, train_data, train_labels):
        """
        Given input_data of binary features (words) and labels
        (binary response variable (0s and 1s)), calculate
        * the logarithm of the empirical class priors, that is,
          the logarithm of the proportions of 0s and 1s:
            log(p(C=0)) and log(p(C=1))
        * the logarithm of the empirical class-conditional likelihoods,
          that is, log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

        Assume a multinomial feature distribution and use Laplace smoothing
        if alpha > 0.

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

        :return
            log_class_priors: a numpy array of length two

            theta:
            a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging
            to class j.
        """
        ### YOUR CODE HERE...
        if train_data is not None and train_labels is not None:
            self.log_class_priors = self.estimate_log_class_priors(train_labels)
            self.theta = self.estimate_log_class_conditional_likelihoods(train_data, train_labels)
            return self.log_class_priors, self.theta

    def predict(self, new_data):
        """
        Given a new data set with binary features, predict the corresponding
        response for each instance (row) of the new_data set.

        :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
        :param log_class_priors: a numpy array of length 2.
        :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
            theta[j, i] corresponds to the logarithm of the probability of feature i appearing
            in a sample belonging to class j.
        :return class_predictions: a numpy array containing the class predictions for each row
            of new_data.
        """
        ### YOUR CODE HERE...
        all_args = np.zeros((self.num_of_classes, new_data.shape[0]))
        for class_i in range(self.num_of_classes):
            all_args[class_i] = self.log_class_priors[class_i] + np.sum(self.theta[class_i] * new_data, axis=1)
        return np.argmax(all_args, axis=0)



def BasicFeatureExtractor(input_data):
    """
    Returns basic extracted features for input_data.
    A binary feature for each pixel: 0 if a pixel is black, 1 otherwise

    :param input_data: a 3-dimensional numpy array of the shape (n, 28, 28)
                       input data of n images of 28x28 pixels

    :return extracted_features: a 2-dimensional numpy array of the shape (n, 784)
                                extracted binary features
    """
    # compute binary features for each pixel
    extracted_features = (input_data > 0).astype(int)

    # flatten images of 28x28 pixels into a vector of 784 length
    extracted_features = np.reshape(extracted_features, (extracted_features.shape[0], -1))

    return extracted_features


NUMBER_OF_CLASSES = 10


def features(input_data, a, b, bias):
    n, HEIGHT, WIDTH = input_data.shape
    mean_per_block = np.zeros((n, (HEIGHT - a) * (WIDTH - b)))
    for i in range(HEIGHT - a):
        for j in range(WIDTH - b):
            x = np.mean(input_data[:, i:i + a, j:j + b], axis=(1, 2))
            mean_per_block[:, i * (WIDTH - b) + j] = x
    return (mean_per_block > bias).astype(int)


def BetterFeatureExtractor(input_data, height=3, width=14, bias=93):
    """
    A function for your improved feature extractor

    :param input_data: a 3-dimensional numpy array of the shape (n, 28, 28)
                       input data of n images of 28x28 pixels

    :return extracted_features: a 2-dimensional numpy array of the shape (n, m)
                                extracted binary m features for n images
    """
    extracted_features = BasicFeatureExtractor(input_data)
    extracted_features = np.hstack((extracted_features, features(input_data, height, width, bias)))
    n = input_data.shape[0]
    # extracted_features = np.ones((n, 0))
    # extracted_features = np.hstack((extracted_features, features(input_data, 1,1), features(input_data, 1,1), features(input_data, 1,15)))
    # extracted_features = np.reshape(input_data,(300,784))
    return extracted_features


# def k_fold(data, batch_size, alpha):
#     accuracies = []
#     for i in range(0, data.shape[0], batch_size):
#         training_batches = np.vstack((data[:i], data[i + batch_size:]))
#         validation_batch = data[i:i + batch_size]
#         spam_classifier = MyClassifier(10, alpha)
#         log_class_priors, theta = spam_classifier.train(training_batches[:, 1:], training_batches[:, 0])
#         class_predictions = spam_classifier.predict(validation_batch[:, 1:])
#         true_classes = validation_batch[:, 0]
#         training_set_accuracy = np.mean(np.equal(class_predictions, true_classes))
#         accuracies.append(training_set_accuracy)
#     return np.mean(np.array(accuracies))



x1 = np.load("data/training_digit_input.npy")
x2 = np.load("data/training_digit_label.npy")
y1 = np.load("data/test_digit_input.npy")
y2 = np.load("data/test_digit_label.npy")
training_val = 600
all_data = np.concatenate((x1, y1), axis=0)
all_labels = np.concatenate((x2, y2), axis=0)
training_digit_input = all_data[:training_val, :, :]
training_digit_label = all_labels[:training_val]
test_digit_input = all_data[training_val:, :, :]
test_digit_label = all_labels[training_val:]
# training_digit_input = x1
# training_digit_label = x2
# test_digit_input = y1
# test_digit_label = y2
(test_digit_input1, test_digit_label1), (test_digit_input2, test_digit_label2) = mnist.load_data()
test_digit_input, test_digit_label = np.concatenate((test_digit_input1, test_digit_input2), axis=0), np.concatenate((test_digit_label1, test_digit_label2), axis=0)



digit_classifier = MyClassifier(10, 300)
training_data = BasicFeatureExtractor(training_digit_input)
digit_classifier.train(training_data, training_digit_label)
test_data = BasicFeatureExtractor(test_digit_input)
basic_feature_predictions = digit_classifier.predict(test_data)

basic_feature_accuracy = np.count_nonzero(
    basic_feature_predictions == test_digit_label)/test_digit_label.shape[0]
print(f"Accuracy on test data using basic features is: {basic_feature_accuracy}")
training_data = BetterFeatureExtractor(training_digit_input)
digit_classifier.train(training_data, training_digit_label)

test_data = BetterFeatureExtractor(test_digit_input)
new_feature_predictions = digit_classifier.predict(test_data)
new_feature_accuracy = np.count_nonzero(
    new_feature_predictions == test_digit_label)/test_digit_label.shape[0]
print(f"Accuracy on test data using new features is: {new_feature_accuracy}")

accuracy_gain = new_feature_accuracy - basic_feature_accuracy
print(f"Accuracy gained by using new features is: {accuracy_gain}")
