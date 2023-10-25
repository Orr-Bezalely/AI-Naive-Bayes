import numpy as np
class knn:
    def __init__(self, classes, k):
        self.classes = classes
        self.k = k
    def standardise(self):
        mean = self.training_inputs.mean(axis=0)
        standard_diviation = self.training_inputs.std(axis=0)
        standard_diviation = np.where(np.isclose(standard_diviation, 0), 1, standard_diviation)
        return mean, standard_diviation

    def train(self, training_inputs, training_labels):
        self.training_inputs = training_inputs
        self.training_labels = training_labels
        # self.mean, self.standard_diviation = self.standardise()
        # self.standardised_data = (self.training_inputs - self.mean) / self.standard_diviation


    def predict(self, test_data):
        rows, columns = self.standardised_data.shape[0], test_data.shape[0]
        test_data_norm = (test_data - self.mean) / self.standard_diviation
        euclidean_distance = np.linalg.norm(self.standardised_data[:, np.newaxis, :] - test_data_norm[np.newaxis,...], axis=-1)
        prediction = np.zeros((columns, ))
        for column in range(columns):
            knn_indeces = np.argpartition(euclidean_distance[:, column], np.arange(self.k))[:self.k]
            # one_votes = np.count_nonzero(self.training_labels[knn_indeces] == 1)
            # prediction[column] = 1 if one_votes > self.k - one_votes else 0
            class_preds = np.array([np.count_nonzero(self.training_labels[knn_indeces] == class_i) for class_i in range(self.classes)])
            prediction[column] = np.argmax(class_preds, axis=0)
        return prediction


def k_N(data, batch_size):
    accuracies = []
    for i in range(0, data.shape[0], batch_size):
        training_batches = np.vstack((data[:i], data[i + batch_size:]))
        test_batch = data[i:i + batch_size]
        spam_classifier = knn(2, 7)
        spam_classifier.train(training_batches[:, 1:], training_batches[:, 0])
        class_predictions = spam_classifier.predict(test_batch[:, 1:])
        true_classes = test_batch[:, 0]
        training_set_accuracy = np.mean(np.equal(class_predictions, true_classes))
        accuracies.append(training_set_accuracy)
        # print(f"Accuracy on the training set: {training_set_accuracy}")
    x = np.array(accuracies)
    print(np.mean(x))


training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
test_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
all_data = np.vstack((training_spam, test_spam))
print("Shape of the all data set:", all_data.shape)
x = k_N(all_data, 150)

# training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
# test_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
# x = knn(2, 7)
# x.train(training_spam[:, 1:], training_spam[:, 0])
# predictions = x.predict(test_spam[:, 1:])
# print((test_spam.shape[0]-np.count_nonzero(predictions-test_spam[:, 0]))/test_spam.shape[0])



################################ PART 2 ##############################################

training_digit_input = np.load("data/training_digit_input.npy")
training_digit_label = np.load("data/training_digit_label.npy")
test_digit_input = np.load("data/test_digit_input.npy")
test_digit_label = np.load("data/test_digit_label.npy")
all_data_input = np.vstack((training_digit_input, test_digit_input))
all_data_label = np.vstack((training_digit_label, test_digit_label))


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


def BetterFeatureExtractor(input_data):
    """
    A function for your improved feature extractor

    :param input_data: a 3-dimensional numpy array of the shape (n, 28, 28)
                       input data of n images of 28x28 pixels

    :return extracted_features: a 2-dimensional numpy array of the shape (n, m)
                                extracted binary m features for n images
    """
    #extracted_features = BasicFeatureExtractor(input_data)
    n = input_data.shape[0]
    #extracted_features = np.zeros((n, 1))


    # SIDE_LENGTH = 28
    # square_size = 4
    # n = input_data.shape[0]
    # mean_per_block = np.zeros((n,(SIDE_LENGTH-square_size)**2))
    # for i in range(SIDE_LENGTH-square_size):
    #     for j in range(SIDE_LENGTH-square_size):
    #         x = np.mean(input_data[:, i:i+square_size, j:j+square_size], axis = (1,2))
    #         mean_per_block[:,i*(SIDE_LENGTH-square_size)+j] = x
    # extracted_features = np.hstack((extracted_features, np.where(mean_per_block > 110, 1, 0)))

    # # MANUALLY
    #
    # #Zero features
    # zero_top =  np.reshape(np.mean(input_data[:, 5:10, 7:23], (1,2)), (n,1))
    # zero_right =  np.reshape(np.mean(input_data[:, 7:23, 17:23], (1,2)), (n,1))
    # zero_bot =  np.reshape(np.mean(input_data[:, 7:23, 17:23], (1,2)), (n,1))
    # zero_left = np.reshape(np.mean(input_data[:, 10:22, 5:10], (1,2)), (n,1))
    # zero_features = np.hstack((np.where(zero_top > 100, 1, 0), np.where(zero_right > 95, 1, 0), np.where(zero_bot > 95, 1, 0), np.where(zero_left > 95, 1, 0)))
    # anti_zero_centre = np.reshape(np.mean(input_data[:, 12:16, 12:16], (1,2)), (n,1))
    # zero_anti_features = np.where(anti_zero_centre < 30, 1, 0)
    # extracted_features = np.hstack((extracted_features, zero_features, zero_anti_features))
    #
    # #One features
    # one_highlight = np.reshape(np.mean(input_data[:, 10:18, 12:16], (1,2)), (n,1))
    # one_bot_left = np.reshape(np.mean(input_data[:, 15:22, 11:15], (1,2)), (n,1))
    # one_top_right = np.reshape(np.mean(input_data[:, 5:12, 13:18], (1,2)), (n,1))
    # one_features = np.hstack((np.where(one_highlight > 135, 1, 0), np.where(one_bot_left > 130, 1, 0), np.where(one_top_right > 90, 1, 0)))
    # anti_one_bot_right = np.reshape(np.mean(input_data[:, 17:, 17:], (1,2)), (n,1))
    # anti_one_top_left = np.reshape(np.mean(input_data[:, :12, :12], (1,2)), (n,1))
    # one_anti_features = np.hstack((np.where(anti_one_bot_right < 7, 1, 0), np.where(anti_one_top_left > 8, 1, 0)))
    # extracted_features = np.hstack((extracted_features, one_features, one_anti_features))
    #
    # #Two features
    # two_top = np.reshape(np.mean(input_data[:, 3:8, 5:20], (1,2)), (n,1))
    # two_mid = np.reshape(np.mean(input_data[:, 5:15, 15:20], (1,2)), (n,1))
    # two_bot_left = np.reshape(np.mean(input_data[:, 15:23, 7:16], (1,2)), (n,1))
    # two_bot_right = np.reshape(np.mean(input_data[:, 16:22, 16:23], (1,2)), (n,1))
    # two_features = np.hstack((np.where(two_top > 65, 1, 0), np.where(two_mid > 100, 1, 0), np.where(two_bot_left > 120, 1, 0), np.where(two_bot_right > 95, 1, 0)))
    # anti_two_left_mid = np.reshape(np.mean(input_data[:, 11:13, :10], (1,2)), (n,1))
    # two_anti_features = np.where(anti_two_left_mid < 15, 1, 0)
    # extracted_features = np.hstack((extracted_features, two_features, two_anti_features))
    #
    # #Three features
    # three_top = np.reshape(np.mean(input_data[:, 4:8, 9:18], (1,2)), (n,1))
    # three_top_right = np.reshape(np.mean(input_data[:, 4:8, 16:20], (1,2)), (n,1))
    # three_mid = np.reshape(np.mean(input_data[:, 11:15, 11:17], (1,2)), (n,1))
    # three_bot_right = np.reshape(np.mean(input_data[:, 14:22, 16:20], (1,2)), (n,1))
    # three_bot = np.reshape(np.mean(input_data[:, 21:24, 7:16], (1,2)), (n,1))
    # three_features = np.hstack((np.where(three_top > 100, 1, 0), np.where(three_top_right > 90, 1, 0), np.where(three_mid > 140, 1, 0), np.where(three_bot_right > 110, 1, 0), np.where(three_bot > 115, 1, 0)))
    # three_anti_top_left = np.reshape(np.mean(input_data[:, 7:13, 5:8], (1,2)), (n,1))
    # three_anti_bot_left = np.reshape(np.mean(input_data[:, 15:25, 4:7], (1,2)), (n,1))
    # three_anti_bot_mid = np.reshape(np.mean(input_data[:, 17:19, 10:16], (1,2)), (n,1))
    # three_anti_features = np.hstack((np.where(three_anti_top_left < 30, 1, 0), np.where(three_anti_bot_left < 35, 1, 0), np.where(three_anti_bot_mid < 40, 1, 0)))
    # extracted_features = np.hstack((extracted_features, three_features, three_anti_features))
    #
    # #Four features
    # four_horizontal = np.reshape(np.mean(input_data[:, 12:17, 7:20], (1,2)), (n,1))
    # four_left_vertical = np.reshape(np.mean(input_data[:, 7:17, 7:12], (1,2)), (n,1))
    # four_top_right_vertical = np.reshape(np.mean(input_data[:, 7:12, 16:21], (1,2)), (n,1))
    # four_bot_right_vertical = np.reshape(np.mean(input_data[:, 16:24, 12:17], (1,2)), (n,1))
    # four_features = np.hstack((np.where(four_horizontal > 130, 1, 0), np.where(four_left_vertical > 95, 1, 0), np.where(four_top_right_vertical > 95, 1, 0), np.where(four_bot_right_vertical > 105, 1, 0)))
    # anti_four_bot_left = np.reshape(np.mean(input_data[:, 16:24, :11], (1,2)), (n,1))
    # anti_four_top_middle = np.reshape(np.mean(input_data[:, 10:12, 14:15], (1,2)), (n,1))
    # four_anti_features = np.hstack((np.where(anti_four_bot_left < 20, 1, 0), np.where(anti_four_top_middle < 30, 1, 0)))
    # extracted_features = np.hstack((extracted_features, four_features, four_anti_features))
    #
    # #Five features
    # five_top = np.reshape(np.mean(input_data[:, 4:8, 11:22], (1,2)), (n,1))
    # five_top_vert = np.reshape(np.mean(input_data[:, 6:15, 9:15], (1,2)), (n,1))
    # five_mid = np.reshape(np.mean(input_data[:, 12:15, 8:20], (1,2)), (n,1))
    # five_mid_vert = np.reshape(np.mean(input_data[:, 12:21, 15:21], (1,2)), (n,1))
    # five_bot = np.reshape(np.mean(input_data[:, 20:25, 8:17], (1,2)), (n,1))
    # five_features = np.hstack((np.where(five_top > 70, 1, 0), np.where(five_top_vert > 100, 1, 0), np.where(five_mid > 80, 1, 0), np.where(five_mid_vert > 65, 1, 0), np.where(five_bot > 85, 1, 0)))
    # five_anti_right_vert = np.reshape(np.mean(input_data[:, 10:13, 17:23], (1,2)), (n,1))
    # five_anti_mid = np.reshape(np.mean(input_data[:, 16:20, 7:13], (1,2)), (n,1))
    # five_anti_features = np.hstack((np.where(five_anti_right_vert < 30, 1, 0), np.where(five_anti_mid < 50, 1, 0)))
    # extracted_features = np.hstack((extracted_features, five_features, five_anti_features))
    #
    # #Six features
    # six_top = np.reshape(np.mean(input_data[:, 3:9, :], (1,2)), (n,1))
    # six_middle = np.reshape(np.mean(input_data[:, 13:17, 8:20], (1,2)), (n,1))
    # six_bot_left = np.reshape(np.mean(input_data[:, 13:22, 9:13], (1,2)), (n,1))
    # six_bot = np.reshape(np.mean(input_data[:, 19:21, 10:17], (1,2)), (n,1))
    # six_bot_right = np.reshape(np.mean(input_data[:, 11:20, 17:22], (1,2)), (n,1))
    # six_features = np.hstack((np.where(six_top > 15, 1, 0), np.where(six_middle > 110, 1, 0), np.where(six_bot_left > 130, 1, 0), np.where(six_bot > 165, 1, 0), np.where(six_bot_right > 80, 1, 0)))
    # six_anti_top_right = np.reshape(np.mean(input_data[:, 5:10, 19:23], (1,2)), (n,1))
    # six_anti_mid = np.reshape(np.mean(input_data[:, 17:19, 13:16], (1,2)), (n,1))
    # six_anti_features = np.hstack((np.where(six_anti_top_right < 15, 1, 0), np.where(six_anti_mid < 80, 1, 0)))
    # extracted_features = np.hstack((extracted_features, six_features, six_anti_features))
    #
    # #Seven features
    # seven_top = np.reshape(np.mean(input_data[:, 7:10, 6:20], (1,2)), (n,1))
    # seven_top_right = np.reshape(np.mean(input_data[:, 8:13, 16:19], (1,2)), (n,1))
    # seven_mid_right = np.reshape(np.mean(input_data[:, 14:17, 15:17], (1,2)), (n,1))
    # seven_mid = np.reshape(np.mean(input_data[:, 17:24, 12:15], (1,2)), (n,1))
    # seven_bot = np.reshape(np.mean(input_data[:, 21:25, 9:15], (1,2)), (n,1))
    # seven_features = np.hstack((np.where(seven_top > 120, 1, 0), np.where(seven_top_right > 155, 1, 0), np.where(seven_mid_right > 170, 1, 0), np.where(seven_mid > 130, 1, 0), np.where(seven_bot > 80, 1, 0)))
    # seven_anti_left = np.reshape(np.mean(input_data[:, 13:18, 5:12], (1,2)), (n,1))
    # seven_anti_right = np.reshape(np.mean(input_data[:, 20:25, 17:21], (1,2)), (n,1))
    # seven_anti_features = np.hstack((np.where(seven_anti_left < 30, 1, 0), np.where(seven_anti_right < 25, 1, 0)))
    # extracted_features = np.hstack((extracted_features, seven_features, seven_anti_features))
    #
    # #Eight features
    # eight_top = np.reshape(np.mean(input_data[:, 6:7, 10:20], (1,2)), (n,1))
    # eight_mid = np.reshape(np.mean(input_data[:, 13:17, 11:16], (1,2)), (n,1))
    # eight_bot = np.reshape(np.mean(input_data[:, 22:24, 9:14], (1,2)), (n,1))
    # eight_top_left = np.reshape(np.mean(input_data[:, 8:12, 9:12], (1,2)), (n,1))
    # eight_top_right = np.reshape(np.mean(input_data[:, 5:12, 17:20], (1,2)), (n,1))
    # eight_bot_left = np.reshape(np.mean(input_data[:, 18:23, 8:11], (1,2)), (n,1))
    # eight_bot_right = np.reshape(np.mean(input_data[:, 18:22, 15:18], (1,2)), (n,1))
    # eight_features = np.hstack((np.where(eight_top > 110, 1, 0), np.where(eight_mid > 180, 1, 0), np.where(eight_bot > 165, 1, 0), np.where(eight_top_left > 120, 1, 0), np.where(eight_top_right > 120, 1, 0), np.where(eight_bot_left > 110, 1, 0), np.where(eight_bot_right > 110, 1, 0)))
    # eight_anti_top = np.reshape(np.mean(input_data[:, 9:12, 13:16], (1,2)), (n,1))
    # eight_anti_bot = np.reshape(np.mean(input_data[:, 19:21, 12:14], (1,2)), (n,1))
    # eight_anti_left = np.reshape(np.mean(input_data[:, 13:17, 5:9], (1,2)), (n,1))
    # eight_anti_right = np.reshape(np.mean(input_data[:, 14:17, 18:21], (1,2)), (n,1))
    # eight_anti_features = np.hstack((np.where(eight_anti_top < 110, 1, 0), np.where(eight_anti_bot < 95, 1, 0), np.where(eight_anti_left < 30, 1, 0), np.where(eight_anti_right < 45, 1, 0)))
    # extracted_features = np.hstack((extracted_features, eight_features, eight_anti_features))
    #
    # #Nine features
    # nine_top = np.reshape(np.mean(input_data[:, 6:8, 10:17], (1,2)), (n,1))
    # nine_mid = np.reshape(np.mean(input_data[:, 13:17, 8:18], (1,2)), (n,1))
    # nine_left = np.reshape(np.mean(input_data[:, 9:15, 7:10], (1,2)), (n,1))
    # nine_top_right = np.reshape(np.mean(input_data[:, 7:15, 17:19], (1,2)), (n,1))
    # nine_mid_right = np.reshape(np.mean(input_data[:, 13:18, 15:17], (1,2)), (n,1))
    # nine_bot_right = np.reshape(np.mean(input_data[:, 14:22, 14:17], (1,2)), (n,1))
    # nine_features = np.hstack((np.where(nine_top > 90, 1, 0), np.where(nine_mid > 120, 1, 0), np.where(nine_left > 75, 1, 0), np.where(nine_top_right > 130, 1, 0), np.where(nine_mid_right > 170, 1, 0), np.where(nine_bot_right > 130, 1, 0)))
    # nine_anti_top = np.reshape(np.mean(input_data[:, 10:12, 12:15], (1,2)), (n,1))
    # nine_anti_left = np.reshape(np.mean(input_data[:, 18:25, 7:12], (1,2)), (n,1))
    # nine_anti_features = np.hstack((np.where(nine_anti_top < 50, 1, 0), np.where(nine_anti_left < 30, 1, 0)))
    # extracted_features = np.hstack((extracted_features, nine_features, nine_anti_features))

    return extracted_features

# digit_classifier = knn(10, 10)
# if True:
#     test_digit_input = np.load("data/test_digit_input.npy")
#     test_digit_label = np.load("data/test_digit_label.npy")
#     #     training_and_test_input = np.vstack((training_digit_input, test_digit_input))
#     #     training_and_test_label = np.vstack((training_digit_label, test_digit_label))
#     #     mean_pictures_mat = mean_pictures1()
#     #     for i in range(NUMBER_OF_CLASSES):
#     #         if True:
#     #             plt.imshow(mean_pictures_mat[i], cmap='gray')
#     #             plt.show()
#
#     # train classifier on basic features
#     training_data = BasicFeatureExtractor(training_digit_input)
#     digit_classifier.train(training_data, training_digit_label)
#
#     # test classifier on basic features
#     test_data = BasicFeatureExtractor(test_digit_input)
#     basic_feature_predictions = digit_classifier.predict(test_data)
#     basic_feature_accuracy = np.count_nonzero(
#         basic_feature_predictions == test_digit_label) / test_digit_label.shape[0]
#     print(f"Accuracy on test data using basic features is: {basic_feature_accuracy}")
#
#     # train classifier on new features
#     training_data = BetterFeatureExtractor(training_digit_input)
#     digit_classifier.train(training_data, training_digit_label)
#
#     # test_classifier on new features
#     test_data = BetterFeatureExtractor(test_digit_input)
#     new_feature_predictions = digit_classifier.predict(test_data)
#     new_feature_accuracy = np.count_nonzero(
#         new_feature_predictions == test_digit_label) / test_digit_label.shape[0]
#     print(f"Accuracy on test data using new features is: {new_feature_accuracy}")
#
#     # compare accuracies achieved on different sets of features
#     accuracy_gain = new_feature_accuracy - basic_feature_accuracy
#     print(f"Accuracy gained by using new features is: {accuracy_gain}")

#
# all_data = np.vstack((training_spam, test_spam))
# total, correct = 0, 0
# for row in test_spam:
#     print(total, correct)
#     prediction = knn_predict(row[1:], training_spam, k=1)
#     if prediction == row[0]:
#         correct += 1
#     total += 1
#
# print(correct/total*100)
