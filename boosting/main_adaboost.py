from boosting.adaboost import AdaBoostClassifier
from datasets import two_moon_dataset
from boosting.utils import *

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # X_train, Y_train, X_test, Y_test = gaussians_dataset(2, [300, 400], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    # print(X_train)
    # X_train, Y_train, X_test, Y_test = h_shaped_dataset()
    X_train, Y_train, X_test, Y_test = two_moon_dataset(n_samples=500, noise=0.2)

    # visualize dataset
    plot_2d_dataset(X_train, Y_train, 'Training')

    # train model and predict
    model = AdaBoostClassifier(n_learners=2)

    model.fit(X_train, Y_train, verbose=True)
    P = model.predict(X_test)

    # visualize the boundary!
    plot_boundary(X_train, Y_train, model)

    # evaluate and print error
    error = 100 * float(np.sum(P == Y_test)) / Y_test.size
    print('Test set - Classification Accuracy: {}% '.format(error))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
