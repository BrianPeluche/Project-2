#main.py
import numpy as np
import time 

from search_algorithms import forward_selection, backward_elimination
from search_algorithms import stub
from validator import validator
from nearest_neighbor import NearestNeighbor

def load_dataset(dataset):
    data = np.txt(dataset)
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    return X, y

def normalize(X):
    means = X.mean(axis = 0)
    stds = X.std(axis = 0)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def main():
    print("Welcome to Bertie Woosters bgarc208/hwhee004 Feature Selection Algorithm.\n")
    #user_feat = int(input("Please enter total number of features: "))
    print("Choose file:")
    print("\n1) large-test-dataset-2.txt")
    print("\n2) small-test-dataset-2-2.txt")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        dataset_path = "large-test-dataset-2.txt"
    elif choice == "2":
        dataset_path = "small-test-dataset-2-2.txt"
    else:
        # type a filename 
        dataset_path = choice

    X, y = load_dataset(dataset_path)
    X = normalize(X)
    num_features = X.shape[1]
    print(f"\nThis dataset has {num_features} features.\n")

    print("\nType the number of the algorithm you want to run.\n")
    print("\tForward Selection\n\tBackward Elimination\n\tBertie\'s Special Algorithm.\n")
    classifier = NearestNeighbor()
    val = validator(classifier)

    def accuracy(feature_set):
        return val.leave_one_out_accuracy(X, y, feature_set)
    
    user_input = int(input("Please enter a number: "))
    start = time.time()

    if user_input == 1:
        forward_selection(num_features, accuracy)
    elif user_input == 2:
        backward_elimination(num_features, accuracy)
    # elif user_input == 3:
    #     classifier = NearestNeighbor()
    #     val = validator(classifier) 
    else:
        print("Invalid")

    print(f"\nTime taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()