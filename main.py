#main.py
import numpy as np
import time 

from search_algorithms import forward_selection, backward_elimination
from search_algorithms import stub
from validator import Validator
from nearest_neighbor import NearestNeighbor

def load_dataset(dataset): # read  file, split into labels and features
    data = np.loadtxt(dataset)
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    return X, y

def normalize(X): # make each feature column have similar scale
    means = X.mean(axis = 0)
    stand_dev = X.std(axis = 0)
    stand_dev[stand_dev == 0] = 1.0
    return (X - means) / stand_dev


def main():
    print("Welcome to Bertie Woosters bgarc208/hwhee004 Feature Selection Algorithm.\n")
    #user_feat = int(input("Please enter total number of features: "))
    print("Choose file:")
    print("\n1) small-test-dataset-2-2.txt")
    print("\n2) large-test-dataset-2.txt")

    user_inpt = input("Enter 1 or 2: ").strip()

    if user_inpt == "1":
        dataset_path = "small-test-dataset-2-2.txt"
    elif user_inpt == "2":
        dataset_path = "large-test-dataset-2.txt"
    else:
        # type a filename 
        #dataset_path = user_inpt
        print("Invalid")
        return

    X, y = load_dataset(dataset_path)
    X = normalize(X)
    num_features = X.shape[1] # get number of columns

    print(f"\nThis dataset has {num_features} features.\n")
    print("\nType the number of the algorithm you want to run.\n")
    print("\t1. Forward Selection\n\t2. Backward Elimination\n\t3. NN on features {3, 5, 7},\n\t4. NN on features {3, 15, 27},\n")
    
    classifier = NearestNeighbor()
    val = Validator(classifier)

    def accuracy(feature_set): # helper function for accuracy hard coded features
        return val.leave_one_out_accuracy(X, y, feature_set)
    
    user_input = int(input("Please enter a number: "))
    start = time.time()

    if user_input == 1:
        forward_selection(num_features, accuracy)
    elif user_input == 2:
        backward_elimination(num_features, accuracy)
    elif user_input == 3:
        feat = [3, 5, 7]   # hard coded
        acc = accuracy(feat)
        print("Using only feature(s) {3, 5, 7}") 
        print(f"Accuracy is {acc:.1f}%")
    elif user_input == 4:
        feat = [1, 15, 27]   # hard coded
        acc = accuracy(feat)
        print("Using only feature(s) {1, 15, 27}") 
        print(f"Accuracy is {acc:.1f}%")
    else:
        print("Invalid")

    print(f"\nTime taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()