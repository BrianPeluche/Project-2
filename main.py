#main.py
from search_algorithms import forward_selection, backward_elimination
from search_algorithms import stub

def main():
    print("Welcome to Bertie Woosters bgarc208/hwhee004 Feature Selection Algorithm.\n")
    user_feat = int(input("Please enter total number of features: "))

    print("\nType the number of the algorithm you want to run.\n")
    print("\tForward Selection\n\tBackward Elimination\n\tBertie\'s Special Algorithm.\n")

    user_input = int(input("Please enter a number: "))
    if user_input == 1:
        forward_selection(user_feat, stub)
    elif user_input == 2:
        backward_elimination(user_feat, stub)
    else:
        print("Invalid")

if __name__ == "__main__":
    main()