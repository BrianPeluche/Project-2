#main.py
from search_algorithms import forward_selection
from search_algorithms import stub

def main():
    print("Welcome to Bertie Woosters bgarc208/hwhee Feature Selection Algorithm.\n")
    user_feat = int(input("Please enter total number of features: "))

    print("\nType the number of the algorithm you want to run.\n")
    print("\tForward Selection\n\tBackward Elimination\n\tBertie\'s Special Algorithm.\n")

    user_input = int(input("Please enter a number: "))
    if user_input == 1:
        forward_selection(user_feat, stub)

if __name__ == "__main__":
    main()