#search_algorithms.py
import random
# stub
def stub(set):
    return random.uniform(30.0, 80.0)

#Greedy Forward selection
def forward_selection(feat_num, feat_idx):
    curr_set = [] # currently selected so far
    best_set = []
    best_accuracy = feat_idx(curr_set)

    print(f"Using no features and \"random\" evaluation, I get an accuracy of {best_accuracy:.1f}%\n")
    print("Beginning search.\n")

    for level in range(1, feat_num + 1): # add feature one at a time
        feature_add_level = None # feature to add at this level
        best_accuracy_level = -1.0 # best accuracy at this level

        for feature in range(1, feat_num + 1): # add feature we dont have
            if feature in curr_set:
                continue # Skip features that are already selected

            temp_set = curr_set + [feature]
            accuracy = feat_idx(temp_set) # Use the provided evaluation function  or stub  to get accuracy

            feat_str = ",".join(str(f) for f in temp_set) # Print the trace line showing the candidate feature set and accuracy
            print(f"Using feature(s) {{{feat_str}}} accuracy is {accuracy:.1f}%")

            if accuracy > best_accuracy_level: # Keep track of the best candidate seen at this level
                best_accuracy_level = accuracy
                feature_add_level = feature

        if feature_add_level is not None: # After testing all candidates at this level, actually add the best feature
            curr_set.append(feature_add_level)

            if best_accuracy_level > best_accuracy: # If this level's best accuracy is better than the best so far,
                best_accuracy = best_accuracy_level
                best_set = curr_set.copy()
                features_str = ",".join(str(f) for f in curr_set)  # update best_accuracy and best_set and print the "was best" message
                print(f"Feature set {{{features_str}}} was best, accuracy is {best_accuracy_level:.1f}%")
            else:
                # Accuracy got worse compared to the best so far
                features_str = ",".join(str(f) for f in curr_set)
                print("(Warning, Accuracy has decreased!)")

    best_features_str = ",".join(str(f) for f in best_set)
    print(f"Finished search!! The best feature subset is {{{best_features_str}}}, " f"which has an accuracy of {best_accuracy:.1f}%")

    return best_set, best_accuracy
