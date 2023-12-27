#!/usr/bin/env python
# coding: utf-8

# # ID3 implementation



# Some required imports.
# Make sure you have these packages installed on your system.
import pandas as pd
import numpy as np
import math 




# Some functions needed when creating the decision tree
class Utils:
    
    #Calculates the entropy for the values in target_column, which should be of type Pandas.DataFrame
    @staticmethod
    def entropy(target_column):
        
        num_instances_all_classes = len(target_column)
    
        value_counts = target_column.value_counts()
        
        entropy = 0  
        
        for target_value in value_counts.keys():         
            num_instances_current_class = value_counts[target_value[0]]
            share_instances_current_class = num_instances_current_class/num_instances_all_classes   
            entropy = entropy - share_instances_current_class * math.log2(share_instances_current_class)
    
        return entropy
    
    # Identifies the dominating class for the entries in target_column
    # target_column should be of typePandas.DataFrame 
    @staticmethod
    def find_dominating_class(target_column):
                
        return_class = None
        count_return_class = 0
        
        value_counts = target_column.value_counts()
        for target_value in value_counts.keys():
            if value_counts[target_value[0]] > count_return_class:
                count_return_class = value_counts[target_value[0]] 
                return_class = target_value        
                
        return return_class
    
    # This method returns a unique node identifier
    @staticmethod
    def get_next_node_id():
        global node_id_counter
        node_id = node_id_counter
        node_id_counter = node_id_counter + 1
        
        return node_id
        
    # Resets the global variable node_counter_id used to assign unique identifiers for the nodes.
    @staticmethod
    def reset_node_id_counter():
        global node_id_counter
        node_id_counter = 1
        



# This class is used to represent a node in the decision tree. 
class Node:
    
    #Constructor for the node class taking the following parameters.
    def __init__(self, height, max_height, input_columns, target_column, parent, parent_split_value):
        
        # Height in the tree for the current node. 
        # It is suggested that the height of the root of the tree is 1. 
        self.height = height
        
        # The maximum height of the decision tree, meaning that no path from the root 
        # to a leaf should contain more than 3 nodes. 
        self.max_height = max_height
        
        # Pandas columns containing the input features. 
        # One column for each input feature, and each row is data point (or instance)
        self.input_columns = input_columns
        
        # Pandas column containing the target variable. 
        # Each row in this column contains the target value corresponding to the same row in self.input_columns 
        self.target_column = target_column
        
        # The variable to split on in this node. 
        # For leaf nodes this is not set.
        self.split_variable = None
        
        # The names of the input features
        self.input_names = self.input_columns.keys()
        
        # The name of the target variable
        self.target_name = self.target_column.keys()[0]
        
        # Reference to the parent of this node. 
        # This is None for the root of the tree
        self.parent = parent
        
        # The class for this node if it is a leaf node. 
        # This is left empty for non leaf nodes. 
        self.class_name = None
        
        # The unique id of this node. Used when printing the tree. 
        self.id = Utils.get_next_node_id()
    
        # Dictionary used to keep track of the child nodes. Empty if the node is a leaf. 
        self.children = {}


        
    # Method is used to expand a node when constucting the decision tree  
    def split(self):
        # Check if maximum height is reached or if there are no features left to split on
        if self.height >= self.max_height or len(self.input_names) == 0:
            # Determine the class of the leaf node
            self.class_name = self.determine_class_name()
            return

        best_feature, best_info_gain = None, -1
        for feature in self.input_names:
            info_gain = self.calculate_information_gain(feature)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature

        # If no feature gives information gain, set as leaf node with class
        if best_feature is None:
            self.class_name = self.determine_class_name()
            return

        self.split_variable = best_feature
        for value in self.input_columns[best_feature].unique():
            subset = self.input_columns[self.input_columns[best_feature] == value]
            target_subset = self.target_column[self.input_columns[best_feature] == value]
            new_input_columns = subset.drop(columns=[best_feature])
            new_height = self.height + 1
            self.children[value] = Node(new_height, self.max_height, new_input_columns, target_subset, self, value)
            self.children[value].split()
            
            # This method creates a text representation of the decision tree.   
            def print(self):
                    print("Node<" + str(self.id) + ">" )
                
            if not self.children:
                    print("  Leaf node - Parent: " + str(self.parent.id) + ", Decision: " + self.class_name)
                    
            else:
                    if self.parent is None:
                        print("  Non leaf node - Parent: None")
                    else:
                        print("  Non leaf node - Parent: " + str(self.parent.id))    
                    print("  Split variable: " + self.split_variable)
                    
                        
                    for child_split_value in self.children.keys():
                        child_node = self.children[child_split_value]  
                        print("    Child_node: " + str(child_node.id) + ", split_value: " + str(child_split_value))


    def calculate_information_gain(self, feature):
        """
        Calculate the information gain for a feature using the entropy function from Utils class.
        """
        # Calculate total entropy using the target column
        total_entropy = Utils.entropy(self.target_column)

        # Get unique values and their counts for the feature
        val, counts = np.unique(self.input_columns[feature], return_counts=True)

        # Calculate weighted entropy for each unique value
        weighted_entropy = sum(
            (counts[i] / sum(counts)) * Utils.entropy(self.target_column[self.input_columns[feature] == val[i]])
            for i in range(len(val))
        )

        # Information gain is the difference between total entropy and weighted entropy
        info_gain = total_entropy - weighted_entropy
        return info_gain
    
    def determine_class_name(self):
        """
        Determine the class name for a leaf node.
        This method finds the most common class in the target column.
        """
        if self.target_column is None or len(self.target_column) == 0:
            return None  # Return None if target_column is empty or not set

        # Find the most common class in the target column
        most_common_class = self.target_column.value_counts().idxmax()
        return most_common_class
    

        
        
    
    
    

    
        


    


#Class used to represent our decision tree generated using the ID3 algorithm

class DecisionTree:

    # Constructor that takes three input variables:
    #   max_height - The maximum height of the decision tree. 
    #   instances - Pandas DataFrame respresenting instance vectors (both input and target)
    #   target_name - Name of the target column.
    def __init__(self, max_height, instances, target_name):
        
        # Reset the global node_id_counter variable to make sure the root node gets id 1
        Utils.reset_node_id_counter()
        
        self.max_height = max_height;
        
        # Create pandas dataframe containing only the target column. 
        self.target_column = instances[[target_name]]
 
        
        if self.target_column[target_name].unique().size != 2:
            print("Error: Only binary target variables are supported")
            exit()
        
        # Create pandas dataframe containing all input feature columns.
        self.input_columns = instances.drop([target_name], axis=1) 
      
        node_id_counter = 1
    
        # Create the root of the tree
        self.root = Node(1, self.max_height, self.input_columns, self.target_column, None, None)

        # Generate the decision tree by calling the self.generate() function.
        self.generate()
    
    # This method is used to generate a decision tree using the ID3 algorithm
    # All tree nodes are generated recursively when adding new nodes.
    def generate(self):
        self.root.split()
    
    
    #This method prints all nodes in the decision tree
    def print_tree(self):
    
        #Check if decision tree is empty
        if self.root is None:
            print("Decision tree is emtpy")
            return
        
        #Otherwise, iterate over all nodes in the decision tree
        
        node_queue = []
        node_queue.append(self.root)
        
        while node_queue:
            current_node = node_queue.pop(0)
            current_node.print()
            
            for child_node in current_node.children.values():
                node_queue.append(child_node)
            




# This function contains the code to create your decision tree 
def main():

    # Read the data from csv file and store in a pandas datafrane
    golf_dataframe = pd.read_csv("golf_dataset.csv")

    print(golf_dataframe)
    print("\n")

    max_height = 3
    
    # Generate decision tree for golf data set, target variable "Play Golf" and max_height=3 
    dt = DecisionTree(max_height, golf_dataframe, 'Play Golf')


    # Print content of the created tree
    dt.print_tree()




# Call the main function run the code
main()

