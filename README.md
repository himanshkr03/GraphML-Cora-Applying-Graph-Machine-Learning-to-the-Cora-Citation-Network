# GraphML-Cora-Applying-Graph-Machine-Learning-to-the-Cora-Citation-Network
## Project Overview

This project explores the application of graph-based machine learning techniques for node classification on the Cora dataset. The Cora dataset is a citation network where nodes represent research papers and edges represent citations between them. Each paper is classified into one of seven academic topics. 

The project utilizes both traditional machine learning models (Random Forest, XGBoost, SVM, Logistic Regression) and a Graph Neural Network (GNN) with a Graph Convolutional Network (GCN) layer for node classification. A Genetic Algorithm (GA) is also employed to optimize the hyperparameters of the GNN.

## Dataset

**Cora Dataset:**

* **Nodes:** 2708 research papers
* **Edges:** 5429 citations (undirected)
* **Features:** 1433 bag-of-words features representing the paper's content
* **Classes:** 7 academic topics

## Algorithms

### 1. Traditional Machine Learning Models

* **Random Forest:** An ensemble learning method that constructs multiple decision trees and combines their predictions.
* **XGBoost:** A gradient boosting algorithm that builds an ensemble of weak learners sequentially, improving upon previous models.
* **Support Vector Machine (SVM):** A supervised learning algorithm that finds an optimal hyperplane to separate data points into different classes.
* **Logistic Regression:** A statistical model used for binary classification, extended to multi-class problems using techniques like one-vs-rest.

### 2. Graph Neural Network (GNN) with GCN

* **Graph Convolutional Network (GCN):** A type of neural network designed to operate on graph-structured data. It leverages the graph's structure by aggregating information from neighboring nodes to learn node representations.
* **GNN Architecture:** The implemented GNN consists of two GCN layers followed by a ReLU activation function and a log-softmax output layer.

### 3. Genetic Algorithm (GA)

* **Purpose:** Used to optimize the hyperparameters of the GNN, including the hidden layer size, learning rate, and weight decay.
* **Process:** The GA creates a population of individuals, each representing a set of hyperparameters. It then iteratively applies selection, crossover (mating), and mutation operators to evolve the population towards better solutions. The fitness of each individual is determined by the accuracy of the GNN trained with the corresponding hyperparameters.

## Analysis and Results

### 1. Graph Structure Analysis

* **Degree Distribution:** The Cora dataset exhibits a power-law degree distribution, indicating the presence of a few highly connected nodes (hubs) and many less connected nodes.
* **Clustering Coefficient:** The average clustering coefficient is relatively high, suggesting the formation of clusters or communities within the network.
* **Centrality:** Nodes with high betweenness centrality act as bridges between different parts of the network and are considered influential.

### 2. Traditional Machine Learning Model Performance

* The traditional ML models achieved decent accuracy on the node classification task.
* Random Forest and XGBoost generally outperformed SVM and Logistic Regression.
* **Accuracy:** Ranged from ~70% to ~80%
* **Precision, Recall, F1-Score:** Consistent with accuracy trends.
* **ROC-AUC:** A metric indicating the model's ability to distinguish between classes (applicable for binary or multi-class problems).

### 3. GNN Performance

* The GNN with GCN achieved competitive performance compared to traditional ML models.
* **Accuracy:** ~80% or higher
* The GNN's ability to leverage the graph structure for learning contributes to its improved performance.

### 4. Genetic Algorithm Optimization

* The GA successfully identified optimal hyperparameters for the GNN, leading to further performance improvements.
* The optimization process involved evolving a population of hyperparameter sets over multiple generations.
* The fitness of each individual was evaluated based on the accuracy of the GNN trained with the corresponding hyperparameters.

## Insights

* **Graph Structure:** The Cora dataset exhibits characteristics typical of real-world networks, such as a power-law degree distribution and community structure.
* **Model Performance:** Both traditional ML models and GNNs can be effective for node classification on the Cora dataset.
* **GNN Advantages:** GNNs leverage the graph structure, leading to improved performance compared to traditional ML models that only consider node features.
* **Hyperparameter Optimization:** The GA demonstrated the benefits of hyperparameter optimization for GNNs.

## Visualizations

The notebook includes various visualizations to illustrate the graph structure, model performance, and optimization progress:

* **Graph Structure Visualization:** Shows the nodes and edges of the Cora network.
* **Degree Distribution:** Histograms displaying the distribution of node degrees.
* **Clustering Coefficient Distribution:** Histograms displaying the distribution of clustering coefficients.
* **Centrality Analysis:** Bar plots highlighting the most influential nodes based on betweenness centrality.
* **t-SNE Visualization:** Scatter plots showing the node features projected into a 2D space using t-SNE for dimensionality reduction.
* **Model Performance Comparison:** Line graphs comparing the accuracy, precision, recall, F1-score, and ROC-AUC of different models.
* **Genetic Algorithm Optimization Progress:** Plots showing the evolution of fitness (accuracy) over generations.
* **Hyperparameter Distribution:** Box plots and histograms visualizing the distribution of hyperparameters within the GA population.

## Future Work

* Explore other GNN architectures and variations.
* Experiment with different hyperparameter optimization techniques.
* Apply the approach to other graph datasets.
* Investigate the interpretability of GNN predictions.


## Conclusion

This project demonstrated the effectiveness of graph-based machine learning for node classification on the Cora dataset. GNNs, particularly GCNs, exhibited competitive performance and the GA proved valuable for hyperparameter optimization. The insights gained from this project can be extended to other graph-related tasks and datasets.
