import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo

def evaluate_model(class_name, predicted_class_name, all_classes):

    true_positives = [0] * len(all_classes)
    false_positives = [0] * len(all_classes)
    true_negatives = [0] * len(all_classes)
    false_negatives = [0] * len(all_classes)

    precision = [0] * len(all_classes)
    recall = [0] * len(all_classes)
    f1_score = [0] * len(all_classes)
    accuracy = [0] * len(all_classes)

    for i, obj in enumerate(all_classes):
        
        # instantiating TP, FP, TN and FN for each class
        n_true_positives = 0
        n_false_positives = 0
        n_true_negatives = 0
        n_false_negatives = 0

        current_class = str(all_classes[i])
            
        for j in range(len(class_name)):

            if (current_class == class_name[j] and current_class == predicted_class_name[j]): print('JEEEEEE')

            print(f"Predicted class: {predicted_class_name[j]}")
            print(f"True class: {class_name[j]}")
            print('------------------')

            if (predicted_class_name[j] == class_name[j] and class_name[j] == current_class): n_true_positives += 1
            if (predicted_class_name[j] == current_class and predicted_class_name[j] != class_name[j]): n_false_positives += 1
            if (predicted_class_name[j] == class_name[j] and predicted_class_name[j] != current_class): n_true_negatives += 1
            if (predicted_class_name[j] != class_name[j] and class_name[j] == current_class): n_false_negatives += 1

        print("For class: " + current_class + " we have:")
        print(f"True positives: {n_true_positives}")
        print(f"False positives: {n_false_positives}")
        print(f"True negatives: {n_true_negatives}")
        print(f"False negatives: {n_false_negatives}")
        print('------------------')

        true_positives[i] = n_true_positives
        false_positives[i] = n_false_positives
        true_negatives[i] = n_true_negatives
        false_negatives[i] = n_false_negatives

    for i in range(len(all_classes)):

        try:
            precision[i] = true_positives[i] / (true_positives[i] + false_positives[i])
        except:
            precision[i] = np.nan
        try:
            recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i])
        except:
            recall[i] = np.nan
        try:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        except:
            f1_score[i] = np.nan
        try:
            accuracy[i] = (true_positives[i] + true_negatives[i]) / (true_positives[i] + true_negatives[i] + false_positives[i] + false_negatives[i])
        except:
            accuracy[i] = np.nan

    return precision, recall, f1_score, accuracy, true_positives, false_positives, true_negatives, false_negatives


def plot_confusion_matrix_static(true_positives, false_positives, true_negatives, false_negatives, classes):
    N = len(classes)  # Number of classes

    # Create the confusion matrix from the provided metrics
    confusion_matrix = np.zeros((N, N))
    for i in range(N):
        confusion_matrix[i, i] = true_positives[i]  # True positives on the diagonal
        for j in range(N):
            if i != j:
                confusion_matrix[i, j] = false_positives[j]  # False positives on the off-diagonal

    # Normalize the confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.Blues  # Choose a colormap

    # Plot the confusion matrix
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set labels and title
    ax.set(xticks=np.arange(N),
           yticks=np.arange(N),
           xticklabels=classes,
           yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_interactive(true_positives, false_positives, true_negatives, false_negatives, class_names):
    N = len(class_names)  # Number of classes

    # Create the confusion matrix from the provided metrics
    confusion_matrix = np.zeros((N, N))
    for i in range(N):
        confusion_matrix[i, i] = true_positives[i]  # True positives on the diagonal
        for j in range(N):
            if i != j:
                confusion_matrix[i, j] = false_positives[j]  # False positives on the off-diagonal

    # Normalize the confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Create a Plotly heatmap figure
    fig = go.Figure(data=go.Heatmap(z=cm_normalized[::-1],  # Reverse the values on the y-axis
                                    x=class_names,
                                    y=class_names[::-1],  # Reverse the class names for y-axis
                                    colorscale='Blues'))

    # Set axis labels and title
    fig.update_layout(title={'text': 'Confusion Matrix', 'x': 0.5},  # Centered plot title
                      xaxis_title='Predicted label',
                      yaxis_title='True label')

    # Save the figure as an HTML file
    pyo.plot(fig, filename='../results/confusion_matrix.html', auto_open=True)
