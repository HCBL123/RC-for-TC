import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the confusion matrix
confusion_matrix = np.array([
    [15807, 2774, 1518, 2286, 1587, 266],
    [2914, 17687, 3620, 1769, 1569, 655],
    [374, 577, 5382, 302, 195, 81],
    [1190, 916, 557, 8025, 679, 96],
    [755, 694, 341, 513, 6487, 752],
    [42, 75, 44, 36, 113, 2684]
])

# Define emotion labels
emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

# Create a figure and axis with larger size
plt.figure(figsize=(12, 10))

# Create heatmap with improved visibility
sns.heatmap(confusion_matrix, 
            annot=True,  # Show numbers in cells
            fmt='d',     # Use integer format
            cmap='YlOrRd',  # Yellow to Orange to Red colormap
            xticklabels=emotion_labels,
            yticklabels=emotion_labels,
            square=True)  # Make cells square

# Customize the plot
plt.title('Emotion Recognition Confusion Matrix', pad=20, size=14)
plt.xlabel('Predicted Emotion', labelpad=10)
plt.ylabel('True Emotion', labelpad=10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with high DPI for better quality
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()