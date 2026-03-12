from .label_display import (
    plot_labels_on_embedding,
    plot_class_distribution,
    plot_crosstab_heatmap,
    plot_predicted_classes_on_embedding,
)
from .classification import (
    prepare_classification_data,
    train_classifier,
    evaluate_classifier,
    plot_confusion_matrix,
    predict_all_samples,
)
from .latent_analysis import (
    compute_2d_embedding,
    build_label_latent_df,
    plot_latent_pairwise_correlation,
    plot_latent_vs_label_boxplots,
)
