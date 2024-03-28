from engine import *
from pathlib import Path
from feature_extraction import *
import sys

def main(argv=None):
    # input_path = argv[0]
    # output_path = argv[1]
    # Construct HHD engine
    HHD_engine = construct_HHD_engine(
        base_dir=Path.cwd() / "HHD_gender",
        image_shape=(400, 400, 1)
    )
    train_images = HHD_engine.train_images

    print("Extracting Haralick features...")
    haralick_features = extract_haralick_features(train_images)
    print(f"haralick_features size: {haralick_features.shape}")
    # first experiment with LBP features
    radius = 1
    points = 8
    print("Extracting LBP features...")
    lbp_features = extract_lbp_features(train_images, radius, points)
    combined_features = np.concatenate((haralick_features, lbp_features), axis=1)
    print(f"combined_features size: {combined_features.shape}, radii: {radius}, points: {points}")

    # second experiment with LBP features
    radius = 8
    points = 24
    lbp_features = extract_lbp_features(train_images, radius, points)
    combined_features = np.concatenate((haralick_features, lbp_features), axis=1)
    print(f"combined_features size: {combined_features.shape}, radii: {radius}, points: {points}")

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    print("Training SVM model...")
    HHD_engine.train_SVM_model(combined_features, param_grid)
    print("Model trained successfully!")

    print("Saving model...")
    filename = "saved_model.pkl"
    HHD_engine.save_model(filename)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
