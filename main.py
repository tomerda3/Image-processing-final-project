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
    if HHD_engine.load_model() :
        print("Model loaded.")
        print("Model testing:")
        features = features_extract_combine(HHD_engine.train_images)
        HHD_engine.train_model(features, HHD_engine.train_labels)

    print("Model validation:")
    HHD_engine.validation_model(HHD_engine.val_images, HHD_engine.val_labels)
    print("Model testing:")
    HHD_engine.test_model(HHD_engine.test_images, HHD_engine.test_labels)
    if HHD_engine.load_model():
        print("Model saved.")
    HHD_engine.save_model()

if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
