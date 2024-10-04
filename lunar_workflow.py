from keras.src.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from data_loading import load_mseed_data, load_catalog, list_files_in_directory
from preprocessing import highpass_filter, wavelet_denoising
from model import build_model
from test_model import test_model_on_data
from seismic_noise_tomography import seismic_noise_tomography  # Import the noise tomography functionality
import numpy as np
import os

def lunar_workflow(training_data_path, catalog_path, test_data_paths, output_path):
    """Main workflow for handling lunar seismic data."""

    # Load and preprocess lunar data for training
    mseed_files = list_files_in_directory(training_data_path)
    print(f"Found {len(mseed_files)} files in the training data directory.")

    if len(mseed_files) == 0:
        raise ValueError(f"No files found in directory: {training_data_path}")

    # Load the quake catalog
    catalog = load_catalog(catalog_path)

    # Prepare training data
    all_denoised_data = []
    for mseed_file in mseed_files:
        seismic_data, _ = load_mseed_data(mseed_file)
        filtered_data = highpass_filter(seismic_data, cutoff=0.1, fs=50)
        denoised_data = wavelet_denoising(filtered_data)
        all_denoised_data.append(denoised_data)

    # Pad or truncate sequences to match the sequence length used in test data
    X_train = pad_sequences(all_denoised_data, maxlen=72000)  # Ensure the same length as Mars data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Add channel dimension for Conv1D

    # Encode string labels to numeric
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(catalog['mq_type'].values)  # Convert string labels to numeric

    # Build and train the model
    model = build_model(input_shape=(72000, 1))  # Use 72000 as the input shape for Lunar data
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

    # Test the model on each test set for lunar data
    for test_path in test_data_paths:
        test_model_on_data(test_path, model, os.path.join(output_path, "lunar_output.csv"))

    # Run Seismic Noise Tomography (if applicable)
    seismic_noise_tomography(r"D:\Nasa space apps\surce code\lunar_velocity_data.xlsx")
