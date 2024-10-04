from data_loading import load_mseed_data, list_files_in_directory
from preprocessing import highpass_filter, wavelet_denoising
import numpy as np
import os
from keras.src.utils import pad_sequences

def test_model_on_data(test_data_path, model, output_file):
    """Test the model on new data and output predictions."""

    # Load the test data (assuming a similar format to training data)
    mseed_files = list_files_in_directory(test_data_path)

    all_test_data = []
    for mseed_file in mseed_files:
        seismic_data, _ = load_mseed_data(mseed_file)
        filtered_data = highpass_filter(seismic_data, cutoff=0.1, fs=50)
        denoised_data = wavelet_denoising(filtered_data)
        all_test_data.append(denoised_data)

    # Pad or truncate sequences to match the training data shape
    X_test = pad_sequences(all_test_data, maxlen=72000)  # Ensure the same length as training data
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Add channel dimension

    # Make predictions
    test_predictions = (model.predict(X_test) > 0.5).astype(int)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save predictions
    np.savetxt(output_file, test_predictions, delimiter=",", fmt="%d")
