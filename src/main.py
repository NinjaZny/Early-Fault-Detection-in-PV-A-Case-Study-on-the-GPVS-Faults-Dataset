import preprocess
import train_autoencoder

if __name__ == "__main__":
    preprocess.preprocess_all_data()
    # train_autoencoder.train_lstm_ae("none_minmax_butterworth_none_Slidingwindow_")