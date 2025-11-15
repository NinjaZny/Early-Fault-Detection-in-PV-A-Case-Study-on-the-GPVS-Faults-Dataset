import preprocess
import train_autoencoder
#import evaluate
#import cross_validation
import evaluation1

if __name__ == "__main__":
    # When you change this pipline you also need to change the param described in preprocess.py
    pipeline = "IQR_minmax_butterworth_none_Slidingwindow_"


    # You can ignore the training since the models has been trained with all Pipelines,
    # but always run the Preprocess.py before evaluation
    #preprocess.preprocess_all_data()
    #train_autoencoder.train_lstm_ae_with_cv(pipeline)
    evaluation1.evaluate_binary_detection()
    #evaluate.evaluate_pipeline(pipeline)


    # cross-validation script to run Preprocess->Train->Evaluate on all the possible pipelines
    #cross_validation.run_cross_validation()

