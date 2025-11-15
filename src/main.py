import preprocess
import train_autoencoder
#import evaluate
#import cross_validation
import evaluation1
from config import CFG

if __name__ == "__main__":
    # When you change this pipline you also need to change the param described in preprocess.py
    pipeline = f"{CFG.OUTLIER_METHOD}_{CFG.NORMALIZE_METHOD}_{CFG.LOWPASS_FILTER}_{CFG.FEATURE_SELECTION}_Slidingwindow_"


    # You can ignore the training since the models has been trained with all Pipelines,
    # but always run the Preprocess.py before evaluation
    preprocess.preprocess_all_data(CFG.OUTLIER_METHOD, CFG.NORMALIZE_METHOD, CFG.LOWPASS_FILTER, CFG.FEATURE_SELECTION)
    train_autoencoder.train_lstm_ae_with_cv(pipeline)
    evaluation1.evaluate_binary_detection()
    #evaluate.evaluate_pipeline(pipeline)


    # cross-validation script to run Preprocess->Train->Evaluate on all the possible pipelines
    #cross_validation.run_cross_validation()

