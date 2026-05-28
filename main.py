# main.py
import train2
import preprocessing

def main():
    processed = preprocessing.process_all()

    model_args = {
        "eeg_ch": 32, "emg_ch": 5, "imu_ch": 36,
        "embed_dim": 128, "num_classes": 10,
    }

    # DL kwargs include model_class and model_args 
    dl_gated = {
        "model_class": train2.GatedFusion,
        "model_args": model_args,
        "epochs": 70, "batch_size": 32,
        "lr": 1e-3, "augment": True, "patience": 10,
    }

    # dl_cross = {**dl_gated, "model_class": train2.CrossAttentionModel}

    # # ---- DL models ----
    # print("\n===== GATED FUSION — LOSO =====")
    # train2.run_loso(processed, train2.train_dl, **dl_gated)

    # print("\n===== CROSS-ATTENTION — LOSO =====")
    # train2.run_loso(processed, train2.train_dl, **dl_cross)

    # ---- ML models ----
    # print("\n===== SVM — LOSO =====")
    # train2.run_loso(processed, train2.train_ml, classifier_name='svm')

    # print("\n===== XGBoost — LOSO =====")
    # train2.run_loso(processed, train2.train_ml, classifier_name='xgb')

    # print("\n===== Random Forest — LOSO =====")
    # train2.run_loso(processed, train2.train_ml, classifier_name='rf')

    print("\n===== ML ABLATION (XGBoost) =====")
    train2.run_ablation(processed, train2.train_ml, runner=train2.run_loso, classifier_name='xgb')   


if __name__ == "__main__":
    main()