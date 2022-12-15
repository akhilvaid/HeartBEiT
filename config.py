# Config class

class Config:
    # Paths
    dir_ecg_plots = ''
    dir_dataframes = ''
    dir_ssd_plots = ''
    dir_ecg_plots_read_from = ''  # Can be /ssd when required
    dir_ecg_plots_read_from_ptb = ''

    file_best_perf = 'Results/BestPerformance.pickle'

    df_patient_list = 'Data/1_PatientList.pickle'
    df_finetuning = {
        'LVEF': 'Data/LVEF.pickle',
        'HCM': 'Data/HCM.pickle',
        'STEMI': 'Data/STEMI-PTBXL.pickle',
    }

    # Finetuning
    ce_loss = True
    checkpoint_path = 'BeitCheckpoints/checkpoint-299.pth'
    imagenet_checkpoint_path = 'BeitCheckpoints/beit_base_patch16_224_pt22k_ft1k.pth'
    image_size = 224
    finetuning_percentage_iter = [.01, .1, .25, .5, 1]  # Use this amount of TRAINING data for finetuning
    batch_size = 32
    ft_epochs = 20
    models = {
        # In use
        'vit_imagenet': {'internal_identifier': 'vit_imagenet', 'weights': imagenet_checkpoint_path, 'name': 'ViT-B/16'},
        'efficientnet_b4': {'internal_identifier': 'efficientnet_b4', 'weights': 'EfficientNet_B4_Weights', 'name': 'EfficientNet-B4'},
        'resnet152': {'internal_identifier': 'resnet152', 'weights': 'ResNet152_Weights', 'name': 'ResNet152'},
        'vit': {'internal_identifier': 'vit', 'weights': None, 'name': 'HeartBEiT'},
    }

    # General
    random_state = 42
    debug = False
