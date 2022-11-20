import os
import dataloader
import data_processor
import dataprocessor224
import dataprocessor128

anns_path = os.path.abspath(os.path.join('config', 'annotations', 'annotations.json'))
tvt_path = os.path.abspath(os.path.join( 'config', 'annotations', 'train_val_test_distribution_file.json'))

def __main__():
    data = dataloader.DataLoader(anns_path, tvt_path)
    dataprocessor224.image_transformer(data)
    return

if __name__ == "__main__":
    __main__()
