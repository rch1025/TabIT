"""
Generative model training algorithm based on the CTGAN and CTABGAN+ Synthesiser

"""
import pandas as pd
import time
from model.data import read_csv, read_tsv, write_tsv
from model.synthesizers.model import model

import warnings

warnings.filterwarnings("ignore")

class Tabular_GAN():

    def __init__(self,
                 raw_csv_path = "",
                 categorical_columns = "",
                 embedding_dim = 128,
                 generator_dim = '256,256',
                 discriminator_dim = '256,256',
                 generator_lr = 2e-4,
                 generator_decay = 1e-6,
                 discriminator_lr = 2e-4,
                 discriminator_decay = 0,
                 batch_size = 500,
                 bins = 10,
                 epochs = 100,
                 private_bool = True
                 ):

        self.__name__ = 'Tabular_GAN'      
        self.generator_dim = [int(x) for x in generator_dim.split(',')]
        self.discriminator_dim = [int(x) for x in discriminator_dim.split(',')]
        self.synthesizer = model(embedding_dim=embedding_dim, generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim, generator_lr=generator_lr,
            generator_decay=generator_decay, discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay, batch_size=batch_size, bins=bins, # Semantic을 위해 args.bins 설정
            epochs=epochs, private_bool=True)
        self.raw_csv_path = raw_csv_path
        self.categorical_columns = categorical_columns
                
    def fit(self):
        start_time = time.time()
        print('Data Load')
        data, cate_col = read_csv(csv_filename = self.raw_csv_path, meta_filename=False, header=True, discrete = self.categorical_columns)

        print('Data reorder')
        self.data = data.reindex(columns=cate_col + list(data.columns.difference(cate_col)))
        
        print('Fitting')
        self.synthesizer.fit(self.data, cate_col)

        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, original=False):
        if original:
            origin_data, sample = self.synthesizer.sample(len(self.data), original=True)
            return origin_data, sample
        else:
            sample = self.synthesizer.sample(len(self.data))
            return sample