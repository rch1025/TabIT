"""model module."""

import warnings
import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from collections import namedtuple
import random
import scipy.sparse as sp # 희소 행렬을 저장하기 위한 값 
from torch.nn import CosineSimilarity

from model.data_sampler_idx import DataSampler # real_idx를 같이 출력함
from model.data_transformer_semantic import DataTransformer
from model.synthesizers.base import BaseSynthesizer, random_state
import time


"""구간값"""
BinInfo = namedtuple('BinInfo', ['index_start_num', 'index_end_num', 'label', 'bin']) # 변수들의 인덱스 정보를 알기 위함

class Semantic(object):
    def __init__(self, output_info, train_data, bins, batch_size, log_frequency=False, cuda=True):
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'
        self._device = torch.device(device)

        self.bins = bins # 구간 수 정의
        self.batch_size = batch_size
        self.output_info  = output_info
        self.train_data = torch.from_numpy(train_data).to(self._device) # cuda 붙이기
        self.continuous_bin = {} # 각 연속형 변수들의 구간 저장 -> 원핫벡터 idx
        self.continuous_label = {} # 각 연속형 변수들의 구간 저장 -> 원핫벡터 idx
        self.continuous_idx_list = [] # 연속형 변수 col index
        self.continuous_num_list = [] # 원핫 인코딩 데이터에서의 연속형 변수들 위치
        self.semantic_change_data = torch.empty((self.train_data.shape[0], 0)).to(self._device)
        self.bin_info_list = []
        
        """연속형 변수 condtion을 만들기"""
        def is_continuous_column(column_info):
            return (column_info[0].activation_fn != 'softmax')
        n_continuous_columns = sum([1 for column_info in output_info if is_continuous_column(column_info)])
        

        """real 데이터의 구간을 미리 선정해놓기"""
        num = 0
        c_num = 0
        for idx, column_info in enumerate(self.output_info):
            if is_continuous_column(column_info):
                self.continuous_idx_list.append(idx)
                span_info = column_info[0]
                mask = ~torch.isnan(self.train_data[:,c_num])
                data = torch.masked_select(self.train_data[:,c_num], mask).to(self._device)
                bin = torch.linspace(data.min(), data.max(), self.bins).to(self._device) # 구간 만들기 # cuda 붙이기
                # bin,_ = torch.sort(torch.cat((bin, torch.tensor([-99999999999, 99999999999]).to(self._device)), dim=0)) # 추가 구간 설정
                label = torch.bucketize(data, bin) # 구간 label 만들기
                one_hot = torch.zeros((label.size(0), self.bins)).to(self._device)
                real_bin_onehot = one_hot.scatter_(1, label.unsqueeze(1), 1)
                self.semantic_change_data = torch.cat((self.semantic_change_data, real_bin_onehot), dim=1)
                self.continuous_num_list.append(c_num)
                self.bin_info_list.append(BinInfo(index_start_num=num, index_end_num=num+self.bins, label = None, bin=bin)) # 연속형 변수의 index_end_num=0
                num+=self.bins
                c_num+=1
            else:
                # 범주형 변수의 경우, 해당 차원을 num에 더해주기
                span_info = column_info[0]
                num += span_info.dim
                c_num += span_info.dim

        self._continuous_column_cond_st = torch.zeros(n_continuous_columns, dtype=torch.int32)
        self._continuous_column_n_category = torch.zeros(n_continuous_columns, dtype=torch.int32)
        self._continuous_column_category_prob = torch.zeros((n_continuous_columns, self.bins))
        self._n_continuous_columns = n_continuous_columns
        self._n_categories = n_continuous_columns * self.bins
        

    def sampling_real_label(self, real_idx):
        return self.semantic_change_data[real_idx].float()
    
    
    """절반 마스킹하기"""
    def sample_continuous_condvec(self, real_idx):
        if self._n_continuous_columns == 0:
            return None
        
        cont_mask = torch.zeros((self.batch_size, self._n_continuous_columns))
        cont_mask[torch.arange(self.batch_size), :] = 1
        # 연속형 변수의 condition에 마스킹을 줄 인덱스 설정
        
        # n_samples 만큼 반복해서 뽑기
        result_list = [torch.randperm(self._n_continuous_columns-1)[:self._n_continuous_columns//3] for _ in range(self.batch_size)]
        indices = torch.stack(result_list)
        
        cont_mask.scatter_(1, indices, 0)
        masking_vector = cont_mask.clone()
        masking_vector = masking_vector.unsqueeze(-1)
        masking_vector = masking_vector.repeat(1, 1, self.bins)
        masking_vector = masking_vector.view(self.batch_size, -1)

        cont_con_origin = self.semantic_change_data[real_idx,:]
        masking_vector = masking_vector.to(self._device)
        cont_con = masking_vector*cont_con_origin # 몇 개의 조건을 마스킹 함
        
        return cont_con, cont_mask.to(self._device), cont_con_origin


    def calculate_fake_label(self, fake):
        self.fake_continuous_label = torch.empty((fake.shape[0], 0)).to(self._device)
        
        """mini-batch 만큼의 fake 데이터 구간화"""
        for index_, idx in enumerate(self.continuous_idx_list):
            bin = self.bin_info_list[index_].bin
            num = self.continuous_num_list[index_]
            fake_label = torch.bucketize(fake[:,num].contiguous(), bin)
            min_label = 0
            max_label = len(bin)
            result = torch.where(fake_label < max_label, fake_label, max_label-1) # max 보다 크면 최대 라벨을 지정
            result = torch.where(result > min_label, result, min_label)
            one_hot = torch.zeros((result.size(0), self.bins)).to(self._device)
            fake_bin_onehot = one_hot.scatter_(1, result.unsqueeze(1), 1)
            self.fake_continuous_label = torch.cat((self.fake_continuous_label, fake_bin_onehot), dim=1)
        
        return self.fake_continuous_label




"""Discriminator for the model."""
class Discriminator(Module):
    def __init__(self, input_dim, info_dim, discriminator_dim):
        super(Discriminator, self).__init__()
        dim = input_dim
        info_dim = info_dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2, inplace=True)] # , Dropout(0.1)
            dim = item
        self.seq = Sequential(*seq)
        self.fc_info = Sequential(Linear(item, info_dim), LeakyReLU(0.2))
        self.fc_final = Sequential(Linear(info_dim, 1), LeakyReLU(0.2)) # 최종 값은 1로 나오게 하기 

        # seq += [Linear(dim, 1)]
        # self.seq_info = Sequential(*layers[:info])

    def calc_gradient_penalty(self, real_data, fake_data, device='cuda', lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0), 1, 1, device=device)
        alpha = alpha.repeat(1, 1, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates, _ = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        seq_output = self.seq(input_)
        seq_info = self.fc_info(seq_output)
        return self.fc_final(seq_info), seq_info
    


"""Residual layer for the model."""
class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


"""fake data 생성을 위한 생성자"""
## 연속형은 연속형끼리, 범주형은 범주형끼리
class Generator(Module):
    """Generator for the model."""

    def __init__(self, embedding_dim, generator_dim, data_dim, semantic):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        self.semantic = semantic
        self.seq = Sequential(*seq)
        # 연속형 변수의 구간을 생성하는 값을 따로 만든 뒤에 residual로 연결
        # seq.append(Linear(dim, data_dim))
        self.cont_cond = Linear(dim, self.semantic._n_categories) # 연속형 변수의 연산
        self.cont_cond2 = Linear(self.semantic._n_categories, self.semantic._n_continuous_columns) # 연속형 변수의 마지막 연산
        # self.conv1d = nn.Conv1d(in_channels = self.semantic._n_categories, out_channels = self.semantic._n_continuous_columns, kernel_size = 3, stride=1, padding=1)
        self.bn = BatchNorm1d(self.semantic._n_categories)
        
        self.cate_final = Linear(dim, data_dim) # 범주형 변수의 연산
        self.cate_final2 = Linear(data_dim, data_dim-self.semantic._n_continuous_columns) # 범주형 변수의 원핫인코딩 차원만 나오게
        self.bn2 = BatchNorm1d(data_dim)
        self.relu = LeakyReLU(0.2)

    ## cont 정보를 따로 뺀 다음에 다시 넣어줌
    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        # 연속형은 따로 연산
        cont_data = self.cont_cond(data) # 연속형 변수의 원핫인코딩 값 출력
        cont_data = self.bn(cont_data)
        cont_data = self.relu(cont_data)
        bcont_data = self.cont_cond2(cont_data)

        cate_data = self.cate_final(data)
        bcate_data = self.bn2(cate_data)
        bcate_data = self.relu(bcate_data)
        bcate_data = self.cate_final2(bcate_data)
        return torch.cat([bcate_data, bcont_data], dim=1), cont_data


class model(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the model project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original model implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), info_dim=128,
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=100, bins = 5, cuda=True, private_bool=True):

        assert batch_size % 2 == 0

        self.bins = bins
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self.info_dim = info_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs

        # clip_coeff and sigma are the hyper-parameters for injecting noise in gradients
        self.private = private_bool
        self.clip_coeff = 1
        self.sigma = 1.02
        self.micro_batch_size = batch_size

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    """_apply_activate 함수"""
    def _apply_activate(self, data):
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh': # sigmoid로 바꾸기
                    ed = st + span_info.dim
                    data_t.append(torch.sigmoid(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013
        return (loss * m).sum() / data.size()[0]


    """연속형 변수의 condition loss"""
    def _cont_cond_loss(self, cont_data, cont_con_origin, cont_mask):
        data_size = cont_data.size()[0]
        mask_arr_list = []
        mask_arr_list2 = [] # 조건 에측 리스트
        loss = []
        st_c = 0

        ## 각 확률값과 조건 간의 크로스 엔트로피 계산
        for index_ in range(len(self.semantic.continuous_num_list)):
            mask_arr = torch.zeros_like(cont_mask)
            mask_arr[:, index_] = cont_mask[:, index_]
            mask_arr_list.append(mask_arr)

            mask_arr2 = torch.zeros_like(cont_mask)
            mask_arr2[:, index_] = (1-cont_mask)[:, index_] # 1-cont_mask로 조건이 없었던 변수 지명
            mask_arr_list2.append(mask_arr2)     

            ed_c = st_c + self.semantic.bins
            tmp = functional.cross_entropy(
                cont_data[:, st_c:ed_c],
                torch.argmax(cont_con_origin[:, st_c:ed_c], dim=1),
                reduction='none')
            loss.append(tmp)
            st_c = ed_c
        loss = torch.stack(loss, dim=1) 

        # mask_cond와 연산하면서 각 조건들에 대한 loss 계산
        continuous_cond_loss = []
        continuous_forecasting_loss = []
        for mask1, mask2 in zip(mask_arr_list, mask_arr_list2):
            continuous_cond_loss.append((loss * mask1).sum() / data_size)
            continuous_forecasting_loss.append((loss * mask2).sum() / data_size)
        return torch.mean(torch.stack(continuous_cond_loss)), torch.mean(torch.stack(continuous_forecasting_loss))


    def compute_cross_moment(self, fake_features, real_features):
        # fake_features: tensor of shape (batch_size, num_fake_features)

        # Compute the mean of the fake_features
        mean_fake_features = torch.mean(fake_features, dim=0, keepdim=True)
        mean_real_features = torch.mean(real_features, dim=0, keepdim=True)

        # Compute the centered fake_features
        centered_fake_features = fake_features - mean_fake_features
        centered_real_features = real_features - mean_real_features

        # Compute the cross-moment
        fake_cross_moment = torch.matmul(centered_fake_features.T, centered_fake_features) / fake_features.size(0)
        real_cross_moment = torch.matmul(centered_real_features.T, centered_real_features) / real_features.size(0)

        return torch.norm(fake_cross_moment - real_cross_moment, 2)
    

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the model Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        print('### Transform begin')
        train_data = self._transformer.transform(train_data)
        print('### Transform complete')

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        """Semantic 설정"""
        self.semantic = Semantic(self._transformer.output_info_list, train_data, bins=self.bins, batch_size = self._batch_size)
        semantic_change_data = self.semantic.semantic_change_data
        
        ## 연속형 변수의 조건을 만드는 dict 만들기
        if self._data_sampler._n_categories != 0:
            self.condition_dict = {key:[] for key in range(0, self._data_sampler._n_categories)} # 범주형 변수의 condition 개수만큼 설정
        else:
            self.condition_dict = {0:[]}

        """Generator과 Discriminator"""
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec() + self.semantic._n_categories, # 연속형 변수에 대한 condition 추가
            self._generator_dim,
            data_dim,
            self.semantic).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec() + self.semantic._n_categories,
            self.info_dim,
            self._discriminator_dim,).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        # 입력 노이즈를 만들기 위한 크기 생성
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        steps = 0

        steps_per_epoch = max(len(train_data) // self._batch_size, 1) # iteration을 의미
        for i in range(epochs):
            for id_ in range(steps_per_epoch):

                """판별자 실행"""
                for n in range(self._discriminator_steps):
                    
                    fakez = torch.normal(mean=mean, std=std) # 노이즈 생성
                    condvec = self._data_sampler.sample_condvec(self._batch_size) # 조건 생성 (4개의 output이 나옴)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real, real_idx = self._data_sampler.sample_data(self._batch_size, col, opt)
                        cont_con, cont_mask, cont_con_origin = self.semantic.sample_continuous_condvec(real_idx) # 연속형 변수 조건 뽑기
                        for idx, _ in enumerate(cont_con):
                            if i % 20 ==0:
                                self.condition_dict[0].append(sp.csr_matrix(cont_con[idx].detach().cpu().numpy())) # 넘파이로 바꾼 다음에 넣어주기
                        fakez = torch.cat([fakez,cont_con], dim=1) # 조건과 노이즈 합치기

                    else:
                        c1, m1, col, opt = condvec
                        # 연속형 변수의 condition 정의
                        _, cate_cond_num = np.where(c1 == 1)

                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        real, real_idx = self._data_sampler.sample_data(self._batch_size, col, opt)
                        # real_sem = semantic_change_data[real_idx,:]
                        cont_con, cont_mask, cont_con_origin = self.semantic.sample_continuous_condvec(real_idx) # 연속형 변수 조건 뽑기
                        # dict에 저장 -> device에 올라간 뒤에 저장하기
                        if i % 20 ==0:
                            for idx, cate_num in enumerate(cate_cond_num):
                                self.condition_dict[cate_num].append(sp.csr_matrix(cont_con[idx].detach().cpu().numpy())) # 넘파이로 바꾼 다음에 넣어주기
                    
                        # cont_mask = torch.from_numpy(cont_mask).to(self._device) # cont_mask는 넘파이 어레이임 -> 판별자에서는 사용되지 않음
                        fakez = torch.cat([fakez, c1, cont_con], dim=1) # 조건과 노이즈 합치기
                        c2 = c1

                    """생성자 실행"""
                    fake, _ = self._generator(fakez)
                    fakeact = self._apply_activate(fake) # 각 변수에 맞는 활성화 함수 적용

                    real = torch.from_numpy(real.astype('float32')).to(self._device) # 실제 데이터 텐서로 변환

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1, cont_con], dim=1)
                        real_cat = torch.cat([real, c2, cont_con], dim=1)
                    else:
                        fake_cat = torch.cat([fakeact, cont_con], dim=1)
                        real_cat = torch.cat([real, cont_con], dim=1)

                    # 판별자 입력
                    y_fake, _ = discriminator(fake_cat)
                    y_real, _ = discriminator(real_cat)

                    # following block cliping gradients and add noises.
                    if self.private:
                        
                        clipped_grads = {
                            name: torch.zeros_like(param) for name, param in discriminator.named_parameters()}

                        for k in range(int(y_real.size(0) / self.micro_batch_size)):
                            err_micro = -1*y_real[k * self.micro_batch_size: (k + 1) * self.micro_batch_size].mean(0).view(1)
                            err_micro.backward(retain_graph=True)
                            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.clip_coeff)
                            for name, param in discriminator.named_parameters():
                                clipped_grads[name] += param.grad
                            discriminator.zero_grad()

                        for name, param in discriminator.named_parameters():
                            param.grad = (clipped_grads[name] + torch.FloatTensor(clipped_grads[name].size()).normal_(0, self.sigma * self.clip_coeff).cuda()) / (y_real.size(0) / self.micro_batch_size)

                        steps += 1
                    ## private이 False이면 pen~ 부터 진행
                    # else: 
                    #     y_real = -torch.mean(y_real)
                    #     y_real.backward() 

                    pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device) # gradient penalty 계산
                    
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake)) # loss 생성

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()
                
                """생성자 학습"""
                fakez = torch.normal(mean=mean, std=std) # 노이즈 생성

                condvec = self._data_sampler.sample_condvec(self._batch_size) # 조건 생성 (4개의 output이 나옴)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real, real_idx = self._data_sampler.sample_data(self._batch_size, col, opt)
                    real_sem = semantic_change_data[real_idx,:]
                    cont_con, cont_mask, cont_con_origin = self.semantic.sample_continuous_condvec(real_idx) # 연속형 변수 조건 뽑기
                    fakez = torch.cat([fakez,cont_con], dim=1) # 조건과 노이즈 합치기
                else:
                    c1, m1, col, opt = condvec
                    # 연속형 변수의 condition 정의
                    _, cate_cond_num = np.where(c1 == 1)

                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    real, real_idx = self._data_sampler.sample_data(self._batch_size, col, opt)
                    real_sem = semantic_change_data[real_idx,:]
                    cont_con, cont_mask, cont_con_origin = self.semantic.sample_continuous_condvec(real_idx) # 연속형 변수 조건 뽑기
                    # dict에 저장
                    # for idx, cate_num in enumerate(cate_cond_num):
                    #     self.condition_dict[cate_num].append(cont_con[idx])
                    # cont_mask = torch.from_numpy(cont_mask).to(self._device)
                    fakez = torch.cat([fakez, c1, cont_con], dim=1) # 조건과 노이즈 합치기
                    c2 = c1

                """생성자 실행"""
                fake, fake_cont = self._generator(fakez)
                fakeact = self._apply_activate(fake) # 각 변수에 맞는 활성화 함수 적용

                real = torch.from_numpy(real.astype('float32')).to(self._device) # 실제 데이터 텐서로 변환

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1, cont_con], dim=1)
                    real_cat = torch.cat([real, c2, cont_con], dim=1)
                else:
                    fake_cat = torch.cat([fakeact, cont_con], dim=1)
                    real_cat = torch.cat([real, cont_con], dim=1)

                # 판별자 입력
                y_fake, info_fake = discriminator(fake_cat) # 
                y_real, info_real = discriminator(real_cat) # 

                loss_mean = torch.norm(torch.mean(info_fake, dim=0) - torch.mean(info_real, dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake, dim=0) - torch.std(info_real, dim=0), 1)
                loss_info = loss_mean + loss_std 

                # loss_computition = self.compute_cross_moment(info_fake, info_real)
                # 코사인 유사도 계산
                # similarity = cos_sim(info_fake, info_real)

                # 유사도를 이용한 loss 계산
                # loss_sim = 1 - similarity.mean()

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)
                
                now = time
                continuous_entropy, continuous_forecasting_entropy = self._cont_cond_loss(fake_cont, cont_con_origin, cont_mask) # 조건 부여 loss와 조건 예측 loss
                fake_continuous_label = self.semantic.calculate_fake_label(fakeact)
                # # continuous_loss2 = self.jaccard_loss(cont_con, fake_continuous_label)
                if (id_ == 1):
                    print('# Condition output')
                    print(torch.where(real_sem[0] == 1))
                    print(torch.where(fake_continuous_label[0] == 1))
                    print(cont_mask[0])
                    print()
                    print('# Condition output')
                    print(torch.where(real_sem[1] == 1))
                    print(torch.where(fake_continuous_label[1] == 1))
                    print(cont_mask[1])
                    print()


                optimizerG.zero_grad()
                loss_g = -torch.mean(y_fake) + cross_entropy + continuous_entropy + continuous_forecasting_entropy
                loss_g.backward(retain_graph = True)
                # loss_info.backward()
                # loss_sim.backward()
                optimizerG.step()

            print(f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},', f'Loss D: {loss_d.detach().cpu(): .4f}', flush=True)
            if condvec is None:
                pass
            else:
                print('#### cross_entropy :', (cross_entropy).detach().cpu())
            print('## continuous_entropy :', (continuous_entropy).detach().cpu())
            print('## continuous_forecasting_entropy :', (continuous_forecasting_entropy).detach().cpu())
            # print('## loss_info   :', (loss_info).detach().cpu())
            print()

    @random_state
    def sample(self, n, condition_column=None, condition_value=None, original = False):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        prev_value = None

        # for key in sorted(self.condition_dict.keys()):
        #     if not self.condition_dict[key]:
        #         if prev_value is not None:
        #             self.condition_dict[key] = prev_value
        #     else:
        #         prev_value = self.condition_dict[key]

        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size) ############## cont_cond 추가

            if condvec is None:
                continuous_condition = [random.choice(self.condition_dict[0]).toarray() for _ in range(self._batch_size)]
                continuous_condition = np.vstack(continuous_condition)
                continuous_condition = torch.from_numpy(continuous_condition).to(self._device)
                fakez = torch.cat([fakez, continuous_condition], dim=1)
            else:
                c1 = condvec
                ## 연속형 변수의 condition 뽑아내기
                _, cond_g_num = np.where(c1)
                continuous_condition = [random.choice(self.condition_dict[key]).toarray() for key in cond_g_num]
                continuous_condition = np.vstack(continuous_condition)
                continuous_condition = torch.from_numpy(continuous_condition).to(self._device)
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1, continuous_condition], dim=1)

            fake, _ = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        print('Inverse 전의 data :', data.shape)
        data = np.asarray(data[:n])
        print('### Inverse start')
        
        if original:
            return data, self._transformer.inverse_transform(data) # 병렬처리로 해줘야 함
        else:
            return self._transformer.inverse_transform(data) # 병렬처리로 해줘야 함

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
