"""DataTransformer module."""
from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import OneHotEncoder
from scipy.interpolate import interp1d


SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
## ColumnTransformInfo 튜풀 저장: 변수별로 특징을 담고있는 튜플임
# 변환함 (많은 인자들이 추가됨)
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
    'column_name', 'column_type', 'transform', 'output_info', 'transformed_data', 'cdf_min', 'output_dimensions', 'original_data'
    ]
)


"""ECDF 적용"""
class DataTransformer(object):

    def __init__(self):
        self.output_info_list = []
        self.output_dimensions = 0
        self._column_transform_info_list = []
        self.dataframe = True
        self.transform_dict = {}
        self.cdf_min_dict = {}

    def cdf_encoding(self, column_name, data): 
        value_counts = data[column_name].value_counts()
        values = value_counts.index.sort_values()
        cumulative_sum = value_counts.loc[values].cumsum()
        cdf = cumulative_sum / cumulative_sum.max()
        normalized_cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        return normalized_cdf.loc[data[column_name]].values.tolist(), cdf.min() # 역변환을 위해 최소값도 같이 반환
    
    #### CDF Normalize가 들어간 연속형 변수 인코더
    def _fit_continuous(self, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        transformed_data, cdf_min = self.cdf_encoding(column_name, data) # 변환 부분

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=self.cdf_encoding,
            output_info=[SpanInfo(1, 'tanh')], transformed_data = transformed_data, cdf_min = cdf_min,
            output_dimensions=1, original_data = data) # original_data는 데이터프레임 형식의 값


    ## 이산형 변수는 그대로 둠
    def _fit_discrete(self, data):
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')], transformed_data = None, cdf_min = None,
            output_dimensions=num_categories, original_data=None)


    ######## fit()
    def fit(self, raw_data, discrete_columns=()):
        self._column_raw_dtypes = raw_data.infer_objects().dtypes

        # raw_data가 데이터프레임인지 확인하는 것
        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        for idx, column_name in enumerate(raw_data.columns):
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]]) # categorical transform
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]]) # continuous transform

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)



    ######## _transform_continuous() : 개별 변수별 반환
    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        transformed = column_transform_info.transformed_data # CDF Normalize 된 데이터
        
        # 변환 부분
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed # 변환 부분
        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()



    ######## _synchronous_transform() : 하나씩 변환, 위의 _transform_continous 함수 사용
    def _synchronous_transform(self, raw_data, column_transform_info_list):
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data)) # continuous 변환 부분
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))
        return column_data_list
    
    

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        column_data_list = self._synchronous_transform(
            raw_data,
            self._column_transform_info_list
        )
        transformed_output = np.concatenate(column_data_list, axis=1).astype(float)

        # self.transform_dict 저장 -> normalized cdf가 저장되는 것
        st=0
        for idx, column_info in enumerate(self._column_transform_info_list):
            if column_info.column_type == 'continuous':
                dim = column_info.output_dimensions
                self.transform_dict[column_info.column_name] = transformed_output[:, st : st+dim]
                self.cdf_min_dict[column_info.column_name] = column_info.cdf_min # 변수명 정보와 함께 cdf의 최소값 저장
                st += dim
            else:
                st += column_info.output_dimensions

        return transformed_output



    """역변환 부분을 오리지널 cdf 출력값으로 바꾼 것"""
    def inverse_cdf(self, interp_func, idx, select_cdf, min_, max_):
        return interp_func(select_cdf)

    def Interpolative_cdf_decoding(self, column_data, min_, max_, sorted_data, sorted_transformed):
        interp_func = interp1d(sorted_transformed, np.array(sorted_data).reshape(-1))
        return [self.inverse_cdf(interp_func, idx, i, min_, max_) for idx, i in enumerate(column_data)]

    ### 연속형 변수 처리
    def _inverse_transform_continuous(self, column_transform_info, column_data):
        original_data = column_transform_info.original_data
        sorted_data = sorted(np.array(original_data).reshape(-1, 1))
        column_name = column_transform_info.column_name
        cdf_min = self.cdf_min_dict[column_name] # cdf의 최소값 불러오기

        transformed = self.transform_dict[column_name]
        transformed = transformed*(1-cdf_min) + cdf_min
        column_data = column_data*(1-cdf_min) + cdf_min # 모델 출력값도 Normalize 해제하고 CDF 풀기
        
        sorted_transformed = np.sort(transformed.flatten())
        min_ = np.min(sorted_transformed)
        max_ = np.max(sorted_transformed)
        
        return self.Interpolative_cdf_decoding(column_data, min_, max_, sorted_data, sorted_transformed)
    

    ### 이산형 변수 처리
    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]


    ### _inverse_transform_continuous와 _inverse_transform_discrete를 사용해서 변환
    def inverse_transform(self, data):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(column_transform_info, column_data) # 역변환 부분
            else:
                recovered_column_data = self._inverse_transform_discrete(column_transform_info, column_data) # 역변환 부분
            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim
        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names).astype(self._column_raw_dtypes))
                          
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()
        print('Inverse_End')
        return recovered_data
    
    """병렬처리 역변환 함수"""
    def _parallel_inverse_transform(self, raw_data):
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = raw_data[:, st:st + dim]
            process = None
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._inverse_transform_continuous)(column_transform_info, column_data) # 역변환 부분
            else:
                process = delayed(self._inverse_transform_discrete)(column_transform_info, column_data) # 역변환 부분
            recovered_column_data_list.append(process)
            column_names.append(column_transform_info.column_name)
            st += dim
        recovered_column_data_list = Parallel(n_jobs=10)(recovered_column_data_list) 
        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()
        return recovered_data
    
    
    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }
