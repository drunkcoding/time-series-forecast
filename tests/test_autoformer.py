import sys
from forecast.data.dataset import Dataset_SNDLib_Multi
from forecast.model.autoformer import Autoformer
from forecast.utils.cmdparser import HfArgumentParser, ModelArguments, DataArguments

parser = HfArgumentParser((ModelArguments, DataArguments))
model_args, data_args = parser.parse_json_file(sys.argv[1])

dataset = Dataset_SNDLib_Multi(
    data_path=data_args.data_path,
    
)