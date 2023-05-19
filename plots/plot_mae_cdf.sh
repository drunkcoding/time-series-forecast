python plots/plot_mae_cdf.py \
    --input data/GEANT-OD_pair.csv \
    --models "('naive','u'):('arima','u'):('prophet','u'):('kalman','u'):('rnn','m'):('deepar','m'):('tcn','m'):('tft','m'):('tf','m'):('convlstm','m')"

python plots/plot_mae_cdf.py \
    --input data/CERNET-OD_pair.csv \
    --models "('naive','u'):('arima','u'):('prophet','u'):('kalman','u'):('rnn','m'):('deepar','m'):('tcn','m'):('tft','m'):('tf','m'):('convlstm','m')"

python plots/plot_mae_cdf.py \
    --input data/Abilene-OD_pair.csv \
    --models "('naive','u'):('arima','u'):('prophet','u'):('kalman','u'):('rnn','m'):('deepar','m'):('tcn','m'):('tft','m'):('tf','m'):('convlstm','m')"