import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import History, Callback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K


# Definir constantes
dias_previstos = 10
penalizacao = 0.5
epocas = 100
steps = 15
total_treino = .75
modelo_path = 'modelo.hdf5'
salvar_modelo = False

# Definir callback para penalização
class PenalizationCallback(Callback):
    def __init__(self, penalty_factor):
        super(PenalizationCallback, self).__init__()
        self.penalty_factor = penalty_factor

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_loss = logs.get('val_loss')
        if val_loss is not None:
            prev_val_loss = getattr(self, 'prev_val_loss', None)
            if prev_val_loss is not None and val_loss > prev_val_loss:
                # Penalizar o modelo
                current_lr = float(K.get_value(self.model.optimizer.lr))
                new_lr = current_lr * self.penalty_factor
                K.set_value(self.model.optimizer.lr, new_lr)
                print(f'Penalização aplicada. Nova taxa de aprendizado: {new_lr}')
        setattr(self, 'prev_val_loss', val_loss)

# Carregar o dataframe
df = pd.read_csv('dados/base.csv', sep=';')
df['data_dia'] = pd.to_datetime(df['data_dia'], format='%d/%m/%Y')

# Filtrar por sigla do indicador
df_acao = df[df['indicador'] == 'ind']

# Selecionar os campos
df_acao_fec = df_acao[['data_dia', 'preco_fechamento']]

# Definir a coluna data_dia como index
df_acao_fec = df_acao_fec.set_index(pd.DatetimeIndex(df_acao_fec['data_dia'].values))

# Retirar coluna data_dia
df_acao_fec.drop('data_dia', axis=1, inplace=True)

# Verificar a quantidade de linhas
qtd_linhas = len(df_acao_fec)

qtd_linhas_treino = round(total_treino * qtd_linhas)
qtd_linhas_teste = qtd_linhas - qtd_linhas_treino

info = (
    f"Linhas para treino: 0:{qtd_linhas_treino}"
    f", Linhas para teste: {qtd_linhas_treino}:{qtd_linhas_treino + qtd_linhas_teste}"
)

# Normalizando os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_acao_fec)

# Separar em treino e teste
train = df_scaled[:qtd_linhas_treino]
test = df_scaled[qtd_linhas_treino: qtd_linhas_treino + qtd_linhas_teste]

print(info)
print("Total de linhas para treino: {0} e teste: {1}".format(len(train), len(test)))

# Função para converter uma matriz de valores em uma matriz df
def create_df(df, steps=1):
    dataX, dataY = [], []
    for i in range(len(df)-steps-1):
        a = df[i:(i+steps), 0]
        dataX.append(a)
        dataY.append(df[i + steps, 0])
    return np.array(dataX), np.array(dataY)

# Gerando dados de treino e teste
X_train, Y_train = create_df(train, steps)
X_test, Y_test = create_df(test, steps)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Gerando os dados que o modelo espera
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

if os.path.exists(modelo_path):
    modelo = load_model(modelo_path)
else:
    # Criar a rede neural
    modelo = Sequential()
    modelo.add(LSTM(35, return_sequences=True, input_shape=(steps, 1)))
    modelo.add(Dropout(0.2))
    modelo.add(LSTM(35, return_sequences=True))
    modelo.add(Dropout(0.2))
    modelo.add(LSTM(35))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(1))

    modelo.compile(optimizer='adam', loss='mse')

    # Incluir penalização
    penalization_callback = PenalizationCallback(penalty_factor=penalizacao)

    # Treinamento do modelo
    history = modelo.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epocas, batch_size=15,
        verbose=2,
        callbacks=[penalization_callback]
    )

    # Salvar o modelo
    modelo.save(modelo_path)

    # Plotar curvas de treinamento
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

# Fazendo a previsão
prev = modelo.predict(X_test)
prev = scaler.inverse_transform(prev)

# Previsão para os próximos 10 dias
lenght_test = len(test)
days_input_steps = lenght_test - steps
input_steps = test[days_input_steps:]
input_steps = np.array(input_steps).reshape(1, -1)

list_output_steps = list(input_steps)
list_output_steps = list_output_steps[0].tolist()

pred_output = []
i = 1

while(i <= dias_previstos):
    if len(list_output_steps) > steps:
        input_steps = np.array(list_output_steps[1:])
        input_steps = input_steps.reshape(1, -1)
        input_steps = input_steps.reshape((1, steps, 1))
        pred = modelo.predict(input_steps, verbose=0)
        list_output_steps.extend(pred[0].tolist())
        list_output_steps = list_output_steps[1:]
        pred_output.extend(pred.tolist())
        i = i + 1
    else:
        input_steps = input_steps.reshape((1, steps, 1))
        pred = modelo.predict(input_steps, verbose=0)
        list_output_steps.extend(pred[0].tolist())
        pred_output.extend(pred.tolist())
        i = i + 1

# Transformar a saída
prev = scaler.inverse_transform(pred_output)
prev = np.array(prev).reshape(1, -1)
list_output_prev = list(prev)
list_output_prev = prev[0].tolist()

# Data da previsão
dates = pd.to_datetime(df_acao['data_dia'])
predict_dates = pd.date_range(list(dates)[-1] + pd.DateOffset(1), periods=10, freq='b').tolist()

# Criar dataframe com as previsões
forecast_dates = []
for i in predict_dates:
    forecast_dates.append(i.date())

df_forecast = pd.DataFrame({'data_dia': np.array(forecast_dates), 'preco_fechamento': list_output_prev})
df_forecast['data_dia'] = pd.to_datetime(df_forecast['data_dia'])

df_forecast = df_forecast.set_index(pd.DatetimeIndex(df_forecast['data_dia'].values))
df_forecast.drop('data_dia', axis=1, inplace=True)

df_acao = df[(df['indicador'] == 'ind') & (df['data_dia'] > '2023-06-01')]
df_acao_fec = df_acao[['data_dia', 'preco_fechamento']]
df_acao_fec = df_acao_fec.set_index(pd.DatetimeIndex(df_acao_fec['data_dia'].values))
df_acao_fec.drop('data_dia', axis=1, inplace=True)

print("############################################################")
print(df_acao_fec['preco_fechamento'])
print("############################################################")
print(df_forecast['preco_fechamento'])

# Plotar o gráfico
plt.figure(figsize=(16, 8))
plt.plot(df_acao_fec['preco_fechamento'])
plt.plot(df_forecast['preco_fechamento'])
plt.legend(['Preço de Fechamento', 'Preço Previsto'])
plt.show()

# Salvar o modelo em um arquivo
modelo.save('modelo.hdf5')