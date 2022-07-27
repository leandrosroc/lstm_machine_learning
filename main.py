#importar as bibliotecas
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#importando arquivo
df = pd.read_csv('dados/base.csv', sep=';')
df['data_dia'] = pd.to_datetime(df['data_dia'], format='%d/%m/%Y')

#filtrar por sigla do indicador
df_acao= df [df['indicador'] == 'ind']

#selecionar os campos
df_acao_fec = df_acao[['data_dia', 'preco_fechamento']]

#definir a coluna data_dia como index
df_acao_fec = df_acao_fec.set_index(pd.DatetimeIndex(df_acao_fec['data_dia'].values))
print(df_acao_fec)

#retirar coluna data_dia
df_acao_fec.drop('data_dia', axis=1, inplace=True)

#plotar informação
plt.figure(figsize=(16,8))
plt.title('Preço de fechamento')
plt.plot(df_acao_fec['preco_fechamento'])
plt.xlabel('data')
plt.show()

#verificar a quantidade de linhas
qtd_linhas = len(df_acao_fec)

qtd_linhas_treino = round(.70 * qtd_linhas)

qtd_linhas_teste = qtd_linhas - qtd_linhas_treino

info = (
    f"Linhas para treino: 0:{qtd_linhas_treino}"
    f", Linhas para teste: {qtd_linhas_treino}:{qtd_linhas_treino + qtd_linhas_teste}"

)

#normalizando os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_acao_fec)

#separa em treino e teste
train = df_scaled[:qtd_linhas_treino]
test = df_scaled[qtd_linhas_treino: qtd_linhas_treino+qtd_linhas_teste]

print(info)
print("Total de linhas para treino: {0} e teste: {1}".format(len(train), len(test)))

#função para converter uma matriz de valores em uma matriz df
def create_df(df, steps=1):
	dataX, dataY = [], []
	for i in range(len(df)-steps-1):
		a = df[i:(i+steps), 0]
		dataX.append(a)
		dataY.append(df[i + steps, 0])
	return np.array(dataX), np.array(dataY)

#gerando dados de treino e teste
steps = 15
X_train, Y_train = create_df(train, steps)
X_test, Y_test = create_df(test, steps)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#gerando os dados que o modelo espera
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#criando a rede neural
modelo = Sequential()
modelo.add(LSTM(35, return_sequences=True, input_shape=(steps, 1)))
modelo.add(LSTM(35, return_sequences=True))
modelo.add(LSTM(35))
modelo.add(Dropout(0.2))
modelo.add(Dense(1))

modelo.compile(optimizer='adam', loss='mse')
modelo.summary()

#treinamento do modelo
validation = modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=15, verbose=2)

plt.plot(validation.history['loss'], label='Training loss')
plt.plot(validation.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

#fazendo a previsão
prev= modelo.predict(X_test)
prev = scaler.inverse_transform(prev)

#previsão para os proximos 10 dias
lenght_test = len(test)

#pegar os ultimos dias que são o tamanho do step
days_input_steps = lenght_test - steps

#transforma em array
input_steps = test[days_input_steps:]
input_steps = np.array(input_steps).reshape(1,-1)
input_steps

#tranformar em lista
list_output_steps = list(input_steps)
list_output_steps = list_output_steps[0].tolist()
list_output_steps

#loop para prever os próximos 10 dias
pred_output=[]
i=1
n_future=10

while(i<n_future):
    
    if(len(list_output_steps) > steps):
        
        input_steps = np.array(list_output_steps[1:])
        print("Dia: {}. Valores de entrada -> {}".format(i,input_steps))
        input_steps = input_steps.reshape(1,-1)
        input_steps = input_steps.reshape((1, steps, 1))
        #print(input_steps)
        pred = modelo.predict(input_steps, verbose=0)
        print("Dia: {}. Valor previsto -> {}".format(i,pred))
        list_output_steps.extend(pred[0].tolist())
        list_output_steps=list_output_steps[1:]
        #print(list_output_steps)
        pred_output.extend(pred.tolist())
        i=i+1
    else:       
        input_steps = input_steps.reshape((1, steps,1))
        pred = modelo.predict(input_steps, verbose=0)
        print(pred[0])
        list_output_steps.extend(pred[0].tolist())
        print(len(list_output_steps))
        pred_output.extend(pred.tolist())
        i=i+1

#tranformar a saída
prev = scaler.inverse_transform(pred_output)
prev = np.array(prev).reshape(1,-1)
list_output_prev = list(prev)
list_output_prev = prev[0].tolist()

#data da previsão
dates = pd.to_datetime(df_acao['data_dia'])
predict_dates = pd.date_range(list(dates)[-1] + pd.DateOffset(1), periods=10, freq='b').tolist()

#criando dataframe com as previsões
forecast_dates = []
for i in predict_dates:
  forecast_dates.append(i.date())

df_forecast = pd.DataFrame({'data_dia': np.array(forecast_dates), 'preco_fechamento': list_output_prev})
df_forecast['data_dia'] = pd.to_datetime(df_forecast['data_dia'])

df_forecast=df_forecast.set_index(pd.DatetimeIndex(df_forecast['data_dia'].values))
df_forecast.drop('data_dia',axis=1,inplace=True)

df_acao= df[ (df['indicador'] == 'ind') & (df['data_dia'] > '2021-05-01')]
df_acao_fec = df_acao[['data_dia', 'preco_fechamento']]
df_acao_fec=df_acao_fec.set_index(pd.DatetimeIndex(df_acao_fec['data_dia'].values))
df_acao_fec.drop('data_dia',axis=1,inplace=True)

#plotar o grafico
plt.figure(figsize=(16,8))
plt.plot(df_acao_fec['preco_fechamento'])
plt.plot(df_forecast['preco_fechamento'])
plt.legend(['Preço de Fechamento', 'Preço Previsto'])
plt.show()

#salvar o modelo em um arquivo
modelo.save('modelo.hdf5')