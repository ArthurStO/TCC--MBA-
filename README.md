# Lista de pacotes necessários
pacotes <- c("rattle", "rnn", "ggplot2", "dplyr", "keras", "quantmod", "tidyquant", "reticulate")

# Verifica se os pacotes estão instalados; se não, instala e carrega os pacotes
if (sum(as.numeric(!pacotes %in% installed.packages())) != 0) {
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for (i in 1:length(instalador)) {
    install.packages(instalador, dependencies = TRUE)  # Instala pacotes que não estão instalados
    break
  }
  sapply(pacotes, require, character.only = TRUE)  # Carrega todos os pacotes
} else {
  sapply(pacotes, require, character.only = TRUE)  # Carrega todos os pacotes
}

library(reticulate)
use_condaenv("r-reticulate", required = TRUE)

# Carregamento das bibliotecas necessárias
library(rnn)
library(dplyr)
library(keras)
library(quantmod)
library(tidyquant)
library(reticulate)

# Baixa dados históricos da Petrobras e armazena no dataset 'data'
data <- tq_get("PETR4.SA", from = "2023-09-30", to = "2024-09-30")

# Visualiza as primeiras linhas dos dados
head(data)

# Verifica a ordem das datas no dataset
head(data$date)  # Mostra as primeiras datas
tail(data$date)  # Mostra as últimas datas

# Ordena os dados pela data em ordem crescente
data <- data[order(data$date, decreasing = FALSE), ]

# Prepara os dados para análise
fechamento <- data$close  # Extrai a coluna de preços de fechamento
fechamento_anterior <- lead(fechamento, n = 1L)  # Cria uma coluna com os preços de fechamento do dia anterior

# Cria um novo dataframe para análise
data_analise <- data.frame(fechamento)
data_analise$fechamento_anterior <- fechamento_anterior


# Exibe um resumo estatístico dos dados
summary(data_analise)
# Remove linhas com valores NA
data_analise <- na.omit(data_analise)


# Separa as colunas em variáveis independentes (x) e dependentes (y)
x <- data_analise[, 2]
y <- data_analise[, 1]
close_anterior <- data_analise[, 2]
close <- data_analise[, 1]


# Converte as variáveis em matrizes
X <- matrix(x, nrow = 31)
Y <- matrix(y, nrow = 31)


# Gráfico de Preços ao Longo do Tempo 
ggplot(data, aes(x = date, y = close)) +
  geom_line(color = "blue") +
  labs(title = "Variação dos Preços de Fechamento ao Longo do Tempo",
       x = "Data",
       y = "Preço de Fechamento") +
  theme_minimal()


# Normaliza os dados


Yscaled <- (Y - min(Y)) / (max(Y) - min(Y))
Xscaled <- (X - min(X)) / (max(X) - min(X))
# Guarda os valores mínimos e máximos para desnormalização futura
min_Y <- min(Y)
max_Y <- max(Y)
min_X <- min(X)
max_X <- max(X)
# Substitui os dados originais pelos dados normalizados
Y <- Yscaled
X <- Xscaled


# Gráfico dos dados originais e normalizados
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(close, main = "Dados Originais - Y", col = "blue", pch = 19, xaxt = 'n', xlab = "", ylab = "")
plot(Yscaled, main = "Dados Normalizados - Y", col = "red", pch = 19, xaxt = 'n', xlab = "", ylab = "")
plot(close_anterior, main = "Dados Originais - X", col = "blue", pch = 19, xaxt = 'n', xlab = "", ylab = "")
plot(Xscaled, main = "Dados Normalizados - X", col = "red", pch = 19, xaxt = 'n', xlab = "", ylab = "")


# Define os índices para os conjuntos de treinamento e teste
train <- 1:6
test <- 7:8

# Criando um novo dataframe para visualização
data_visualizacao <- data_analise
data_visualizacao$set <- c(rep("Treinamento", length(close) * 0.75), rep("Teste", length(close) * 0.25))

ggplot(data_visualizacao, aes(x = 1:nrow(data_visualizacao), y = fechamento, color = set)) +
  geom_point() +
  labs(title = "Divisão dos Conjuntos de Treinamento e Teste",
       x = "",
       y = "") +
  scale_color_manual(values = c("Treinamento" = "blue", "Teste" = "red")) +
  theme_minimal()


# Define a semente para reprodutibilidade
set.seed(12)

# Define o callback de early stopping para interromper o treinamento se não houver melhora
early_stopping <- callback_early_stopping(
  monitor = 'val_loss',
  patience = 10,
  restore_best_weights = TRUE
)

# Treina o modelo LSTM
model <- trainr(
  X = X[, train],
  Y = Y[, train],
  learningrate = 0.005,
  hidden_dim = 47,
  numepochs = 1500,
  network_type = "lstm",
  dropout = 0.3,
  update_rule = "sgd",
  momentum = 0.8,
  callbacks = list(early_stopping)
)


par(mfrow = c(1, 1))
# Plota o erro médio por época durante o treinamento
plot(colMeans(model$error), type = 'l', xlab = 'Epoch', ylab = 'Error', 
     main = 'Erro Médio por Época Durante o Treinamento')


plot(model$error, type = 'l', xlab = 'Batch', ylab = 'Error', 
     main = 'Erro por Batch Durante o Treinamento')



# Avalia o modelo no conjunto de treinamento
Ytrain <- t(matrix(predictr(model, X[, train]), nrow = 1))
Yreal <- t(matrix(Y[, train], nrow = 1))

# Função para calcular o coeficiente de determinação (R²)
rsq <- function(y_actual, y_predict) {
  cor(y_actual, y_predict)^2
}

# Calcula o R² para o conjunto de treinamento
rsq(Yreal, Ytrain)

# Calcula o R² para o conjunto de treinamento
r_squared_treino <- cor(Ytrain, Yreal)^2
r_squared_treino_percentual <- paste("R² = ", round(r_squared_treino, 3), " (", round(r_squared_treino * 100, 2), "%)", sep="")

# Plota a comparação entre os valores reais e preditos no conjunto de treinamento
plot(Ytrain, type = "l", col = "darkred", lwd = 2,
     main = "Valores Previstos x Reais - Treinamento",
     xlab = "Índice", ylab = "Valor",
     ylim = range(c(Ytrain, Yreal)))
lines(Yreal, col = "darkblue", lwd = 2)
legend("topright", legend = c("Previsto", "Reais"),
       col = c("darkred", "darkblue"), lty = 1, lwd = 2,
       cex = 0.6)
text(x = max(seq_along(Ytrain)) * 0.15, y = max(Ytrain) * 1.15, labels = r_squared_treino_percentual, col = "darkred", cex = 1.1)


# Dados para o gráfico (valores reais e previstos do conjunto de treinamento)
train_data <- data.frame(Real = Yreal, Previsto = Ytrain)

# Criando o gráfico de dispersão com linha de regressão
ggplot(train_data, aes(x = Real, y = Previsto)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Valores Reais vs. Valores Previstos no Conjunto de Treinamento",
       x = "Valores Reais",
       y = "Valores Previstos") +
  theme_minimal()


# Avalia o modelo no conjunto de teste
Ytest <- matrix(Y[, test], nrow = 1)
Ytest <- t(Ytest)
Yp <- predictr(model, X[, test])
Ypredicted <- matrix(Yp, nrow = 1)
Ypredicted <- t(Ypredicted)

# Cria um dataframe com os valores reais e preditos para o conjunto de teste
result_data <- data.frame(Ytest)
result_data$Ypredicted <- Ypredicted

# Calcula o R² para o conjunto de teste
rsq(result_data$Ytest, result_data$Ypredicted)

# Calcula o R² para o conjunto de teste
r_squared_test <- cor(Ytest, Ypredicted)^2
r_squared_test_percentual <- paste("R² = ", round(r_squared_test, 3), " (", round(r_squared_test * 100, 2), "%)", sep="")

# Plota a comparação entre os valores reais e preditos no conjunto de teste
plot(Ytest, type = "l", col = "darkred", lwd = 2,
     main = "Valores Previstos x Reais - Teste",
     xlab = "Índice", ylab = "Valor",
     ylim = range(c(Ytest, Ypredicted)))
lines(Ypredicted, col = "darkblue", lwd = 2)
legend("topright", legend = c("Previsto", "Real"),
       col = c("darkred", "darkblue"), lty = 1, lwd = 2,
       cex = 0.6)
text(x = max(seq_along(Ytest)) * 0.15, y = max(Ytest) * 0.98, labels = r_squared_test_percentual, col = "darkred", cex = 1.1)


############################### PREVER PREÇO ############################################


# Baixa dados históricos da Petrobras e armazena no dataset 'data'
data <- tq_get("PETR4.SA", from = "2024-08-30", to = "2024-10-01")

# Prepara os dados para análise
fechamento <- data$close  # Extrai a coluna de preços de fechamento
fechamento_anterior <- lead(fechamento, n = 1L)  # Cria uma coluna com os preços de fechamento do dia anterior

# Usar dados existentes para prever futuros
new_data_analise <- data.frame(fechamento)
new_data_analise$fechamento_anterior <- fechamento_anterior

# Normalizar os últimos valores de fechamento
ultimo_valor <- tail(new_data_analise$fechamento, 1)
ultimo_valor_normalizado <- (ultimo_valor - min_Y) / (max_Y - min_Y)

# Prever os próximos dias
previsoes_futuras <- numeric()
dias_futuros <- 2

for (i in 1:dias_futuros) {
  input <- matrix(ultimo_valor_normalizado, ncol = 1, nrow = 1)
  pred <- predictr(model, input)
  pred_desnormalizado <- pred * (max_Y - min_Y) + min_Y
  previsoes_futuras <- c(previsoes_futuras, pred_desnormalizado)
  
  # Atualizar o último valor normalizado para a próxima iteração
  ultimo_valor <- pred_desnormalizado
  ultimo_valor_normalizado <- (ultimo_valor - min_Y) / (max_Y - min_Y)
}

# Criar um dataframe para as previsões futuras
futuras_datas <- seq.Date(from = max(data$date) + 1, by = "day", length.out = dias_futuros)
previsoes_df <- data.frame(Data = futuras_datas, Previsao = previsoes_futuras)



data_filtered <- data %>% filter(date >= max(data$date) - 18) %>% rename(fechamento = close)
previsoes_df <- previsoes_df %>% rename(fechamento = Previsao, date = Data)
previsoes_df_last <- tail(previsoes_df, 1)
data_real <- tq_get("PETR4.SA", from = "2024-09-30", to = "2024-10-02")
fechamento_data_real <- data_real %>% select(date, close)
data_real <- fechamento_data_real
data_real <- data_real %>% rename(fechamento = close)

# Visualizar os resultados ajustados para os próximos dias
ggplot() +
  geom_line(data = data_filtered, aes(x = date, y = fechamento, color = "Real")) +
  geom_line(data = previsoes_df, aes(x = date, y = fechamento, color = "Previsto")) +
  geom_text(data = previsoes_df_last, aes(x = date, y = fechamento, label = round(fechamento, 2)), vjust = -1.0, hjust = 0.43, color = "red", size = 4) +
  geom_line(data = data_real, aes(x = date, y = fechamento, color = "black")) +
  geom_text(data = data_real, aes(x = date, y = fechamento, label = round(fechamento, 2)), vjust = -1, hjust = 0.65, color = "darkblue", size = 3.5) +
  scale_x_date(date_breaks = "1 day", date_labels = "%d-%m") +
  labs(title = "Previsão de Preços Futuros de Ações (Próximos Dois Dias)",
       x = "Data",
       y = "Preço",
       color = "Legenda") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_color_manual(values = c("Real" = "blue", "Previsto" = "red"))
