# Ambiente Cloud (Google Colab)

Esta pasta contém as versões dos notebooks preparadas para rodar no **Google Colaboratory** extraindo todo o potencial de aceleração por Hardware gratuita.

## Arquivos:
1. `01_CNN_Training_Multimodal.ipynb`: O script pesado de treinamento da CNN Multimodal. Salva os artefatos de saída (pesos e gráficos) mapeando diretamente para o seu `Google Drive` conectado na nuvem.
2. `02_Evaluation_and_Comparison.ipynb`: Validação Final. Carrega o modelo do Google Drive para aferir sua Acurácia, F1-Score e plotar as matrizes de confusão.

## Como Rodar:
1. Faça o Upload de ambos os arquivos desta pasta para o seu [Google Colab](https://colab.research.google.com/).
2. No menu superior esquerdo: **Runtime (Ambiente de Execução) > Change runtime type (Alterar tipo de ambiente de execução)**.
3. Selecione **T4 GPU** ou superior.
4. Execute as células sequencialmente. Ele pedirá autorização na primeira rodada para conectar o seu Google Drive à nuvem. Todas as suas matrizes e modelos serão salvos lá protegidos!
