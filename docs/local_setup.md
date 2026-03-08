# Ambiente Local de Experimentação

Esta pasta contém as versões dos notebooks de Iniciação Científica configuradas especificamente para rodar **na sua própria máquina** (Sem necessidade do Google Drive).

## Arquivos:
1. `01_CNN_Training_Multimodal.ipynb`: O script de treinamento da Rede Neural EfficientNet. Está configurado para salvar o arquivo de pesos treinado (`.keras`) diretamente na pasta `/models` do projeto, e os gráficos de função de perda na pasta `/results`.
   - **Nota**: Este treinamento requer preferencialmente uma Placa de Vídeo (GPU) dedicada no seu computador.
2. `02_Evaluation_and_Comparison.ipynb`: Pipeline para extração de dados do HAM10000, extração profunda de texturas da CNN, e comparação rigorosa vs os classificadores Clássicos (RF, SVM, Naive Bayes e XGBoost).

## Como Rodar:
Basta ter inicializado o ambiente virtual do Python no seu Windows e rodar o `jupyter notebook`:
```bash
# Na raiz do seu projeto `ICV`
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
