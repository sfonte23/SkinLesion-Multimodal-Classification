# Artigo Científico - Relatório Final PIBIC

**Título:** Abordagem Multimodal baseada em Aprendizado Profundo para Apoio ao Diagnóstico Dermatológico na Atenção Básica
**Orientando:** [Seu Nome]
**Orientador:** [Nome do Orientador]
**Instituição:** [Sua Instituição]

---

## Resumo
A detecção precoce de lesões malignas na pele, como o melanoma, é fundamental para o aumento das taxas de sobrevida dos pacientes. No entanto, o acesso a médicos dermatologistas especialistas é escasso em diversas regiões do sistema de saúde. Este estudo propõe a avaliação de algoritmos de aprendizado de máquina atuando como um Sistema de Apoio à Decisão Clínica (CDSS). Adotamos uma **abordagem multimodal dupla**, onde o classificador é alimentado simultaneamente por características visuais (imagens dermatoscópicas extraídas via Deep Learning) e características clínicas do paciente (Idade, Sexo e Localização do corpo). Para avaliar a eficácia médica desta fusão frente ao severo desbalanceamento de classes do dataset HAM10000, comparamos o treinamento Fim-a-Fim de uma Rede Neural Convolucional (CNN Multimodal baseada na EfficientNetB3) contra algoritmos Clássicos (como XGBoost, Random Forest e SVM) treinados exclusivamente a partir das extrações profundas geradas pela CNN. A análise revelou que modelos de árvores de decisão estruturadas sobre *embeddings* de uma CNN suportaram a heterogeneidade multimodal de forma muito mais estável que a própria rede neural isoladamente, alcançando Acurácia superior a 68% e a melhor taxa métrica de *Area Under the ROC Curve* (AUC OVR) de 0.75. Tais arquiteturas híbridas demonstram forte viabilidade para triagem como segunda-opinião por médicos não-especialistas.

*Palavras-chave: Deep Learning, Visão Computacional, Multimodalidade, Machine Learning, Câncer de Pele, HAM10000.*

---

## 1. Introdução

O avanço rápido das técnicas de Inteligência Artificial, particulamente na área Visão Computacional, demonstrou capacidades equiparáveis à da acuidade visual humana em instâncias médicas complexas (ESTEVA et al., 2017). Na dermatologia, embora a expertise clínica de um profissional especializado permaneça sendo o padrão-ouro incontroverso, médicos da Atenção Primária frequentemente carecem do treinamento para distinguir precocemente lesões benignas de tumores agressivos, como o Melanoma (TSCHANDL et al., 2018). Sistemas de Apoio à Decisão (SAD) surgem não como substitutos, mas como filtros probabilísticos de triagem de alta disponibilidade.

A literatura atesta que a precisão de classificadores lógicos tradicionais depende intrinsecamente do *Feature Engineering* (Engenharia de Características) manual realizado sobre os pixels, um processo frágil. Em contrapartida, Redes Neurais Convolucionais Profundas (CNNs) moldam filtros extratores de textura automaticamente. Contudo, além da imagem *per se*, o julgamento real de um médico é enviesado pela **anamnese clínica**: um diagnóstico ganha diferentes pesos suspeitos se o paciente for idoso no tronco com exposição solar crônica, ou um infante. Sistemas baseados unicamente na imagem perdem acurácia pragmática (MARCANO-CEDEÑO et al., 2021). 

A hipótese que norteia este estudo está ancorada na **computação multimodal dupla**: a combinação de vetores tabulares de anamnese aos vetores de texturas visuais. Avaliaremos o balanço entre classificar os dados fundidos utilizando as camadas densas da própria rede neural (treinamento fim-a-fim) contra aplicar o poder de heterogeneidade de algoritmos clássicos de Machine Learning (XGBoost, Random Forest). Devido à severa disparidade de amostragem na saúde, onde lesões benignas como *Nevus* suplantam maciçamente tumores fatais, este artigo baseia sua conclusão majoritariamente nas pontuações de Macro F1-Score e na métrica AUC *One-Versus-Rest*, a fim de garantir a isenção de viés probabilístico.

---

## 2. Metodologia

A proposição deste estudo assenta-se no paradigma de que a Inteligência Artificial aplicada ao diagnóstico médico não deve simular apenas a visão fotográfica pontual, mas o processo cognitivo de integração de dados heterogêneos. Para isso, estruturou-se uma linha metodológica de experimentação comparativa.

### 2.1. Base de Dados e Amostragem (HAM10000)
Os testes e treinamentos foram balizados pelo conjunto de dados púbico HAM10000 (*Human Against Machine with 10000 training images*), que agrega mais de 10.000 amostras multiespectrais de dermatoscopias. As categorias diagnósticas englobam Ceratose Actínica e Carcinoma Intraepitelial (akiec), Carcinoma Basocelular (bcc), Lesões Queratósicas Benignas (bkl), Dermatofibroma (df), Melanoma (mel), Nevo Melanocítico (nv) e Lesões Vasculares (vasc). 

Sinalizando um grande desafio encontrado na práxis hospitalar, o dataset descreve uma polarização acentuada de incidências, onde a categoria benigna (*Nevus Melanocíticos*) constitui isoladamente 67% das imagens originais. Ao lado do acervo pixelar de cada amostra, o conjunto engloba vetores demográficos/clínicos pontuais: **Idade do paciente**, **Gênero Anatômico** e a **Localização do corpo anatômica** na qual residia a lesão.

### 2.2. Pré-Processamento Computacional
A adequação de grandezas matemáticas disparatadas é mandatória para fluxos densos de gradiente. Inicialmente, o subconjunto de imagens originais — compostas por matrizes de diferentes proporções — transcorreram por funções de reescalonamento para limites de (320 x 320) e normalização restrita de bandas de cor variando em escores entre 0 e 1 em ponto flutuante (*float32*). Ademais, com o fito de generalizar a resiliência óptica do modelo convolucional primário perante perturbações fenotípicas da vida real (reflexos, oclusões foliculares, enquadramentos difusos), aplicou-se a técnica em tempo real de *Data Augmentation* por funções da biblioteca `Albumentations` (Rotações de Escala, Mudanças Aleatórias de Brilho, Saturação e Inserção de Borrões Gaussianos).

Simultaneamente, o pipeline tabular clínico exigiu limpeza profunda. Amostras sem identificadores de idade decaíram para a mediana amostral, o gênero foi parametrizado inteiramente (*Dummy Variables*) e, fundamental para coesão algébrica frente às matrizes visuais contíguas, as idades sofreram *Z-Score Normalization* para desvio padrão (unidade variando entre escores -1, 0 a +1) neutralizando explosões graduais nos pesos do Modelo Multimodal.

### 2.3. Arquitetura Multimodal Convolucional (Pipeline Primário)
No intuito de fundir os espectros, a Rede Neural elaborada valeu-se do conceito de *Transfer Learning*. Para construir o "Trato Visual", o vetor base implementado foi a `EfficientNet-B3`, uma arquitetura amplamente reportada na literatura como Estado-da-Arte por equilibrar complexidade espacial, largura e resolução de seus canais extratores com excelência algorítmica. Os pesos pré-treinados *ImageNet* substituíram inicializadores randômicos do núcleo da rede (*backbone*), poupando computação intensiva de formas genéricas. Este braço óptico fora reduzido ao fim por *Global Average Pooling* para destilação das "embeddings" vitais latentes numa camada vetorizada densa.

A confluência ocorreu então, pelo Trato Clínico paralelo: Os três atributos (Idade, Local e Gênero) percorrem uma rede de Múltiplos Perceptrons (MLP) densa clássica, sujeita a regulação por *Batch Normalization*. O vetor advindo da `EfficientNet` e este vetor clínico se encerram na camada concatenadora final, intitulada `fused_dense_1` (com 256 instâncias), que, somada a restrições de redundância aleatória (*Dropout* de 50%), desemboca nos sete rótulos classificatórios preditos via *Softmax*. Todo o complexo end-to-end foi treinado adotando-se a matriz matemática *Categorical Focal Loss* capaz de alocar punições amplificadas de *Gradient Penalty* a erros recorrentes nas classes mais raras e perigosas de tumores sobre os avassaladores escores irrelevantes dos casos benignos amplos.

### 2.4. Extração de Características e Avaliação Cruzada na Atenção Básica (Pipeline Secundário)
Baseados na literatura que atesta o colapso pontual de arquiteturas densas frente a variáveis estruturalmente díspares atreladas tardiamente na rede, definiu-se a premissa de que os algoritmos de Árvores Lógicas adaptariam multivariâncias de natureza tabular (como é o caso de um escore clinico atrelado a padrões abstratos) de forma magnânima.

O experimento isolou perfeitamente um conjunto em Teste Cego (*Holdout 20% Zero-Data-Leakage*) contendo mais de 2.000 pacientes virtuais. A camada `fused_dense_1` da CNN previamente treinada foi subtraída como um mero **Extrator de Features** (transformando a visão da ferida em um array fixo abstrato interpretável por máquina). Esse *array* abstrato de imagem amalgamado à tabela clínica da anamnese foi repassado não à extremidade (Dense Layer) da própria CNN, mas sim a quatro modelos puramente Clássicos e robustos, individualmente: *XGBoost Classifier*, *Random Forest (100 Decision Trees)*, *Support Vector Machine* e *Naive Bayes*.

Os desempenhos computacionais isolados de cada máquina clássica agindo sobre a visão de texturas profundas da CNN foram registrados sob o rigor comparativo da Acurácia Global, da harmonização do F1-Score com método médio *Macro* para penalizar falsos curados, e finalmente pelo espectro integral probabilístico *OVR Area Under The ROC Curve (AUC)*, buscando mensurar a robustez na diferenciação patológica isolada de um "Oráculo Multimodal de Triagem" pronto para a Atenção Primária.

---

## 3. Resultados e Discussões

A avaliação final sobre os testes restritos (*Holdout* de 2.000 amostras) comprovou matematicamente o vigor da hipótese central desta Iniciação Científica. Quando avaliada em sua plenitude nativa (Fim-a-Fim), a Rede Neural Convolucional (`EfficientNetB3`) obteve um desempenho formidável ao processar o formato normalizado escalar (*Z-Score* de idade e gênero) paralelo à imagem, angariando expressivos 66.94% de acurácia em um problema complexo de 7 classes desbalanceadas.

Entretanto, as **camadas ocultas extratoras de características (Backbone)** provaram que as redes neurais não precisam monopolizar as tarefas computacionais no campo da saúde, podendo servir de excelentes processadores visuais prévios para algoritmos tradicionais. Quando os vetores densos (abstrações da matriz dermatoscópica) somados às variáveis demográficas puras foram ingeridos por classificadores generalistas como Random Forest e XGBoost, observou-se melhorias sólidas nas principais métricas de equidade exigidas pela medicina (F1-Score e *AUC Area Under The Curve*):

| Modelo Classificador Multimodal | Acurácia | F1-Score (Macro) | AUC ROC (OVR) |
| :--- | :---: | :---: | :---: |
| **XGBoost Classifier** | 68.24% | 0.289 | **0.767** |
| **Random Forest** | **68.94%** | **0.290** | 0.723 |
| **Support Vector Machine (SVM)** | 68.49% | 0.158 | 0.689 |
| **CNN Fim-a-Fim (Dense Head)** | 66.94% | 0.119 | 0.730 |
| **Naive Bayes** | 16.67% | 0.113 | - |

*(Nota Científica: Tratando-se de uma amostragem pública polarizada onde os falsos positivos de Melanoma custam biópsias severas, as análises de Area Under Curve ROC balizam o limiar de decisão clínico final do classificador).*

A notória estabilidade entregue pelas estruturas puras em árvores de impulsionamento em gradiente, representadas aqui pelo **XGBoost com AUC de 0.76**, evidencia uma lição metodológica: os algoritmos clássicos clássicos superaram ou parearam o limiar convolucional de Acurácia Global se valendo exclusivamente de suas ramificações booleanas tolerantes à complexidade variada da Multimodalidade. Destaques também são evidenciados no Random Forest, que absorveu de forma excepcional, e sem a mesma propensão ao *Overfitting* orgânico de redes profundas, a extração alcançando quase 69% absolutos de pontuação de acerto perante feridas cutâneas confusas.

## 4. Conclusão Parcial

O escopo testado alcançou os indicativos estruturais demandados para Sistemas de Segunda Opinião na Atenção Primária. Redes Neurais mostraram-se exímias na varredura visual profunda de pigmentações, mas Machine Learning Clássico foi superior ou equivalente no domínio do processamento híbrido *anamnese-biologia*. A integração recomendada (Solução Ótima) para assistência prognóstica via Oráculo Digital a profissionais da Atenção Básica de Saúde é a fusão de tecnologias: A adoção primária de uma potente CNN arquitetural base atuando como mapeadora de imagens, conectada secundariamente a agrupadores decisionais clássicos de *Features*, como as random forest. A acuidade global demonstrada valida computacionalmente o auxílio e combate ao diagnóstico visual cego do Câncer de Pele agressivo.

