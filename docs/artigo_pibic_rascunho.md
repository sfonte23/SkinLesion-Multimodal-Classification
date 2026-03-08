# Artigo Científico - Relatório Final PIBIC

**Título (Sugestão):** Abordagem Multimodal baseada em Aprendizado Profundo para Apoio ao Diagnóstico Dermatológico por Médicos Não Especialistas
**Orientando:** [Seu Nome]
**Orientador:** [Nome do Orientador]
**Instituição:** [Sua Faculdade]

---

## Resumo
A detecção precoce de lesões malignas na pele, como o melanoma, é fundamental para o aumento das taxas de sobrevida dos pacientes. No entanto, o acesso a médicos dermatologistas especialistas é escasso em diversas regiões do sistema de saúde. Este estudo propõe a avaliação de um Sistema de Apoio à Decisão Clínica (CDSS - *Clinical Decision Support System*) alimentado por algoritmos de aprendizado de máquina. Diferente das abordagens clássicas focadas estritamente em atributos visuais ou textuais isolados, este documento arquiteta e compara o desempenho de uma Rede Neural Convolucional (CNN - EfficientNetB3) multimodal, que funde dados de imagens dermatoscópicas extraídas do dataset HAM10000 com dados demográficos e clínicos (Idade, Sexo e Localização do corpo). Para validar a superioridade da extração profunda, as características ocultas (*embeddings*) geradas pela CNN são comparadas com algoritmos puros de Machine Learning Clássico (Random Forest, SVM, XGBoost, Naive Bayes). O objetivo primário é identificar qual arquitetura oferece a maior confiabilidade métrica (Acurácia, macro F1-Score e *Area Under the Curve* OVR) para operar como um oráculo de segunda-opinião confíavel na triagem realizada por profissionais não-especialistas da Atenção Básica.

*Palavras-chave: Deep Learning, Visão Computacional, Multimodalidade, Diagnóstico Médico Assitido, Câncer de Pele.*

---

## 1. Introdução

O avanço rápido das técnicas de Inteligência Artificial, particulamente em subcampos de Visão Computacional, demonstrou capacidades equiparáveis à da acuidade visual humana em diversas áreas da saúde. Na dermatologia diagnóstica, embora a expertise clínica de um profissional especializado permaneça sendo o padrão-ouro incontroverso, médicos da Atenção Primária ou em Unidades de Pronto Atendimento frequentemente carecem de treinamento extenso ou equipamentos (dermatoscópios de precisão) necessários para distinguir precocemente lesões benignas de tumores agressivos agressivos, como o Carcinoma Basocelular ou o Melanoma. 

Nesse contexto computacional, os Sistemas de Apoio à Decisão (SAD) surgem não como substitutos ao juízo clínico, mas como filtros probabilísticos de triagem e "segunda opinião" de baixo custo e alta disponibilidade. 

A literatura atesta que a precisão de classificadores lógicos tradicionais (como Support Vector Machines ou Random Forests) depende intrinsecamente do *Feature Engineering* (Engenharia de Características) manual realizado sobre os pixels da imagem — como cálculo de assimetria, bordas e cores —, um processo frágil e sujeito a perdas de informação. Em contrapartida, as Redes Neurais Convolucionais Profundas (CNNs) transferem essa carga analítica para o próprio algoritmo, moldando filtros extratores de textura baseados em descida de gradiente durante sua fase de treinamento.

Contudo, além da imagem per se, o julgamento de um médico é fortemente enviesado pela anamnese: um diagnóstico visual de uma lesão neoplásica ganha pesos diferentes de suspeita caso o paciente seja um idoso no tronco, ou um jovem no rosto. Sistemas monocromáticos baseados unicamente na imagem perdem considerável acurácia. A hipótese que move este trabalho permeia a Ciência da Computação Multimodal: o condicionamento da matriz visual de uma Rede Neural consolidada com os vetores clínicos tabulares tradicionais.

O presente artigo avalia o balanço estritamente arquitetural entre arquiteturas Convolucionais (EfficientNetB3) comparadas a modelos de inteligência artificial clássicos ao atuarem sob características extraídas profundas. O objetivo central é eleger através de matriz de métricas severa, focada em penalizar desbalanceamentos da área da saúde (F1-score Macro e AUC tipo *One-Versus-Rest*), a abordagem computacional ótima para guiar as diretrizes de ação (e.g., recomendação de biópsia rápida) por um não especialista diante do conjunto público internacional de desafios diagnósticos *Human Against Machine* (HAM10000).

---

## 2. Metodologia (Rascunho)
*(A ser detalhado com as infos do notebook, uso de class weights / focal loss, e a separação dos algoritmos na Extração de Features)*

## 3. Resultados 
*(Serão populados com a tabela de Acurácia, AUC e o F1-Score gerados e a argumentação do porquê o Modelo X ganhou do Y baseado no gráfico `comparativo_final_plot.png`)*

## 4. Conclusão
*(Será descrita de acordo com os resultados apontando a viabilidade do sistema de Apoio à Decisão para Clínicos Gerais)*
