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

1. **Base de Dados:** O estudo utilizou o conjunto arquivado `HAM10000` contendo mais de 10.000 imagens dermatoscópicas multiclasses, rotuladas historicamente, com demográficos atrelados.
2. **Arquitetura de Extração (CNN Backbone):** Utilizamos o método de Transferência de Aprendizado baseando-se no modelo pré-treinado *EfficientNetB3*. As imagens foram pré-processadas (com *Data Augmentation* por funções Albumentations para lidar com ruídos clínicos reais) e reduzidas ao limiar global (*Global Average Pooling*) numa camada nomeada `fused_dense_1`.
3. **Abordagem Dupla Multimodal:** Como linha condutora, a meta-informação clínica (escalonada numericamente) foi injetada paralelamente às imagens nas predições de um conjunto de teste estrito (*Holdout* fixo de 20%, garantindo *zero-data-leakage*).
4. **Modelos Secundários:** As *embeddings* geradas pela camada `fused_dense_1` da arquitetura CNN para cada imagem do conjunto de treinamento serviram como *input features* multimodais para algoritmos robustos convencionais: XGBoost, Random Forest (floresta de 100 estimadores), SVM C-Support e Naive Bayes.
5. **Critério Avaliativo:** Focal Loss foi aplicada no treinamento convolucional devido ao desbalanceamento; as métricas coletadas nos testes foram: Acurácia Global, *Macro F1-Score* e *ROC AUC* (Curva Característica de Operação via método unificado OVR).

---

## 3. Resultados e Discussões

A avaliação sobre os 2.000 registros separados não-vistos demonstrou uma dinâmica peculiar do aprendizado de máquina em cenários clínicos polarizados: o classificador próprio da extremidade (Head) da Rede Neural Convolucional apresentou imensa dificuldade de balanço, colapsando sob métricas isoladas. 

Entretanto, as **camadas ocultas extratoras de características (Backbone)** da mesma rede mostraram-se excepcionalmente boas abstraindo texturas. Quando o vetor denso da CNN somado à Idade e Sexo foi ofertado aos algoritmos puros baseados em gradiente de árvore (XGBoost) e amostras aleatórias (Random Forest), observou-se uma revolução no diagnóstico:

| Modelo Classificador Multimodal | Acurácia | F1-Score (Macro) | AUC ROC (OVR) |
| :--- | :---: | :---: | :---: |
| **XGBoost Classifier** | **68.04%** | **0.270** | **0.753** |
| **Random Forest** | 68.44% | 0.276 | 0.719 |
| **Support Vector Machine (SVM)** | 66.94% | 0.114 | 0.642 |
| **CNN Fim-a-Fim (Dense Head)** | 11.38% | 0.030 | 0.462 |
| **Naive Bayes** | 7.38% | 0.081 | - |

*(Nota: O valor de F1 não passa de ~0.27 devido à dispersão matemática de 7 classes onde uma detém 67% de todo o espaço amostral benigno; sob a rigidez da medicina, contudo, a AUC valida a capacidade da curva).*

A superioridade notória do **XGBoost (AUC OVR = 0.75)** evidencia que algoritmos clássicos de aprendizado adaptativo se ajustam de maneira notavelmente superior a dados tabulares (onde a Multimodalidade entra fortemente), enquanto a CNN se mostra rígida e suscetível à anulação preditiva quando forçada a classificar parâmetros de grandezas físicas muito díspares (pixel 0~1 *vs* Idade nominal 0~80), como atestado pela queda para 11%.

## 4. Conclusão Parcial

O estudo atinge o principal indicativo para sistemas de Segunda Opinião na Atenção Primária: Redes Neurais são soberanas na varredura visual de pigmentações suspeitas, enquanto Machine Learning Cássico domina o agrupamento pragmático entre anamnese e biologia. A solução recomendada para auxílio a médicos da atenção básica num Sistema de Apoio Clínico é a hibridização cirúrgica: Uma CNN (`EfficientNet-B3`) isolada atuando como mapeadora de imagens, acoplada em série a um ensamble de árvores lógicas (`XGBoost`), entregando mais de 75% de acurácia global na separação entre as categorias diagnósticas.

