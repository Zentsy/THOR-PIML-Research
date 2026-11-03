# ESCOPO E DIRETRIZES TÉCNICAS — SEMCITEC 2026

**Projeto:** THOR-PIML (*Taylor-Hurdle Optimized Regional Physics-Informed Machine Learning*)  
**Artigo:** *"Projeção do regime pluviométrico e indicadores bioclimáticos sob cenários SSP (2026–2100) em Guarulhos-SP via downscaling com o modelo THOR-PIML"*  
**Meta:** Artigo expandido / Relatório técnico completo de até **25 páginas**.

---

## 1. Fundamentação e Narrativa Científica

### A. Por que o Downscaling Estatístico é Essencial?
* Os modelos climáticos globais (GCMs / CMIP6) e regionais (RCMs / CORDEX) possuem resolução espacial grosseira ($25\text{ a }100\text{ km}$), gerando:
  1. **Drizzle Bias Crônico:** Superestimação de dias de chuva fraca ($0.1\text{ a }2\text{ mm}$);
  2. **Subestimação de Picos Convectivos:** Incapacidade de resolver tempestades convectivas locais e efeitos orográficos da Serra da Cantareira / Bacia de Guarulhos.
* O modelo **THOR-PIML** atua como operador de transferência física $\mathcal{F}_{\theta}(\mathbf{X}_{\text{GCM}}) \to y_{\text{local}}$, corrigindo a distribuição probabilística da precipitação sem violar os limites termodinâmicos de Clausius-Clapeyron.

### B. Postura Científica & Escala Temporal
* **Invariante Metodológica:** Não realizamos nem apresentamos previsão meteorológica determinística de dia individual para 2026–2100 (a atmosfera futura é caótica e estocástica).
* **Foco Central:** **Agregações Mensais, Sazonais (DJF vs JJA), Curvas de Permanência (FDC) e Variáveis Bioclimáticas Oficiais.**

---

## 2. Variáveis Bioclimáticas de Precipitação (BIO12 a BIO19 - WMO/WorldClim)

As variáveis bioclimáticas de precipitação são calculadas sobre agregações mensais e trimestrais, medindo a magnitude, sazonalidade e contrastes do regime hídrico:

$$\begin{array}{|l|l|l|}
\hline
\textbf{Código} & \textbf{Nome da Variável} & \textbf{Definição Matemática} \\ \hline
\text{BIO12} & \text{Precipitação Anual Total} & P_{\text{total}} = \sum_{m=1}^{12} P_m \text{ (mm/ano)} \\ \hline
\text{BIO13} & \text{Precipitação do Mês Mais Chuvoso} & P_{\text{max}} = \max(P_1, \dots, P_{12}) \text{ (mm)} \\ \hline
\text{BIO14} & \text{Precipitação do Mês Mais Seco} & P_{\text{min}} = \min(P_1, \dots, P_{12}) \text{ (mm)} \\ \hline
\text{BIO15} & \text{Sazonalidade da Precipitação} & CV = \frac{\sigma(P_m)}{\mu(P_m)} \times 100 \text{ (\%)} \\ \hline
\text{BIO16} & \text{Precipitação do Trimestre Mais Úmido} & \max \sum_{k=m}^{m+2} P_k \text{ (ex: Verão DJF em mm)} \\ \hline
\text{BIO17} & \text{Precipitação do Trimestre Mais Seco} & \min \sum_{k=m}^{m+2} P_k \text{ (ex: Inverno JJA em mm)} \\ \hline
\text{BIO18} & \text{Precipitação do Trimestre Mais Quente} & \sum P_k \text{ no trimestre de maior temperatura} \\ \hline
\text{BIO19} & \text{Precipitação do Trimestre Mais Frio} & \sum P_k \text{ no trimestre de menor temperatura} \\ \hline
\end{array}$$

---

## 3. Indicadores Complementares de Extremos (ETCCDI)

* **$R10\text{mm}$:** Número anual de dias com chuva forte ($P \ge 10\text{ mm/dia}$).
* **$R20\text{mm}$:** Número anual de dias com tempestades severas ($P \ge 20\text{ mm/dia}$).
* **$RX1\text{day}$:** Precipitação máxima diária acumulada em 1 dia no ano ($\text{mm}$).
* **$RX5\text{day}$:** Precipitação máxima acumulada em 5 dias consecutivos ($\text{mm}$ — gatilho de transbordamento de bacias urbanas).
* **$CDD$ (*Consecutive Dry Days*):** Maior sequência de dias secos consecutivos ($P < 1.0\text{ mm}$) no ano.
* **$CWD$ (*Consecutive Wet Days*):** Maior sequência de dias chuvosos consecutivos ($P \ge 1.0\text{ mm}$) no ano.
* **$R95p$ / $R99p$:** Percentual do volume anual precipitado em dias muito úmidos ($> p95$) e extremamente úmidos ($> p99$).

---

## 4. Divisão Temporal das Projeções (2026–2100)

A análise comparativa sob os cenários **SSP2-4.5** (estabilização média) e **SSP5-8.5** (alta emissão fóssil) divide-se em 4 épocas climatológicas:

1. **Linha de Base Histórica (Baseline):** $1981\text{–}2014$ ($34\text{ anos}$ via ERA5-Land / CHIRPS).
2. **Futuro Próximo:** $2026\text{–}2050$ ($25\text{ anos}$).
3. **Meio do Século:** $2051\text{–}2075$ ($25\text{ anos}$).
4. **Final do Século:** $2076\text{–}2100$ ($25\text{ anos}$).

---

## 5. Estrutura das 25 Páginas para o Artigo SEMCITEC

1. **Introdução & Motivação:** Vulnerabilidade climática na RMSP, bacia do Rio Baquirivu-Guaçu / Alto Tietê, limitações de GCMs/RCMs brutos e a relevância de PIML.
2. **Revisão Bibliográfica:** Downscaling estatístico com redes neurais, abordagens Hurdle e restrições de física informada.
3. **Metodologia Completa:**
   * Formulação da arquitetura **THOR-V8** (2D Synoptic CNN + ResLSTM + Multi-Scale TCN + Gated Fusion + SDPA + Hurdle).
   * Função de perda com barreira de Clausius-Clapeyron ($\mathcal{L}_{\text{phys}}$).
   * Protocolo de validação com Zero Data Leakage.
4. **Validação Histórica & Benchmark:** Resultados do teste cego ($1981\text{–}2026$), comparação com EQM, ResLSTM, TCN e análise de ablação.
5. **Projeções Climáticas (2026–2100):**
   * Séries decadais e anomalias percentuais de $\text{PRCPTOT}$ sob SSP2-4.5 vs SSP5-8.5.
   * Análise detalhada das variáveis bioclimáticas de precipitação (**BIO12 a BIO19**).
   * Evolução dos índices de extremos ETCCDI ($R10\text{mm}, R20\text{mm}, RX1\text{day}, CDD, CWD$).
   * Comparativo: CORDEX Bruto vs. THOR-PIML Downscaled.
6. **Discussão de Impactos Urbanos e Conclusão:** Riscos hidrológicos, vulnerabilidade a inundações urbanas e mitigação de desastres em Guarulhos.
