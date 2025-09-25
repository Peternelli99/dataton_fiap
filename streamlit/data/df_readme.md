## **COLUNAS DO DATASET — DESCRIÇÃO COMPLETA**

### **📋 CHAVES RELACIONAIS (2 colunas)**

**1. `vaga_id` (string)**
- **Origem:** Chave primária de vagas.json
- **Significado:** Identificador único da vaga no sistema ATS da Decision
- **Exemplo:** "4530", "10976" 
- **Uso:** Permite agrupar candidatos por vaga e calcular métricas de performance por posição

**2. `codigo_candidato` (string)** 
- **Origem:** Chave primária de applicants.json, harmonizada com prospects.json
- **Significado:** Identificador único do candidato no sistema
- **Exemplo:** "31000", "25632"
- **Uso:** Rastreia histórico do candidato e permite joins com dados complementares

### **🔧 COMPATIBILIDADE TÉCNICA (4 colunas)**

**3. `tech_overlap_count` (int8)**
- **Fórmula:** Interseção entre termos técnicos da vaga e do candidato
- **Range:** 0-20 (quantidade de tecnologias em comum)
- **Vocabulário:** SAP, SQL, Python, Java, .NET, AWS, Azure, Docker, etc.
- **Interpretação:** Quanto maior, melhor o match técnico vaga-candidato
- **Exemplo:** Vaga exige "SAP, SQL, Python" + Candidato tem "SAP, Python, Oracle" = 2

**4. `cand_has_sap` (int8 - binária)**
- **Origem:** Busca regex r"\bsap\b" no CV e conhecimentos técnicos do candidato 
- **Valores:** 0 (não tem), 1 (tem SAP)
- **Justificativa:** SAP é competência premium no mercado TI brasileiro
- **Uso:** Feature isolada para capturar especialização SAP independente de overlap geral

**5. `is_sap_vaga` (int8 - binária)**
- **Origem:** Campo "informacoes_basicas.vaga_sap" harmonizado ("Sim"→1, "Não"→0)
- **Valores:** 0 (vaga não-SAP), 1 (vaga SAP)
- **Uso:** Identifica demanda SAP na posição para cruzar com capacidade do candidato

**6. `sap_pair` (int8 - binária)**
- **Fórmula:** `is_sap_vaga` AND `cand_has_sap` 
- **Significado:** Match perfeito SAP (vaga exige E candidato possui)
- **Interpretação:** Feature de interação para capturar sinergia especializada
- **Valor de Negócio:** Candidatos SAP em vagas SAP têm maior probabilidade de contratação

### **🌍 COMPATIBILIDADE DE IDIOMAS (6 colunas)**

**7. `vaga_ing_rank` (Int8)**
- **Origem:** Campo "perfil_vaga.nivel_ingles" mapeado para escala ordinal
- **Mapeamento:** básico=1, intermediário=3, avançado=5, fluente=6, c1=6, c2=7
- **Range:** 0-7 (0 = sem requisito)
- **Uso:** Nível mínimo de inglês exigido na vaga

**8. `cand_ing_rank` (Int8)**
- **Origem:** Campo "formacao_e_idiomas.nivel_ingles" do candidato
- **Mapeamento:** Mesma escala ordinal da vaga
- **Range:** 0-7 (0 = sem inglês)
- **Uso:** Nível de inglês declarado pelo candidato

**9. `vaga_esp_rank` (Int8)**
- **Origem:** Campo "perfil_vaga.nivel_espanhol" 
- **Função:** Análogo ao inglês para mercado latino-americano
- **Range:** 0-7

**10. `cand_esp_rank` (Int8)**
- **Origem:** Campo "formacao_e_idiomas.nivel_espanhol" do candidato
- **Range:** 0-7

**11. `ingles_ok` (int8 - binária)**
- **Fórmula:** `cand_ing_rank` >= `vaga_ing_rank`
- **Significado:** Candidato ATENDE requisito mínimo de inglês
- **Interpretação:** 1 = apto, 0 = insuficiente
- **Valor:** Elimina candidatos que não atendem hard requirement

**12. `espanhol_ok` (int8 - binária)**
- **Fórmula:** `cand_esp_rank` >= `vaga_esp_rank`
- **Função:** Análoga ao inglês para espanhol

### **👔 ANÁLISE DE SENIORIDADE (4 colunas)**

**13. `vaga_sen_rank` (Int8)**
- **Origem:** Campo "perfil_vaga.nivel_profissional" com NLP básico
- **Mapeamento:** júnior=1, pleno=2, sênior=3, outros=0
- **Uso:** Senioridade exigida na vaga

**14. `cand_sen_rank` (Int8)**
- **Origem:** Campo "titulo_profissional" do candidato com busca por palavras-chave
- **Extração:** Busca "sen", "plen", "jun" no título do candidato
- **Limitação:** Baseado em título, não em anos de experiência
- **Range:** 0-3

**15. `senioridade_gap` (Int8)**
- **Fórmula:** `cand_sen_rank` - `vaga_sen_rank`
- **Interpretação:** 
  - Positivo = candidato more senior (over-qualified)
  - Zero = match perfeito
  - Negativo = candidato under-qualified
- **Range:** -3 a +3

**16. `senioridade_ok` (int8 - binária)**
- **Fórmula:** `cand_sen_rank` >= `vaga_sen_rank`
- **Significado:** Candidato tem senioridade suficiente
- **Uso:** Hard requirement para evitar under-qualification

### **⏱️ DINÂMICA DO FUNIL TEMPORAL (2 colunas)**

**17. `days_update` (int16)**
- **Origem:** Diferença entre "data_candidatura" e "ultima_atualizacao" 
- **Range:** 0-365+ dias
- **Interpretação:** 
  - 0 = candidatura recente sem movimento
  - Valores altos = processo longo (pode indicar interesse mútuo ou complexity)
- **Valor:** Velocidade no funil indica engajamento e fit

**18. `situacao_ord` (Int8)**
- **Origem:** Campo "situacao_candidato" mapeado para escala ordinal
- **Mapeamento:** Cadastrado=0, Contato=1, Encaminhado=2, Entrevista=3, Aprovado=4, Contratado=5, Reprovado=6
- **Interpretação:** Posição atual no funil de recrutamento  
- **Uso:** Variável target potencial e feature de posição no processo

### **📊 BINNING E INTERAÇÕES (3 colunas)**

**19. `len_cv_bin` (Int8)**
- **Origem:** Tamanho do CV em caracteres dividido em 4 quartis
- **Range:** 0-3 (quartis)
- **Rationale:** Binning captura relação não-linear entre tamanho CV e experiência
- **Uso:** Proxy para experiência sem usar variável contínua bruta

**20. `ok_eng_sen` (int8 - binária)**
- **Fórmula:** `ingles_ok` AND `senioridade_ok`
- **Significado:** Candidato atende AMBOS requisitos críticos
- **Interpretação:** Feature de interação para capturar efeito combinado
- **Valor:** Candidatos que passam nas duas barreiras têm perfil mais forte

### **📈 NORMALIZAÇÃO (1 coluna)**

**21. `len_cv_pt_z` (float32)**
- **Origem:** Tamanho do CV normalizado via StandardScaler (Z-score)
- **Fórmula:** (len_cv_pt - média) / desvio_padrão
- **Range:** Tipicamente -3 a +3
- **Interpretação:** 
  - 0 = tamanho médio de CV
  - Positivo = CV mais longo que média
  - Negativo = CV mais curto que média
- **Uso:** Versão normalizada para algoritmos sensíveis à escala

## **📋 RESUMO ESTATÍSTICO DO DATASET**

- **Dimensões:** 53.759 registros × 21 colunas
- **Tipos:** 2 chaves string + 19 features numéricas (int8/Int8/int16/float32)
- **Densidade:** ~90%+ de dados válidos pós-tratamento de ausentes
- **Target sugerido:** `situacao_ord` ≥ 5 (Contratado) para classificação binária
- **Granularidade:** Uma linha = um candidato aplicado a uma vaga específica
- **Cobertura temporal:** 2021 baseado nas datas de candidatura dos prospects[3]

## **🎯 APLICAÇÕES PRINCIPAIS**

- **Ranking de candidatos:** Usar probabilidades do modelo para ordenar candidatos por vaga
- **Triagem automática:** Filtros em `ingles_ok`, `senioridade_ok`, `tech_overlap_count`
- **Insights de processo:** Análise de `days_update` e `situacao_ord` por perfil de vaga
- **Matching inteligente:** Features de compatibilidade alimentam algoritmos de recomendação[2][4][3]

