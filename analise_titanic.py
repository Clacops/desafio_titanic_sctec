# criar ambiente virtual : python- m venv venv
# ativar o ambiennte (Windows) : .\venv\Scripts\activate
# instalar bibliotecas : "pip install numpy pandas matplotlib seaborn" e ja deixei pronto o seaborn instalado tambem
# Foi necessario - Erro: desinstalar tudo e reinstalar tudo de novo,
# pip uninstall pandas numpy  - pip cache purge reinstalar do zero.

import pandas as pd
import seaborn as sns
#import numpy as np - programa não utilizou o numpy, então optei por retirar a importação para deixar o código mais limpo.
import matplotlib
matplotlib.use('Agg') # evita comp travar por loop de gráficos, e permite salvar os gráficos sem exibir janelas pop-up, 
#o que é ideal para scripts que geram múltiplos gráficos em sequência.
import matplotlib.pyplot as plt


print("Fase 1 - Tratamento dos Dados")
 
# buscando o dataset no Drive, e lendo o arquivo CSV
df = pd.read_csv('titanic_dataset.csv')

# Exibir as primeiras 5 linhas e o resumo estrutural 
print("Visualização inicial dos dados:")
print(df.head()) 


print("\nVerificação de tipos e nulos:")
print(df.info()) #estrutura dataset, tipos de dados e contagem de nulos por coluna
#aqui percebi que a coluna 'Cabin' tem muitos nulos, e a coluna 'Age' tem 177 nulos, e 'Embarked' tem 2 nulos.

# iniciando a retirada dos nulls

print("\nTabela 1 - Valores nulos por coluna antes do tratamento:")
print("="*90)
print(df.isnull().sum())


# Tratando a coluna 'Age' (Idade)
# preencher as idades faltantes com a mediana 
df['Age'] = df['Age'].fillna(df['Age'].median()) 
# manter as 177 idades faltantes com a utilização da mediana para evitar distorção causada por outliers
# valores muitos extremos estarao fora da distribuição, mais realista possível.

# Tratando a coluna 'Embarked' (Embarque)
# preencher com o porto mais frequente (moda)
porto_frequente = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(porto_frequente)

# Removendo duplicatas - registro unicos -importante
#df = df.drop_duplicates()

# obtive a coluna cabin com muitos nulos = 687, cosniderando que nao possuo os dados,
# optei por preencher os nulos da cabine com uma categoria genérica de "Unknown" - nao diminuir a amostra 
df['Cabin'] = df['Cabin'].fillna('Unknown')

print("\n" + "="*90)
print("\nTabela 2 - Confirmação de nulos")
print(df.isnull().sum())
print("\n" + "="*90)

print("\nFase 2 - Construindo a Tabela 3 e  Primeiros Gráficos ") #age e survivers em taxas % e ns absolutos

#print(df.groupby('Sex')['Survived'].mean()) optei por utilizar numeros absultos e taxa na mesma tabela

# Agrupar 'Sex' / contagem total e a média de sobrevivência relacao entre toral masculino e feminino prorporcional a sobreviventes
tabela_sexo = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']) # acrescentado agg pois travando muito o código, e com agg ficou 
#mais fluida a execução, e consegui calcular as 3 métricas em um único passo.

tabela_sexo.columns = ['Total Passageiros', 'Total Sobreviventes', 'Taxa de Sobrevivência']

tabela_sexo['Taxa de Sobrevivência'] = tabela_sexo['Taxa de Sobrevivência'] * 100 # ajustar casas decimais

print("Tabela 3: Sobrevivência por Sexo")
print ("Média de sobrevivência por sexo: Números Absolutos e Percentuais")

print(tabela_sexo)
print("\n" + "="*90)

#criando o grafico de barras
#  estilo do gráfico
sns.set_theme(style="whitegrid") 

# personalizadando as cores para cada sexo
cores_sexo = {'female': 'red', 'male': 'blue'}

# Versão atualizada seaborn q solicita o hue para evitar o Warning  (depois de muitos warnings!!!)
# hue para diferenciar as barras por sexo,  legend=False para evitar legenda redundante.
sns.barplot(x='Sex', y='Survived', data=df, hue='Sex', palette=cores_sexo, legend=False)

#títulos e rótulos 
plt.title('Taxa de Sobrevivência por Sexo - Desafio Titanic 2026')
plt.xlabel('Sexo (Feminino vs Masculino)')
plt.ylabel('Proporção de Sobreviventes')

# Salvar como imagem 
plt.savefig('graf1_taxa_sobreviventes_sexo.png')
plt.close() 
# retirei o plt.show() para evitar que o gráfico seja exibido em janelas pop-up, e fica aparecendo warning de loop.

# Percebi que poderia ser visualemtne melhor utiizar taxas + dados absolutos no grafico - graf1A o
plt.figure(figsize=(8, 6))

# gráfico de barras (que mostra a média/taxa de sobrevivência)
ax = sns.barplot(x='Sex', y='Survived', data=df, hue='Sex', palette=cores_sexo, legend=False)

#  valores absolutos para colocar no topo das barras
# sobreviventes em cada grupo / contagem /sex/survived

# Percebi que poderia ser visualemtne melhor utiizar taxas + dados absolutos no grafico - graf1A o
plt.figure(figsize=(8, 6))

# gráfico de barras (que mostra a média/taxa de sobrevivência)
ax = sns.barplot(x='Sex', y='Survived', data=df, hue='Sex', palette=cores_sexo, legend=False)

# valores absolutos para colocar no topo das barras
# sobreviventes em cada grupo / contagem /sex/survived

# CCORREÇÂO!!! grafico com valores invertidos female/male, entao para garantir que 
# os valores absolutos correspondam a barra correta, utilizei o método value_counts() filtrando apenas os sobreviventes (Survived == 1) e contando por sexo. Assim, a contagem de #
# sobreviventes para cada sexo estará na ordem correta para ser associada às barras do gráfico.
contagem_sobreviventes = df[df['Survived'] == 1]['Sex'].value_counts()

# ordem exata das categorias que o Seaborn usou no eixo X
# ax.get_xticklabels() retorna objetos de texto, usamos .get_text() para pegar a string ('male', 'female')
ordem_categorias = [label.get_text() for label in ax.get_xticklabels()]

# Iterar sobre as barras E a ordem ao mesmo tempo usando zip
# ax.patches contém os retângulos (barras) na ordem do desenho
for p, sexo_barra in zip(ax.patches, ordem_categorias):
    
    # quantidade correta baseada no sexo que corresponde a ESTA barra
    qtd = contagem_sobreviventes[sexo_barra]
    
    # Pegar a porcentagem (altura da barra * 100)
    porcentagem = p.get_height() * 100
    
    # Criar o rótulo "Qtd vivos (Porcentagem%)"
    # f-string para formatar (:.1f para uma casa decimal)
    rotulo_texto = f'{qtd} vivos ({porcentagem:.1f}%)'
    
    # Adicionar o texto (rótulo) no topo de cada barra
    ax.annotate(rotulo_texto, 
                # Coordenadas: (meio da largura da barra, altura da barra)
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center',       # Alinhamento horizontal: centro
                va='baseline',     # Alinhamento vertical: base
                fontsize=11, fontweight='bold', color='black', 
                xytext=(0, 14),    # Afastar 14 pontos para cima (y)
                textcoords='offset points') # Usar pontos como referência de afastamento

plt.title('Análise de Sobrevivência: Total Absoluto e Taxa Relativa por Sexo')
plt.ylabel('Taxa de Sobrevivência (0.0 a 1.0)')
plt.ylim(0, 1.1) 
plt.savefig('graf1A_n.absoluto_taxa_sobreviventes_sexo.png', dpi=200) 
plt.close()

print("\nTabela 4: Estatísticas de Idade por Sobrevivência e Sexo")

# Gerando média, mediana, mínimo e máximo de idade para cada grupo
tabela_idade_stats = df.groupby(['Survived', 'Sex'])['Age'].agg(['mean', 'median', 'min', 'max']).round(1)

# Renomeando as colunas para o português
tabela_idade_stats.columns = ['Média de Idade', 'Mediana', 'Idade Mín.', 'Idade Máx.']

# Ajustando o índice para ficar legível
tabela_idade_stats.index = [
    ('Não Sobreviveu', 'Feminino'), ('Não Sobreviveu', 'Masculino'),
    ('Sobreviveu', 'Feminino'), ('Sobreviveu', 'Masculino')
]

print(tabela_idade_stats)
print("="*90)   

print("\nFase 3 - Construindo Tabela 5 e Gráfico 2 - Sobrevivência por Classe")
#  taxa de sobrevivência por Classe (1ª, 2ª e 3ª)


# sobreviventes / sex / taxa de sobrevivência por Classe (1ª, 2ª e 3ª)
tabela_classe = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])

# Renomear as colunas 
tabela_classe.columns = ['Total Passageiros', 'Total Sobreviventes', 'Taxa de Sobrevivência (%)']

# Convertendo a média para porcentagem com uma casa decimal
tabela_classe['Taxa de Sobrevivência (%)'] = (tabela_classe['Taxa de Sobrevivência (%)'] * 100).round(1)

# garantir que a 1ª classe apareça no topo
tabela_classe = tabela_classe.sort_index()

print("\nTabela 5: Análise de Sobrevivência por Classe Social")

print(tabela_classe)
print("\n" + "="*90)

# personalizar cores para sobrevivência
cores_sobrevivencia = {0: 'red', 1: 'green'}

# Criar figura e o gráficopython- m venv venv
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', hue='Sex', data=df, palette=cores_sexo)

plt.title('Distribuição de Idade por Sobrevivência e Sexo')
plt.xlabel('Sobreviveu (0 = Não, 1 = Sim)')
plt.ylabel('Idade')
plt.legend(title='Sexo')



# Salvar 
plt.savefig('graf2_sex_age_survived.png')
plt.close() 
#  Limpeza total, tenho tipo experiencias de nao encerrar totlamente os gráficos, então para evitar qualquer tipo de confusão ou sobreposição, vou garantir que todos os gráficos sejam fechados antes de criar o próximo. Isso é especialmente importante quando se trabalha com múltiplos gráficos em sequência.

print("\nSegundo gráfico gerado: 'graf2_sex_age_survived.png'")
print('\n' + "="*90)

print("\nFase 4 - Construindo gráfico 3 - Sobrevivência por Sexo, Idade e Classe")
# - Boxplot - Outliers
cores_status = {'Sobreviveu': 'green', 'Não Sobreviveu': 'red'}
if 'Status' not in df.columns:
    df['Status'] = df['Survived'].map({0: 'Não Sobreviveu', 1: 'Sobreviveu'})

plt.close('all')

# criar grafico 3
g = sns.catplot(
    x='Sex', y='Age', hue='Status', col='Pclass', 
    data=df, kind='box', palette=cores_status,
    height=4, aspect=0.7, showfliers=True
)

# bloquear calc layout (Isso evita o travamento)
g.figure.set_layout_engine(None) 

# ajustes para grafico3 - sem bordas - sem o bbox_inches)
g.savefig('graf3_surv_sex_age_class.png', dpi=100) # dpi menor para ser mais rápido
plt.close('all')
print("Gráfico 3 salvo.")
print("="*90)

# Teste: Mediana apenas de quem já tinha idade preenchida no CSV todas estao iguais a 28 e sugere que a adequacao nos 20% (177)
#print(df.dropna(subset=['Age']).groupby(['Survived', 'Sex'])['Age'].median())

print("\nFase 5: Construindo Tabela 6 - analise embarcados, classe, sobreviventes, sex, age") # tabela_analise_porto_detalhada.png

print("\n Tabela 6 - analise embarcados, classe, sobreviventes, sex")

#Agrupar Base (Médias e Somas)
resumo_pct = (df.groupby(['Embarked', 'Pclass', 'Sex'])['Survived'].mean() * 100).unstack()
resumo_sobreviventes = df.groupby(['Embarked', 'Pclass', 'Sex'])['Survived'].sum().unstack()

total_qtd_mulheres = resumo_sobreviventes['female'].sum()
total_qtd_homens = resumo_sobreviventes['male'].sum()

# Média ponderada real do que está exibido
total_pct_mulheres = (total_qtd_mulheres / df[df['Sex'] == 'female']['PassengerId'].count()) * 100
total_pct_homens = (total_qtd_homens / df[df['Sex'] == 'male']['PassengerId'].count()) * 100

# Inseri a linha de Total Geral
resumo_pct.loc[('Geral', 'Total'), :] = [total_pct_mulheres, total_pct_homens]
resumo_sobreviventes.loc[('Geral', 'Total'), :] = [total_qtd_mulheres, total_qtd_homens]

# visualização final para terminal e imagem !!!!!!
resumo_visual = resumo_pct.round(1).astype(str) + "% (" + resumo_sobreviventes.astype(int).astype(str) + ")"

# Formatando os nomes das linhas (Índices)
novos_indices = []
for porto, classe in resumo_visual.index:
    if porto == 'Geral':
        novos_indices.append("TOTAL GERAL DO NAVIO")
    else:
        n_porto = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}.get(porto, porto)
        novos_indices.append(f"Porto: {n_porto} | Classe: {int(classe)}")

resumo_visual.index = novos_indices
resumo_visual.columns = ['Sobrev. Mulheres (Taxa e Qtd)', 'Sobrev. Homens (Taxa e Qtd)']

# EXIBIÇÃO NO TERMINAL

print(resumo_visual)
print("="*90)

# imagem tabela 6
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')
tb = ax.table(
    cellText=resumo_visual.values, 
    rowLabels=resumo_visual.index, 
    colLabels=resumo_visual.columns, 
    cellLoc='center', loc='center',
    colColours=['#2c3e50', '#2c3e50']
)

tb.auto_set_font_size(False)
tb.set_fontsize(10)
tb.scale(1.2, 2.5)

# Branco no cabeçalho e destaque no Total Geral
for (row, col), cell in tb.get_celld().items():
    if row == 0: 
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
    if row == len(resumo_visual): # Linha do Total
        cell.set_facecolor('#ecf0f1')
        cell.get_text().set_weight('bold')
    if col == -1: 
        cell.get_text().set_weight('bold')

plt.subplots_adjust(left=0.25)
plt.savefig('tabela6_analise_porto_class_age.png', dpi=150)
plt.close('all')

# Verificando a classe dos homens de Queenstown
homens_q = df[(df['Embarked'] == 'Q') & (df['Sex'] == 'male')]

print("\nTabela 7 - Distribuição de Classe dos Homens em Queenstown:")
print(homens_q['Pclass'].value_counts())

#queenstown teve 77 homens, e a maioria estava na 3ª classe

print("\nTotal de Homens Sobreviventes em Queenstown:")
print(homens_q['Survived'].sum())

# Homens que embarcaram em Queenstown (Porto Q) 
print("\n" + "="*90)
print("Análise de homens que embarcaram em Queenstown (Porto Q) - Sobreviventes e Classe")

# Filtrando apenas homens de Queenstown
homens_q = df[(df['Embarked'] == 'Q') & (df['Sex'] == 'male')]

# Criando a comparação entre Sobreviventes e Não Sobreviventes
comparativo_q = homens_q.groupby('Survived').agg({
    'PassengerId': 'count',
    'Age': ['mean', 'median', 'std'],
    'Pclass': lambda x: x.mode()[0] # Mostra a classe mais comum
})

# Renomeando colunas para clareza
comparativo_q.columns = ['Total de Homens', 'Média Idade', 'Mediana Idade', 'Desvio Padrão', 'Classe Predominante']
comparativo_q.index = ['Não Sobreviveu', 'Sobreviveu']

print(comparativo_q)
print("="*90)

# Verificação de Classe (Por que morreram tanto?)
print("\nDistribuição por Classe dos Homens em Queenstown (Total):")
print(homens_q['Pclass'].value_counts())
print("\n✅ PROJETO FINALIZADO! Todas as tabelas foram exibidas no terminal e salvas como imagem.")





