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
print("="*80)
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
df = df.drop_duplicates()

# obtive a coluna cabin com muitos nulos = 687, cosniderando que nao possuo os dados,
# optei por preencher os nulos da cabine com uma categoria genérica de "Unknown" - nao diminuir a amostra 
df['Cabin'] = df['Cabin'].fillna('Unknown')

print("\n" + "="*80)
print("\nTabela 2 - Confirmação de nulos")
print(df.isnull().sum())
print("\n" + "="*80)

print("\nFase 2 - Construindo a Tabela 3 e Primeiro Gráfico ") #age e survivers em taxas % e ns absolutos
print ("\nMédia de sobrevivência por sexo: Números Absolutos e Percentuais")


#print(df.groupby('Sex')['Survived'].mean()) optei por utilizar numeros absultos e taxa na mesma tabela

# Agrupar 'Sex' / contagem total e a média de sobrevivência relacao entre toral masculino e feminino prorporcional a sobreviventes
tabela_sexo = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']) # acrescentado agg pois travando muito o código, e com agg ficou 
#mais fluida a execução, e consegui calcular as 3 métricas em um único passo.

tabela_sexo.columns = ['Total Passageiros', 'Total Sobreviventes', 'Taxa de Sobrevivência']

tabela_sexo['Taxa de Sobrevivência'] = tabela_sexo['Taxa de Sobrevivência'] * 100 # ajustar casas decimais

print("Tabela 3: Sobrevivência por Sexo")
print(tabela_sexo)
print("\n" + "="*80)

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
plt.savefig('graf1_taxa_sobrevivencia_sexo.png')
plt.close() 
# retirei o plt.show() para evitar que o gráfico seja exibido em janelas pop-up, e fica aparecendo warning de loop.

# Percebi que poderia ser visualemtne melhor utiizar taxas + dados absolutos no grafico - graf1A o
plt.figure(figsize=(8, 6))

# gráfico de barras (que mostra a média/taxa de sobrevivência)
ax = sns.barplot(x='Sex', y='Survived', data=df, hue='Sex', palette=cores_sexo, legend=False)

#  valores absolutos para colocar no topo das barras
# sobreviventes em cada grupo / contagem /sex/survived
contagem = df[df['Survived'] == 1]['Sex'].value_counts()

# rótulos (Texto) no topo de cada barra
for i, p in enumerate(ax.patches): # condicional enumerate = iterar sobre as barras do gráfico
    # nome do sexo baseado na posição da barra
    sexo = 'female' if i == 0 else 'male' 
    qtd = contagem[sexo]
    porcentagem = p.get_height() * 100
    
    # "Qtd Sobreviventes (Porcentagem%)"
    ax.annotate(f'{qtd} vivas ({porcentagem:.1f}%)', #uma casa decimal
                (p.get_x() + p.get_width() / 2., p.get_height()), # coordenadas altura e largura da barra
                ha='center', va='baseline', 
                fontsize=11, fontweight='bold', color='black', xytext=(0, 14), 
                textcoords='offset points')

plt.title('Análise de Sobrevivência: Total Absoluto e Taxa Relativa por Sexo')
plt.ylabel('Taxa de Sobrevivência (0.0 a 1.0)')
plt.ylim(0, 1.1) 
plt.savefig('graf1A_n.absoluto_taxa_sobreviventes_sexo.png', dpi=200) 
#mantive o grafico de taxa (%) pq ja estava pronto e foi aprendizado para este
plt.close()

print("\nFase 3 - Construindo Tabela 4 e Gráfico 2 - Sobrevivência por Classe")
#  taxa de sobrevivência por Classe (1ª, 2ª e 3ª)

# sobreviventes / sex / taxa de sobrevivência por Classe (1ª, 2ª e 3ª)
tabela_classe = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])

# Renomear as colunas 
tabela_classe.columns = ['Total Passageiros', 'Total Sobreviventes', 'Taxa de Sobrevivência (%)']

# Convertendo a média para porcentagem com uma casa decimal
tabela_classe['Taxa de Sobrevivência (%)'] = (tabela_classe['Taxa de Sobrevivência (%)'] * 100).round(1)

# garantir que a 1ª classe apareça no topo
tabela_classe = tabela_classe.sort_index()

print("\nTabela 4: Análise de Sobrevivência por Classe Social")
print("\n" + "="*80)
print(tabela_classe)
print("\n" + "="*80)

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
print("="*80)

print("\nFase 5: Construindo tabela 5 - analise embarcados, sobreviventes, sex, age") # tabela_analise_porto_detalhada.png

# Agruparl (sempre partindo do df para evitar erros de repetição)
resumo_pct = (df.groupby(['Embarked', 'Pclass', 'Sex'])['Survived'].mean().unstack() * 100).round(1)
resumo_qtd = df.groupby(['Embarked', 'Pclass', 'Sex'])['Survived'].count().unstack()

# Criar a visualização dos dados (Taxa + Quantidade entre parênteses)
resumo_visual = resumo_pct.astype(str) + "% (" + resumo_qtd.astype(str) + ")"

# transformar indices para nomes mais amigáveis
novos_indices = []
for porto, classe in resumo_visual.index:
    nome_porto = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}.get(porto, porto)
    novos_indices.append(f"Porto: {nome_porto} | Classe: {int(classe)}")

# Aplicar os novos nomes 
resumo_visual.index = novos_indices
resumo_visual.columns = ['Taxa Sobrev. Mulheres', 'Taxa Sobrev. Homens']
resumo_visual.index.name = 'Origem e Categoria Social'

# Exibir no terminal 
print("\nTabela 5 - análise embarcados, sobreviventes, sex, age")
#print(resumo_visual)
#nomes das colunas - faltava para o terminal
resumo_visual.columns = ['Taxa Sobrev. Mulheres', 'Taxa Sobrev. Homens']

# nomes dos portos e classes 

novos_indices = []
for porto, classe in resumo_pct.index: 
    nome_porto = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}.get(porto, porto)
    novos_indices.append(f"Porto: {nome_porto} | Classe: {int(classe)}")

# Aplicar nomes ao índice e  título da primeira coluna
resumo_visual.index = novos_indices
resumo_visual.index.name = 'Origem e Categoria Social'

# exibir tabela
print( "="*80)
print(resumo_visual)
print("\n" + "="*80)
# criar imagem tabela

plt.close('all')
fig, ax = plt.subplots(figsize=(16, 10)) # Aumentamos o "papel" da imagem para caber tudo
ax.axis('off')

# Criar a tabela com as cores de cabeçalho profissionais
tb = ax.table(
    cellText=resumo_visual.values, 
    rowLabels=resumo_visual.index, 
    colLabels=resumo_visual.columns, 
    cellLoc='center', 
    loc='center',
    colColours=['#2c3e50', '#2c3e50'] # Azul escuro
)

# Estilização para ficar legível
tb.auto_set_font_size(False)
tb.set_fontsize(11)
tb.scale(1.2, 2.5) # Dá altura às linhas para o texto não ficar "espremido"

# Pintar o texto do cabeçalho de branco e colocar negrito
for (row, col), cell in tb.get_celld().items():
    if row == 0: # Cabeçalho
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
    if col == -1: # Nomes das linhas (Portos/Classes)
        cell.get_text().set_weight('bold')
        cell.set_facecolor('#ffffff') # Fundo branco para garantir leitura

plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)

# salvar
plt.savefig('tabela4_analise_porto_detalhada.png', dpi=150)
plt.close('all')

print("\n✅ PROJETO FINALIZADO COM SUCESSO! Verifique os arquivos .png na sua pasta.")




