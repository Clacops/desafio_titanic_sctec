# Utilizar o venv : "pip install numpy pandas matplotlib seaborn" e ja deixei pronto o seaborn isnatalado tambem
# Erro: desinstalar tudo e reinstalar tudo de novo, "pip uninstall pandas numpy -" "pip cache purge" reinstalar do zero.
import pandas as pd

#import matplotlib
import matplotlib.pyplot as plt

print("Fase 1 - Tratamento dos Dados")
# dataset
df = pd.read_csv('titanic_dataset.csv')

# Exibindo as primeiras linhas e o resumo estrutural 
print("Visualização inicial dos dados:")
print(df.head())

print("\nVerificação de tipos e nulos:")
print(df.info())

# iniciando a retirada dos nulls
# quantidade de nulos por coluna
print("\nValores nulos por coluna antes do tratamento:")
print(df.isnull().sum())

# Tratando a coluna 'Age' (Idade)
# preencher as idades faltantes com a mediana 
df['Age'] = df['Age'].fillna(df['Age'].median())

#Tratando a coluna 'Embarked' (Embarque)
# Vamos preencher com o porto mais frequente (moda)
porto_frequente = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(porto_frequente)

# Removendo duplicatas - importante!
df = df.drop_duplicates()

# obtive a coluna cabin com muitos nulos = 687, cosniderando que nao possuo os dados optei por preencher os nulos da cabine com uma categoria genérica
df['Cabin'] = df['Cabin'].fillna('Unknown')

print(df.isnull().sum())

print("\nFase 2 - Construindo o primeiro Grafico ")
import matplotlib.pyplot as plt
import seaborn as sns

#  estilo do gráfico
sns.set_theme(style="whitegrid")

# personalizadando as cores para cada sexo
cores_sexo = {'female': 'red', 'male': 'blue'}

# Versão atualizada para evitar o Warning
sns.barplot(x='Sex', y='Survived', data=df, hue='Sex', palette=cores_sexo, legend=False)

#títulos e rótulos
plt.title('Taxa de Sobrevivência por Sexo - Desafio Titanic 2026')
plt.xlabel('Sexo (Feminino vs Masculino)')
plt.ylabel('Proporção de Sobreviventes')

# Salvar como imagem 
plt.savefig('graf1_sobrevivencia_sexo.png')
plt.close() 
# retirei o plt.show() para evitar que o gráfico seja exibido em janelas pop-up, e fica aparecendo warning de loop.

print ("\nFase 3 - análise dos resultados ")
print("\nMédia de sobrevivência por sexo:")
print(df.groupby('Sex')['Survived'].mean())

#  taxa de sobrevivência por Classe (1ª, 2ª e 3ª)
print("\nTaxa de Sobrevivência por Classe:")
print(df.groupby('Pclass')['Survived'].mean())

# sobreviventes X sex X taxa de sobrevivência por Classe (1ª, 2ª e 3ª)

print("\n*Analise relação sex,age e survided*")
print("\nSobrevivência por Sexo e Classe ")
analise_sex_age_survived = df.groupby(['Sex', 'Pclass'])['Survived'].mean()
print(analise_sex_age_survived)

print("\nFase 4  - Grafico 2  - com 3 variaveis")


# personalizar cores para sobrevivência
cores_sobrevivencia = {0: 'red', 1: 'green'}

# Criar figura e o gráfico
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', hue='Sex', data=df, palette=cores_sexo)

plt.title('Distribuição de Idade por Sobrevivência e Sexo')
plt.xlabel('Sobreviveu (0 = Não, 1 = Sim)')
plt.ylabel('Idade')
plt.legend(title='Sexo')

# Salvar 
plt.savefig('graf2_sex_age_survived.png')
plt.close() 
print("\nSegundo gráfico gerado: 'graf2_sex_age_survived.png'")


print("\nFase 5 - analise 4 variaveis - sex, age, class e survived- Boxplot - Outliers e tabela")

#  Limpeza total, tenho tipo experiencias de naoo encerrar totlamente os gráficos, então para evitar qualquer tipo de confusão ou sobreposição, vou garantir que todos os gráficos sejam fechados antes de criar o próximo. Isso é especialmente importante quando se trabalha com múltiplos gráficos em sequência.
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')

# Tabela 
print("\n--- Tabela sobreviventes: Idade Média por Classe, Sexo e Status ---")

df['Status'] = df['Survived'].map({0: 'Não Sobreviveu', 1: 'Sobreviveu'})


tabela_4_variaveis = df.groupby(['Pclass', 'Sex', 'Status'])['Age'].mean().unstack()
print(tabela_4_variaveis)

# Criar Gráfico 
cores_status = {'Sobreviveu': 'green', 'Não Sobreviveu': 'red'}

g = sns.catplot(
    x='Sex', 
    y='Age', 
    hue='Status', 
    col='Pclass', 
    data=df, 
    kind='box', 
    palette=cores_status,
    height=4.5, 
    aspect=0.8,
    showfliers=True #  outliers 
)

g.figure.set_layout_engine('none') 

# Títulos 
g.set_axis_labels("Sexo", "Idade")
g.set_titles("Classe {col_name}")

# Salvar
g.savefig('graf3_surv_sex_age_class.png', bbox_inches='tight')

plt.close('all')
print("\nSucesso! Tabela exibida no terminal e gráfico salvo como 'graf3_sur_sex_age_class.png'")

# Fase 6 -  analise de sobreviventes por embarque, classe e sexo
print(" \nFase 6 - Análise por Porto de Embarque, Classe e Sexo")

# Limpeza preventiva de fechamento de gráficos para evitar sobreposição
plt.close('all')

# Criando o gráfico 
# contagem de sobreviventes (Survived) separada por Porto (col), Classe (row) e Sexo (hue)
g = sns.catplot(
    data=df,
    x="Status", 
    hue="Sex", 
    col="Embarked", 
    row="Pclass",
    kind="count",
    palette="viridis",
    height=3, 
    aspect=1.2
)

# títulos e eixos
g.set_axis_labels("Resultado", "Nº de Passageiros")
g.set_titles("Porto {col_name} | Classe {row_name}")

# Salvar
g.savefig('graf4_embarque_analise.png', bbox_inches='tight')

# Tabela 
print("\n--- Tabela: Sobrevivência por Porto e Classe ---")
tabela_porto = df.groupby(['Embarked', 'Pclass', 'Sex'])['Survived'].mean().unstack()
print(tabela_porto)

plt.close('all')
print("\nGráfico 4 gerado com sucesso!")



