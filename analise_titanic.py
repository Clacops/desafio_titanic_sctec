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

# Salvando e FECHANDO para liberar memória
plt.savefig('graf2_sex_age_survived.png')
plt.close() 
print("\nSegundo gráfico gerado: 'graf2_sex_age_survived.png'")

print("\nFase 5 - analise 4 variaveis - sex, age, class e survived")

print(" \nFase 5 - Análise 4 variáveis - Sex, Age, Class e Survived")

# 1. Criando rótulos amigáveis para o gráfico
df['Status'] = df['Survived'].map({0: 'Não Sobreviveu', 1: 'Sobreviveu'})


# 2. Ranking numérico (O uso do sort_values que você queria)
print("\n--- Idade Média dos Sobreviventes (Ordenado por Idade) ---")
# Filtramos apenas quem sobreviveu e agrupamos
analise_vivos = df[df['Survived'] == 1].groupby(['Pclass', 'Sex'])['Age'].mean().sort_values(ascending=False)
print(analise_vivos)

# 3. Criação do Gráfico 3 - Versão Estabilizada
cores_status = {'Sobreviveu': 'green', 'Não Sobreviveu': 'red'}

# Usamos o 'kind=bar' e 'errorbar=None' para garantir que não trave
g = sns.catplot(
    x='Sex', 
    y='Age', 
    hue='Status', 
    col='Pclass', 
    data=df, 
    kind='bar', 
    palette=cores_status,
    errorbar=None, # Remove cálculos pesados de erro
    height=5, 
    aspect=0.8
)

# Ajustando títulos e eixos de forma profissional
g.set_axis_labels("Sexo", "Idade Média")
g.set_titles("Classe {col_name}")

# 4. Salvando o resultado final
plt.savefig('graf3_4_variaveis.png', bbox_inches='tight')
plt.close('all') # Limpa a memória para encerrar com segurança

print("\nSucesso absoluto! O gráfico final foi salvo como 'graf3_4_variaveis.png'")
