import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

# 1. Charger le CSV nettoyé
df = pd.read_csv("adult_clean.csv")

# 2. Connexion MySQL
connexion = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)
# cursor() est un objet qui permet d'envoyer des commandes à MySQL
cursor = connexion.cursor()

# 3. Parcourir les lignes du DataFrame (le "_" pour signifier qu'il n'y pas besoin de l'index)
for _, row in df.iloc[5:].iterrows():

    # Insertion dans personne
    sql_personne = """
        INSERT INTO personne (
            age,
            race,
            sex,
            native_country,
            fnlwgt,
            education,
            education_num,
            marital_status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
# %s sont des places holders a qui le connecteur va associer dans l'ordre les valeurs du second argument de " cursor.execute(sql_personne, values_personne)"

    values_personne = (
        int(row["age"]),
        row["race"],
        row["sex"],
        row["native_country"],
        int(row["fnlwgt"]),
        row["education"],
        int(row["education_num"]),
        row["marital_status"]
    )
    # modification apportée à MySQL ici 
    cursor.execute(sql_personne, values_personne)

    # Récupère l'id généré automatiquement
    id_pers = cursor.lastrowid

    # Insertion dans situation
    sql_situation = """
        INSERT INTO situation (
            relationship,
            class,
            workclass,
            occupation,
            capital_gain,
            capital_loss,
            hours_per_week,
            id_pers
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    values_situation = (
        row["relationship"],
        row["class"],
        row["workclass"],
        row["occupation"],
        int(row["capital_gain"]),
        int(row["capital_loss"]),
        int(row["hours_per_week"]),
        id_pers
    )

    cursor.execute(sql_situation, values_situation)

# 4. Validation des insertions 
connexion.commit()

print(f"{len(df)} lignes chargées dans MySQL.")

# on ferm la connexion et le curseur pour libérer les ressources associées
cursor.close()
connexion.close()


