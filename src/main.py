from sklearn.datasets import fetch_openml
data = fetch_openml(data_id=43898, as_frame=True)

df = data.frame

# création csv original

df.to_csv("adult_brut.csv", index=False)


import uvicorn
import os
import mysql.connector
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# création de l'application
app = FastAPI()

# Chargement des variables du fichier .env
load_dotenv()

# Connexion à MySQL
def get_connexion():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )


# endpoint d'accueil 
@app.get("/")
def accueil():
    return {"message": "API Census Income"}


@app.get("/personnes")
def get_personnes():

# objet pour communiquer avec mysql
    connexion = get_connexion()

# cursor permet  d'envoyer des commandes sql
    cursor = connexion.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM personne
        LIMIT 10
    """)

# recuperer toutes les lignes de-u résultat de la requête
    personnes = cursor.fetchall()

    cursor.close()
    connexion.close()

    return personnes

@app.get("/personnes/{id_pers}")
def get_personne(id_pers: int):

    connexion = get_connexion()
    cursor = connexion.cursor(dictionary=True)

# mysqlconnector fait le lien entrre la requête et la valeur saisie (id_pers) pour la placer sur le placeholder %s
    cursor.execute("""
        SELECT *
        FROM personne
        WHERE id_pers = %s
    """, (id_pers,))

    personne = cursor.fetchone()

    cursor.close()
    connexion.close()

    return personne



# démarrage du serveur web pour exécuter la variable app
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )




