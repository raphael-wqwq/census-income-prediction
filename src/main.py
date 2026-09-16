from sklearn.datasets import fetch_openml
data = fetch_openml(data_id=43898, as_frame=True)

df = data.frame

# création csv original

df.to_csv("adult_brut.csv", index=False)


import uvicorn
import os
import mysql.connector
from fastapi import FastAPI
from dotenv import load_dotenv

# création de l'application
app = FastAPI()

# Chargement des variables du fichier .env
load_dotenv()

# endpoint d'accueil 
@app.get("/")
def accueil():
    return {"message": "API Census Income"}


@app.get("/personnes")
def get_personnes():

    connexion = get_connexion()

    cursor = connexion.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM personne
        LIMIT 10
    """)

    personnes = cursor.fetchall()

    cursor.close()
    connexion.close()

    return personnes

# Connexion à MySQL
def get_connexion():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )


# démarrage du serveur web pour exécuter la variable app
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )




