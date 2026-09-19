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
from pydantic import BaseModel
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

class PersonneCreate(BaseModel):
    age: int
    race: str
    sex: str
    native_country: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str

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

    if personne is None:
        raise HTTPException(
            status_code=404,
            detail="Personne non trouvée"
        )
    return personne


@app.post("/personnes")
def create_personne(personne: PersonneCreate):

    connexion = get_connexion()
    cursor = connexion.cursor()

    cursor.execute("""
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
    """, (
        personne.age,
        personne.race,
        personne.sex,
        personne.native_country,
        personne.fnlwgt,
        personne.education,
        personne.education_num,
        personne.marital_status
    ))

# validation du insert via commit
    connexion.commit()

# récupère l'id auto-incrémenter que mysql vient de générer 

    id_pers = cursor.lastrowid

    cursor.close()
    connexion.close()

    return {
        "message": "Personne créée",
        "id_pers": id_pers
    }

@app.put("/personnes/{id_pers}")
def update_personne(id_pers: int, personne: PersonneCreate):

    connexion = get_connexion()
    cursor = connexion.cursor()

    cursor.execute("""
        UPDATE personne
        SET age = %s,
            race = %s,
            sex = %s,
            native_country = %s,
            fnlwgt = %s,
            education = %s,
            education_num = %s,
            marital_status = %s
        WHERE id_pers = %s
    """, (
        personne.age,
        personne.race,
        personne.sex,
        personne.native_country,
        personne.fnlwgt,
        personne.education,
        personne.education_num,
        personne.marital_status,
        id_pers
    ))

    connexion.commit()

# je cherche si une personne a été modifiée
    if cursor.rowcount == 0:
        cursor.close()
        connexion.close()

        raise HTTPException(
            status_code=404,
            detail="Personne non trouvée"
        )


    cursor.close()
    connexion.close()

    return {"message": "Personne modifiée"}



@app.delete("/personnes/{id_pers}")
def delete_personne(id_pers: int):

    connexion = get_connexion()
    cursor = connexion.cursor()

    try:
        cursor.execute("""
            DELETE FROM personne
            WHERE id_pers = %s
        """, (id_pers,))

        connexion.commit()

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Personne non trouvée"
            )

    except mysql.connector.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="Impossible de supprimer cette personne : une situation lui est associée."
        )

    finally:
        cursor.close()
        connexion.close()

    return {"message": "Personne supprimée"}

# démarrage du serveur web pour exécuter la variable app
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )




