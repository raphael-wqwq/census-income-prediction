USE census_income;

CREATE TABLE personne (
    id_pers INT AUTO_INCREMENT PRIMARY KEY,
    age INT,
    race VARCHAR(50),
    sex VARCHAR(20),
    native_country VARCHAR(100),
    fnlwgt INT,
    education VARCHAR(100),
    education_num INT,
    marital_status VARCHAR(100)
);

CREATE TABLE situation (
    id_situation INT AUTO_INCREMENT PRIMARY KEY,
    relationship VARCHAR(100),
    class VARCHAR(20),
    workclass VARCHAR(100),
    occupation VARCHAR(100),
    capital_gain INT,
    capital_loss INT,
    hours_per_week INT,
    id_pers INT NOT NULL,

    CONSTRAINT fk_situation_personne
        FOREIGN KEY (id_pers)
        REFERENCES personne(id_pers)
);