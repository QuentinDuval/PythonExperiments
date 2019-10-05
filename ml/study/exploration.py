"""
Statistics of INSEE 2018:
https://www.insee.fr/fr/statistiques/4191029

COURED;Individu en couple dans le ménage au trimestre courant;1;Oui;CHAR;1
COURED;Individu en couple dans le ménage au trimestre courant;2;Non;CHAR;1

ENFRED;Individu avec au moins un enfant dans le ménage ou en garde alternée, au trimestre courant;1;Oui;CHAR;1
ENFRED;Individu avec au moins un enfant dans le ménage ou en garde alternée, au trimestre courant;2;Non;CHAR;1

SEXE;Sexe;1;Masculin;CHAR;1
SEXE;Sexe;2;Féminin;CHAR;1

ACTEU;Statut d'activité au sens du Bureau International du Travail (BIT) selon l'interprétation communautaire;;Sans objet (ACTEU non renseigné, individus de 15 ans et plus nécessairement non pondérés);CHAR;1
ACTEU;Statut d'activité au sens du Bureau International du Travail (BIT) selon l'interprétation communautaire;1;Actif occupé;CHAR;1
ACTEU;Statut d'activité au sens du Bureau International du Travail (BIT) selon l'interprétation communautaire;2;Chômeur;CHAR;1
ACTEU;Statut d'activité au sens du Bureau International du Travail (BIT) selon l'interprétation communautaire;3;Inactif;CHAR;1

TYPMEN7;Type de ménage (7 postes);1;Ménage d'une seule personne;CHAR;1
TYPMEN7;Type de ménage (7 postes);2;Famille monoparentale;CHAR;1
TYPMEN7;Type de ménage (7 postes);3;Couple sans enfant;CHAR;1
TYPMEN7;Type de ménage (7 postes);4;Couple avec enfant(s);CHAR;1
TYPMEN7;Type de ménage (7 postes);5;Ménage de plusieurs personnes ayant toutes un lien de parenté avec la personne de référence du ménage, ni couple, ni famille monoparentale ;CHAR;1
TYPMEN7;Type de ménage (7 postes);6;Ménage de plusieurs personnes n'ayant pas toutes un lien de parenté avec la personne de référence du ménage, ni couple, ni famille monoparentale ;CHAR;1
TYPMEN7;Type de ménage (7 postes);9;Autres ménages complexes de plus d'un personne;CHAR;1

ANCCHOM;Ancienneté de chômage en 8 postes;;Sans objet (ACTEU distinct de 2) ou non renseigné;CHAR;1
ANCCHOM;Ancienneté de chômage en 8 postes;1;Moins d'un mois;CHAR;1
ANCCHOM;Ancienneté de chômage en 8 postes;2;De 1 mois à moins de 3 mois;CHAR;1
ANCCHOM;Ancienneté de chômage en 8 postes;3;De 3 mois à moins de 6 mois;CHAR;1
ANCCHOM;Ancienneté de chômage en 8 postes;4;De 6 mois à moins d'un an;CHAR;1
ANCCHOM;Ancienneté de chômage en 8 postes;5;De 1 an à moins d'un an et demi;CHAR;1
ANCCHOM;Ancienneté de chômage en 8 postes;6;De 1 an et demi à moins de 2 ans;CHAR;1
ANCCHOM;Ancienneté de chômage en 8 postes;7;De 2 ans à moins de 3 ans;CHAR;1
ANCCHOM;Ancienneté de chômage en 8 postes;8;3 ans ou plus;CHAR;1

ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);;Sans objet (personnes non actives occupées, travailleurs informels et travailleurs intérimaires,en activité temporaire ou d'appoint) ou non renseigné;CHAR;1
ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);1;Moins d'un an;CHAR;1
ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);2;De 1 an à moins de 5 ans;CHAR;1
ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);3;De 5 ans à moins de 10 ans;CHAR;1
ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);4;10 ans ou plus;CHAR;1

QPRC;Position professionnelle dans l'emploi principal;;Sans objet (personnes non actives occupées et actifs occupés exerçant pour le compte d'un particulier ou de plusieurs employeurs sans principal) ;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;1;Manoeuvre ou ouvrier spécialisé;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;2;Ouvrier qualifié ou hautement qualifié, technicien d'atelier;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;3;Technicien;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;4;Employé de bureau, employé de commerce, personnel de services, personnel de catégorie C dans la fonction publique ;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;5;Agent de maîtrise, maîtrise administrative ou commerciale, VRP (non cadre), personnel de catégorie B dans la fonction publique ;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;6;Ingénieur, cadre (à l'exception des directeurs ou de leurs adjoints directs) personnel de catégorie A dans la fonction publique ;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;7;Directeur général, adjoint direct;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;8;Autre;CHAR;1
QPRC;Position professionnelle dans l'emploi principal;9;Non renseigné;CHAR;1

DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);;Sans objet (personnes non actives occupées);CHAR;1
DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);1;Temps partiel de moins de 15 heures;CHAR;1
DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);2;Temps partiel de 15 à 29 heures;CHAR;1
DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);3;Temps partiel de 30 heures ou plus;CHAR;1
DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);4;Temps complet de moins de 30 heures;CHAR;1
DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);5;Temps complet de 30 à 34 heures;CHAR;1
DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);6;Temps complet de 35 à 39 heures;CHAR;1
DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);7;Temps complet de 40 heures ou plus;CHAR;1
DUHAB;Type d'horaires de travail (temps complet ou temps partiel en tranches);9;Pas d'horaire habituel ou horaire habituel non déclaré;CHAR;1

ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);;Sans objet (personnes non actives occupées, travailleurs informels et travailleurs intérimaires,en activité temporaire ou d'appoint) ou non renseigné;CHAR;1
ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);1;Moins d'un an;CHAR;1
ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);2;De 1 an à moins de 5 ans;CHAR;1
ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);3;De 5 ans à moins de 10 ans;CHAR;1
ANCENTR4;Ancienneté dans l'entreprise ou dans la fonction publique (4 postes);4;10 ans ou plus;CHAR;1

DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);;Non renseigné;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);10;Licence (L3), Maitrise (M1), Master (recherche ou professionnel), DEA, DESS, Doctorat;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);11;Ecoles niveau licence et au-delà;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);30;DEUG;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);31;BTS, DUT ou équivalent;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);33;Paramédical et social (niveau bac+2);CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);41;Baccalauréat général;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);42;Baccalauréat technologique, bac professionnel ou équivalents;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);50;CAP, BEP ou équivalents;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);60;Brevet des collèges;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);70;Certificat d'Etudes Primaires;CHAR;2
DIP11;Diplôme le plus élevé obtenu (2 chiffres, 11 postes);71;Sans diplôme;CHAR;2

CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;;Vide Statut d'activité BIT non renseigné pour la personne de référence du ménage (ACTEU6PRM);CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;00;non renseigné);CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;11;Agriculteurs sur petite exploitation;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;12;Agriculteurs sur moyenne exploitation;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;13;Agriculteurs sur grande exploitation;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;21;Artisans;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;22;Commerçants et assimilés;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;23;Chefs d'entreprise de 10 salariés ou plus;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;31;Professions libérales;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;33;Cadres de la fonction publique;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;34;Professeurs, professions scientifiques;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;35;Professions de l'information, des arts et des spectacles;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;37;Cadres administratifs et commerciaux d'entreprise;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;38;Ingénieurs et cadres techniques d'entreprise;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;42;Professeurs des écoles, instituteurs et assimilés;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;43;Professions intermédiaires de la santé et du travail social;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;44;Clergé, religieux;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;45;Professions intermédiaires administratives de la fonction publique;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;46;Professions intermédiaires administratives et commerciales des entreprises;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;47;Techniciens;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;48;Contremaîtres, agents de maîtrise;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;52;Employés civils et agents de service de la fonction publique;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;53;Policiers et militaires;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;54;Employés administratifs d'entreprise;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;55;Employés de commerce;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;56;Personnels des services directs aux particuliers;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;62;Ouvriers qualifiés de type industriel;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;63;Ouvriers qualifiés de type artisanal;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;64;Chauffeurs;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;65;Ouvriers qualifiés de la manutention, du magasinage et du transport;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;67;Ouvriers non qualifiés de type industriel;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;68;Ouvriers non qualifiés de type artisanal;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;69;Ouvriers agricoles;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;71;Anciens agriculteurs exploitants;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;72;Anciens artisans, commerçants, chefs d'entreprise;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;74;Anciens cadres;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;75;Anciennes professions intermédiaires;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;77;Anciens employés;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;78;Anciens ouvriers;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;81;Chômeurs n'ayant jamais travaillé;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;83;Militaires du contingent;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;84;Elèves, étudiants;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;85;Personnes de moins de 60 ans n'ayant jamais travaillé;CHAR;2
CSTOTPRM;Catégorie socioprofessionnelle (2 chiffres, niveau détaillé) de la personne de référence du ménage;86;Personnes de 60 ans ou plus n'ayant jamais travaillé;CHAR;2

HHCE;Nombre d'heures travaillées en moyenne par semaine dans l'emploi principal, heures supplémentaires comprises;;Sans objet (personnes non actives occupées) ou non réponse;NUM;3.1

JOURTR;Nombre de jours travaillés habituellement par semaine;;Sans objet (personnes non actives occupées) ou non réponse;NUM;2.1

HPLUSA;Nombre d'heures de travail souhaitées par semaine dans l'idéal (avec la variation de revenus correspondante), pour ceux qui souhaitent faire plus ou moins d'heures;;Sans objet (STPLC distinct de 1 et STMN distinct de 1);NUM;3.1

NBTOTE;Nombre d'heures travaillées en moyenne par semaine pour l'ensemble des activités professionnelles de l'individu (pour ceux ayant plusieurs activités professionnelles);;Sans objet (voir commentaires);NUM;3.1
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


df = pd.read_csv("FD_csv_EEC18.csv", sep=';')


def get_ration_male_female(df):
    males = df.loc[df['SEXE'] == 1]
    females = df.loc[df['SEXE'] == 2]
    return len(males.index) / (len(males.index) + len(females.index))


def amount_of_work_desired(df):
    active = df.loc[df["ACTEU"] == 1]
    reduced = active[["HHCE",       # Worked hours
                      "HPLUSA",     # Desired worked hours
                      "SEXE",       # 1 for male, 2 for females
                      "COURED",     # 1 for couple, 2 if not in couple
                      "ENFRED"      # 1 if children, 2 if no children
                      ]].dropna()
    reduced = reduced.loc[(reduced["HPLUSA"] != 0) & (reduced["HHCE"] != 0)]

    # Interestingly, those who answer the question "how much to increase" have lower work amount
    worked = reduced["HHCE"]
    worked_desired = reduced["HPLUSA"]
    ratio = worked_desired / worked
    print(df['HHCE'].describe())
    print(worked.describe())
    print(worked_desired.describe())
    print(ratio.describe())

    # Trying a linear regression but it is actually not very well adapted
    reg = linear_model.LinearRegression()
    reg.fit(X=worked.values.reshape(-1, 1), y=worked_desired.values)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    lo, hi = worked.min(), worked.max()
    lo_val = lo * slope + intercept
    hi_val = hi * slope + intercept

    plt.scatter(x=worked, y=worked_desired, alpha=0.01, marker='.')
    # plt.hexbin(x=worked, y=worked_desired, gridsize=(10, 10))
    plt.plot([lo, hi], [lo_val, hi_val], linestyle='-', color='orange')
    plt.plot([lo, hi], [lo, hi], linestyle='-', color='darkgreen')
    plt.show()


amount_of_work_desired(df)


'''
type_menage = df['TYPMEN7']
type_menage.plot.hist(bins=9)
plt.show()
'''

