#!/bin/bash

# SLURM options:

#SBATCH --job-name=CIMP_1.3-3mA    # Nom du job



#SBATCH --output=./logs/Output_%j_%x.log   # Standard output et error log
#SBATCH --error=./logs/Error_%j_%x.log
#SBATCH --partition=htc               # Choix de partition (htc par défaut)



#SBATCH --time=6-01:00:00             # Délai max = 7 jours
#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --mem=4000                   # Mémoire en MB par défaut

#SBATCH --mail-user=<e-mail>          # Où envoyer l'e-mail
#SBATCH --mail-type=END,FAIL          # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --licenses=sps                # Déclaration des ressources de stockage et/ou logicielles

# Commandes à soumettre :
# python script.py <turns> <mp> <current> <model>

python mbtrack_run.py 300_000 10_000 1.3245e-3 CIMP

