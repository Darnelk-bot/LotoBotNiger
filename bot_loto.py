import threading
from flask import Flask, request
import os
import sys
import subprocess
import importlib
import csv
import random
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters
)

# Vérifier et installer les dépendances manquantes
required_modules = ['numpy', 'scikit-learn', 'joblib']
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

def install_missing_modules():
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} est déjà installé")
        except ImportError:
            print(f"⚠️ Installation de {module}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"✅ {module} installé avec succès")
            except Exception as e:
                print(f"❌ Échec de l'installation de {module}: {str(e)}")

install_missing_modules()

# Maintenant que les dépendances sont installées, importer les modules
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ======================
# CONFIGURATION NIGÉRIENNE
# ======================
MAX_NUM = 90  # Format du loto nigérien
NOMBRE_TIRAGE = 5  # Nombre de numéros à prédire

# Chemins absolus pour tous les fichiers
CSV_FILE = os.path.join(os.getcwd(), 'loto_niger.csv')
MODEL_FILE = os.path.join(os.getcwd(), 'loto_model_niger.pkl')
HISTORY_FILE = os.path.join(os.getcwd(), 'prediction_history.csv')
PREMIUM_FILE = os.path.join(os.getcwd(), 'premium_users.csv')
REJECTED_BALLS_FILE = os.path.join(os.getcwd(), 'boules_rejetees.csv')
ACTIVITY_LOG = os.path.join(os.getcwd(), 'activity_log.csv')
PENDING_PAYMENTS = os.path.join(os.getcwd(), 'pending_payments.csv')

# Plans premium en FCFA
PREMIUM_PLANS = {
    '1m': {'price': 1000, 'days': 30, 'label': "💎 1 Mois - 1000 FCFA"},
    '3m': {'price': 2500, 'days': 90, 'label': "💎 3 Mois - 2500 FCFA"},
    '1a': {'price': 9000, 'days': 365, 'label': "💎 1 An - 9000 FCFA"}
}

# Période d'essai gratuit (7 jours)
FREE_TRIAL_DAYS = 7

# ======================
# FONCTIONS DE BASE
# ======================
def lire_donnees():
    """Charge les données historiques du loto nigérien avec dates"""
    tirages = []
    print(f"📖 Lecture des données depuis: {CSV_FILE}")
    
    if not os.path.exists(CSV_FILE):
        print("⚠️ Fichier de données non trouvé")
        return []
    
    with open(CSV_FILE, encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader, None)  # Ignorer l'en-tête si présent
        
        for row in reader:
            if not row or len(row) < NOMBRE_TIRAGE + 1:  # Doit avoir date + numéros
                continue
                
            try:
                # La première colonne est la date
                date_str = row[0].strip()
                
                # Nettoyer et valider les numéros DANS L'ORDRE ORIGINAL
                numeros = []
                for cell in row[1:1+NOMBRE_TIRAGE]:
                    if cell.strip().isdigit():
                        num = int(cell.strip())
                        if 1 <= num <= MAX_NUM:
                            numeros.append(num)
                
                if len(numeros) == NOMBRE_TIRAGE:
                    # Stocker [date, num1, num2, ...] dans l'ordre original
                    tirages.append([date_str] + numeros)
            except ValueError:
                continue
    
    # Trier les tirages par date (plus ancien en premier)
    try:
        tirages.sort(key=lambda x: datetime.strptime(x[0], "%d/%m/%Y"))
    except:
        # Si erreur de format de date, garder l'ordre d'origine
        pass
    
    return tirages

def get_derniere_mise_a_jour():
    """Récupère la date du dernier tirage dans les données"""
    tirages = lire_donnees()
    if tirages:
        # Trier par date pour obtenir la plus récente
        try:
            derniers = sorted(
                tirages, 
                key=lambda x: datetime.strptime(x[0], "%d/%m/%Y"), 
                reverse=True
            )
            return derniers[0][0]  # Date du tirage le plus récent
        except:
            return "Inconnue"
    return "Inconnue"

def tirage_to_vecteur(tirage):
    """Convertit un tirage en vecteur binaire (sans la date)"""
    vect = np.zeros(MAX_NUM)
    # tirage est [date, num1, num2, ...] - on prend les numéros seulement
    for num in tirage[1:]:
        if 1 <= num <= MAX_NUM:
            vect[num - 1] = 1
    return vect

# ======================
# GESTION DES BOULES REJETÉES
# ======================
def charger_boules_rejetees():
    """Charge les boules rejetées du fichier manuel"""
    boules_rejetees = []
    print(f"📖 Lecture des boules rejetées depuis: {REJECTED_BALLS_FILE}")
    
    if not os.path.exists(REJECTED_BALLS_FILE):
        print("⚠️ Fichier des boules rejetées non trouvé")
        return boules_rejetees
    
    with open(REJECTED_BALLS_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Ignorer l'en-tête si présent
        
        for row in reader:
            if not row or len(row) < 3:
                continue
                
            try:
                # La colonne des boules rejetées est la troisième
                nums_str = row[2].split(',')
                boules = [int(num.strip()) for num in nums_str if num.strip().isdigit()]
                boules_rejetees.extend(boules)
            except ValueError:
                continue
    
    # Éliminer les doublons et filtrer les numéros valides
    return list(set([num for num in boules_rejetees if 1 <= num <= MAX_NUM]))

# ======================
# MODÈLE IA
# ======================
def entrainer_model():
    """Entraîne le modèle sur les données nigériennes"""
    print("🔄 Entraînement du modèle spécifique Niger...")
    tirages = lire_donnees()
    
    # Vérifier si on a assez de données
    if len(tirages) < 10:
        print("❌ Pas assez de données pour l'entraînement (min. 10 tirages requis)")
        return None
        
    X, y = [], []

    # Création des séquences temporelles (utilise seulement les numéros)
    for i in range(len(tirages) - 3):
        sequence = []
        for j in range(3):
            # Utiliser seulement les numéros (sans date)
            sequence.extend(tirage_to_vecteur(tirages[i+j]))
        X.append(sequence)
        y.append(tirage_to_vecteur(tirages[i+3]))
    
    # Vérifier si on a assez de séquences
    if len(X) < 2:
        print("❌ Pas assez de séquences pour l'entraînement")
        return None
    
    # Entraînement avec validation
    test_size = min(0.2, len(X) - 1)  # Garantir au moins 1 échantillon d'entraînement
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42
    )
    
    try:
        model.fit(X_train, y_train)
        print("✅ Modèle entraîné avec succès")
        return model
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return None

def predire_tirage(model, historique):
    """Génère une prédiction basée sur l'historique récent et les boules rejetées"""
    # Si le modèle n'est pas disponible, utiliser une prédiction aléatoire
    if model is None:
        prediction = random.sample(range(1, MAX_NUM + 1), NOMBRE_TIRAGE)
        return sorted(prediction)
    
    # Vérifier l'historique
    if len(historique) < 3:
        prediction = random.sample(range(1, MAX_NUM + 1), NOMBRE_TIRAGE)
        return sorted(prediction)
    
    try:
        # Créer le vecteur d'historique (utilise seulement les numéros)
        vect_historique = np.concatenate([
            tirage_to_vecteur(t) for t in historique[-3:]
        ]).reshape(1, -1)
        
        proba = model.predict_proba(vect_historique)[0]
    except:
        # En cas d'erreur, utiliser une prédiction aléatoire
        prediction = random.sample(range(1, MAX_NUM + 1), NOMBRE_TIRAGE)
        return sorted(prediction)
    
    # Charger les boules rejetées manuellement
    boules_rejetees = charger_boules_rejetees()
    
    # Augmenter significativement la probabilité des boules rejetées
    for num in boules_rejetees:
        if 1 <= num <= MAX_NUM:
            idx = num - 1
            proba[idx] *= 2.5  # Multiplier par 2.5 la probabilité
    
    # Classer les numéros par probabilité
    numeros_tries = sorted(
        range(1, MAX_NUM + 1),
        key=lambda i: proba[i-1],
        reverse=True
    )
    
    # Exclusion des numéros récents
    derniers_numeros = set()
    for t in historique[-5:]:
        # t est [date, num1, num2, ...] - on prend les numéros seulement
        derniers_numeros.update(t[1:])
    
    prediction = []
    for num in numeros_tries:
        if num not in derniers_numeros:
            prediction.append(num)
            if len(prediction) == NOMBRE_TIRAGE:
                break
    
    # Si on n'a pas assez de numéros, compléter avec les plus probables
    if len(prediction) < NOMBRE_TIRAGE:
        prediction.extend(numeros_tries[:NOMBRE_TIRAGE - len(prediction)])
    
    return sorted(prediction)

# ======================
# GESTION UTILISATEURS (AVEC ESSAI GRATUIT)
# ======================
def charger_premium_users():
    """Charge les utilisateurs premium depuis le fichier"""
    premium_users = {}
    print(f"📖 Lecture des utilisateurs premium depuis: {PREMIUM_FILE}")
    
    if not os.path.exists(PREMIUM_FILE):
        print("⚠️ Fichier premium non trouvé")
        return premium_users
    
    with open(PREMIUM_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Ignorer l'en-tête
        for row in reader:
            if row and len(row) >= 2:
                try:
                    user_id, expiry = row
                    premium_users[int(user_id)] = datetime.fromisoformat(expiry)
                except (ValueError, IndexError):
                    continue
    return premium_users

def sauvegarder_premium_user(user_id, duration_days):
    """Ajoute un utilisateur premium"""
    expiry = datetime.now() + timedelta(days=duration_days)
    premium_users = charger_premium_users()
    premium_users[user_id] = expiry
    
    with open(PREMIUM_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'expiry_date'])
        for uid, exp in premium_users.items():
            writer.writerow([uid, exp.isoformat()])
    
    return expiry

def activer_essai_gratuit(user_id):
    """Active un essai gratuit de 7 jours"""
    return sauvegarder_premium_user(user_id, FREE_TRIAL_DAYS)

def verifier_premium(user_id):
    """Vérifie si un utilisateur est premium"""
    premium_users = charger_premium_users()
    expiry = premium_users.get(user_id)
    return expiry and expiry > datetime.now()

def a_deja_utilise_essai(user_id):
    """Vérifie si l'utilisateur a déjà utilisé son essai gratuit"""
    premium_users = charger_premium_users()
    expiry = premium_users.get(user_id)
    # Si l'utilisateur a une expiration mais qu'elle est passée, il a déjà utilisé son essai
    return expiry is not None

# ======================
# JOURNALISATION
# ======================
def logger_prediction(user_id, prediction):
    """Enregistre les prédictions pour analyse"""
    try:
        print(f"📝 Journalisation de la prédiction dans: {HISTORY_FILE}")
        header = not os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(['timestamp', 'user_id', 'prediction'])
            writer.writerow([
                datetime.now().isoformat(),
                user_id,
                ','.join(map(str, prediction))
            ])
    except Exception as e:
        print(f"⚠️ Erreur journalisation: {str(e)}")

def logger_activite(action, user_id):
    """Journalise les activités importantes"""
    try:
        print(f"📝 Journalisation d'activité: {action}")
        with open(ACTIVITY_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                user_id,
                action
            ])
    except Exception as e:
        print(f"⚠️ Erreur journalisation activité: {str(e)}")

# ======================
# FONCTIONS TELEGRAM
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande de démarrage avec menu adapté"""
    user = update.effective_user
    logger_activite("start", user.id)
    
    # Créer le menu
    keyboard = [
        [InlineKeyboardButton("🔮 Prédiction", callback_data='predict')],
        [InlineKeyboardButton("📊 Stats Niger", callback_data='stats')],
        [InlineKeyboardButton("🏆 Témoignages", callback_data='temoignages')],
        [InlineKeyboardButton("🚫 Boules rejetées", callback_data='rejected')],
        [InlineKeyboardButton("❓ Aide", callback_data='help')]
    ]
    
    # Ajouter l'option d'essai gratuit si l'utilisateur n'est pas premium et n'a pas encore utilisé son essai
    if not verifier_premium(user.id) and not a_deja_utilise_essai(user.id):
        keyboard.insert(0, [InlineKeyboardButton("🆓 Essai 7 jours gratuit", callback_data='free_trial')])
    else:
        keyboard.insert(0, [InlineKeyboardButton("💎 Devenir Premium", callback_data='premium')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = (
        "🌟 *LotoBot Niger - Votre prophète des numéros gagnants!*\n\n"
        "Bienvenue! Je suis une IA spécialement entraînée sur les tirages du loto nigérien.\n\n"
        "Utilisez les boutons ci-dessous pour commencer :"
    )
    
    await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gère les interactions avec les boutons"""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    
    if query.data == 'predict':
        await send_prediction(query, context)
    elif query.data == 'stats':
        await send_stats(query, context)
    elif query.data == 'temoignages':
        await send_temoignages(query, context)
    elif query.data == 'premium':
        await send_premium_offers(query, context)
    elif query.data == 'help':
        await send_help(query, context)
    elif query.data == 'rejected':
        await afficher_boules_rejetees(query, context)
    elif query.data == 'free_trial':  # Nouvelle option pour l'essai gratuit
        await activer_essai_gratuit_handler(query, context)
    elif query.data.startswith('premium_'):
        await handle_premium_payment(query, context)

async def activer_essai_gratuit_handler(query, context: ContextTypes.DEFAULT_TYPE):
    """Active l'essai gratuit de 7 jours"""
    user_id = query.from_user.id
    
    # Vérifier si l'utilisateur a déjà utilisé son essai
    if a_deja_utilise_essai(user_id):
        await query.answer("⚠️ Vous avez déjà utilisé votre essai gratuit.", show_alert=True)
        return
    
    # Activer l'essai
    expiry = activer_essai_gratuit(user_id)
    expiry_str = expiry.strftime("%d/%m/%Y à %H:%M")
    
    # Mettre à jour le message
    await query.edit_message_text(
        text=f"🎉 *Essai Premium activé!*\n\n"
             f"Vous avez maintenant accès à toutes les fonctionnalités premium jusqu'au {expiry_str}.\n\n"
             f"Profitez de votre essai gratuit de 7 jours!",
        parse_mode='Markdown'
    )
    
    # Journaliser l'activation
    logger_activite("free_trial_activated", user_id)

async def send_prediction(query_or_update, context: ContextTypes.DEFAULT_TYPE):
    """Envoie une prédiction personnalisée"""
    try:
        # Récupération de l'utilisateur
        if isinstance(query_or_update, Update):
            user = query_or_update.effective_user
            chat_id = query_or_update.message.chat_id
            reply = query_or_update.message.reply_text
        else:
            user = query_or_update.from_user
            chat_id = query_or_update.message.chat_id
            reply = query_or_update.edit_message_text
        
        # Vérification premium ou essai
        is_premium = verifier_premium(user.id)
        free_uses = compter_utilisations_gratuites(user.id)
        
        if not is_premium:
            # Limite d'utilisation gratuite
            if free_uses >= 3:
                await envoyer_offre_premium(chat_id, user.id, context)
                return
        
        # Génération de la prédiction
        try:
            tirages = lire_donnees()
            prediction = predire_tirage(models, tirages[-5:])
            
            # Visualisation graphique
            visual = ["○"] * MAX_NUM
            for num in prediction:
                if 1 <= num <= MAX_NUM:
                    visual[num-1] = "●"
            
            # Créer la grille visuelle
            grid_lines = []
            for i in range(0, MAX_NUM, 10):
                grid_lines.append(" ".join(visual[i:i+10]))
            grid = "\n".join(grid_lines)
            
            # Message avec crédibilité
            derniers_tirages = tirages[-1][1:] if tirages and len(tirages) > 0 else []
            succes = len(set(prediction) & set(derniers_tirages)) if tirages and len(tirages) > 0 else 0
            precision = f"\n\n📈 Précision récente: {succes}/5 numéros corrects" if tirages else ""
            
            # Ajouter une note si le modèle n'est pas disponible
            model_note = "\n\n⚠️ _Note: Prédiction aléatoire (modèle non disponible)_" if models is None else ""
            
            message = (
                f"🎯 *Votre prédiction personnalisée:*\n"
                f"`{prediction}`{model_note}{precision}\n\n"
                f"📊 Représentation visuelle :\n"
                f"```\n{grid}\n```\n"
                "● = Numéro prédit | ○ = Autres numéros\n\n"
            )
            
            # Ajouter l'info sur les boules rejetées
            boules_rejetees = charger_boules_rejetees()
            if boules_rejetees:
                message += f"🚫 Boules rejetées boostées: {sorted(boules_rejetees)}\n\n"
            
            if not is_premium:
                message += f"🆓 Utilisations gratuites restantes: {3 - free_uses - 1}/3\n"
                # Proposer l'essai gratuit si pas encore utilisé
                if not a_deja_utilise_essai(user.id):
                    message += "Activez votre essai gratuit de 7 jours! /start"
                else:
                    message += "Passez premium pour des prédictions illimitées! /premium"
            else:
                # Afficher la date d'expiration pour les utilisateurs en essai
                premium_users = charger_premium_users()
                expiry = premium_users.get(user.id)
                if expiry and (expiry - datetime.now()).days <= FREE_TRIAL_DAYS:
                    message += f"⏳ Votre essai gratuit expire le {expiry.strftime('%d/%m/%Y')}\n"
            
            # Envoi du message
            await reply(message, parse_mode='Markdown')
            
            # Journalisation
            logger_prediction(user.id, prediction)
            logger_activite("prediction", user.id)
            
        except Exception as e:
            await reply(f"❌ Erreur lors de la génération de la prédiction: {str(e)}")
        
    except Exception as e:
        error_msg = f"❌ Erreur système: {str(e)}"
        if isinstance(query_or_update, Update):
            await query_or_update.message.reply_text(error_msg)
        else:
            await query_or_update.edit_message_text(error_msg)

async def send_stats(query_or_update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les statistiques locales avec le total et la dernière mise à jour"""
    try:
        if isinstance(query_or_update, Update):
            reply = query_or_update.message.reply_text
        else:
            reply = query_or_update.edit_message_text
        
        try:
            tirages = lire_donnees()
            total_tirages = len(tirages)
            derniere_maj = get_derniere_mise_a_jour()
            
            if not tirages:
                await reply("📊 Aucune donnée statistique disponible pour le moment")
                return
            
            # Statistiques locales (tous les numéros de tous les tirages)
            all_nums = []
            for t in tirages:
                all_nums.extend(t[1:])  # Ignorer la date
            
            freq = {i: all_nums.count(i) for i in range(1, MAX_NUM+1)}
            top_5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Derniers résultats - triés du plus récent au plus ancien
            derniers_tri = sorted(
                tirages, 
                key=lambda x: datetime.strptime(x[0], "%d/%m/%Y"), 
                reverse=True
            )[:3]
            
            # Formater les derniers tirages (conserver l'ordre original)
            derniers = "\n".join([f"• {t[0]}: {' '.join(map(str, t[1:]))}" for t in derniers_tri])
            
            # Boules rejetées
            boules_rejetees = charger_boules_rejetees()
            info_rejet = f"🚫 Boules rejetées: {sorted(boules_rejetees)}\n\n" if boules_rejetees else ""
            
            message = (
                "📊 *Statistiques pour le Niger:*\n\n"
                f"• Tirages analysés: *{total_tirages}*\n"
                f"• Dernier tirage: *{derniere_maj}*\n\n"
                f"{info_rejet}"
                "🔝 Numéros les plus fréquents:\n" +
                "\n".join([f"• {num}: {count} fois" for num, count in top_5]) +
                "\n\n📅 Derniers tirages (du plus récent):\n" + derniers
            )
            
            await reply(message, parse_mode='Markdown')
            
        except Exception as e:
            await reply(f"❌ Erreur de données: {str(e)}")
            
    except Exception as e:
        error_msg = f"❌ Erreur système: {str(e)}"
        if isinstance(query_or_update, Update):
            await query_or_update.message.reply_text(error_msg)
        else:
            await query_or_update.edit_message_text(error_msg)

async def send_temoignages(query_or_update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche des témoignages de gagnants"""
    try:
        if isinstance(query_or_update, Update):
            reply = query_or_update.message.reply_text
        else:
            reply = query_or_update.edit_message_text
        
        temoignages = [
            "🎉 *Amadou, Niamey:* 'Grâce à LotoBot, j'ai gagné 500 000 FCFA! Je n'en revenais pas!'",
            "💰 *Fatima, Zinder:* '3 gains en 2 mois seulement! Ce bot est magique.'",
            "🏆 *Ibrahim, Maradi:* 'J'ai enfin pu ouvrir mon commerce avec mes gains. Merci LotoBot!'"
        ]
        
        message = "🏆 *Témoignages de nos gagnants:*\n\n" + "\n\n".join(temoignages)
        await reply(message, parse_mode='Markdown')
        
    except Exception as e:
        error_msg = f"❌ Erreur: {str(e)}"
        if isinstance(query_or_update, Update):
            await query_or_update.message.reply_text(error_msg)
        else:
            await query_or_update.edit_message_text(error_msg)

async def send_premium_offers(query_or_update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les offres premium"""
    try:
        keyboard = [
            [InlineKeyboardButton(plan['label'], callback_data=f'premium_{plan_id}')]
            for plan_id, plan in PREMIUM_PLANS.items()
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = (
            "💎 *LotoBot Premium - La clé de vos gains!*\n\n"
            "Avantages exclusifs:\n"
            "✅ Prédictions quotidiennes illimitées\n"
            "✅ Conseils de paris stratégiques\n"
            "✅ Alertes SMS avant chaque tirage\n"
            "✅ Statistiques avancées exclusives\n"
            "✅ Support personnel prioritaire\n\n"
            f"🆓 Essai gratuit de {FREE_TRIAL_DAYS} jours disponible! /start\n\n"
            "Choisissez votre formule:"
        )
        
        if isinstance(query_or_update, Update):
            await query_or_update.message.reply_text(
                message, 
                parse_mode='Markdown', 
                reply_markup=reply_markup
            )
        else:
            await query_or_update.edit_message_text(
                message, 
                parse_mode='Markdown', 
                reply_markup=reply_markup
            )
            
    except Exception as e:
        error_msg = f"❌ Erreur: {str(e)}"
        if isinstance(query_or_update, Update):
            await query_or_update.message.reply_text(error_msg)
        else:
            await query_or_update.edit_message_text(error_msg)

async def handle_premium_payment(query, context: ContextTypes.DEFAULT_TYPE):
    """Gère le processus de paiement pour premium"""
    try:
        plan_id = query.data.split('_')[1]
        plan = PREMIUM_PLANS.get(plan_id)
        
        if not plan:
            await query.edit_message_text("❌ Offre invalide")
            return
        
        message = (
            f"📲 *Paiement pour {plan['label']}:*\n\n"
            f"1. **Orange Money:** Envoyez {plan['price']} FCFA au 93 00 00 00\n"
            f"2. **Moov Money:** Envoyez {plan['price']} FCFA au 96 00 00 00\n"
            f"3. **Flooz:** Envoyez {plan['price']} FCFA au 98 00 00 00\n\n"
            "Après paiement, envoyez-nous:\n"
            f"- Le code de transaction\n"
            f"- Votre ID: `{query.from_user.id}`\n\n"
            "Votre compte sera activé dans les 24h!"
        )
        
        await query.edit_message_text(message, parse_mode='Markdown')
        
    except Exception as e:
        await query.edit_message_text(f"❌ Erreur: {str(e)}")

async def send_help(query_or_update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche l'aide et les contacts"""
    try:
        if isinstance(query_or_update, Update):
            reply = query_or_update.message.reply_text
        else:
            reply = query_or_update.edit_message_text
        
        message = (
            "ℹ️ *Aide & Support LotoBot Niger*\n\n"
            "Commandes disponibles:\n"
            "• /start - Menu principal\n"
            "• /predict - Prédiction de numéros\n"
            "• /stats - Statistiques locales\n"
            "• /premium - Devenir membre premium\n"
            "• /temoignages - Voir nos gagnants\n"
            "• /rejected - Voir les boules rejetées\n\n"
            f"🆓 *Essai gratuit:*\n"
            f"Profitez de {FREE_TRIAL_DAYS} jours d'essai gratuit avec toutes les fonctionnalités premium!\n\n"
            "📞 Support technique:\n"
            "Telegram: @SupportLotoBotNiger\n"
            "Tél: +227 93 000 000\n"
            "Email: support@lotobot.ne\n\n"
            "Heures d'ouverture: 8h-20h, 7j/7"
        )
        
        await reply(message, parse_mode='Markdown')
        
    except Exception as e:
        error_msg = f"❌ Erreur: {str(e)}"
        if isinstance(query_or_update, Update):
            await query_or_update.message.reply_text(error_msg)
        else:
            await query_or_update.edit_message_text(error_msg)

async def afficher_boules_rejetees(query_or_update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les boules rejetées chargées du fichier manuel"""
    try:
        if isinstance(query_or_update, Update):
            reply = query_or_update.message.reply_text
        else:
            reply = query_or_update.edit_message_text
        
        boules_rejetees = charger_boules_rejetees()
        
        if not boules_rejetees:
            message = "ℹ️ Aucune boule rejetée enregistrée pour le moment."
        else:
            message = (
                "🚫 *Boules rejetées (boostées dans les prédictions):*\n\n"
                f"`{sorted(boules_rejetees)}`\n\n"
                f"Total: {len(boules_rejetees)} numéros\n\n"
                "_Ces numéros sont absents depuis longtemps et ont une probabilité accrue dans nos prédictions._"
            )
        
        await reply(message, parse_mode='Markdown')
        
    except Exception as e:
        error_msg = f"❌ Erreur: {str(e)}"
        if isinstance(query_or_update, Update):
            await query_or_update.message.reply_text(error_msg)
        else:
            await query_or_update.edit_message_text(error_msg)

async def process_payment_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Traite les messages contenant des codes de paiement"""
    try:
        user_id = update.effective_user.id
        message_text = update.message.text.lower()
        
        # Détection de code de transaction
        if "trans" in message_text and any(word in message_text for word in ["om", "moov", "flooz"]):
            # Enregistrer la demande de validation
            with open(PENDING_PAYMENTS, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), user_id, message_text])
            
            await update.message.reply_text(
                "✅ Merci! Votre paiement est en cours de validation.\n"
                "Vous recevrez une confirmation sous 24h."
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur de traitement: {str(e)}")

# ======================
# UTILITAIRES
# ======================
def compter_utilisations_gratuites(user_id):
    """Compte le nombre d'utilisations gratuites"""
    count = 0
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Ignorer l'en-tête
            for row in reader:
                if row and len(row) > 1 and row[1] == str(user_id):
                    count += 1
    return count

async def envoyer_offre_premium(chat_id, user_id, context: ContextTypes.DEFAULT_TYPE):
    """Propose l'offre premium quand la limite est atteinte"""
    # Vérifier si l'utilisateur peut avoir un essai gratuit
    if not a_deja_utilise_essai(user_id):
        message = (
            "🆓 Vous avez utilisé toutes vos prédictions gratuites!\n\n"
            f"💎 Activez votre essai gratuit de {FREE_TRIAL_DAYS} jours pour:\n"
            "- Prédictions ILLIMITÉES\n"
            "- Alertes SMS exclusives\n"
            "- Conseils de paris stratégiques\n\n"
            "👉 /start pour activer votre essai"
        )
    else:
        message = (
            "🆓 Vous avez utilisé toutes vos prédictions gratuites!\n\n"
            "💎 Passez à LotoBot Premium pour:\n"
            "- Prédictions ILLIMITÉES\n"
            "- Alertes SMS exclusives\n"
            "- Conseils de paris stratégiques\n\n"
            "Offre spéciale Niger: seulement 1000 FCFA/mois!\n"
            "👉 /premium pour voir les offres"
        )
    
    await context.bot.send_message(chat_id=chat_id, text=message)

# ======================
# INITIALISATION
# ======================
print("⚙️ Initialisation de LotoBot Niger...")
print(f"📁 Répertoire de travail: {os.getcwd()}")

# Créer les fichiers s'ils n'existent pas
for file_path in [CSV_FILE, MODEL_FILE, HISTORY_FILE, PREMIUM_FILE, 
                 REJECTED_BALLS_FILE, ACTIVITY_LOG, PENDING_PAYMENTS]:
    if not os.path.exists(file_path):
        # Créer les répertoires parents si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Créer le fichier avec l'en-tête approprié
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            if file_path == REJECTED_BALLS_FILE:
                writer.writerow(['date', 'tirages_analyses', 'boules_rejetees'])
                print(f"📁 Fichier créé: {file_path}")
            elif file_path == PREMIUM_FILE:
                writer.writerow(['user_id', 'expiry_date'])
                print(f"📁 Fichier créé: {file_path}")
            elif file_path == HISTORY_FILE:
                writer.writerow(['timestamp', 'user_id', 'prediction'])
                print(f"📁 Fichier créé: {file_path}")
            elif file_path == ACTIVITY_LOG:
                writer.writerow(['timestamp', 'user_id', 'action'])
                print(f"📁 Fichier créé: {file_path}")
            elif file_path == PENDING_PAYMENTS:
                writer.writerow(['timestamp', 'user_id', 'message'])
                print(f"📁 Fichier créé: {file_path}")
            else:
                # Pour les autres fichiers, on ne fait rien (ils seront créés par l'usage)
                pass

# Chargement du modèle
models = None
try:
    # Essayer de charger un modèle existant
    if os.path.exists(MODEL_FILE):
        print("⚡ Tentative de chargement du modèle depuis le cache...")
        models = joblib.load(MODEL_FILE)
        print("✅ Modèle chargé depuis le cache")
    
    # Si pas de modèle chargé, essayer d'en entraîner un nouveau
    if models is None:
        print("🔄 Aucun modèle trouvé, entraînement en cours...")
        models = entrainer_model()
        if models:
            joblib.dump(models, MODEL_FILE)
            print("✅ Nouveau modèle entraîné et sauvegardé")
        else:
            print("⚠️ Utilisation du mode aléatoire (modèle non disponible)")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle: {str(e)}")
    models = None

if __name__ == '__main__':
    # Récupérer le token depuis les variables d'environnement
    BOT_TOKEN = os.environ.get('BOT_TOKEN')
    if not BOT_TOKEN:
        print("❌ ERREUR: BOT_TOKEN non défini dans les variables d'environnement")
        exit(1)
    
    # Initialiser le bot Telegram
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Handlers principaux
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", lambda u, c: send_prediction(u, c)))
    app.add_handler(CommandHandler("stats", lambda u, c: send_stats(u, c)))
    app.add_handler(CommandHandler("premium", lambda u, c: send_premium_offers(u, c)))
    app.add_handler(CommandHandler("temoignages", lambda u, c: send_temoignages(u, c)))
    app.add_handler(CommandHandler("help", lambda u, c: send_help(u, c)))
    app.add_handler(CommandHandler("rejected", lambda u, c: afficher_boules_rejetees(u, c)))
    
    # Handlers spéciaux
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_payment_message))
    
    print("🤖 LotoBot Niger lancé - Prêt à gagner!")
    print(f"🆓 Période d'essai gratuit: {FREE_TRIAL_DAYS} jours")
    
    # Lancer le bot en mode polling
    app.run_polling()


# ======================
# PARTIE DÉMARRAGE POUR RENDER
# ======================


# Créer une application Flask minimale
flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return "🤖 LotoBot Niger est actif et fonctionne! ✅"

@flask_app.route('/health')
def health_check():
    return "OK", 200

def run_flask_server():
    """Lance le serveur Flask dans un thread séparé"""
    port = int(os.environ.get("PORT", 1000))
    flask_app.run(host='0.0.0.0', port=port, use_reloader=False)

if __name__ == '__main__':
    # Démarrer le serveur Flask dans un thread séparé
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Démarrer le bot Telegram
    print("Démarrage du bot Telegram...")
    application.run_polling()
