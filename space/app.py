"""Deux onglets Gradio reliés au modèle d'abandon dans les 28 jours."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import hashlib
from io import BytesIO
import numpy as np
import math
from pathlib import Path
from functools import lru_cache
from threading import RLock

import gradio as gr
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
default_extract = BASE / 'demo.parquet'
if not default_extract.exists():
    default_extract = BASE / 'generated' / 'demo.parquet'
EXTRACT = Path(os.environ.get('MOODLE_DEMO_PARQUET', default_extract))
if not EXTRACT.is_file():
    raise FileNotFoundError('Extrait Parquet absent. Lancez : python outputs/moodle_schema_poc/export_parquet.py')
parquet_metadata = pq.read_schema(EXTRACT).metadata or {}
if b'moodle_model' not in parquet_metadata:
    raise ValueError('Ancien format Parquet : régénérez l’extrait pour inclure le modèle.')
extract_metadata = json.loads(parquet_metadata[b'moodle_extract'])
model_bytes = parquet_metadata[b'moodle_model']
if hashlib.sha256(model_bytes).hexdigest() != extract_metadata['model_sha256']:
    raise ValueError('Le modèle intégré au Parquet ne correspond pas à son empreinte.')
# Le Parquet doit provenir du générateur du projet, comme tout modèle joblib chargé.
BUNDLE = joblib.load(BytesIO(model_bytes))
MODEL = BUNDLE['model']
EXPERIENCE = extract_metadata
if extract_metadata['features'] != BUNDLE['features']:
    raise ValueError('Les variables du Parquet ne correspondent pas au modèle.')
# Les issues futures sont exclues ; seules les variables à X servent à prédire.
SCORES = pd.read_parquet(EXTRACT)
SCORES['risque_abandon_28j'] = MODEL.predict_proba(SCORES[BUNDLE['features']])[:, 1]
if not np.allclose(SCORES.risque_abandon_28j, SCORES.risque_reference):
    raise ValueError('Les prédictions diffèrent du notebook.')
EVALUATION = pd.DataFrame(extract_metadata['evaluation'])
USER_RISKS = SCORES.sort_values('risque_abandon_28j', ascending=False).drop_duplicates('userid')
TOP3 = USER_RISKS.head(3)
# Pour chaque stagiaire, comparer son inscription au risque maximal.
LOW3 = USER_RISKS.sort_values(['risque_abandon_28j', 'userid'], kind='stable').head(3)
DATE = pd.Timestamp(SCORES.as_of.iloc[0])
END = DATE + pd.DateOffset(days=28)
SEARCH_COLUMNS = ['Formation', 'Financement', 'Risque d’abandon', 'Progression', 'Jours sans activité']
TOP_COLUMNS = ['Rang', 'Identifiant', 'Formation', 'Risque estimé']
EXAMPLES = TOP3.userid.astype(int).tolist()
LOW_EXAMPLES = LOW3.userid.astype(int).tolist()
CSS = '.gradio-container {max-width: 1280px !important;} footer {display: none !important;}'
SHAP_LOCK = RLock()
# Référence fixe et reproductible ; seul le calcul d’explication change, pas le modèle.
SHAP_BACKGROUND = shap.sample(SCORES[BUNDLE['features']], 100, random_state=42)
EXPLAINER = shap.TreeExplainer(
    MODEL, data=SHAP_BACKGROUND,
    feature_perturbation='interventional', model_output='probability')
FEATURE_LABELS = {
    'enrolment_age_days': 'Ancienneté de l’inscription (jours)',
    'days_to_expected_end': 'Jours avant la fin prévue',
    'days_since_activity': 'Jours depuis la dernière activité',
    'sessions_7d': 'Sessions sur 7 jours',
    'sessions_previous_7d': 'Sessions sur les 7 jours précédents',
    'active_days_28d': 'Jours actifs sur 28 jours',
    'progress': 'Progression', 'retard_relatif': 'Retard relatif',
    'late_session_ratio_28d': 'Part des sessions tardives',
    'session_gap_std_days_28d': 'Irrégularité des connexions',
    'intention_poursuite': 'Dernière intention de poursuivre',
    'difficulte_declaree': 'Difficulté déclarée',
    'disponibilite_relative': 'Disponibilité relative',
    'intention_moyenne_21j': 'Intention moyenne sur 21 jours',
    'evolution_intention_21j': 'Évolution de l’intention sur 21 jours',
    'diversite_proposee': 'Diversité proposée à X',
    'diversite_rencontree': 'Diversité rencontrée à X',
    'formation_professionnalisante': 'Formation professionnalisante',
    'financement_CPF': 'Financement CPF', 'financement_PDC': 'Financement PDC',
    'financement_fonds_propres': 'Financement sur fonds propres',
}


def explain_rows(rows):
    """SHAP en probabilité, puis affichage en points de pourcentage."""
    features = BUNDLE['features']
    with SHAP_LOCK:
        explanation = EXPLAINER(rows[features])
    probability = np.asarray(explanation.base_values).reshape(-1) + explanation.values.sum(axis=1)
    mismatched = ~np.isclose(
        probability, MODEL.predict_proba(rows[features])[:, 1], rtol=0, atol=1e-6)
    if mismatched.any():
        # TreeSHAP interventional peut diverger aux seuils numériques des arbres.
        # Recalculer ces seuls cas par permutations sur la même référence : les
        # contributions sont estimées, mais leur somme retrouve la prédiction.
        def predict_probability(values):
            return MODEL.predict_proba(pd.DataFrame(values, columns=features))[:, 1]

        with SHAP_LOCK:
            for position in np.flatnonzero(mismatched):
                # Même estimation dans les vues individuelle et globale.
                fallback = shap.PermutationExplainer(
                    predict_probability, SHAP_BACKGROUND, seed=42)(
                        rows.iloc[[position]][features],
                        max_evals=10 * (2 * len(features) + 1), silent=True)
                explanation.values[position] = fallback.values[0]
                explanation.base_values[position] = fallback.base_values[0]
        probability = np.asarray(explanation.base_values).reshape(-1) + explanation.values.sum(axis=1)
    if not np.allclose(probability, MODEL.predict_proba(rows[features])[:, 1], rtol=0, atol=1e-6):
        raise ValueError('Les contributions SHAP ne reconstituent pas le risque estimé.')
    # Ne pas convertir chaque contribution par une sigmoïde : elles doivent rester additives.
    explanation.values *= 100
    explanation.base_values *= 100
    explanation.feature_names = [FEATURE_LABELS.get(f, f) for f in features]
    return explanation


def render_shap(explanation, *, individual=False, title=''):
    # Matplotlib utilise un état global : sérialiser les rendus concurrents.
    with SHAP_LOCK:
        fig = plt.figure()
        try:
            if individual:
                shap.plots.waterfall(explanation, max_display=12, show=False)
                axes = plt.gcf().axes
                axes[0].set_xlabel(
                    'Risque estimé (%) — contributions en points de pourcentage', labelpad=55)
                reference = float(explanation.base_values)
                risk = reference + float(explanation.values.sum())
                axes[1].set_xticks([reference], [f'Référence : {percent(reference / 100)}'])
                axes[1].tick_params(axis='x', pad=22)
                axes[2].set_xticks([risk], [f'Risque estimé : {percent(risk / 100)}'])
                # Les libellés SHAP sont superposés par défaut ; un seul repère suffit.
                labels = [t.get_text().replace('other features', 'autres variables')
                          for t in axes[0].get_yticklabels()]
                axes[0].set_yticks(axes[0].get_yticks(), labels)
            else:
                shap.plots.beeswarm(explanation, max_display=15, show=False)
                plt.xlabel('Variation du risque estimé (points de pourcentage)')
            plt.gcf().axes[0].set_title(title, pad=45 if individual else 10)
            fig = plt.gcf()
            fig.set_size_inches(12, fig.get_figheight())
            fig.tight_layout()
            fig.canvas.draw()
            return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        finally:
            plt.close(fig)


@lru_cache(maxsize=1)
def global_shap_image():
    return render_shap(explain_rows(SCORES), title='Sens des contributions — cohorte de démonstration')


def local_shap_image(row):
    features = BUNDLE['features']
    sample = row[features].to_frame().T.astype(SCORES[features].dtypes)
    return render_shap(
        explain_rows(sample)[0], individual=True,
        title=f'Utilisateur {int(row.userid)} — inscription {int(row.userenrolmentid)} '
              f'— risque {percent(row.risque_abandon_28j)}')


def percent(value):
    return '<0,1 %' if 0 < value < .001 else f'{value:.1%}'


def inactivity_duration(days):
    if pd.isna(days) or not math.isfinite(float(days)):
        return 'Non renseigné'
    total_hours = math.floor(max(0., float(days)) * 24 + 1e-8)
    if total_hours == 0:
        return 'Moins d’une heure' if days > 0 else '0 heure'
    full_days, hours = divmod(total_hours, 24)
    parts = []
    if full_days:
        parts.append(f'{full_days}j')
    if hours:
        parts.append(f'{hours} heure' + ('s' if hours > 1 else ''))
    return ' et '.join(parts)


def search_user(user_id):
    try:
        number = float(user_id)
        if not math.isfinite(number) or not number.is_integer() or number < 1:
            raise ValueError
        user_id = int(number)
    except (TypeError, ValueError, OverflowError):
        return 'Saisissez un identifiant entier positif.', pd.DataFrame(columns=SEARCH_COLUMNS), None
    rows = SCORES[SCORES.userid.eq(user_id)].copy()
    if rows.empty:
        return ('Aucune inscription dans la cohorte de démonstration pour cet identifiant. '
                'Cela ne signifie pas que son risque est nul.', pd.DataFrame(columns=SEARCH_COLUMNS), None)
    # Inférence réelle sur la photographie figée, sans réentraînement.
    rows['risque_abandon_28j'] = MODEL.predict_proba(rows[EXPERIENCE['features']])[:, 1]
    rows = rows.sort_values('risque_abandon_28j', ascending=False)
    table = pd.DataFrame({
        'Formation': rows.formation,
        'Financement': np.select([rows.financement_CPF.eq(1), rows.financement_PDC.eq(1), rows.financement_fonds_propres.eq(1)], ['CPF', 'PDC', 'Fonds propres'], default='Non renseigné'),
        'Risque d’abandon': rows.risque_abandon_28j.map(percent),
        'Progression': rows.progress.map(percent),
        'Jours sans activité': rows.days_since_activity.map(inactivity_duration),
    })
    return (f'### Identifiant {user_id}\n'
            f'Risque maximal parmi ses inscriptions : **{percent(rows.risque_abandon_28j.max())}**. '
            f'Estimation au {DATE:%d/%m/%Y}, pour un abandon effectif d’ici au {END:%d/%m/%Y}. '
            'Il s’agit d’un risque estimé non calibré, pas d’un départ certain.'), table, local_shap_image(rows.iloc[0])


def risk_table(rows):
    return pd.DataFrame({'Rang': range(1, len(rows)+1), 'Identifiant': rows.userid.astype(int),
                         'Formation': rows.formation, 'Risque estimé': rows.risque_abandon_28j.map(percent)})


def platform_view():
    metric = EVALUATION.iloc[0]
    summary = (f'### Abandon administratif dans les 28 prochains jours\n'
        f'**{len(SCORES)} inscriptions · {SCORES.userid.nunique()} stagiaires** dans les deux populations de démonstration évaluables au {DATE:%d/%m/%Y}, inscrits depuis au moins 21 jours.\n\n'
        f'Précision : **{metric.precision:.1%}** · rappel : **{metric.rappel:.1%}** · F1 : **{metric.F1:.1%}**.\n\n'
        'Le SHAP global décrit cette cohorte, pas toute la plateforme. '
        'À droite, une contribution augmente le risque ; à gauche, elle le diminue. '
        'Rouge indique une valeur élevée, bleu une valeur faible. '
        'Les contributions sont en points de pourcentage : +5 signifie une hausse de 5 points '
        'du risque estimé (par exemple, de 10 % à 15 %). '
        'SHAP explique les associations apprises, pas les causes des abandons.\n\n'
        'Le financement, les signes d’engagement, la diversité et la finalité professionnelle influencent cette simulation. '
        'Ces résultats ne constituent pas une validation sur le terrain. '
        'Deux populations ont été évaluées avec le même modèle : F1 de 95,0 % et 85,4 %. '
        'Le bilan commun est affiché ci-dessus. La convention professionnalisante vaut oui pour formateur et management, non pour bureautique.')
    table = risk_table(TOP3)
    return summary, global_shap_image(), table, *[
        local_shap_image(row) for _, row in TOP3.iterrows()]


def create_app():
    with gr.Blocks(title='Moodle — abandon à 28 jours', analytics_enabled=False) as demo:
        gr.Markdown('# Risque d’abandon dans les 28 prochains jours\n'
                    f'Photographie au **{DATE:%d/%m/%Y}** · données synthétiques · cohorte de test')
        with gr.Tabs(selected='individual'):
            with gr.Tab('Risque individuel', id='individual'):
                gr.Markdown('Saisissez un identifiant présent dans la cohorte de démonstration.')
                with gr.Row():
                    user_id = gr.Number(value=EXAMPLES[0], label='Identifiant utilisateur', precision=0, minimum=1)
                    search = gr.Button('Afficher le risque', variant='primary')
                initial_summary, initial_table, initial_shap = search_user(EXAMPLES[0])
                user_summary = gr.Markdown(initial_summary)
                user_table = gr.Dataframe(value=initial_table, headers=SEARCH_COLUMNS, interactive=False, wrap=True)
                gr.Markdown('### Explication de l’inscription au risque maximal\n'
                            'Le graphique part d’un risque de référence et aboutit au risque estimé pour cette inscription. '
                            'Les contributions sont en points de pourcentage : +5 fait passer, par exemple, de 10 % à 15 %. '
                            'Rouge augmente le risque, bleu le diminue. '
                            'SHAP explique les associations apprises, pas les causes des abandons.')
                user_shap = gr.Image(value=initial_shap, label='SHAP individuel', interactive=False, buttons=['fullscreen'])
                gr.Examples(examples=[[i] for i in EXAMPLES], inputs=user_id, label='Exemples — trois risques les plus élevés')
                gr.Examples(examples=[[i] for i in LOW_EXAMPLES], inputs=user_id, label='Exemples — trois risques les plus faibles')
                search.click(search_user, user_id, [user_summary, user_table, user_shap], api_name='user_risk')
                user_id.submit(search_user, user_id, [user_summary, user_table, user_shap])
            with gr.Tab('État de la plateforme et explications', id='platform'):
                summary, global_image, top_table, *local_images = platform_view()
                gr.Markdown(summary)
                gr.Image(value=global_image, label='SHAP global — sens des contributions', interactive=False, buttons=['fullscreen'])
                gr.Markdown('### Trois stagiaires aux risques les plus élevés dans cette cohorte')
                gr.Dataframe(value=top_table, headers=TOP_COLUMNS, interactive=False, wrap=True)
                gr.Markdown('### Trois stagiaires aux risques les plus faibles dans cette cohorte')
                gr.Markdown('Pour chaque stagiaire, le classement retient le risque maximal parmi ses inscriptions.')
                gr.Dataframe(value=risk_table(LOW3), headers=TOP_COLUMNS, interactive=False, wrap=True)
                gr.Markdown('### Explications locales des trois risques les plus élevés')
                for i, path in enumerate(local_images, 1):
                    gr.Image(value=path, label=f'SHAP local — rang {i}', interactive=False, buttons=['fullscreen'])
                gr.Markdown('Les résultats sur ce jeu fictif restent exploratoires. Aucun seuil d’alerte automatique n’est appliqué.')
    return demo.queue(max_size=16)


if __name__ == '__main__':
    create_app().launch(css=CSS)
