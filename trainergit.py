import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict, cross_val_score # Dodan cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline # Use pipeline from imblearn

# Postavke za ljepši izgled grafova
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# --- 1. Učitavanje i osnovna priprema podataka ---
print("--- 1. Učitavanje i osnovna priprema podataka ---")
try:
    df = pd.read_csv('train_and_test2.csv')
    print("Učitana datoteka 'train_and_test2.csv'.")
except FileNotFoundError:
    print("Greška: Datoteka 'train_and_test2.csv' nije pronađena.")
    exit()
except Exception as e:
    print(f"Greška prilikom čitanja CSV datoteke: {e}")
    exit()

# Directly use the target column name from your header
target_col_name = 'Survived'

if target_col_name not in df.columns:
    print(f"Greška: Ciljni stupac '{target_col_name}' nije pronađen u učitanim stupcima: {df.columns.tolist()}")
    exit()

# Čišćenje ciljne varijable
if df[target_col_name].isnull().any():
    print(f"Pronađene NaN vrijednosti u '{target_col_name}', uklanjam te retke.")
    df.dropna(subset=[target_col_name], inplace=True)
if df.empty:
    print(f"Dataset je prazan nakon uklanjanja redaka s NaN '{target_col_name}'.")
    exit()
df[target_col_name] = df[target_col_name].astype(int)
print(f"Ciljna varijabla '{target_col_name}' pripremljena.")
print(f"Ukupno redaka nakon pripreme ciljne varijable: {len(df)}")


# --- 2. Osnovna Eksploratorna Analiza Podataka (EDA) s Plotovima ---
print("\n--- 2. Osnovna Eksploratorna Analiza Podataka (EDA) ---")
# Distribucija ciljne varijable
plt.figure(figsize=(6,4)); sns.countplot(x=target_col_name, data=df); plt.title(f'Distribucija ciljne varijable ({target_col_name})'); plt.xlabel(f"{target_col_name} (0 = Ne, 1 = Da)"); plt.ylabel("Broj putnika"); plt.show()


age_col_eda = 'Age'
fare_col_eda = 'Fare'
sex_col_eda = 'Sex'
pclass_col_eda = 'Pclass'
embarked_col_eda = 'Embarked'

if age_col_eda in df.columns:
    plt.figure(figsize=(8,5)); sns.histplot(df[age_col_eda].dropna(), kde=True, bins=30); plt.title(f'Distribucija godina ({age_col_eda})'); plt.xlabel("Godine"); plt.ylabel("Frekvencija"); plt.show()
else:
    print(f"Stupac '{age_col_eda}' nije pronađen za EDA.")

if fare_col_eda in df.columns:
    plt.figure(figsize=(8,5)); sns.histplot(df[df[fare_col_eda] < 300][fare_col_eda].dropna(), kde=True, bins=40); plt.title(f'Distribucija cijene karte ({fare_col_eda}) (Karte < 300)'); plt.xlabel("Cijena karte"); plt.ylabel("Frekvencija"); plt.show()
else:
    print(f"Stupac '{fare_col_eda}' nije pronađen za EDA.")

# Stopa preživljavanja po kategorijskim značajkama
categorical_eda_plot_features = []
if sex_col_eda in df.columns: categorical_eda_plot_features.append(sex_col_eda)
if pclass_col_eda in df.columns: categorical_eda_plot_features.append(pclass_col_eda)
if embarked_col_eda in df.columns: categorical_eda_plot_features.append(embarked_col_eda)

for feature_for_plot in categorical_eda_plot_features:
    plt.figure(figsize=(8,5)); sns.barplot(x=feature_for_plot, y=target_col_name, data=df, errorbar=None); plt.title(f'Stopa preživljavanja po {feature_for_plot}'); plt.ylabel(f'Prosječna stopa preživljavanja ({target_col_name})'); plt.show()


# --- 3. Odabir značajki i ciljne varijable za model ---
print("\n--- 3. Odabir značajki za model ---")
features_from_csv = ['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']

# Provjerimo koje od ovih značajki stvarno postoje u DataFrameu
actual_features_list = []
for feature_name in features_from_csv:
    if feature_name in df.columns:
        actual_features_list.append(feature_name)
    else:
        print(f"Upozorenje: Očekivana značajka '{feature_name}' nije pronađena u DataFrameu.")

if not actual_features_list:
    print("Greška: Nijedna od specificiranih značajki nije pronađena."); exit()

# Sortiramo radi konzistentnosti (opcionalno, ali dobra praksa)
actual_features_list = sorted(list(set(actual_features_list)))

print(f"Odabrane značajke za X: {actual_features_list}")
X = df[actual_features_list].copy()
y = df[target_col_name]



# --- 4. Definiranje numeričkih i kategorijskih značajki za Pipeline ---
# Logika kakva je bila u tvom "najboljem" kodu
print("\n--- 4. Definiranje numeričkih i kategorijskih značajki ---")
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()


pclass_like_cols = [col for col in numeric_features if 'pclass' in col.lower()]
for p_col in pclass_like_cols:
    if p_col in numeric_features:
        numeric_features.remove(p_col)
    if p_col not in categorical_features:
        categorical_features.append(p_col)

numeric_features = sorted(list(set(numeric_features)))
categorical_features = sorted(list(set(categorical_features)))

print(f"Konačne numeričke značajke za model (prema originalnoj logici): {numeric_features}")
print(f"Konačne kategorijske značajke za model (prema originalnoj logici): {categorical_features}")


# --- 5. Preprocessing Pipeline ---
print("\n--- 5. Definiranje Preprocessing Pipeline-a ---")

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# --- 6. Model Pipeline ---
print("\n--- 6. Definiranje Model Pipeline-a ---")

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced',n_estimators=30))
])

# --- 7. Unakrsna validacija (osnovni model s defaultnim parametrima) ---
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_scoring_metrics_list = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
#validacija - nije potrbna, bilo za test
#print("\n--- 7. Unakrsna validacija (Random Forest - default parametri) ---")
#try:
#    print("Rezultati za model s defaultnim parametrima (koristeći cross_val_score):")
#    for metric_name in all_scoring_metrics_list:
#
#        scores = cross_val_score(
#            rf_pipeline, X, y, cv=cv_strategy, scoring=metric_name, n_jobs=-1, error_score='raise'
#        )
#        mean_val = scores.mean()
#        std_val = scores.std()
#        display_metric_name = metric_name.replace('_macro', ' (macro)').capitalize()
#        print(f"{display_metric_name:<20}: {mean_val:.4f} ± {std_val:.4f}")
#except Exception as e_cv_default:
#    print(f"Greška tijekom unakrsne validacije defaultnog modela (s cross_val_score): {e_cv_default}")

# --- 8. Optimizacija Hiperparametara ---
print("\n--- 8. Optimizacija Hiperparametara ---")

param_dist_rf_search = {
    'classifier__n_estimators': randint(50, 350),
    'classifier__max_depth': list(range(3, 21)) + [None],
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 20),
    'classifier__class_weight': ['balanced', 'balanced_subsample', None]

}

scoring_dict_for_search = {
    'f1_macro': 'f1_macro',
    'roc_auc': 'roc_auc'   ,
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',

}
refit_this_metric = 'f1_macro'

random_search_optimizer = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_dist_rf_search,
    n_iter=150,
    scoring=scoring_dict_for_search,
    refit=refit_this_metric,
    cv=cv_strategy,
    n_jobs=-1,
    random_state=42,
    verbose=1,
    return_train_score=True,
)

try:
    print("Započinjanje RandomizedSearchCV...")
    random_search_optimizer.fit(X, y)
    print(f"\nNajbolji {refit_this_metric} (optimizacijska metrika): {random_search_optimizer.best_score_:.4f}")
    best_classifier_params = {k.split('__', 1)[1]: v for k, v in random_search_optimizer.best_params_.items()}
    print(f"Najbolji parametri za klasifikator: {best_classifier_params}")

    cv_search_results_df = pd.DataFrame(random_search_optimizer.cv_results_)
    print(f"\nTop 5 kombinacija parametara prema '{refit_this_metric}' (iz RandomizedSearchCV):")
    mean_test_score_cols_to_show = [f'mean_test_{key}' for key in scoring_dict_for_search.keys()]
    print(cv_search_results_df[['params'] + mean_test_score_cols_to_show].nlargest(5, f'mean_test_{refit_this_metric}'))

    best_model_after_optimization = random_search_optimizer.best_estimator_

    # --- 9. Finalna evaluacija i VIZUALIZACIJE za najbolji optimizirani model ---
    print(f"\n--- 9. Konačna evaluacija najboljeg modela (optimiziranog za {refit_this_metric}) ---")
    print("Rezultati za optimizirani model (koristeći cross_val_score - Min/Srednja/Max):")
    for metric_name in all_scoring_metrics_list:
        # cross_val_score vraća niz score-ova, po jedan za svaki fold
        scores = cross_val_score(
            best_model_after_optimization, X, y, cv=cv_strategy, scoring=metric_name, n_jobs=-1
        )
        min_val = scores.min()
        mean_val = scores.mean()
        max_val = scores.max()
        display_metric_name = metric_name.replace('_macro', ' (macro)').capitalize()
        print(
            f"{display_metric_name:<20}: Min={min_val:.4f}, Srednja={mean_val:.4f}, Max={max_val:.4f} (Raspon: {max_val - min_val:.4f})")

    print("\nGeneriranje matrice konfuzije i izvještaja o klasifikaciji...")
    # y_predictions_cv se generira za matricu konfuzije i classification_report,
    # ovaj dio ostaje isti jer se ne odnosi na prikaz Min/Mean/Max za cross_val_score.
    y_predictions_cv = cross_val_predict(best_model_after_optimization, X, y, cv=cv_strategy)

    cm_optimized = confusion_matrix(y, y_predictions_cv, labels=best_model_after_optimization.classes_)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm_optimized, display_labels=['Not Survived', 'Survived'])
    disp_cm.plot(cmap=plt.cm.Blues);
    plt.title("Matrica zabune za najbolji optimizirani model");
    plt.show()
    print("\nIzvještaj o klasifikaciji za najbolji optimizirani model (unakrsna validacija):")
    print(classification_report(y, y_predictions_cv, target_names=['Not Survived', 'Survived']))

    print("\nPokušaj generiranja grafa važnosti značajki...")
    try:

        if isinstance(best_model_after_optimization, ImbPipeline):
            preprocessor_from_best_model = best_model_after_optimization.named_steps['pipeline'].named_steps['preprocessor']
            classifier_from_best_model = best_model_after_optimization.named_steps['pipeline'].named_steps['classifier']
        else:
            preprocessor_from_best_model = best_model_after_optimization.named_steps['preprocessor']
            classifier_from_best_model = best_model_after_optimization.named_steps['classifier']

        transformed_feature_names = preprocessor_from_best_model.get_feature_names_out()
        feature_importances = classifier_from_best_model.feature_importances_
        importances_series = pd.Series(feature_importances, index=transformed_feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, max(6, len(importances_series[:15]) * 0.4)))
        sns.barplot(x=importances_series.values[:15], y=importances_series.index[:15])
        plt.title("Važnost značajki za Random Forest model"); plt.xlabel("Prosječno smanjenje nečistoće")
        plt.ylabel("Značajke"); plt.tight_layout(); plt.show()
    except Exception as e_feature_importance:
        print(f"Nije moguće generirati graf važnosti značajki: {e_feature_importance}")

except Exception as e_optimization:
    print(f"Greška tijekom optimizacije hiperparametara ili finalne evaluacije: {e_optimization}")
    if 'X' in locals() and 'preprocessor' in locals():
        print("Pokušavam transformirati X s definiranim preprocesorom da vidim gdje je problem:")
        try:
            X_transformed_debug = preprocessor.fit_transform(X)
            print(f"Oblik transformiranog X (debug): {X_transformed_debug.shape}")
            if np.isnan(X_transformed_debug).any():
                print("UPOZORENJE: Transformirani X (debug) sadrži NaN vrijednosti!")
        except Exception as e_transform_debug:
            print(f"Greška prilikom transformacije X (debug): {e_transform_debug}")

print("\n--- Skripta je završila ---")