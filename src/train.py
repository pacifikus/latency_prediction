import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score


def get_model_coef(model, feature_names):
    coefs = pd.DataFrame(
        model.coef_,
        columns=["Coefficients"],
        index=feature_names,
    )
    return coefs


def prepare_data(config):
    data = pd.read_csv(config['data']['processed_df_path'])

    X_train_, X_test_, y_train_, y_test_ = train_test_split(
        data.drop(['model_uuid', 'time'], axis=1),
        data['time'],
        random_state=config['base']['seed'],
        test_size=config['training']['test_size']
    )
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)
    X_test_ = scaler.transform(X_test_)
    features_names = data.drop(['model_uuid', 'time'], axis=1).columns.tolist()
    return X_train_, X_test_, y_train_, y_test_, features_names


def train_lr(X_train, X_test, y_train, y_test, f_names, config):
    lr = Ridge(
        random_state=config['base']['seed'],
        alpha=config['training']['ridge_alpha']
    )
    scores = cross_val_score(
        lr,
        X_train,
        y_train,
        cv=config['training']['cv'],
        scoring=config['training']['cv_scoring']
    )
    print(f'Cross-validateed scores: {scores}')
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    print(f'MAPE: {mean_absolute_percentage_error(y_test, preds)}')
    print(f'Model coeffs: {get_model_coef(lr, f_names)}')
