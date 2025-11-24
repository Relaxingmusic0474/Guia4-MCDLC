import pandas as pd
from scipy import stats
from math import sqrt

def get_range_outlier(
        q1:float, 
        q3:float, 
        IQR:float, 
        factor_value:float=1.5):

    min_value = q1 - IQR*factor_value
    max_value = q3 + IQR*factor_value

    return min_value, max_value

def check_is_outlier(value, min_value, max_value):
    
    if value > max_value or value < min_value:
        return True
    else:
        return False

def generate_df_count_nulls(df):
    rows = df.shape[0]

    df_count_nulls = pd.DataFrame()

    for column in df.columns:
        df_count_nulls[column] = [df[column].isna().sum()]

    # Se transpone y se agrega nombre de columna
    df_count_nulls = df_count_nulls.T
    df_count_nulls.columns = ["count_nulls"]

    # Columna con el porcentaje de datos nulos para cada descriptor
    df_count_nulls["pct_nulls"] = round(df_count_nulls["count_nulls"] / rows * 100, 1)

    return df_count_nulls    

def generate_df_counts(df_values, columns_name, verbose:bool=False):
    data_rows = []

    for column in df_values.columns:
        counts = df_values[column].value_counts()

        row = [column, 0, 0] # generamos una fila

        if 1 in counts.index: # preguntamos si se identificaron nulos
            row[1] = counts[1] # lo asignamos al espacio del nulos en la fila
        if 0 in counts.index: # preguntamos si se identificaron no nulos
            row[2] = counts[0] # lo asignamos al espacio de no nulos
        
        if verbose:
            print(row)
        data_rows.append(row) # la fila la agregamos a la matriz

    # usamos la matriz para generar un data frame.
    df_counts = pd.DataFrame(data=data_rows, columns=columns_name)
    return df_counts


def categorize_iqr(value):
    if value >0:
        return 1
    else:
        return 0
        
def calculate_ic_known_std(muestral_mean, population_std, n, trust_level=0.95):
    z_crit = stats.norm.ppf((1+trust_level)/2)
    min = muestral_mean - z_crit * (population_std/sqrt(n))
    max = muestral_mean + z_crit * (population_std/sqrt(n))
    return (min, max)

def calculate_ic_unknown_std(muestral_mean, muestral_std, n, trust_level=0.95):
    freedom_degrees = n-1
    t_crit = stats.t.ppf((1+trust_level)/2, freedom_degrees)
    min = muestral_mean - t_crit * (muestral_std/sqrt(n))
    max = muestral_mean + t_crit * (muestral_std/sqrt(n))
    return (min, max)

def calculate_ic_mean(mean, std, n, trust_level=0.95, known_std=False):
    if known_std:
        crit = stats.norm.ppf((1 + trust_level) / 2)
    else:
        crit = stats.t.ppf((1 + trust_level) / 2, df=n - 1)

    margin_error = crit * (std / sqrt(n))
    min = mean - margin_error
    max = mean + margin_error
    return (min, max)

def calculate_ic_variance(muestral_std, n, trust_level=0.95):
    freedom_degrees = n - 1
    alpha = 1 - trust_level

    chi2_low = stats.chi2.ppf(alpha / 2, df=freedom_degrees)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=freedom_degrees)

    s2 = muestral_std ** 2
    var_min = (freedom_degrees * s2) / chi2_high
    var_max = (freedom_degrees * s2) / chi2_low

    return (var_min, var_max)

def calculate_ic_std(muestral_std, n, trust_level=0.95):
    min, max = calculate_ic_variance(muestral_std, n, trust_level)
    std_min = sqrt(min)
    std_max = sqrt(max)
    return (std_min, std_max)