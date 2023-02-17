import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def game_domain_category(df, k, nb_players, id_selected_game):
    # TREATMENT SUGGESTED PLAYER
    # for empty list, fill with all numbers of player
    # and transform int in str before ','.join()
    df['suggested_player'] = df.apply(
        lambda row: row['suggested_player'] if len(row['suggested_player']) > 0 else [i for i in
                                                                                      range(row['min_player'],
                                                                                            row['max_player'] + 1)],
        axis=1)
    # si on a renseigné un nombre de joueurs pour jouer, on flag les jeux recommandés pour ce nombre de joueurs
    if nb_players is not None:
        df['player_flag'] = df['suggested_player'].apply(lambda mylist: 1 if nb_players in mylist else 0)
    # sinon on met 1 tout le temps
    else:
        df['player_flag'] = 1

    # TREATMENT DOMAIN (get dummies)
    df['domain_list_str'] = df['domain_list'].apply(lambda x: ','.join(x))
    df_domain = df['domain_list_str'].str.get_dummies(',')
    col_domain = df_domain.columns.to_list()
    df = pd.concat([df, df_domain], axis=1)
    df.drop(['domain_list_str'], axis=1, inplace=True)

    # TREATMENT CAT_LIST (get dummies)
    df['cat_list_str'] = df['cat_list'].apply(lambda x: ','.join(x))
    df_cat = df['cat_list_str'].str.get_dummies(',')
    col_cat = df_cat.columns.to_list()
    df = pd.concat([df, df_cat], axis=1)
    df.drop(['cat_list_str'], axis=1, inplace=True)
    df.fillna(0, inplace=True)

    # TREATMENT MECHA with Cosine_similarity
    # transform list into string
    df['mecha_list_str'] = df['mecha_list'].apply(lambda x: ','.join(x))

    # Vectorize & similarity calculation
    count = CountVectorizer()
    count_matrix = count.fit_transform(df['mecha_list_str'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    idx_selected_game = df[df['id'] == id_selected_game].index.to_list()[0]
    score_series = pd.Series(cosine_sim[idx_selected_game]).sort_values(
        ascending=False)  # similarity scores in descending order
    score_series = score_series.to_frame()
    score_series.rename({0: 'match_mecha'}, axis=1, inplace=True)
    df = df.join(score_series)

    # RECOMMANDATION
    # split the selected game from the others
    df_game = df[df['id'] == id_selected_game]

    # delete game where nb players is not selected
    df_other_game = df[df['id'] != id_selected_game]
    df_other_game = df_other_game[df_other_game['player_flag'] == 1]
    print('other_game_shape', df_other_game.shape)

    col_X = ['bgg_rank',
             'ratio_play',
             'year',
             'age',
             'awards',
             'weight',
             'geek_rating',
             'match_mecha'] + col_domain + col_cat

    # Definition X
    X = df_other_game[col_X]
    X_game = df_game[col_X]

    scaler = MinMaxScaler()

    # fit & transform train
    X_scaled = scaler.fit_transform(X)
    X_game_scaled = scaler.transform(X_game)

    # weight on some columns
    X_scaled[:, 5] = X_scaled[:, 5] * 7
    X_scaled[:, 7] = X_scaled[:, 7] * 10

    X_game_scaled[:, 5] = X_game_scaled[:, 5] * 7
    X_game_scaled[:, 7] = X_game_scaled[:, 7] * 10

    # fit the model
    modelKNN = NearestNeighbors(n_neighbors=k).fit(X_scaled)

    dist, indice = modelKNN.kneighbors(X_game_scaled)

    df_result = df_other_game.iloc[indice[0], :]

    return df_result
