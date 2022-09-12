from scipy import stats


MODEL_NAME_LIST = ['{}_{}_{}'.format(model_structure_name, season_name, building_name)
                   for model_structure_name in ['lstm', 'sparse_lstm', 'sparse_ed', 'seq2seq_with_attention']
                   for season_name in ['CP4', 'DOH', 'OIE']
                   for building_name in ['spring', 'summer']]


def jaccard_similarity_coefficient(rank_1, rank_2, k):
    """
    smaller -> ???
    :param rank_1:
    :param rank_2:
    :param k:
    :return:
    """
    temp_rank_1 = rank_1[:k]
    temp_rank_2 = rank_2[:k]
    numerator = set(temp_rank_1).intersection(set(temp_rank_2))
    denominator = set(temp_rank_1).union(set(temp_rank_2))
    return '{:.4f}'.format(len(numerator) / len(denominator))


def spearman_correlation_coefficient(rank_1, rank_2):
    temp_rank_1 = []
    temp_rank_2 = []
    for model in MODEL_NAME_LIST:
        for i in range(len(rank_1)):
            if rank_1[i] == model:
                temp_rank_1.append(i + 1)
                break
        for j in range(len(rank_2)):
            if rank_2[j] == model:
                temp_rank_2.append(j + 1)
                break
    return '{:.4f}'.format(stats.spearmanr(temp_rank_1, temp_rank_2)[0])


def kendall_correlation_coefficient(rank_1, rank_2):
    temp_rank_1 = []
    temp_rank_2 = []
    for model in MODEL_NAME_LIST:
        for i in range(len(rank_1)):
            if rank_1[i] == model:
                temp_rank_1.append(i + 1)
                break
        for j in range(len(rank_2)):
            if rank_2[j] == model:
                temp_rank_2.append(j + 1)
                break
    return '{:.4f}'.format(stats.kendalltau(temp_rank_1, temp_rank_2)[0])


if __name__ == '__main__':
    rank_1 = ['lstm_CP4_spring', 'lstm_CP4_summer', 'lstm_DOH_spring', 'lstm_DOH_summer', 'lstm_OIE_spring',
              'lstm_OIE_summer', 'sparse_lstm_CP4_spring']
    rank_2 = ['lstm_OIE_summer', 'sparse_lstm_CP4_spring', 'sparse_lstm_CP4_summer', 'sparse_lstm_DOH_spring',
              'sparse_lstm_DOH_summer', 'sparse_lstm_OIE_summer', 'lstm_CP4_spring']
    print('jaccard_similarity_coefficient of top {} models: {}'.format(
        3, jaccard_similarity_coefficient(rank_1=rank_1, rank_2=rank_2, k=5)
    ))
    print('jaccard_similarity_coefficient of top {} models: {}'.format(
        5, jaccard_similarity_coefficient(rank_1=rank_1, rank_2=rank_2, k=7)
    ))
    print('spearman_correlation_coefficient: {}'.format(spearman_correlation_coefficient(rank_1, rank_2)))
    print('kendall_correlation_coefficient: {}'.format(kendall_correlation_coefficient(rank_1, rank_2)))

