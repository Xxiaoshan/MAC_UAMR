import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 定义函数计算余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def cos_midu(epoch, vectors, label_c, args):
    # 筛选相同调制类型
    if args.ab_choose  == "RML201610A":
        label = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    else:
        label = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    num_label = len(label)
    m = 200
    histogram_sum = []

    for sig_type in range(0, num_label, 1):
        index_list = np.where(label_c == sig_type)
        index_num = index_list[0]
        vectors_choose = vectors[index_num]
        # n = len(vectors_choose)
        n_inter = 1000
        n = 500
        #随机生成和类内样本个数一致的类间样本标号
        vectors_left = np.delete(vectors, index_num, axis= 0)
        num_inter_class = np.random.randint(0, len(vectors_left), n_inter)
        vectors_inter_class_choose = vectors_left[num_inter_class]

        # n = 10
        # 计算两两向量的余弦相似度
        similarities = np.zeros((n, n))
        similarities_inter = np.zeros(n_inter)
        for i in range(n):
            for j in range(n):
                if i != j:
                # similarities_inter = np.zeros(n)
                #计算类内和类间的比值
                    for k in range(n_inter):
                         similarities_inter[k] = cosine_similarity(vectors_choose[i], vectors_inter_class_choose[k])
                    similarities_inter_mean = np.mean(similarities_inter)
                    similarities[i, j] = cosine_similarity(vectors_choose[i], vectors_choose[j]) / similarities_inter_mean

        eye_similarity = 1112 * np.eye(n)
        similarities = similarities + eye_similarity
        triang = np.ravel(similarities)  #展开矩阵
        flattend_similarity = np.delete(triang, np.where(triang > 1000))
        # flattend_similarity = similarities_inter_mean
        # flattend_similarity_sum = np.mean(flattend_similarity)



        # # 统计每个区间存在的个数
        # histogram = np.zeros(m)
        # for i in range(n):
        #     for j in range(n):
        #         if i != j:
        #             similarity = similarities[i, j]
        #             index = int((similarity - min_similarity) // interval)
        #             histogram[index-1] += 1
        if sig_type == 0:
            histogram_sum = flattend_similarity
        else:
            histogram_sum = np.hstack([histogram_sum, flattend_similarity])
    # 绘制密度分布柱状图
    # colors = ["black", "red"]
    d = plt.figure(figsize=(8, 6))
    # sns.kdeplot(histogram,shade=True,color='red')
    sns.set(style="whitegrid", palette=["#184991", "#745ea6", "#cd4f27"])
    sns.kdeplot(histogram_sum, fill=True)
    # plt.show()
    name_mx = 'MD_view/SP_b_' + str(m) + 'epoch' + str(epoch) + '_model_' + args.ab_choose + '_snr_' + str(args.snr_tat) + '.png'
    np.save("histogram_SP_b_" + str(epoch) + '_snr_' + str(args.snr_tat) + '_model_' + args.ab_choose + ".npy", histogram_sum)
    plt.xlabel("intra_cosine_similarity")
    plt.ylabel("Density")
    x = np.arange(m)
    # plt.xticks(x, [f"{min_similarity + i * interval:.2f}-{min_similarity + (i + 1) * interval:.2f}" for i in range(m)])
    # plt.show()
    plt.savefig(name_mx, transparent=True, dpi=800)
    plt.close(d)
# plt.bar(x, histogram)
# plt.xticks(x, [f"{min_similarity + i*interval:.2f}-{min_similarity + (i+1)*interval:.2f}" for i in range(m)])
# plt.xlabel("Similarity Range")
# plt.ylabel("Frequency")
# plt.title("Density Distribution of Cosine Similarity")
# plt.show()
