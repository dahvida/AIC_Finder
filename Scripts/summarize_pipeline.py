from summarize import *
import os

def main():
    path = "../Results/summary/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    mean_list = get_means()
    mean_list[0].to_csv(path + "mvsa.csv")
    mean_list[1].to_csv(path + "catboost.csv")
    mean_list[2].to_csv(path + "score.csv")
    mean_list[3].to_csv(path + "filter.csv")

    stats1, stats2, stats3 = get_stats()
    stats1.to_csv(path + "mvsa_vs_catboost.csv")
    stats2.to_csv(path + "mvsa_vs_score.csv")
    stats3.to_csv(path + "mvsa_vs_filter.csv")

    all1, all2, all3, all4 = get_all()
    all1.to_csv(path + "mvsa_all.csv")
    all2.to_csv(path + "catboost_all.csv")
    all3.to_csv(path + "score_all.csv")
    all4.to_csv(path + "filter_all.csv")

    print("[summarize]: Data summarized at", path)

if __name__ == "__main__":
    main()
