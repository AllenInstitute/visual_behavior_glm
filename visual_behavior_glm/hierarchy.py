import matplotlib.pyplot as plt

def plot_hierarchy(df):
    plt.figure(figsize=(8,3))
    count = 0
    areas = ['VISp','VISl']
    depths = ['75','175','275','375']

    for area in areas:
        for depth in depths:
            plt.plot(count, df.loc[area+'_'+depth]['hit'],'bo')
            plt.plot(count, df.loc[area+'_'+depth]['miss'],'rx')
            plt.plot(count, df.loc[area+'_'+depth]['repeat'],'o',color='gray')
            count +=1

    plt.ylabel('response')
    plt.xlabel('location')
    plt.tight_layout()

