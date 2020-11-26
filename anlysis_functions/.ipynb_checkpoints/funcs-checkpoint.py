import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bidi import algorithm as bidialg
from IPython.display import display
from scipy.stats import chi2_contingency
from scipy.stats import chi2
sns.color_palette("pastel")

def get_groupby_data(df, cols):
    tmp = df.groupby(cols).count()[["id"]]
    tmp = tmp.assign(prec=lambda x: x/tmp.id.sum())
    tmp.columns = ["count(id)", "prec%"]
    tmp.reset_index(inplace=True)
    
    return tmp

def get_groupby_plot(df, cols, figsize=(6,4), with_prec=True, colors="Blues"):
    """ RETURN barplot of the groupby data based on cols[0] (count(id)) when the second column is the hue"""
    sns.set(font_scale=0.7)  # crazy big
    tmp = get_groupby_data(df,cols)
    plt.figure(figsize=figsize)
    
    if len(cols) == 1:
        ax = sns.barplot(x=cols[0], y="count(id)", data=tmp, 
                         ci=False, palette=colors)
        xlabels = []
        if with_prec:
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                text = str(round(tmp["prec%"][i]*100,2)) + "%"               
                ax.text(p.get_x()+p.get_width()/2., height-(height/5), text, ha="center", color="black")
                xlabels.append(bidialg.get_display(str(tmp[cols[0]][i])))
            ax.set_xticklabels(xlabels, ha="center")
        else:
            for i, p in enumerate(ax.patches):
                xlabels.append(bidialg.get_display(str(tmp[cols[0]][i])))
            ax.set_xticklabels(xlabels, ha="center")

    else:
        ax = sns.barplot(x=cols[0], y="count(id)", data=tmp, hue=tmp[cols[1]], ci=False, palette=colors)
        xlabels=[]
        if with_prec:
            text_lables = tmp.sort_values(cols[1])["prec%"].values
            text_lables = [str(round(i*100,2))+"%" for i in text_lables]
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2., height, text_lables[i],
                        ha="center", color="black")
        
        for i in tmp.sort_values(cols[1])[cols[0]].unique():
            xlabels.append(bidialg.get_display(str(i)))
        ax.set_xticklabels(xlabels, ha="center")
        
        h, l = ax.get_legend_handles_labels()
        legend_labels = [bidialg.get_display(str(i)) for i in tmp[cols[1]].unique()]
        ax.legend(h, legend_labels, title=cols[1], loc='best', bbox_to_anchor=(1, 0.5), ncol=1)
    
    ax.set_title("Group By " + cols[0], fontsize=15)
    return ax

def get_distributions(df, col):
    """ RETURN the distribution of the column """
    tmp=df.groupby(col).count()[["id"]]
    tmp.insert(loc = 1, column = 'prec%', value = (tmp[["id"]]/tmp.sum()['id'])*100)

    return tmp.round(2)

def get_crosstab_table(df, cols, target_col):
    """ RETURN 2 DF's crosstable and chi-square table """
    res = pd.DataFrame()
    res_cols = [df[col] for col in cols]
    tmp1 = pd.crosstab(res_cols, df[target_col], normalize='index')
    tmp2 = pd.crosstab(res_cols, df[target_col])
    tmp1 = pd.concat([tmp2,tmp1],axis=1)
    res = pd.concat([res,tmp1])
    res.columns = ['no', 'yes', 'no%','yes%']

    res.columns.name = target_col
    if len(cols) == 1:
        res.index.name = cols[0]
    res["total"] = res.drop(['no%','yes%'], axis=1).sum(axis=1)
    
    stat, p, dof, expected = chi2_contingency(res[["yes","no"]].to_numpy())
    prob = 0.95
    alpha = 1.0 - prob
    if p > alpha:
        conc = "independent"
    else: 
        conc = "dependent"

    tmp_df = pd.DataFrame(data={"p_value": p, "conclusion": conc}, index=[cols[0]])
    return res.round(2), tmp_df



def get_barplot(df, x,y, figsize=(6,4), colors="viridis_r", hue=None, rotation=0, legend=False):
    """ RETURN barplot based on seaborn lib"""
    
    sns.set(font_scale=0.7)
    plt.figure(figsize=figsize)
    
    ax = sns.barplot(x=x, y=y, data=df, hue=hue, ci=False, palette=colors)
    labels=[] 
    if df[x].dtype == "object":
        for i in df.sort_values(x)[x].unique():
            labels.append(bidialg.get_display(str(i)))
            ax.set_xticklabels(labels, ha="right")
            plt.xticks(rotation=45, ha="right")

    else:
        for i in df.sort_values(y)[y].unique():
            labels.append(bidialg.get_display(str(i)))
            ax.set_yticklabels(labels, ha="right")

    if legend:
        h, l = ax.get_legend_handles_labels()
        legend_labels = [bidialg.get_display(str(i)) for i in df[x].unique()]
        ax.legend(h, legend_labels, title=x, loc='best', bbox_to_anchor=(1, 0.5), ncol=1)

    ax.set_title(x + " ~ " + y, fontsize=15)
    return ax
