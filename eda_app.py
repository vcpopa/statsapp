# Import dependencies
from multiprocessing.sharedctypes import Value
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import scipy.stats as stats
from scipy.stats import t
from scipy.stats import f_oneway
import statsmodels.api as sm
import scikit_posthocs as sp
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from PIL import Image
import sys
sys.tracebacklimit = 0

st.set_option('deprecation.showPyplotGlobalUse', False)

sns.set()
px.defaults.width = 850
px.defaults.height = 650



# Functions Exploratory Analysis
class EDA:

    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = self.df.columns
        self.num_vars = self.df.select_dtypes(include=[np.number]).columns
        self.cat_vars = self.df.select_dtypes(include=[np.object]).columns


        

    def box_plot(self, main_var, col_x=None, hue=None):
        return px.box(self.df, x=col_x, y=main_var, color=hue)

    def violin(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.violinplot(x=col_x, y=main_var, hue=hue,
                    data=self.df, palette="husl", split=split)

    def swarmplot(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.swarmplot(x=col_x, y=main_var, hue=hue,
                    data=self.df, palette="husl", dodge=split)
    
    def histogram_num(self, main_var, hue=None, bins = None, ranger=None):
        return  px.histogram(self.df[self.df[main_var].between(left = ranger[0], right = ranger[1])], \
            x=main_var, nbins =bins , color=hue, marginal='violin')

    def scatter_plot(self, col_x,col_y,hue=None, size=None):
        return px.scatter(self.df, x=col_x, y=col_y, color=hue,size=size)

    def bar_plot(self, col_y, col_x, hue=None):
        return px.bar(self.df, x=col_x, y=col_y,color=hue)
        
    def line_plot(self, col_y,col_x,hue=None, group=None):
        return px.line(self.df, x=col_x, y=col_y,color=hue, line_group=group)

    def CountPlot(self, main_var, hue=None):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.countplot(x=main_var, data=self.df, hue=hue, palette='pastel')
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def heatmap_vars(self,cols, func = np.mean):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.heatmap(self.df.pivot_table(index =cols[0], columns =cols[1],  values =cols[2], aggfunc=func, fill_value=0).dropna(axis=1), annot=True, annot_kws={"size": 7}, linewidths=.5)
        
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def Corr(self, cols=None, method = 'pearson'):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        if len(cols) != 0:
            corr = self.df[cols].corr(method = method)
        else:
            corr = self.df.corr(method = method)
        chart = sns.heatmap(corr, annot=True, annot_kws={"size": 7}, linewidths=.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30)
        chart.set_yticklabels(chart.get_yticklabels(), rotation=30)
        return chart
   
    def DistPlot(self, main_var):
        sns.set(style="whitegrid")
        return sns.distplot(self.df[main_var], color='c', rug=True)

    def one_sample_t_test(self,col_x,mean):
       
        if col_x is not None and is_numeric_dtype(self.df[col_x]):
            output_dict = {'Group':[],'Statistic':[],'P-value':[]}
            a=self.df[col_x].to_numpy()
            output=stats.ttest_1samp(a=a, popmean=mean,nan_policy='omit')
            output_dict['Group'].append(col_x)
            output_dict['Statistic'].append(output[0])
            output_dict['P-value'].append(output[1])
            output_table=pd.DataFrame.from_dict(output_dict)
            return output_table
            
        if is_numeric_dtype(self.df[col_x]) is False:
            raise Exception("One sample t-test can only be run against numeric columns!")

        if col_x is None:
            raise Exception ("Please input a column!")


    def two_sample_t_test(self,col_x,col_y):
    
        if is_numeric_dtype(self.df[col_x]) and is_numeric_dtype(self.df[col_y]):
            n1 = self.df[col_x].size
            n2 = self.df[col_y].size
            m1 = np.mean(self.df[col_x])
            m2 = np.mean(self.df[col_y])
            
            v1 = np.var(self.df[col_x], ddof=1)
            v2 = np.var(self.df[col_y], ddof=1)
            
            pooled_se = np.sqrt(v1 / n1 + v2 / n2)
            delta = m1-m2
            
            tstat = delta /  pooled_se
            df = (v1 / n1 + v2 / n2)**2 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))
            
            # two side t-test
            p = 2 * t.cdf(-abs(tstat), df)
            
            # upper and lower bounds
            lb = delta - t.ppf(0.975,df)*pooled_se 
            ub = delta + t.ppf(0.975,df)*pooled_se
        
            return pd.DataFrame(np.array([col_x,col_y,tstat,df,p,delta,lb,ub]).reshape(1,-1),
                                columns=['Group 1','Group 2','T statistic','df','P-value','MeanDiff','LowerBound','UpperBound'])
        if is_numeric_dtype(self.df[col_x]) is False or is_numeric_dtype(self.df[col_y]) is False:
            raise Exception ("Make sure both columns are numeric!")

    def ANOVA(self,col_x,col_y):
        output_dict = {'Group 1':[],'Group 2':[],'F-Value':[],'P-value':[]}
    
        output = f_oneway(self.df[col_x],self.df[col_y])
        
        output_dict['Group 1'].append(col_x)
        output_dict['Group 2'].append(col_y)
        output_dict['F-Value'].append(output[0])
        output_dict['P-value'].append(output[1])
        return pd.DataFrame.from_dict(output_dict)


    def tukey(self,col_x,col_y):
        tukey_output= pairwise_tukeyhsd(endog=self.df[col_x],
                          groups=self.df[col_y],
                          alpha=0.05)

        return pd.DataFrame(data=tukey_output._results_table.data[1:], columns=tukey_output._results_table.data[0])
        # tukey_output = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        # tukey_output.columns=['Group 1','Group 2','MeanDiff','P-value','Lower','Upper','RejectNullHypothesis']

    def dunn(self,col_x,col_y,correction='bonferroni'):
        dunn_output=sp.posthoc_dunn(a=self.df,val_col=col_x,group_col=col_y,p_adjust=correction)
        return dunn_output
                    
    def kw(self,col_x,col_y):
        output_dict = {'Group 1':[],'Group 2':[],'H-value':[],'P-value':[]}
        a = self.df[col_x]
        b = self.df[col_y]
        a=a.to_numpy()
        b=b.to_numpy()
        output=stats.kruskal(a,b)
        output_dict['Group 1'].append(col_x)
        output_dict['Group 2'].append(col_y)
        output_dict['H-value'].append(output[0])
        output_dict['P-value'].append(output[1])
        return pd.DataFrame.from_dict(output_dict)


    def chi_squarred(self,col_x,col_y):
        cross=pd.crosstab(self.df[col_x],self.df[col_y],margins=True,margins_name='Total')
        chi_square = 0
        rows = self.df[col_x].unique()
        columns = self.df[col_y].unique()
        for i in columns:
            for j in rows:
                O = cross[i][j]
                E = cross[i]['Total'] * cross['Total'][j] / cross['Total']['Total']
                chi_square += (O-E)**2/E
        return chi_square,len(rows),len(columns)

    def adf(self,col):
        data=self.df[col]
        stat, p, lags, obs, crit, t = adfuller(data)
        return stat,p

    def mw(self,col_x,col_y):
        a = self.df[col_x]
        b = self.df[col_y]
        stat, p = mannwhitneyu(a, b)
        return stat,p

    def shapiro(self,col):
        data=self.df[col]
        stat,p=stats.shapiro(data)
        return stat,p
   
def get_data(file):   
    
    read_cache_csv = st.cache(pd.read_csv, allow_output_mutation = True)
    read_cache_feather=st.cache(pd.read_feather,allow_output_mutation=True)
    read_cache_parquet=st.cache(pd.read_parquet,allow_output_mutation=True)
    if file.name.endswith('.csv'):
        df = read_cache_csv(file)
    if file.name.endswith('.feather'):
        df=read_cache_feather(file)
    if file.name.endswith('.parquet'):
        df=read_cache_parquet(file)
    
    return df

@st.cache
def get_stats(df):
    stats_num = df.describe()
    if df.select_dtypes(np.object).empty :
        return stats_num.transpose(), None
    if df.select_dtypes(np.number).empty :
        return None, df.describe(include=np.object).transpose()
    else:
        return stats_num.transpose(), df.describe(include=np.object).transpose()

@st.cache
def get_info(df):
    return pd.DataFrame({'types': df.dtypes, 'nan': df.isna().sum(), 'nan%': round((df.isna().sum()/len(df))*100,2), 'unique':df.nunique()})

def input_null(df, col, radio):
    df_inp = df.copy()

    if radio == 'Mean':
        st.write("Mean:", df[col].mean())
        df_inp[col] = df[col].fillna(df[col].mean())
    
    elif radio == 'Median':
        st.write("Median:", df[col].median())
        df_inp[col] = df[col].fillna(df[col].median())

    elif radio == 'Mode':
        for i in col:
            st.write(f"Mode {i}:", df[i].mode()[0])
            df_inp[i] = df[i].fillna(df[i].mode()[0])
        
    elif radio == 'Repeat last valid value':
        df_inp[col] = df[col].fillna(method = 'ffill')

    elif radio == 'Repeat next valid value':
        df_inp[col] = df[col].fillna(method = 'bfill')

    elif radio == 'Value':
        for i in col:
            number = st.number_input(f'Insert a number to fill missing values in {i}', format='%f', key=i)
            df_inp[i] = df[i].fillna(number)
    
    elif radio == 'Drop rows with missing values':
        if type(col) != list:
            col = [col]
        df_inp = df.dropna(axis=0, subset=col)
        st.markdown("Rows dropped!")
        st.write('raw # of rows ', df.shape[0], ' || preproc # of rows ', df_inp.shape[0])

    st.table(get_na_info(df_inp, df, col)) 
    
    return df_inp

def input_null_cat(df, col, radio):
    df_inp = df.copy()

    if radio == 'Text':
        for i in col:
            user_text = st.text_input(f'Replace missing values in {i} with', key=i)
            df_inp[i] = df[i].fillna(user_text)
    
    elif radio == 'Drop rows with missing values':
        if type(col) != list:
            col = [col]
        df_inp = df.dropna(axis=0, subset=col)
        st.markdown("Rows dropped!")
        st.write('raw # of rows ', df.shape[0], ' || preproc # of rows ', df_inp.shape[0])

    st.table(pd.concat([get_info(df[col]),get_info(df_inp[col])], axis=0))
    
    return df_inp

@st.cache
def get_na_info(df_preproc, df, col):
    raw_info = pd_of_stats(df, col)
    prep_info = pd_of_stats(df_preproc,col)
    return raw_info.join(prep_info, lsuffix= '_raw', rsuffix='_prep').T

@st.cache     
def pd_of_stats(df,col):
    #Descriptive Statistics
    stats = dict()
    stats['Mean']  = df[col].mean()
    stats['Std']   = df[col].std()
    stats['Var'] = df[col].var()
    stats['Kurtosis'] = df[col].kurtosis()
    stats['Skewness'] = df[col].skew()
    stats['Coefficient Variance'] = stats['Std'] / stats['Mean']
    return pd.DataFrame(stats, index = col).T.round(2)

@st.cache   
def pf_of_info(df,col):
    info = dict()
    info['Type'] =  df[col].dtypes
    info['Unique'] = df[col].nunique()
    info['n_zeros'] = (len(df) - np.count_nonzero(df[col]))
    info['p_zeros'] = round(info['n_zeros'] * 100 / len(df),2)
    info['nan'] = df[col].isna().sum()
    info['p_nan'] =  (df[col].isna().sum() / df.shape[0]) * 100
    return pd.DataFrame(info, index = col).T.round(2)

@st.cache     
def pd_of_stats_quantile(df,col):
    df_no_na = df[col].dropna()
    stats_q = dict()

    stats_q['Min'] = df[col].min()
    label = {0.25:"Q1", 0.5:'Median', 0.75:"Q3"}
    for percentile in np.array([0.25, 0.5, 0.75]):
        stats_q[label[percentile]] = df_no_na.quantile(percentile)
    stats_q['Max'] = df[col].max()
    stats_q['Range'] = stats_q['Max']-stats_q['Min']
    stats_q['IQR'] = stats_q['Q3']-stats_q['Q1']
    return pd.DataFrame(stats_q, index = col).T.round(2)    

@st.cache
def make_downloadable_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def plot_univariate(obj_plot, main_var, radio_plot_uni):
    
    if radio_plot_uni == 'Histogram' :
        st.subheader('Histogram')
        bins, range_ = None, None
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None))
        bins_ = st.sidebar.slider('Number of bins optional', value = 50)
        range_ = st.sidebar.slider('Choose range optional', float(obj_plot.df[main_var].min()), \
            float(obj_plot.df[main_var].max()),(float(obj_plot.df[main_var].min()),float(obj_plot.df[main_var].max())))    
        if st.sidebar.button('Plot histogram chart'):
            st.plotly_chart(obj_plot.histogram_num(main_var, hue_opt, bins_, range_))
    
    if radio_plot_uni ==('Distribution Plot'):
        st.subheader('Distribution Plot')
        if st.sidebar.button('Plot distribution'):
            fig = obj_plot.DistPlot(main_var)
            st.pyplot()  

    if radio_plot_uni == 'BoxPlot' :
        st.subheader('Boxplot')
        # col_x, hue_opt = None, None
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None), key ='boxplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key ='boxplot')
        if st.sidebar.button('Plot boxplot chart'):
            st.plotly_chart(obj_plot.box_plot(main_var,col_x, hue_opt))

def plot_multivariate(obj_plot, radio_plot):

    if radio_plot == ('Boxplot'):
        st.subheader('Boxplot')
        col_y  = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key ='boxplot')
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None), key ='boxplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key ='boxplot')
        if st.sidebar.button('Plot boxplot chart'):
            st.plotly_chart(obj_plot.box_plot(col_y,col_x, hue_opt))
    
    if radio_plot == ('Violin'):
        st.subheader('Violin')
        col_y  = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='violin')
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None),key='violin')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='violin')
        split = st.sidebar.checkbox("Split",key='violin')
        if st.sidebar.button('Plot violin chart'):
            fig = obj_plot.violin(col_y,col_x, hue_opt, split)
            st.pyplot()
    
    if radio_plot == ('Swarmplot'):
        st.subheader('Swarmplot')
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='swarmplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None),key='swarmplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='swarmplot')
        split = st.sidebar.checkbox("Split", key ='swarmplot')
        if st.sidebar.button('Plot swarmplot chart'):
            fig = obj_plot.swarmplot(col_y,col_x, hue_opt, split)
            st.pyplot()

    def pretty(method):
        return method.capitalize()

    if radio_plot == ('Correlation'):
        st.subheader('Heatmap Correlation Plot')
        correlation = st.sidebar.selectbox("Choose the correlation method", ('pearson', 'kendall','spearman'), format_func=pretty)
        cols_list = st.sidebar.multiselect("Select columns",obj_plot.columns)
        st.sidebar.markdown("If None selected, it will plot the correlation of all numeric variables.")
        if st.sidebar.button('Plot heatmap chart'):
            fig = obj_plot.Corr(cols_list, correlation)
            st.pyplot()

    def map_func(function):
        dic = {np.mean:'Mean', np.sum:'Sum', np.median:'Median'}
        return dic[function]
    
    if radio_plot == ('Heatmap'):
        st.subheader('Heatmap between vars')
        st.markdown(" In order to plot this chart remember that the order of the selection matters, \
            chooose in order the variables that will build the pivot table: row, column and value.")
        cols_list = st.sidebar.multiselect("Select 3 variables (2 categorical and 1 numeric)",obj_plot.columns, key= 'heatmapvars')
        agg_func = st.sidebar.selectbox("Choose one function to aggregate the data", (np.mean, np.sum, np.median), format_func=map_func)
        if st.sidebar.button('Plot heatmap between vars'):
            fig = obj_plot.heatmap_vars(cols_list, agg_func)
            st.pyplot()
    
    if radio_plot == ('Histogram'):
        st.subheader('Histogram')
        col_hist = st.sidebar.selectbox("Choose main variable", obj_plot.num_vars, key = 'hist')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'hist')
        bins_, range_ = None, None
        bins_ = st.sidebar.slider('Number of bins optional', value = 30)
        range_ = st.sidebar.slider('Choose range optional', int(obj_plot.df[col_hist].min()), int(obj_plot.df[col_hist].max()),\
                (int(obj_plot.df[col_hist].min()),int(obj_plot.df[col_hist].max())))    
        if st.sidebar.button('Plot histogram chart'):
                st.plotly_chart(obj_plot.histogram_num(col_hist, hue_opt, bins_, range_))

    if radio_plot == ('Scatterplot'): 
        st.subheader('Scatter plot')
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.num_vars, key = 'scatter')
        col_y = st.sidebar.selectbox("Choose y variable (numerical)", obj_plot.num_vars, key = 'scatter')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key = 'scatter')
        size_opt = st.sidebar.selectbox("Size (numerical) optional",obj_plot.columns.insert(0,None), key = 'scatter')
        if st.sidebar.button('Plot scatter chart'):
            st.plotly_chart(obj_plot.scatter_plot(col_x,col_y, hue_opt, size_opt))

    if radio_plot == ('Countplot'):
        st.subheader('Count Plot')
        col_count_plot = st.sidebar.selectbox("Choose main variable",obj_plot.columns, key = 'countplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'countplot')
        if st.sidebar.button('Plot Countplot'):
            fig = obj_plot.CountPlot(col_count_plot, hue_opt)
            st.pyplot()
    
    if radio_plot == ('Barplot'):
        st.subheader('Barplot') 
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='barplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns,key='barplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical/numerical) optional", obj_plot.columns.insert(0,None),key='barplot')
        if st.sidebar.button('Plot barplot chart'):
            st.plotly_chart(obj_plot.bar_plot(col_y,col_x, hue_opt))

    if radio_plot == ('Lineplot'):
        st.subheader('Lineplot') 
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='lineplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns,key='lineplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='lineplot')
        group = st.sidebar.selectbox("Group color (categorical) optional", obj_plot.columns.insert(0,None),key='lineplot')
        if st.sidebar.button('Plot lineplot chart'):
            st.plotly_chart(obj_plot.line_plot(col_y,col_x, hue_opt, group))
    

def do_statistics(obj_plot, test_options):
    if test_options==('One-sample t-test'):
        st.subheader('One sample t-test')
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.columns,key='One sample t-test')
        mean=st.sidebar.number_input("Input a population mean to compare against",min_value=None,max_value=None)
        if st.sidebar.button("Run one sample t-test"):
            test_result=obj_plot.one_sample_t_test(col_x=col_x,mean=mean)
            st.table(test_result.astype(str))

    if test_options==('2 sample t-test'):
        st.subheader('2 sample t-test')
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.columns,key='2 sample t-test')
        col_y = st.sidebar.selectbox("Choose y variable (numerical)", obj_plot.columns,key='2 sample t-test')
        if st.sidebar.button("Run 2 sample t-test"):
            test_result=obj_plot.two_sample_t_test(col_x=col_x,col_y=col_y)
            st.table(test_result.astype(str))

    if test_options==('ANOVA'):
        st.subheader("Pairwise Anova")
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.columns,key='Pairwise ANOVA')
        col_y = st.sidebar.selectbox("Choose y variable (numerical)", obj_plot.columns,key='Pairwise ANOVA')
        if st.sidebar.button("Run ANOVA"):
            test_result=obj_plot.ANOVA(col_x=col_x,col_y=col_y)
            st.table(test_result.astype(str))

    if test_options==('Tukey Test'):
        st.subheader("Tukey Test")
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.columns,key='Tukey Test')
        col_y = st.sidebar.selectbox("Choose y variable (categorical)", obj_plot.columns,key='Tukey Test')
        if st.sidebar.button("Run Tukey Test"):
            test_result=obj_plot.tukey(col_x=col_x,col_y=col_y)
            st.table(test_result)

    if test_options==("Kruskal-Wallis Test"):
        st.subheader("Kruskal-Wallis Test")
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.columns,key='Kruskal-Wallis Test')
        col_y = st.sidebar.selectbox("Choose y variable (numerical)", obj_plot.columns,key='Kruskal-Wallis Test')
        if st.sidebar.button("Run Kruskal-Wallis Test"):
            test_result=obj_plot.kw(col_x=col_x,col_y=col_y)
            st.table(test_result.astype(str))
            

    if test_options==("Dunn's Test"):
        st.subheader("Dunn's Test")
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.columns,key='Kruskal-Wallis Test')
        col_y = st.sidebar.selectbox("Choose y variable (categorical)", obj_plot.columns,key='Kruskal-Wallis Test')
        if st.sidebar.button("Run Dunn's Test"):
            test_result=obj_plot.dunn(col_x,col_y)
            st.table(test_result.astype(str))

    if test_options==("Chi-Squarred Test"):
        st.subheader("Chi-Squarred Test")
        col_x=st.sidebar.selectbox("Choose a row variable", obj_plot.columns,key='Chi-Squarred')
        col_y=st.sidebar.selectbox("Choose a column variable", obj_plot.columns,key='Chi-Squarred')
        alpha=0.05
        if st.sidebar.button("Run Chi-Squarred Test"):
            chi_result,chi_row,chi_col=obj_plot.chi_squarred(col_x,col_y)
            p_value = 1 - stats.chi2.cdf(chi_result, (chi_row-1)*(chi_col-1))
            conclusion = "Failed to reject the null hypothesis."
            if p_value <= alpha:
                conclusion = "Null Hypothesis is rejected."
            st.markdown(f"Chi-Squarred score: {chi_result}. P-value: {p_value}")
            st.markdown(conclusion)

    if test_options==("Augmented Dickey-Fuller"):
        st.subheader("Augmented Dickey-Fuller")
        col=st.sidebar.selectbox("Choose a numeric column",obj_plot.columns,key='Dickey-Fuller')
        if st.sidebar.button("Run Augmented Dickey-Fuller Unit Root Test"):
            adf_stat,adf_p=obj_plot.adf(col)
            st.markdown(f"Stat: {adf_stat}. P-value: {adf_p}")
            if (adf_p > 0.05):
                st.markdown('Probably not Stationary')
            else:
                st.markdown('Probably Stationary')

    if test_options==("Mann-Whitney"):
        st.subheader("Mann-Whitney")
        col_x=st.sidebar.selectbox("Choose a numeric variable", obj_plot.columns,key='Mann-Whitney')
        col_y=st.sidebar.selectbox("Choose a numeric variable", obj_plot.columns,key='Mann-Whitney2')
        if st.sidebar.button("Run Mann-Whitney Test"):
            mw_stat,mw_p=obj_plot.mw(col_x,col_y)
            st.markdown(f"Stat: {mw_stat}. P-value: {mw_p}")
            if (mw_p > 0.05):
                st.markdown('The 2 columns have the same distribution')
            else:
                st.markdown('The 2 columns do not have the same distribution')

    if test_options==('Shapiro-Willks Normality Test'):
        st.subheader('Shapiro-Willks Normality Test')
        col=st.sidebar.selectbox("Choose a variable",obj_plot.columns,key='Shapiro')
        if st.sidebar.button('Run Shapiro-Willks'):
            sw_stat,sw_p=obj_plot.shapiro(col)
            st.markdown(f"Stat: {sw_stat}. P-value: {sw_p}")
            if (sw_p > 0.05):
                st.markdown('Gaussian distribution')
            else:
                st.markdown('Non-Gaussian distribution')



def main():
    # hide_streamlit_style = """
    #         <style>
    #         #MainMenu {visibility: hidden;}
    #         footer {visibility: hidden;}
    #         </style>
    #         """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    st.title('Exploratory Data Analysis Tool')
    # st.header('Analyze the descriptive statistics and the dsistribution of your data. Preview and save your graphics.')
    image_main = Image.open("./logo.png").resize((800, 200),Image.ANTIALIAS)
    image_side=Image.open("./logo2.png")

    st.image(image_main)
    st.sidebar.image(image_side)
    file  = st.sidebar.file_uploader(' ', type = ['csv','feather','parquet'])
    
    
 
    if file is not None:
        df = get_data(file)
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=[np.object]).columns

        def basic_info(df):
            # st.header("Data")
            st.write('Number of observations', df.shape[0]) 
            st.write('Number of variables', df.shape[1])
            st.write('Number of missing (%)',((df.isna().sum().sum()/df.size)*100).round(2))
        
        #Sidebar Menu
        options = ["View statistics","Data aggregation", "Statistic univariate", "Statistic multivariate",'Statistical tests']
        menu = st.sidebar.selectbox("Menu options", options)

        #Data statistics
        df_info = get_info(df)   
        if (menu == "View statistics"):

            # st.subheader("INTERACTIVE DATAFRAME")
            gb=GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
            gb.configure_default_column(groupable=False,editable=False)
            # gb.configure_side_bar() #Add a sidebar
            gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
            gridOptions = gb.build()
            grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT', 
            update_mode='MODEL_CHANGED', 
            fit_columns_on_grid_load=False,
            theme='alpine', #Add theme color to the table
            enable_enterprise_modules=True,
            height=550, 
            width='100%',
            reload_data=True
        )

            data = grid_response['data']
            selected = grid_response['selected_rows'] 
            df_interactive = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
            df_stat_num, df_stat_obj = get_stats(df)
             #Visualize data
            st.subheader("BASIC INFO")
            basic_info(df)
            st.subheader("NUMERICAL SUMMARY")
            st.table(df_stat_num)

            st.subheader("CATEGORICAL SUMMARY")
            st.table(df_stat_obj)
            st.subheader("MISSING VALUES")
            st.table(df_info.astype(str))
           
        if (menu=='Data aggregation'):
            st.header("DATA AGGREGATION")
            #Sidebar Menu
            agg_options = ["Group by","Pivot", "Pivot wide to long"]
            agg_menu = st.selectbox("Wrangling options", agg_options)
            if agg_menu=='Group by':
                groupby_cols=st.multiselect("Choose columns to aggregate data by",df.columns)
                agg_funcs=['min','max','sum','size','mean','median','sem','std','pct_change']
                agg_func=st.multiselect('Choose aggregation functions',agg_funcs)

                #get numeric cols
                if agg_func is not None and groupby_cols is not None and st.button("Aggregate data"):
                    gdf=pd.DataFrame(df.groupby(groupby_cols,as_index=False)[numeric_features].agg(agg_func))
                    st.table(gdf.astype(str).head())

                #download button
                    csv=gdf.to_csv(index=False).encode('utf-8')
                    st.download_button(
                "Press to Download aggregated data",
                    csv,
                        f"{', '.join(agg_func)} by {', '.join(groupby_cols)}.csv",
                        "text/csv",
                            key='download-csv'
    )

            if agg_menu=='Pivot wide to long':
                    id_cols=st.multiselect("Choose ID variables",df.columns)
                    val_cols=st.multiselect('Choose value variables',df.columns)

                    #get numeric cols
                    if id_cols is not None and val_cols is not None and st.button("Pivot data"):
                        melt_df=pd.melt(df,id_vars=id_cols,value_vars=val_cols)
                        st.table(melt_df.astype(str).head())

                    #download button
                        csv=melt_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                    "Press to Download aggregated data",
                        csv,
                            f"{', '.join(id_cols)} by {', '.join(val_cols)}.csv",
                            "text/csv",
                                key='download-csv'
        )

            if agg_menu=='Pivot':
                    id_cols=st.multiselect("Choose ID variables",df.columns)
                    val_cols=st.multiselect('Choose value variables',df.columns)
                    agg_funcs_piv=st.multiselect('Choose an aggregation function',['min','max','sum','size','mean','median','sem','std','pct_change'])

                    #get numeric cols
                    if id_cols is not None and val_cols is not None and agg_func is not None and st.button("Pivot data"):
                        piv_df=pd.pivot_table(df,index=id_cols,values=val_cols,aggfunc=agg_funcs_piv)
                        st.table(piv_df.astype(str).head())

                    #download button
                        csv=piv_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                    "Press to Download aggregated data",
                        csv,
                            f"{', '.join(id_cols)} by {', '.join(val_cols)}.csv",
                            "text/csv",
                                key='download-csv'
        )

        eda_plot = EDA(df) 


        # Visualize data

       


        if (menu =="Statistic univariate" ):
            st.header("Statistic univariate")
            st.markdown("Provides summary statistics of only one variable in the raw dataset.")
            main_var = st.selectbox("Choose one numeric variable to analyze:", df[numeric_features].columns.insert(0,None))

            if main_var in numeric_features:
                if main_var != None:
                    st.subheader("Variable info")

                    
                    st.table(pf_of_info(df, [main_var]).T.astype(str))
                    st.subheader("Descriptive Statistics")
                    st.table((pd_of_stats(df, [main_var])).T.astype(str))
                    st.subheader("Quantile Statistics") 
                    st.table((pd_of_stats_quantile(df, [main_var])).T.astype(str)) 
                else:
                    st.markdown('Numerical statistics unavailable for categorical variables')

                    
                    chart_univariate = st.sidebar.radio('Chart', ('None','Histogram', 'BoxPlot', 'Distribution Plot'))
                    
                    plot_univariate(eda_plot, main_var, chart_univariate)

            if main_var in categorical_features:
                st.table(df[main_var].describe(include = np.object))
                st.bar_chart(df[main_var].value_counts().to_frame())

            st.sidebar.subheader("Explore other categorical variables!")
            var = st.sidebar.selectbox("Check its unique values and its frequency:", df.columns.insert(0,None))
            if var !=None:
                aux_chart = df[var].value_counts(dropna=False).to_frame()
                data = st.sidebar.table(aux_chart.style.bar(color='#3d66af'))

        if (menu =="Statistic multivariate" ):
            st.header("Statistic multivariate")

            st.markdown('Here you can visualize your data by choosing one of the chart options available on the sidebar!')
               
            st.sidebar.subheader('Data visualization options')
            radio_plot = st.sidebar.radio('Choose plot style', ('Correlation', 'Boxplot', 'Violin', 'Swarmplot', 'Heatmap', 'Histogram', \
                'Scatterplot', 'Countplot', 'Barplot', 'Lineplot'))

            plot_multivariate(eda_plot, radio_plot)

        if (menu =='Statistical tests'):
            st.header("Statistical tests")
            st.markdown('Here you can visualize your statistics test results by choosing one of the test options available on the sidebar!')
            st.sidebar.subheader('Statistical tests')
            test_options = st.sidebar.radio('Choose test to run', ('One-sample t-test', '2 sample t-test', 'ANOVA', 'Tukey Test', 'Kruskal-Wallis Test', "Dunn's Test","Chi-Squarred Test","Augmented Dickey-Fuller","Mann-Whitney",'Shapiro-Willks Normality Test'))
            do_statistics(eda_plot,test_options)
if __name__ == '__main__':
    main()