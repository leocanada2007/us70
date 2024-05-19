# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:54:31 2024

@author: Administrator
"""



from pandas_datareader import data as pdr
import streamlit as st, pandas as pd, numpy as np, yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
yf.pdr_override() # <== that's all it takes :-)
import cufflinks as cf
import datetime
from streamlit_autorefresh import st_autorefresh


import altair as alt



#==============================================================================
# Background Color Function
#==============================================================================

def bgLevels(df, fig, variable, level, mode, fillcolor, layer):
    """
    Set a specified color as background for given
    levels of a specified variable using a shape.
    
    Keyword arguments:
    ==================
    fig -- plotly figure
    variable -- column name in a pandas dataframe
    level -- int or float
    mode -- set threshold above or below
    fillcolor -- any color type that plotly can handle
    layer -- position of shape in plotly fiugre, like "below"
    
    """
    
    if mode == 'above':
        m = df[variable].gt(level)
    
    if mode == 'below':
        m = df[variable].lt(level)
        
    df1 = df[m].groupby((~m).cumsum())['DATE'].agg(['first','last'])

    for index, row in df1.iterrows():
        #print(row['first'], row['last'])
        fig.add_shape(type="rect",
                        xref="x",
                        yref="paper",
                        x0=row['first'],
                        y0=0,
                        x1=row['last'],
                        y1=1,
                        line=dict(color="rgba(0,0,0,0)",width=3,),
                        fillcolor=fillcolor,
                        layer=layer) 
    return(fig)


#%%
#==============================================================================
# Tab 1 Overall
#==============================================================================

def tab1():
    
    pct_columns = ['DJI', 'SP500', 'SP500PE', 'Gold_reserves_billion']
    diff_columns= ['Discount_Rate', 
                   'Indus_Production_YoY',
                   'CPI_YoY',
                   'Core_CPI_YoY',
                   'PPI_YoY',
                   'Unemployment_Rate',
                   'PCE_YoY',
                   'Real_GDP_YoY']
    
    df = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\daily.csv')
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    
    df1 = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\monthly.csv')
    df1['DATE'] = pd.to_datetime(df1['DATE']).dt.date
    
    df2 = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\quarterly.csv')
    df2['DATE'] = pd.to_datetime(df2['DATE']).dt.date
    
    df3 = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\weekly.csv')
    df3['DATE'] = pd.to_datetime(df3['DATE']).dt.date
    
    df_events = pd.read_excel(r'E:\Investments\Readings\US_Equity_70\data\events.xlsx')
    df_events['start'] = pd.to_datetime(df_events['start']).dt.date
    df_events['end'] = pd.to_datetime(df_events['end']).dt.date
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1945, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(2020, 12, 31))

    df = df[df['DATE'] >= start_date]
    df = df[df['DATE'] <= end_date]
    
    df1 = df1[df1['DATE'] >= start_date] 
    df1 = df1[df1['DATE'] <= end_date]
    
    df2 = df2[df2['DATE'] >= start_date]
    df2 = df2[df2['DATE'] <= end_date]
    
    df3 = df3[df3['DATE'] >= start_date]
    df3 = df3[df3['DATE'] <= end_date]
    
    df_events = df_events[df_events['end'] >= start_date]
    df_events = df_events[df_events['start'] <= end_date]
    
    
    delta = pd.DataFrame([])
    
    for c in range(2,len(df.columns)):
        
        ds = pd.DataFrame([[df.columns[c], df.iloc[-1,c] - df.iloc[0,c]]])
        delta = pd.concat([delta, ds])
        
    for c in range(2,len(df1.columns)):
        
        ds = pd.DataFrame([[df1.columns[c], df1.iloc[-1,c] - df1.iloc[0,c]]])
        delta = pd.concat([delta, ds])

    for c in range(2,len(df2.columns)):
        
        ds = pd.DataFrame([[df2.columns[c], df2.iloc[-1,c] - df2.iloc[0,c]]])
        delta = pd.concat([delta, ds])

    for c in range(2,len(df3.columns)):
        
        ds = pd.DataFrame([[df3.columns[c], df3.iloc[-1,c] - df3.iloc[0,c]]])
        delta = pd.concat([delta, ds]) 
        
    delta.columns = ['index', 'Growth']
    delta = delta[delta['index'].isin(diff_columns)]
        
    pct = pd.DataFrame([])
    
    for c in range(2,len(df.columns)):
        
        ds = pd.DataFrame([[df.columns[c], 100*(df.iloc[-1,c] - df.iloc[0,c])/df.iloc[0,c]]])
        pct = pd.concat([pct, ds])
        
    for c in range(2,len(df1.columns)):
        
        ds = pd.DataFrame([[df1.columns[c], 100*(df1.iloc[-1,c] - df1.iloc[0,c])/df1.iloc[0,c]]])
        pct = pd.concat([pct, ds])

    for c in range(2,len(df2.columns)):
        
        ds = pd.DataFrame([[df2.columns[c], 100*(df2.iloc[-1,c] - df2.iloc[0,c])/df2.iloc[0,c]]])
        pct = pd.concat([pct, ds])

    for c in range(2,len(df3.columns)):
        
        ds = pd.DataFrame([[df3.columns[c], 100*(df3.iloc[-1,c] - df3.iloc[0,c])/df3.iloc[0,c]]])
        pct = pd.concat([pct, ds])  
        
    pct.columns = ['index', 'Growth']
    pct = pct[pct['index'].isin(pct_columns)]
        
 
    change = pd.concat([pct, delta])
    change['color'] = np.where(change["Growth"]<0, 'red', 'green')
    # change.plot(x = 'index', y = 'Growth', kind='bar', title="Summary", color=change.positive.map({True: 'g', False: 'r'}), legend = False)
    fig_summary = px.bar(change,
                  x="index",
                  y="Growth",
                  color = 'color',
                  color_discrete_map = {"green":"green", "red":"red"})
    fig_summary.update_layout(showlegend=False)
    
    fig_event = px.timeline(df_events.sort_values('start'),
                  x_start="start",
                  x_end="end",
                  y="event",
                  text="remark",
                  color_discrete_sequence=["tan"])




    # Display
    st.title('概览') 
    st.plotly_chart(fig_summary)
    
    
    

    
    st.title('事件')
    st.plotly_chart(fig_event)
        

        


    

    
#==============================================================================
# Tab 2 Index
#==============================================================================

def tab2():
    
    df = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\daily.csv')
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1945, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(2020, 12, 31))

    df = df[df['DATE'] >= start_date]
    df = df[df['DATE'] <= end_date]
    
    # plotly setup DJI
    fig_dji = px.line(df, x=df['DATE'], y=['DJI'])
    fig_dji.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_dji.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_dji = bgLevels(df=df, fig = fig_dji, variable = 'USRECDM', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')

    # plotly setup SP500
    fig_spx = px.line(df, x=df['DATE'], y=['SP500'])
    fig_spx.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_spx.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_spx = bgLevels(df=df, fig = fig_spx, variable = 'USRECDM', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    






    # Display

    st.title('道琼斯')
    st.plotly_chart(fig_dji)
        
    st.title('标普500')
    st.plotly_chart(fig_spx)        
        

        


    

    

    
#==============================================================================
# Tab 3 Macro
#==============================================================================

def tab3():
    
    macro_columns = ['Indus_Production_YoY', 
                     'CPI_YoY',
                     'Core_CPI_YoY',
                     'PPI_YoY', 
                     'Unemployment_Rate', 
                     'Disposable_Income_YoY',
                     'PCE_YoY',
                     'Nominal_GDP_YoY', 
                     'Real_GDP_YoY']
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1945, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(2020, 12, 31))
    
    # Import Data
    
    df1 = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\monthly.csv')
    df1['DATE'] = pd.to_datetime(df1['DATE']).dt.date
    
    df2 = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\quarterly.csv')
    df2['DATE'] = pd.to_datetime(df2['DATE']).dt.date
    



    # Filter Data    

    
    df1 = df1[df1['DATE'] >= start_date]
    df1 = df1[df1['DATE'] <= end_date] 
    
    df2 = df2[df2['DATE'] >= start_date]
    df2 = df2[df2['DATE'] <= end_date] 
    

    
    
    # Mean Table
    df1_mean = df1.iloc[:, 2:].mean()
    df2_mean = df2.iloc[:, 2:].mean()
    df_mean = pd.concat([df1_mean, df2_mean])
    df_mean = pd.DataFrame(df_mean)
    df_mean.reset_index(inplace=True)
    df_mean.columns = ['Indicator', 'Mean']
    df_mean = df_mean[df_mean['Indicator'].isin(macro_columns)]
    
    
    # Plots
    
    # plotly setup Industrial Production 
    fig_ind = px.line(df1, x=df1['DATE'], y=['Indus_Production_YoY'])
    fig_ind.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_ind.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_ind = bgLevels(df=df1, fig = fig_ind, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    
    # plotly setup CPI PPI
    fig_inflation = px.line(df1, x=df1['DATE'], y=['CPI_YoY', 'PPI_YoY', 'Core_CPI_YoY'])
    fig_inflation.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_inflation.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_inflation = bgLevels(df=df1, fig = fig_inflation, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    
    # plotly setup GDP
    fig_gdp = px.line(df2, x=df2['DATE'], y=['Nominal_GDP_YoY', 'Real_GDP_YoY'])
    fig_gdp.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_gdp.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_gdp = bgLevels(df=df1, fig = fig_gdp, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    # plotly setup Unemployment
    fig_unemp = px.line(df1, x=df1['DATE'], y=['Unemployment_Rate'])
    fig_unemp.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_unemp.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_unemp = bgLevels(df=df1, fig = fig_unemp, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    # plotly setup Income & Outlay
    fig_IO = px.line(df1, x=df1['DATE'], y=['Disposable_Income_YoY', 'PCE_YoY'])
    fig_IO.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_IO.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_IO = bgLevels(df=df1, fig = fig_IO, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    

    
    
    
    # Display    
    
  

    st.title('GDP 同比')
    st.plotly_chart(fig_gdp)

    st.title('失业率')
    st.plotly_chart(fig_unemp)
    
    st.title('CPI PPI 同比')
    st.plotly_chart(fig_inflation)
    
    st.title('收入支出同比')
    st.plotly_chart(fig_IO)
    
    st.title('工业生产指数同比')
    st.plotly_chart(fig_ind)

    
    st.title('均值')
    st.table(df_mean)






#==============================================================================
# Tab 4 Rates
#==============================================================================

def tab4():
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1945, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(2020, 12, 31))
    
    df = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\daily.csv')
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    
    df = df[df['DATE'] >= start_date]
    df = df[df['DATE'] <= end_date]
    

    df1 = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\monthly.csv')
    df1['DATE'] = pd.to_datetime(df1['DATE']).dt.date
    
    df3 = pd.read_csv(r'E:\Investments\Readings\US_Equity_70\data\weekly.csv')
    df3['DATE'] = pd.to_datetime(df3['DATE']).dt.date
    
    df1 = df1[df1['DATE'] >= start_date]
    df1 = df1[df1['DATE'] <= end_date] 
    
    df3 = df3[df3['DATE'] >= start_date]
    df3 = df3[df3['DATE'] <= end_date] 
    
    
    # plotly setup Discount rate
    fid_discount_rate = px.line(df1, x=df1['DATE'], y=['Discount_Rate'])
    fid_discount_rate.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fid_discount_rate.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fid_discount_rate = bgLevels(df=df1, fig = fid_discount_rate, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    
    # plotly setup 10 Year Treasury
    fig_10y = px.line(df, x=df['DATE'], y=['DGS10'])
    fig_10y.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_10y.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_10y = bgLevels(df=df, fig = fig_10y, variable = 'USRECDM', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')  
    
    
    # plotly setup PE
    fig_pe = px.line(df1, x=df1['DATE'], y=['SP500PE'])
    fig_pe.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_pe.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_pe = bgLevels(df=df1, fig = fig_pe, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    
    
    # plotly setup Income & Outlay
    fig_gold = px.line(df3, x=df3['DATE'], y=['Gold_reserves_billion'])
    fig_gold.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_gold.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_gold = bgLevels(df=df1, fig = fig_gold, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    
    
    
    
    # Display
    
    
    
    st.title('美联储贴现利率')
    st.plotly_chart(fid_discount_rate)
    
    st.title('10年期国债收益')
    st.plotly_chart(fig_10y)
    
    st.title('黄金储备（十亿）')
    st.plotly_chart(fig_gold)
    
    st.title('标普500市盈率')
    st.plotly_chart(fig_pe)
    

    
    
    
    

        

#==============================================================================
# Main body
#==============================================================================

def run():
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    # ticker_list = ['AGG', 'PSI', 'SGOV', 'SPHY', 'VTHR']
    
# =============================================================================
#     # Add selection box
#     global ticker
#     ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
# =============================================================================
    
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['概览', '指数', '宏观', '资本市场'])

    # Show the selected tab
    if select_tab == '概览':
        tab1()
    elif select_tab == '指数':
        tab2()
    elif select_tab == '宏观':
        tab3()
    elif select_tab == '资本市场':
        tab4()      
        
      
 
       
    
if __name__ == "__main__":
    run()              