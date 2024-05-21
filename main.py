import streamlit as st, pandas as pd, numpy as np
import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import pandas_ta as ta
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

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
    
    pct_columns = ['DJI', 'SP500', 'Gold_reserves_billion']
    diff_columns= ['SP500PE',
                   'Discount_Rate', 
                   'Indus_Production_YoY',
                   'CPI_YoY',
                   'Core_CPI_YoY',
                   'PPI_YoY',
                   'Unemployment_Rate',
                   'PCE_YoY',
                   'Real_GDP_YoY']
    
    cols = pct_columns + diff_columns
    
    df = pd.read_csv('data/daily.csv')
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    
    df1 = pd.read_csv('data/monthly.csv')
    df1['DATE'] = pd.to_datetime(df1['DATE']).dt.date
    
    df2 = pd.read_csv('data/quarterly.csv')
    df2['DATE'] = pd.to_datetime(df2['DATE']).dt.date
    
    df3 = pd.read_csv('data/weekly.csv')
    df3['DATE'] = pd.to_datetime(df3['DATE']).dt.date
    
    df_events = pd.read_csv('data/events.csv')
    df_events['start'] = pd.to_datetime(df_events['start']).dt.date
    df_events['end'] = pd.to_datetime(df_events['end']).dt.date
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1948, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(1949, 1, 1))

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
    
    std = pd.DataFrame([])
    
    for c in range(2,len(df.columns)):
        
        ds = pd.DataFrame([[df.columns[c], df.iloc[-1,c] - df.iloc[0,c]]])
        var = pd.DataFrame([[df.columns[c], df.iloc[:,c].std()]])
        delta = pd.concat([delta, ds])
        std = pd.concat([std, var])
        
    for c in range(2,len(df1.columns)):
        
        try:
            ds = pd.DataFrame([[df1.columns[c], df1.iloc[-1,c] - df1.iloc[0,c]]])
            var = pd.DataFrame([[df1.columns[c], df1.iloc[:,c].std()]])
            delta = pd.concat([delta, ds])
            std = pd.concat([std, var])
        except:
            pass
        
    for c in range(2,len(df2.columns)):
        
        try:
            ds = pd.DataFrame([[df2.columns[c], df2.iloc[-1,c] - df2.iloc[0,c]]])
            delta = pd.concat([delta, ds])
            var = pd.DataFrame([[df2.columns[c], df2.iloc[:,c].std()]])
            std = pd.concat([std, var])
        except:
            pass

    for c in range(2,len(df3.columns)):
        
        try:
            ds = pd.DataFrame([[df3.columns[c], df3.iloc[-1,c] - df3.iloc[0,c]]])
            delta = pd.concat([delta, ds]) 
            var = pd.DataFrame([[df3.columns[c], df3.iloc[:,c].std()]])
            std = pd.concat([std, var])
        except:
            pass
        
    delta.columns = ['index', 'value']
    std.columns = ['index', 'value']
    delta = delta[delta['index'].isin(diff_columns)]
    std = std[std['index'].isin(cols)]

        
    pct = pd.DataFrame([])
    
    for c in range(2,len(df.columns)):
        
        ds = pd.DataFrame([[df.columns[c], 100*(df.iloc[-1,c] - df.iloc[0,c])/df.iloc[0,c]]])
        pct = pd.concat([pct, ds])
        
    for c in range(2,len(df1.columns)):
        
        try:
            ds = pd.DataFrame([[df1.columns[c], 100*(df1.iloc[-1,c] - df1.iloc[0,c])/df1.iloc[0,c]]])
            pct = pd.concat([pct, ds])
        except:
            pass

    for c in range(2,len(df2.columns)):
        
        try:
            ds = pd.DataFrame([[df2.columns[c], 100*(df2.iloc[-1,c] - df2.iloc[0,c])/df2.iloc[0,c]]])
            pct = pd.concat([pct, ds])
        except:
            pass
        
    for c in range(2,len(df3.columns)):
        
        try:
            ds = pd.DataFrame([[df3.columns[c], 100*(df3.iloc[-1,c] - df3.iloc[0,c])/df3.iloc[0,c]]])
            pct = pd.concat([pct, ds])  
        except:
            pass
        
    pct.columns = ['index', 'value']
    pct = pct[pct['index'].isin(pct_columns)]
        
 
    change = pd.concat([pct, delta])

    
    change['stats'] = 'Delta'
    std['stats'] = 'STD'
    
    change = pd.concat([change, std])
    
    change['color'] = np.where(change["value"]<0, 'down', 'up')
    
    change.columns = ['Indicator', 'Value', 'Stats', 'color']
    

    
    fig_summary = px.bar(change,
                  x="Indicator",
                  y='Value',
                  color = 'Stats',
                  barmode = 'group',
                  pattern_shape="Stats"
                  )
    
    fig_summary.update_layout(legend_title="Stats", bargap=0.5,bargroupgap=0.1)
    
    fig_summary.for_each_trace(
        lambda trace: trace.update(marker_color=np.where(change.loc[change['Stats'].eq(trace.name), 'Value'] < 0, 'red', 'green'))
    )
    
    fig_event = px.timeline(df_events.sort_values('start'),
                  x_start="start",
                  x_end="end",
                  y="event",
                  text="remark",
                  color_discrete_sequence=["tan"])
    
    
    # plotly setup DJI
    fig_dji = px.line(df, x=df['DATE'], y=['DJI'])
    fig_dji.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_dji.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_dji = bgLevels(df=df, fig = fig_dji, variable = 'USRECDM', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')




    # Display

    
    
    
    
    st.title('概览') 
    st.plotly_chart(fig_summary)
    
    
    

    
    st.title('事件')
    st.plotly_chart(fig_event)
    
    
        
    st.title('道琼斯指数')
    st.plotly_chart(fig_dji)
        

        



    

    
#==============================================================================
# Tab 2 Index
#==============================================================================

def tab2():
    
    df = pd.read_csv('data/daily.csv')
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1948, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(1949, 1, 1))

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
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1948, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(1949, 1, 1))
    
    # Import Data
    
    df1 = pd.read_csv('data/monthly.csv')
    df1['DATE'] = pd.to_datetime(df1['DATE']).dt.date
    
    df2 = pd.read_csv('data/quarterly.csv')
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
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1948, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(1949, 1, 1))
    
    df = pd.read_csv('data/daily.csv')
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    
    df = df[df['DATE'] >= start_date]
    df = df[df['DATE'] <= end_date]
    

    df1 = pd.read_csv('data/monthly.csv')
    df1['DATE'] = pd.to_datetime(df1['DATE']).dt.date
    
    df3 = pd.read_csv('data/weekly.csv')
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
# Tab 5 Trends
#==============================================================================

def tab5():
    
    analysis_year = st.sidebar.number_input('Year', 1948)
    
    df = pd.read_csv('data/DJI.csv')
    
    df['year'] = pd.to_datetime(df['Date']).dt.year
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['DJI'] = df['DJI'].astype(float)
    
    df['atr'] = ta.atr(high = df['DJI'], low = df['DJI'], close = df['DJI'])
    df['atr'] = df.atr.rolling(window = 30).mean()
    
    df = df[df['Date'] >= '1948-01-01']
    df = df[df['Date'] <= '2018-12-31']
    df.set_index('Date', inplace = True)
    
    df_temp = df[df.year == analysis_year]
    
    fig, ax = plt.subplots()
    plt.xticks(rotation = -30)
    price, = ax.plot(df_temp.index, df_temp.DJI, c='grey', lw = 2, alpha=0.5, zorder = 5)
    
    df_temp['smoothed'] = savgol_filter(df_temp.DJI, 60, 5)
    fig, ax = plt.subplots()
    plt.xticks(rotation = -30)
    price, = ax.plot(df_temp.index, df_temp.DJI, c='grey', lw = 2, alpha=0.5, zorder = 5) 
    price_smooth, = ax.plot(df_temp.index, df_temp.smoothed, c='b', lw = 2, zorder = 5) 
    
    
    atr = df_temp.atr.iloc[-1]
    
    peaks_idx, _ = find_peaks(df_temp.smoothed, distance = 15, width = 3, prominence = atr)
    
    troughs_idx, _ = find_peaks(-1*df_temp.smoothed, distance = 15, width = 3, prominence = atr)
    
    peaks, = ax.plot(df_temp.index[peaks_idx], df_temp.smoothed.iloc[peaks_idx], \
                     c = 'r', linestyle = 'None', markersize = 10, marker = 'o', zorder = 10)
    
    troughs, = ax.plot(df_temp.index[troughs_idx], df_temp.smoothed.iloc[troughs_idx], \
                     c = 'g', linestyle = 'None', markersize = 10, marker = 'o', zorder = 10)
        
    ax.set_ylabel('DJI')    
    ax.set_title(analysis_year)  
    
   
    df_peak = pd.DataFrame(df_temp.smoothed.iloc[peaks_idx])
    df_peak['type'] = 'peak'
    df_trough = pd.DataFrame(df_temp.smoothed.iloc[troughs_idx])
    df_trough['type'] = 'trough'
    
    tp = pd.concat([df_peak, df_trough])
    
    tp = tp.sort_index()
    
    
    
    
    
    # Display
    
    
    
    st.title('区间划分')
    st.pyplot(fig)
    
    st.title('拐点')
    st.table(tp)
    
    
    
    

        

#==============================================================================
# Main body
#==============================================================================

def run():
    
    
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['概览', '指数', '宏观', '资本市场','区间划分'])

    # Show the selected tab
    if select_tab == '概览':
        tab1()
    elif select_tab == '指数':
        tab2()
    elif select_tab == '宏观':
        tab3()
    elif select_tab == '资本市场':
        tab4()  
    elif select_tab == '区间划分':
        tab5()    
        
if __name__ == "__main__":
    run()   
