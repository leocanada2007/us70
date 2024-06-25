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

    intervals = pd.read_csv(r'data/intervals.csv')
    intervals['Start'] = pd.to_datetime(intervals['Start']).dt.date
    intervals['End'] = pd.to_datetime(intervals['End']).dt.date     
    
    coll, colr = st.columns(2)
    
    with coll:   
        start_date = st.selectbox('Start date', options = intervals['Start'])
    
    intervals1 = intervals.loc[intervals['End']>start_date]    
    
    with colr:    
        end_date = st.selectbox('End date', options = intervals1['End'])
    
    timeline = intervals[intervals['Start'] >= start_date]
    timeline = timeline[timeline['End'] <= end_date]
    
    fig_event = px.timeline(timeline.sort_values('Start'),
                  x_start="Start",
                  x_end="End",
                  y="Summary",
                  # text="remark",
                  color_discrete_sequence=["tan"])
    
    
    st.plotly_chart(fig_event)

    start_year = start_date.replace(month=1, day=1) 
    end_year = end_date.replace(month=12, day=31) 
    
    df = df[df['DATE'] >= start_year]
    df = df[df['DATE'] <= end_year]
    
    df1 = df1[df1['DATE'] >= start_year] 
    df1 = df1[df1['DATE'] <= end_year]
    
    df2 = df2[df2['DATE'] >= start_year]
    df2 = df2[df2['DATE'] <= end_year]
    
    df3 = df3[df3['DATE'] >= start_year]
    df3 = df3[df3['DATE'] <= end_year]
    
    df_events = df_events[df_events['end'] >= start_year]
    df_events = df_events[df_events['start'] <= end_year]
    
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Market Plots
    
    df.plot(x = 'DATE', y = 'SP500')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("SP500", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    with col1:
        st.pyplot(fig=plt)
        
    df1.plot(x = 'DATE', y = 'SP500PE', title='SP500PE')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("SP500PE", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)

    
    with col2:
        st.pyplot(fig=plt)  
        
    if start_year <= datetime.date(1982, 12, 31):    
        
        df1.plot(x = 'DATE', y = 'Discount_Rate', title='Discount Rate')
        plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
        plt.xticks(rotation=45)
        plt.title("Discount Rate", fontsize = 24)
        plt.tick_params(axis='both', which='major', labelsize=16)        
            
        with col3:
            st.pyplot(fig=plt)
    else:
         df.plot(x = 'DATE', y = ['DFEDTARL', 'DFEDTARU'], title='Federal Fund Rate')
         plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
         plt.xticks(rotation=45)
         plt.title("Federal Fund Rate", fontsize = 24)
         plt.tick_params(axis='both', which='major', labelsize=16)
         
         with col3:
             st.pyplot(fig=plt) 
    
    
    df.plot(x = 'DATE', y = 'DJI', title='Dow Jones')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("Dow Jones", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    with col4:
        st.pyplot(fig=plt) 
        
    col5, col6, col7, col8 = st.columns(4)        
        
    df.plot(x = 'DATE', y = ['DTB3', 'DGS10'], title='Bond Yield: 3M Vs 10Y')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("Bond Yield: 3M Vs 10Y", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    with col5:
        st.pyplot(fig=plt)     
        
    df1.plot(x = 'DATE', y = ['CPI_YoY', 'PPI_YoY', 'Core_CPI_YoY'], title='(Core) CPI and PPI, YoY')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("'(Core) CPI and PPI, YoY'", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    with col6:
        st.pyplot(fig=plt)          
        
    df1.plot(x = 'DATE', y = ['Unemployment_Rate'], title='Unemployment Rate')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("Unemployment Rate", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)    

    
    with col7:
        st.pyplot(fig=plt)  
        
    df2.plot(x = 'DATE', y = ['Nominal_GDP_YoY', 'Real_GDP_YoY'], title='GDP, YoY')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("GDP, YoY", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    with col8:
        st.pyplot(fig=plt)     
        
    col9, col10, col11, col12 = st.columns(4)
        
    df1.plot(x = 'DATE', y = ['Indus_Production_YoY'], title='Industrial Production, YoY')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("Industrial Production, YoY", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    with col9:
        st.pyplot(fig=plt)    
        
    df2.plot(x = 'DATE', y = ['CP_QoQ'], title='Corporate Profit, QoQ')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("Corporate Profit, QoQ", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    with col10:
        st.pyplot(fig=plt)  


            

    df1.plot(x = 'DATE', y = ['RSAFS_MoM'], title='Retail Sales, MoM')
    plt.axvspan(start_date, end_date, color = 'red', alpha = 0.25)
    plt.xticks(rotation=45)
    plt.title("Retail Sales, MoM", fontsize = 24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    with col11:
        st.pyplot(fig=plt)   


    

    
#==============================================================================
# Tab 2 Index
#==============================================================================

def tab2():
    
    df = pd.read_csv('data/daily.csv')
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date

    df_events = pd.read_csv('data/events.csv')
    df_events['start'] = pd.to_datetime(df_events['start']).dt.date
    df_events['end'] = pd.to_datetime(df_events['end']).dt.date    
    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1948, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(1949, 1, 1))

    df = df[df['DATE'] >= start_date]
    df = df[df['DATE'] <= end_date]

    df_events = df_events[df_events['end'] >= start_date]
    df_events = df_events[df_events['start'] <= end_date]    
    
    # plotly setup DJI
    fig_dji = px.line(df, x=df['DATE'], y=['DJI'])
    fig_dji.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_dji.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_dji = bgLevels(df=df, fig = fig_dji, variable = 'USRECDM', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')


    # plotly setup Nasdaq
    fig_ixic = px.line(df, x=df['DATE'], y=['IXIC'])
    fig_ixic.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_ixic.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_ixic = bgLevels(df=df, fig = fig_ixic, variable = 'USRECDM', level = 0.5, mode = 'above',
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

    st.title('纳斯达克')
    st.plotly_chart(fig_ixic)
    
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

    
     # plotly setup Coporate Profit
    fig_cp = px.line(df2, x=df2['DATE'], y=['CP_QoQ'])
    fig_cp.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_cp.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_cp = bgLevels(df=df1, fig = fig_cp, variable = 'Recession', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')   
    
    # plotly setup Retail Sales
    fig_rs = px.line(df1, x=df1['DATE'], y=['RSAFS_MoM'])
    fig_rs.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_rs.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_rs = bgLevels(df=df1, fig = fig_rs, variable = 'Recession', level = 0.5, mode = 'above',
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

    st.title('企业盈利环比')
    st.plotly_chart(fig_cp)
    
    st.title('零售品总额环比')
    st.plotly_chart(fig_rs)
    
    # st.title('均值')
    # st.table(df_mean)






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

    # plotly setup Federal Fund rate
    fid_ff_rate = px.line(df, x=df['DATE'], y=['DFEDTARL', 'DFEDTARU'])
    fid_ff_rate.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fid_ff_rate.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fid_ff_rate = bgLevels(df=df, fig = fid_ff_rate, variable = 'USRECDM', level = 0.5, mode = 'above',
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


    # plotly setup 3 Month Year Treasury
    fig_3m = px.line(df, x=df['DATE'], y=['DTB3', 'DGS10'])
    fig_3m.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_3m.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_3m = bgLevels(df=df, fig = fig_3m, variable = 'USRECDM', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')


    # plotly setup WTI
    fig_wti = px.line(df, x=df['DATE'], y=['WTI'])
    fig_wti.update_xaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    fig_wti.update_yaxes(showgrid=False, gridwidth=1, gridcolor='rgba(0,0,255,0.1)')
    
    fig_wti = bgLevels(df=df, fig = fig_wti, variable = 'USRECDM', level = 0.5, mode = 'above',
                   fillcolor = 'rgba(100,100,100,0.2)', layer = 'below')
    
    
    
    
    
    # Display
    
    
    
    
    st.title('美联储贴现利率 (Discount Rate)')
    st.plotly_chart(fid_discount_rate)

    st.title('联邦基金目标利率（通道）')
    st.plotly_chart(fid_ff_rate)
    
    st.title('长(10Y)短(3M)期国债收益')
    st.plotly_chart(fig_3m)
    
    st.title('WTI原油价格')
    st.plotly_chart(fig_wti)

    st.title('标普500市盈率')
    st.plotly_chart(fig_pe)
    
    st.title('黄金储备（十亿）')
    st.plotly_chart(fig_gold)
    

    
    

    
#==============================================================================
# Tab 5 DJI Intervals
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
    
    
    
    st.title('道琼斯区间划分')
    st.pyplot(fig)
    
    st.title('拐点')
    st.table(tp)
    
    
    

#==============================================================================
# Tab 6 Nasdaq Intervals
#==============================================================================

def tab6():
    
    analysis_year = st.sidebar.number_input('Year', 1971)
    
    df = pd.read_csv('data/IXIC.csv')
    
    df['year'] = pd.to_datetime(df['Date']).dt.year
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['IXIC'] = df['IXIC'].astype(float)
    
    df['atr'] = ta.atr(high = df['IXIC'], low = df['IXIC'], close = df['IXIC'])
    df['atr'] = df.atr.rolling(window = 30).mean()
    
    df = df[df['Date'] >= '1971-02-05']
    df = df[df['Date'] <= '2018-12-31']
    df.set_index('Date', inplace = True)
    
    df_temp = df[df.year == analysis_year]
    
    fig, ax = plt.subplots()
    plt.xticks(rotation = -30)
    price, = ax.plot(df_temp.index, df_temp.IXIC, c='grey', lw = 2, alpha=0.5, zorder = 5)
    
    df_temp['smoothed'] = savgol_filter(df_temp.IXIC, 60, 5)
    fig, ax = plt.subplots()
    plt.xticks(rotation = -30)
    price, = ax.plot(df_temp.index, df_temp.IXIC, c='grey', lw = 2, alpha=0.5, zorder = 5) 
    price_smooth, = ax.plot(df_temp.index, df_temp.smoothed, c='b', lw = 2, zorder = 5) 
    
    
    atr = df_temp.atr.iloc[-1]
    
    peaks_idx, _ = find_peaks(df_temp.smoothed, distance = 15, width = 3, prominence = atr)
    
    troughs_idx, _ = find_peaks(-1*df_temp.smoothed, distance = 15, width = 3, prominence = atr)
    
    peaks, = ax.plot(df_temp.index[peaks_idx], df_temp.smoothed.iloc[peaks_idx], \
                     c = 'r', linestyle = 'None', markersize = 10, marker = 'o', zorder = 10)
    
    troughs, = ax.plot(df_temp.index[troughs_idx], df_temp.smoothed.iloc[troughs_idx], \
                     c = 'g', linestyle = 'None', markersize = 10, marker = 'o', zorder = 10)
        
    ax.set_ylabel('IXIC')    
    ax.set_title(analysis_year)  
    
   
    df_peak = pd.DataFrame(df_temp.smoothed.iloc[peaks_idx])
    df_peak['type'] = 'peak'
    df_trough = pd.DataFrame(df_temp.smoothed.iloc[troughs_idx])
    df_trough['type'] = 'trough'
    
    tp = pd.concat([df_peak, df_trough])
    
    tp = tp.sort_index()
    
    
    
    
    
    # Display
    
    
    
    st.title('纳斯达克区间划分')
    st.pyplot(fig)
    
    st.title('拐点')
    st.table(tp)    

#==============================================================================
# Tab 7 SP500 Intervals
#==============================================================================

def tab7():
    
    analysis_year = st.sidebar.number_input('Year', 1948)
    
    df = pd.read_csv('data/SP500.csv')
    
    df['year'] = pd.to_datetime(df['Date']).dt.year
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['SP500'] = df['SP500'].astype(float)
    
    df['atr'] = ta.atr(high = df['SP500'], low = df['SP500'], close = df['SP500'])
    df['atr'] = df.atr.rolling(window = 30).mean()
    
    df = df[df['Date'] >= '1948-01-01']
    df = df[df['Date'] <= '2018-12-31']
    df.set_index('Date', inplace = True)
    
    df_temp = df[df.year == analysis_year]
    
    fig, ax = plt.subplots()
    plt.xticks(rotation = -30)
    price, = ax.plot(df_temp.index, df_temp.SP500, c='grey', lw = 2, alpha=0.5, zorder = 5)
    
    df_temp['smoothed'] = savgol_filter(df_temp.SP500, 60, 5)
    fig, ax = plt.subplots()
    plt.xticks(rotation = -30)
    price, = ax.plot(df_temp.index, df_temp.SP500, c='grey', lw = 2, alpha=0.5, zorder = 5) 
    price_smooth, = ax.plot(df_temp.index, df_temp.smoothed, c='b', lw = 2, zorder = 5) 
    
    
    atr = df_temp.atr.iloc[-1]
    
    peaks_idx, _ = find_peaks(df_temp.smoothed, distance = 15, width = 3, prominence = atr)
    
    troughs_idx, _ = find_peaks(-1*df_temp.smoothed, distance = 15, width = 3, prominence = atr)
    
    peaks, = ax.plot(df_temp.index[peaks_idx], df_temp.smoothed.iloc[peaks_idx], \
                     c = 'r', linestyle = 'None', markersize = 10, marker = 'o', zorder = 10)
    
    troughs, = ax.plot(df_temp.index[troughs_idx], df_temp.smoothed.iloc[troughs_idx], \
                     c = 'g', linestyle = 'None', markersize = 10, marker = 'o', zorder = 10)
        
    ax.set_ylabel('SP500')    
    ax.set_title(analysis_year)  
    
   
    df_peak = pd.DataFrame(df_temp.smoothed.iloc[peaks_idx])
    df_peak['type'] = 'peak'
    df_trough = pd.DataFrame(df_temp.smoothed.iloc[troughs_idx])
    df_trough['type'] = 'trough'
    
    tp = pd.concat([df_peak, df_trough])
    
    tp = tp.sort_index()
    
    
    
    
    
    # Display
    
    
    
    st.title('SP500区间划分')
    st.pyplot(fig)
    
    st.title('拐点')
    st.table(tp)    

        

#==============================================================================
# Main body
#==============================================================================

def run():
    
    
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['概览', '指数', '宏观', '其它','道琼斯区间划分','纳斯达克区间划分','SP500区间划分'])

    # Show the selected tab
    if select_tab == '概览':
        tab1()
    elif select_tab == '指数':
        tab2()
    elif select_tab == '宏观':
        tab3()
    elif select_tab == '其它':
        tab4()  
    elif select_tab == '道琼斯区间划分':
        tab5()    
    elif select_tab == '纳斯达克区间划分':
        tab6()    
    elif select_tab == 'SP500区间划分':
        tab7()            
        
if __name__ == "__main__":
    run()   
