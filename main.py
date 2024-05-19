

import streamlit as st, pandas as pd, numpy as np
import plotly.express as px
import datetime




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

    
    start_date = st.sidebar.date_input('Start date', datetime.datetime(1945, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.datetime(2020, 12, 31))

    df = df[df['DATE'] >= start_date]
    df = df[df['DATE'] <= end_date]
    


        
    pct = pd.DataFrame([])
    
    for c in range(2,len(df.columns)):
        
        ds = pd.DataFrame([[df.columns[c], 100*(df.iloc[-1,c] - df.iloc[0,c])/df.iloc[0,c]]])
        pct = pd.concat([pct, ds])
        

        
    pct.columns = ['index', 'Growth']
    pct = pct[pct['index'].isin(pct_columns)]

    pct['color'] = np.where(pct["Growth"]<0, 'red', 'green')
        

    fig_summary = px.bar(pct,
                  x="index",
                  y="Growth",
                  color = 'color',
                  color_discrete_map = {"green":"green", "red":"red"})
    fig_summary.update_layout(showlegend=False)
    





    # Display
    st.title('概览') 
    st.plotly_chart(fig_summary)
    

        


    

        

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
    select_tab = st.sidebar.radio("Select tab", ['概览'])

    # Show the selected tab
    if select_tab == '概览':
        tab1()
 
        
      
 
       
    
if __name__ == "__main__":
    run()              
