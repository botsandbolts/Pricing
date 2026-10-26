"""Interactive Plotly charts. Every axis states its actual prediction scale."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pricing.visual_explanations import label

RED='#e76f51';BLUE='#3478c7';GREY='#8b96a5'

def style(fig,title,xaxis=None,yaxis=None):
    fig.update_layout(title=title,template='plotly_white',height=480,
        margin=dict(l=15,r=20,t=65,b=45),font=dict(size=13),legend=dict(orientation='h'))
    if xaxis:fig.update_xaxes(title=xaxis)
    if yaxis:fig.update_yaxes(title=yaxis)
    return fig

def waterfall(local):
    order=np.argsort(-np.abs(local['values']));names=[local['names'][i] for i in order]
    values=local['values'][order]
    fig=go.Figure(go.Waterfall(orientation='h',
        measure=['absolute']+['relative']*len(values)+['total'],
        y=['Reference cars']+names+['Your estimate'],x=[local['base']]+values.tolist()+[0],
        text=[f"${local['base']:,.0f}"]+[f'{v:+,.0f}' for v in values]+[f"${local['prediction']:,.0f}"],
        textposition='outside',increasing=dict(marker_color=RED),decreasing=dict(marker_color=BLUE),
        totals=dict(marker_color=GREY),connector=dict(line=dict(color='#b7bec9')),
        customdata=[local['base']]+values.tolist()+[local['prediction']],
        hovertemplate='%{y}<br>USD value / change: %{customdata:,.2f}<extra></extra>'))
    fig.update_yaxes(autorange='reversed');fig.update_xaxes(tickprefix='$')
    return style(fig,'Why this asking price? · complete model','Predicted asking price / contribution (USD)')

def swarm_offsets(values):
    """Stack near-overlapping points symmetrically in narrow contribution bins."""
    values=np.asarray(values,float);span=np.ptp(values)
    bins=np.floor((values-values.min())/(span or 1)*35).astype(int);offset=np.zeros(len(values))
    for bucket in np.unique(bins):
        members=np.flatnonzero(bins==bucket)
        positions=np.arange(len(members));signed=np.ceil(positions/2)*np.where(positions%2,1,-1)
        offset[members]=signed*min(.06,.34/max(1,np.max(abs(signed))))
    return offset

def beeswarm(native,top=12):
    values=native['values'];x=native['x'];order=np.argsort(-np.mean(abs(values),axis=0))[:top]
    fig=go.Figure();first_colour=True
    for row,j in enumerate(order):
        name=native['names'][j];v=values[:,j];data=x.iloc[:,j]
        numeric=pd.api.types.is_numeric_dtype(data)
        if numeric:
            raw=pd.to_numeric(data,errors='coerce').to_numpy(dtype=float)
            valid=raw[np.isfinite(raw)];lo,hi=np.quantile(valid,[.05,.95]) if len(valid) else (0,1)
            colour=np.clip((raw-lo)/(hi-lo or 1),0,1)
            marker=dict(color=colour,colorscale=[[0,BLUE],[1,RED]],cmin=0,cmax=1,size=7,opacity=.8,
                        showscale=first_colour,colorbar=dict(title='Feature value',tickvals=[0,1],ticktext=['Low','High']))
            first_colour=False
        else:marker=dict(color=GREY,size=7,opacity=.8)
        display=data.astype('string').fillna('Missing').to_numpy()
        fig.add_trace(go.Scatter(x=v,y=row+swarm_offsets(v),mode='markers',marker=marker,
            customdata=display,name=label(name),showlegend=False,
            hovertemplate=label(name)+' = %{customdata}<br>SHAP: %{x:.4f}<extra></extra>'))
    fig.add_vline(x=0,line_color=GREY,line_dash='dot')
    fig.update_yaxes(tickvals=list(range(len(order))),ticktext=[label(native['names'][i]) for i in order],autorange='reversed')
    return style(fig,'Which features move predictions? · reference gallery',f"Native member SHAP ({native['unit']})")

def dependence(native,feature,colour_feature=None):
    index=native['names'].index(feature);data=native['x'][feature]
    fig=go.Figure();marker=dict(color=BLUE,size=9,opacity=.8)
    if colour_feature is not None:
        colour=pd.to_numeric(native['x'][colour_feature],errors='coerce')
        marker.update(color=colour,colorscale='Viridis',showscale=True,colorbar=dict(title=label(colour_feature)))
    fig.add_trace(go.Scatter(x=data.astype('string').fillna('Missing') if not pd.api.types.is_numeric_dtype(data) else data,
        y=native['values'][:,index],mode='markers',marker=marker,
        hovertemplate=label(feature)+' = %{x}<br>SHAP: %{y:.4f}<extra></extra>'))
    fig.add_hline(y=0,line_color=GREY,line_dash='dot')
    return style(fig,'How does this feature behave?',label(feature),f"Native member SHAP ({native['unit']})")

def decision(native,comparison,top=12):
    order=np.argsort(-np.mean(abs(native['values']),axis=0))[:top]
    other=[i for i in range(len(native['names'])) if i not in order]
    labels=['Reference']+[label(native['names'][i]) for i in order]
    matrix=native['values'][:,order]
    if other:labels+=['Other features'];matrix=np.column_stack([matrix,native['values'][:,other].sum(axis=1)])
    cumulative=np.column_stack([native['base'],native['base'][:,None]+np.cumsum(matrix,axis=1)])
    fig=go.Figure()
    for i,row in comparison.iterrows():
        name='Your car' if i==0 else f"Reference {i}: {row.manufacturer} {row.model}"
        fig.add_trace(go.Scatter(x=cumulative[i],y=list(range(len(labels))),mode='lines+markers',name=name,
            line=dict(width=4 if i==0 else 1.5),opacity=1 if i==0 else .65,
            hovertemplate='%{fullData.name}<br>Cumulative score: %{x:.4f}<extra></extra>'))
    fig.update_yaxes(tickvals=list(range(len(labels))),ticktext=labels,autorange='reversed')
    return style(fig,'Different cars, different reasons · one member',f"Cumulative native score ({native['unit']})")

def mileage_curve(ice,current_mileage,current_price):
    fig=go.Figure(go.Scatter(x=ice['mileage'],y=ice['price'],mode='lines',name='Model sensitivity',line=dict(color=BLUE,width=3)))
    fig.add_trace(go.Scatter(x=[current_mileage],y=[current_price],mode='markers',name='Submitted car',marker=dict(color=RED,size=12)))
    fig.update_yaxes(tickprefix='$')
    return style(fig,'What if this car had different mileage?','Mileage (miles)','Predicted asking price (USD)')

def interaction_grid(grid):
    fig=go.Figure(go.Heatmap(x=grid['mileage'],y=grid['years'],z=grid['price'],colorscale='Viridis',
        colorbar=dict(title='USD'),hovertemplate='Year: %{y}<br>Mileage: %{x:,.0f}<br>Estimate: $%{z:,.0f}<extra></extra>'))
    return style(fig,'Year × mileage · price sensitivity, not causal depreciation','Mileage (miles)','Model year')
