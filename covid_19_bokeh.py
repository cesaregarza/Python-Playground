# %%
import pandas as pd
import numpy as np

df = pd.read_excel("F:\\Dev\\Data\\Texas COVID-19 Case Count Data by County.xlsx", skiprows=2).iloc[:254]
df.columns = df.columns.str.replace('\n', ' ')

for col in df.iloc[:, 1:].columns:
    df[col] = df[col].astype(int)

df = df.set_index("County Name")
col_length = len(df.columns) - 1
date_list = df.columns.str.extract(r'(\d{2}\-\d{2})').iloc[:, 0].to_list()[1:]

delta_df = df.iloc[:, 1:] - df.iloc[:, 1:].shift(axis=1)


# %%
from bokeh.io import show
from bokeh.models import LogColorMapper, ColumnDataSource, Slider, CustomJS, Div
from bokeh.palettes import Viridis9 as palette
from bokeh.plotting import figure, output_file
from bokeh.layouts import column, row
from bokeh.sampledata.us_counties import data as counties
from bokeh.models.formatters import TickFormatter

palette = tuple(reversed(palette))

counties = {
    code: county for code, county in counties.items() if county["state"] == "tx"
}

county_xs = [county["lons"] for county in counties.values()]
county_ys = [county["lats"] for county in counties.values()]

county_names = [county['name'] for county in counties.values()]
county_rates = df.iloc[:, -1].to_list()
color_mapper = LogColorMapper(palette=palette)


source=ColumnDataSource(dict(
    x=county_xs,
    y=county_ys,
    name=county_names,
    df = df.values.tolist(),
    rate=df.iloc[:, col_length].to_list(),
    pop = df.iloc[:, 0].to_list()
))

TOOLS = "pan,wheel_zoom,reset,hover,save"

div = Div(text=f'<b>{date_list[-1]}</b>')

p = figure(
    title="Texas COVID-19 Cases, 2020", tools=TOOLS,
    x_axis_location=None, y_axis_location=None,
    tooltips=[
        ("Name", "@name"), ("Covid 19 Cases", "@rate{0,0}"), ("Population", "@pop{0,0}")
    ])

slider = Slider(start=1, end=col_length, step=1, value=col_length, title="Date")

callback = CustomJS(args=dict(source=source, slider=slider, div = div, date_list = date_list),
                    code="""
                    const data = source.data;
                    const col = slider.value;
                    const new_rate = data['df'].map(function(value, index){return value[col]});

                    var rate = data['rate']
                    rate.splice(0, rate.length, ...new_rate)
                    
                    div.text = date_list[col - 1]

                    source.change.emit();
                    """)

slider.js_on_change('value', callback)

p.grid.grid_line_color = None
p.hover.point_policy = "follow_mouse"

p.patches('x', 'y', source=source,
          fill_color={'field': 'rate', 'transform': color_mapper},
          fill_alpha=0.7, line_color="white", line_width=0.5)

layout= row(p, column(slider), div)

# %%
# output_file('texas_covid.html')
show(layout)

# %%
