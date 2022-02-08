#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

import plotly.graph_objects as go
import panel as pn
from autograd import grad, numpy as autograd_np
import matplotlib.cm


# Solo necesario para notebook:
#pn.extension("plotly")


# Funciones auxiliares:

def rgba_a_plotly_rgba_string(rgba_tuple):
    els = ",".join([str(int(i*255)) for i in rgba_tuple[:-1]])
    return "rgb({})".format(els)

def crear_colormap_plotly_de_matplotlib(cmap_name, resolution=256):
    return [(idx, rgba_a_plotly_rgba_string(matplotlib.cm.get_cmap(cmap_name)(idx)))
            for idx in np.linspace(0.0, 1.0, resolution)]

def calcular_rango_ejes(rango_valores_eje):
    return (rango_valores_eje.min()-((rango_valores_eje.max()- rango_valores_eje.min())*0.1),
            rango_valores_eje.max()+((rango_valores_eje.max()- rango_valores_eje.min())*0.1)
           )

def calcular_norma_escaladora(vector_orig, vector_dest, rangos_ejes, dimensiones_figura):
    return np.sqrt((np.linalg.norm(vector_orig/rangos_ejes*dimensiones_figura)**2) / 
                     np.sum((vector_dest/rangos_ejes*dimensiones_figura)**2))


## Elementos del dashboard:

# Selección de la loss function:
seleccion_loss_function = pn.widgets.RadioBoxGroup(name='loss_function', 
                                                   options=['Regresión lineal (teoría)', 
                                                            'Perceptrón multicapa (teoría)', 
                                                            'Personalizada'], 
                                                   inline=True)

input_loss_function = pn.widgets.input.TextAreaInput(name='Loss function', 
                                                     placeholder=("Introduce aquí una expresión para el loss en base " + 
                                                                  "a w1 y w2. Puedes usar funciones de numpy (como np.loquesea). " +
                                                                  "Por ejemplo: np.log(np.abs(w2)+1) + 0.1 * np.abs(w1**3) - 0.5 * w1**2 + 1e-4 * w2 **2 * w1**2 + 12"),
                                                     value="np.log(np.abs(w2)+1) + 0.1 * np.abs(w1**3) - 0.5 * w1**2 + 1e-4 * w2 **2 * w1**2 + 12",
                                                     sizing_mode="stretch_width")
pane_beta_inicial_1_custom_loss = pn.widgets.FloatInput(name="w₁ inicial:", 
                                                        sizing_mode="stretch_width",
                                                        value=-5.5,
                                                        height=50)
pane_beta_inicial_2_custom_loss = pn.widgets.FloatInput(name="w₂ inicial:", 
                                                        sizing_mode="stretch_width",
                                                        value=4.0,
                                                        height=50)
w1_min = pn.widgets.FloatInput(name="Eje w₁ (min):", 
                               sizing_mode="stretch_width",
                               value=-6.0,
                               height=50)
w1_max = pn.widgets.FloatInput(name="Eje w₁ (max)", 
                               sizing_mode="stretch_width",
                               value=6.0,
                               height=50)
w2_min = pn.widgets.FloatInput(name="Eje w₂ (min):", 
                               sizing_mode="stretch_width",
                               value=-6.0,
                               height=50)
w2_max = pn.widgets.FloatInput(name="Eje w₂ (max):", 
                               sizing_mode="stretch_width",
                               value=6.0,
                               height=50)

grid_loss = pn.GridSpec(height=120, sizing_mode="stretch_width"
                       )
grid_loss[0:14, :2] = input_loss_function
grid_loss[0:8, 2:3] = w1_min
grid_loss[0:8, 3:4] = w1_max
grid_loss[0:8, 4:5] = w2_min
grid_loss[0:8, 5:6] = w2_max
grid_loss[8:16, 2:4] = pane_beta_inicial_1_custom_loss
grid_loss[8:16, 4:6] = pane_beta_inicial_2_custom_loss

generacion_loss_function = pn.widgets.Button(name="¡Generar!", sizing_mode="stretch_width")
grid_boton = pn.GridSpec(nrows=1, ncols=6, height=35, sizing_mode="stretch_width")

grid_boton[0, :1] = None
grid_boton[0, 1:5] = generacion_loss_function
grid_boton[0, 5:] = None

# Iteraciones de gradient descent:
pane_beta_1 = pn.widgets.FloatInput(name="w₁:", sizing_mode="stretch_width")
pane_beta_2 = pn.widgets.FloatInput(name="w₂:", sizing_mode="stretch_width")
pane_valor_loss = pn.pane.Markdown(object="Valor loss: ", sizing_mode="stretch_width")
pane_vector_gradiente = pn.pane.Markdown(object="Vector gradiente: ", sizing_mode="stretch_width")

slider_step_size = pn.widgets.DiscreteSlider(name='Tamaño step (learning rate)', 
                                             options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 80.0], 
                                             value=0.1, 
                                             sizing_mode="stretch_width")
boton_step = pn.widgets.Button(name="¡Calcular gradiente!",  
                               sizing_mode="stretch_width")

caja_lateral = pn.WidgetBox(pane_beta_1, 
                            pane_beta_2,
                            pane_valor_loss, 
                            pane_vector_gradiente, 
                            slider_step_size, 
                            boton_step,
                            pn.pane.Markdown(object=""),
                            sizing_mode="stretch_width"
                           )

# Gráfico + caja lateral:
row_plot = pn.Row(pn.Spacer(sizing_mode="stretch_both"),
                  caja_lateral,
                  sizing_mode="stretch_height"
                 )

# Footer:

# Dashboard completo:
dashboard = pn.Column(pn.pane.HTML(object="<h1>Demo Gradient Descent</h1>",
                                   style={"text-align": "center",
                                          "margin-bottom": "0"},
                                   sizing_mode="stretch_width"),
                      pn.pane.HTML(object='<h3>Generación problema (loss function)</h3>',
                                   sizing_mode="stretch_width",
                                   style={"margin-top": "0"}),
                      seleccion_loss_function,
                      grid_boton,
                      pn.pane.HTML(object='<br><h3>Algoritmo</h3>',
                                   sizing_mode="stretch_width",
                                   style={"margin-top": "0"}),
                      row_plot,
                      pn.pane.HTML(object="""<p>Demo por <a href="https://twitter.com/julio_a_soto">Julio Antonio Soto</a> 
                                             para <a href="https://www.datahack.es/">datahack</a>. Realizada con <a href="https://panel.holoviz.org/">Panel</a>, <a href="https://plotly.com/graphing-libraries/">Plotly</a>, 
                                             <a href="https://numpy.org/">numpy</a> y <a href="https://github.com/HIPS/autograd">autograd</a>.</p>"""),
                      sizing_mode="stretch_width"
                      )


# Callbacks

def añadir_opciones_personalizadas_loss(*events):
    for event in events:
        if event.name=="value":
            if event.new == "Personalizada":
                dashboard.insert(3, grid_loss)
                fig_inicial["layout"]["height"] = 401
                if plotly_grafico.relayout_data is not None:
                    fig_inicial["layout"]["scene"]["camera"]["eye"] = plotly_grafico.relayout_data["scene.camera"]["eye"]
                plotly_grafico.object = fig_inicial
            else:
                if len(dashboard) > 7:
                    dashboard.pop(3)
                    fig_inicial["layout"]["height"] = 400
                    if plotly_grafico.relayout_data is not None:
                        fig_inicial["layout"]["scene"]["camera"]["eye"] = plotly_grafico.relayout_data["scene.camera"]["eye"]
                    plotly_grafico.object = fig_inicial
                    
def pintar_loss_function():
    beta_1, beta_2 = np.meshgrid(rango_beta_1, rango_beta_2)
    loss = loss_function(beta_1, beta_2)
    
    loss_superficie = go.Surface(x=beta_1,
                                 y=beta_2,
                                 z=loss,
                                 opacity=1.0,
                                 autocolorscale=False,
                                 colorscale=crear_colormap_plotly_de_matplotlib("RdBu_r", resolution=256),
                                 showscale=False,
                                 name="Loss function",
                                 hovertemplate = ('w₁: %{x}<br>'+
                                                  'w₂: %{y}<br>'+
                                                  'Error (loss): %{z}'),
                                )
    
    global rango_eje_x
    global rango_eje_y
    global rango_eje_z
    
    rango_eje_x = calcular_rango_ejes(rango_beta_1)
    rango_eje_y = calcular_rango_ejes(rango_beta_2)
    rango_eje_z = calcular_rango_ejes(loss)
    
    layout = go.Layout(scene_xaxis_title="w₁",
                       scene_yaxis_title="w₂",
                       scene_zaxis_title="Error (loss)",
                       margin=dict(l=0, r=0, b=0, t=0),
                       scene=dict(xaxis_range=rango_eje_x,
                                  yaxis_range=rango_eje_y,
                                  zaxis_range=rango_eje_z
                                  ),
                       uirevision=True,  # Allows for modifying elements without changing axis and zoom etc
                       autosize=True,
                       scene_aspectmode="manual",
                       scene_aspectratio={"x": 4, "y":4, "z":3},
                       scene_camera_eye={"x": 3, "y": -6, "z": 3},
                       height=400
                       )
    fig = {"data": [loss_superficie], "layout": layout}

    return fig

def pintar_elementos_gradient_descent(valores_betas):
    valor_loss = loss_function(*valores_betas)
    
    # Betas actuales:
    punto_beta = go.Scatter3d(x=[valores_betas[0]],
                              y=[valores_betas[1]],
                              z=[valor_loss],
                              marker_color="purple",
                              showlegend=False,
                              name="Valor actual weights",
                              hovertemplate = ('w₁: %{x}<br>'+
                                               'w₂: %{y}<br>'+
                                               'Error (loss): %{z}')
                             )
    
    valor_gradiente = np.nan_to_num(np.array(gradiente_loss_function(*valores_betas)))
    if np.linalg.norm(valor_gradiente) == 0.0:
        return (punto_beta, None, None)
    valor_gradiente_normalizado = valor_gradiente/np.linalg.norm(valor_gradiente)

    # Computar flecha del vector gradiente:
    flecha = [[valores_betas[0]-valor_gradiente_normalizado[0], 
               valores_betas[0]+valor_gradiente_normalizado[0]], 
              [valores_betas[1]-valor_gradiente_normalizado[1], 
               valores_betas[1]+valor_gradiente_normalizado[1]], 
              [valor_loss+np.dot(valor_gradiente, -valor_gradiente_normalizado), 
               valor_loss+np.dot(valor_gradiente, valor_gradiente_normalizado)]
             ]
    
    # Nuestro vector algebraico por tanto es:
    vector_gradiente = np.array([flecha[0][1] - flecha[0][0], 
                                 flecha[1][1] - flecha[1][0],
                                 flecha[2][1] - flecha[2][0]])
    
    # Para que siempre tenga un tamaño razonable en el gráfico,
    # debemos escalarlo acorde al tamaño de los ejes y al aspect
    # ratio del gráfico:
    vector_gradiente_normalizado = vector_gradiente / np.linalg.norm(vector_gradiente)
    
    longitud_eje_x = rango_eje_x[1] - rango_eje_x[0]
    longitud_eje_y = rango_eje_y[1] - rango_eje_y[0]
    longitud_eje_z = rango_eje_z[1] - rango_eje_z[0]
    cubo = np.array([longitud_eje_x, longitud_eje_y, longitud_eje_z]) / 5.0
    
    norma_escaladora = calcular_norma_escaladora(cubo, 
                                                 vector_gradiente_normalizado, 
                                                 cubo,
                                                 np.array([1, 1, 1]))
    
    vector_gradiente_escalado = (vector_gradiente_normalizado
                                 * norma_escaladora
                                 * 0.5
                                )
    
    # Coordenadas de la línea final para el vector:
    flecha_final = [[valores_betas[0]-vector_gradiente_escalado[0], valores_betas[0]+vector_gradiente_escalado[0]], 
                    [valores_betas[1]-vector_gradiente_escalado[1], valores_betas[1]+vector_gradiente_escalado[1]], 
                    [valor_loss-vector_gradiente_escalado[2], valor_loss+vector_gradiente_escalado[2]]
                    ]
    
    # Pintar línea de la flecha:
    linea_flecha = go.Scatter3d(x=flecha_final[0],
                                y=flecha_final[1],
                                z=flecha_final[2],
                                mode='lines',
                                line_color="yellow",
                                line_width=6,
                                hoverinfo="skip",
                                showlegend=False)

    # Pintar punta de la flecha (cono):
    punta_flecha = go.Cone(x=[flecha_final[0][0]],
                           y=[flecha_final[1][0]],
                           z=[flecha_final[2][0]],
                           u=[-vector_gradiente_escalado[0]],
                           v=[-vector_gradiente_escalado[1]],
                           w=[-vector_gradiente_escalado[2]],
                           anchor="tail",
                           showscale=False,
                           showlegend=False,
                           hoverinfo="skip",
                           colorscale=[[0, "yellow"], [1, "yellow"]],
                           sizemode="scaled"
                          )
    return punto_beta, linea_flecha, punta_flecha

def iniciar_figura(click):
    global loss_function
    global rango_beta_1
    global rango_beta_2
    global betas_actuales
    global plotly_grafico
    global fig_inicial
    global gradiente_loss_function

    if seleccion_loss_function.value == "Regresión lineal (teoría)":
        loss_function = lambda w1, w2: 0.01 * ( (w1-14.5)**2 + (w2+11.3)**2 + 0.99)
        rango_beta_1 = np.linspace(-15, 45, 100)
        rango_beta_2 = np.linspace(-45, 20, 100)
        betas_actuales = np.array([43.0, 19])
    elif seleccion_loss_function.value == "Perceptrón multicapa (teoría)":
        loss_function = lambda w1, w2: (2*autograd_np.sin(w1) 
                                         + 0.05*autograd_np.sin(w2)*w2
                                         + 3*(1-w1)**2 * autograd_np.exp(-(w1**2) - (w2+1)**2) 
                                         - 10*(w1/5 - w1**3 - w2**5) * autograd_np.exp(-(w1+0)**2-w2**2)
                                         - 4*autograd_np.exp(-(w1+1)**2 -w2**2)
                                         + autograd_np.exp(-4*autograd_np.log(2) * ((w1+2)**2 + (w2+3)**2) / 3**2)
                                         + autograd_np.exp(-2 * ((w1+2)**3 + (w2+3)**2) / 3**2)
                                         + 5 * autograd_np.exp(-2 *((w1-3)**2 + (w2+3)**2) / 3**2)
                                         + 3 * autograd_np.exp(-2 *((w1-4)**2 + (w2-3)**2) / 3**2)
                                         + 2 * autograd_np.exp(-2 *((w1+4)**2 + (w2-3)**2) / 3.5**2)
                                         - 6.5 * autograd_np.exp(-2 *((w1-2)**2 + (w2-1.5)**2) / 1.5**2)
                                         + 8.3
                                         )
        rango_beta_1 = np.linspace(-4., 4, 100)
        rango_beta_2 = np.linspace(-4., 4, 100)
        betas_actuales = np.array([0.36, 1.17])
    elif seleccion_loss_function.value == "Personalizada":
        loss_function_custom = input_loss_function.value
        loss_function_custom = ("lambda w1, w2: " + 
                                loss_function_custom.replace("np.", "autograd_np.").replace("\n", " ")
                               )
        loss_function = eval(loss_function_custom)
        rango_beta_1 = np.linspace(w1_min.value, w1_max.value, 100)
        rango_beta_2 = np.linspace(w2_min.value, w2_max.value, 100)
        betas_actuales = np.array([float(pane_beta_inicial_1_custom_loss.value), 
                                   float(pane_beta_inicial_2_custom_loss.value)])
        

    fig_inicial = pintar_loss_function()
    gradiente_loss_function = grad(loss_function, argnum=[0,1])

    elementos = pintar_elementos_gradient_descent(betas_actuales)
    fig_inicial["data"].append(elementos[0])
    plotly_grafico = pn.pane.Plotly(config={'responsive': True})
    plotly_grafico.object = fig_inicial
    dashboard[-2][0] = plotly_grafico
    fig_inicial["layout"]["height"] = 402
    plotly_grafico.object = fig_inicial
    boton_step.name = "¡Calcular gradiente!"
    pane_beta_1.value = betas_actuales[0]
    pane_beta_2.value = betas_actuales[1]
    pane_beta_1.disabled=False
    pane_beta_2.disabled=False
    pane_valor_loss.object = ("Valor loss: " +  "%.4f" % loss_function(*betas_actuales)
                              )
    
def actualizar_beta_1(*events):
    for event in events:
        if event.name=="value":
            global betas_actuales
            global betas_cambiadas_automaticamente
            betas_actuales[0] = event.new
            punto_beta, _, _ = pintar_elementos_gradient_descent(betas_actuales)
            fig_inicial["data"][1].update({"x": punto_beta["x"],
                                           "y": punto_beta["y"],
                                           "z": punto_beta["z"]})
            if len(fig_inicial["data"]) == 5 and betas_cambiadas_automaticamente == False:
                del fig_inicial["data"][4]
            plotly_grafico.object = fig_inicial
            boton_step.name="¡Calcular gradiente!"
            boton_step.disabled=False
            
def actualizar_beta_2(*events):
    for event in events:
        if event.name=="value":
            global betas_actuales
            global betas_cambiadas_automaticamente
            betas_actuales[1] = event.new
            punto_beta, _, _ = pintar_elementos_gradient_descent(betas_actuales)
            fig_inicial["data"][1].update({"x": punto_beta["x"],
                                           "y": punto_beta["y"],
                                           "z": punto_beta["z"]})
            if len(fig_inicial["data"]) == 5 and betas_cambiadas_automaticamente == False:
                del fig_inicial["data"][4]
            plotly_grafico.object = fig_inicial
            boton_step.name="¡Calcular gradiente!"
            boton_step.disabled=False

def actualizar_betas():
    global betas_actuales
    global betas_cambiadas_automaticamente
    gradiente = np.nan_to_num(np.array(gradiente_loss_function(*betas_actuales)))
    nuevas_betas = betas_actuales - slider_step_size.value * gradiente
    punto_beta, _, _ = pintar_elementos_gradient_descent(nuevas_betas)
    fig_inicial["data"][1].update({"x": punto_beta["x"],
                                   "y": punto_beta["y"],
                                   "z": punto_beta["z"]})
    if len(fig_inicial["data"]) > 3:
        fig_inicial["data"][2].update({"visible": False})
        fig_inicial["data"][3].update({"visible": False})
        
    if len(fig_inicial["data"]) == 5:
        fig_inicial["data"][4].update({"x": list(fig_inicial["data"][4]["x"]) + list(np.linspace(fig_inicial["data"][4]["x"][-1], punto_beta["x"][0], 100)),
                                       "y": list(fig_inicial["data"][4]["y"]) + list(np.linspace(fig_inicial["data"][4]["y"][-1], punto_beta["y"][0], 100)),
                                       "z": list(fig_inicial["data"][4]["z"]) + [loss_function(*p) for p in 
                                                                                 zip(list(np.linspace(fig_inicial["data"][4]["x"][-1], punto_beta["x"][0], 100)),
                                                                                     list(np.linspace(fig_inicial["data"][4]["y"][-1], punto_beta["y"][0], 100)))
                                                                                ]
                                       })
    else:
        fig_inicial["data"].append(go.Scatter3d(x=list(np.linspace(betas_actuales[0], punto_beta["x"][0], 100)),
                                                y=list(np.linspace(betas_actuales[1], punto_beta["y"][0], 100)),
												z=[loss_function(*p) for p in 
                                                   zip(list(np.linspace(betas_actuales[0], punto_beta["x"][0], 100)),
                                                       list(np.linspace(betas_actuales[1], punto_beta["y"][0], 100))
													   )],
                                                mode='lines',
                                                line_color="black",
												#marker_symbol="cross",
												line_dash="dash",
                                                line_width=4,
                                                showlegend=False,
                                                hoverinfo="skip"
                                               )
                                  )
        
    plotly_grafico.object = fig_inicial
    betas_actuales = nuevas_betas
    betas_cambiadas_automaticamente = True
    pane_beta_1.value = betas_actuales[0]
    pane_beta_2.value = betas_actuales[1]
    betas_cambiadas_automaticamente = False
    pane_beta_1.disabled=False
    pane_beta_2.disabled=False
    pane_valor_loss.object = ("Valor loss: " +  "%.4f" % loss_function(*betas_actuales)
                              )
    pane_vector_gradiente.object = "Vector gradiente: "
    
def actualizar_gradiente():
    _, linea_flecha, punta_flecha = pintar_elementos_gradient_descent(betas_actuales)
    if None in (linea_flecha, punta_flecha):
        raise ValueError
    if len(fig_inicial["data"]) < 3:
        fig_inicial["data"].append(linea_flecha)
        fig_inicial["data"].append(punta_flecha)
    else:
        fig_inicial["data"][2].update({"x": linea_flecha["x"],
                                       "y": linea_flecha["y"],
                                       "z": linea_flecha["z"],
                                       "visible": True}
                                       )
        fig_inicial["data"][3].update({"x": punta_flecha["x"],
                                       "y": punta_flecha["y"],
                                       "z": punta_flecha["z"],
                                       "u": punta_flecha["u"],
                                       "v": punta_flecha["v"],
                                       "w": punta_flecha["w"],
                                       "visible": True}
                                      )
    plotly_grafico.object = fig_inicial
    pane_vector_gradiente.object = ("Vector gradiente: " +  "\[{}\]"
                                    .format(", ".join(["%.7f" % i for i in 
                                                       np.nan_to_num(np.array(gradiente_loss_function(*betas_actuales)))]))
                                    )
    pane_beta_1.disabled=True
    pane_beta_2.disabled=True
    
def gradient_descent_paso(event):
    if boton_step.name == "¡Actualizar weights!":
        actualizar_betas()
        boton_step.name="¡Calcular gradiente!"
    else:
        try:
            actualizar_gradiente()
            boton_step.name="¡Actualizar weights!"
        except ValueError:
            boton_step.name="¡Convergido!"
            boton_step.disabled=True

# Anclar callbacks a elementos:
seleccion_loss_function.param.watch(añadir_opciones_personalizadas_loss, ["value"], onlychanged=True)
generacion_loss_function.on_click(iniciar_figura)
pane_beta_1.param.watch(actualizar_beta_1, ["value"], onlychanged=True)
pane_beta_2.param.watch(actualizar_beta_2, ["value"], onlychanged=True)
boton_step.on_click(gradient_descent_paso)


# Inicialización dashboard:
generacion_loss_function.clicks = 1
betas_cambiadas_automaticamente = False
dashboard.servable(title="Gradient Descent Demo")

