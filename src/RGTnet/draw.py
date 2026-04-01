import os
import torch
import random
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage import measure

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_img(img, msk=None, cmap="jet", method="bilinear"):
    plt.imshow(img,cmap=cmap, interpolation=method)
    if msk is not None:
        plt.imshow(msk, alpha=0.4, cmap='jet', interpolation='nearest')  
    plt.colorbar(fraction=0.023,pad=0.02) 
    
def draw_slice(volume, x_slice, y_slice, z_slice, cmap='jet',clab=None):
    if len(volume.shape) > 3:
        volume = volume.squeeze()
    z, y, x = volume.shape
    cmin=np.min(volume)
    cmax=np.max(volume)
    
    if clab is None:
        showscale = False
    else:
        showscale = True
        
    # x-slice
    yy = np.arange(0, y, 1)
    zz = np.arange(0, z, 1)
    yy,zz = np.meshgrid(yy,zz)
    xx = x_slice * np.ones((y, z)).T
    vv = volume[:,:,x_slice]
    fig = go.Figure(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=showscale,
        colorbar={"title":clab, 
                  "title_side":'right',
                  "len": 0.8,
                  "thickness": 8,
                  "xanchor":"right"}))

    # y-slice
    xx = np.arange(0, x, 1)
    zz = np.arange(0, z, 1)
    xx,zz = np.meshgrid(xx,zz)
    yy = y_slice * np.ones((x, z)).T
    vv = volume[:,y_slice,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    # z-slice
    xx = np.arange(0, x, 1)
    yy = np.arange(0, y, 1)
    xx,yy = np.meshgrid(xx,yy)
    zz = z_slice * np.ones((x, y)).T
    vv = volume[z_slice,:,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    fig.update_layout(
            height=400,
            width=600,
            scene = {
            "xaxis": {"nticks": 5, "title":"Corssline"},
            "yaxis": {"nticks": 5, "title":"Inline"},
            "zaxis": {"nticks": 5, "autorange":'reversed', "title":"Sample"},
            'camera_eye': {"x": 1.25, "y": 1.25, "z": 1.25},
            'camera_up': {"x": 0, "y": 0, "z": 1},
            "aspectratio": {"x": 1, "y": 1, "z": 1.05}
            },
            margin=dict(t=0, l=0, b=0))
    fig.show()
    
def draw_slice_surf(volume, x_slice, y_slice, z_slice, cmap='jet', color='cyan',
                    clab=None, isofs=None, surfs=None, volume2=None):
    if len(volume.shape) > 3:
        volume = volume.squeeze()
    nz, ny, nx = volume.shape
    cmin=np.min(volume)
    cmax=np.max(volume)
    
    if clab is None:
        showscale = False
    else:
        showscale = True
      
    fig = go.Figure()  
    
    # surf
    if surfs is not None:
        for surf in surfs:
            xx,yy,zz = [],[],[]
            for ix in range(0,nx,2):
                for iy in range(0,ny,2):
                    if surf[iy][ix]>0 and surf[iy][ix]<nz-1:
                        xx.append(ix)
                        yy.append(iy)
                        zz.append(surf[iy][ix])
            obj = {}        
            obj.update({"type": "mesh3d",
                        "x": xx,
                        "y": yy,
                        "z": zz,
                        "color": color,
                        "opacity": 0.5})         
            fig.add_trace(obj)
            
    # iso-surf
    if isofs is not None:
        if volume2 is not None:
            cube = volume2
        else:
            cube = volume       
        for isof in isofs:
            obj = {}
            verts, faces, normals, values = measure.marching_cubes(cube.transpose(2,1,0), isof, step_size=2)
            obj.update({"type": "mesh3d",
                        "x": verts[:, 0],
                        "y": verts[:, 1],
                        "z": verts[:, 2],
                        "i": faces[:, 0],
                        "j": faces[:, 1],
                        "k": faces[:, 2],
                        "intensity": np.ones(len(verts[:, 0])) * isof,
                        "colorscale": "jet",
                        "showscale": False,
                        "cmin": cmin,
                        "cmax": cmax,
                        "opacity": 0.5})
            fig.add_trace(obj)
    
    # x-slice
    yy = np.arange(0, ny, 1)
    zz = np.arange(0, nz, 1)
    yy,zz = np.meshgrid(yy,zz)
    xx = x_slice * np.ones((ny, nz)).T
    vv = volume[:,:,x_slice]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=showscale,
        colorbar={"title":clab, 
                  "title_side":'right',
                  "len": 0.8,
                  "thickness": 8,
                  "xanchor":"right"}))

    # y-slice
    xx = np.arange(0, nx, 1)
    zz = np.arange(0, nz, 1)
    xx,zz = np.meshgrid(xx,zz)
    yy = y_slice * np.ones((nx, nz)).T
    vv = volume[:,y_slice,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    # z-slice
    xx = np.arange(0, nx, 1)
    yy = np.arange(0, ny, 1)
    xx,yy = np.meshgrid(xx,yy)
    zz = z_slice * np.ones((nx, ny)).T
    vv = volume[z_slice,:,:]
    fig.add_trace(go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=vv,
        colorscale=cmap,
        cmin=cmin, cmax=cmax,
        showscale=False))

    fig.update_layout(
            height=400,
            width=600,
            scene = {
            "xaxis": {"nticks": 5, "title":"Corssline"},
            "yaxis": {"nticks": 5, "title":"Inline"},
            "zaxis": {"nticks": 5, "autorange":'reversed', "title":"Sample"},
            'camera_eye': {"x": 1.25, "y": 1.25, "z": 1.25},
            'camera_up': {"x": 0, "y": 0, "z": 1},
            "aspectratio": {"x": 1, "y": 1, "z": 1.05}
            },
            margin=dict(t=0, l=0, b=0))
    fig.show()