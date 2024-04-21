#!/usr/bin/env python3
"""
Efficient data processing, visualisation and retrieval
Autor: xzdebs00

"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any
from scipy.integrate import trapz


def integrate(f: Callable[[NDArray], NDArray], 
              a: float, b: float, steps=1000) -> float:
    '''
    Numerical integration of a function f(x) using the trapezoidal rule.
    
    Parameters
    ----------
    f : Callable[[NDArray], NDArray]
        Function to be integrated
    a : float
        Lower bound of the integral
    b : float
        Upper bound of the integral
    steps : int, optional
        Number of steps, by default 1000
    
    Returns
    -------
    float
        Value of the integral
    '''
    array_x = np.linspace(a, b, steps)
    delta_x = array_x[1] - array_x[0]
    values_f = f((array_x[:-1] + array_x[1:])/2)
    return np.sum(delta_x * values_f)

def f_a(x, a):
    '''
    Function f_a(x) = a^2 * x^3 * sin(x)
    
    Parameters
    ----------
    x : float
        Value of x
    a : float
        Coefficient
    
    Returns
    -------
    float
        Value of the function
    '''
    return a**2 * x**3 * np.sin(x)

def generate_graph(a: List[float], 
                   show_figure: bool = False, 
                   save_path: str | None = None):
    '''
    Generating a graph with different coefficients

    Generate a graph of the function f_a(x) = a^2 * x^3 * sin(x) for a in a.
    
    Parameters
    ----------
    a : List[float]
        List of coefficients
    show_figure : bool, optional
        Display the figure, by default False
    save_path : str, optional
        Save the figure to the specified path, by default None
    '''
    x = np.linspace(-3, 3, 1000)
    y = np.array([f_a(x, ai) for ai in a])

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, ai in enumerate(a):
        ax.plot(x, y[i], label=f'$y_{{{ai}}}(x)$')
        ax.fill_between(x, y[i], alpha=0.1)
        
    ax.set_xlabel('$x$')
    ax.set_xlim(left=-3, right=5)
    x_ticks = np.arange(-3, 4)
    ax.set_xticks(x_ticks)
    ax.set_ylim(0, 40)
    ax.set_ylabel('$f_a(x)$')
    ax.grid(False)
    ax.legend(loc ="upper center", bbox_to_anchor=(0.5, 1.13), ncols = 3)

    integrals = [trapz(f, x) for f in y]
    for i, ai in enumerate(a):
        ax.annotate(f'∫ $f_{{{ai}}}(x)dx$ = {integrals[i]:.2f}', (3, y[i][-1]), fontsize=12, ha='left')

    if show_figure:
        plt.show()

    if save_path:
        fig.savefig(save_path)
        
def f1(t):
    '''
    First sine wave function.
    
    Parameters
    ----------
    t : float
        Time
    
    Returns
    -------
    float
        Value of the function at time t
    '''
    return 0.5 * np.cos(np.pi * 0.02 * t)

def f2(t):
    '''
    Second sine wave function.
    
    Parameters
    ----------
    t : float
        Time
    
    Returns
    -------
    float
        Value of the function at time t
    '''
    return 0.25 * (np.sin(np.pi * t) + np.sin(1.5 * np.pi * t))

def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    '''
    Advanced sine wave visualization.
    
    Parameters
    ----------
    show_figure : bool, optional
        Display the figure, by default False
    save_path : str, optional
        Save the figure to the specified path, by default None
    '''
    t = np.linspace(0, 100, 1000)

    y1 = f1(t)
    y2 = f2(t)

    y_sum = y1 + y2

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    ax1.plot(t, y1, color='blue')
    ax1.set_ylabel('f1(t)')
    ax1.set_xlim(0, 100)
    ax1.grid(True)

    ax2.plot(t, y2, color='orange')
    ax2.set_ylabel('f2(t)')
    ax2.set_xlim(0, 100)
    ax2.grid(True)

    ax3.fill_between(t, y_sum, y1, where=(y_sum > y1), interpolate=True, color='green')
    ax3.fill_between(t, y_sum, y1, where=(y_sum <= y1), interpolate=True, color='red')
    ax3.set_ylabel('f1(t) + f2(t)')
    ax3.set_xlim(0, 100)
    ax3.set_xlabel('t')
    ax3.grid(True)

    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.8))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.8))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(0.8))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    if show_figure:
        plt.show()

    if save_path:
        plt.savefig(save_path)


def download_data() -> List[Dict[str, Any]]:
    '''
    Download data from the website.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with station records
    '''
    url = "https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html"

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to retrieve data from the website.")

    soup = BeautifulSoup(response.content, "html.parser")
    
    table = soup.find_all('table')[1]
    station_records = []

    for row in table.find_all('tr')[1:]:
        columns = row.find_all('td')
        if len(columns) >= 4:
            position = columns[0].text.strip()
            lat = float(columns[2].text.strip().replace(',', '.').strip('°'))
            long = float(columns[4].text.strip().replace(',', '.').strip('°'))
            height = float(columns[6].text.strip().replace(',', '.').strip('°'))
            station_record = {
                'position': position,
                'lat': lat,
                'long': long,
                'height': height
            }
            station_records.append(station_record)

    return station_records